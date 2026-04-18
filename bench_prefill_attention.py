#!/usr/bin/env python3
"""Benchmark text-prefill attention backends on real benchmark-path tensors.

This tool:
1. Loads one real sample from `./data`
2. Replays the real prompt-prefill path once
3. Captures one text decoder layer's prefill attention payload (q/k/v)
4. Benchmarks `current interface / SDPA auto / forced backends` directly

It is intended to answer whether SDPA's automatic backend selection is already
good enough for the text-prefill attention shapes seen in this repo.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("ENABLE_CUSTOM_DECODE", "1")
os.environ.setdefault("ENABLE_CUDAGRAPH", "0")
os.environ.setdefault("CUDAGRAPH_PREBUILD_MAX_CACHE_LEN", "0")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.cache_utils import StaticCache
from datasets import load_from_disk

from bench_decode_attention import _build_bucketed_mask, _prepare_inputs
from benchmark import get_system_info
from benchmark_cuda_utils import (
    DEFAULT_BENCHMARK_L2_FLUSH_MB,
    benchmark_style_cuda_reset,
    make_l2_flush_runner,
)
from evaluation_wrapper import VLMModel


def _shape_list(x):
    return list(x.shape) if torch.is_tensor(x) else None


def _repeat_kv(k: torch.Tensor, v: torch.Tensor, num_key_value_groups: int):
    batch, num_kv_heads, seq_len, head_dim = k.shape
    kk = (
        k[:, :, None, :, :]
        .expand(batch, num_kv_heads, num_key_value_groups, seq_len, head_dim)
        .reshape(batch, num_kv_heads * num_key_value_groups, seq_len, head_dim)
    )
    vv = (
        v[:, :, None, :, :]
        .expand(batch, num_kv_heads, num_key_value_groups, seq_len, head_dim)
        .reshape(batch, num_kv_heads * num_key_value_groups, seq_len, head_dim)
    )
    return kk, vv


class PrefillAttentionCapture:
    def __init__(self):
        self.payload = None

    def maybe_capture(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        is_causal: bool,
        num_key_value_groups: int,
    ):
        if self.payload is not None:
            return
        if q.ndim != 4 or q.shape[2] <= 1:
            return
        self.payload = {
            "q": q.detach().clone(),
            "k": k.detach().clone(),
            "v": v.detach().clone(),
            "attention_mask": attention_mask.detach().clone() if attention_mask is not None else None,
            "scaling": float(scaling),
            "is_causal": bool(is_causal),
            "num_key_value_groups": int(num_key_value_groups),
        }


def _run_prefill_capture(
    vlm: VLMModel,
    inputs,
    layer_idx: int,
    max_new_tokens: int,
    bucketed: bool,
):
    prompt_attention_mask, _full_attention_mask, max_cache_len = _build_bucketed_mask(
        vlm,
        inputs,
        max_new_tokens=max_new_tokens,
        bucketed=bucketed,
    )
    static_cache = StaticCache(config=vlm._model.config, max_cache_len=max_cache_len)
    model = vlm._model
    layer = model.model.language_model.layers[layer_idx]
    capture = PrefillAttentionCapture()
    original_interface = layer._attention_interface

    def wrapped_attention_interface(module, q, k, v, attention_mask, **kwargs):
        capture.maybe_capture(
            q=q,
            k=k,
            v=v,
            attention_mask=attention_mask,
            scaling=kwargs.get("scaling", module.scaling),
            is_causal=kwargs.get("is_causal", module.is_causal),
            num_key_value_groups=module.num_key_value_groups,
        )
        return original_interface(module, q, k, v, attention_mask, **kwargs)

    layer._attention_interface = wrapped_attention_interface
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=prompt_attention_mask,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                mm_token_type_ids=inputs.get("mm_token_type_ids"),
                past_key_values=static_cache,
                use_cache=True,
                logits_to_keep=1,
            )
    finally:
        layer._attention_interface = original_interface

    if capture.payload is None:
        raise RuntimeError(
            f"Failed to capture prefill attention payload for layer_idx={layer_idx}"
        )

    next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    meta = {
        "layer_idx": int(layer_idx),
        "prompt_length": int(inputs.input_ids.shape[1]),
        "max_cache_len": int(max_cache_len),
        "first_token_id": int(next_tokens[0, 0].item()),
    }
    return capture.payload, meta, original_interface, layer


def _current_interface(module, interface, q, k, v, attention_mask, scaling, is_causal, _num_key_value_groups):
    out, _ = interface(
        module,
        q,
        k,
        v,
        attention_mask,
        scaling=scaling,
        dropout=0.0,
        is_causal=is_causal,
    )
    return out


def _sdpa_auto_gqa(_module, _interface, q, k, v, attention_mask, scaling, is_causal, _num_key_value_groups):
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attention_mask,
        dropout_p=0.0,
        scale=scaling,
        is_causal=is_causal,
        enable_gqa=True,
    ).transpose(1, 2).contiguous()


def _sdpa_backend_gqa(backend):
    def _impl(_module, _interface, q, k, v, attention_mask, scaling, is_causal, _num_key_value_groups):
        with sdpa_kernel([backend]):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=0.0,
                scale=scaling,
                is_causal=is_causal,
                enable_gqa=True,
            ).transpose(1, 2).contiguous()

    return _impl


def _sdpa_backend_repeat_kv(backend):
    def _impl(_module, _interface, q, k, v, attention_mask, scaling, is_causal, num_key_value_groups):
        kk, vv = _repeat_kv(k, v, num_key_value_groups)
        with sdpa_kernel([backend]):
            return F.scaled_dot_product_attention(
                q,
                kk,
                vv,
                attn_mask=attention_mask,
                dropout_p=0.0,
                scale=scaling,
                is_causal=is_causal,
                enable_gqa=False,
            ).transpose(1, 2).contiguous()

    return _impl


def _bench_impl_graph(
    fn,
    module,
    interface,
    q,
    k,
    v,
    attention_mask,
    scaling,
    is_causal,
    num_key_value_groups,
    warmup: int,
    iters: int,
    capture_repeats: int,
    pre_replay=None,
):
    for _ in range(warmup):
        fn(module, interface, q, k, v, attention_mask, scaling, is_causal, num_key_value_groups)
    torch.cuda.synchronize()

    benchmark_style_cuda_reset()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(capture_repeats):
                out = fn(module, interface, q, k, v, attention_mask, scaling, is_causal, num_key_value_groups)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    benchmark_style_cuda_reset()
    for _ in range(warmup):
        if pre_replay is not None:
            pre_replay()
        graph.replay()
    torch.cuda.synchronize()

    benchmark_style_cuda_reset()
    if pre_replay is None:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        end.synchronize()
        elapsed_ms = start.elapsed_time(end) / (iters * capture_repeats)
    else:
        start_events = []
        end_events = []
        for _ in range(iters):
            pre_replay()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            start_events.append(start)
            end_events.append(end)
        torch.cuda.synchronize()
        elapsed_ms = sum(
            start.elapsed_time(end) for start, end in zip(start_events, end_events)
        ) / (iters * capture_repeats)
    return out, elapsed_ms


def _run_benchmarks(
    payload,
    module,
    interface,
    warmup: int,
    iters: int,
    graph_capture_repeats: int,
    l2_flush_mb: int,
):
    q = payload["q"]
    k = payload["k"]
    v = payload["v"]
    attention_mask = payload["attention_mask"]
    scaling = payload["scaling"]
    is_causal = payload["is_causal"]
    num_key_value_groups = payload["num_key_value_groups"]

    variants = {
        "current_interface_auto": _current_interface,
        "sdpa_auto_gqa": _sdpa_auto_gqa,
        "sdpa_flash_gqa": _sdpa_backend_gqa(SDPBackend.FLASH_ATTENTION),
        "sdpa_cudnn_gqa": _sdpa_backend_gqa(SDPBackend.CUDNN_ATTENTION),
        "sdpa_math_gqa": _sdpa_backend_gqa(SDPBackend.MATH),
        "sdpa_flash_repeat_kv": _sdpa_backend_repeat_kv(SDPBackend.FLASH_ATTENTION),
        "sdpa_efficient_repeat_kv": _sdpa_backend_repeat_kv(SDPBackend.EFFICIENT_ATTENTION),
        "sdpa_math_repeat_kv": _sdpa_backend_repeat_kv(SDPBackend.MATH),
    }

    pre_replay = make_l2_flush_runner(q.device, l2_flush_mb)
    results = {}
    ref_name = "current_interface_auto"
    ref_out, ref_ms = _bench_impl_graph(
        variants[ref_name],
        module,
        interface,
        q,
        k,
        v,
        attention_mask,
        scaling,
        is_causal,
        num_key_value_groups,
        warmup=warmup,
        iters=iters,
        capture_repeats=graph_capture_repeats,
        pre_replay=pre_replay,
    )
    results[ref_name] = {
        "status": "ok",
        "time_ms": ref_ms,
        "max_abs_diff_vs_current": 0.0,
    }

    for name, fn in variants.items():
        if name == ref_name:
            continue
        try:
            out, elapsed_ms = _bench_impl_graph(
                fn,
                module,
                interface,
                q,
                k,
                v,
                attention_mask,
                scaling,
                is_causal,
                num_key_value_groups,
                warmup=warmup,
                iters=iters,
                capture_repeats=graph_capture_repeats,
                pre_replay=pre_replay,
            )
            diff = float((out - ref_out).abs().max().item())
            results[name] = {
                "status": "ok",
                "time_ms": elapsed_ms,
                "max_abs_diff_vs_current": diff,
                "speedup_vs_current": (ref_ms / elapsed_ms) if elapsed_ms > 0 else None,
            }
        except Exception as exc:
            results[name] = {
                "status": "error",
                "error": str(exc),
            }
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark real prefill attention backends on captured text-layer tensors")
    parser.add_argument("--model-path", type=str, default="./Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", type=str, default="./data")
    parser.add_argument("--sample-idx", type=int, default=32)
    parser.add_argument("--layer-idx", type=int, default=14)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--graph-capture-repeats", type=int, default=1)
    parser.add_argument("--bucketed", action="store_true", default=True)
    parser.add_argument("--no-bucketed", dest="bucketed", action="store_false")
    parser.add_argument("--l2-flush-mb", type=int, default=DEFAULT_BENCHMARK_L2_FLUSH_MB)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = (
            f"results/prefill_attention_sample{args.sample_idx}_"
            f"layer{args.layer_idx}.json"
        )

    print("[PrefillAttentionBench] Loading model...")
    vlm = VLMModel(args.model_path)
    dataset = load_from_disk(args.dataset_path)
    sample = dataset[args.sample_idx]
    inputs = _prepare_inputs(vlm, sample["image"], sample["question"])

    print(
        "[PrefillAttentionBench] Capturing payload: "
        f"sample_idx={args.sample_idx}, question_id={sample['question_id']}, "
        f"layer_idx={args.layer_idx}, prompt_length={inputs.input_ids.shape[1]}"
    )
    payload, meta, interface, layer = _run_prefill_capture(
        vlm,
        inputs,
        layer_idx=args.layer_idx,
        max_new_tokens=args.max_new_tokens,
        bucketed=args.bucketed,
    )

    print(
        "[PrefillAttentionBench] Benchmarking variants "
        f"(warmup={args.warmup}, iters={args.iters}, l2_flush_mb={args.l2_flush_mb})..."
    )
    results = _run_benchmarks(
        payload,
        module=layer,
        interface=interface,
        warmup=args.warmup,
        iters=args.iters,
        graph_capture_repeats=args.graph_capture_repeats,
        l2_flush_mb=args.l2_flush_mb,
    )

    output = {
        "sample_idx": int(args.sample_idx),
        "question_id": int(sample["question_id"]),
        "layer_idx": int(args.layer_idx),
        "prompt_length": int(meta["prompt_length"]),
        "max_cache_len": int(meta["max_cache_len"]),
        "first_token_id": int(meta["first_token_id"]),
        "bucketed": bool(args.bucketed),
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "graph_capture_repeats": int(args.graph_capture_repeats),
        "l2_flush_mb": int(args.l2_flush_mb),
        "payload": {
            "q_shape": _shape_list(payload["q"]),
            "k_shape": _shape_list(payload["k"]),
            "v_shape": _shape_list(payload["v"]),
            "attention_mask_shape": _shape_list(payload["attention_mask"]),
            "scaling": float(payload["scaling"]),
            "is_causal": bool(payload["is_causal"]),
            "num_key_value_groups": int(payload["num_key_value_groups"]),
        },
        "variants": results,
        "system_info": get_system_info(),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"[PrefillAttentionBench] Saved results to {output_path}")
    for name, result in results.items():
        if result.get("status") == "ok":
            print(
                f"  {name}: {result['time_ms']:.4f} ms"
                + (
                    f", speedup_vs_current={result['speedup_vs_current']:.3f}x"
                    if "speedup_vs_current" in result
                    else ""
                )
                + f", max_abs_diff={result['max_abs_diff_vs_current']:.3e}"
            )
        else:
            print(f"  {name}: ERROR {result.get('error')}")


if __name__ == "__main__":
    main()
