#!/usr/bin/env python3
"""Benchmark decode attention implementations on real benchmark-like tensors.

This tool:
1. Loads one real sample from `./data`
2. Replays the benchmark-style prefill + custom decode loop
3. Captures attention inputs (q/k/v/mask) from one decode step + layer
4. Benchmarks several attention implementations directly on the captured tensors

The capture path is eager for observability, but it reuses the same model, sample
preparation, StaticCache layout, and optional graph-style bucketed cache length.

Usage:
    TQDM_DISABLE=1 ./venv/bin/python bench_decode_attention.py
    TQDM_DISABLE=1 ./venv/bin/python bench_decode_attention.py --sample-idx 0 --layer-idx 0 --decode-step 0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("ENABLE_CUSTOM_DECODE", "1")
os.environ.setdefault("ENABLE_CUDAGRAPH", "0")
os.environ.setdefault("CUDAGRAPH_PREBUILD_MAX_CACHE_LEN", "0")

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers.cache_utils import StaticCache

import optimized_layer
from benchmark import get_system_info
from evaluation_wrapper import VLMModel
from optimized_layer import set_small_decode_triton_enabled


def _prepare_inputs(vlm: VLMModel, image, question: str):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }]
    return vlm.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(vlm.device)


class AttentionCapture:
    def __init__(self, target_call: int):
        self.target_call = target_call
        self.decode_call_index = 0
        self.payload = None

    def maybe_capture(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        num_key_value_groups: int,
        cache_position: torch.Tensor | None,
    ):
        if q.shape[2] != 1:
            return
        if self.payload is None and self.decode_call_index == self.target_call:
            self.payload = {
                "q": q.detach().clone(),
                "k": k.detach().clone(),
                "v": v.detach().clone(),
                "attention_mask": attention_mask.detach().clone() if attention_mask is not None else None,
                "scaling": float(scaling),
                "num_key_value_groups": int(num_key_value_groups),
                "cache_position": cache_position.detach().clone() if cache_position is not None else None,
            }
        self.decode_call_index += 1


def _install_attention_capture(capture: AttentionCapture):
    original = optimized_layer._small_decode_attention

    def wrapped(
        q,
        k,
        v,
        attention_mask,
        scaling,
        num_key_value_groups,
        prefer_small_triton=False,
        cache_position=None,
        splitkv_workspace=None,
    ):
        capture.maybe_capture(q, k, v, attention_mask, scaling, num_key_value_groups, cache_position)
        return original(
            q,
            k,
            v,
            attention_mask,
            scaling,
            num_key_value_groups,
            prefer_small_triton=prefer_small_triton,
            cache_position=cache_position,
            splitkv_workspace=splitkv_workspace,
        )

    optimized_layer._small_decode_attention = wrapped
    return original


def _restore_attention_capture(original):
    optimized_layer._small_decode_attention = original


def _build_bucketed_mask(vlm: VLMModel, inputs, max_new_tokens: int, bucketed: bool):
    prompt_length = inputs.input_ids.shape[1]
    engine = vlm.model._custom_decode
    max_cache_len = prompt_length + max_new_tokens
    if bucketed:
        max_cache_len = engine._get_graph_bucket_len(max_cache_len)

    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        prompt_attention_mask = torch.ones((1, prompt_length), dtype=torch.bool, device=vlm.device)
    else:
        prompt_attention_mask = attention_mask.to(device=vlm.device, dtype=torch.bool)

    full_attention_mask = torch.zeros((1, max_cache_len), dtype=torch.bool, device=vlm.device)
    full_attention_mask[:, :prompt_length] = prompt_attention_mask
    return prompt_attention_mask, full_attention_mask, max_cache_len


def _run_capture(
    vlm: VLMModel,
    inputs,
    decode_step: int,
    max_new_tokens: int,
    bucketed: bool,
    layer_idx: int,
):
    prompt_length = inputs.input_ids.shape[1]
    prompt_attention_mask, full_attention_mask, max_cache_len = _build_bucketed_mask(
        vlm,
        inputs,
        max_new_tokens=max_new_tokens,
        bucketed=bucketed,
    )
    static_cache = StaticCache(config=vlm._model.config, max_cache_len=max_cache_len)
    model = vlm._model
    target_call = decode_step * len(model.model.language_model.layers) + layer_idx
    capture = AttentionCapture(target_call=target_call)
    original_attention = _install_attention_capture(capture)

    previous_small_decode_triton = set_small_decode_triton_enabled(model, True)
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

            rope_deltas = outputs.rope_deltas
            next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            current_position = prompt_length
            engine = vlm.model._custom_decode

            for step in range(max_new_tokens - 1):
                full_attention_mask[:, current_position] = True
                position_ids = engine._build_decode_position_ids(
                    current_position=current_position,
                    rope_deltas=rope_deltas,
                    device=torch.device(vlm.device),
                )
                cache_position = torch.tensor([current_position], dtype=torch.long, device=vlm.device)
                outputs = model(
                    input_ids=next_tokens,
                    attention_mask=full_attention_mask,
                    position_ids=position_ids,
                    past_key_values=static_cache,
                    use_cache=True,
                    cache_position=cache_position,
                    logits_to_keep=1,
                )
                next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                if capture.payload is not None:
                    return capture.payload, {
                        "prompt_length": int(prompt_length),
                        "max_cache_len": int(max_cache_len),
                        "captured_decode_step": int(step),
                        "layer_idx": int(layer_idx),
                    }
                current_position += 1
    finally:
        set_small_decode_triton_enabled(model, previous_small_decode_triton)
        _restore_attention_capture(original_attention)

    raise RuntimeError(
        "Failed to capture attention tensors. "
        f"decode_step={decode_step}, layer_idx={layer_idx}, target_call={target_call}"
    )


def _is_prefix_mask(mask_1d: torch.Tensor) -> bool:
    valid_len = int(mask_1d.sum().item())
    return bool(mask_1d[:valid_len].all() and not mask_1d[valid_len:].any())


def _hf_repeat_kv_attention(q, k, v, attention_mask, scaling, num_key_value_groups):
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
    return F.scaled_dot_product_attention(
        q,
        kk,
        vv,
        attn_mask=attention_mask,
        dropout_p=0.0,
        scale=scaling,
        is_causal=False,
    ).transpose(1, 2).contiguous()


def _sdpa_gqa_mask(q, k, v, attention_mask, scaling, _num_key_value_groups):
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attention_mask,
        dropout_p=0.0,
        scale=scaling,
        is_causal=False,
        enable_gqa=True,
    ).transpose(1, 2).contiguous()


def _sdpa_gqa_valid_only(q, k, v, attention_mask, scaling, _num_key_value_groups):
    mask_1d = attention_mask[0, 0, 0]
    valid_len = int(mask_1d.sum().item())
    if not _is_prefix_mask(mask_1d):
        raise RuntimeError("attention_mask is not a simple prefix mask; cannot slice valid prefix safely")
    return F.scaled_dot_product_attention(
        q,
        k[:, :, :valid_len, :],
        v[:, :, :valid_len, :],
        attn_mask=None,
        dropout_p=0.0,
        scale=scaling,
        is_causal=False,
        enable_gqa=True,
    ).transpose(1, 2).contiguous()


def _triton_per_q_head(q, k, v, attention_mask, scaling, num_key_value_groups):
    out = optimized_layer._small_decode_attention_per_head(
        q,
        k,
        v,
        attention_mask,
        scaling,
        num_key_value_groups,
        prefer_small_triton=True,
    )
    if out is None:
        raise RuntimeError("Per-head Triton decode attention returned None for captured tensors")
    return out


def _triton_grouped_gqa(q, k, v, attention_mask, scaling, num_key_value_groups):
    out = optimized_layer._small_decode_attention_grouped(
        q,
        k,
        v,
        attention_mask,
        scaling,
        num_key_value_groups,
        prefer_small_triton=True,
    )
    if out is None:
        raise RuntimeError("Grouped-GQA Triton decode attention returned None for captured tensors")
    return out


def _triton_grouped_cachepos_prefix(q, k, v, attention_mask, scaling, num_key_value_groups, cache_position):
    out = optimized_layer._small_decode_attention_grouped_prefix_cachepos(
        q,
        k,
        v,
        scaling,
        num_key_value_groups,
        cache_position,
        prefer_small_triton=True,
    )
    if out is None:
        raise RuntimeError("Grouped cache-position Triton decode attention returned None for captured tensors")
    return out


def _triton_grouped_cachepos_prefix_dot(q, k, v, attention_mask, scaling, num_key_value_groups, cache_position):
    out = optimized_layer._small_decode_attention_grouped_prefix_cachepos_dot(
        q,
        k,
        v,
        scaling,
        num_key_value_groups,
        cache_position,
        prefer_small_triton=True,
    )
    if out is None:
        raise RuntimeError("Grouped cache-position dot Triton decode attention returned None for captured tensors")
    return out


def _triton_grouped_cachepos_splitkv(q, k, v, attention_mask, scaling, num_key_value_groups, cache_position):
    previous = optimized_layer._ENABLE_TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV
    optimized_layer._ENABLE_TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV = True
    try:
        out = optimized_layer._small_decode_attention_grouped_prefix_cachepos_splitkv(
            q,
            k,
            v,
            scaling,
            num_key_value_groups,
            cache_position,
            prefer_small_triton=True,
        )
    finally:
        optimized_layer._ENABLE_TRITON_SMALL_DECODE_ATTENTION_SPLIT_KV = previous
    if out is None:
        raise RuntimeError("Grouped cache-position split-KV Triton decode attention returned None for captured tensors")
    return out


def _triton_current(q, k, v, attention_mask, scaling, num_key_value_groups, cache_position):
    out = optimized_layer._small_decode_attention(
        q,
        k,
        v,
        attention_mask,
        scaling,
        num_key_value_groups,
        prefer_small_triton=True,
        cache_position=cache_position,
    )
    if out is None:
        raise RuntimeError("Default Triton decode attention returned None for captured tensors")
    return out


def _bench_impl(
    fn,
    q,
    k,
    v,
    attention_mask,
    scaling,
    num_key_value_groups,
    cache_position,
    warmup: int,
    iters: int,
):
    for _ in range(warmup):
        fn(q, k, v, attention_mask, scaling, num_key_value_groups, cache_position)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = None
    for _ in range(iters):
        out = fn(q, k, v, attention_mask, scaling, num_key_value_groups, cache_position)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / iters
    return out, elapsed_ms


def _bench_impl_graph(
    fn,
    q,
    k,
    v,
    attention_mask,
    scaling,
    num_key_value_groups,
    cache_position,
    warmup: int,
    iters: int,
):
    """Benchmark using CUDAGraph replay to eliminate Python dispatch overhead."""
    # Eager warmup to JIT-compile Triton kernels
    for _ in range(warmup):
        fn(q, k, v, attention_mask, scaling, num_key_value_groups, cache_position)
    torch.cuda.synchronize()

    # Capture CUDAGraph
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fn(q, k, v, attention_mask, scaling, num_key_value_groups, cache_position)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # Warmup replays
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    # Timed replays
    t0 = time.perf_counter()
    for _ in range(iters):
        graph.replay()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / iters
    return out, elapsed_ms


def _run_benchmarks(payload, warmup: int, iters: int, use_graph: bool = False):
    q = payload["q"]
    k = payload["k"]
    v = payload["v"]
    attention_mask = payload["attention_mask"]
    scaling = payload["scaling"]
    num_key_value_groups = payload["num_key_value_groups"]
    cache_position = payload["cache_position"]

    variants = {
        "hf_repeat_kv_sdpa": lambda q, k, v, attention_mask, scaling, num_key_value_groups, _cache_position: _hf_repeat_kv_attention(q, k, v, attention_mask, scaling, num_key_value_groups),
        "triton_per_q_head": lambda q, k, v, attention_mask, scaling, num_key_value_groups, _cache_position: _triton_per_q_head(q, k, v, attention_mask, scaling, num_key_value_groups),
        "triton_grouped_gqa": lambda q, k, v, attention_mask, scaling, num_key_value_groups, _cache_position: _triton_grouped_gqa(q, k, v, attention_mask, scaling, num_key_value_groups),
        "triton_grouped_cachepos_prefix": _triton_grouped_cachepos_prefix,
        "triton_grouped_cachepos_prefix_dot": _triton_grouped_cachepos_prefix_dot,
        "triton_grouped_cachepos_splitkv": _triton_grouped_cachepos_splitkv,
        "triton_current": _triton_current,
        "sdpa_enable_gqa_mask": lambda q, k, v, attention_mask, scaling, num_key_value_groups, _cache_position: _sdpa_gqa_mask(q, k, v, attention_mask, scaling, num_key_value_groups),
        "sdpa_enable_gqa_valid_prefix": lambda q, k, v, attention_mask, scaling, num_key_value_groups, _cache_position: _sdpa_gqa_valid_only(q, k, v, attention_mask, scaling, num_key_value_groups),
    }

    bench_fn = _bench_impl_graph if use_graph else _bench_impl
    results = {}
    ref_out, ref_ms = bench_fn(
        variants["hf_repeat_kv_sdpa"],
        q, k, v, attention_mask, scaling, num_key_value_groups, cache_position,
        warmup=warmup,
        iters=iters,
    )
    results["hf_repeat_kv_sdpa"] = {
        "time_ms": ref_ms,
        "status": "ok",
    }

    for name, fn in variants.items():
        if name == "hf_repeat_kv_sdpa":
            continue
        try:
            out, ms = bench_fn(
                fn,
                q, k, v, attention_mask, scaling, num_key_value_groups, cache_position,
                warmup=warmup,
                iters=iters,
            )
            diff = (out - ref_out).abs().float()
            results[name] = {
                "time_ms": ms,
                "status": "ok",
                "max_abs_diff": float(diff.max().item()),
                "mean_abs_diff": float(diff.mean().item()),
                "speedup_vs_hf": ref_ms / ms if ms > 0 else None,
            }
        except Exception as exc:
            results[name] = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark decode attention implementations on real captured tensors")
    parser.add_argument("--model-path", type=str, default="./Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", type=str, default="./data")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--decode-step", type=int, default=0, help="0 means the first decode step after prefill")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--bucketed", action="store_true", default=True, help="Use graph-style bucketed max_cache_len")
    parser.add_argument("--no-bucketed", dest="bucketed", action="store_false")
    parser.add_argument("--graph", action="store_true", default=True, help="Use CUDAGraph replay to eliminate dispatch overhead (default)")
    parser.add_argument("--no-graph", dest="graph", action="store_false", help="Use eager mode (includes Python dispatch overhead)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (default: results/...)")
    args = parser.parse_args()

    if args.output is None:
        args.output = (
            f"results/attention_microbench_sample{args.sample_idx}_"
            f"layer{args.layer_idx}_step{args.decode_step}.json"
        )

    print("[AttentionBench] Loading model...")
    vlm = VLMModel(args.model_path)
    dataset = load_from_disk(args.dataset_path)
    sample = dataset[args.sample_idx]
    inputs = _prepare_inputs(vlm, sample["image"], sample["question"])

    print(
        "[AttentionBench] Capturing tensors: "
        f"sample_idx={args.sample_idx}, question_id={sample['question_id']}, "
        f"layer_idx={args.layer_idx}, decode_step={args.decode_step}, bucketed={args.bucketed}"
    )
    payload, capture_meta = _run_capture(
        vlm,
        inputs,
        decode_step=args.decode_step,
        max_new_tokens=args.max_new_tokens,
        bucketed=args.bucketed,
        layer_idx=args.layer_idx,
    )

    mask_1d = payload["attention_mask"][0, 0, 0]
    metadata = {
        "question_id": int(sample["question_id"]),
        "sample_idx": int(args.sample_idx),
        "layer_idx": int(args.layer_idx),
        "decode_step": int(args.decode_step),
        "bucketed": bool(args.bucketed),
        "q_shape": list(payload["q"].shape),
        "k_shape": list(payload["k"].shape),
        "v_shape": list(payload["v"].shape),
        "attention_mask_shape": list(payload["attention_mask"].shape),
        "dtype": str(payload["q"].dtype),
        "device": str(payload["q"].device),
        "valid_len": int(mask_1d.sum().item()),
        "prefix_mask": bool(_is_prefix_mask(mask_1d)),
        "scaling": float(payload["scaling"]),
        "num_key_value_groups": int(payload["num_key_value_groups"]),
        "prompt_length": capture_meta["prompt_length"],
        "max_cache_len": capture_meta["max_cache_len"],
        "captured_decode_step": capture_meta["captured_decode_step"],
        "cache_position": int(payload["cache_position"].item()) if payload["cache_position"] is not None else None,
        "bench_mode": "graph" if args.graph else "eager",
    }

    print(
        "[AttentionBench] Captured: "
        f"valid_len={metadata['valid_len']}, max_cache_len={metadata['max_cache_len']}, "
        f"q={tuple(metadata['q_shape'])}, k={tuple(metadata['k_shape'])}"
    )
    bench_mode = "graph" if args.graph else "eager"
    print(f"[AttentionBench] Benchmarking implementations (mode={bench_mode})...")
    benchmarks = _run_benchmarks(payload, warmup=args.warmup, iters=args.iters, use_graph=args.graph)

    print("\n[AttentionBench] Summary")
    print(f"{'Implementation':<28} {'Time(ms)':>10} {'Speedup':>10} {'Status':>10}")
    print(f"{'-'*28} {'-'*10} {'-'*10} {'-'*10}")
    for name, info in benchmarks.items():
        if info["status"] == "ok":
            speedup = info.get("speedup_vs_hf")
            speedup_text = "-" if speedup is None else f"{speedup:.2f}x"
            print(f"{name:<28} {info['time_ms']:>10.4f} {speedup_text:>10} {'ok':>10}")
        else:
            print(f"{name:<28} {'-':>10} {'-':>10} {'error':>10}")

    output = {
        "system_info": get_system_info(),
        "metadata": metadata,
        "benchmarks": benchmarks,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n[AttentionBench] Saved to: {output_path}")


if __name__ == "__main__":
    main()
