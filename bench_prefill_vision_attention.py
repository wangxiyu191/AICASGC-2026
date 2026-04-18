#!/usr/bin/env python3
"""Benchmark vision-prefill attention backends on real benchmark-path tensors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("ENABLE_CUSTOM_DECODE", "1")
os.environ.setdefault("ENABLE_CUDAGRAPH", "0")
os.environ.setdefault("CUDAGraph_PREBUILD_MAX_CACHE_LEN", "0")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from datasets import load_from_disk

from bench_decode_attention import _prepare_inputs
from benchmark import get_system_info
from benchmark_cuda_utils import (
    DEFAULT_BENCHMARK_L2_FLUSH_MB,
    benchmark_style_cuda_reset,
    make_l2_flush_runner,
)
from evaluation_wrapper import VLMModel
from optimized_layer import fused_vision_rope


def _shape_list(x):
    return list(x.shape) if torch.is_tensor(x) else None


def _run_capture(
    vlm: VLMModel,
    inputs,
    block_idx: int,
    split_idx: int,
):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        apply_rotary_pos_emb_vision,
        eager_attention_forward,
        is_flash_attention_requested,
    )

    visual = vlm._model.model.visual
    blk = visual.blocks[block_idx]
    current_forward = blk.attn.forward
    capture = {"payload": None, "split_sizes": None}

    def wrapped_forward(
        this_attn,
        hidden_states,
        cu_seqlens,
        rotary_pos_emb=None,
        position_embeddings=None,
        **kwargs,
    ):
        prefill_split_sizes = kwargs.pop("prefill_split_sizes", None)
        if (
            prefill_split_sizes is None
            or torch.is_grad_enabled()
            or is_flash_attention_requested(this_attn.config)
        ):
            return current_forward(
                hidden_states,
                cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            this_attn.qkv(hidden_states)
            .reshape(seq_length, 3, this_attn.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        fused_rope_result = fused_vision_rope(query_states, key_states, cos, sin)
        if fused_rope_result is not None:
            query_states, key_states = fused_rope_result
        else:
            query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            this_attn.config._attn_implementation,
            eager_attention_forward,
        )
        split_tensors = [
            torch.split(tensor, prefill_split_sizes, dim=2)
            for tensor in (query_states, key_states, value_states)
        ]
        capture["split_sizes"] = [int(v) for v in prefill_split_sizes]
        attn_outputs = []
        for idx, (q, k, v) in enumerate(zip(*split_tensors)):
            if capture["payload"] is None and idx == split_idx:
                capture["payload"] = {
                    "q": q.detach().clone(),
                    "k": k.detach().clone(),
                    "v": v.detach().clone(),
                    "scaling": float(this_attn.scaling),
                    "is_causal": False,
                }
            attn_outputs.append(
                attention_interface(
                    this_attn,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=this_attn.scaling,
                    dropout=0.0 if not this_attn.training else this_attn.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
            )

        attn_output = torch.cat(attn_outputs, dim=1)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return this_attn.proj(attn_output)

    blk.attn.forward = wrapped_forward.__get__(blk.attn, type(blk.attn))
    try:
        with torch.no_grad():
            _ = vlm._model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.get("attention_mask"),
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                mm_token_type_ids=inputs.get("mm_token_type_ids"),
                use_cache=True,
                logits_to_keep=1,
            )
    finally:
        blk.attn.forward = current_forward

    if capture["payload"] is None:
        raise RuntimeError(
            f"Failed to capture vision attention payload for block_idx={block_idx}, split_idx={split_idx}"
        )
    meta = {
        "block_idx": int(block_idx),
        "split_idx": int(split_idx),
        "split_sizes": capture["split_sizes"],
        "image_grid_thw": inputs.get("image_grid_thw").detach().cpu().tolist(),
    }
    return capture["payload"], meta, blk.attn


def _current_interface(module, q, k, v, scaling, is_causal):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3_vl.modeling_qwen3_vl import eager_attention_forward

    interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        module.config._attn_implementation,
        eager_attention_forward,
    )
    out, _ = interface(
        module,
        q,
        k,
        v,
        attention_mask=None,
        scaling=scaling,
        dropout=0.0,
        is_causal=is_causal,
    )
    return out


def _sdpa_auto(q, k, v, scaling, is_causal):
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        scale=scaling,
        is_causal=is_causal,
    ).transpose(1, 2).contiguous()


def _sdpa_backend(backend):
    def _impl(q, k, v, scaling, is_causal):
        with sdpa_kernel([backend]):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                scale=scaling,
                is_causal=is_causal,
            ).transpose(1, 2).contiguous()

    return _impl


def _bench_impl_graph(
    fn,
    module,
    q,
    k,
    v,
    scaling,
    is_causal,
    warmup: int,
    iters: int,
    capture_repeats: int,
    pre_replay=None,
):
    for _ in range(warmup):
        out = fn(module, q, k, v, scaling, is_causal)
    torch.cuda.synchronize()

    benchmark_style_cuda_reset()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(capture_repeats):
                out = fn(module, q, k, v, scaling, is_causal)
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
        starts = []
        ends = []
        for _ in range(iters):
            pre_replay()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            starts.append(start)
            ends.append(end)
        torch.cuda.synchronize()
        elapsed_ms = sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / (iters * capture_repeats)
    return out, elapsed_ms


def _run_benchmarks(
    payload,
    module,
    warmup: int,
    iters: int,
    graph_capture_repeats: int,
    l2_flush_mb: int,
):
    q = payload["q"]
    k = payload["k"]
    v = payload["v"]
    scaling = payload["scaling"]
    is_causal = payload["is_causal"]

    variants = {
        "current_interface_auto": lambda module, q, k, v, scaling, is_causal: _current_interface(module, q, k, v, scaling, is_causal),
        "sdpa_auto": lambda _module, q, k, v, scaling, is_causal: _sdpa_auto(q, k, v, scaling, is_causal),
        "sdpa_flash": lambda _module, q, k, v, scaling, is_causal: _sdpa_backend(SDPBackend.FLASH_ATTENTION)(q, k, v, scaling, is_causal),
        "sdpa_cudnn": lambda _module, q, k, v, scaling, is_causal: _sdpa_backend(SDPBackend.CUDNN_ATTENTION)(q, k, v, scaling, is_causal),
        "sdpa_efficient": lambda _module, q, k, v, scaling, is_causal: _sdpa_backend(SDPBackend.EFFICIENT_ATTENTION)(q, k, v, scaling, is_causal),
        "sdpa_math": lambda _module, q, k, v, scaling, is_causal: _sdpa_backend(SDPBackend.MATH)(q, k, v, scaling, is_causal),
    }

    pre_replay = make_l2_flush_runner(q.device, l2_flush_mb)
    results = {}
    ref_out, ref_ms = _bench_impl_graph(
        variants["current_interface_auto"],
        module,
        q,
        k,
        v,
        scaling,
        is_causal,
        warmup=warmup,
        iters=iters,
        capture_repeats=graph_capture_repeats,
        pre_replay=pre_replay,
    )
    results["current_interface_auto"] = {
        "status": "ok",
        "time_ms": ref_ms,
        "max_abs_diff_vs_current": 0.0,
    }

    for name, fn in variants.items():
        if name == "current_interface_auto":
            continue
        try:
            out, elapsed_ms = _bench_impl_graph(
                fn,
                module,
                q,
                k,
                v,
                scaling,
                is_causal,
                warmup=warmup,
                iters=iters,
                capture_repeats=graph_capture_repeats,
                pre_replay=pre_replay,
            )
            results[name] = {
                "status": "ok",
                "time_ms": elapsed_ms,
                "max_abs_diff_vs_current": float((out - ref_out).abs().max().item()),
                "speedup_vs_current": (ref_ms / elapsed_ms) if elapsed_ms > 0 else None,
            }
        except Exception as exc:
            results[name] = {
                "status": "error",
                "error": str(exc),
            }
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark real vision-prefill attention backends on captured block tensors")
    parser.add_argument("--model-path", type=str, default="./Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", type=str, default="./data")
    parser.add_argument("--sample-idx", type=int, default=32)
    parser.add_argument("--block-idx", type=int, default=0)
    parser.add_argument("--split-idx", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--graph-capture-repeats", type=int, default=1)
    parser.add_argument("--l2-flush-mb", type=int, default=DEFAULT_BENCHMARK_L2_FLUSH_MB)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = (
            f"results/prefill_vision_attention_sample{args.sample_idx}_"
            f"block{args.block_idx}_split{args.split_idx}.json"
        )

    print("[VisionAttentionBench] Loading model...")
    vlm = VLMModel(args.model_path)
    dataset = load_from_disk(args.dataset_path)
    sample = dataset[args.sample_idx]
    inputs = _prepare_inputs(vlm, sample["image"], sample["question"])

    print(
        "[VisionAttentionBench] Capturing payload: "
        f"sample_idx={args.sample_idx}, question_id={sample['question_id']}, "
        f"block_idx={args.block_idx}, split_idx={args.split_idx}"
    )
    payload, meta, attn_module = _run_capture(
        vlm,
        inputs,
        block_idx=args.block_idx,
        split_idx=args.split_idx,
    )

    print(
        "[VisionAttentionBench] Benchmarking variants "
        f"(warmup={args.warmup}, iters={args.iters}, l2_flush_mb={args.l2_flush_mb})..."
    )
    results = _run_benchmarks(
        payload,
        module=attn_module,
        warmup=args.warmup,
        iters=args.iters,
        graph_capture_repeats=args.graph_capture_repeats,
        l2_flush_mb=args.l2_flush_mb,
    )

    output = {
        "sample_idx": int(args.sample_idx),
        "question_id": int(sample["question_id"]),
        "block_idx": int(args.block_idx),
        "split_idx": int(args.split_idx),
        "split_sizes": meta["split_sizes"],
        "image_grid_thw": meta["image_grid_thw"],
        "payload": {
            "q_shape": _shape_list(payload["q"]),
            "k_shape": _shape_list(payload["k"]),
            "v_shape": _shape_list(payload["v"]),
            "scaling": float(payload["scaling"]),
            "is_causal": bool(payload["is_causal"]),
        },
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "graph_capture_repeats": int(args.graph_capture_repeats),
        "l2_flush_mb": int(args.l2_flush_mb),
        "variants": results,
        "system_info": get_system_info(),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"[VisionAttentionBench] Saved results to {output_path}")
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
