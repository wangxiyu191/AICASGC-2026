#!/usr/bin/env python3
"""Run one captured decode-attention variant repeatedly for Nsight Compute profiling."""

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
from datasets import load_from_disk

import optimized_layer
from bench_decode_attention import (
    _cuda_v4_cta_qhead,
    _cuda_v4_cta_qhead_w12,
    _cuda_v4_cta_qhead_w12_contig,
    _cuda_v4_cta_qhead_w12_g8,
    _cuda_v4_cta_qhead_splitkv4,
    _cuda_v4_cta_qhead_w8,
    _cuda_v4_fused,
    _cuda_v4_splitkv,
    _hf_repeat_kv_attention,
    _prepare_inputs,
    _run_capture,
    _sdpa_gqa_mask,
    _sdpa_gqa_valid_only,
    _triton_current,
    _triton_grouped_cachepos_prefix,
    _triton_grouped_cachepos_splitkv,
    _triton_grouped_cachepos_prefix_dot,
    _triton_grouped_gqa,
    _triton_per_q_head,
)
from evaluation_wrapper import VLMModel


def _build_variant(name: str):
    variants = {
        "hf_repeat_kv_sdpa": lambda q, k, v, attention_mask, scaling, num_key_value_groups, _cache_position: _hf_repeat_kv_attention(
            q, k, v, attention_mask, scaling, num_key_value_groups
        ),
        "triton_per_q_head": lambda q, k, v, attention_mask, scaling, num_key_value_groups, _cache_position: _triton_per_q_head(
            q, k, v, attention_mask, scaling, num_key_value_groups
        ),
        "triton_grouped_gqa": lambda q, k, v, attention_mask, scaling, num_key_value_groups, _cache_position: _triton_grouped_gqa(
            q, k, v, attention_mask, scaling, num_key_value_groups
        ),
        "triton_grouped_cachepos_prefix": _triton_grouped_cachepos_prefix,
        "triton_grouped_cachepos_splitkv": _triton_grouped_cachepos_splitkv,
        "triton_grouped_cachepos_prefix_dot": _triton_grouped_cachepos_prefix_dot,
        "triton_current": _triton_current,
        "cuda_v4_cta_qhead": _cuda_v4_cta_qhead,
        "cuda_v4_cta_qhead_w8": _cuda_v4_cta_qhead_w8,
        "cuda_v4_cta_qhead_w12": _cuda_v4_cta_qhead_w12,
        "cuda_v4_cta_qhead_w12_contig": _cuda_v4_cta_qhead_w12_contig,
        "cuda_v4_cta_qhead_w12_g8": _cuda_v4_cta_qhead_w12_g8,
        "cuda_v4_cta_qhead_splitkv4": _cuda_v4_cta_qhead_splitkv4,
        "cuda_v4_splitkv": _cuda_v4_splitkv,
        "cuda_v4_fused": _cuda_v4_fused,
        "sdpa_enable_gqa_mask": lambda q, k, v, attention_mask, scaling, num_key_value_groups, _cache_position: _sdpa_gqa_mask(
            q, k, v, attention_mask, scaling, num_key_value_groups
        ),
        "sdpa_enable_gqa_valid_prefix": lambda q, k, v, attention_mask, scaling, num_key_value_groups, _cache_position: _sdpa_gqa_valid_only(
            q, k, v, attention_mask, scaling, num_key_value_groups
        ),
    }
    try:
        return variants[name]
    except KeyError as exc:
        raise SystemExit(f"Unsupported variant: {name}") from exc


def _dump_metadata(path: str | None, metadata: dict):
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Profile one decode-attention variant with real captured tensors")
    parser.add_argument("--model-path", type=str, default="./Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", type=str, default="./data")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--layer-idx", type=int, default=14)
    parser.add_argument("--decode-step", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--bucketed", action="store_true", default=True)
    parser.add_argument("--no-bucketed", dest="bucketed", action="store_false")
    parser.add_argument("--variant", type=str, default="triton_grouped_cachepos_prefix_dot")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--disable-triton-during-capture", action="store_true", default=True)
    parser.add_argument("--enable-triton-during-capture", dest="disable_triton_during_capture", action="store_false")
    parser.add_argument("--metadata-output", type=str, default=None)
    args = parser.parse_args()

    print(
        "[AttentionNCU] Loading capture case: "
        f"sample={args.sample_idx}, layer={args.layer_idx}, decode_step={args.decode_step}, "
        f"variant={args.variant}, warmup={args.warmup}, iters={args.iters}"
    )
    vlm = VLMModel(args.model_path)
    dataset = load_from_disk(args.dataset_path)
    sample = dataset[args.sample_idx]
    inputs = _prepare_inputs(vlm, sample["image"], sample["question"])
    prev_enable = optimized_layer._ENABLE_TRITON_SMALL_DECODE_ATTENTION
    prev_dot = optimized_layer._ENABLE_TRITON_SMALL_DECODE_ATTENTION_DOT
    try:
        if args.disable_triton_during_capture:
            optimized_layer._ENABLE_TRITON_SMALL_DECODE_ATTENTION = False
            optimized_layer._ENABLE_TRITON_SMALL_DECODE_ATTENTION_DOT = False
        payload, capture_meta = _run_capture(
            vlm,
            inputs,
            decode_step=args.decode_step,
            max_new_tokens=args.max_new_tokens,
            bucketed=args.bucketed,
            layer_idx=args.layer_idx,
        )
    finally:
        optimized_layer._ENABLE_TRITON_SMALL_DECODE_ATTENTION = prev_enable
        optimized_layer._ENABLE_TRITON_SMALL_DECODE_ATTENTION_DOT = prev_dot

    q = payload["q"]
    k = payload["k"]
    v = payload["v"]
    attention_mask = payload["attention_mask"]
    scaling = payload["scaling"]
    num_key_value_groups = payload["num_key_value_groups"]
    cache_position = payload["cache_position"]
    valid_len = int(attention_mask[0, 0, 0].sum().item()) if attention_mask is not None else None
    metadata = {
        "question_id": int(sample["question_id"]),
        "sample_idx": int(args.sample_idx),
        "layer_idx": int(args.layer_idx),
        "decode_step": int(args.decode_step),
        "variant": args.variant,
        "bucketed": bool(args.bucketed),
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "q_shape": list(q.shape),
        "k_shape": list(k.shape),
        "v_shape": list(v.shape),
        "attention_mask_shape": list(attention_mask.shape) if attention_mask is not None else None,
        "dtype": str(q.dtype),
        "device": str(q.device),
        "valid_len": valid_len,
        "num_key_value_groups": int(num_key_value_groups),
        "cache_position": int(cache_position.item()) if cache_position is not None else None,
        "prompt_length": int(capture_meta["prompt_length"]),
        "max_cache_len": int(capture_meta["max_cache_len"]),
    }
    _dump_metadata(args.metadata_output, metadata)

    print(
        "[AttentionNCU] Captured tensors: "
        f"q={tuple(q.shape)}, k={tuple(k.shape)}, valid_len={valid_len}, "
        f"cache_position={metadata['cache_position']}, "
        f"capture_triton_enabled={not args.disable_triton_during_capture}"
    )
    fn = _build_variant(args.variant)

    with torch.no_grad():
        for _ in range(args.warmup):
            fn(q, k, v, attention_mask, scaling, num_key_value_groups, cache_position)
        torch.cuda.synchronize()

        for idx in range(args.iters):
            out = fn(q, k, v, attention_mask, scaling, num_key_value_groups, cache_position)
            if idx == 0:
                print(
                    "[AttentionNCU] First output: "
                    f"shape={tuple(out.shape)}, dtype={out.dtype}, finite={bool(torch.isfinite(out).all().item())}"
                )
        torch.cuda.synchronize()

    print("[AttentionNCU] Done")


if __name__ == "__main__":
    main()
