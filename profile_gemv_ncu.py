#!/usr/bin/env python3
"""Run one captured decode-GEMV variant repeatedly for Nsight Compute profiling.

Usage pattern (standalone dry-run):
    TQDM_DISABLE=1 python profile_gemv_ncu.py --target o_proj --variant f_linear --warmup 5 --iters 10

With NCU:
    sudo /usr/local/cuda-12.6/bin/ncu -f --target-processes all \
      --launch-skip 5 --launch-count 1 \
      --metrics sm__cycles_elapsed.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warps_active.avg.pct_of_peak_sustained_active \
      --export results/ncu/gemv_o_proj_cublas \
      ./venv/bin/python profile_gemv_ncu.py --target o_proj --variant f_linear --warmup 5 --iters 1
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
os.environ.setdefault("ENABLE_CUDA_NOSPLIT_GEMV", "0")

import torch
import torch.nn.functional as F
from datasets import load_from_disk

from bench_decode_attention import _prepare_inputs
from bench_decode_linears import (
    LinearCapture,
    _run_capture,
    _cuda_nosplit_hfma2_matvec,
    _cuda_nosplit_k6144_n8k1024_xw_async_matvec,
    _cuda_nosplit_k6144_n8k1024_xwarp_w_async_matvec,
    _cuda_nosplit_k6144_n8k1024_xwarp_w_async_sms2_matvec,
    _cuda_nosplit_k6144_n8k1024_xwarp_w_async_sms2_balanced_matvec,
    _cuda_nosplit_k6144_persistent_xfull_n8k1024_w_async_matvec,
    _cuda_nosplit_k6144_persistent_xfull_w2_async_matvec,
    _cuda_nosplit_k6144_persistent_xfull_row1_warp8_async_matvec,
    _cuda_nosplit_matvec,
    _triton_matvec,
    _triton_splitk4_matvec,
    _triton_splitk8_matvec,
)
from evaluation_wrapper import VLMModel
from sweep_gemv import run_gemv_v2


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

def _f_linear(x, weight, bias):
    return F.linear(x, weight, bias)


def _triton_gemv_v2(x, weight, bias):
    """v2 kernel with small BLOCK_M for high grid parallelism."""
    return run_gemv_v2(x, weight, bias, block_m=4, block_k=256, num_warps=4, num_stages=2)


VARIANTS = {
    "f_linear": _f_linear,
    "cuda_nosplit_matvec": _cuda_nosplit_matvec,
    "cuda_nosplit_hfma2_matvec": _cuda_nosplit_hfma2_matvec,
    "cuda_nosplit_k6144_n8k1024_xw_async_matvec": _cuda_nosplit_k6144_n8k1024_xw_async_matvec,
    "cuda_nosplit_k6144_n8k1024_xwarp_w_async_matvec": _cuda_nosplit_k6144_n8k1024_xwarp_w_async_matvec,
    "cuda_nosplit_k6144_n8k1024_xwarp_w_async_sms2_matvec": _cuda_nosplit_k6144_n8k1024_xwarp_w_async_sms2_matvec,
    "cuda_nosplit_k6144_n8k1024_xwarp_w_async_sms2_balanced_matvec": _cuda_nosplit_k6144_n8k1024_xwarp_w_async_sms2_balanced_matvec,
    "cuda_nosplit_k6144_persistent_xfull_n8k1024_w_async_matvec": _cuda_nosplit_k6144_persistent_xfull_n8k1024_w_async_matvec,
    "cuda_nosplit_k6144_persistent_xfull_w2_async_matvec": _cuda_nosplit_k6144_persistent_xfull_w2_async_matvec,
    "cuda_nosplit_k6144_persistent_xfull_row1_warp8_async_matvec": _cuda_nosplit_k6144_persistent_xfull_row1_warp8_async_matvec,
    "triton_matvec": _triton_matvec,
    "triton_gemv_v2": _triton_gemv_v2,
    "triton_splitk4": _triton_splitk4_matvec,
    "triton_splitk8": _triton_splitk8_matvec,
}


def _build_variant(name: str):
    try:
        return VARIANTS[name]
    except KeyError as exc:
        available = ", ".join(sorted(VARIANTS))
        raise SystemExit(
            f"Unsupported variant: {name}. Available: {available}"
        ) from exc


# ---------------------------------------------------------------------------
# Metadata dump
# ---------------------------------------------------------------------------

def _dump_metadata(path: str | None, metadata: dict):
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Profile one decode-GEMV variant with real captured tensors for NCU"
    )
    parser.add_argument("--model-path", type=str, default="./Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", type=str, default="./data")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--layer-idx", type=int, default=14)
    parser.add_argument("--decode-step", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--bucketed", action="store_true", default=True)
    parser.add_argument("--no-bucketed", dest="bucketed", action="store_false")
    parser.add_argument(
        "--target",
        type=str,
        default="o_proj",
        choices=["qkv", "gate_up", "o_proj", "down_proj", "lm_head"],
        help="Which GEMV target to profile",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="f_linear",
        help=f"Variant to run. Available: {', '.join(sorted(VARIANTS))}",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--metadata-output", type=str, default=None)
    args = parser.parse_args()

    print(
        f"[GEMV-NCU] Loading capture case: "
        f"sample={args.sample_idx}, layer={args.layer_idx}, decode_step={args.decode_step}, "
        f"target={args.target}, variant={args.variant}, "
        f"warmup={args.warmup}, iters={args.iters}"
    )

    vlm = VLMModel(args.model_path)
    dataset = load_from_disk(args.dataset_path)
    sample = dataset[args.sample_idx]
    inputs = _prepare_inputs(vlm, sample["image"], sample["question"])

    payloads, capture_meta = _run_capture(
        vlm,
        inputs,
        decode_step=args.decode_step,
        max_new_tokens=args.max_new_tokens,
        bucketed=args.bucketed,
        layer_idx=args.layer_idx,
    )

    target = args.target
    if target not in payloads:
        raise SystemExit(f"Target '{target}' not found in captured payloads. Available: {list(payloads.keys())}")

    payload = payloads[target]
    x = payload["x"]
    weight = payload["weight"]
    bias = payload["bias"]

    metadata = {
        "question_id": int(sample["question_id"]),
        "sample_idx": int(args.sample_idx),
        "layer_idx": int(args.layer_idx),
        "decode_step": int(args.decode_step),
        "target": target,
        "variant": args.variant,
        "bucketed": bool(args.bucketed),
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "x_shape": list(x.shape),
        "weight_shape": list(weight.shape),
        "has_bias": bias is not None,
        "dtype": str(x.dtype),
        "device": str(x.device),
        "prompt_length": int(capture_meta["prompt_length"]),
        "max_cache_len": int(capture_meta["max_cache_len"]),
    }
    _dump_metadata(args.metadata_output, metadata)

    print(
        f"[GEMV-NCU] Captured: target={target}, "
        f"x={tuple(x.shape)}, weight={tuple(weight.shape)}, "
        f"bias={'yes' if bias is not None else 'no'}, dtype={x.dtype}"
    )

    fn = _build_variant(args.variant)

    # --- Warmup (outside profiler range) ---
    with torch.no_grad():
        for _ in range(args.warmup):
            fn(x, weight, bias)
        torch.cuda.synchronize()

    # --- Profiling iterations ---
    # Use cudaProfilerStart/Stop so NCU with --profile-from-start off
    # only captures kernels in this section.
    print(f"[GEMV-NCU] Starting {args.iters} profiling iterations...")
    with torch.no_grad():
        torch.cuda.cudart().cudaProfilerStart()
        for idx in range(args.iters):
            torch.cuda.nvtx.range_push(f"gemv_{target}_{args.variant}")
            out = fn(x, weight, bias)
            torch.cuda.nvtx.range_pop()
            if idx == 0:
                print(
                    f"[GEMV-NCU] First output: "
                    f"shape={tuple(out.shape)}, dtype={out.dtype}, "
                    f"finite={bool(torch.isfinite(out).all().item())}"
                )
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()

    print("[GEMV-NCU] Done")


if __name__ == "__main__":
    main()
