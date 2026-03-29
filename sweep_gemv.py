#!/usr/bin/env python3
"""Sweep Triton GEMV configurations to find optimal settings for o_proj/down_proj.

NCU analysis revealed cuBLAS uses WMMA GEMM (not GEMV) for M=1 small-N shapes,
achieving only 52-59% DRAM throughput and 10% occupancy. A proper GEMV kernel
with higher grid parallelism should beat it.

Usage:
    TQDM_DISABLE=1 ./venv/bin/python sweep_gemv.py
    TQDM_DISABLE=1 ./venv/bin/python sweep_gemv.py --target o_proj --iters 500
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
import triton
import triton.language as tl
from datasets import load_from_disk

from bench_decode_attention import _prepare_inputs
from bench_decode_linears import LinearCapture, _run_capture
from evaluation_wrapper import VLMModel


# ---------------------------------------------------------------------------
# Triton GEMV kernel: high-parallelism design
#
# Key difference from the old matvec kernel:
#   - Old: BLOCK_M=64 → grid=32 for N=2048 → 70% SMs idle
#   - New: BLOCK_M=1~8 → grid=256~2048 → all SMs busy
# ---------------------------------------------------------------------------

@triton.jit
def _gemv_v2_kernel(
    X,       # [K] input vector
    W,       # [N, K] weight matrix (row-major)
    OUT,     # [N] output vector
    K: tl.constexpr,       # input dimension
    N,                     # output dimension
    stride_wn,             # W.stride(0) = K for row-major
    stride_wk,             # W.stride(1) = 1 for row-major
    HAS_BIAS: tl.constexpr,
    BIAS,
    BLOCK_M: tl.constexpr, # output rows per block
    BLOCK_K: tl.constexpr, # K elements per iteration
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for start_k in range(0, K, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        # Load x vector segment (broadcast across all blocks via L2 cache)
        x = tl.load(X + offs_k).to(tl.float32)
        # Load weight tile [BLOCK_M, BLOCK_K]
        w_ptrs = W + offs_m[:, None] * stride_wn + offs_k[None, :] * stride_wk
        w = tl.load(w_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
        # Dot product: reduce over K dimension
        acc += tl.sum(w * x[None, :], axis=1)

    if HAS_BIAS:
        bias = tl.load(BIAS + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        acc += bias

    tl.store(OUT + offs_m, acc.to(OUT.dtype.element_ty), mask=mask_m)


def run_gemv_v2(x, weight, bias, block_m, block_k, num_warps, num_stages):
    """Run the v2 GEMV kernel with given config."""
    x_vec = x.reshape(-1).contiguous()
    N, K = weight.shape
    out_vec = torch.empty((N,), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(N, block_m),)
    _gemv_v2_kernel[grid](
        x_vec,
        weight,
        out_vec,
        K,
        N,
        weight.stride(0),
        weight.stride(1),
        HAS_BIAS=bias is not None,
        BIAS=bias if bias is not None else out_vec,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out_vec.view(*x.shape[:-1], N)


# ---------------------------------------------------------------------------
# Alternative GEMV implementations: torch.mv / addmv (calls cublasGemv)
# ---------------------------------------------------------------------------

def _torch_mv(x, weight, bias):
    """Use torch.mv which calls cublasGemv directly instead of cublasGemm."""
    x_1d = x.reshape(-1)
    out = torch.mv(weight, x_1d)
    if bias is not None:
        out = out + bias
    return out.view(*x.shape[:-1], weight.shape[0])


def _torch_addmv(x, weight, bias):
    """Use torch.addmv (alpha*mat@vec + beta*vec)."""
    x_1d = x.reshape(-1)
    dummy = torch.zeros(weight.shape[0], dtype=x.dtype, device=x.device)
    out = torch.addmv(dummy, weight, x_1d)
    if bias is not None:
        out = out + bias
    return out.view(*x.shape[:-1], weight.shape[0])


# ---------------------------------------------------------------------------
# Benchmark infrastructure
# ---------------------------------------------------------------------------

def bench_fn(fn, x, weight, bias, warmup: int, iters: int):
    for _ in range(warmup):
        fn(x, weight, bias)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(x, weight, bias)
    torch.cuda.synchronize()
    elapsed_us = (time.perf_counter() - t0) * 1e6 / iters
    return out, elapsed_us


def main():
    parser = argparse.ArgumentParser(description="Sweep GEMV configurations")
    parser.add_argument("--model-path", type=str, default="./Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", type=str, default="./data")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--layer-idx", type=int, default=14)
    parser.add_argument("--decode-step", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--target", type=str, default="all",
                        choices=["o_proj", "down_proj", "lm_head", "all"])
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--output", type=str, default="results/gemv_sweep.json")
    args = parser.parse_args()

    print("[GEMV-Sweep] Loading model and capturing tensors...")
    vlm = VLMModel(args.model_path)
    dataset = load_from_disk(args.dataset_path)
    sample = dataset[args.sample_idx]
    inputs = _prepare_inputs(vlm, sample["image"], sample["question"])

    payloads, capture_meta = _run_capture(
        vlm, inputs,
        decode_step=args.decode_step,
        max_new_tokens=args.max_new_tokens,
        bucketed=True,
        layer_idx=args.layer_idx,
    )

    targets = ["o_proj", "down_proj", "lm_head"] if args.target == "all" else [args.target]

    # Configurations to sweep
    # (block_m, block_k, num_warps, num_stages)
    configs = [
        # High-parallelism configs (small BLOCK_M → large grid)
        (1,  128, 2, 2),
        (1,  256, 2, 2),
        (1,  256, 4, 2),
        (1,  512, 4, 2),
        (2,  128, 2, 2),
        (2,  256, 2, 2),
        (2,  256, 4, 2),
        (2,  512, 4, 2),
        (4,  128, 2, 2),
        (4,  128, 4, 2),
        (4,  256, 2, 2),
        (4,  256, 4, 2),
        (4,  512, 4, 2),
        (8,  128, 4, 2),
        (8,  256, 4, 2),
        (8,  512, 4, 2),
        # Medium-parallelism configs
        (16, 128, 4, 2),
        (16, 256, 4, 2),
        (32, 128, 4, 2),
        (32, 256, 4, 2),
        # Pipeline stages sweep
        (4,  256, 4, 1),
        (4,  256, 4, 3),
        (4,  256, 4, 4),
        # More warps
        (4,  256, 8, 2),
        (8,  256, 8, 2),
    ]

    results = {}

    for target in targets:
        payload = payloads[target]
        x = payload["x"]
        weight = payload["weight"]
        bias = payload["bias"]
        N, K = weight.shape

        print(f"\n{'='*60}")
        print(f"[GEMV-Sweep] Target: {target} | shape: [1,{K}] x [{N},{K}]")
        print(f"{'='*60}")

        # Baseline: F.linear (cuBLAS)
        ref_out, ref_us = bench_fn(
            lambda x, w, b: F.linear(x, w, b),
            x, weight, bias,
            warmup=args.warmup, iters=args.iters,
        )
        print(f"  {'cuBLAS F.linear':<40s} {ref_us:8.2f} us  (baseline)")

        target_results = {"cublas_baseline_us": ref_us, "configs": []}

        # Test torch.mv and addmv (these call cublasGemv instead of cublasGemm)
        for name, fn in [("torch.mv", _torch_mv), ("torch.addmv", _torch_addmv)]:
            try:
                out, us = bench_fn(fn, x, weight, bias, warmup=args.warmup, iters=args.iters)
                diff = (out - ref_out).abs().float()
                max_diff = float(diff.max().item())
                speedup = ref_us / us if us > 0 else 0
                marker = " ***" if speedup > 1.0 else ""
                print(f"  {name:<40s} {us:8.2f} us  {speedup:5.2f}x  diff={max_diff:.1e}{marker}")
                target_results["configs"].append({
                    "name": name, "time_us": us, "speedup": speedup,
                    "max_abs_diff": max_diff,
                })
            except Exception as exc:
                print(f"  {name:<40s} ERROR: {exc}")

        for block_m, block_k, nw, ns in configs:
            # Skip configs that don't divide evenly (avoid masking overhead)
            if K % block_k != 0:
                continue
            grid_size = triton.cdiv(N, block_m)
            label = f"bm={block_m:>3d} bk={block_k:>4d} w={nw} s={ns} grid={grid_size}"

            def make_fn(bm, bk, nw_, ns_):
                return lambda x, w, b: run_gemv_v2(x, w, b, bm, bk, nw_, ns_)

            fn = make_fn(block_m, block_k, nw, ns)
            try:
                out, us = bench_fn(fn, x, weight, bias, warmup=args.warmup, iters=args.iters)
                diff = (out - ref_out).abs().float()
                max_diff = float(diff.max().item())
                speedup = ref_us / us if us > 0 else 0
                status = "OK" if max_diff < 1e-2 else "DIFF"
                marker = " ***" if speedup > 1.0 else ""
                print(f"  {label:<40s} {us:8.2f} us  {speedup:5.2f}x  diff={max_diff:.1e} {status}{marker}")
                target_results["configs"].append({
                    "block_m": block_m, "block_k": block_k,
                    "num_warps": nw, "num_stages": ns,
                    "grid_size": grid_size,
                    "time_us": us, "speedup": speedup,
                    "max_abs_diff": max_diff,
                })
            except Exception as exc:
                print(f"  {label:<40s} ERROR: {exc}")
                target_results["configs"].append({
                    "block_m": block_m, "block_k": block_k,
                    "num_warps": nw, "num_stages": ns,
                    "grid_size": grid_size,
                    "status": "error", "error": str(exc),
                })

        # Find best
        valid = [c for c in target_results["configs"]
                 if "speedup" in c and c.get("max_abs_diff", 999) < 1e-2]
        if valid:
            best = max(valid, key=lambda c: c["speedup"])
            print(f"\n  BEST: bm={best['block_m']} bk={best['block_k']} "
                  f"w={best['num_warps']} s={best['num_stages']} → "
                  f"{best['time_us']:.2f} us ({best['speedup']:.2f}x vs cuBLAS)")
        results[target] = target_results

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[GEMV-Sweep] Results saved to: {output_path}")


if __name__ == "__main__":
    main()
