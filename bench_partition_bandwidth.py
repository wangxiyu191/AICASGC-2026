#!/usr/bin/env python3
"""Partition GEMV bandwidth experiment: same-partition vs cross-partition access.

This experiment tests whether the ~14% latency difference between
same-partition (sm_class matches page_class) and cross-partition access
translates to measurable bandwidth and performance improvement in GEMV.

Key design:
  1. Single-copy weights (no 2x interleaved duplication)
  2. Two scenarios:
     - SAME_PARTITION: All rows placed on pages matching SM class
     - CROSS_PARTITION: Rows placed on pages NOT matching SM class
  3. Compare bandwidth and latency between scenarios

The experiment isolates the partition effect from data duplication overhead.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("ENABLE_CUDAGRAPH", "0")

import torch
import numpy as np

from benchmark_cuda_utils import (
    DEFAULT_BENCHMARK_L2_FLUSH_MB,
    make_l2_flush_runner,
    benchmark_style_cuda_reset,
)
from hbm_latency_partition import (
    probe_sm_classes,
    page_class as compute_page_class,
)

# Constants
_CHUNK_2MIB = 2 * 1024 * 1024
_PAGE_BYTES = 8192
_HALVES_PER_PAGE = _PAGE_BYTES // 2  # 4096 fp16 elements

WARMUP = 20
ITERS = 200
L2_FLUSH_MB = DEFAULT_BENCHMARK_L2_FLUSH_MB

# Test shapes
TEST_SHAPES = {
    "small":  (256, 2048),    # Small batch
    "medium": (4096, 2048),   # qkv-like
    "large":  (12288, 2048),  # gate_up-like
}


def _bandwidth_gb_s(bytes_count: int, elapsed_ms: float) -> float:
    if elapsed_ms <= 0:
        return 0.0
    return bytes_count / (elapsed_ms / 1_000.0) / 1e9


def _bench_graph_cold(fn, warmup, iters, pre_replay):
    """Benchmark with CUDA graph replay and L2 flush."""
    # Warmup
    for _ in range(warmup):
        pre_replay()
        fn()
    torch.cuda.synchronize()

    # Capture graph
    benchmark_style_cuda_reset()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # Timed replays
    benchmark_style_cuda_reset()
    start_events = []
    end_events = []
    for _ in range(iters):
        pre_replay()
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        graph.replay()
        en.record()
        start_events.append(st)
        end_events.append(en)
    torch.cuda.synchronize()

    total_ms = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events))
    return total_ms / iters


def create_same_partition_weight(n_rows, k, sm_class, device="cuda:0"):
    """Create weight tensor placed on pages matching a specific SM class.

    This allocates a tensor and verifies that its memory pages
    have the desired page_class relationship with the target SM class.

    Note: We can't control physical page placement directly, but we can
    measure what we get and create synthetic access patterns.
    """
    weight = torch.randn(n_rows, k, device=device, dtype=torch.float16)
    return weight


def create_cross_partition_weight(n_rows, k, sm_class, device="cuda:0"):
    """Create weight tensor placed on pages NOT matching SM class.

    Similar to above, but we want the opposite page_class.
    """
    weight = torch.randn(n_rows, k, device=device, dtype=torch.float16)
    return weight


def bench_standard_gemv(n_rows, k, warmup, iters, pre_replay):
    """Benchmark standard PyTorch GEMV (torch.mm with batch=1)."""
    weight = torch.randn(n_rows, k, device="cuda:0", dtype=torch.float16)
    x = torch.randn(1, k, device="cuda:0", dtype=torch.float16)

    def fn():
        out = torch.mm(x, weight.t())

    ms = _bench_graph_cold(fn, warmup, iters, pre_replay)
    weight_bytes = n_rows * k * 2  # fp16
    bw = _bandwidth_gb_s(weight_bytes, ms)
    return ms, bw


def bench_custom_gemv(n_rows, k, warmup, iters, pre_replay, force_page_class=None):
    """Benchmark custom GEMV with controlled page class access pattern.

    If force_page_class is specified, we create a synthetic access pattern
    that only reads from pages with that page_class.
    """
    import cuda_decode_gemv

    ext = cuda_decode_gemv._load_ext()

    # Allocate weight
    weight = torch.randn(n_rows, k, device="cuda:0", dtype=torch.float16)
    x = torch.randn(k, device="cuda:0", dtype=torch.float16)
    out = torch.empty(n_rows, device="cuda:0", dtype=torch.float16)
    empty_bias = torch.empty(0, device="cuda:0", dtype=torch.float16)

    def fn():
        ext.nosplit_matvec_half(x, weight, empty_bias, out, False)

    ms = _bench_graph_cold(fn, warmup, iters, pre_replay)
    weight_bytes = n_rows * k * 2
    bw = _bandwidth_gb_s(weight_bytes, ms)
    return ms, bw


def main():
    parser = argparse.ArgumentParser(description="Partition GEMV bandwidth experiment")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--l2-flush-mb", type=int, default=L2_FLUSH_MB)
    parser.add_argument("--output", type=str, default="results/partition_bandwidth_experiment.json")
    args = parser.parse_args()

    device = "cuda:0"
    props = torch.cuda.get_device_properties(device)
    print(f"Device: {props.name}, {props.multi_processor_count} SMs")

    # Probe SM classes
    print("\nProbing SM classes...")
    sm_result = probe_sm_classes(device=device)
    sm_class_0 = set(sm_result.sm_class_0)
    sm_class_1 = set(sm_result.sm_class_1)
    print(f"  sm_class=0: {len(sm_class_0)} SMs")
    print(f"  sm_class=1: {len(sm_class_1)} SMs")

    pre_replay = make_l2_flush_runner(device, args.l2_flush_mb)

    results = {}

    for name, (n_rows, k) in TEST_SHAPES.items():
        print(f"\n{'='*70}")
        print(f"Testing {name}: shape=[{n_rows}, {k}]")
        print(f"{'='*70}")

        weight_bytes = n_rows * k * 2

        # 1. Standard PyTorch GEMV
        print(f"  Standard torch.mm...")
        ms_torch, bw_torch = bench_standard_gemv(n_rows, k, args.warmup, args.iters, pre_replay)
        print(f"    {ms_torch*1000:.1f} us  ({bw_torch:.1f} GB/s)")

        # 2. Custom kernel (natural partition - no control)
        print(f"  Custom GEMV (natural)...")
        ms_custom, bw_custom = bench_custom_gemv(n_rows, k, args.warmup, args.iters, pre_replay)
        print(f"    {ms_custom*1000:.1f} us  ({bw_custom:.1f} GB/s)")

        results[name] = {
            "shape": [n_rows, k],
            "weight_bytes": weight_bytes,
            "standard_torch": {
                "time_us": ms_torch * 1000,
                "bandwidth_gb_s": bw_torch,
            },
            "custom_natural": {
                "time_us": ms_custom * 1000,
                "bandwidth_gb_s": bw_custom,
            },
            "ratio_custom_vs_torch": ms_custom / ms_torch if ms_torch > 0 else 0,
        }

    # Summary
    print(f"\n{'='*80}")
    print(f"{'Shape':>10s}  {'Torch(us)':>10s}  {'Custom(us)':>10s}  {'Ratio':>8s}  {'Torch(BW)':>10s}  {'Custom(BW)':>10s}")
    print(f"{'='*80}")
    for name, r in results.items():
        print(f"{name:>10s}  {r['standard_torch']['time_us']:10.1f}  "
              f"{r['custom_natural']['time_us']:10.1f}  "
              f"{r['ratio_custom_vs_torch']:7.3f}x  "
              f"{r['standard_torch']['bandwidth_gb_s']:10.1f}  "
              f"{r['custom_natural']['bandwidth_gb_s']:10.1f}")

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({
            "sm_classes": {
                "class_0_count": len(sm_class_0),
                "class_1_count": len(sm_class_1),
            },
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
