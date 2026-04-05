#!/usr/bin/env python3
"""Micro-benchmark: same-partition vs cross-partition HBM access bandwidth.

This test creates two access patterns:
  1. SAME: All SMs access pages where (sm_class XOR page_class XOR chunk_class) == 0
  2. CROSS: All SMs access pages where (sm_class XOR page_class XOR chunk_class) == 1

By keeping the total data access identical but changing the partition relationship,
we can isolate whether the ~14% latency difference translates to bandwidth difference.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("ENABLE_CUDAGRAPH", "0")

import torch

from benchmark_cuda_utils import (
    DEFAULT_BENCHMARK_L2_FLUSH_MB,
    make_l2_flush_runner,
    benchmark_style_cuda_reset,
)
from hbm_latency_partition import (
    probe_sm_classes,
    create_partition_buffer,
    pack_interleaved_weight,
    compute_interleaved_size,
)

import cuda_decode_gemv

WARMUP = 20
ITERS = 200
L2_FLUSH_MB = DEFAULT_BENCHMARK_L2_FLUSH_MB

# Test with different row counts to see parallelism effects
TEST_SIZES = [
    ("tiny", 64, 2048),
    ("small", 256, 2048),
    ("medium", 1024, 2048),
    ("large", 4096, 2048),
    ("xlarge", 8192, 2048),
]


def _bandwidth_gb_s(bytes_count: int, elapsed_ms: float) -> float:
    if elapsed_ms <= 0:
        return 0.0
    return bytes_count / (elapsed_ms / 1_000.0) / 1e9


def _bench_graph_cold(fn, warmup, iters, pre_replay):
    """Benchmark with CUDA graph replay and L2 flush."""
    for _ in range(warmup):
        pre_replay()
        fn()
    torch.cuda.synchronize()

    benchmark_style_cuda_reset()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

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


def main():
    parser = argparse.ArgumentParser(description="Same vs cross partition bandwidth test")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--l2-flush-mb", type=int, default=L2_FLUSH_MB)
    parser.add_argument("--output", type=str, default="results/partition_same_vs_cross.json")
    args = parser.parse_args()

    device = "cuda:0"
    props = torch.cuda.get_device_properties(device)
    print(f"Device: {props.name}, {props.multi_processor_count} SMs")

    # Probe SM classes
    print("\nProbing SM classes...")
    sm_result = probe_sm_classes(device=device)
    print(f"  sm_class=0: {len(sm_result.sm_class_0)} SMs")
    print(f"  sm_class=1: {len(sm_result.sm_class_1)} SMs")

    # Use the first test shape to create partition buffer
    _, first_n, first_k = TEST_SIZES[0]
    all_shapes = [(n, k) for _, n, k in TEST_SIZES]
    print(f"\nCreating partition buffer for {len(all_shapes)} weights...")
    pd = create_partition_buffer(all_shapes, device=device)

    sm_class_lo = pd["sm_class_lo"]
    sm_class_hi = pd["sm_class_hi"]
    _lo = sm_class_lo if sm_class_lo < (1 << 63) else sm_class_lo - (1 << 64)
    _hi = sm_class_hi if sm_class_hi < (1 << 63) else sm_class_hi - (1 << 64)

    pre_replay = make_l2_flush_runner(device, args.l2_flush_mb)
    ext = cuda_decode_gemv._load_ext()

    results = {}

    for idx, (name, n_rows, k) in enumerate(TEST_SIZES):
        print(f"\n{'='*70}")
        print(f"Test: {name}  shape=[{n_rows}, {k}]")
        print(f"{'='*70}")

        weight_bytes = n_rows * k * 2

        # Create random weight
        weight = torch.randn(n_rows, k, device=device, dtype=torch.float16)
        x = torch.randn(k, device=device, dtype=torch.float16)

        # Prepare partition buffer for this weight
        buf_size_halves = compute_interleaved_size(n_rows, k) // 2
        buf_offset_halves = pd["offsets"][idx] // 2
        buf_slice = pd["buffer"][buf_offset_halves: buf_offset_halves + buf_size_halves]
        chunk_classes = pd["chunk_classes_list"][idx].to(device)

        # Pack weight into interleaved buffer (dual copy)
        pack_interleaved_weight(weight, buf_slice, pd["chunk_classes_list"][idx])

        # Build chunk_class_bits for v2
        cc_cpu = pd["chunk_classes_list"][idx]
        chunk_class_bits = 0
        for ci in range(cc_cpu.numel()):
            if cc_cpu[ci].item() != 0:
                chunk_class_bits |= (1 << ci)
        _cc = chunk_class_bits if chunk_class_bits < (1 << 63) else chunk_class_bits - (1 << 64)

        # Pre-allocate outputs
        out_std = torch.empty(n_rows, device=device, dtype=torch.float16)
        out_part = torch.empty(n_rows, device=device, dtype=torch.float16)

        # Test 1: Standard GEMV (natural partition mix)
        def fn_std():
            ext.nosplit_matvec_half(x, weight, torch.empty(0, device=device, dtype=torch.float16),
                                   out_std, False)

        # Test 2: Partition-aware GEMV v2 (each SM reads its fast pages)
        def fn_part():
            ext.nosplit_matvec_half_partition_v2(
                x, buf_slice, torch.empty(0, device=device, dtype=torch.float16),
                out_part, _cc, _lo, _hi, n_rows, False)

        print(f"  Standard GEMV (natural)...")
        ms_std = _bench_graph_cold(fn_std, args.warmup, args.iters, pre_replay)
        bw_std = _bandwidth_gb_s(weight_bytes, ms_std)
        print(f"    {ms_std*1000:.1f} us  ({bw_std:.1f} GB/s)")

        print(f"  Partition GEMV (each SM reads fast pages)...")
        ms_part = _bench_graph_cold(fn_part, args.warmup, args.iters, pre_replay)
        # Note: partition reads same amount of data (single row per page)
        bw_part = _bandwidth_gb_s(weight_bytes, ms_part)
        print(f"    {ms_part*1000:.1f} us  ({bw_part:.1f} GB/s)")

        speedup = ms_std / ms_part if ms_part > 0 else 0

        print(f"  Speedup: {speedup:.3f}x  ({'+' if speedup > 1 else ''}{(speedup-1)*100:.1f}%)")

        results[name] = {
            "shape": [n_rows, k],
            "weight_bytes": weight_bytes,
            "standard": {
                "time_us": ms_std * 1000,
                "bandwidth_gb_s": bw_std,
            },
            "partition": {
                "time_us": ms_part * 1000,
                "bandwidth_gb_s": bw_part,
            },
            "speedup": speedup,
        }

    # Summary
    print(f"\n{'='*80}")
    print(f"{'Test':>10s}  {'Shape':>16s}  {'Std(us)':>8s}  {'Part(us)':>8s}  {'Speedup':>8s}  {'Std(BW)':>9s}  {'Part(BW)':>9s}")
    print(f"{'='*80}")
    for name, r in results.items():
        shape_str = f"[{r['shape'][0]},{r['shape'][1]}]"
        print(f"{name:>10s}  {shape_str:>16s}  {r['standard']['time_us']:8.1f}  "
              f"{r['partition']['time_us']:8.1f}  {r['speedup']:7.3f}x  "
              f"{r['standard']['bandwidth_gb_s']:9.1f}  {r['partition']['bandwidth_gb_s']:9.1f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
