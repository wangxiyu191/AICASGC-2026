#!/usr/bin/env python3
"""Micro-benchmark: partition GEMV vs standard row2 GEMV.

Compares cold-L2 (HBM-bound) latency for k=2048 GEMV operators:
  1. Standard nosplit row2 GEMV (reading from original weight)
  2. Partition-aware GEMV v1 (global read for chunk_classes + LUT)
  3. Partition-aware GEMV v2 (zero extra memory reads, pure register)

Uses CUDAGraph replay + L2 flush for accurate HBM-bottleneck measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("ENABLE_CUDAGRAPH", "0")
os.environ.setdefault("CUDAGRAPH_PREBUILD_MAX_CACHE_LEN", "0")

import torch

from benchmark_cuda_utils import (
    DEFAULT_BENCHMARK_L2_FLUSH_MB,
    make_l2_flush_runner,
    benchmark_style_cuda_reset,
)
import cuda_decode_gemv
from hbm_latency_partition import (
    compute_interleaved_size,
    create_partition_buffer,
    pack_interleaved_weight,
)

# ---- Shapes from Qwen3-VL-2B-Instruct (k=2048 targets) ----
WEIGHT_SHAPES = {
    "qkv":     (4096, 2048),
    "o_proj":   (2048, 2048),
    "gate_up": (12288, 2048),
}

WARMUP = 20
ITERS = 300
L2_FLUSH_MB = DEFAULT_BENCHMARK_L2_FLUSH_MB


def _bandwidth_gb_s(weight_bytes: int, elapsed_ms: float) -> float:
    return weight_bytes / (elapsed_ms / 1_000.0) / 1e9


def _bench_graph_cold(fn, warmup, iters, pre_replay):
    """CUDAGraph replay benchmark with L2 flush between each iteration."""
    for _ in range(warmup):
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

    # Warmup replays
    benchmark_style_cuda_reset()
    for _ in range(warmup):
        pre_replay()
        graph.replay()
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


def _bench_graph_hot(fn, warmup, iters):
    """CUDAGraph replay benchmark WITHOUT L2 flush (hot cache)."""
    for _ in range(warmup):
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
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        graph.replay()
    en.record()
    en.synchronize()
    return st.elapsed_time(en) / iters


def main():
    parser = argparse.ArgumentParser(description="Partition GEMV micro-benchmark")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--l2-flush-mb", type=int, default=L2_FLUSH_MB)
    parser.add_argument("--target", type=str, default="all",
                        choices=["all"] + list(WEIGHT_SHAPES.keys()))
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = "cuda:0"
    targets = list(WEIGHT_SHAPES.keys()) if args.target == "all" else [args.target]

    # Build partition buffer for ALL targets at once (same as model init)
    all_shapes = [WEIGHT_SHAPES[t] for t in targets]
    print(f"Allocating partition buffer for {len(all_shapes)} weights...")
    pd = create_partition_buffer(all_shapes, device=device)

    # Use probed SM classification from create_partition_buffer().
    # SM classification is hardware-specific and must be probed at runtime.
    sm_class_lo = pd["sm_class_lo"]
    sm_class_hi = pd["sm_class_hi"]
    print(f"SM class bitfields: lo=0x{sm_class_lo:016x} hi=0x{sm_class_hi:016x}")

    # Convert to signed for pybind11
    _lo = sm_class_lo if sm_class_lo < (1 << 63) else sm_class_lo - (1 << 64)
    _hi = sm_class_hi if sm_class_hi < (1 << 63) else sm_class_hi - (1 << 64)

    pre_replay = make_l2_flush_runner(device, args.l2_flush_mb)
    results_all = {}

    for idx, target in enumerate(targets):
        n_rows, k = WEIGHT_SHAPES[target]
        weight_bytes = n_rows * k * 2  # fp16

        # Random weight (standard layout)
        weight = torch.randn(n_rows, k, device=device, dtype=torch.float16)
        x = torch.randn(1, 1, k, device=device, dtype=torch.float16)

        # Pack into interleaved partition buffer
        buf_size_halves = compute_interleaved_size(n_rows, k) // 2
        buf_offset_halves = pd["offsets"][idx] // 2
        buf_slice = pd["buffer"][buf_offset_halves: buf_offset_halves + buf_size_halves]
        chunk_classes = pd["chunk_classes_list"][idx].to(device)
        pack_interleaved_weight(weight, buf_slice, pd["chunk_classes_list"][idx])

        # Build chunk_class_bits (uint64 bitfield) for v2 kernel
        cc_cpu = pd["chunk_classes_list"][idx]
        chunk_class_bits = 0
        for ci in range(cc_cpu.numel()):
            if cc_cpu[ci].item() != 0:
                chunk_class_bits |= (1 << ci)
        # Convert to signed for pybind11
        _cc = chunk_class_bits if chunk_class_bits < (1 << 63) else chunk_class_bits - (1 << 64)

        # Pre-allocate outputs
        out_std = torch.empty(n_rows, device=device, dtype=torch.float16)
        out_part = torch.empty(n_rows, device=device, dtype=torch.float16)
        out_part_v2 = torch.empty(n_rows, device=device, dtype=torch.float16)
        x_vec = x.reshape(-1).contiguous()
        empty_bias = torch.empty(0, device=device, dtype=torch.float16)

        ext = cuda_decode_gemv._load_ext()

        def _fn_standard(x_vec=x_vec, weight=weight, out=out_std, ext=ext,
                         eb=empty_bias):
            ext.nosplit_matvec_half(x_vec, weight, eb, out, False)

        def _fn_partition(x_vec=x_vec, buf=buf_slice, out=out_part, ext=ext,
                          cc=chunk_classes, lo=_lo, hi=_hi, nr=n_rows,
                          eb=empty_bias):
            ext.nosplit_matvec_half_partition(
                x_vec, buf, eb, out, cc, lo, hi, nr, False)

        def _fn_partition_v2(x_vec=x_vec, buf=buf_slice, out=out_part_v2, ext=ext,
                             ccb=_cc, lo=_lo, hi=_hi, nr=n_rows,
                             eb=empty_bias):
            ext.nosplit_matvec_half_partition_v2(
                x_vec, buf, eb, out, ccb, lo, hi, nr, False)

        print(f"\n{'='*60}")
        print(f"Target: {target}  weight=[{n_rows}, {k}]  ({weight_bytes/1024:.0f} KB)")
        print(f"{'='*60}")

        # ---- Cold L2 (HBM-bound, the metric we care about) ----
        ms_std_cold = _bench_graph_cold(_fn_standard, args.warmup, args.iters, pre_replay)
        ms_part_cold = _bench_graph_cold(_fn_partition, args.warmup, args.iters, pre_replay)
        ms_part_v2_cold = _bench_graph_cold(_fn_partition_v2, args.warmup, args.iters, pre_replay)

        bw_std_cold = _bandwidth_gb_s(weight_bytes, ms_std_cold)
        bw_part_cold = _bandwidth_gb_s(weight_bytes, ms_part_cold)
        bw_part_v2_cold = _bandwidth_gb_s(weight_bytes, ms_part_v2_cold)

        speedup_cold = ms_std_cold / ms_part_cold
        speedup_v2_cold = ms_std_cold / ms_part_v2_cold

        print(f"  Cold L2 (HBM-bound):")
        print(f"    Standard row2:    {ms_std_cold*1000:.1f} us  "
              f"({bw_std_cold:.1f} GB/s effective)")
        print(f"    Partition v1:     {ms_part_cold*1000:.1f} us  "
              f"({bw_part_cold:.1f} GB/s effective)")
        print(f"      v1 speedup: {speedup_cold:.3f}x  "
              f"({'FASTER' if speedup_cold > 1.0 else 'SLOWER'} by {abs(speedup_cold-1)*100:.1f}%)")
        print(f"    Partition v2:     {ms_part_v2_cold*1000:.1f} us  "
              f"({bw_part_v2_cold:.1f} GB/s effective)")
        print(f"      v2 speedup: {speedup_v2_cold:.3f}x  "
              f"({'FASTER' if speedup_v2_cold > 1.0 else 'SLOWER'} by {abs(speedup_v2_cold-1)*100:.1f}%)")

        # ---- Hot L2 (compute-bound baseline) ----
        ms_std_hot = _bench_graph_hot(_fn_standard, args.warmup, args.iters)
        ms_part_hot = _bench_graph_hot(_fn_partition, args.warmup, args.iters)
        ms_part_v2_hot = _bench_graph_hot(_fn_partition_v2, args.warmup, args.iters)

        print(f"  Hot L2 (compute-bound):")
        print(f"    Standard row2:    {ms_std_hot*1000:.1f} us")
        print(f"    Partition v1:     {ms_part_hot*1000:.1f} us")
        print(f"    Partition v2:     {ms_part_v2_hot*1000:.1f} us")

        results_all[target] = {
            "shape": [n_rows, k],
            "weight_bytes": weight_bytes,
            "cold_standard_us": ms_std_cold * 1000,
            "cold_partition_v1_us": ms_part_cold * 1000,
            "cold_partition_v2_us": ms_part_v2_cold * 1000,
            "cold_standard_bw_gbs": bw_std_cold,
            "cold_partition_v1_bw_gbs": bw_part_cold,
            "cold_partition_v2_bw_gbs": bw_part_v2_cold,
            "cold_speedup_v1": speedup_cold,
            "cold_speedup_v2": speedup_v2_cold,
            "hot_standard_us": ms_std_hot * 1000,
            "hot_partition_v1_us": ms_part_hot * 1000,
            "hot_partition_v2_us": ms_part_v2_hot * 1000,
        }

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Target':>10s}  {'Shape':>14s}  {'Std(us)':>8s}  {'v1(us)':>8s}  {'v2(us)':>8s}  {'v1 spd':>7s}  {'v2 spd':>7s}")
    print(f"{'='*80}")
    for t, r in results_all.items():
        shape_str = f"[{r['shape'][0]},{r['shape'][1]}]"
        print(f"{t:>10s}  {shape_str:>14s}  {r['cold_standard_us']:8.1f}  {r['cold_partition_v1_us']:8.1f}  "
              f"{r['cold_partition_v2_us']:8.1f}  {r['cold_speedup_v1']:6.3f}x  {r['cold_speedup_v2']:6.3f}x")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(results_all, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
