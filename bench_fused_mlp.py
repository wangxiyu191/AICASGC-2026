#!/usr/bin/env python3
"""Benchmark fused MLP (gateup+silu+downproj) variants vs 3 separate ops.

Compares:
  - 3 separate ops: gateup GEMV + torch.silu + downproj GEMV
  - fused softsync: single kernel with software barrier
  - fused gridsync: single kernel with cooperative_groups grid sync

Uses cold-L2 CUDAGraph replay consistent with bench_gemv_bandwidth.py.
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

from benchmark_cuda_utils import (
    DEFAULT_BENCHMARK_L2_FLUSH_MB,
    benchmark_style_cuda_reset,
    make_l2_flush_runner,
)
import cuda_decode_gemv


# ---------------------------------------------------------------------------
# Variant runners - each returns a callable(x_vec, gate_up_w, down_proj_w)
# ---------------------------------------------------------------------------

def _make_3sep_runner():
    """3 separate kernels: gateup GEMV + silu*up + downproj GEMV."""
    ws = {}

    def _run(x_vec, gate_up_w, down_proj_w):
        gu = cuda_decode_gemv.nosplit_matvec_half(
            x_vec, gate_up_w, None, workspace=ws,
        )
        gu_flat = gu.view(-1)
        gate = gu_flat[:6144]
        up = gu_flat[6144:]
        hidden = F.silu(gate) * up
        out = cuda_decode_gemv.nosplit_matvec_half(
            hidden.unsqueeze(0), down_proj_w, None,
        )
        return out

    return _run


def _make_softsync_runner():
    """Fused softsync kernel."""
    ws = {}

    def _run(x_vec, gate_up_w, down_proj_w):
        _, out = cuda_decode_gemv.fused_gateup_silu_downproj_softsync_half(
            x_vec, gate_up_w, down_proj_w, workspace=ws,
        )
        return out

    return _run


def _make_gridsync_runner():
    """Fused gridsync kernel (cooperative_groups)."""
    ws = {}

    def _run(x_vec, gate_up_w, down_proj_w):
        _, out = cuda_decode_gemv.fused_gateup_silu_downproj_gridsync_half(
            x_vec, gate_up_w, down_proj_w, workspace=ws,
        )
        return out

    return _run


def _make_persistent_runner():
    """Fused persistent kernel (1 block/SM, cooperative_groups)."""
    ws = {}

    def _run(x_vec, gate_up_w, down_proj_w):
        _, out = cuda_decode_gemv.fused_gateup_silu_downproj_persistent_half(
            x_vec, gate_up_w, down_proj_w, workspace=ws,
        )
        return out

    return _run


# ---------------------------------------------------------------------------
# Benchmark helper (adapted from _bench_impl_graph in bench_decode_linears.py)
# ---------------------------------------------------------------------------

def _bench_fused_graph(
    fn,
    x_vec: torch.Tensor,
    gate_up_w: torch.Tensor,
    down_proj_w: torch.Tensor,
    warmup: int,
    iters: int,
    capture_repeats: int,
    pre_replay=None,
):
    """Benchmark using CUDAGraph replay and GPU-side timing."""
    # Eager warmup
    for _ in range(warmup):
        fn(x_vec, gate_up_w, down_proj_w)
    torch.cuda.synchronize()

    # Capture CUDAGraph
    benchmark_style_cuda_reset()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(capture_repeats):
                out = fn(x_vec, gate_up_w, down_proj_w)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # Warmup replays
    benchmark_style_cuda_reset()
    for _ in range(warmup):
        if pre_replay is not None:
            pre_replay()
        graph.replay()
    torch.cuda.synchronize()

    # Timed replays
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
        total_elapsed_ms = sum(
            se.elapsed_time(ee) for se, ee in zip(start_events, end_events)
        )
        elapsed_ms = total_elapsed_ms / (iters * capture_repeats)

    return out, elapsed_ms


def _bandwidth_gb_s(gate_up_w: torch.Tensor, down_proj_w: torch.Tensor, elapsed_ms: float) -> float:
    """Effective weight bandwidth: total weight bytes / elapsed time."""
    total_bytes = (gate_up_w.numel() + down_proj_w.numel()) * gate_up_w.element_size()
    return total_bytes / (elapsed_ms / 1_000.0) / 1e9


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark fused MLP variants")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--capture-repeats", type=int, default=1)
    parser.add_argument("--l2-flush-mb", type=int, default=DEFAULT_BENCHMARK_L2_FLUSH_MB)
    parser.add_argument("--output", type=str, default="results/bench_fused_mlp.json")
    parser.add_argument(
        "--use-real-weights",
        action="store_true",
        default=False,
        help="Load real model weights via _run_capture (requires model + dataset)",
    )
    parser.add_argument("--model-path", type=str, default="./Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", type=str, default="./data")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--layer-idx", type=int, default=14)
    parser.add_argument("--decode-step", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    if args.l2_flush_mb > 0 and args.capture_repeats != 1:
        print(
            f"[bench_fused_mlp] l2 flush enabled ({args.l2_flush_mb} MiB); "
            f"forcing capture_repeats=1 for cold-L2 measurement."
        )
        args.capture_repeats = 1

    device = torch.device("cuda:0")

    # Prepare inputs
    if args.use_real_weights:
        from datasets import load_from_disk
        from bench_decode_attention import _prepare_inputs
        from bench_decode_linears import _run_capture
        from evaluation_wrapper import VLMModel

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
        x_vec = payloads["gate_up"]["x"]
        gate_up_w = payloads["gate_up"]["weight"]
        down_proj_w = payloads["down_proj"]["weight"]
        print(f"Loaded real weights: x={list(x_vec.shape)} gate_up={list(gate_up_w.shape)} down_proj={list(down_proj_w.shape)}")
    else:
        torch.manual_seed(42)
        x_vec = torch.randn(1, 1, 2048, dtype=torch.float16, device=device) * 0.01
        gate_up_w = torch.randn(12288, 2048, dtype=torch.float16, device=device) * 0.01
        down_proj_w = torch.randn(2048, 6144, dtype=torch.float16, device=device) * 0.01
        print(f"Using random weights: x={list(x_vec.shape)} gate_up={list(gate_up_w.shape)} down_proj={list(down_proj_w.shape)}")

    pre_replay = make_l2_flush_runner(device, args.l2_flush_mb)

    variants = {
        "3sep_ops": _make_3sep_runner(),
        "fused_softsync": _make_softsync_runner(),
        "fused_gridsync": _make_gridsync_runner(),
        "fused_persistent": _make_persistent_runner(),
    }

    results = {}
    for name, fn in variants.items():
        print(f"Benchmarking {name} ...", end="", flush=True)
        try:
            out, elapsed_ms = _bench_fused_graph(
                fn, x_vec, gate_up_w, down_proj_w,
                warmup=args.warmup,
                iters=args.iters,
                capture_repeats=args.capture_repeats,
                pre_replay=pre_replay,
            )
            bw = _bandwidth_gb_s(gate_up_w, down_proj_w, elapsed_ms)
            results[name] = {
                "elapsed_ms": elapsed_ms,
                "elapsed_us": elapsed_ms * 1000,
                "effective_weight_bandwidth_gb_s": bw,
            }
            print(f" {elapsed_ms * 1000:.1f} us  ({bw:.1f} GB/s)")
        except Exception as e:
            results[name] = {"error": str(e)}
            print(f" FAILED: {e}")

    # Derive comparisons
    baseline_key = "3sep_ops"
    if baseline_key in results and "elapsed_ms" in results[baseline_key]:
        baseline_us = results[baseline_key]["elapsed_us"]
        for name, r in results.items():
            if "elapsed_us" in r:
                r["vs_3sep_pct"] = (r["elapsed_us"] / baseline_us - 1) * 100

    # Print summary
    print()
    print("=" * 72)
    print(f"{'Variant':<25} {'Latency':>10} {'BW (GB/s)':>12} {'vs 3sep':>10}")
    print("-" * 72)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<25} {'ERROR':>10}")
            continue
        delta_str = "baseline" if name == baseline_key else f"{r.get('vs_3sep_pct', 0):+.1f}%"
        print(
            f"{name:<25} {r['elapsed_us']:>9.1f}us "
            f"{r['effective_weight_bandwidth_gb_s']:>11.1f} "
            f"{delta_str:>10}"
        )

    # Verdict
    if all("elapsed_us" in results.get(k, {}) for k in ["3sep_ops", "fused_gridsync"]):
        sep = results["3sep_ops"]["elapsed_us"]
        gs = results["fused_gridsync"]["elapsed_us"]
        diff = sep - gs
        print()
        if diff > 0:
            print(f"VERDICT: fused gridsync saves {diff:.1f} us ({diff/sep*100:.1f}%) vs 3 separate ops under CUDA graph")
        else:
            print(f"VERDICT: 3 separate ops is {-diff:.1f} us ({-diff/sep*100:.1f}%) faster than fused gridsync under CUDA graph")

    # Gridsync profile breakdown
    print()
    print("=== Gridsync Internal Phase Profile ===")
    _, _, prof, meta = cuda_decode_gemv.fused_gateup_silu_downproj_gridsync_profile_half(
        x_vec, gate_up_w, down_proj_w,
    )
    slot_names = meta["slot_names"]
    prof_cpu = prof.cpu()

    def _phase_stats(start_slot, end_slot):
        si = slot_names.index(start_slot)
        ei = slot_names.index(end_slot)
        vals = (prof_cpu[:, ei] - prof_cpu[:, si]).float()
        valid = vals[vals > 0]
        if len(valid) == 0:
            return {"mean": 0, "min": 0, "max": 0}
        return {
            "mean": valid.mean().item() / 1000,
            "min": valid.min().item() / 1000,
            "max": valid.max().item() / 1000,
        }

    phases = [
        ("x_load", "kernel_start", "x_load_done"),
        ("gate_up_compute", "x_load_done", "gate_up_compute_done"),
        ("barrier", "barrier_start", "barrier_done"),
        ("hidden_load", "barrier_done", "hidden_load_done"),
        ("down_proj", "hidden_load_done", "kernel_done"),
        ("total", "kernel_start", "kernel_done"),
    ]
    print(f"{'Phase':<20} {'Mean(us)':>10} {'Min(us)':>10} {'Max(us)':>10}")
    print("-" * 52)
    for label, s_slot, e_slot in phases:
        st = _phase_stats(s_slot, e_slot)
        print(f"{label:<20} {st['mean']:>10.2f} {st['min']:>10.2f} {st['max']:>10.2f}")

    # Save output
    output = {
        "meta": {
            "x_shape": list(x_vec.shape),
            "gate_up_shape": list(gate_up_w.shape),
            "down_proj_shape": list(down_proj_w.shape),
            "warmup": args.warmup,
            "iters": args.iters,
            "capture_repeats": args.capture_repeats,
            "l2_flush_mb": args.l2_flush_mb,
            "use_real_weights": args.use_real_weights,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
        },
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
