#!/usr/bin/env python3
"""Profile internal phase timing for fused_gateup_silu_downproj_softsync on real payloads."""

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
import cuda_decode_gemv


def _compute_softsync_phase_durations_us(profile: torch.Tensor, slot_names: list[str]) -> dict[str, torch.Tensor]:
    """Compute per-block phase durations from softsync profile timestamps.

    profile: [grid_blocks, n_slots] int64 timestamps (ns)
    Returns durations per block in microseconds.
    """
    ts = profile.to(torch.int64).cpu()
    n_blocks = ts.shape[0]
    zeros = torch.zeros(n_blocks, dtype=torch.float64)

    phases = {}

    # x_load: kernel_start -> x_load_done
    mask = (ts[:, 1] > 0) & (ts[:, 0] > 0)
    x_load = zeros.clone()
    x_load[mask] = (ts[mask, 1] - ts[mask, 0]).to(torch.float64) / 1000.0
    phases["x_load_us"] = x_load

    # gate_up_compute: x_load_done -> gate_up_compute_done
    mask = (ts[:, 2] > 0) & (ts[:, 1] > 0)
    gate_up = zeros.clone()
    gate_up[mask] = (ts[mask, 2] - ts[mask, 1]).to(torch.float64) / 1000.0
    phases["gate_up_compute_us"] = gate_up

    # barrier: barrier_start -> barrier_done
    mask = (ts[:, 4] > 0) & (ts[:, 3] > 0)
    barrier = zeros.clone()
    barrier[mask] = (ts[mask, 4] - ts[mask, 3]).to(torch.float64) / 1000.0
    phases["barrier_us"] = barrier

    # hidden_load: barrier_done -> hidden_load_done
    mask = (ts[:, 5] > 0) & (ts[:, 4] > 0)
    hidden_load = zeros.clone()
    hidden_load[mask] = (ts[mask, 5] - ts[mask, 4]).to(torch.float64) / 1000.0
    phases["hidden_load_us"] = hidden_load

    # tile0 compute: hidden_load_done -> tile0_compute_done
    mask = (ts[:, 7] > 0) & (ts[:, 5] > 0)
    tile0 = zeros.clone()
    tile0[mask] = (ts[mask, 7] - ts[mask, 5]).to(torch.float64) / 1000.0
    phases["tile0_compute_us"] = tile0

    # tile1 compute: tile0_compute_done -> tile1_compute_done
    mask = (ts[:, 9] > 0) & (ts[:, 7] > 0)
    tile1 = zeros.clone()
    tile1[mask] = (ts[mask, 9] - ts[mask, 7]).to(torch.float64) / 1000.0
    phases["tile1_compute_us"] = tile1

    # tile2+ compute: tile1_compute_done -> kernel_done (remaining 4 tiles + epilogue)
    mask = (ts[:, 11] > 0) & (ts[:, 9] > 0)
    tile2_rest = zeros.clone()
    tile2_rest[mask] = (ts[mask, 11] - ts[mask, 9]).to(torch.float64) / 1000.0
    phases["tile2_rest_compute_us"] = tile2_rest

    # total: kernel_start -> kernel_done
    mask = (ts[:, 11] > 0) & (ts[:, 0] > 0)
    total = zeros.clone()
    total[mask] = (ts[mask, 11] - ts[mask, 0]).to(torch.float64) / 1000.0
    phases["total_us"] = total

    return phases


def _summarize_phases(phases: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    """Summarize phase durations across all blocks."""
    summary = {}
    for name, vals in phases.items():
        nonzero = vals[vals > 0]
        if len(nonzero) == 0:
            summary[name] = {"count": 0}
            continue
        summary[name] = {
            "count": int(len(nonzero)),
            "mean_us": float(nonzero.mean().item()),
            "median_us": float(nonzero.median().item()),
            "min_us": float(nonzero.min().item()),
            "max_us": float(nonzero.max().item()),
            "total_us": float(nonzero.sum().item()),
        }
    return summary


def run_softsync_profile(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Run the softsync profile and return (hidden_out, out, profile, meta)."""
    return cuda_decode_gemv.fused_gateup_silu_downproj_softsync_profile_half(
        x,
        gate_up_weight,
        down_proj_weight,
    )


def main():
    parser = argparse.ArgumentParser(description="Profile softsync fused MLP phases")
    parser.add_argument("--x-size", type=int, default=2048, help="Input hidden dim K")
    parser.add_argument("--gate-up-rows", type=int, default=12288, help="gate_up_weight N")
    parser.add_argument("--down-rows", type=int, default=2048, help="down_proj_weight N")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    device = torch.device("cuda:0")

    # Create inputs
    x = torch.randn((1, 1, args.x_size), dtype=torch.float16, device=device)
    gate_up_weight = torch.randn(
        (args.gate_up_rows, args.x_size), dtype=torch.float16, device=device
    )
    down_proj_weight = torch.randn(
        (args.down_rows, args.gate_up_rows // 2), dtype=torch.float16, device=device
    )  # [2048, 6144] = [2048, gate_up_rows // 2]

    # Warmup
    for _ in range(args.warmup):
        run_softsync_profile(x, gate_up_weight, down_proj_weight)
    torch.cuda.synchronize()

    # Profile
    hidden_out, out, profile, meta = run_softsync_profile(x, gate_up_weight, down_proj_weight)
    torch.cuda.synchronize()

    # Analyze
    phases = _compute_softsync_phase_durations_us(profile, meta["slot_names"])
    summary = _summarize_phases(phases)

    result = {
        "meta": meta,
        "phase_summary": summary,
        "validation": {
            "hidden_out_shape": list(hidden_out.shape),
            "out_shape": list(out.shape),
            "profile_shape": list(profile.shape),
            "profile_has_zeros": bool((profile == 0).any().item()),
        },
    }

    # Print summary
    print(f"\n=== Softsync Fused MLP Phase Timing (grid_blocks={meta['grid_blocks']}) ===")
    print(f"{'Phase':<30} {'Count':>6} {'Mean(us)':>10} {'Median(us)':>11} {'Min(us)':>9} {'Max(us)':>9}")
    print("-" * 80)
    for phase_name in [
        "x_load_us", "gate_up_compute_us", "barrier_us",
        "hidden_load_us", "tile0_compute_us", "tile1_compute_us",
        "tile2_rest_compute_us", "total_us"
    ]:
        if phase_name in summary:
            s = summary[phase_name]
            if s["count"] > 0:
                print(
                    f"{phase_name:<30} {s['count']:6d} {s['mean_us']:10.2f} "
                    f"{s['median_us']:11.2f} {s['min_us']:9.2f} {s['max_us']:9.2f}"
                )
            else:
                print(f"{phase_name:<30} {'':>6} {'':>10} {'':>11} {'':>9} {'':>9}")

    # Full per-block breakdown table
    n_blocks = profile.shape[0]
    print(f"\n=== Per-block Phase Timing (all {n_blocks} blocks) ===")
    print(f"{'Block':>5} {'x_load':>9} {'gate_up':>9} {'barrier':>9} "
          f"{'hidden_ld':>9} {'tile0':>9} {'tile1':>9} {'tile2+':>9} {'total':>9}")
    print("-" * 80)
    for b in range(n_blocks):
        print(
            f"{b:5d} "
            f"{phases['x_load_us'][b]:9.2f} "
            f"{phases['gate_up_compute_us'][b]:9.2f} "
            f"{phases['barrier_us'][b]:9.2f} "
            f"{phases['hidden_load_us'][b]:9.2f} "
            f"{phases['tile0_compute_us'][b]:9.2f} "
            f"{phases['tile1_compute_us'][b]:9.2f} "
            f"{phases['tile2_rest_compute_us'][b]:9.2f} "
            f"{phases['total_us'][b]:9.2f}"
        )

    # ASCII heatmap: show relative timing across blocks for each phase
    print(f"\n=== ASCII Heatmap (each block = one char, '#' = slow, '.' = fast) ===")
    phases_heatmap = [
        ("gate_up_compute_us", "gate_up"),
        ("barrier_us", "barrier"),
        ("tile0_compute_us", "tile0"),
        ("tile1_compute_us", "tile1"),
        ("tile2_rest_compute_us", "tile2+"),
        ("total_us", "total"),
    ]
    for phase_key, label in phases_heatmap:
        vals = phases[phase_key].numpy()
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax - vmin < 1e-6:
            chars = ["." if v > 0 else " " for v in vals]
        else:
            chars = []
            for v in vals:
                if v <= 0:
                    chars.append(" ")
                else:
                    norm = (v - vmin) / (vmax - vmin)
                    idx = int(norm * 5)
                    chars.append(" .oO#@"[min(idx, 5)])
        print(f"  {label:<12} | {''.join(chars)} |")

    # Per-phase distribution analysis: show block ranges that correspond to SM scheduling
    print(f"\n=== Block Timing by 32-block groups (8 SMs × 32 blocks/SM ≈ scheduling pattern) ===")
    group_size = 32
    for g_start in range(0, n_blocks, group_size):
        g_end = min(g_start + group_size, n_blocks)
        print(f"  Blocks [{g_start:3d}-{g_end-1:3d}] (count={g_end-g_start}):")
        for phase_name in ["x_load_us", "gate_up_compute_us", "barrier_us", "hidden_load_us",
                          "tile0_compute_us", "tile1_compute_us", "tile2_rest_compute_us", "total_us"]:
            vals = phases[phase_name][g_start:g_end]
            nonzero = vals[vals > 0]
            if len(nonzero) == 0:
                continue
            print(f"    {phase_name:<28} mean={nonzero.mean():7.2f}us  median={nonzero.median():7.2f}us  "
                  f"min={nonzero.min():7.2f}us  max={nonzero.max():7.2f}us  stddev={nonzero.std():7.2f}us")

    # Save output
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert tensors to lists for JSON
        json_result = {
            "meta": meta,
            "phase_summary": summary,
            "validation": result["validation"],
        }
        with open(out_path, "w") as f:
            json.dump(json_result, f, indent=2, default=str)
        print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
