#!/usr/bin/env python3
"""Profile per-iteration phase timing for the persistent fused gateup+silu+downproj kernel.

Measures effective HBM bandwidth for each phase:
  - gate_up iter0 (all SMs active)
  - gate_up iter1 (subset of SMs active)
  - grid sync
  - down_proj hidden load
  - down_proj compute (subset of SMs active)
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
import cuda_decode_gemv

# A800 80GB PCIe peak HBM bandwidth
PEAK_BW_GBS = 2039.0


def _load_real_weights(
    model_path: str, layer_idx: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Try to load gate_up_proj and down_proj weights from a real model."""
    try:
        from safetensors import safe_open

        idx_path = Path(model_path) / "model.safetensors.index.json"
        if not idx_path.exists():
            return None
        with open(idx_path) as f:
            index = json.load(f)
        wm = index.get("weight_map", {})
        gu_key = f"model.layers.{layer_idx}.mlp.gate_up_proj.weight"
        dp_key = f"model.layers.{layer_idx}.mlp.down_proj.weight"
        gu_file = wm.get(gu_key)
        dp_file = wm.get(dp_key)
        if gu_file is None or dp_file is None:
            return None
        gu_path = Path(model_path) / gu_file
        dp_path = Path(model_path) / dp_file
        with safe_open(str(gu_path), framework="pt", device=str(device)) as f:
            gate_up_w = f.get_tensor(gu_key).to(torch.float16).contiguous()
        with safe_open(str(dp_path), framework="pt", device=str(device)) as f:
            down_proj_w = f.get_tensor(dp_key).to(torch.float16).contiguous()
        return gate_up_w, down_proj_w
    except Exception as e:
        print(f"Warning: failed to load real weights: {e}")
        return None


def _extract_phases(profile: torch.Tensor, num_sms: int) -> dict:
    """Extract per-block phase durations from profile tensor [num_sms, 9]."""
    ts = profile.to(torch.int64).cpu().to(torch.float64)  # nanoseconds
    n = ts.shape[0]

    phases = {}

    # x_load: slot 0 -> slot 1
    phases["x_load"] = {
        "dur_ns": ts[:, 1] - ts[:, 0],
        "blocks": list(range(n)),
        "label": "x_load",
    }

    # gate_up iter0: slot 1 -> slot 2 (all blocks)
    phases["gateup_iter0"] = {
        "dur_ns": ts[:, 2] - ts[:, 1],
        "blocks": list(range(n)),
        "label": "gate_up iter0",
    }

    # gate_up iter1: slot 2 -> slot 3 (only blocks with slot 3 > 0)
    iter1_mask = ts[:, 3] > 0
    iter1_blocks = iter1_mask.nonzero(as_tuple=True)[0].tolist()
    iter1_dur = torch.zeros(n, dtype=torch.float64)
    if iter1_blocks:
        iter1_dur[iter1_mask] = ts[iter1_mask, 3] - ts[iter1_mask, 2]
    phases["gateup_iter1"] = {
        "dur_ns": iter1_dur,
        "blocks": iter1_blocks,
        "label": "gate_up iter1",
    }

    # grid sync: slot 4 -> slot 5
    phases["gridsync"] = {
        "dur_ns": ts[:, 5] - ts[:, 4],
        "blocks": list(range(n)),
        "label": "grid_sync",
    }

    # down hidden load: slot 5 -> slot 6
    phases["down_hidden_load"] = {
        "dur_ns": ts[:, 6] - ts[:, 5],
        "blocks": list(range(n)),
        "label": "down hidden load",
    }

    # down_proj compute: slot 6 -> slot 7
    # blocks 0..63 are active (row < 2048), blocks 64+ idle
    down_active_limit = (2048 + 31) // 32  # = 64
    down_active_blocks = list(range(min(down_active_limit, n)))
    down_idle_blocks = list(range(down_active_limit, n))
    phases["down_compute"] = {
        "dur_ns": ts[:, 7] - ts[:, 6],
        "blocks": down_active_blocks,
        "label": "down_proj compute",
    }
    phases["down_compute_idle"] = {
        "dur_ns": ts[:, 7] - ts[:, 6],
        "blocks": down_idle_blocks,
        "label": "down_proj (idle SMs)",
    }

    # total: slot 0 -> slot 8
    phases["total"] = {
        "dur_ns": ts[:, 8] - ts[:, 0],
        "blocks": list(range(n)),
        "label": "total kernel",
    }

    return phases


def _compute_bandwidth(
    phases: dict,
    num_sms: int,
) -> list[dict]:
    """Compute wall-time bandwidth for each phase."""
    # Data volume per phase (bytes)
    warps_per_block = 32
    stride = num_sms * warps_per_block  # 108 * 32 = 3456

    iter0_rows = min(stride, 6144)
    iter1_rows = max(0, 6144 - iter0_rows)

    data_bytes = {
        "x_load": 2048 * 2,  # x[2048] fp16
        "gateup_iter0": iter0_rows * 2 * 2048 * 2,  # gate + up, K=2048, fp16
        "gateup_iter1": iter1_rows * 2 * 2048 * 2 if iter1_rows > 0 else 0,
        "gridsync": 0,
        "down_hidden_load": 6144 * 2,  # hidden[6144] fp16
        "down_compute": 2048 * 6144 * 2,  # weight[2048, 6144] fp16
        "down_compute_idle": 0,
        "total": 12288 * 2048 * 2 + 2048 * 6144 * 2,  # gate_up + down_proj weights
    }

    results = []
    for key in ["x_load", "gateup_iter0", "gateup_iter1", "gridsync",
                "down_hidden_load", "down_compute", "total"]:
        phase = phases[key]
        blocks = phase["blocks"]
        if not blocks:
            continue
        dur = phase["dur_ns"]
        active_dur = dur[blocks]

        # Wall time: max end - min start across participating blocks
        # Since timestamps are absolute, wall time = max(dur) for per-block durations
        # (all blocks start roughly simultaneously after sync points)
        wall_ns = float(active_dur.max())
        median_ns = float(active_dur.median())
        p95_ns = float(active_dur.quantile(0.95)) if len(blocks) > 1 else wall_ns
        mean_ns = float(active_dur.mean())

        db = data_bytes.get(key, 0)
        wall_us = wall_ns / 1000.0
        median_us = median_ns / 1000.0
        p95_us = p95_ns / 1000.0

        bw_gbs = (db / wall_ns) if wall_ns > 0 and db > 0 else 0.0
        pct_peak = (bw_gbs / PEAK_BW_GBS * 100) if bw_gbs > 0 else 0.0

        results.append({
            "phase": phase["label"],
            "active_sms": len(blocks),
            "data_mb": db / 1e6,
            "wall_us": wall_us,
            "median_us": median_us,
            "p95_us": p95_us,
            "bw_gbs": bw_gbs,
            "pct_peak": pct_peak,
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Profile persistent fused MLP kernel phases and compute effective bandwidth"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./Qwen3-VL-2B-Instruct",
        help="Path to model for real weights",
    )
    parser.add_argument("--layer-idx", type=int, default=0, help="MLP layer index")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=20, help="Profile iterations")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    print(f"Device: {torch.cuda.get_device_name(device)}, SMs: {num_sms}")

    # Load weights
    weights = _load_real_weights(args.model_path, args.layer_idx, device)
    if weights is not None:
        gate_up_w, down_proj_w = weights
        print(f"Loaded real weights from layer {args.layer_idx}: "
              f"gate_up={list(gate_up_w.shape)}, down_proj={list(down_proj_w.shape)}")
    else:
        print("Using random weights")
        gate_up_w = torch.randn((12288, 2048), dtype=torch.float16, device=device)
        down_proj_w = torch.randn((2048, 6144), dtype=torch.float16, device=device)

    x = torch.randn((1, 1, 2048), dtype=torch.float16, device=device)

    # Warmup
    print(f"Warmup ({args.warmup} iters)...")
    for _ in range(args.warmup):
        cuda_decode_gemv.fused_gateup_silu_downproj_persistent_half(
            x, gate_up_w, down_proj_w
        )
    torch.cuda.synchronize()

    # Profile runs
    print(f"Profiling ({args.iters} runs)...")
    all_results = []
    all_profiles = []
    for i in range(args.iters):
        _, _, profile, meta = cuda_decode_gemv.fused_gateup_silu_downproj_persistent_profile_half(
            x, gate_up_w, down_proj_w
        )
        torch.cuda.synchronize()
        phases = _extract_phases(profile, num_sms)
        bw_results = _compute_bandwidth(phases, num_sms)
        all_results.append(bw_results)
        all_profiles.append(profile.cpu())

    # Aggregate across runs (median of each metric)
    phase_names = [r["phase"] for r in all_results[0]]
    aggregated = []
    for pi, pname in enumerate(phase_names):
        vals = {k: [] for k in ["wall_us", "median_us", "p95_us", "bw_gbs", "pct_peak"]}
        for run in all_results:
            for k in vals:
                vals[k].append(run[pi][k])
        agg = {
            "phase": pname,
            "active_sms": all_results[0][pi]["active_sms"],
            "data_mb": all_results[0][pi]["data_mb"],
        }
        for k in vals:
            sorted_v = sorted(vals[k])
            mid = len(sorted_v) // 2
            agg[k] = sorted_v[mid]  # median across runs
        aggregated.append(agg)

    # Print results
    print(f"\n{'='*95}")
    print(f"Persistent Kernel Phase Bandwidth Analysis ({args.iters} runs, median)")
    print(f"{'='*95}")
    hdr = (f"{'Phase':<22} {'SMs':>4} {'Data(MB)':>9} {'Wall(us)':>9} "
           f"{'BW(GB/s)':>9} {'%Peak':>6} {'Med(us)':>9} {'P95(us)':>9}")
    print(hdr)
    print("-" * 95)
    for r in aggregated:
        bw_str = f"{r['bw_gbs']:9.1f}" if r["bw_gbs"] > 0 else f"{'--':>9}"
        pct_str = f"{r['pct_peak']:5.1f}%" if r["pct_peak"] > 0 else f"{'--':>6}"
        print(
            f"{r['phase']:<22} {r['active_sms']:4d} {r['data_mb']:9.2f} "
            f"{r['wall_us']:9.2f} {bw_str} {pct_str} "
            f"{r['median_us']:9.2f} {r['p95_us']:9.2f}"
        )
    print(f"{'='*95}")

    # Per-block detail for a single representative run (last run)
    last_profile = all_profiles[-1]
    last_phases = _extract_phases(last_profile.to(device), num_sms)
    print(f"\n--- Per-block detail (last run) ---")
    print(f"{'Block':>5} {'iter0(us)':>10} {'iter1(us)':>10} {'sync(us)':>10} "
          f"{'down_ld(us)':>11} {'down_cp(us)':>11} {'total(us)':>10}")
    print("-" * 72)
    for b in range(num_sms):
        iter0_us = last_phases["gateup_iter0"]["dur_ns"][b].item() / 1000
        iter1_us = last_phases["gateup_iter1"]["dur_ns"][b].item() / 1000
        sync_us = last_phases["gridsync"]["dur_ns"][b].item() / 1000
        dld_us = last_phases["down_hidden_load"]["dur_ns"][b].item() / 1000
        dcp_us = last_phases["down_compute"]["dur_ns"][b].item() / 1000
        tot_us = last_phases["total"]["dur_ns"][b].item() / 1000
        iter1_str = f"{iter1_us:10.2f}" if iter1_us > 0 else f"{'--':>10}"
        print(
            f"{b:5d} {iter0_us:10.2f} {iter1_str} {sync_us:10.2f} "
            f"{dld_us:11.2f} {dcp_us:11.2f} {tot_us:10.2f}"
        )

    # Summary: iter1 blocks vs no-iter1 blocks
    iter1_blocks = last_phases["gateup_iter1"]["blocks"]
    no_iter1_blocks = [b for b in range(num_sms) if b not in iter1_blocks]
    print(f"\n--- gate_up iteration distribution ---")
    print(f"  Blocks with 2 iterations: {len(iter1_blocks)} (blocks 0-{max(iter1_blocks) if iter1_blocks else 'N/A'})")
    print(f"  Blocks with 1 iteration:  {len(no_iter1_blocks)} (blocks {min(no_iter1_blocks) if no_iter1_blocks else 'N/A'}-{max(no_iter1_blocks) if no_iter1_blocks else 'N/A'})")

    down_active = last_phases["down_compute"]["blocks"]
    down_idle = last_phases["down_compute_idle"]["blocks"]
    print(f"\n--- down_proj SM utilization ---")
    print(f"  Active SMs: {len(down_active)} (blocks 0-{max(down_active) if down_active else 'N/A'})")
    print(f"  Idle SMs:   {len(down_idle)} (blocks {min(down_idle) if down_idle else 'N/A'}-{max(down_idle) if down_idle else 'N/A'})")

    # Save JSON
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json_result = {
            "device": torch.cuda.get_device_name(device),
            "num_sms": num_sms,
            "iters": args.iters,
            "peak_bw_gbs": PEAK_BW_GBS,
            "aggregated": aggregated,
        }
        with open(out_path, "w") as f:
            json.dump(json_result, f, indent=2, default=str)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
