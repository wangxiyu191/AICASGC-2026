#!/usr/bin/env python3
"""Profile internal phase timing for active nosplit GEMV kernels on real payloads."""

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
from datasets import load_from_disk

import cuda_decode_gemv
from bench_decode_attention import _prepare_inputs
from bench_decode_linears import _run_capture
from evaluation_wrapper import VLMModel


def _compute_phase_durations_us(timestamps: torch.Tensor) -> dict[str, torch.Tensor]:
    ts = timestamps.to(torch.int64)
    zeros = torch.zeros(ts.shape[0], dtype=torch.float64)

    load_us = zeros.clone()
    compute_us = zeros.clone()

    for load_slot, compute_slot in ((1, 2), (3, 4), (5, 6)):
        load_mask = (ts[:, load_slot] > 0) & (ts[:, load_slot - 1] > 0)
        compute_mask = (ts[:, compute_slot] > 0) & (ts[:, load_slot] > 0)
        load_us[load_mask] += (ts[load_mask, load_slot] - ts[load_mask, load_slot - 1]).to(torch.float64) / 1000.0
        compute_us[compute_mask] += (ts[compute_mask, compute_slot] - ts[compute_mask, load_slot]).to(torch.float64) / 1000.0

    last_mainloop = torch.where(
        ts[:, 6] > 0,
        ts[:, 6],
        torch.where(ts[:, 4] > 0, ts[:, 4], ts[:, 2]),
    )
    epilogue_mask = (ts[:, 7] > 0) & (last_mainloop > 0)
    epilogue_us = zeros.clone()
    epilogue_us[epilogue_mask] = (ts[epilogue_mask, 7] - last_mainloop[epilogue_mask]).to(torch.float64) / 1000.0

    total_mask = (ts[:, 7] > 0) & (ts[:, 0] > 0)
    total_us = zeros.clone()
    total_us[total_mask] = (ts[total_mask, 7] - ts[total_mask, 0]).to(torch.float64) / 1000.0

    return {
        "load_us": load_us,
        "compute_us": compute_us,
        "epilogue_us": epilogue_us,
        "total_us": total_us,
    }


def _tensor_stats(values: torch.Tensor) -> dict[str, float]:
    vals = values[torch.isfinite(values)]
    if vals.numel() == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(vals.mean().item()),
        "median": float(vals.median().item()),
        "p95": float(torch.quantile(vals, 0.95).item()),
        "max": float(vals.max().item()),
    }


def _profile_target(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, repeats: int) -> tuple[dict, torch.Tensor]:
    ref = F.linear(x, weight, bias)

    raw_timestamps = []
    max_abs_diff = 0.0
    mean_abs_diff = 0.0
    meta = None

    with torch.no_grad():
        cuda_decode_gemv.nosplit_matvec_half(x, weight, bias)
        torch.cuda.synchronize()

        for rep in range(repeats):
            out, timestamps, meta = cuda_decode_gemv.nosplit_matvec_half_profile(x, weight, bias)
            torch.cuda.synchronize()
            if rep == 0:
                diff = (out - ref).abs()
                max_abs_diff = float(diff.max().item())
                mean_abs_diff = float(diff.float().mean().item())
            raw_timestamps.append(timestamps.cpu())

    stacked = torch.stack(raw_timestamps, dim=0)
    load_all = []
    compute_all = []
    epilogue_all = []
    total_all = []
    critical_load = []
    critical_compute = []
    critical_epilogue = []
    critical_total = []

    for rep_idx in range(stacked.shape[0]):
        durations = _compute_phase_durations_us(stacked[rep_idx])
        load_all.append(durations["load_us"])
        compute_all.append(durations["compute_us"])
        epilogue_all.append(durations["epilogue_us"])
        total_all.append(durations["total_us"])

        critical_idx = int(durations["total_us"].argmax().item())
        critical_load.append(float(durations["load_us"][critical_idx].item()))
        critical_compute.append(float(durations["compute_us"][critical_idx].item()))
        critical_epilogue.append(float(durations["epilogue_us"][critical_idx].item()))
        critical_total.append(float(durations["total_us"][critical_idx].item()))

    load_flat = torch.cat(load_all)
    compute_flat = torch.cat(compute_all)
    epilogue_flat = torch.cat(epilogue_all)
    total_flat = torch.cat(total_all)
    critical_total_tensor = torch.tensor(critical_total, dtype=torch.float64)
    critical_load_tensor = torch.tensor(critical_load, dtype=torch.float64)
    critical_compute_tensor = torch.tensor(critical_compute, dtype=torch.float64)
    critical_epilogue_tensor = torch.tensor(critical_epilogue, dtype=torch.float64)

    critical_share_denom = torch.clamp_min(critical_total_tensor, 1e-9)
    summary = {
        "kernel_name": meta["kernel_name"],
        "grid_blocks": int(meta["grid_blocks"]),
        "slot_names": meta["slot_names"],
        "max_abs_diff_vs_f_linear": max_abs_diff,
        "mean_abs_diff_vs_f_linear": mean_abs_diff,
        "block_phase_us": {
            "load": _tensor_stats(load_flat),
            "compute": _tensor_stats(compute_flat),
            "epilogue": _tensor_stats(epilogue_flat),
            "total": _tensor_stats(total_flat),
        },
        "critical_path_us": {
            "load": _tensor_stats(critical_load_tensor),
            "compute": _tensor_stats(critical_compute_tensor),
            "epilogue": _tensor_stats(critical_epilogue_tensor),
            "total": _tensor_stats(critical_total_tensor),
        },
        "critical_path_share_pct": {
            "load": float((critical_load_tensor / critical_share_denom).mean().item() * 100.0),
            "compute": float((critical_compute_tensor / critical_share_denom).mean().item() * 100.0),
            "epilogue": float((critical_epilogue_tensor / critical_share_denom).mean().item() * 100.0),
        },
    }
    return summary, stacked


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile internal timing for current nosplit GEMV kernels")
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
        default="all",
        choices=["qkv", "gate_up", "o_proj", "down_proj", "all"],
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--output-json",
        type=str,
        default="results/nosplit_internal_profile_layer14_step16.json",
    )
    parser.add_argument(
        "--output-pt",
        type=str,
        default="results/nosplit_internal_profile_layer14_step16.pt",
    )
    args = parser.parse_args()

    targets = ["qkv", "gate_up", "o_proj", "down_proj"] if args.target == "all" else [args.target]

    print(
        f"[Nosplit-Internal] sample={args.sample_idx} layer={args.layer_idx} "
        f"step={args.decode_step} targets={targets} repeats={args.repeats}"
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

    output = {
        "metadata": {
            "question_id": int(sample["question_id"]),
            "sample_idx": int(args.sample_idx),
            "layer_idx": int(args.layer_idx),
            "decode_step": int(args.decode_step),
            "bucketed": bool(args.bucketed),
            "repeats": int(args.repeats),
            "prompt_length": int(capture_meta["prompt_length"]),
            "max_cache_len": int(capture_meta["max_cache_len"]),
        },
        "targets": {},
    }
    raw_output = {
        "metadata": output["metadata"],
        "targets": {},
    }

    for target in targets:
        payload = payloads[target]
        x = payload["x"]
        weight = payload["weight"]
        bias = payload["bias"]
        print(f"[Nosplit-Internal] profiling {target}: x={tuple(x.shape)} weight={tuple(weight.shape)}")
        summary, raw = _profile_target(x, weight, bias, repeats=args.repeats)
        summary["x_shape"] = list(x.shape)
        summary["weight_shape"] = list(weight.shape)
        summary["has_bias"] = bias is not None
        output["targets"][target] = summary
        raw_output["targets"][target] = raw

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.save(raw_output, args.output_pt)
    print(f"[Nosplit-Internal] summary saved to {output_json}")
    print(f"[Nosplit-Internal] raw timestamps saved to {args.output_pt}")


if __name__ == "__main__":
    main()
