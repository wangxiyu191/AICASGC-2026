#!/usr/bin/env python3
"""Benchmark decode GEMV weight bandwidth on real captured payloads."""

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

from benchmark_cuda_utils import (
    DEFAULT_BENCHMARK_L2_FLUSH_MB,
    make_l2_flush_runner,
)
import cuda_decode_gemv
from bench_decode_attention import _prepare_inputs
from bench_decode_linears import (
    _bench_impl_graph,
    _cuda_nosplit_k2048_row2_matvec,
    _cuda_nosplit_k2048_row4_matvec,
    _cuda_nosplit_k6144_n8k1024_xwarp_w_async_matvec,
    _cuda_nosplit_k6144_row4_matvec,
    _run_capture,
)
from evaluation_wrapper import VLMModel

ALL_TARGETS = ("qkv", "gate_up", "o_proj", "down_proj", "lm_head")
DEFAULT_L2_FLUSH_MB = DEFAULT_BENCHMARK_L2_FLUSH_MB


def _f_linear(x, weight, bias):
    return F.linear(x, weight, bias)


def _make_weight_only_runner(variant: str | None):
    workspace = {}

    def _runner(_x, weight, _bias):
        if variant is None:
            return cuda_decode_gemv.weight_stream_read_half(weight, workspace=workspace)
        return cuda_decode_gemv.nosplit_weight_only_half_variant(weight, variant, workspace=workspace)

    return _runner


def _bandwidth_gb_s(weight: torch.Tensor, elapsed_ms: float) -> float:
    weight_bytes = weight.numel() * weight.element_size()
    return weight_bytes / (elapsed_ms / 1_000.0) / 1e9


def _build_variants(target: str, weight: torch.Tensor) -> tuple[dict[str, callable], dict[str, str]]:
    n, k = weight.shape
    variants: dict[str, callable] = {
        "flat_weight_stream": _make_weight_only_runner(None),
        "f_linear_full": _f_linear,
    }
    labels = {
        "default_full": "f_linear_full",
    }

    if k == 2048:
        variants["row2_weight_only"] = _make_weight_only_runner("k2048_row2")
        variants["row2_gemv"] = _cuda_nosplit_k2048_row2_matvec
        variants["row4_weight_only"] = _make_weight_only_runner("k2048_row4")
        variants["row4_gemv"] = _cuda_nosplit_k2048_row4_matvec

        if target == "o_proj":
            labels["default_pattern"] = "row2_weight_only"
            labels["default_full"] = "row2_gemv"
            labels["candidate_pattern"] = "row4_weight_only"
            labels["candidate_full"] = "row4_gemv"
        elif target in ("qkv", "gate_up"):
            labels["default_pattern"] = "row4_weight_only"
            labels["default_full"] = "row4_gemv"
            labels["candidate_pattern"] = "row2_weight_only"
            labels["candidate_full"] = "row2_gemv"
        else:
            labels["candidate_pattern"] = "row4_weight_only"
            labels["candidate_full"] = "row4_gemv"
    elif k == 6144:
        variants["row4_weight_only"] = _make_weight_only_runner("k6144_row4")
        variants["row4_gemv"] = _cuda_nosplit_k6144_row4_matvec
        variants["xwarp_weight_only"] = _make_weight_only_runner("k6144_n8k1024_xwarp_w_async")
        variants["xwarp_gemv"] = _cuda_nosplit_k6144_n8k1024_xwarp_w_async_matvec

        labels["default_pattern"] = "row4_weight_only"
        labels["default_full"] = "row4_gemv"
        labels["candidate_pattern"] = "xwarp_weight_only"
        labels["candidate_full"] = "xwarp_gemv"
    else:
        raise RuntimeError(f"Unsupported K={k} for target={target}")

    if target == "lm_head":
        labels["used_path_full"] = "f_linear_full"
    else:
        labels["used_path_full"] = labels["default_full"]

    return variants, labels


def _derive_results(results: dict[str, dict], labels: dict[str, str]) -> dict[str, float]:
    derived: dict[str, float] = {}
    flat_bw = results["flat_weight_stream"]["effective_weight_bandwidth_gb_s"]
    memory_only_names = [name for name in results if name.endswith("_weight_only") or name == "flat_weight_stream"]
    ceiling_name = max(
        memory_only_names,
        key=lambda name: results[name]["effective_weight_bandwidth_gb_s"],
    )
    ceiling_bw = results[ceiling_name]["effective_weight_bandwidth_gb_s"]
    used_full = labels["used_path_full"]
    used_full_bw = results[used_full]["effective_weight_bandwidth_gb_s"]

    derived["memory_only_ceiling_gb_s"] = ceiling_bw
    derived["memory_only_ceiling_source"] = ceiling_name
    derived["current_kernel_gb_s"] = used_full_bw
    derived["current_kernel_source"] = used_full
    derived["current_kernel_gap_gb_s"] = ceiling_bw - used_full_bw
    derived["current_kernel_gap_ratio_vs_ceiling"] = 1.0 - (used_full_bw / ceiling_bw)
    derived["current_kernel_bw_vs_ceiling"] = used_full_bw / ceiling_bw
    derived["flat_stream_gb_s"] = flat_bw

    if "default_pattern" in labels:
        default_pattern = labels["default_pattern"]
        default_full = labels["default_full"]
        derived["default_pattern_loss_vs_flat"] = 1.0 - results[default_pattern]["bw_vs_flat"]
        derived["default_compute_loss_vs_pattern"] = 1.0 - (
            results[default_full]["effective_weight_bandwidth_gb_s"]
            / results[default_pattern]["effective_weight_bandwidth_gb_s"]
        )

    if "candidate_pattern" in labels:
        candidate_pattern = labels["candidate_pattern"]
        candidate_full = labels["candidate_full"]
        derived["candidate_pattern_loss_vs_flat"] = 1.0 - results[candidate_pattern]["bw_vs_flat"]
        derived["candidate_compute_loss_vs_pattern"] = 1.0 - (
            results[candidate_full]["effective_weight_bandwidth_gb_s"]
            / results[candidate_pattern]["effective_weight_bandwidth_gb_s"]
        )

    derived["used_path_bw_vs_flat"] = results[used_full]["effective_weight_bandwidth_gb_s"] / flat_bw
    return derived


def _print_summary(output: dict):
    print("\n=== Memory Ceiling Summary ===")
    for target, entry in output["per_target"].items():
        meta = entry["meta"]
        derived = entry["derived"]
        weight_shape = tuple(meta["weight_shape"])
        print(
            f"{target:9s} weight={weight_shape} "
            f"ceiling={derived['memory_only_ceiling_gb_s']:.1f} GB/s "
            f"({derived['memory_only_ceiling_source']}) "
            f"current={derived['current_kernel_gb_s']:.1f} GB/s "
            f"({derived['current_kernel_source']}) "
            f"gap={derived['current_kernel_gap_gb_s']:.1f} GB/s "
            f"({derived['current_kernel_gap_ratio_vs_ceiling']*100:.2f}%)"
        )


def _run_one_target(
    target: str,
    payload: dict,
    sample: dict,
    capture_meta: dict,
    args,
) -> dict:
    x = payload["x"]
    weight = payload["weight"]
    bias = payload["bias"]
    variants, labels = _build_variants(target, weight)

    results: dict[str, dict] = {}
    pre_replay = _make_l2_flush_runner(args.l2_flush_mb)
    for name, fn in variants.items():
        _, elapsed_ms = _bench_impl_graph(
            fn,
            x,
            weight,
            bias,
            warmup=args.warmup,
            iters=args.iters,
            capture_repeats=args.capture_repeats,
            pre_replay=pre_replay,
        )
        results[name] = {
            "elapsed_ms": elapsed_ms,
            "effective_weight_bandwidth_gb_s": _bandwidth_gb_s(weight, elapsed_ms),
        }

    flat_bw = results["flat_weight_stream"]["effective_weight_bandwidth_gb_s"]
    for result in results.values():
        bw = result["effective_weight_bandwidth_gb_s"]
        result["bw_vs_flat"] = bw / flat_bw

    return {
        "meta": {
            "question_id": int(sample["question_id"]),
            "target": target,
            "x_shape": list(x.shape),
            "weight_shape": list(weight.shape),
            "has_bias": bias is not None,
            "prompt_length": int(capture_meta["prompt_length"]),
            "max_cache_len": int(capture_meta["max_cache_len"]),
            "l2_flush_mb": int(args.l2_flush_mb),
            "labels": labels,
        },
        "results": results,
        "derived": _derive_results(results, labels),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark decode GEMV weight bandwidth on real payloads")
    parser.add_argument("--model-path", type=str, default="./Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", type=str, default="./data")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--layer-idx", type=int, default=14)
    parser.add_argument("--decode-step", type=int, default=16)
    parser.add_argument("--target", type=str, default="all", choices=("all",) + ALL_TARGETS)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--capture-repeats", type=int, default=20)
    parser.add_argument("--l2-flush-mb", type=int, default=DEFAULT_L2_FLUSH_MB)
    parser.add_argument("--bucketed", action="store_true", default=True)
    parser.add_argument("--no-bucketed", dest="bucketed", action="store_false")
    parser.add_argument(
        "--output",
        type=str,
        default="results/gemv_bandwidth_all_sample0_layer14_step16.json",
    )
    args = parser.parse_args()
    if args.l2_flush_mb > 0 and args.capture_repeats != 1:
        print(
            f"[bench_gemv_bandwidth] l2 flush enabled ({args.l2_flush_mb} MiB); "
            f"forcing capture_repeats from {args.capture_repeats} to 1 so every measured kernel runs cold."
        )
        args.capture_repeats = 1

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

    targets = ALL_TARGETS if args.target == "all" else (args.target,)
    per_target = {
        target: _run_one_target(target, payloads[target], sample, capture_meta, args)
        for target in targets
    }

    output = {
        "meta": {
            "sample_idx": int(args.sample_idx),
            "question_id": int(sample["question_id"]),
            "layer_idx": int(args.layer_idx),
            "decode_step": int(args.decode_step),
            "targets": list(targets),
            "warmup": int(args.warmup),
            "iters": int(args.iters),
            "capture_repeats": int(args.capture_repeats),
            "device": str(next(iter(payloads.values()))["x"].device),
            "device_name": torch.cuda.get_device_name(next(iter(payloads.values()))["x"].device),
        },
        "per_target": per_target,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    _print_summary(output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
