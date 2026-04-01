#!/usr/bin/env python3
"""Benchmark cold-L2 weight-read bandwidth for bundled K=2048 decode weights."""

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
from datasets import load_from_disk

from benchmark_cuda_utils import DEFAULT_BENCHMARK_L2_FLUSH_MB, make_l2_flush_runner
import cuda_decode_gemv
from bench_decode_attention import _prepare_inputs
from bench_decode_linears import _bench_impl_graph, _run_capture
from evaluation_wrapper import VLMModel

DEFAULT_BUNDLE = ("qkv", "o_proj", "gate_up")
K2048_VARIANTS = (
    ("flat_weight_stream", None),
    ("row2_weight_only", "k2048_row2"),
    ("row4_weight_only", "k2048_row4"),
    ("row4_persistent_weight_only", "k2048_row4_persistent"),
    ("row8_persistent_weight_only", "k2048_row8_persistent"),
)


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


def main():
    parser = argparse.ArgumentParser(description="Benchmark cold-L2 bandwidth for bundled K=2048 decode weights")
    parser.add_argument("--model-path", type=str, default="./Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", type=str, default="./data")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--layer-idx", type=int, default=14)
    parser.add_argument("--decode-step", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--capture-repeats", type=int, default=10)
    parser.add_argument("--l2-flush-mb", type=int, default=DEFAULT_BENCHMARK_L2_FLUSH_MB)
    parser.add_argument("--bundle", nargs="+", default=list(DEFAULT_BUNDLE), choices=("qkv", "o_proj", "gate_up"))
    parser.add_argument(
        "--output",
        type=str,
        default="results/k2048_bundle_bandwidth_sample0_layer14_step16_a800_flushl2.json",
    )
    args = parser.parse_args()

    if args.l2_flush_mb > 0 and args.capture_repeats != 1:
        print(
            f"[bench_k2048_bundle_bandwidth] l2 flush enabled ({args.l2_flush_mb} MiB); "
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
        bucketed=True,
        layer_idx=args.layer_idx,
    )

    bundle = []
    for name in args.bundle:
        weight = payloads[name]["weight"]
        if tuple(weight.shape[1:]) != (2048,):
            raise RuntimeError(f"{name} weight has unsupported shape {tuple(weight.shape)} for k2048 bundle")
        bundle.append((name, weight))
    weight = torch.cat([w for _name, w in bundle], dim=0).contiguous()
    dummy_x = torch.empty((1, 1, weight.shape[1]), device=weight.device, dtype=payloads[args.bundle[0]]["x"].dtype)
    pre_replay = make_l2_flush_runner(weight.device, args.l2_flush_mb)

    results = {}
    for name, variant in K2048_VARIANTS:
        fn = _make_weight_only_runner(variant)
        _, elapsed_ms = _bench_impl_graph(
            fn,
            dummy_x,
            weight,
            None,
            warmup=args.warmup,
            iters=args.iters,
            capture_repeats=args.capture_repeats,
            pre_replay=pre_replay,
        )
        results[name] = {
            "elapsed_ms": elapsed_ms,
            "effective_weight_bandwidth_gb_s": _bandwidth_gb_s(weight, elapsed_ms),
        }

    ceiling_name = max(results, key=lambda key: results[key]["effective_weight_bandwidth_gb_s"])
    output = {
        "meta": {
            "sample_idx": int(args.sample_idx),
            "question_id": int(sample["question_id"]),
            "layer_idx": int(args.layer_idx),
            "decode_step": int(args.decode_step),
            "bundle": list(args.bundle),
            "component_shapes": {name: list(w.shape) for name, w in bundle},
            "bundle_shape": list(weight.shape),
            "bundle_weight_bytes": int(weight.numel() * weight.element_size()),
            "bundle_weight_mb": float(weight.numel() * weight.element_size() / 1e6),
            "warmup": int(args.warmup),
            "iters": int(args.iters),
            "capture_repeats": int(args.capture_repeats),
            "l2_flush_mb": int(args.l2_flush_mb),
            "prompt_length": int(capture_meta["prompt_length"]),
            "max_cache_len": int(capture_meta["max_cache_len"]),
            "device": str(weight.device),
            "device_name": torch.cuda.get_device_name(weight.device),
        },
        "results": results,
        "derived": {
            "memory_only_ceiling_gb_s": results[ceiling_name]["effective_weight_bandwidth_gb_s"],
            "memory_only_ceiling_source": ceiling_name,
        },
    }
    print("\n=== K2048 Bundle Summary ===")
    print(
        f"bundle={args.bundle} shape={tuple(weight.shape)} weight_mb={output['meta']['bundle_weight_mb']:.1f} "
        f"ceiling={output['derived']['memory_only_ceiling_gb_s']:.1f} GB/s "
        f"({output['derived']['memory_only_ceiling_source']})"
    )
    print(json.dumps(output, indent=2))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
