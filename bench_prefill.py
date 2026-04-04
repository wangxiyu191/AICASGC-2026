#!/usr/bin/env python3
"""Benchmark real benchmark-path prefill forward on one sample.

This tool isolates the model-side prefill forward from the benchmark path:
1. Load one real sample from `./data`
2. Prepare multimodal inputs with the same chat template as benchmark
3. Run only the prompt prefill forward with `StaticCache`
4. Report host-side and GPU-side timings, plus optional VizTracer output

Unlike `benchmark.py`, this script does not include processor/input-prep time.
It is intended to study the model-side prefill path that feeds the first token.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from statistics import mean, median

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("ENABLE_CUSTOM_DECODE", "1")
os.environ.setdefault("ENABLE_CUDAGRAPH", "0")
os.environ.setdefault("CUDAGRAPH_PREBUILD_MAX_CACHE_LEN", "0")

import torch
from datasets import load_from_disk
from transformers.cache_utils import StaticCache

from bench_decode_attention import _build_bucketed_mask, _prepare_inputs
from benchmark import get_system_info
from benchmark_cuda_utils import (
    DEFAULT_BENCHMARK_L2_FLUSH_MB,
    benchmark_style_cuda_reset,
    make_l2_flush_runner,
)
from evaluation_wrapper import VLMModel


def _event_stats_ms(values_ms: list[float]) -> dict[str, float]:
    ordered = sorted(values_ms)
    idx_95 = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return {
        "mean_ms": float(mean(values_ms)),
        "median_ms": float(median(values_ms)),
        "p95_ms": float(ordered[idx_95]),
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
    }


def _shape_list(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return list(value.shape)
    return None


def _run_prefill_once(
    vlm: VLMModel,
    inputs,
    max_new_tokens: int,
    bucketed: bool,
):
    prompt_attention_mask, _full_attention_mask, max_cache_len = _build_bucketed_mask(
        vlm,
        inputs,
        max_new_tokens=max_new_tokens,
        bucketed=bucketed,
    )
    static_cache = StaticCache(config=vlm._model.config, max_cache_len=max_cache_len)
    with torch.no_grad():
        outputs = vlm._model(
            input_ids=inputs.input_ids,
            attention_mask=prompt_attention_mask,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
            past_key_values=static_cache,
            use_cache=True,
            logits_to_keep=1,
        )
    next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    return {
        "outputs": outputs,
        "next_tokens": next_tokens,
        "prompt_attention_mask": prompt_attention_mask,
        "max_cache_len": int(max_cache_len),
        "prompt_length": int(inputs.input_ids.shape[1]),
    }


def _measure_prefill(
    vlm: VLMModel,
    inputs,
    max_new_tokens: int,
    bucketed: bool,
    warmup: int,
    iters: int,
    l2_flush_mb: int,
):
    pre_replay = make_l2_flush_runner(vlm.device, l2_flush_mb)

    for _ in range(warmup):
        benchmark_style_cuda_reset(pre_replay)
        _run_prefill_once(
            vlm,
            inputs,
            max_new_tokens=max_new_tokens,
            bucketed=bucketed,
        )
        torch.cuda.synchronize()

    host_values_ms = []
    gpu_values_ms = []
    first_token_ids = []
    last_result = None

    for _ in range(iters):
        benchmark_style_cuda_reset(pre_replay)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        host_start = time.perf_counter()
        start.record()
        result = _run_prefill_once(
            vlm,
            inputs,
            max_new_tokens=max_new_tokens,
            bucketed=bucketed,
        )
        end.record()
        torch.cuda.synchronize()
        host_values_ms.append((time.perf_counter() - host_start) * 1000.0)
        gpu_values_ms.append(start.elapsed_time(end))
        first_token_ids.append(int(result["next_tokens"][0, 0].item()))
        last_result = result

    return {
        "host_ms": _event_stats_ms(host_values_ms),
        "gpu_ms": _event_stats_ms(gpu_values_ms),
        "first_token_ids": first_token_ids,
        "last_result": last_result,
    }


def _trace_prefill(
    vlm: VLMModel,
    inputs,
    max_new_tokens: int,
    bucketed: bool,
    warmup: int,
    trace_iters: int,
    trace_output: Path,
    l2_flush_mb: int,
):
    from viztracer import VizTracer

    pre_replay = make_l2_flush_runner(vlm.device, l2_flush_mb)
    for _ in range(warmup):
        benchmark_style_cuda_reset(pre_replay)
        _run_prefill_once(
            vlm,
            inputs,
            max_new_tokens=max_new_tokens,
            bucketed=bucketed,
        )
        torch.cuda.synchronize()

    tracer = VizTracer(log_torch=True, output_file=str(trace_output), max_stack_depth=15, verbose=0)
    tracer.start()
    try:
        for _ in range(trace_iters):
            benchmark_style_cuda_reset(pre_replay)
            _run_prefill_once(
                vlm,
                inputs,
                max_new_tokens=max_new_tokens,
                bucketed=bucketed,
            )
            torch.cuda.synchronize()
    finally:
        tracer.stop()
        tracer.save()


def main():
    parser = argparse.ArgumentParser(description="Benchmark model-side prefill on real benchmark-path inputs")
    parser.add_argument("--model-path", type=str, default="./Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", type=str, default="./data")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--bucketed", action="store_true", default=True, help="Use graph-style bucketed max_cache_len")
    parser.add_argument("--no-bucketed", dest="bucketed", action="store_false")
    parser.add_argument("--l2-flush-mb", type=int, default=DEFAULT_BENCHMARK_L2_FLUSH_MB)
    parser.add_argument("--viztrace", action="store_true", help="Capture VizTracer trace for the prefill loop.")
    parser.add_argument("--viztrace-iters", type=int, default=1, help="Number of traced prefill runs.")
    parser.add_argument("--viztrace-output", type=str, default=None, help="Trace JSON output path under traces/.")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (default: results/...)")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results/prefill_bench_sample{args.sample_idx}.json"
    if args.viztrace_output is None:
        args.viztrace_output = f"traces/prefill/prefill_bench_sample{args.sample_idx}.json"

    print("[PrefillBench] Loading model...")
    vlm = VLMModel(args.model_path)
    dataset = load_from_disk(args.dataset_path)
    sample = dataset[args.sample_idx]
    inputs = _prepare_inputs(vlm, sample["image"], sample["question"])

    print(
        "[PrefillBench] Prepared inputs: "
        f"sample_idx={args.sample_idx}, question_id={sample['question_id']}, "
        f"prompt_length={inputs.input_ids.shape[1]}, bucketed={args.bucketed}"
    )
    print(
        "[PrefillBench] Benchmarking prefill "
        f"(warmup={args.warmup}, iters={args.iters}, reset=cold_l2_flush({args.l2_flush_mb}MiB))..."
    )
    measured = _measure_prefill(
        vlm,
        inputs,
        max_new_tokens=args.max_new_tokens,
        bucketed=args.bucketed,
        warmup=args.warmup,
        iters=args.iters,
        l2_flush_mb=args.l2_flush_mb,
    )

    last_result = measured["last_result"]
    outputs = last_result["outputs"]
    first_token_ids = measured["first_token_ids"]
    first_token_unique = sorted(set(first_token_ids))

    metadata = {
        "question_id": int(sample["question_id"]),
        "sample_idx": int(args.sample_idx),
        "bucketed": bool(args.bucketed),
        "prompt_length": int(last_result["prompt_length"]),
        "max_cache_len": int(last_result["max_cache_len"]),
        "max_new_tokens": int(args.max_new_tokens),
        "input_ids_shape": list(inputs.input_ids.shape),
        "prompt_attention_mask_shape": list(last_result["prompt_attention_mask"].shape),
        "pixel_values_shape": _shape_list(inputs.get("pixel_values")),
        "image_grid_thw_shape": _shape_list(inputs.get("image_grid_thw")),
        "mm_token_type_ids_shape": _shape_list(inputs.get("mm_token_type_ids")),
        "logits_shape": list(outputs.logits.shape),
        "rope_deltas_shape": _shape_list(getattr(outputs, "rope_deltas", None)),
        "dtype": str(outputs.logits.dtype),
        "device": str(outputs.logits.device),
        "l2_flush_mb": int(args.l2_flush_mb),
        "timers": ["perf_counter", "cuda_event"],
        "benchmark_reset": "torch.cuda.empty_cache + torch.cuda.synchronize + optional_l2_flush",
        "measured_path": "model_prefill_forward_only",
        "excludes": ["processor.apply_chat_template", "python-side image/question preprocessing"],
        "first_token_id_last": int(first_token_ids[-1]),
        "first_token_ids_unique": first_token_unique,
        "first_token_stable": len(first_token_unique) == 1,
        "warmup": int(args.warmup),
        "iters": int(args.iters),
    }

    print("\n[PrefillBench] Summary")
    print(f"  host mean/median : {measured['host_ms']['mean_ms']:.3f} / {measured['host_ms']['median_ms']:.3f} ms")
    print(f"  gpu  mean/median : {measured['gpu_ms']['mean_ms']:.3f} / {measured['gpu_ms']['median_ms']:.3f} ms")
    print(
        "  first token      : "
        f"{first_token_ids[-1]} (stable={metadata['first_token_stable']}, unique={first_token_unique})"
    )

    trace_path = None
    if args.viztrace:
        trace_path = Path(args.viztrace_output)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            "[PrefillBench] Capturing viztrace "
            f"(warmup={args.warmup}, iters={args.viztrace_iters})..."
        )
        _trace_prefill(
            vlm,
            inputs,
            max_new_tokens=args.max_new_tokens,
            bucketed=args.bucketed,
            warmup=args.warmup,
            trace_iters=args.viztrace_iters,
            trace_output=trace_path,
            l2_flush_mb=args.l2_flush_mb,
        )
        print(f"[PrefillBench] Trace saved to: {trace_path}")

    output = {
        "system_info": get_system_info(),
        "metadata": metadata,
        "timings": {
            "host_ms": measured["host_ms"],
            "gpu_ms": measured["gpu_ms"],
        },
    }
    if trace_path is not None:
        output["trace"] = {
            "enabled": True,
            "iters": int(args.viztrace_iters),
            "output": str(trace_path),
        }
    else:
        output["trace"] = {
            "enabled": False,
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n[PrefillBench] Saved to: {output_path}")


if __name__ == "__main__":
    main()
