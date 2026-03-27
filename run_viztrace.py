#!/usr/bin/env python3
"""Focused viztracer profiling: one TTFT trace + one throughput trace.

Produces two trace files:
  traces/generate_1.json  - TTFT (max_new_tokens=1)
  traces/generate_2.json  - Throughput (max_new_tokens=128)

Usage:
    TQDM_DISABLE=1 python run_viztrace.py [--warmup N] [--sample-idx IDX]
"""
import argparse
import os
import time

import torch
from datasets import load_from_disk
from viztracer import VizTracer

os.environ.setdefault("TQDM_DISABLE", "1")

from evaluation_wrapper import VLMModel  # noqa: E402

TRACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces")


def prepare_inputs(model, image, question):
    """Replicate benchmark's input preparation."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }]
    inputs = model.processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    return inputs


def run_generate(model_obj, inputs, max_new_tokens):
    """Run generate matching benchmark settings."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model_obj.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=0.0, use_cache=True,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return output_ids, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--sample-idx", type=int, default=0, help="Dataset sample index to profile")
    parser.add_argument("--model-path", type=str, default="./Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", type=str, default="./data")
    parser.add_argument("--max-stack-depth", type=int, default=40)
    args = parser.parse_args()

    os.makedirs(TRACE_DIR, exist_ok=True)

    # Load model (uses the real _model, not the ModelWrapper)
    print("[Profile] Loading model...")
    vlm = VLMModel(args.model_path)
    real_model = vlm._model  # bypass ModelWrapper to avoid double-tracing

    # Load dataset sample
    dataset = load_from_disk(args.dataset_path)
    sample = dataset[args.sample_idx]
    inputs = prepare_inputs(vlm, sample["image"], sample["question"])
    print(f"[Profile] Sample {args.sample_idx}: input_ids shape={inputs.input_ids.shape}")

    # Warmup
    print(f"[Profile] Warming up ({args.warmup} iterations)...")
    for i in range(args.warmup):
        torch.cuda.empty_cache()
        _, t = run_generate(real_model, inputs, max_new_tokens=1)
        if i == 0:
            print(f"  warmup 0 (cold): {t*1000:.1f} ms")
    _, t = run_generate(real_model, inputs, max_new_tokens=128)
    print(f"  warmup throughput: {t*1000:.1f} ms")

    # Baseline (no trace) for comparison
    torch.cuda.empty_cache()
    _, t_ttft = run_generate(real_model, inputs, max_new_tokens=1)
    torch.cuda.empty_cache()
    out_full, t_full = run_generate(real_model, inputs, max_new_tokens=128)
    n_tok = out_full.shape[1] - inputs.input_ids.shape[1]
    print(f"[Profile] Baseline (no trace): TTFT={t_ttft*1000:.1f}ms, "
          f"Throughput={n_tok/t_full:.1f} tok/s ({t_full*1000:.1f}ms for {n_tok} tokens)")

    # === Trace 1: TTFT (1 token) ===
    trace_ttft = os.path.join(TRACE_DIR, "generate_1.json")
    print(f"\n[Profile] Tracing TTFT → {trace_ttft}")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    tracer = VizTracer(
        log_torch=True, output_file=trace_ttft,
        max_stack_depth=args.max_stack_depth, verbose=0,
    )
    tracer.start()
    _, t = run_generate(real_model, inputs, max_new_tokens=1)
    tracer.stop()
    tracer.save()
    print(f"  TTFT (traced): {t*1000:.1f} ms  [file: {os.path.getsize(trace_ttft)/1e6:.1f} MB]")

    # === Trace 2: Throughput (128 tokens) ===
    trace_tput = os.path.join(TRACE_DIR, "generate_2.json")
    print(f"\n[Profile] Tracing throughput → {trace_tput}")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    tracer = VizTracer(
        log_torch=True, output_file=trace_tput,
        max_stack_depth=args.max_stack_depth, verbose=0,
    )
    tracer.start()
    out_tput, t = run_generate(real_model, inputs, max_new_tokens=128)
    tracer.stop()
    tracer.save()
    n_tok_tput = out_tput.shape[1] - inputs.input_ids.shape[1]
    print(f"  Throughput (traced): {n_tok_tput/t:.1f} tok/s ({t*1000:.1f} ms, {n_tok_tput} tokens)  "
          f"[file: {os.path.getsize(trace_tput)/1e6:.1f} MB]")

    print("\n[Profile] Done. Run: python analyze_traces.py ./traces")


if __name__ == "__main__":
    main()
