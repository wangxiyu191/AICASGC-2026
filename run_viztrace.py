#!/usr/bin/env python3
"""Focused trace runner aligned with the real ModelWrapper generate path.

Produces two trace files via ModelWrapper's built-in VizTracer integration:
  <output_dir>/generate_1.json  - TTFT (max_new_tokens=1)
  <output_dir>/generate_2.json  - Throughput (max_new_tokens=128)

Usage:
    TQDM_DISABLE=1 python run_viztrace.py [--warmup N] [--sample-idx IDX] [--mode cudagraph]
"""
import argparse
import os
import time

import torch
from datasets import load_from_disk

os.environ.setdefault("TQDM_DISABLE", "1")

DEFAULT_TRACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces")


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
    parser.add_argument(
        "--mode",
        choices=("hf", "eager", "cudagraph"),
        default="cudagraph",
        help="Generate path to trace: raw HF generate, custom eager decode, or custom decode with CUDA Graph",
    )
    parser.add_argument(
        "--keep-existing-traces",
        action="store_true",
        help="Keep existing traces/generate_*.json instead of deleting them before profiling",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_TRACE_DIR,
        help="Directory for generate_*.json trace outputs",
    )
    args = parser.parse_args()

    trace_dir = os.path.abspath(args.output_dir)
    os.environ["ENABLE_VIZTRACE"] = "1"
    os.environ["VIZTRACE_OUTPUT_DIR"] = trace_dir
    os.environ["VIZTRACE_WARMUP_CALLS"] = str(10 ** 9)
    if args.mode == "hf":
        os.environ["ENABLE_CUSTOM_DECODE"] = "0"
        os.environ["ENABLE_CUDAGRAPH"] = "0"
        os.environ["ENABLE_CUDAGRAPH_WITH_VIZTRACE"] = "0"
    elif args.mode == "eager":
        os.environ["ENABLE_CUSTOM_DECODE"] = "1"
        os.environ["ENABLE_CUDAGRAPH"] = "0"
        os.environ["ENABLE_CUDAGRAPH_WITH_VIZTRACE"] = "0"
    else:
        os.environ["ENABLE_CUSTOM_DECODE"] = "1"
        os.environ["ENABLE_CUDAGRAPH"] = "1"
        os.environ["ENABLE_CUDAGRAPH_WITH_VIZTRACE"] = "1"

    from evaluation_wrapper import VLMModel  # noqa: E402

    os.makedirs(trace_dir, exist_ok=True)
    if not args.keep_existing_traces:
        for filename in os.listdir(trace_dir):
            if filename.startswith("generate_") and filename.endswith(".json"):
                os.remove(os.path.join(trace_dir, filename))

    print(f"[Profile] Mode: {args.mode}")
    print("[Profile] Loading model...")
    vlm = VLMModel(args.model_path)
    real_model = vlm._model
    traced_model = vlm.model

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

    # Reset wrapper trace counters so the next two generate() calls become generate_1/2.json.
    traced_model._call_count = 0
    traced_model._trace_count = 0
    traced_model._warmup_calls = 0

    # === Trace 1: TTFT (1 token) ===
    trace_ttft = os.path.join(trace_dir, "generate_1.json")
    print(f"\n[Profile] Tracing TTFT → {trace_ttft}")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    _, t = run_generate(traced_model, inputs, max_new_tokens=1)
    print(f"  TTFT (traced): {t*1000:.1f} ms  [file: {os.path.getsize(trace_ttft)/1e6:.1f} MB]")

    # === Trace 2: Throughput (128 tokens) ===
    trace_tput = os.path.join(trace_dir, "generate_2.json")
    print(f"\n[Profile] Tracing throughput → {trace_tput}")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    out_tput, t = run_generate(traced_model, inputs, max_new_tokens=128)
    n_tok_tput = out_tput.shape[1] - inputs.input_ids.shape[1]
    print(f"  Throughput (traced): {n_tok_tput/t:.1f} tok/s ({t*1000:.1f} ms, {n_tok_tput} tokens)  "
          f"[file: {os.path.getsize(trace_tput)/1e6:.1f} MB]")

    print(f"\n[Profile] Done. Run: python analyze_traces.py {trace_dir}")


if __name__ == "__main__":
    main()
