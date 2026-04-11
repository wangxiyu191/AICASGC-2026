#!/usr/bin/env python3
"""Low-overhead prefill profiler using nsys + CUPTI.

Unlike VizTracer's log_torch=True (which inflates CPU time 6-10x via
torch.profiler RecordFunction hooks), this script uses NVTX annotations
with nsys's CUPTI-based GPU kernel collection. CPU overhead is ~1.4x
instead of ~7x, giving much more realistic wall-clock timing.

Usage:
    # Collect trace (output: traces/nsys_prefill/report.nsys-rep)
    TQDM_DISABLE=1 /opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys profile \
        --trace=cuda,nvtx --force-overwrite=true \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        -o traces/nsys_prefill/report \
        ./venv/bin/python run_nsys_prefill.py --warmup 3 --sample-idx 0

    # Export to chrome trace JSON for viewing in chrome://tracing
    /opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys export \
        --type=json traces/nsys_prefill/report.nsys-rep \
        > traces/nsys_prefill/report.json

    # Or just use the built-in sqlite export + analysis
    /opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys stats \
        traces/nsys_prefill/report.nsys-rep

Alternatively, run without nsys for cudaEvent-based timing (no trace file,
just prints per-phase wall-clock and GPU times):
    TQDM_DISABLE=1 ./venv/bin/python run_nsys_prefill.py --warmup 3 --sample-idx 0 --no-nsys
"""
import argparse
import os
import time

import torch
from datasets import load_from_disk

os.environ.setdefault("TQDM_DISABLE", "1")


def prepare_inputs(model, image, question):
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
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--model-path", type=str, default="./Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", type=str, default="./data")
    parser.add_argument("--no-nsys", action="store_true",
                        help="Run without nsys, use cudaEvent timing only")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Number of profiled repeats (for --no-nsys averaging)")
    args = parser.parse_args()

    # Keep standard generate path (cudagraph mode)
    os.environ.setdefault("ENABLE_CUSTOM_DECODE", "1")
    os.environ.setdefault("ENABLE_CUDAGRAPH", "1")
    # Disable viztrace to avoid overhead
    os.environ["ENABLE_VIZTRACE"] = "0"

    from evaluation_wrapper import VLMModel

    print("[Profile] Loading model...")
    vlm = VLMModel(args.model_path)
    # Use vlm.model (ModelWrapper → GreedyDecodeEngine) to match benchmark.py
    # path.  Previously used vlm._model (raw HF generate) which bypasses our
    # custom prefill fast-paths (text_fastpath, mm_slice_fastpath) and yields
    # a different GPU-kernel profile than actual evaluation.
    real_model = vlm.model

    dataset = load_from_disk(args.dataset_path)
    sample = dataset[args.sample_idx]
    inputs = prepare_inputs(vlm, sample["image"], sample["question"])
    print(f"[Profile] Sample {args.sample_idx}: input_ids shape={inputs.input_ids.shape}")

    # Warmup
    print(f"[Profile] Warming up ({args.warmup} iters)...")
    for i in range(args.warmup):
        _, t = run_generate(real_model, inputs, max_new_tokens=1)
        if i == 0:
            print(f"  warmup 0 (cold): {t*1000:.1f} ms")

    # Baseline timing (no profiler at all)
    times_base = []
    for _ in range(5):
        _, t = run_generate(real_model, inputs, max_new_tokens=1)
        times_base.append(t)
    base = sum(times_base) / len(times_base)
    print(f"[Profile] Baseline TTFT (no profiler): {base*1000:.2f} ms")

    if args.no_nsys:
        # cudaEvent-based timing: just wall-clock, no trace file
        print(f"\n[Profile] Running {args.repeats} profiled TTFT calls (cudaEvent)...")
        times = []
        for i in range(args.repeats):
            _, t = run_generate(real_model, inputs, max_new_tokens=1)
            times.append(t)
            print(f"  run {i}: {t*1000:.2f} ms")
        avg = sum(times) / len(times)
        print(f"\n[Profile] Avg TTFT: {avg*1000:.2f} ms (baseline: {base*1000:.2f} ms, "
              f"overhead: {(avg/base - 1)*100:.1f}%)")
    else:
        # nsys mode: signal profiler start/stop via cudaProfilerApi
        print(f"\n[Profile] Starting nsys-captured region ({args.repeats} TTFT calls)...")
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()

        times = []
        for i in range(args.repeats):
            torch.cuda.nvtx.range_push(f"TTFT_run_{i}")
            _, t = run_generate(real_model, inputs, max_new_tokens=1)
            torch.cuda.nvtx.range_pop()
            times.append(t)
            print(f"  run {i}: {t*1000:.2f} ms")

        torch.cuda.cudart().cudaProfilerStop()
        avg = sum(times) / len(times)
        print(f"\n[Profile] Avg TTFT: {avg*1000:.2f} ms (baseline: {base*1000:.2f} ms, "
              f"overhead: {(avg/base - 1)*100:.1f}%)")
        print("[Profile] nsys trace captured. Process with:")
        print("  nsys stats <report.nsys-rep>")
        print("  nsys export --type=json <report.nsys-rep>")


if __name__ == "__main__":
    main()
