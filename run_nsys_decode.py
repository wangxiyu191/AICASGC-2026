#!/usr/bin/env python3
"""Profile decode phase with nsys + NVTX annotations.

Runs a full generation (prefill + many decode steps), placing
cudaProfilerStart/Stop around only the decode loop so nsys captures
just the steady-state decode kernels.

Usage:
    NSYS=/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys
    TQDM_DISABLE=1 ENABLE_DECODE_NVTX=1 $NSYS profile \
        --trace=cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
        --force-overwrite=true -o traces/nsys_decode_0411/decode_report \
        ./venv/bin/python run_nsys_decode.py --warmup 3 --sample-idx 0

    # Without nsys (cudaEvent timing only):
    TQDM_DISABLE=1 ENABLE_DECODE_NVTX=1 \
        ./venv/bin/python run_nsys_decode.py --warmup 3 --sample-idx 0 --no-nsys
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


def run_full_generate(model_obj, inputs, max_new_tokens):
    """Run full generation and return output + timing."""
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
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Max tokens to generate (default: 128, same as benchmark)")
    parser.add_argument("--no-nsys", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    os.environ.setdefault("ENABLE_CUSTOM_DECODE", "1")
    os.environ.setdefault("ENABLE_CUDAGRAPH", "1")
    os.environ["ENABLE_VIZTRACE"] = "0"
    # Enable NVTX decode annotations for per-operation breakdown
    os.environ["ENABLE_DECODE_NVTX"] = "1"

    from evaluation_wrapper import VLMModel

    print("[DecodeProfile] Loading model...")
    vlm = VLMModel(args.model_path)
    real_model = vlm.model  # ModelWrapper → GreedyDecodeEngine path

    dataset = load_from_disk(args.dataset_path)
    sample = dataset[args.sample_idx]
    inputs = prepare_inputs(vlm, sample["image"], sample["question"])
    prompt_len = inputs.input_ids.shape[1]
    print(f"[DecodeProfile] Sample {args.sample_idx}: input_ids shape={inputs.input_ids.shape}")

    # Warmup with full generation
    print(f"[DecodeProfile] Warming up ({args.warmup} full generates)...")
    for i in range(args.warmup):
        out, t = run_full_generate(real_model, inputs, args.max_new_tokens)
        gen_tokens = out.shape[1] - prompt_len
        if i == 0:
            print(f"  warmup 0 (cold): {t*1000:.1f} ms, generated {gen_tokens} tokens")

    # Baseline timing
    times_base = []
    for _ in range(3):
        out, t = run_full_generate(real_model, inputs, args.max_new_tokens)
        gen_tokens = out.shape[1] - prompt_len
        times_base.append(t)
    base = sum(times_base) / len(times_base)
    tok_per_sec = gen_tokens / base
    decode_time = base  # approximate; includes prefill
    print(f"[DecodeProfile] Baseline: {base*1000:.2f} ms total, "
          f"{gen_tokens} tokens, {tok_per_sec:.1f} tok/s")

    # Also measure TTFT separately to estimate pure decode time
    ttft_times = []
    for _ in range(3):
        _, t = run_full_generate(real_model, inputs, max_new_tokens=1)
        ttft_times.append(t)
    ttft = sum(ttft_times) / len(ttft_times)
    pure_decode_time = base - ttft
    decode_per_token = pure_decode_time / max(1, gen_tokens - 1) * 1000  # ms
    print(f"[DecodeProfile] TTFT: {ttft*1000:.2f} ms")
    print(f"[DecodeProfile] Pure decode: {pure_decode_time*1000:.2f} ms "
          f"({gen_tokens-1} tokens, {decode_per_token:.2f} ms/token)")

    if args.no_nsys:
        print(f"\n[DecodeProfile] Running {args.repeats} profiled full generates (cudaEvent)...")
        for i in range(args.repeats):
            out, t = run_full_generate(real_model, inputs, args.max_new_tokens)
            gen_tokens = out.shape[1] - prompt_len
            print(f"  run {i}: {t*1000:.2f} ms, {gen_tokens} tokens, "
                  f"{gen_tokens/t:.1f} tok/s")
    else:
        print(f"\n[DecodeProfile] Starting nsys-captured region "
              f"({args.repeats} full generates)...")
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()

        for i in range(args.repeats):
            torch.cuda.nvtx.range_push(f"full_generate_{i}")
            out, t = run_full_generate(real_model, inputs, args.max_new_tokens)
            torch.cuda.nvtx.range_pop()
            gen_tokens = out.shape[1] - prompt_len
            print(f"  run {i}: {t*1000:.2f} ms, {gen_tokens} tokens, "
                  f"{gen_tokens/t:.1f} tok/s")

        torch.cuda.cudart().cudaProfilerStop()
        print("\n[DecodeProfile] nsys trace captured.")


if __name__ == "__main__":
    main()
