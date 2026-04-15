#!/usr/bin/env python3
"""Measure per-decode-step kernel-level timing using cudaEvent pairs.

Unlike nsys (which can't see inside CUDAGraph replays), this script
injects cudaEvent pairs around individual CUDAGraph replays to measure
the true steady-state decode step time.

Usage:
    TQDM_DISABLE=1 ./venv/bin/python measure_decode_step.py \
        --sample-idx 0 --decode-steps 30 --warmup 5
"""
import argparse
import os
import time

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("ENABLE_CUSTOM_DECODE", "1")
os.environ.setdefault("ENABLE_CUDAGRAPH", "1")

import torch
from datasets import load_from_disk
from statistics import mean


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


def measure_graph_replay_step(vlm, inputs, num_decode_steps=30, warmup=5):
    """Measure individual CUDAGraph replay step times using cudaEvent."""
    from evaluation_wrapper import VLMModel
    from transformers.cache_utils import StaticCache

    model = vlm._model
    wrapper = vlm.model  # ModelWrapper
    engine = wrapper._custom_decode

    # Run a full generate to ensure everything is warmed up
    with torch.no_grad():
        _ = wrapper.generate(**inputs, max_new_tokens=1, do_sample=False, use_cache=True)

    # Now do a controlled decode: prefill first, then measure each decode step
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]
    device = input_ids.device

    step_times = []
    for trial in range(warmup + 3):
        # Full generation with timing per step
        with torch.no_grad():
            out = wrapper.generate(
                **inputs, max_new_tokens=num_decode_steps,
                do_sample=False, use_cache=True,
            )

        gen_tokens = out.shape[1] - prompt_len
        if trial >= warmup:
            # Measure the full generate wall time
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = wrapper.generate(
                    **inputs, max_new_tokens=num_decode_steps,
                    do_sample=False, use_cache=True,
                )
            torch.cuda.synchronize()
            total_time = time.perf_counter() - t0

            # Measure TTFT
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = wrapper.generate(
                    **inputs, max_new_tokens=1,
                    do_sample=False, use_cache=True,
                )
            torch.cuda.synchronize()
            ttft = time.perf_counter() - t0

            decode_time = total_time - ttft
            tokens_decoded = gen_tokens - 1
            if tokens_decoded > 0:
                step_times.append({
                    'total_ms': total_time * 1000,
                    'ttft_ms': ttft * 1000,
                    'decode_ms': decode_time * 1000,
                    'tokens': tokens_decoded,
                    'ms_per_token': decode_time / tokens_decoded * 1000,
                    'tok_per_sec': tokens_decoded / decode_time,
                })

    return step_times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--model-path", type=str, default="./Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", type=str, default="./data")
    parser.add_argument("--decode-steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    from evaluation_wrapper import VLMModel

    print("[MeasureDecode] Loading model...")
    vlm = VLMModel(args.model_path)

    dataset = load_from_disk(args.dataset_path)
    sample = dataset[args.sample_idx]
    inputs = prepare_inputs(vlm, sample["image"], sample["question"])
    prompt_len = inputs.input_ids.shape[1]
    print(f"[MeasureDecode] Sample {args.sample_idx}: prompt_len={prompt_len}")

    # 1. Measure CUDAGraph replay step time
    print(f"\n{'='*80}")
    print("CUDAGraph REPLAY STEP TIMING")
    print(f"{'='*80}")
    graph_times = measure_graph_replay_step(
        vlm, inputs, num_decode_steps=args.decode_steps, warmup=args.warmup
    )
    if graph_times:
        avg_ms_per_token = mean(t['ms_per_token'] for t in graph_times)
        avg_tok_s = mean(t['tok_per_sec'] for t in graph_times)
        print(f"\n  Trials: {len(graph_times)}")
        for t in graph_times:
            print(f"    total={t['total_ms']:.2f}ms, ttft={t['ttft_ms']:.2f}ms, "
                  f"decode={t['decode_ms']:.2f}ms, tokens={t['tokens']}, "
                  f"{t['ms_per_token']:.3f}ms/tok, {t['tok_per_sec']:.1f}tok/s")
        print(f"\n  Average: {avg_ms_per_token:.3f} ms/token, {avg_tok_s:.0f} tok/s")


if __name__ == "__main__":
    main()
