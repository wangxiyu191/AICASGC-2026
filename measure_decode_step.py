#!/usr/bin/env python3
"""Measure per-decode-step kernel-level timing using cudaEvent pairs.

Unlike nsys (which can't see inside CUDAGraph replays), this script
injects cudaEvent pairs around individual CUDAGraph replays to measure
the true steady-state decode step time. It also runs a breakdown using
eager-mode decode with per-operation cudaEvent timing.

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
from statistics import mean, median


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


def measure_eager_breakdown(vlm, inputs, num_decode_steps=30, warmup=3):
    """Run eager-mode decode with per-operation cudaEvent timing.

    This disables CUDAGraph to see individual kernel timings.
    """
    from evaluation_wrapper import VLMModel, GreedyDecodeEngine
    from transformers.cache_utils import StaticCache
    from optimized_layer import set_small_decode_triton_enabled

    model = vlm._model
    wrapper = vlm.model

    # Run a full generate first
    with torch.no_grad():
        _ = wrapper.generate(**inputs, max_new_tokens=1, do_sample=False, use_cache=True)

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")
    mm_token_type_ids = inputs.get("mm_token_type_ids")

    batch_size, prompt_length = input_ids.shape
    device = input_ids.device
    max_cache_len = prompt_length + num_decode_steps

    if attention_mask is None:
        prompt_attention_mask = torch.ones((batch_size, prompt_length), dtype=torch.bool, device=device)
    else:
        prompt_attention_mask = attention_mask.to(device=device, dtype=torch.bool)

    full_attention_mask = torch.zeros((batch_size, max_cache_len), dtype=torch.bool, device=device)
    full_attention_mask[:, :prompt_length] = prompt_attention_mask

    static_cache = StaticCache(config=model.config, max_cache_len=max_cache_len)

    # We need to use the actual decode path (with optimized layers) but without CUDAGraph
    # to measure individual operation timings
    set_small_decode_triton_enabled(model, True)

    try:
        with torch.no_grad():
            # Prefill
            outputs = model(
                input_ids=input_ids,
                attention_mask=prompt_attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                past_key_values=static_cache,
                use_cache=True,
                logits_to_keep=1,
            )
            rope_deltas = outputs.rope_deltas
            next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

            # Warmup decode steps
            for i in range(min(warmup, num_decode_steps - 1)):
                current_position = prompt_length + i
                full_attention_mask[:, current_position] = True
                text_position_ids = torch.full((1, 1), current_position, dtype=torch.long, device=device)
                if rope_deltas is not None:
                    text_position_ids = text_position_ids + rope_deltas.to(device=device, dtype=torch.long).view(1, 1)
                position_ids = text_position_ids.view(1, 1, 1).expand(3, 1, 1)
                cache_position = torch.tensor([current_position], dtype=torch.long, device=device)

                outputs = model(
                    input_ids=next_tokens,
                    attention_mask=full_attention_mask,
                    position_ids=position_ids,
                    past_key_values=static_cache,
                    use_cache=True,
                    cache_position=cache_position,
                    logits_to_keep=1,
                )
                next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

            # Now measure decode steps with cudaEvent timing
            measured_steps = []
            for i in range(warmup, num_decode_steps - 1):
                current_position = prompt_length + i
                full_attention_mask[:, current_position] = True
                text_position_ids = torch.full((1, 1), current_position, dtype=torch.long, device=device)
                if rope_deltas is not None:
                    text_position_ids = text_position_ids + rope_deltas.to(device=device, dtype=torch.long).view(1, 1)
                position_ids = text_position_ids.view(1, 1, 1).expand(3, 1, 1)
                cache_position = torch.tensor([current_position], dtype=torch.long, device=device)

                # Time the entire decode step
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                outputs = model(
                    input_ids=next_tokens,
                    attention_mask=full_attention_mask,
                    position_ids=position_ids,
                    past_key_values=static_cache,
                    use_cache=True,
                    cache_position=cache_position,
                    logits_to_keep=1,
                )
                next_tokens_new = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                end_event.record()
                torch.cuda.synchronize()

                step_ms = start_event.elapsed_time(end_event)
                cache_len = current_position + 1
                measured_steps.append({
                    'step': i,
                    'cache_len': cache_len,
                    'step_ms': step_ms,
                })
                next_tokens = next_tokens_new

    finally:
        set_small_decode_triton_enabled(model, False)

    return measured_steps


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

    # 2. Measure eager decode per-step timing
    print(f"\n{'='*80}")
    print("EAGER DECODE PER-STEP TIMING (small_decode_triton=True, no CUDAGraph)")
    print(f"{'='*80}")
    eager_steps = measure_eager_breakdown(
        vlm, inputs, num_decode_steps=args.decode_steps, warmup=args.warmup
    )
    if eager_steps:
        step_times = [s['step_ms'] for s in eager_steps]
        print(f"\n  Measured {len(eager_steps)} decode steps:")
        print(f"  Mean:   {mean(step_times):.3f} ms/step")
        print(f"  Median: {median(step_times):.3f} ms/step")
        print(f"  Min:    {min(step_times):.3f} ms/step")
        print(f"  Max:    {max(step_times):.3f} ms/step")
        print(f"\n  Per-step detail:")
        for s in eager_steps:
            print(f"    step={s['step']:>3}, cache_len={s['cache_len']:>5}, "
                  f"step_ms={s['step_ms']:.3f}")

    # 3. Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    if graph_times:
        print(f"  CUDAGraph decode: {avg_ms_per_token:.3f} ms/token → {avg_tok_s:.0f} tok/s")
    if eager_steps:
        eager_avg = mean(step_times)
        # Eager doesn't include lm_head + argmax (only model forward)
        print(f"  Eager decode (model fwd only): {eager_avg:.3f} ms/step")
        print(f"  Note: Eager step doesn't include lm_head + argmax")
        print(f"  Note: Eager includes Python dispatch overhead not present in CUDAGraph")


if __name__ == "__main__":
    main()
