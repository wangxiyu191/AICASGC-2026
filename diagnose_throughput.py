#!/usr/bin/env python3
"""Diagnose throughput gap between profiling script and benchmark.

Checks:
1. Input length variation across dataset samples
2. Same-input vs different-input throughput
3. Impact of empty_cache() between calls
4. ModelWrapper overhead
"""
import os, time, torch
os.environ["TQDM_DISABLE"] = "1"

from datasets import load_from_disk
from evaluation_wrapper import VLMModel

NUM_SAMPLES = 10

def prepare_inputs(model, image, question):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]}]
    return model.processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

def timed_generate(model_obj, inputs, max_new_tokens):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model_obj.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False, temperature=0.0, use_cache=True)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, out

print("Loading model...")
vlm = VLMModel("./Qwen3-VL-2B-Instruct")
real_model = vlm._model  # bypass ModelWrapper
wrapped_model = vlm.model  # through ModelWrapper

dataset = load_from_disk("./data")

# 1. Check input lengths across samples
print(f"\n{'='*70}")
print("1. Input sequence lengths across first {NUM_SAMPLES} samples")
print(f"{'='*70}")
all_inputs = []
for i in range(NUM_SAMPLES):
    sample = dataset[i]
    inp = prepare_inputs(vlm, sample["image"], sample["question"])
    seq_len = inp.input_ids.shape[1]
    print(f"  Sample {i}: seq_len={seq_len}")
    all_inputs.append(inp)

# 2. Warmup
print(f"\nWarming up (5 iters on sample 0)...")
for _ in range(5):
    timed_generate(real_model, all_inputs[0], max_new_tokens=1)
timed_generate(real_model, all_inputs[0], max_new_tokens=128)

# 3. Same input repeated (what profiling script does)
print(f"\n{'='*70}")
print("2. Same input repeated (sample 0) - like profiling script")
print(f"{'='*70}")
times_same = []
for i in range(5):
    t, out = timed_generate(real_model, all_inputs[0], max_new_tokens=128)
    n_tok = out.shape[1] - all_inputs[0].input_ids.shape[1]
    tput = n_tok / t
    times_same.append(tput)
    print(f"  Run {i}: {tput:.1f} tok/s ({t*1000:.1f}ms, {n_tok} tokens)")
print(f"  Avg: {sum(times_same)/len(times_same):.1f} tok/s")

# 4. Different inputs (what benchmark does) - no cache clearing
print(f"\n{'='*70}")
print("3. Different inputs, NO empty_cache - just generate()")
print(f"{'='*70}")
times_diff = []
for i in range(NUM_SAMPLES):
    t, out = timed_generate(real_model, all_inputs[i], max_new_tokens=128)
    n_tok = out.shape[1] - all_inputs[i].input_ids.shape[1]
    tput = n_tok / t
    times_diff.append(tput)
    seq_len = all_inputs[i].input_ids.shape[1]
    print(f"  Sample {i} (seq={seq_len}): {tput:.1f} tok/s ({t*1000:.1f}ms, {n_tok} tokens)")
print(f"  Avg: {sum(times_diff)/len(times_diff):.1f} tok/s")

# 5. Different inputs WITH empty_cache (benchmark style)
print(f"\n{'='*70}")
print("4. Different inputs, WITH empty_cache + sync + sleep (benchmark style)")
print(f"{'='*70}")
times_bench = []
for i in range(NUM_SAMPLES):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    # Also do TTFT first like benchmark
    timed_generate(real_model, all_inputs[i], max_new_tokens=1)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(0.005)
    # Then throughput
    t, out = timed_generate(real_model, all_inputs[i], max_new_tokens=128)
    n_tok = out.shape[1] - all_inputs[i].input_ids.shape[1]
    tput = n_tok / t
    times_bench.append(tput)
    seq_len = all_inputs[i].input_ids.shape[1]
    print(f"  Sample {i} (seq={seq_len}): {tput:.1f} tok/s ({t*1000:.1f}ms, {n_tok} tokens)")
print(f"  Avg: {sum(times_bench)/len(times_bench):.1f} tok/s")

# 6. Through ModelWrapper (to check overhead)
print(f"\n{'='*70}")
print("5. Through ModelWrapper (same as benchmark uses)")
print(f"{'='*70}")
times_wrap = []
for i in range(3):
    t, out = timed_generate(wrapped_model, all_inputs[i], max_new_tokens=128)
    n_tok = out.shape[1] - all_inputs[i].input_ids.shape[1]
    tput = n_tok / t
    times_wrap.append(tput)
    print(f"  Sample {i}: {tput:.1f} tok/s ({t*1000:.1f}ms)")
print(f"  Avg: {sum(times_wrap)/len(times_wrap):.1f} tok/s")

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"  Same input repeated:        {sum(times_same)/len(times_same):.1f} tok/s")
print(f"  Different inputs (no clear): {sum(times_diff)/len(times_diff):.1f} tok/s")
print(f"  Different inputs (bench):    {sum(times_bench)/len(times_bench):.1f} tok/s")
print(f"  Through ModelWrapper:        {sum(times_wrap)/len(times_wrap):.1f} tok/s")
