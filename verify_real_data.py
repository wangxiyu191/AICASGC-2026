"""
Compare generation output: original model vs fused-layer model on real data.

Strategy: load model once, run N samples baseline, then apply fused layers
in-place and re-run same samples. Compare generated text token-by-token.
"""
import torch
import os
import sys

os.environ["TQDM_DISABLE"] = "1"

from transformers import AutoModelForImageTextToText, AutoProcessor
from datasets import load_from_disk


def generate_one(model, processor, image, question, device, max_new_tokens=128):
    """Run single-sample generation, return token ids and text."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    ).to(device)
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=0.0, use_cache=True,
        )
    gen_ids = output_ids[0][input_len:].tolist()
    text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)
    return gen_ids, text


def main():
    model_path = "./Qwen3-VL-2B-Instruct"
    data_path = "./data"
    device = "cuda:0"
    N = 20  # number of samples

    print("Loading dataset...")
    dataset = load_from_disk(data_path)

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_path)

    print("Loading model (no fused layers)...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.float16,
        device_map=device, attn_implementation="sdpa",
    )
    model.eval()

    # ---- Phase 1: baseline (original layers) ----
    print(f"\n{'='*60}")
    print(f"Phase 1: Baseline (original layers) — {N} samples")
    print(f"{'='*60}")

    # Warmup
    for i in range(3):
        generate_one(model, processor, dataset[i]["image"],
                     dataset[i]["question"], device, max_new_tokens=8)

    baseline_results = []
    for i in range(N):
        torch.cuda.empty_cache()
        ids, text = generate_one(model, processor, dataset[i]["image"],
                                 dataset[i]["question"], device)
        baseline_results.append({"ids": ids, "text": text})
        print(f"  [{i:>2}] {len(ids):>3} tokens | {text[:80]}...")

    # ---- Phase 2: apply fused layers ----
    print(f"\n{'='*60}")
    print("Applying fused decoder layers...")
    print(f"{'='*60}")

    from optimized_layer import replace_decoder_layers
    count = replace_decoder_layers(model)
    print(f"Replaced {count} layers")

    # Warmup again (to be fair)
    for i in range(3):
        generate_one(model, processor, dataset[i]["image"],
                     dataset[i]["question"], device, max_new_tokens=8)

    # ---- Phase 3: fused layers ----
    print(f"\n{'='*60}")
    print(f"Phase 2: Fused layers — {N} samples")
    print(f"{'='*60}")

    fused_results = []
    for i in range(N):
        torch.cuda.empty_cache()
        ids, text = generate_one(model, processor, dataset[i]["image"],
                                 dataset[i]["question"], device)
        fused_results.append({"ids": ids, "text": text})
        print(f"  [{i:>2}] {len(ids):>3} tokens | {text[:80]}...")

    # ---- Compare ----
    print(f"\n{'='*60}")
    print("Comparison")
    print(f"{'='*60}")

    exact_match = 0
    total_tokens_baseline = 0
    total_tokens_matched = 0
    diverge_positions = []

    for i in range(N):
        b_ids = baseline_results[i]["ids"]
        f_ids = fused_results[i]["ids"]
        b_text = baseline_results[i]["text"]
        f_text = fused_results[i]["text"]

        total_tokens_baseline += len(b_ids)

        # Token-level comparison
        min_len = min(len(b_ids), len(f_ids))
        matched = 0
        first_diff_pos = -1
        for j in range(min_len):
            if b_ids[j] == f_ids[j]:
                matched += 1
            elif first_diff_pos == -1:
                first_diff_pos = j

        total_tokens_matched += matched

        if b_ids == f_ids:
            exact_match += 1
            print(f"  [{i:>2}] EXACT MATCH ({len(b_ids)} tokens)")
        else:
            if first_diff_pos == -1:
                first_diff_pos = min_len  # length mismatch
            print(f"  [{i:>2}] DIFFER at token {first_diff_pos}/{min_len}"
                  f"  matched={matched}/{min_len}"
                  f"  len: {len(b_ids)} vs {len(f_ids)}")
            diverge_positions.append(first_diff_pos)
            # Show the diff context
            if first_diff_pos < min_len:
                ctx_start = max(0, first_diff_pos - 2)
                b_ctx = processor.tokenizer.decode(b_ids[ctx_start:first_diff_pos+3],
                                                    skip_special_tokens=True)
                f_ctx = processor.tokenizer.decode(f_ids[ctx_start:first_diff_pos+3],
                                                    skip_special_tokens=True)
                print(f"        baseline: ...{b_ctx}...")
                print(f"        fused:    ...{f_ctx}...")

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Samples:           {N}")
    print(f"  Exact match:       {exact_match}/{N} ({100*exact_match/N:.1f}%)")
    print(f"  Token match rate:  {total_tokens_matched}/{total_tokens_baseline}"
          f" ({100*total_tokens_matched/total_tokens_baseline:.2f}%)")
    if diverge_positions:
        avg_div = sum(diverge_positions) / len(diverge_positions)
        print(f"  Avg diverge pos:   {avg_div:.1f} (of ~128 tokens)")
        print(f"  Min diverge pos:   {min(diverge_positions)}")
        print(f"  Max diverge pos:   {max(diverge_positions)}")


if __name__ == "__main__":
    main()
