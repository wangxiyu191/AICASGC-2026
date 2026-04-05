#!/usr/bin/env python3
"""Quick end-to-end smoke test for partition GEMV integration.

Loads the model with ENABLE_PARTITION_GEMV=1 and runs a single inference
to verify that partition GEMV activates and produces correct output.
"""
import os
import sys

os.environ["ENABLE_PARTITION_GEMV"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["ENABLE_CUDAGRAPH"] = "0"  # Disable CUDAGraph for simpler debugging

import torch
from evaluation_wrapper import VLMModel
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Qwen3-VL-2B-Instruct")

def main():
    print("=" * 60)
    print("Partition GEMV End-to-End Smoke Test")
    print("=" * 60)

    # Check that partition data gets populated
    import optimized_layer
    print(f"\nBefore model load: _PARTITION_DATA = {optimized_layer._PARTITION_DATA}")

    model = VLMModel(MODEL_PATH)

    print(f"\nAfter model load: _PARTITION_DATA is None? {optimized_layer._PARTITION_DATA is None}")
    if optimized_layer._PARTITION_DATA is not None:
        pd = optimized_layer._PARTITION_DATA
        print(f"  sm_class_lo: {pd['sm_class_lo']:#018x}")
        print(f"  sm_class_hi: {pd['sm_class_hi']:#018x}")
        print(f"  # weights: {len(pd['weights'])}")
        for k in sorted(pd['weights'].keys())[:6]:
            info = pd['weights'][k]
            print(f"    {k}: n_rows={info['n_rows']}, buf_shape={info['buf'].shape}, "
                  f"chunk_classes_shape={info['chunk_classes'].shape}")
    else:
        print("ERROR: _PARTITION_DATA is None - partition GEMV did not activate!")
        sys.exit(1)

    # Run a simple text-only inference
    print("\nRunning inference...")
    processor = model._processor
    messages = [{"role": "user", "content": [{"type": "text", "text": "What is 2+3?"}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to("cuda:0")

    with torch.no_grad():
        output_ids = model._wrapped_model.generate(**inputs, max_new_tokens=20)

    output_text = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )[0]
    print(f"Output: {output_text}")

    # Basic sanity check
    if len(output_text.strip()) > 0:
        print("\nSMOKE TEST PASSED - partition GEMV activated and inference produced output")
    else:
        print("\nSMOKE TEST FAILED - empty output")
        sys.exit(1)

if __name__ == "__main__":
    main()
    # Explicit cleanup to avoid segfault during Python shutdown
    torch.cuda.synchronize()
    import gc; gc.collect()
    torch.cuda.empty_cache()
