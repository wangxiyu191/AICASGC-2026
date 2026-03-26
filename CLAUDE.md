# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AICAS 2026 competition codebase for optimizing a Vision-Language Model (Qwen3-VL-2B-Instruct) inference performance. The goal is to improve TTFT (Time-To-First-Token) and throughput while maintaining VQA accuracy.

**Scoring formula:** `Final Score = 0.4 × Accuracy + 0.3 × TTFT_Improvement + 0.3 × Throughput_Improvement`

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download model (~4-5GB, first time only)
export HF_ENDPOINT=https://hf-mirror.com  # For users in China
huggingface-cli download --resume-download Qwen/Qwen3-VL-2B-Instruct \
  --local-dir ./Qwen3-VL-2B-Instruct --local-dir-use-symlinks False
```

## Running the Benchmark

```bash
# Quick test (100 samples, ~a few minutes)
python benchmark.py \
    --model-path ./Qwen3-VL-2B-Instruct \
    --dataset-path ./data \
    --output result.json \
    --num-samples 100

# Full evaluation for submission (5000 samples)
python benchmark.py \
    --model-path ./Qwen3-VL-2B-Instruct \
    --dataset-path ./data \
    --output result.json \
    --num-samples 5000
```

## Architecture

**Only `evaluation_wrapper.py` should be modified.** `benchmark.py` is off-limits.

### How the benchmark interacts with participant code

The benchmark instantiates `VLMModel` from `evaluation_wrapper.py` and then:
- Calls `vlm.processor` to prepare inputs (via `AutoProcessor.apply_chat_template`)
- Calls `vlm.model.generate(**inputs, ...)` directly for both TTFT and throughput measurement
- Calls `vlm.device` for tensor placement

### Optimization pattern (Monkey Patching)

All optimizations are applied in `VLMModel.__init__()` by calling the provided placeholder methods. Each method modifies `self._model` in place:

```python
# In __init__, uncomment to enable:
self._optimize_vision_encoder()      # Patch vision encoder attention/conv layers
self._optimize_kv_cache()            # Configure KV cache, patch attention layers
self._optimize_cross_modal_connector() # Patch cross-modal projection/attention
self._enable_flash_attention()       # Replace scaled_dot_product_attention
self._apply_quantization()           # Reload model with quantization config
```

The `generate()` method on `VLMModel` is not used by the benchmark — it's only for debugging.

### Required interface

`VLMModel` must expose three properties: `processor`, `model`, `device`.

### Performance baselines

- TTFT: ~80ms
- Throughput: ~55 tokens/sec
- Warmup: 10 samples before measurement
- TTFT measures time to first token (includes all preprocessing + prefill)
- Throughput measures full 128-token generation

## Competition Constraints

**Prohibited:** external APIs/network calls, hardcoding answers, modifying dataset, replacing the base model, GPU frequency locking, overfitting to specific samples, black-box config-only changes.

**Encouraged:** Triton/CUDA custom kernels, Flash Attention, KV cache memory layout optimization, `torch.compile`, quantization (INT8/FP8 via BitsAndBytes/TensorRT), operator replacement for Attention/LayerNorm/Conv2d.
