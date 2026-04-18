# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AICAS 2026 competition entry — optimize Qwen3-VL-2B-Instruct inference on a single NVIDIA A800 80GB PCIe for **TTFT**, **throughput** (tokens/sec), and accuracy. The scoring formula is `0.4 × accuracy_ratio + 0.3 × TTFT_improvement + 0.3 × throughput_improvement`. Only `evaluation_wrapper.py` (and files it imports) are in scope — **never modify `benchmark.py`**; the official evaluator ignores any uploaded copy.

Competition ground rules (see `README_CN.md`):
- No quantization, sparsification, distillation, or any change to model parameters/architecture.
- No external API/network calls, no precomputed or hardcoded answers, no per-sample branching.
- Eval hardware is a single A800 80GB PCIe, batch size 1. The eval worker machine has **no internet** — extensions must be pre-built.

## Environment

- **All development must use the `./venv` virtualenv.** `venv/bin/python` is the correct interpreter.
- Target environment: Python 3.12.3, PyTorch 2.8.0+cu128, CUDA Toolkit 12.8 (`/usr/local/cuda/bin/nvcc`). `flash-attn 2.8.3` is pre-installed on the eval machine.
- **Always set `TQDM_DISABLE=1`** when any script path hits `from_pretrained` / benchmarks — otherwise stdout blocks and the run hangs forever.
- Model weights live in `./Qwen3-VL-2B-Instruct/`, validation data in `./data/` (neither is in git).

## Common commands

### Full benchmark (TTFT / throughput / accuracy)
```bash
TQDM_DISABLE=1 ./venv/bin/python benchmark.py \
  --model-path ./Qwen3-VL-2B-Instruct \
  --dataset-path ./data \
  --output results/result_<label>.json \
  --num-samples 10          # 10 for smoke, 100 for ablation, 5000 for final
```
Benchmark samples are a **fixed prefix** (samples 0..N-1). `--random-seed` only affects PyTorch RNG, not sample selection. `results/` is the canonical home for JSON outputs.

### Microbenchmarks (real decode tensors, `cudaEvent` timing, graph-capture aware)
```bash
# Attention variants
TQDM_DISABLE=1 ./venv/bin/python bench_decode_attention.py \
  --sample-idx 0 --layer-idx 14 --decode-step 36 \
  --warmup 5 --iters 50 --graph --graph-capture-repeats 64 \
  --output results/attention_microbench_<label>.json

# Linear/GEMV variants (qkv / gate_up / o_proj / down_proj / lm_head)
TQDM_DISABLE=1 ./venv/bin/python bench_decode_linears.py \
  --sample-idx 0 --layer-idx 14 --decode-step 36 \
  --warmup 5 --iters 50 --graph --graph-capture-repeats 64 \
  --output results/linear_microbench_<label>.json

# Single-layer decode chain (CUDAGraph replay timing, with optional viztrace)
./venv/bin/python bench_decode_layer_chain.py --help
```

### Trace capture & analysis
```bash
# Standalone viztracer runner
TQDM_DISABLE=1 ./venv/bin/python run_viztrace.py \
  --warmup 5 --sample-idx 0 --mode cudagraph \
  --output-dir traces/compare/<label>

./venv/bin/python analyze_traces.py traces/compare/<label> \
  --top 12 --step-samples 3 --step-op-limit 35 \
  --layer-samples 3 --layer-neighborhood 8

# Or drive benchmark.py with trace env vars
TQDM_DISABLE=1 ENABLE_VIZTRACE=1 \
  VIZTRACE_OUTPUT_DIR=./traces/benchmark_env/<label> \
  VIZTRACE_WARMUP_CALLS=1 ENABLE_CUDAGRAPH_WITH_VIZTRACE=1 \
  ./venv/bin/python benchmark.py \
  --model-path ./Qwen3-VL-2B-Instruct --dataset-path ./data \
  --output results/result_trace_<label>.json --num-samples 1
```
Trace files go in **`traces/`** (use a subdirectory for comparisons — never create sibling `trace*/` dirs at the repo root). Trace runs are much slower than normal — treat their tok/s as hot-path ordering info only.

### Build CUDA extensions
Loaders in `cuda_decode_gemv.py` / `cuda_decode_attention.py` auto-rebuild on source-mtime change. To force a manual rebuild:
```bash
cd cuda_extensions
PATH=/usr/local/cuda/bin:$PATH ../venv/bin/python setup.py build_ext --inplace
PATH=/usr/local/cuda/bin:$PATH ../venv/bin/python setup_decode_gemv.py build_ext --inplace
```
Setup scripts explicitly disable `ninja` and use `TORCH_CUDA_ARCH_LIST=8.0` (A800 / SM80).

### Create submission ZIP
```bash
./create_submission.sh            # emits submission_<short-sha>.zip
```
This script builds both CUDA extensions, then packs `evaluation_wrapper.py`, `optimized_layer.py`, the CUDA loaders, `cuda_extensions/` (sources + prebuilt `.so`), `requirements.txt`, and `docs/optimization_report.md`. The ZIP must NOT contain `benchmark.py` (the eval worker ignores it anyway). Precompile `.so` files against CUDA 12.8 to match the eval machine ABI.

## Architecture

The hot path is a single file chain; all optimizations land in **`evaluation_wrapper.py`** and **`optimized_layer.py`**. For the final implementation summary, read [`docs/optimization_report.md`](docs/optimization_report.md), especially Sections 3 and 4.

### High-level call graph
```
benchmark.py (official, immutable)
  └── from evaluation_wrapper import VLMModel
      VLMModel.__init__  →  _configure_cuda
                            AutoProcessor / AutoModelForImageTextToText (FP16, SDPA)
                            _optimize_kv_cache          (use_cache=True, pad_token fix)
                            _replace_decoder_layers     → optimized_layer.replace_decoder_layers
                            _optimize_prefill_path      (monkey-patches on vision/text model)
                            ModelWrapper(model)         (wraps .generate)

      vlm.model.generate(...)
        → ModelWrapper.generate
            → GreedyDecodeEngine.generate                (custom greedy + StaticCache + CUDAGraph)
                → DecodeCudaGraphRunner.replay           (one captured step per replay)
                → async EOS polling via pinned host ring
            → model.generate(...) fallback               (only when request unsupported)
```

### Two execution phases
1. **Prefill (eager)** — runs through `_optimize_prefill_path` monkey-patches that cache vision grid metadata (`fast_pos_embed_interpolate`, `rot_pos_emb`, cu_seqlens, per-frame split_sizes), rewrite `patch_embed` as `F.linear`, skip HF's generic multimodal merge / causal-mask builder, and cache RoPE position deltas. Vision encoder uses a fused Triton RoPE kernel (`fused_vision_rope`) replacing 14 PyTorch ops per ViT block. Text decoder prefill uses `_language_model_forward_prefill_fastpath` with Triton fused QK-norm+RoPE (non-contiguous-input aware, separate output strides to skip DirectCopy), fused add+RMSNorm, and packed SiLU*mul. Prefill attention runs SDPA on the valid-prompt slice then writes back into StaticCache.

2. **Decode (CUDAGraph replay)** — `DecodeCudaGraphRunner` pre-captures **32 graph buckets** (`cache_len` 64..2048 in 64-step increments, tunable via `CUDAGRAPH_PREBUILD_MAX_CACHE_LEN` / `CUDAGRAPH_BUCKET_SIZE`) during `VLMModel.__init__`. Each request picks the smallest bucket ≥ its cache length, resets the static cache/mask, then runs `graph.replay() → argmax → next_token_buf` in a tight loop. EOS detection runs asynchronously via pinned host copies + CUDA events so the decode loop never stalls on `cudaEventSynchronize`.

### OptimizedDecoderLayer (optimized_layer.py)
Each Qwen3-VL decoder layer is swapped for `OptimizedDecoderLayer`. The decode chain per layer is:
```
input_rmsnorm → qkv (1 fused GEMM/GEMV) → fused q_norm+k_norm+RoPE
 → single-token KV cache update (1 Triton kernel) → decode attention
 → o_proj → post_attn add+rmsnorm → fused gate_up → silu_mul → down_proj → residual_add
```
Key fusions and when they fire are summarized in `docs/optimization_report.md` Sections 3.1, 3.2, and 4.2. The single biggest wins layered in over time (see `result.csv`): fused QKV/gate_up, Triton small-shape RMSNorm/QK-norm+RoPE/decode-attention, split-KV attention with persistent workspace + online merge, Triton single-token StaticCache writer, CUDA nosplit GEMV (replacing cuBLAS M=1 WMMA GEMM), and persistent cooperative CUDA kernels that fuse `post_attn add+rmsnorm → gate_up → silu_mul → down_proj → residual_add` into a single launch, with an experimental variant that also absorbs `o_proj` upstream. The prefill path also has Triton kernels: fused QK-norm+RoPE with separate output strides (eliminates DirectCopy), packed SiLU*mul (eliminates chunk copies), fused add+RMSNorm, and fused vision RoPE (replaces 14 PyTorch ops per ViT block).

### Decode attention dispatch (`_small_decode_attention`)
Tried in order; each can be individually toggled via env var:
1. persistent **split-KV** (packed stats, one-pass online merge)
2. **dot-style** (`tl.dot` block)
3. **grouped prefix + cache_position**
4. **grouped** (GQA K/V reuse)
5. **per-head** (last resort fallback)

### CUDA extensions (`cuda_extensions/`)
Two PyTorch `BuildExtension` modules, each with Python loaders at repo root:

- **`cuda_decode_gemv_ext`** (`cuda_decode_gemv.py`, `decode_gemv_kernel.cu`, `decode_gemv_gridsync.cu`):
  - `nosplit_matvec_half()` — primary M=1 GEMV; auto-routes `down_proj` (2048×6144) to the `k6144_n8k1024_xwarp_w_async` variant, and has k2048/k6144 row2/row4/xwarp variants plus internal `%globaltimer` per-block profiling.
  - Fused MLP entry points: `fused_add_rmsnorm_gateup_half`, `fused_silu_downproj_half`, `fused_gateup_silu_downproj_half`, persistent `fused_post_attn_mlp_persistent`, and the newer `fused_o_proj_post_attn_mlp_persistent`.

- **`cuda_decode_attention_ext`** (`cuda_decode_attention.py`, `decode_attention_kernel.cu`, `decode_attention_v4_kernel.cu`):
  - Experimental split-KV decode attention with 32-warp tree reduction. Works but currently still slower than the Triton split-KV path on A800 — keep as reference implementation.

### Environment variable toggles
The decode-path flags are enabled by default via `os.environ.setdefault` at the top of `evaluation_wrapper.py` and inside `optimized_layer.py`. Full table in `docs/optimization_report.md §4.3`. The currently-default ones you will likely touch:
- `ENABLE_CUSTOM_DECODE`, `ENABLE_CUDAGRAPH`
- `ENABLE_CUDA_NOSPLIT_GEMV`
- `ENABLE_CUDA_FUSED_GATEUP_SILU_DOWNPROJ` + `CUDA_FUSED_GATEUP_SILU_DOWNPROJ_VARIANT=persistent`
- `ENABLE_CUDA_FUSED_POST_ATTN_MLP_PERSISTENT`
- `ENABLE_TRITON_SMALL_RMSNORM` / `_ADD_RMSNORM` / `_QK_NORM_ROPE` / `_DECODE_ATTENTION{,_DOT,_SPLIT_KV}` / `_DECODE_STATIC_KV` / `_DECODE_GEMV`
- `ENABLE_PREFILL_{VISION_GRID_CACHE,ROPE_FASTPATH,PATCH_EMBED_LINEAR,MM_SLICE_FASTPATH,TEXT_FASTPATH}`
- `CUDAGRAPH_PREBUILD_MAX_CACHE_LEN` (2048), `CUDAGRAPH_BUCKET_SIZE` (64), `CUDAGRAPH_ASYNC_EOS_WINDOW` (2)

## Workflow conventions

- **Track every optimization in one of three CSVs** (see `AGENTS.md`):
  - `result.csv` — milestones: baselines, official evaluations, mainline-merged optimizations, current default configuration benchmarks.
  - `result_experiments.csv` — exploratory / ablation / A-B / reverted experiments.
  - `result_tooling.csv` — NCU profiling, trace tooling, microbenches, research notes.
  Each row should have step, optimization, TTFT, throughput, num_samples, GPU, and enough notes to explain *why* and link to the result JSON.

- **Commit after each landed optimization.** Small commits make rollback trivial when a "win" turns out to regress on larger sample counts.

- **Results directories are the source of truth**, not stdout: JSON benchmark output → `results/`, traces → `traces/` (new comparisons get a subdirectory like `traces/compare/<label>`). Keep the repo root clean.

- **Before promoting a flag to default-on**, run at minimum a 100-sample benchmark and verify answers are bit-identical (or equivalent) vs the previous default. The pattern is visible throughout `result.csv`: step `XXa` does the A/B at 100 samples, `XXb` promotes the flag.

- **Historical banned optimization**: a prompt-prefix KV reuse trick between TTFT / throughput / accuracy calls (step 84) was removed per organizer rules on 2026-04-08. Do not resurrect it — it is referenced in `result.csv` for history only. `[historical commit]` is the removal commit.

- **Reference code available locally** (read-only, not ours): `./reference/vllm/`, `./reference/sglang/`, `./reference/llama.cpp/` — useful for checking how established engines schedule decode.

## Key docs

- [`docs/README.md`](docs/README.md) — index of active and stale design / analysis docs
- [`docs/decode_fusion_plan.md`](docs/decode_fusion_plan.md) — in-flight fusion priorities
- [`docs/decode_attention_optimization.md`](docs/decode_attention_optimization.md) — attention / split-KV route, NCU findings, failed experiments
- [`docs/decode_gemv_analysis.md`](docs/decode_gemv_analysis.md) — GEMV analysis and CUDA/fusion verdicts
- [`docs/nsys_profiling_guide.md`](docs/nsys_profiling_guide.md) — nsys trace capture & SQLite analysis workflow
- [`FILES.md`](FILES.md) — complete index of every benchmark/profile/tooling script
- [`AGENTS.md`](AGENTS.md) — project-wide working conventions (source of truth over anything here that drifts)
