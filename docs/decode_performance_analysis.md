# Decode Phase Performance Analysis (2026-04-11)

## 1. Overview

**Goal**: Profile decode phase, measure per-step timing, compute theoretical limits, identify optimization directions.

**Current Performance (CUDAGraph decode, batch=1, A800 80GB PCIe)**:
| Metric | Value |
|--------|-------|
| Throughput (5000-sample official) | 255.65 tok/s |
| Throughput (10-sample local) | 240.52 tok/s |
| CUDAGraph decode step (cudaEvent) | 3.362 ms/token = 297 tok/s |
| Theoretical roofline | 1.72 ms/token = 582 tok/s |
| **Efficiency vs roofline** | **51%** |

> Note: Benchmark throughput (240-256 tok/s) is lower than raw decode (297 tok/s) because
> it includes TTFT (prefill) amortized into the throughput metric.

---

## 2. Measurement Methodology

Three complementary approaches used:

1. **cudaEvent timing** (`measure_decode_step.py`): Direct measurement of CUDAGraph replay step time using `cudaEventRecord/ElapsedTime` pairs around each `graph.replay()`. Most accurate end-to-end per-token decode timing.

2. **nsys kernel profiling** (eager mode, `ENABLE_CUDAGRAPH=0`): Full kernel-level breakdown from Nsight Systems trace export. Provides per-kernel GPU time for each decode operation. Note: eager mode has large Python dispatch overhead (15.9 ms/step vs 3.4 ms CUDAGraph) but individual GPU kernel timings are representative.

3. **nsys full benchmark** (CUDAGraph enabled): Full trace without capture-range, capturing all kernels including CUDAGraph capture and replay. Provides CUDAGraph-mode kernel timings.

4. **Theoretical roofline** (`analyze_decode_nsys.py --theory-only`): Memory-bandwidth-bound analysis. For batch=1 decode, all operations are bandwidth-limited on A800. `time = bytes_transferred / peak_HBM_bandwidth`.

---

## 3. Per-Token CUDAGraph Decode Breakdown

Based on nsys full benchmark trace kernel averages (1792 persistent kernel calls = 64 decode tokens x 28 layers):

### 3.1 Per-Layer Operations (x28 layers per token)

| Operation | Kernel | us/layer | us/token | Theory us/tok | Eff. | % Total |
|-----------|--------|----------|----------|---------------|------|---------|
| **fused_post_attn_mlp** | persistent_k2048_k6144 | 55.5 | 1554 | 1037 | 67% | 46.2% |
| **qkv_proj** | nosplit_matvec_k2048 (g=1024) | 12.73 | 356 | 231 | 65% | 10.6% |
| **o_proj** | nosplit_matvec_k2048_row2 (g=1024) | 7.98 | 223 | 115 | 52% | 6.6% |
| **attention** | flash_fwd_splitkv + combine | ~13.5 | 378 | ~38* | ~10%* | 11.2% |
| **qk_norm_rope** | _fused_qk_rms_norm_rope (Triton) | 4.61 | 129 | 0.2 | <1% | 3.8% |
| **input_rmsnorm** | _rms_norm_fwd (Triton) | 3.61 | 101 | 0.2 | <1% | 3.0% |
| **kv_cache + misc** | CatArrayBatchedCopy + fill + etc. | ~5.5 | ~154 | ~0.1 | - | 4.6% |

### 3.2 Per-Token Operations (x1 per token)

| Operation | Kernel | us/token | Theory us/tok | Eff. | % Total |
|-----------|--------|----------|---------------|------|---------|
| **lm_head** | ampere_fp16_64x64_sliced | 368 | 305 | 83% | 10.9% |
| **argmax + overhead** | argmax + graph gaps | ~99 | ~0.3 | - | 2.9% |

### 3.3 Summary

| Category | us/token | % Total |
|----------|----------|---------|
| Weight GEMV (fused_mlp + qkv + o_proj + lm_head) | 2501 | 74.4% |
| Attention | 378 | 11.2% |
| Small kernels (rmsnorm, qk_norm_rope) | 230 | 6.8% |
| KV cache + fill + misc | 154 | 4.6% |
| Argmax + graph overhead | 99 | 2.9% |
| **Total** | **3362** | **100%** |

> \* Attention "theoretical" is pure KV cache bandwidth limit; actual attention involves compute (softmax, QK^T, xV)
> and memory access patterns that are inherently less efficient for M=1 single-query decode.

---

## 4. Theoretical Roofline Analysis

**A800 80GB PCIe**: HBM BW = 2039 GB/s, 108 SMs, FP16 Tensor Core 312 TFLOPS

All decode operations are **memory-bandwidth-bound** at batch=1. Theoretical time = bytes / bandwidth.

### 4.1 Weight-Dominated Operations (per token, 28 layers)

| Operation | Weight Size | Total Bytes | Theory us/tok | Actual us/tok | Eff. |
|-----------|-------------|-------------|---------------|---------------|------|
| gate_up_proj [12288x2048] | 48 MB/layer | 1.31 GB | 691.6 | ~980* | ~71%* |
| down_proj [2048x6144] | 24 MB/layer | 0.66 GB | 345.8 | ~574* | ~60%* |
| lm_head [151936x2048] | 593 MB | 0.59 GB | 305.4 | 368 | 83% |
| qkv_proj [4096x2048] | 16 MB/layer | 0.44 GB | 230.6 | 356 | 65% |
| o_proj [2048x2048] | 8 MB/layer | 0.22 GB | 115.3 | 223 | 52% |
| **Sub-total (weights)** | | **3.22 GB** | **1688.7** | **2501** | **68%** |

> \* gate_up and down_proj are inside fused_post_attn_mlp_persistent; individual times estimated
> from persistent kernel overhead decomposition.

### 4.2 Activation-Dominated Operations (per token)

| Operation | Data Size/layer | Total Bytes | Theory us/tok | Actual us/tok | Overhead |
|-----------|-----------------|-------------|---------------|---------------|----------|
| attention (S=700) | ~2.9 MB/layer | ~77 MB | 37.7 | 378 | 10x |
| qk_norm_rope | ~14 KB/layer | ~373 KB | 0.2 | 129 | 645x |
| input_rmsnorm | ~12 KB/layer | ~344 KB | 0.2 | 101 | 505x |
| kv_cache_update | ~4 KB/layer | ~115 KB | 0.1 | ~50 | 500x |
| residual_add | ~12 KB/layer | ~344 KB | 0.2 | fused | - |
| argmax | ~304 KB | ~304 KB | 0.15 | 12 | 80x |

> Small activation operations have extreme overhead ratios because:
> 1. GPU kernel launch latency (~3-5 us minimum) dominates when data is tiny
> 2. Even in CUDAGraph, each graph node has ~1-2 us replay overhead
> 3. Theoretical roofline assumes zero startup cost

### 4.3 Total Theoretical vs Actual

| Scenario | us/token | tok/s | vs Current |
|----------|----------|-------|------------|
| **Current (CUDAGraph)** | **3362** | **297** | **1.00x** |
| Theoretical non-fused | 1719 | 582 | 1.96x |
| Theoretical best-fused | 1718 | 582 | 1.96x |

> The theoretical limit doesn't change much with fusion because weight reads dominate (98% of bytes).
> Fusion's main benefit is **eliminating kernel launch overhead**, not reducing data movement.

---

## 5. Efficiency Gap Analysis

Total gap: **3362 - 1727 = 1635 us/token**

### 5.1 Gap Decomposition

| Source | Gap (us) | % of Gap | Explanation |
|--------|----------|----------|-------------|
| **fused_post_attn_mlp (67% eff.)** | 517 | 31.6% | Persistent kernel reads gate_up(48MB) + down_proj(24MB) + o_proj norms etc. per layer. Grid sync overhead between phases. Memory access pattern not fully coalesced across 108 SM cooperative grid. |
| **Attention overhead** | 340 | 20.8% | flash_fwd_splitkv designed for long-seq; single-query has poor tile utilization. Split-KV needs K passes + combine kernel. Softmax + normalization overhead per tile. |
| **qkv_proj GEMV (65% eff.)** | 125 | 7.6% | nosplit_matvec at 65% bandwidth. Grid launch overhead for 1024 blocks. Suboptimal warp-level reduction for K=2048. |
| **o_proj GEMV (52% eff.)** | 108 | 6.6% | row2 variant for small N=2048. Only 1024 blocks, low occupancy. k2048_row2 is the weakest GEMV variant. |
| **Small kernel overhead** | 230 | 14.1% | rmsnorm(101us) + qk_norm_rope(129us): data is <15KB/layer but kernel launch minimum is ~3.5us. 28 layers x 3.5us x 2 ops = 196us unavoidable. |
| **KV cache + misc** | 154 | 9.4% | CatArrayBatchedCopy for K+V scatter, elementwise fill ops, etc. |
| **lm_head gap** | 63 | 3.9% | cuBLAS at 83%, already near optimal for M=1 GEMV via GEMM. |
| **Graph overhead + gaps** | 98 | 6.0% | Inter-kernel gaps in CUDAGraph replay, argmax kernel overhead. |

### 5.2 Key Insight: Weight Read Bandwidth is THE Bottleneck

Weight reads account for **3.22 GB per decode token** (out of ~3.3 GB total memory traffic).

- At 2039 GB/s theoretical: 1.58 ms minimum
- At 68% average efficiency: 2.33 ms = 69% of actual decode time
- This means **74% of decode time is spent reading weight matrices from HBM**

The weight bandwidth efficiency (68%) already includes our CUDA nosplit GEMV + persistent fused kernel optimizations. Further improvement requires better memory access patterns or reducing total bytes read.

---

## 6. Optimization Opportunities (Ranked by Impact)

### Tier 1: High Impact (>100 us savings)

#### 6.1 Improve fused_post_attn_mlp_persistent efficiency (67% -> 80%)
- **Potential**: ~250 us/token (1554 * (1 - 0.67/0.80))
- **Current**: persistent_k2048_k6144 at 55.5 us/layer
- **Approaches**:
  - Optimize memory access pattern in cooperative kernel: better L2 cache utilization, streaming stores
  - Reduce grid_sync overhead between gate_up -> silu -> down_proj phases
  - Pipeline double-buffering: overlap next weight tile load with current compute
  - Investigate if warp-level scheduling can reduce bank conflicts
- **Risk**: Medium. Requires deep CUDA kernel optimization.

#### 6.2 Fuse o_proj into persistent kernel (fused_o_proj_post_attn_mlp_persistent)
- **Potential**: ~120 us/token (saves 223us o_proj GEMV, persistent gets ~100us slower from extra weight)
- **Current**: Experimental `fused_o_proj_post_attn_mlp_persistent` exists but not default
- **Approach**: Enable and tune the existing experimental kernel
- **Risk**: Low. Code exists, just needs validation and tuning.
- **Note**: This also eliminates 28 kernel launch gaps per token.

#### 6.3 Optimize decode attention for single-query
- **Potential**: ~150 us/token
- **Current**: flash_fwd_splitkv at ~14.37 us/layer (split + combine kernels)
- **Approaches**:
  - Custom single-query attention kernel: avoid split-KV overhead for small batch
  - Use existing Triton small_decode_attention variants (some already faster at ~12.66 us/layer)
  - PagedAttention-style flat KV cache for better coalescing
  - Explore tree-reduction attention for GQA (8 KV heads, 16 Q heads)
- **Risk**: Medium. Need to handle varying seq_len CUDAGraph buckets.

#### 6.4 Fuse small kernels (rmsnorm + qkv, or rmsnorm + qk_norm_rope)
- **Potential**: ~100 us/token
- **Current**: input_rmsnorm (101us) and qk_norm_rope (129us) dominated by launch overhead
- **Approaches**:
  - Fuse input_rmsnorm into qkv GEMV kernel (add normalization in GEMV prologue)
  - Fuse qk_norm_rope + kv_cache_update into single kernel
  - Reduce total kernel count per layer from 7+ to 4-5
- **Risk**: High. Requires significant kernel redesign.

### Tier 2: Medium Impact (50-100 us savings)

#### 6.5 Improve qkv GEMV efficiency (65% -> 80%)
- **Potential**: ~80 us/token
- **Approaches**:
  - Optimize nosplit_matvec_k2048 for N=4096: better warp reduction
  - Use persistent GEMV for QKV (avoid launch overhead)
  - Async weight prefetch with L2 residency hints
- **Risk**: Medium. Previous GEMV analysis shows we're near diminishing returns.

#### 6.6 Improve o_proj GEMV efficiency (52% -> 75%)
- **Potential**: ~70 us/token (if not fused into persistent)
- **Current**: k2048_row2 variant is weakest at 52% bandwidth
- **Approaches**:
  - Tune row2 variant for N=2048 specifically
  - Try row4 variant or non-row-blocked GEMV
  - Or: just fuse into persistent kernel (6.2)
- **Risk**: Low-Medium.

### Tier 3: Lower Impact (<50 us savings)

#### 6.7 lm_head optimization (83% -> 90%)
- **Potential**: ~30 us/token
- **Approaches**: Replace cuBLAS with custom GEMV for [151936 x 2048]; partial vocab (top-K decode)
- **Risk**: High for little gain.

#### 6.8 Reduce CUDAGraph overhead
- **Potential**: ~30 us/token
- **Approaches**: Fewer graph nodes (kernel fusion reduces this), larger graph buckets
- **Risk**: Low.

---

## 7. Combined Optimization Scenarios

| Scenario | Changes | Est. us/tok | Est. tok/s | Improvement |
|----------|---------|-------------|------------|-------------|
| **Current** | - | 3362 | 297 | 1.00x |
| **Conservative** | 6.2 (fuse o_proj) + 6.3 (better attn) | ~2990 | ~335 | 1.13x |
| **Moderate** | Above + 6.1 (persistent 80%) + 6.5 (qkv) | ~2650 | ~377 | 1.27x |
| **Aggressive** | Above + 6.4 (fuse small kernels) + 6.6 | ~2400 | ~417 | 1.40x |
| **Theoretical limit** | 100% HBM bandwidth | 1720 | 582 | 1.96x |

> For benchmark throughput: raw decode tok/s -> benchmark throughput requires accounting for TTFT.
> With TTFT ~36ms and ~50 token generation avg: benchmark_throughput = 50 / (0.036 + 50/decode_tok_s)
> Conservative: ~310 tok/s benchmark, Moderate: ~340 tok/s, Aggressive: ~370 tok/s

---

## 8. Recommended Priority Order

1. **[Quick win] Enable fused_o_proj_post_attn_mlp_persistent** (6.2)
   - Code already exists, just validate correctness + benchmark
   - Expected: ~120 us/token saving

2. **[Medium effort] Tune persistent kernel memory access** (6.1)
   - Profile with NCU to identify specific bottleneck (L2 miss rate, bank conflicts, warp stalls)
   - Expected: 150-250 us/token saving

3. **[Medium effort] Optimize decode attention** (6.3)
   - Try enabling Triton splitkv variants uniformly across graph buckets
   - Consider custom GQA attention for exact (16Q, 8KV, dim=128) configuration
   - Expected: 100-150 us/token saving

4. **[Lower priority] Kernel fusion for small ops** (6.4)
   - Only worthwhile after bigger gains are captured
   - High implementation effort for ~100 us return

---

## Appendix A: Raw Kernel Timing Data

### A.1 CUDAGraph Mode (from nsys full benchmark trace)

```
Operation                                      Count   Avg(us)  Total(ms)
fused_post_attn_mlp_persistent_k2048_k6144      1792    56.16    100.63
nosplit_matvec_k2048_hfma2 (grid=3072,gate_up)  1792    32.92     58.99
nosplit_matvec_k2048_hfma2 (grid=1024,qkv)      3584    12.73     45.61
nosplit_matvec_k6144 (down_proj)                 1792    17.92     32.11
nosplit_matvec_k2048_row2 (o_proj)               3584     7.98     28.60
CatArrayBatchedCopy (kv_cache)                   4091     6.99     28.59
flash_fwd_splitkv_kernel (decode attn)           1792     8.33     14.93
_small_decode_attn_splitkv (Triton)              1400     9.65     13.51
flash_fwd_splitkv_combine                        1792     6.04     10.82
_fused_silu_mul_packed                           1904     5.56     10.59
_fused_rms_norm_rope (vision)                    3584     2.72      9.73
_fused_qk_rms_norm_rope                          1904     4.61      8.78
_rms_norm_fwd (input_rmsnorm)                    1904     3.61      6.87
_fused_add_rms_norm                              1904     3.01      5.74
ampere_fp16_64x64 (lm_head)                       132   367.87     48.56
```

### A.2 Persistent Kernel Stability

```
Capture phase (first 896 calls): avg = 55.78 us/layer
Replay phase (remaining 896 calls): avg = 55.53 us/layer
Range: 51.0 - 66.0 us/layer (max during first few warmup calls)
```

### A.3 Eager Mode (Python dispatch overhead)

```
CUDAGraph decode: 3.362 ms/token (297 tok/s)
Eager decode (GPU kernel only): 6.8 ms/token
Eager decode (wall clock): 15.85 ms/token
→ CUDAGraph eliminates ~4.7x Python dispatch overhead
```

### A.4 Theoretical Roofline Constants

```
A800 80GB PCIe HBM BW: 2039 GB/s
Qwen3-VL-2B: hidden=2048, heads=16, kv_heads=8, head_dim=128,
             intermediate=6144, vocab=151936, layers=28
Total weight bytes per decode token: ~3.22 GB
Total KV cache bytes per token (S=700): ~77 MB
```
