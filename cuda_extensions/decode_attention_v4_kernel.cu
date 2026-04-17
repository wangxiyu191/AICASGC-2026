// decode_attention_v4_kernel.cu — tiled two-pass decode attention, smem-free
//
// Architecture: thread-group=4, 2 warps (one per q-head)
//
// Key design: TILED two-pass within each 64-token tile:
//   Pass 1 (K): compute scores, store in 8 register slots
//   Cross-group max via 3-round shfl_xor(4,8,16)
//   ONE rescale per tile (same frequency as Triton & v3)
//   Pass 2 (V): load V, weighted accumulate using stored scores
//
// vs v3: NO smem, NO __syncthreads(), scores in registers
// vs v4-prev: fewer rescales via tiled processing

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <cuda_fp16.h>

namespace {

constexpr int kHeadDim      = 128;
constexpr int kWarpSize     = 32;
constexpr int kThreadGroup  = 4;
constexpr int kGroupsPerWarp = kWarpSize / kThreadGroup;  // 8
constexpr int kDimsPerThread = kHeadDim / kThreadGroup;   // 32
constexpr int kTileSteps    = 8;                          // steps per tile
constexpr int kTileN        = kGroupsPerWarp * kTileSteps; // 64 tokens per tile
constexpr int kQHeadsPerKV  = 2;
constexpr int kCTALocalWarpsPerQHead = 4;
constexpr int kCTALocalWarps = kQHeadsPerKV * kCTALocalWarpsPerQHead;
constexpr int kCTAQHeadWarpsDefault = 4;
constexpr int kCTAQHeadWarpsWide = 8;
constexpr int kCTAQHeadWarpsXL = 16;
constexpr int kThreadGroupWide  = 8;
constexpr int kGroupsPerWarpWide = kWarpSize / kThreadGroupWide; // 4
constexpr int kDimsPerThreadWide = kHeadDim / kThreadGroupWide;  // 16
constexpr int kTileStepsWide    = 16;
constexpr int kTileNWide        = kGroupsPerWarpWide * kTileStepsWide; // 64 tokens
constexpr int kContigDimsPerLane = kHeadDim / kWarpSize; // 4
constexpr int kPrefetchLineBytes = 128;
constexpr float kNegLarge   = -1.0e20f;

__device__ __forceinline__ void prefetch_global_l2(const void* ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr));
#else
    (void)ptr;
#endif
}

__device__ __forceinline__ void prefetch_weight_lines_l2(
    const __half* weight,
    int64_t weight_numel,
    int prefetch_block_idx,
    int prefetch_blocks) {
    if (weight == nullptr || weight_numel <= 0 || prefetch_blocks <= 0) {
        return;
    }

    const char* weight_bytes = reinterpret_cast<const char*>(weight);
    const int64_t total_bytes = weight_numel * static_cast<int64_t>(sizeof(__half));
    const int64_t total_lines = (total_bytes + kPrefetchLineBytes - 1) / kPrefetchLineBytes;
    const int64_t thread_idx =
        static_cast<int64_t>(prefetch_block_idx) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(prefetch_blocks) * blockDim.x;

    for (int64_t line_idx = thread_idx; line_idx < total_lines; line_idx += stride) {
        prefetch_global_l2(weight_bytes + line_idx * kPrefetchLineBytes);
    }
}

__device__ __forceinline__ uint32_t smem_u32addr(const void* ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
#else
    return 0;
#endif
}

__device__ __forceinline__ void cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32addr(smem_ptr)), "l"(gmem_ptr));
#else
    *reinterpret_cast<uint4*>(smem_ptr) = *reinterpret_cast<const uint4*>(gmem_ptr);
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

__device__ __forceinline__ float warp_sum_full(float val) {
    #pragma unroll
    for (int offset = kWarpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_max_full(float val) {
    #pragma unroll
    for (int offset = kWarpSize >> 1; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ void load_half4_vec(
    const __half* ptr,
    float& x0,
    float& x1,
    float& x2,
    float& x3) {
    const uint2 d = __ldg(reinterpret_cast<const uint2*>(ptr));
    const __half* h = reinterpret_cast<const __half*>(&d);
    x0 = __half2float(h[0]);
    x1 = __half2float(h[1]);
    x2 = __half2float(h[2]);
    x3 = __half2float(h[3]);
}

// ---------------------------------------------------------------------------
// Forward kernel — tiled two-pass, always 64-token tiles
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(64)
decode_attn_v4_fwd_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const __half* __restrict__ prefetch_weight,
    const int64_t* __restrict__ cache_position,
    float* __restrict__ partial_acc,
    float* __restrict__ partial_stats,
    int64_t q_stride_h,
    int64_t k_stride_h, int64_t k_stride_s,
    int64_t v_stride_h, int64_t v_stride_s,
    int64_t pa_stride_kh, int64_t pa_stride_split, int64_t pa_stride_head,
    int64_t ps_stride_kh, int64_t ps_stride_split,
    int64_t ps_stride_head, int64_t ps_stride_stat,
    int64_t prefetch_weight_numel,
    float scale,
    int max_seq_len,
    int split_k,
    int attention_blocks) {

    const int linear_block_idx = blockIdx.x;
    if (linear_block_idx >= attention_blocks) {
        prefetch_weight_lines_l2(
            prefetch_weight,
            prefetch_weight_numel,
            linear_block_idx - attention_blocks,
            gridDim.x - attention_blocks);
        return;
    }

    const int kv_head_idx = linear_block_idx / split_k;
    const int split_idx  = linear_block_idx - kv_head_idx * split_k;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int group_id   = lane / kThreadGroup;   // 0-7
    const int group_lane = lane % kThreadGroup;   // 0-3

    // Chunk boundaries
    const int valid_len  = min(static_cast<int>(cache_position[0] + 1), max_seq_len);
    const int chunk_size = (valid_len + split_k - 1) / split_k;
    const int chunk_start = split_idx * chunk_size;
    const int chunk_end   = min(chunk_start + chunk_size, valid_len);

    const int dim_offset = group_lane * kDimsPerThread;

    // Handle empty chunks
    if (chunk_start >= chunk_end) {
        if (group_id == 0) {
            float* pa = partial_acc + kv_head_idx * pa_stride_kh
                      + split_idx * pa_stride_split
                      + warp_id * pa_stride_head + dim_offset;
            #pragma unroll
            for (int i = 0; i < kDimsPerThread; ++i) pa[i] = 0.0f;
            if (group_lane == 0) {
                float* ps = partial_stats + kv_head_idx * ps_stride_kh
                          + split_idx * ps_stride_split
                          + warp_id * ps_stride_head;
                ps[0] = 0.0f;
                ps[1 * ps_stride_stat] = kNegLarge;
            }
        }
        return;
    }

    // Load Q for this warp's Q-head
    const int q_head_idx = kv_head_idx * 2 + warp_id;
    float q_reg[kDimsPerThread];
    {
        const __half* qp = q + q_head_idx * q_stride_h + dim_offset;
        #pragma unroll
        for (int i = 0; i < kDimsPerThread; i += 8) {
            uint4 d = __ldg(reinterpret_cast<const uint4*>(qp + i));
            const __half* h = reinterpret_cast<const __half*>(&d);
            #pragma unroll
            for (int j = 0; j < 8; ++j) q_reg[i + j] = __half2float(h[j]);
        }
    }

    // Online softmax state
    float m_val = kNegLarge;
    float l_val = 0.0f;
    float acc[kDimsPerThread];
    #pragma unroll
    for (int i = 0; i < kDimsPerThread; ++i) acc[i] = 0.0f;

    const __half* k_base = k + kv_head_idx * k_stride_h;
    const __half* v_base = v + kv_head_idx * v_stride_h;

    // ---- Main loop: process 64 tokens per tile ----
    for (int tile_start = chunk_start; tile_start < chunk_end; tile_start += kTileN) {
        const int tile_end = min(tile_start + kTileN, chunk_end);

        // ==== Pass 1: K dot products -> scores in registers ====
        float local_scores[kTileSteps];
        float tile_max = kNegLarge;

        #pragma unroll
        for (int step = 0; step < kTileSteps; ++step) {
            const int t_global = tile_start + step * kGroupsPerWarp + group_id;
            const bool valid = (t_global < tile_end);

            float dot = 0.0f;
            if (valid) {
                const __half* kp = k_base + t_global * k_stride_s + dim_offset;
                #pragma unroll
                for (int i = 0; i < kDimsPerThread; i += 8) {
                    uint4 d = __ldg(reinterpret_cast<const uint4*>(kp + i));
                    const __half* h = reinterpret_cast<const __half*>(&d);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) dot += q_reg[i + j] * __half2float(h[j]);
                }
            }
            // Intra-group reduce
            dot += __shfl_xor_sync(0xffffffff, dot, 1);
            dot += __shfl_xor_sync(0xffffffff, dot, 2);

            float score = valid ? (dot * scale) : kNegLarge;
            local_scores[step] = score;
            tile_max = fmaxf(tile_max, score);
        }

        // ==== Cross-group max via shfl (3 rounds) ====
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 4));
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 8));
        tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 16));

        // ==== ONE rescale per tile ====
        float new_m = fmaxf(m_val, tile_max);
        float alpha = expf(m_val - new_m);
        #pragma unroll
        for (int i = 0; i < kDimsPerThread; ++i) acc[i] *= alpha;
        l_val *= alpha;
        m_val = new_m;

        // ==== Pass 2: V load + weighted accumulate ====
        #pragma unroll
        for (int step = 0; step < kTileSteps; ++step) {
            const int t_global = tile_start + step * kGroupsPerWarp + group_id;
            const bool valid = (t_global < tile_end);

            if (valid) {
                float p = expf(local_scores[step] - m_val);
                l_val += p;
                const __half* vp = v_base + t_global * v_stride_s + dim_offset;
                #pragma unroll
                for (int i = 0; i < kDimsPerThread; i += 8) {
                    uint4 d = __ldg(reinterpret_cast<const uint4*>(vp + i));
                    const __half* h = reinterpret_cast<const __half*>(&d);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) acc[i + j] += p * __half2float(h[j]);
                }
            }
        }
    }

    // ---- Merge 8 thread groups via log-step shfl ----
    #pragma unroll
    for (int xor_mask = kThreadGroup; xor_mask < kWarpSize; xor_mask <<= 1) {
        float pm = __shfl_xor_sync(0xffffffff, m_val, xor_mask);
        float pl = __shfl_xor_sync(0xffffffff, l_val, xor_mask);
        float nm  = fmaxf(m_val, pm);
        float s_me = expf(m_val - nm);
        float s_pa = expf(pm - nm);
        #pragma unroll
        for (int i = 0; i < kDimsPerThread; ++i) {
            float pa = __shfl_xor_sync(0xffffffff, acc[i], xor_mask);
            acc[i] = acc[i] * s_me + pa * s_pa;
        }
        l_val = l_val * s_me + pl * s_pa;
        m_val = nm;
    }

    // ---- Write partial results ----
    if (group_id == 0) {
        float* pa = partial_acc + kv_head_idx * pa_stride_kh
                  + split_idx * pa_stride_split
                  + warp_id * pa_stride_head + dim_offset;
        #pragma unroll
        for (int i = 0; i < kDimsPerThread; ++i) pa[i] = acc[i];
        if (group_lane == 0) {
            float* ps = partial_stats + kv_head_idx * ps_stride_kh
                      + split_idx * ps_stride_split
                      + warp_id * ps_stride_head;
            ps[0] = l_val;
            ps[1 * ps_stride_stat] = m_val;
        }
    }
}

// ---------------------------------------------------------------------------
// Merge kernel: combines split-K partials (same as v3)
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(128)
decode_attn_v4_merge_kernel(
    const float* __restrict__ partial_acc,
    const float* __restrict__ partial_stats,
    __half* __restrict__ out,
    int64_t pa_stride_kh, int64_t pa_stride_split, int64_t pa_stride_head,
    int64_t ps_stride_kh, int64_t ps_stride_split,
    int64_t ps_stride_head, int64_t ps_stride_stat,
    int64_t out_stride_h,
    int split_k) {

    const int kv_head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= kHeadDim) return;

    float max0 = kNegLarge, max1 = kNegLarge;
    float sum0 = 0.0f, sum1 = 0.0f;
    float a0 = 0.0f, a1 = 0.0f;

    for (int s = 0; s < split_k; ++s) {
        const int64_t ps_base = kv_head_idx * ps_stride_kh + s * ps_stride_split;
        const int64_t pa_base = kv_head_idx * pa_stride_kh + s * pa_stride_split;

        const float lv0 = partial_stats[ps_base + 0 * ps_stride_head];
        const float mv0 = partial_stats[ps_base + 0 * ps_stride_head + 1 * ps_stride_stat];
        const float lv1 = partial_stats[ps_base + 1 * ps_stride_head];
        const float mv1 = partial_stats[ps_base + 1 * ps_stride_head + 1 * ps_stride_stat];
        const float pv0 = partial_acc[pa_base + 0 * pa_stride_head + tid];
        const float pv1 = partial_acc[pa_base + 1 * pa_stride_head + tid];

        const float nm0 = fmaxf(max0, mv0);
        const float nm1 = fmaxf(max1, mv1);
        a0 = a0 * expf(max0 - nm0) + pv0 * expf(mv0 - nm0);
        a1 = a1 * expf(max1 - nm1) + pv1 * expf(mv1 - nm1);
        sum0 = sum0 * expf(max0 - nm0) + lv0 * expf(mv0 - nm0);
        sum1 = sum1 * expf(max1 - nm1) + lv1 * expf(mv1 - nm1);
        max0 = nm0;
        max1 = nm1;
    }

    const int q0 = kv_head_idx * 2;
    out[q0 * out_stride_h + tid] = __float2half(a0 / sum0);
    out[(q0 + 1) * out_stride_h + tid] = __float2half(a1 / sum1);
}

// ---------------------------------------------------------------------------
// Fused kernel: fwd + per-kv-head atomic last-block merge (single launch)
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(64)
decode_attn_v4_fused_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const __half* __restrict__ prefetch_weight,
    const int64_t* __restrict__ cache_position,
    float* __restrict__ partial_acc,
    float* __restrict__ partial_stats,
    __half* __restrict__ out,
    int* __restrict__ done_count,
    int64_t q_stride_h,
    int64_t k_stride_h, int64_t k_stride_s,
    int64_t v_stride_h, int64_t v_stride_s,
    int64_t pa_stride_kh, int64_t pa_stride_split, int64_t pa_stride_head,
    int64_t ps_stride_kh, int64_t ps_stride_split,
    int64_t ps_stride_head, int64_t ps_stride_stat,
    int64_t out_stride_h,
    int64_t prefetch_weight_numel,
    float scale,
    int max_seq_len,
    int split_k,
    int attention_blocks) {

    const int linear_block_idx = blockIdx.x;
    if (linear_block_idx >= attention_blocks) {
        prefetch_weight_lines_l2(
            prefetch_weight,
            prefetch_weight_numel,
            linear_block_idx - attention_blocks,
            gridDim.x - attention_blocks);
        return;
    }

    const int kv_head_idx = linear_block_idx / split_k;
    const int split_idx  = linear_block_idx - kv_head_idx * split_k;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int group_id   = lane / kThreadGroup;
    const int group_lane = lane % kThreadGroup;
    const int dim_offset = group_lane * kDimsPerThread;

    const int valid_len  = min(static_cast<int>(cache_position[0] + 1), max_seq_len);
    const int chunk_size = (valid_len + split_k - 1) / split_k;
    const int chunk_start = split_idx * chunk_size;
    const int chunk_end   = min(chunk_start + chunk_size, valid_len);

    if (chunk_start >= chunk_end) {
        // Empty chunk: write zero partials
        if (group_id == 0) {
            float* pa = partial_acc + kv_head_idx * pa_stride_kh
                      + split_idx * pa_stride_split
                      + warp_id * pa_stride_head + dim_offset;
            #pragma unroll
            for (int i = 0; i < kDimsPerThread; ++i) pa[i] = 0.0f;
            if (group_lane == 0) {
                float* ps = partial_stats + kv_head_idx * ps_stride_kh
                          + split_idx * ps_stride_split
                          + warp_id * ps_stride_head;
                ps[0] = 0.0f;
                ps[1 * ps_stride_stat] = kNegLarge;
            }
        }
    } else {
        // ---- Full fwd computation (identical to fwd kernel) ----
        const int q_head_idx = kv_head_idx * 2 + warp_id;
        float q_reg[kDimsPerThread];
        {
            const __half* qp = q + q_head_idx * q_stride_h + dim_offset;
            #pragma unroll
            for (int i = 0; i < kDimsPerThread; i += 8) {
                uint4 d = __ldg(reinterpret_cast<const uint4*>(qp + i));
                const __half* h = reinterpret_cast<const __half*>(&d);
                #pragma unroll
                for (int j = 0; j < 8; ++j) q_reg[i + j] = __half2float(h[j]);
            }
        }

        float m_val = kNegLarge;
        float l_val = 0.0f;
        float acc[kDimsPerThread];
        #pragma unroll
        for (int i = 0; i < kDimsPerThread; ++i) acc[i] = 0.0f;

        const __half* k_base = k + kv_head_idx * k_stride_h;
        const __half* v_base = v + kv_head_idx * v_stride_h;

        for (int tile_start = chunk_start; tile_start < chunk_end; tile_start += kTileN) {
            const int tile_end = min(tile_start + kTileN, chunk_end);

            float local_scores[kTileSteps];
            float tile_max = kNegLarge;

            #pragma unroll
            for (int step = 0; step < kTileSteps; ++step) {
                const int t_global = tile_start + step * kGroupsPerWarp + group_id;
                const bool valid = (t_global < tile_end);

                float dot = 0.0f;
                if (valid) {
                    const __half* kp = k_base + t_global * k_stride_s + dim_offset;
                    #pragma unroll
                    for (int i = 0; i < kDimsPerThread; i += 8) {
                        uint4 d = __ldg(reinterpret_cast<const uint4*>(kp + i));
                        const __half* h = reinterpret_cast<const __half*>(&d);
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) dot += q_reg[i + j] * __half2float(h[j]);
                    }
                }
                dot += __shfl_xor_sync(0xffffffff, dot, 1);
                dot += __shfl_xor_sync(0xffffffff, dot, 2);

                float score = valid ? (dot * scale) : kNegLarge;
                local_scores[step] = score;
                tile_max = fmaxf(tile_max, score);
            }

            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 4));
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 8));
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 16));

            float new_m = fmaxf(m_val, tile_max);
            float alpha = expf(m_val - new_m);
            #pragma unroll
            for (int i = 0; i < kDimsPerThread; ++i) acc[i] *= alpha;
            l_val *= alpha;
            m_val = new_m;

            #pragma unroll
            for (int step = 0; step < kTileSteps; ++step) {
                const int t_global = tile_start + step * kGroupsPerWarp + group_id;
                const bool valid = (t_global < tile_end);

                if (valid) {
                    float p = expf(local_scores[step] - m_val);
                    l_val += p;
                    const __half* vp = v_base + t_global * v_stride_s + dim_offset;
                    #pragma unroll
                    for (int i = 0; i < kDimsPerThread; i += 8) {
                        uint4 d = __ldg(reinterpret_cast<const uint4*>(vp + i));
                        const __half* h = reinterpret_cast<const __half*>(&d);
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) acc[i + j] += p * __half2float(h[j]);
                    }
                }
            }
        }

        // Merge 8 thread groups via log-step shfl
        #pragma unroll
        for (int xor_mask = kThreadGroup; xor_mask < kWarpSize; xor_mask <<= 1) {
            float pm = __shfl_xor_sync(0xffffffff, m_val, xor_mask);
            float pl = __shfl_xor_sync(0xffffffff, l_val, xor_mask);
            float nm  = fmaxf(m_val, pm);
            float s_me = expf(m_val - nm);
            float s_pa = expf(pm - nm);
            #pragma unroll
            for (int i = 0; i < kDimsPerThread; ++i) {
                float pa = __shfl_xor_sync(0xffffffff, acc[i], xor_mask);
                acc[i] = acc[i] * s_me + pa * s_pa;
            }
            l_val = l_val * s_me + pl * s_pa;
            m_val = nm;
        }

        // Write partial results
        if (group_id == 0) {
            float* pa = partial_acc + kv_head_idx * pa_stride_kh
                      + split_idx * pa_stride_split
                      + warp_id * pa_stride_head + dim_offset;
            #pragma unroll
            for (int i = 0; i < kDimsPerThread; ++i) pa[i] = acc[i];
            if (group_lane == 0) {
                float* ps = partial_stats + kv_head_idx * ps_stride_kh
                          + split_idx * ps_stride_split
                          + warp_id * ps_stride_head;
                ps[0] = l_val;
                ps[1 * ps_stride_stat] = m_val;
            }
        }
    }

    // ---- Per-kv-head atomic: determine if this is the last block ----
    __threadfence();
    __syncthreads();

    __shared__ int s_is_last;
    if (tid == 0) {
        int arrived = atomicAdd(&done_count[kv_head_idx], 1);
        s_is_last = (arrived == split_k - 1) ? 1 : 0;
    }
    __syncthreads();

    if (!s_is_last) return;

    // ---- Last block: reset counter + merge all split_k partials ----
    if (tid == 0) {
        done_count[kv_head_idx] = 0;
    }
    __threadfence();

    // Each warp merges one q_head (warp0 -> head0, warp1 -> head1)
    // 32 lanes * 4 dims = 128 dims per warp
    const int merge_dim = lane * 4;
    const int head_off = warp_id;

    float m_merge = kNegLarge;
    float l_merge = 0.0f;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    for (int s = 0; s < split_k; ++s) {
        const int64_t ps_base = kv_head_idx * ps_stride_kh
                              + s * ps_stride_split
                              + head_off * ps_stride_head;
        const int64_t pa_base = kv_head_idx * pa_stride_kh
                              + s * pa_stride_split
                              + head_off * pa_stride_head;

        float lv = partial_stats[ps_base];
        float mv = partial_stats[ps_base + ps_stride_stat];

        float nm = fmaxf(m_merge, mv);
        float s_old = expf(m_merge - nm);
        float s_new = expf(mv - nm);

        const float4 pv = *reinterpret_cast<const float4*>(
            partial_acc + pa_base + merge_dim);
        a0 = a0 * s_old + pv.x * s_new;
        a1 = a1 * s_old + pv.y * s_new;
        a2 = a2 * s_old + pv.z * s_new;
        a3 = a3 * s_old + pv.w * s_new;

        l_merge = l_merge * s_old + lv * s_new;
        m_merge = nm;
    }

    // Write final output as half
    const int q_head = kv_head_idx * 2 + head_off;
    __half* out_ptr = out + q_head * out_stride_h + merge_dim;
    float inv_l = __fdividef(1.0f, l_merge);
    __half2 h01 = __floats2half2_rn(a0 * inv_l, a1 * inv_l);
    __half2 h23 = __floats2half2_rn(a2 * inv_l, a3 * inv_l);
    *reinterpret_cast<__half2*>(out_ptr)     = h01;
    *reinterpret_cast<__half2*>(out_ptr + 2) = h23;
}

// ---------------------------------------------------------------------------
// CTA-local experiment: one CTA owns one KV head, does intra-CTA shard merge
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kCTALocalWarps * kWarpSize)
decode_attn_v4_cta_local_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const int64_t* __restrict__ cache_position,
    __half* __restrict__ out,
    int64_t q_stride_h,
    int64_t k_stride_h, int64_t k_stride_s,
    int64_t v_stride_h, int64_t v_stride_s,
    int64_t out_stride_h,
    float scale,
    int max_seq_len) {

    const int kv_head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int q_local_idx = warp_id / kCTALocalWarpsPerQHead;
    const int shard_idx = warp_id - q_local_idx * kCTALocalWarpsPerQHead;
    const int group_id = lane / kThreadGroup;
    const int group_lane = lane % kThreadGroup;
    const int dim_offset = group_lane * kDimsPerThread;
    const int valid_len = min(static_cast<int>(cache_position[0] + 1), max_seq_len);

    __shared__ float s_partial_acc[kQHeadsPerKV][kCTALocalWarpsPerQHead][kHeadDim];
    __shared__ float s_partial_l[kQHeadsPerKV][kCTALocalWarpsPerQHead];
    __shared__ float s_partial_m[kQHeadsPerKV][kCTALocalWarpsPerQHead];

    float m_val = kNegLarge;
    float l_val = 0.0f;
    float acc[kDimsPerThread];
    #pragma unroll
    for (int i = 0; i < kDimsPerThread; ++i) {
        acc[i] = 0.0f;
    }

    const int shard_size = (valid_len + kCTALocalWarpsPerQHead - 1) / kCTALocalWarpsPerQHead;
    const int shard_start = shard_idx * shard_size;
    const int shard_end = min(shard_start + shard_size, valid_len);

    if (shard_start < shard_end) {
        const int q_head_idx = kv_head_idx * kQHeadsPerKV + q_local_idx;
        float q_reg[kDimsPerThread];
        const __half* qp = q + q_head_idx * q_stride_h + dim_offset;
        #pragma unroll
        for (int i = 0; i < kDimsPerThread; i += 8) {
            uint4 d = __ldg(reinterpret_cast<const uint4*>(qp + i));
            const __half* h = reinterpret_cast<const __half*>(&d);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                q_reg[i + j] = __half2float(h[j]);
            }
        }

        const __half* k_base = k + kv_head_idx * k_stride_h;
        const __half* v_base = v + kv_head_idx * v_stride_h;

        for (int tile_start = shard_start; tile_start < shard_end; tile_start += kTileN) {
            const int tile_end = min(tile_start + kTileN, shard_end);
            float local_scores[kTileSteps];
            float tile_max = kNegLarge;

            #pragma unroll
            for (int step = 0; step < kTileSteps; ++step) {
                const int t_global = tile_start + step * kGroupsPerWarp + group_id;
                const bool valid = (t_global < tile_end);

                float dot = 0.0f;
                if (valid) {
                    const __half* kp = k_base + t_global * k_stride_s + dim_offset;
                    #pragma unroll
                    for (int i = 0; i < kDimsPerThread; i += 8) {
                        uint4 d = __ldg(reinterpret_cast<const uint4*>(kp + i));
                        const __half* h = reinterpret_cast<const __half*>(&d);
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            dot += q_reg[i + j] * __half2float(h[j]);
                        }
                    }
                }
                dot += __shfl_xor_sync(0xffffffff, dot, 1);
                dot += __shfl_xor_sync(0xffffffff, dot, 2);

                const float score = valid ? (dot * scale) : kNegLarge;
                local_scores[step] = score;
                tile_max = fmaxf(tile_max, score);
            }

            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 4));
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 8));
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 16));

            const float new_m = fmaxf(m_val, tile_max);
            const float alpha = expf(m_val - new_m);
            #pragma unroll
            for (int i = 0; i < kDimsPerThread; ++i) {
                acc[i] *= alpha;
            }
            l_val *= alpha;
            m_val = new_m;

            #pragma unroll
            for (int step = 0; step < kTileSteps; ++step) {
                const int t_global = tile_start + step * kGroupsPerWarp + group_id;
                const bool valid = (t_global < tile_end);
                if (!valid) {
                    continue;
                }

                const float p = expf(local_scores[step] - m_val);
                l_val += p;
                const __half* vp = v_base + t_global * v_stride_s + dim_offset;
                #pragma unroll
                for (int i = 0; i < kDimsPerThread; i += 8) {
                    uint4 d = __ldg(reinterpret_cast<const uint4*>(vp + i));
                    const __half* h = reinterpret_cast<const __half*>(&d);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        acc[i + j] += p * __half2float(h[j]);
                    }
                }
            }
        }

        #pragma unroll
        for (int xor_mask = kThreadGroup; xor_mask < kWarpSize; xor_mask <<= 1) {
            const float pm = __shfl_xor_sync(0xffffffff, m_val, xor_mask);
            const float pl = __shfl_xor_sync(0xffffffff, l_val, xor_mask);
            const float nm = fmaxf(m_val, pm);
            const float s_me = expf(m_val - nm);
            const float s_pa = expf(pm - nm);
            #pragma unroll
            for (int i = 0; i < kDimsPerThread; ++i) {
                const float pa = __shfl_xor_sync(0xffffffff, acc[i], xor_mask);
                acc[i] = acc[i] * s_me + pa * s_pa;
            }
            l_val = l_val * s_me + pl * s_pa;
            m_val = nm;
        }
    }

    if (group_id == 0) {
        float* sm_acc = &s_partial_acc[q_local_idx][shard_idx][dim_offset];
        #pragma unroll
        for (int i = 0; i < kDimsPerThread; ++i) {
            sm_acc[i] = acc[i];
        }
        if (group_lane == 0) {
            s_partial_l[q_local_idx][shard_idx] = l_val;
            s_partial_m[q_local_idx][shard_idx] = m_val;
        }
    }

    __syncthreads();

    if (shard_idx != 0) {
        return;
    }

    const int merge_dim = lane * 4;
    float m_merge = kNegLarge;
    float l_merge = 0.0f;
    float a0 = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;

    #pragma unroll
    for (int s = 0; s < kCTALocalWarpsPerQHead; ++s) {
        const float lv = s_partial_l[q_local_idx][s];
        const float mv = s_partial_m[q_local_idx][s];
        const float nm = fmaxf(m_merge, mv);
        const float s_old = expf(m_merge - nm);
        const float s_new = expf(mv - nm);

        const float* pv = &s_partial_acc[q_local_idx][s][merge_dim];
        a0 = a0 * s_old + pv[0] * s_new;
        a1 = a1 * s_old + pv[1] * s_new;
        a2 = a2 * s_old + pv[2] * s_new;
        a3 = a3 * s_old + pv[3] * s_new;
        l_merge = l_merge * s_old + lv * s_new;
        m_merge = nm;
    }

    const int q_head = kv_head_idx * kQHeadsPerKV + q_local_idx;
    __half* out_ptr = out + q_head * out_stride_h + merge_dim;
    if (l_merge > 0.0f) {
        const float inv_l = __fdividef(1.0f, l_merge);
        *reinterpret_cast<__half2*>(out_ptr) = __floats2half2_rn(a0 * inv_l, a1 * inv_l);
        *reinterpret_cast<__half2*>(out_ptr + 2) = __floats2half2_rn(a2 * inv_l, a3 * inv_l);
    } else {
        *reinterpret_cast<__half2*>(out_ptr) = __float2half2_rn(0.0f);
        *reinterpret_cast<__half2*>(out_ptr + 2) = __float2half2_rn(0.0f);
    }
}

// ---------------------------------------------------------------------------
// CTA-qhead experiment: one CTA owns one q-head, does intra-CTA shard merge
// ---------------------------------------------------------------------------
template <int WarpsPerCTA, int PrefetchMode>
__global__ void
decode_attn_v4_cta_qhead_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const int64_t* __restrict__ cache_position,
    __half* __restrict__ out,
    int64_t q_stride_h,
    int64_t k_stride_h, int64_t k_stride_s,
    int64_t v_stride_h, int64_t v_stride_s,
    int64_t out_stride_h,
    float scale,
    int max_seq_len,
    int num_kv_groups,
    int merge_mode) {

    const int q_head_idx = blockIdx.x;
    const int kv_head_idx = q_head_idx / num_kv_groups;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int shard_idx = warp_id;
    const int group_id = lane / kThreadGroup;
    const int group_lane = lane % kThreadGroup;
    const int dim_offset = group_lane * kDimsPerThread;
    const int valid_len = min(static_cast<int>(cache_position[0] + 1), max_seq_len);

    __shared__ float s_partial_acc[WarpsPerCTA][kHeadDim];
    __shared__ float s_partial_l[WarpsPerCTA];
    __shared__ float s_partial_m[WarpsPerCTA];

    float m_val = kNegLarge;
    float l_val = 0.0f;
    float acc[kDimsPerThread];
    #pragma unroll
    for (int i = 0; i < kDimsPerThread; ++i) {
        acc[i] = 0.0f;
    }

    const int shard_size = (valid_len + WarpsPerCTA - 1) / WarpsPerCTA;
    const int shard_start = shard_idx * shard_size;
    const int shard_end = min(shard_start + shard_size, valid_len);

    if (shard_start < shard_end) {
        float q_reg[kDimsPerThread];
        const __half* qp = q + q_head_idx * q_stride_h + dim_offset;
        #pragma unroll
        for (int i = 0; i < kDimsPerThread; i += 8) {
            uint4 d = __ldg(reinterpret_cast<const uint4*>(qp + i));
            const __half* h = reinterpret_cast<const __half*>(&d);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                q_reg[i + j] = __half2float(h[j]);
            }
        }

        const __half* k_base = k + kv_head_idx * k_stride_h;
        const __half* v_base = v + kv_head_idx * v_stride_h;

        for (int tile_start = shard_start; tile_start < shard_end; tile_start += kTileN) {
            const int tile_end = min(tile_start + kTileN, shard_end);
            if constexpr (PrefetchMode == 2) {
                const int pf_tile_start = tile_start + kTileN;
                if (pf_tile_start < shard_end && group_lane == 0) {
                    #pragma unroll
                    for (int step = 0; step < kTileSteps; ++step) {
                        const int t_pf = pf_tile_start + step * kGroupsPerWarp + group_id;
                        if (t_pf < shard_end) {
                            const __half* kp_pf = k_base + t_pf * k_stride_s;
                            prefetch_global_l2(kp_pf);
                            prefetch_global_l2(kp_pf + 64);
                        }
                    }
                }
            }
            float local_scores[kTileSteps];
            float tile_max = kNegLarge;

            #pragma unroll
            for (int step = 0; step < kTileSteps; ++step) {
                const int t_global = tile_start + step * kGroupsPerWarp + group_id;
                const bool valid = (t_global < tile_end);

                float dot = 0.0f;
                if (valid) {
                    const __half* kp = k_base + t_global * k_stride_s + dim_offset;
                    if constexpr (PrefetchMode == 1) {
                        if (group_lane == 0) {
                            const __half* vp_pf = v_base + t_global * v_stride_s;
                            prefetch_global_l2(vp_pf);
                            prefetch_global_l2(vp_pf + 64);
                        }
                    }
                    #pragma unroll
                    for (int i = 0; i < kDimsPerThread; i += 8) {
                        uint4 d = __ldg(reinterpret_cast<const uint4*>(kp + i));
                        const __half* h = reinterpret_cast<const __half*>(&d);
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            dot += q_reg[i + j] * __half2float(h[j]);
                        }
                    }
                }
                dot += __shfl_xor_sync(0xffffffff, dot, 1);
                dot += __shfl_xor_sync(0xffffffff, dot, 2);

                const float score = valid ? (dot * scale) : kNegLarge;
                local_scores[step] = score;
                tile_max = fmaxf(tile_max, score);
            }

            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 4));
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 8));
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 16));

            const float new_m = fmaxf(m_val, tile_max);
            const float alpha = expf(m_val - new_m);
            #pragma unroll
            for (int i = 0; i < kDimsPerThread; ++i) {
                acc[i] *= alpha;
            }
            l_val *= alpha;
            m_val = new_m;

            #pragma unroll
            for (int step = 0; step < kTileSteps; ++step) {
                const int t_global = tile_start + step * kGroupsPerWarp + group_id;
                const bool valid = (t_global < tile_end);
                if (!valid) {
                    continue;
                }

                const float p = expf(local_scores[step] - m_val);
                l_val += p;
                const __half* vp = v_base + t_global * v_stride_s + dim_offset;
                #pragma unroll
                for (int i = 0; i < kDimsPerThread; i += 8) {
                    uint4 d = __ldg(reinterpret_cast<const uint4*>(vp + i));
                    const __half* h = reinterpret_cast<const __half*>(&d);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        acc[i + j] += p * __half2float(h[j]);
                    }
                }
            }
        }

        #pragma unroll
        for (int xor_mask = kThreadGroup; xor_mask < kWarpSize; xor_mask <<= 1) {
            const float pm = __shfl_xor_sync(0xffffffff, m_val, xor_mask);
            const float pl = __shfl_xor_sync(0xffffffff, l_val, xor_mask);
            const float nm = fmaxf(m_val, pm);
            const float s_me = expf(m_val - nm);
            const float s_pa = expf(pm - nm);
            #pragma unroll
            for (int i = 0; i < kDimsPerThread; ++i) {
                const float pa = __shfl_xor_sync(0xffffffff, acc[i], xor_mask);
                acc[i] = acc[i] * s_me + pa * s_pa;
            }
            l_val = l_val * s_me + pl * s_pa;
            m_val = nm;
        }
    }

    if (group_id == 0) {
        float* sm_acc = &s_partial_acc[shard_idx][dim_offset];
        #pragma unroll
        for (int i = 0; i < kDimsPerThread; ++i) {
            sm_acc[i] = acc[i];
        }
        if (group_lane == 0) {
            s_partial_l[shard_idx] = l_val;
            s_partial_m[shard_idx] = m_val;
        }
    }

    __syncthreads();

    if (merge_mode == 1 || merge_mode == 2) {
        if (merge_mode == 1) {
            constexpr int kParallelMergeWarps = 4;
            if (warp_id >= kParallelMergeWarps) {
                return;
            }

            const int merge_dim = warp_id * kWarpSize + lane;
            float m_merge = kNegLarge;
            float l_merge = 0.0f;
            float acc_merge = 0.0f;

            #pragma unroll
            for (int s = 0; s < WarpsPerCTA; ++s) {
                const float lv = s_partial_l[s];
                const float mv = s_partial_m[s];
                const float nm = fmaxf(m_merge, mv);
                const float s_old = expf(m_merge - nm);
                const float s_new = expf(mv - nm);

                const float pv = s_partial_acc[s][merge_dim];
                acc_merge = acc_merge * s_old + pv * s_new;
                l_merge = l_merge * s_old + lv * s_new;
                m_merge = nm;
            }

            __half* out_ptr = out + q_head_idx * out_stride_h + merge_dim;
            *out_ptr = (l_merge > 0.0f)
                ? __float2half(acc_merge * __fdividef(1.0f, l_merge))
                : __float2half(0.0f);
        } else {
            constexpr int kParallelMergeWarps = 2;
            if (warp_id >= kParallelMergeWarps) {
                return;
            }

            const int merge_dim = warp_id * 64 + lane * 2;
            float m_merge = kNegLarge;
            float l_merge = 0.0f;
            float a0 = 0.0f;
            float a1 = 0.0f;

            #pragma unroll
            for (int s = 0; s < WarpsPerCTA; ++s) {
                const float lv = s_partial_l[s];
                const float mv = s_partial_m[s];
                const float nm = fmaxf(m_merge, mv);
                const float s_old = expf(m_merge - nm);
                const float s_new = expf(mv - nm);

                const float* pv = &s_partial_acc[s][merge_dim];
                a0 = a0 * s_old + pv[0] * s_new;
                a1 = a1 * s_old + pv[1] * s_new;
                l_merge = l_merge * s_old + lv * s_new;
                m_merge = nm;
            }

            __half* out_ptr = out + q_head_idx * out_stride_h + merge_dim;
            if (l_merge > 0.0f) {
                const float inv_l = __fdividef(1.0f, l_merge);
                *reinterpret_cast<__half2*>(out_ptr) = __floats2half2_rn(a0 * inv_l, a1 * inv_l);
            } else {
                *reinterpret_cast<__half2*>(out_ptr) = __float2half2_rn(0.0f);
            }
        }
    } else {
        if (warp_id != 0) {
            return;
        }

        const int merge_dim = lane * 4;
        float m_merge = kNegLarge;
        float l_merge = 0.0f;
        float a0 = 0.0f;
        float a1 = 0.0f;
        float a2 = 0.0f;
        float a3 = 0.0f;

        #pragma unroll
        for (int s = 0; s < WarpsPerCTA; ++s) {
            const float lv = s_partial_l[s];
            const float mv = s_partial_m[s];
            const float nm = fmaxf(m_merge, mv);
            const float s_old = expf(m_merge - nm);
            const float s_new = expf(mv - nm);

            const float* pv = &s_partial_acc[s][merge_dim];
            a0 = a0 * s_old + pv[0] * s_new;
            a1 = a1 * s_old + pv[1] * s_new;
            a2 = a2 * s_old + pv[2] * s_new;
            a3 = a3 * s_old + pv[3] * s_new;
            l_merge = l_merge * s_old + lv * s_new;
            m_merge = nm;
        }

        __half* out_ptr = out + q_head_idx * out_stride_h + merge_dim;
        if (l_merge > 0.0f) {
            const float inv_l = __fdividef(1.0f, l_merge);
            *reinterpret_cast<__half2*>(out_ptr) = __floats2half2_rn(a0 * inv_l, a1 * inv_l);
            *reinterpret_cast<__half2*>(out_ptr + 2) = __floats2half2_rn(a2 * inv_l, a3 * inv_l);
        } else {
            *reinterpret_cast<__half2*>(out_ptr) = __float2half2_rn(0.0f);
            *reinterpret_cast<__half2*>(out_ptr + 2) = __float2half2_rn(0.0f);
        }
    }
}

template <int WarpsPerCTA>
__global__ void
decode_attn_v4_cta_qhead_cpasync_k_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const int64_t* __restrict__ cache_position,
    __half* __restrict__ out,
    int64_t q_stride_h,
    int64_t k_stride_h, int64_t k_stride_s,
    int64_t v_stride_h, int64_t v_stride_s,
    int64_t out_stride_h,
    float scale,
    int max_seq_len,
    int num_kv_groups) {

    constexpr int kStepRows = kGroupsPerWarp;
    constexpr int kStepElems = kStepRows * kHeadDim;
    constexpr int kStepBytes = kStepElems * static_cast<int>(sizeof(__half));
    constexpr int kCpAsyncRounds = kStepBytes / (kWarpSize * 16);

    const int q_head_idx = blockIdx.x;
    const int kv_head_idx = q_head_idx / num_kv_groups;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int shard_idx = warp_id;
    const int group_id = lane / kThreadGroup;
    const int group_lane = lane % kThreadGroup;
    const int dim_offset = group_lane * kDimsPerThread;
    const int valid_len = min(static_cast<int>(cache_position[0] + 1), max_seq_len);

    __shared__ __align__(16) __half s_k_stage[WarpsPerCTA][2][kStepElems];
    __shared__ float s_partial_acc[WarpsPerCTA][kHeadDim];
    __shared__ float s_partial_l[WarpsPerCTA];
    __shared__ float s_partial_m[WarpsPerCTA];

    float m_val = kNegLarge;
    float l_val = 0.0f;
    float acc[kDimsPerThread];
    #pragma unroll
    for (int i = 0; i < kDimsPerThread; ++i) {
        acc[i] = 0.0f;
    }

    const int shard_size = (valid_len + WarpsPerCTA - 1) / WarpsPerCTA;
    const int shard_start = shard_idx * shard_size;
    const int shard_end = min(shard_start + shard_size, valid_len);

    if (shard_start < shard_end) {
        float q_reg[kDimsPerThread];
        const __half* qp = q + q_head_idx * q_stride_h + dim_offset;
        #pragma unroll
        for (int i = 0; i < kDimsPerThread; i += 8) {
            uint4 d = __ldg(reinterpret_cast<const uint4*>(qp + i));
            const __half* h = reinterpret_cast<const __half*>(&d);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                q_reg[i + j] = __half2float(h[j]);
            }
        }

        const __half* k_base = k + kv_head_idx * k_stride_h;
        const __half* v_base = v + kv_head_idx * v_stride_h;

        for (int tile_start = shard_start; tile_start < shard_end; tile_start += kTileN) {
            const int tile_end = min(tile_start + kTileN, shard_end);
            float local_scores[kTileSteps];
            float tile_max = kNegLarge;

            int stage_buf = 0;
            const bool first_full_step = (tile_start + kStepRows <= tile_end);
            if (first_full_step) {
                char* smem_bytes = reinterpret_cast<char*>(&s_k_stage[warp_id][stage_buf][0]);
                const char* gmem_bytes = reinterpret_cast<const char*>(k_base + tile_start * k_stride_s);
                #pragma unroll
                for (int round = 0; round < kCpAsyncRounds; ++round) {
                    const int byte_off = (round * kWarpSize + lane) * 16;
                    cp_async_cg_16(smem_bytes + byte_off, gmem_bytes + byte_off);
                }
                cp_async_commit();
            }

            #pragma unroll
            for (int step = 0; step < kTileSteps; ++step) {
                const int step_base = tile_start + step * kStepRows;
                const int t_global = step_base + group_id;
                const bool valid = (t_global < tile_end);
                const bool full_step = (step_base + kStepRows <= tile_end);

                if (full_step) {
                    cp_async_wait_all();
                    __syncwarp();
                }

                const int next_step_base = step_base + kStepRows;
                const bool next_full_step = (next_step_base + kStepRows <= tile_end);
                if (next_full_step) {
                    const int next_buf = stage_buf ^ 1;
                    char* smem_bytes = reinterpret_cast<char*>(&s_k_stage[warp_id][next_buf][0]);
                    const char* gmem_bytes = reinterpret_cast<const char*>(k_base + next_step_base * k_stride_s);
                    #pragma unroll
                    for (int round = 0; round < kCpAsyncRounds; ++round) {
                        const int byte_off = (round * kWarpSize + lane) * 16;
                        cp_async_cg_16(smem_bytes + byte_off, gmem_bytes + byte_off);
                    }
                    cp_async_commit();
                }

                float dot = 0.0f;
                if (valid) {
                    if (full_step) {
                        const __half* kp = &s_k_stage[warp_id][stage_buf][group_id * kHeadDim + dim_offset];
                        #pragma unroll
                        for (int i = 0; i < kDimsPerThread; i += 8) {
                            uint4 d = *reinterpret_cast<const uint4*>(kp + i);
                            const __half* h = reinterpret_cast<const __half*>(&d);
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                dot += q_reg[i + j] * __half2float(h[j]);
                            }
                        }
                    } else {
                        const __half* kp = k_base + t_global * k_stride_s + dim_offset;
                        #pragma unroll
                        for (int i = 0; i < kDimsPerThread; i += 8) {
                            uint4 d = __ldg(reinterpret_cast<const uint4*>(kp + i));
                            const __half* h = reinterpret_cast<const __half*>(&d);
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                dot += q_reg[i + j] * __half2float(h[j]);
                            }
                        }
                    }
                }
                dot += __shfl_xor_sync(0xffffffff, dot, 1);
                dot += __shfl_xor_sync(0xffffffff, dot, 2);

                const float score = valid ? (dot * scale) : kNegLarge;
                local_scores[step] = score;
                tile_max = fmaxf(tile_max, score);
                stage_buf ^= next_full_step ? 1 : 0;
            }

            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 4));
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 8));
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 16));

            const float new_m = fmaxf(m_val, tile_max);
            const float alpha = expf(m_val - new_m);
            #pragma unroll
            for (int i = 0; i < kDimsPerThread; ++i) {
                acc[i] *= alpha;
            }
            l_val *= alpha;
            m_val = new_m;

            #pragma unroll
            for (int step = 0; step < kTileSteps; ++step) {
                const int t_global = tile_start + step * kGroupsPerWarp + group_id;
                const bool valid = (t_global < tile_end);
                if (!valid) {
                    continue;
                }

                const float p = expf(local_scores[step] - m_val);
                l_val += p;
                const __half* vp = v_base + t_global * v_stride_s + dim_offset;
                #pragma unroll
                for (int i = 0; i < kDimsPerThread; i += 8) {
                    uint4 d = __ldg(reinterpret_cast<const uint4*>(vp + i));
                    const __half* h = reinterpret_cast<const __half*>(&d);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        acc[i + j] += p * __half2float(h[j]);
                    }
                }
            }
        }

        #pragma unroll
        for (int xor_mask = kThreadGroup; xor_mask < kWarpSize; xor_mask <<= 1) {
            const float pm = __shfl_xor_sync(0xffffffff, m_val, xor_mask);
            const float pl = __shfl_xor_sync(0xffffffff, l_val, xor_mask);
            const float nm = fmaxf(m_val, pm);
            const float s_me = expf(m_val - nm);
            const float s_pa = expf(pm - nm);
            #pragma unroll
            for (int i = 0; i < kDimsPerThread; ++i) {
                const float pa = __shfl_xor_sync(0xffffffff, acc[i], xor_mask);
                acc[i] = acc[i] * s_me + pa * s_pa;
            }
            l_val = l_val * s_me + pl * s_pa;
            m_val = nm;
        }
    }

    if (group_id == 0) {
        float* sm_acc = &s_partial_acc[shard_idx][dim_offset];
        #pragma unroll
        for (int i = 0; i < kDimsPerThread; ++i) {
            sm_acc[i] = acc[i];
        }
        if (group_lane == 0) {
            s_partial_l[shard_idx] = l_val;
            s_partial_m[shard_idx] = m_val;
        }
    }

    __syncthreads();

    if (warp_id != 0) {
        return;
    }

    const int merge_dim = lane * 4;
    float m_merge = kNegLarge;
    float l_merge = 0.0f;
    float a0 = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;

    #pragma unroll
    for (int s = 0; s < WarpsPerCTA; ++s) {
        const float lv = s_partial_l[s];
        const float mv = s_partial_m[s];
        const float nm = fmaxf(m_merge, mv);
        const float s_old = expf(m_merge - nm);
        const float s_new = expf(mv - nm);

        const float* pv = &s_partial_acc[s][merge_dim];
        a0 = a0 * s_old + pv[0] * s_new;
        a1 = a1 * s_old + pv[1] * s_new;
        a2 = a2 * s_old + pv[2] * s_new;
        a3 = a3 * s_old + pv[3] * s_new;
        l_merge = l_merge * s_old + lv * s_new;
        m_merge = nm;
    }

    __half* out_ptr = out + q_head_idx * out_stride_h + merge_dim;
    if (l_merge > 0.0f) {
        const float inv_l = __fdividef(1.0f, l_merge);
        *reinterpret_cast<__half2*>(out_ptr) = __floats2half2_rn(a0 * inv_l, a1 * inv_l);
        *reinterpret_cast<__half2*>(out_ptr + 2) = __floats2half2_rn(a2 * inv_l, a3 * inv_l);
    } else {
        *reinterpret_cast<__half2*>(out_ptr) = __float2half2_rn(0.0f);
        *reinterpret_cast<__half2*>(out_ptr + 2) = __float2half2_rn(0.0f);
    }
}

template <int WarpsPerCTA>
__global__ void
decode_attn_v4_cta_qhead_contig_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const int64_t* __restrict__ cache_position,
    __half* __restrict__ out,
    int64_t q_stride_h,
    int64_t k_stride_h, int64_t k_stride_s,
    int64_t v_stride_h, int64_t v_stride_s,
    int64_t out_stride_h,
    float scale,
    int max_seq_len,
    int num_kv_groups) {

    const int q_head_idx = blockIdx.x;
    const int kv_head_idx = q_head_idx / num_kv_groups;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int shard_idx = warp_id;
    const int valid_len = min(static_cast<int>(cache_position[0] + 1), max_seq_len);

    __shared__ float s_partial_acc[WarpsPerCTA][kHeadDim];
    __shared__ float s_partial_l[WarpsPerCTA];
    __shared__ float s_partial_m[WarpsPerCTA];

    float m_val = kNegLarge;
    float l_val = 0.0f;
    float acc[kContigDimsPerLane];
    #pragma unroll
    for (int i = 0; i < kContigDimsPerLane; ++i) {
        acc[i] = 0.0f;
    }

    const int shard_size = (valid_len + WarpsPerCTA - 1) / WarpsPerCTA;
    const int shard_start = shard_idx * shard_size;
    const int shard_end = min(shard_start + shard_size, valid_len);
    const int dim_offset = lane * kContigDimsPerLane;

    if (shard_start < shard_end) {
        float q_reg[kContigDimsPerLane];
        const __half* qp = q + q_head_idx * q_stride_h + dim_offset;
        load_half4_vec(qp, q_reg[0], q_reg[1], q_reg[2], q_reg[3]);

        const __half* k_base = k + kv_head_idx * k_stride_h;
        const __half* v_base = v + kv_head_idx * v_stride_h;

        for (int tile_start = shard_start; tile_start < shard_end; tile_start += kTileSteps) {
            const int tile_end = min(tile_start + kTileSteps, shard_end);
            float local_scores[kTileSteps];
            float tile_max = kNegLarge;

            #pragma unroll
            for (int step = 0; step < kTileSteps; ++step) {
                const int t_global = tile_start + step;
                const bool valid = (t_global < tile_end);
                float dot = 0.0f;
                if (valid) {
                    const __half* kp = k_base + t_global * k_stride_s + dim_offset;
                    float k0, k1, k2, k3;
                    load_half4_vec(kp, k0, k1, k2, k3);
                    dot += q_reg[0] * k0;
                    dot += q_reg[1] * k1;
                    dot += q_reg[2] * k2;
                    dot += q_reg[3] * k3;
                }
                dot = warp_sum_full(dot);
                const float score = valid ? (dot * scale) : kNegLarge;
                local_scores[step] = score;
                tile_max = fmaxf(tile_max, score);
            }

            tile_max = warp_max_full(tile_max);

            const float new_m = fmaxf(m_val, tile_max);
            const float alpha = expf(m_val - new_m);
            #pragma unroll
            for (int i = 0; i < kContigDimsPerLane; ++i) {
                acc[i] *= alpha;
            }
            l_val *= alpha;
            m_val = new_m;

            #pragma unroll
            for (int step = 0; step < kTileSteps; ++step) {
                const int t_global = tile_start + step;
                const bool valid = (t_global < tile_end);
                if (!valid) {
                    continue;
                }
                const float p = expf(local_scores[step] - m_val);
                l_val += p;
                const __half* vp = v_base + t_global * v_stride_s + dim_offset;
                float v0, v1, v2, v3;
                load_half4_vec(vp, v0, v1, v2, v3);
                acc[0] += p * v0;
                acc[1] += p * v1;
                acc[2] += p * v2;
                acc[3] += p * v3;
            }
        }
    }

    float* sm_acc = &s_partial_acc[shard_idx][dim_offset];
    #pragma unroll
    for (int i = 0; i < kContigDimsPerLane; ++i) {
        sm_acc[i] = acc[i];
    }
    if (lane == 0) {
        s_partial_l[shard_idx] = l_val;
        s_partial_m[shard_idx] = m_val;
    }

    __syncthreads();

    if (warp_id != 0) {
        return;
    }

    const int merge_dim = lane * kContigDimsPerLane;
    float m_merge = kNegLarge;
    float l_merge = 0.0f;
    float merged[kContigDimsPerLane];
    #pragma unroll
    for (int i = 0; i < kContigDimsPerLane; ++i) {
        merged[i] = 0.0f;
    }

    #pragma unroll
    for (int s = 0; s < WarpsPerCTA; ++s) {
        const float lv = s_partial_l[s];
        const float mv = s_partial_m[s];
        const float nm = fmaxf(m_merge, mv);
        const float s_old = expf(m_merge - nm);
        const float s_new = expf(mv - nm);
        const float* pv = &s_partial_acc[s][merge_dim];
        #pragma unroll
        for (int i = 0; i < kContigDimsPerLane; ++i) {
            merged[i] = merged[i] * s_old + pv[i] * s_new;
        }
        l_merge = l_merge * s_old + lv * s_new;
        m_merge = nm;
    }

    __half* out_ptr = out + q_head_idx * out_stride_h + merge_dim;
    if (l_merge > 0.0f) {
        const float inv_l = __fdividef(1.0f, l_merge);
        #pragma unroll
        for (int i = 0; i < kContigDimsPerLane; ++i) {
            out_ptr[i] = __float2half(merged[i] * inv_l);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < kContigDimsPerLane; ++i) {
            out_ptr[i] = __float2half(0.0f);
        }
    }
}

template <int WarpsPerCTA>
__global__ void
decode_attn_v4_cta_qhead_group8_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const int64_t* __restrict__ cache_position,
    __half* __restrict__ out,
    int64_t q_stride_h,
    int64_t k_stride_h, int64_t k_stride_s,
    int64_t v_stride_h, int64_t v_stride_s,
    int64_t out_stride_h,
    float scale,
    int max_seq_len,
    int num_kv_groups) {

    const int q_head_idx = blockIdx.x;
    const int kv_head_idx = q_head_idx / num_kv_groups;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int shard_idx = warp_id;
    const int group_id = lane / kThreadGroupWide;
    const int group_lane = lane % kThreadGroupWide;
    const int dim_offset = group_lane * kDimsPerThreadWide;
    const int valid_len = min(static_cast<int>(cache_position[0] + 1), max_seq_len);

    __shared__ float s_partial_acc[WarpsPerCTA][kHeadDim];
    __shared__ float s_partial_l[WarpsPerCTA];
    __shared__ float s_partial_m[WarpsPerCTA];

    float m_val = kNegLarge;
    float l_val = 0.0f;
    float acc[kDimsPerThreadWide];
    #pragma unroll
    for (int i = 0; i < kDimsPerThreadWide; ++i) {
        acc[i] = 0.0f;
    }

    const int shard_size = (valid_len + WarpsPerCTA - 1) / WarpsPerCTA;
    const int shard_start = shard_idx * shard_size;
    const int shard_end = min(shard_start + shard_size, valid_len);

    if (shard_start < shard_end) {
        float q_reg[kDimsPerThreadWide];
        const __half* qp = q + q_head_idx * q_stride_h + dim_offset;
        #pragma unroll
        for (int i = 0; i < kDimsPerThreadWide; i += 8) {
            uint4 d = __ldg(reinterpret_cast<const uint4*>(qp + i));
            const __half* h = reinterpret_cast<const __half*>(&d);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                q_reg[i + j] = __half2float(h[j]);
            }
        }

        const __half* k_base = k + kv_head_idx * k_stride_h;
        const __half* v_base = v + kv_head_idx * v_stride_h;

        for (int tile_start = shard_start; tile_start < shard_end; tile_start += kTileNWide) {
            const int tile_end = min(tile_start + kTileNWide, shard_end);
            float local_scores[kTileStepsWide];
            float tile_max = kNegLarge;

            #pragma unroll
            for (int step = 0; step < kTileStepsWide; ++step) {
                const int t_global = tile_start + step * kGroupsPerWarpWide + group_id;
                const bool valid = (t_global < tile_end);

                float dot = 0.0f;
                if (valid) {
                    const __half* kp = k_base + t_global * k_stride_s + dim_offset;
                    #pragma unroll
                    for (int i = 0; i < kDimsPerThreadWide; i += 8) {
                        uint4 d = __ldg(reinterpret_cast<const uint4*>(kp + i));
                        const __half* h = reinterpret_cast<const __half*>(&d);
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            dot += q_reg[i + j] * __half2float(h[j]);
                        }
                    }
                }
                dot += __shfl_xor_sync(0xffffffff, dot, 1);
                dot += __shfl_xor_sync(0xffffffff, dot, 2);
                dot += __shfl_xor_sync(0xffffffff, dot, 4);

                const float score = valid ? (dot * scale) : kNegLarge;
                local_scores[step] = score;
                tile_max = fmaxf(tile_max, score);
            }

            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 8));
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 16));

            const float new_m = fmaxf(m_val, tile_max);
            const float alpha = expf(m_val - new_m);
            #pragma unroll
            for (int i = 0; i < kDimsPerThreadWide; ++i) {
                acc[i] *= alpha;
            }
            l_val *= alpha;
            m_val = new_m;

            #pragma unroll
            for (int step = 0; step < kTileStepsWide; ++step) {
                const int t_global = tile_start + step * kGroupsPerWarpWide + group_id;
                const bool valid = (t_global < tile_end);
                if (!valid) {
                    continue;
                }

                const float p = expf(local_scores[step] - m_val);
                l_val += p;
                const __half* vp = v_base + t_global * v_stride_s + dim_offset;
                #pragma unroll
                for (int i = 0; i < kDimsPerThreadWide; i += 8) {
                    uint4 d = __ldg(reinterpret_cast<const uint4*>(vp + i));
                    const __half* h = reinterpret_cast<const __half*>(&d);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        acc[i + j] += p * __half2float(h[j]);
                    }
                }
            }
        }

        #pragma unroll
        for (int xor_mask = kThreadGroupWide; xor_mask < kWarpSize; xor_mask <<= 1) {
            const float pm = __shfl_xor_sync(0xffffffff, m_val, xor_mask);
            const float pl = __shfl_xor_sync(0xffffffff, l_val, xor_mask);
            const float nm = fmaxf(m_val, pm);
            const float s_me = expf(m_val - nm);
            const float s_pa = expf(pm - nm);
            #pragma unroll
            for (int i = 0; i < kDimsPerThreadWide; ++i) {
                const float pa = __shfl_xor_sync(0xffffffff, acc[i], xor_mask);
                acc[i] = acc[i] * s_me + pa * s_pa;
            }
            l_val = l_val * s_me + pl * s_pa;
            m_val = nm;
        }
    }

    if (group_id == 0) {
        float* sm_acc = &s_partial_acc[shard_idx][dim_offset];
        #pragma unroll
        for (int i = 0; i < kDimsPerThreadWide; ++i) {
            sm_acc[i] = acc[i];
        }
        if (group_lane == 0) {
            s_partial_l[shard_idx] = l_val;
            s_partial_m[shard_idx] = m_val;
        }
    }

    __syncthreads();

    if (warp_id != 0) {
        return;
    }

    const int merge_dim = lane * 4;
    float m_merge = kNegLarge;
    float l_merge = 0.0f;
    float a0 = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;

    #pragma unroll
    for (int s = 0; s < WarpsPerCTA; ++s) {
        const float lv = s_partial_l[s];
        const float mv = s_partial_m[s];
        const float nm = fmaxf(m_merge, mv);
        const float s_old = expf(m_merge - nm);
        const float s_new = expf(mv - nm);

        const float* pv = &s_partial_acc[s][merge_dim];
        a0 = a0 * s_old + pv[0] * s_new;
        a1 = a1 * s_old + pv[1] * s_new;
        a2 = a2 * s_old + pv[2] * s_new;
        a3 = a3 * s_old + pv[3] * s_new;
        l_merge = l_merge * s_old + lv * s_new;
        m_merge = nm;
    }

    __half* out_ptr = out + q_head_idx * out_stride_h + merge_dim;
    if (l_merge > 0.0f) {
        const float inv_l = __fdividef(1.0f, l_merge);
        *reinterpret_cast<__half2*>(out_ptr) = __floats2half2_rn(a0 * inv_l, a1 * inv_l);
        *reinterpret_cast<__half2*>(out_ptr + 2) = __floats2half2_rn(a2 * inv_l, a3 * inv_l);
    } else {
        *reinterpret_cast<__half2*>(out_ptr) = __float2half2_rn(0.0f);
        *reinterpret_cast<__half2*>(out_ptr + 2) = __float2half2_rn(0.0f);
    }
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Host dispatch
// ---------------------------------------------------------------------------
torch::Tensor decode_attn_v4_splitkv_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cache_position,
    double scale,
    const torch::Tensor& partial_acc,
    const torch::Tensor& partial_stats,
    const torch::Tensor& out,
    int64_t num_kv_groups,
    int64_t split_k,
    const torch::Tensor& prefetch_weight,
    int64_t prefetch_blocks) {

    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kHalf, "only float16");
    TORCH_CHECK(q.size(0) == 1 && q.size(2) == 1, "q must be [1, H, 1, D]");
    TORCH_CHECK(q.size(3) == kHeadDim, "head_dim must be 128");

    const int64_t num_kv_heads = k.size(1);
    const int64_t max_seq_len = k.size(2);
    TORCH_CHECK(split_k > 1, "split_k must be > 1");
    TORCH_CHECK(split_k <= partial_acc.size(1), "split_k exceeds partial_acc capacity");
    TORCH_CHECK(split_k <= partial_stats.size(1), "split_k exceeds partial_stats capacity");
    TORCH_CHECK(prefetch_blocks >= 0, "prefetch_blocks must be >= 0");

    c10::cuda::CUDAGuard device_guard(q.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto qp = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
    auto kp = reinterpret_cast<const __half*>(k.data_ptr<at::Half>());
    auto vp = reinterpret_cast<const __half*>(v.data_ptr<at::Half>());
    auto op = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    const __half* prefetch_ptr = nullptr;
    int64_t prefetch_numel = 0;
    if (prefetch_blocks > 0 && prefetch_weight.numel() > 0) {
        TORCH_CHECK(prefetch_weight.is_cuda(), "prefetch_weight must be CUDA");
        TORCH_CHECK(prefetch_weight.device() == q.device(), "prefetch_weight must match q device");
        TORCH_CHECK(prefetch_weight.scalar_type() == torch::kHalf, "prefetch_weight must be float16");
        TORCH_CHECK(
            prefetch_weight.dim() == 2
                && prefetch_weight.size(0) == 2048
                && prefetch_weight.size(1) == 2048
                && prefetch_weight.stride(0) == 2048
                && prefetch_weight.stride(1) == 1,
            "prefetch_weight must be contiguous [2048, 2048]");
        prefetch_ptr = reinterpret_cast<const __half*>(prefetch_weight.data_ptr<at::Half>());
        prefetch_numel = prefetch_weight.numel();
    } else {
        prefetch_blocks = 0;
    }

    const int attention_blocks = static_cast<int>(num_kv_heads * split_k);

    decode_attn_v4_fwd_kernel<<<attention_blocks + static_cast<int>(prefetch_blocks), 64, 0, stream>>>(
        qp, kp, vp, prefetch_ptr,
        cache_position.data_ptr<int64_t>(),
        partial_acc.data_ptr<float>(),
        partial_stats.data_ptr<float>(),
        q.stride(1),
        k.stride(1), k.stride(2),
        v.stride(1), v.stride(2),
        partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2),
        partial_stats.stride(0), partial_stats.stride(1),
        partial_stats.stride(2), partial_stats.stride(3),
        prefetch_numel,
        static_cast<float>(scale),
        static_cast<int>(max_seq_len),
        static_cast<int>(split_k),
        attention_blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    decode_attn_v4_merge_kernel<<<num_kv_heads, 128, 0, stream>>>(
        partial_acc.data_ptr<float>(),
        partial_stats.data_ptr<float>(),
        op,
        partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2),
        partial_stats.stride(0), partial_stats.stride(1),
        partial_stats.stride(2), partial_stats.stride(3),
        out.stride(1),
        static_cast<int>(split_k));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

// ---------------------------------------------------------------------------
// Host dispatch — fused fwd+merge (single kernel launch)
// ---------------------------------------------------------------------------
torch::Tensor decode_attn_v4_fused_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cache_position,
    double scale,
    const torch::Tensor& partial_acc,
    const torch::Tensor& partial_stats,
    const torch::Tensor& out,
    const torch::Tensor& done_count,
    int64_t num_kv_groups,
    int64_t split_k,
    const torch::Tensor& prefetch_weight,
    int64_t prefetch_blocks) {

    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kHalf, "only float16");
    TORCH_CHECK(q.size(0) == 1 && q.size(2) == 1, "q must be [1, H, 1, D]");
    TORCH_CHECK(q.size(3) == kHeadDim, "head_dim must be 128");
    TORCH_CHECK(done_count.is_cuda() && done_count.scalar_type() == torch::kInt32,
                "done_count must be CUDA int32");

    const int64_t num_kv_heads = k.size(1);
    const int64_t max_seq_len = k.size(2);
    TORCH_CHECK(split_k > 1, "split_k must be > 1");
    TORCH_CHECK(split_k <= partial_acc.size(1), "split_k exceeds partial_acc capacity");
    TORCH_CHECK(split_k <= partial_stats.size(1), "split_k exceeds partial_stats capacity");
    TORCH_CHECK(done_count.numel() >= num_kv_heads, "done_count too small");
    TORCH_CHECK(prefetch_blocks >= 0, "prefetch_blocks must be >= 0");

    c10::cuda::CUDAGuard device_guard(q.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto qp = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
    auto kp = reinterpret_cast<const __half*>(k.data_ptr<at::Half>());
    auto vp = reinterpret_cast<const __half*>(v.data_ptr<at::Half>());
    auto op = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    const __half* prefetch_ptr = nullptr;
    int64_t prefetch_numel = 0;
    if (prefetch_blocks > 0 && prefetch_weight.numel() > 0) {
        TORCH_CHECK(prefetch_weight.is_cuda(), "prefetch_weight must be CUDA");
        TORCH_CHECK(prefetch_weight.device() == q.device(), "prefetch_weight must match q device");
        TORCH_CHECK(prefetch_weight.scalar_type() == torch::kHalf, "prefetch_weight must be float16");
        TORCH_CHECK(
            prefetch_weight.dim() == 2
                && prefetch_weight.size(0) == 2048
                && prefetch_weight.size(1) == 2048
                && prefetch_weight.stride(0) == 2048
                && prefetch_weight.stride(1) == 1,
            "prefetch_weight must be contiguous [2048, 2048]");
        prefetch_ptr = reinterpret_cast<const __half*>(prefetch_weight.data_ptr<at::Half>());
        prefetch_numel = prefetch_weight.numel();
    } else {
        prefetch_blocks = 0;
    }

    const int attention_blocks = static_cast<int>(num_kv_heads * split_k);

    decode_attn_v4_fused_kernel<<<attention_blocks + static_cast<int>(prefetch_blocks), 64, 0, stream>>>(
        qp, kp, vp, prefetch_ptr,
        cache_position.data_ptr<int64_t>(),
        partial_acc.data_ptr<float>(),
        partial_stats.data_ptr<float>(),
        op,
        done_count.data_ptr<int>(),
        q.stride(1),
        k.stride(1), k.stride(2),
        v.stride(1), v.stride(2),
        partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2),
        partial_stats.stride(0), partial_stats.stride(1),
        partial_stats.stride(2), partial_stats.stride(3),
        out.stride(1),
        prefetch_numel,
        static_cast<float>(scale),
        static_cast<int>(max_seq_len),
        static_cast<int>(split_k),
        attention_blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

// ---------------------------------------------------------------------------
// Host dispatch — CTA-local experiment (one CTA per KV head)
// ---------------------------------------------------------------------------
torch::Tensor decode_attn_v4_cta_local_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cache_position,
    double scale,
    const torch::Tensor& out,
    int64_t num_kv_groups) {

    TORCH_CHECK(num_kv_groups == kQHeadsPerKV, "CTA-local kernel currently requires num_kv_groups == 2");
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && out.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(cache_position.is_cuda() && cache_position.scalar_type() == torch::kLong,
                "cache_position must be CUDA int64");
    TORCH_CHECK(q.scalar_type() == torch::kHalf, "only float16");
    TORCH_CHECK(k.scalar_type() == torch::kHalf && v.scalar_type() == torch::kHalf, "k/v must be float16");
    TORCH_CHECK(out.scalar_type() == torch::kHalf, "out must be float16");
    TORCH_CHECK(q.size(0) == 1 && q.size(2) == 1, "q must be [1, H, 1, D]");
    TORCH_CHECK(q.size(3) == kHeadDim, "head_dim must be 128");
    TORCH_CHECK(k.size(0) == 1 && v.size(0) == 1, "k/v batch must be 1");
    TORCH_CHECK(k.size(1) == v.size(1), "k/v head mismatch");
    TORCH_CHECK(k.size(2) == v.size(2), "k/v seq mismatch");
    TORCH_CHECK(k.size(3) == kHeadDim && v.size(3) == kHeadDim, "k/v head_dim must be 128");
    TORCH_CHECK(q.size(1) == k.size(1) * num_kv_groups, "q heads must equal kv_heads * num_kv_groups");
    TORCH_CHECK(out.dim() == 3 && out.size(0) == 1, "out must be [1, H, D]");
    TORCH_CHECK(out.size(1) == q.size(1) && out.size(2) == kHeadDim, "out shape mismatch");

    const int64_t num_kv_heads = k.size(1);
    const int64_t max_seq_len = k.size(2);

    c10::cuda::CUDAGuard device_guard(q.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    decode_attn_v4_cta_local_kernel<<<num_kv_heads, kCTALocalWarps * kWarpSize, 0, stream>>>(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        q.stride(1),
        k.stride(1), k.stride(2),
        v.stride(1), v.stride(2),
        out.stride(1),
        static_cast<float>(scale),
        static_cast<int>(max_seq_len));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

// ---------------------------------------------------------------------------
// Host dispatch — CTA-qhead experiment (one CTA per q-head)
// ---------------------------------------------------------------------------
torch::Tensor decode_attn_v4_cta_qhead_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cache_position,
    double scale,
    const torch::Tensor& out,
    int64_t num_kv_groups,
    int64_t warps_per_cta,
    int64_t lane_layout,
    int64_t prefetch_mode,
    int64_t merge_mode) {

    TORCH_CHECK(num_kv_groups == kQHeadsPerKV, "CTA-qhead kernel currently requires num_kv_groups == 2");
    TORCH_CHECK(
        warps_per_cta == kCTAQHeadWarpsDefault
            || warps_per_cta == kCTAQHeadWarpsWide
            || warps_per_cta == kCTAQHeadWarpsXL,
        "CTA-qhead kernel currently supports warps_per_cta in {4, 8, 16}");
    TORCH_CHECK(lane_layout == 0 || lane_layout == 1 || lane_layout == 2,
        "CTA-qhead kernel currently supports lane_layout in {0=grouped, 1=contig, 2=group8}");
    TORCH_CHECK(prefetch_mode == 0 || prefetch_mode == 1 || prefetch_mode == 2 || prefetch_mode == 3,
        "CTA-qhead kernel currently supports prefetch_mode in {0=off, 1=v-prefetch, 2=next-tile-k-prefetch, 3=cpasync-k-stage}");
    TORCH_CHECK(prefetch_mode == 0 || lane_layout == 0,
        "CTA-qhead prefetch_mode is currently only supported for lane_layout=0");
    TORCH_CHECK(prefetch_mode != 3 || warps_per_cta == kCTAQHeadWarpsWide,
        "CTA-qhead cpasync-k-stage is currently only supported for warps_per_cta=8");
    TORCH_CHECK(merge_mode == 0 || merge_mode == 1 || merge_mode == 2,
        "CTA-qhead kernel currently supports merge_mode in {0=warp0-serial, 1=4warp-parallel, 2=2warp-parallel}");
    TORCH_CHECK(merge_mode == 0 || (lane_layout == 0 && prefetch_mode == 0),
        "CTA-qhead merge_mode!=0 is currently only supported for lane_layout=0 and prefetch_mode=0");
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && out.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(cache_position.is_cuda() && cache_position.scalar_type() == torch::kLong,
                "cache_position must be CUDA int64");
    TORCH_CHECK(q.scalar_type() == torch::kHalf, "only float16");
    TORCH_CHECK(k.scalar_type() == torch::kHalf && v.scalar_type() == torch::kHalf, "k/v must be float16");
    TORCH_CHECK(out.scalar_type() == torch::kHalf, "out must be float16");
    TORCH_CHECK(q.size(0) == 1 && q.size(2) == 1, "q must be [1, H, 1, D]");
    TORCH_CHECK(q.size(3) == kHeadDim, "head_dim must be 128");
    TORCH_CHECK(k.size(0) == 1 && v.size(0) == 1, "k/v batch must be 1");
    TORCH_CHECK(k.size(1) == v.size(1), "k/v head mismatch");
    TORCH_CHECK(k.size(2) == v.size(2), "k/v seq mismatch");
    TORCH_CHECK(k.size(3) == kHeadDim && v.size(3) == kHeadDim, "k/v head_dim must be 128");
    TORCH_CHECK(q.size(1) == k.size(1) * num_kv_groups, "q heads must equal kv_heads * num_kv_groups");
    TORCH_CHECK(out.dim() == 3 && out.size(0) == 1, "out must be [1, H, D]");
    TORCH_CHECK(out.size(1) == q.size(1) && out.size(2) == kHeadDim, "out shape mismatch");

    const int64_t num_q_heads = q.size(1);
    const int64_t max_seq_len = k.size(2);

    c10::cuda::CUDAGuard device_guard(q.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    if (warps_per_cta == kCTAQHeadWarpsDefault) {
        if (lane_layout == 0) {
            if (prefetch_mode == 0) {
                decode_attn_v4_cta_qhead_kernel<kCTAQHeadWarpsDefault, 0><<<
                    num_q_heads, kCTAQHeadWarpsDefault * kWarpSize, 0, stream>>>(
                    reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
                    cache_position.data_ptr<int64_t>(),
                    reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                    q.stride(1),
                    k.stride(1), k.stride(2),
                    v.stride(1), v.stride(2),
                    out.stride(1),
                    static_cast<float>(scale),
                    static_cast<int>(max_seq_len),
                    static_cast<int>(num_kv_groups),
                    static_cast<int>(merge_mode));
            } else if (prefetch_mode == 1) {
                decode_attn_v4_cta_qhead_kernel<kCTAQHeadWarpsDefault, 1><<<
                    num_q_heads, kCTAQHeadWarpsDefault * kWarpSize, 0, stream>>>(
                    reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
                    cache_position.data_ptr<int64_t>(),
                    reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                    q.stride(1),
                    k.stride(1), k.stride(2),
                    v.stride(1), v.stride(2),
                    out.stride(1),
                    static_cast<float>(scale),
                    static_cast<int>(max_seq_len),
                    static_cast<int>(num_kv_groups),
                    0);
            } else if (prefetch_mode == 2) {
                decode_attn_v4_cta_qhead_kernel<kCTAQHeadWarpsDefault, 2><<<
                    num_q_heads, kCTAQHeadWarpsDefault * kWarpSize, 0, stream>>>(
                    reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
                    cache_position.data_ptr<int64_t>(),
                    reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                    q.stride(1),
                    k.stride(1), k.stride(2),
                    v.stride(1), v.stride(2),
                    out.stride(1),
                    static_cast<float>(scale),
                    static_cast<int>(max_seq_len),
                    static_cast<int>(num_kv_groups),
                    0);
            } else {
                TORCH_CHECK(false, "cpasync-k-stage only supports warps_per_cta=8");
            }
        } else if (lane_layout == 1) {
            decode_attn_v4_cta_qhead_contig_kernel<kCTAQHeadWarpsDefault><<<
                num_q_heads, kCTAQHeadWarpsDefault * kWarpSize, 0, stream>>>(
                reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
                cache_position.data_ptr<int64_t>(),
                reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                q.stride(1),
                k.stride(1), k.stride(2),
                v.stride(1), v.stride(2),
                out.stride(1),
                static_cast<float>(scale),
                static_cast<int>(max_seq_len),
                static_cast<int>(num_kv_groups));
        } else {
            decode_attn_v4_cta_qhead_group8_kernel<kCTAQHeadWarpsDefault><<<
                num_q_heads, kCTAQHeadWarpsDefault * kWarpSize, 0, stream>>>(
                reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
                cache_position.data_ptr<int64_t>(),
                reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                q.stride(1),
                k.stride(1), k.stride(2),
                v.stride(1), v.stride(2),
                out.stride(1),
                static_cast<float>(scale),
                static_cast<int>(max_seq_len),
                static_cast<int>(num_kv_groups));
        }
    } else if (warps_per_cta == kCTAQHeadWarpsWide) {
        if (lane_layout == 0) {
            if (prefetch_mode == 0) {
                decode_attn_v4_cta_qhead_kernel<kCTAQHeadWarpsWide, 0><<<
                    num_q_heads, kCTAQHeadWarpsWide * kWarpSize, 0, stream>>>(
                    reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
                    cache_position.data_ptr<int64_t>(),
                    reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                    q.stride(1),
                    k.stride(1), k.stride(2),
                    v.stride(1), v.stride(2),
                    out.stride(1),
                    static_cast<float>(scale),
                    static_cast<int>(max_seq_len),
                    static_cast<int>(num_kv_groups),
                    static_cast<int>(merge_mode));
            } else if (prefetch_mode == 1) {
                decode_attn_v4_cta_qhead_kernel<kCTAQHeadWarpsWide, 1><<<
                    num_q_heads, kCTAQHeadWarpsWide * kWarpSize, 0, stream>>>(
                    reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
                    cache_position.data_ptr<int64_t>(),
                    reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                    q.stride(1),
                    k.stride(1), k.stride(2),
                    v.stride(1), v.stride(2),
                    out.stride(1),
                    static_cast<float>(scale),
                    static_cast<int>(max_seq_len),
                    static_cast<int>(num_kv_groups),
                    0);
            } else if (prefetch_mode == 2) {
                decode_attn_v4_cta_qhead_kernel<kCTAQHeadWarpsWide, 2><<<
                    num_q_heads, kCTAQHeadWarpsWide * kWarpSize, 0, stream>>>(
                    reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
                    cache_position.data_ptr<int64_t>(),
                    reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                    q.stride(1),
                    k.stride(1), k.stride(2),
                    v.stride(1), v.stride(2),
                    out.stride(1),
                    static_cast<float>(scale),
                    static_cast<int>(max_seq_len),
                    static_cast<int>(num_kv_groups),
                    0);
            } else {
                decode_attn_v4_cta_qhead_cpasync_k_kernel<kCTAQHeadWarpsWide><<<
                    num_q_heads, kCTAQHeadWarpsWide * kWarpSize, 0, stream>>>(
                    reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
                    cache_position.data_ptr<int64_t>(),
                    reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                    q.stride(1),
                    k.stride(1), k.stride(2),
                    v.stride(1), v.stride(2),
                    out.stride(1),
                    static_cast<float>(scale),
                    static_cast<int>(max_seq_len),
                    static_cast<int>(num_kv_groups));
            }
        } else if (lane_layout == 1) {
            decode_attn_v4_cta_qhead_contig_kernel<kCTAQHeadWarpsWide><<<
                num_q_heads, kCTAQHeadWarpsWide * kWarpSize, 0, stream>>>(
                reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
                cache_position.data_ptr<int64_t>(),
                reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                q.stride(1),
                k.stride(1), k.stride(2),
                v.stride(1), v.stride(2),
                out.stride(1),
                static_cast<float>(scale),
                static_cast<int>(max_seq_len),
                static_cast<int>(num_kv_groups));
        } else {
            decode_attn_v4_cta_qhead_group8_kernel<kCTAQHeadWarpsWide><<<
                num_q_heads, kCTAQHeadWarpsWide * kWarpSize, 0, stream>>>(
                reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
                cache_position.data_ptr<int64_t>(),
                reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                q.stride(1),
                k.stride(1), k.stride(2),
                v.stride(1), v.stride(2),
                out.stride(1),
                static_cast<float>(scale),
                static_cast<int>(max_seq_len),
                static_cast<int>(num_kv_groups));
        }
    } else {
        if (lane_layout == 0) {
            if (prefetch_mode == 0) {
                decode_attn_v4_cta_qhead_kernel<kCTAQHeadWarpsXL, 0><<<
                    num_q_heads, kCTAQHeadWarpsXL * kWarpSize, 0, stream>>>(
                    reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
                    cache_position.data_ptr<int64_t>(),
                    reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                    q.stride(1),
                    k.stride(1), k.stride(2),
                    v.stride(1), v.stride(2),
                    out.stride(1),
                    static_cast<float>(scale),
                    static_cast<int>(max_seq_len),
                    static_cast<int>(num_kv_groups),
                    static_cast<int>(merge_mode));
            } else if (prefetch_mode == 1) {
                decode_attn_v4_cta_qhead_kernel<kCTAQHeadWarpsXL, 1><<<
                    num_q_heads, kCTAQHeadWarpsXL * kWarpSize, 0, stream>>>(
                    reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
                    cache_position.data_ptr<int64_t>(),
                    reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                    q.stride(1),
                    k.stride(1), k.stride(2),
                    v.stride(1), v.stride(2),
                    out.stride(1),
                    static_cast<float>(scale),
                    static_cast<int>(max_seq_len),
                    static_cast<int>(num_kv_groups),
                    0);
            } else if (prefetch_mode == 2) {
                decode_attn_v4_cta_qhead_kernel<kCTAQHeadWarpsXL, 2><<<
                    num_q_heads, kCTAQHeadWarpsXL * kWarpSize, 0, stream>>>(
                    reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
                    cache_position.data_ptr<int64_t>(),
                    reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                    q.stride(1),
                    k.stride(1), k.stride(2),
                    v.stride(1), v.stride(2),
                    out.stride(1),
                    static_cast<float>(scale),
                    static_cast<int>(max_seq_len),
                    static_cast<int>(num_kv_groups),
                    0);
            } else {
                TORCH_CHECK(false, "cpasync-k-stage only supports warps_per_cta=8");
            }
        } else if (lane_layout == 1) {
            decode_attn_v4_cta_qhead_contig_kernel<kCTAQHeadWarpsXL><<<
                num_q_heads, kCTAQHeadWarpsXL * kWarpSize, 0, stream>>>(
                reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
                cache_position.data_ptr<int64_t>(),
                reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                q.stride(1),
                k.stride(1), k.stride(2),
                v.stride(1), v.stride(2),
                out.stride(1),
                static_cast<float>(scale),
                static_cast<int>(max_seq_len),
                static_cast<int>(num_kv_groups));
        } else {
            decode_attn_v4_cta_qhead_group8_kernel<kCTAQHeadWarpsXL><<<
                num_q_heads, kCTAQHeadWarpsXL * kWarpSize, 0, stream>>>(
                reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
                cache_position.data_ptr<int64_t>(),
                reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                q.stride(1),
                k.stride(1), k.stride(2),
                v.stride(1), v.stride(2),
                out.stride(1),
                static_cast<float>(scale),
                static_cast<int>(max_seq_len),
                static_cast<int>(num_kv_groups));
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
