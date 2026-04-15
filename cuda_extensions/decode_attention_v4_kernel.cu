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
