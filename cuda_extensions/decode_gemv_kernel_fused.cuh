__global__ void fused_add_rmsnorm_gateup_k2048_hfma2_fwd_kernel(
    const at::Half* __restrict__ residual,
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ norm_weight,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ residual_out,
    at::Half* __restrict__ gate_up_out,
    float eps,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __half summed_sh[kNosplitSpecializedK2048];
    __shared__ __half normed_sh[kNosplitSpecializedK2048];
    __shared__ float warp_sums[kRowsPerBlock];
    __shared__ float inv_rms_sh;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * 4;
    const int row = row_base + warp;

    const __half* residual_half = reinterpret_cast<const __half*>(residual);
    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* norm_half = reinterpret_cast<const __half*>(norm_weight);
    const __half* w_half = reinterpret_cast<const __half*>(weight);
    __half* residual_out_half = reinterpret_cast<__half*>(residual_out);
    __half* gate_up_out_half = reinterpret_cast<__half*>(gate_up_out);

    float thread_sum_sq = 0.0f;
    for (int idx = threadIdx.x; idx < kNosplitSpecializedK2048; idx += kRow4Threads) {
        const __half summed = __hadd(residual_half[idx], x_half[idx]);
        summed_sh[idx] = summed;
        if (blockIdx.x == 0) {
            residual_out_half[idx] = summed;
        }
        const float summed_f = __half2float(summed);
        thread_sum_sq += summed_f * summed_f;
    }

    thread_sum_sq = warp_sum(thread_sum_sq);
    if (lane == 0) {
        warp_sums[warp] = thread_sum_sq;
    }
    __syncthreads();

    if (warp == 0) {
        float block_sum_sq = (lane < kRowsPerBlock) ? warp_sums[lane] : 0.0f;
        block_sum_sq = warp_sum(block_sum_sq);
        if (lane == 0) {
            inv_rms_sh = rsqrtf(block_sum_sq / static_cast<float>(kNosplitSpecializedK2048) + eps);
        }
    }
    __syncthreads();

    const float inv_rms = inv_rms_sh;
    for (int idx = threadIdx.x; idx < kNosplitSpecializedK2048; idx += kRow4Threads) {
        const float summed_f = __half2float(summed_sh[idx]);
        const float scale_f = __half2float(norm_half[idx]);
        normed_sh[idx] = __float2half(summed_f * inv_rms * scale_f);
    }
    __syncthreads();

    if (row < n) {
        const uint4* normed_vec = reinterpret_cast<const uint4*>(normed_sh);
        const uint4* w_row_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn);
        constexpr int vecs_k2048 = kNosplitSpecializedK2048 / kHalfPerVec128;
        __half2 h2_acc0 = __float2half2_rn(0.0f);
        __half2 h2_acc1 = __float2half2_rn(0.0f);

        #pragma unroll
        for (int vec_idx = lane; vec_idx < vecs_k2048; vec_idx += 64) {
            const uint4 x_vec0 = normed_vec[vec_idx];
            const uint4 w_vec0 = w_row_vec[vec_idx];
            const uint4 x_vec1 = normed_vec[vec_idx + 32];
            const uint4 w_vec1 = w_row_vec[vec_idx + 32];
            accumulate_hfma2_uint4(x_vec0, w_vec0, h2_acc0);
            accumulate_hfma2_uint4(x_vec1, w_vec1, h2_acc1);
        }

        const float2 f2_0 = __half22float2(h2_acc0);
        const float2 f2_1 = __half22float2(h2_acc1);
        float sum = f2_0.x + f2_0.y + f2_1.x + f2_1.y;
        sum = warp_sum(sum);
        if (lane == 0) {
            float result = sum;
            if (has_bias) {
                result += __half2float(reinterpret_cast<const __half*>(bias)[row]);
            }
            gate_up_out_half[row] = __float2half(result);
        }
    }
}

// ---------------------------------------------------------------------------
// Fused Input-RMSNorm + QKV GEMV: norm in shared memory, no write-back
// Grid: (n+3)/4 blocks, 128 threads (4 warps)
// ---------------------------------------------------------------------------
__global__ void fused_rmsnorm_qkv_k2048_hfma2_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ norm_weight,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    float eps,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __half x_sh[kNosplitSpecializedK2048];
    __shared__ float warp_sums[kRowsPerBlock];
    __shared__ float inv_rms_sh;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + warp;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* norm_half = reinterpret_cast<const __half*>(norm_weight);
    const __half* w_half = reinterpret_cast<const __half*>(weight);

    // Phase 1: Load x into shared memory + accumulate sum_of_squares
    float thread_sum_sq = 0.0f;
    for (int idx = threadIdx.x; idx < kNosplitSpecializedK2048; idx += kRow4Threads) {
        const __half val = x_half[idx];
        x_sh[idx] = val;
        const float val_f = __half2float(val);
        thread_sum_sq += val_f * val_f;
    }

    // Phase 2: Cross-warp reduction -> inv_rms
    thread_sum_sq = warp_sum(thread_sum_sq);
    if (lane == 0) {
        warp_sums[warp] = thread_sum_sq;
    }
    __syncthreads();

    if (warp == 0) {
        float block_sum_sq = (lane < kRowsPerBlock) ? warp_sums[lane] : 0.0f;
        block_sum_sq = warp_sum(block_sum_sq);
        if (lane == 0) {
            inv_rms_sh = rsqrtf(block_sum_sq / static_cast<float>(kNosplitSpecializedK2048) + eps);
        }
    }
    __syncthreads();

    // Phase 3: In-place normalization in shared memory
    const float inv_rms = inv_rms_sh;
    for (int idx = threadIdx.x; idx < kNosplitSpecializedK2048; idx += kRow4Threads) {
        const float val_f = __half2float(x_sh[idx]);
        const float scale_f = __half2float(norm_half[idx]);
        x_sh[idx] = __float2half(val_f * inv_rms * scale_f);
    }
    __syncthreads();

    // Phase 4: Standard hfma2 GEMV using normalized x_sh
    if (row < n) {
        const uint4* x_vec = reinterpret_cast<const uint4*>(x_sh);
        const uint4* w_row_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn);
        constexpr int vecs_k2048 = kNosplitSpecializedK2048 / kHalfPerVec128;
        __half2 h2_acc0 = __float2half2_rn(0.0f);
        __half2 h2_acc1 = __float2half2_rn(0.0f);

        #pragma unroll
        for (int vec_idx = lane; vec_idx < vecs_k2048; vec_idx += 64) {
            const uint4 x_vec0 = x_vec[vec_idx];
            const uint4 w_vec0 = w_row_vec[vec_idx];
            const uint4 x_vec1 = x_vec[vec_idx + 32];
            const uint4 w_vec1 = w_row_vec[vec_idx + 32];
            accumulate_hfma2_uint4(x_vec0, w_vec0, h2_acc0);
            accumulate_hfma2_uint4(x_vec1, w_vec1, h2_acc1);
        }

        const float2 f2_0 = __half22float2(h2_acc0);
        const float2 f2_1 = __half22float2(h2_acc1);
        float sum = f2_0.x + f2_0.y + f2_1.x + f2_1.y;
        sum = warp_sum(sum);
        if (lane == 0) {
            float result = sum;
            if (has_bias) {
                result += __half2float(reinterpret_cast<const __half*>(bias)[row]);
            }
            reinterpret_cast<__half*>(out)[row] = __float2half(result);
        }
    }
}

__global__ void fused_silu_downproj_k6144_hfma2_fwd_kernel(
    const at::Half* __restrict__ gate_up,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __half x_sh[kNosplitSpecializedTileK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * 4;
    const int row = row_base + warp;

    const __half* gate_half = reinterpret_cast<const __half*>(gate_up);
    const __half* up_half = gate_half + 6144;
    const __half* w_half = reinterpret_cast<const __half*>(weight);
    uint4* x_sh_vec = reinterpret_cast<uint4*>(x_sh);

    const int vecs_per_tile = kNosplitSpecializedTileK6144 / kHalfPerVec128;
    float sum = 0.0f;

    #pragma unroll
    for (int tile_idx = 0; tile_idx < 3; ++tile_idx) {
        const int tile_start = tile_idx * kNosplitSpecializedTileK6144;

        for (int idx = threadIdx.x; idx < kNosplitSpecializedTileK6144; idx += kRow4Threads) {
            const float gate_f = __half2float(gate_half[tile_start + idx]);
            const float up_f = __half2float(up_half[tile_start + idx]);
            const float silu_f = gate_f / (1.0f + __expf(-gate_f));
            x_sh[idx] = __float2half(silu_f * up_f);
        }
        __syncthreads();

        if (row < n) {
            const uint4* w_row_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn + tile_start);
            __half2 h2_acc0 = __float2half2_rn(0.0f);
            __half2 h2_acc1 = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane; vec_idx < vecs_per_tile; vec_idx += 64) {
                const uint4 x_vec0 = x_sh_vec[vec_idx];
                const uint4 w_vec0 = w_row_vec[vec_idx];
                const uint4 x_vec1 = x_sh_vec[vec_idx + 32];
                const uint4 w_vec1 = w_row_vec[vec_idx + 32];
                accumulate_hfma2_uint4(x_vec0, w_vec0, h2_acc0);
                accumulate_hfma2_uint4(x_vec1, w_vec1, h2_acc1);
            }

            const float2 f2_0 = __half22float2(h2_acc0);
            const float2 f2_1 = __half22float2(h2_acc1);
            sum += f2_0.x + f2_0.y + f2_1.x + f2_1.y;
        }
        __syncthreads();
    }

    if (row < n) {
        sum = warp_sum(sum);
        if (lane == 0) {
            float result = sum;
            if (has_bias) {
                result += __half2float(reinterpret_cast<const __half*>(bias)[row]);
            }
            reinterpret_cast<__half*>(out)[row] = __float2half(result);
        }
    }
}

__global__ void fused_gateup_silu_k2048_hfma2_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ gate_up_weight,
    at::Half* __restrict__ hidden_out,
    int64_t stride_wn) {
    __shared__ __half x_sh[kNosplitSpecializedK2048];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * 4;
    const int row = row_base + warp;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(gate_up_weight);
    __half* out_half = reinterpret_cast<__half*>(hidden_out);

    const uint4* x_global_vec = reinterpret_cast<const uint4*>(x_half);
    uint4* x_sh_vec = reinterpret_cast<uint4*>(x_sh);
    constexpr int vecs_k2048 = kNosplitSpecializedK2048 / kHalfPerVec128;

    if (threadIdx.x < vecs_k2048 / 2) {
        x_sh_vec[threadIdx.x] = x_global_vec[threadIdx.x];
        x_sh_vec[threadIdx.x + vecs_k2048 / 2] = x_global_vec[threadIdx.x + vecs_k2048 / 2];
    }
    __syncthreads();

    if (row < 6144) {
        const uint4* gate_row_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn);
        const uint4* up_row_vec = reinterpret_cast<const uint4*>(w_half + (row + 6144) * stride_wn);
        __half2 gate_acc0 = __float2half2_rn(0.0f);
        __half2 gate_acc1 = __float2half2_rn(0.0f);
        __half2 up_acc0 = __float2half2_rn(0.0f);
        __half2 up_acc1 = __float2half2_rn(0.0f);

        #pragma unroll
        for (int vec_idx = lane; vec_idx < vecs_k2048; vec_idx += 64) {
            const uint4 x_vec0 = x_sh_vec[vec_idx];
            const uint4 x_vec1 = x_sh_vec[vec_idx + 32];
            accumulate_hfma2_uint4(x_vec0, gate_row_vec[vec_idx], gate_acc0);
            accumulate_hfma2_uint4(x_vec1, gate_row_vec[vec_idx + 32], gate_acc1);
            accumulate_hfma2_uint4(x_vec0, up_row_vec[vec_idx], up_acc0);
            accumulate_hfma2_uint4(x_vec1, up_row_vec[vec_idx + 32], up_acc1);
        }

        const float2 gate_f0 = __half22float2(gate_acc0);
        const float2 gate_f1 = __half22float2(gate_acc1);
        const float2 up_f0 = __half22float2(up_acc0);
        const float2 up_f1 = __half22float2(up_acc1);
        float gate_sum = gate_f0.x + gate_f0.y + gate_f1.x + gate_f1.y;
        float up_sum = up_f0.x + up_f0.y + up_f1.x + up_f1.y;
        gate_sum = warp_sum(gate_sum);
        up_sum = warp_sum(up_sum);
        if (lane == 0) {
            const float silu = gate_sum / (1.0f + __expf(-gate_sum));
            out_half[row] = __float2half(silu * up_sum);
        }
    }
}

__global__ void fused_gateup_silu_downproj_mega_k2048_k6144_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ gate_up_weight,
    const at::Half* __restrict__ down_proj_weight,
    at::Half* __restrict__ out,
    int64_t gate_up_stride_wn,
    int64_t down_proj_stride_wn) {
    __shared__ __half x_sh[kNosplitSpecializedK2048];
    __shared__ float z_sh[kFusedMegaHiddenBatch];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int hidden_base = blockIdx.x * kFusedMegaHiddenBatch;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* gate_w_half = reinterpret_cast<const __half*>(gate_up_weight);
    const __half* down_w_half = reinterpret_cast<const __half*>(down_proj_weight);
    __half* out_half = reinterpret_cast<__half*>(out);

    const uint4* x_global_vec = reinterpret_cast<const uint4*>(x_half);
    uint4* x_sh_vec = reinterpret_cast<uint4*>(x_sh);
    constexpr int vecs_k2048 = kNosplitSpecializedK2048 / kHalfPerVec128;

    if (threadIdx.x < vecs_k2048 / 2) {
        x_sh_vec[threadIdx.x] = x_global_vec[threadIdx.x];
        x_sh_vec[threadIdx.x + vecs_k2048 / 2] = x_global_vec[threadIdx.x + vecs_k2048 / 2];
    }
    __syncthreads();

    #pragma unroll
    for (int sub = 0; sub < kFusedMegaHiddenBatch / kRowsPerBlockAsync; ++sub) {
        const int hidden_idx = hidden_base + sub * kRowsPerBlockAsync + warp;
        if (hidden_idx < 6144) {
            const uint4* gate_row_vec = reinterpret_cast<const uint4*>(gate_w_half + hidden_idx * gate_up_stride_wn);
            const uint4* up_row_vec = reinterpret_cast<const uint4*>(gate_w_half + (hidden_idx + 6144) * gate_up_stride_wn);
            __half2 gate_acc0 = __float2half2_rn(0.0f);
            __half2 gate_acc1 = __float2half2_rn(0.0f);
            __half2 up_acc0 = __float2half2_rn(0.0f);
            __half2 up_acc1 = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane; vec_idx < vecs_k2048; vec_idx += 64) {
                const uint4 x_vec0 = x_sh_vec[vec_idx];
                const uint4 x_vec1 = x_sh_vec[vec_idx + 32];
                accumulate_hfma2_uint4(x_vec0, gate_row_vec[vec_idx], gate_acc0);
                accumulate_hfma2_uint4(x_vec1, gate_row_vec[vec_idx + 32], gate_acc1);
                accumulate_hfma2_uint4(x_vec0, up_row_vec[vec_idx], up_acc0);
                accumulate_hfma2_uint4(x_vec1, up_row_vec[vec_idx + 32], up_acc1);
            }

            const float2 gate_f0 = __half22float2(gate_acc0);
            const float2 gate_f1 = __half22float2(gate_acc1);
            const float2 up_f0 = __half22float2(up_acc0);
            const float2 up_f1 = __half22float2(up_acc1);
            float gate_sum = gate_f0.x + gate_f0.y + gate_f1.x + gate_f1.y;
            float up_sum = up_f0.x + up_f0.y + up_f1.x + up_f1.y;
            gate_sum = warp_sum(gate_sum);
            up_sum = warp_sum(up_sum);
            if (lane == 0) {
                z_sh[sub * kRowsPerBlockAsync + warp] = gate_sum / (1.0f + __expf(-gate_sum)) * up_sum;
            }
        } else if (lane == 0) {
            z_sh[sub * kRowsPerBlockAsync + warp] = 0.0f;
        }
    }
    __syncthreads();

    for (int row = warp; row < 2048; row += kRowsPerBlockAsync) {
        float contrib = 0.0f;
        #pragma unroll
        for (int z_offset = 0; z_offset < kFusedMegaHiddenBatch; z_offset += 32) {
            const int hidden_idx = hidden_base + z_offset + lane;
            if (hidden_idx < 6144) {
                contrib += z_sh[z_offset + lane] * __half2float(down_w_half[row * down_proj_stride_wn + hidden_idx]);
            }
        }
        contrib = warp_sum(contrib);
        if (lane == 0) {
            atomicAdd(out_half + row, __float2half(contrib));
        }
    }
}

__global__ void fused_gateup_silu_downproj_mega_partial_k2048_k6144_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ gate_up_weight,
    const at::Half* __restrict__ down_proj_weight,
    at::Half* __restrict__ partial_out,
    int64_t gate_up_stride_wn,
    int64_t down_proj_stride_wn) {
    __shared__ __align__(16) __half x_sh[kNosplitSpecializedK2048];
    __shared__ float z_sh[kFusedMegaHiddenBatch];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int hidden_base = blockIdx.x * kFusedMegaHiddenBatch;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* gate_w_half = reinterpret_cast<const __half*>(gate_up_weight);
    const __half* down_w_half = reinterpret_cast<const __half*>(down_proj_weight);
    __half* partial_half = reinterpret_cast<__half*>(partial_out);

    const uint4* x_global_vec = reinterpret_cast<const uint4*>(x_half);
    uint4* x_sh_vec = reinterpret_cast<uint4*>(x_sh);
    constexpr int vecs_k2048 = kNosplitSpecializedK2048 / kHalfPerVec128;

    if (threadIdx.x < vecs_k2048 / 2) {
        x_sh_vec[threadIdx.x] = x_global_vec[threadIdx.x];
        x_sh_vec[threadIdx.x + vecs_k2048 / 2] = x_global_vec[threadIdx.x + vecs_k2048 / 2];
    }
    __syncthreads();

    #pragma unroll
    for (int sub = 0; sub < kFusedMegaHiddenBatch / kRowsPerBlockAsync; ++sub) {
        const int hidden_idx = hidden_base + sub * kRowsPerBlockAsync + warp;
        if (hidden_idx < 6144) {
            const uint4* gate_row_vec = reinterpret_cast<const uint4*>(gate_w_half + hidden_idx * gate_up_stride_wn);
            const uint4* up_row_vec = reinterpret_cast<const uint4*>(gate_w_half + (hidden_idx + 6144) * gate_up_stride_wn);
            __half2 gate_acc0 = __float2half2_rn(0.0f);
            __half2 gate_acc1 = __float2half2_rn(0.0f);
            __half2 up_acc0 = __float2half2_rn(0.0f);
            __half2 up_acc1 = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane; vec_idx < vecs_k2048; vec_idx += 64) {
                const uint4 x_vec0 = x_sh_vec[vec_idx];
                const uint4 x_vec1 = x_sh_vec[vec_idx + 32];
                accumulate_hfma2_uint4(x_vec0, gate_row_vec[vec_idx], gate_acc0);
                accumulate_hfma2_uint4(x_vec1, gate_row_vec[vec_idx + 32], gate_acc1);
                accumulate_hfma2_uint4(x_vec0, up_row_vec[vec_idx], up_acc0);
                accumulate_hfma2_uint4(x_vec1, up_row_vec[vec_idx + 32], up_acc1);
            }

            const float2 gate_f0 = __half22float2(gate_acc0);
            const float2 gate_f1 = __half22float2(gate_acc1);
            const float2 up_f0 = __half22float2(up_acc0);
            const float2 up_f1 = __half22float2(up_acc1);
            float gate_sum = gate_f0.x + gate_f0.y + gate_f1.x + gate_f1.y;
            float up_sum = up_f0.x + up_f0.y + up_f1.x + up_f1.y;
            gate_sum = warp_sum(gate_sum);
            up_sum = warp_sum(up_sum);
            if (lane == 0) {
                z_sh[sub * kRowsPerBlockAsync + warp] = gate_sum / (1.0f + __expf(-gate_sum)) * up_sum;
            }
        } else if (lane == 0) {
            z_sh[sub * kRowsPerBlockAsync + warp] = 0.0f;
        }
    }
    __syncthreads();

    for (int row = warp; row < 2048; row += kRowsPerBlockAsync) {
        float contrib = 0.0f;
        #pragma unroll
        for (int z_offset = 0; z_offset < kFusedMegaHiddenBatch; z_offset += 32) {
            const int hidden_idx = hidden_base + z_offset + lane;
            if (hidden_idx < 6144) {
                contrib += z_sh[z_offset + lane] * __half2float(down_w_half[row * down_proj_stride_wn + hidden_idx]);
            }
        }
        contrib = warp_sum(contrib);
        if (lane == 0) {
            partial_half[blockIdx.x * 2048 + row] = __float2half(contrib);
        }
    }
}

constexpr int kSoftsyncProfileSlots = 12;
enum SoftsyncProfileSlot : int {
    SS_PROF_KERNEL_START = 0,
    SS_PROF_X_LOAD_DONE = 1,
    SS_PROF_GATE_UP_COMPUTE_DONE = 2,
    SS_PROF_BARRIER_START = 3,
    SS_PROF_BARRIER_DONE = 4,
    SS_PROF_HIDDEN_LOAD_DONE = 5,
    SS_PROF_TILE0_LOAD_DONE = 6,
    SS_PROF_TILE0_COMPUTE_DONE = 7,
    SS_PROF_TILE1_LOAD_DONE = 8,
    SS_PROF_TILE1_COMPUTE_DONE = 9,
    SS_PROF_TILE2_COMPUTE_DONE = 10,
    SS_PROF_KERNEL_DONE = 11,
};

__global__ void fused_gateup_silu_downproj_softsync_k2048_k6144_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ gate_up_weight,
    const at::Half* __restrict__ down_proj_weight,
    int* __restrict__ barrier_count,
    int* __restrict__ barrier_phase,
    at::Half* __restrict__ hidden_out,
    at::Half* __restrict__ out,
    int64_t gate_up_stride_wn,
    int64_t down_proj_stride_wn) {
    __shared__ __align__(16) __half x_small_sh[kNosplitSpecializedK2048];
    __shared__ __align__(16) uint4 x_tile_sh[2][kAsyncVecsPerTileK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* gate_w_half = reinterpret_cast<const __half*>(gate_up_weight);
    const __half* down_w_half = reinterpret_cast<const __half*>(down_proj_weight);
    __half* hidden_half = reinterpret_cast<__half*>(hidden_out);
    __half* out_half = reinterpret_cast<__half*>(out);

    const uint4* x_global_vec = reinterpret_cast<const uint4*>(x_half);
    uint4* x_small_vec = reinterpret_cast<uint4*>(x_small_sh);
    constexpr int vecs_k2048 = kNosplitSpecializedK2048 / kHalfPerVec128;

    if (threadIdx.x < vecs_k2048 / 2) {
        x_small_vec[threadIdx.x] = x_global_vec[threadIdx.x];
        x_small_vec[threadIdx.x + vecs_k2048 / 2] = x_global_vec[threadIdx.x + vecs_k2048 / 2];
    }
    __syncthreads();

    for (int hidden_base = blockIdx.x * kRowsPerBlockAsync; hidden_base < 6144; hidden_base += gridDim.x * kRowsPerBlockAsync) {
        const int hidden_idx = hidden_base + warp;
        if (hidden_idx < 6144) {
            const uint4* gate_row_vec = reinterpret_cast<const uint4*>(gate_w_half + hidden_idx * gate_up_stride_wn);
            const uint4* up_row_vec = reinterpret_cast<const uint4*>(gate_w_half + (hidden_idx + 6144) * gate_up_stride_wn);
            __half2 gate_acc0 = __float2half2_rn(0.0f);
            __half2 gate_acc1 = __float2half2_rn(0.0f);
            __half2 up_acc0 = __float2half2_rn(0.0f);
            __half2 up_acc1 = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane; vec_idx < vecs_k2048; vec_idx += 64) {
                const uint4 x_vec0 = x_small_vec[vec_idx];
                const uint4 x_vec1 = x_small_vec[vec_idx + 32];
                accumulate_hfma2_uint4(x_vec0, gate_row_vec[vec_idx], gate_acc0);
                accumulate_hfma2_uint4(x_vec1, gate_row_vec[vec_idx + 32], gate_acc1);
                accumulate_hfma2_uint4(x_vec0, up_row_vec[vec_idx], up_acc0);
                accumulate_hfma2_uint4(x_vec1, up_row_vec[vec_idx + 32], up_acc1);
            }

            const float2 gate_f0 = __half22float2(gate_acc0);
            const float2 gate_f1 = __half22float2(gate_acc1);
            const float2 up_f0 = __half22float2(up_acc0);
            const float2 up_f1 = __half22float2(up_acc1);
            float gate_sum = gate_f0.x + gate_f0.y + gate_f1.x + gate_f1.y;
            float up_sum = up_f0.x + up_f0.y + up_f1.x + up_f1.y;
            gate_sum = warp_sum(gate_sum);
            up_sum = warp_sum(up_sum);
            if (lane == 0) {
                hidden_half[hidden_idx] = __float2half(gate_sum / (1.0f + __expf(-gate_sum)) * up_sum);
            }
        }
    }

    software_grid_sync(barrier_count, barrier_phase, gridDim.x);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    const int row_base = blockIdx.x * kRowsPerBlockAsync;
    const int row = row_base + warp;

    for (int vec_idx = threadIdx.x; vec_idx < kAsyncVecsPerTileK6144; vec_idx += kRow8Threads) {
        if constexpr (kUseCpAsync) {
            cp_async_cg_16(&x_tile_sh[0][vec_idx], reinterpret_cast<const uint4*>(hidden_half) + vec_idx);
        } else {
            x_tile_sh[0][vec_idx] = reinterpret_cast<const uint4*>(hidden_half)[vec_idx];
        }
    }
    if constexpr (kUseCpAsync) {
        cp_async_commit_group();
        cp_async_wait_group<0>();
    }
    __syncthreads();

    float sum = 0.0f;
    #pragma unroll
    for (int tile_idx = 0; tile_idx < 6; ++tile_idx) {
        const int stage = tile_idx & 1;
        const int tile_start = tile_idx * kNosplitAsyncTileK6144;

        if (tile_idx + 1 < 6) {
            const int next_tile_start = tile_start + kNosplitAsyncTileK6144;
            const uint4* x_next_vec = reinterpret_cast<const uint4*>(hidden_half + next_tile_start);
            for (int vec_idx = threadIdx.x; vec_idx < kAsyncVecsPerTileK6144; vec_idx += kRow8Threads) {
                if constexpr (kUseCpAsync) {
                    cp_async_cg_16(&x_tile_sh[stage ^ 1][vec_idx], x_next_vec + vec_idx);
                } else {
                    x_tile_sh[stage ^ 1][vec_idx] = x_next_vec[vec_idx];
                }
            }
            if constexpr (kUseCpAsync) {
                cp_async_commit_group();
            }
        }

        if (row < 2048) {
            const uint4* x_sh_stage_vec = x_tile_sh[stage];
            const uint4* w_row_vec = reinterpret_cast<const uint4*>(down_w_half + row * down_proj_stride_wn + tile_start);
            __half2 h2_acc0 = __float2half2_rn(0.0f);
            __half2 h2_acc1 = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 64) {
                const uint4 x_vec0 = x_sh_stage_vec[vec_idx];
                const uint4 w_vec0 = w_row_vec[vec_idx];
                const uint4 x_vec1 = x_sh_stage_vec[vec_idx + 32];
                const uint4 w_vec1 = w_row_vec[vec_idx + 32];
                accumulate_hfma2_uint4(x_vec0, w_vec0, h2_acc0);
                accumulate_hfma2_uint4(x_vec1, w_vec1, h2_acc1);
            }

            const float2 f2_0 = __half22float2(h2_acc0);
            const float2 f2_1 = __half22float2(h2_acc1);
            sum += f2_0.x + f2_0.y + f2_1.x + f2_1.y;
        }

        if constexpr (kUseCpAsync) {
            if (tile_idx + 1 < 6) {
                cp_async_wait_group<0>();
            }
        }
        __syncthreads();
    }

    if (row < 2048) {
        sum = warp_sum(sum);
        if (lane == 0) {
            out_half[row] = __float2half(sum);
        }
    }
}

template <bool kProfile>
__global__ void fused_gateup_silu_downproj_softsync_profile_k2048_k6144_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ gate_up_weight,
    const at::Half* __restrict__ down_proj_weight,
    int* __restrict__ barrier_count,
    int* __restrict__ barrier_phase,
    at::Half* __restrict__ hidden_out,
    at::Half* __restrict__ out,
    uint64_t* __restrict__ profile,
    int64_t gate_up_stride_wn,
    int64_t down_proj_stride_wn) {
    __shared__ __align__(16) __half x_small_sh[kNosplitSpecializedK2048];
    __shared__ __align__(16) uint4 x_tile_sh[2][kAsyncVecsPerTileK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* gate_w_half = reinterpret_cast<const __half*>(gate_up_weight);
    const __half* down_w_half = reinterpret_cast<const __half*>(down_proj_weight);
    __half* hidden_half = reinterpret_cast<__half*>(hidden_out);
    __half* out_half = reinterpret_cast<__half*>(out);

    const uint4* x_global_vec = reinterpret_cast<const uint4*>(x_half);
    uint4* x_small_vec = reinterpret_cast<uint4*>(x_small_sh);
    constexpr int vecs_k2048 = kNosplitSpecializedK2048 / kHalfPerVec128;

    // Record kernel start (thread 0 only)
    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kSoftsyncProfileSlots + SS_PROF_KERNEL_START] = read_globaltimer();
        }
    }

    if (threadIdx.x < vecs_k2048 / 2) {
        x_small_vec[threadIdx.x] = x_global_vec[threadIdx.x];
        x_small_vec[threadIdx.x + vecs_k2048 / 2] = x_global_vec[threadIdx.x + vecs_k2048 / 2];
    }
    __syncthreads();

    // Record x load done
    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kSoftsyncProfileSlots + SS_PROF_X_LOAD_DONE] = read_globaltimer();
        }
    }

    // Phase 1: gate_up GEMV + SiLU*mul
    for (int hidden_base = blockIdx.x * kRowsPerBlockAsync; hidden_base < 6144; hidden_base += gridDim.x * kRowsPerBlockAsync) {
        const int hidden_idx = hidden_base + warp;
        if (hidden_idx < 6144) {
            const uint4* gate_row_vec = reinterpret_cast<const uint4*>(gate_w_half + hidden_idx * gate_up_stride_wn);
            const uint4* up_row_vec = reinterpret_cast<const uint4*>(gate_w_half + (hidden_idx + 6144) * gate_up_stride_wn);
            __half2 gate_acc0 = __float2half2_rn(0.0f);
            __half2 gate_acc1 = __float2half2_rn(0.0f);
            __half2 up_acc0 = __float2half2_rn(0.0f);
            __half2 up_acc1 = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane; vec_idx < vecs_k2048; vec_idx += 64) {
                const uint4 x_vec0 = x_small_vec[vec_idx];
                const uint4 x_vec1 = x_small_vec[vec_idx + 32];
                accumulate_hfma2_uint4(x_vec0, gate_row_vec[vec_idx], gate_acc0);
                accumulate_hfma2_uint4(x_vec1, gate_row_vec[vec_idx + 32], gate_acc1);
                accumulate_hfma2_uint4(x_vec0, up_row_vec[vec_idx], up_acc0);
                accumulate_hfma2_uint4(x_vec1, up_row_vec[vec_idx + 32], up_acc1);
            }

            const float2 gate_f0 = __half22float2(gate_acc0);
            const float2 gate_f1 = __half22float2(gate_acc1);
            const float2 up_f0 = __half22float2(up_acc0);
            const float2 up_f1 = __half22float2(up_acc1);
            float gate_sum = gate_f0.x + gate_f0.y + gate_f1.x + gate_f1.y;
            float up_sum = up_f0.x + up_f0.y + up_f1.x + up_f1.y;
            gate_sum = warp_sum(gate_sum);
            up_sum = warp_sum(up_sum);
            if (lane == 0) {
                hidden_half[hidden_idx] = __float2half(gate_sum / (1.0f + __expf(-gate_sum)) * up_sum);
            }
        }
    }

    // Record gate_up compute done
    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kSoftsyncProfileSlots + SS_PROF_GATE_UP_COMPUTE_DONE] = read_globaltimer();
            profile[static_cast<int64_t>(blockIdx.x) * kSoftsyncProfileSlots + SS_PROF_BARRIER_START] = read_globaltimer();
        }
    }

    software_grid_sync(barrier_count, barrier_phase, gridDim.x);

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kSoftsyncProfileSlots + SS_PROF_BARRIER_DONE] = read_globaltimer();
        }
    }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    const int row_base = blockIdx.x * kRowsPerBlockAsync;
    const int row = row_base + warp;

    for (int vec_idx = threadIdx.x; vec_idx < kAsyncVecsPerTileK6144; vec_idx += kRow8Threads) {
        if constexpr (kUseCpAsync) {
            cp_async_cg_16(&x_tile_sh[0][vec_idx], reinterpret_cast<const uint4*>(hidden_half) + vec_idx);
        } else {
            x_tile_sh[0][vec_idx] = reinterpret_cast<const uint4*>(hidden_half)[vec_idx];
        }
    }
    if constexpr (kUseCpAsync) {
        cp_async_commit_group();
        cp_async_wait_group<0>();
    }
    __syncthreads();

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kSoftsyncProfileSlots + SS_PROF_HIDDEN_LOAD_DONE] = read_globaltimer();
        }
    }

    float sum = 0.0f;
    #pragma unroll
    for (int tile_idx = 0; tile_idx < 6; ++tile_idx) {
        const int stage = tile_idx & 1;
        const int tile_start = tile_idx * kNosplitAsyncTileK6144;

        if (tile_idx + 1 < 6) {
            const int next_tile_start = tile_start + kNosplitAsyncTileK6144;
            const uint4* x_next_vec = reinterpret_cast<const uint4*>(hidden_half + next_tile_start);
            for (int vec_idx = threadIdx.x; vec_idx < kAsyncVecsPerTileK6144; vec_idx += kRow8Threads) {
                if constexpr (kUseCpAsync) {
                    cp_async_cg_16(&x_tile_sh[stage ^ 1][vec_idx], x_next_vec + vec_idx);
                } else {
                    x_tile_sh[stage ^ 1][vec_idx] = x_next_vec[vec_idx];
                }
            }
            if constexpr (kUseCpAsync) {
                cp_async_commit_group();
            }
        }

        if (row < 2048) {
            const uint4* x_sh_stage_vec = x_tile_sh[stage];
            const uint4* w_row_vec = reinterpret_cast<const uint4*>(down_w_half + row * down_proj_stride_wn + tile_start);
            __half2 h2_acc0 = __float2half2_rn(0.0f);
            __half2 h2_acc1 = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 64) {
                const uint4 x_vec0 = x_sh_stage_vec[vec_idx];
                const uint4 w_vec0 = w_row_vec[vec_idx];
                const uint4 x_vec1 = x_sh_stage_vec[vec_idx + 32];
                const uint4 w_vec1 = w_row_vec[vec_idx + 32];
                accumulate_hfma2_uint4(x_vec0, w_vec0, h2_acc0);
                accumulate_hfma2_uint4(x_vec1, w_vec1, h2_acc1);
            }

            const float2 f2_0 = __half22float2(h2_acc0);
            const float2 f2_1 = __half22float2(h2_acc1);
            sum += f2_0.x + f2_0.y + f2_1.x + f2_1.y;
        }

        if constexpr (kUseCpAsync) {
            if (tile_idx + 1 < 6) {
                cp_async_wait_group<0>();
            }
        }
        __syncthreads();

        // Record per-tile compute done for first 3 tiles
        if constexpr (kProfile) {
            if (threadIdx.x == 0 && profile != nullptr) {
                if (tile_idx == 0) {
                    profile[static_cast<int64_t>(blockIdx.x) * kSoftsyncProfileSlots + SS_PROF_TILE0_LOAD_DONE] = 0;
                    profile[static_cast<int64_t>(blockIdx.x) * kSoftsyncProfileSlots + SS_PROF_TILE0_COMPUTE_DONE] = read_globaltimer();
                } else if (tile_idx == 1) {
                    profile[static_cast<int64_t>(blockIdx.x) * kSoftsyncProfileSlots + SS_PROF_TILE1_LOAD_DONE] = 0;
                    profile[static_cast<int64_t>(blockIdx.x) * kSoftsyncProfileSlots + SS_PROF_TILE1_COMPUTE_DONE] = read_globaltimer();
                } else if (tile_idx == 2) {
                    profile[static_cast<int64_t>(blockIdx.x) * kSoftsyncProfileSlots + SS_PROF_TILE2_COMPUTE_DONE] = read_globaltimer();
                }
            }
        }
    }

    if (row < 2048) {
        sum = warp_sum(sum);
        if (lane == 0) {
            out_half[row] = __float2half(sum);
        }
    }

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kSoftsyncProfileSlots + SS_PROF_KERNEL_DONE] = read_globaltimer();
        }
    }
}

// ---------------------------------------------------------------------------
// fused_post_attn_mlp softsync variant (AtomicGridSync, no -rdc / cooperative)
// Fuses: residual_add + RMSNorm + gate_up GEMV + SiLU*mul + down_proj GEMV + residual epilogue
// Template: kBlockThreads threads, kBlockWarps warps per block
// Supported: <1024,32> and <512,16>
// ---------------------------------------------------------------------------

template <int kBlockThreads, int kBlockWarps>
__launch_bounds__(kBlockThreads, 1)
__global__ void fused_post_attn_mlp_softsync_k2048_k6144_fwd_kernel(
    const at::Half* __restrict__ residual,
    const at::Half* __restrict__ attn_output,
    const at::Half* __restrict__ norm_weight,
    const at::Half* __restrict__ gate_up_weight,
    const at::Half* __restrict__ down_proj_weight,
    const at::Half* __restrict__ bias,
    unsigned int* __restrict__ barrier_count,
    unsigned int* __restrict__ barrier_generation,
    at::Half* __restrict__ hidden_out,
    at::Half* __restrict__ out,
    float eps,
    bool has_bias,
    int64_t gate_up_stride_wn,
    int64_t down_proj_stride_wn) {
    __shared__ __align__(16) __half residual_sh[kNosplitSpecializedK2048];
    __shared__ __align__(16) __half x_small_sh[kNosplitSpecializedK2048];
    __shared__ float warp_sums[kBlockWarps];
    __shared__ float inv_rms_sh;
    __shared__ __align__(16) uint4 x_tile_sh[2][kAsyncVecsPerTileK6144];
    __shared__ unsigned int local_gen_sh;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const __half* residual_half = reinterpret_cast<const __half*>(residual);
    const __half* attn_half = reinterpret_cast<const __half*>(attn_output);
    const __half* norm_half = reinterpret_cast<const __half*>(norm_weight);
    const __half* gate_w_half = reinterpret_cast<const __half*>(gate_up_weight);
    const __half* down_w_half = reinterpret_cast<const __half*>(down_proj_weight);
    const __half* bias_half = reinterpret_cast<const __half*>(bias);
    __half* hidden_half = reinterpret_cast<__half*>(hidden_out);
    __half* out_half = reinterpret_cast<__half*>(out);
    uint4* x_small_vec = reinterpret_cast<uint4*>(x_small_sh);
    constexpr int vecs_k2048 = kNosplitSpecializedK2048 / kHalfPerVec128;  // 256

    // Initialize AtomicGridSync
    if (threadIdx.x == 0) {
        local_gen_sh = barrier_generation[0];
    }
    __syncthreads();

    AtomicGridSync grid_sync{
        barrier_count,
        barrier_generation,
        static_cast<unsigned int>(gridDim.x),
        &local_gen_sh,
    };

    // --- Phase 0: residual add + RMSNorm ---
    float thread_sum_sq = 0.0f;
    for (int idx = threadIdx.x; idx < kNosplitSpecializedK2048; idx += kBlockThreads) {
        const __half summed = __hadd(residual_half[idx], attn_half[idx]);
        residual_sh[idx] = summed;
        const float summed_f = __half2float(summed);
        thread_sum_sq += summed_f * summed_f;
    }

    thread_sum_sq = warp_sum(thread_sum_sq);
    if (lane == 0) {
        warp_sums[warp] = thread_sum_sq;
    }
    __syncthreads();

    if (warp == 0) {
        float block_sum_sq = lane < kBlockWarps ? warp_sums[lane] : 0.0f;
        block_sum_sq = warp_sum(block_sum_sq);
        if (lane == 0) {
            inv_rms_sh = rsqrtf(block_sum_sq / static_cast<float>(kNosplitSpecializedK2048) + eps);
        }
    }
    __syncthreads();

    const float inv_rms = inv_rms_sh;
    for (int idx = threadIdx.x; idx < kNosplitSpecializedK2048; idx += kBlockThreads) {
        const float summed_f = __half2float(residual_sh[idx]);
        const float scale_f = __half2float(norm_half[idx]);
        x_small_sh[idx] = __float2half(summed_f * inv_rms * scale_f);
    }
    __syncthreads();

    // --- Phase 1: gate_up GEMV + SiLU fusion (grid-cycle) ---
    for (int hidden_base = blockIdx.x * kBlockWarps;
         hidden_base < 6144;
         hidden_base += gridDim.x * kBlockWarps) {
        const int hidden_idx = hidden_base + warp;
        if (hidden_idx < 6144) {
            const uint4* gate_row_vec = reinterpret_cast<const uint4*>(
                gate_w_half + hidden_idx * gate_up_stride_wn);
            const uint4* up_row_vec = reinterpret_cast<const uint4*>(
                gate_w_half + (hidden_idx + 6144) * gate_up_stride_wn);
            __half2 gate_acc0 = __float2half2_rn(0.0f);
            __half2 gate_acc1 = __float2half2_rn(0.0f);
            __half2 up_acc0 = __float2half2_rn(0.0f);
            __half2 up_acc1 = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane; vec_idx < vecs_k2048; vec_idx += 64) {
                const uint4 x_vec0 = x_small_vec[vec_idx];
                const uint4 x_vec1 = x_small_vec[vec_idx + 32];
                accumulate_hfma2_uint4(x_vec0, gate_row_vec[vec_idx], gate_acc0);
                accumulate_hfma2_uint4(x_vec1, gate_row_vec[vec_idx + 32], gate_acc1);
                accumulate_hfma2_uint4(x_vec0, up_row_vec[vec_idx], up_acc0);
                accumulate_hfma2_uint4(x_vec1, up_row_vec[vec_idx + 32], up_acc1);
            }

            const float2 gate_f0 = __half22float2(gate_acc0);
            const float2 gate_f1 = __half22float2(gate_acc1);
            const float2 up_f0 = __half22float2(up_acc0);
            const float2 up_f1 = __half22float2(up_acc1);
            float gate_sum = gate_f0.x + gate_f0.y + gate_f1.x + gate_f1.y;
            float up_sum = up_f0.x + up_f0.y + up_f1.x + up_f1.y;
            gate_sum = warp_sum(gate_sum);
            up_sum = warp_sum(up_sum);
            if (lane == 0) {
                hidden_half[hidden_idx] =
                    __float2half(gate_sum / (1.0f + __expf(-gate_sum)) * up_sum);
            }
        }
    }

    // --- AtomicGridSync barrier ---
    grid_sync.sync();

    // --- Phase 2: down_proj GEMV (hidden[6144] -> out[2048]) + residual epilogue ---
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    // Grid-cycle over output rows (needed when gridDim.x * kBlockWarps < 2048)
    for (int row_base = blockIdx.x * kBlockWarps;
         row_base < 2048;
         row_base += gridDim.x * kBlockWarps) {
        const int row = row_base + warp;

        // Load first tile into x_tile_sh[0]
        for (int vec_idx = threadIdx.x; vec_idx < kAsyncVecsPerTileK6144;
             vec_idx += kBlockThreads) {
            if constexpr (kUseCpAsync) {
                cp_async_cg_16(&x_tile_sh[0][vec_idx],
                               reinterpret_cast<const uint4*>(hidden_half) + vec_idx);
            } else {
                x_tile_sh[0][vec_idx] =
                    reinterpret_cast<const uint4*>(hidden_half)[vec_idx];
            }
        }
        if constexpr (kUseCpAsync) {
            cp_async_commit_group();
            cp_async_wait_group<0>();
        }
        __syncthreads();

        float sum = 0.0f;
        #pragma unroll
        for (int tile_idx = 0; tile_idx < 6; ++tile_idx) {
            const int stage = tile_idx & 1;
            const int tile_start = tile_idx * kNosplitAsyncTileK6144;

            if (tile_idx + 1 < 6) {
                const int next_tile_start = tile_start + kNosplitAsyncTileK6144;
                const uint4* x_next_vec =
                    reinterpret_cast<const uint4*>(hidden_half + next_tile_start);
                for (int vec_idx = threadIdx.x; vec_idx < kAsyncVecsPerTileK6144;
                     vec_idx += kBlockThreads) {
                    if constexpr (kUseCpAsync) {
                        cp_async_cg_16(&x_tile_sh[stage ^ 1][vec_idx], x_next_vec + vec_idx);
                    } else {
                        x_tile_sh[stage ^ 1][vec_idx] = x_next_vec[vec_idx];
                    }
                }
                if constexpr (kUseCpAsync) {
                    cp_async_commit_group();
                }
            }

            if (row < 2048) {
                const uint4* x_sh_stage_vec = x_tile_sh[stage];
                const uint4* w_row_vec = reinterpret_cast<const uint4*>(
                    down_w_half + row * down_proj_stride_wn + tile_start);
                __half2 h2_acc0 = __float2half2_rn(0.0f);
                __half2 h2_acc1 = __float2half2_rn(0.0f);

                #pragma unroll
                for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 64) {
                    const uint4 x_vec0 = x_sh_stage_vec[vec_idx];
                    const uint4 w_vec0 = w_row_vec[vec_idx];
                    const uint4 x_vec1 = x_sh_stage_vec[vec_idx + 32];
                    const uint4 w_vec1 = w_row_vec[vec_idx + 32];
                    accumulate_hfma2_uint4(x_vec0, w_vec0, h2_acc0);
                    accumulate_hfma2_uint4(x_vec1, w_vec1, h2_acc1);
                }

                const float2 f2_0 = __half22float2(h2_acc0);
                const float2 f2_1 = __half22float2(h2_acc1);
                sum += f2_0.x + f2_0.y + f2_1.x + f2_1.y;
            }

            if constexpr (kUseCpAsync) {
                if (tile_idx + 1 < 6) {
                    cp_async_wait_group<0>();
                }
            }
            __syncthreads();
        }

        if (row < 2048) {
            sum = warp_sum(sum);
            if (lane == 0) {
                float result = sum + __half2float(residual_sh[row]);
                if (has_bias) {
                    result += __half2float(bias_half[row]);
                }
                out_half[row] = __float2half(result);
            }
        }
    } // end row grid-cycle
}

