template <typename scalar_t>
__global__ void splitk_matvec_fwd_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    float* __restrict__ scratch,
    int64_t stride_wn,
    int64_t stride_wk,
    int n,
    int k,
    int split_k) {
    const int row = blockIdx.x;
    const int split_idx = blockIdx.y;
    const int lane = threadIdx.x;
    if (row >= n) {
        return;
    }

    const int chunk_k = (k + split_k - 1) / split_k;
    const int start = split_idx * chunk_k;
    const int end = min(start + chunk_k, k);
    float sum = 0.0f;
    for (int idx = start + lane * 4; idx < end; idx += kThreads * 4) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int col = idx + i;
            if (col < end) {
                sum += to_float(x[col]) * to_float(weight[row * stride_wn + col * stride_wk]);
            }
        }
    }
    sum = warp_sum(sum);
    if (lane == 0) {
        atomicAdd(scratch + row, sum);
    }
}

template <typename scalar_t>
__global__ void splitk_matvec_finalize_kernel(
    const float* __restrict__ scratch,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int n,
    bool has_bias) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float acc = scratch[idx];
    if (has_bias) {
        acc += to_float(bias[idx]);
    }
    out[idx] = static_cast<scalar_t>(acc);
}

template <typename scalar_t>
__global__ void splitk_matvec_row4_fwd_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    float* __restrict__ scratch,
    int64_t stride_wn,
    int64_t stride_wk,
    int n,
    int k,
    int split_k) {
    const int row_base = blockIdx.x * 4;
    const int split_idx = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = row_base + warp;
    if (row >= n) {
        return;
    }

    const int chunk_k = (k + split_k - 1) / split_k;
    const int start = split_idx * chunk_k;
    const int end = min(start + chunk_k, k);
    float sum = 0.0f;
    for (int idx = start + lane * 4; idx < end; idx += kThreads * 4) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int col = idx + i;
            if (col < end) {
                sum += to_float(x[col]) * to_float(weight[row * stride_wn + col * stride_wk]);
            }
        }
    }
    sum = warp_sum(sum);
    if (lane == 0) {
        atomicAdd(scratch + row, sum);
    }
}

template <typename scalar_t>
__global__ void splitk_matvec_row8_shared_fwd_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    float* __restrict__ scratch,
    int64_t stride_wn,
    int64_t stride_wk,
    int n,
    int k,
    int split_k) {
    __shared__ float x_sh[kSharedTileK];

    const int row_base = blockIdx.x * 8;
    const int split_idx = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = row_base + warp;
    if (row >= n) {
        return;
    }

    const int chunk_k = (k + split_k - 1) / split_k;
    const int chunk_start = split_idx * chunk_k;
    const int chunk_end = min(chunk_start + chunk_k, k);
    float sum = 0.0f;

    for (int tile_start = chunk_start; tile_start < chunk_end; tile_start += kSharedTileK) {
        const int tile_elems = min(kSharedTileK, chunk_end - tile_start);
        if (threadIdx.x < tile_elems) {
            x_sh[threadIdx.x] = to_float(x[tile_start + threadIdx.x]);
        }
        __syncthreads();

        for (int idx = lane * 4; idx < tile_elems; idx += 32 * 4) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int local_k = idx + i;
                if (local_k < tile_elems) {
                    const int col = tile_start + local_k;
                    sum += x_sh[local_k] * to_float(weight[row * stride_wn + col * stride_wk]);
                }
            }
        }
        __syncthreads();
    }

    sum = warp_sum(sum);
    if (lane == 0) {
        atomicAdd(scratch + row, sum);
    }
}

__global__ void splitk_matvec_row4_shared_vec_half_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    float* __restrict__ scratch,
    int64_t stride_wn,
    int64_t stride_wk,
    int n,
    int k,
    int split_k) {
    __shared__ __half x_sh[kVecTileK];

    const int row_base = blockIdx.x * 4;
    const int split_idx = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = row_base + warp;
    if (row >= n) {
        return;
    }

    const int chunk_k = (k + split_k - 1) / split_k;
    const int chunk_start = split_idx * chunk_k;
    const int chunk_end = min(chunk_start + chunk_k, k);
    float sum = 0.0f;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);

    for (int tile_start = chunk_start; tile_start < chunk_end; tile_start += kVecTileK) {
        const int tile_elems = min(kVecTileK, chunk_end - tile_start);
        for (int idx = threadIdx.x; idx < tile_elems; idx += kRow4Threads) {
            x_sh[idx] = x_half[tile_start + idx];
        }
        __syncthreads();

        const bool aligned_vec = ((tile_start & 7) == 0);
        const int vec_limit = aligned_vec ? (tile_elems & ~7) : 0;
        for (int idx = lane * 8; idx < vec_limit; idx += 32 * 8) {
            const __half2* x2 = reinterpret_cast<const __half2*>(&x_sh[idx]);
            const __half2* w2 = reinterpret_cast<const __half2*>(
                &w_half[row * stride_wn + (tile_start + idx) * stride_wk]
            );
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 xf = __half22float2(x2[i]);
                const float2 wf = __half22float2(w2[i]);
                sum += xf.x * wf.x + xf.y * wf.y;
            }
        }

        for (int idx = vec_limit + lane; idx < tile_elems; idx += 32) {
            sum += __half2float(x_sh[idx]) * __half2float(w_half[row * stride_wn + (tile_start + idx) * stride_wk]);
        }
        __syncthreads();
    }

    sum = warp_sum(sum);
    if (lane == 0) {
        atomicAdd(scratch + row, sum);
    }
}

constexpr int kVecTileK_v2 = 1024;

template <bool kProfile>
__global__ void nosplit_matvec_k2048_hfma2_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    uint64_t* __restrict__ profile,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __half x_sh[kNosplitSpecializedK2048];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * 4;
    const int row = row_base + warp;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);

    const uint4* x_global_vec = reinterpret_cast<const uint4*>(x_half);
    uint4* x_sh_vec = reinterpret_cast<uint4*>(x_sh);

    const int vecs_k2048 = kNosplitSpecializedK2048 / kHalfPerVec128;
    maybe_record_profile<kProfile>(profile, kProfileKernelStart);
    if (threadIdx.x < vecs_k2048 / 2) {
        x_sh_vec[threadIdx.x] = x_global_vec[threadIdx.x];
        x_sh_vec[threadIdx.x + vecs_k2048 / 2] = x_global_vec[threadIdx.x + vecs_k2048 / 2];
    }
    __syncthreads();
    maybe_record_profile<kProfile>(profile, kProfileTile0LoadDone);

    if (row < n) {
        const uint4* w_row_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn);
        __half2 h2_acc0 = __float2half2_rn(0.0f);
        __half2 h2_acc1 = __float2half2_rn(0.0f);

        #pragma unroll
        for (int vec_idx = lane; vec_idx < vecs_k2048; vec_idx += 64) {
            const uint4 x_vec0 = x_sh_vec[vec_idx];
            const uint4 w_vec0 = w_row_vec[vec_idx];
            const uint4 x_vec1 = x_sh_vec[vec_idx + 32];
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
    if constexpr (kProfile) {
        __syncthreads();
        maybe_record_profile<kProfile>(profile, kProfileTile0ComputeDone);
        __syncthreads();
        maybe_record_profile<kProfile>(profile, kProfileKernelDone);
    }
}

template <bool kProfile>
__global__ void nosplit_matvec_k2048_row2_hfma2_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    uint64_t* __restrict__ profile,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __half x_sh[kNosplitSpecializedK2048];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * 2;
    const int row = row_base + warp;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);

    const uint4* x_global_vec = reinterpret_cast<const uint4*>(x_half);
    uint4* x_sh_vec = reinterpret_cast<uint4*>(x_sh);

    constexpr int vecs_k2048 = kNosplitSpecializedK2048 / kHalfPerVec128;
    maybe_record_profile<kProfile>(profile, kProfileKernelStart);
    if (threadIdx.x < vecs_k2048 / 4) {
        x_sh_vec[threadIdx.x] = x_global_vec[threadIdx.x];
        x_sh_vec[threadIdx.x + vecs_k2048 / 4] = x_global_vec[threadIdx.x + vecs_k2048 / 4];
        x_sh_vec[threadIdx.x + vecs_k2048 / 2] = x_global_vec[threadIdx.x + vecs_k2048 / 2];
        x_sh_vec[threadIdx.x + (vecs_k2048 * 3) / 4] = x_global_vec[threadIdx.x + (vecs_k2048 * 3) / 4];
    }
    __syncthreads();
    maybe_record_profile<kProfile>(profile, kProfileTile0LoadDone);

    if (row < n) {
        const uint4* w_row_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn);
        __half2 h2_acc0 = __float2half2_rn(0.0f);
        __half2 h2_acc1 = __float2half2_rn(0.0f);

        #pragma unroll
        for (int vec_idx = lane; vec_idx < vecs_k2048; vec_idx += 64) {
            const uint4 x_vec0 = x_sh_vec[vec_idx];
            const uint4 w_vec0 = w_row_vec[vec_idx];
            const uint4 x_vec1 = x_sh_vec[vec_idx + 32];
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
    if constexpr (kProfile) {
        __syncthreads();
        maybe_record_profile<kProfile>(profile, kProfileTile0ComputeDone);
        __syncthreads();
        maybe_record_profile<kProfile>(profile, kProfileKernelDone);
    }
}

// Page-class LUT for interleaved partition layout.
// Within each 256KB period (32 pages), pages with page_class==0 and ==1.
__device__ constexpr int kPc0Pages[16] = {0,2,4,6,8,10,12,14,17,19,21,23,25,27,29,31};
__device__ constexpr int kPc1Pages[16] = {1,3,5,7,9,11,13,15,16,18,20,22,24,26,28,30};

// HBM partition-aware k=2048 row2 GEMV kernel.
// Identical compute to k2048_row2_hfma2_fwd_kernel, but uses interleaved
// memory layout where each SM reads only from its "fast" 8KB pages.
__global__ void nosplit_matvec_k2048_row2_partition_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight_buf,   // flat interleaved 2x buffer
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    const int32_t* __restrict__ chunk_classes, // chunk_class per 2MB chunk
    int64_t sm_class_lo,                       // SM class bitfield bits 0-63
    int64_t sm_class_hi,                       // SM class bitfield bits 64-107
    int n,
    bool has_bias) {
    __shared__ __half x_sh[kNosplitSpecializedK2048];
    __shared__ int s_abs_page;  // absolute page index computed by thread 0

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * 2;
    const int row = row_base + warp;

    const __half* x_half = reinterpret_cast<const __half*>(x);

    const uint4* x_global_vec = reinterpret_cast<const uint4*>(x_half);
    uint4* x_sh_vec = reinterpret_cast<uint4*>(x_sh);

    constexpr int vecs_k2048 = kNosplitSpecializedK2048 / kHalfPerVec128;

    // Load x vector into shared memory (same as standard kernel).
    if (threadIdx.x < vecs_k2048 / 4) {
        x_sh_vec[threadIdx.x] = x_global_vec[threadIdx.x];
        x_sh_vec[threadIdx.x + vecs_k2048 / 4] = x_global_vec[threadIdx.x + vecs_k2048 / 4];
        x_sh_vec[threadIdx.x + vecs_k2048 / 2] = x_global_vec[threadIdx.x + vecs_k2048 / 2];
        x_sh_vec[threadIdx.x + (vecs_k2048 * 3) / 4] = x_global_vec[threadIdx.x + (vecs_k2048 * 3) / 4];
    }

    // Thread 0 computes the interleaved page address for this block's row-pair.
    if (threadIdx.x == 0) {
        const int smid = read_smid_int();
        const int sm_cls = (smid < 64)
            ? static_cast<int>((static_cast<uint64_t>(sm_class_lo) >> smid) & 1)
            : static_cast<int>((static_cast<uint64_t>(sm_class_hi) >> (smid - 64)) & 1);

        const int pair_idx = blockIdx.x;       // global row-pair index
        const int chunk_idx = pair_idx >> 7;   // / 128
        const int pair_in_chunk = pair_idx & 127;
        const int cc = chunk_classes[chunk_idx];
        const int target_pc = sm_cls ^ cc;

        const int* lut = (target_pc == 0) ? kPc0Pages : kPc1Pages;
        const int period = pair_in_chunk >> 4;     // / 16
        const int intra  = pair_in_chunk & 15;
        const int page_in_chunk = period * 32 + lut[intra];

        s_abs_page = (chunk_idx << 8) + page_in_chunk;  // chunk_idx * 256 + page_in_chunk
    }
    __syncthreads();

    if (row < n) {
        // Compute weight row address from interleaved layout.
        // Each page = 8KB = 4096 halves. Row-pair occupies 1 page, each row = 2048 halves.
        const __half* w_buf_half = reinterpret_cast<const __half*>(weight_buf);
        const uint4* w_row_vec = reinterpret_cast<const uint4*>(
            w_buf_half + static_cast<int64_t>(s_abs_page) * 4096 + warp * 2048);

        __half2 h2_acc0 = __float2half2_rn(0.0f);
        __half2 h2_acc1 = __float2half2_rn(0.0f);

        #pragma unroll
        for (int vec_idx = lane; vec_idx < vecs_k2048; vec_idx += 64) {
            const uint4 x_vec0 = x_sh_vec[vec_idx];
            const uint4 w_vec0 = w_row_vec[vec_idx];
            const uint4 x_vec1 = x_sh_vec[vec_idx + 32];
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

// V2 HBM partition-aware k=2048 row2 GEMV kernel.
// Eliminates ALL global/constant memory reads for page address computation.
// Key optimizations vs V1:
//   1) chunk_class_bits: uint64 bitfield kernel arg replaces chunk_classes[] global read
//   2) Arithmetic formula replaces device constant LUT (kPc0Pages/kPc1Pages)
//   3) Every thread computes page address independently from registers (no shmem, no shuffle)
//   4) Only ONE __syncthreads for x loading — identical to standard kernel
__global__ void nosplit_matvec_k2048_row2_partition_v2_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight_buf,   // flat interleaved 2x buffer
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    int64_t chunk_class_bits,                  // bitfield: bit i = chunk_class of chunk i
    int64_t sm_class_lo,                       // SM class bitfield bits 0-63
    int64_t sm_class_hi,                       // SM class bitfield bits 64-107
    int n,
    bool has_bias) {
    __shared__ __half x_sh[kNosplitSpecializedK2048];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * 2;
    const int row = row_base + warp;

    const __half* x_half = reinterpret_cast<const __half*>(x);

    const uint4* x_global_vec = reinterpret_cast<const uint4*>(x_half);
    uint4* x_sh_vec = reinterpret_cast<uint4*>(x_sh);

    constexpr int vecs_k2048 = kNosplitSpecializedK2048 / kHalfPerVec128;

    // Load x vector into shared memory (same as standard kernel).
    if (threadIdx.x < vecs_k2048 / 4) {
        x_sh_vec[threadIdx.x] = x_global_vec[threadIdx.x];
        x_sh_vec[threadIdx.x + vecs_k2048 / 4] = x_global_vec[threadIdx.x + vecs_k2048 / 4];
        x_sh_vec[threadIdx.x + vecs_k2048 / 2] = x_global_vec[threadIdx.x + vecs_k2048 / 2];
        x_sh_vec[threadIdx.x + (vecs_k2048 * 3) / 4] = x_global_vec[threadIdx.x + (vecs_k2048 * 3) / 4];
    }

    // Compute interleaved page address entirely from registers (zero extra memory).
    const int smid = read_smid_int();
    const int sm_cls = (smid < 64)
        ? static_cast<int>((static_cast<uint64_t>(sm_class_lo) >> smid) & 1)
        : static_cast<int>((static_cast<uint64_t>(sm_class_hi) >> (smid - 64)) & 1);

    const int pair_idx = blockIdx.x;
    const int chunk_idx = pair_idx >> 7;           // / 128 pairs per 2MB chunk
    const int pair_in_chunk = pair_idx & 127;
    const int cc = static_cast<int>((static_cast<uint64_t>(chunk_class_bits) >> chunk_idx) & 1);
    const int target_pc = sm_cls ^ cc;

    const int period = pair_in_chunk >> 4;         // / 16
    const int intra  = pair_in_chunk & 15;
    // Arithmetic formula equivalent to kPc0Pages[intra] / kPc1Pages[intra]:
    const int page_in_chunk = period * 32 + intra * 2 + ((intra >> 3) ^ target_pc);
    const int abs_page = (chunk_idx << 8) + page_in_chunk;

    __syncthreads();  // Wait for x loading (same as standard kernel, NO extra sync)

    if (row < n) {
        const __half* w_buf_half = reinterpret_cast<const __half*>(weight_buf);
        const uint4* w_row_vec = reinterpret_cast<const uint4*>(
            w_buf_half + static_cast<int64_t>(abs_page) * 4096 + warp * 2048);

        __half2 h2_acc0 = __float2half2_rn(0.0f);
        __half2 h2_acc1 = __float2half2_rn(0.0f);

        #pragma unroll
        for (int vec_idx = lane; vec_idx < vecs_k2048; vec_idx += 64) {
            const uint4 x_vec0 = x_sh_vec[vec_idx];
            const uint4 w_vec0 = w_row_vec[vec_idx];
            const uint4 x_vec1 = x_sh_vec[vec_idx + 32];
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

template <bool kProfile>
__global__ void nosplit_matvec_k6144_hfma2_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    uint64_t* __restrict__ profile,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __half x_sh[kNosplitSpecializedTileK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * 4;
    const int row = row_base + warp;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);
    uint4* x_sh_vec = reinterpret_cast<uint4*>(x_sh);

    const int vecs_per_tile = kNosplitSpecializedTileK6144 / kHalfPerVec128;
    float sum = 0.0f;
    maybe_record_profile<kProfile>(profile, kProfileKernelStart);

    #pragma unroll
    for (int tile_idx = 0; tile_idx < 3; ++tile_idx) {
        const int tile_start = tile_idx * kNosplitSpecializedTileK6144;
        const uint4* x_tile_vec = reinterpret_cast<const uint4*>(x_half + tile_start);

        if (threadIdx.x < vecs_per_tile / 2) {
            x_sh_vec[threadIdx.x] = x_tile_vec[threadIdx.x];
            x_sh_vec[threadIdx.x + vecs_per_tile / 2] = x_tile_vec[threadIdx.x + vecs_per_tile / 2];
        }
        __syncthreads();
        if constexpr (kProfile) {
            maybe_record_profile<kProfile>(profile, kProfileTile0LoadDone + tile_idx * 2);
        }

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
        if constexpr (kProfile) {
            maybe_record_profile<kProfile>(profile, kProfileTile0ComputeDone + tile_idx * 2);
        }
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
    if constexpr (kProfile) {
        __syncthreads();
        maybe_record_profile<kProfile>(profile, kProfileKernelDone);
    }
}

