#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <cuda_fp16.h>

namespace {

constexpr int kThreads = 32;
constexpr int kRow2Threads = 64;
constexpr int kRow4Threads = 128;
constexpr int kRow8Threads = 256;
constexpr int kRowsPerBlock = 4;
constexpr int kRowsPerBlockAsync = 8;
constexpr int kSharedTileK = 256;
constexpr int kVecTileK = 256;
constexpr int kNosplitSpecializedK2048 = 2048;
constexpr int kNosplitSpecializedTileK6144 = 2048;
constexpr int kNosplitAsyncTileK6144 = 1024;
constexpr int kNosplitAsyncTileK6144Small = 512;
constexpr int kHalfPerVec128 = 8;
constexpr int kNosplitProfileSlots = 14;
constexpr int kFullVecsK6144 = 6144 / kHalfPerVec128;
constexpr int kPersistentRowsPerBatchK6144 = 2;
constexpr int kPersistentWarp8PerRowK6144 = kRow8Threads / 32;
constexpr int kWeightVecsPerTileK6144 = kNosplitSpecializedTileK6144 / kHalfPerVec128;
constexpr int kWeightTileVecsK6144 = kRowsPerBlock * kWeightVecsPerTileK6144;
constexpr int kAsyncVecsPerTileK6144 = kNosplitAsyncTileK6144 / kHalfPerVec128;
constexpr int kCpAsyncBytes = sizeof(uint4);
constexpr int kXWarpDetailPrologueSlots = 8;
constexpr int kXWarpDetailSlotsPerTile = 10;
constexpr int kXWarpDetailTileCount = 6;
constexpr int kXWarpDetailSlots =
    kXWarpDetailPrologueSlots + kXWarpDetailTileCount * kXWarpDetailSlotsPerTile + 2;

enum NosplitProfileSlot : int {
    kProfileKernelStart = 0,
    kProfileTile0LoadDone = 1,
    kProfileTile0ComputeDone = 2,
    kProfileTile1LoadDone = 3,
    kProfileTile1ComputeDone = 4,
    kProfileTile2LoadDone = 5,
    kProfileTile2ComputeDone = 6,
    kProfileTile3LoadDone = 7,
    kProfileTile3ComputeDone = 8,
    kProfileTile4LoadDone = 9,
    kProfileTile4ComputeDone = 10,
    kProfileTile5LoadDone = 11,
    kProfileTile5ComputeDone = 12,
    kProfileKernelDone = 13,
};

enum NosplitVariant : int64_t {
    kNosplitVariantAuto = 0,
    kNosplitVariantK2048Row2 = 1,
    kNosplitVariantK2048Row4 = 2,
    kNosplitVariantK6144Row4 = 3,
    kNosplitVariantPersistent = 4,
    kNosplitVariantPersistentHfma2 = 5,
    kNosplitVariantK6144WeightAsync = 6,
    kNosplitVariantK6144N8K1024Async = 7,
    kNosplitVariantK6144N8K1024XWAsync = 8,
    kNosplitVariantK6144N8K512S3XWAsync = 9,
    kNosplitVariantK6144PersistentXFullW2Async = 10,
    kNosplitVariantK6144PersistentXFullRow1Warp8Async = 11,
    kNosplitVariantK6144PersistentXFullN8K1024WAsync = 12,
    kNosplitVariantK6144N8K1024XWarpWAsync = 13,
    kNosplitVariantK6144N8K1024XWarpWAsyncSms2 = 14,
    kNosplitVariantK6144N8K1024XWarpWAsyncSms2Balanced = 15,
};

template <typename scalar_t>
__device__ inline float to_float(scalar_t v) {
    return static_cast<float>(v);
}

__device__ inline float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ inline unsigned int warp_sum_u32(unsigned int v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ inline void accumulate_hfma2_uint4(
    const uint4& x_vec,
    const uint4& w_vec,
    __half2& acc) {
    const __half2* x2 = reinterpret_cast<const __half2*>(&x_vec);
    const __half2* w2 = reinterpret_cast<const __half2*>(&w_vec);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        acc = __hfma2(x2[i], w2[i], acc);
    }
}

__device__ inline uint64_t read_globaltimer() {
    unsigned long long ts;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts));
    return static_cast<uint64_t>(ts);
}

__device__ inline unsigned int to_smem_ptr(const void* ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return static_cast<unsigned int>(__cvta_generic_to_shared(ptr));
#else
    (void)ptr;
    return 0;
#endif
}

__device__ inline void cp_async_cg_16(void* smem_dst, const void* gmem_src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned int smem_addr = to_smem_ptr(smem_dst);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2;\n"
        :
        : "r"(smem_addr), "l"(gmem_src), "n"(kCpAsyncBytes)
    );
#else
    *reinterpret_cast<uint4*>(smem_dst) = *reinterpret_cast<const uint4*>(gmem_src);
#endif
}

__device__ inline void cp_async_commit_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int kPendingGroups>
__device__ inline void cp_async_wait_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(kPendingGroups));
#endif
}

__device__ inline uint4 zero_uint4() {
    uint4 value;
    value.x = 0;
    value.y = 0;
    value.z = 0;
    value.w = 0;
    return value;
}

template <bool kUseCpAsync, int kVecsPerTileT, int kThreadsPerBlockT>
__device__ inline void prefetch_x_tile_vecs(
    uint4* __restrict__ x_sh_stage,
    const uint4* __restrict__ x_tile_vec) {
    for (int vec_idx = threadIdx.x; vec_idx < kVecsPerTileT; vec_idx += kThreadsPerBlockT) {
        if constexpr (kUseCpAsync) {
            cp_async_cg_16(x_sh_stage + vec_idx, x_tile_vec + vec_idx);
        } else {
            x_sh_stage[vec_idx] = x_tile_vec[vec_idx];
        }
    }
}

template <bool kUseCpAsync>
__device__ inline void prefetch_k6144_weight_tile(
    uint4* __restrict__ w_sh_stage,
    const __half* __restrict__ w_half,
    int64_t stride_wn,
    int row_base,
    int tile_start,
    int n) {
    for (int flat_idx = threadIdx.x; flat_idx < kWeightTileVecsK6144; flat_idx += kRow4Threads) {
        const int row_slot = flat_idx / kWeightVecsPerTileK6144;
        const int vec_idx = flat_idx - row_slot * kWeightVecsPerTileK6144;
        uint4* smem_dst = w_sh_stage + flat_idx;
        if (row_base + row_slot < n) {
            const uint4* gmem_src = reinterpret_cast<const uint4*>(
                w_half + (row_base + row_slot) * stride_wn + tile_start) + vec_idx;
            if constexpr (kUseCpAsync) {
                cp_async_cg_16(smem_dst, gmem_src);
            } else {
                *smem_dst = *gmem_src;
            }
        } else {
            *smem_dst = zero_uint4();
        }
    }
    if constexpr (kUseCpAsync) {
        cp_async_commit_group();
    }
}

template <bool kUseCpAsync, int kRowsPerBlockT, int kVecsPerTileT>
__device__ inline void prefetch_weight_tile_rows(
    uint4* __restrict__ w_sh_stage,
    const __half* __restrict__ w_half,
    int64_t stride_wn,
    int row_base,
    int tile_start,
    int n,
    int threads_per_block) {
    const int tile_vecs = kRowsPerBlockT * kVecsPerTileT;
    for (int flat_idx = threadIdx.x; flat_idx < tile_vecs; flat_idx += threads_per_block) {
        const int row_slot = flat_idx / kVecsPerTileT;
        const int vec_idx = flat_idx - row_slot * kVecsPerTileT;
        uint4* smem_dst = w_sh_stage + flat_idx;
        if (row_base + row_slot < n) {
            const uint4* gmem_src = reinterpret_cast<const uint4*>(
                w_half + (row_base + row_slot) * stride_wn + tile_start) + vec_idx;
            if constexpr (kUseCpAsync) {
                cp_async_cg_16(smem_dst, gmem_src);
            } else {
                *smem_dst = *gmem_src;
            }
        } else {
            *smem_dst = zero_uint4();
        }
    }
    if constexpr (kUseCpAsync) {
        cp_async_commit_group();
    }
}

template <bool kUseCpAsync, int kRowsPerBlockT, int kVecsPerTileT>
__device__ inline void prefetch_weight_tile_rows_no_commit(
    uint4* __restrict__ w_sh_stage,
    const __half* __restrict__ w_half,
    int64_t stride_wn,
    int row_base,
    int tile_start,
    int n,
    int threads_per_block) {
    const int tile_vecs = kRowsPerBlockT * kVecsPerTileT;
    for (int flat_idx = threadIdx.x; flat_idx < tile_vecs; flat_idx += threads_per_block) {
        const int row_slot = flat_idx / kVecsPerTileT;
        const int vec_idx = flat_idx - row_slot * kVecsPerTileT;
        uint4* smem_dst = w_sh_stage + flat_idx;
        if (row_base + row_slot < n) {
            const uint4* gmem_src = reinterpret_cast<const uint4*>(
                w_half + (row_base + row_slot) * stride_wn + tile_start) + vec_idx;
            if constexpr (kUseCpAsync) {
                cp_async_cg_16(smem_dst, gmem_src);
            } else {
                *smem_dst = *gmem_src;
            }
        } else {
            *smem_dst = zero_uint4();
        }
    }
}

template <bool kProfile>
__device__ inline void maybe_record_profile(uint64_t* profile, int slot) {
    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kNosplitProfileSlots + slot] = read_globaltimer();
        }
    }
}

__device__ inline void maybe_record_warp_detail(int64_t* trace, int slot) {
    if (trace != nullptr && (threadIdx.x & 31) == 0) {
        const int warp = threadIdx.x >> 5;
        const int64_t trace_idx =
            (static_cast<int64_t>(blockIdx.x) * kRowsPerBlockAsync + warp) * kXWarpDetailSlots + slot;
        trace[trace_idx] = static_cast<int64_t>(read_globaltimer());
    }
}

__device__ inline void maybe_record_warp_smid(int32_t* smid_out) {
    if (smid_out != nullptr && (threadIdx.x & 31) == 0) {
        const int warp = threadIdx.x >> 5;
        const int64_t smid_idx = static_cast<int64_t>(blockIdx.x) * kRowsPerBlockAsync + warp;
        uint32_t smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        smid_out[smid_idx] = static_cast<int32_t>(smid);
    }
}

__device__ inline void maybe_record_block_smid(int32_t* smid_out) {
    if (smid_out != nullptr && threadIdx.x == 0) {
        uint32_t smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        smid_out[blockIdx.x] = static_cast<int32_t>(smid);
    }
}

__device__ inline void maybe_record_block_iter_trace(
    int64_t* trace_out,
    int32_t* row_base_out,
    int max_iters,
    int iter_idx,
    int row_base,
    bool is_start) {
    if (threadIdx.x == 0 && trace_out != nullptr && iter_idx >= 0 && iter_idx < max_iters) {
        const int64_t iter_linear = static_cast<int64_t>(blockIdx.x) * max_iters + iter_idx;
        if (is_start) {
            trace_out[iter_linear * 2] = static_cast<int64_t>(read_globaltimer());
            if (row_base_out != nullptr) {
                row_base_out[iter_linear] = row_base;
            }
        } else {
            trace_out[iter_linear * 2 + 1] = static_cast<int64_t>(read_globaltimer());
        }
    }
}

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

__global__ void weight_stream_read_half_fwd_kernel(
    const at::Half* __restrict__ weight,
    int32_t* __restrict__ out,
    int64_t num_vecs) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int64_t global_tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const uint4* weight_vec = reinterpret_cast<const uint4*>(weight);

    unsigned int sum = 0;
    for (int64_t vec_idx = global_tid; vec_idx < num_vecs; vec_idx += stride) {
        const uint4 value = weight_vec[vec_idx];
        sum += value.x + value.y + value.z + value.w;
    }
    sum = warp_sum_u32(sum);
    if (lane == 0) {
        out[blockIdx.x * warps_per_block + warp] = static_cast<int32_t>(sum);
    }
}

__global__ void nosplit_weight_only_k6144_row4_fwd_kernel(
    const at::Half* __restrict__ weight,
    int32_t* __restrict__ out,
    int64_t stride_wn,
    int n) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * kRowsPerBlock;
    const int row = row_base + warp;
    const __half* w_half = reinterpret_cast<const __half*>(weight);
    constexpr int kVecsPerTile = kNosplitSpecializedTileK6144 / kHalfPerVec128;

    unsigned int sum = 0;
    if (row < n) {
        #pragma unroll
        for (int tile_idx = 0; tile_idx < 3; ++tile_idx) {
            const int tile_start = tile_idx * kNosplitSpecializedTileK6144;
            const uint4* w_row_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn + tile_start);
            #pragma unroll
            for (int vec_idx = lane; vec_idx < kVecsPerTile; vec_idx += 64) {
                const uint4 w_vec0 = w_row_vec[vec_idx];
                const uint4 w_vec1 = w_row_vec[vec_idx + 32];
                sum += w_vec0.x + w_vec0.y + w_vec0.z + w_vec0.w;
                sum += w_vec1.x + w_vec1.y + w_vec1.z + w_vec1.w;
            }
        }
        sum = warp_sum_u32(sum);
        if (lane == 0) {
            out[row] = static_cast<int32_t>(sum);
        }
    }
}

__global__ void nosplit_weight_only_k2048_row4_fwd_kernel(
    const at::Half* __restrict__ weight,
    int32_t* __restrict__ out,
    int64_t stride_wn,
    int n) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * kRowsPerBlock;
    const int row = row_base + warp;
    const __half* w_half = reinterpret_cast<const __half*>(weight);
    constexpr int kVecsK2048 = kNosplitSpecializedK2048 / kHalfPerVec128;

    unsigned int sum = 0;
    if (row < n) {
        const uint4* w_row_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn);
        #pragma unroll
        for (int vec_idx = lane; vec_idx < kVecsK2048; vec_idx += 64) {
            const uint4 w_vec0 = w_row_vec[vec_idx];
            const uint4 w_vec1 = w_row_vec[vec_idx + 32];
            sum += w_vec0.x + w_vec0.y + w_vec0.z + w_vec0.w;
            sum += w_vec1.x + w_vec1.y + w_vec1.z + w_vec1.w;
        }
        sum = warp_sum_u32(sum);
        if (lane == 0) {
            out[row] = static_cast<int32_t>(sum);
        }
    }
}

__global__ void nosplit_weight_only_k2048_row2_fwd_kernel(
    const at::Half* __restrict__ weight,
    int32_t* __restrict__ out,
    int64_t stride_wn,
    int n) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * 2;
    const int row = row_base + warp;
    const __half* w_half = reinterpret_cast<const __half*>(weight);
    constexpr int kVecsK2048 = kNosplitSpecializedK2048 / kHalfPerVec128;

    unsigned int sum = 0;
    if (row < n) {
        const uint4* w_row_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn);
        #pragma unroll
        for (int vec_idx = lane; vec_idx < kVecsK2048; vec_idx += 64) {
            const uint4 w_vec0 = w_row_vec[vec_idx];
            const uint4 w_vec1 = w_row_vec[vec_idx + 32];
            sum += w_vec0.x + w_vec0.y + w_vec0.z + w_vec0.w;
            sum += w_vec1.x + w_vec1.y + w_vec1.z + w_vec1.w;
        }
        sum = warp_sum_u32(sum);
        if (lane == 0) {
            out[row] = static_cast<int32_t>(sum);
        }
    }
}

template <bool kProfile>
__global__ void nosplit_matvec_k6144_weight_async_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    uint64_t* __restrict__ profile,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __half x_sh[kNosplitSpecializedTileK6144];
    __shared__ __align__(16) uint4 w_sh[2][kWeightTileVecsK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * kRowsPerBlock;
    const int row = row_base + warp;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);
    const uint4* x_sh_vec = reinterpret_cast<const uint4*>(x_sh);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    float sum = 0.0f;
    maybe_record_profile<kProfile>(profile, kProfileKernelStart);

    prefetch_k6144_weight_tile<kUseCpAsync>(&w_sh[0][0], w_half, stride_wn, row_base, 0, n);
    if constexpr (kUseCpAsync) {
        cp_async_wait_group<0>();
    }
    __syncthreads();

    #pragma unroll
    for (int tile_idx = 0; tile_idx < 3; ++tile_idx) {
        const int stage = tile_idx & 1;
        const int tile_start = tile_idx * kNosplitSpecializedTileK6144;
        const uint4* x_tile_vec = reinterpret_cast<const uint4*>(x_half + tile_start);

        if (threadIdx.x < kWeightVecsPerTileK6144 / 2) {
            reinterpret_cast<uint4*>(x_sh)[threadIdx.x] = x_tile_vec[threadIdx.x];
            reinterpret_cast<uint4*>(x_sh)[threadIdx.x + kWeightVecsPerTileK6144 / 2] =
                x_tile_vec[threadIdx.x + kWeightVecsPerTileK6144 / 2];
        }
        __syncthreads();

        if (tile_idx + 1 < 3) {
            const int next_tile_start = tile_start + kNosplitSpecializedTileK6144;
            prefetch_k6144_weight_tile<kUseCpAsync>(
                &w_sh[stage ^ 1][0],
                w_half,
                stride_wn,
                row_base,
                next_tile_start,
                n);
        }

        if constexpr (kProfile) {
            maybe_record_profile<kProfile>(profile, kProfileTile0LoadDone + tile_idx * 2);
        }

        if (row < n) {
            const uint4* w_sh_row_vec = &w_sh[stage][warp * kWeightVecsPerTileK6144];
            __half2 h2_acc0 = __float2half2_rn(0.0f);
            __half2 h2_acc1 = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane; vec_idx < kWeightVecsPerTileK6144; vec_idx += 64) {
                const uint4 x_vec0 = x_sh_vec[vec_idx];
                const uint4 w_vec0 = w_sh_row_vec[vec_idx];
                const uint4 x_vec1 = x_sh_vec[vec_idx + 32];
                const uint4 w_vec1 = w_sh_row_vec[vec_idx + 32];
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
        if constexpr (kUseCpAsync) {
            if (tile_idx + 1 < 3) {
                cp_async_wait_group<0>();
            }
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
    if constexpr (kProfile) {
        __syncthreads();
        maybe_record_profile<kProfile>(profile, kProfileKernelDone);
    }
}

template <bool kProfile>
__global__ void nosplit_matvec_k6144_n8k1024_async_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    uint64_t* __restrict__ profile,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __align__(16) uint4 x_sh[2][kAsyncVecsPerTileK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * kRowsPerBlockAsync;
    const int row = row_base + warp;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    maybe_record_profile<kProfile>(profile, kProfileKernelStart);
    for (int vec_idx = threadIdx.x; vec_idx < kAsyncVecsPerTileK6144; vec_idx += kRow8Threads) {
        if constexpr (kUseCpAsync) {
            cp_async_cg_16(&x_sh[0][vec_idx], reinterpret_cast<const uint4*>(x_half) + vec_idx);
        } else {
            x_sh[0][vec_idx] = reinterpret_cast<const uint4*>(x_half)[vec_idx];
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
            const uint4* x_next_vec = reinterpret_cast<const uint4*>(x_half + next_tile_start);
            for (int vec_idx = threadIdx.x; vec_idx < kAsyncVecsPerTileK6144; vec_idx += kRow8Threads) {
                if constexpr (kUseCpAsync) {
                    cp_async_cg_16(&x_sh[stage ^ 1][vec_idx], x_next_vec + vec_idx);
                } else {
                    x_sh[stage ^ 1][vec_idx] = x_next_vec[vec_idx];
                }
            }
            if constexpr (kUseCpAsync) {
                cp_async_commit_group();
            }
        }

        if constexpr (kProfile) {
            maybe_record_profile<kProfile>(profile, kProfileTile0LoadDone + tile_idx * 2);
        }

        if (row < n) {
            const uint4* x_sh_stage_vec = x_sh[stage];
            const uint4* w_row_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn + tile_start);
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

        if constexpr (kProfile) {
            maybe_record_profile<kProfile>(profile, kProfileTile0ComputeDone + tile_idx * 2);
        }
        if constexpr (kUseCpAsync) {
            if (tile_idx + 1 < 6) {
                cp_async_wait_group<0>();
            }
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
    if constexpr (kProfile) {
        __syncthreads();
        maybe_record_profile<kProfile>(profile, kProfileKernelDone);
    }
}

template <bool kProfile>
__global__ void nosplit_matvec_k6144_n8k1024_xw_async_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    uint64_t* __restrict__ profile,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __align__(16) uint4 x_sh[2][kAsyncVecsPerTileK6144];
    __shared__ __align__(16) uint4 w_sh[2][kRowsPerBlockAsync * kAsyncVecsPerTileK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * kRowsPerBlockAsync;
    const int row = row_base + warp;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    maybe_record_profile<kProfile>(profile, kProfileKernelStart);
    for (int vec_idx = threadIdx.x; vec_idx < kAsyncVecsPerTileK6144; vec_idx += kRow8Threads) {
        if constexpr (kUseCpAsync) {
            cp_async_cg_16(&x_sh[0][vec_idx], reinterpret_cast<const uint4*>(x_half) + vec_idx);
        } else {
            x_sh[0][vec_idx] = reinterpret_cast<const uint4*>(x_half)[vec_idx];
        }
    }
    if constexpr (kUseCpAsync) {
        cp_async_commit_group();
    }
    prefetch_weight_tile_rows<kUseCpAsync, kRowsPerBlockAsync, kAsyncVecsPerTileK6144>(
        &w_sh[0][0],
        w_half,
        stride_wn,
        row_base,
        0,
        n,
        kRow8Threads);
    if constexpr (kUseCpAsync) {
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
            const uint4* x_next_vec = reinterpret_cast<const uint4*>(x_half + next_tile_start);
            for (int vec_idx = threadIdx.x; vec_idx < kAsyncVecsPerTileK6144; vec_idx += kRow8Threads) {
                if constexpr (kUseCpAsync) {
                    cp_async_cg_16(&x_sh[stage ^ 1][vec_idx], x_next_vec + vec_idx);
                } else {
                    x_sh[stage ^ 1][vec_idx] = x_next_vec[vec_idx];
                }
            }
            if constexpr (kUseCpAsync) {
                cp_async_commit_group();
            }
            prefetch_weight_tile_rows<kUseCpAsync, kRowsPerBlockAsync, kAsyncVecsPerTileK6144>(
                &w_sh[stage ^ 1][0],
                w_half,
                stride_wn,
                row_base,
                next_tile_start,
                n,
                kRow8Threads);
        }

        if constexpr (kProfile) {
            maybe_record_profile<kProfile>(profile, kProfileTile0LoadDone + tile_idx * 2);
        }

        if (row < n) {
            const uint4* x_sh_stage_vec = x_sh[stage];
            const uint4* w_sh_row_vec = &w_sh[stage][warp * kAsyncVecsPerTileK6144];
            __half2 h2_acc0 = __float2half2_rn(0.0f);
            __half2 h2_acc1 = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 64) {
                const uint4 x_vec0 = x_sh_stage_vec[vec_idx];
                const uint4 w_vec0 = w_sh_row_vec[vec_idx];
                const uint4 x_vec1 = x_sh_stage_vec[vec_idx + 32];
                const uint4 w_vec1 = w_sh_row_vec[vec_idx + 32];
                accumulate_hfma2_uint4(x_vec0, w_vec0, h2_acc0);
                accumulate_hfma2_uint4(x_vec1, w_vec1, h2_acc1);
            }

            const float2 f2_0 = __half22float2(h2_acc0);
            const float2 f2_1 = __half22float2(h2_acc1);
            sum += f2_0.x + f2_0.y + f2_1.x + f2_1.y;
        }

        if constexpr (kProfile) {
            maybe_record_profile<kProfile>(profile, kProfileTile0ComputeDone + tile_idx * 2);
        }
        if constexpr (kUseCpAsync) {
            if (tile_idx + 1 < 6) {
                cp_async_wait_group<0>();
            }
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
    if constexpr (kProfile) {
        __syncthreads();
        maybe_record_profile<kProfile>(profile, kProfileKernelDone);
    }
}

template <bool kProfile>
__global__ void nosplit_matvec_k6144_n8k1024_xwarp_w_async_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    uint64_t* __restrict__ profile,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __align__(16) uint4 x_sh[2][kAsyncVecsPerTileK6144];
    __shared__ __align__(16) uint4 w_sh[2][kRowsPerBlockAsync][kAsyncVecsPerTileK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * kRowsPerBlockAsync;
    const int row = row_base + warp;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    maybe_record_profile<kProfile>(profile, kProfileKernelStart);

    prefetch_x_tile_vecs<kUseCpAsync, kAsyncVecsPerTileK6144, kRow8Threads>(
        &x_sh[0][0],
        reinterpret_cast<const uint4*>(x_half));
    if (row < n) {
        const uint4* w_row_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn);
        for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
            if constexpr (kUseCpAsync) {
                cp_async_cg_16(&w_sh[0][warp][vec_idx], w_row_vec + vec_idx);
            } else {
                w_sh[0][warp][vec_idx] = w_row_vec[vec_idx];
            }
        }
    } else {
        for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
            w_sh[0][warp][vec_idx] = zero_uint4();
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
            const uint4* x_next_vec = reinterpret_cast<const uint4*>(x_half + next_tile_start);
            prefetch_x_tile_vecs<kUseCpAsync, kAsyncVecsPerTileK6144, kRow8Threads>(
                &x_sh[stage ^ 1][0],
                x_next_vec);

            if (row < n) {
                const uint4* w_row_next_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn + next_tile_start);
                for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                    if constexpr (kUseCpAsync) {
                        cp_async_cg_16(&w_sh[stage ^ 1][warp][vec_idx], w_row_next_vec + vec_idx);
                    } else {
                        w_sh[stage ^ 1][warp][vec_idx] = w_row_next_vec[vec_idx];
                    }
                }
            } else {
                for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                    w_sh[stage ^ 1][warp][vec_idx] = zero_uint4();
                }
            }
            if constexpr (kUseCpAsync) {
                cp_async_commit_group();
            }
        }

        if constexpr (kProfile) {
            maybe_record_profile<kProfile>(profile, kProfileTile0LoadDone + tile_idx * 2);
        }

        if (row < n) {
            const uint4* x_sh_stage_vec = x_sh[stage];
            const uint4* w_sh_row_vec = &w_sh[stage][warp][0];
            __half2 h2_acc0 = __float2half2_rn(0.0f);
            __half2 h2_acc1 = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 64) {
                const uint4 x_vec0 = x_sh_stage_vec[vec_idx];
                const uint4 w_vec0 = w_sh_row_vec[vec_idx];
                const uint4 x_vec1 = x_sh_stage_vec[vec_idx + 32];
                const uint4 w_vec1 = w_sh_row_vec[vec_idx + 32];
                accumulate_hfma2_uint4(x_vec0, w_vec0, h2_acc0);
                accumulate_hfma2_uint4(x_vec1, w_vec1, h2_acc1);
            }

            const float2 f2_0 = __half22float2(h2_acc0);
            const float2 f2_1 = __half22float2(h2_acc1);
            sum += f2_0.x + f2_0.y + f2_1.x + f2_1.y;
        }

        if constexpr (kProfile) {
            maybe_record_profile<kProfile>(profile, kProfileTile0ComputeDone + tile_idx * 2);
        }
        if constexpr (kUseCpAsync) {
            if (tile_idx + 1 < 6) {
                cp_async_wait_group<0>();
            }
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
    if constexpr (kProfile) {
        __syncthreads();
        maybe_record_profile<kProfile>(profile, kProfileKernelDone);
    }
}

__global__ void nosplit_weight_only_k6144_n8k1024_xwarp_w_async_fwd_kernel(
    const at::Half* __restrict__ weight,
    int32_t* __restrict__ out,
    int64_t stride_wn,
    int n) {
    __shared__ __align__(16) uint4 w_sh[2][kRowsPerBlockAsync][kAsyncVecsPerTileK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * kRowsPerBlockAsync;
    const int row = row_base + warp;
    const __half* w_half = reinterpret_cast<const __half*>(weight);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    if (row < n) {
        const uint4* w_row_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn);
        for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
            if constexpr (kUseCpAsync) {
                cp_async_cg_16(&w_sh[0][warp][vec_idx], w_row_vec + vec_idx);
            } else {
                w_sh[0][warp][vec_idx] = w_row_vec[vec_idx];
            }
        }
    } else {
        for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
            w_sh[0][warp][vec_idx] = zero_uint4();
        }
    }
    if constexpr (kUseCpAsync) {
        cp_async_commit_group();
        cp_async_wait_group<0>();
    }
    __syncthreads();

    unsigned int sum = 0;
    #pragma unroll
    for (int tile_idx = 0; tile_idx < 6; ++tile_idx) {
        const int stage = tile_idx & 1;
        const int tile_start = tile_idx * kNosplitAsyncTileK6144;

        if (tile_idx + 1 < 6) {
            const int next_tile_start = tile_start + kNosplitAsyncTileK6144;
            if (row < n) {
                const uint4* w_row_next_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn + next_tile_start);
                for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                    if constexpr (kUseCpAsync) {
                        cp_async_cg_16(&w_sh[stage ^ 1][warp][vec_idx], w_row_next_vec + vec_idx);
                    } else {
                        w_sh[stage ^ 1][warp][vec_idx] = w_row_next_vec[vec_idx];
                    }
                }
            } else {
                for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                    w_sh[stage ^ 1][warp][vec_idx] = zero_uint4();
                }
            }
            if constexpr (kUseCpAsync) {
                cp_async_commit_group();
            }
        }

        if (row < n) {
            const uint4* w_sh_row_vec = &w_sh[stage][warp][0];
            #pragma unroll
            for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 64) {
                const uint4 w_vec0 = w_sh_row_vec[vec_idx];
                const uint4 w_vec1 = w_sh_row_vec[vec_idx + 32];
                sum += w_vec0.x + w_vec0.y + w_vec0.z + w_vec0.w;
                sum += w_vec1.x + w_vec1.y + w_vec1.z + w_vec1.w;
            }
        }

        if constexpr (kUseCpAsync) {
            if (tile_idx + 1 < 6) {
                cp_async_wait_group<0>();
            }
        }
        __syncthreads();
    }

    if (row < n) {
        sum = warp_sum_u32(sum);
        if (lane == 0) {
            out[row] = static_cast<int32_t>(sum);
        }
    }
}

template <bool kProfile>
__global__ void nosplit_matvec_k6144_n8k1024_xwarp_w_async_sms2_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    uint64_t* __restrict__ profile,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __align__(16) uint4 x_sh[2][kAsyncVecsPerTileK6144];
    __shared__ __align__(16) uint4 w_sh[2][kRowsPerBlockAsync][kAsyncVecsPerTileK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    maybe_record_profile<kProfile>(profile, kProfileKernelStart);

    const int row_stride = gridDim.x * kRowsPerBlockAsync;
    for (int row_base = blockIdx.x * kRowsPerBlockAsync; row_base < n; row_base += row_stride) {
        const int row = row_base + warp;

        prefetch_x_tile_vecs<kUseCpAsync, kAsyncVecsPerTileK6144, kRow8Threads>(
            &x_sh[0][0],
            reinterpret_cast<const uint4*>(x_half));
        if (row < n) {
            const uint4* w_row_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn);
            for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                if constexpr (kUseCpAsync) {
                    cp_async_cg_16(&w_sh[0][warp][vec_idx], w_row_vec + vec_idx);
                } else {
                    w_sh[0][warp][vec_idx] = w_row_vec[vec_idx];
                }
            }
        } else {
            for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                w_sh[0][warp][vec_idx] = zero_uint4();
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
                const uint4* x_next_vec = reinterpret_cast<const uint4*>(x_half + next_tile_start);
                prefetch_x_tile_vecs<kUseCpAsync, kAsyncVecsPerTileK6144, kRow8Threads>(
                    &x_sh[stage ^ 1][0],
                    x_next_vec);

                if (row < n) {
                    const uint4* w_row_next_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn + next_tile_start);
                    for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                        if constexpr (kUseCpAsync) {
                            cp_async_cg_16(&w_sh[stage ^ 1][warp][vec_idx], w_row_next_vec + vec_idx);
                        } else {
                            w_sh[stage ^ 1][warp][vec_idx] = w_row_next_vec[vec_idx];
                        }
                    }
                } else {
                    for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                        w_sh[stage ^ 1][warp][vec_idx] = zero_uint4();
                    }
                }
                if constexpr (kUseCpAsync) {
                    cp_async_commit_group();
                }
            }

            if constexpr (kProfile) {
                maybe_record_profile<kProfile>(profile, kProfileTile0LoadDone + tile_idx * 2);
            }

            if (row < n) {
                const uint4* x_sh_stage_vec = x_sh[stage];
                const uint4* w_sh_row_vec = &w_sh[stage][warp][0];
                __half2 h2_acc0 = __float2half2_rn(0.0f);
                __half2 h2_acc1 = __float2half2_rn(0.0f);

                #pragma unroll
                for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 64) {
                    const uint4 x_vec0 = x_sh_stage_vec[vec_idx];
                    const uint4 w_vec0 = w_sh_row_vec[vec_idx];
                    const uint4 x_vec1 = x_sh_stage_vec[vec_idx + 32];
                    const uint4 w_vec1 = w_sh_row_vec[vec_idx + 32];
                    accumulate_hfma2_uint4(x_vec0, w_vec0, h2_acc0);
                    accumulate_hfma2_uint4(x_vec1, w_vec1, h2_acc1);
                }

                const float2 f2_0 = __half22float2(h2_acc0);
                const float2 f2_1 = __half22float2(h2_acc1);
                sum += f2_0.x + f2_0.y + f2_1.x + f2_1.y;
            }

            if constexpr (kProfile) {
                maybe_record_profile<kProfile>(profile, kProfileTile0ComputeDone + tile_idx * 2);
            }
            if constexpr (kUseCpAsync) {
                if (tile_idx + 1 < 6) {
                    cp_async_wait_group<0>();
                }
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
        __syncthreads();
    }

    if constexpr (kProfile) {
        maybe_record_profile<kProfile>(profile, kProfileKernelDone);
    }
}

template <bool kUseCpAsync>
__device__ inline void run_k6144_xwarp_async_row_batch(
    const __half* x_half,
    const __half* w_half,
    const __half* bias_half,
    __half* out_half,
    uint4 x_sh[2][kAsyncVecsPerTileK6144],
    uint4 w_sh[2][kRowsPerBlockAsync][kAsyncVecsPerTileK6144],
    int warp,
    int lane,
    int row_base,
    int active_rows,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    const int row = row_base + warp;
    const bool row_valid = (warp < active_rows) && (row < n);

    prefetch_x_tile_vecs<kUseCpAsync, kAsyncVecsPerTileK6144, kRow8Threads>(
        &x_sh[0][0],
        reinterpret_cast<const uint4*>(x_half));
    if (row_valid) {
        const uint4* w_row_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn);
        for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
            if constexpr (kUseCpAsync) {
                cp_async_cg_16(&w_sh[0][warp][vec_idx], w_row_vec + vec_idx);
            } else {
                w_sh[0][warp][vec_idx] = w_row_vec[vec_idx];
            }
        }
    } else {
        for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
            w_sh[0][warp][vec_idx] = zero_uint4();
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
            const uint4* x_next_vec = reinterpret_cast<const uint4*>(x_half + next_tile_start);
            prefetch_x_tile_vecs<kUseCpAsync, kAsyncVecsPerTileK6144, kRow8Threads>(
                &x_sh[stage ^ 1][0],
                x_next_vec);

            if (row_valid) {
                const uint4* w_row_next_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn + next_tile_start);
                for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                    if constexpr (kUseCpAsync) {
                        cp_async_cg_16(&w_sh[stage ^ 1][warp][vec_idx], w_row_next_vec + vec_idx);
                    } else {
                        w_sh[stage ^ 1][warp][vec_idx] = w_row_next_vec[vec_idx];
                    }
                }
            } else {
                for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                    w_sh[stage ^ 1][warp][vec_idx] = zero_uint4();
                }
            }
            if constexpr (kUseCpAsync) {
                cp_async_commit_group();
            }
        }

        if (row_valid) {
            const uint4* x_sh_stage_vec = x_sh[stage];
            const uint4* w_sh_row_vec = &w_sh[stage][warp][0];
            __half2 h2_acc0 = __float2half2_rn(0.0f);
            __half2 h2_acc1 = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 64) {
                const uint4 x_vec0 = x_sh_stage_vec[vec_idx];
                const uint4 w_vec0 = w_sh_row_vec[vec_idx];
                const uint4 x_vec1 = x_sh_stage_vec[vec_idx + 32];
                const uint4 w_vec1 = w_sh_row_vec[vec_idx + 32];
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

    if (row_valid) {
        sum = warp_sum(sum);
        if (lane == 0) {
            float result = sum;
            if (has_bias) {
                result += __half2float(bias_half[row]);
            }
            out_half[row] = __float2half(result);
        }
    }
    __syncthreads();
}

__global__ void nosplit_matvec_k6144_n8k1024_xwarp_w_async_sms2_balanced_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __align__(16) uint4 x_sh[2][kAsyncVecsPerTileK6144];
    __shared__ __align__(16) uint4 w_sh[2][kRowsPerBlockAsync][kAsyncVecsPerTileK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);
    const __half* bias_half = reinterpret_cast<const __half*>(bias);
    __half* out_half = reinterpret_cast<__half*>(out);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    const int block_count = gridDim.x;
    const int base_rows = n / block_count;
    const int extra_rows = n % block_count;
    const int row_begin = blockIdx.x * base_rows + min(static_cast<int>(blockIdx.x), extra_rows);
    const int row_count = base_rows + (static_cast<int>(blockIdx.x) < extra_rows ? 1 : 0);

    const int first_rows = min(row_count, kRowsPerBlockAsync);
    const int second_rows = max(0, row_count - kRowsPerBlockAsync);

    if (first_rows > 0) {
        run_k6144_xwarp_async_row_batch<kUseCpAsync>(
            x_half, w_half, bias_half, out_half, x_sh, w_sh, warp, lane,
            row_begin, first_rows, stride_wn, n, has_bias);
    }
    if (second_rows > 0) {
        run_k6144_xwarp_async_row_batch<kUseCpAsync>(
            x_half, w_half, bias_half, out_half, x_sh, w_sh, warp, lane,
            row_begin + kRowsPerBlockAsync, second_rows, stride_wn, n, has_bias);
    }
}

__global__ void nosplit_matvec_k6144_n8k1024_xwarp_w_async_sms2_balanced_cta_trace_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    int64_t* __restrict__ trace,
    int32_t* __restrict__ smid_out,
    int32_t* __restrict__ row_base_out,
    int max_iters,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __align__(16) uint4 x_sh[2][kAsyncVecsPerTileK6144];
    __shared__ __align__(16) uint4 w_sh[2][kRowsPerBlockAsync][kAsyncVecsPerTileK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);
    const __half* bias_half = reinterpret_cast<const __half*>(bias);
    __half* out_half = reinterpret_cast<__half*>(out);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    maybe_record_block_smid(smid_out);

    const int block_count = gridDim.x;
    const int base_rows = n / block_count;
    const int extra_rows = n % block_count;
    const int row_begin = blockIdx.x * base_rows + min(static_cast<int>(blockIdx.x), extra_rows);
    const int row_count = base_rows + (static_cast<int>(blockIdx.x) < extra_rows ? 1 : 0);

    const int first_rows = min(row_count, kRowsPerBlockAsync);
    const int second_rows = max(0, row_count - kRowsPerBlockAsync);

    if (first_rows > 0) {
        maybe_record_block_iter_trace(trace, row_base_out, max_iters, 0, row_begin, true);
        run_k6144_xwarp_async_row_batch<kUseCpAsync>(
            x_half, w_half, bias_half, out_half, x_sh, w_sh, warp, lane,
            row_begin, first_rows, stride_wn, n, has_bias);
        maybe_record_block_iter_trace(trace, row_base_out, max_iters, 0, row_begin, false);
    }
    if (second_rows > 0) {
        maybe_record_block_iter_trace(trace, row_base_out, max_iters, 1, row_begin + kRowsPerBlockAsync, true);
        run_k6144_xwarp_async_row_batch<kUseCpAsync>(
            x_half, w_half, bias_half, out_half, x_sh, w_sh, warp, lane,
            row_begin + kRowsPerBlockAsync, second_rows, stride_wn, n, has_bias);
        maybe_record_block_iter_trace(trace, row_base_out, max_iters, 1, row_begin + kRowsPerBlockAsync, false);
    }
}

__global__ void nosplit_matvec_k6144_n8k1024_xwarp_w_async_sms2_cta_trace_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    int64_t* __restrict__ trace,
    int32_t* __restrict__ smid_out,
    int32_t* __restrict__ row_base_out,
    int max_iters,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __align__(16) uint4 x_sh[2][kAsyncVecsPerTileK6144];
    __shared__ __align__(16) uint4 w_sh[2][kRowsPerBlockAsync][kAsyncVecsPerTileK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    maybe_record_block_smid(smid_out);

    const int row_stride = gridDim.x * kRowsPerBlockAsync;
    int iter_idx = 0;
    for (int row_base = blockIdx.x * kRowsPerBlockAsync; row_base < n; row_base += row_stride, ++iter_idx) {
        const int row = row_base + warp;
        maybe_record_block_iter_trace(trace, row_base_out, max_iters, iter_idx, row_base, true);

        prefetch_x_tile_vecs<kUseCpAsync, kAsyncVecsPerTileK6144, kRow8Threads>(
            &x_sh[0][0],
            reinterpret_cast<const uint4*>(x_half));
        if (row < n) {
            const uint4* w_row_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn);
            for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                if constexpr (kUseCpAsync) {
                    cp_async_cg_16(&w_sh[0][warp][vec_idx], w_row_vec + vec_idx);
                } else {
                    w_sh[0][warp][vec_idx] = w_row_vec[vec_idx];
                }
            }
        } else {
            for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                w_sh[0][warp][vec_idx] = zero_uint4();
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
                const uint4* x_next_vec = reinterpret_cast<const uint4*>(x_half + next_tile_start);
                prefetch_x_tile_vecs<kUseCpAsync, kAsyncVecsPerTileK6144, kRow8Threads>(
                    &x_sh[stage ^ 1][0],
                    x_next_vec);

                if (row < n) {
                    const uint4* w_row_next_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn + next_tile_start);
                    for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                        if constexpr (kUseCpAsync) {
                            cp_async_cg_16(&w_sh[stage ^ 1][warp][vec_idx], w_row_next_vec + vec_idx);
                        } else {
                            w_sh[stage ^ 1][warp][vec_idx] = w_row_next_vec[vec_idx];
                        }
                    }
                } else {
                    for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                        w_sh[stage ^ 1][warp][vec_idx] = zero_uint4();
                    }
                }
                if constexpr (kUseCpAsync) {
                    cp_async_commit_group();
                }
            }

            if (row < n) {
                const uint4* x_sh_stage_vec = x_sh[stage];
                const uint4* w_sh_row_vec = &w_sh[stage][warp][0];
                __half2 h2_acc0 = __float2half2_rn(0.0f);
                __half2 h2_acc1 = __float2half2_rn(0.0f);

                #pragma unroll
                for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 64) {
                    const uint4 x_vec0 = x_sh_stage_vec[vec_idx];
                    const uint4 w_vec0 = w_sh_row_vec[vec_idx];
                    const uint4 x_vec1 = x_sh_stage_vec[vec_idx + 32];
                    const uint4 w_vec1 = w_sh_row_vec[vec_idx + 32];
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
        __syncthreads();
        maybe_record_block_iter_trace(trace, row_base_out, max_iters, iter_idx, row_base, false);
    }
}

__global__ void nosplit_matvec_k6144_n8k1024_xwarp_w_async_trace_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    int64_t* __restrict__ trace,
    int32_t* __restrict__ smid_out,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __align__(16) uint4 x_sh[2][kAsyncVecsPerTileK6144];
    __shared__ __align__(16) uint4 w_sh[2][kRowsPerBlockAsync][kAsyncVecsPerTileK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * kRowsPerBlockAsync;
    const int row = row_base + warp;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    maybe_record_warp_smid(smid_out);
    maybe_record_warp_detail(trace, 0);
    prefetch_x_tile_vecs<kUseCpAsync, kAsyncVecsPerTileK6144, kRow8Threads>(
        &x_sh[0][0],
        reinterpret_cast<const uint4*>(x_half));
    maybe_record_warp_detail(trace, 1);

    maybe_record_warp_detail(trace, 2);
    if (row < n) {
        const uint4* w_row_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn);
        for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
            if constexpr (kUseCpAsync) {
                cp_async_cg_16(&w_sh[0][warp][vec_idx], w_row_vec + vec_idx);
            } else {
                w_sh[0][warp][vec_idx] = w_row_vec[vec_idx];
            }
        }
    } else {
        for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
            w_sh[0][warp][vec_idx] = zero_uint4();
        }
    }
    maybe_record_warp_detail(trace, 3);

    maybe_record_warp_detail(trace, 4);
    if constexpr (kUseCpAsync) {
        cp_async_commit_group();
        cp_async_wait_group<0>();
    }
    maybe_record_warp_detail(trace, 5);

    maybe_record_warp_detail(trace, 6);
    __syncthreads();
    maybe_record_warp_detail(trace, 7);

    float sum = 0.0f;
    #pragma unroll
    for (int tile_idx = 0; tile_idx < 6; ++tile_idx) {
        const int stage = tile_idx & 1;
        const int tile_start = tile_idx * kNosplitAsyncTileK6144;
        const int slot_base = kXWarpDetailPrologueSlots + tile_idx * kXWarpDetailSlotsPerTile;

        if (tile_idx + 1 < 6) {
            const int next_tile_start = tile_start + kNosplitAsyncTileK6144;
            const uint4* x_next_vec = reinterpret_cast<const uint4*>(x_half + next_tile_start);

            maybe_record_warp_detail(trace, slot_base + 0);
            prefetch_x_tile_vecs<kUseCpAsync, kAsyncVecsPerTileK6144, kRow8Threads>(
                &x_sh[stage ^ 1][0],
                x_next_vec);
            maybe_record_warp_detail(trace, slot_base + 1);

            maybe_record_warp_detail(trace, slot_base + 2);
            if (row < n) {
                const uint4* w_row_next_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn + next_tile_start);
                for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                    if constexpr (kUseCpAsync) {
                        cp_async_cg_16(&w_sh[stage ^ 1][warp][vec_idx], w_row_next_vec + vec_idx);
                    } else {
                        w_sh[stage ^ 1][warp][vec_idx] = w_row_next_vec[vec_idx];
                    }
                }
            } else {
                for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                    w_sh[stage ^ 1][warp][vec_idx] = zero_uint4();
                }
            }
            maybe_record_warp_detail(trace, slot_base + 3);

            if constexpr (kUseCpAsync) {
                cp_async_commit_group();
            }
        }

        maybe_record_warp_detail(trace, slot_base + 4);
        if (row < n) {
            const uint4* x_sh_stage_vec = x_sh[stage];
            const uint4* w_sh_row_vec = &w_sh[stage][warp][0];
            __half2 h2_acc0 = __float2half2_rn(0.0f);
            __half2 h2_acc1 = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 64) {
                const uint4 x_vec0 = x_sh_stage_vec[vec_idx];
                const uint4 w_vec0 = w_sh_row_vec[vec_idx];
                const uint4 x_vec1 = x_sh_stage_vec[vec_idx + 32];
                const uint4 w_vec1 = w_sh_row_vec[vec_idx + 32];
                accumulate_hfma2_uint4(x_vec0, w_vec0, h2_acc0);
                accumulate_hfma2_uint4(x_vec1, w_vec1, h2_acc1);
            }

            const float2 f2_0 = __half22float2(h2_acc0);
            const float2 f2_1 = __half22float2(h2_acc1);
            sum += f2_0.x + f2_0.y + f2_1.x + f2_1.y;
        }
        maybe_record_warp_detail(trace, slot_base + 5);

        maybe_record_warp_detail(trace, slot_base + 6);
        if constexpr (kUseCpAsync) {
            if (tile_idx + 1 < 6) {
                cp_async_wait_group<0>();
            }
        }
        maybe_record_warp_detail(trace, slot_base + 7);

        maybe_record_warp_detail(trace, slot_base + 8);
        __syncthreads();
        maybe_record_warp_detail(trace, slot_base + 9);
    }

    const int epilogue_slot = kXWarpDetailPrologueSlots + kXWarpDetailTileCount * kXWarpDetailSlotsPerTile;
    maybe_record_warp_detail(trace, epilogue_slot + 0);
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
    maybe_record_warp_detail(trace, epilogue_slot + 1);
}

__global__ void nosplit_xwarp_wait_sync_calibration_kernel(
    int64_t* __restrict__ trace) {
    maybe_record_warp_detail(trace, 4);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    cp_async_wait_group<0>();
#endif
    maybe_record_warp_detail(trace, 5);

    maybe_record_warp_detail(trace, 6);
    __syncthreads();
    maybe_record_warp_detail(trace, 7);

    #pragma unroll
    for (int tile_idx = 0; tile_idx < 6; ++tile_idx) {
        const int slot_base = kXWarpDetailPrologueSlots + tile_idx * kXWarpDetailSlotsPerTile;
        maybe_record_warp_detail(trace, slot_base + 6);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        cp_async_wait_group<0>();
#endif
        maybe_record_warp_detail(trace, slot_base + 7);

        maybe_record_warp_detail(trace, slot_base + 8);
        __syncthreads();
        maybe_record_warp_detail(trace, slot_base + 9);
    }
}

template <int kTileK, int kStages, bool kProfile>
__global__ void nosplit_matvec_k6144_n8_xw_async_multi_stage_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    uint64_t* __restrict__ profile,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    static_assert(kStages >= 2, "multi-stage kernel requires at least 2 stages");
    constexpr int kTotalK = 6144;
    static_assert(kTotalK % kTileK == 0, "kTileK must evenly divide k=6144");

    constexpr int kVecsPerTile = kTileK / kHalfPerVec128;
    constexpr int kTileCount = kTotalK / kTileK;
    constexpr int kThreadsPerBlock = kRow8Threads;

    __shared__ __align__(16) uint4 x_sh[kStages][kVecsPerTile];
    __shared__ __align__(16) uint4 w_sh[kStages][kRowsPerBlockAsync * kVecsPerTile];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * kRowsPerBlockAsync;
    const int row = row_base + warp;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    maybe_record_profile<kProfile>(profile, kProfileKernelStart);

    #pragma unroll
    for (int prefetch_tile = 0; prefetch_tile < kStages; ++prefetch_tile) {
        if (prefetch_tile < kTileCount) {
            const int tile_start = prefetch_tile * kTileK;
            prefetch_x_tile_vecs<kUseCpAsync, kVecsPerTile, kThreadsPerBlock>(
                &x_sh[prefetch_tile][0],
                reinterpret_cast<const uint4*>(x_half + tile_start));
            prefetch_weight_tile_rows_no_commit<kUseCpAsync, kRowsPerBlockAsync, kVecsPerTile>(
                &w_sh[prefetch_tile][0],
                w_half,
                stride_wn,
                row_base,
                tile_start,
                n,
                kThreadsPerBlock);
            if constexpr (kUseCpAsync) {
                cp_async_commit_group();
            }
        }
    }
    if constexpr (kUseCpAsync) {
        cp_async_wait_group<kStages - 2>();
    }
    __syncthreads();

    float sum = 0.0f;
    #pragma unroll
    for (int tile_idx = 0; tile_idx < kTileCount; ++tile_idx) {
        const int stage = tile_idx % kStages;

        if constexpr (kProfile) {
            maybe_record_profile<kProfile>(profile, kProfileTile0LoadDone + (tile_idx >> 1) * 2);
        }

        if (row < n) {
            const uint4* x_sh_stage_vec = x_sh[stage];
            const uint4* w_sh_row_vec = &w_sh[stage][warp * kVecsPerTile];
            __half2 h2_acc0 = __float2half2_rn(0.0f);
            __half2 h2_acc1 = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane; vec_idx < kVecsPerTile; vec_idx += 64) {
                const uint4 x_vec0 = x_sh_stage_vec[vec_idx];
                const uint4 w_vec0 = w_sh_row_vec[vec_idx];
                const uint4 x_vec1 = x_sh_stage_vec[vec_idx + 32];
                const uint4 w_vec1 = w_sh_row_vec[vec_idx + 32];
                accumulate_hfma2_uint4(x_vec0, w_vec0, h2_acc0);
                accumulate_hfma2_uint4(x_vec1, w_vec1, h2_acc1);
            }

            const float2 f2_0 = __half22float2(h2_acc0);
            const float2 f2_1 = __half22float2(h2_acc1);
            sum += f2_0.x + f2_0.y + f2_1.x + f2_1.y;
        }

        if constexpr (kProfile) {
            maybe_record_profile<kProfile>(profile, kProfileTile0ComputeDone + (tile_idx >> 1) * 2);
        }

        // All warps must finish consuming the current stage before it is recycled.
        __syncthreads();

        const int next_tile = tile_idx + kStages;
        if (next_tile < kTileCount) {
            const int next_tile_start = next_tile * kTileK;
            prefetch_x_tile_vecs<kUseCpAsync, kVecsPerTile, kThreadsPerBlock>(
                &x_sh[stage][0],
                reinterpret_cast<const uint4*>(x_half + next_tile_start));
            prefetch_weight_tile_rows_no_commit<kUseCpAsync, kRowsPerBlockAsync, kVecsPerTile>(
                &w_sh[stage][0],
                w_half,
                stride_wn,
                row_base,
                next_tile_start,
                n,
                kThreadsPerBlock);
            if constexpr (kUseCpAsync) {
                cp_async_commit_group();
            }
        }

        if constexpr (kUseCpAsync) {
            if (tile_idx + 1 < kTileCount) {
                const int future_tiles = kTileCount - (tile_idx + 1);
                if constexpr (kStages == 3) {
                    if (future_tiles > 1) {
                        cp_async_wait_group<1>();
                    } else {
                        cp_async_wait_group<0>();
                    }
                } else {
                    cp_async_wait_group<kStages - 2>();
                }
            }
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
    if constexpr (kProfile) {
        __syncthreads();
        maybe_record_profile<kProfile>(profile, kProfileKernelDone);
    }
}

template <bool kProfile>
__global__ void nosplit_matvec_k6144_persistent_xfull_w2_async_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    uint64_t* __restrict__ profile,
    int64_t stride_wn,
    int n,
    bool has_bias) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
    __shared__ __align__(16) uint4 x_sh[kFullVecsK6144];
    __shared__ __align__(16) uint4 w_sh[2][kPersistentRowsPerBatchK6144 * kFullVecsK6144];
    __shared__ float warp_partials[kPersistentRowsPerBatchK6144][2];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_slot = warp >> 1;
    const int warp_in_row = warp & 1;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    maybe_record_profile<kProfile>(profile, kProfileKernelStart);

    prefetch_x_tile_vecs<kUseCpAsync, kFullVecsK6144, kRow4Threads>(
        &x_sh[0],
        reinterpret_cast<const uint4*>(x_half));
    if constexpr (kUseCpAsync) {
        cp_async_commit_group();
        cp_async_wait_group<0>();
    }
    __syncthreads();

    const int row_stride = gridDim.x * kPersistentRowsPerBatchK6144;
    int row_base = blockIdx.x * kPersistentRowsPerBatchK6144;
    if (row_base < n) {
        prefetch_weight_tile_rows_no_commit<kUseCpAsync, kPersistentRowsPerBatchK6144, kFullVecsK6144>(
            &w_sh[0][0],
            w_half,
            stride_wn,
            row_base,
            0,
            n,
            kRow4Threads);
        if constexpr (kUseCpAsync) {
            cp_async_commit_group();
            cp_async_wait_group<0>();
        }
        __syncthreads();
    }

    for (int iter = 0; row_base < n; row_base += row_stride, ++iter) {
        const int stage = iter & 1;
        const int row = row_base + row_slot;
        const int next_row_base = row_base + row_stride;

        if (next_row_base < n) {
            prefetch_weight_tile_rows_no_commit<kUseCpAsync, kPersistentRowsPerBatchK6144, kFullVecsK6144>(
                &w_sh[stage ^ 1][0],
                w_half,
                stride_wn,
                next_row_base,
                0,
                n,
                kRow4Threads);
            if constexpr (kUseCpAsync) {
                cp_async_commit_group();
            }
        }

        if constexpr (kProfile) {
            maybe_record_profile<kProfile>(profile, kProfileTile0LoadDone);
        }

        float sum = 0.0f;
        if (row < n) {
            const uint4* x_sh_vec = x_sh;
            const uint4* w_row_vec = &w_sh[stage][row_slot * kFullVecsK6144];
            __half2 h2_acc = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane + (warp_in_row << 5); vec_idx < kFullVecsK6144; vec_idx += 64) {
                const uint4 x_vec = x_sh_vec[vec_idx];
                const uint4 w_vec = w_row_vec[vec_idx];
                accumulate_hfma2_uint4(x_vec, w_vec, h2_acc);
            }

            const float2 f2 = __half22float2(h2_acc);
            sum = f2.x + f2.y;
        }

        if constexpr (kProfile) {
            maybe_record_profile<kProfile>(profile, kProfileTile0ComputeDone);
        }

        sum = warp_sum(sum);
        if (lane == 0 && row_slot < kPersistentRowsPerBatchK6144) {
            warp_partials[row_slot][warp_in_row] = sum;
        }
        __syncthreads();

        if (row < n && lane == 0 && warp_in_row == 0) {
            float result = warp_partials[row_slot][0] + warp_partials[row_slot][1];
            if (has_bias) {
                result += __half2float(reinterpret_cast<const __half*>(bias)[row]);
            }
            reinterpret_cast<__half*>(out)[row] = __float2half(result);
        }

        if constexpr (kUseCpAsync) {
            if (next_row_base < n) {
                cp_async_wait_group<0>();
            }
        }
        __syncthreads();
    }

    if constexpr (kProfile) {
        __syncthreads();
        maybe_record_profile<kProfile>(profile, kProfileKernelDone);
    }
#else
    (void)x;
    (void)weight;
    (void)bias;
    (void)out;
    (void)profile;
    (void)stride_wn;
    (void)n;
    (void)has_bias;
#endif
}

template <bool kProfile>
__global__ void nosplit_matvec_k6144_persistent_xfull_row1_warp8_async_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    uint64_t* __restrict__ profile,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __align__(16) uint4 x_sh[kFullVecsK6144];
    __shared__ __align__(16) uint4 w_sh[2][kFullVecsK6144];
    __shared__ float warp_partials[kPersistentWarp8PerRowK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    maybe_record_profile<kProfile>(profile, kProfileKernelStart);

    prefetch_x_tile_vecs<kUseCpAsync, kFullVecsK6144, kRow8Threads>(
        &x_sh[0],
        reinterpret_cast<const uint4*>(x_half));
    if constexpr (kUseCpAsync) {
        cp_async_commit_group();
        cp_async_wait_group<0>();
    }
    __syncthreads();

    int row = blockIdx.x;
    if (row < n) {
        prefetch_weight_tile_rows_no_commit<kUseCpAsync, 1, kFullVecsK6144>(
            &w_sh[0][0],
            w_half,
            stride_wn,
            row,
            0,
            n,
            kRow8Threads);
        if constexpr (kUseCpAsync) {
            cp_async_commit_group();
            cp_async_wait_group<0>();
        }
        __syncthreads();
    }

    for (int iter = 0; row < n; row += gridDim.x, ++iter) {
        const int stage = iter & 1;
        const int next_row = row + gridDim.x;

        if (next_row < n) {
            prefetch_weight_tile_rows_no_commit<kUseCpAsync, 1, kFullVecsK6144>(
                &w_sh[stage ^ 1][0],
                w_half,
                stride_wn,
                next_row,
                0,
                n,
                kRow8Threads);
            if constexpr (kUseCpAsync) {
                cp_async_commit_group();
            }
        }

        if constexpr (kProfile) {
            maybe_record_profile<kProfile>(profile, kProfileTile0LoadDone);
        }

        float sum = 0.0f;
        const uint4* w_row_vec = &w_sh[stage][0];
        __half2 h2_acc = __float2half2_rn(0.0f);

        #pragma unroll
        for (int vec_idx = lane + warp * 32; vec_idx < kFullVecsK6144; vec_idx += kRow8Threads) {
            const uint4 x_vec = x_sh[vec_idx];
            const uint4 w_vec = w_row_vec[vec_idx];
            accumulate_hfma2_uint4(x_vec, w_vec, h2_acc);
        }

        const float2 f2 = __half22float2(h2_acc);
        sum = f2.x + f2.y;

        if constexpr (kProfile) {
            maybe_record_profile<kProfile>(profile, kProfileTile0ComputeDone);
        }

        sum = warp_sum(sum);
        if (lane == 0) {
            warp_partials[warp] = sum;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            float result = 0.0f;
            #pragma unroll
            for (int warp_idx = 0; warp_idx < kPersistentWarp8PerRowK6144; ++warp_idx) {
                result += warp_partials[warp_idx];
            }
            if (has_bias) {
                result += __half2float(reinterpret_cast<const __half*>(bias)[row]);
            }
            reinterpret_cast<__half*>(out)[row] = __float2half(result);
        }

        if constexpr (kUseCpAsync) {
            if (next_row < n) {
                cp_async_wait_group<0>();
            }
        }
        __syncthreads();
    }

    if constexpr (kProfile) {
        __syncthreads();
        maybe_record_profile<kProfile>(profile, kProfileKernelDone);
    }
}

template <bool kProfile>
__global__ void nosplit_matvec_k6144_persistent_xfull_n8k1024_w_async_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    uint64_t* __restrict__ profile,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __align__(16) uint4 x_sh[kFullVecsK6144];
    __shared__ __align__(16) uint4 w_sh[2][kRowsPerBlockAsync * kAsyncVecsPerTileK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_slot = warp;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    maybe_record_profile<kProfile>(profile, kProfileKernelStart);

    prefetch_x_tile_vecs<kUseCpAsync, kFullVecsK6144, kRow8Threads>(
        &x_sh[0],
        reinterpret_cast<const uint4*>(x_half));
    if constexpr (kUseCpAsync) {
        cp_async_commit_group();
        cp_async_wait_group<0>();
    }
    __syncthreads();

    const int row_stride = gridDim.x * kRowsPerBlockAsync;
    for (int row_base = blockIdx.x * kRowsPerBlockAsync; row_base < n; row_base += row_stride) {
        const int row = row_base + row_slot;

        prefetch_weight_tile_rows_no_commit<kUseCpAsync, kRowsPerBlockAsync, kAsyncVecsPerTileK6144>(
            &w_sh[0][0],
            w_half,
            stride_wn,
            row_base,
            0,
            n,
            kRow8Threads);
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
            const int x_vec_base = tile_idx * kAsyncVecsPerTileK6144;

            if (tile_idx + 1 < 6) {
                const int next_tile_start = tile_start + kNosplitAsyncTileK6144;
                prefetch_weight_tile_rows_no_commit<kUseCpAsync, kRowsPerBlockAsync, kAsyncVecsPerTileK6144>(
                    &w_sh[stage ^ 1][0],
                    w_half,
                    stride_wn,
                    row_base,
                    next_tile_start,
                    n,
                    kRow8Threads);
                if constexpr (kUseCpAsync) {
                    cp_async_commit_group();
                }
            }

            if constexpr (kProfile) {
                maybe_record_profile<kProfile>(profile, kProfileTile0LoadDone + tile_idx * 2);
            }

            if (row < n) {
                const uint4* x_sh_stage_vec = &x_sh[x_vec_base];
                const uint4* w_sh_row_vec = &w_sh[stage][warp * kAsyncVecsPerTileK6144];
                __half2 h2_acc0 = __float2half2_rn(0.0f);
                __half2 h2_acc1 = __float2half2_rn(0.0f);

                #pragma unroll
                for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 64) {
                    const uint4 x_vec0 = x_sh_stage_vec[vec_idx];
                    const uint4 w_vec0 = w_sh_row_vec[vec_idx];
                    const uint4 x_vec1 = x_sh_stage_vec[vec_idx + 32];
                    const uint4 w_vec1 = w_sh_row_vec[vec_idx + 32];
                    accumulate_hfma2_uint4(x_vec0, w_vec0, h2_acc0);
                    accumulate_hfma2_uint4(x_vec1, w_vec1, h2_acc1);
                }

                const float2 f2_0 = __half22float2(h2_acc0);
                const float2 f2_1 = __half22float2(h2_acc1);
                sum += f2_0.x + f2_0.y + f2_1.x + f2_1.y;
            }

            if constexpr (kProfile) {
                maybe_record_profile<kProfile>(profile, kProfileTile0ComputeDone + tile_idx * 2);
            }

            if constexpr (kUseCpAsync) {
                if (tile_idx + 1 < 6) {
                    cp_async_wait_group<0>();
                }
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
        __syncthreads();
    }

    if constexpr (kProfile) {
        __syncthreads();
        maybe_record_profile<kProfile>(profile, kProfileKernelDone);
    }
}

__global__ void splitk_matvec_row4_shared_vec_half_v2_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    float* __restrict__ scratch,
    int64_t stride_wn,
    int n,
    int k,
    int split_k) {
    __shared__ __half x_sh[kVecTileK_v2];

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

    for (int tile_start = chunk_start; tile_start < chunk_end; tile_start += kVecTileK_v2) {
        const int tile_elems = min(kVecTileK_v2, chunk_end - tile_start);

        // --- 128-bit vectorized load of x into shared memory ---
        // Each thread loads 8 halfs (uint4 = 128 bits)
        // 128 threads * 8 halfs = 1024 halfs per pass (exactly one tile)
        const int vec8_limit_load = tile_elems & ~7;
        for (int idx = threadIdx.x * 8; idx < vec8_limit_load; idx += kRow4Threads * 8) {
            *reinterpret_cast<uint4*>(&x_sh[idx]) =
                *reinterpret_cast<const uint4*>(&x_half[tile_start + idx]);
        }
        // scalar tail for x load
        for (int idx = vec8_limit_load + threadIdx.x; idx < tile_elems; idx += kRow4Threads) {
            x_sh[idx] = x_half[tile_start + idx];
        }
        __syncthreads();

        // --- Vectorized compute with 128-bit loads ---
        const int compute_vec_limit = tile_elems & ~7;
        for (int idx = lane * 8; idx < compute_vec_limit; idx += 32 * 8) {
            // 128-bit load from shared memory (x)
            const uint4 x_vec = *reinterpret_cast<const uint4*>(&x_sh[idx]);
            // 128-bit load from global memory (weight) -- assumes stride_wk == 1
            const uint4 w_vec = *reinterpret_cast<const uint4*>(
                &w_half[row * stride_wn + tile_start + idx]);

            const __half2* x2 = reinterpret_cast<const __half2*>(&x_vec);
            const __half2* w2 = reinterpret_cast<const __half2*>(&w_vec);

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 xf = __half22float2(x2[i]);
                const float2 wf = __half22float2(w2[i]);
                sum += xf.x * wf.x + xf.y * wf.y;
            }
        }

        // scalar tail
        for (int idx = compute_vec_limit + lane; idx < tile_elems; idx += 32) {
            sum += __half2float(x_sh[idx]) * __half2float(w_half[row * stride_wn + tile_start + idx]);
        }
        __syncthreads();
    }

    sum = warp_sum(sum);
    if (lane == 0) {
        atomicAdd(scratch + row, sum);
    }
}

// ============================================================
// Persistent GEMV kernel: launch exactly #SM blocks, no split-K
// - 1 kernel launch (no scratch.zero_ + no finalize)
// - No atomicAdd
// - Each block loops over row groups, full K reduction per row
// ============================================================
__global__ void persistent_matvec_half_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    int64_t stride_wn,
    int n,
    int k,
    bool has_bias) {
    __shared__ __half x_sh[kVecTileK_v2];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);

    // Each block loops over row groups, stride = gridDim.x * 4
    for (int row_base = blockIdx.x * 4; row_base < n; row_base += gridDim.x * 4) {
        const int row = row_base + warp;
        float sum = 0.0f;

        // Tile over the full K dimension
        for (int tile_start = 0; tile_start < k; tile_start += kVecTileK_v2) {
            const int tile_elems = min(kVecTileK_v2, k - tile_start);

            // 128-bit vectorized load of x into shared memory
            const int vec8_limit_load = tile_elems & ~7;
            for (int idx = threadIdx.x * 8; idx < vec8_limit_load; idx += kRow4Threads * 8) {
                *reinterpret_cast<uint4*>(&x_sh[idx]) =
                    *reinterpret_cast<const uint4*>(&x_half[tile_start + idx]);
            }
            for (int idx = vec8_limit_load + threadIdx.x; idx < tile_elems; idx += kRow4Threads) {
                x_sh[idx] = x_half[tile_start + idx];
            }
            __syncthreads();

            if (row < n) {
                // 128-bit vectorized compute
                const int compute_vec_limit = tile_elems & ~7;
                for (int idx = lane * 8; idx < compute_vec_limit; idx += 32 * 8) {
                    const uint4 x_vec = *reinterpret_cast<const uint4*>(&x_sh[idx]);
                    const uint4 w_vec = *reinterpret_cast<const uint4*>(
                        &w_half[row * stride_wn + tile_start + idx]);

                    const __half2* x2 = reinterpret_cast<const __half2*>(&x_vec);
                    const __half2* w2 = reinterpret_cast<const __half2*>(&w_vec);

                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float2 xf = __half22float2(x2[i]);
                        const float2 wf = __half22float2(w2[i]);
                        sum += xf.x * wf.x + xf.y * wf.y;
                    }
                }

                // scalar tail
                for (int idx = compute_vec_limit + lane; idx < tile_elems; idx += 32) {
                    sum += __half2float(x_sh[idx]) * __half2float(w_half[row * stride_wn + tile_start + idx]);
                }
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
}

// ============================================================
// hfma2 variant: accumulate in half2, per-tile flush to float
// - Reduces instruction count (~6x fewer ALU ops in inner loop)
// - Per-tile flush keeps precision safe (max 16 hfma2 per flush)
// ============================================================
__global__ void persistent_matvec_hfma2_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    int64_t stride_wn,
    int n,
    int k,
    bool has_bias) {
    __shared__ __half x_sh[kVecTileK_v2];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);

    for (int row_base = blockIdx.x * 4; row_base < n; row_base += gridDim.x * 4) {
        const int row = row_base + warp;
        float sum = 0.0f;

        for (int tile_start = 0; tile_start < k; tile_start += kVecTileK_v2) {
            const int tile_elems = min(kVecTileK_v2, k - tile_start);

            // 128-bit vectorized load of x into shared memory
            const int vec8_limit_load = tile_elems & ~7;
            for (int idx = threadIdx.x * 8; idx < vec8_limit_load; idx += kRow4Threads * 8) {
                *reinterpret_cast<uint4*>(&x_sh[idx]) =
                    *reinterpret_cast<const uint4*>(&x_half[tile_start + idx]);
            }
            for (int idx = vec8_limit_load + threadIdx.x; idx < tile_elems; idx += kRow4Threads) {
                x_sh[idx] = x_half[tile_start + idx];
            }
            __syncthreads();

            if (row < n) {
                // hfma2 accumulation with per-tile flush
                __half2 h2_acc = __float2half2_rn(0.0f);
                const int compute_vec_limit = tile_elems & ~7;
                for (int idx = lane * 8; idx < compute_vec_limit; idx += 32 * 8) {
                    const uint4 x_vec = *reinterpret_cast<const uint4*>(&x_sh[idx]);
                    const uint4 w_vec = *reinterpret_cast<const uint4*>(
                        &w_half[row * stride_wn + tile_start + idx]);

                    const __half2* x2 = reinterpret_cast<const __half2*>(&x_vec);
                    const __half2* w2 = reinterpret_cast<const __half2*>(&w_vec);

                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        h2_acc = __hfma2(x2[i], w2[i], h2_acc);
                    }
                }
                // flush half2 accumulator to float
                const float2 f2 = __half22float2(h2_acc);
                sum += f2.x + f2.y;

                // scalar tail (stays in float)
                for (int idx = compute_vec_limit + lane; idx < tile_elems; idx += 32) {
                    sum += __half2float(x_sh[idx]) * __half2float(w_half[row * stride_wn + tile_start + idx]);
                }
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
}

void launch_nosplit_variant(
    const at::Half* x,
    const at::Half* weight,
    const at::Half* bias,
    at::Half* out,
    int64_t stride_wn,
    int n,
    int k,
    bool has_bias,
    int64_t variant,
    int num_sms,
    cudaStream_t stream) {
    const bool auto_use_row2_k2048 = (k == kNosplitSpecializedK2048 && n <= 2048);
    const int row2_grid_blocks = (n + 1) / 2;
    const int row4_grid_blocks = (n + 3) / 4;

    switch (variant) {
        case kNosplitVariantAuto:
            if (auto_use_row2_k2048) {
                nosplit_matvec_k2048_row2_hfma2_fwd_kernel<false><<<row2_grid_blocks, kRow2Threads, 0, stream>>>(
                    x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            } else if (k == kNosplitSpecializedK2048) {
                nosplit_matvec_k2048_hfma2_fwd_kernel<false><<<row4_grid_blocks, kRow4Threads, 0, stream>>>(
                    x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            } else if (k == 6144) {
                nosplit_matvec_k6144_hfma2_fwd_kernel<false><<<row4_grid_blocks, kRow4Threads, 0, stream>>>(
                    x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            } else {
                persistent_matvec_half_fwd_kernel<<<num_sms, kRow4Threads, 0, stream>>>(
                    x, weight, bias, out, stride_wn, n, k, has_bias);
            }
            break;
        case kNosplitVariantK2048Row2:
            TORCH_CHECK(k == kNosplitSpecializedK2048, "k2048_row2 variant requires k=2048");
            nosplit_matvec_k2048_row2_hfma2_fwd_kernel<false><<<row2_grid_blocks, kRow2Threads, 0, stream>>>(
                x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            break;
        case kNosplitVariantK2048Row4:
            TORCH_CHECK(k == kNosplitSpecializedK2048, "k2048_row4 variant requires k=2048");
            nosplit_matvec_k2048_hfma2_fwd_kernel<false><<<row4_grid_blocks, kRow4Threads, 0, stream>>>(
                x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            break;
        case kNosplitVariantK6144Row4:
            TORCH_CHECK(k == 6144, "k6144_row4 variant requires k=6144");
            nosplit_matvec_k6144_hfma2_fwd_kernel<false><<<row4_grid_blocks, kRow4Threads, 0, stream>>>(
                x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            break;
        case kNosplitVariantK6144WeightAsync:
            TORCH_CHECK(k == 6144, "k6144_weight_async variant requires k=6144");
            nosplit_matvec_k6144_weight_async_fwd_kernel<false><<<row4_grid_blocks, kRow4Threads, 0, stream>>>(
                x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            break;
        case kNosplitVariantK6144N8K1024Async: {
            TORCH_CHECK(k == 6144, "k6144_n8k1024_async variant requires k=6144");
            const int row8_grid_blocks = (n + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;
            nosplit_matvec_k6144_n8k1024_async_fwd_kernel<false><<<row8_grid_blocks, kRow8Threads, 0, stream>>>(
                x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            break;
        }
        case kNosplitVariantK6144N8K1024XWAsync: {
            TORCH_CHECK(k == 6144, "k6144_n8k1024_xw_async variant requires k=6144");
            const int row8_grid_blocks = (n + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;
            nosplit_matvec_k6144_n8k1024_xw_async_fwd_kernel<false><<<row8_grid_blocks, kRow8Threads, 0, stream>>>(
                x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            break;
        }
        case kNosplitVariantK6144N8K512S3XWAsync: {
            TORCH_CHECK(k == 6144, "k6144_n8k512_s3_xw_async variant requires k=6144");
            const int row8_grid_blocks = (n + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;
            nosplit_matvec_k6144_n8_xw_async_multi_stage_fwd_kernel<kNosplitAsyncTileK6144Small, 3, false>
                <<<row8_grid_blocks, kRow8Threads, 0, stream>>>(
                    x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            break;
        }
        case kNosplitVariantK6144PersistentXFullW2Async:
            TORCH_CHECK(k == 6144, "k6144_persistent_xfull_w2_async variant requires k=6144");
            {
                int cc_major = 0;
                int cc_minor = 0;
                cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, at::cuda::current_device());
                cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, at::cuda::current_device());
                TORCH_CHECK(
                    (cc_major > 8) || (cc_major == 8 && cc_minor >= 6),
                    "k6144_persistent_xfull_w2_async variant requires sm86+ due to static shared-memory footprint");
            }
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                nosplit_matvec_k6144_persistent_xfull_w2_async_fwd_kernel<false>,
                cudaFuncAttributePreferredSharedMemoryCarveout,
                100));
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                nosplit_matvec_k6144_persistent_xfull_w2_async_fwd_kernel<false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                0));
            nosplit_matvec_k6144_persistent_xfull_w2_async_fwd_kernel<false>
                <<<num_sms, kRow4Threads, 0, stream>>>(
                    x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            break;
        case kNosplitVariantK6144PersistentXFullRow1Warp8Async:
            TORCH_CHECK(k == 6144, "k6144_persistent_xfull_row1_warp8_async variant requires k=6144");
            nosplit_matvec_k6144_persistent_xfull_row1_warp8_async_fwd_kernel<false>
                <<<num_sms, kRow8Threads, 0, stream>>>(
                    x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            break;
        case kNosplitVariantK6144PersistentXFullN8K1024WAsync:
            TORCH_CHECK(k == 6144, "k6144_persistent_xfull_n8k1024_w_async variant requires k=6144");
            nosplit_matvec_k6144_persistent_xfull_n8k1024_w_async_fwd_kernel<false>
                <<<num_sms, kRow8Threads, 0, stream>>>(
                    x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            break;
        case kNosplitVariantK6144N8K1024XWarpWAsync: {
            TORCH_CHECK(k == 6144, "k6144_n8k1024_xwarp_w_async variant requires k=6144");
            const int row8_grid_blocks = (n + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;
            nosplit_matvec_k6144_n8k1024_xwarp_w_async_fwd_kernel<false><<<row8_grid_blocks, kRow8Threads, 0, stream>>>(
                x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            break;
        }
        case kNosplitVariantK6144N8K1024XWarpWAsyncSms2: {
            TORCH_CHECK(k == 6144, "k6144_n8k1024_xwarp_w_async_sms2 variant requires k=6144");
            const int row8_grid_blocks = (n + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;
            const int launch_grid_blocks = std::min(row8_grid_blocks, std::max(1, num_sms * 2));
            nosplit_matvec_k6144_n8k1024_xwarp_w_async_sms2_fwd_kernel<false>
                <<<launch_grid_blocks, kRow8Threads, 0, stream>>>(
                    x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            break;
        }
        case kNosplitVariantK6144N8K1024XWarpWAsyncSms2Balanced: {
            TORCH_CHECK(k == 6144, "k6144_n8k1024_xwarp_w_async_sms2_balanced variant requires k=6144");
            const int row8_grid_blocks = (n + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;
            const int launch_grid_blocks = std::min(row8_grid_blocks, std::max(1, num_sms * 2));
            nosplit_matvec_k6144_n8k1024_xwarp_w_async_sms2_balanced_fwd_kernel
                <<<launch_grid_blocks, kRow8Threads, 0, stream>>>(
                    x, weight, bias, out, stride_wn, n, has_bias);
            break;
        }
        case kNosplitVariantPersistent:
            persistent_matvec_half_fwd_kernel<<<num_sms, kRow4Threads, 0, stream>>>(
                x, weight, bias, out, stride_wn, n, k, has_bias);
            break;
        case kNosplitVariantPersistentHfma2:
            persistent_matvec_hfma2_fwd_kernel<<<num_sms, kRow4Threads, 0, stream>>>(
                x, weight, bias, out, stride_wn, n, k, has_bias);
            break;
        default:
            TORCH_CHECK(false, "unknown nosplit variant: ", variant);
    }
}

}  // namespace

torch::Tensor splitk_matvec_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t split_k,
    const torch::Tensor& scratch,
    const torch::Tensor& out,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(scratch.is_cuda(), "scratch must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(scratch.scalar_type() == torch::kFloat32, "scratch must be float32");
    TORCH_CHECK(out.scalar_type() == x.scalar_type(), "out dtype mismatch");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    scratch.zero_();

    const dim3 grid_fwd(n, split_k);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        x.scalar_type(),
        "splitk_matvec_fwd",
        [&] {
            splitk_matvec_fwd_kernel<scalar_t><<<grid_fwd, kThreads, 0, stream>>>(
                x.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                scratch.data_ptr<float>(),
                weight.stride(0),
                weight.stride(1),
                n,
                k,
                static_cast<int>(split_k));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int block = 256;
    const int grid = (n + block - 1) / block;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        x.scalar_type(),
        "splitk_matvec_finalize",
        [&] {
            splitk_matvec_finalize_kernel<scalar_t><<<grid, block, 0, stream>>>(
                scratch.data_ptr<float>(),
                has_bias ? bias.data_ptr<scalar_t>() : nullptr,
                out.data_ptr<scalar_t>(),
                n,
                has_bias);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor splitk_matvec_row4_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t split_k,
    const torch::Tensor& scratch,
    const torch::Tensor& out,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(scratch.is_cuda(), "scratch must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(scratch.scalar_type() == torch::kFloat32, "scratch must be float32");
    TORCH_CHECK(out.scalar_type() == x.scalar_type(), "out dtype mismatch");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    scratch.zero_();

    const dim3 grid_fwd((n + 3) / 4, split_k);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        x.scalar_type(),
        "splitk_matvec_row4_fwd",
        [&] {
            splitk_matvec_row4_fwd_kernel<scalar_t><<<grid_fwd, kRow4Threads, 0, stream>>>(
                x.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                scratch.data_ptr<float>(),
                weight.stride(0),
                weight.stride(1),
                n,
                k,
                static_cast<int>(split_k));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int block = 256;
    const int grid = (n + block - 1) / block;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        x.scalar_type(),
        "splitk_matvec_row4_finalize",
        [&] {
            splitk_matvec_finalize_kernel<scalar_t><<<grid, block, 0, stream>>>(
                scratch.data_ptr<float>(),
                has_bias ? bias.data_ptr<scalar_t>() : nullptr,
                out.data_ptr<scalar_t>(),
                n,
                has_bias);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor splitk_matvec_row8_shared_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t split_k,
    const torch::Tensor& scratch,
    const torch::Tensor& out,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(scratch.is_cuda(), "scratch must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(scratch.scalar_type() == torch::kFloat32, "scratch must be float32");
    TORCH_CHECK(out.scalar_type() == x.scalar_type(), "out dtype mismatch");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    scratch.zero_();

    const dim3 grid_fwd((n + 7) / 8, split_k);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        x.scalar_type(),
        "splitk_matvec_row8_shared_fwd",
        [&] {
            splitk_matvec_row8_shared_fwd_kernel<scalar_t><<<grid_fwd, kRow8Threads, 0, stream>>>(
                x.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                scratch.data_ptr<float>(),
                weight.stride(0),
                weight.stride(1),
                n,
                k,
                static_cast<int>(split_k));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int block = 256;
    const int grid = (n + block - 1) / block;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        x.scalar_type(),
        "splitk_matvec_row8_shared_finalize",
        [&] {
            splitk_matvec_finalize_kernel<scalar_t><<<grid, block, 0, stream>>>(
                scratch.data_ptr<float>(),
                has_bias ? bias.data_ptr<scalar_t>() : nullptr,
                out.data_ptr<scalar_t>(),
                n,
                has_bias);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor splitk_matvec_row4_shared_vec_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t split_k,
    const torch::Tensor& scratch,
    const torch::Tensor& out,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(scratch.is_cuda(), "scratch must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "row4_shared_vec_half only supports fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "row4_shared_vec_half only supports fp16");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(scratch.scalar_type() == torch::kFloat32, "scratch must be float32");
    TORCH_CHECK(out.scalar_type() == x.scalar_type(), "out dtype mismatch");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    scratch.zero_();

    const dim3 grid_fwd((n + 3) / 4, split_k);
    splitk_matvec_row4_shared_vec_half_fwd_kernel<<<grid_fwd, kRow4Threads, 0, stream>>>(
        x.data_ptr<at::Half>(),
        weight.data_ptr<at::Half>(),
        scratch.data_ptr<float>(),
        weight.stride(0),
        weight.stride(1),
        n,
        k,
        static_cast<int>(split_k)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int block = 256;
    const int grid = (n + block - 1) / block;
    splitk_matvec_finalize_kernel<at::Half><<<grid, block, 0, stream>>>(
        scratch.data_ptr<float>(),
        has_bias ? bias.data_ptr<at::Half>() : nullptr,
        out.data_ptr<at::Half>(),
        n,
        has_bias
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor splitk_matvec_row4_shared_vec_half_v2_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t split_k,
    const torch::Tensor& scratch,
    const torch::Tensor& out,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(scratch.is_cuda(), "scratch must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "v2 only supports fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "v2 only supports fp16");
    TORCH_CHECK(weight.stride(1) == 1, "v2 requires contiguous weight (stride_wk==1)");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(scratch.scalar_type() == torch::kFloat32, "scratch must be float32");
    TORCH_CHECK(out.scalar_type() == x.scalar_type(), "out dtype mismatch");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    scratch.zero_();

    const dim3 grid_fwd((n + 3) / 4, split_k);
    splitk_matvec_row4_shared_vec_half_v2_fwd_kernel<<<grid_fwd, kRow4Threads, 0, stream>>>(
        x.data_ptr<at::Half>(),
        weight.data_ptr<at::Half>(),
        scratch.data_ptr<float>(),
        weight.stride(0),
        n,
        k,
        static_cast<int>(split_k)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int block_f = 256;
    const int grid_f = (n + block_f - 1) / block_f;
    splitk_matvec_finalize_kernel<at::Half><<<grid_f, block_f, 0, stream>>>(
        scratch.data_ptr<float>(),
        has_bias ? bias.data_ptr<at::Half>() : nullptr,
        out.data_ptr<at::Half>(),
        n,
        has_bias
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor persistent_matvec_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "persistent_matvec only supports fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "persistent_matvec only supports fp16");
    TORCH_CHECK(weight.stride(1) == 1, "persistent_matvec requires contiguous weight (stride_wk==1)");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    // Launch exactly #SM blocks (query device)
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, x.device().index());
    const int grid_blocks = num_sms;

    persistent_matvec_half_fwd_kernel<<<grid_blocks, kRow4Threads, 0, stream>>>(
        x.data_ptr<at::Half>(),
        weight.data_ptr<at::Half>(),
        has_bias ? bias.data_ptr<at::Half>() : nullptr,
        out.data_ptr<at::Half>(),
        weight.stride(0),
        n,
        k,
        has_bias
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor nosplit_matvec_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "nosplit_matvec only supports fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "nosplit_matvec only supports fp16");
    TORCH_CHECK(weight.stride(1) == 1, "nosplit_matvec requires contiguous weight");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    // 1 block per 4 rows — no split-K, no scratch, no atomicAdd, no finalize
    const bool use_row2_k2048 = (k == kNosplitSpecializedK2048 && n <= 2048);
    const int grid_blocks = use_row2_k2048 ? ((n + 1) / 2) : ((n + 3) / 4);

    if (use_row2_k2048) {
        nosplit_matvec_k2048_row2_hfma2_fwd_kernel<false><<<grid_blocks, kRow2Threads, 0, stream>>>(
            x.data_ptr<at::Half>(),
            weight.data_ptr<at::Half>(),
            has_bias ? bias.data_ptr<at::Half>() : nullptr,
            out.data_ptr<at::Half>(),
            nullptr,
            weight.stride(0),
            n,
            has_bias
        );
    } else if (k == kNosplitSpecializedK2048) {
        nosplit_matvec_k2048_hfma2_fwd_kernel<false><<<grid_blocks, kRow4Threads, 0, stream>>>(
            x.data_ptr<at::Half>(),
            weight.data_ptr<at::Half>(),
            has_bias ? bias.data_ptr<at::Half>() : nullptr,
            out.data_ptr<at::Half>(),
            nullptr,
            weight.stride(0),
            n,
            has_bias
        );
    } else if (k == 6144) {
        nosplit_matvec_k6144_hfma2_fwd_kernel<false><<<grid_blocks, kRow4Threads, 0, stream>>>(
            x.data_ptr<at::Half>(),
            weight.data_ptr<at::Half>(),
            has_bias ? bias.data_ptr<at::Half>() : nullptr,
            out.data_ptr<at::Half>(),
            nullptr,
            weight.stride(0),
            n,
            has_bias
        );
    } else {
        persistent_matvec_half_fwd_kernel<<<grid_blocks, kRow4Threads, 0, stream>>>(
            x.data_ptr<at::Half>(),
            weight.data_ptr<at::Half>(),
            has_bias ? bias.data_ptr<at::Half>() : nullptr,
            out.data_ptr<at::Half>(),
            weight.stride(0),
            n,
            k,
            has_bias
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor nosplit_hfma2_matvec_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "hfma2 nosplit only supports fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "hfma2 nosplit only supports fp16");
    TORCH_CHECK(weight.stride(1) == 1, "hfma2 nosplit requires contiguous weight");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const bool use_row2_k2048 = (k == kNosplitSpecializedK2048 && n <= 2048);
    const int grid_blocks = use_row2_k2048 ? ((n + 1) / 2) : ((n + 3) / 4);

    if (use_row2_k2048) {
        nosplit_matvec_k2048_row2_hfma2_fwd_kernel<false><<<grid_blocks, kRow2Threads, 0, stream>>>(
            x.data_ptr<at::Half>(),
            weight.data_ptr<at::Half>(),
            has_bias ? bias.data_ptr<at::Half>() : nullptr,
            out.data_ptr<at::Half>(),
            nullptr,
            weight.stride(0),
            n,
            has_bias
        );
    } else if (k == kNosplitSpecializedK2048) {
        nosplit_matvec_k2048_hfma2_fwd_kernel<false><<<grid_blocks, kRow4Threads, 0, stream>>>(
            x.data_ptr<at::Half>(),
            weight.data_ptr<at::Half>(),
            has_bias ? bias.data_ptr<at::Half>() : nullptr,
            out.data_ptr<at::Half>(),
            nullptr,
            weight.stride(0),
            n,
            has_bias
        );
    } else if (k == 6144) {
        nosplit_matvec_k6144_hfma2_fwd_kernel<false><<<grid_blocks, kRow4Threads, 0, stream>>>(
            x.data_ptr<at::Half>(),
            weight.data_ptr<at::Half>(),
            has_bias ? bias.data_ptr<at::Half>() : nullptr,
            out.data_ptr<at::Half>(),
            nullptr,
            weight.stride(0),
            n,
            has_bias
        );
    } else {
        persistent_matvec_hfma2_fwd_kernel<<<grid_blocks, kRow4Threads, 0, stream>>>(
            x.data_ptr<at::Half>(),
            weight.data_ptr<at::Half>(),
            has_bias ? bias.data_ptr<at::Half>() : nullptr,
            out.data_ptr<at::Half>(),
            weight.stride(0),
            n,
            k,
            has_bias
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor nosplit_matvec_half_variant_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    bool has_bias,
    int64_t variant) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "nosplit variant only supports fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "nosplit variant only supports fp16");
    TORCH_CHECK(weight.stride(1) == 1, "nosplit variant requires contiguous weight");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, x.device().index());
    launch_nosplit_variant(
        x.data_ptr<at::Half>(),
        weight.data_ptr<at::Half>(),
        has_bias ? bias.data_ptr<at::Half>() : nullptr,
        out.data_ptr<at::Half>(),
        weight.stride(0),
        n,
        k,
        has_bias,
        variant,
        num_sms,
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor nosplit_matvec_half_profile_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& profile,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(profile.is_cuda(), "profile must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "nosplit profile only supports fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "nosplit profile only supports fp16");
    TORCH_CHECK(weight.stride(1) == 1, "nosplit profile requires contiguous weight");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(profile.scalar_type() == torch::kInt64, "profile must be int64");
    TORCH_CHECK(profile.is_contiguous(), "profile must be contiguous");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    const bool use_row2_k2048 = (k == kNosplitSpecializedK2048 && n <= 2048);
    const int grid_blocks = use_row2_k2048 ? ((n + 1) / 2) : ((n + 3) / 4);
    TORCH_CHECK(
        k == kNosplitSpecializedK2048 || k == 6144,
        "nosplit internal profiling currently supports the active k=2048/6144 specialized kernels only");
    TORCH_CHECK(
        profile.numel() == static_cast<int64_t>(grid_blocks) * kNosplitProfileSlots,
        "profile tensor shape mismatch");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    profile.zero_();

    if (use_row2_k2048) {
        nosplit_matvec_k2048_row2_hfma2_fwd_kernel<true><<<grid_blocks, kRow2Threads, 0, stream>>>(
            x.data_ptr<at::Half>(),
            weight.data_ptr<at::Half>(),
            has_bias ? bias.data_ptr<at::Half>() : nullptr,
            out.data_ptr<at::Half>(),
            reinterpret_cast<uint64_t*>(profile.data_ptr<int64_t>()),
            weight.stride(0),
            n,
            has_bias
        );
    } else if (k == kNosplitSpecializedK2048) {
        nosplit_matvec_k2048_hfma2_fwd_kernel<true><<<grid_blocks, kRow4Threads, 0, stream>>>(
            x.data_ptr<at::Half>(),
            weight.data_ptr<at::Half>(),
            has_bias ? bias.data_ptr<at::Half>() : nullptr,
            out.data_ptr<at::Half>(),
            reinterpret_cast<uint64_t*>(profile.data_ptr<int64_t>()),
            weight.stride(0),
            n,
            has_bias
        );
    } else {
        nosplit_matvec_k6144_hfma2_fwd_kernel<true><<<grid_blocks, kRow4Threads, 0, stream>>>(
            x.data_ptr<at::Half>(),
            weight.data_ptr<at::Half>(),
            has_bias ? bias.data_ptr<at::Half>() : nullptr,
            out.data_ptr<at::Half>(),
            reinterpret_cast<uint64_t*>(profile.data_ptr<int64_t>()),
            weight.stride(0),
            n,
            has_bias
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor nosplit_matvec_half_profile_variant_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& profile,
    bool has_bias,
    int64_t variant) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(profile.is_cuda(), "profile must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "nosplit profile only supports fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "nosplit profile only supports fp16");
    TORCH_CHECK(weight.stride(1) == 1, "nosplit profile requires contiguous weight");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(profile.scalar_type() == torch::kInt64, "profile must be int64");
    TORCH_CHECK(profile.is_contiguous(), "profile must be contiguous");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    TORCH_CHECK(k == 6144, "variant profiling currently supports only k=6144");

    int grid_blocks = 0;
    switch (variant) {
        case kNosplitVariantK6144Row4:
            grid_blocks = (n + 3) / 4;
            break;
        case kNosplitVariantK6144N8K1024Async:
        case kNosplitVariantK6144N8K1024XWAsync:
        case kNosplitVariantK6144N8K1024XWarpWAsync:
            grid_blocks = (n + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;
            break;
        default:
            TORCH_CHECK(false, "variant profiling unsupported for variant: ", variant);
    }
    TORCH_CHECK(
        profile.numel() == static_cast<int64_t>(grid_blocks) * kNosplitProfileSlots,
        "profile tensor shape mismatch");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    profile.zero_();

    switch (variant) {
        case kNosplitVariantK6144Row4:
            nosplit_matvec_k6144_hfma2_fwd_kernel<true><<<grid_blocks, kRow4Threads, 0, stream>>>(
                x.data_ptr<at::Half>(),
                weight.data_ptr<at::Half>(),
                has_bias ? bias.data_ptr<at::Half>() : nullptr,
                out.data_ptr<at::Half>(),
                reinterpret_cast<uint64_t*>(profile.data_ptr<int64_t>()),
                weight.stride(0),
                n,
                has_bias
            );
            break;
        case kNosplitVariantK6144N8K1024Async:
            nosplit_matvec_k6144_n8k1024_async_fwd_kernel<true><<<grid_blocks, kRow8Threads, 0, stream>>>(
                x.data_ptr<at::Half>(),
                weight.data_ptr<at::Half>(),
                has_bias ? bias.data_ptr<at::Half>() : nullptr,
                out.data_ptr<at::Half>(),
                reinterpret_cast<uint64_t*>(profile.data_ptr<int64_t>()),
                weight.stride(0),
                n,
                has_bias
            );
            break;
        case kNosplitVariantK6144N8K1024XWAsync:
            nosplit_matvec_k6144_n8k1024_xw_async_fwd_kernel<true><<<grid_blocks, kRow8Threads, 0, stream>>>(
                x.data_ptr<at::Half>(),
                weight.data_ptr<at::Half>(),
                has_bias ? bias.data_ptr<at::Half>() : nullptr,
                out.data_ptr<at::Half>(),
                reinterpret_cast<uint64_t*>(profile.data_ptr<int64_t>()),
                weight.stride(0),
                n,
                has_bias
            );
            break;
        case kNosplitVariantK6144N8K1024XWarpWAsync:
            nosplit_matvec_k6144_n8k1024_xwarp_w_async_fwd_kernel<true><<<grid_blocks, kRow8Threads, 0, stream>>>(
                x.data_ptr<at::Half>(),
                weight.data_ptr<at::Half>(),
                has_bias ? bias.data_ptr<at::Half>() : nullptr,
                out.data_ptr<at::Half>(),
                reinterpret_cast<uint64_t*>(profile.data_ptr<int64_t>()),
                weight.stride(0),
                n,
                has_bias
            );
            break;
        default:
            TORCH_CHECK(false, "variant profiling unsupported for variant: ", variant);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor nosplit_matvec_half_profile_xwarp_detail_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& trace,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(trace.is_cuda(), "trace must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "xwarp detail profile only supports fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "xwarp detail profile only supports fp16");
    TORCH_CHECK(weight.stride(1) == 1, "xwarp detail profile requires contiguous weight");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(trace.scalar_type() == torch::kInt64, "trace must be int64");
    TORCH_CHECK(trace.is_contiguous(), "trace must be contiguous");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    TORCH_CHECK(k == 6144, "xwarp detail profile currently supports only k=6144");

    const int grid_blocks = (n + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;
    TORCH_CHECK(
        trace.numel() == static_cast<int64_t>(grid_blocks) * kRowsPerBlockAsync * kXWarpDetailSlots,
        "trace tensor shape mismatch");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    trace.zero_();

    nosplit_matvec_k6144_n8k1024_xwarp_w_async_trace_fwd_kernel<<<grid_blocks, kRow8Threads, 0, stream>>>(
        x.data_ptr<at::Half>(),
        weight.data_ptr<at::Half>(),
        has_bias ? bias.data_ptr<at::Half>() : nullptr,
        out.data_ptr<at::Half>(),
        trace.data_ptr<int64_t>(),
        nullptr,
        weight.stride(0),
        n,
        has_bias
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor nosplit_xwarp_wait_sync_calibration_cuda(
    const torch::Tensor& trace) {
    TORCH_CHECK(trace.is_cuda(), "trace must be CUDA");
    TORCH_CHECK(trace.scalar_type() == torch::kInt64, "trace must be int64");
    TORCH_CHECK(trace.is_contiguous(), "trace must be contiguous");
    TORCH_CHECK(trace.dim() == 3, "trace must be [grid_blocks, warp_groups, slots]");
    TORCH_CHECK(trace.size(1) == kRowsPerBlockAsync, "trace warp_groups must be 8");
    TORCH_CHECK(trace.size(2) == kXWarpDetailSlots, "trace slot count mismatch");

    const int grid_blocks = static_cast<int>(trace.size(0));
    c10::cuda::CUDAGuard device_guard(trace.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    trace.zero_();

    nosplit_xwarp_wait_sync_calibration_kernel<<<grid_blocks, kRow8Threads, 0, stream>>>(
        trace.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return trace;
}

torch::Tensor nosplit_matvec_half_profile_xwarp_detail_with_smid_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& trace,
    const torch::Tensor& smid,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(trace.is_cuda(), "trace must be CUDA");
    TORCH_CHECK(smid.is_cuda(), "smid must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "xwarp detail profile only supports fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "xwarp detail profile only supports fp16");
    TORCH_CHECK(weight.stride(1) == 1, "xwarp detail profile requires contiguous weight");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(trace.scalar_type() == torch::kInt64, "trace must be int64");
    TORCH_CHECK(trace.is_contiguous(), "trace must be contiguous");
    TORCH_CHECK(smid.scalar_type() == torch::kInt, "smid must be int32");
    TORCH_CHECK(smid.is_contiguous(), "smid must be contiguous");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    TORCH_CHECK(k == 6144, "xwarp detail profile currently supports only k=6144");

    const int grid_blocks = (n + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;
    TORCH_CHECK(
        trace.numel() == static_cast<int64_t>(grid_blocks) * kRowsPerBlockAsync * kXWarpDetailSlots,
        "trace tensor shape mismatch");
    TORCH_CHECK(
        smid.numel() == static_cast<int64_t>(grid_blocks) * kRowsPerBlockAsync,
        "smid tensor shape mismatch");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    trace.zero_();
    smid.fill_(-1);

    nosplit_matvec_k6144_n8k1024_xwarp_w_async_trace_fwd_kernel<<<grid_blocks, kRow8Threads, 0, stream>>>(
        x.data_ptr<at::Half>(),
        weight.data_ptr<at::Half>(),
        has_bias ? bias.data_ptr<at::Half>() : nullptr,
        out.data_ptr<at::Half>(),
        trace.data_ptr<int64_t>(),
        smid.data_ptr<int32_t>(),
        weight.stride(0),
        n,
        has_bias
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor nosplit_matvec_half_profile_xwarp_sms2_cta_trace_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& trace,
    const torch::Tensor& smid,
    const torch::Tensor& row_base,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(trace.is_cuda(), "trace must be CUDA");
    TORCH_CHECK(smid.is_cuda(), "smid must be CUDA");
    TORCH_CHECK(row_base.is_cuda(), "row_base must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "sms2 CTA trace only supports fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "sms2 CTA trace only supports fp16");
    TORCH_CHECK(weight.stride(1) == 1, "sms2 CTA trace requires contiguous weight");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(trace.scalar_type() == torch::kInt64, "trace must be int64");
    TORCH_CHECK(trace.is_contiguous(), "trace must be contiguous");
    TORCH_CHECK(smid.scalar_type() == torch::kInt, "smid must be int32");
    TORCH_CHECK(smid.is_contiguous(), "smid must be contiguous");
    TORCH_CHECK(row_base.scalar_type() == torch::kInt, "row_base must be int32");
    TORCH_CHECK(row_base.is_contiguous(), "row_base must be contiguous");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    TORCH_CHECK(k == 6144, "sms2 CTA trace currently supports only k=6144");

    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, x.device().index());
    const int full_grid_blocks = (n + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;
    const int launch_grid_blocks = std::min(full_grid_blocks, std::max(1, num_sms * 2));
    const int max_iters = (full_grid_blocks + launch_grid_blocks - 1) / launch_grid_blocks;

    TORCH_CHECK(trace.numel() == static_cast<int64_t>(launch_grid_blocks) * max_iters * 2, "trace tensor shape mismatch");
    TORCH_CHECK(smid.numel() == static_cast<int64_t>(launch_grid_blocks), "smid tensor shape mismatch");
    TORCH_CHECK(row_base.numel() == static_cast<int64_t>(launch_grid_blocks) * max_iters, "row_base tensor shape mismatch");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    trace.zero_();
    smid.fill_(-1);
    row_base.fill_(-1);

    nosplit_matvec_k6144_n8k1024_xwarp_w_async_sms2_cta_trace_fwd_kernel<<<launch_grid_blocks, kRow8Threads, 0, stream>>>(
        x.data_ptr<at::Half>(),
        weight.data_ptr<at::Half>(),
        has_bias ? bias.data_ptr<at::Half>() : nullptr,
        out.data_ptr<at::Half>(),
        trace.data_ptr<int64_t>(),
        smid.data_ptr<int32_t>(),
        row_base.data_ptr<int32_t>(),
        max_iters,
        weight.stride(0),
        n,
        has_bias
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor nosplit_matvec_half_profile_xwarp_sms2_balanced_cta_trace_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& trace,
    const torch::Tensor& smid,
    const torch::Tensor& row_base,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(trace.is_cuda(), "trace must be CUDA");
    TORCH_CHECK(smid.is_cuda(), "smid must be CUDA");
    TORCH_CHECK(row_base.is_cuda(), "row_base must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "sms2 balanced CTA trace only supports fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "sms2 balanced CTA trace only supports fp16");
    TORCH_CHECK(weight.stride(1) == 1, "sms2 balanced CTA trace requires contiguous weight");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(trace.scalar_type() == torch::kInt64, "trace must be int64");
    TORCH_CHECK(trace.is_contiguous(), "trace must be contiguous");
    TORCH_CHECK(smid.scalar_type() == torch::kInt, "smid must be int32");
    TORCH_CHECK(smid.is_contiguous(), "smid must be contiguous");
    TORCH_CHECK(row_base.scalar_type() == torch::kInt, "row_base must be int32");
    TORCH_CHECK(row_base.is_contiguous(), "row_base must be contiguous");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    TORCH_CHECK(k == 6144, "sms2 balanced CTA trace currently supports only k=6144");

    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, x.device().index());
    const int full_grid_blocks = (n + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;
    const int launch_grid_blocks = std::min(full_grid_blocks, std::max(1, num_sms * 2));
    const int max_rows_per_block = (n + launch_grid_blocks - 1) / launch_grid_blocks;
    const int max_iters = (max_rows_per_block + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;
    TORCH_CHECK(max_iters <= 2, "sms2 balanced CTA trace expects at most 2 row batches per CTA");

    TORCH_CHECK(trace.numel() == static_cast<int64_t>(launch_grid_blocks) * max_iters * 2, "trace tensor shape mismatch");
    TORCH_CHECK(smid.numel() == static_cast<int64_t>(launch_grid_blocks), "smid tensor shape mismatch");
    TORCH_CHECK(row_base.numel() == static_cast<int64_t>(launch_grid_blocks) * max_iters, "row_base tensor shape mismatch");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    trace.zero_();
    smid.fill_(-1);
    row_base.fill_(-1);

    nosplit_matvec_k6144_n8k1024_xwarp_w_async_sms2_balanced_cta_trace_fwd_kernel<<<launch_grid_blocks, kRow8Threads, 0, stream>>>(
        x.data_ptr<at::Half>(),
        weight.data_ptr<at::Half>(),
        has_bias ? bias.data_ptr<at::Half>() : nullptr,
        out.data_ptr<at::Half>(),
        trace.data_ptr<int64_t>(),
        smid.data_ptr<int32_t>(),
        row_base.data_ptr<int32_t>(),
        max_iters,
        weight.stride(0),
        n,
        has_bias
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor weight_stream_read_half_cuda(
    const torch::Tensor& weight,
    const torch::Tensor& out) {
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "weight stream read only supports fp16");
    TORCH_CHECK(weight.is_contiguous(), "weight stream read requires contiguous weight");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(out.scalar_type() == torch::kInt, "out must be int32");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    constexpr int kWeightStreamThreads = 256;
    const int warps_per_block = kWeightStreamThreads / 32;
    TORCH_CHECK(out.numel() % warps_per_block == 0, "out size must be divisible by warps_per_block");

    const int64_t num_vecs = weight.numel() / kHalfPerVec128;
    TORCH_CHECK(
        num_vecs * kHalfPerVec128 == weight.numel(),
        "weight stream read requires element count divisible by 8");

    const int grid_blocks = static_cast<int>(out.numel() / warps_per_block);
    TORCH_CHECK(grid_blocks > 0, "weight stream read requires non-empty output");

    c10::cuda::CUDAGuard device_guard(weight.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    weight_stream_read_half_fwd_kernel<<<grid_blocks, kWeightStreamThreads, 0, stream>>>(
        weight.data_ptr<at::Half>(),
        out.data_ptr<int32_t>(),
        num_vecs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor nosplit_weight_only_half_variant_cuda(
    const torch::Tensor& weight,
    const torch::Tensor& out,
    int64_t variant) {
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "nosplit weight-only variant only supports fp16");
    TORCH_CHECK(weight.stride(1) == 1, "nosplit weight-only variant requires contiguous weight");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(out.scalar_type() == torch::kInt, "out must be int32");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    TORCH_CHECK(k == 2048 || k == 6144, "nosplit weight-only variant currently supports only k=2048/6144");
    TORCH_CHECK(out.numel() == n, "out size mismatch");

    c10::cuda::CUDAGuard device_guard(weight.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    switch (variant) {
        case kNosplitVariantK2048Row2: {
            const int grid_blocks = (n + 1) / 2;
            nosplit_weight_only_k2048_row2_fwd_kernel<<<grid_blocks, kRow2Threads, 0, stream>>>(
                weight.data_ptr<at::Half>(),
                out.data_ptr<int32_t>(),
                weight.stride(0),
                n);
            break;
        }
        case kNosplitVariantK2048Row4: {
            const int grid_blocks = (n + kRowsPerBlock - 1) / kRowsPerBlock;
            nosplit_weight_only_k2048_row4_fwd_kernel<<<grid_blocks, kRow4Threads, 0, stream>>>(
                weight.data_ptr<at::Half>(),
                out.data_ptr<int32_t>(),
                weight.stride(0),
                n);
            break;
        }
        case kNosplitVariantK6144Row4: {
            const int grid_blocks = (n + kRowsPerBlock - 1) / kRowsPerBlock;
            nosplit_weight_only_k6144_row4_fwd_kernel<<<grid_blocks, kRow4Threads, 0, stream>>>(
                weight.data_ptr<at::Half>(),
                out.data_ptr<int32_t>(),
                weight.stride(0),
                n);
            break;
        }
        case kNosplitVariantK6144N8K1024XWarpWAsync: {
            const int grid_blocks = (n + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;
            nosplit_weight_only_k6144_n8k1024_xwarp_w_async_fwd_kernel<<<grid_blocks, kRow8Threads, 0, stream>>>(
                weight.data_ptr<at::Half>(),
                out.data_ptr<int32_t>(),
                weight.stride(0),
                n);
            break;
        }
        default:
            TORCH_CHECK(
                false,
                "nosplit weight-only variant currently supports only k2048_row2, k2048_row4, k6144_row4 and k6144_n8k1024_xwarp_w_async");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
