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
constexpr int kFusedMegaHiddenBatch = 256;
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
constexpr int kL2ParityProbeThreads = 256;
constexpr int kL2ParityProbeDynamicSmemBytes = 96 * 1024;

__device__ inline void software_grid_sync(
    int* barrier_count,
    int* barrier_phase,
    int expected_blocks) {
    __shared__ int phase_snapshot;
    __shared__ int is_last_block;
    if (threadIdx.x == 0) {
        phase_snapshot = atomicAdd(barrier_phase, 0);
        const int ticket = atomicAdd(barrier_count, 1);
        is_last_block = (ticket == expected_blocks - 1) ? 1 : 0;
        if (is_last_block) {
            __threadfence();
            *barrier_count = 0;
            atomicAdd(barrier_phase, 1);
        }
    }
    __syncthreads();
    if (!is_last_block) {
        while (atomicAdd(barrier_phase, 0) == phase_snapshot) {
        }
    }
    __syncthreads();
}

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
    kNosplitVariantK2048Row8Persistent = 16,
    kNosplitVariantK2048Row4Persistent = 17,
};

enum DualCopyRouteMode : int64_t {
    kDualCopyRouteForceA = 0,
    kDualCopyRouteForceB = 1,
    kDualCopyRouteSmidHalf = 2,
    kDualCopyRouteSmidParity = 3,
    kDualCopyRouteTable = 4,
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

__device__ inline uint64_t read_clock64_cycles() {
    return static_cast<uint64_t>(clock64());
}

__device__ inline int read_smid_int() {
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return static_cast<int>(smid);
}

__device__ inline uint32_t load_global_cg_u32(const uint32_t* ptr) {
#if defined(__CUDA_ARCH__)
    uint32_t value;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(ptr));
    return value;
#else
    return *ptr;
#endif
}

__device__ inline uint64_t load_global_cg_u64(const uint64_t* ptr) {
#if defined(__CUDA_ARCH__)
    uint64_t value;
    asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(value) : "l"(ptr));
    return value;
#else
    return *ptr;
#endif
}

__device__ inline int pick_dualcopy_route(
    int smid,
    const int32_t* route_table,
    int route_mode,
    int smid_threshold) {
    switch (route_mode) {
        case kDualCopyRouteForceA:
            return 0;
        case kDualCopyRouteForceB:
            return 1;
        case kDualCopyRouteSmidHalf:
            return smid >= smid_threshold ? 1 : 0;
        case kDualCopyRouteSmidParity:
            return smid & 1;
        case kDualCopyRouteTable:
            return route_table == nullptr ? 0 : (route_table[smid] & 1);
        default:
            return 0;
    }
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

__global__ void weight_stream_read_dual_half_fwd_kernel(
    const at::Half* __restrict__ weight_a,
    const at::Half* __restrict__ weight_b,
    const int32_t* __restrict__ route_table,
    int32_t* __restrict__ out,
    int64_t* __restrict__ trace,
    int32_t* __restrict__ smid_out,
    int32_t* __restrict__ route_out,
    int64_t vecs_per_block,
    int route_mode,
    int smid_threshold) {
    __shared__ uint64_t block_start;
    __shared__ int block_smid;
    __shared__ int selected_copy;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int64_t block_vec_base = static_cast<int64_t>(blockIdx.x) * vecs_per_block;

    if (threadIdx.x == 0) {
        block_smid = read_smid_int();
        selected_copy = pick_dualcopy_route(block_smid, route_table, route_mode, smid_threshold);
        if (smid_out != nullptr) {
            smid_out[blockIdx.x] = block_smid;
        }
        if (route_out != nullptr) {
            route_out[blockIdx.x] = selected_copy;
        }
        block_start = read_globaltimer();
    }
    __syncthreads();

    const uint4* weight_vec = reinterpret_cast<const uint4*>(selected_copy ? weight_b : weight_a);
    unsigned int sum = 0;
    for (int64_t vec_idx = threadIdx.x; vec_idx < vecs_per_block; vec_idx += blockDim.x) {
        const uint4 value = weight_vec[block_vec_base + vec_idx];
        sum += value.x + value.y + value.z + value.w;
    }
    sum = warp_sum_u32(sum);
    if (lane == 0) {
        out[blockIdx.x * warps_per_block + warp] = static_cast<int32_t>(sum);
    }

    __syncthreads();
    if (threadIdx.x == 0 && trace != nullptr) {
        trace[static_cast<int64_t>(blockIdx.x) * 2] = static_cast<int64_t>(block_start);
        trace[static_cast<int64_t>(blockIdx.x) * 2 + 1] = static_cast<int64_t>(read_globaltimer());
    }
}

__device__ inline void flush_l2_probe_block_segment(
    const int32_t* flush_buf,
    int64_t flush_elems_per_block,
    int32_t* flush_sink) {
    const uint32_t* flush_u32 = reinterpret_cast<const uint32_t*>(flush_buf);
    const int64_t block_base = static_cast<int64_t>(blockIdx.x) * flush_elems_per_block;

    unsigned int sum = 0;
    for (int64_t idx = threadIdx.x; idx < flush_elems_per_block; idx += blockDim.x) {
        sum += load_global_cg_u32(flush_u32 + block_base + idx);
    }
    if (flush_sink != nullptr) {
        atomicAdd(flush_sink + blockIdx.x, static_cast<int32_t>(sum));
    }
}

__global__ void l2_sequential_pointer_chase_trace_fwd_kernel(
    const int64_t* __restrict__ chain,
    const int64_t* __restrict__ start,
    const int32_t* __restrict__ flush_buf,
    int64_t flush_elems_per_block,
    int64_t chain_len,
    int64_t* __restrict__ latency_out,
    int32_t* __restrict__ smid_out,
    int64_t* __restrict__ final_idx_out,
    int32_t* __restrict__ flush_sink) {
    extern __shared__ uint8_t smem_dummy[];
    (void)smem_dummy;

    if (threadIdx.x == 0) {
        smid_out[blockIdx.x] = read_smid_int();

        // Warm page-table/TLB state once so the timed phase mostly reflects
        // cache / memory latency rather than first-touch translation overhead.
        int64_t idx = start[blockIdx.x];
        #pragma unroll 1
        for (int64_t step = 0; step < chain_len; ++step) {
            idx = static_cast<int64_t>(load_global_cg_u64(reinterpret_cast<const uint64_t*>(chain) + idx));
        }
        final_idx_out[blockIdx.x] = idx;
    }
    __syncthreads();

    flush_l2_probe_block_segment(flush_buf, flush_elems_per_block, flush_sink);
    __syncthreads();

    if (threadIdx.x == 0) {
        int64_t idx = start[blockIdx.x];
        const int64_t trace_base = static_cast<int64_t>(blockIdx.x) * chain_len;
        #pragma unroll 1
        for (int64_t step = 0; step < chain_len; ++step) {
            const uint64_t t0 = read_clock64_cycles();
            idx = static_cast<int64_t>(load_global_cg_u64(reinterpret_cast<const uint64_t*>(chain) + idx));
            const uint64_t t1 = read_clock64_cycles();
            latency_out[trace_base + step] = static_cast<int64_t>(t1 - t0);
        }
        final_idx_out[blockIdx.x] = idx;
    }
}

__global__ void l2_8kb_pointer_chase_pair_fwd_kernel(
    const int64_t* __restrict__ chain,
    const int64_t* __restrict__ start_even,
    const int64_t* __restrict__ start_odd,
    const int32_t* __restrict__ flush_buf,
    int64_t flush_elems_per_block,
    int64_t chain_len,
    int swap_order,
    int64_t* __restrict__ trace,
    int32_t* __restrict__ smid_out,
    int64_t* __restrict__ final_idx_out,
    int32_t* __restrict__ flush_sink) {
    extern __shared__ uint8_t smem_dummy[];
    (void)smem_dummy;

    if (threadIdx.x == 0) {
        smid_out[blockIdx.x] = read_smid_int();

        // Warm page-table/TLB state once so the timed phase mostly reflects
        // global-memory latency, not first-touch translation overhead.
        int64_t idx = start_even[blockIdx.x];
        #pragma unroll 1
        for (int64_t step = 0; step < chain_len; ++step) {
            idx = static_cast<int64_t>(load_global_cg_u64(reinterpret_cast<const uint64_t*>(chain) + idx));
        }
        final_idx_out[static_cast<int64_t>(blockIdx.x) * 2] = idx;

        idx = start_odd[blockIdx.x];
        #pragma unroll 1
        for (int64_t step = 0; step < chain_len; ++step) {
            idx = static_cast<int64_t>(load_global_cg_u64(reinterpret_cast<const uint64_t*>(chain) + idx));
        }
        final_idx_out[static_cast<int64_t>(blockIdx.x) * 2 + 1] = idx;
    }
    __syncthreads();

    flush_l2_probe_block_segment(flush_buf, flush_elems_per_block, flush_sink);
    __syncthreads();

    if (threadIdx.x == 0) {
        const int64_t* first_start = swap_order ? start_odd : start_even;
        int64_t idx = first_start[blockIdx.x];
        const uint64_t t0 = read_globaltimer();
        #pragma unroll 1
        for (int64_t step = 0; step < chain_len; ++step) {
            idx = static_cast<int64_t>(load_global_cg_u64(reinterpret_cast<const uint64_t*>(chain) + idx));
        }
        const uint64_t t1 = read_globaltimer();
        trace[static_cast<int64_t>(blockIdx.x) * 4] = static_cast<int64_t>(t0);
        trace[static_cast<int64_t>(blockIdx.x) * 4 + 1] = static_cast<int64_t>(t1);
        final_idx_out[static_cast<int64_t>(blockIdx.x) * 2 + (swap_order ? 1 : 0)] = idx;
    }
    __syncthreads();

    flush_l2_probe_block_segment(flush_buf, flush_elems_per_block, flush_sink);
    __syncthreads();

    if (threadIdx.x == 0) {
        const int64_t* second_start = swap_order ? start_even : start_odd;
        int64_t idx = second_start[blockIdx.x];
        const uint64_t t0 = read_globaltimer();
        #pragma unroll 1
        for (int64_t step = 0; step < chain_len; ++step) {
            idx = static_cast<int64_t>(load_global_cg_u64(reinterpret_cast<const uint64_t*>(chain) + idx));
        }
        const uint64_t t1 = read_globaltimer();
        trace[static_cast<int64_t>(blockIdx.x) * 4 + 2] = static_cast<int64_t>(t0);
        trace[static_cast<int64_t>(blockIdx.x) * 4 + 3] = static_cast<int64_t>(t1);
        final_idx_out[static_cast<int64_t>(blockIdx.x) * 2 + (swap_order ? 0 : 1)] = idx;
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

__global__ void nosplit_weight_only_k2048_row8_persistent_fwd_kernel(
    const at::Half* __restrict__ weight,
    int32_t* __restrict__ out,
    int64_t stride_wn,
    int n) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const __half* w_half = reinterpret_cast<const __half*>(weight);
    constexpr int kVecsK2048 = kNosplitSpecializedK2048 / kHalfPerVec128;
    const int row_stride = gridDim.x * kRowsPerBlockAsync;

    for (int row_base = blockIdx.x * kRowsPerBlockAsync; row_base < n; row_base += row_stride) {
        const int row = row_base + warp;
        unsigned int sum = 0;
        if (row < n) {
            const uint4* w_row_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn);
            #pragma unroll
            for (int vec_idx = lane; vec_idx < kVecsK2048; vec_idx += 32) {
                const uint4 w_vec = w_row_vec[vec_idx];
                sum += w_vec.x + w_vec.y + w_vec.z + w_vec.w;
            }
            sum = warp_sum_u32(sum);
            if (lane == 0) {
                out[row] = static_cast<int32_t>(sum);
            }
        }
    }
}

__global__ void nosplit_weight_only_k2048_row4_persistent_fwd_kernel(
    const at::Half* __restrict__ weight,
    int32_t* __restrict__ out,
    int64_t stride_wn,
    int n) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const __half* w_half = reinterpret_cast<const __half*>(weight);
    constexpr int kVecsK2048 = kNosplitSpecializedK2048 / kHalfPerVec128;
    const int row_stride = gridDim.x * kRowsPerBlock;

    for (int row_base = blockIdx.x * kRowsPerBlock; row_base < n; row_base += row_stride) {
        const int row = row_base + warp;
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

template <bool kUseCpAsync>
__device__ inline void run_k6144_xwarp_async_dualcopy_block(
    const __half* x_half,
    const __half* weight_a_half,
    const __half* weight_b_half,
    const __half* bias_half,
    __half* out_half,
    const int32_t* route_table,
    int route_mode,
    int smid_threshold,
    uint4 x_sh[2][kAsyncVecsPerTileK6144],
    uint4 w_sh[2][kRowsPerBlockAsync][kAsyncVecsPerTileK6144],
    int warp,
    int lane,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ int selected_copy;
    if (threadIdx.x == 0) {
        const int smid = read_smid_int();
        selected_copy = pick_dualcopy_route(smid, route_table, route_mode, smid_threshold);
    }
    __syncthreads();

    const __half* w_half = selected_copy ? weight_b_half : weight_a_half;
    const int row_base = blockIdx.x * kRowsPerBlockAsync;
    const int active_rows = max(0, min(kRowsPerBlockAsync, n - row_base));
    if (active_rows > 0) {
        run_k6144_xwarp_async_row_batch<kUseCpAsync>(
            x_half,
            w_half,
            bias_half,
            out_half,
            x_sh,
            w_sh,
            warp,
            lane,
            row_base,
            active_rows,
            stride_wn,
            n,
            has_bias);
    }
}

__global__ void nosplit_matvec_k6144_n8k1024_xwarp_w_async_dualcopy_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight_a,
    const at::Half* __restrict__ weight_b,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    const int32_t* __restrict__ route_table,
    int route_mode,
    int smid_threshold,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __align__(16) uint4 x_sh[2][kAsyncVecsPerTileK6144];
    __shared__ __align__(16) uint4 w_sh[2][kRowsPerBlockAsync][kAsyncVecsPerTileK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* weight_a_half = reinterpret_cast<const __half*>(weight_a);
    const __half* weight_b_half = reinterpret_cast<const __half*>(weight_b);
    const __half* bias_half = reinterpret_cast<const __half*>(bias);
    __half* out_half = reinterpret_cast<__half*>(out);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    run_k6144_xwarp_async_dualcopy_block<kUseCpAsync>(
        x_half,
        weight_a_half,
        weight_b_half,
        bias_half,
        out_half,
        route_table,
        route_mode,
        smid_threshold,
        x_sh,
        w_sh,
        warp,
        lane,
        stride_wn,
        n,
        has_bias);
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

torch::Tensor nosplit_matvec_half_dualcopy_routed_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight_a,
    const torch::Tensor& weight_b,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& route_table,
    bool has_bias,
    int64_t route_mode,
    int64_t smid_threshold) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight_a.is_cuda(), "weight_a must be CUDA");
    TORCH_CHECK(weight_b.is_cuda(), "weight_b must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "dualcopy nosplit matvec only supports fp16");
    TORCH_CHECK(weight_a.scalar_type() == torch::kFloat16, "dualcopy nosplit matvec only supports fp16");
    TORCH_CHECK(weight_b.scalar_type() == torch::kFloat16, "dualcopy nosplit matvec only supports fp16");
    TORCH_CHECK(weight_a.stride(1) == 1, "dualcopy nosplit matvec requires contiguous weight_a");
    TORCH_CHECK(weight_b.stride(1) == 1, "dualcopy nosplit matvec requires contiguous weight_b");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight_a.dim() == 2, "weight_a must be [N, K]");
    TORCH_CHECK(weight_b.sizes() == weight_a.sizes(), "weight_b size mismatch");
    TORCH_CHECK(weight_a.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(
        route_mode >= kDualCopyRouteForceA && route_mode <= kDualCopyRouteTable,
        "invalid dualcopy route_mode: ",
        route_mode);
    if (route_mode == kDualCopyRouteTable) {
        TORCH_CHECK(route_table.is_cuda(), "route_table must be CUDA for table routing");
        TORCH_CHECK(route_table.scalar_type() == torch::kInt, "route_table must be int32");
        TORCH_CHECK(route_table.is_contiguous(), "route_table must be contiguous");
    }
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight_a.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight_a.size(0));
    const int k = static_cast<int>(weight_a.size(1));
    TORCH_CHECK(k == 6144, "dualcopy nosplit matvec currently supports only k=6144");
    TORCH_CHECK(out.numel() == n, "out size mismatch");

    int num_sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, x.device().index()));
    if (route_mode == kDualCopyRouteTable) {
        TORCH_CHECK(route_table.numel() >= num_sms, "route_table too small for SM count");
    }
    const int grid_blocks = (n + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    nosplit_matvec_k6144_n8k1024_xwarp_w_async_dualcopy_fwd_kernel<<<grid_blocks, kRow8Threads, 0, stream>>>(
        x.data_ptr<at::Half>(),
        weight_a.data_ptr<at::Half>(),
        weight_b.data_ptr<at::Half>(),
        has_bias ? bias.data_ptr<at::Half>() : nullptr,
        out.data_ptr<at::Half>(),
        route_mode == kDualCopyRouteTable ? route_table.data_ptr<int32_t>() : nullptr,
        static_cast<int>(route_mode),
        static_cast<int>(smid_threshold),
        weight_a.stride(0),
        n,
        has_bias);
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

std::vector<torch::Tensor> fused_add_rmsnorm_gateup_half_cuda(
    const torch::Tensor& residual,
    const torch::Tensor& x,
    const torch::Tensor& norm_weight,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& residual_out,
    const torch::Tensor& gate_up_out,
    double eps,
    bool has_bias) {
    TORCH_CHECK(residual.is_cuda(), "residual must be CUDA");
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(norm_weight.is_cuda(), "norm_weight must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(residual_out.is_cuda(), "residual_out must be CUDA");
    TORCH_CHECK(gate_up_out.is_cuda(), "gate_up_out must be CUDA");
    TORCH_CHECK(residual.scalar_type() == torch::kFloat16, "fused_add_rmsnorm_gateup only supports fp16");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "fused_add_rmsnorm_gateup only supports fp16");
    TORCH_CHECK(norm_weight.scalar_type() == torch::kFloat16, "fused_add_rmsnorm_gateup only supports fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "fused_add_rmsnorm_gateup only supports fp16");
    TORCH_CHECK(weight.stride(1) == 1, "fused_add_rmsnorm_gateup requires contiguous weight");
    TORCH_CHECK(residual.dim() == 1, "residual must be flattened [K]");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(norm_weight.dim() == 1, "norm_weight must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(residual.numel() == kNosplitSpecializedK2048, "fused_add_rmsnorm_gateup expects K=2048");
    TORCH_CHECK(x.numel() == kNosplitSpecializedK2048, "fused_add_rmsnorm_gateup expects K=2048");
    TORCH_CHECK(norm_weight.numel() == kNosplitSpecializedK2048, "fused_add_rmsnorm_gateup expects K=2048 norm_weight");
    TORCH_CHECK(weight.size(1) == kNosplitSpecializedK2048, "fused_add_rmsnorm_gateup expects weight K=2048");
    TORCH_CHECK(residual_out.scalar_type() == torch::kFloat16, "residual_out must be fp16");
    TORCH_CHECK(gate_up_out.scalar_type() == torch::kFloat16, "gate_up_out must be fp16");
    TORCH_CHECK(residual_out.numel() == kNosplitSpecializedK2048, "residual_out size mismatch");
    TORCH_CHECK(gate_up_out.numel() == weight.size(0), "gate_up_out size mismatch");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int grid_blocks = (n + 3) / 4;
    c10::cuda::CUDAGuard device_guard(residual.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    fused_add_rmsnorm_gateup_k2048_hfma2_fwd_kernel<<<grid_blocks, kRow4Threads, 0, stream>>>(
        residual.data_ptr<at::Half>(),
        x.data_ptr<at::Half>(),
        norm_weight.data_ptr<at::Half>(),
        weight.data_ptr<at::Half>(),
        has_bias ? bias.data_ptr<at::Half>() : nullptr,
        residual_out.data_ptr<at::Half>(),
        gate_up_out.data_ptr<at::Half>(),
        static_cast<float>(eps),
        weight.stride(0),
        n,
        has_bias
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {residual_out, gate_up_out};
}

torch::Tensor fused_silu_downproj_half_cuda(
    const torch::Tensor& gate_up,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    bool has_bias) {
    TORCH_CHECK(gate_up.is_cuda(), "gate_up must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(gate_up.scalar_type() == torch::kFloat16, "fused_silu_downproj only supports fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "fused_silu_downproj only supports fp16");
    TORCH_CHECK(weight.stride(1) == 1, "fused_silu_downproj requires contiguous weight");
    TORCH_CHECK(gate_up.dim() == 1, "gate_up must be flattened [12288]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(gate_up.numel() == 12288, "fused_silu_downproj expects gate_up size 12288");
    TORCH_CHECK(weight.size(1) == 6144, "fused_silu_downproj expects weight K=6144");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(out.numel() == weight.size(0), "out size mismatch");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int grid_blocks = (n + 3) / 4;
    c10::cuda::CUDAGuard device_guard(gate_up.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    fused_silu_downproj_k6144_hfma2_fwd_kernel<<<grid_blocks, kRow4Threads, 0, stream>>>(
        gate_up.data_ptr<at::Half>(),
        weight.data_ptr<at::Half>(),
        has_bias ? bias.data_ptr<at::Half>() : nullptr,
        out.data_ptr<at::Half>(),
        weight.stride(0),
        n,
        has_bias
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_gateup_silu_downproj_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& bias,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(gate_up_weight.is_cuda(), "gate_up_weight must be CUDA");
    TORCH_CHECK(down_proj_weight.is_cuda(), "down_proj_weight must be CUDA");
    TORCH_CHECK(hidden_out.is_cuda(), "hidden_out must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "fused_gateup_silu_downproj only supports fp16");
    TORCH_CHECK(gate_up_weight.scalar_type() == torch::kFloat16, "fused_gateup_silu_downproj only supports fp16");
    TORCH_CHECK(down_proj_weight.scalar_type() == torch::kFloat16, "fused_gateup_silu_downproj only supports fp16");
    TORCH_CHECK(gate_up_weight.stride(1) == 1, "fused_gateup_silu_downproj requires contiguous gate_up_weight");
    TORCH_CHECK(down_proj_weight.stride(1) == 1, "fused_gateup_silu_downproj requires contiguous down_proj_weight");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(gate_up_weight.dim() == 2, "gate_up_weight must be [12288, 2048]");
    TORCH_CHECK(down_proj_weight.dim() == 2, "down_proj_weight must be [2048, 6144]");
    TORCH_CHECK(x.numel() == kNosplitSpecializedK2048, "fused_gateup_silu_downproj expects K=2048");
    TORCH_CHECK(gate_up_weight.size(0) == 12288 && gate_up_weight.size(1) == kNosplitSpecializedK2048,
        "fused_gateup_silu_downproj expects gate_up_weight [12288, 2048]");
    TORCH_CHECK(down_proj_weight.size(0) == 2048 && down_proj_weight.size(1) == 6144,
        "fused_gateup_silu_downproj expects down_proj_weight [2048, 6144]");
    TORCH_CHECK(hidden_out.scalar_type() == torch::kFloat16, "hidden_out must be fp16");
    TORCH_CHECK(hidden_out.numel() == 6144, "hidden_out size mismatch");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(out.numel() == down_proj_weight.size(0), "out size mismatch");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == down_proj_weight.size(0), "bias size mismatch");
    }

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int hidden_grid_blocks = (6144 + 3) / 4;
    fused_gateup_silu_k2048_hfma2_fwd_kernel<<<hidden_grid_blocks, kRow4Threads, 0, stream>>>(
        x.data_ptr<at::Half>(),
        gate_up_weight.data_ptr<at::Half>(),
        hidden_out.data_ptr<at::Half>(),
        gate_up_weight.stride(0)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, x.device().index());
    launch_nosplit_variant(
        hidden_out.data_ptr<at::Half>(),
        down_proj_weight.data_ptr<at::Half>(),
        has_bias ? bias.data_ptr<at::Half>() : nullptr,
        out.data_ptr<at::Half>(),
        down_proj_weight.stride(0),
        static_cast<int>(down_proj_weight.size(0)),
        static_cast<int>(down_proj_weight.size(1)),
        has_bias,
        kNosplitVariantK6144N8K1024XWarpWAsync,
        num_sms,
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_gateup_silu_downproj_mega_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& out) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(gate_up_weight.is_cuda(), "gate_up_weight must be CUDA");
    TORCH_CHECK(down_proj_weight.is_cuda(), "down_proj_weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "fused_gateup_silu_downproj_mega only supports fp16");
    TORCH_CHECK(gate_up_weight.scalar_type() == torch::kFloat16, "fused_gateup_silu_downproj_mega only supports fp16");
    TORCH_CHECK(down_proj_weight.scalar_type() == torch::kFloat16, "fused_gateup_silu_downproj_mega only supports fp16");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(gate_up_weight.stride(1) == 1, "fused_gateup_silu_downproj_mega requires contiguous gate_up_weight");
    TORCH_CHECK(down_proj_weight.stride(1) == 1, "fused_gateup_silu_downproj_mega requires contiguous down_proj_weight");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(gate_up_weight.dim() == 2, "gate_up_weight must be [12288, 2048]");
    TORCH_CHECK(down_proj_weight.dim() == 2, "down_proj_weight must be [2048, 6144]");
    TORCH_CHECK(x.numel() == kNosplitSpecializedK2048, "fused_gateup_silu_downproj_mega expects K=2048");
    TORCH_CHECK(gate_up_weight.size(0) == 12288 && gate_up_weight.size(1) == kNosplitSpecializedK2048,
        "fused_gateup_silu_downproj_mega expects gate_up_weight [12288, 2048]");
    TORCH_CHECK(down_proj_weight.size(0) == 2048 && down_proj_weight.size(1) == 6144,
        "fused_gateup_silu_downproj_mega expects down_proj_weight [2048, 6144]");
    TORCH_CHECK(out.numel() == down_proj_weight.size(0), "out size mismatch");

    const int grid_blocks = (6144 + kFusedMegaHiddenBatch - 1) / kFusedMegaHiddenBatch;
    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    fused_gateup_silu_downproj_mega_k2048_k6144_fwd_kernel<<<grid_blocks, kRow8Threads, 0, stream>>>(
        x.data_ptr<at::Half>(),
        gate_up_weight.data_ptr<at::Half>(),
        down_proj_weight.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        gate_up_weight.stride(0),
        down_proj_weight.stride(0)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_gateup_silu_downproj_mega_partial_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& partial_out) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(gate_up_weight.is_cuda(), "gate_up_weight must be CUDA");
    TORCH_CHECK(down_proj_weight.is_cuda(), "down_proj_weight must be CUDA");
    TORCH_CHECK(partial_out.is_cuda(), "partial_out must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "fused_gateup_silu_downproj_mega_partial only supports fp16");
    TORCH_CHECK(gate_up_weight.scalar_type() == torch::kFloat16, "fused_gateup_silu_downproj_mega_partial only supports fp16");
    TORCH_CHECK(down_proj_weight.scalar_type() == torch::kFloat16, "fused_gateup_silu_downproj_mega_partial only supports fp16");
    TORCH_CHECK(partial_out.scalar_type() == torch::kFloat16, "partial_out must be fp16");
    TORCH_CHECK(gate_up_weight.stride(1) == 1, "fused_gateup_silu_downproj_mega_partial requires contiguous gate_up_weight");
    TORCH_CHECK(down_proj_weight.stride(1) == 1, "fused_gateup_silu_downproj_mega_partial requires contiguous down_proj_weight");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(gate_up_weight.dim() == 2, "gate_up_weight must be [12288, 2048]");
    TORCH_CHECK(down_proj_weight.dim() == 2, "down_proj_weight must be [2048, 6144]");
    TORCH_CHECK(x.numel() == kNosplitSpecializedK2048, "fused_gateup_silu_downproj_mega_partial expects K=2048");
    const int grid_blocks = (6144 + kFusedMegaHiddenBatch - 1) / kFusedMegaHiddenBatch;
    TORCH_CHECK(partial_out.dim() == 2, "partial_out must be [grid_blocks, 2048]");
    TORCH_CHECK(partial_out.size(0) == grid_blocks && partial_out.size(1) == 2048, "partial_out shape mismatch");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    fused_gateup_silu_downproj_mega_partial_k2048_k6144_fwd_kernel<<<grid_blocks, kRow8Threads, 0, stream>>>(
        x.data_ptr<at::Half>(),
        gate_up_weight.data_ptr<at::Half>(),
        down_proj_weight.data_ptr<at::Half>(),
        partial_out.data_ptr<at::Half>(),
        gate_up_weight.stride(0),
        down_proj_weight.stride(0)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return partial_out;
}

torch::Tensor fused_gateup_silu_downproj_softsync_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& barrier_count,
    const torch::Tensor& barrier_phase,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(gate_up_weight.is_cuda(), "gate_up_weight must be CUDA");
    TORCH_CHECK(down_proj_weight.is_cuda(), "down_proj_weight must be CUDA");
    TORCH_CHECK(barrier_count.is_cuda(), "barrier_count must be CUDA");
    TORCH_CHECK(barrier_phase.is_cuda(), "barrier_phase must be CUDA");
    TORCH_CHECK(hidden_out.is_cuda(), "hidden_out must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "fused_gateup_silu_downproj_softsync only supports fp16");
    TORCH_CHECK(gate_up_weight.scalar_type() == torch::kFloat16, "fused_gateup_silu_downproj_softsync only supports fp16");
    TORCH_CHECK(down_proj_weight.scalar_type() == torch::kFloat16, "fused_gateup_silu_downproj_softsync only supports fp16");
    TORCH_CHECK(barrier_count.scalar_type() == torch::kInt32, "barrier_count must be int32");
    TORCH_CHECK(barrier_phase.scalar_type() == torch::kInt32, "barrier_phase must be int32");
    TORCH_CHECK(hidden_out.scalar_type() == torch::kFloat16, "hidden_out must be fp16");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(gate_up_weight.stride(1) == 1, "fused_gateup_silu_downproj_softsync requires contiguous gate_up_weight");
    TORCH_CHECK(down_proj_weight.stride(1) == 1, "fused_gateup_silu_downproj_softsync requires contiguous down_proj_weight");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(gate_up_weight.dim() == 2, "gate_up_weight must be [12288, 2048]");
    TORCH_CHECK(down_proj_weight.dim() == 2, "down_proj_weight must be [2048, 6144]");
    TORCH_CHECK(x.numel() == kNosplitSpecializedK2048, "fused_gateup_silu_downproj_softsync expects K=2048");
    TORCH_CHECK(gate_up_weight.size(0) == 12288 && gate_up_weight.size(1) == kNosplitSpecializedK2048,
        "fused_gateup_silu_downproj_softsync expects gate_up_weight [12288, 2048]");
    TORCH_CHECK(down_proj_weight.size(0) == 2048 && down_proj_weight.size(1) == 6144,
        "fused_gateup_silu_downproj_softsync expects down_proj_weight [2048, 6144]");
    TORCH_CHECK(barrier_count.dim() == 1 && barrier_count.numel() == 1, "barrier_count must be [1]");
    TORCH_CHECK(barrier_phase.dim() == 1 && barrier_phase.numel() == 1, "barrier_phase must be [1]");
    TORCH_CHECK(hidden_out.dim() == 1 && hidden_out.numel() == 6144, "hidden_out must be [6144]");
    TORCH_CHECK(out.dim() == 1 && out.numel() == 2048, "out must be [2048]");

    constexpr int grid_blocks = 2048 / kRowsPerBlockAsync;
    static_assert(grid_blocks * kRowsPerBlockAsync == 2048, "softsync launch grid must fully cover 2048 rows");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    int num_sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, x.device().index()));

    int max_blocks_per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        fused_gateup_silu_downproj_softsync_k2048_k6144_fwd_kernel,
        kRow8Threads,
        0));
    TORCH_CHECK(
        max_blocks_per_sm * num_sms >= grid_blocks,
        "fused_gateup_silu_downproj_softsync launch would deadlock: resident blocks ",
        max_blocks_per_sm * num_sms,
        " < required grid blocks ",
        grid_blocks);

    C10_CUDA_CHECK(cudaMemsetAsync(barrier_count.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(barrier_phase.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));

    fused_gateup_silu_downproj_softsync_k2048_k6144_fwd_kernel<<<grid_blocks, kRow8Threads, 0, stream>>>(
        x.data_ptr<at::Half>(),
        gate_up_weight.data_ptr<at::Half>(),
        down_proj_weight.data_ptr<at::Half>(),
        barrier_count.data_ptr<int32_t>(),
        barrier_phase.data_ptr<int32_t>(),
        hidden_out.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        gate_up_weight.stride(0),
        down_proj_weight.stride(0));
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

torch::Tensor weight_stream_read_dual_half_cuda(
    const torch::Tensor& weight_a,
    const torch::Tensor& weight_b,
    const torch::Tensor& out,
    const torch::Tensor& trace,
    const torch::Tensor& smid,
    const torch::Tensor& route,
    const torch::Tensor& route_table,
    int64_t vecs_per_block,
    int64_t route_mode,
    int64_t smid_threshold) {
    TORCH_CHECK(weight_a.is_cuda(), "weight_a must be CUDA");
    TORCH_CHECK(weight_b.is_cuda(), "weight_b must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(trace.is_cuda(), "trace must be CUDA");
    TORCH_CHECK(smid.is_cuda(), "smid must be CUDA");
    TORCH_CHECK(route.is_cuda(), "route must be CUDA");
    TORCH_CHECK(weight_a.scalar_type() == torch::kFloat16, "dual weight stream only supports fp16");
    TORCH_CHECK(weight_b.scalar_type() == torch::kFloat16, "dual weight stream only supports fp16");
    TORCH_CHECK(weight_a.is_contiguous(), "dual weight stream requires contiguous weight_a");
    TORCH_CHECK(weight_b.is_contiguous(), "dual weight stream requires contiguous weight_b");
    TORCH_CHECK(weight_a.sizes() == weight_b.sizes(), "dual weight stream weight size mismatch");
    TORCH_CHECK(weight_a.dim() == 2, "weight_a must be [N, K]");
    TORCH_CHECK(out.scalar_type() == torch::kInt, "out must be int32");
    TORCH_CHECK(trace.scalar_type() == torch::kInt64, "trace must be int64");
    TORCH_CHECK(smid.scalar_type() == torch::kInt, "smid must be int32");
    TORCH_CHECK(route.scalar_type() == torch::kInt, "route must be int32");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(trace.is_contiguous(), "trace must be contiguous");
    TORCH_CHECK(smid.is_contiguous(), "smid must be contiguous");
    TORCH_CHECK(route.is_contiguous(), "route must be contiguous");
    TORCH_CHECK(vecs_per_block > 0, "vecs_per_block must be > 0");
    TORCH_CHECK(
        route_mode >= kDualCopyRouteForceA && route_mode <= kDualCopyRouteTable,
        "invalid dual weight stream route_mode: ",
        route_mode);

    constexpr int kWeightStreamThreads = 256;
    const int warps_per_block = kWeightStreamThreads / 32;
    TORCH_CHECK(out.numel() % warps_per_block == 0, "out size must be divisible by warps_per_block");
    const int grid_blocks = static_cast<int>(out.numel() / warps_per_block);
    TORCH_CHECK(grid_blocks > 0, "dual weight stream requires non-empty output");
    TORCH_CHECK(trace.numel() == static_cast<int64_t>(grid_blocks) * 2, "trace size mismatch");
    TORCH_CHECK(smid.numel() == grid_blocks, "smid size mismatch");
    TORCH_CHECK(route.numel() == grid_blocks, "route size mismatch");

    const int64_t total_vecs = weight_a.numel() / kHalfPerVec128;
    TORCH_CHECK(total_vecs * kHalfPerVec128 == weight_a.numel(), "dual weight stream requires element count divisible by 8");
    TORCH_CHECK(
        static_cast<int64_t>(grid_blocks) * vecs_per_block <= total_vecs,
        "dual weight stream grid * vecs_per_block exceeds tensor size");

    int num_sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, weight_a.device().index()));
    if (route_mode == kDualCopyRouteTable) {
        TORCH_CHECK(route_table.is_cuda(), "route_table must be CUDA for table routing");
        TORCH_CHECK(route_table.scalar_type() == torch::kInt, "route_table must be int32");
        TORCH_CHECK(route_table.is_contiguous(), "route_table must be contiguous");
        TORCH_CHECK(route_table.numel() >= num_sms, "route_table too small for SM count");
    }

    c10::cuda::CUDAGuard device_guard(weight_a.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    weight_stream_read_dual_half_fwd_kernel<<<grid_blocks, kWeightStreamThreads, 0, stream>>>(
        weight_a.data_ptr<at::Half>(),
        weight_b.data_ptr<at::Half>(),
        route_mode == kDualCopyRouteTable ? route_table.data_ptr<int32_t>() : nullptr,
        out.data_ptr<int32_t>(),
        trace.data_ptr<int64_t>(),
        smid.data_ptr<int32_t>(),
        route.data_ptr<int32_t>(),
        vecs_per_block,
        static_cast<int>(route_mode),
        static_cast<int>(smid_threshold));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor l2_8kb_pointer_chase_pair_cuda(
    const torch::Tensor& chain,
    const torch::Tensor& start_even,
    const torch::Tensor& start_odd,
    const torch::Tensor& flush_buf,
    const torch::Tensor& trace,
    const torch::Tensor& smid,
    const torch::Tensor& final_idx,
    const torch::Tensor& flush_sink,
    int64_t chain_len,
    int64_t swap_order) {
    TORCH_CHECK(chain.is_cuda(), "chain must be CUDA");
    TORCH_CHECK(start_even.is_cuda(), "start_even must be CUDA");
    TORCH_CHECK(start_odd.is_cuda(), "start_odd must be CUDA");
    TORCH_CHECK(flush_buf.is_cuda(), "flush_buf must be CUDA");
    TORCH_CHECK(trace.is_cuda(), "trace must be CUDA");
    TORCH_CHECK(smid.is_cuda(), "smid must be CUDA");
    TORCH_CHECK(final_idx.is_cuda(), "final_idx must be CUDA");
    TORCH_CHECK(flush_sink.is_cuda(), "flush_sink must be CUDA");
    TORCH_CHECK(chain.scalar_type() == torch::kInt64, "chain must be int64");
    TORCH_CHECK(start_even.scalar_type() == torch::kInt64, "start_even must be int64");
    TORCH_CHECK(start_odd.scalar_type() == torch::kInt64, "start_odd must be int64");
    TORCH_CHECK(flush_buf.scalar_type() == torch::kInt32, "flush_buf must be int32");
    TORCH_CHECK(trace.scalar_type() == torch::kInt64, "trace must be int64");
    TORCH_CHECK(smid.scalar_type() == torch::kInt, "smid must be int32");
    TORCH_CHECK(final_idx.scalar_type() == torch::kInt64, "final_idx must be int64");
    TORCH_CHECK(flush_sink.scalar_type() == torch::kInt, "flush_sink must be int32");
    TORCH_CHECK(chain.is_contiguous(), "chain must be contiguous");
    TORCH_CHECK(start_even.is_contiguous(), "start_even must be contiguous");
    TORCH_CHECK(start_odd.is_contiguous(), "start_odd must be contiguous");
    TORCH_CHECK(flush_buf.is_contiguous(), "flush_buf must be contiguous");
    TORCH_CHECK(trace.is_contiguous(), "trace must be contiguous");
    TORCH_CHECK(smid.is_contiguous(), "smid must be contiguous");
    TORCH_CHECK(final_idx.is_contiguous(), "final_idx must be contiguous");
    TORCH_CHECK(flush_sink.is_contiguous(), "flush_sink must be contiguous");
    TORCH_CHECK(chain.dim() == 1, "chain must be 1D");
    TORCH_CHECK(start_even.dim() == 1, "start_even must be 1D");
    TORCH_CHECK(start_odd.dim() == 1, "start_odd must be 1D");
    TORCH_CHECK(chain_len > 0, "chain_len must be > 0");

    const int64_t num_blocks = start_even.numel();
    TORCH_CHECK(start_odd.numel() == num_blocks, "start_odd size mismatch");
    TORCH_CHECK(trace.numel() == num_blocks * 4, "trace size mismatch");
    TORCH_CHECK(smid.numel() == num_blocks, "smid size mismatch");
    TORCH_CHECK(final_idx.numel() == num_blocks * 2, "final_idx size mismatch");
    TORCH_CHECK(flush_sink.numel() == num_blocks, "flush_sink size mismatch");
    TORCH_CHECK(flush_buf.numel() % num_blocks == 0, "flush_buf size must divide evenly by block count");

    const int64_t flush_elems_per_block = flush_buf.numel() / num_blocks;
    TORCH_CHECK(flush_elems_per_block > 0, "flush buffer per block must be > 0");

    c10::cuda::CUDAGuard device_guard(chain.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        l2_8kb_pointer_chase_pair_fwd_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        l2_8kb_pointer_chase_pair_fwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kL2ParityProbeDynamicSmemBytes));

    int max_blocks_per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        l2_8kb_pointer_chase_pair_fwd_kernel,
        kL2ParityProbeThreads,
        kL2ParityProbeDynamicSmemBytes));
    TORCH_CHECK(
        max_blocks_per_sm == 1,
        "l2_8kb pointer-chase probe expected 1 resident block/SM, got ",
        max_blocks_per_sm);

    trace.zero_();
    smid.fill_(-1);
    flush_sink.zero_();
    l2_8kb_pointer_chase_pair_fwd_kernel<<<
        static_cast<int>(num_blocks),
        kL2ParityProbeThreads,
        kL2ParityProbeDynamicSmemBytes,
        stream>>>(
            chain.data_ptr<int64_t>(),
            start_even.data_ptr<int64_t>(),
            start_odd.data_ptr<int64_t>(),
            flush_buf.data_ptr<int32_t>(),
            flush_elems_per_block,
            chain_len,
            static_cast<int>(swap_order != 0),
            trace.data_ptr<int64_t>(),
            smid.data_ptr<int32_t>(),
            final_idx.data_ptr<int64_t>(),
            flush_sink.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return trace;
}

torch::Tensor l2_sequential_pointer_chase_trace_cuda(
    const torch::Tensor& chain,
    const torch::Tensor& start,
    const torch::Tensor& flush_buf,
    const torch::Tensor& latency,
    const torch::Tensor& smid,
    const torch::Tensor& final_idx,
    const torch::Tensor& flush_sink,
    int64_t chain_len) {
    TORCH_CHECK(chain.is_cuda(), "chain must be CUDA");
    TORCH_CHECK(start.is_cuda(), "start must be CUDA");
    TORCH_CHECK(flush_buf.is_cuda(), "flush_buf must be CUDA");
    TORCH_CHECK(latency.is_cuda(), "latency must be CUDA");
    TORCH_CHECK(smid.is_cuda(), "smid must be CUDA");
    TORCH_CHECK(final_idx.is_cuda(), "final_idx must be CUDA");
    TORCH_CHECK(flush_sink.is_cuda(), "flush_sink must be CUDA");
    TORCH_CHECK(chain.scalar_type() == torch::kInt64, "chain must be int64");
    TORCH_CHECK(start.scalar_type() == torch::kInt64, "start must be int64");
    TORCH_CHECK(flush_buf.scalar_type() == torch::kInt32, "flush_buf must be int32");
    TORCH_CHECK(latency.scalar_type() == torch::kInt64, "latency must be int64");
    TORCH_CHECK(smid.scalar_type() == torch::kInt, "smid must be int32");
    TORCH_CHECK(final_idx.scalar_type() == torch::kInt64, "final_idx must be int64");
    TORCH_CHECK(flush_sink.scalar_type() == torch::kInt, "flush_sink must be int32");
    TORCH_CHECK(chain.is_contiguous(), "chain must be contiguous");
    TORCH_CHECK(start.is_contiguous(), "start must be contiguous");
    TORCH_CHECK(flush_buf.is_contiguous(), "flush_buf must be contiguous");
    TORCH_CHECK(latency.is_contiguous(), "latency must be contiguous");
    TORCH_CHECK(smid.is_contiguous(), "smid must be contiguous");
    TORCH_CHECK(final_idx.is_contiguous(), "final_idx must be contiguous");
    TORCH_CHECK(flush_sink.is_contiguous(), "flush_sink must be contiguous");
    TORCH_CHECK(chain.dim() == 1, "chain must be 1D");
    TORCH_CHECK(start.dim() == 1, "start must be 1D");
    TORCH_CHECK(chain_len > 0, "chain_len must be > 0");

    const int64_t num_blocks = start.numel();
    TORCH_CHECK(num_blocks > 0, "start must be non-empty");
    TORCH_CHECK(latency.numel() == num_blocks * chain_len, "latency size mismatch");
    TORCH_CHECK(smid.numel() == num_blocks, "smid size mismatch");
    TORCH_CHECK(final_idx.numel() == num_blocks, "final_idx size mismatch");
    TORCH_CHECK(flush_sink.numel() == num_blocks, "flush_sink size mismatch");
    TORCH_CHECK(flush_buf.numel() % num_blocks == 0, "flush_buf size must divide evenly by block count");

    const int64_t flush_elems_per_block = flush_buf.numel() / num_blocks;
    TORCH_CHECK(flush_elems_per_block > 0, "flush buffer per block must be > 0");

    c10::cuda::CUDAGuard device_guard(chain.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        l2_sequential_pointer_chase_trace_fwd_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        l2_sequential_pointer_chase_trace_fwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kL2ParityProbeDynamicSmemBytes));

    int max_blocks_per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        l2_sequential_pointer_chase_trace_fwd_kernel,
        kL2ParityProbeThreads,
        kL2ParityProbeDynamicSmemBytes));
    TORCH_CHECK(
        max_blocks_per_sm == 1,
        "l2 sequential pointer-chase trace expected 1 resident block/SM, got ",
        max_blocks_per_sm);

    latency.zero_();
    smid.fill_(-1);
    flush_sink.zero_();
    l2_sequential_pointer_chase_trace_fwd_kernel<<<
        static_cast<int>(num_blocks),
        kL2ParityProbeThreads,
        kL2ParityProbeDynamicSmemBytes,
        stream>>>(
            chain.data_ptr<int64_t>(),
            start.data_ptr<int64_t>(),
            flush_buf.data_ptr<int32_t>(),
            flush_elems_per_block,
            chain_len,
            latency.data_ptr<int64_t>(),
            smid.data_ptr<int32_t>(),
            final_idx.data_ptr<int64_t>(),
            flush_sink.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return latency;
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
        case kNosplitVariantK2048Row8Persistent: {
            TORCH_CHECK(k == kNosplitSpecializedK2048, "k2048_row8_persistent variant requires k=2048");
            int num_sms = 0;
            cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, weight.device().index());
            const int full_grid_blocks = (n + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;
            const int launch_grid_blocks = std::min(full_grid_blocks, std::max(1, num_sms * 2));
            nosplit_weight_only_k2048_row8_persistent_fwd_kernel<<<launch_grid_blocks, kRow8Threads, 0, stream>>>(
                weight.data_ptr<at::Half>(),
                out.data_ptr<int32_t>(),
                weight.stride(0),
                n);
            break;
        }
        case kNosplitVariantK2048Row4Persistent: {
            TORCH_CHECK(k == kNosplitSpecializedK2048, "k2048_row4_persistent variant requires k=2048");
            int num_sms = 0;
            cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, weight.device().index());
            const int full_grid_blocks = (n + kRowsPerBlock - 1) / kRowsPerBlock;
            const int launch_grid_blocks = std::min(full_grid_blocks, std::max(1, num_sms * 2));
            nosplit_weight_only_k2048_row4_persistent_fwd_kernel<<<launch_grid_blocks, kRow4Threads, 0, stream>>>(
                weight.data_ptr<at::Half>(),
                out.data_ptr<int32_t>(),
                weight.stride(0),
                n);
            break;
        }
        default:
            TORCH_CHECK(
                false,
                "nosplit weight-only variant currently supports only k2048_row2, k2048_row4, k2048_row4_persistent, k2048_row8_persistent, "
                "k6144_row4 and k6144_n8k1024_xwarp_w_async");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
