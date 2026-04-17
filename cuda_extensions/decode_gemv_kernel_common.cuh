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

// Hardware grid sync kernels are in decode_gemv_gridsync.cu (separate TU, compiled with -rdc=true)

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

constexpr int kPersistentThreads = 1024;
constexpr int kPersistentWarps   = 32;  // 1024 / 32

struct AtomicGridSync {
    unsigned int* counter;
    unsigned int* generation;
    unsigned int nblocks;
    unsigned int* local_gen;

    __device__ __forceinline__ void sync() const {
        __syncthreads();
        if (threadIdx.x == 0) {
            const unsigned int my_gen = *local_gen;
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
            const unsigned int arrived = atomicAdd(counter, 1u);
            if (arrived == nblocks - 1) {
                *counter = 0;
                asm volatile("fence.acq_rel.gpu;" ::: "memory");
                atomicAdd(generation, 1u);
            } else {
                volatile unsigned int* vgen =
                    reinterpret_cast<volatile unsigned int*>(generation);
                while (*vgen <= my_gen) {
                }
            }
            *local_gen = my_gen + 1;
        }
        __syncthreads();
    }
};

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

