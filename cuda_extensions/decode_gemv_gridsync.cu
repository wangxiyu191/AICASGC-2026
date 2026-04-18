// Separate TU for cooperative-groups grid-sync kernels.
// Must be compiled with -rdc=true (relocatable device code).

#include <cooperative_groups.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cstdlib>

namespace cg = cooperative_groups;

namespace {

// Constants duplicated from decode_gemv_kernel.cu (they live in an anonymous
// namespace there, so we must re-declare them here).
constexpr int kRow8Threads = 256;
constexpr int kWarpSize = 32;
constexpr int kRowsPerBlockAsync = 8;
constexpr int kNosplitSpecializedK2048 = 2048;
constexpr int kNosplitAsyncTileK6144 = 1024;
constexpr int kHalfPerVec128 = 8;
constexpr int kAsyncVecsPerTileK6144 = kNosplitAsyncTileK6144 / kHalfPerVec128;

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

// Persistent kernel profile slots (per-iteration granularity)
constexpr int kPersistentProfileSlots = 9;
enum PersistentProfileSlot : int {
    PP_KERNEL_START       = 0,
    PP_X_LOAD_DONE        = 1,
    PP_GATEUP_ITER0_DONE  = 2,
    PP_GATEUP_ITER1_DONE  = 3,  // stays 0 if block does only 1 iteration
    PP_GRIDSYNC_START     = 4,
    PP_GRIDSYNC_DONE      = 5,
    PP_DOWN_HIDDEN_LOADED = 6,
    PP_DOWN_COMPUTE_DONE  = 7,
    PP_KERNEL_DONE        = 8,
};

// O-proj + post-attn MLP persistent kernel profile slots
// Slots 0-12: recorded with clock64() (SM cycle counter, ~0.7ns resolution)
// Slots 13-14: recorded with globaltimer (1024ns quantization) for calibration
constexpr int kOProjMlpProfileSlots = 16;
enum OProjMlpProfileSlot : int {
    OP_KERNEL_START       = 0,
    OP_X_LOAD_DONE        = 1,
    OP_OPROJ_DONE         = 2,
    OP_GRIDSYNC1_START    = 3,
    OP_GRIDSYNC1_DONE     = 4,
    OP_RESNORM_DONE       = 5,
    OP_GATEUP_ITER0_DONE  = 6,
    OP_GATEUP_ITER1_DONE  = 7,  // stays 0 if block does only 1 iteration
    OP_GRIDSYNC2_START    = 8,
    OP_GRIDSYNC2_DONE     = 9,
    OP_DOWN_HIDDEN_LOADED = 10,
    OP_DOWN_COMPUTE_DONE  = 11,
    OP_KERNEL_DONE        = 12,
    OP_GT_START           = 13,  // globaltimer at kernel start (calibration)
    OP_GT_END             = 14,  // globaltimer at kernel end (calibration)
    OP_SM_ID              = 15,  // SM ID from %smid PTX register
};

// Full-layer softsync profile slots
// Slots 0-19: recorded with clock64() (SM cycle counter, ~0.7ns resolution)
// Slots 20-21: recorded with globaltimer (1024ns quantization) for calibration
constexpr int kFullLayerProfileSlots = 24;
enum FullLayerProfileSlot : int {
    FL_KERNEL_START       = 0,
    FL_INPUT_RMSNORM_DONE = 1,
    FL_QKV_DONE           = 2,
    FL_SYNC1_START        = 3,
    FL_SYNC1_DONE         = 4,
    FL_QKVCACHE_DONE      = 5,
    FL_SYNC2_START        = 6,
    FL_SYNC2_DONE         = 7,
    FL_ATTENTION_DONE     = 8,
    FL_SYNC3_START        = 9,
    FL_SYNC3_DONE         = 10,
    FL_OPROJ_DONE         = 11,
    FL_SYNC4_START        = 12,
    FL_SYNC4_DONE         = 13,
    FL_POST_RMSNORM_DONE  = 14,
    FL_GATEUP_SILU_DONE   = 15,
    FL_SYNC5_START        = 16,
    FL_SYNC5_DONE         = 17,
    FL_DOWNPROJ_DONE      = 18,
    FL_KERNEL_DONE        = 19,
    FL_GT_START           = 20,  // globaltimer at kernel start (calibration)
    FL_GT_END             = 21,  // globaltimer at kernel end (calibration)
    FL_SM_ID              = 22,  // SM ID from %smid PTX register
    FL_ATTN_PREFETCH_LINES = 23, // issued prefetch.global.L2 cache-line count during attention
};

__device__ __forceinline__ uint64_t read_globaltimer() {
    uint64_t ts;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts));
    return ts;
}

__device__ __forceinline__ uint64_t read_clock64() {
    return static_cast<uint64_t>(clock64());
}

constexpr int kPrefetchLineBytes = 128;

__device__ __forceinline__ void prefetch_global_l2(const void* ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr));
#else
    (void)ptr;
#endif
}

__device__ __forceinline__ unsigned int prefetch_oproj_rows_for_block_l2_until(
    const __half* weight,
    int64_t weight_stride_wn,
    int grid_blocks,
    int future_block_idx,
    uint64_t deadline_gt) {
    if (weight == nullptr || weight_stride_wn <= 0 || grid_blocks <= 0) {
        return 0;
    }

    constexpr int kOProjRowsPerWave = 16;
    const int row_lines = static_cast<int>(
        (weight_stride_wn * static_cast<int64_t>(sizeof(__half)) + kPrefetchLineBytes - 1)
        / kPrefetchLineBytes);
    const int items_per_wave = kOProjRowsPerWave * row_lines;
    const int row_stride_bytes = static_cast<int>(weight_stride_wn * sizeof(__half));

    unsigned int issued_lines = 0;
    for (int row_base = future_block_idx * kOProjRowsPerWave;
         row_base < kNosplitSpecializedK2048;
         row_base += grid_blocks * kOProjRowsPerWave) {
        for (int item_idx = threadIdx.x; item_idx < items_per_wave; item_idx += blockDim.x) {
            if (read_globaltimer() >= deadline_gt) {
                return issued_lines;
            }
            const int row_offset = item_idx / row_lines;
            const int line_idx = item_idx - row_offset * row_lines;
            const int row_idx = row_base + row_offset;
            if (row_idx >= kNosplitSpecializedK2048) {
                continue;
            }
            const char* row_bytes = reinterpret_cast<const char*>(weight) +
                static_cast<int64_t>(row_idx) * row_stride_bytes;
            prefetch_global_l2(row_bytes + line_idx * kPrefetchLineBytes);
            ++issued_lines;
        }
    }
    return issued_lines;
}

inline int getenv_int_or(const char* name, int default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value) {
        return default_value;
    }
    return static_cast<int>(parsed);
}

__device__ __forceinline__ void accumulate_hfma2_uint4(
    const uint4& x_vec, const uint4& w_vec, __half2& acc) {
    const __half2* xh = reinterpret_cast<const __half2*>(&x_vec);
    const __half2* wh = reinterpret_cast<const __half2*>(&w_vec);
    acc = __hfma2(xh[0], wh[0], acc);
    acc = __hfma2(xh[1], wh[1], acc);
    acc = __hfma2(xh[2], wh[2], acc);
    acc = __hfma2(xh[3], wh[3], acc);
}

__device__ __forceinline__ float warp_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ void cp_async_cg_16(void* dst, const void* src) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;"
        :
        : "r"(static_cast<unsigned>(__cvta_generic_to_shared(dst))),
          "l"(src)
        : "memory");
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;");
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
}

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

__device__ __forceinline__ void monotonic_counter_arrive(
    unsigned int* counter) {
    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        atomicAdd(counter, 1u);
    }
    __syncthreads();
}

__device__ __forceinline__ void monotonic_counter_wait_ge(
    const unsigned int* counter,
    unsigned int target) {
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile const unsigned int* vcounter =
            reinterpret_cast<volatile const unsigned int*>(counter);
        while (*vcounter < target) {
        }
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Non-profile kernel
// ---------------------------------------------------------------------------

__global__ void fused_gateup_silu_downproj_gridsync_k2048_k6144_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ gate_up_weight,
    const at::Half* __restrict__ down_proj_weight,
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

    for (int hidden_base = blockIdx.x * kRowsPerBlockAsync; hidden_base < 6144;
         hidden_base += gridDim.x * kRowsPerBlockAsync) {
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

    // Hardware grid sync
    cg::this_grid().sync();

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    const int row_base = blockIdx.x * kRowsPerBlockAsync;
    const int row = row_base + warp;

    for (int vec_idx = threadIdx.x; vec_idx < kAsyncVecsPerTileK6144;
         vec_idx += kRow8Threads) {
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
                 vec_idx += kRow8Threads) {
                if constexpr (kUseCpAsync) {
                    cp_async_cg_16(&x_tile_sh[stage ^ 1][vec_idx],
                                   x_next_vec + vec_idx);
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
            for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144;
                 vec_idx += 64) {
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

// ---------------------------------------------------------------------------
// Profile kernel (template)
// ---------------------------------------------------------------------------

template <bool kProfile>
__global__ void fused_gateup_silu_downproj_gridsync_profile_k2048_k6144_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ gate_up_weight,
    const at::Half* __restrict__ down_proj_weight,
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

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kSoftsyncProfileSlots +
                    SS_PROF_KERNEL_START] = read_globaltimer();
        }
    }

    if (threadIdx.x < vecs_k2048 / 2) {
        x_small_vec[threadIdx.x] = x_global_vec[threadIdx.x];
        x_small_vec[threadIdx.x + vecs_k2048 / 2] =
            x_global_vec[threadIdx.x + vecs_k2048 / 2];
    }
    __syncthreads();

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kSoftsyncProfileSlots +
                    SS_PROF_X_LOAD_DONE] = read_globaltimer();
        }
    }

    for (int hidden_base = blockIdx.x * kRowsPerBlockAsync; hidden_base < 6144;
         hidden_base += gridDim.x * kRowsPerBlockAsync) {
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

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kSoftsyncProfileSlots +
                    SS_PROF_GATE_UP_COMPUTE_DONE] = read_globaltimer();
            profile[static_cast<int64_t>(blockIdx.x) * kSoftsyncProfileSlots +
                    SS_PROF_BARRIER_START] = read_globaltimer();
        }
    }

    cg::this_grid().sync();

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kSoftsyncProfileSlots +
                    SS_PROF_BARRIER_DONE] = read_globaltimer();
        }
    }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    const int row_base = blockIdx.x * kRowsPerBlockAsync;
    const int row = row_base + warp;

    for (int vec_idx = threadIdx.x; vec_idx < kAsyncVecsPerTileK6144;
         vec_idx += kRow8Threads) {
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

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kSoftsyncProfileSlots +
                    SS_PROF_HIDDEN_LOAD_DONE] = read_globaltimer();
        }
    }

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
                 vec_idx += kRow8Threads) {
                if constexpr (kUseCpAsync) {
                    cp_async_cg_16(&x_tile_sh[stage ^ 1][vec_idx],
                                   x_next_vec + vec_idx);
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
            for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144;
                 vec_idx += 64) {
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

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kSoftsyncProfileSlots +
                    SS_PROF_KERNEL_DONE] = read_globaltimer();
        }
    }
}

// ---------------------------------------------------------------------------
// Persistent kernel: 1 block per SM, 1024 threads (32 warps)
// ---------------------------------------------------------------------------

constexpr int kPersistentThreads = 1024;
constexpr int kPersistentWarps   = 32;  // 1024 / 32

__launch_bounds__(1024, 1)
__global__ void fused_post_attn_mlp_persistent_k2048_k6144_fwd_kernel(
    const at::Half* __restrict__ residual,
    const at::Half* __restrict__ attn_output,
    const at::Half* __restrict__ norm_weight,
    const at::Half* __restrict__ gate_up_weight,
    const at::Half* __restrict__ down_proj_weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ hidden_out,
    at::Half* __restrict__ out,
    float eps,
    bool has_bias,
    int64_t gate_up_stride_wn,
    int64_t down_proj_stride_wn) {
    __shared__ __align__(16) __half residual_sh[kNosplitSpecializedK2048];
    __shared__ __align__(16) __half x_small_sh[kNosplitSpecializedK2048];
    __shared__ float warp_sums[kPersistentWarps];
    __shared__ float inv_rms_sh;
    __shared__ __align__(16) uint4 x_tile_sh[2][kAsyncVecsPerTileK6144];

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

    float thread_sum_sq = 0.0f;
    for (int idx = threadIdx.x; idx < kNosplitSpecializedK2048; idx += kPersistentThreads) {
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
        float block_sum_sq = warp_sums[lane];
        block_sum_sq = warp_sum(block_sum_sq);
        if (lane == 0) {
            inv_rms_sh = rsqrtf(block_sum_sq / static_cast<float>(kNosplitSpecializedK2048) + eps);
        }
    }
    __syncthreads();

    const float inv_rms = inv_rms_sh;
    for (int idx = threadIdx.x; idx < kNosplitSpecializedK2048; idx += kPersistentThreads) {
        const float summed_f = __half2float(residual_sh[idx]);
        const float scale_f = __half2float(norm_half[idx]);
        x_small_sh[idx] = __float2half(summed_f * inv_rms * scale_f);
    }
    __syncthreads();

    // --- Phase 1: gate_up GEMV + SiLU fusion (grid-cycle) ---
    for (int hidden_base = blockIdx.x * kPersistentWarps;
         hidden_base < 6144;
         hidden_base += gridDim.x * kPersistentWarps) {
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

    // --- Hardware grid sync ---
    cg::this_grid().sync();

    // --- Phase 2: down_proj GEMV (hidden[6144] -> out[2048]) + residual epilogue ---
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    const int row = blockIdx.x * kPersistentWarps + warp;

    for (int vec_idx = threadIdx.x; vec_idx < kAsyncVecsPerTileK6144;
         vec_idx += kPersistentThreads) {
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
                 vec_idx += kPersistentThreads) {
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
}

__launch_bounds__(1024, 1)
__global__ void fused_o_proj_post_attn_mlp_persistent_k2048_k6144_fwd_kernel(
    const at::Half* __restrict__ residual,
    const at::Half* __restrict__ attn_input,
    const at::Half* __restrict__ o_proj_weight,
    const at::Half* __restrict__ o_proj_bias,
    const at::Half* __restrict__ norm_weight,
    const at::Half* __restrict__ gate_up_weight,
    const at::Half* __restrict__ down_proj_weight,
    const at::Half* __restrict__ down_proj_bias,
    at::Half* __restrict__ hidden_out,
    at::Half* __restrict__ out,
    float eps,
    bool has_o_proj_bias,
    bool has_down_proj_bias,
    int64_t o_proj_stride_wn,
    int64_t gate_up_stride_wn,
    int64_t down_proj_stride_wn) {
    __shared__ __align__(16) __half residual_sh[kNosplitSpecializedK2048];
    __shared__ __align__(16) __half x_small_sh[kNosplitSpecializedK2048];
    __shared__ float warp_sums[kPersistentWarps];
    __shared__ float inv_rms_sh;
    __shared__ __align__(16) uint4 x_tile_sh[2][kAsyncVecsPerTileK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const __half* residual_half = reinterpret_cast<const __half*>(residual);
    const __half* attn_half = reinterpret_cast<const __half*>(attn_input);
    const __half* o_w_half = reinterpret_cast<const __half*>(o_proj_weight);
    const __half* o_bias_half = reinterpret_cast<const __half*>(o_proj_bias);
    const __half* norm_half = reinterpret_cast<const __half*>(norm_weight);
    const __half* gate_w_half = reinterpret_cast<const __half*>(gate_up_weight);
    const __half* down_w_half = reinterpret_cast<const __half*>(down_proj_weight);
    const __half* down_bias_half = reinterpret_cast<const __half*>(down_proj_bias);
    __half* hidden_half = reinterpret_cast<__half*>(hidden_out);
    __half* out_half = reinterpret_cast<__half*>(out);
    uint4* x_small_vec = reinterpret_cast<uint4*>(x_small_sh);
    constexpr int vecs_k2048 = kNosplitSpecializedK2048 / kHalfPerVec128;  // 256

    for (int idx = threadIdx.x; idx < vecs_k2048; idx += kPersistentThreads) {
        x_small_vec[idx] = reinterpret_cast<const uint4*>(attn_half)[idx];
    }
    __syncthreads();

    // Phase 0: o_proj GEMV. Materialize to hidden_out[0:2048] as scratch so
    // every CTA can consume the same post-attention residual after grid-sync.
    for (int out_base = blockIdx.x * kPersistentWarps;
         out_base < kNosplitSpecializedK2048;
         out_base += gridDim.x * kPersistentWarps) {
        const int out_idx = out_base + warp;
        if (out_idx < kNosplitSpecializedK2048) {
            const uint4* o_row_vec = reinterpret_cast<const uint4*>(
                o_w_half + out_idx * o_proj_stride_wn);
            __half2 acc0 = __float2half2_rn(0.0f);
            __half2 acc1 = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane; vec_idx < vecs_k2048; vec_idx += 64) {
                const uint4 x_vec0 = x_small_vec[vec_idx];
                const uint4 x_vec1 = x_small_vec[vec_idx + 32];
                accumulate_hfma2_uint4(x_vec0, o_row_vec[vec_idx], acc0);
                accumulate_hfma2_uint4(x_vec1, o_row_vec[vec_idx + 32], acc1);
            }

            const float2 f0 = __half22float2(acc0);
            const float2 f1 = __half22float2(acc1);
            float sum = f0.x + f0.y + f1.x + f1.y;
            sum = warp_sum(sum);
            if (lane == 0) {
                if (has_o_proj_bias) {
                    sum += __half2float(o_bias_half[out_idx]);
                }
                hidden_half[out_idx] = __float2half(sum);
            }
        }
    }

    cg::this_grid().sync();

    // Phase 1: post-attn residual add + RMSNorm.
    float thread_sum_sq = 0.0f;
    for (int idx = threadIdx.x; idx < kNosplitSpecializedK2048; idx += kPersistentThreads) {
        const float summed_f =
            __half2float(residual_half[idx]) + __half2float(hidden_half[idx]);
        residual_sh[idx] = __float2half(summed_f);
        thread_sum_sq += summed_f * summed_f;
    }

    thread_sum_sq = warp_sum(thread_sum_sq);
    if (lane == 0) {
        warp_sums[warp] = thread_sum_sq;
    }
    __syncthreads();

    if (warp == 0) {
        float block_sum_sq = lane < kPersistentWarps ? warp_sums[lane] : 0.0f;
        block_sum_sq = warp_sum(block_sum_sq);
        if (lane == 0) {
            inv_rms_sh = rsqrtf(
                block_sum_sq / static_cast<float>(kNosplitSpecializedK2048) + eps);
        }
    }
    __syncthreads();

    const float inv_rms = inv_rms_sh;
    for (int idx = threadIdx.x; idx < kNosplitSpecializedK2048; idx += kPersistentThreads) {
        const float summed_f = __half2float(residual_sh[idx]);
        const float scale_f = __half2float(norm_half[idx]);
        x_small_sh[idx] = __float2half(summed_f * inv_rms * scale_f);
    }
    __syncthreads();

    // Phase 2: gate_up GEMV + SiLU fusion.
    for (int hidden_base = blockIdx.x * kPersistentWarps;
         hidden_base < 6144;
         hidden_base += gridDim.x * kPersistentWarps) {
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

    cg::this_grid().sync();

    // Phase 3: down_proj GEMV from hidden_out[6144] + post-attn residual epilogue.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    const int row = blockIdx.x * kPersistentWarps + warp;

    for (int vec_idx = threadIdx.x; vec_idx < kAsyncVecsPerTileK6144;
         vec_idx += kPersistentThreads) {
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
                 vec_idx += kPersistentThreads) {
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
            if (has_down_proj_bias) {
                result += __half2float(down_bias_half[row]);
            }
            out_half[row] = __float2half(result);
        }
    }
}

// ---- Profile variant of fused_o_proj_post_attn_mlp_persistent ----
template <bool kProfile = false>
__launch_bounds__(1024, 1)
__global__ void fused_o_proj_post_attn_mlp_persistent_profile_k2048_k6144_fwd_kernel(
    const at::Half* __restrict__ residual,
    const at::Half* __restrict__ attn_input,
    const at::Half* __restrict__ o_proj_weight,
    const at::Half* __restrict__ o_proj_bias,
    const at::Half* __restrict__ norm_weight,
    const at::Half* __restrict__ gate_up_weight,
    const at::Half* __restrict__ down_proj_weight,
    const at::Half* __restrict__ down_proj_bias,
    at::Half* __restrict__ hidden_out,
    at::Half* __restrict__ out,
    uint64_t* __restrict__ profile,
    float eps,
    bool has_o_proj_bias,
    bool has_down_proj_bias,
    int64_t o_proj_stride_wn,
    int64_t gate_up_stride_wn,
    int64_t down_proj_stride_wn,
    int gate_up_active_sms) {
    __shared__ __align__(16) __half residual_sh[kNosplitSpecializedK2048];
    __shared__ __align__(16) __half x_small_sh[kNosplitSpecializedK2048];
    __shared__ float warp_sums[kPersistentWarps];
    __shared__ float inv_rms_sh;
    __shared__ __align__(16) uint4 x_tile_sh[2][kAsyncVecsPerTileK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const __half* residual_half = reinterpret_cast<const __half*>(residual);
    const __half* attn_half = reinterpret_cast<const __half*>(attn_input);
    const __half* o_w_half = reinterpret_cast<const __half*>(o_proj_weight);
    const __half* o_bias_half = reinterpret_cast<const __half*>(o_proj_bias);
    const __half* norm_half = reinterpret_cast<const __half*>(norm_weight);
    const __half* gate_w_half = reinterpret_cast<const __half*>(gate_up_weight);
    const __half* down_w_half = reinterpret_cast<const __half*>(down_proj_weight);
    const __half* down_bias_half = reinterpret_cast<const __half*>(down_proj_bias);
    __half* hidden_half = reinterpret_cast<__half*>(hidden_out);
    __half* out_half = reinterpret_cast<__half*>(out);
    uint4* x_small_vec = reinterpret_cast<uint4*>(x_small_sh);
    constexpr int vecs_k2048 = kNosplitSpecializedK2048 / kHalfPerVec128;  // 256

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                    OP_GT_START] = read_globaltimer();
            profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                    OP_KERNEL_START] = read_clock64();
            unsigned int smid;
            asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
            profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                    OP_SM_ID] = static_cast<uint64_t>(smid);
        }
    }

    for (int idx = threadIdx.x; idx < vecs_k2048; idx += kPersistentThreads) {
        x_small_vec[idx] = reinterpret_cast<const uint4*>(attn_half)[idx];
    }
    __syncthreads();

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                    OP_X_LOAD_DONE] = read_clock64();
        }
    }

    // Phase 0: o_proj GEMV
    for (int out_base = blockIdx.x * kPersistentWarps;
         out_base < kNosplitSpecializedK2048;
         out_base += gridDim.x * kPersistentWarps) {
        const int out_idx = out_base + warp;
        if (out_idx < kNosplitSpecializedK2048) {
            const uint4* o_row_vec = reinterpret_cast<const uint4*>(
                o_w_half + out_idx * o_proj_stride_wn);
            __half2 acc0 = __float2half2_rn(0.0f);
            __half2 acc1 = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane; vec_idx < vecs_k2048; vec_idx += 64) {
                const uint4 x_vec0 = x_small_vec[vec_idx];
                const uint4 x_vec1 = x_small_vec[vec_idx + 32];
                accumulate_hfma2_uint4(x_vec0, o_row_vec[vec_idx], acc0);
                accumulate_hfma2_uint4(x_vec1, o_row_vec[vec_idx + 32], acc1);
            }

            const float2 f0 = __half22float2(acc0);
            const float2 f1 = __half22float2(acc1);
            float sum = f0.x + f0.y + f1.x + f1.y;
            sum = warp_sum(sum);
            if (lane == 0) {
                if (has_o_proj_bias) {
                    sum += __half2float(o_bias_half[out_idx]);
                }
                hidden_half[out_idx] = __float2half(sum);
            }
        }
    }

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                    OP_OPROJ_DONE] = read_clock64();
        }
    }

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                    OP_GRIDSYNC1_START] = read_clock64();
        }
    }

    cg::this_grid().sync();

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                    OP_GRIDSYNC1_DONE] = read_clock64();
        }
    }

    // Phase 1: post-attn residual add + RMSNorm
    float thread_sum_sq = 0.0f;
    for (int idx = threadIdx.x; idx < kNosplitSpecializedK2048; idx += kPersistentThreads) {
        const float summed_f =
            __half2float(residual_half[idx]) + __half2float(hidden_half[idx]);
        residual_sh[idx] = __float2half(summed_f);
        thread_sum_sq += summed_f * summed_f;
    }

    thread_sum_sq = warp_sum(thread_sum_sq);
    if (lane == 0) {
        warp_sums[warp] = thread_sum_sq;
    }
    __syncthreads();

    if (warp == 0) {
        float block_sum_sq = lane < kPersistentWarps ? warp_sums[lane] : 0.0f;
        block_sum_sq = warp_sum(block_sum_sq);
        if (lane == 0) {
            inv_rms_sh = rsqrtf(
                block_sum_sq / static_cast<float>(kNosplitSpecializedK2048) + eps);
        }
    }
    __syncthreads();

    const float inv_rms = inv_rms_sh;
    for (int idx = threadIdx.x; idx < kNosplitSpecializedK2048; idx += kPersistentThreads) {
        const float summed_f = __half2float(residual_sh[idx]);
        const float scale_f = __half2float(norm_half[idx]);
        x_small_sh[idx] = __float2half(summed_f * inv_rms * scale_f);
    }
    __syncthreads();

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                    OP_RESNORM_DONE] = read_clock64();
        }
    }

    // Phase 2: gate_up GEMV + SiLU fusion
    // gate_up_active_sms <= 0 means use all SMs (baseline).
    int gate_up_iter = 0;
    const int gu_sms = (gate_up_active_sms > 0 && gate_up_active_sms < static_cast<int>(gridDim.x))
                        ? gate_up_active_sms : static_cast<int>(gridDim.x);
    if (static_cast<int>(blockIdx.x) < gu_sms) {
        for (int hidden_base = blockIdx.x * kPersistentWarps;
             hidden_base < 6144;
             hidden_base += gu_sms * kPersistentWarps) {
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
            if constexpr (kProfile) {
                if (threadIdx.x == 0 && profile != nullptr) {
                    profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                            OP_GATEUP_ITER0_DONE + gate_up_iter] = read_clock64();
                }
            }
            ++gate_up_iter;
        }
    }

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                    OP_GRIDSYNC2_START] = read_clock64();
        }
    }

    cg::this_grid().sync();

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                    OP_GRIDSYNC2_DONE] = read_clock64();
        }
    }

    // Phase 3: down_proj GEMV
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    const int row = blockIdx.x * kPersistentWarps + warp;

    for (int vec_idx = threadIdx.x; vec_idx < kAsyncVecsPerTileK6144;
         vec_idx += kPersistentThreads) {
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

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                    OP_DOWN_HIDDEN_LOADED] = read_clock64();
        }
    }

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
                 vec_idx += kPersistentThreads) {
                if constexpr (kUseCpAsync) {
                    cp_async_cg_16(&x_tile_sh[stage ^ 1][vec_idx],
                                   x_next_vec + vec_idx);
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
            for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144;
                 vec_idx += 64) {
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
            if (has_down_proj_bias) {
                result += __half2float(down_bias_half[row]);
            }
            out_half[row] = __float2half(result);
        }
    }

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                    OP_DOWN_COMPUTE_DONE] = read_clock64();
            profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                    OP_KERNEL_DONE] = read_clock64();
            profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                    OP_GT_END] = read_globaltimer();
        }
    }
}

__launch_bounds__(1024, 1)
__global__ void fused_o_proj_post_attn_mlp_softsync_k2048_k6144_fwd_kernel(
    const at::Half* __restrict__ residual,
    const at::Half* __restrict__ attn_input,
    const at::Half* __restrict__ o_proj_weight,
    const at::Half* __restrict__ o_proj_bias,
    const at::Half* __restrict__ norm_weight,
    const at::Half* __restrict__ gate_up_weight,
    const at::Half* __restrict__ down_proj_weight,
    const at::Half* __restrict__ down_proj_bias,
    unsigned int* __restrict__ barrier_count,
    unsigned int* __restrict__ barrier_generation,
    at::Half* __restrict__ hidden_out,
    at::Half* __restrict__ out,
    float eps,
    bool has_o_proj_bias,
    bool has_down_proj_bias,
    int64_t o_proj_stride_wn,
    int64_t gate_up_stride_wn,
    int64_t down_proj_stride_wn,
    int gate_up_active_sms) {
    __shared__ __align__(16) __half residual_sh[kNosplitSpecializedK2048];
    __shared__ __align__(16) __half x_small_sh[kNosplitSpecializedK2048];
    __shared__ float warp_sums[kPersistentWarps];
    __shared__ float inv_rms_sh;
    __shared__ __align__(16) uint4 x_tile_sh[2][kAsyncVecsPerTileK6144];
    __shared__ unsigned int local_gen_sh;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const __half* residual_half = reinterpret_cast<const __half*>(residual);
    const __half* attn_half = reinterpret_cast<const __half*>(attn_input);
    const __half* o_w_half = reinterpret_cast<const __half*>(o_proj_weight);
    const __half* o_bias_half = reinterpret_cast<const __half*>(o_proj_bias);
    const __half* norm_half = reinterpret_cast<const __half*>(norm_weight);
    const __half* gate_w_half = reinterpret_cast<const __half*>(gate_up_weight);
    const __half* down_w_half = reinterpret_cast<const __half*>(down_proj_weight);
    const __half* down_bias_half = reinterpret_cast<const __half*>(down_proj_bias);
    __half* hidden_half = reinterpret_cast<__half*>(hidden_out);
    __half* out_half = reinterpret_cast<__half*>(out);
    uint4* x_small_vec = reinterpret_cast<uint4*>(x_small_sh);
    constexpr int vecs_k2048 = kNosplitSpecializedK2048 / kHalfPerVec128;  // 256

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

    for (int idx = threadIdx.x; idx < vecs_k2048; idx += kPersistentThreads) {
        x_small_vec[idx] = reinterpret_cast<const uint4*>(attn_half)[idx];
    }
    __syncthreads();

    // Phase 0: o_proj GEMV. Materialize to hidden_out[0:2048] as scratch so
    // every CTA can consume the same post-attention residual after sync.
    for (int out_base = blockIdx.x * kPersistentWarps;
         out_base < kNosplitSpecializedK2048;
         out_base += gridDim.x * kPersistentWarps) {
        const int out_idx = out_base + warp;
        if (out_idx < kNosplitSpecializedK2048) {
            const uint4* o_row_vec = reinterpret_cast<const uint4*>(
                o_w_half + out_idx * o_proj_stride_wn);
            __half2 acc0 = __float2half2_rn(0.0f);
            __half2 acc1 = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane; vec_idx < vecs_k2048; vec_idx += 64) {
                const uint4 x_vec0 = x_small_vec[vec_idx];
                const uint4 x_vec1 = x_small_vec[vec_idx + 32];
                accumulate_hfma2_uint4(x_vec0, o_row_vec[vec_idx], acc0);
                accumulate_hfma2_uint4(x_vec1, o_row_vec[vec_idx + 32], acc1);
            }

            const float2 f0 = __half22float2(acc0);
            const float2 f1 = __half22float2(acc1);
            float sum = f0.x + f0.y + f1.x + f1.y;
            sum = warp_sum(sum);
            if (lane == 0) {
                if (has_o_proj_bias) {
                    sum += __half2float(o_bias_half[out_idx]);
                }
                hidden_half[out_idx] = __float2half(sum);
            }
        }
    }

    grid_sync.sync();

    // Phase 1: post-attn residual add + RMSNorm.
    float thread_sum_sq = 0.0f;
    for (int idx = threadIdx.x; idx < kNosplitSpecializedK2048; idx += kPersistentThreads) {
        const float summed_f =
            __half2float(residual_half[idx]) + __half2float(hidden_half[idx]);
        residual_sh[idx] = __float2half(summed_f);
        thread_sum_sq += summed_f * summed_f;
    }

    thread_sum_sq = warp_sum(thread_sum_sq);
    if (lane == 0) {
        warp_sums[warp] = thread_sum_sq;
    }
    __syncthreads();

    if (warp == 0) {
        float block_sum_sq = lane < kPersistentWarps ? warp_sums[lane] : 0.0f;
        block_sum_sq = warp_sum(block_sum_sq);
        if (lane == 0) {
            inv_rms_sh = rsqrtf(
                block_sum_sq / static_cast<float>(kNosplitSpecializedK2048) + eps);
        }
    }
    __syncthreads();

    const float inv_rms = inv_rms_sh;
    for (int idx = threadIdx.x; idx < kNosplitSpecializedK2048; idx += kPersistentThreads) {
        const float summed_f = __half2float(residual_sh[idx]);
        const float scale_f = __half2float(norm_half[idx]);
        x_small_sh[idx] = __float2half(summed_f * inv_rms * scale_f);
    }
    __syncthreads();

    // Phase 2: gate_up GEMV + SiLU fusion.
    const int gu_sms = (gate_up_active_sms > 0 && gate_up_active_sms < static_cast<int>(gridDim.x))
        ? gate_up_active_sms : static_cast<int>(gridDim.x);
    if (static_cast<int>(blockIdx.x) < gu_sms) {
        for (int hidden_base = blockIdx.x * kPersistentWarps;
             hidden_base < 6144;
             hidden_base += gu_sms * kPersistentWarps) {
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
    }

    grid_sync.sync();

    // Phase 3: down_proj GEMV from hidden_out[6144] + post-attn residual epilogue.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    const int row = blockIdx.x * kPersistentWarps + warp;

    for (int vec_idx = threadIdx.x; vec_idx < kAsyncVecsPerTileK6144;
         vec_idx += kPersistentThreads) {
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
                 vec_idx += kPersistentThreads) {
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
            if (has_down_proj_bias) {
                result += __half2float(down_bias_half[row]);
            }
            out_half[row] = __float2half(result);
        }
    }
}

template <bool kProfile = false>
__launch_bounds__(1024, 1)
__global__ void fused_o_proj_post_attn_mlp_softsync_profile_k2048_k6144_fwd_kernel(
    const at::Half* __restrict__ residual,
    const at::Half* __restrict__ attn_input,
    const at::Half* __restrict__ o_proj_weight,
    const at::Half* __restrict__ o_proj_bias,
    const at::Half* __restrict__ norm_weight,
    const at::Half* __restrict__ gate_up_weight,
    const at::Half* __restrict__ down_proj_weight,
    const at::Half* __restrict__ down_proj_bias,
    unsigned int* __restrict__ barrier_count,
    unsigned int* __restrict__ barrier_generation,
    at::Half* __restrict__ hidden_out,
    at::Half* __restrict__ out,
    uint64_t* __restrict__ profile,
    float eps,
    bool has_o_proj_bias,
    bool has_down_proj_bias,
    int64_t o_proj_stride_wn,
    int64_t gate_up_stride_wn,
    int64_t down_proj_stride_wn,
    int gate_up_active_sms) {
    __shared__ __align__(16) __half residual_sh[kNosplitSpecializedK2048];
    __shared__ __align__(16) __half x_small_sh[kNosplitSpecializedK2048];
    __shared__ float warp_sums[kPersistentWarps];
    __shared__ float inv_rms_sh;
    __shared__ __align__(16) uint4 x_tile_sh[2][kAsyncVecsPerTileK6144];
    __shared__ unsigned int local_gen_sh;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const __half* residual_half = reinterpret_cast<const __half*>(residual);
    const __half* attn_half = reinterpret_cast<const __half*>(attn_input);
    const __half* o_w_half = reinterpret_cast<const __half*>(o_proj_weight);
    const __half* o_bias_half = reinterpret_cast<const __half*>(o_proj_bias);
    const __half* norm_half = reinterpret_cast<const __half*>(norm_weight);
    const __half* gate_w_half = reinterpret_cast<const __half*>(gate_up_weight);
    const __half* down_w_half = reinterpret_cast<const __half*>(down_proj_weight);
    const __half* down_bias_half = reinterpret_cast<const __half*>(down_proj_bias);
    __half* hidden_half = reinterpret_cast<__half*>(hidden_out);
    __half* out_half = reinterpret_cast<__half*>(out);
    uint4* x_small_vec = reinterpret_cast<uint4*>(x_small_sh);
    constexpr int vecs_k2048 = kNosplitSpecializedK2048 / kHalfPerVec128;  // 256

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

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                    OP_GT_START] = read_globaltimer();
            profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                    OP_KERNEL_START] = read_clock64();
            unsigned int smid;
            asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
            profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                    OP_SM_ID] = static_cast<uint64_t>(smid);
        }
    }

    for (int idx = threadIdx.x; idx < vecs_k2048; idx += kPersistentThreads) {
        x_small_vec[idx] = reinterpret_cast<const uint4*>(attn_half)[idx];
    }
    __syncthreads();

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                    OP_X_LOAD_DONE] = read_clock64();
        }
    }

    // Phase 0: o_proj GEMV
    for (int out_base = blockIdx.x * kPersistentWarps;
         out_base < kNosplitSpecializedK2048;
         out_base += gridDim.x * kPersistentWarps) {
        const int out_idx = out_base + warp;
        if (out_idx < kNosplitSpecializedK2048) {
            const uint4* o_row_vec = reinterpret_cast<const uint4*>(
                o_w_half + out_idx * o_proj_stride_wn);
            __half2 acc0 = __float2half2_rn(0.0f);
            __half2 acc1 = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane; vec_idx < vecs_k2048; vec_idx += 64) {
                const uint4 x_vec0 = x_small_vec[vec_idx];
                const uint4 x_vec1 = x_small_vec[vec_idx + 32];
                accumulate_hfma2_uint4(x_vec0, o_row_vec[vec_idx], acc0);
                accumulate_hfma2_uint4(x_vec1, o_row_vec[vec_idx + 32], acc1);
            }

            const float2 f0 = __half22float2(acc0);
            const float2 f1 = __half22float2(acc1);
            float sum = f0.x + f0.y + f1.x + f1.y;
            sum = warp_sum(sum);
            if (lane == 0) {
                if (has_o_proj_bias) {
                    sum += __half2float(o_bias_half[out_idx]);
                }
                hidden_half[out_idx] = __float2half(sum);
            }
        }
    }

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                    OP_OPROJ_DONE] = read_clock64();
            profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                    OP_GRIDSYNC1_START] = read_clock64();
        }
    }

    grid_sync.sync();

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                    OP_GRIDSYNC1_DONE] = read_clock64();
        }
    }

    // Phase 1: post-attn residual add + RMSNorm
    float thread_sum_sq = 0.0f;
    for (int idx = threadIdx.x; idx < kNosplitSpecializedK2048; idx += kPersistentThreads) {
        const float summed_f =
            __half2float(residual_half[idx]) + __half2float(hidden_half[idx]);
        residual_sh[idx] = __float2half(summed_f);
        thread_sum_sq += summed_f * summed_f;
    }

    thread_sum_sq = warp_sum(thread_sum_sq);
    if (lane == 0) {
        warp_sums[warp] = thread_sum_sq;
    }
    __syncthreads();

    if (warp == 0) {
        float block_sum_sq = lane < kPersistentWarps ? warp_sums[lane] : 0.0f;
        block_sum_sq = warp_sum(block_sum_sq);
        if (lane == 0) {
            inv_rms_sh = rsqrtf(
                block_sum_sq / static_cast<float>(kNosplitSpecializedK2048) + eps);
        }
    }
    __syncthreads();

    const float inv_rms = inv_rms_sh;
    for (int idx = threadIdx.x; idx < kNosplitSpecializedK2048; idx += kPersistentThreads) {
        const float summed_f = __half2float(residual_sh[idx]);
        const float scale_f = __half2float(norm_half[idx]);
        x_small_sh[idx] = __float2half(summed_f * inv_rms * scale_f);
    }
    __syncthreads();

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                    OP_RESNORM_DONE] = read_clock64();
        }
    }

    // Phase 2: gate_up GEMV + SiLU fusion
    int gate_up_iter = 0;
    const int gu_sms = (gate_up_active_sms > 0 && gate_up_active_sms < static_cast<int>(gridDim.x))
        ? gate_up_active_sms : static_cast<int>(gridDim.x);
    if (static_cast<int>(blockIdx.x) < gu_sms) {
        for (int hidden_base = blockIdx.x * kPersistentWarps;
             hidden_base < 6144;
             hidden_base += gu_sms * kPersistentWarps) {
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
            if constexpr (kProfile) {
                if (threadIdx.x == 0 && profile != nullptr) {
                    profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                            OP_GATEUP_ITER0_DONE + gate_up_iter] = read_clock64();
                }
            }
            ++gate_up_iter;
        }
    }

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                    OP_GRIDSYNC2_START] = read_clock64();
        }
    }

    grid_sync.sync();

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                    OP_GRIDSYNC2_DONE] = read_clock64();
        }
    }

    // Phase 3: down_proj GEMV
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    const int row = blockIdx.x * kPersistentWarps + warp;

    for (int vec_idx = threadIdx.x; vec_idx < kAsyncVecsPerTileK6144;
         vec_idx += kPersistentThreads) {
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

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                    OP_DOWN_HIDDEN_LOADED] = read_clock64();
        }
    }

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
                 vec_idx += kPersistentThreads) {
                if constexpr (kUseCpAsync) {
                    cp_async_cg_16(&x_tile_sh[stage ^ 1][vec_idx],
                                   x_next_vec + vec_idx);
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
            for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144;
                 vec_idx += 64) {
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
            if (has_down_proj_bias) {
                result += __half2float(down_bias_half[row]);
            }
            out_half[row] = __float2half(result);
        }
    }

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                    OP_DOWN_COMPUTE_DONE] = read_clock64();
            profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                    OP_KERNEL_DONE] = read_clock64();
            profile[static_cast<int64_t>(blockIdx.x) * kOProjMlpProfileSlots +
                    OP_GT_END] = read_globaltimer();
        }
    }
}

__launch_bounds__(1024, 1)
__global__ void fused_gateup_silu_downproj_persistent_k2048_k6144_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ gate_up_weight,
    const at::Half* __restrict__ down_proj_weight,
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
    constexpr int vecs_k2048 = kNosplitSpecializedK2048 / kHalfPerVec128;  // 256

    // --- Phase 1: Load x[2048] into shared memory ---
    if (threadIdx.x < vecs_k2048 / 2) {  // threads 0..127
        x_small_vec[threadIdx.x] = x_global_vec[threadIdx.x];
        x_small_vec[threadIdx.x + vecs_k2048 / 2] = x_global_vec[threadIdx.x + vecs_k2048 / 2];
    }
    __syncthreads();

    // --- Phase 1: gate_up GEMV + SiLU fusion (grid-cycle) ---
    for (int hidden_base = blockIdx.x * kPersistentWarps;
         hidden_base < 6144;
         hidden_base += gridDim.x * kPersistentWarps) {
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

    // --- Hardware grid sync ---
    cg::this_grid().sync();

    // --- Phase 2: down_proj GEMV (hidden[6144] -> out[2048]) ---
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    const int row = blockIdx.x * kPersistentWarps + warp;

    // Prefetch first x tile
    for (int vec_idx = threadIdx.x; vec_idx < kAsyncVecsPerTileK6144;
         vec_idx += kPersistentThreads) {
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

        // Prefetch next x tile (all threads participate)
        if (tile_idx + 1 < 6) {
            const int next_tile_start = tile_start + kNosplitAsyncTileK6144;
            const uint4* x_next_vec =
                reinterpret_cast<const uint4*>(hidden_half + next_tile_start);
            for (int vec_idx = threadIdx.x; vec_idx < kAsyncVecsPerTileK6144;
                 vec_idx += kPersistentThreads) {
                if constexpr (kUseCpAsync) {
                    cp_async_cg_16(&x_tile_sh[stage ^ 1][vec_idx],
                                   x_next_vec + vec_idx);
                } else {
                    x_tile_sh[stage ^ 1][vec_idx] = x_next_vec[vec_idx];
                }
            }
            if constexpr (kUseCpAsync) {
                cp_async_commit_group();
            }
        }

        // Compute (only active warps)
        if (row < 2048) {
            const uint4* x_sh_stage_vec = x_tile_sh[stage];
            const uint4* w_row_vec = reinterpret_cast<const uint4*>(
                down_w_half + row * down_proj_stride_wn + tile_start);
            __half2 h2_acc0 = __float2half2_rn(0.0f);
            __half2 h2_acc1 = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144;
                 vec_idx += 64) {
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

// ---------------------------------------------------------------------------
// Persistent profile kernel
// ---------------------------------------------------------------------------

template <bool kProfile>
__launch_bounds__(1024, 1)
__global__ void fused_gateup_silu_downproj_persistent_profile_k2048_k6144_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ gate_up_weight,
    const at::Half* __restrict__ down_proj_weight,
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

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kPersistentProfileSlots +
                    PP_KERNEL_START] = read_globaltimer();
        }
    }

    if (threadIdx.x < vecs_k2048 / 2) {
        x_small_vec[threadIdx.x] = x_global_vec[threadIdx.x];
        x_small_vec[threadIdx.x + vecs_k2048 / 2] =
            x_global_vec[threadIdx.x + vecs_k2048 / 2];
    }
    __syncthreads();

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kPersistentProfileSlots +
                    PP_X_LOAD_DONE] = read_globaltimer();
        }
    }

    // --- gate_up GEMV with per-iteration profiling ---
    int gate_up_iter = 0;
    for (int hidden_base = blockIdx.x * kPersistentWarps;
         hidden_base < 6144;
         hidden_base += gridDim.x * kPersistentWarps) {
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
        if constexpr (kProfile) {
            if (threadIdx.x == 0 && profile != nullptr) {
                profile[static_cast<int64_t>(blockIdx.x) * kPersistentProfileSlots +
                        PP_GATEUP_ITER0_DONE + gate_up_iter] = read_globaltimer();
            }
        }
        ++gate_up_iter;
    }

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kPersistentProfileSlots +
                    PP_GRIDSYNC_START] = read_globaltimer();
        }
    }

    cg::this_grid().sync();

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kPersistentProfileSlots +
                    PP_GRIDSYNC_DONE] = read_globaltimer();
        }
    }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    const int row = blockIdx.x * kPersistentWarps + warp;

    for (int vec_idx = threadIdx.x; vec_idx < kAsyncVecsPerTileK6144;
         vec_idx += kPersistentThreads) {
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

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kPersistentProfileSlots +
                    PP_DOWN_HIDDEN_LOADED] = read_globaltimer();
        }
    }

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
                 vec_idx += kPersistentThreads) {
                if constexpr (kUseCpAsync) {
                    cp_async_cg_16(&x_tile_sh[stage ^ 1][vec_idx],
                                   x_next_vec + vec_idx);
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
            for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144;
                 vec_idx += 64) {
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

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[static_cast<int64_t>(blockIdx.x) * kPersistentProfileSlots +
                    PP_DOWN_COMPUTE_DONE] = read_globaltimer();
            profile[static_cast<int64_t>(blockIdx.x) * kPersistentProfileSlots +
                    PP_KERNEL_DONE] = read_globaltimer();
        }
    }
}

// ---------------------------------------------------------------------------
// Persistent attention kernel: 1 block per SM, 1024 threads (32 warps)
// All 32 warps participate in attention.  Grid cycles over q-heads (not
// kv-heads) for maximum SM utilization.  Within each block, 32 warps split
// the KV sequence, then tree-reduce in shared memory (5 rounds for 32-way).
//
// Layout: 1 block per q-head, all 32 warps split KV for that head.
// Two blocks sharing a kv-group read the same K/V → L2 cache reuse.
// ---------------------------------------------------------------------------

constexpr int kAttnHeadDim      = 128;
constexpr int kAttnWarpSize     = 32;
constexpr int kAttnThreadGroup  = 4;
constexpr int kAttnGroupsPerWarp = kAttnWarpSize / kAttnThreadGroup;  // 8
constexpr int kAttnDimsPerThread = kAttnHeadDim / kAttnThreadGroup;   // 32
constexpr int kAttnTileSteps    = 8;
constexpr int kAttnTileN        = kAttnGroupsPerWarp * kAttnTileSteps; // 64
constexpr float kAttnNegLarge   = -1.0e20f;
constexpr int kAttnHalfHeadDim  = kAttnHeadDim / 2;

constexpr int kAttnGqaRatio       = 2;
constexpr int kAttnInternalSplits = kPersistentWarps;  // 32 (all warps)
constexpr int kFullLayerThreads = 512;
constexpr int kFullLayerWarps = kFullLayerThreads / kWarpSize;  // 16
constexpr int kFullLayerInternalSplits = kFullLayerWarps;
constexpr int kFullLayerQHeadSplitWarps = 12;

__device__ __forceinline__ __half rope_out_half(
    __half a,
    __half b,
    __half cos,
    __half sin,
    bool second_half) {
    const float af = __half2float(a);
    const float bf = __half2float(b);
    const float cf = __half2float(cos);
    const float sf = __half2float(sin);
    const float out = second_half ? (bf * cf + af * sf) : (af * cf - bf * sf);
    return __float2half_rn(out);
}

__device__ __forceinline__ void fused_rms_norm_rope_one_head(
    const __half* __restrict__ x,
    const __half* __restrict__ weight,
    const __half* __restrict__ cos,
    const __half* __restrict__ sin,
    __half* __restrict__ out,
    float eps,
    int lane) {
    float sum_sq = 0.0f;
    #pragma unroll
    for (int idx = lane; idx < kAttnHeadDim; idx += kWarpSize) {
        const float v = __half2float(x[idx]);
        sum_sq += v * v;
    }
    sum_sq = warp_sum(sum_sq);
    const float rstd = rsqrtf(
        __shfl_sync(0xffffffff, sum_sq / static_cast<float>(kAttnHeadDim) + eps, 0));

    for (int rk = lane; rk < kAttnHalfHeadDim; rk += kWarpSize) {
        const float n0f =
            __half2float(x[rk]) * rstd * __half2float(weight[rk]);
        const float n1f =
            __half2float(x[kAttnHalfHeadDim + rk]) * rstd
            * __half2float(weight[kAttnHalfHeadDim + rk]);
        const __half n0 = __float2half_rn(n0f);
        const __half n1 = __float2half_rn(n1f);
        const __half c = cos[rk];
        const __half s = sin[rk];
        out[rk] = rope_out_half(n0, n1, c, s, false);
        out[kAttnHalfHeadDim + rk] = rope_out_half(n0, n1, c, s, true);
    }
}

__launch_bounds__(1024, 1)
__global__ void decode_attn_persistent_fwd_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    const int64_t* __restrict__ cache_position,
    __half* __restrict__ out,
    int64_t q_stride_h,
    int64_t k_stride_h, int64_t k_stride_s,
    int64_t v_stride_h, int64_t v_stride_s,
    int64_t out_stride_h,
    float scale,
    int max_seq_len,
    int num_kv_heads) {

    // Shared memory for within-block tree-reduction of 32 warp partials
    __shared__ float smem_acc[kPersistentWarps][kAttnHeadDim];  // 16384B
    __shared__ float smem_m[kPersistentWarps];                  // 128B
    __shared__ float smem_l[kPersistentWarps];                  // 128B

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;     // 0..31
    const int lane = tid & 31;

    // Thread-group layout within each warp (same as v4)
    const int group_id   = lane / kAttnThreadGroup;    // 0..7
    const int group_lane = lane % kAttnThreadGroup;    // 0..3
    const int dim_offset = group_lane * kAttnDimsPerThread;  // 0,32,64,96

    const int valid_len = min(
        static_cast<int>(cache_position[0] + 1), max_seq_len);

    const int num_q_heads = num_kv_heads * kAttnGqaRatio;

    // ===== Grid-cycle over q-heads (all 32 warps on same q-head) ===========
    for (int q_head_idx = blockIdx.x; q_head_idx < num_q_heads;
         q_head_idx += gridDim.x) {

        const int kv_head_idx = q_head_idx / kAttnGqaRatio;

        // All 32 warps split the KV sequence
        const int chunk_size  = (valid_len + kAttnInternalSplits - 1) / kAttnInternalSplits;
        const int chunk_start = warp * chunk_size;
        const int chunk_end   = min(chunk_start + chunk_size, valid_len);

        // Load Q for this q-head
        float q_reg[kAttnDimsPerThread];
        {
            const __half* qp = q + q_head_idx * q_stride_h + dim_offset;
            #pragma unroll
            for (int i = 0; i < kAttnDimsPerThread; i += 8) {
                uint4 d = __ldg(reinterpret_cast<const uint4*>(qp + i));
                const __half* h = reinterpret_cast<const __half*>(&d);
                #pragma unroll
                for (int j = 0; j < 8; ++j) q_reg[i + j] = __half2float(h[j]);
            }
        }

        // Online softmax state
        float m_val = kAttnNegLarge;
        float l_val = 0.0f;
        float acc[kAttnDimsPerThread];
        #pragma unroll
        for (int i = 0; i < kAttnDimsPerThread; ++i) acc[i] = 0.0f;

        const __half* k_base = k_cache + kv_head_idx * k_stride_h;
        const __half* v_base = v_cache + kv_head_idx * v_stride_h;

        // Tiled two-pass attention over this warp's chunk
        if (chunk_start < chunk_end) {
            for (int tile_start = chunk_start; tile_start < chunk_end;
                 tile_start += kAttnTileN) {
                const int tile_end = min(tile_start + kAttnTileN, chunk_end);

                // Pass 1: Q·K scores
                float local_scores[kAttnTileSteps];
                float tile_max = kAttnNegLarge;

                #pragma unroll
                for (int step = 0; step < kAttnTileSteps; ++step) {
                    const int t = tile_start + step * kAttnGroupsPerWarp + group_id;
                    const bool valid = (t < tile_end);
                    float dot = 0.0f;
                    if (valid) {
                        const __half* kp = k_base + t * k_stride_s + dim_offset;
                        #pragma unroll
                        for (int i = 0; i < kAttnDimsPerThread; i += 8) {
                            uint4 d = __ldg(reinterpret_cast<const uint4*>(kp + i));
                            const __half* h = reinterpret_cast<const __half*>(&d);
                            #pragma unroll
                            for (int j = 0; j < 8; ++j)
                                dot += q_reg[i + j] * __half2float(h[j]);
                        }
                    }
                    dot += __shfl_xor_sync(0xffffffff, dot, 1);
                    dot += __shfl_xor_sync(0xffffffff, dot, 2);
                    float score = valid ? (dot * scale) : kAttnNegLarge;
                    local_scores[step] = score;
                    tile_max = fmaxf(tile_max, score);
                }

                // Cross-group max
                tile_max = fmaxf(tile_max,
                    __shfl_xor_sync(0xffffffff, tile_max, 4));
                tile_max = fmaxf(tile_max,
                    __shfl_xor_sync(0xffffffff, tile_max, 8));
                tile_max = fmaxf(tile_max,
                    __shfl_xor_sync(0xffffffff, tile_max, 16));

                // Rescale running state
                float new_m = fmaxf(m_val, tile_max);
                float alpha = expf(m_val - new_m);
                #pragma unroll
                for (int i = 0; i < kAttnDimsPerThread; ++i) acc[i] *= alpha;
                l_val *= alpha;
                m_val = new_m;

                // Pass 2: V accumulate
                #pragma unroll
                for (int step = 0; step < kAttnTileSteps; ++step) {
                    const int t = tile_start + step * kAttnGroupsPerWarp + group_id;
                    const bool valid = (t < tile_end);
                    if (valid) {
                        float p = expf(local_scores[step] - m_val);
                        l_val += p;
                        const __half* vp = v_base + t * v_stride_s + dim_offset;
                        #pragma unroll
                        for (int i = 0; i < kAttnDimsPerThread; i += 8) {
                            uint4 d = __ldg(reinterpret_cast<const uint4*>(vp + i));
                            const __half* h = reinterpret_cast<const __half*>(&d);
                            #pragma unroll
                            for (int j = 0; j < 8; ++j)
                                acc[i + j] += p * __half2float(h[j]);
                        }
                    }
                }
            }
        }
        // Empty chunk: initial values (m=-inf, l=0, acc=0) are correct

        // --- Intra-warp merge: 8 thread groups → 1 via shuffle ---
        #pragma unroll
        for (int xor_mask = kAttnThreadGroup; xor_mask < kAttnWarpSize;
             xor_mask <<= 1) {
            float pm = __shfl_xor_sync(0xffffffff, m_val, xor_mask);
            float pl = __shfl_xor_sync(0xffffffff, l_val, xor_mask);
            float nm  = fmaxf(m_val, pm);
            float s_me = expf(m_val - nm);
            float s_pa = expf(pm - nm);
            #pragma unroll
            for (int i = 0; i < kAttnDimsPerThread; ++i) {
                float pa_v = __shfl_xor_sync(0xffffffff, acc[i], xor_mask);
                acc[i] = acc[i] * s_me + pa_v * s_pa;
            }
            l_val = l_val * s_me + pl * s_pa;
            m_val = nm;
        }

        // --- Write per-warp partial to shared memory ---
        if (group_id == 0) {
            #pragma unroll
            for (int i = 0; i < kAttnDimsPerThread; ++i) {
                smem_acc[warp][dim_offset + i] = acc[i];
            }
            if (group_lane == 0) {
                smem_m[warp] = m_val;
                smem_l[warp] = l_val;
            }
        }
        __syncthreads();

        // --- Tree reduction: 32 → 1 in 5 rounds ---
        constexpr int kDimsPerLane = kAttnHeadDim / kAttnWarpSize;  // 4
        #pragma unroll
        for (int stride = kAttnInternalSplits / 2; stride >= 1; stride >>= 1) {
            if (warp < stride) {
                const int src = warp + stride;
                const float pm = smem_m[src];
                const float pl = smem_l[src];
                const float my_m = smem_m[warp];
                const float my_l = smem_l[warp];
                const float nm = fmaxf(my_m, pm);
                const float s_me = expf(my_m - nm);
                const float s_pa = expf(pm - nm);
                #pragma unroll
                for (int i = 0; i < kDimsPerLane; ++i) {
                    const float my_a = smem_acc[warp][lane * kDimsPerLane + i];
                    const float pa_v = smem_acc[src][lane * kDimsPerLane + i];
                    smem_acc[warp][lane * kDimsPerLane + i] = my_a * s_me + pa_v * s_pa;
                }
                if (lane == 0) {
                    smem_m[warp] = nm;
                    smem_l[warp] = my_l * s_me + pl * s_pa;
                }
            }
            __syncthreads();
        }

        // --- Warp 0 writes final output ---
        if (warp == 0) {
            const float inv_l = (smem_l[0] > 0.0f) ? (1.0f / smem_l[0]) : 0.0f;
            __half* op = out + q_head_idx * out_stride_h;
            #pragma unroll
            for (int i = 0; i < kDimsPerLane; ++i) {
                op[lane * kDimsPerLane + i] =
                    __float2half(smem_acc[0][lane * kDimsPerLane + i] * inv_l);
            }
        }
        __syncthreads();  // sync before next grid-cycle iteration
    }
}

template <bool kProfile>
__launch_bounds__(kFullLayerThreads, 1)
__global__ void fused_decode_full_layer_softsync_k2048_fwd_kernel(
    const at::Half* __restrict__ hidden_states,
    const at::Half* __restrict__ qkv_weight,
    const at::Half* __restrict__ input_layernorm_weight,
    const at::Half* __restrict__ q_norm_weight,
    const at::Half* __restrict__ k_norm_weight,
    const at::Half* __restrict__ cos,
    const at::Half* __restrict__ sin,
    at::Half* __restrict__ k_cache,
    at::Half* __restrict__ v_cache,
    const int64_t* __restrict__ cache_position,
    const at::Half* __restrict__ o_proj_weight,
    const at::Half* __restrict__ o_proj_bias,
    const at::Half* __restrict__ post_attn_norm_weight,
    const at::Half* __restrict__ gate_up_weight,
    const at::Half* __restrict__ down_proj_weight,
    const at::Half* __restrict__ down_proj_bias,
    unsigned int* __restrict__ barrier_count,
    unsigned int* __restrict__ barrier_generation,
    unsigned int* __restrict__ kv_ready_count,
    unsigned int* __restrict__ attn_done_count,
    float* __restrict__ attn_partial_acc,
    float* __restrict__ attn_partial_stats,
    int* __restrict__ attn_split_done_count,
    at::Half* __restrict__ qkv_out,
    at::Half* __restrict__ hidden_out,
    at::Half* __restrict__ out,
    uint64_t* __restrict__ profile,
    float eps,
    float attn_scale,
    bool has_o_proj_bias,
    bool has_down_proj_bias,
    int64_t qkv_stride_wn,
    int64_t k_cache_stride_h,
    int64_t k_cache_stride_s,
    int64_t v_cache_stride_h,
    int64_t v_cache_stride_s,
    int64_t o_proj_stride_wn,
    int64_t gate_up_stride_wn,
    int64_t down_proj_stride_wn,
    bool partial_attn_flags,
    bool o_proj_idle_only,
    int o_proj_prefetch_blocks,
    uint64_t o_proj_prefetch_start_delay_ns,
    uint64_t o_proj_prefetch_budget_ns,
    int max_seq_len,
    int num_kv_heads,
    bool use_attn_qhead_splitkv,
    int attn_qhead_split_k) {
    __shared__ __align__(16) __half residual_sh[kNosplitSpecializedK2048];
    __shared__ __align__(16) __half x_small_sh[kNosplitSpecializedK2048];
    __shared__ float warp_sums[kFullLayerWarps];
    __shared__ float inv_rms_sh;
    __shared__ __align__(16) uint4 x_tile_sh[2][kAsyncVecsPerTileK6144];
    __shared__ unsigned int local_gen_sh;
    __shared__ float smem_acc[kFullLayerWarps][kAttnHeadDim];
    __shared__ float smem_m[kFullLayerWarps];
    __shared__ float smem_l[kFullLayerWarps];
    __shared__ unsigned int attn_prefetch_lines_sh;
    __shared__ unsigned int kv_ready_target_sh;
    __shared__ unsigned int attn_done_target_sh;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int64_t profile_base =
        static_cast<int64_t>(blockIdx.x) * kFullLayerProfileSlots;

    const __half* residual_half = reinterpret_cast<const __half*>(hidden_states);
    const __half* qkv_w_half = reinterpret_cast<const __half*>(qkv_weight);
    const __half* input_norm_half = reinterpret_cast<const __half*>(input_layernorm_weight);
    const __half* q_norm_half = reinterpret_cast<const __half*>(q_norm_weight);
    const __half* k_norm_half = reinterpret_cast<const __half*>(k_norm_weight);
    const __half* cos_half = reinterpret_cast<const __half*>(cos);
    const __half* sin_half = reinterpret_cast<const __half*>(sin);
    __half* k_cache_half = reinterpret_cast<__half*>(k_cache);
    __half* v_cache_half = reinterpret_cast<__half*>(v_cache);
    const __half* o_w_half = reinterpret_cast<const __half*>(o_proj_weight);
    const __half* o_bias_half = reinterpret_cast<const __half*>(o_proj_bias);
    const __half* post_norm_half = reinterpret_cast<const __half*>(post_attn_norm_weight);
    const __half* gate_w_half = reinterpret_cast<const __half*>(gate_up_weight);
    const __half* down_w_half = reinterpret_cast<const __half*>(down_proj_weight);
    const __half* down_bias_half = reinterpret_cast<const __half*>(down_proj_bias);
    __half* qkv_half = reinterpret_cast<__half*>(qkv_out);
    __half* hidden_half = reinterpret_cast<__half*>(hidden_out);
    __half* out_half = reinterpret_cast<__half*>(out);
    uint4* x_small_vec = reinterpret_cast<uint4*>(x_small_sh);
    constexpr int vecs_k2048 = kNosplitSpecializedK2048 / kHalfPerVec128;  // 256
    constexpr int kQRows = 2048;
    constexpr int kKRows = 1024;
    constexpr int kVRows = 1024;
    constexpr int kQKVRows = kQRows + kKRows + kVRows;
    constexpr int kKOffset = kQRows;
    constexpr int kVOffset = kQRows + kKRows;
    const int num_q_heads = num_kv_heads * kAttnGqaRatio;
    const int kv_ready_blocks = min(static_cast<int>(gridDim.x), num_kv_heads);
    const int attn_block_limit = use_attn_qhead_splitkv
        ? num_q_heads * attn_qhead_split_k
        : num_q_heads;
    const int attn_blocks = min(static_cast<int>(gridDim.x), attn_block_limit);

    if (tid == 0) {
        local_gen_sh = barrier_generation[0];
        kv_ready_target_sh = kv_ready_count[0] + static_cast<unsigned int>(kv_ready_blocks);
        attn_done_target_sh = attn_done_count[0] + static_cast<unsigned int>(attn_blocks);
    }
    __syncthreads();

    AtomicGridSync grid_sync{
        barrier_count,
        barrier_generation,
        static_cast<unsigned int>(gridDim.x),
        &local_gen_sh,
    };

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[profile_base + FL_GT_START] = read_globaltimer();
            profile[profile_base + FL_KERNEL_START] = read_clock64();
            unsigned int smid;
            asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
            profile[profile_base + FL_SM_ID] = static_cast<uint64_t>(smid);
        }
    }

    // Phase 0: input RMSNorm on the decoder residual input.
    float thread_sum_sq = 0.0f;
    for (int idx = tid; idx < kNosplitSpecializedK2048; idx += kFullLayerThreads) {
        const float v = __half2float(residual_half[idx]);
        thread_sum_sq += v * v;
    }
    thread_sum_sq = warp_sum(thread_sum_sq);
    if (lane == 0) {
        warp_sums[warp] = thread_sum_sq;
    }
    __syncthreads();

    if (warp == 0) {
        float block_sum_sq = lane < kFullLayerWarps ? warp_sums[lane] : 0.0f;
        block_sum_sq = warp_sum(block_sum_sq);
        if (lane == 0) {
            inv_rms_sh = rsqrtf(
                block_sum_sq / static_cast<float>(kNosplitSpecializedK2048) + eps);
        }
    }
    __syncthreads();

    const float input_inv_rms = inv_rms_sh;
    for (int idx = tid; idx < kNosplitSpecializedK2048; idx += kFullLayerThreads) {
        const float v = __half2float(residual_half[idx]);
        const float scale_f = __half2float(input_norm_half[idx]);
        x_small_sh[idx] = __float2half(v * input_inv_rms * scale_f);
    }
    __syncthreads();

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[profile_base + FL_INPUT_RMSNORM_DONE] = read_clock64();
        }
    }

    // Phase 1: fused QKV GEMV into qkv_out[4096].
    for (int out_base = blockIdx.x * kFullLayerWarps;
         out_base < kQKVRows;
         out_base += gridDim.x * kFullLayerWarps) {
        const int out_idx = out_base + warp;
        if (out_idx < kQKVRows) {
            const uint4* qkv_row_vec = reinterpret_cast<const uint4*>(
                qkv_w_half + out_idx * qkv_stride_wn);
            __half2 acc0 = __float2half2_rn(0.0f);
            __half2 acc1 = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane; vec_idx < vecs_k2048; vec_idx += 64) {
                const uint4 x_vec0 = x_small_vec[vec_idx];
                const uint4 x_vec1 = x_small_vec[vec_idx + 32];
                accumulate_hfma2_uint4(x_vec0, qkv_row_vec[vec_idx], acc0);
                accumulate_hfma2_uint4(x_vec1, qkv_row_vec[vec_idx + 32], acc1);
            }

            const float2 f0 = __half22float2(acc0);
            const float2 f1 = __half22float2(acc1);
            float sum = f0.x + f0.y + f1.x + f1.y;
            sum = warp_sum(sum);
            if (lane == 0) {
                qkv_half[out_idx] = __float2half(sum);
            }
        }
    }

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[profile_base + FL_QKV_DONE] = read_clock64();
            profile[profile_base + FL_SYNC1_START] = read_clock64();
        }
    }
    grid_sync.sync();
    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[profile_base + FL_SYNC1_DONE] = read_clock64();
        }
    }

    // Phase 2: q_norm+rope, k_norm+rope->cache, v->cache.
    if (static_cast<int>(blockIdx.x) < num_kv_heads) {
        const int kv_h = static_cast<int>(blockIdx.x);
        const int64_t pos = cache_position[0];
        const __half* q0_in = qkv_half + (kv_h * kAttnGqaRatio + 0) * kAttnHeadDim;
        const __half* q1_in = qkv_half + (kv_h * kAttnGqaRatio + 1) * kAttnHeadDim;
        const __half* k_in = qkv_half + kKOffset + kv_h * kAttnHeadDim;
        const __half* v_in = qkv_half + kVOffset + kv_h * kAttnHeadDim;
        __half* q0_out = qkv_half + (kv_h * kAttnGqaRatio + 0) * kAttnHeadDim;
        __half* q1_out = qkv_half + (kv_h * kAttnGqaRatio + 1) * kAttnHeadDim;
        __half* k_dst =
            k_cache_half + kv_h * k_cache_stride_h + pos * k_cache_stride_s;
        __half* v_dst =
            v_cache_half + kv_h * v_cache_stride_h + pos * v_cache_stride_s;

        if (warp == 0) {
            fused_rms_norm_rope_one_head(
                k_in,
                k_norm_half,
                cos_half,
                sin_half,
                k_dst,
                eps,
                lane);
        } else if (warp == 1) {
            fused_rms_norm_rope_one_head(
                q0_in,
                q_norm_half,
                cos_half,
                sin_half,
                q0_out,
                eps,
                lane);
        } else if (warp == 2) {
            fused_rms_norm_rope_one_head(
                q1_in,
                q_norm_half,
                cos_half,
                sin_half,
                q1_out,
                eps,
                lane);
        } else if (warp == 3) {
            #pragma unroll
            for (int idx = lane; idx < kAttnHeadDim; idx += kWarpSize) {
                v_dst[idx] = v_in[idx];
            }
        }
    }

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[profile_base + FL_QKVCACHE_DONE] = read_clock64();
            profile[profile_base + FL_SYNC2_START] = read_clock64();
        }
    }
    if (partial_attn_flags) {
        if (static_cast<int>(blockIdx.x) < kv_ready_blocks) {
            monotonic_counter_arrive(kv_ready_count);
        }
        if (static_cast<int>(blockIdx.x) < attn_blocks) {
            monotonic_counter_wait_ge(kv_ready_count, kv_ready_target_sh);
        }
    } else {
        grid_sync.sync();
    }
    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[profile_base + FL_SYNC2_DONE] = read_clock64();
        }
    }

    // Phase 3: persistent attention. Reuse qkv_out[0:2048] as the attention output buffer.
    const int valid_len = min(
        static_cast<int>(cache_position[0] + 1), max_seq_len);
    const int group_id = lane / kAttnThreadGroup;
    const int group_lane = lane % kAttnThreadGroup;
    const int dim_offset = group_lane * kAttnDimsPerThread;
    const int idle_block_idx = static_cast<int>(blockIdx.x) - attn_blocks;
    const int o_proj_grid_blocks = o_proj_idle_only
        ? max(0, static_cast<int>(gridDim.x) - attn_blocks)
        : static_cast<int>(gridDim.x);
    const int o_proj_block_idx = o_proj_idle_only
        ? idle_block_idx
        : static_cast<int>(blockIdx.x);
    unsigned int thread_prefetch_lines = 0;

    if constexpr (kProfile) {
        if (tid == 0) {
            attn_prefetch_lines_sh = 0;
        }
        __syncthreads();
    }

    if (idle_block_idx >= 0
        && idle_block_idx < o_proj_prefetch_blocks
        && o_proj_block_idx >= 0
        && o_proj_grid_blocks > 0
        && o_proj_prefetch_budget_ns > 0) {
        const uint64_t prefetch_start_gt =
            read_globaltimer() + o_proj_prefetch_start_delay_ns;
        while (read_globaltimer() < prefetch_start_gt) {
        }
        const uint64_t prefetch_deadline_gt =
            prefetch_start_gt + o_proj_prefetch_budget_ns;
        thread_prefetch_lines = prefetch_oproj_rows_for_block_l2_until(
            o_w_half,
            o_proj_stride_wn,
            o_proj_grid_blocks,
            o_proj_block_idx,
            prefetch_deadline_gt);
        if constexpr (kProfile) {
            if (thread_prefetch_lines > 0) {
                atomicAdd(&attn_prefetch_lines_sh, thread_prefetch_lines);
            }
        }
    }

    if constexpr (kProfile) {
        __syncthreads();
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[profile_base + FL_ATTN_PREFETCH_LINES] =
                static_cast<uint64_t>(attn_prefetch_lines_sh);
        }
    }

    if (use_attn_qhead_splitkv) {
        if (static_cast<int>(blockIdx.x) < attn_blocks) {
            const int q_head_idx = static_cast<int>(blockIdx.x) / attn_qhead_split_k;
            const int split_idx = static_cast<int>(blockIdx.x) - q_head_idx * attn_qhead_split_k;
            const int kv_head_idx = q_head_idx / kAttnGqaRatio;
            const int merge_dim = lane * 4;

            float m_val = kAttnNegLarge;
            float l_val = 0.0f;
            float acc[kAttnDimsPerThread];
            #pragma unroll
            for (int i = 0; i < kAttnDimsPerThread; ++i) {
                acc[i] = 0.0f;
            }

            if (warp < kFullLayerQHeadSplitWarps) {
                const int chunk_size =
                    (valid_len + attn_qhead_split_k - 1) / attn_qhead_split_k;
                const int chunk_start = split_idx * chunk_size;
                const int chunk_end = min(chunk_start + chunk_size, valid_len);

                if (chunk_start < chunk_end) {
                    const int chunk_len = chunk_end - chunk_start;
                    const int shard_size =
                        (chunk_len + kFullLayerQHeadSplitWarps - 1) / kFullLayerQHeadSplitWarps;
                    const int shard_start = chunk_start + warp * shard_size;
                    const int shard_end = min(shard_start + shard_size, chunk_end);

                    if (shard_start < shard_end) {
                        float q_reg[kAttnDimsPerThread];
                        const __half* qp = qkv_half + q_head_idx * kAttnHeadDim + dim_offset;
                        #pragma unroll
                        for (int i = 0; i < kAttnDimsPerThread; i += 8) {
                            uint4 d = __ldg(reinterpret_cast<const uint4*>(qp + i));
                            const __half* h = reinterpret_cast<const __half*>(&d);
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                q_reg[i + j] = __half2float(h[j]);
                            }
                        }

                        const __half* k_base = k_cache_half + kv_head_idx * k_cache_stride_h;
                        const __half* v_base = v_cache_half + kv_head_idx * v_cache_stride_h;

                        for (int tile_start = shard_start; tile_start < shard_end;
                             tile_start += kAttnTileN) {
                            const int tile_end = min(tile_start + kAttnTileN, shard_end);
                            float local_scores[kAttnTileSteps];
                            float tile_max = kAttnNegLarge;

                            #pragma unroll
                            for (int step = 0; step < kAttnTileSteps; ++step) {
                                const int t = tile_start + step * kAttnGroupsPerWarp + group_id;
                                const bool valid = (t < tile_end);
                                float dot = 0.0f;
                                if (valid) {
                                    const __half* kp = k_base + t * k_cache_stride_s + dim_offset;
                                    #pragma unroll
                                    for (int i = 0; i < kAttnDimsPerThread; i += 8) {
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
                                const float score = valid ? (dot * attn_scale) : kAttnNegLarge;
                                local_scores[step] = score;
                                tile_max = fmaxf(tile_max, score);
                            }

                            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 4));
                            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 8));
                            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 16));

                            const float new_m = fmaxf(m_val, tile_max);
                            const float alpha = expf(m_val - new_m);
                            #pragma unroll
                            for (int i = 0; i < kAttnDimsPerThread; ++i) {
                                acc[i] *= alpha;
                            }
                            l_val *= alpha;
                            m_val = new_m;

                            #pragma unroll
                            for (int step = 0; step < kAttnTileSteps; ++step) {
                                const int t = tile_start + step * kAttnGroupsPerWarp + group_id;
                                const bool valid = (t < tile_end);
                                if (!valid) {
                                    continue;
                                }
                                const float p = expf(local_scores[step] - m_val);
                                l_val += p;
                                const __half* vp = v_base + t * v_cache_stride_s + dim_offset;
                                #pragma unroll
                                for (int i = 0; i < kAttnDimsPerThread; i += 8) {
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
                        for (int xor_mask = kAttnThreadGroup; xor_mask < kAttnWarpSize;
                             xor_mask <<= 1) {
                            const float pm = __shfl_xor_sync(0xffffffff, m_val, xor_mask);
                            const float pl = __shfl_xor_sync(0xffffffff, l_val, xor_mask);
                            const float nm = fmaxf(m_val, pm);
                            const float s_me = expf(m_val - nm);
                            const float s_pa = expf(pm - nm);
                            #pragma unroll
                            for (int i = 0; i < kAttnDimsPerThread; ++i) {
                                const float pa_v = __shfl_xor_sync(0xffffffff, acc[i], xor_mask);
                                acc[i] = acc[i] * s_me + pa_v * s_pa;
                            }
                            l_val = l_val * s_me + pl * s_pa;
                            m_val = nm;
                        }
                    }
                }
            }

            if (group_id == 0) {
                #pragma unroll
                for (int i = 0; i < kAttnDimsPerThread; ++i) {
                    smem_acc[warp][dim_offset + i] = acc[i];
                }
                if (group_lane == 0) {
                    smem_m[warp] = m_val;
                    smem_l[warp] = l_val;
                }
            }
            __syncthreads();

            if (warp == 0) {
                float m_merge = kAttnNegLarge;
                float l_merge = 0.0f;
                float a0 = 0.0f;
                float a1 = 0.0f;
                float a2 = 0.0f;
                float a3 = 0.0f;

                #pragma unroll
                for (int s = 0; s < kFullLayerQHeadSplitWarps; ++s) {
                    const float lv = smem_l[s];
                    const float mv = smem_m[s];
                    const float nm = fmaxf(m_merge, mv);
                    const float s_old = expf(m_merge - nm);
                    const float s_new = expf(mv - nm);
                    const float* pv = &smem_acc[s][merge_dim];
                    a0 = a0 * s_old + pv[0] * s_new;
                    a1 = a1 * s_old + pv[1] * s_new;
                    a2 = a2 * s_old + pv[2] * s_new;
                    a3 = a3 * s_old + pv[3] * s_new;
                    l_merge = l_merge * s_old + lv * s_new;
                    m_merge = nm;
                }

                float* pa = attn_partial_acc
                    + (q_head_idx * attn_qhead_split_k + split_idx) * kAttnHeadDim
                    + merge_dim;
                pa[0] = a0;
                pa[1] = a1;
                pa[2] = a2;
                pa[3] = a3;
                if (lane == 0) {
                    float* ps = attn_partial_stats
                        + (q_head_idx * attn_qhead_split_k + split_idx) * 2;
                    ps[0] = l_merge;
                    ps[1] = m_merge;
                }

                __syncwarp();
                __threadfence();

                int is_last = 0;
                if (lane == 0) {
                    const int arrived = atomicAdd(&attn_split_done_count[q_head_idx], 1);
                    is_last = (arrived == attn_qhead_split_k - 1) ? 1 : 0;
                    if (is_last) {
                        attn_split_done_count[q_head_idx] = 0;
                    }
                }
                is_last = __shfl_sync(0xffffffff, is_last, 0);

                if (is_last) {
                    float out_m = kAttnNegLarge;
                    float out_l = 0.0f;
                    float out0 = 0.0f;
                    float out1 = 0.0f;
                    float out2 = 0.0f;
                    float out3 = 0.0f;

                    for (int s = 0; s < attn_qhead_split_k; ++s) {
                        const float* ps = attn_partial_stats
                            + (q_head_idx * attn_qhead_split_k + s) * 2;
                        const float* pv = attn_partial_acc
                            + (q_head_idx * attn_qhead_split_k + s) * kAttnHeadDim
                            + merge_dim;
                        const float lv = ps[0];
                        const float mv = ps[1];
                        const float nm = fmaxf(out_m, mv);
                        const float s_old = expf(out_m - nm);
                        const float s_new = expf(mv - nm);
                        out0 = out0 * s_old + pv[0] * s_new;
                        out1 = out1 * s_old + pv[1] * s_new;
                        out2 = out2 * s_old + pv[2] * s_new;
                        out3 = out3 * s_old + pv[3] * s_new;
                        out_l = out_l * s_old + lv * s_new;
                        out_m = nm;
                    }

                    __half* op = qkv_half + q_head_idx * kAttnHeadDim + merge_dim;
                    if (out_l > 0.0f) {
                        const float inv_l = __fdividef(1.0f, out_l);
                        op[0] = __float2half(out0 * inv_l);
                        op[1] = __float2half(out1 * inv_l);
                        op[2] = __float2half(out2 * inv_l);
                        op[3] = __float2half(out3 * inv_l);
                    } else {
                        op[0] = __float2half(0.0f);
                        op[1] = __float2half(0.0f);
                        op[2] = __float2half(0.0f);
                        op[3] = __float2half(0.0f);
                    }
                }
            }
        }
        __syncthreads();
    } else {
        for (int q_head_idx = blockIdx.x; q_head_idx < num_q_heads; q_head_idx += gridDim.x) {
            const int kv_head_idx = q_head_idx / kAttnGqaRatio;
            const int chunk_size =
                (valid_len + kFullLayerInternalSplits - 1) / kFullLayerInternalSplits;
            const int chunk_start = warp * chunk_size;
            const int chunk_end = min(chunk_start + chunk_size, valid_len);

            float q_reg[kAttnDimsPerThread];
            {
                const __half* qp = qkv_half + q_head_idx * kAttnHeadDim + dim_offset;
                #pragma unroll
                for (int i = 0; i < kAttnDimsPerThread; i += 8) {
                    uint4 d = __ldg(reinterpret_cast<const uint4*>(qp + i));
                    const __half* h = reinterpret_cast<const __half*>(&d);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        q_reg[i + j] = __half2float(h[j]);
                    }
                }
            }

            float m_val = kAttnNegLarge;
            float l_val = 0.0f;
            float acc[kAttnDimsPerThread];
            #pragma unroll
            for (int i = 0; i < kAttnDimsPerThread; ++i) {
                acc[i] = 0.0f;
            }

            const __half* k_base = k_cache_half + kv_head_idx * k_cache_stride_h;
            const __half* v_base = v_cache_half + kv_head_idx * v_cache_stride_h;

            if (chunk_start < chunk_end) {
                for (int tile_start = chunk_start; tile_start < chunk_end;
                     tile_start += kAttnTileN) {
                    const int tile_end = min(tile_start + kAttnTileN, chunk_end);
                    float local_scores[kAttnTileSteps];
                    float tile_max = kAttnNegLarge;

                    #pragma unroll
                    for (int step = 0; step < kAttnTileSteps; ++step) {
                        const int t = tile_start + step * kAttnGroupsPerWarp + group_id;
                        const bool valid = (t < tile_end);
                        float dot = 0.0f;
                        if (valid) {
                            const __half* kp = k_base + t * k_cache_stride_s + dim_offset;
                            #pragma unroll
                            for (int i = 0; i < kAttnDimsPerThread; i += 8) {
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
                        const float score = valid ? (dot * attn_scale) : kAttnNegLarge;
                        local_scores[step] = score;
                        tile_max = fmaxf(tile_max, score);
                    }

                    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 4));
                    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 8));
                    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, 16));

                    const float new_m = fmaxf(m_val, tile_max);
                    const float alpha = expf(m_val - new_m);
                    #pragma unroll
                    for (int i = 0; i < kAttnDimsPerThread; ++i) {
                        acc[i] *= alpha;
                    }
                    l_val *= alpha;
                    m_val = new_m;

                    #pragma unroll
                    for (int step = 0; step < kAttnTileSteps; ++step) {
                        const int t = tile_start + step * kAttnGroupsPerWarp + group_id;
                        const bool valid = (t < tile_end);
                        if (valid) {
                            const float p = expf(local_scores[step] - m_val);
                            l_val += p;
                            const __half* vp = v_base + t * v_cache_stride_s + dim_offset;
                            #pragma unroll
                            for (int i = 0; i < kAttnDimsPerThread; i += 8) {
                                uint4 d = __ldg(reinterpret_cast<const uint4*>(vp + i));
                                const __half* h = reinterpret_cast<const __half*>(&d);
                                #pragma unroll
                                for (int j = 0; j < 8; ++j) {
                                    acc[i + j] += p * __half2float(h[j]);
                                }
                            }
                        }
                    }
                }
            }

            #pragma unroll
            for (int xor_mask = kAttnThreadGroup; xor_mask < kAttnWarpSize;
                 xor_mask <<= 1) {
                const float pm = __shfl_xor_sync(0xffffffff, m_val, xor_mask);
                const float pl = __shfl_xor_sync(0xffffffff, l_val, xor_mask);
                const float nm = fmaxf(m_val, pm);
                const float s_me = expf(m_val - nm);
                const float s_pa = expf(pm - nm);
                #pragma unroll
                for (int i = 0; i < kAttnDimsPerThread; ++i) {
                    const float pa_v = __shfl_xor_sync(0xffffffff, acc[i], xor_mask);
                    acc[i] = acc[i] * s_me + pa_v * s_pa;
                }
                l_val = l_val * s_me + pl * s_pa;
                m_val = nm;
            }

            if (group_id == 0) {
                #pragma unroll
                for (int i = 0; i < kAttnDimsPerThread; ++i) {
                    smem_acc[warp][dim_offset + i] = acc[i];
                }
                if (group_lane == 0) {
                    smem_m[warp] = m_val;
                    smem_l[warp] = l_val;
                }
            }
            __syncthreads();

            constexpr int kDimsPerLane = kAttnHeadDim / kAttnWarpSize;  // 4
            #pragma unroll
            for (int stride = kFullLayerInternalSplits / 2; stride >= 1; stride >>= 1) {
                if (warp < stride) {
                    const int src = warp + stride;
                    const float pm = smem_m[src];
                    const float pl = smem_l[src];
                    const float my_m = smem_m[warp];
                    const float my_l = smem_l[warp];
                    const float nm = fmaxf(my_m, pm);
                    const float s_me = expf(my_m - nm);
                    const float s_pa = expf(pm - nm);
                    #pragma unroll
                    for (int i = 0; i < kDimsPerLane; ++i) {
                        const float my_a = smem_acc[warp][lane * kDimsPerLane + i];
                        const float pa_v = smem_acc[src][lane * kDimsPerLane + i];
                        smem_acc[warp][lane * kDimsPerLane + i] = my_a * s_me + pa_v * s_pa;
                    }
                    if (lane == 0) {
                        smem_m[warp] = nm;
                        smem_l[warp] = my_l * s_me + pl * s_pa;
                    }
                }
                __syncthreads();
            }

            if (warp == 0) {
                const float inv_l = (smem_l[0] > 0.0f) ? (1.0f / smem_l[0]) : 0.0f;
                __half* op = qkv_half + q_head_idx * kAttnHeadDim;
                #pragma unroll
                for (int i = 0; i < kDimsPerLane; ++i) {
                    op[lane * kDimsPerLane + i] =
                        __float2half(smem_acc[0][lane * kDimsPerLane + i] * inv_l);
                }
            }
            __syncthreads();
        }
    }

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[profile_base + FL_ATTENTION_DONE] = read_clock64();
            profile[profile_base + FL_SYNC3_START] = read_clock64();
        }
    }
    if (partial_attn_flags) {
        if (static_cast<int>(blockIdx.x) < attn_blocks) {
            monotonic_counter_arrive(attn_done_count);
        }
        monotonic_counter_wait_ge(attn_done_count, attn_done_target_sh);
    } else {
        grid_sync.sync();
    }
    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[profile_base + FL_SYNC3_DONE] = read_clock64();
        }
    }

    // Phase 4: existing new softsync path for o_proj -> post_attn_mlp -> residual.
    for (int idx = tid; idx < vecs_k2048; idx += kFullLayerThreads) {
        x_small_vec[idx] = reinterpret_cast<const uint4*>(qkv_half)[idx];
    }
    __syncthreads();

    for (int out_base = o_proj_block_idx * kFullLayerWarps;
         o_proj_block_idx >= 0 && out_base < kNosplitSpecializedK2048;
         out_base += o_proj_grid_blocks * kFullLayerWarps) {
        const int out_idx = out_base + warp;
        if (out_idx < kNosplitSpecializedK2048) {
            const uint4* o_row_vec = reinterpret_cast<const uint4*>(
                o_w_half + out_idx * o_proj_stride_wn);
            __half2 acc0 = __float2half2_rn(0.0f);
            __half2 acc1 = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane; vec_idx < vecs_k2048; vec_idx += 64) {
                const uint4 x_vec0 = x_small_vec[vec_idx];
                const uint4 x_vec1 = x_small_vec[vec_idx + 32];
                accumulate_hfma2_uint4(x_vec0, o_row_vec[vec_idx], acc0);
                accumulate_hfma2_uint4(x_vec1, o_row_vec[vec_idx + 32], acc1);
            }

            const float2 f0 = __half22float2(acc0);
            const float2 f1 = __half22float2(acc1);
            float sum = f0.x + f0.y + f1.x + f1.y;
            sum = warp_sum(sum);
            if (lane == 0) {
                if (has_o_proj_bias) {
                    sum += __half2float(o_bias_half[out_idx]);
                }
                hidden_half[out_idx] = __float2half(sum);
            }
        }
    }

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[profile_base + FL_OPROJ_DONE] = read_clock64();
            profile[profile_base + FL_SYNC4_START] = read_clock64();
        }
    }
    grid_sync.sync();
    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[profile_base + FL_SYNC4_DONE] = read_clock64();
        }
    }

    thread_sum_sq = 0.0f;
    for (int idx = tid; idx < kNosplitSpecializedK2048; idx += kFullLayerThreads) {
        const float summed_f =
            __half2float(residual_half[idx]) + __half2float(hidden_half[idx]);
        residual_sh[idx] = __float2half(summed_f);
        thread_sum_sq += summed_f * summed_f;
    }

    thread_sum_sq = warp_sum(thread_sum_sq);
    if (lane == 0) {
        warp_sums[warp] = thread_sum_sq;
    }
    __syncthreads();

    if (warp == 0) {
        float block_sum_sq = lane < kFullLayerWarps ? warp_sums[lane] : 0.0f;
        block_sum_sq = warp_sum(block_sum_sq);
        if (lane == 0) {
            inv_rms_sh = rsqrtf(
                block_sum_sq / static_cast<float>(kNosplitSpecializedK2048) + eps);
        }
    }
    __syncthreads();

    const float post_inv_rms = inv_rms_sh;
    for (int idx = tid; idx < kNosplitSpecializedK2048; idx += kFullLayerThreads) {
        const float summed_f = __half2float(residual_sh[idx]);
        const float scale_f = __half2float(post_norm_half[idx]);
        x_small_sh[idx] = __float2half(summed_f * post_inv_rms * scale_f);
    }
    __syncthreads();

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[profile_base + FL_POST_RMSNORM_DONE] = read_clock64();
        }
    }

    {
        for (int hidden_base = blockIdx.x * kFullLayerWarps;
             hidden_base < 6144;
             hidden_base += gridDim.x * kFullLayerWarps) {
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
    }

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[profile_base + FL_GATEUP_SILU_DONE] = read_clock64();
            profile[profile_base + FL_SYNC5_START] = read_clock64();
        }
    }
    grid_sync.sync();
    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[profile_base + FL_SYNC5_DONE] = read_clock64();
        }
    }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    // For the common >=64-block tuning range, down_proj has at most 2 waves.
    // Keep the staged-x reuse, but spread the tail rows across the whole grid
    // so the extra wave does not sit entirely on a small prefix of blocks.
    if (gridDim.x >= 64) {
        const int first_wave_rows = gridDim.x * kFullLayerWarps;
        const int extra_rows =
            first_wave_rows < kNosplitSpecializedK2048
                ? (kNosplitSpecializedK2048 - first_wave_rows)
                : 0;
        const int extra_row_start = static_cast<int>(
            (static_cast<int64_t>(blockIdx.x) * extra_rows) / gridDim.x);
        const int extra_row_end = static_cast<int>(
            (static_cast<int64_t>(blockIdx.x + 1) * extra_rows) / gridDim.x);
        const int extra_row_count = extra_row_end - extra_row_start;
        const int row0 = blockIdx.x * kFullLayerWarps + warp;
        const bool row1_valid = warp < extra_row_count;
        const int row1 = first_wave_rows + extra_row_start + warp;
        for (int vec_idx = tid; vec_idx < kAsyncVecsPerTileK6144;
             vec_idx += kFullLayerThreads) {
            if constexpr (kUseCpAsync) {
                cp_async_cg_16(
                    &x_tile_sh[0][vec_idx],
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

        float sum0 = 0.0f;
        float sum1 = 0.0f;
        #pragma unroll
        for (int tile_idx = 0; tile_idx < 6; ++tile_idx) {
            const int stage = tile_idx & 1;
            const int tile_start = tile_idx * kNosplitAsyncTileK6144;

            if (tile_idx + 1 < 6) {
                const int next_tile_start = tile_start + kNosplitAsyncTileK6144;
                const uint4* x_next_vec =
                    reinterpret_cast<const uint4*>(hidden_half + next_tile_start);
                for (int vec_idx = tid; vec_idx < kAsyncVecsPerTileK6144;
                     vec_idx += kFullLayerThreads) {
                    if constexpr (kUseCpAsync) {
                        cp_async_cg_16(
                            &x_tile_sh[stage ^ 1][vec_idx],
                            x_next_vec + vec_idx);
                    } else {
                        x_tile_sh[stage ^ 1][vec_idx] = x_next_vec[vec_idx];
                    }
                }
                if constexpr (kUseCpAsync) {
                    cp_async_commit_group();
                }
            }

            if (row0 < kNosplitSpecializedK2048) {
                const uint4* x_sh_stage_vec = x_tile_sh[stage];
                const uint4* w_row_vec = reinterpret_cast<const uint4*>(
                    down_w_half + row0 * down_proj_stride_wn + tile_start);
                __half2 h2_acc0 = __float2half2_rn(0.0f);
                __half2 h2_acc1 = __float2half2_rn(0.0f);

                #pragma unroll
                for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144;
                     vec_idx += 64) {
                    const uint4 x_vec0 = x_sh_stage_vec[vec_idx];
                    const uint4 w_vec0 = w_row_vec[vec_idx];
                    const uint4 x_vec1 = x_sh_stage_vec[vec_idx + 32];
                    const uint4 w_vec1 = w_row_vec[vec_idx + 32];
                    accumulate_hfma2_uint4(x_vec0, w_vec0, h2_acc0);
                    accumulate_hfma2_uint4(x_vec1, w_vec1, h2_acc1);
                }

                const float2 f2_0 = __half22float2(h2_acc0);
                const float2 f2_1 = __half22float2(h2_acc1);
                sum0 += f2_0.x + f2_0.y + f2_1.x + f2_1.y;
            }

            if (row1_valid) {
                const uint4* x_sh_stage_vec = x_tile_sh[stage];
                const uint4* w_row_vec = reinterpret_cast<const uint4*>(
                    down_w_half + row1 * down_proj_stride_wn + tile_start);
                __half2 h2_acc0 = __float2half2_rn(0.0f);
                __half2 h2_acc1 = __float2half2_rn(0.0f);

                #pragma unroll
                for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144;
                     vec_idx += 64) {
                    const uint4 x_vec0 = x_sh_stage_vec[vec_idx];
                    const uint4 w_vec0 = w_row_vec[vec_idx];
                    const uint4 x_vec1 = x_sh_stage_vec[vec_idx + 32];
                    const uint4 w_vec1 = w_row_vec[vec_idx + 32];
                    accumulate_hfma2_uint4(x_vec0, w_vec0, h2_acc0);
                    accumulate_hfma2_uint4(x_vec1, w_vec1, h2_acc1);
                }

                const float2 f2_0 = __half22float2(h2_acc0);
                const float2 f2_1 = __half22float2(h2_acc1);
                sum1 += f2_0.x + f2_0.y + f2_1.x + f2_1.y;
            }

            if constexpr (kUseCpAsync) {
                if (tile_idx + 1 < 6) {
                    cp_async_wait_group<0>();
                }
            }
            __syncthreads();
        }

        if (row0 < kNosplitSpecializedK2048) {
            sum0 = warp_sum(sum0);
            if (lane == 0) {
                float result = sum0 + __half2float(residual_sh[row0]);
                if (has_down_proj_bias) {
                    result += __half2float(down_bias_half[row0]);
                }
                out_half[row0] = __float2half(result);
            }
        }
        if (row1_valid) {
            sum1 = warp_sum(sum1);
            if (lane == 0) {
                float result = sum1 + __half2float(residual_sh[row1]);
                if (has_down_proj_bias) {
                    result += __half2float(down_bias_half[row1]);
                }
                out_half[row1] = __float2half(result);
            }
        }
        __syncthreads();
    } else {
        // Fallback for low-block sweeps that still need 3+ row waves.
        for (int row_base = blockIdx.x * kFullLayerWarps;
             row_base < kNosplitSpecializedK2048;
             row_base += gridDim.x * kFullLayerWarps) {
            const int row = row_base + warp;
            for (int vec_idx = tid; vec_idx < kAsyncVecsPerTileK6144;
                 vec_idx += kFullLayerThreads) {
                if constexpr (kUseCpAsync) {
                    cp_async_cg_16(
                        &x_tile_sh[0][vec_idx],
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
                    for (int vec_idx = tid; vec_idx < kAsyncVecsPerTileK6144;
                         vec_idx += kFullLayerThreads) {
                        if constexpr (kUseCpAsync) {
                            cp_async_cg_16(
                                &x_tile_sh[stage ^ 1][vec_idx],
                                x_next_vec + vec_idx);
                        } else {
                            x_tile_sh[stage ^ 1][vec_idx] = x_next_vec[vec_idx];
                        }
                    }
                    if constexpr (kUseCpAsync) {
                        cp_async_commit_group();
                    }
                }

                if (row < kNosplitSpecializedK2048) {
                    const uint4* x_sh_stage_vec = x_tile_sh[stage];
                    const uint4* w_row_vec = reinterpret_cast<const uint4*>(
                        down_w_half + row * down_proj_stride_wn + tile_start);
                    __half2 h2_acc0 = __float2half2_rn(0.0f);
                    __half2 h2_acc1 = __float2half2_rn(0.0f);

                    #pragma unroll
                    for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144;
                         vec_idx += 64) {
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

            if (row < kNosplitSpecializedK2048) {
                sum = warp_sum(sum);
                if (lane == 0) {
                    float result = sum + __half2float(residual_sh[row]);
                    if (has_down_proj_bias) {
                        result += __half2float(down_bias_half[row]);
                    }
                    out_half[row] = __float2half(result);
                }
            }
            __syncthreads();
        }
    }

    if constexpr (kProfile) {
        if (threadIdx.x == 0 && profile != nullptr) {
            profile[profile_base + FL_DOWNPROJ_DONE] = read_clock64();
            profile[profile_base + FL_KERNEL_DONE] = read_clock64();
            profile[profile_base + FL_GT_END] = read_globaltimer();
        }
    }

}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Host-side launch wrappers (linked from decode_gemv_ext.cpp)
// ---------------------------------------------------------------------------

torch::Tensor decode_attn_persistent_half_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& out,
    double scale) {

    TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda(),
        "inputs must be CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16, "persistent attn fp16 only");
    TORCH_CHECK(q.dim() == 2 && q.size(1) == 128,
        "q must be [num_q_heads, 128]");

    const int64_t num_kv_heads = k_cache.size(1);
    const int64_t max_seq_len  = k_cache.size(2);

    int device_idx = static_cast<int>(q.device().index());
    int num_sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &num_sms, cudaDevAttrMultiProcessorCount, device_idx));

    c10::cuda::CUDAGuard device_guard(q.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto qp  = reinterpret_cast<const __half*>(q.data_ptr<at::Half>());
    auto kp  = reinterpret_cast<const __half*>(k_cache.data_ptr<at::Half>());
    auto vp  = reinterpret_cast<const __half*>(v_cache.data_ptr<at::Half>());
    auto op  = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    auto* cp = cache_position.data_ptr<int64_t>();

    float scale_f = static_cast<float>(scale);
    int max_sl = static_cast<int>(max_seq_len);
    int n_kv   = static_cast<int>(num_kv_heads);

    int64_t q_s0 = q.stride(0);
    int64_t k_s1 = k_cache.stride(1), k_s2 = k_cache.stride(2);
    int64_t v_s1 = v_cache.stride(1), v_s2 = v_cache.stride(2);
    int64_t o_s0 = out.stride(0);

    // Standalone attention: no grid sync needed, use regular launch.
    // Grid = num_q_heads (one block per q-head) for maximum SM utilization.
    const int num_q_heads = static_cast<int>(q.size(0));
    const int grid = num_q_heads;
    decode_attn_persistent_fwd_kernel<<<grid, kPersistentThreads, 0, stream>>>(
        qp, kp, vp, cp, op,
        q_s0,
        k_s1, k_s2,
        v_s1, v_s2,
        o_s0,
        scale_f,
        max_sl,
        n_kv);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_decode_full_layer_softsync_half_cuda(
    const torch::Tensor& hidden_states,
    const torch::Tensor& qkv_weight,
    const torch::Tensor& input_layernorm_weight,
    const torch::Tensor& q_norm_weight,
    const torch::Tensor& k_norm_weight,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& o_proj_weight,
    const torch::Tensor& o_proj_bias,
    const torch::Tensor& post_attn_norm_weight,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& down_proj_bias,
    const torch::Tensor& barrier_count,
    const torch::Tensor& barrier_generation,
    const torch::Tensor& kv_ready_count,
    const torch::Tensor& attn_done_count,
    const torch::Tensor& attn_partial_acc,
    const torch::Tensor& attn_partial_stats,
    const torch::Tensor& attn_split_done_count,
    const torch::Tensor& qkv_out,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out,
    double eps,
    double attn_scale,
    bool has_o_proj_bias,
    bool has_down_proj_bias,
    int64_t launch_blocks) {
    TORCH_CHECK(
        hidden_states.is_cuda() && qkv_weight.is_cuda() && input_layernorm_weight.is_cuda() && q_norm_weight.is_cuda()
            && k_norm_weight.is_cuda() && cos.is_cuda() && sin.is_cuda()
            && k_cache.is_cuda() && v_cache.is_cuda() && cache_position.is_cuda()
            && o_proj_weight.is_cuda() && post_attn_norm_weight.is_cuda()
            && gate_up_weight.is_cuda() && down_proj_weight.is_cuda()
            && barrier_count.is_cuda() && barrier_generation.is_cuda()
            && kv_ready_count.is_cuda() && attn_done_count.is_cuda()
            && attn_partial_acc.is_cuda() && attn_partial_stats.is_cuda()
            && attn_split_done_count.is_cuda()
            && qkv_out.is_cuda() && hidden_out.is_cuda() && out.is_cuda(),
        "fused full layer softsync expects CUDA tensors");
    TORCH_CHECK(hidden_states.scalar_type() == torch::kFloat16, "hidden_states must be fp16");
    TORCH_CHECK(qkv_weight.scalar_type() == torch::kFloat16, "qkv_weight must be fp16");
    TORCH_CHECK(input_layernorm_weight.scalar_type() == torch::kFloat16, "input_layernorm_weight must be fp16");
    TORCH_CHECK(q_norm_weight.scalar_type() == torch::kFloat16, "q_norm_weight must be fp16");
    TORCH_CHECK(k_norm_weight.scalar_type() == torch::kFloat16, "k_norm_weight must be fp16");
    TORCH_CHECK(cos.scalar_type() == torch::kFloat16, "cos must be fp16");
    TORCH_CHECK(sin.scalar_type() == torch::kFloat16, "sin must be fp16");
    TORCH_CHECK(k_cache.scalar_type() == torch::kFloat16, "k_cache must be fp16");
    TORCH_CHECK(v_cache.scalar_type() == torch::kFloat16, "v_cache must be fp16");
    TORCH_CHECK(o_proj_weight.scalar_type() == torch::kFloat16, "o_proj_weight must be fp16");
    TORCH_CHECK(post_attn_norm_weight.scalar_type() == torch::kFloat16, "post_attn_norm_weight must be fp16");
    TORCH_CHECK(gate_up_weight.scalar_type() == torch::kFloat16, "gate_up_weight must be fp16");
    TORCH_CHECK(down_proj_weight.scalar_type() == torch::kFloat16, "down_proj_weight must be fp16");
    TORCH_CHECK(barrier_count.scalar_type() == torch::kInt32, "barrier_count must be int32");
    TORCH_CHECK(barrier_generation.scalar_type() == torch::kInt32, "barrier_generation must be int32");
    TORCH_CHECK(kv_ready_count.scalar_type() == torch::kInt32, "kv_ready_count must be int32");
    TORCH_CHECK(attn_done_count.scalar_type() == torch::kInt32, "attn_done_count must be int32");
    TORCH_CHECK(attn_partial_acc.scalar_type() == torch::kFloat, "attn_partial_acc must be fp32");
    TORCH_CHECK(attn_partial_stats.scalar_type() == torch::kFloat, "attn_partial_stats must be fp32");
    TORCH_CHECK(attn_split_done_count.scalar_type() == torch::kInt32, "attn_split_done_count must be int32");
    TORCH_CHECK(qkv_out.scalar_type() == torch::kFloat16, "qkv_out must be fp16");
    TORCH_CHECK(hidden_out.scalar_type() == torch::kFloat16, "hidden_out must be fp16");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(cache_position.scalar_type() == torch::kLong, "cache_position must be int64");

    TORCH_CHECK(hidden_states.dim() == 1 && hidden_states.numel() == kNosplitSpecializedK2048,
        "hidden_states must be flattened [2048]");
    TORCH_CHECK(
        qkv_weight.dim() == 2
            && qkv_weight.size(0) == 4096
            && qkv_weight.size(1) == kNosplitSpecializedK2048
            && qkv_weight.stride(1) == 1,
        "qkv_weight must be contiguous [4096, 2048]");
    TORCH_CHECK(
        input_layernorm_weight.dim() == 1
            && input_layernorm_weight.numel() == kNosplitSpecializedK2048
            && input_layernorm_weight.stride(0) == 1,
        "input_layernorm_weight must be contiguous [2048]");
    TORCH_CHECK(q_norm_weight.dim() == 1 && q_norm_weight.numel() == kAttnHeadDim,
        "q_norm_weight must be [128]");
    TORCH_CHECK(k_norm_weight.dim() == 1 && k_norm_weight.numel() == kAttnHeadDim,
        "k_norm_weight must be [128]");
    TORCH_CHECK(cos.numel() == kAttnHeadDim, "cos must have 128 elements");
    TORCH_CHECK(sin.numel() == kAttnHeadDim, "sin must have 128 elements");
    TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4, "k_cache/v_cache must be [1, kv_heads, seq, 128]");
    TORCH_CHECK(k_cache.size(0) == 1 && v_cache.size(0) == 1, "only batch=1 cache is supported");
    TORCH_CHECK(k_cache.size(1) == 8 && v_cache.size(1) == 8, "k_cache/v_cache must have 8 kv heads");
    TORCH_CHECK(k_cache.size(3) == kAttnHeadDim && v_cache.size(3) == kAttnHeadDim,
        "k_cache/v_cache head_dim must be 128");
    const int attn_qhead_split_k = max(
        2, getenv_int_or("CUDA_FUSED_FULL_LAYER_SOFTSYNC_ATTN_QHEAD_SPLITKV_SPLIT_K", 4));
    const int64_t num_q_heads = k_cache.size(1) * kAttnGqaRatio;
    TORCH_CHECK(cache_position.dim() == 1 && cache_position.numel() == 1, "cache_position must be [1]");
    TORCH_CHECK(
        o_proj_weight.dim() == 2
            && o_proj_weight.size(0) == 2048
            && o_proj_weight.size(1) == kNosplitSpecializedK2048
            && o_proj_weight.stride(1) == 1,
        "o_proj_weight must be contiguous [2048, 2048]");
    TORCH_CHECK(
        post_attn_norm_weight.dim() == 1
            && post_attn_norm_weight.numel() == kNosplitSpecializedK2048,
        "post_attn_norm_weight must be [2048]");
    TORCH_CHECK(
        gate_up_weight.dim() == 2
            && gate_up_weight.size(0) == 12288
            && gate_up_weight.size(1) == kNosplitSpecializedK2048
            && gate_up_weight.stride(1) == 1,
        "gate_up_weight must be contiguous [12288, 2048]");
    TORCH_CHECK(
        down_proj_weight.dim() == 2
            && down_proj_weight.size(0) == 2048
            && down_proj_weight.size(1) == 6144
            && down_proj_weight.stride(1) == 1,
        "down_proj_weight must be contiguous [2048, 6144]");
    TORCH_CHECK(barrier_count.dim() == 1 && barrier_count.numel() == 1, "barrier_count must be [1]");
    TORCH_CHECK(barrier_generation.dim() == 1 && barrier_generation.numel() == 1,
        "barrier_generation must be [1]");
    TORCH_CHECK(kv_ready_count.dim() == 1 && kv_ready_count.numel() == 1,
        "kv_ready_count must be [1]");
    TORCH_CHECK(attn_done_count.dim() == 1 && attn_done_count.numel() == 1,
        "attn_done_count must be [1]");
    TORCH_CHECK(
        attn_partial_acc.dim() == 3
            && attn_partial_acc.size(0) == num_q_heads
            && attn_partial_acc.size(1) == attn_qhead_split_k
            && attn_partial_acc.size(2) == kAttnHeadDim
            && attn_partial_acc.stride(2) == 1,
        "attn_partial_acc must be contiguous [num_q_heads, split_k, 128]");
    TORCH_CHECK(
        attn_partial_stats.dim() == 3
            && attn_partial_stats.size(0) == num_q_heads
            && attn_partial_stats.size(1) == attn_qhead_split_k
            && attn_partial_stats.size(2) == 2
            && attn_partial_stats.stride(2) == 1,
        "attn_partial_stats must be contiguous [num_q_heads, split_k, 2]");
    TORCH_CHECK(
        attn_split_done_count.dim() == 1 && attn_split_done_count.numel() == num_q_heads,
        "attn_split_done_count must be [num_q_heads]");
    TORCH_CHECK(qkv_out.dim() == 1 && qkv_out.numel() == 4096, "qkv_out must be [4096]");
    TORCH_CHECK(hidden_out.dim() == 1 && hidden_out.numel() == 6144, "hidden_out must be [6144]");
    TORCH_CHECK(out.dim() == 1 && out.numel() == 2048, "out must be [2048]");
    TORCH_CHECK(
        !has_o_proj_bias
            || (o_proj_bias.is_cuda() && o_proj_bias.scalar_type() == torch::kFloat16
                && o_proj_bias.dim() == 1 && o_proj_bias.numel() == 2048
                && o_proj_bias.stride(0) == 1),
        "o_proj_bias must be contiguous [2048] when provided");
    TORCH_CHECK(
        !has_down_proj_bias
            || (down_proj_bias.is_cuda() && down_proj_bias.scalar_type() == torch::kFloat16
                && down_proj_bias.dim() == 1 && down_proj_bias.numel() == 2048
                && down_proj_bias.stride(0) == 1),
        "down_proj_bias must be contiguous [2048] when provided");

    int device_idx = static_cast<int>(hidden_states.device().index());
    int num_sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_idx));
    const int requested_blocks = static_cast<int>(launch_blocks);
    const int grid_blocks = max(
        1,
        min(num_sms, requested_blocks > 0 ? requested_blocks : num_sms));

    int max_blocks_per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        fused_decode_full_layer_softsync_k2048_fwd_kernel<false>,
        kFullLayerThreads,
        0));
    TORCH_CHECK(
        max_blocks_per_sm * num_sms >= grid_blocks,
        "fused_decode_full_layer_softsync launch would deadlock: resident blocks ",
        max_blocks_per_sm * num_sms,
        " < required grid blocks ",
        grid_blocks);

    c10::cuda::CUDAGuard device_guard(hidden_states.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto* hidden_ptr = hidden_states.data_ptr<at::Half>();
    auto* qkv_w_ptr = qkv_weight.data_ptr<at::Half>();
    auto* input_norm_ptr = input_layernorm_weight.data_ptr<at::Half>();
    auto* q_norm_ptr = q_norm_weight.data_ptr<at::Half>();
    auto* k_norm_ptr = k_norm_weight.data_ptr<at::Half>();
    auto* cos_ptr = cos.data_ptr<at::Half>();
    auto* sin_ptr = sin.data_ptr<at::Half>();
    auto* k_cache_ptr = k_cache.data_ptr<at::Half>();
    auto* v_cache_ptr = v_cache.data_ptr<at::Half>();
    auto* cache_pos_ptr = cache_position.data_ptr<int64_t>();
    auto* o_w_ptr = o_proj_weight.data_ptr<at::Half>();
    auto* o_b_ptr = has_o_proj_bias ? o_proj_bias.data_ptr<at::Half>() : nullptr;
    auto* post_norm_ptr = post_attn_norm_weight.data_ptr<at::Half>();
    auto* gate_up_ptr = gate_up_weight.data_ptr<at::Half>();
    auto* down_w_ptr = down_proj_weight.data_ptr<at::Half>();
    auto* down_b_ptr = has_down_proj_bias ? down_proj_bias.data_ptr<at::Half>() : nullptr;
    auto* barrier_count_ptr = reinterpret_cast<unsigned int*>(barrier_count.data_ptr<int32_t>());
    auto* barrier_generation_ptr = reinterpret_cast<unsigned int*>(barrier_generation.data_ptr<int32_t>());
    auto* kv_ready_count_ptr = reinterpret_cast<unsigned int*>(kv_ready_count.data_ptr<int32_t>());
    auto* attn_done_count_ptr = reinterpret_cast<unsigned int*>(attn_done_count.data_ptr<int32_t>());
    auto* attn_partial_acc_ptr = attn_partial_acc.data_ptr<float>();
    auto* attn_partial_stats_ptr = attn_partial_stats.data_ptr<float>();
    auto* attn_split_done_count_ptr = attn_split_done_count.data_ptr<int32_t>();
    auto* qkv_out_ptr = qkv_out.data_ptr<at::Half>();
    auto* hidden_out_ptr = hidden_out.data_ptr<at::Half>();
    auto* out_ptr = out.data_ptr<at::Half>();
    float eps_f = static_cast<float>(eps);
    float attn_scale_f = static_cast<float>(attn_scale);
    int64_t qkv_s0 = qkv_weight.stride(0);
    int64_t k_cache_s1 = k_cache.stride(1);
    int64_t k_cache_s2 = k_cache.stride(2);
    int64_t v_cache_s1 = v_cache.stride(1);
    int64_t v_cache_s2 = v_cache.stride(2);
    int64_t o_s0 = o_proj_weight.stride(0);
    int64_t gate_up_s0 = gate_up_weight.stride(0);
    int64_t down_s0 = down_proj_weight.stride(0);
    int max_sl = static_cast<int>(k_cache.size(2));
    int n_kv = static_cast<int>(k_cache.size(1));
    const bool enable_o_proj_prefetch =
        getenv_int_or("ENABLE_CUDA_FUSED_FULL_LAYER_SOFTSYNC_O_PROJ_PREFETCH", 0) == 1;
    const bool partial_attn_flags =
        getenv_int_or("ENABLE_CUDA_FUSED_FULL_LAYER_SOFTSYNC_PARTIAL_ATTN_FLAGS", 0) == 1;
    const bool o_proj_idle_only =
        getenv_int_or("ENABLE_CUDA_FUSED_FULL_LAYER_SOFTSYNC_O_PROJ_IDLE_ONLY", 0) == 1;
    const bool enable_attn_qhead_splitkv =
        getenv_int_or("ENABLE_CUDA_FUSED_FULL_LAYER_SOFTSYNC_ATTN_QHEAD_SPLITKV", 1) == 1;
    const int attn_qhead_splitkv_max_seq_len = getenv_int_or(
        "CUDA_FUSED_FULL_LAYER_SOFTSYNC_ATTN_QHEAD_SPLITKV_MAX_SEQ_LEN", 4096);
    const bool use_attn_qhead_splitkv =
        enable_attn_qhead_splitkv
        && grid_blocks >= n_kv * kAttnGqaRatio * attn_qhead_split_k
        && (attn_qhead_splitkv_max_seq_len <= 0 || max_sl <= attn_qhead_splitkv_max_seq_len);
    const int attn_active_blocks = min(
        grid_blocks,
        use_attn_qhead_splitkv ? n_kv * kAttnGqaRatio * attn_qhead_split_k : n_kv * kAttnGqaRatio);
    const int idle_blocks = max(0, grid_blocks - attn_active_blocks);
    const int requested_o_proj_prefetch_blocks = getenv_int_or(
        "CUDA_FUSED_FULL_LAYER_SOFTSYNC_O_PROJ_PREFETCH_BLOCKS", 0);
    const int o_proj_prefetch_blocks = enable_o_proj_prefetch
        ? (requested_o_proj_prefetch_blocks > 0
            ? min(idle_blocks, requested_o_proj_prefetch_blocks)
            : idle_blocks)
        : 0;
    const uint64_t o_proj_prefetch_start_delay_ns =
        enable_o_proj_prefetch
            ? static_cast<uint64_t>(max(0, getenv_int_or(
                "CUDA_FUSED_FULL_LAYER_SOFTSYNC_O_PROJ_PREFETCH_START_DELAY_NS", 0)))
            : 0;
    const uint64_t o_proj_prefetch_budget_ns =
        enable_o_proj_prefetch
            ? static_cast<uint64_t>(max(0, getenv_int_or(
                "CUDA_FUSED_FULL_LAYER_SOFTSYNC_O_PROJ_PREFETCH_BUDGET_NS", 12000)))
            : 0;

    fused_decode_full_layer_softsync_k2048_fwd_kernel<false><<<grid_blocks, kFullLayerThreads, 0, stream>>>(
        hidden_ptr,
        qkv_w_ptr,
        input_norm_ptr,
        q_norm_ptr,
        k_norm_ptr,
        cos_ptr,
        sin_ptr,
        k_cache_ptr,
        v_cache_ptr,
        cache_pos_ptr,
        o_w_ptr,
        o_b_ptr,
        post_norm_ptr,
        gate_up_ptr,
        down_w_ptr,
        down_b_ptr,
        barrier_count_ptr,
        barrier_generation_ptr,
        kv_ready_count_ptr,
        attn_done_count_ptr,
        attn_partial_acc_ptr,
        attn_partial_stats_ptr,
        attn_split_done_count_ptr,
        qkv_out_ptr,
        hidden_out_ptr,
        out_ptr,
        nullptr,
        eps_f,
        attn_scale_f,
        has_o_proj_bias,
        has_down_proj_bias,
        qkv_s0,
        k_cache_s1,
        k_cache_s2,
        v_cache_s1,
        v_cache_s2,
        o_s0,
        gate_up_s0,
        down_s0,
        partial_attn_flags,
        o_proj_idle_only,
        o_proj_prefetch_blocks,
        o_proj_prefetch_start_delay_ns,
        o_proj_prefetch_budget_ns,
        max_sl,
        n_kv,
        use_attn_qhead_splitkv,
        attn_qhead_split_k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_decode_full_layer_softsync_profile_half_cuda(
    const torch::Tensor& hidden_states,
    const torch::Tensor& qkv_weight,
    const torch::Tensor& input_layernorm_weight,
    const torch::Tensor& q_norm_weight,
    const torch::Tensor& k_norm_weight,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& o_proj_weight,
    const torch::Tensor& o_proj_bias,
    const torch::Tensor& post_attn_norm_weight,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& down_proj_bias,
    const torch::Tensor& barrier_count,
    const torch::Tensor& barrier_generation,
    const torch::Tensor& kv_ready_count,
    const torch::Tensor& attn_done_count,
    const torch::Tensor& attn_partial_acc,
    const torch::Tensor& attn_partial_stats,
    const torch::Tensor& attn_split_done_count,
    const torch::Tensor& qkv_out,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out,
    const torch::Tensor& profile,
    double eps,
    double attn_scale,
    bool has_o_proj_bias,
    bool has_down_proj_bias,
    int64_t launch_blocks) {
    TORCH_CHECK(
        hidden_states.is_cuda() && qkv_weight.is_cuda() && input_layernorm_weight.is_cuda() && q_norm_weight.is_cuda()
            && k_norm_weight.is_cuda() && cos.is_cuda() && sin.is_cuda()
            && k_cache.is_cuda() && v_cache.is_cuda() && cache_position.is_cuda()
            && o_proj_weight.is_cuda() && post_attn_norm_weight.is_cuda()
            && gate_up_weight.is_cuda() && down_proj_weight.is_cuda()
            && barrier_count.is_cuda() && barrier_generation.is_cuda()
            && kv_ready_count.is_cuda() && attn_done_count.is_cuda()
            && attn_partial_acc.is_cuda() && attn_partial_stats.is_cuda()
            && attn_split_done_count.is_cuda()
            && qkv_out.is_cuda() && hidden_out.is_cuda() && out.is_cuda()
            && profile.is_cuda(),
        "fused full layer softsync profile expects CUDA tensors");
    TORCH_CHECK(hidden_states.scalar_type() == torch::kFloat16, "hidden_states must be fp16");
    TORCH_CHECK(qkv_weight.scalar_type() == torch::kFloat16, "qkv_weight must be fp16");
    TORCH_CHECK(input_layernorm_weight.scalar_type() == torch::kFloat16, "input_layernorm_weight must be fp16");
    TORCH_CHECK(q_norm_weight.scalar_type() == torch::kFloat16, "q_norm_weight must be fp16");
    TORCH_CHECK(k_norm_weight.scalar_type() == torch::kFloat16, "k_norm_weight must be fp16");
    TORCH_CHECK(cos.scalar_type() == torch::kFloat16, "cos must be fp16");
    TORCH_CHECK(sin.scalar_type() == torch::kFloat16, "sin must be fp16");
    TORCH_CHECK(k_cache.scalar_type() == torch::kFloat16, "k_cache must be fp16");
    TORCH_CHECK(v_cache.scalar_type() == torch::kFloat16, "v_cache must be fp16");
    TORCH_CHECK(o_proj_weight.scalar_type() == torch::kFloat16, "o_proj_weight must be fp16");
    TORCH_CHECK(post_attn_norm_weight.scalar_type() == torch::kFloat16, "post_attn_norm_weight must be fp16");
    TORCH_CHECK(gate_up_weight.scalar_type() == torch::kFloat16, "gate_up_weight must be fp16");
    TORCH_CHECK(down_proj_weight.scalar_type() == torch::kFloat16, "down_proj_weight must be fp16");
    TORCH_CHECK(barrier_count.scalar_type() == torch::kInt32, "barrier_count must be int32");
    TORCH_CHECK(barrier_generation.scalar_type() == torch::kInt32, "barrier_generation must be int32");
    TORCH_CHECK(kv_ready_count.scalar_type() == torch::kInt32, "kv_ready_count must be int32");
    TORCH_CHECK(attn_done_count.scalar_type() == torch::kInt32, "attn_done_count must be int32");
    TORCH_CHECK(attn_partial_acc.scalar_type() == torch::kFloat, "attn_partial_acc must be fp32");
    TORCH_CHECK(attn_partial_stats.scalar_type() == torch::kFloat, "attn_partial_stats must be fp32");
    TORCH_CHECK(attn_split_done_count.scalar_type() == torch::kInt32, "attn_split_done_count must be int32");
    TORCH_CHECK(qkv_out.scalar_type() == torch::kFloat16, "qkv_out must be fp16");
    TORCH_CHECK(hidden_out.scalar_type() == torch::kFloat16, "hidden_out must be fp16");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(profile.scalar_type() == torch::kLong, "profile must be int64");
    TORCH_CHECK(cache_position.scalar_type() == torch::kLong, "cache_position must be int64");

    TORCH_CHECK(hidden_states.dim() == 1 && hidden_states.numel() == kNosplitSpecializedK2048,
        "hidden_states must be flattened [2048]");
    TORCH_CHECK(
        qkv_weight.dim() == 2
            && qkv_weight.size(0) == 4096
            && qkv_weight.size(1) == kNosplitSpecializedK2048
            && qkv_weight.stride(1) == 1,
        "qkv_weight must be contiguous [4096, 2048]");
    TORCH_CHECK(
        input_layernorm_weight.dim() == 1
            && input_layernorm_weight.numel() == kNosplitSpecializedK2048
            && input_layernorm_weight.stride(0) == 1,
        "input_layernorm_weight must be contiguous [2048]");
    TORCH_CHECK(q_norm_weight.dim() == 1 && q_norm_weight.numel() == kAttnHeadDim,
        "q_norm_weight must be [128]");
    TORCH_CHECK(k_norm_weight.dim() == 1 && k_norm_weight.numel() == kAttnHeadDim,
        "k_norm_weight must be [128]");
    TORCH_CHECK(cos.numel() == kAttnHeadDim, "cos must have 128 elements");
    TORCH_CHECK(sin.numel() == kAttnHeadDim, "sin must have 128 elements");
    TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4, "k_cache/v_cache must be [1, kv_heads, seq, 128]");
    TORCH_CHECK(k_cache.size(0) == 1 && v_cache.size(0) == 1, "only batch=1 cache is supported");
    TORCH_CHECK(k_cache.size(1) == 8 && v_cache.size(1) == 8, "k_cache/v_cache must have 8 kv heads");
    TORCH_CHECK(k_cache.size(3) == kAttnHeadDim && v_cache.size(3) == kAttnHeadDim,
        "k_cache/v_cache head_dim must be 128");
    const int attn_qhead_split_k = max(
        2, getenv_int_or("CUDA_FUSED_FULL_LAYER_SOFTSYNC_ATTN_QHEAD_SPLITKV_SPLIT_K", 4));
    const int64_t num_q_heads = k_cache.size(1) * kAttnGqaRatio;
    TORCH_CHECK(cache_position.dim() == 1 && cache_position.numel() == 1, "cache_position must be [1]");
    TORCH_CHECK(
        o_proj_weight.dim() == 2
            && o_proj_weight.size(0) == 2048
            && o_proj_weight.size(1) == kNosplitSpecializedK2048
            && o_proj_weight.stride(1) == 1,
        "o_proj_weight must be contiguous [2048, 2048]");
    TORCH_CHECK(
        post_attn_norm_weight.dim() == 1
            && post_attn_norm_weight.numel() == kNosplitSpecializedK2048,
        "post_attn_norm_weight must be [2048]");
    TORCH_CHECK(
        gate_up_weight.dim() == 2
            && gate_up_weight.size(0) == 12288
            && gate_up_weight.size(1) == kNosplitSpecializedK2048
            && gate_up_weight.stride(1) == 1,
        "gate_up_weight must be contiguous [12288, 2048]");
    TORCH_CHECK(
        down_proj_weight.dim() == 2
            && down_proj_weight.size(0) == 2048
            && down_proj_weight.size(1) == 6144
            && down_proj_weight.stride(1) == 1,
        "down_proj_weight must be contiguous [2048, 6144]");
    TORCH_CHECK(barrier_count.dim() == 1 && barrier_count.numel() == 1, "barrier_count must be [1]");
    TORCH_CHECK(barrier_generation.dim() == 1 && barrier_generation.numel() == 1,
        "barrier_generation must be [1]");
    TORCH_CHECK(kv_ready_count.dim() == 1 && kv_ready_count.numel() == 1,
        "kv_ready_count must be [1]");
    TORCH_CHECK(attn_done_count.dim() == 1 && attn_done_count.numel() == 1,
        "attn_done_count must be [1]");
    TORCH_CHECK(
        attn_partial_acc.dim() == 3
            && attn_partial_acc.size(0) == num_q_heads
            && attn_partial_acc.size(1) == attn_qhead_split_k
            && attn_partial_acc.size(2) == kAttnHeadDim
            && attn_partial_acc.stride(2) == 1,
        "attn_partial_acc must be contiguous [num_q_heads, split_k, 128]");
    TORCH_CHECK(
        attn_partial_stats.dim() == 3
            && attn_partial_stats.size(0) == num_q_heads
            && attn_partial_stats.size(1) == attn_qhead_split_k
            && attn_partial_stats.size(2) == 2
            && attn_partial_stats.stride(2) == 1,
        "attn_partial_stats must be contiguous [num_q_heads, split_k, 2]");
    TORCH_CHECK(
        attn_split_done_count.dim() == 1 && attn_split_done_count.numel() == num_q_heads,
        "attn_split_done_count must be [num_q_heads]");
    TORCH_CHECK(qkv_out.dim() == 1 && qkv_out.numel() == 4096, "qkv_out must be [4096]");
    TORCH_CHECK(hidden_out.dim() == 1 && hidden_out.numel() == 6144, "hidden_out must be [6144]");
    TORCH_CHECK(out.dim() == 1 && out.numel() == 2048, "out must be [2048]");
    TORCH_CHECK(
        !has_o_proj_bias
            || (o_proj_bias.is_cuda() && o_proj_bias.scalar_type() == torch::kFloat16
                && o_proj_bias.dim() == 1 && o_proj_bias.numel() == 2048
                && o_proj_bias.stride(0) == 1),
        "o_proj_bias must be contiguous [2048] when provided");
    TORCH_CHECK(
        !has_down_proj_bias
            || (down_proj_bias.is_cuda() && down_proj_bias.scalar_type() == torch::kFloat16
                && down_proj_bias.dim() == 1 && down_proj_bias.numel() == 2048
                && down_proj_bias.stride(0) == 1),
        "down_proj_bias must be contiguous [2048] when provided");

    int device_idx = static_cast<int>(hidden_states.device().index());
    int num_sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_idx));
    const int requested_blocks = static_cast<int>(launch_blocks);
    const int grid_blocks = max(
        1,
        min(num_sms, requested_blocks > 0 ? requested_blocks : num_sms));
    TORCH_CHECK(
        profile.dim() == 2
            && profile.size(0) == grid_blocks
            && profile.size(1) == kFullLayerProfileSlots,
        "profile must be [grid_blocks, kFullLayerProfileSlots]");

    int max_blocks_per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        fused_decode_full_layer_softsync_k2048_fwd_kernel<true>,
        kFullLayerThreads,
        0));
    TORCH_CHECK(
        max_blocks_per_sm * num_sms >= grid_blocks,
        "fused_decode_full_layer_softsync_profile launch would deadlock: resident blocks ",
        max_blocks_per_sm * num_sms,
        " < required grid blocks ",
        grid_blocks);

    c10::cuda::CUDAGuard device_guard(hidden_states.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto* hidden_ptr = hidden_states.data_ptr<at::Half>();
    auto* qkv_w_ptr = qkv_weight.data_ptr<at::Half>();
    auto* input_norm_ptr = input_layernorm_weight.data_ptr<at::Half>();
    auto* q_norm_ptr = q_norm_weight.data_ptr<at::Half>();
    auto* k_norm_ptr = k_norm_weight.data_ptr<at::Half>();
    auto* cos_ptr = cos.data_ptr<at::Half>();
    auto* sin_ptr = sin.data_ptr<at::Half>();
    auto* k_cache_ptr = k_cache.data_ptr<at::Half>();
    auto* v_cache_ptr = v_cache.data_ptr<at::Half>();
    auto* cache_pos_ptr = cache_position.data_ptr<int64_t>();
    auto* o_w_ptr = o_proj_weight.data_ptr<at::Half>();
    auto* o_b_ptr = has_o_proj_bias ? o_proj_bias.data_ptr<at::Half>() : nullptr;
    auto* post_norm_ptr = post_attn_norm_weight.data_ptr<at::Half>();
    auto* gate_up_ptr = gate_up_weight.data_ptr<at::Half>();
    auto* down_w_ptr = down_proj_weight.data_ptr<at::Half>();
    auto* down_b_ptr = has_down_proj_bias ? down_proj_bias.data_ptr<at::Half>() : nullptr;
    auto* barrier_count_ptr = reinterpret_cast<unsigned int*>(barrier_count.data_ptr<int32_t>());
    auto* barrier_generation_ptr = reinterpret_cast<unsigned int*>(barrier_generation.data_ptr<int32_t>());
    auto* kv_ready_count_ptr = reinterpret_cast<unsigned int*>(kv_ready_count.data_ptr<int32_t>());
    auto* attn_done_count_ptr = reinterpret_cast<unsigned int*>(attn_done_count.data_ptr<int32_t>());
    auto* attn_partial_acc_ptr = attn_partial_acc.data_ptr<float>();
    auto* attn_partial_stats_ptr = attn_partial_stats.data_ptr<float>();
    auto* attn_split_done_count_ptr = attn_split_done_count.data_ptr<int32_t>();
    auto* qkv_out_ptr = qkv_out.data_ptr<at::Half>();
    auto* hidden_out_ptr = hidden_out.data_ptr<at::Half>();
    auto* out_ptr = out.data_ptr<at::Half>();
    auto* profile_ptr = reinterpret_cast<uint64_t*>(profile.data_ptr<int64_t>());
    float eps_f = static_cast<float>(eps);
    float attn_scale_f = static_cast<float>(attn_scale);
    int64_t qkv_s0 = qkv_weight.stride(0);
    int64_t k_cache_s1 = k_cache.stride(1);
    int64_t k_cache_s2 = k_cache.stride(2);
    int64_t v_cache_s1 = v_cache.stride(1);
    int64_t v_cache_s2 = v_cache.stride(2);
    int64_t o_s0 = o_proj_weight.stride(0);
    int64_t gate_up_s0 = gate_up_weight.stride(0);
    int64_t down_s0 = down_proj_weight.stride(0);
    int max_sl = static_cast<int>(k_cache.size(2));
    int n_kv = static_cast<int>(k_cache.size(1));
    const bool enable_o_proj_prefetch =
        getenv_int_or("ENABLE_CUDA_FUSED_FULL_LAYER_SOFTSYNC_O_PROJ_PREFETCH", 0) == 1;
    const bool partial_attn_flags =
        getenv_int_or("ENABLE_CUDA_FUSED_FULL_LAYER_SOFTSYNC_PARTIAL_ATTN_FLAGS", 0) == 1;
    const bool o_proj_idle_only =
        getenv_int_or("ENABLE_CUDA_FUSED_FULL_LAYER_SOFTSYNC_O_PROJ_IDLE_ONLY", 0) == 1;
    const bool enable_attn_qhead_splitkv =
        getenv_int_or("ENABLE_CUDA_FUSED_FULL_LAYER_SOFTSYNC_ATTN_QHEAD_SPLITKV", 1) == 1;
    const int attn_qhead_splitkv_max_seq_len = getenv_int_or(
        "CUDA_FUSED_FULL_LAYER_SOFTSYNC_ATTN_QHEAD_SPLITKV_MAX_SEQ_LEN", 4096);
    const bool use_attn_qhead_splitkv =
        enable_attn_qhead_splitkv
        && grid_blocks >= n_kv * kAttnGqaRatio * attn_qhead_split_k
        && (attn_qhead_splitkv_max_seq_len <= 0 || max_sl <= attn_qhead_splitkv_max_seq_len);
    const int attn_active_blocks = min(
        grid_blocks,
        use_attn_qhead_splitkv ? n_kv * kAttnGqaRatio * attn_qhead_split_k : n_kv * kAttnGqaRatio);
    const int idle_blocks = max(0, grid_blocks - attn_active_blocks);
    const int requested_o_proj_prefetch_blocks = getenv_int_or(
        "CUDA_FUSED_FULL_LAYER_SOFTSYNC_O_PROJ_PREFETCH_BLOCKS", 0);
    const int o_proj_prefetch_blocks = enable_o_proj_prefetch
        ? (requested_o_proj_prefetch_blocks > 0
            ? min(idle_blocks, requested_o_proj_prefetch_blocks)
            : idle_blocks)
        : 0;
    const uint64_t o_proj_prefetch_start_delay_ns =
        enable_o_proj_prefetch
            ? static_cast<uint64_t>(max(0, getenv_int_or(
                "CUDA_FUSED_FULL_LAYER_SOFTSYNC_O_PROJ_PREFETCH_START_DELAY_NS", 0)))
            : 0;
    const uint64_t o_proj_prefetch_budget_ns =
        enable_o_proj_prefetch
            ? static_cast<uint64_t>(max(0, getenv_int_or(
                "CUDA_FUSED_FULL_LAYER_SOFTSYNC_O_PROJ_PREFETCH_BUDGET_NS", 12000)))
            : 0;

    fused_decode_full_layer_softsync_k2048_fwd_kernel<true><<<grid_blocks, kFullLayerThreads, 0, stream>>>(
        hidden_ptr,
        qkv_w_ptr,
        input_norm_ptr,
        q_norm_ptr,
        k_norm_ptr,
        cos_ptr,
        sin_ptr,
        k_cache_ptr,
        v_cache_ptr,
        cache_pos_ptr,
        o_w_ptr,
        o_b_ptr,
        post_norm_ptr,
        gate_up_ptr,
        down_w_ptr,
        down_b_ptr,
        barrier_count_ptr,
        barrier_generation_ptr,
        kv_ready_count_ptr,
        attn_done_count_ptr,
        attn_partial_acc_ptr,
        attn_partial_stats_ptr,
        attn_split_done_count_ptr,
        qkv_out_ptr,
        hidden_out_ptr,
        out_ptr,
        profile_ptr,
        eps_f,
        attn_scale_f,
        has_o_proj_bias,
        has_down_proj_bias,
        qkv_s0,
        k_cache_s1,
        k_cache_s2,
        v_cache_s1,
        v_cache_s2,
        o_s0,
        gate_up_s0,
        down_s0,
        partial_attn_flags,
        o_proj_idle_only,
        o_proj_prefetch_blocks,
        o_proj_prefetch_start_delay_ns,
        o_proj_prefetch_budget_ns,
        max_sl,
        n_kv,
        use_attn_qhead_splitkv,
        attn_qhead_split_k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_gateup_silu_downproj_gridsync_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out) {
    TORCH_CHECK(x.is_cuda() && gate_up_weight.is_cuda() && down_proj_weight.is_cuda(),
        "inputs must be CUDA");
    TORCH_CHECK(hidden_out.is_cuda() && out.is_cuda(), "outputs must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "gridsync only supports fp16");
    TORCH_CHECK(gate_up_weight.scalar_type() == torch::kFloat16, "gridsync fp16");
    TORCH_CHECK(down_proj_weight.scalar_type() == torch::kFloat16, "gridsync fp16");
    TORCH_CHECK(x.dim() == 1 && x.numel() == kNosplitSpecializedK2048, "x [2048]");
    TORCH_CHECK(gate_up_weight.dim() == 2 && gate_up_weight.size(0) == 12288 &&
        gate_up_weight.size(1) == kNosplitSpecializedK2048 && gate_up_weight.stride(1) == 1,
        "gate_up [12288,2048] contiguous");
    TORCH_CHECK(down_proj_weight.dim() == 2 && down_proj_weight.size(0) == 2048 &&
        down_proj_weight.size(1) == 6144 && down_proj_weight.stride(1) == 1,
        "down_proj [2048,6144] contiguous");
    TORCH_CHECK(hidden_out.dim() == 1 && hidden_out.numel() == 6144, "hidden [6144]");
    TORCH_CHECK(out.dim() == 1 && out.numel() == 2048, "out [2048]");

    constexpr int grid_blocks = 2048 / kRowsPerBlockAsync;

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto* x_ptr = x.data_ptr<at::Half>();
    auto* gu_ptr = gate_up_weight.data_ptr<at::Half>();
    auto* dp_ptr = down_proj_weight.data_ptr<at::Half>();
    auto* h_ptr = hidden_out.data_ptr<at::Half>();
    auto* o_ptr = out.data_ptr<at::Half>();
    int64_t gu_s0 = gate_up_weight.stride(0);
    int64_t dp_s0 = down_proj_weight.stride(0);

    void* args[] = {&x_ptr, &gu_ptr, &dp_ptr, &h_ptr, &o_ptr, &gu_s0, &dp_s0};

    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(
            fused_gateup_silu_downproj_gridsync_k2048_k6144_fwd_kernel),
        grid_blocks, kRow8Threads, args, 0, stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_gateup_silu_downproj_gridsync_profile_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out,
    const torch::Tensor& profile) {
    TORCH_CHECK(x.is_cuda() && gate_up_weight.is_cuda() && down_proj_weight.is_cuda(),
        "inputs must be CUDA");
    TORCH_CHECK(hidden_out.is_cuda() && out.is_cuda() && profile.is_cuda(),
        "outputs must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "gridsync_profile fp16");
    TORCH_CHECK(gate_up_weight.scalar_type() == torch::kFloat16, "gridsync_profile fp16");
    TORCH_CHECK(down_proj_weight.scalar_type() == torch::kFloat16, "gridsync_profile fp16");
    TORCH_CHECK(profile.scalar_type() == torch::kInt64 && profile.is_contiguous(),
        "profile int64 contiguous");
    TORCH_CHECK(x.dim() == 1 && x.numel() == kNosplitSpecializedK2048, "x [2048]");
    TORCH_CHECK(gate_up_weight.dim() == 2 && gate_up_weight.size(0) == 12288 &&
        gate_up_weight.size(1) == kNosplitSpecializedK2048 && gate_up_weight.stride(1) == 1,
        "gate_up [12288,2048] contiguous");
    TORCH_CHECK(down_proj_weight.dim() == 2 && down_proj_weight.size(0) == 2048 &&
        down_proj_weight.size(1) == 6144 && down_proj_weight.stride(1) == 1,
        "down_proj [2048,6144] contiguous");
    TORCH_CHECK(hidden_out.dim() == 1 && hidden_out.numel() == 6144, "hidden [6144]");
    TORCH_CHECK(out.dim() == 1 && out.numel() == 2048, "out [2048]");

    constexpr int grid_blocks = 2048 / kRowsPerBlockAsync;
    TORCH_CHECK(profile.numel() == static_cast<int64_t>(grid_blocks) * kSoftsyncProfileSlots,
        "profile shape mismatch");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    profile.zero_();

    auto* x_ptr = x.data_ptr<at::Half>();
    auto* gu_ptr = gate_up_weight.data_ptr<at::Half>();
    auto* dp_ptr = down_proj_weight.data_ptr<at::Half>();
    auto* h_ptr = hidden_out.data_ptr<at::Half>();
    auto* o_ptr = out.data_ptr<at::Half>();
    auto* p_ptr = reinterpret_cast<uint64_t*>(profile.data_ptr<int64_t>());
    int64_t gu_s0 = gate_up_weight.stride(0);
    int64_t dp_s0 = down_proj_weight.stride(0);

    void* args[] = {&x_ptr, &gu_ptr, &dp_ptr, &h_ptr, &o_ptr, &p_ptr, &gu_s0, &dp_s0};

    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(
            fused_gateup_silu_downproj_gridsync_profile_k2048_k6144_fwd_kernel<true>),
        grid_blocks, kRow8Threads, args, 0, stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// ---------------------------------------------------------------------------
// Persistent kernel host wrappers
// ---------------------------------------------------------------------------

torch::Tensor fused_post_attn_mlp_persistent_half_cuda(
    const torch::Tensor& residual,
    const torch::Tensor& attn_output,
    const torch::Tensor& norm_weight,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& bias,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out,
    double eps,
    bool has_bias) {
    TORCH_CHECK(
        residual.is_cuda() && attn_output.is_cuda() && norm_weight.is_cuda()
            && gate_up_weight.is_cuda() && down_proj_weight.is_cuda(),
        "inputs must be CUDA");
    TORCH_CHECK(hidden_out.is_cuda() && out.is_cuda(), "outputs must be CUDA");
    TORCH_CHECK(residual.scalar_type() == torch::kFloat16, "persistent only supports fp16");
    TORCH_CHECK(attn_output.scalar_type() == torch::kFloat16, "persistent only supports fp16");
    TORCH_CHECK(norm_weight.scalar_type() == torch::kFloat16, "persistent only supports fp16");
    TORCH_CHECK(gate_up_weight.scalar_type() == torch::kFloat16, "persistent fp16");
    TORCH_CHECK(down_proj_weight.scalar_type() == torch::kFloat16, "persistent fp16");
    TORCH_CHECK(residual.dim() == 1 && residual.numel() == kNosplitSpecializedK2048, "residual [2048]");
    TORCH_CHECK(attn_output.dim() == 1 && attn_output.numel() == kNosplitSpecializedK2048, "attn_output [2048]");
    TORCH_CHECK(norm_weight.dim() == 1 && norm_weight.numel() == kNosplitSpecializedK2048, "norm_weight [2048]");
    TORCH_CHECK(gate_up_weight.dim() == 2 && gate_up_weight.size(0) == 12288 &&
        gate_up_weight.size(1) == kNosplitSpecializedK2048 && gate_up_weight.stride(1) == 1,
        "gate_up [12288,2048] contiguous");
    TORCH_CHECK(down_proj_weight.dim() == 2 && down_proj_weight.size(0) == 2048 &&
        down_proj_weight.size(1) == 6144 && down_proj_weight.stride(1) == 1,
        "down_proj [2048,6144] contiguous");
    TORCH_CHECK(hidden_out.dim() == 1 && hidden_out.numel() == 6144, "hidden [6144]");
    TORCH_CHECK(out.dim() == 1 && out.numel() == 2048, "out [2048]");
    TORCH_CHECK(!has_bias || (bias.is_cuda() && bias.scalar_type() == torch::kFloat16 &&
        bias.dim() == 1 && bias.numel() == 2048 && bias.stride(0) == 1),
        "bias [2048] contiguous when provided");

    int device_idx = static_cast<int>(residual.device().index());
    int num_sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_idx));

    c10::cuda::CUDAGuard device_guard(residual.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto* residual_ptr = residual.data_ptr<at::Half>();
    auto* attn_ptr = attn_output.data_ptr<at::Half>();
    auto* norm_ptr = norm_weight.data_ptr<at::Half>();
    auto* gu_ptr = gate_up_weight.data_ptr<at::Half>();
    auto* dp_ptr = down_proj_weight.data_ptr<at::Half>();
    auto* bias_ptr = has_bias ? bias.data_ptr<at::Half>() : nullptr;
    auto* h_ptr = hidden_out.data_ptr<at::Half>();
    auto* o_ptr = out.data_ptr<at::Half>();
    float eps_f = static_cast<float>(eps);
    int64_t gu_s0 = gate_up_weight.stride(0);
    int64_t dp_s0 = down_proj_weight.stride(0);

    void* args[] = {
        &residual_ptr,
        &attn_ptr,
        &norm_ptr,
        &gu_ptr,
        &dp_ptr,
        &bias_ptr,
        &h_ptr,
        &o_ptr,
        &eps_f,
        &has_bias,
        &gu_s0,
        &dp_s0,
    };

    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(fused_post_attn_mlp_persistent_k2048_k6144_fwd_kernel),
        num_sms, kPersistentThreads, args, 0, stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_o_proj_post_attn_mlp_persistent_half_cuda(
    const torch::Tensor& residual,
    const torch::Tensor& attn_input,
    const torch::Tensor& o_proj_weight,
    const torch::Tensor& o_proj_bias,
    const torch::Tensor& norm_weight,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& down_proj_bias,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out,
    double eps,
    bool has_o_proj_bias,
    bool has_down_proj_bias) {
    TORCH_CHECK(
        residual.is_cuda() && attn_input.is_cuda() && o_proj_weight.is_cuda()
            && norm_weight.is_cuda() && gate_up_weight.is_cuda()
            && down_proj_weight.is_cuda(),
        "inputs must be CUDA");
    TORCH_CHECK(hidden_out.is_cuda() && out.is_cuda(), "outputs must be CUDA");
    TORCH_CHECK(residual.scalar_type() == torch::kFloat16, "persistent only supports fp16");
    TORCH_CHECK(attn_input.scalar_type() == torch::kFloat16, "persistent only supports fp16");
    TORCH_CHECK(o_proj_weight.scalar_type() == torch::kFloat16, "persistent fp16");
    TORCH_CHECK(norm_weight.scalar_type() == torch::kFloat16, "persistent fp16");
    TORCH_CHECK(gate_up_weight.scalar_type() == torch::kFloat16, "persistent fp16");
    TORCH_CHECK(down_proj_weight.scalar_type() == torch::kFloat16, "persistent fp16");
    TORCH_CHECK(residual.dim() == 1 && residual.numel() == kNosplitSpecializedK2048, "residual [2048]");
    TORCH_CHECK(attn_input.dim() == 1 && attn_input.numel() == kNosplitSpecializedK2048, "attn_input [2048]");
    TORCH_CHECK(o_proj_weight.dim() == 2 && o_proj_weight.size(0) == 2048 &&
        o_proj_weight.size(1) == kNosplitSpecializedK2048 && o_proj_weight.stride(1) == 1,
        "o_proj [2048,2048] contiguous");
    TORCH_CHECK(norm_weight.dim() == 1 && norm_weight.numel() == kNosplitSpecializedK2048, "norm_weight [2048]");
    TORCH_CHECK(gate_up_weight.dim() == 2 && gate_up_weight.size(0) == 12288 &&
        gate_up_weight.size(1) == kNosplitSpecializedK2048 && gate_up_weight.stride(1) == 1,
        "gate_up [12288,2048] contiguous");
    TORCH_CHECK(down_proj_weight.dim() == 2 && down_proj_weight.size(0) == 2048 &&
        down_proj_weight.size(1) == 6144 && down_proj_weight.stride(1) == 1,
        "down_proj [2048,6144] contiguous");
    TORCH_CHECK(hidden_out.dim() == 1 && hidden_out.numel() == 6144, "hidden [6144]");
    TORCH_CHECK(out.dim() == 1 && out.numel() == 2048, "out [2048]");
    TORCH_CHECK(!has_o_proj_bias || (o_proj_bias.is_cuda() && o_proj_bias.scalar_type() == torch::kFloat16 &&
        o_proj_bias.dim() == 1 && o_proj_bias.numel() == 2048 && o_proj_bias.stride(0) == 1),
        "o_proj_bias [2048] contiguous when provided");
    TORCH_CHECK(!has_down_proj_bias || (down_proj_bias.is_cuda() && down_proj_bias.scalar_type() == torch::kFloat16 &&
        down_proj_bias.dim() == 1 && down_proj_bias.numel() == 2048 && down_proj_bias.stride(0) == 1),
        "down_proj_bias [2048] contiguous when provided");

    int device_idx = static_cast<int>(residual.device().index());
    int num_sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_idx));

    c10::cuda::CUDAGuard device_guard(residual.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto* residual_ptr = residual.data_ptr<at::Half>();
    auto* attn_ptr = attn_input.data_ptr<at::Half>();
    auto* o_w_ptr = o_proj_weight.data_ptr<at::Half>();
    auto* o_b_ptr = has_o_proj_bias ? o_proj_bias.data_ptr<at::Half>() : nullptr;
    auto* norm_ptr = norm_weight.data_ptr<at::Half>();
    auto* gu_ptr = gate_up_weight.data_ptr<at::Half>();
    auto* dp_ptr = down_proj_weight.data_ptr<at::Half>();
    auto* dp_b_ptr = has_down_proj_bias ? down_proj_bias.data_ptr<at::Half>() : nullptr;
    auto* h_ptr = hidden_out.data_ptr<at::Half>();
    auto* o_ptr = out.data_ptr<at::Half>();
    float eps_f = static_cast<float>(eps);
    int64_t o_s0 = o_proj_weight.stride(0);
    int64_t gu_s0 = gate_up_weight.stride(0);
    int64_t dp_s0 = down_proj_weight.stride(0);

    void* args[] = {
        &residual_ptr,
        &attn_ptr,
        &o_w_ptr,
        &o_b_ptr,
        &norm_ptr,
        &gu_ptr,
        &dp_ptr,
        &dp_b_ptr,
        &h_ptr,
        &o_ptr,
        &eps_f,
        &has_o_proj_bias,
        &has_down_proj_bias,
        &o_s0,
        &gu_s0,
        &dp_s0,
    };

    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(fused_o_proj_post_attn_mlp_persistent_k2048_k6144_fwd_kernel),
        num_sms, kPersistentThreads, args, 0, stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_o_proj_post_attn_mlp_persistent_profile_half_cuda(
    const torch::Tensor& residual,
    const torch::Tensor& attn_input,
    const torch::Tensor& o_proj_weight,
    const torch::Tensor& o_proj_bias,
    const torch::Tensor& norm_weight,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& down_proj_bias,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out,
    const torch::Tensor& profile,
    double eps,
    bool has_o_proj_bias,
    bool has_down_proj_bias,
    int64_t gate_up_active_sms) {
    TORCH_CHECK(
        residual.is_cuda() && attn_input.is_cuda() && o_proj_weight.is_cuda()
            && norm_weight.is_cuda() && gate_up_weight.is_cuda()
            && down_proj_weight.is_cuda(),
        "inputs must be CUDA");
    TORCH_CHECK(hidden_out.is_cuda() && out.is_cuda() && profile.is_cuda(),
        "outputs must be CUDA");
    TORCH_CHECK(residual.scalar_type() == torch::kFloat16, "persistent_profile fp16");
    TORCH_CHECK(attn_input.scalar_type() == torch::kFloat16, "persistent_profile fp16");
    TORCH_CHECK(o_proj_weight.scalar_type() == torch::kFloat16, "persistent_profile fp16");
    TORCH_CHECK(norm_weight.scalar_type() == torch::kFloat16, "persistent_profile fp16");
    TORCH_CHECK(gate_up_weight.scalar_type() == torch::kFloat16, "persistent_profile fp16");
    TORCH_CHECK(down_proj_weight.scalar_type() == torch::kFloat16, "persistent_profile fp16");
    TORCH_CHECK(profile.scalar_type() == torch::kInt64 && profile.is_contiguous(),
        "profile int64 contiguous");
    TORCH_CHECK(residual.dim() == 1 && residual.numel() == kNosplitSpecializedK2048, "residual [2048]");
    TORCH_CHECK(attn_input.dim() == 1 && attn_input.numel() == kNosplitSpecializedK2048, "attn_input [2048]");
    TORCH_CHECK(o_proj_weight.dim() == 2 && o_proj_weight.size(0) == 2048 &&
        o_proj_weight.size(1) == kNosplitSpecializedK2048 && o_proj_weight.stride(1) == 1,
        "o_proj [2048,2048] contiguous");
    TORCH_CHECK(norm_weight.dim() == 1 && norm_weight.numel() == kNosplitSpecializedK2048, "norm_weight [2048]");
    TORCH_CHECK(gate_up_weight.dim() == 2 && gate_up_weight.size(0) == 12288 &&
        gate_up_weight.size(1) == kNosplitSpecializedK2048 && gate_up_weight.stride(1) == 1,
        "gate_up [12288,2048] contiguous");
    TORCH_CHECK(down_proj_weight.dim() == 2 && down_proj_weight.size(0) == 2048 &&
        down_proj_weight.size(1) == 6144 && down_proj_weight.stride(1) == 1,
        "down_proj [2048,6144] contiguous");
    TORCH_CHECK(hidden_out.dim() == 1 && hidden_out.numel() == 6144, "hidden [6144]");
    TORCH_CHECK(out.dim() == 1 && out.numel() == 2048, "out [2048]");
    TORCH_CHECK(!has_o_proj_bias || (o_proj_bias.is_cuda() && o_proj_bias.scalar_type() == torch::kFloat16 &&
        o_proj_bias.dim() == 1 && o_proj_bias.numel() == 2048 && o_proj_bias.stride(0) == 1),
        "o_proj_bias [2048] contiguous when provided");
    TORCH_CHECK(!has_down_proj_bias || (down_proj_bias.is_cuda() && down_proj_bias.scalar_type() == torch::kFloat16 &&
        down_proj_bias.dim() == 1 && down_proj_bias.numel() == 2048 && down_proj_bias.stride(0) == 1),
        "down_proj_bias [2048] contiguous when provided");

    int device_idx = static_cast<int>(residual.device().index());
    int num_sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_idx));

    TORCH_CHECK(profile.numel() == static_cast<int64_t>(num_sms) * kOProjMlpProfileSlots,
        "profile shape mismatch: expected ", num_sms * kOProjMlpProfileSlots,
        " got ", profile.numel());

    c10::cuda::CUDAGuard device_guard(residual.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    profile.zero_();

    auto* residual_ptr = residual.data_ptr<at::Half>();
    auto* attn_ptr = attn_input.data_ptr<at::Half>();
    auto* o_w_ptr = o_proj_weight.data_ptr<at::Half>();
    auto* o_b_ptr = has_o_proj_bias ? o_proj_bias.data_ptr<at::Half>() : nullptr;
    auto* norm_ptr = norm_weight.data_ptr<at::Half>();
    auto* gu_ptr = gate_up_weight.data_ptr<at::Half>();
    auto* dp_ptr = down_proj_weight.data_ptr<at::Half>();
    auto* dp_b_ptr = has_down_proj_bias ? down_proj_bias.data_ptr<at::Half>() : nullptr;
    auto* h_ptr = hidden_out.data_ptr<at::Half>();
    auto* o_ptr = out.data_ptr<at::Half>();
    auto* p_ptr = reinterpret_cast<uint64_t*>(profile.data_ptr<int64_t>());
    float eps_f = static_cast<float>(eps);
    int64_t o_s0 = o_proj_weight.stride(0);
    int64_t gu_s0 = gate_up_weight.stride(0);
    int64_t dp_s0 = down_proj_weight.stride(0);
    int gu_active_sms = static_cast<int>(gate_up_active_sms);

    void* args[] = {
        &residual_ptr, &attn_ptr, &o_w_ptr, &o_b_ptr, &norm_ptr,
        &gu_ptr, &dp_ptr, &dp_b_ptr, &h_ptr, &o_ptr, &p_ptr,
        &eps_f, &has_o_proj_bias, &has_down_proj_bias, &o_s0, &gu_s0, &dp_s0,
        &gu_active_sms,
    };

    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(
            fused_o_proj_post_attn_mlp_persistent_profile_k2048_k6144_fwd_kernel<true>),
        num_sms, kPersistentThreads, args, 0, stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_o_proj_post_attn_mlp_softsync_half_cuda(
    const torch::Tensor& residual,
    const torch::Tensor& attn_input,
    const torch::Tensor& o_proj_weight,
    const torch::Tensor& o_proj_bias,
    const torch::Tensor& norm_weight,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& down_proj_bias,
    const torch::Tensor& barrier_count,
    const torch::Tensor& barrier_generation,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out,
    double eps,
    bool has_o_proj_bias,
    bool has_down_proj_bias,
    int64_t gate_up_active_sms) {
    TORCH_CHECK(
        residual.is_cuda() && attn_input.is_cuda() && o_proj_weight.is_cuda()
            && norm_weight.is_cuda() && gate_up_weight.is_cuda()
            && down_proj_weight.is_cuda(),
        "inputs must be CUDA");
    TORCH_CHECK(hidden_out.is_cuda() && out.is_cuda(), "outputs must be CUDA");
    TORCH_CHECK(barrier_count.is_cuda() && barrier_generation.is_cuda(),
        "barrier tensors must be CUDA");
    TORCH_CHECK(residual.scalar_type() == torch::kFloat16, "softsync only supports fp16");
    TORCH_CHECK(attn_input.scalar_type() == torch::kFloat16, "softsync only supports fp16");
    TORCH_CHECK(o_proj_weight.scalar_type() == torch::kFloat16, "softsync fp16");
    TORCH_CHECK(norm_weight.scalar_type() == torch::kFloat16, "softsync fp16");
    TORCH_CHECK(gate_up_weight.scalar_type() == torch::kFloat16, "softsync fp16");
    TORCH_CHECK(down_proj_weight.scalar_type() == torch::kFloat16, "softsync fp16");
    TORCH_CHECK(barrier_count.scalar_type() == torch::kInt32, "barrier_count int32");
    TORCH_CHECK(barrier_generation.scalar_type() == torch::kInt32, "barrier_generation int32");
    TORCH_CHECK(residual.dim() == 1 && residual.numel() == kNosplitSpecializedK2048, "residual [2048]");
    TORCH_CHECK(attn_input.dim() == 1 && attn_input.numel() == kNosplitSpecializedK2048, "attn_input [2048]");
    TORCH_CHECK(o_proj_weight.dim() == 2 && o_proj_weight.size(0) == 2048 &&
        o_proj_weight.size(1) == kNosplitSpecializedK2048 && o_proj_weight.stride(1) == 1,
        "o_proj [2048,2048] contiguous");
    TORCH_CHECK(norm_weight.dim() == 1 && norm_weight.numel() == kNosplitSpecializedK2048, "norm_weight [2048]");
    TORCH_CHECK(gate_up_weight.dim() == 2 && gate_up_weight.size(0) == 12288 &&
        gate_up_weight.size(1) == kNosplitSpecializedK2048 && gate_up_weight.stride(1) == 1,
        "gate_up [12288,2048] contiguous");
    TORCH_CHECK(down_proj_weight.dim() == 2 && down_proj_weight.size(0) == 2048 &&
        down_proj_weight.size(1) == 6144 && down_proj_weight.stride(1) == 1,
        "down_proj [2048,6144] contiguous");
    TORCH_CHECK(barrier_count.dim() == 1 && barrier_count.numel() == 1, "barrier_count [1]");
    TORCH_CHECK(barrier_generation.dim() == 1 && barrier_generation.numel() == 1,
        "barrier_generation [1]");
    TORCH_CHECK(hidden_out.dim() == 1 && hidden_out.numel() == 6144, "hidden [6144]");
    TORCH_CHECK(out.dim() == 1 && out.numel() == 2048, "out [2048]");
    TORCH_CHECK(!has_o_proj_bias || (o_proj_bias.is_cuda() && o_proj_bias.scalar_type() == torch::kFloat16 &&
        o_proj_bias.dim() == 1 && o_proj_bias.numel() == 2048 && o_proj_bias.stride(0) == 1),
        "o_proj_bias [2048] contiguous when provided");
    TORCH_CHECK(!has_down_proj_bias || (down_proj_bias.is_cuda() && down_proj_bias.scalar_type() == torch::kFloat16 &&
        down_proj_bias.dim() == 1 && down_proj_bias.numel() == 2048 && down_proj_bias.stride(0) == 1),
        "down_proj_bias [2048] contiguous when provided");

    int device_idx = static_cast<int>(residual.device().index());
    int num_sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_idx));

    int max_blocks_per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        fused_o_proj_post_attn_mlp_softsync_k2048_k6144_fwd_kernel,
        kPersistentThreads,
        0));
    TORCH_CHECK(
        max_blocks_per_sm * num_sms >= num_sms,
        "fused_o_proj_post_attn_mlp_softsync launch would deadlock: resident blocks ",
        max_blocks_per_sm * num_sms,
        " < required grid blocks ",
        num_sms);

    c10::cuda::CUDAGuard device_guard(residual.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto* residual_ptr = residual.data_ptr<at::Half>();
    auto* attn_ptr = attn_input.data_ptr<at::Half>();
    auto* o_w_ptr = o_proj_weight.data_ptr<at::Half>();
    auto* o_b_ptr = has_o_proj_bias ? o_proj_bias.data_ptr<at::Half>() : nullptr;
    auto* norm_ptr = norm_weight.data_ptr<at::Half>();
    auto* gu_ptr = gate_up_weight.data_ptr<at::Half>();
    auto* dp_ptr = down_proj_weight.data_ptr<at::Half>();
    auto* dp_b_ptr = has_down_proj_bias ? down_proj_bias.data_ptr<at::Half>() : nullptr;
    auto* bc_ptr = reinterpret_cast<unsigned int*>(barrier_count.data_ptr<int32_t>());
    auto* bg_ptr = reinterpret_cast<unsigned int*>(barrier_generation.data_ptr<int32_t>());
    auto* h_ptr = hidden_out.data_ptr<at::Half>();
    auto* o_ptr = out.data_ptr<at::Half>();
    float eps_f = static_cast<float>(eps);
    int64_t o_s0 = o_proj_weight.stride(0);
    int64_t gu_s0 = gate_up_weight.stride(0);
    int64_t dp_s0 = down_proj_weight.stride(0);
    int gu_active_sms = static_cast<int>(gate_up_active_sms);

    fused_o_proj_post_attn_mlp_softsync_k2048_k6144_fwd_kernel<<<num_sms, kPersistentThreads, 0, stream>>>(
        residual_ptr,
        attn_ptr,
        o_w_ptr,
        o_b_ptr,
        norm_ptr,
        gu_ptr,
        dp_ptr,
        dp_b_ptr,
        bc_ptr,
        bg_ptr,
        h_ptr,
        o_ptr,
        eps_f,
        has_o_proj_bias,
        has_down_proj_bias,
        o_s0,
        gu_s0,
        dp_s0,
        gu_active_sms);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_o_proj_post_attn_mlp_softsync_profile_half_cuda(
    const torch::Tensor& residual,
    const torch::Tensor& attn_input,
    const torch::Tensor& o_proj_weight,
    const torch::Tensor& o_proj_bias,
    const torch::Tensor& norm_weight,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& down_proj_bias,
    const torch::Tensor& barrier_count,
    const torch::Tensor& barrier_generation,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out,
    const torch::Tensor& profile,
    double eps,
    bool has_o_proj_bias,
    bool has_down_proj_bias,
    int64_t gate_up_active_sms) {
    TORCH_CHECK(
        residual.is_cuda() && attn_input.is_cuda() && o_proj_weight.is_cuda()
            && norm_weight.is_cuda() && gate_up_weight.is_cuda()
            && down_proj_weight.is_cuda(),
        "inputs must be CUDA");
    TORCH_CHECK(hidden_out.is_cuda() && out.is_cuda() && profile.is_cuda(),
        "outputs must be CUDA");
    TORCH_CHECK(barrier_count.is_cuda() && barrier_generation.is_cuda(),
        "barrier tensors must be CUDA");
    TORCH_CHECK(residual.scalar_type() == torch::kFloat16, "softsync_profile fp16");
    TORCH_CHECK(attn_input.scalar_type() == torch::kFloat16, "softsync_profile fp16");
    TORCH_CHECK(o_proj_weight.scalar_type() == torch::kFloat16, "softsync_profile fp16");
    TORCH_CHECK(norm_weight.scalar_type() == torch::kFloat16, "softsync_profile fp16");
    TORCH_CHECK(gate_up_weight.scalar_type() == torch::kFloat16, "softsync_profile fp16");
    TORCH_CHECK(down_proj_weight.scalar_type() == torch::kFloat16, "softsync_profile fp16");
    TORCH_CHECK(barrier_count.scalar_type() == torch::kInt32, "barrier_count int32");
    TORCH_CHECK(barrier_generation.scalar_type() == torch::kInt32, "barrier_generation int32");
    TORCH_CHECK(profile.scalar_type() == torch::kInt64 && profile.is_contiguous(),
        "profile int64 contiguous");
    TORCH_CHECK(residual.dim() == 1 && residual.numel() == kNosplitSpecializedK2048, "residual [2048]");
    TORCH_CHECK(attn_input.dim() == 1 && attn_input.numel() == kNosplitSpecializedK2048, "attn_input [2048]");
    TORCH_CHECK(o_proj_weight.dim() == 2 && o_proj_weight.size(0) == 2048 &&
        o_proj_weight.size(1) == kNosplitSpecializedK2048 && o_proj_weight.stride(1) == 1,
        "o_proj [2048,2048] contiguous");
    TORCH_CHECK(norm_weight.dim() == 1 && norm_weight.numel() == kNosplitSpecializedK2048, "norm_weight [2048]");
    TORCH_CHECK(gate_up_weight.dim() == 2 && gate_up_weight.size(0) == 12288 &&
        gate_up_weight.size(1) == kNosplitSpecializedK2048 && gate_up_weight.stride(1) == 1,
        "gate_up [12288,2048] contiguous");
    TORCH_CHECK(down_proj_weight.dim() == 2 && down_proj_weight.size(0) == 2048 &&
        down_proj_weight.size(1) == 6144 && down_proj_weight.stride(1) == 1,
        "down_proj [2048,6144] contiguous");
    TORCH_CHECK(barrier_count.dim() == 1 && barrier_count.numel() == 1, "barrier_count [1]");
    TORCH_CHECK(barrier_generation.dim() == 1 && barrier_generation.numel() == 1,
        "barrier_generation [1]");
    TORCH_CHECK(hidden_out.dim() == 1 && hidden_out.numel() == 6144, "hidden [6144]");
    TORCH_CHECK(out.dim() == 1 && out.numel() == 2048, "out [2048]");
    TORCH_CHECK(!has_o_proj_bias || (o_proj_bias.is_cuda() && o_proj_bias.scalar_type() == torch::kFloat16 &&
        o_proj_bias.dim() == 1 && o_proj_bias.numel() == 2048 && o_proj_bias.stride(0) == 1),
        "o_proj_bias [2048] contiguous when provided");
    TORCH_CHECK(!has_down_proj_bias || (down_proj_bias.is_cuda() && down_proj_bias.scalar_type() == torch::kFloat16 &&
        down_proj_bias.dim() == 1 && down_proj_bias.numel() == 2048 && down_proj_bias.stride(0) == 1),
        "down_proj_bias [2048] contiguous when provided");

    int device_idx = static_cast<int>(residual.device().index());
    int num_sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_idx));

    TORCH_CHECK(profile.numel() == static_cast<int64_t>(num_sms) * kOProjMlpProfileSlots,
        "profile shape mismatch: expected ", num_sms * kOProjMlpProfileSlots,
        " got ", profile.numel());

    int max_blocks_per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        fused_o_proj_post_attn_mlp_softsync_profile_k2048_k6144_fwd_kernel<true>,
        kPersistentThreads,
        0));
    TORCH_CHECK(
        max_blocks_per_sm * num_sms >= num_sms,
        "fused_o_proj_post_attn_mlp_softsync_profile launch would deadlock: resident blocks ",
        max_blocks_per_sm * num_sms,
        " < required grid blocks ",
        num_sms);

    c10::cuda::CUDAGuard device_guard(residual.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    profile.zero_();

    auto* residual_ptr = residual.data_ptr<at::Half>();
    auto* attn_ptr = attn_input.data_ptr<at::Half>();
    auto* o_w_ptr = o_proj_weight.data_ptr<at::Half>();
    auto* o_b_ptr = has_o_proj_bias ? o_proj_bias.data_ptr<at::Half>() : nullptr;
    auto* norm_ptr = norm_weight.data_ptr<at::Half>();
    auto* gu_ptr = gate_up_weight.data_ptr<at::Half>();
    auto* dp_ptr = down_proj_weight.data_ptr<at::Half>();
    auto* dp_b_ptr = has_down_proj_bias ? down_proj_bias.data_ptr<at::Half>() : nullptr;
    auto* bc_ptr = reinterpret_cast<unsigned int*>(barrier_count.data_ptr<int32_t>());
    auto* bg_ptr = reinterpret_cast<unsigned int*>(barrier_generation.data_ptr<int32_t>());
    auto* h_ptr = hidden_out.data_ptr<at::Half>();
    auto* o_ptr = out.data_ptr<at::Half>();
    auto* p_ptr = reinterpret_cast<uint64_t*>(profile.data_ptr<int64_t>());
    float eps_f = static_cast<float>(eps);
    int64_t o_s0 = o_proj_weight.stride(0);
    int64_t gu_s0 = gate_up_weight.stride(0);
    int64_t dp_s0 = down_proj_weight.stride(0);
    int gu_active_sms = static_cast<int>(gate_up_active_sms);

    fused_o_proj_post_attn_mlp_softsync_profile_k2048_k6144_fwd_kernel<true><<<num_sms, kPersistentThreads, 0, stream>>>(
        residual_ptr,
        attn_ptr,
        o_w_ptr,
        o_b_ptr,
        norm_ptr,
        gu_ptr,
        dp_ptr,
        dp_b_ptr,
        bc_ptr,
        bg_ptr,
        h_ptr,
        o_ptr,
        p_ptr,
        eps_f,
        has_o_proj_bias,
        has_down_proj_bias,
        o_s0,
        gu_s0,
        dp_s0,
        gu_active_sms);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_gateup_silu_downproj_persistent_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out) {
    TORCH_CHECK(x.is_cuda() && gate_up_weight.is_cuda() && down_proj_weight.is_cuda(),
        "inputs must be CUDA");
    TORCH_CHECK(hidden_out.is_cuda() && out.is_cuda(), "outputs must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "persistent only supports fp16");
    TORCH_CHECK(gate_up_weight.scalar_type() == torch::kFloat16, "persistent fp16");
    TORCH_CHECK(down_proj_weight.scalar_type() == torch::kFloat16, "persistent fp16");
    TORCH_CHECK(x.dim() == 1 && x.numel() == kNosplitSpecializedK2048, "x [2048]");
    TORCH_CHECK(gate_up_weight.dim() == 2 && gate_up_weight.size(0) == 12288 &&
        gate_up_weight.size(1) == kNosplitSpecializedK2048 && gate_up_weight.stride(1) == 1,
        "gate_up [12288,2048] contiguous");
    TORCH_CHECK(down_proj_weight.dim() == 2 && down_proj_weight.size(0) == 2048 &&
        down_proj_weight.size(1) == 6144 && down_proj_weight.stride(1) == 1,
        "down_proj [2048,6144] contiguous");
    TORCH_CHECK(hidden_out.dim() == 1 && hidden_out.numel() == 6144, "hidden [6144]");
    TORCH_CHECK(out.dim() == 1 && out.numel() == 2048, "out [2048]");

    int device_idx = static_cast<int>(x.device().index());
    int num_sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_idx));

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto* x_ptr = x.data_ptr<at::Half>();
    auto* gu_ptr = gate_up_weight.data_ptr<at::Half>();
    auto* dp_ptr = down_proj_weight.data_ptr<at::Half>();
    auto* h_ptr = hidden_out.data_ptr<at::Half>();
    auto* o_ptr = out.data_ptr<at::Half>();
    int64_t gu_s0 = gate_up_weight.stride(0);
    int64_t dp_s0 = down_proj_weight.stride(0);

    void* args[] = {&x_ptr, &gu_ptr, &dp_ptr, &h_ptr, &o_ptr, &gu_s0, &dp_s0};

    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(
            fused_gateup_silu_downproj_persistent_k2048_k6144_fwd_kernel),
        num_sms, kPersistentThreads, args, 0, stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_gateup_silu_downproj_persistent_profile_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out,
    const torch::Tensor& profile) {
    TORCH_CHECK(x.is_cuda() && gate_up_weight.is_cuda() && down_proj_weight.is_cuda(),
        "inputs must be CUDA");
    TORCH_CHECK(hidden_out.is_cuda() && out.is_cuda() && profile.is_cuda(),
        "outputs must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "persistent_profile fp16");
    TORCH_CHECK(gate_up_weight.scalar_type() == torch::kFloat16, "persistent_profile fp16");
    TORCH_CHECK(down_proj_weight.scalar_type() == torch::kFloat16, "persistent_profile fp16");
    TORCH_CHECK(profile.scalar_type() == torch::kInt64 && profile.is_contiguous(),
        "profile int64 contiguous");
    TORCH_CHECK(x.dim() == 1 && x.numel() == kNosplitSpecializedK2048, "x [2048]");
    TORCH_CHECK(gate_up_weight.dim() == 2 && gate_up_weight.size(0) == 12288 &&
        gate_up_weight.size(1) == kNosplitSpecializedK2048 && gate_up_weight.stride(1) == 1,
        "gate_up [12288,2048] contiguous");
    TORCH_CHECK(down_proj_weight.dim() == 2 && down_proj_weight.size(0) == 2048 &&
        down_proj_weight.size(1) == 6144 && down_proj_weight.stride(1) == 1,
        "down_proj [2048,6144] contiguous");
    TORCH_CHECK(hidden_out.dim() == 1 && hidden_out.numel() == 6144, "hidden [6144]");
    TORCH_CHECK(out.dim() == 1 && out.numel() == 2048, "out [2048]");

    int device_idx = static_cast<int>(x.device().index());
    int num_sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_idx));

    TORCH_CHECK(profile.numel() == static_cast<int64_t>(num_sms) * kPersistentProfileSlots,
        "profile shape mismatch: expected ", num_sms * kPersistentProfileSlots,
        " got ", profile.numel());

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    profile.zero_();

    auto* x_ptr = x.data_ptr<at::Half>();
    auto* gu_ptr = gate_up_weight.data_ptr<at::Half>();
    auto* dp_ptr = down_proj_weight.data_ptr<at::Half>();
    auto* h_ptr = hidden_out.data_ptr<at::Half>();
    auto* o_ptr = out.data_ptr<at::Half>();
    auto* p_ptr = reinterpret_cast<uint64_t*>(profile.data_ptr<int64_t>());
    int64_t gu_s0 = gate_up_weight.stride(0);
    int64_t dp_s0 = down_proj_weight.stride(0);

    void* args[] = {&x_ptr, &gu_ptr, &dp_ptr, &h_ptr, &o_ptr, &p_ptr, &gu_s0, &dp_s0};

    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(
            fused_gateup_silu_downproj_persistent_profile_k2048_k6144_fwd_kernel<true>),
        num_sms, kPersistentThreads, args, 0, stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
