// Separate TU for cooperative-groups grid-sync kernels.
// Must be compiled with -rdc=true (relocatable device code).

#include <cooperative_groups.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <cuda_fp16.h>

namespace cg = cooperative_groups;

namespace {

// Constants duplicated from decode_gemv_kernel.cu (they live in an anonymous
// namespace there, so we must re-declare them here).
constexpr int kRow8Threads = 256;
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

__device__ __forceinline__ uint64_t read_globaltimer() {
    uint64_t ts;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts));
    return ts;
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

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Host-side launch wrappers (linked from decode_gemv_ext.cpp)
// ---------------------------------------------------------------------------

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
