#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <cuda_fp16.h>

namespace {

constexpr int kThreads = 32;
constexpr int kRow4Threads = 128;
constexpr int kRow8Threads = 256;
constexpr int kSharedTileK = 256;
constexpr int kVecTileK = 256;

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
    const int grid_blocks = (n + 3) / 4;

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

    const int grid_blocks = (n + 3) / 4;

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
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
