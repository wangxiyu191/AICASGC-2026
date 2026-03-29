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
