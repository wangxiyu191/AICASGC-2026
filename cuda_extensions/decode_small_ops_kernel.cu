#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

constexpr int kWarpSize = 32;
constexpr int kHeadDim = 128;
constexpr int kHalfHeadDim = kHeadDim / 2;
constexpr int kSupportedGroups = 2;
constexpr int kWarpsPerBlock = 3;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr int kElementwiseThreads = 256;
constexpr int kReductionThreads = 256;

__device__ inline float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ inline __half rope_out(
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

__device__ inline void fused_rms_norm_rope_one_head(
    const __half* __restrict__ x,
    const __half* __restrict__ weight,
    const __half* __restrict__ cos,
    const __half* __restrict__ sin,
    __half* __restrict__ out,
    float eps,
    int lane) {
    float sum_sq = 0.0f;
    #pragma unroll
    for (int idx = lane; idx < kHeadDim; idx += kWarpSize) {
        const float v = __half2float(x[idx]);
        sum_sq += v * v;
    }
    sum_sq = warp_sum(sum_sq);
    const float rstd = rsqrtf(__shfl_sync(0xffffffff, sum_sq / static_cast<float>(kHeadDim) + eps, 0));

    for (int rk = lane; rk < kHalfHeadDim; rk += kWarpSize) {
        const float n0f = __half2float(x[rk]) * rstd * __half2float(weight[rk]);
        const float n1f = __half2float(x[kHalfHeadDim + rk]) * rstd * __half2float(weight[kHalfHeadDim + rk]);
        const __half n0 = __float2half_rn(n0f);
        const __half n1 = __float2half_rn(n1f);
        const __half c = cos[rk];
        const __half s = sin[rk];
        out[rk] = rope_out(n0, n1, c, s, false);
        out[kHalfHeadDim + rk] = rope_out(n0, n1, c, s, true);
    }
}

__global__ void fused_qk_rms_norm_rope_fwd_kernel(
    const at::Half* __restrict__ q,
    const at::Half* __restrict__ k,
    const at::Half* __restrict__ q_weight,
    const at::Half* __restrict__ k_weight,
    const at::Half* __restrict__ cos,
    const at::Half* __restrict__ sin,
    at::Half* __restrict__ q_out,
    at::Half* __restrict__ k_out,
    int64_t q_stride_m,
    int64_t q_stride_h,
    int64_t k_stride_m,
    int64_t k_stride_h,
    int64_t cos_stride_m,
    int kv_heads,
    float eps) {
    const int group_idx = blockIdx.x;
    const int m = group_idx / kv_heads;
    const int kv_h = group_idx % kv_heads;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const __half* cos_row = reinterpret_cast<const __half*>(cos + m * cos_stride_m);
    const __half* sin_row = reinterpret_cast<const __half*>(sin + m * cos_stride_m);
    const __half* q_weight_half = reinterpret_cast<const __half*>(q_weight);
    const __half* k_weight_half = reinterpret_cast<const __half*>(k_weight);
    const __half* q_half = reinterpret_cast<const __half*>(q);
    const __half* k_half = reinterpret_cast<const __half*>(k);
    __half* q_out_half = reinterpret_cast<__half*>(q_out);
    __half* k_out_half = reinterpret_cast<__half*>(k_out);

    if (warp == 0) {
        const __half* k_in = k_half + m * k_stride_m + kv_h * k_stride_h;
        __half* k_dst = k_out_half + m * k_stride_m + kv_h * k_stride_h;
        fused_rms_norm_rope_one_head(k_in, k_weight_half, cos_row, sin_row, k_dst, eps, lane);
        return;
    }

    const int q_group = warp - 1;
    if (q_group < kSupportedGroups) {
        const int q_h = kv_h * kSupportedGroups + q_group;
        const __half* q_in = q_half + m * q_stride_m + q_h * q_stride_h;
        __half* q_dst = q_out_half + m * q_stride_m + q_h * q_stride_h;
        fused_rms_norm_rope_one_head(q_in, q_weight_half, cos_row, sin_row, q_dst, eps, lane);
    }
}

}  // namespace

void fused_qk_rms_norm_rope_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& q_weight,
    const torch::Tensor& k_weight,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& q_out,
    const torch::Tensor& k_out,
    double eps) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(k.is_cuda(), "k must be CUDA");
    TORCH_CHECK(q_weight.is_cuda(), "q_weight must be CUDA");
    TORCH_CHECK(k_weight.is_cuda(), "k_weight must be CUDA");
    TORCH_CHECK(cos.is_cuda(), "cos must be CUDA");
    TORCH_CHECK(sin.is_cuda(), "sin must be CUDA");
    TORCH_CHECK(q_out.is_cuda(), "q_out must be CUDA");
    TORCH_CHECK(k_out.is_cuda(), "k_out must be CUDA");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16, "q must be fp16");
    TORCH_CHECK(k.scalar_type() == torch::kFloat16, "k must be fp16");
    TORCH_CHECK(q_weight.scalar_type() == torch::kFloat16, "q_weight must be fp16");
    TORCH_CHECK(k_weight.scalar_type() == torch::kFloat16, "k_weight must be fp16");
    TORCH_CHECK(cos.scalar_type() == torch::kFloat16, "cos must be fp16");
    TORCH_CHECK(sin.scalar_type() == torch::kFloat16, "sin must be fp16");
    TORCH_CHECK(q_out.scalar_type() == torch::kFloat16, "q_out must be fp16");
    TORCH_CHECK(k_out.scalar_type() == torch::kFloat16, "k_out must be fp16");
    TORCH_CHECK(q.dim() == 3, "q must be [M, QH, D]");
    TORCH_CHECK(k.dim() == 3, "k must be [M, KVH, D]");
    TORCH_CHECK(cos.dim() == 2, "cos must be [M, D]");
    TORCH_CHECK(sin.dim() == 2, "sin must be [M, D]");
    TORCH_CHECK(q.size(0) == k.size(0), "q/k M mismatch");
    TORCH_CHECK(q.size(2) == kHeadDim, "q head_dim must be 128");
    TORCH_CHECK(k.size(2) == kHeadDim, "k head_dim must be 128");
    TORCH_CHECK(cos.size(0) == q.size(0) && cos.size(1) == kHeadDim, "cos shape mismatch");
    TORCH_CHECK(sin.size(0) == q.size(0) && sin.size(1) == kHeadDim, "sin shape mismatch");
    TORCH_CHECK(q.size(1) == k.size(1) * kSupportedGroups, "q_heads must equal 2 * kv_heads");
    TORCH_CHECK(q_weight.numel() == kHeadDim, "q_weight shape mismatch");
    TORCH_CHECK(k_weight.numel() == kHeadDim, "k_weight shape mismatch");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(cos.is_contiguous(), "cos must be contiguous");
    TORCH_CHECK(sin.is_contiguous(), "sin must be contiguous");
    TORCH_CHECK(q_weight.is_contiguous(), "q_weight must be contiguous");
    TORCH_CHECK(k_weight.is_contiguous(), "k_weight must be contiguous");
    TORCH_CHECK(q_out.is_contiguous(), "q_out must be contiguous");
    TORCH_CHECK(k_out.is_contiguous(), "k_out must be contiguous");
    TORCH_CHECK(q_out.sizes() == q.sizes(), "q_out shape mismatch");
    TORCH_CHECK(k_out.sizes() == k.sizes(), "k_out shape mismatch");

    c10::cuda::CUDAGuard device_guard(q.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int m = static_cast<int>(q.size(0));
    const int kv_heads = static_cast<int>(k.size(1));
    const dim3 grid(m * kv_heads);
    fused_qk_rms_norm_rope_fwd_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const at::Half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const at::Half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const at::Half*>(q_weight.data_ptr<at::Half>()),
        reinterpret_cast<const at::Half*>(k_weight.data_ptr<at::Half>()),
        reinterpret_cast<const at::Half*>(cos.data_ptr<at::Half>()),
        reinterpret_cast<const at::Half*>(sin.data_ptr<at::Half>()),
        reinterpret_cast<at::Half*>(q_out.data_ptr<at::Half>()),
        reinterpret_cast<at::Half*>(k_out.data_ptr<at::Half>()),
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        cos.stride(0),
        kv_heads,
        static_cast<float>(eps));
    C10_CUDA_CHECK(cudaGetLastError());
}

__global__ void fused_silu_mul_fwd_kernel(
    const at::Half* __restrict__ gate,
    const at::Half* __restrict__ up,
    at::Half* __restrict__ out,
    int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const float gate_f = __half2float(reinterpret_cast<const __half*>(gate)[idx]);
    const float up_f = __half2float(reinterpret_cast<const __half*>(up)[idx]);
    const float silu = gate_f / (1.0f + expf(-gate_f));
    reinterpret_cast<__half*>(out)[idx] = __float2half_rn(silu * up_f);
}

void fused_silu_mul_cuda(
    const torch::Tensor& gate,
    const torch::Tensor& up,
    const torch::Tensor& out) {
    TORCH_CHECK(gate.is_cuda(), "gate must be CUDA");
    TORCH_CHECK(up.is_cuda(), "up must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(gate.scalar_type() == torch::kFloat16, "gate must be fp16");
    TORCH_CHECK(up.scalar_type() == torch::kFloat16, "up must be fp16");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(gate.sizes() == up.sizes(), "gate/up shape mismatch");
    TORCH_CHECK(gate.sizes() == out.sizes(), "gate/out shape mismatch");
    TORCH_CHECK(gate.is_contiguous(), "gate must be contiguous");
    TORCH_CHECK(up.is_contiguous(), "up must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    c10::cuda::CUDAGuard device_guard(gate.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int total = static_cast<int>(gate.numel());
    const dim3 block(kElementwiseThreads);
    const dim3 grid((total + block.x - 1) / block.x);
    fused_silu_mul_fwd_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const at::Half*>(gate.data_ptr<at::Half>()),
        reinterpret_cast<const at::Half*>(up.data_ptr<at::Half>()),
        reinterpret_cast<at::Half*>(out.data_ptr<at::Half>()),
        total);
    C10_CUDA_CHECK(cudaGetLastError());
}

__global__ void fused_add_rms_norm_fwd_kernel(
    const at::Half* __restrict__ residual,
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    at::Half* __restrict__ out_sum,
    at::Half* __restrict__ out_norm,
    int cols,
    float eps) {
    __shared__ float warp_sums[kReductionThreads / kWarpSize];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row_base = row * cols;

    float sum_sq = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        const float summed =
            __half2float(reinterpret_cast<const __half*>(residual)[row_base + col]) +
            __half2float(reinterpret_cast<const __half*>(x)[row_base + col]);
        sum_sq += summed * summed;
        reinterpret_cast<__half*>(out_sum)[row_base + col] = __float2half_rn(summed);
    }

    sum_sq = warp_sum(sum_sq);
    if (lane == 0) {
        warp_sums[warp] = sum_sq;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (warp == 0) {
        block_sum = (lane < (kReductionThreads / kWarpSize)) ? warp_sums[lane] : 0.0f;
        block_sum = warp_sum(block_sum);
        if (lane == 0) {
            warp_sums[0] = rsqrtf(block_sum / static_cast<float>(cols) + eps);
        }
    }
    __syncthreads();
    const float rstd = warp_sums[0];

    for (int col = tid; col < cols; col += blockDim.x) {
        const float summed = __half2float(reinterpret_cast<const __half*>(out_sum)[row_base + col]);
        const float w = __half2float(reinterpret_cast<const __half*>(weight)[col]);
        reinterpret_cast<__half*>(out_norm)[row_base + col] = __float2half_rn(summed * rstd * w);
    }
}

__global__ void rms_norm_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    at::Half* __restrict__ out,
    int cols,
    float eps) {
    __shared__ float warp_sums[kReductionThreads / kWarpSize];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row_base = row * cols;

    float sum_sq = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        const float v = __half2float(reinterpret_cast<const __half*>(x)[row_base + col]);
        sum_sq += v * v;
    }

    sum_sq = warp_sum(sum_sq);
    if (lane == 0) {
        warp_sums[warp] = sum_sq;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (warp == 0) {
        block_sum = (lane < (kReductionThreads / kWarpSize)) ? warp_sums[lane] : 0.0f;
        block_sum = warp_sum(block_sum);
        if (lane == 0) {
            warp_sums[0] = rsqrtf(block_sum / static_cast<float>(cols) + eps);
        }
    }
    __syncthreads();
    const float rstd = warp_sums[0];

    for (int col = tid; col < cols; col += blockDim.x) {
        const float v = __half2float(reinterpret_cast<const __half*>(x)[row_base + col]);
        const float w = __half2float(reinterpret_cast<const __half*>(weight)[col]);
        reinterpret_cast<__half*>(out)[row_base + col] = __float2half_rn(v * rstd * w);
    }
}

void fused_add_rms_norm_cuda(
    const torch::Tensor& residual,
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& out_sum,
    const torch::Tensor& out_norm,
    double eps) {
    TORCH_CHECK(residual.is_cuda(), "residual must be CUDA");
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out_sum.is_cuda(), "out_sum must be CUDA");
    TORCH_CHECK(out_norm.is_cuda(), "out_norm must be CUDA");
    TORCH_CHECK(residual.scalar_type() == torch::kFloat16, "residual must be fp16");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "weight must be fp16");
    TORCH_CHECK(out_sum.scalar_type() == torch::kFloat16, "out_sum must be fp16");
    TORCH_CHECK(out_norm.scalar_type() == torch::kFloat16, "out_norm must be fp16");
    TORCH_CHECK(residual.dim() == 2, "residual must be [M, N]");
    TORCH_CHECK(x.dim() == 2, "x must be [M, N]");
    TORCH_CHECK(weight.dim() == 1, "weight must be [N]");
    TORCH_CHECK(residual.sizes() == x.sizes(), "residual/x shape mismatch");
    TORCH_CHECK(residual.sizes() == out_sum.sizes(), "residual/out_sum shape mismatch");
    TORCH_CHECK(residual.sizes() == out_norm.sizes(), "residual/out_norm shape mismatch");
    TORCH_CHECK(residual.size(1) == weight.size(0), "weight size mismatch");
    TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(out_sum.is_contiguous(), "out_sum must be contiguous");
    TORCH_CHECK(out_norm.is_contiguous(), "out_norm must be contiguous");

    c10::cuda::CUDAGuard device_guard(residual.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int rows = static_cast<int>(residual.size(0));
    const int cols = static_cast<int>(residual.size(1));
    fused_add_rms_norm_fwd_kernel<<<rows, kReductionThreads, 0, stream>>>(
        reinterpret_cast<const at::Half*>(residual.data_ptr<at::Half>()),
        reinterpret_cast<const at::Half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const at::Half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<at::Half*>(out_sum.data_ptr<at::Half>()),
        reinterpret_cast<at::Half*>(out_norm.data_ptr<at::Half>()),
        cols,
        static_cast<float>(eps));
    C10_CUDA_CHECK(cudaGetLastError());
}

void rms_norm_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& out,
    double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "x must be fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "weight must be fp16");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(x.dim() == 2, "x must be [M, N]");
    TORCH_CHECK(weight.dim() == 1, "weight must be [N]");
    TORCH_CHECK(x.sizes() == out.sizes(), "x/out shape mismatch");
    TORCH_CHECK(x.size(1) == weight.size(0), "weight size mismatch");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int rows = static_cast<int>(x.size(0));
    const int cols = static_cast<int>(x.size(1));
    rms_norm_fwd_kernel<<<rows, kReductionThreads, 0, stream>>>(
        reinterpret_cast<const at::Half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const at::Half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<at::Half*>(out.data_ptr<at::Half>()),
        cols,
        static_cast<float>(eps));
    C10_CUDA_CHECK(cudaGetLastError());
}

__global__ void add_fwd_kernel(
    const at::Half* __restrict__ a,
    const at::Half* __restrict__ b,
    at::Half* __restrict__ out,
    int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const float av = __half2float(reinterpret_cast<const __half*>(a)[idx]);
    const float bv = __half2float(reinterpret_cast<const __half*>(b)[idx]);
    reinterpret_cast<__half*>(out)[idx] = __float2half_rn(av + bv);
}

void add_cuda(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& out) {
    TORCH_CHECK(a.is_cuda(), "a must be CUDA");
    TORCH_CHECK(b.is_cuda(), "b must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(a.scalar_type() == torch::kFloat16, "a must be fp16");
    TORCH_CHECK(b.scalar_type() == torch::kFloat16, "b must be fp16");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(a.sizes() == b.sizes(), "a/b shape mismatch");
    TORCH_CHECK(a.sizes() == out.sizes(), "a/out shape mismatch");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    c10::cuda::CUDAGuard device_guard(a.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int total = static_cast<int>(a.numel());
    const dim3 block(kElementwiseThreads);
    const dim3 grid((total + block.x - 1) / block.x);
    add_fwd_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const at::Half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const at::Half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<at::Half*>(out.data_ptr<at::Half>()),
        total);
    C10_CUDA_CHECK(cudaGetLastError());
}
