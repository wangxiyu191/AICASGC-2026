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
