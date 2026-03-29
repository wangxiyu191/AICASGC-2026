#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

constexpr int kForwardThreads = 64;
constexpr int kQHeadForwardThreads = 32;
constexpr int kMergeThreads = 128;
constexpr float kNegLarge = -1.0e20f;

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
__global__ void grouped_prefix_cachepos_splitkv_fwd_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int64_t* __restrict__ cache_position,
    float* __restrict__ partial_acc,
    float* __restrict__ partial_stats,
    int64_t q_stride_h,
    int64_t k_stride_h,
    int64_t k_stride_s,
    int64_t v_stride_h,
    int64_t v_stride_s,
    int64_t pa_stride_kh,
    int64_t pa_stride_split,
    int64_t pa_stride_head,
    int64_t ps_stride_kh,
    int64_t ps_stride_split,
    int64_t ps_stride_head,
    int64_t ps_stride_stat,
    float scale,
    int head_dim,
    int max_seq_len,
    int split_k) {
    const int kv_head_idx = blockIdx.x;
    const int split_idx = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (warp >= 2) {
        return;
    }

    const int valid_len = min(static_cast<int>(cache_position[0] + 1), max_seq_len);
    const int chunk_size = (valid_len + split_k - 1) / split_k;
    const int chunk_start = split_idx * chunk_size;
    const int chunk_end = min(chunk_start + chunk_size, valid_len);

    const int q_head_idx = kv_head_idx * 2 + warp;
    const int dim_base = lane * 4;
    float q_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = dim_base + i;
        if (d < head_dim) {
            q_reg[i] = to_float(q[q_head_idx * q_stride_h + d]);
        }
    }

    float m = kNegLarge;
    float l = 0.0f;

    for (int token_idx = chunk_start; token_idx < chunk_end; ++token_idx) {
        float dot = 0.0f;
        float v_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int d = dim_base + i;
            if (d < head_dim) {
                const float kval = to_float(k[kv_head_idx * k_stride_h + token_idx * k_stride_s + d]);
                dot += q_reg[i] * kval;
                v_reg[i] = to_float(v[kv_head_idx * v_stride_h + token_idx * v_stride_s + d]);
            }
        }
        float score = warp_sum(dot) * scale;
        score = __shfl_sync(0xffffffff, score, 0);
        const float next_m = fmaxf(m, score);
        const float alpha = expf(m - next_m);
        const float p = expf(score - next_m);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc_reg[i] = acc_reg[i] * alpha + p * v_reg[i];
        }
        l = l * alpha + p;
        m = next_m;
    }

    const int64_t pa_base = kv_head_idx * pa_stride_kh + split_idx * pa_stride_split;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = dim_base + i;
        if (d < head_dim) {
            partial_acc[pa_base + warp * pa_stride_head + d] = acc_reg[i];
        }
    }

    if (lane == 0) {
        const int64_t ps_base = kv_head_idx * ps_stride_kh + split_idx * ps_stride_split;
        partial_stats[ps_base + warp * ps_stride_head + 0 * ps_stride_stat] = l;
        partial_stats[ps_base + warp * ps_stride_head + 1 * ps_stride_stat] = m;
    }
}

template <typename scalar_t>
__global__ void grouped_prefix_cachepos_splitkv_merge_kernel(
    const float* __restrict__ partial_acc,
    const float* __restrict__ partial_stats,
    scalar_t* __restrict__ out,
    int64_t pa_stride_kh,
    int64_t pa_stride_split,
    int64_t pa_stride_head,
    int64_t ps_stride_kh,
    int64_t ps_stride_split,
    int64_t ps_stride_head,
    int64_t ps_stride_stat,
    int64_t out_stride_h,
    int split_k,
    int head_dim) {
    const int kv_head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= head_dim) {
        return;
    }

    float max0 = kNegLarge;
    float max1 = kNegLarge;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int split_idx = 0; split_idx < split_k; ++split_idx) {
        const int64_t ps_base = kv_head_idx * ps_stride_kh + split_idx * ps_stride_split;
        const int64_t pa_base = kv_head_idx * pa_stride_kh + split_idx * pa_stride_split;
        const float l0 = partial_stats[ps_base + 0 * ps_stride_head + 0 * ps_stride_stat];
        const float m0 = partial_stats[ps_base + 0 * ps_stride_head + 1 * ps_stride_stat];
        const float l1 = partial_stats[ps_base + 1 * ps_stride_head + 0 * ps_stride_stat];
        const float m1 = partial_stats[ps_base + 1 * ps_stride_head + 1 * ps_stride_stat];
        const float part_acc0 = partial_acc[pa_base + 0 * pa_stride_head + tid];
        const float part_acc1 = partial_acc[pa_base + 1 * pa_stride_head + tid];

        const float next_max0 = fmaxf(max0, m0);
        const float next_max1 = fmaxf(max1, m1);
        const float prev_scale0 = expf(max0 - next_max0);
        const float prev_scale1 = expf(max1 - next_max1);
        const float part_scale0 = expf(m0 - next_max0);
        const float part_scale1 = expf(m1 - next_max1);
        acc0 = acc0 * prev_scale0 + part_acc0 * part_scale0;
        acc1 = acc1 * prev_scale1 + part_acc1 * part_scale1;
        sum0 = sum0 * prev_scale0 + l0 * part_scale0;
        sum1 = sum1 * prev_scale1 + l1 * part_scale1;
        max0 = next_max0;
        max1 = next_max1;
    }

    const int q_head_idx0 = kv_head_idx * 2;
    const int q_head_idx1 = q_head_idx0 + 1;
    out[q_head_idx0 * out_stride_h + tid] = static_cast<scalar_t>(acc0 / sum0);
    out[q_head_idx1 * out_stride_h + tid] = static_cast<scalar_t>(acc1 / sum1);
}

template <typename scalar_t>
__global__ void grouped_prefix_cachepos_splitkv_qhead_fwd_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int64_t* __restrict__ cache_position,
    float* __restrict__ partial_acc,
    float* __restrict__ partial_stats,
    int64_t q_stride_h,
    int64_t k_stride_h,
    int64_t k_stride_s,
    int64_t v_stride_h,
    int64_t v_stride_s,
    int64_t pa_stride_h,
    int64_t pa_stride_split,
    int64_t ps_stride_h,
    int64_t ps_stride_split,
    int64_t ps_stride_stat,
    float scale,
    int head_dim,
    int max_seq_len,
    int split_k) {
    const int q_head_idx = blockIdx.x;
    const int split_idx = blockIdx.y;
    const int kv_head_idx = q_head_idx >> 1;
    const int lane = threadIdx.x;

    const int valid_len = min(static_cast<int>(cache_position[0] + 1), max_seq_len);
    const int chunk_size = (valid_len + split_k - 1) / split_k;
    const int chunk_start = split_idx * chunk_size;
    const int chunk_end = min(chunk_start + chunk_size, valid_len);

    const int dim_base = lane * 4;
    float q_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = dim_base + i;
        if (d < head_dim) {
            q_reg[i] = to_float(q[q_head_idx * q_stride_h + d]);
        }
    }

    float m = kNegLarge;
    float l = 0.0f;
    for (int token_idx = chunk_start; token_idx < chunk_end; ++token_idx) {
        float dot = 0.0f;
        float v_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int d = dim_base + i;
            if (d < head_dim) {
                const float kval = to_float(k[kv_head_idx * k_stride_h + token_idx * k_stride_s + d]);
                dot += q_reg[i] * kval;
                v_reg[i] = to_float(v[kv_head_idx * v_stride_h + token_idx * v_stride_s + d]);
            }
        }
        float score = warp_sum(dot) * scale;
        score = __shfl_sync(0xffffffff, score, 0);
        const float next_m = fmaxf(m, score);
        const float alpha = expf(m - next_m);
        const float p = expf(score - next_m);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc_reg[i] = acc_reg[i] * alpha + p * v_reg[i];
        }
        l = l * alpha + p;
        m = next_m;
    }

    const int64_t pa_base = q_head_idx * pa_stride_h + split_idx * pa_stride_split;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = dim_base + i;
        if (d < head_dim) {
            partial_acc[pa_base + d] = acc_reg[i];
        }
    }
    if (lane == 0) {
        const int64_t ps_base = q_head_idx * ps_stride_h + split_idx * ps_stride_split;
        partial_stats[ps_base + 0 * ps_stride_stat] = l;
        partial_stats[ps_base + 1 * ps_stride_stat] = m;
    }
}

template <typename scalar_t>
__global__ void grouped_prefix_cachepos_splitkv_qhead_merge_kernel(
    const float* __restrict__ partial_acc,
    const float* __restrict__ partial_stats,
    scalar_t* __restrict__ out,
    int64_t pa_stride_h,
    int64_t pa_stride_split,
    int64_t ps_stride_h,
    int64_t ps_stride_split,
    int64_t ps_stride_stat,
    int64_t out_stride_h,
    int split_k,
    int head_dim) {
    const int q_head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= head_dim) {
        return;
    }

    float max_v = kNegLarge;
    float sum_v = 0.0f;
    float acc_v = 0.0f;
    for (int split_idx = 0; split_idx < split_k; ++split_idx) {
        const int64_t ps_base = q_head_idx * ps_stride_h + split_idx * ps_stride_split;
        const int64_t pa_base = q_head_idx * pa_stride_h + split_idx * pa_stride_split;
        const float l = partial_stats[ps_base + 0 * ps_stride_stat];
        const float m = partial_stats[ps_base + 1 * ps_stride_stat];
        const float part_acc = partial_acc[pa_base + tid];
        const float next_max = fmaxf(max_v, m);
        const float prev_scale = expf(max_v - next_max);
        const float part_scale = expf(m - next_max);
        acc_v = acc_v * prev_scale + part_acc * part_scale;
        sum_v = sum_v * prev_scale + l * part_scale;
        max_v = next_max;
    }
    out[q_head_idx * out_stride_h + tid] = static_cast<scalar_t>(acc_v / sum_v);
}

}  // namespace

torch::Tensor grouped_prefix_cachepos_splitkv_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cache_position,
    double scale,
    const torch::Tensor& partial_acc,
    const torch::Tensor& partial_stats,
    const torch::Tensor& out) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(k.is_cuda(), "k must be CUDA");
    TORCH_CHECK(v.is_cuda(), "v must be CUDA");
    TORCH_CHECK(cache_position.is_cuda(), "cache_position must be CUDA");
    TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q/k dtype mismatch");
    TORCH_CHECK(q.scalar_type() == v.scalar_type(), "q/v dtype mismatch");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be 4D");
    TORCH_CHECK(q.size(0) == 1 && q.size(2) == 1, "expected q shape [1, heads, 1, head_dim]");
    TORCH_CHECK(k.size(0) == 1 && v.size(0) == 1, "expected k/v batch size 1");
    TORCH_CHECK(q.size(1) == k.size(1) * 2, "expected q heads = kv heads * 2");
    TORCH_CHECK(q.size(3) <= 128, "head_dim > 128 not supported");
    TORCH_CHECK(partial_acc.scalar_type() == torch::kFloat32, "partial_acc must be float32");
    TORCH_CHECK(partial_stats.scalar_type() == torch::kFloat32, "partial_stats must be float32");
    TORCH_CHECK(out.scalar_type() == q.scalar_type(), "out dtype mismatch");

    const int64_t split_k = partial_acc.size(1);
    const int64_t head_dim = q.size(3);
    const int64_t max_seq_len = k.size(2);
    const int64_t num_kv_heads = k.size(1);

    c10::cuda::CUDAGuard device_guard(q.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const dim3 grid_fwd(num_kv_heads, split_k);
    const dim3 block_fwd(kForwardThreads);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        q.scalar_type(),
        "grouped_prefix_cachepos_splitkv_fwd",
        [&] {
            grouped_prefix_cachepos_splitkv_fwd_kernel<scalar_t><<<grid_fwd, block_fwd, 0, stream>>>(
                q.data_ptr<scalar_t>(),
                k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                cache_position.data_ptr<int64_t>(),
                partial_acc.data_ptr<float>(),
                partial_stats.data_ptr<float>(),
                q.stride(1),
                k.stride(1),
                k.stride(2),
                v.stride(1),
                v.stride(2),
                partial_acc.stride(0),
                partial_acc.stride(1),
                partial_acc.stride(2),
                partial_stats.stride(0),
                partial_stats.stride(1),
                partial_stats.stride(2),
                partial_stats.stride(3),
                static_cast<float>(scale),
                static_cast<int>(head_dim),
                static_cast<int>(max_seq_len),
                static_cast<int>(split_k));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const dim3 grid_merge(num_kv_heads);
    const dim3 block_merge(kMergeThreads);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        q.scalar_type(),
        "grouped_prefix_cachepos_splitkv_merge",
        [&] {
            grouped_prefix_cachepos_splitkv_merge_kernel<scalar_t><<<grid_merge, block_merge, 0, stream>>>(
                partial_acc.data_ptr<float>(),
                partial_stats.data_ptr<float>(),
                out.data_ptr<scalar_t>(),
                partial_acc.stride(0),
                partial_acc.stride(1),
                partial_acc.stride(2),
                partial_stats.stride(0),
                partial_stats.stride(1),
                partial_stats.stride(2),
                partial_stats.stride(3),
                out.stride(1),
                static_cast<int>(split_k),
                static_cast<int>(head_dim));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor grouped_prefix_cachepos_splitkv_qhead_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cache_position,
    double scale,
    const torch::Tensor& partial_acc,
    const torch::Tensor& partial_stats,
    const torch::Tensor& out) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(k.is_cuda(), "k must be CUDA");
    TORCH_CHECK(v.is_cuda(), "v must be CUDA");
    TORCH_CHECK(cache_position.is_cuda(), "cache_position must be CUDA");
    TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q/k dtype mismatch");
    TORCH_CHECK(q.scalar_type() == v.scalar_type(), "q/v dtype mismatch");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be 4D");
    TORCH_CHECK(q.size(0) == 1 && q.size(2) == 1, "expected q shape [1, heads, 1, head_dim]");
    TORCH_CHECK(k.size(0) == 1 && v.size(0) == 1, "expected k/v batch size 1");
    TORCH_CHECK(q.size(1) == k.size(1) * 2, "expected q heads = kv heads * 2");
    TORCH_CHECK(q.size(3) <= 128, "head_dim > 128 not supported");
    TORCH_CHECK(partial_acc.scalar_type() == torch::kFloat32, "partial_acc must be float32");
    TORCH_CHECK(partial_stats.scalar_type() == torch::kFloat32, "partial_stats must be float32");
    TORCH_CHECK(out.scalar_type() == q.scalar_type(), "out dtype mismatch");

    const int64_t split_k = partial_acc.size(1);
    const int64_t head_dim = q.size(3);
    const int64_t max_seq_len = k.size(2);
    const int64_t num_heads = q.size(1);

    c10::cuda::CUDAGuard device_guard(q.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const dim3 grid_fwd(num_heads, split_k);
    const dim3 block_fwd(kQHeadForwardThreads);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        q.scalar_type(),
        "grouped_prefix_cachepos_splitkv_qhead_fwd",
        [&] {
            grouped_prefix_cachepos_splitkv_qhead_fwd_kernel<scalar_t><<<grid_fwd, block_fwd, 0, stream>>>(
                q.data_ptr<scalar_t>(),
                k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                cache_position.data_ptr<int64_t>(),
                partial_acc.data_ptr<float>(),
                partial_stats.data_ptr<float>(),
                q.stride(1),
                k.stride(1),
                k.stride(2),
                v.stride(1),
                v.stride(2),
                partial_acc.stride(0),
                partial_acc.stride(1),
                partial_stats.stride(0),
                partial_stats.stride(1),
                partial_stats.stride(2),
                static_cast<float>(scale),
                static_cast<int>(head_dim),
                static_cast<int>(max_seq_len),
                static_cast<int>(split_k));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const dim3 grid_merge(num_heads);
    const dim3 block_merge(kMergeThreads);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        q.scalar_type(),
        "grouped_prefix_cachepos_splitkv_qhead_merge",
        [&] {
            grouped_prefix_cachepos_splitkv_qhead_merge_kernel<scalar_t><<<grid_merge, block_merge, 0, stream>>>(
                partial_acc.data_ptr<float>(),
                partial_stats.data_ptr<float>(),
                out.data_ptr<scalar_t>(),
                partial_acc.stride(0),
                partial_acc.stride(1),
                partial_stats.stride(0),
                partial_stats.stride(1),
                partial_stats.stride(2),
                out.stride(1),
                static_cast<int>(split_k),
                static_cast<int>(head_dim));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
