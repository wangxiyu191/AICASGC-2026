#include <torch/extension.h>

torch::Tensor grouped_prefix_cachepos_splitkv_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cache_position,
    double scale,
    const torch::Tensor& partial_acc,
    const torch::Tensor& partial_stats,
    const torch::Tensor& out);

torch::Tensor grouped_prefix_cachepos_splitkv_qhead_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cache_position,
    double scale,
    const torch::Tensor& partial_acc,
    const torch::Tensor& partial_stats,
    const torch::Tensor& out);

torch::Tensor grouped_prefix_cachepos_splitkv(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cache_position,
    double scale,
    const torch::Tensor& partial_acc,
    const torch::Tensor& partial_stats,
    const torch::Tensor& out) {
    return grouped_prefix_cachepos_splitkv_cuda(
        q,
        k,
        v,
        cache_position,
        scale,
        partial_acc,
        partial_stats,
        out);
}

torch::Tensor grouped_prefix_cachepos_splitkv_qhead(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cache_position,
    double scale,
    const torch::Tensor& partial_acc,
    const torch::Tensor& partial_stats,
    const torch::Tensor& out) {
    return grouped_prefix_cachepos_splitkv_qhead_cuda(
        q,
        k,
        v,
        cache_position,
        scale,
        partial_acc,
        partial_stats,
        out);
}

// v4 tiled two-pass kernel declaration
torch::Tensor decode_attn_v4_splitkv_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cache_position,
    double scale,
    const torch::Tensor& partial_acc,
    const torch::Tensor& partial_stats,
    const torch::Tensor& out,
    int64_t num_kv_groups);

torch::Tensor decode_attn_v4_splitkv(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cache_position,
    double scale,
    const torch::Tensor& partial_acc,
    const torch::Tensor& partial_stats,
    const torch::Tensor& out,
    int64_t num_kv_groups) {
    return decode_attn_v4_splitkv_cuda(
        q, k, v, cache_position, scale,
        partial_acc, partial_stats, out, num_kv_groups);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped_prefix_cachepos_splitkv", &grouped_prefix_cachepos_splitkv);
    m.def("grouped_prefix_cachepos_splitkv_qhead", &grouped_prefix_cachepos_splitkv_qhead);
    m.def("decode_attn_v4_splitkv", &decode_attn_v4_splitkv);
}
