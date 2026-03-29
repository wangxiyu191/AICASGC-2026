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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped_prefix_cachepos_splitkv", &grouped_prefix_cachepos_splitkv);
    m.def("grouped_prefix_cachepos_splitkv_qhead", &grouped_prefix_cachepos_splitkv_qhead);
}
