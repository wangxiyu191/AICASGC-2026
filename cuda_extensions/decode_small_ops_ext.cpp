#include <torch/extension.h>

void fused_qk_rms_norm_rope_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& q_weight,
    const torch::Tensor& k_weight,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& q_out,
    const torch::Tensor& k_out,
    double eps);

void fused_qk_rms_norm_rope(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& q_weight,
    const torch::Tensor& k_weight,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& q_out,
    const torch::Tensor& k_out,
    double eps) {
    fused_qk_rms_norm_rope_cuda(q, k, q_weight, k_weight, cos, sin, q_out, k_out, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_qk_rms_norm_rope", &fused_qk_rms_norm_rope);
}
