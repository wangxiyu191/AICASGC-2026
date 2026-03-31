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

void fused_silu_mul_cuda(
    const torch::Tensor& gate,
    const torch::Tensor& up,
    const torch::Tensor& out);

void fused_add_rms_norm_cuda(
    const torch::Tensor& residual,
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& out_sum,
    const torch::Tensor& out_norm,
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

void fused_silu_mul(
    const torch::Tensor& gate,
    const torch::Tensor& up,
    const torch::Tensor& out) {
    fused_silu_mul_cuda(gate, up, out);
}

void fused_add_rms_norm(
    const torch::Tensor& residual,
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& out_sum,
    const torch::Tensor& out_norm,
    double eps) {
    fused_add_rms_norm_cuda(residual, x, weight, out_sum, out_norm, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_qk_rms_norm_rope", &fused_qk_rms_norm_rope);
    m.def("fused_silu_mul", &fused_silu_mul);
    m.def("fused_add_rms_norm", &fused_add_rms_norm);
}
