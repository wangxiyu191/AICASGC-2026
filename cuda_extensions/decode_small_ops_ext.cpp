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

void fused_q_rope_kv_cache_update_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& q_weight,
    const torch::Tensor& k_weight,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& q_out,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& cache_position,
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

void rms_norm_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& out,
    double eps);

void add_cuda(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& out);

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

void fused_q_rope_kv_cache_update(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& q_weight,
    const torch::Tensor& k_weight,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& q_out,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& cache_position,
    double eps) {
    fused_q_rope_kv_cache_update_cuda(
        q,
        k,
        v,
        q_weight,
        k_weight,
        cos,
        sin,
        q_out,
        k_cache,
        v_cache,
        cache_position,
        eps);
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

void rms_norm(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& out,
    double eps) {
    rms_norm_cuda(x, weight, out, eps);
}

void add(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& out) {
    add_cuda(a, b, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_qk_rms_norm_rope", &fused_qk_rms_norm_rope);
    m.def("fused_q_rope_kv_cache_update", &fused_q_rope_kv_cache_update);
    m.def("fused_silu_mul", &fused_silu_mul);
    m.def("fused_add_rms_norm", &fused_add_rms_norm);
    m.def("rms_norm", &rms_norm);
    m.def("add", &add);
}
