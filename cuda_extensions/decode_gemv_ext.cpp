#include <torch/extension.h>

torch::Tensor splitk_matvec_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t split_k,
    const torch::Tensor& scratch,
    const torch::Tensor& out,
    bool has_bias);

torch::Tensor splitk_matvec_row4_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t split_k,
    const torch::Tensor& scratch,
    const torch::Tensor& out,
    bool has_bias);

torch::Tensor splitk_matvec_row8_shared_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t split_k,
    const torch::Tensor& scratch,
    const torch::Tensor& out,
    bool has_bias);

torch::Tensor splitk_matvec_row4_shared_vec_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t split_k,
    const torch::Tensor& scratch,
    const torch::Tensor& out,
    bool has_bias);

torch::Tensor splitk_matvec(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t split_k,
    const torch::Tensor& scratch,
    const torch::Tensor& out,
    bool has_bias) {
    return splitk_matvec_cuda(x, weight, bias, split_k, scratch, out, has_bias);
}

torch::Tensor splitk_matvec_row4(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t split_k,
    const torch::Tensor& scratch,
    const torch::Tensor& out,
    bool has_bias) {
    return splitk_matvec_row4_cuda(x, weight, bias, split_k, scratch, out, has_bias);
}

torch::Tensor splitk_matvec_row8_shared(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t split_k,
    const torch::Tensor& scratch,
    const torch::Tensor& out,
    bool has_bias) {
    return splitk_matvec_row8_shared_cuda(x, weight, bias, split_k, scratch, out, has_bias);
}

torch::Tensor splitk_matvec_row4_shared_vec_half(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t split_k,
    const torch::Tensor& scratch,
    const torch::Tensor& out,
    bool has_bias) {
    return splitk_matvec_row4_shared_vec_half_cuda(x, weight, bias, split_k, scratch, out, has_bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("splitk_matvec", &splitk_matvec);
    m.def("splitk_matvec_row4", &splitk_matvec_row4);
    m.def("splitk_matvec_row8_shared", &splitk_matvec_row8_shared);
    m.def("splitk_matvec_row4_shared_vec_half", &splitk_matvec_row4_shared_vec_half);
}
