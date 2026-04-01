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

torch::Tensor splitk_matvec_row4_shared_vec_half_v2_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t split_k,
    const torch::Tensor& scratch,
    const torch::Tensor& out,
    bool has_bias);

torch::Tensor persistent_matvec_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    bool has_bias);

torch::Tensor nosplit_matvec_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    bool has_bias);

torch::Tensor nosplit_hfma2_matvec_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    bool has_bias);

torch::Tensor nosplit_matvec_half_variant_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    bool has_bias,
    int64_t variant);

torch::Tensor nosplit_matvec_half_profile_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& profile,
    bool has_bias);

torch::Tensor nosplit_matvec_half_profile_variant_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& profile,
    bool has_bias,
    int64_t variant);

torch::Tensor nosplit_matvec_half_profile_xwarp_detail_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& trace,
    bool has_bias);

torch::Tensor nosplit_matvec_half_profile_xwarp_detail_with_smid_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& trace,
    const torch::Tensor& smid,
    bool has_bias);

torch::Tensor nosplit_matvec_half_profile_xwarp_sms2_cta_trace_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& trace,
    const torch::Tensor& smid,
    const torch::Tensor& row_base,
    bool has_bias);

torch::Tensor nosplit_matvec_half_profile_xwarp_sms2_balanced_cta_trace_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& trace,
    const torch::Tensor& smid,
    const torch::Tensor& row_base,
    bool has_bias);

torch::Tensor nosplit_xwarp_wait_sync_calibration_cuda(
    const torch::Tensor& trace);

torch::Tensor weight_stream_read_half_cuda(
    const torch::Tensor& weight,
    const torch::Tensor& out);

torch::Tensor nosplit_weight_only_half_variant_cuda(
    const torch::Tensor& weight,
    const torch::Tensor& out,
    int64_t variant);

std::vector<torch::Tensor> fused_add_rmsnorm_gateup_half_cuda(
    const torch::Tensor& residual,
    const torch::Tensor& x,
    const torch::Tensor& norm_weight,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& residual_out,
    const torch::Tensor& gate_up_out,
    double eps,
    bool has_bias);

torch::Tensor fused_silu_downproj_half_cuda(
    const torch::Tensor& gate_up,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
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

torch::Tensor splitk_matvec_row4_shared_vec_half_v2(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t split_k,
    const torch::Tensor& scratch,
    const torch::Tensor& out,
    bool has_bias) {
    return splitk_matvec_row4_shared_vec_half_v2_cuda(x, weight, bias, split_k, scratch, out, has_bias);
}

torch::Tensor persistent_matvec_half(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    bool has_bias) {
    return persistent_matvec_half_cuda(x, weight, bias, out, has_bias);
}

torch::Tensor nosplit_matvec_half(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    bool has_bias) {
    return nosplit_matvec_half_cuda(x, weight, bias, out, has_bias);
}

torch::Tensor nosplit_hfma2_matvec_half(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    bool has_bias) {
    return nosplit_hfma2_matvec_half_cuda(x, weight, bias, out, has_bias);
}

torch::Tensor nosplit_matvec_half_variant(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    bool has_bias,
    int64_t variant) {
    return nosplit_matvec_half_variant_cuda(x, weight, bias, out, has_bias, variant);
}

torch::Tensor nosplit_matvec_half_profile(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& profile,
    bool has_bias) {
    return nosplit_matvec_half_profile_cuda(x, weight, bias, out, profile, has_bias);
}

torch::Tensor nosplit_matvec_half_profile_variant(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& profile,
    bool has_bias,
    int64_t variant) {
    return nosplit_matvec_half_profile_variant_cuda(x, weight, bias, out, profile, has_bias, variant);
}

torch::Tensor nosplit_matvec_half_profile_xwarp_detail(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& trace,
    bool has_bias) {
    return nosplit_matvec_half_profile_xwarp_detail_cuda(x, weight, bias, out, trace, has_bias);
}

torch::Tensor nosplit_matvec_half_profile_xwarp_detail_with_smid(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& trace,
    const torch::Tensor& smid,
    bool has_bias) {
    return nosplit_matvec_half_profile_xwarp_detail_with_smid_cuda(x, weight, bias, out, trace, smid, has_bias);
}

torch::Tensor nosplit_matvec_half_profile_xwarp_sms2_cta_trace(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& trace,
    const torch::Tensor& smid,
    const torch::Tensor& row_base,
    bool has_bias) {
    return nosplit_matvec_half_profile_xwarp_sms2_cta_trace_cuda(
        x, weight, bias, out, trace, smid, row_base, has_bias);
}

torch::Tensor nosplit_matvec_half_profile_xwarp_sms2_balanced_cta_trace(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& trace,
    const torch::Tensor& smid,
    const torch::Tensor& row_base,
    bool has_bias) {
    return nosplit_matvec_half_profile_xwarp_sms2_balanced_cta_trace_cuda(
        x, weight, bias, out, trace, smid, row_base, has_bias);
}

torch::Tensor nosplit_xwarp_wait_sync_calibration(
    const torch::Tensor& trace) {
    return nosplit_xwarp_wait_sync_calibration_cuda(trace);
}

torch::Tensor weight_stream_read_half(
    const torch::Tensor& weight,
    const torch::Tensor& out) {
    return weight_stream_read_half_cuda(weight, out);
}

torch::Tensor nosplit_weight_only_half_variant(
    const torch::Tensor& weight,
    const torch::Tensor& out,
    int64_t variant) {
    return nosplit_weight_only_half_variant_cuda(weight, out, variant);
}

std::vector<torch::Tensor> fused_add_rmsnorm_gateup_half(
    const torch::Tensor& residual,
    const torch::Tensor& x,
    const torch::Tensor& norm_weight,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& residual_out,
    const torch::Tensor& gate_up_out,
    double eps,
    bool has_bias) {
    return fused_add_rmsnorm_gateup_half_cuda(
        residual,
        x,
        norm_weight,
        weight,
        bias,
        residual_out,
        gate_up_out,
        eps,
        has_bias);
}

torch::Tensor fused_silu_downproj_half(
    const torch::Tensor& gate_up,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    bool has_bias) {
    return fused_silu_downproj_half_cuda(gate_up, weight, bias, out, has_bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("splitk_matvec", &splitk_matvec);
    m.def("splitk_matvec_row4", &splitk_matvec_row4);
    m.def("splitk_matvec_row8_shared", &splitk_matvec_row8_shared);
    m.def("splitk_matvec_row4_shared_vec_half", &splitk_matvec_row4_shared_vec_half);
    m.def("splitk_matvec_row4_shared_vec_half_v2", &splitk_matvec_row4_shared_vec_half_v2);
    m.def("persistent_matvec_half", &persistent_matvec_half);
    m.def("nosplit_matvec_half", &nosplit_matvec_half);
    m.def("nosplit_hfma2_matvec_half", &nosplit_hfma2_matvec_half);
    m.def("nosplit_matvec_half_variant", &nosplit_matvec_half_variant);
    m.def("nosplit_matvec_half_profile", &nosplit_matvec_half_profile);
    m.def("nosplit_matvec_half_profile_variant", &nosplit_matvec_half_profile_variant);
    m.def("nosplit_matvec_half_profile_xwarp_detail", &nosplit_matvec_half_profile_xwarp_detail);
    m.def("nosplit_matvec_half_profile_xwarp_detail_with_smid", &nosplit_matvec_half_profile_xwarp_detail_with_smid);
    m.def("nosplit_matvec_half_profile_xwarp_sms2_cta_trace", &nosplit_matvec_half_profile_xwarp_sms2_cta_trace);
    m.def("nosplit_matvec_half_profile_xwarp_sms2_balanced_cta_trace", &nosplit_matvec_half_profile_xwarp_sms2_balanced_cta_trace);
    m.def("nosplit_xwarp_wait_sync_calibration", &nosplit_xwarp_wait_sync_calibration);
    m.def("weight_stream_read_half", &weight_stream_read_half);
    m.def("nosplit_weight_only_half_variant", &nosplit_weight_only_half_variant);
    m.def("fused_add_rmsnorm_gateup_half", &fused_add_rmsnorm_gateup_half);
    m.def("fused_silu_downproj_half", &fused_silu_downproj_half);
}
