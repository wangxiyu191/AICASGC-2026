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

torch::Tensor nosplit_matvec_half_dualcopy_routed_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight_a,
    const torch::Tensor& weight_b,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& route_table,
    bool has_bias,
    int64_t route_mode,
    int64_t smid_threshold);

torch::Tensor nosplit_matvec_half_partition_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight_buf,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& chunk_classes,
    int64_t sm_class_lo,
    int64_t sm_class_hi,
    int64_t n_rows,
    bool has_bias);

torch::Tensor nosplit_matvec_half_partition_v2_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight_buf,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    int64_t chunk_class_bits,
    int64_t sm_class_lo,
    int64_t sm_class_hi,
    int64_t n_rows,
    bool has_bias);

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

torch::Tensor weight_stream_read_dual_half_cuda(
    const torch::Tensor& weight_a,
    const torch::Tensor& weight_b,
    const torch::Tensor& out,
    const torch::Tensor& trace,
    const torch::Tensor& smid,
    const torch::Tensor& route,
    const torch::Tensor& route_table,
    int64_t vecs_per_block,
    int64_t route_mode,
    int64_t smid_threshold);

torch::Tensor l2_8kb_pointer_chase_pair_cuda(
    const torch::Tensor& chain,
    const torch::Tensor& start_even,
    const torch::Tensor& start_odd,
    const torch::Tensor& flush_buf,
    const torch::Tensor& trace,
    const torch::Tensor& smid,
    const torch::Tensor& final_idx,
    const torch::Tensor& flush_sink,
    int64_t chain_len,
    int64_t swap_order);

torch::Tensor l2_sequential_pointer_chase_trace_cuda(
    const torch::Tensor& chain,
    const torch::Tensor& start,
    const torch::Tensor& flush_buf,
    const torch::Tensor& latency,
    const torch::Tensor& smid,
    const torch::Tensor& final_idx,
    const torch::Tensor& flush_sink,
    int64_t chain_len);

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

torch::Tensor fused_gateup_silu_downproj_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& bias,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out,
    bool has_bias);

torch::Tensor fused_gateup_silu_downproj_mega_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& out);

torch::Tensor fused_gateup_silu_downproj_mega_partial_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& partial_out);

torch::Tensor fused_gateup_silu_downproj_softsync_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& barrier_count,
    const torch::Tensor& barrier_phase,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out);

torch::Tensor fused_gateup_silu_downproj_softsync_profile_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& barrier_count,
    const torch::Tensor& barrier_phase,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out,
    const torch::Tensor& profile);

torch::Tensor fused_gateup_silu_downproj_gridsync_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out);

torch::Tensor fused_gateup_silu_downproj_gridsync_profile_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out,
    const torch::Tensor& profile);

torch::Tensor fused_post_attn_mlp_persistent_half_cuda(
    const torch::Tensor& residual,
    const torch::Tensor& attn_output,
    const torch::Tensor& norm_weight,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& bias,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out,
    double eps,
    bool has_bias);

torch::Tensor fused_gateup_silu_downproj_persistent_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out);

torch::Tensor fused_gateup_silu_downproj_persistent_profile_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out,
    const torch::Tensor& profile);

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

torch::Tensor nosplit_matvec_half_dualcopy_routed(
    const torch::Tensor& x,
    const torch::Tensor& weight_a,
    const torch::Tensor& weight_b,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& route_table,
    bool has_bias,
    int64_t route_mode,
    int64_t smid_threshold) {
    return nosplit_matvec_half_dualcopy_routed_cuda(
        x,
        weight_a,
        weight_b,
        bias,
        out,
        route_table,
        has_bias,
        route_mode,
        smid_threshold);
}

torch::Tensor nosplit_matvec_half_partition(
    const torch::Tensor& x,
    const torch::Tensor& weight_buf,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& chunk_classes,
    int64_t sm_class_lo,
    int64_t sm_class_hi,
    int64_t n_rows,
    bool has_bias) {
    return nosplit_matvec_half_partition_cuda(
        x,
        weight_buf,
        bias,
        out,
        chunk_classes,
        sm_class_lo,
        sm_class_hi,
        n_rows,
        has_bias);
}

torch::Tensor nosplit_matvec_half_partition_v2(
    const torch::Tensor& x,
    const torch::Tensor& weight_buf,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    int64_t chunk_class_bits,
    int64_t sm_class_lo,
    int64_t sm_class_hi,
    int64_t n_rows,
    bool has_bias) {
    return nosplit_matvec_half_partition_v2_cuda(
        x,
        weight_buf,
        bias,
        out,
        chunk_class_bits,
        sm_class_lo,
        sm_class_hi,
        n_rows,
        has_bias);
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

torch::Tensor weight_stream_read_dual_half(
    const torch::Tensor& weight_a,
    const torch::Tensor& weight_b,
    const torch::Tensor& out,
    const torch::Tensor& trace,
    const torch::Tensor& smid,
    const torch::Tensor& route,
    const torch::Tensor& route_table,
    int64_t vecs_per_block,
    int64_t route_mode,
    int64_t smid_threshold) {
    return weight_stream_read_dual_half_cuda(
        weight_a,
        weight_b,
        out,
        trace,
        smid,
        route,
        route_table,
        vecs_per_block,
        route_mode,
        smid_threshold);
}

torch::Tensor l2_8kb_pointer_chase_pair(
    const torch::Tensor& chain,
    const torch::Tensor& start_even,
    const torch::Tensor& start_odd,
    const torch::Tensor& flush_buf,
    const torch::Tensor& trace,
    const torch::Tensor& smid,
    const torch::Tensor& final_idx,
    const torch::Tensor& flush_sink,
    int64_t chain_len,
    int64_t swap_order) {
    return l2_8kb_pointer_chase_pair_cuda(
        chain,
        start_even,
        start_odd,
        flush_buf,
        trace,
        smid,
        final_idx,
        flush_sink,
        chain_len,
        swap_order);
}

torch::Tensor l2_sequential_pointer_chase_trace(
    const torch::Tensor& chain,
    const torch::Tensor& start,
    const torch::Tensor& flush_buf,
    const torch::Tensor& latency,
    const torch::Tensor& smid,
    const torch::Tensor& final_idx,
    const torch::Tensor& flush_sink,
    int64_t chain_len) {
    return l2_sequential_pointer_chase_trace_cuda(
        chain,
        start,
        flush_buf,
        latency,
        smid,
        final_idx,
        flush_sink,
        chain_len);
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

torch::Tensor fused_gateup_silu_downproj_half(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& bias,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out,
    bool has_bias) {
    return fused_gateup_silu_downproj_half_cuda(
        x,
        gate_up_weight,
        down_proj_weight,
        bias,
        hidden_out,
        out,
        has_bias);
}

torch::Tensor fused_gateup_silu_downproj_mega_half(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& out) {
    return fused_gateup_silu_downproj_mega_half_cuda(
        x,
        gate_up_weight,
        down_proj_weight,
        out);
}

torch::Tensor fused_gateup_silu_downproj_mega_partial_half(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& partial_out) {
    return fused_gateup_silu_downproj_mega_partial_half_cuda(
        x,
        gate_up_weight,
        down_proj_weight,
        partial_out);
}

torch::Tensor fused_gateup_silu_downproj_softsync_half(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& barrier_count,
    const torch::Tensor& barrier_phase,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out) {
    return fused_gateup_silu_downproj_softsync_half_cuda(
        x,
        gate_up_weight,
        down_proj_weight,
        barrier_count,
        barrier_phase,
        hidden_out,
        out);
}

torch::Tensor fused_gateup_silu_downproj_softsync_profile_half(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& barrier_count,
    const torch::Tensor& barrier_phase,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out,
    const torch::Tensor& profile) {
    return fused_gateup_silu_downproj_softsync_profile_half_cuda(
        x,
        gate_up_weight,
        down_proj_weight,
        barrier_count,
        barrier_phase,
        hidden_out,
        out,
        profile);
}

torch::Tensor fused_gateup_silu_downproj_gridsync_half(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out) {
    return fused_gateup_silu_downproj_gridsync_half_cuda(
        x,
        gate_up_weight,
        down_proj_weight,
        hidden_out,
        out);
}

torch::Tensor fused_gateup_silu_downproj_gridsync_profile_half(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out,
    const torch::Tensor& profile) {
    return fused_gateup_silu_downproj_gridsync_profile_half_cuda(
        x,
        gate_up_weight,
        down_proj_weight,
        hidden_out,
        out,
        profile);
}

torch::Tensor fused_gateup_silu_downproj_persistent_half(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out) {
    return fused_gateup_silu_downproj_persistent_half_cuda(
        x,
        gate_up_weight,
        down_proj_weight,
        hidden_out,
        out);
}

torch::Tensor fused_post_attn_mlp_persistent_half(
    const torch::Tensor& residual,
    const torch::Tensor& attn_output,
    const torch::Tensor& norm_weight,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& bias,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out,
    double eps,
    bool has_bias) {
    return fused_post_attn_mlp_persistent_half_cuda(
        residual,
        attn_output,
        norm_weight,
        gate_up_weight,
        down_proj_weight,
        bias,
        hidden_out,
        out,
        eps,
        has_bias);
}

torch::Tensor fused_gateup_silu_downproj_persistent_profile_half(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out,
    const torch::Tensor& profile) {
    return fused_gateup_silu_downproj_persistent_profile_half_cuda(
        x,
        gate_up_weight,
        down_proj_weight,
        hidden_out,
        out,
        profile);
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
    m.def("nosplit_matvec_half_dualcopy_routed", &nosplit_matvec_half_dualcopy_routed);
    m.def("nosplit_matvec_half_partition", &nosplit_matvec_half_partition);
    m.def("nosplit_matvec_half_partition_v2", &nosplit_matvec_half_partition_v2);
    m.def("nosplit_matvec_half_profile", &nosplit_matvec_half_profile);
    m.def("nosplit_matvec_half_profile_variant", &nosplit_matvec_half_profile_variant);
    m.def("nosplit_matvec_half_profile_xwarp_detail", &nosplit_matvec_half_profile_xwarp_detail);
    m.def("nosplit_matvec_half_profile_xwarp_detail_with_smid", &nosplit_matvec_half_profile_xwarp_detail_with_smid);
    m.def("nosplit_matvec_half_profile_xwarp_sms2_cta_trace", &nosplit_matvec_half_profile_xwarp_sms2_cta_trace);
    m.def("nosplit_matvec_half_profile_xwarp_sms2_balanced_cta_trace", &nosplit_matvec_half_profile_xwarp_sms2_balanced_cta_trace);
    m.def("nosplit_xwarp_wait_sync_calibration", &nosplit_xwarp_wait_sync_calibration);
    m.def("weight_stream_read_half", &weight_stream_read_half);
    m.def("weight_stream_read_dual_half", &weight_stream_read_dual_half);
    m.def("l2_8kb_pointer_chase_pair", &l2_8kb_pointer_chase_pair);
    m.def("l2_sequential_pointer_chase_trace", &l2_sequential_pointer_chase_trace);
    m.def("nosplit_weight_only_half_variant", &nosplit_weight_only_half_variant);
    m.def("fused_add_rmsnorm_gateup_half", &fused_add_rmsnorm_gateup_half);
    m.def("fused_silu_downproj_half", &fused_silu_downproj_half);
    m.def("fused_gateup_silu_downproj_half", &fused_gateup_silu_downproj_half);
    m.def("fused_gateup_silu_downproj_mega_half", &fused_gateup_silu_downproj_mega_half);
    m.def("fused_gateup_silu_downproj_mega_partial_half", &fused_gateup_silu_downproj_mega_partial_half);
    m.def("fused_gateup_silu_downproj_softsync_half", &fused_gateup_silu_downproj_softsync_half);
    m.def("fused_gateup_silu_downproj_softsync_profile_half", &fused_gateup_silu_downproj_softsync_profile_half);
    m.def("fused_gateup_silu_downproj_gridsync_half", &fused_gateup_silu_downproj_gridsync_half);
    m.def("fused_gateup_silu_downproj_gridsync_profile_half", &fused_gateup_silu_downproj_gridsync_profile_half);
    m.def("fused_post_attn_mlp_persistent_half", &fused_post_attn_mlp_persistent_half);
    m.def("fused_gateup_silu_downproj_persistent_half", &fused_gateup_silu_downproj_persistent_half);
    m.def("fused_gateup_silu_downproj_persistent_profile_half", &fused_gateup_silu_downproj_persistent_profile_half);
}
