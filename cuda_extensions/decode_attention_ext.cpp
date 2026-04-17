#include <torch/extension.h>

torch::Tensor grouped_prefix_cachepos_splitkv_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cache_position,
    double scale,
    const torch::Tensor& partial_acc,
    const torch::Tensor& partial_stats,
    const torch::Tensor& out,
    int64_t split_k);

torch::Tensor grouped_prefix_cachepos_splitkv_qhead_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cache_position,
    double scale,
    const torch::Tensor& partial_acc,
    const torch::Tensor& partial_stats,
    const torch::Tensor& out,
    int64_t split_k);

torch::Tensor grouped_prefix_cachepos_splitkv(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cache_position,
    double scale,
    const torch::Tensor& partial_acc,
    const torch::Tensor& partial_stats,
    const torch::Tensor& out,
    int64_t split_k) {
    return grouped_prefix_cachepos_splitkv_cuda(
        q,
        k,
        v,
        cache_position,
        scale,
        partial_acc,
        partial_stats,
        out,
        split_k);
}

torch::Tensor grouped_prefix_cachepos_splitkv_qhead(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cache_position,
    double scale,
    const torch::Tensor& partial_acc,
    const torch::Tensor& partial_stats,
    const torch::Tensor& out,
    int64_t split_k) {
    return grouped_prefix_cachepos_splitkv_qhead_cuda(
        q,
        k,
        v,
        cache_position,
        scale,
        partial_acc,
        partial_stats,
        out,
        split_k);
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
    int64_t num_kv_groups,
    int64_t split_k,
    const torch::Tensor& prefetch_weight,
    int64_t prefetch_blocks);

// v4 fused fwd+merge (single kernel launch)
torch::Tensor decode_attn_v4_fused_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cache_position,
    double scale,
    const torch::Tensor& partial_acc,
    const torch::Tensor& partial_stats,
    const torch::Tensor& out,
    const torch::Tensor& done_count,
    int64_t num_kv_groups,
    int64_t split_k,
    const torch::Tensor& prefetch_weight,
    int64_t prefetch_blocks);

// v4 CTA-local experiment: one CTA per KV head, intra-CTA seqlen split + merge
torch::Tensor decode_attn_v4_cta_local_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cache_position,
    double scale,
    const torch::Tensor& out,
    int64_t num_kv_groups);

// v4 CTA-qhead experiment: one CTA per q-head, intra-CTA seqlen split + merge
torch::Tensor decode_attn_v4_cta_qhead_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cache_position,
    double scale,
    const torch::Tensor& out,
    int64_t num_kv_groups,
    int64_t warps_per_cta,
    int64_t lane_layout,
    int64_t prefetch_mode,
    int64_t merge_mode);

torch::Tensor decode_attn_v4_splitkv(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cache_position,
    double scale,
    const torch::Tensor& partial_acc,
    const torch::Tensor& partial_stats,
    const torch::Tensor& out,
    int64_t num_kv_groups,
    int64_t split_k,
    const torch::Tensor& prefetch_weight,
    int64_t prefetch_blocks) {
    return decode_attn_v4_splitkv_cuda(
        q, k, v, cache_position, scale,
        partial_acc, partial_stats, out, num_kv_groups, split_k,
        prefetch_weight, prefetch_blocks);
}

torch::Tensor decode_attn_v4_fused(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cache_position,
    double scale,
    const torch::Tensor& partial_acc,
    const torch::Tensor& partial_stats,
    const torch::Tensor& out,
    const torch::Tensor& done_count,
    int64_t num_kv_groups,
    int64_t split_k,
    const torch::Tensor& prefetch_weight,
    int64_t prefetch_blocks) {
    return decode_attn_v4_fused_cuda(
        q, k, v, cache_position, scale,
        partial_acc, partial_stats, out, done_count, num_kv_groups, split_k,
        prefetch_weight, prefetch_blocks);
}

torch::Tensor decode_attn_v4_cta_local(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cache_position,
    double scale,
    const torch::Tensor& out,
    int64_t num_kv_groups) {
    return decode_attn_v4_cta_local_cuda(
        q, k, v, cache_position, scale, out, num_kv_groups);
}

torch::Tensor decode_attn_v4_cta_qhead(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cache_position,
    double scale,
    const torch::Tensor& out,
    int64_t num_kv_groups,
    int64_t warps_per_cta,
    int64_t lane_layout,
    int64_t prefetch_mode,
    int64_t merge_mode) {
    return decode_attn_v4_cta_qhead_cuda(
        q, k, v, cache_position, scale, out, num_kv_groups, warps_per_cta, lane_layout, prefetch_mode, merge_mode);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped_prefix_cachepos_splitkv", &grouped_prefix_cachepos_splitkv);
    m.def("grouped_prefix_cachepos_splitkv_qhead", &grouped_prefix_cachepos_splitkv_qhead);
    m.def("decode_attn_v4_splitkv", &decode_attn_v4_splitkv);
    m.def("decode_attn_v4_fused", &decode_attn_v4_fused);
    m.def("decode_attn_v4_cta_local", &decode_attn_v4_cta_local);
    m.def("decode_attn_v4_cta_qhead", &decode_attn_v4_cta_qhead);
}
