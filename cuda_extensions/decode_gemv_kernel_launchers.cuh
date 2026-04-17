torch::Tensor splitk_matvec_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t split_k,
    const torch::Tensor& scratch,
    const torch::Tensor& out,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(scratch.is_cuda(), "scratch must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(scratch.scalar_type() == torch::kFloat32, "scratch must be float32");
    TORCH_CHECK(out.scalar_type() == x.scalar_type(), "out dtype mismatch");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    scratch.zero_();

    const dim3 grid_fwd(n, split_k);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        x.scalar_type(),
        "splitk_matvec_fwd",
        [&] {
            splitk_matvec_fwd_kernel<scalar_t><<<grid_fwd, kThreads, 0, stream>>>(
                x.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                scratch.data_ptr<float>(),
                weight.stride(0),
                weight.stride(1),
                n,
                k,
                static_cast<int>(split_k));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int block = 256;
    const int grid = (n + block - 1) / block;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        x.scalar_type(),
        "splitk_matvec_finalize",
        [&] {
            splitk_matvec_finalize_kernel<scalar_t><<<grid, block, 0, stream>>>(
                scratch.data_ptr<float>(),
                has_bias ? bias.data_ptr<scalar_t>() : nullptr,
                out.data_ptr<scalar_t>(),
                n,
                has_bias);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor splitk_matvec_row4_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t split_k,
    const torch::Tensor& scratch,
    const torch::Tensor& out,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(scratch.is_cuda(), "scratch must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(scratch.scalar_type() == torch::kFloat32, "scratch must be float32");
    TORCH_CHECK(out.scalar_type() == x.scalar_type(), "out dtype mismatch");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    scratch.zero_();

    const dim3 grid_fwd((n + 3) / 4, split_k);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        x.scalar_type(),
        "splitk_matvec_row4_fwd",
        [&] {
            splitk_matvec_row4_fwd_kernel<scalar_t><<<grid_fwd, kRow4Threads, 0, stream>>>(
                x.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                scratch.data_ptr<float>(),
                weight.stride(0),
                weight.stride(1),
                n,
                k,
                static_cast<int>(split_k));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int block = 256;
    const int grid = (n + block - 1) / block;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        x.scalar_type(),
        "splitk_matvec_row4_finalize",
        [&] {
            splitk_matvec_finalize_kernel<scalar_t><<<grid, block, 0, stream>>>(
                scratch.data_ptr<float>(),
                has_bias ? bias.data_ptr<scalar_t>() : nullptr,
                out.data_ptr<scalar_t>(),
                n,
                has_bias);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor splitk_matvec_row8_shared_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t split_k,
    const torch::Tensor& scratch,
    const torch::Tensor& out,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(scratch.is_cuda(), "scratch must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(scratch.scalar_type() == torch::kFloat32, "scratch must be float32");
    TORCH_CHECK(out.scalar_type() == x.scalar_type(), "out dtype mismatch");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    scratch.zero_();

    const dim3 grid_fwd((n + 7) / 8, split_k);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        x.scalar_type(),
        "splitk_matvec_row8_shared_fwd",
        [&] {
            splitk_matvec_row8_shared_fwd_kernel<scalar_t><<<grid_fwd, kRow8Threads, 0, stream>>>(
                x.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                scratch.data_ptr<float>(),
                weight.stride(0),
                weight.stride(1),
                n,
                k,
                static_cast<int>(split_k));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int block = 256;
    const int grid = (n + block - 1) / block;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        x.scalar_type(),
        "splitk_matvec_row8_shared_finalize",
        [&] {
            splitk_matvec_finalize_kernel<scalar_t><<<grid, block, 0, stream>>>(
                scratch.data_ptr<float>(),
                has_bias ? bias.data_ptr<scalar_t>() : nullptr,
                out.data_ptr<scalar_t>(),
                n,
                has_bias);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor splitk_matvec_row4_shared_vec_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t split_k,
    const torch::Tensor& scratch,
    const torch::Tensor& out,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(scratch.is_cuda(), "scratch must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "row4_shared_vec_half only supports fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "row4_shared_vec_half only supports fp16");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(scratch.scalar_type() == torch::kFloat32, "scratch must be float32");
    TORCH_CHECK(out.scalar_type() == x.scalar_type(), "out dtype mismatch");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    scratch.zero_();

    const dim3 grid_fwd((n + 3) / 4, split_k);
    splitk_matvec_row4_shared_vec_half_fwd_kernel<<<grid_fwd, kRow4Threads, 0, stream>>>(
        x.data_ptr<at::Half>(),
        weight.data_ptr<at::Half>(),
        scratch.data_ptr<float>(),
        weight.stride(0),
        weight.stride(1),
        n,
        k,
        static_cast<int>(split_k)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int block = 256;
    const int grid = (n + block - 1) / block;
    splitk_matvec_finalize_kernel<at::Half><<<grid, block, 0, stream>>>(
        scratch.data_ptr<float>(),
        has_bias ? bias.data_ptr<at::Half>() : nullptr,
        out.data_ptr<at::Half>(),
        n,
        has_bias
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor splitk_matvec_row4_shared_vec_half_v2_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t split_k,
    const torch::Tensor& scratch,
    const torch::Tensor& out,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(scratch.is_cuda(), "scratch must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "v2 only supports fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "v2 only supports fp16");
    TORCH_CHECK(weight.stride(1) == 1, "v2 requires contiguous weight (stride_wk==1)");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(scratch.scalar_type() == torch::kFloat32, "scratch must be float32");
    TORCH_CHECK(out.scalar_type() == x.scalar_type(), "out dtype mismatch");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    scratch.zero_();

    const dim3 grid_fwd((n + 3) / 4, split_k);
    splitk_matvec_row4_shared_vec_half_v2_fwd_kernel<<<grid_fwd, kRow4Threads, 0, stream>>>(
        x.data_ptr<at::Half>(),
        weight.data_ptr<at::Half>(),
        scratch.data_ptr<float>(),
        weight.stride(0),
        n,
        k,
        static_cast<int>(split_k)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int block_f = 256;
    const int grid_f = (n + block_f - 1) / block_f;
    splitk_matvec_finalize_kernel<at::Half><<<grid_f, block_f, 0, stream>>>(
        scratch.data_ptr<float>(),
        has_bias ? bias.data_ptr<at::Half>() : nullptr,
        out.data_ptr<at::Half>(),
        n,
        has_bias
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor persistent_matvec_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "persistent_matvec only supports fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "persistent_matvec only supports fp16");
    TORCH_CHECK(weight.stride(1) == 1, "persistent_matvec requires contiguous weight (stride_wk==1)");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    // Launch exactly #SM blocks (query device)
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, x.device().index());
    const int grid_blocks = num_sms;

    persistent_matvec_half_fwd_kernel<<<grid_blocks, kRow4Threads, 0, stream>>>(
        x.data_ptr<at::Half>(),
        weight.data_ptr<at::Half>(),
        has_bias ? bias.data_ptr<at::Half>() : nullptr,
        out.data_ptr<at::Half>(),
        weight.stride(0),
        n,
        k,
        has_bias
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor nosplit_matvec_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "nosplit_matvec only supports fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "nosplit_matvec only supports fp16");
    TORCH_CHECK(weight.stride(1) == 1, "nosplit_matvec requires contiguous weight");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    // 1 block per 4 rows — no split-K, no scratch, no atomicAdd, no finalize
    const bool use_row2_k2048 = (k == kNosplitSpecializedK2048 && n <= 2048);
    const int grid_blocks = use_row2_k2048 ? ((n + 1) / 2) : ((n + 3) / 4);

    if (use_row2_k2048) {
        nosplit_matvec_k2048_row2_hfma2_fwd_kernel<false><<<grid_blocks, kRow2Threads, 0, stream>>>(
            x.data_ptr<at::Half>(),
            weight.data_ptr<at::Half>(),
            has_bias ? bias.data_ptr<at::Half>() : nullptr,
            out.data_ptr<at::Half>(),
            nullptr,
            weight.stride(0),
            n,
            has_bias
        );
    } else if (k == kNosplitSpecializedK2048) {
        nosplit_matvec_k2048_hfma2_fwd_kernel<false><<<grid_blocks, kRow4Threads, 0, stream>>>(
            x.data_ptr<at::Half>(),
            weight.data_ptr<at::Half>(),
            has_bias ? bias.data_ptr<at::Half>() : nullptr,
            out.data_ptr<at::Half>(),
            nullptr,
            weight.stride(0),
            n,
            has_bias
        );
    } else if (k == 6144) {
        nosplit_matvec_k6144_hfma2_fwd_kernel<false><<<grid_blocks, kRow4Threads, 0, stream>>>(
            x.data_ptr<at::Half>(),
            weight.data_ptr<at::Half>(),
            has_bias ? bias.data_ptr<at::Half>() : nullptr,
            out.data_ptr<at::Half>(),
            nullptr,
            weight.stride(0),
            n,
            has_bias
        );
    } else {
        persistent_matvec_half_fwd_kernel<<<grid_blocks, kRow4Threads, 0, stream>>>(
            x.data_ptr<at::Half>(),
            weight.data_ptr<at::Half>(),
            has_bias ? bias.data_ptr<at::Half>() : nullptr,
            out.data_ptr<at::Half>(),
            weight.stride(0),
            n,
            k,
            has_bias
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor nosplit_hfma2_matvec_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "hfma2 nosplit only supports fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "hfma2 nosplit only supports fp16");
    TORCH_CHECK(weight.stride(1) == 1, "hfma2 nosplit requires contiguous weight");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const bool use_row2_k2048 = (k == kNosplitSpecializedK2048 && n <= 2048);
    const int grid_blocks = use_row2_k2048 ? ((n + 1) / 2) : ((n + 3) / 4);

    if (use_row2_k2048) {
        nosplit_matvec_k2048_row2_hfma2_fwd_kernel<false><<<grid_blocks, kRow2Threads, 0, stream>>>(
            x.data_ptr<at::Half>(),
            weight.data_ptr<at::Half>(),
            has_bias ? bias.data_ptr<at::Half>() : nullptr,
            out.data_ptr<at::Half>(),
            nullptr,
            weight.stride(0),
            n,
            has_bias
        );
    } else if (k == kNosplitSpecializedK2048) {
        nosplit_matvec_k2048_hfma2_fwd_kernel<false><<<grid_blocks, kRow4Threads, 0, stream>>>(
            x.data_ptr<at::Half>(),
            weight.data_ptr<at::Half>(),
            has_bias ? bias.data_ptr<at::Half>() : nullptr,
            out.data_ptr<at::Half>(),
            nullptr,
            weight.stride(0),
            n,
            has_bias
        );
    } else if (k == 6144) {
        nosplit_matvec_k6144_hfma2_fwd_kernel<false><<<grid_blocks, kRow4Threads, 0, stream>>>(
            x.data_ptr<at::Half>(),
            weight.data_ptr<at::Half>(),
            has_bias ? bias.data_ptr<at::Half>() : nullptr,
            out.data_ptr<at::Half>(),
            nullptr,
            weight.stride(0),
            n,
            has_bias
        );
    } else {
        persistent_matvec_hfma2_fwd_kernel<<<grid_blocks, kRow4Threads, 0, stream>>>(
            x.data_ptr<at::Half>(),
            weight.data_ptr<at::Half>(),
            has_bias ? bias.data_ptr<at::Half>() : nullptr,
            out.data_ptr<at::Half>(),
            weight.stride(0),
            n,
            k,
            has_bias
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor nosplit_matvec_half_variant_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    bool has_bias,
    int64_t variant) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "nosplit variant only supports fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "nosplit variant only supports fp16");
    TORCH_CHECK(weight.stride(1) == 1, "nosplit variant requires contiguous weight");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, x.device().index());
    launch_nosplit_variant(
        x.data_ptr<at::Half>(),
        weight.data_ptr<at::Half>(),
        has_bias ? bias.data_ptr<at::Half>() : nullptr,
        out.data_ptr<at::Half>(),
        weight.stride(0),
        n,
        k,
        has_bias,
        variant,
        num_sms,
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor nosplit_matvec_half_dualcopy_routed_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight_a,
    const torch::Tensor& weight_b,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& route_table,
    bool has_bias,
    int64_t route_mode,
    int64_t smid_threshold) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight_a.is_cuda(), "weight_a must be CUDA");
    TORCH_CHECK(weight_b.is_cuda(), "weight_b must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "dualcopy nosplit matvec only supports fp16");
    TORCH_CHECK(weight_a.scalar_type() == torch::kFloat16, "dualcopy nosplit matvec only supports fp16");
    TORCH_CHECK(weight_b.scalar_type() == torch::kFloat16, "dualcopy nosplit matvec only supports fp16");
    TORCH_CHECK(weight_a.stride(1) == 1, "dualcopy nosplit matvec requires contiguous weight_a");
    TORCH_CHECK(weight_b.stride(1) == 1, "dualcopy nosplit matvec requires contiguous weight_b");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight_a.dim() == 2, "weight_a must be [N, K]");
    TORCH_CHECK(weight_b.sizes() == weight_a.sizes(), "weight_b size mismatch");
    TORCH_CHECK(weight_a.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(
        route_mode >= kDualCopyRouteForceA && route_mode <= kDualCopyRouteTable,
        "invalid dualcopy route_mode: ",
        route_mode);
    if (route_mode == kDualCopyRouteTable) {
        TORCH_CHECK(route_table.is_cuda(), "route_table must be CUDA for table routing");
        TORCH_CHECK(route_table.scalar_type() == torch::kInt, "route_table must be int32");
        TORCH_CHECK(route_table.is_contiguous(), "route_table must be contiguous");
    }
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight_a.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight_a.size(0));
    const int k = static_cast<int>(weight_a.size(1));
    TORCH_CHECK(k == 6144, "dualcopy nosplit matvec currently supports only k=6144");
    TORCH_CHECK(out.numel() == n, "out size mismatch");

    int num_sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, x.device().index()));
    if (route_mode == kDualCopyRouteTable) {
        TORCH_CHECK(route_table.numel() >= num_sms, "route_table too small for SM count");
    }
    const int grid_blocks = (n + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    nosplit_matvec_k6144_n8k1024_xwarp_w_async_dualcopy_fwd_kernel<<<grid_blocks, kRow8Threads, 0, stream>>>(
        x.data_ptr<at::Half>(),
        weight_a.data_ptr<at::Half>(),
        weight_b.data_ptr<at::Half>(),
        has_bias ? bias.data_ptr<at::Half>() : nullptr,
        out.data_ptr<at::Half>(),
        route_mode == kDualCopyRouteTable ? route_table.data_ptr<int32_t>() : nullptr,
        static_cast<int>(route_mode),
        static_cast<int>(smid_threshold),
        weight_a.stride(0),
        n,
        has_bias);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor nosplit_matvec_half_partition_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight_buf,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& chunk_classes,
    int64_t sm_class_lo,
    int64_t sm_class_hi,
    int64_t n_rows,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight_buf.is_cuda(), "weight_buf must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(chunk_classes.is_cuda(), "chunk_classes must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "partition matvec only supports fp16");
    TORCH_CHECK(weight_buf.scalar_type() == torch::kFloat16, "partition matvec only supports fp16 weight_buf");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(chunk_classes.scalar_type() == torch::kInt, "chunk_classes must be int32");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(x.size(0) == kNosplitSpecializedK2048, "partition matvec requires K=2048");
    TORCH_CHECK(weight_buf.is_contiguous(), "weight_buf must be contiguous");
    TORCH_CHECK(chunk_classes.is_contiguous(), "chunk_classes must be contiguous");
    TORCH_CHECK(n_rows > 0, "n_rows must be > 0");
    TORCH_CHECK(out.numel() >= n_rows, "out too small");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() >= n_rows, "bias size mismatch");
    }

    const int n = static_cast<int>(n_rows);
    const int grid_blocks = (n + 1) / 2;

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    nosplit_matvec_k2048_row2_partition_fwd_kernel<<<grid_blocks, kRow2Threads, 0, stream>>>(
        x.data_ptr<at::Half>(),
        weight_buf.data_ptr<at::Half>(),
        has_bias ? bias.data_ptr<at::Half>() : nullptr,
        out.data_ptr<at::Half>(),
        chunk_classes.data_ptr<int32_t>(),
        sm_class_lo,
        sm_class_hi,
        n,
        has_bias);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor nosplit_matvec_half_partition_v2_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight_buf,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    int64_t chunk_class_bits,
    int64_t sm_class_lo,
    int64_t sm_class_hi,
    int64_t n_rows,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight_buf.is_cuda(), "weight_buf must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "partition v2 matvec only supports fp16");
    TORCH_CHECK(weight_buf.scalar_type() == torch::kFloat16, "partition v2 matvec only supports fp16 weight_buf");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(x.size(0) == kNosplitSpecializedK2048, "partition v2 matvec requires K=2048");
    TORCH_CHECK(weight_buf.is_contiguous(), "weight_buf must be contiguous");
    TORCH_CHECK(n_rows > 0, "n_rows must be > 0");
    TORCH_CHECK(out.numel() >= n_rows, "out too small");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() >= n_rows, "bias size mismatch");
    }

    const int n = static_cast<int>(n_rows);
    const int grid_blocks = (n + 1) / 2;

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    nosplit_matvec_k2048_row2_partition_v2_fwd_kernel<<<grid_blocks, kRow2Threads, 0, stream>>>(
        x.data_ptr<at::Half>(),
        weight_buf.data_ptr<at::Half>(),
        has_bias ? bias.data_ptr<at::Half>() : nullptr,
        out.data_ptr<at::Half>(),
        chunk_class_bits,
        sm_class_lo,
        sm_class_hi,
        n,
        has_bias);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor nosplit_matvec_half_profile_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& profile,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(profile.is_cuda(), "profile must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "nosplit profile only supports fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "nosplit profile only supports fp16");
    TORCH_CHECK(weight.stride(1) == 1, "nosplit profile requires contiguous weight");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(profile.scalar_type() == torch::kInt64, "profile must be int64");
    TORCH_CHECK(profile.is_contiguous(), "profile must be contiguous");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    const bool use_row2_k2048 = (k == kNosplitSpecializedK2048 && n <= 2048);
    const int grid_blocks = use_row2_k2048 ? ((n + 1) / 2) : ((n + 3) / 4);
    TORCH_CHECK(
        k == kNosplitSpecializedK2048 || k == 6144,
        "nosplit internal profiling currently supports the active k=2048/6144 specialized kernels only");
    TORCH_CHECK(
        profile.numel() == static_cast<int64_t>(grid_blocks) * kNosplitProfileSlots,
        "profile tensor shape mismatch");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    profile.zero_();

    if (use_row2_k2048) {
        nosplit_matvec_k2048_row2_hfma2_fwd_kernel<true><<<grid_blocks, kRow2Threads, 0, stream>>>(
            x.data_ptr<at::Half>(),
            weight.data_ptr<at::Half>(),
            has_bias ? bias.data_ptr<at::Half>() : nullptr,
            out.data_ptr<at::Half>(),
            reinterpret_cast<uint64_t*>(profile.data_ptr<int64_t>()),
            weight.stride(0),
            n,
            has_bias
        );
    } else if (k == kNosplitSpecializedK2048) {
        nosplit_matvec_k2048_hfma2_fwd_kernel<true><<<grid_blocks, kRow4Threads, 0, stream>>>(
            x.data_ptr<at::Half>(),
            weight.data_ptr<at::Half>(),
            has_bias ? bias.data_ptr<at::Half>() : nullptr,
            out.data_ptr<at::Half>(),
            reinterpret_cast<uint64_t*>(profile.data_ptr<int64_t>()),
            weight.stride(0),
            n,
            has_bias
        );
    } else {
        nosplit_matvec_k6144_hfma2_fwd_kernel<true><<<grid_blocks, kRow4Threads, 0, stream>>>(
            x.data_ptr<at::Half>(),
            weight.data_ptr<at::Half>(),
            has_bias ? bias.data_ptr<at::Half>() : nullptr,
            out.data_ptr<at::Half>(),
            reinterpret_cast<uint64_t*>(profile.data_ptr<int64_t>()),
            weight.stride(0),
            n,
            has_bias
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor nosplit_matvec_half_profile_variant_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& profile,
    bool has_bias,
    int64_t variant) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(profile.is_cuda(), "profile must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "nosplit profile only supports fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "nosplit profile only supports fp16");
    TORCH_CHECK(weight.stride(1) == 1, "nosplit profile requires contiguous weight");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(profile.scalar_type() == torch::kInt64, "profile must be int64");
    TORCH_CHECK(profile.is_contiguous(), "profile must be contiguous");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    TORCH_CHECK(k == 6144, "variant profiling currently supports only k=6144");

    int grid_blocks = 0;
    switch (variant) {
        case kNosplitVariantK6144Row4:
            grid_blocks = (n + 3) / 4;
            break;
        case kNosplitVariantK6144N8K1024Async:
        case kNosplitVariantK6144N8K1024XWAsync:
        case kNosplitVariantK6144N8K1024XWarpWAsync:
            grid_blocks = (n + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;
            break;
        default:
            TORCH_CHECK(false, "variant profiling unsupported for variant: ", variant);
    }
    TORCH_CHECK(
        profile.numel() == static_cast<int64_t>(grid_blocks) * kNosplitProfileSlots,
        "profile tensor shape mismatch");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    profile.zero_();

    switch (variant) {
        case kNosplitVariantK6144Row4:
            nosplit_matvec_k6144_hfma2_fwd_kernel<true><<<grid_blocks, kRow4Threads, 0, stream>>>(
                x.data_ptr<at::Half>(),
                weight.data_ptr<at::Half>(),
                has_bias ? bias.data_ptr<at::Half>() : nullptr,
                out.data_ptr<at::Half>(),
                reinterpret_cast<uint64_t*>(profile.data_ptr<int64_t>()),
                weight.stride(0),
                n,
                has_bias
            );
            break;
        case kNosplitVariantK6144N8K1024Async:
            nosplit_matvec_k6144_n8k1024_async_fwd_kernel<true><<<grid_blocks, kRow8Threads, 0, stream>>>(
                x.data_ptr<at::Half>(),
                weight.data_ptr<at::Half>(),
                has_bias ? bias.data_ptr<at::Half>() : nullptr,
                out.data_ptr<at::Half>(),
                reinterpret_cast<uint64_t*>(profile.data_ptr<int64_t>()),
                weight.stride(0),
                n,
                has_bias
            );
            break;
        case kNosplitVariantK6144N8K1024XWAsync:
            nosplit_matvec_k6144_n8k1024_xw_async_fwd_kernel<true><<<grid_blocks, kRow8Threads, 0, stream>>>(
                x.data_ptr<at::Half>(),
                weight.data_ptr<at::Half>(),
                has_bias ? bias.data_ptr<at::Half>() : nullptr,
                out.data_ptr<at::Half>(),
                reinterpret_cast<uint64_t*>(profile.data_ptr<int64_t>()),
                weight.stride(0),
                n,
                has_bias
            );
            break;
        case kNosplitVariantK6144N8K1024XWarpWAsync:
            nosplit_matvec_k6144_n8k1024_xwarp_w_async_fwd_kernel<true><<<grid_blocks, kRow8Threads, 0, stream>>>(
                x.data_ptr<at::Half>(),
                weight.data_ptr<at::Half>(),
                has_bias ? bias.data_ptr<at::Half>() : nullptr,
                out.data_ptr<at::Half>(),
                reinterpret_cast<uint64_t*>(profile.data_ptr<int64_t>()),
                weight.stride(0),
                n,
                has_bias
            );
            break;
        default:
            TORCH_CHECK(false, "variant profiling unsupported for variant: ", variant);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor nosplit_matvec_half_profile_xwarp_detail_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& trace,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(trace.is_cuda(), "trace must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "xwarp detail profile only supports fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "xwarp detail profile only supports fp16");
    TORCH_CHECK(weight.stride(1) == 1, "xwarp detail profile requires contiguous weight");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(trace.scalar_type() == torch::kInt64, "trace must be int64");
    TORCH_CHECK(trace.is_contiguous(), "trace must be contiguous");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    TORCH_CHECK(k == 6144, "xwarp detail profile currently supports only k=6144");

    const int grid_blocks = (n + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;
    TORCH_CHECK(
        trace.numel() == static_cast<int64_t>(grid_blocks) * kRowsPerBlockAsync * kXWarpDetailSlots,
        "trace tensor shape mismatch");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    trace.zero_();

    nosplit_matvec_k6144_n8k1024_xwarp_w_async_trace_fwd_kernel<<<grid_blocks, kRow8Threads, 0, stream>>>(
        x.data_ptr<at::Half>(),
        weight.data_ptr<at::Half>(),
        has_bias ? bias.data_ptr<at::Half>() : nullptr,
        out.data_ptr<at::Half>(),
        trace.data_ptr<int64_t>(),
        nullptr,
        weight.stride(0),
        n,
        has_bias
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor nosplit_xwarp_wait_sync_calibration_cuda(
    const torch::Tensor& trace) {
    TORCH_CHECK(trace.is_cuda(), "trace must be CUDA");
    TORCH_CHECK(trace.scalar_type() == torch::kInt64, "trace must be int64");
    TORCH_CHECK(trace.is_contiguous(), "trace must be contiguous");
    TORCH_CHECK(trace.dim() == 3, "trace must be [grid_blocks, warp_groups, slots]");
    TORCH_CHECK(trace.size(1) == kRowsPerBlockAsync, "trace warp_groups must be 8");
    TORCH_CHECK(trace.size(2) == kXWarpDetailSlots, "trace slot count mismatch");

    const int grid_blocks = static_cast<int>(trace.size(0));
    c10::cuda::CUDAGuard device_guard(trace.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    trace.zero_();

    nosplit_xwarp_wait_sync_calibration_kernel<<<grid_blocks, kRow8Threads, 0, stream>>>(
        trace.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return trace;
}

torch::Tensor nosplit_matvec_half_profile_xwarp_detail_with_smid_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& trace,
    const torch::Tensor& smid,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(trace.is_cuda(), "trace must be CUDA");
    TORCH_CHECK(smid.is_cuda(), "smid must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "xwarp detail profile only supports fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "xwarp detail profile only supports fp16");
    TORCH_CHECK(weight.stride(1) == 1, "xwarp detail profile requires contiguous weight");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(trace.scalar_type() == torch::kInt64, "trace must be int64");
    TORCH_CHECK(trace.is_contiguous(), "trace must be contiguous");
    TORCH_CHECK(smid.scalar_type() == torch::kInt, "smid must be int32");
    TORCH_CHECK(smid.is_contiguous(), "smid must be contiguous");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    TORCH_CHECK(k == 6144, "xwarp detail profile currently supports only k=6144");

    const int grid_blocks = (n + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;
    TORCH_CHECK(
        trace.numel() == static_cast<int64_t>(grid_blocks) * kRowsPerBlockAsync * kXWarpDetailSlots,
        "trace tensor shape mismatch");
    TORCH_CHECK(
        smid.numel() == static_cast<int64_t>(grid_blocks) * kRowsPerBlockAsync,
        "smid tensor shape mismatch");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    trace.zero_();
    smid.fill_(-1);

    nosplit_matvec_k6144_n8k1024_xwarp_w_async_trace_fwd_kernel<<<grid_blocks, kRow8Threads, 0, stream>>>(
        x.data_ptr<at::Half>(),
        weight.data_ptr<at::Half>(),
        has_bias ? bias.data_ptr<at::Half>() : nullptr,
        out.data_ptr<at::Half>(),
        trace.data_ptr<int64_t>(),
        smid.data_ptr<int32_t>(),
        weight.stride(0),
        n,
        has_bias
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor nosplit_matvec_half_profile_xwarp_sms2_cta_trace_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& trace,
    const torch::Tensor& smid,
    const torch::Tensor& row_base,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(trace.is_cuda(), "trace must be CUDA");
    TORCH_CHECK(smid.is_cuda(), "smid must be CUDA");
    TORCH_CHECK(row_base.is_cuda(), "row_base must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "sms2 CTA trace only supports fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "sms2 CTA trace only supports fp16");
    TORCH_CHECK(weight.stride(1) == 1, "sms2 CTA trace requires contiguous weight");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(trace.scalar_type() == torch::kInt64, "trace must be int64");
    TORCH_CHECK(trace.is_contiguous(), "trace must be contiguous");
    TORCH_CHECK(smid.scalar_type() == torch::kInt, "smid must be int32");
    TORCH_CHECK(smid.is_contiguous(), "smid must be contiguous");
    TORCH_CHECK(row_base.scalar_type() == torch::kInt, "row_base must be int32");
    TORCH_CHECK(row_base.is_contiguous(), "row_base must be contiguous");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    TORCH_CHECK(k == 6144, "sms2 CTA trace currently supports only k=6144");

    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, x.device().index());
    const int full_grid_blocks = (n + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;
    const int launch_grid_blocks = std::min(full_grid_blocks, std::max(1, num_sms * 2));
    const int max_iters = (full_grid_blocks + launch_grid_blocks - 1) / launch_grid_blocks;

    TORCH_CHECK(trace.numel() == static_cast<int64_t>(launch_grid_blocks) * max_iters * 2, "trace tensor shape mismatch");
    TORCH_CHECK(smid.numel() == static_cast<int64_t>(launch_grid_blocks), "smid tensor shape mismatch");
    TORCH_CHECK(row_base.numel() == static_cast<int64_t>(launch_grid_blocks) * max_iters, "row_base tensor shape mismatch");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    trace.zero_();
    smid.fill_(-1);
    row_base.fill_(-1);

    nosplit_matvec_k6144_n8k1024_xwarp_w_async_sms2_cta_trace_fwd_kernel<<<launch_grid_blocks, kRow8Threads, 0, stream>>>(
        x.data_ptr<at::Half>(),
        weight.data_ptr<at::Half>(),
        has_bias ? bias.data_ptr<at::Half>() : nullptr,
        out.data_ptr<at::Half>(),
        trace.data_ptr<int64_t>(),
        smid.data_ptr<int32_t>(),
        row_base.data_ptr<int32_t>(),
        max_iters,
        weight.stride(0),
        n,
        has_bias
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

torch::Tensor nosplit_matvec_half_profile_xwarp_sms2_balanced_cta_trace_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    const torch::Tensor& trace,
    const torch::Tensor& smid,
    const torch::Tensor& row_base,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(trace.is_cuda(), "trace must be CUDA");
    TORCH_CHECK(smid.is_cuda(), "smid must be CUDA");
    TORCH_CHECK(row_base.is_cuda(), "row_base must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "sms2 balanced CTA trace only supports fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "sms2 balanced CTA trace only supports fp16");
    TORCH_CHECK(weight.stride(1) == 1, "sms2 balanced CTA trace requires contiguous weight");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(weight.size(1) == x.size(0), "weight/x shape mismatch");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(trace.scalar_type() == torch::kInt64, "trace must be int64");
    TORCH_CHECK(trace.is_contiguous(), "trace must be contiguous");
    TORCH_CHECK(smid.scalar_type() == torch::kInt, "smid must be int32");
    TORCH_CHECK(smid.is_contiguous(), "smid must be contiguous");
    TORCH_CHECK(row_base.scalar_type() == torch::kInt, "row_base must be int32");
    TORCH_CHECK(row_base.is_contiguous(), "row_base must be contiguous");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    TORCH_CHECK(k == 6144, "sms2 balanced CTA trace currently supports only k=6144");

    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, x.device().index());
    const int full_grid_blocks = (n + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;
    const int launch_grid_blocks = std::min(full_grid_blocks, std::max(1, num_sms * 2));
    const int max_rows_per_block = (n + launch_grid_blocks - 1) / launch_grid_blocks;
    const int max_iters = (max_rows_per_block + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;
    TORCH_CHECK(max_iters <= 2, "sms2 balanced CTA trace expects at most 2 row batches per CTA");

    TORCH_CHECK(trace.numel() == static_cast<int64_t>(launch_grid_blocks) * max_iters * 2, "trace tensor shape mismatch");
    TORCH_CHECK(smid.numel() == static_cast<int64_t>(launch_grid_blocks), "smid tensor shape mismatch");
    TORCH_CHECK(row_base.numel() == static_cast<int64_t>(launch_grid_blocks) * max_iters, "row_base tensor shape mismatch");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    trace.zero_();
    smid.fill_(-1);
    row_base.fill_(-1);

    nosplit_matvec_k6144_n8k1024_xwarp_w_async_sms2_balanced_cta_trace_fwd_kernel<<<launch_grid_blocks, kRow8Threads, 0, stream>>>(
        x.data_ptr<at::Half>(),
        weight.data_ptr<at::Half>(),
        has_bias ? bias.data_ptr<at::Half>() : nullptr,
        out.data_ptr<at::Half>(),
        trace.data_ptr<int64_t>(),
        smid.data_ptr<int32_t>(),
        row_base.data_ptr<int32_t>(),
        max_iters,
        weight.stride(0),
        n,
        has_bias
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

std::vector<torch::Tensor> fused_add_rmsnorm_gateup_half_cuda(
    const torch::Tensor& residual,
    const torch::Tensor& x,
    const torch::Tensor& norm_weight,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& residual_out,
    const torch::Tensor& gate_up_out,
    double eps,
    bool has_bias) {
    TORCH_CHECK(residual.is_cuda(), "residual must be CUDA");
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(norm_weight.is_cuda(), "norm_weight must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(residual_out.is_cuda(), "residual_out must be CUDA");
    TORCH_CHECK(gate_up_out.is_cuda(), "gate_up_out must be CUDA");
    TORCH_CHECK(residual.scalar_type() == torch::kFloat16, "fused_add_rmsnorm_gateup only supports fp16");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "fused_add_rmsnorm_gateup only supports fp16");
    TORCH_CHECK(norm_weight.scalar_type() == torch::kFloat16, "fused_add_rmsnorm_gateup only supports fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "fused_add_rmsnorm_gateup only supports fp16");
    TORCH_CHECK(weight.stride(1) == 1, "fused_add_rmsnorm_gateup requires contiguous weight");
    TORCH_CHECK(residual.dim() == 1, "residual must be flattened [K]");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(norm_weight.dim() == 1, "norm_weight must be flattened [K]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(residual.numel() == kNosplitSpecializedK2048, "fused_add_rmsnorm_gateup expects K=2048");
    TORCH_CHECK(x.numel() == kNosplitSpecializedK2048, "fused_add_rmsnorm_gateup expects K=2048");
    TORCH_CHECK(norm_weight.numel() == kNosplitSpecializedK2048, "fused_add_rmsnorm_gateup expects K=2048 norm_weight");
    TORCH_CHECK(weight.size(1) == kNosplitSpecializedK2048, "fused_add_rmsnorm_gateup expects weight K=2048");
    TORCH_CHECK(residual_out.scalar_type() == torch::kFloat16, "residual_out must be fp16");
    TORCH_CHECK(gate_up_out.scalar_type() == torch::kFloat16, "gate_up_out must be fp16");
    TORCH_CHECK(residual_out.numel() == kNosplitSpecializedK2048, "residual_out size mismatch");
    TORCH_CHECK(gate_up_out.numel() == weight.size(0), "gate_up_out size mismatch");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int grid_blocks = (n + 3) / 4;
    c10::cuda::CUDAGuard device_guard(residual.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    fused_add_rmsnorm_gateup_k2048_hfma2_fwd_kernel<<<grid_blocks, kRow4Threads, 0, stream>>>(
        residual.data_ptr<at::Half>(),
        x.data_ptr<at::Half>(),
        norm_weight.data_ptr<at::Half>(),
        weight.data_ptr<at::Half>(),
        has_bias ? bias.data_ptr<at::Half>() : nullptr,
        residual_out.data_ptr<at::Half>(),
        gate_up_out.data_ptr<at::Half>(),
        static_cast<float>(eps),
        weight.stride(0),
        n,
        has_bias
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {residual_out, gate_up_out};
}

torch::Tensor fused_silu_downproj_half_cuda(
    const torch::Tensor& gate_up,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& out,
    bool has_bias) {
    TORCH_CHECK(gate_up.is_cuda(), "gate_up must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(gate_up.scalar_type() == torch::kFloat16, "fused_silu_downproj only supports fp16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "fused_silu_downproj only supports fp16");
    TORCH_CHECK(weight.stride(1) == 1, "fused_silu_downproj requires contiguous weight");
    TORCH_CHECK(gate_up.dim() == 1, "gate_up must be flattened [12288]");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(gate_up.numel() == 12288, "fused_silu_downproj expects gate_up size 12288");
    TORCH_CHECK(weight.size(1) == 6144, "fused_silu_downproj expects weight K=6144");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(out.numel() == weight.size(0), "out size mismatch");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == weight.size(0), "bias size mismatch");
    }

    const int n = static_cast<int>(weight.size(0));
    const int grid_blocks = (n + 3) / 4;
    c10::cuda::CUDAGuard device_guard(gate_up.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    fused_silu_downproj_k6144_hfma2_fwd_kernel<<<grid_blocks, kRow4Threads, 0, stream>>>(
        gate_up.data_ptr<at::Half>(),
        weight.data_ptr<at::Half>(),
        has_bias ? bias.data_ptr<at::Half>() : nullptr,
        out.data_ptr<at::Half>(),
        weight.stride(0),
        n,
        has_bias
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_gateup_silu_downproj_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& bias,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out,
    bool has_bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(gate_up_weight.is_cuda(), "gate_up_weight must be CUDA");
    TORCH_CHECK(down_proj_weight.is_cuda(), "down_proj_weight must be CUDA");
    TORCH_CHECK(hidden_out.is_cuda(), "hidden_out must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "fused_gateup_silu_downproj only supports fp16");
    TORCH_CHECK(gate_up_weight.scalar_type() == torch::kFloat16, "fused_gateup_silu_downproj only supports fp16");
    TORCH_CHECK(down_proj_weight.scalar_type() == torch::kFloat16, "fused_gateup_silu_downproj only supports fp16");
    TORCH_CHECK(gate_up_weight.stride(1) == 1, "fused_gateup_silu_downproj requires contiguous gate_up_weight");
    TORCH_CHECK(down_proj_weight.stride(1) == 1, "fused_gateup_silu_downproj requires contiguous down_proj_weight");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(gate_up_weight.dim() == 2, "gate_up_weight must be [12288, 2048]");
    TORCH_CHECK(down_proj_weight.dim() == 2, "down_proj_weight must be [2048, 6144]");
    TORCH_CHECK(x.numel() == kNosplitSpecializedK2048, "fused_gateup_silu_downproj expects K=2048");
    TORCH_CHECK(gate_up_weight.size(0) == 12288 && gate_up_weight.size(1) == kNosplitSpecializedK2048,
        "fused_gateup_silu_downproj expects gate_up_weight [12288, 2048]");
    TORCH_CHECK(down_proj_weight.size(0) == 2048 && down_proj_weight.size(1) == 6144,
        "fused_gateup_silu_downproj expects down_proj_weight [2048, 6144]");
    TORCH_CHECK(hidden_out.scalar_type() == torch::kFloat16, "hidden_out must be fp16");
    TORCH_CHECK(hidden_out.numel() == 6144, "hidden_out size mismatch");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(out.numel() == down_proj_weight.size(0), "out size mismatch");
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA when present");
        TORCH_CHECK(bias.numel() == down_proj_weight.size(0), "bias size mismatch");
    }

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int hidden_grid_blocks = (6144 + 3) / 4;
    fused_gateup_silu_k2048_hfma2_fwd_kernel<<<hidden_grid_blocks, kRow4Threads, 0, stream>>>(
        x.data_ptr<at::Half>(),
        gate_up_weight.data_ptr<at::Half>(),
        hidden_out.data_ptr<at::Half>(),
        gate_up_weight.stride(0)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, x.device().index());
    launch_nosplit_variant(
        hidden_out.data_ptr<at::Half>(),
        down_proj_weight.data_ptr<at::Half>(),
        has_bias ? bias.data_ptr<at::Half>() : nullptr,
        out.data_ptr<at::Half>(),
        down_proj_weight.stride(0),
        static_cast<int>(down_proj_weight.size(0)),
        static_cast<int>(down_proj_weight.size(1)),
        has_bias,
        kNosplitVariantK6144N8K1024XWarpWAsync,
        num_sms,
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_gateup_silu_downproj_mega_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& out) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(gate_up_weight.is_cuda(), "gate_up_weight must be CUDA");
    TORCH_CHECK(down_proj_weight.is_cuda(), "down_proj_weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "fused_gateup_silu_downproj_mega only supports fp16");
    TORCH_CHECK(gate_up_weight.scalar_type() == torch::kFloat16, "fused_gateup_silu_downproj_mega only supports fp16");
    TORCH_CHECK(down_proj_weight.scalar_type() == torch::kFloat16, "fused_gateup_silu_downproj_mega only supports fp16");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(gate_up_weight.stride(1) == 1, "fused_gateup_silu_downproj_mega requires contiguous gate_up_weight");
    TORCH_CHECK(down_proj_weight.stride(1) == 1, "fused_gateup_silu_downproj_mega requires contiguous down_proj_weight");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(gate_up_weight.dim() == 2, "gate_up_weight must be [12288, 2048]");
    TORCH_CHECK(down_proj_weight.dim() == 2, "down_proj_weight must be [2048, 6144]");
    TORCH_CHECK(x.numel() == kNosplitSpecializedK2048, "fused_gateup_silu_downproj_mega expects K=2048");
    TORCH_CHECK(gate_up_weight.size(0) == 12288 && gate_up_weight.size(1) == kNosplitSpecializedK2048,
        "fused_gateup_silu_downproj_mega expects gate_up_weight [12288, 2048]");
    TORCH_CHECK(down_proj_weight.size(0) == 2048 && down_proj_weight.size(1) == 6144,
        "fused_gateup_silu_downproj_mega expects down_proj_weight [2048, 6144]");
    TORCH_CHECK(out.numel() == down_proj_weight.size(0), "out size mismatch");

    const int grid_blocks = (6144 + kFusedMegaHiddenBatch - 1) / kFusedMegaHiddenBatch;
    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    fused_gateup_silu_downproj_mega_k2048_k6144_fwd_kernel<<<grid_blocks, kRow8Threads, 0, stream>>>(
        x.data_ptr<at::Half>(),
        gate_up_weight.data_ptr<at::Half>(),
        down_proj_weight.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        gate_up_weight.stride(0),
        down_proj_weight.stride(0)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_gateup_silu_downproj_mega_partial_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& partial_out) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(gate_up_weight.is_cuda(), "gate_up_weight must be CUDA");
    TORCH_CHECK(down_proj_weight.is_cuda(), "down_proj_weight must be CUDA");
    TORCH_CHECK(partial_out.is_cuda(), "partial_out must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "fused_gateup_silu_downproj_mega_partial only supports fp16");
    TORCH_CHECK(gate_up_weight.scalar_type() == torch::kFloat16, "fused_gateup_silu_downproj_mega_partial only supports fp16");
    TORCH_CHECK(down_proj_weight.scalar_type() == torch::kFloat16, "fused_gateup_silu_downproj_mega_partial only supports fp16");
    TORCH_CHECK(partial_out.scalar_type() == torch::kFloat16, "partial_out must be fp16");
    TORCH_CHECK(gate_up_weight.stride(1) == 1, "fused_gateup_silu_downproj_mega_partial requires contiguous gate_up_weight");
    TORCH_CHECK(down_proj_weight.stride(1) == 1, "fused_gateup_silu_downproj_mega_partial requires contiguous down_proj_weight");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(gate_up_weight.dim() == 2, "gate_up_weight must be [12288, 2048]");
    TORCH_CHECK(down_proj_weight.dim() == 2, "down_proj_weight must be [2048, 6144]");
    TORCH_CHECK(x.numel() == kNosplitSpecializedK2048, "fused_gateup_silu_downproj_mega_partial expects K=2048");
    const int grid_blocks = (6144 + kFusedMegaHiddenBatch - 1) / kFusedMegaHiddenBatch;
    TORCH_CHECK(partial_out.dim() == 2, "partial_out must be [grid_blocks, 2048]");
    TORCH_CHECK(partial_out.size(0) == grid_blocks && partial_out.size(1) == 2048, "partial_out shape mismatch");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    fused_gateup_silu_downproj_mega_partial_k2048_k6144_fwd_kernel<<<grid_blocks, kRow8Threads, 0, stream>>>(
        x.data_ptr<at::Half>(),
        gate_up_weight.data_ptr<at::Half>(),
        down_proj_weight.data_ptr<at::Half>(),
        partial_out.data_ptr<at::Half>(),
        gate_up_weight.stride(0),
        down_proj_weight.stride(0)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return partial_out;
}

// gridsync kernels + launch wrappers are in decode_gemv_gridsync.cu

torch::Tensor fused_gateup_silu_downproj_softsync_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& barrier_count,
    const torch::Tensor& barrier_phase,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(gate_up_weight.is_cuda(), "gate_up_weight must be CUDA");
    TORCH_CHECK(down_proj_weight.is_cuda(), "down_proj_weight must be CUDA");
    TORCH_CHECK(barrier_count.is_cuda(), "barrier_count must be CUDA");
    TORCH_CHECK(barrier_phase.is_cuda(), "barrier_phase must be CUDA");
    TORCH_CHECK(hidden_out.is_cuda(), "hidden_out must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "fused_gateup_silu_downproj_softsync only supports fp16");
    TORCH_CHECK(gate_up_weight.scalar_type() == torch::kFloat16, "fused_gateup_silu_downproj_softsync only supports fp16");
    TORCH_CHECK(down_proj_weight.scalar_type() == torch::kFloat16, "fused_gateup_silu_downproj_softsync only supports fp16");
    TORCH_CHECK(barrier_count.scalar_type() == torch::kInt32, "barrier_count must be int32");
    TORCH_CHECK(barrier_phase.scalar_type() == torch::kInt32, "barrier_phase must be int32");
    TORCH_CHECK(hidden_out.scalar_type() == torch::kFloat16, "hidden_out must be fp16");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(gate_up_weight.stride(1) == 1, "fused_gateup_silu_downproj_softsync requires contiguous gate_up_weight");
    TORCH_CHECK(down_proj_weight.stride(1) == 1, "fused_gateup_silu_downproj_softsync requires contiguous down_proj_weight");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(gate_up_weight.dim() == 2, "gate_up_weight must be [12288, 2048]");
    TORCH_CHECK(down_proj_weight.dim() == 2, "down_proj_weight must be [2048, 6144]");
    TORCH_CHECK(x.numel() == kNosplitSpecializedK2048, "fused_gateup_silu_downproj_softsync expects K=2048");
    TORCH_CHECK(gate_up_weight.size(0) == 12288 && gate_up_weight.size(1) == kNosplitSpecializedK2048,
        "fused_gateup_silu_downproj_softsync expects gate_up_weight [12288, 2048]");
    TORCH_CHECK(down_proj_weight.size(0) == 2048 && down_proj_weight.size(1) == 6144,
        "fused_gateup_silu_downproj_softsync expects down_proj_weight [2048, 6144]");
    TORCH_CHECK(barrier_count.dim() == 1 && barrier_count.numel() == 1, "barrier_count must be [1]");
    TORCH_CHECK(barrier_phase.dim() == 1 && barrier_phase.numel() == 1, "barrier_phase must be [1]");
    TORCH_CHECK(hidden_out.dim() == 1 && hidden_out.numel() == 6144, "hidden_out must be [6144]");
    TORCH_CHECK(out.dim() == 1 && out.numel() == 2048, "out must be [2048]");

    constexpr int grid_blocks = 2048 / kRowsPerBlockAsync;
    static_assert(grid_blocks * kRowsPerBlockAsync == 2048, "softsync launch grid must fully cover 2048 rows");

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    int num_sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, x.device().index()));

    int max_blocks_per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        fused_gateup_silu_downproj_softsync_k2048_k6144_fwd_kernel,
        kRow8Threads,
        0));
    TORCH_CHECK(
        max_blocks_per_sm * num_sms >= grid_blocks,
        "fused_gateup_silu_downproj_softsync launch would deadlock: resident blocks ",
        max_blocks_per_sm * num_sms,
        " < required grid blocks ",
        grid_blocks);

    C10_CUDA_CHECK(cudaMemsetAsync(barrier_count.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(barrier_phase.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));

    fused_gateup_silu_downproj_softsync_k2048_k6144_fwd_kernel<<<grid_blocks, kRow8Threads, 0, stream>>>(
        x.data_ptr<at::Half>(),
        gate_up_weight.data_ptr<at::Half>(),
        down_proj_weight.data_ptr<at::Half>(),
        barrier_count.data_ptr<int32_t>(),
        barrier_phase.data_ptr<int32_t>(),
        hidden_out.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        gate_up_weight.stride(0),
        down_proj_weight.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_gateup_silu_downproj_softsync_profile_half_cuda(
    const torch::Tensor& x,
    const torch::Tensor& gate_up_weight,
    const torch::Tensor& down_proj_weight,
    const torch::Tensor& barrier_count,
    const torch::Tensor& barrier_phase,
    const torch::Tensor& hidden_out,
    const torch::Tensor& out,
    const torch::Tensor& profile) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(gate_up_weight.is_cuda(), "gate_up_weight must be CUDA");
    TORCH_CHECK(down_proj_weight.is_cuda(), "down_proj_weight must be CUDA");
    TORCH_CHECK(barrier_count.is_cuda(), "barrier_count must be CUDA");
    TORCH_CHECK(barrier_phase.is_cuda(), "barrier_phase must be CUDA");
    TORCH_CHECK(hidden_out.is_cuda(), "hidden_out must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(profile.is_cuda(), "profile must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "fused_gateup_silu_downproj_softsync_profile only supports fp16");
    TORCH_CHECK(gate_up_weight.scalar_type() == torch::kFloat16, "softsync_profile only supports fp16 gate_up_weight");
    TORCH_CHECK(down_proj_weight.scalar_type() == torch::kFloat16, "softsync_profile only supports fp16 down_proj_weight");
    TORCH_CHECK(barrier_count.scalar_type() == torch::kInt32, "barrier_count must be int32");
    TORCH_CHECK(barrier_phase.scalar_type() == torch::kInt32, "barrier_phase must be int32");
    TORCH_CHECK(hidden_out.scalar_type() == torch::kFloat16, "hidden_out must be fp16");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(profile.scalar_type() == torch::kInt64, "profile must be int64");
    TORCH_CHECK(profile.is_contiguous(), "profile must be contiguous");
    TORCH_CHECK(gate_up_weight.stride(1) == 1, "softsync_profile requires contiguous gate_up_weight");
    TORCH_CHECK(down_proj_weight.stride(1) == 1, "softsync_profile requires contiguous down_proj_weight");
    TORCH_CHECK(x.dim() == 1, "x must be flattened [K]");
    TORCH_CHECK(gate_up_weight.dim() == 2, "gate_up_weight must be [12288, 2048]");
    TORCH_CHECK(down_proj_weight.dim() == 2, "down_proj_weight must be [2048, 6144]");
    TORCH_CHECK(x.numel() == kNosplitSpecializedK2048, "softsync_profile expects K=2048");
    TORCH_CHECK(gate_up_weight.size(0) == 12288 && gate_up_weight.size(1) == kNosplitSpecializedK2048,
        "softsync_profile expects gate_up_weight [12288, 2048]");
    TORCH_CHECK(down_proj_weight.size(0) == 2048 && down_proj_weight.size(1) == 6144,
        "softsync_profile expects down_proj_weight [2048, 6144]");
    TORCH_CHECK(barrier_count.dim() == 1 && barrier_count.numel() == 1, "barrier_count must be [1]");
    TORCH_CHECK(barrier_phase.dim() == 1 && barrier_phase.numel() == 1, "barrier_phase must be [1]");
    TORCH_CHECK(hidden_out.dim() == 1 && hidden_out.numel() == 6144, "hidden_out must be [6144]");
    TORCH_CHECK(out.dim() == 1 && out.numel() == 2048, "out must be [2048]");

    constexpr int grid_blocks = 2048 / kRowsPerBlockAsync;
    static_assert(grid_blocks * kRowsPerBlockAsync == 2048, "softsync_profile launch grid must fully cover 2048 rows");
    TORCH_CHECK(
        profile.numel() == static_cast<int64_t>(grid_blocks) * kSoftsyncProfileSlots,
        "profile tensor shape mismatch: expected [grid_blocks, kSoftsyncProfileSlots]");

    int num_sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, x.device().index()));

    int max_blocks_per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        fused_gateup_silu_downproj_softsync_profile_k2048_k6144_fwd_kernel<true>,
        kRow8Threads,
        0));
    TORCH_CHECK(
        max_blocks_per_sm * num_sms >= grid_blocks,
        "fused_gateup_silu_downproj_softsync_profile launch would deadlock: resident blocks ",
        max_blocks_per_sm * num_sms,
        " < required grid blocks ",
        grid_blocks);

    c10::cuda::CUDAGuard device_guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    C10_CUDA_CHECK(cudaMemsetAsync(barrier_count.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(barrier_phase.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
    profile.zero_();

    fused_gateup_silu_downproj_softsync_profile_k2048_k6144_fwd_kernel<true><<<grid_blocks, kRow8Threads, 0, stream>>>(
        x.data_ptr<at::Half>(),
        gate_up_weight.data_ptr<at::Half>(),
        down_proj_weight.data_ptr<at::Half>(),
        barrier_count.data_ptr<int32_t>(),
        barrier_phase.data_ptr<int32_t>(),
        hidden_out.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        reinterpret_cast<uint64_t*>(profile.data_ptr<int64_t>()),
        gate_up_weight.stride(0),
        down_proj_weight.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor weight_stream_read_half_cuda(
    const torch::Tensor& weight,
    const torch::Tensor& out) {
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "weight stream read only supports fp16");
    TORCH_CHECK(weight.is_contiguous(), "weight stream read requires contiguous weight");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(out.scalar_type() == torch::kInt, "out must be int32");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    constexpr int kWeightStreamThreads = 256;
    const int warps_per_block = kWeightStreamThreads / 32;
    TORCH_CHECK(out.numel() % warps_per_block == 0, "out size must be divisible by warps_per_block");

    const int64_t num_vecs = weight.numel() / kHalfPerVec128;
    TORCH_CHECK(
        num_vecs * kHalfPerVec128 == weight.numel(),
        "weight stream read requires element count divisible by 8");

    const int grid_blocks = static_cast<int>(out.numel() / warps_per_block);
    TORCH_CHECK(grid_blocks > 0, "weight stream read requires non-empty output");

    c10::cuda::CUDAGuard device_guard(weight.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    weight_stream_read_half_fwd_kernel<<<grid_blocks, kWeightStreamThreads, 0, stream>>>(
        weight.data_ptr<at::Half>(),
        out.data_ptr<int32_t>(),
        num_vecs);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

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
    int64_t smid_threshold) {
    TORCH_CHECK(weight_a.is_cuda(), "weight_a must be CUDA");
    TORCH_CHECK(weight_b.is_cuda(), "weight_b must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(trace.is_cuda(), "trace must be CUDA");
    TORCH_CHECK(smid.is_cuda(), "smid must be CUDA");
    TORCH_CHECK(route.is_cuda(), "route must be CUDA");
    TORCH_CHECK(weight_a.scalar_type() == torch::kFloat16, "dual weight stream only supports fp16");
    TORCH_CHECK(weight_b.scalar_type() == torch::kFloat16, "dual weight stream only supports fp16");
    TORCH_CHECK(weight_a.is_contiguous(), "dual weight stream requires contiguous weight_a");
    TORCH_CHECK(weight_b.is_contiguous(), "dual weight stream requires contiguous weight_b");
    TORCH_CHECK(weight_a.sizes() == weight_b.sizes(), "dual weight stream weight size mismatch");
    TORCH_CHECK(weight_a.dim() == 2, "weight_a must be [N, K]");
    TORCH_CHECK(out.scalar_type() == torch::kInt, "out must be int32");
    TORCH_CHECK(trace.scalar_type() == torch::kInt64, "trace must be int64");
    TORCH_CHECK(smid.scalar_type() == torch::kInt, "smid must be int32");
    TORCH_CHECK(route.scalar_type() == torch::kInt, "route must be int32");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(trace.is_contiguous(), "trace must be contiguous");
    TORCH_CHECK(smid.is_contiguous(), "smid must be contiguous");
    TORCH_CHECK(route.is_contiguous(), "route must be contiguous");
    TORCH_CHECK(vecs_per_block > 0, "vecs_per_block must be > 0");
    TORCH_CHECK(
        route_mode >= kDualCopyRouteForceA && route_mode <= kDualCopyRouteTable,
        "invalid dual weight stream route_mode: ",
        route_mode);

    constexpr int kWeightStreamThreads = 256;
    const int warps_per_block = kWeightStreamThreads / 32;
    TORCH_CHECK(out.numel() % warps_per_block == 0, "out size must be divisible by warps_per_block");
    const int grid_blocks = static_cast<int>(out.numel() / warps_per_block);
    TORCH_CHECK(grid_blocks > 0, "dual weight stream requires non-empty output");
    TORCH_CHECK(trace.numel() == static_cast<int64_t>(grid_blocks) * 2, "trace size mismatch");
    TORCH_CHECK(smid.numel() == grid_blocks, "smid size mismatch");
    TORCH_CHECK(route.numel() == grid_blocks, "route size mismatch");

    const int64_t total_vecs = weight_a.numel() / kHalfPerVec128;
    TORCH_CHECK(total_vecs * kHalfPerVec128 == weight_a.numel(), "dual weight stream requires element count divisible by 8");
    TORCH_CHECK(
        static_cast<int64_t>(grid_blocks) * vecs_per_block <= total_vecs,
        "dual weight stream grid * vecs_per_block exceeds tensor size");

    int num_sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, weight_a.device().index()));
    if (route_mode == kDualCopyRouteTable) {
        TORCH_CHECK(route_table.is_cuda(), "route_table must be CUDA for table routing");
        TORCH_CHECK(route_table.scalar_type() == torch::kInt, "route_table must be int32");
        TORCH_CHECK(route_table.is_contiguous(), "route_table must be contiguous");
        TORCH_CHECK(route_table.numel() >= num_sms, "route_table too small for SM count");
    }

    c10::cuda::CUDAGuard device_guard(weight_a.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    weight_stream_read_dual_half_fwd_kernel<<<grid_blocks, kWeightStreamThreads, 0, stream>>>(
        weight_a.data_ptr<at::Half>(),
        weight_b.data_ptr<at::Half>(),
        route_mode == kDualCopyRouteTable ? route_table.data_ptr<int32_t>() : nullptr,
        out.data_ptr<int32_t>(),
        trace.data_ptr<int64_t>(),
        smid.data_ptr<int32_t>(),
        route.data_ptr<int32_t>(),
        vecs_per_block,
        static_cast<int>(route_mode),
        static_cast<int>(smid_threshold));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

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
    int64_t swap_order) {
    TORCH_CHECK(chain.is_cuda(), "chain must be CUDA");
    TORCH_CHECK(start_even.is_cuda(), "start_even must be CUDA");
    TORCH_CHECK(start_odd.is_cuda(), "start_odd must be CUDA");
    TORCH_CHECK(flush_buf.is_cuda(), "flush_buf must be CUDA");
    TORCH_CHECK(trace.is_cuda(), "trace must be CUDA");
    TORCH_CHECK(smid.is_cuda(), "smid must be CUDA");
    TORCH_CHECK(final_idx.is_cuda(), "final_idx must be CUDA");
    TORCH_CHECK(flush_sink.is_cuda(), "flush_sink must be CUDA");
    TORCH_CHECK(chain.scalar_type() == torch::kInt64, "chain must be int64");
    TORCH_CHECK(start_even.scalar_type() == torch::kInt64, "start_even must be int64");
    TORCH_CHECK(start_odd.scalar_type() == torch::kInt64, "start_odd must be int64");
    TORCH_CHECK(flush_buf.scalar_type() == torch::kInt32, "flush_buf must be int32");
    TORCH_CHECK(trace.scalar_type() == torch::kInt64, "trace must be int64");
    TORCH_CHECK(smid.scalar_type() == torch::kInt, "smid must be int32");
    TORCH_CHECK(final_idx.scalar_type() == torch::kInt64, "final_idx must be int64");
    TORCH_CHECK(flush_sink.scalar_type() == torch::kInt, "flush_sink must be int32");
    TORCH_CHECK(chain.is_contiguous(), "chain must be contiguous");
    TORCH_CHECK(start_even.is_contiguous(), "start_even must be contiguous");
    TORCH_CHECK(start_odd.is_contiguous(), "start_odd must be contiguous");
    TORCH_CHECK(flush_buf.is_contiguous(), "flush_buf must be contiguous");
    TORCH_CHECK(trace.is_contiguous(), "trace must be contiguous");
    TORCH_CHECK(smid.is_contiguous(), "smid must be contiguous");
    TORCH_CHECK(final_idx.is_contiguous(), "final_idx must be contiguous");
    TORCH_CHECK(flush_sink.is_contiguous(), "flush_sink must be contiguous");
    TORCH_CHECK(chain.dim() == 1, "chain must be 1D");
    TORCH_CHECK(start_even.dim() == 1, "start_even must be 1D");
    TORCH_CHECK(start_odd.dim() == 1, "start_odd must be 1D");
    TORCH_CHECK(chain_len > 0, "chain_len must be > 0");

    const int64_t num_blocks = start_even.numel();
    TORCH_CHECK(start_odd.numel() == num_blocks, "start_odd size mismatch");
    TORCH_CHECK(trace.numel() == num_blocks * 4, "trace size mismatch");
    TORCH_CHECK(smid.numel() == num_blocks, "smid size mismatch");
    TORCH_CHECK(final_idx.numel() == num_blocks * 2, "final_idx size mismatch");
    TORCH_CHECK(flush_sink.numel() == num_blocks, "flush_sink size mismatch");
    TORCH_CHECK(flush_buf.numel() % num_blocks == 0, "flush_buf size must divide evenly by block count");

    const int64_t flush_elems_per_block = flush_buf.numel() / num_blocks;
    TORCH_CHECK(flush_elems_per_block > 0, "flush buffer per block must be > 0");

    c10::cuda::CUDAGuard device_guard(chain.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        l2_8kb_pointer_chase_pair_fwd_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        l2_8kb_pointer_chase_pair_fwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kL2ParityProbeDynamicSmemBytes));

    int max_blocks_per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        l2_8kb_pointer_chase_pair_fwd_kernel,
        kL2ParityProbeThreads,
        kL2ParityProbeDynamicSmemBytes));
    TORCH_CHECK(
        max_blocks_per_sm == 1,
        "l2_8kb pointer-chase probe expected 1 resident block/SM, got ",
        max_blocks_per_sm);

    trace.zero_();
    smid.fill_(-1);
    flush_sink.zero_();
    l2_8kb_pointer_chase_pair_fwd_kernel<<<
        static_cast<int>(num_blocks),
        kL2ParityProbeThreads,
        kL2ParityProbeDynamicSmemBytes,
        stream>>>(
            chain.data_ptr<int64_t>(),
            start_even.data_ptr<int64_t>(),
            start_odd.data_ptr<int64_t>(),
            flush_buf.data_ptr<int32_t>(),
            flush_elems_per_block,
            chain_len,
            static_cast<int>(swap_order != 0),
            trace.data_ptr<int64_t>(),
            smid.data_ptr<int32_t>(),
            final_idx.data_ptr<int64_t>(),
            flush_sink.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return trace;
}

torch::Tensor l2_sequential_pointer_chase_trace_cuda(
    const torch::Tensor& chain,
    const torch::Tensor& start,
    const torch::Tensor& flush_buf,
    const torch::Tensor& latency,
    const torch::Tensor& smid,
    const torch::Tensor& final_idx,
    const torch::Tensor& flush_sink,
    int64_t chain_len) {
    TORCH_CHECK(chain.is_cuda(), "chain must be CUDA");
    TORCH_CHECK(start.is_cuda(), "start must be CUDA");
    TORCH_CHECK(flush_buf.is_cuda(), "flush_buf must be CUDA");
    TORCH_CHECK(latency.is_cuda(), "latency must be CUDA");
    TORCH_CHECK(smid.is_cuda(), "smid must be CUDA");
    TORCH_CHECK(final_idx.is_cuda(), "final_idx must be CUDA");
    TORCH_CHECK(flush_sink.is_cuda(), "flush_sink must be CUDA");
    TORCH_CHECK(chain.scalar_type() == torch::kInt64, "chain must be int64");
    TORCH_CHECK(start.scalar_type() == torch::kInt64, "start must be int64");
    TORCH_CHECK(flush_buf.scalar_type() == torch::kInt32, "flush_buf must be int32");
    TORCH_CHECK(latency.scalar_type() == torch::kInt64, "latency must be int64");
    TORCH_CHECK(smid.scalar_type() == torch::kInt, "smid must be int32");
    TORCH_CHECK(final_idx.scalar_type() == torch::kInt64, "final_idx must be int64");
    TORCH_CHECK(flush_sink.scalar_type() == torch::kInt, "flush_sink must be int32");
    TORCH_CHECK(chain.is_contiguous(), "chain must be contiguous");
    TORCH_CHECK(start.is_contiguous(), "start must be contiguous");
    TORCH_CHECK(flush_buf.is_contiguous(), "flush_buf must be contiguous");
    TORCH_CHECK(latency.is_contiguous(), "latency must be contiguous");
    TORCH_CHECK(smid.is_contiguous(), "smid must be contiguous");
    TORCH_CHECK(final_idx.is_contiguous(), "final_idx must be contiguous");
    TORCH_CHECK(flush_sink.is_contiguous(), "flush_sink must be contiguous");
    TORCH_CHECK(chain.dim() == 1, "chain must be 1D");
    TORCH_CHECK(start.dim() == 1, "start must be 1D");
    TORCH_CHECK(chain_len > 0, "chain_len must be > 0");

    const int64_t num_blocks = start.numel();
    TORCH_CHECK(num_blocks > 0, "start must be non-empty");
    TORCH_CHECK(latency.numel() == num_blocks * chain_len, "latency size mismatch");
    TORCH_CHECK(smid.numel() == num_blocks, "smid size mismatch");
    TORCH_CHECK(final_idx.numel() == num_blocks, "final_idx size mismatch");
    TORCH_CHECK(flush_sink.numel() == num_blocks, "flush_sink size mismatch");
    TORCH_CHECK(flush_buf.numel() % num_blocks == 0, "flush_buf size must divide evenly by block count");

    const int64_t flush_elems_per_block = flush_buf.numel() / num_blocks;
    TORCH_CHECK(flush_elems_per_block > 0, "flush buffer per block must be > 0");

    c10::cuda::CUDAGuard device_guard(chain.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        l2_sequential_pointer_chase_trace_fwd_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        l2_sequential_pointer_chase_trace_fwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kL2ParityProbeDynamicSmemBytes));

    int max_blocks_per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        l2_sequential_pointer_chase_trace_fwd_kernel,
        kL2ParityProbeThreads,
        kL2ParityProbeDynamicSmemBytes));
    TORCH_CHECK(
        max_blocks_per_sm == 1,
        "l2 sequential pointer-chase trace expected 1 resident block/SM, got ",
        max_blocks_per_sm);

    latency.zero_();
    smid.fill_(-1);
    flush_sink.zero_();
    l2_sequential_pointer_chase_trace_fwd_kernel<<<
        static_cast<int>(num_blocks),
        kL2ParityProbeThreads,
        kL2ParityProbeDynamicSmemBytes,
        stream>>>(
            chain.data_ptr<int64_t>(),
            start.data_ptr<int64_t>(),
            flush_buf.data_ptr<int32_t>(),
            flush_elems_per_block,
            chain_len,
            latency.data_ptr<int64_t>(),
            smid.data_ptr<int32_t>(),
            final_idx.data_ptr<int64_t>(),
            flush_sink.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return latency;
}

torch::Tensor nosplit_weight_only_half_variant_cuda(
    const torch::Tensor& weight,
    const torch::Tensor& out,
    int64_t variant) {
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "nosplit weight-only variant only supports fp16");
    TORCH_CHECK(weight.stride(1) == 1, "nosplit weight-only variant requires contiguous weight");
    TORCH_CHECK(weight.dim() == 2, "weight must be [N, K]");
    TORCH_CHECK(out.scalar_type() == torch::kInt, "out must be int32");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    const int n = static_cast<int>(weight.size(0));
    const int k = static_cast<int>(weight.size(1));
    TORCH_CHECK(k == 2048 || k == 6144, "nosplit weight-only variant currently supports only k=2048/6144");
    TORCH_CHECK(out.numel() == n, "out size mismatch");

    c10::cuda::CUDAGuard device_guard(weight.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    switch (variant) {
        case kNosplitVariantK2048Row2: {
            const int grid_blocks = (n + 1) / 2;
            nosplit_weight_only_k2048_row2_fwd_kernel<<<grid_blocks, kRow2Threads, 0, stream>>>(
                weight.data_ptr<at::Half>(),
                out.data_ptr<int32_t>(),
                weight.stride(0),
                n);
            break;
        }
        case kNosplitVariantK2048Row4: {
            const int grid_blocks = (n + kRowsPerBlock - 1) / kRowsPerBlock;
            nosplit_weight_only_k2048_row4_fwd_kernel<<<grid_blocks, kRow4Threads, 0, stream>>>(
                weight.data_ptr<at::Half>(),
                out.data_ptr<int32_t>(),
                weight.stride(0),
                n);
            break;
        }
        case kNosplitVariantK6144Row4: {
            const int grid_blocks = (n + kRowsPerBlock - 1) / kRowsPerBlock;
            nosplit_weight_only_k6144_row4_fwd_kernel<<<grid_blocks, kRow4Threads, 0, stream>>>(
                weight.data_ptr<at::Half>(),
                out.data_ptr<int32_t>(),
                weight.stride(0),
                n);
            break;
        }
        case kNosplitVariantK6144N8K1024XWarpWAsync: {
            const int grid_blocks = (n + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;
            nosplit_weight_only_k6144_n8k1024_xwarp_w_async_fwd_kernel<<<grid_blocks, kRow8Threads, 0, stream>>>(
                weight.data_ptr<at::Half>(),
                out.data_ptr<int32_t>(),
                weight.stride(0),
                n);
            break;
        }
        case kNosplitVariantK2048Row8Persistent: {
            TORCH_CHECK(k == kNosplitSpecializedK2048, "k2048_row8_persistent variant requires k=2048");
            int num_sms = 0;
            cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, weight.device().index());
            const int full_grid_blocks = (n + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;
            const int launch_grid_blocks = std::min(full_grid_blocks, std::max(1, num_sms * 2));
            nosplit_weight_only_k2048_row8_persistent_fwd_kernel<<<launch_grid_blocks, kRow8Threads, 0, stream>>>(
                weight.data_ptr<at::Half>(),
                out.data_ptr<int32_t>(),
                weight.stride(0),
                n);
            break;
        }
        case kNosplitVariantK2048Row4Persistent: {
            TORCH_CHECK(k == kNosplitSpecializedK2048, "k2048_row4_persistent variant requires k=2048");
            int num_sms = 0;
            cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, weight.device().index());
            const int full_grid_blocks = (n + kRowsPerBlock - 1) / kRowsPerBlock;
            const int launch_grid_blocks = std::min(full_grid_blocks, std::max(1, num_sms * 2));
            nosplit_weight_only_k2048_row4_persistent_fwd_kernel<<<launch_grid_blocks, kRow4Threads, 0, stream>>>(
                weight.data_ptr<at::Half>(),
                out.data_ptr<int32_t>(),
                weight.stride(0),
                n);
            break;
        }
        default:
            TORCH_CHECK(
                false,
                "nosplit weight-only variant currently supports only k2048_row2, k2048_row4, k2048_row4_persistent, k2048_row8_persistent, "
                "k6144_row4 and k6144_n8k1024_xwarp_w_async");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
