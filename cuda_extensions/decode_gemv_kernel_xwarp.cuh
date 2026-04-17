__global__ void weight_stream_read_half_fwd_kernel(
    const at::Half* __restrict__ weight,
    int32_t* __restrict__ out,
    int64_t num_vecs) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int64_t global_tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const uint4* weight_vec = reinterpret_cast<const uint4*>(weight);

    unsigned int sum = 0;
    for (int64_t vec_idx = global_tid; vec_idx < num_vecs; vec_idx += stride) {
        const uint4 value = weight_vec[vec_idx];
        sum += value.x + value.y + value.z + value.w;
    }
    sum = warp_sum_u32(sum);
    if (lane == 0) {
        out[blockIdx.x * warps_per_block + warp] = static_cast<int32_t>(sum);
    }
}

__global__ void weight_stream_read_dual_half_fwd_kernel(
    const at::Half* __restrict__ weight_a,
    const at::Half* __restrict__ weight_b,
    const int32_t* __restrict__ route_table,
    int32_t* __restrict__ out,
    int64_t* __restrict__ trace,
    int32_t* __restrict__ smid_out,
    int32_t* __restrict__ route_out,
    int64_t vecs_per_block,
    int route_mode,
    int smid_threshold) {
    __shared__ uint64_t block_start;
    __shared__ int block_smid;
    __shared__ int selected_copy;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int64_t block_vec_base = static_cast<int64_t>(blockIdx.x) * vecs_per_block;

    if (threadIdx.x == 0) {
        block_smid = read_smid_int();
        selected_copy = pick_dualcopy_route(block_smid, route_table, route_mode, smid_threshold);
        if (smid_out != nullptr) {
            smid_out[blockIdx.x] = block_smid;
        }
        if (route_out != nullptr) {
            route_out[blockIdx.x] = selected_copy;
        }
        block_start = read_globaltimer();
    }
    __syncthreads();

    const uint4* weight_vec = reinterpret_cast<const uint4*>(selected_copy ? weight_b : weight_a);
    unsigned int sum = 0;
    for (int64_t vec_idx = threadIdx.x; vec_idx < vecs_per_block; vec_idx += blockDim.x) {
        const uint4 value = weight_vec[block_vec_base + vec_idx];
        sum += value.x + value.y + value.z + value.w;
    }
    sum = warp_sum_u32(sum);
    if (lane == 0) {
        out[blockIdx.x * warps_per_block + warp] = static_cast<int32_t>(sum);
    }

    __syncthreads();
    if (threadIdx.x == 0 && trace != nullptr) {
        trace[static_cast<int64_t>(blockIdx.x) * 2] = static_cast<int64_t>(block_start);
        trace[static_cast<int64_t>(blockIdx.x) * 2 + 1] = static_cast<int64_t>(read_globaltimer());
    }
}

__device__ inline void flush_l2_probe_block_segment(
    const int32_t* flush_buf,
    int64_t flush_elems_per_block,
    int32_t* flush_sink) {
    const uint32_t* flush_u32 = reinterpret_cast<const uint32_t*>(flush_buf);
    const int64_t block_base = static_cast<int64_t>(blockIdx.x) * flush_elems_per_block;

    unsigned int sum = 0;
    for (int64_t idx = threadIdx.x; idx < flush_elems_per_block; idx += blockDim.x) {
        sum += load_global_cg_u32(flush_u32 + block_base + idx);
    }
    if (flush_sink != nullptr) {
        atomicAdd(flush_sink + blockIdx.x, static_cast<int32_t>(sum));
    }
}

__global__ void l2_sequential_pointer_chase_trace_fwd_kernel(
    const int64_t* __restrict__ chain,
    const int64_t* __restrict__ start,
    const int32_t* __restrict__ flush_buf,
    int64_t flush_elems_per_block,
    int64_t chain_len,
    int64_t* __restrict__ latency_out,
    int32_t* __restrict__ smid_out,
    int64_t* __restrict__ final_idx_out,
    int32_t* __restrict__ flush_sink) {
    extern __shared__ uint8_t smem_dummy[];
    (void)smem_dummy;

    if (threadIdx.x == 0) {
        smid_out[blockIdx.x] = read_smid_int();

        // Warm page-table/TLB state once so the timed phase mostly reflects
        // cache / memory latency rather than first-touch translation overhead.
        int64_t idx = start[blockIdx.x];
        #pragma unroll 1
        for (int64_t step = 0; step < chain_len; ++step) {
            idx = static_cast<int64_t>(load_global_cg_u64(reinterpret_cast<const uint64_t*>(chain) + idx));
        }
        final_idx_out[blockIdx.x] = idx;
    }
    __syncthreads();

    flush_l2_probe_block_segment(flush_buf, flush_elems_per_block, flush_sink);
    __syncthreads();

    if (threadIdx.x == 0) {
        int64_t idx = start[blockIdx.x];
        const int64_t trace_base = static_cast<int64_t>(blockIdx.x) * chain_len;
        #pragma unroll 1
        for (int64_t step = 0; step < chain_len; ++step) {
            const uint64_t t0 = read_clock64_cycles();
            idx = static_cast<int64_t>(load_global_cg_u64(reinterpret_cast<const uint64_t*>(chain) + idx));
            const uint64_t t1 = read_clock64_cycles();
            latency_out[trace_base + step] = static_cast<int64_t>(t1 - t0);
        }
        final_idx_out[blockIdx.x] = idx;
    }
}

__global__ void l2_8kb_pointer_chase_pair_fwd_kernel(
    const int64_t* __restrict__ chain,
    const int64_t* __restrict__ start_even,
    const int64_t* __restrict__ start_odd,
    const int32_t* __restrict__ flush_buf,
    int64_t flush_elems_per_block,
    int64_t chain_len,
    int swap_order,
    int64_t* __restrict__ trace,
    int32_t* __restrict__ smid_out,
    int64_t* __restrict__ final_idx_out,
    int32_t* __restrict__ flush_sink) {
    extern __shared__ uint8_t smem_dummy[];
    (void)smem_dummy;

    if (threadIdx.x == 0) {
        smid_out[blockIdx.x] = read_smid_int();

        // Warm page-table/TLB state once so the timed phase mostly reflects
        // global-memory latency, not first-touch translation overhead.
        int64_t idx = start_even[blockIdx.x];
        #pragma unroll 1
        for (int64_t step = 0; step < chain_len; ++step) {
            idx = static_cast<int64_t>(load_global_cg_u64(reinterpret_cast<const uint64_t*>(chain) + idx));
        }
        final_idx_out[static_cast<int64_t>(blockIdx.x) * 2] = idx;

        idx = start_odd[blockIdx.x];
        #pragma unroll 1
        for (int64_t step = 0; step < chain_len; ++step) {
            idx = static_cast<int64_t>(load_global_cg_u64(reinterpret_cast<const uint64_t*>(chain) + idx));
        }
        final_idx_out[static_cast<int64_t>(blockIdx.x) * 2 + 1] = idx;
    }
    __syncthreads();

    flush_l2_probe_block_segment(flush_buf, flush_elems_per_block, flush_sink);
    __syncthreads();

    if (threadIdx.x == 0) {
        const int64_t* first_start = swap_order ? start_odd : start_even;
        int64_t idx = first_start[blockIdx.x];
        const uint64_t t0 = read_globaltimer();
        #pragma unroll 1
        for (int64_t step = 0; step < chain_len; ++step) {
            idx = static_cast<int64_t>(load_global_cg_u64(reinterpret_cast<const uint64_t*>(chain) + idx));
        }
        const uint64_t t1 = read_globaltimer();
        trace[static_cast<int64_t>(blockIdx.x) * 4] = static_cast<int64_t>(t0);
        trace[static_cast<int64_t>(blockIdx.x) * 4 + 1] = static_cast<int64_t>(t1);
        final_idx_out[static_cast<int64_t>(blockIdx.x) * 2 + (swap_order ? 1 : 0)] = idx;
    }
    __syncthreads();

    flush_l2_probe_block_segment(flush_buf, flush_elems_per_block, flush_sink);
    __syncthreads();

    if (threadIdx.x == 0) {
        const int64_t* second_start = swap_order ? start_even : start_odd;
        int64_t idx = second_start[blockIdx.x];
        const uint64_t t0 = read_globaltimer();
        #pragma unroll 1
        for (int64_t step = 0; step < chain_len; ++step) {
            idx = static_cast<int64_t>(load_global_cg_u64(reinterpret_cast<const uint64_t*>(chain) + idx));
        }
        const uint64_t t1 = read_globaltimer();
        trace[static_cast<int64_t>(blockIdx.x) * 4 + 2] = static_cast<int64_t>(t0);
        trace[static_cast<int64_t>(blockIdx.x) * 4 + 3] = static_cast<int64_t>(t1);
        final_idx_out[static_cast<int64_t>(blockIdx.x) * 2 + (swap_order ? 0 : 1)] = idx;
    }
}

__global__ void nosplit_weight_only_k6144_row4_fwd_kernel(
    const at::Half* __restrict__ weight,
    int32_t* __restrict__ out,
    int64_t stride_wn,
    int n) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * kRowsPerBlock;
    const int row = row_base + warp;
    const __half* w_half = reinterpret_cast<const __half*>(weight);
    constexpr int kVecsPerTile = kNosplitSpecializedTileK6144 / kHalfPerVec128;

    unsigned int sum = 0;
    if (row < n) {
        #pragma unroll
        for (int tile_idx = 0; tile_idx < 3; ++tile_idx) {
            const int tile_start = tile_idx * kNosplitSpecializedTileK6144;
            const uint4* w_row_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn + tile_start);
            #pragma unroll
            for (int vec_idx = lane; vec_idx < kVecsPerTile; vec_idx += 64) {
                const uint4 w_vec0 = w_row_vec[vec_idx];
                const uint4 w_vec1 = w_row_vec[vec_idx + 32];
                sum += w_vec0.x + w_vec0.y + w_vec0.z + w_vec0.w;
                sum += w_vec1.x + w_vec1.y + w_vec1.z + w_vec1.w;
            }
        }
        sum = warp_sum_u32(sum);
        if (lane == 0) {
            out[row] = static_cast<int32_t>(sum);
        }
    }
}

__global__ void nosplit_weight_only_k2048_row4_fwd_kernel(
    const at::Half* __restrict__ weight,
    int32_t* __restrict__ out,
    int64_t stride_wn,
    int n) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * kRowsPerBlock;
    const int row = row_base + warp;
    const __half* w_half = reinterpret_cast<const __half*>(weight);
    constexpr int kVecsK2048 = kNosplitSpecializedK2048 / kHalfPerVec128;

    unsigned int sum = 0;
    if (row < n) {
        const uint4* w_row_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn);
        #pragma unroll
        for (int vec_idx = lane; vec_idx < kVecsK2048; vec_idx += 64) {
            const uint4 w_vec0 = w_row_vec[vec_idx];
            const uint4 w_vec1 = w_row_vec[vec_idx + 32];
            sum += w_vec0.x + w_vec0.y + w_vec0.z + w_vec0.w;
            sum += w_vec1.x + w_vec1.y + w_vec1.z + w_vec1.w;
        }
        sum = warp_sum_u32(sum);
        if (lane == 0) {
            out[row] = static_cast<int32_t>(sum);
        }
    }
}

__global__ void nosplit_weight_only_k2048_row2_fwd_kernel(
    const at::Half* __restrict__ weight,
    int32_t* __restrict__ out,
    int64_t stride_wn,
    int n) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * 2;
    const int row = row_base + warp;
    const __half* w_half = reinterpret_cast<const __half*>(weight);
    constexpr int kVecsK2048 = kNosplitSpecializedK2048 / kHalfPerVec128;

    unsigned int sum = 0;
    if (row < n) {
        const uint4* w_row_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn);
        #pragma unroll
        for (int vec_idx = lane; vec_idx < kVecsK2048; vec_idx += 64) {
            const uint4 w_vec0 = w_row_vec[vec_idx];
            const uint4 w_vec1 = w_row_vec[vec_idx + 32];
            sum += w_vec0.x + w_vec0.y + w_vec0.z + w_vec0.w;
            sum += w_vec1.x + w_vec1.y + w_vec1.z + w_vec1.w;
        }
        sum = warp_sum_u32(sum);
        if (lane == 0) {
            out[row] = static_cast<int32_t>(sum);
        }
    }
}

__global__ void nosplit_weight_only_k2048_row8_persistent_fwd_kernel(
    const at::Half* __restrict__ weight,
    int32_t* __restrict__ out,
    int64_t stride_wn,
    int n) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const __half* w_half = reinterpret_cast<const __half*>(weight);
    constexpr int kVecsK2048 = kNosplitSpecializedK2048 / kHalfPerVec128;
    const int row_stride = gridDim.x * kRowsPerBlockAsync;

    for (int row_base = blockIdx.x * kRowsPerBlockAsync; row_base < n; row_base += row_stride) {
        const int row = row_base + warp;
        unsigned int sum = 0;
        if (row < n) {
            const uint4* w_row_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn);
            #pragma unroll
            for (int vec_idx = lane; vec_idx < kVecsK2048; vec_idx += 32) {
                const uint4 w_vec = w_row_vec[vec_idx];
                sum += w_vec.x + w_vec.y + w_vec.z + w_vec.w;
            }
            sum = warp_sum_u32(sum);
            if (lane == 0) {
                out[row] = static_cast<int32_t>(sum);
            }
        }
    }
}

__global__ void nosplit_weight_only_k2048_row4_persistent_fwd_kernel(
    const at::Half* __restrict__ weight,
    int32_t* __restrict__ out,
    int64_t stride_wn,
    int n) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const __half* w_half = reinterpret_cast<const __half*>(weight);
    constexpr int kVecsK2048 = kNosplitSpecializedK2048 / kHalfPerVec128;
    const int row_stride = gridDim.x * kRowsPerBlock;

    for (int row_base = blockIdx.x * kRowsPerBlock; row_base < n; row_base += row_stride) {
        const int row = row_base + warp;
        unsigned int sum = 0;
        if (row < n) {
            const uint4* w_row_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn);
            #pragma unroll
            for (int vec_idx = lane; vec_idx < kVecsK2048; vec_idx += 64) {
                const uint4 w_vec0 = w_row_vec[vec_idx];
                const uint4 w_vec1 = w_row_vec[vec_idx + 32];
                sum += w_vec0.x + w_vec0.y + w_vec0.z + w_vec0.w;
                sum += w_vec1.x + w_vec1.y + w_vec1.z + w_vec1.w;
            }
            sum = warp_sum_u32(sum);
            if (lane == 0) {
                out[row] = static_cast<int32_t>(sum);
            }
        }
    }
}

template <bool kProfile>
__global__ void nosplit_matvec_k6144_weight_async_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    uint64_t* __restrict__ profile,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __half x_sh[kNosplitSpecializedTileK6144];
    __shared__ __align__(16) uint4 w_sh[2][kWeightTileVecsK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * kRowsPerBlock;
    const int row = row_base + warp;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);
    const uint4* x_sh_vec = reinterpret_cast<const uint4*>(x_sh);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    float sum = 0.0f;
    maybe_record_profile<kProfile>(profile, kProfileKernelStart);

    prefetch_k6144_weight_tile<kUseCpAsync>(&w_sh[0][0], w_half, stride_wn, row_base, 0, n);
    if constexpr (kUseCpAsync) {
        cp_async_wait_group<0>();
    }
    __syncthreads();

    #pragma unroll
    for (int tile_idx = 0; tile_idx < 3; ++tile_idx) {
        const int stage = tile_idx & 1;
        const int tile_start = tile_idx * kNosplitSpecializedTileK6144;
        const uint4* x_tile_vec = reinterpret_cast<const uint4*>(x_half + tile_start);

        if (threadIdx.x < kWeightVecsPerTileK6144 / 2) {
            reinterpret_cast<uint4*>(x_sh)[threadIdx.x] = x_tile_vec[threadIdx.x];
            reinterpret_cast<uint4*>(x_sh)[threadIdx.x + kWeightVecsPerTileK6144 / 2] =
                x_tile_vec[threadIdx.x + kWeightVecsPerTileK6144 / 2];
        }
        __syncthreads();

        if (tile_idx + 1 < 3) {
            const int next_tile_start = tile_start + kNosplitSpecializedTileK6144;
            prefetch_k6144_weight_tile<kUseCpAsync>(
                &w_sh[stage ^ 1][0],
                w_half,
                stride_wn,
                row_base,
                next_tile_start,
                n);
        }

        if constexpr (kProfile) {
            maybe_record_profile<kProfile>(profile, kProfileTile0LoadDone + tile_idx * 2);
        }

        if (row < n) {
            const uint4* w_sh_row_vec = &w_sh[stage][warp * kWeightVecsPerTileK6144];
            __half2 h2_acc0 = __float2half2_rn(0.0f);
            __half2 h2_acc1 = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane; vec_idx < kWeightVecsPerTileK6144; vec_idx += 64) {
                const uint4 x_vec0 = x_sh_vec[vec_idx];
                const uint4 w_vec0 = w_sh_row_vec[vec_idx];
                const uint4 x_vec1 = x_sh_vec[vec_idx + 32];
                const uint4 w_vec1 = w_sh_row_vec[vec_idx + 32];
                accumulate_hfma2_uint4(x_vec0, w_vec0, h2_acc0);
                accumulate_hfma2_uint4(x_vec1, w_vec1, h2_acc1);
            }

            const float2 f2_0 = __half22float2(h2_acc0);
            const float2 f2_1 = __half22float2(h2_acc1);
            sum += f2_0.x + f2_0.y + f2_1.x + f2_1.y;
        }

        __syncthreads();
        if constexpr (kProfile) {
            maybe_record_profile<kProfile>(profile, kProfileTile0ComputeDone + tile_idx * 2);
        }
        if constexpr (kUseCpAsync) {
            if (tile_idx + 1 < 3) {
                cp_async_wait_group<0>();
            }
        }
        __syncthreads();
    }

    if (row < n) {
        sum = warp_sum(sum);
        if (lane == 0) {
            float result = sum;
            if (has_bias) {
                result += __half2float(reinterpret_cast<const __half*>(bias)[row]);
            }
            reinterpret_cast<__half*>(out)[row] = __float2half(result);
        }
    }
    if constexpr (kProfile) {
        __syncthreads();
        maybe_record_profile<kProfile>(profile, kProfileKernelDone);
    }
}

template <bool kProfile>
__global__ void nosplit_matvec_k6144_n8k1024_async_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    uint64_t* __restrict__ profile,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __align__(16) uint4 x_sh[2][kAsyncVecsPerTileK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * kRowsPerBlockAsync;
    const int row = row_base + warp;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    maybe_record_profile<kProfile>(profile, kProfileKernelStart);
    for (int vec_idx = threadIdx.x; vec_idx < kAsyncVecsPerTileK6144; vec_idx += kRow8Threads) {
        if constexpr (kUseCpAsync) {
            cp_async_cg_16(&x_sh[0][vec_idx], reinterpret_cast<const uint4*>(x_half) + vec_idx);
        } else {
            x_sh[0][vec_idx] = reinterpret_cast<const uint4*>(x_half)[vec_idx];
        }
    }
    if constexpr (kUseCpAsync) {
        cp_async_commit_group();
        cp_async_wait_group<0>();
    }
    __syncthreads();

    float sum = 0.0f;
    #pragma unroll
    for (int tile_idx = 0; tile_idx < 6; ++tile_idx) {
        const int stage = tile_idx & 1;
        const int tile_start = tile_idx * kNosplitAsyncTileK6144;

        if (tile_idx + 1 < 6) {
            const int next_tile_start = tile_start + kNosplitAsyncTileK6144;
            const uint4* x_next_vec = reinterpret_cast<const uint4*>(x_half + next_tile_start);
            for (int vec_idx = threadIdx.x; vec_idx < kAsyncVecsPerTileK6144; vec_idx += kRow8Threads) {
                if constexpr (kUseCpAsync) {
                    cp_async_cg_16(&x_sh[stage ^ 1][vec_idx], x_next_vec + vec_idx);
                } else {
                    x_sh[stage ^ 1][vec_idx] = x_next_vec[vec_idx];
                }
            }
            if constexpr (kUseCpAsync) {
                cp_async_commit_group();
            }
        }

        if constexpr (kProfile) {
            maybe_record_profile<kProfile>(profile, kProfileTile0LoadDone + tile_idx * 2);
        }

        if (row < n) {
            const uint4* x_sh_stage_vec = x_sh[stage];
            const uint4* w_row_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn + tile_start);
            __half2 h2_acc0 = __float2half2_rn(0.0f);
            __half2 h2_acc1 = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 64) {
                const uint4 x_vec0 = x_sh_stage_vec[vec_idx];
                const uint4 w_vec0 = w_row_vec[vec_idx];
                const uint4 x_vec1 = x_sh_stage_vec[vec_idx + 32];
                const uint4 w_vec1 = w_row_vec[vec_idx + 32];
                accumulate_hfma2_uint4(x_vec0, w_vec0, h2_acc0);
                accumulate_hfma2_uint4(x_vec1, w_vec1, h2_acc1);
            }

            const float2 f2_0 = __half22float2(h2_acc0);
            const float2 f2_1 = __half22float2(h2_acc1);
            sum += f2_0.x + f2_0.y + f2_1.x + f2_1.y;
        }

        if constexpr (kProfile) {
            maybe_record_profile<kProfile>(profile, kProfileTile0ComputeDone + tile_idx * 2);
        }
        if constexpr (kUseCpAsync) {
            if (tile_idx + 1 < 6) {
                cp_async_wait_group<0>();
            }
        }
        __syncthreads();
    }

    if (row < n) {
        sum = warp_sum(sum);
        if (lane == 0) {
            float result = sum;
            if (has_bias) {
                result += __half2float(reinterpret_cast<const __half*>(bias)[row]);
            }
            reinterpret_cast<__half*>(out)[row] = __float2half(result);
        }
    }
    if constexpr (kProfile) {
        __syncthreads();
        maybe_record_profile<kProfile>(profile, kProfileKernelDone);
    }
}

template <bool kProfile>
__global__ void nosplit_matvec_k6144_n8k1024_xw_async_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    uint64_t* __restrict__ profile,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __align__(16) uint4 x_sh[2][kAsyncVecsPerTileK6144];
    __shared__ __align__(16) uint4 w_sh[2][kRowsPerBlockAsync * kAsyncVecsPerTileK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * kRowsPerBlockAsync;
    const int row = row_base + warp;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    maybe_record_profile<kProfile>(profile, kProfileKernelStart);
    for (int vec_idx = threadIdx.x; vec_idx < kAsyncVecsPerTileK6144; vec_idx += kRow8Threads) {
        if constexpr (kUseCpAsync) {
            cp_async_cg_16(&x_sh[0][vec_idx], reinterpret_cast<const uint4*>(x_half) + vec_idx);
        } else {
            x_sh[0][vec_idx] = reinterpret_cast<const uint4*>(x_half)[vec_idx];
        }
    }
    if constexpr (kUseCpAsync) {
        cp_async_commit_group();
    }
    prefetch_weight_tile_rows<kUseCpAsync, kRowsPerBlockAsync, kAsyncVecsPerTileK6144>(
        &w_sh[0][0],
        w_half,
        stride_wn,
        row_base,
        0,
        n,
        kRow8Threads);
    if constexpr (kUseCpAsync) {
        cp_async_wait_group<0>();
    }
    __syncthreads();

    float sum = 0.0f;
    #pragma unroll
    for (int tile_idx = 0; tile_idx < 6; ++tile_idx) {
        const int stage = tile_idx & 1;
        const int tile_start = tile_idx * kNosplitAsyncTileK6144;

        if (tile_idx + 1 < 6) {
            const int next_tile_start = tile_start + kNosplitAsyncTileK6144;
            const uint4* x_next_vec = reinterpret_cast<const uint4*>(x_half + next_tile_start);
            for (int vec_idx = threadIdx.x; vec_idx < kAsyncVecsPerTileK6144; vec_idx += kRow8Threads) {
                if constexpr (kUseCpAsync) {
                    cp_async_cg_16(&x_sh[stage ^ 1][vec_idx], x_next_vec + vec_idx);
                } else {
                    x_sh[stage ^ 1][vec_idx] = x_next_vec[vec_idx];
                }
            }
            if constexpr (kUseCpAsync) {
                cp_async_commit_group();
            }
            prefetch_weight_tile_rows<kUseCpAsync, kRowsPerBlockAsync, kAsyncVecsPerTileK6144>(
                &w_sh[stage ^ 1][0],
                w_half,
                stride_wn,
                row_base,
                next_tile_start,
                n,
                kRow8Threads);
        }

        if constexpr (kProfile) {
            maybe_record_profile<kProfile>(profile, kProfileTile0LoadDone + tile_idx * 2);
        }

        if (row < n) {
            const uint4* x_sh_stage_vec = x_sh[stage];
            const uint4* w_sh_row_vec = &w_sh[stage][warp * kAsyncVecsPerTileK6144];
            __half2 h2_acc0 = __float2half2_rn(0.0f);
            __half2 h2_acc1 = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 64) {
                const uint4 x_vec0 = x_sh_stage_vec[vec_idx];
                const uint4 w_vec0 = w_sh_row_vec[vec_idx];
                const uint4 x_vec1 = x_sh_stage_vec[vec_idx + 32];
                const uint4 w_vec1 = w_sh_row_vec[vec_idx + 32];
                accumulate_hfma2_uint4(x_vec0, w_vec0, h2_acc0);
                accumulate_hfma2_uint4(x_vec1, w_vec1, h2_acc1);
            }

            const float2 f2_0 = __half22float2(h2_acc0);
            const float2 f2_1 = __half22float2(h2_acc1);
            sum += f2_0.x + f2_0.y + f2_1.x + f2_1.y;
        }

        if constexpr (kProfile) {
            maybe_record_profile<kProfile>(profile, kProfileTile0ComputeDone + tile_idx * 2);
        }
        if constexpr (kUseCpAsync) {
            if (tile_idx + 1 < 6) {
                cp_async_wait_group<0>();
            }
        }
        __syncthreads();
    }

    if (row < n) {
        sum = warp_sum(sum);
        if (lane == 0) {
            float result = sum;
            if (has_bias) {
                result += __half2float(reinterpret_cast<const __half*>(bias)[row]);
            }
            reinterpret_cast<__half*>(out)[row] = __float2half(result);
        }
    }
    if constexpr (kProfile) {
        __syncthreads();
        maybe_record_profile<kProfile>(profile, kProfileKernelDone);
    }
}

template <bool kProfile>
__global__ void nosplit_matvec_k6144_n8k1024_xwarp_w_async_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    uint64_t* __restrict__ profile,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __align__(16) uint4 x_sh[2][kAsyncVecsPerTileK6144];
    __shared__ __align__(16) uint4 w_sh[2][kRowsPerBlockAsync][kAsyncVecsPerTileK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * kRowsPerBlockAsync;
    const int row = row_base + warp;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    maybe_record_profile<kProfile>(profile, kProfileKernelStart);

    prefetch_x_tile_vecs<kUseCpAsync, kAsyncVecsPerTileK6144, kRow8Threads>(
        &x_sh[0][0],
        reinterpret_cast<const uint4*>(x_half));
    if (row < n) {
        const uint4* w_row_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn);
        for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
            if constexpr (kUseCpAsync) {
                cp_async_cg_16(&w_sh[0][warp][vec_idx], w_row_vec + vec_idx);
            } else {
                w_sh[0][warp][vec_idx] = w_row_vec[vec_idx];
            }
        }
    } else {
        for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
            w_sh[0][warp][vec_idx] = zero_uint4();
        }
    }
    if constexpr (kUseCpAsync) {
        cp_async_commit_group();
        cp_async_wait_group<0>();
    }
    __syncthreads();

    float sum = 0.0f;
    #pragma unroll
    for (int tile_idx = 0; tile_idx < 6; ++tile_idx) {
        const int stage = tile_idx & 1;
        const int tile_start = tile_idx * kNosplitAsyncTileK6144;

        if (tile_idx + 1 < 6) {
            const int next_tile_start = tile_start + kNosplitAsyncTileK6144;
            const uint4* x_next_vec = reinterpret_cast<const uint4*>(x_half + next_tile_start);
            prefetch_x_tile_vecs<kUseCpAsync, kAsyncVecsPerTileK6144, kRow8Threads>(
                &x_sh[stage ^ 1][0],
                x_next_vec);

            if (row < n) {
                const uint4* w_row_next_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn + next_tile_start);
                for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                    if constexpr (kUseCpAsync) {
                        cp_async_cg_16(&w_sh[stage ^ 1][warp][vec_idx], w_row_next_vec + vec_idx);
                    } else {
                        w_sh[stage ^ 1][warp][vec_idx] = w_row_next_vec[vec_idx];
                    }
                }
            } else {
                for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                    w_sh[stage ^ 1][warp][vec_idx] = zero_uint4();
                }
            }
            if constexpr (kUseCpAsync) {
                cp_async_commit_group();
            }
        }

        if constexpr (kProfile) {
            maybe_record_profile<kProfile>(profile, kProfileTile0LoadDone + tile_idx * 2);
        }

        if (row < n) {
            const uint4* x_sh_stage_vec = x_sh[stage];
            const uint4* w_sh_row_vec = &w_sh[stage][warp][0];
            __half2 h2_acc0 = __float2half2_rn(0.0f);
            __half2 h2_acc1 = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 64) {
                const uint4 x_vec0 = x_sh_stage_vec[vec_idx];
                const uint4 w_vec0 = w_sh_row_vec[vec_idx];
                const uint4 x_vec1 = x_sh_stage_vec[vec_idx + 32];
                const uint4 w_vec1 = w_sh_row_vec[vec_idx + 32];
                accumulate_hfma2_uint4(x_vec0, w_vec0, h2_acc0);
                accumulate_hfma2_uint4(x_vec1, w_vec1, h2_acc1);
            }

            const float2 f2_0 = __half22float2(h2_acc0);
            const float2 f2_1 = __half22float2(h2_acc1);
            sum += f2_0.x + f2_0.y + f2_1.x + f2_1.y;
        }

        if constexpr (kProfile) {
            maybe_record_profile<kProfile>(profile, kProfileTile0ComputeDone + tile_idx * 2);
        }
        if constexpr (kUseCpAsync) {
            if (tile_idx + 1 < 6) {
                cp_async_wait_group<0>();
            }
        }
        __syncthreads();
    }

    if (row < n) {
        sum = warp_sum(sum);
        if (lane == 0) {
            float result = sum;
            if (has_bias) {
                result += __half2float(reinterpret_cast<const __half*>(bias)[row]);
            }
            reinterpret_cast<__half*>(out)[row] = __float2half(result);
        }
    }
    if constexpr (kProfile) {
        __syncthreads();
        maybe_record_profile<kProfile>(profile, kProfileKernelDone);
    }
}

__global__ void nosplit_weight_only_k6144_n8k1024_xwarp_w_async_fwd_kernel(
    const at::Half* __restrict__ weight,
    int32_t* __restrict__ out,
    int64_t stride_wn,
    int n) {
    __shared__ __align__(16) uint4 w_sh[2][kRowsPerBlockAsync][kAsyncVecsPerTileK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * kRowsPerBlockAsync;
    const int row = row_base + warp;
    const __half* w_half = reinterpret_cast<const __half*>(weight);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    if (row < n) {
        const uint4* w_row_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn);
        for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
            if constexpr (kUseCpAsync) {
                cp_async_cg_16(&w_sh[0][warp][vec_idx], w_row_vec + vec_idx);
            } else {
                w_sh[0][warp][vec_idx] = w_row_vec[vec_idx];
            }
        }
    } else {
        for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
            w_sh[0][warp][vec_idx] = zero_uint4();
        }
    }
    if constexpr (kUseCpAsync) {
        cp_async_commit_group();
        cp_async_wait_group<0>();
    }
    __syncthreads();

    unsigned int sum = 0;
    #pragma unroll
    for (int tile_idx = 0; tile_idx < 6; ++tile_idx) {
        const int stage = tile_idx & 1;
        const int tile_start = tile_idx * kNosplitAsyncTileK6144;

        if (tile_idx + 1 < 6) {
            const int next_tile_start = tile_start + kNosplitAsyncTileK6144;
            if (row < n) {
                const uint4* w_row_next_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn + next_tile_start);
                for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                    if constexpr (kUseCpAsync) {
                        cp_async_cg_16(&w_sh[stage ^ 1][warp][vec_idx], w_row_next_vec + vec_idx);
                    } else {
                        w_sh[stage ^ 1][warp][vec_idx] = w_row_next_vec[vec_idx];
                    }
                }
            } else {
                for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                    w_sh[stage ^ 1][warp][vec_idx] = zero_uint4();
                }
            }
            if constexpr (kUseCpAsync) {
                cp_async_commit_group();
            }
        }

        if (row < n) {
            const uint4* w_sh_row_vec = &w_sh[stage][warp][0];
            #pragma unroll
            for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 64) {
                const uint4 w_vec0 = w_sh_row_vec[vec_idx];
                const uint4 w_vec1 = w_sh_row_vec[vec_idx + 32];
                sum += w_vec0.x + w_vec0.y + w_vec0.z + w_vec0.w;
                sum += w_vec1.x + w_vec1.y + w_vec1.z + w_vec1.w;
            }
        }

        if constexpr (kUseCpAsync) {
            if (tile_idx + 1 < 6) {
                cp_async_wait_group<0>();
            }
        }
        __syncthreads();
    }

    if (row < n) {
        sum = warp_sum_u32(sum);
        if (lane == 0) {
            out[row] = static_cast<int32_t>(sum);
        }
    }
}

template <bool kProfile>
__global__ void nosplit_matvec_k6144_n8k1024_xwarp_w_async_sms2_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    uint64_t* __restrict__ profile,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __align__(16) uint4 x_sh[2][kAsyncVecsPerTileK6144];
    __shared__ __align__(16) uint4 w_sh[2][kRowsPerBlockAsync][kAsyncVecsPerTileK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    maybe_record_profile<kProfile>(profile, kProfileKernelStart);

    const int row_stride = gridDim.x * kRowsPerBlockAsync;
    for (int row_base = blockIdx.x * kRowsPerBlockAsync; row_base < n; row_base += row_stride) {
        const int row = row_base + warp;

        prefetch_x_tile_vecs<kUseCpAsync, kAsyncVecsPerTileK6144, kRow8Threads>(
            &x_sh[0][0],
            reinterpret_cast<const uint4*>(x_half));
        if (row < n) {
            const uint4* w_row_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn);
            for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                if constexpr (kUseCpAsync) {
                    cp_async_cg_16(&w_sh[0][warp][vec_idx], w_row_vec + vec_idx);
                } else {
                    w_sh[0][warp][vec_idx] = w_row_vec[vec_idx];
                }
            }
        } else {
            for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                w_sh[0][warp][vec_idx] = zero_uint4();
            }
        }
        if constexpr (kUseCpAsync) {
            cp_async_commit_group();
            cp_async_wait_group<0>();
        }
        __syncthreads();

        float sum = 0.0f;
        #pragma unroll
        for (int tile_idx = 0; tile_idx < 6; ++tile_idx) {
            const int stage = tile_idx & 1;
            const int tile_start = tile_idx * kNosplitAsyncTileK6144;

            if (tile_idx + 1 < 6) {
                const int next_tile_start = tile_start + kNosplitAsyncTileK6144;
                const uint4* x_next_vec = reinterpret_cast<const uint4*>(x_half + next_tile_start);
                prefetch_x_tile_vecs<kUseCpAsync, kAsyncVecsPerTileK6144, kRow8Threads>(
                    &x_sh[stage ^ 1][0],
                    x_next_vec);

                if (row < n) {
                    const uint4* w_row_next_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn + next_tile_start);
                    for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                        if constexpr (kUseCpAsync) {
                            cp_async_cg_16(&w_sh[stage ^ 1][warp][vec_idx], w_row_next_vec + vec_idx);
                        } else {
                            w_sh[stage ^ 1][warp][vec_idx] = w_row_next_vec[vec_idx];
                        }
                    }
                } else {
                    for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                        w_sh[stage ^ 1][warp][vec_idx] = zero_uint4();
                    }
                }
                if constexpr (kUseCpAsync) {
                    cp_async_commit_group();
                }
            }

            if constexpr (kProfile) {
                maybe_record_profile<kProfile>(profile, kProfileTile0LoadDone + tile_idx * 2);
            }

            if (row < n) {
                const uint4* x_sh_stage_vec = x_sh[stage];
                const uint4* w_sh_row_vec = &w_sh[stage][warp][0];
                __half2 h2_acc0 = __float2half2_rn(0.0f);
                __half2 h2_acc1 = __float2half2_rn(0.0f);

                #pragma unroll
                for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 64) {
                    const uint4 x_vec0 = x_sh_stage_vec[vec_idx];
                    const uint4 w_vec0 = w_sh_row_vec[vec_idx];
                    const uint4 x_vec1 = x_sh_stage_vec[vec_idx + 32];
                    const uint4 w_vec1 = w_sh_row_vec[vec_idx + 32];
                    accumulate_hfma2_uint4(x_vec0, w_vec0, h2_acc0);
                    accumulate_hfma2_uint4(x_vec1, w_vec1, h2_acc1);
                }

                const float2 f2_0 = __half22float2(h2_acc0);
                const float2 f2_1 = __half22float2(h2_acc1);
                sum += f2_0.x + f2_0.y + f2_1.x + f2_1.y;
            }

            if constexpr (kProfile) {
                maybe_record_profile<kProfile>(profile, kProfileTile0ComputeDone + tile_idx * 2);
            }
            if constexpr (kUseCpAsync) {
                if (tile_idx + 1 < 6) {
                    cp_async_wait_group<0>();
                }
            }
            __syncthreads();
        }

        if (row < n) {
            sum = warp_sum(sum);
            if (lane == 0) {
                float result = sum;
                if (has_bias) {
                    result += __half2float(reinterpret_cast<const __half*>(bias)[row]);
                }
                reinterpret_cast<__half*>(out)[row] = __float2half(result);
            }
        }
        __syncthreads();
    }

    if constexpr (kProfile) {
        maybe_record_profile<kProfile>(profile, kProfileKernelDone);
    }
}

template <bool kUseCpAsync>
__device__ inline void run_k6144_xwarp_async_row_batch(
    const __half* x_half,
    const __half* w_half,
    const __half* bias_half,
    __half* out_half,
    uint4 x_sh[2][kAsyncVecsPerTileK6144],
    uint4 w_sh[2][kRowsPerBlockAsync][kAsyncVecsPerTileK6144],
    int warp,
    int lane,
    int row_base,
    int active_rows,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    const int row = row_base + warp;
    const bool row_valid = (warp < active_rows) && (row < n);

    prefetch_x_tile_vecs<kUseCpAsync, kAsyncVecsPerTileK6144, kRow8Threads>(
        &x_sh[0][0],
        reinterpret_cast<const uint4*>(x_half));
    if (row_valid) {
        const uint4* w_row_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn);
        for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
            if constexpr (kUseCpAsync) {
                cp_async_cg_16(&w_sh[0][warp][vec_idx], w_row_vec + vec_idx);
            } else {
                w_sh[0][warp][vec_idx] = w_row_vec[vec_idx];
            }
        }
    } else {
        for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
            w_sh[0][warp][vec_idx] = zero_uint4();
        }
    }
    if constexpr (kUseCpAsync) {
        cp_async_commit_group();
        cp_async_wait_group<0>();
    }
    __syncthreads();

    float sum = 0.0f;
    #pragma unroll
    for (int tile_idx = 0; tile_idx < 6; ++tile_idx) {
        const int stage = tile_idx & 1;
        const int tile_start = tile_idx * kNosplitAsyncTileK6144;

        if (tile_idx + 1 < 6) {
            const int next_tile_start = tile_start + kNosplitAsyncTileK6144;
            const uint4* x_next_vec = reinterpret_cast<const uint4*>(x_half + next_tile_start);
            prefetch_x_tile_vecs<kUseCpAsync, kAsyncVecsPerTileK6144, kRow8Threads>(
                &x_sh[stage ^ 1][0],
                x_next_vec);

            if (row_valid) {
                const uint4* w_row_next_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn + next_tile_start);
                for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                    if constexpr (kUseCpAsync) {
                        cp_async_cg_16(&w_sh[stage ^ 1][warp][vec_idx], w_row_next_vec + vec_idx);
                    } else {
                        w_sh[stage ^ 1][warp][vec_idx] = w_row_next_vec[vec_idx];
                    }
                }
            } else {
                for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                    w_sh[stage ^ 1][warp][vec_idx] = zero_uint4();
                }
            }
            if constexpr (kUseCpAsync) {
                cp_async_commit_group();
            }
        }

        if (row_valid) {
            const uint4* x_sh_stage_vec = x_sh[stage];
            const uint4* w_sh_row_vec = &w_sh[stage][warp][0];
            __half2 h2_acc0 = __float2half2_rn(0.0f);
            __half2 h2_acc1 = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 64) {
                const uint4 x_vec0 = x_sh_stage_vec[vec_idx];
                const uint4 w_vec0 = w_sh_row_vec[vec_idx];
                const uint4 x_vec1 = x_sh_stage_vec[vec_idx + 32];
                const uint4 w_vec1 = w_sh_row_vec[vec_idx + 32];
                accumulate_hfma2_uint4(x_vec0, w_vec0, h2_acc0);
                accumulate_hfma2_uint4(x_vec1, w_vec1, h2_acc1);
            }

            const float2 f2_0 = __half22float2(h2_acc0);
            const float2 f2_1 = __half22float2(h2_acc1);
            sum += f2_0.x + f2_0.y + f2_1.x + f2_1.y;
        }

        if constexpr (kUseCpAsync) {
            if (tile_idx + 1 < 6) {
                cp_async_wait_group<0>();
            }
        }
        __syncthreads();
    }

    if (row_valid) {
        sum = warp_sum(sum);
        if (lane == 0) {
            float result = sum;
            if (has_bias) {
                result += __half2float(bias_half[row]);
            }
            out_half[row] = __float2half(result);
        }
    }
    __syncthreads();
}

template <bool kUseCpAsync>
__device__ inline void run_k6144_xwarp_async_dualcopy_block(
    const __half* x_half,
    const __half* weight_a_half,
    const __half* weight_b_half,
    const __half* bias_half,
    __half* out_half,
    const int32_t* route_table,
    int route_mode,
    int smid_threshold,
    uint4 x_sh[2][kAsyncVecsPerTileK6144],
    uint4 w_sh[2][kRowsPerBlockAsync][kAsyncVecsPerTileK6144],
    int warp,
    int lane,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ int selected_copy;
    if (threadIdx.x == 0) {
        const int smid = read_smid_int();
        selected_copy = pick_dualcopy_route(smid, route_table, route_mode, smid_threshold);
    }
    __syncthreads();

    const __half* w_half = selected_copy ? weight_b_half : weight_a_half;
    const int row_base = blockIdx.x * kRowsPerBlockAsync;
    const int active_rows = max(0, min(kRowsPerBlockAsync, n - row_base));
    if (active_rows > 0) {
        run_k6144_xwarp_async_row_batch<kUseCpAsync>(
            x_half,
            w_half,
            bias_half,
            out_half,
            x_sh,
            w_sh,
            warp,
            lane,
            row_base,
            active_rows,
            stride_wn,
            n,
            has_bias);
    }
}

__global__ void nosplit_matvec_k6144_n8k1024_xwarp_w_async_dualcopy_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight_a,
    const at::Half* __restrict__ weight_b,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    const int32_t* __restrict__ route_table,
    int route_mode,
    int smid_threshold,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __align__(16) uint4 x_sh[2][kAsyncVecsPerTileK6144];
    __shared__ __align__(16) uint4 w_sh[2][kRowsPerBlockAsync][kAsyncVecsPerTileK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* weight_a_half = reinterpret_cast<const __half*>(weight_a);
    const __half* weight_b_half = reinterpret_cast<const __half*>(weight_b);
    const __half* bias_half = reinterpret_cast<const __half*>(bias);
    __half* out_half = reinterpret_cast<__half*>(out);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    run_k6144_xwarp_async_dualcopy_block<kUseCpAsync>(
        x_half,
        weight_a_half,
        weight_b_half,
        bias_half,
        out_half,
        route_table,
        route_mode,
        smid_threshold,
        x_sh,
        w_sh,
        warp,
        lane,
        stride_wn,
        n,
        has_bias);
}

__global__ void nosplit_matvec_k6144_n8k1024_xwarp_w_async_sms2_balanced_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __align__(16) uint4 x_sh[2][kAsyncVecsPerTileK6144];
    __shared__ __align__(16) uint4 w_sh[2][kRowsPerBlockAsync][kAsyncVecsPerTileK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);
    const __half* bias_half = reinterpret_cast<const __half*>(bias);
    __half* out_half = reinterpret_cast<__half*>(out);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    const int block_count = gridDim.x;
    const int base_rows = n / block_count;
    const int extra_rows = n % block_count;
    const int row_begin = blockIdx.x * base_rows + min(static_cast<int>(blockIdx.x), extra_rows);
    const int row_count = base_rows + (static_cast<int>(blockIdx.x) < extra_rows ? 1 : 0);

    const int first_rows = min(row_count, kRowsPerBlockAsync);
    const int second_rows = max(0, row_count - kRowsPerBlockAsync);

    if (first_rows > 0) {
        run_k6144_xwarp_async_row_batch<kUseCpAsync>(
            x_half, w_half, bias_half, out_half, x_sh, w_sh, warp, lane,
            row_begin, first_rows, stride_wn, n, has_bias);
    }
    if (second_rows > 0) {
        run_k6144_xwarp_async_row_batch<kUseCpAsync>(
            x_half, w_half, bias_half, out_half, x_sh, w_sh, warp, lane,
            row_begin + kRowsPerBlockAsync, second_rows, stride_wn, n, has_bias);
    }
}

__global__ void nosplit_matvec_k6144_n8k1024_xwarp_w_async_sms2_balanced_cta_trace_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    int64_t* __restrict__ trace,
    int32_t* __restrict__ smid_out,
    int32_t* __restrict__ row_base_out,
    int max_iters,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __align__(16) uint4 x_sh[2][kAsyncVecsPerTileK6144];
    __shared__ __align__(16) uint4 w_sh[2][kRowsPerBlockAsync][kAsyncVecsPerTileK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);
    const __half* bias_half = reinterpret_cast<const __half*>(bias);
    __half* out_half = reinterpret_cast<__half*>(out);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    maybe_record_block_smid(smid_out);

    const int block_count = gridDim.x;
    const int base_rows = n / block_count;
    const int extra_rows = n % block_count;
    const int row_begin = blockIdx.x * base_rows + min(static_cast<int>(blockIdx.x), extra_rows);
    const int row_count = base_rows + (static_cast<int>(blockIdx.x) < extra_rows ? 1 : 0);

    const int first_rows = min(row_count, kRowsPerBlockAsync);
    const int second_rows = max(0, row_count - kRowsPerBlockAsync);

    if (first_rows > 0) {
        maybe_record_block_iter_trace(trace, row_base_out, max_iters, 0, row_begin, true);
        run_k6144_xwarp_async_row_batch<kUseCpAsync>(
            x_half, w_half, bias_half, out_half, x_sh, w_sh, warp, lane,
            row_begin, first_rows, stride_wn, n, has_bias);
        maybe_record_block_iter_trace(trace, row_base_out, max_iters, 0, row_begin, false);
    }
    if (second_rows > 0) {
        maybe_record_block_iter_trace(trace, row_base_out, max_iters, 1, row_begin + kRowsPerBlockAsync, true);
        run_k6144_xwarp_async_row_batch<kUseCpAsync>(
            x_half, w_half, bias_half, out_half, x_sh, w_sh, warp, lane,
            row_begin + kRowsPerBlockAsync, second_rows, stride_wn, n, has_bias);
        maybe_record_block_iter_trace(trace, row_base_out, max_iters, 1, row_begin + kRowsPerBlockAsync, false);
    }
}

__global__ void nosplit_matvec_k6144_n8k1024_xwarp_w_async_sms2_cta_trace_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    int64_t* __restrict__ trace,
    int32_t* __restrict__ smid_out,
    int32_t* __restrict__ row_base_out,
    int max_iters,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __align__(16) uint4 x_sh[2][kAsyncVecsPerTileK6144];
    __shared__ __align__(16) uint4 w_sh[2][kRowsPerBlockAsync][kAsyncVecsPerTileK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    maybe_record_block_smid(smid_out);

    const int row_stride = gridDim.x * kRowsPerBlockAsync;
    int iter_idx = 0;
    for (int row_base = blockIdx.x * kRowsPerBlockAsync; row_base < n; row_base += row_stride, ++iter_idx) {
        const int row = row_base + warp;
        maybe_record_block_iter_trace(trace, row_base_out, max_iters, iter_idx, row_base, true);

        prefetch_x_tile_vecs<kUseCpAsync, kAsyncVecsPerTileK6144, kRow8Threads>(
            &x_sh[0][0],
            reinterpret_cast<const uint4*>(x_half));
        if (row < n) {
            const uint4* w_row_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn);
            for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                if constexpr (kUseCpAsync) {
                    cp_async_cg_16(&w_sh[0][warp][vec_idx], w_row_vec + vec_idx);
                } else {
                    w_sh[0][warp][vec_idx] = w_row_vec[vec_idx];
                }
            }
        } else {
            for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                w_sh[0][warp][vec_idx] = zero_uint4();
            }
        }
        if constexpr (kUseCpAsync) {
            cp_async_commit_group();
            cp_async_wait_group<0>();
        }
        __syncthreads();

        float sum = 0.0f;
        #pragma unroll
        for (int tile_idx = 0; tile_idx < 6; ++tile_idx) {
            const int stage = tile_idx & 1;
            const int tile_start = tile_idx * kNosplitAsyncTileK6144;

            if (tile_idx + 1 < 6) {
                const int next_tile_start = tile_start + kNosplitAsyncTileK6144;
                const uint4* x_next_vec = reinterpret_cast<const uint4*>(x_half + next_tile_start);
                prefetch_x_tile_vecs<kUseCpAsync, kAsyncVecsPerTileK6144, kRow8Threads>(
                    &x_sh[stage ^ 1][0],
                    x_next_vec);

                if (row < n) {
                    const uint4* w_row_next_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn + next_tile_start);
                    for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                        if constexpr (kUseCpAsync) {
                            cp_async_cg_16(&w_sh[stage ^ 1][warp][vec_idx], w_row_next_vec + vec_idx);
                        } else {
                            w_sh[stage ^ 1][warp][vec_idx] = w_row_next_vec[vec_idx];
                        }
                    }
                } else {
                    for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                        w_sh[stage ^ 1][warp][vec_idx] = zero_uint4();
                    }
                }
                if constexpr (kUseCpAsync) {
                    cp_async_commit_group();
                }
            }

            if (row < n) {
                const uint4* x_sh_stage_vec = x_sh[stage];
                const uint4* w_sh_row_vec = &w_sh[stage][warp][0];
                __half2 h2_acc0 = __float2half2_rn(0.0f);
                __half2 h2_acc1 = __float2half2_rn(0.0f);

                #pragma unroll
                for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 64) {
                    const uint4 x_vec0 = x_sh_stage_vec[vec_idx];
                    const uint4 w_vec0 = w_sh_row_vec[vec_idx];
                    const uint4 x_vec1 = x_sh_stage_vec[vec_idx + 32];
                    const uint4 w_vec1 = w_sh_row_vec[vec_idx + 32];
                    accumulate_hfma2_uint4(x_vec0, w_vec0, h2_acc0);
                    accumulate_hfma2_uint4(x_vec1, w_vec1, h2_acc1);
                }

                const float2 f2_0 = __half22float2(h2_acc0);
                const float2 f2_1 = __half22float2(h2_acc1);
                sum += f2_0.x + f2_0.y + f2_1.x + f2_1.y;
            }

            if constexpr (kUseCpAsync) {
                if (tile_idx + 1 < 6) {
                    cp_async_wait_group<0>();
                }
            }
            __syncthreads();
        }

        if (row < n) {
            sum = warp_sum(sum);
            if (lane == 0) {
                float result = sum;
                if (has_bias) {
                    result += __half2float(reinterpret_cast<const __half*>(bias)[row]);
                }
                reinterpret_cast<__half*>(out)[row] = __float2half(result);
            }
        }
        __syncthreads();
        maybe_record_block_iter_trace(trace, row_base_out, max_iters, iter_idx, row_base, false);
    }
}

__global__ void nosplit_matvec_k6144_n8k1024_xwarp_w_async_trace_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    int64_t* __restrict__ trace,
    int32_t* __restrict__ smid_out,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __align__(16) uint4 x_sh[2][kAsyncVecsPerTileK6144];
    __shared__ __align__(16) uint4 w_sh[2][kRowsPerBlockAsync][kAsyncVecsPerTileK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * kRowsPerBlockAsync;
    const int row = row_base + warp;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    maybe_record_warp_smid(smid_out);
    maybe_record_warp_detail(trace, 0);
    prefetch_x_tile_vecs<kUseCpAsync, kAsyncVecsPerTileK6144, kRow8Threads>(
        &x_sh[0][0],
        reinterpret_cast<const uint4*>(x_half));
    maybe_record_warp_detail(trace, 1);

    maybe_record_warp_detail(trace, 2);
    if (row < n) {
        const uint4* w_row_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn);
        for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
            if constexpr (kUseCpAsync) {
                cp_async_cg_16(&w_sh[0][warp][vec_idx], w_row_vec + vec_idx);
            } else {
                w_sh[0][warp][vec_idx] = w_row_vec[vec_idx];
            }
        }
    } else {
        for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
            w_sh[0][warp][vec_idx] = zero_uint4();
        }
    }
    maybe_record_warp_detail(trace, 3);

    maybe_record_warp_detail(trace, 4);
    if constexpr (kUseCpAsync) {
        cp_async_commit_group();
        cp_async_wait_group<0>();
    }
    maybe_record_warp_detail(trace, 5);

    maybe_record_warp_detail(trace, 6);
    __syncthreads();
    maybe_record_warp_detail(trace, 7);

    float sum = 0.0f;
    #pragma unroll
    for (int tile_idx = 0; tile_idx < 6; ++tile_idx) {
        const int stage = tile_idx & 1;
        const int tile_start = tile_idx * kNosplitAsyncTileK6144;
        const int slot_base = kXWarpDetailPrologueSlots + tile_idx * kXWarpDetailSlotsPerTile;

        if (tile_idx + 1 < 6) {
            const int next_tile_start = tile_start + kNosplitAsyncTileK6144;
            const uint4* x_next_vec = reinterpret_cast<const uint4*>(x_half + next_tile_start);

            maybe_record_warp_detail(trace, slot_base + 0);
            prefetch_x_tile_vecs<kUseCpAsync, kAsyncVecsPerTileK6144, kRow8Threads>(
                &x_sh[stage ^ 1][0],
                x_next_vec);
            maybe_record_warp_detail(trace, slot_base + 1);

            maybe_record_warp_detail(trace, slot_base + 2);
            if (row < n) {
                const uint4* w_row_next_vec = reinterpret_cast<const uint4*>(w_half + row * stride_wn + next_tile_start);
                for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                    if constexpr (kUseCpAsync) {
                        cp_async_cg_16(&w_sh[stage ^ 1][warp][vec_idx], w_row_next_vec + vec_idx);
                    } else {
                        w_sh[stage ^ 1][warp][vec_idx] = w_row_next_vec[vec_idx];
                    }
                }
            } else {
                for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 32) {
                    w_sh[stage ^ 1][warp][vec_idx] = zero_uint4();
                }
            }
            maybe_record_warp_detail(trace, slot_base + 3);

            if constexpr (kUseCpAsync) {
                cp_async_commit_group();
            }
        }

        maybe_record_warp_detail(trace, slot_base + 4);
        if (row < n) {
            const uint4* x_sh_stage_vec = x_sh[stage];
            const uint4* w_sh_row_vec = &w_sh[stage][warp][0];
            __half2 h2_acc0 = __float2half2_rn(0.0f);
            __half2 h2_acc1 = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 64) {
                const uint4 x_vec0 = x_sh_stage_vec[vec_idx];
                const uint4 w_vec0 = w_sh_row_vec[vec_idx];
                const uint4 x_vec1 = x_sh_stage_vec[vec_idx + 32];
                const uint4 w_vec1 = w_sh_row_vec[vec_idx + 32];
                accumulate_hfma2_uint4(x_vec0, w_vec0, h2_acc0);
                accumulate_hfma2_uint4(x_vec1, w_vec1, h2_acc1);
            }

            const float2 f2_0 = __half22float2(h2_acc0);
            const float2 f2_1 = __half22float2(h2_acc1);
            sum += f2_0.x + f2_0.y + f2_1.x + f2_1.y;
        }
        maybe_record_warp_detail(trace, slot_base + 5);

        maybe_record_warp_detail(trace, slot_base + 6);
        if constexpr (kUseCpAsync) {
            if (tile_idx + 1 < 6) {
                cp_async_wait_group<0>();
            }
        }
        maybe_record_warp_detail(trace, slot_base + 7);

        maybe_record_warp_detail(trace, slot_base + 8);
        __syncthreads();
        maybe_record_warp_detail(trace, slot_base + 9);
    }

    const int epilogue_slot = kXWarpDetailPrologueSlots + kXWarpDetailTileCount * kXWarpDetailSlotsPerTile;
    maybe_record_warp_detail(trace, epilogue_slot + 0);
    if (row < n) {
        sum = warp_sum(sum);
        if (lane == 0) {
            float result = sum;
            if (has_bias) {
                result += __half2float(reinterpret_cast<const __half*>(bias)[row]);
            }
            reinterpret_cast<__half*>(out)[row] = __float2half(result);
        }
    }
    maybe_record_warp_detail(trace, epilogue_slot + 1);
}

__global__ void nosplit_xwarp_wait_sync_calibration_kernel(
    int64_t* __restrict__ trace) {
    maybe_record_warp_detail(trace, 4);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    cp_async_wait_group<0>();
#endif
    maybe_record_warp_detail(trace, 5);

    maybe_record_warp_detail(trace, 6);
    __syncthreads();
    maybe_record_warp_detail(trace, 7);

    #pragma unroll
    for (int tile_idx = 0; tile_idx < 6; ++tile_idx) {
        const int slot_base = kXWarpDetailPrologueSlots + tile_idx * kXWarpDetailSlotsPerTile;
        maybe_record_warp_detail(trace, slot_base + 6);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        cp_async_wait_group<0>();
#endif
        maybe_record_warp_detail(trace, slot_base + 7);

        maybe_record_warp_detail(trace, slot_base + 8);
        __syncthreads();
        maybe_record_warp_detail(trace, slot_base + 9);
    }
}

template <int kTileK, int kStages, bool kProfile>
__global__ void nosplit_matvec_k6144_n8_xw_async_multi_stage_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    uint64_t* __restrict__ profile,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    static_assert(kStages >= 2, "multi-stage kernel requires at least 2 stages");
    constexpr int kTotalK = 6144;
    static_assert(kTotalK % kTileK == 0, "kTileK must evenly divide k=6144");

    constexpr int kVecsPerTile = kTileK / kHalfPerVec128;
    constexpr int kTileCount = kTotalK / kTileK;
    constexpr int kThreadsPerBlock = kRow8Threads;

    __shared__ __align__(16) uint4 x_sh[kStages][kVecsPerTile];
    __shared__ __align__(16) uint4 w_sh[kStages][kRowsPerBlockAsync * kVecsPerTile];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = blockIdx.x * kRowsPerBlockAsync;
    const int row = row_base + warp;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    maybe_record_profile<kProfile>(profile, kProfileKernelStart);

    #pragma unroll
    for (int prefetch_tile = 0; prefetch_tile < kStages; ++prefetch_tile) {
        if (prefetch_tile < kTileCount) {
            const int tile_start = prefetch_tile * kTileK;
            prefetch_x_tile_vecs<kUseCpAsync, kVecsPerTile, kThreadsPerBlock>(
                &x_sh[prefetch_tile][0],
                reinterpret_cast<const uint4*>(x_half + tile_start));
            prefetch_weight_tile_rows_no_commit<kUseCpAsync, kRowsPerBlockAsync, kVecsPerTile>(
                &w_sh[prefetch_tile][0],
                w_half,
                stride_wn,
                row_base,
                tile_start,
                n,
                kThreadsPerBlock);
            if constexpr (kUseCpAsync) {
                cp_async_commit_group();
            }
        }
    }
    if constexpr (kUseCpAsync) {
        cp_async_wait_group<kStages - 2>();
    }
    __syncthreads();

    float sum = 0.0f;
    #pragma unroll
    for (int tile_idx = 0; tile_idx < kTileCount; ++tile_idx) {
        const int stage = tile_idx % kStages;

        if constexpr (kProfile) {
            maybe_record_profile<kProfile>(profile, kProfileTile0LoadDone + (tile_idx >> 1) * 2);
        }

        if (row < n) {
            const uint4* x_sh_stage_vec = x_sh[stage];
            const uint4* w_sh_row_vec = &w_sh[stage][warp * kVecsPerTile];
            __half2 h2_acc0 = __float2half2_rn(0.0f);
            __half2 h2_acc1 = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane; vec_idx < kVecsPerTile; vec_idx += 64) {
                const uint4 x_vec0 = x_sh_stage_vec[vec_idx];
                const uint4 w_vec0 = w_sh_row_vec[vec_idx];
                const uint4 x_vec1 = x_sh_stage_vec[vec_idx + 32];
                const uint4 w_vec1 = w_sh_row_vec[vec_idx + 32];
                accumulate_hfma2_uint4(x_vec0, w_vec0, h2_acc0);
                accumulate_hfma2_uint4(x_vec1, w_vec1, h2_acc1);
            }

            const float2 f2_0 = __half22float2(h2_acc0);
            const float2 f2_1 = __half22float2(h2_acc1);
            sum += f2_0.x + f2_0.y + f2_1.x + f2_1.y;
        }

        if constexpr (kProfile) {
            maybe_record_profile<kProfile>(profile, kProfileTile0ComputeDone + (tile_idx >> 1) * 2);
        }

        // All warps must finish consuming the current stage before it is recycled.
        __syncthreads();

        const int next_tile = tile_idx + kStages;
        if (next_tile < kTileCount) {
            const int next_tile_start = next_tile * kTileK;
            prefetch_x_tile_vecs<kUseCpAsync, kVecsPerTile, kThreadsPerBlock>(
                &x_sh[stage][0],
                reinterpret_cast<const uint4*>(x_half + next_tile_start));
            prefetch_weight_tile_rows_no_commit<kUseCpAsync, kRowsPerBlockAsync, kVecsPerTile>(
                &w_sh[stage][0],
                w_half,
                stride_wn,
                row_base,
                next_tile_start,
                n,
                kThreadsPerBlock);
            if constexpr (kUseCpAsync) {
                cp_async_commit_group();
            }
        }

        if constexpr (kUseCpAsync) {
            if (tile_idx + 1 < kTileCount) {
                const int future_tiles = kTileCount - (tile_idx + 1);
                if constexpr (kStages == 3) {
                    if (future_tiles > 1) {
                        cp_async_wait_group<1>();
                    } else {
                        cp_async_wait_group<0>();
                    }
                } else {
                    cp_async_wait_group<kStages - 2>();
                }
            }
        }
        __syncthreads();
    }

    if (row < n) {
        sum = warp_sum(sum);
        if (lane == 0) {
            float result = sum;
            if (has_bias) {
                result += __half2float(reinterpret_cast<const __half*>(bias)[row]);
            }
            reinterpret_cast<__half*>(out)[row] = __float2half(result);
        }
    }
    if constexpr (kProfile) {
        __syncthreads();
        maybe_record_profile<kProfile>(profile, kProfileKernelDone);
    }
}

template <bool kProfile>
__global__ void nosplit_matvec_k6144_persistent_xfull_w2_async_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    uint64_t* __restrict__ profile,
    int64_t stride_wn,
    int n,
    bool has_bias) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
    __shared__ __align__(16) uint4 x_sh[kFullVecsK6144];
    __shared__ __align__(16) uint4 w_sh[2][kPersistentRowsPerBatchK6144 * kFullVecsK6144];
    __shared__ float warp_partials[kPersistentRowsPerBatchK6144][2];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_slot = warp >> 1;
    const int warp_in_row = warp & 1;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    maybe_record_profile<kProfile>(profile, kProfileKernelStart);

    prefetch_x_tile_vecs<kUseCpAsync, kFullVecsK6144, kRow4Threads>(
        &x_sh[0],
        reinterpret_cast<const uint4*>(x_half));
    if constexpr (kUseCpAsync) {
        cp_async_commit_group();
        cp_async_wait_group<0>();
    }
    __syncthreads();

    const int row_stride = gridDim.x * kPersistentRowsPerBatchK6144;
    int row_base = blockIdx.x * kPersistentRowsPerBatchK6144;
    if (row_base < n) {
        prefetch_weight_tile_rows_no_commit<kUseCpAsync, kPersistentRowsPerBatchK6144, kFullVecsK6144>(
            &w_sh[0][0],
            w_half,
            stride_wn,
            row_base,
            0,
            n,
            kRow4Threads);
        if constexpr (kUseCpAsync) {
            cp_async_commit_group();
            cp_async_wait_group<0>();
        }
        __syncthreads();
    }

    for (int iter = 0; row_base < n; row_base += row_stride, ++iter) {
        const int stage = iter & 1;
        const int row = row_base + row_slot;
        const int next_row_base = row_base + row_stride;

        if (next_row_base < n) {
            prefetch_weight_tile_rows_no_commit<kUseCpAsync, kPersistentRowsPerBatchK6144, kFullVecsK6144>(
                &w_sh[stage ^ 1][0],
                w_half,
                stride_wn,
                next_row_base,
                0,
                n,
                kRow4Threads);
            if constexpr (kUseCpAsync) {
                cp_async_commit_group();
            }
        }

        if constexpr (kProfile) {
            maybe_record_profile<kProfile>(profile, kProfileTile0LoadDone);
        }

        float sum = 0.0f;
        if (row < n) {
            const uint4* x_sh_vec = x_sh;
            const uint4* w_row_vec = &w_sh[stage][row_slot * kFullVecsK6144];
            __half2 h2_acc = __float2half2_rn(0.0f);

            #pragma unroll
            for (int vec_idx = lane + (warp_in_row << 5); vec_idx < kFullVecsK6144; vec_idx += 64) {
                const uint4 x_vec = x_sh_vec[vec_idx];
                const uint4 w_vec = w_row_vec[vec_idx];
                accumulate_hfma2_uint4(x_vec, w_vec, h2_acc);
            }

            const float2 f2 = __half22float2(h2_acc);
            sum = f2.x + f2.y;
        }

        if constexpr (kProfile) {
            maybe_record_profile<kProfile>(profile, kProfileTile0ComputeDone);
        }

        sum = warp_sum(sum);
        if (lane == 0 && row_slot < kPersistentRowsPerBatchK6144) {
            warp_partials[row_slot][warp_in_row] = sum;
        }
        __syncthreads();

        if (row < n && lane == 0 && warp_in_row == 0) {
            float result = warp_partials[row_slot][0] + warp_partials[row_slot][1];
            if (has_bias) {
                result += __half2float(reinterpret_cast<const __half*>(bias)[row]);
            }
            reinterpret_cast<__half*>(out)[row] = __float2half(result);
        }

        if constexpr (kUseCpAsync) {
            if (next_row_base < n) {
                cp_async_wait_group<0>();
            }
        }
        __syncthreads();
    }

    if constexpr (kProfile) {
        __syncthreads();
        maybe_record_profile<kProfile>(profile, kProfileKernelDone);
    }
#else
    (void)x;
    (void)weight;
    (void)bias;
    (void)out;
    (void)profile;
    (void)stride_wn;
    (void)n;
    (void)has_bias;
#endif
}

template <bool kProfile>
__global__ void nosplit_matvec_k6144_persistent_xfull_row1_warp8_async_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    uint64_t* __restrict__ profile,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __align__(16) uint4 x_sh[kFullVecsK6144];
    __shared__ __align__(16) uint4 w_sh[2][kFullVecsK6144];
    __shared__ float warp_partials[kPersistentWarp8PerRowK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    maybe_record_profile<kProfile>(profile, kProfileKernelStart);

    prefetch_x_tile_vecs<kUseCpAsync, kFullVecsK6144, kRow8Threads>(
        &x_sh[0],
        reinterpret_cast<const uint4*>(x_half));
    if constexpr (kUseCpAsync) {
        cp_async_commit_group();
        cp_async_wait_group<0>();
    }
    __syncthreads();

    int row = blockIdx.x;
    if (row < n) {
        prefetch_weight_tile_rows_no_commit<kUseCpAsync, 1, kFullVecsK6144>(
            &w_sh[0][0],
            w_half,
            stride_wn,
            row,
            0,
            n,
            kRow8Threads);
        if constexpr (kUseCpAsync) {
            cp_async_commit_group();
            cp_async_wait_group<0>();
        }
        __syncthreads();
    }

    for (int iter = 0; row < n; row += gridDim.x, ++iter) {
        const int stage = iter & 1;
        const int next_row = row + gridDim.x;

        if (next_row < n) {
            prefetch_weight_tile_rows_no_commit<kUseCpAsync, 1, kFullVecsK6144>(
                &w_sh[stage ^ 1][0],
                w_half,
                stride_wn,
                next_row,
                0,
                n,
                kRow8Threads);
            if constexpr (kUseCpAsync) {
                cp_async_commit_group();
            }
        }

        if constexpr (kProfile) {
            maybe_record_profile<kProfile>(profile, kProfileTile0LoadDone);
        }

        float sum = 0.0f;
        const uint4* w_row_vec = &w_sh[stage][0];
        __half2 h2_acc = __float2half2_rn(0.0f);

        #pragma unroll
        for (int vec_idx = lane + warp * 32; vec_idx < kFullVecsK6144; vec_idx += kRow8Threads) {
            const uint4 x_vec = x_sh[vec_idx];
            const uint4 w_vec = w_row_vec[vec_idx];
            accumulate_hfma2_uint4(x_vec, w_vec, h2_acc);
        }

        const float2 f2 = __half22float2(h2_acc);
        sum = f2.x + f2.y;

        if constexpr (kProfile) {
            maybe_record_profile<kProfile>(profile, kProfileTile0ComputeDone);
        }

        sum = warp_sum(sum);
        if (lane == 0) {
            warp_partials[warp] = sum;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            float result = 0.0f;
            #pragma unroll
            for (int warp_idx = 0; warp_idx < kPersistentWarp8PerRowK6144; ++warp_idx) {
                result += warp_partials[warp_idx];
            }
            if (has_bias) {
                result += __half2float(reinterpret_cast<const __half*>(bias)[row]);
            }
            reinterpret_cast<__half*>(out)[row] = __float2half(result);
        }

        if constexpr (kUseCpAsync) {
            if (next_row < n) {
                cp_async_wait_group<0>();
            }
        }
        __syncthreads();
    }

    if constexpr (kProfile) {
        __syncthreads();
        maybe_record_profile<kProfile>(profile, kProfileKernelDone);
    }
}

template <bool kProfile>
__global__ void nosplit_matvec_k6144_persistent_xfull_n8k1024_w_async_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    uint64_t* __restrict__ profile,
    int64_t stride_wn,
    int n,
    bool has_bias) {
    __shared__ __align__(16) uint4 x_sh[kFullVecsK6144];
    __shared__ __align__(16) uint4 w_sh[2][kRowsPerBlockAsync * kAsyncVecsPerTileK6144];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_slot = warp;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr bool kUseCpAsync = true;
#else
    constexpr bool kUseCpAsync = false;
#endif

    maybe_record_profile<kProfile>(profile, kProfileKernelStart);

    prefetch_x_tile_vecs<kUseCpAsync, kFullVecsK6144, kRow8Threads>(
        &x_sh[0],
        reinterpret_cast<const uint4*>(x_half));
    if constexpr (kUseCpAsync) {
        cp_async_commit_group();
        cp_async_wait_group<0>();
    }
    __syncthreads();

    const int row_stride = gridDim.x * kRowsPerBlockAsync;
    for (int row_base = blockIdx.x * kRowsPerBlockAsync; row_base < n; row_base += row_stride) {
        const int row = row_base + row_slot;

        prefetch_weight_tile_rows_no_commit<kUseCpAsync, kRowsPerBlockAsync, kAsyncVecsPerTileK6144>(
            &w_sh[0][0],
            w_half,
            stride_wn,
            row_base,
            0,
            n,
            kRow8Threads);
        if constexpr (kUseCpAsync) {
            cp_async_commit_group();
            cp_async_wait_group<0>();
        }
        __syncthreads();

        float sum = 0.0f;
        #pragma unroll
        for (int tile_idx = 0; tile_idx < 6; ++tile_idx) {
            const int stage = tile_idx & 1;
            const int tile_start = tile_idx * kNosplitAsyncTileK6144;
            const int x_vec_base = tile_idx * kAsyncVecsPerTileK6144;

            if (tile_idx + 1 < 6) {
                const int next_tile_start = tile_start + kNosplitAsyncTileK6144;
                prefetch_weight_tile_rows_no_commit<kUseCpAsync, kRowsPerBlockAsync, kAsyncVecsPerTileK6144>(
                    &w_sh[stage ^ 1][0],
                    w_half,
                    stride_wn,
                    row_base,
                    next_tile_start,
                    n,
                    kRow8Threads);
                if constexpr (kUseCpAsync) {
                    cp_async_commit_group();
                }
            }

            if constexpr (kProfile) {
                maybe_record_profile<kProfile>(profile, kProfileTile0LoadDone + tile_idx * 2);
            }

            if (row < n) {
                const uint4* x_sh_stage_vec = &x_sh[x_vec_base];
                const uint4* w_sh_row_vec = &w_sh[stage][warp * kAsyncVecsPerTileK6144];
                __half2 h2_acc0 = __float2half2_rn(0.0f);
                __half2 h2_acc1 = __float2half2_rn(0.0f);

                #pragma unroll
                for (int vec_idx = lane; vec_idx < kAsyncVecsPerTileK6144; vec_idx += 64) {
                    const uint4 x_vec0 = x_sh_stage_vec[vec_idx];
                    const uint4 w_vec0 = w_sh_row_vec[vec_idx];
                    const uint4 x_vec1 = x_sh_stage_vec[vec_idx + 32];
                    const uint4 w_vec1 = w_sh_row_vec[vec_idx + 32];
                    accumulate_hfma2_uint4(x_vec0, w_vec0, h2_acc0);
                    accumulate_hfma2_uint4(x_vec1, w_vec1, h2_acc1);
                }

                const float2 f2_0 = __half22float2(h2_acc0);
                const float2 f2_1 = __half22float2(h2_acc1);
                sum += f2_0.x + f2_0.y + f2_1.x + f2_1.y;
            }

            if constexpr (kProfile) {
                maybe_record_profile<kProfile>(profile, kProfileTile0ComputeDone + tile_idx * 2);
            }

            if constexpr (kUseCpAsync) {
                if (tile_idx + 1 < 6) {
                    cp_async_wait_group<0>();
                }
            }
            __syncthreads();
        }

        if (row < n) {
            sum = warp_sum(sum);
            if (lane == 0) {
                float result = sum;
                if (has_bias) {
                    result += __half2float(reinterpret_cast<const __half*>(bias)[row]);
                }
                reinterpret_cast<__half*>(out)[row] = __float2half(result);
            }
        }
        __syncthreads();
    }

    if constexpr (kProfile) {
        __syncthreads();
        maybe_record_profile<kProfile>(profile, kProfileKernelDone);
    }
}

__global__ void splitk_matvec_row4_shared_vec_half_v2_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    float* __restrict__ scratch,
    int64_t stride_wn,
    int n,
    int k,
    int split_k) {
    __shared__ __half x_sh[kVecTileK_v2];

    const int row_base = blockIdx.x * 4;
    const int split_idx = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = row_base + warp;
    if (row >= n) {
        return;
    }

    const int chunk_k = (k + split_k - 1) / split_k;
    const int chunk_start = split_idx * chunk_k;
    const int chunk_end = min(chunk_start + chunk_k, k);
    float sum = 0.0f;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);

    for (int tile_start = chunk_start; tile_start < chunk_end; tile_start += kVecTileK_v2) {
        const int tile_elems = min(kVecTileK_v2, chunk_end - tile_start);

        // --- 128-bit vectorized load of x into shared memory ---
        // Each thread loads 8 halfs (uint4 = 128 bits)
        // 128 threads * 8 halfs = 1024 halfs per pass (exactly one tile)
        const int vec8_limit_load = tile_elems & ~7;
        for (int idx = threadIdx.x * 8; idx < vec8_limit_load; idx += kRow4Threads * 8) {
            *reinterpret_cast<uint4*>(&x_sh[idx]) =
                *reinterpret_cast<const uint4*>(&x_half[tile_start + idx]);
        }
        // scalar tail for x load
        for (int idx = vec8_limit_load + threadIdx.x; idx < tile_elems; idx += kRow4Threads) {
            x_sh[idx] = x_half[tile_start + idx];
        }
        __syncthreads();

        // --- Vectorized compute with 128-bit loads ---
        const int compute_vec_limit = tile_elems & ~7;
        for (int idx = lane * 8; idx < compute_vec_limit; idx += 32 * 8) {
            // 128-bit load from shared memory (x)
            const uint4 x_vec = *reinterpret_cast<const uint4*>(&x_sh[idx]);
            // 128-bit load from global memory (weight) -- assumes stride_wk == 1
            const uint4 w_vec = *reinterpret_cast<const uint4*>(
                &w_half[row * stride_wn + tile_start + idx]);

            const __half2* x2 = reinterpret_cast<const __half2*>(&x_vec);
            const __half2* w2 = reinterpret_cast<const __half2*>(&w_vec);

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 xf = __half22float2(x2[i]);
                const float2 wf = __half22float2(w2[i]);
                sum += xf.x * wf.x + xf.y * wf.y;
            }
        }

        // scalar tail
        for (int idx = compute_vec_limit + lane; idx < tile_elems; idx += 32) {
            sum += __half2float(x_sh[idx]) * __half2float(w_half[row * stride_wn + tile_start + idx]);
        }
        __syncthreads();
    }

    sum = warp_sum(sum);
    if (lane == 0) {
        atomicAdd(scratch + row, sum);
    }
}

// ============================================================
// Persistent GEMV kernel: launch exactly #SM blocks, no split-K
// - 1 kernel launch (no scratch.zero_ + no finalize)
// - No atomicAdd
// - Each block loops over row groups, full K reduction per row
// ============================================================
__global__ void persistent_matvec_half_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    int64_t stride_wn,
    int n,
    int k,
    bool has_bias) {
    __shared__ __half x_sh[kVecTileK_v2];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);

    // Each block loops over row groups, stride = gridDim.x * 4
    for (int row_base = blockIdx.x * 4; row_base < n; row_base += gridDim.x * 4) {
        const int row = row_base + warp;
        float sum = 0.0f;

        // Tile over the full K dimension
        for (int tile_start = 0; tile_start < k; tile_start += kVecTileK_v2) {
            const int tile_elems = min(kVecTileK_v2, k - tile_start);

            // 128-bit vectorized load of x into shared memory
            const int vec8_limit_load = tile_elems & ~7;
            for (int idx = threadIdx.x * 8; idx < vec8_limit_load; idx += kRow4Threads * 8) {
                *reinterpret_cast<uint4*>(&x_sh[idx]) =
                    *reinterpret_cast<const uint4*>(&x_half[tile_start + idx]);
            }
            for (int idx = vec8_limit_load + threadIdx.x; idx < tile_elems; idx += kRow4Threads) {
                x_sh[idx] = x_half[tile_start + idx];
            }
            __syncthreads();

            if (row < n) {
                // 128-bit vectorized compute
                const int compute_vec_limit = tile_elems & ~7;
                for (int idx = lane * 8; idx < compute_vec_limit; idx += 32 * 8) {
                    const uint4 x_vec = *reinterpret_cast<const uint4*>(&x_sh[idx]);
                    const uint4 w_vec = *reinterpret_cast<const uint4*>(
                        &w_half[row * stride_wn + tile_start + idx]);

                    const __half2* x2 = reinterpret_cast<const __half2*>(&x_vec);
                    const __half2* w2 = reinterpret_cast<const __half2*>(&w_vec);

                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float2 xf = __half22float2(x2[i]);
                        const float2 wf = __half22float2(w2[i]);
                        sum += xf.x * wf.x + xf.y * wf.y;
                    }
                }

                // scalar tail
                for (int idx = compute_vec_limit + lane; idx < tile_elems; idx += 32) {
                    sum += __half2float(x_sh[idx]) * __half2float(w_half[row * stride_wn + tile_start + idx]);
                }
            }
            __syncthreads();
        }

        if (row < n) {
            sum = warp_sum(sum);
            if (lane == 0) {
                float result = sum;
                if (has_bias) {
                    result += __half2float(reinterpret_cast<const __half*>(bias)[row]);
                }
                reinterpret_cast<__half*>(out)[row] = __float2half(result);
            }
        }
    }
}

// ============================================================
// hfma2 variant: accumulate in half2, per-tile flush to float
// - Reduces instruction count (~6x fewer ALU ops in inner loop)
// - Per-tile flush keeps precision safe (max 16 hfma2 per flush)
// ============================================================
__global__ void persistent_matvec_hfma2_fwd_kernel(
    const at::Half* __restrict__ x,
    const at::Half* __restrict__ weight,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ out,
    int64_t stride_wn,
    int n,
    int k,
    bool has_bias) {
    __shared__ __half x_sh[kVecTileK_v2];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const __half* x_half = reinterpret_cast<const __half*>(x);
    const __half* w_half = reinterpret_cast<const __half*>(weight);

    for (int row_base = blockIdx.x * 4; row_base < n; row_base += gridDim.x * 4) {
        const int row = row_base + warp;
        float sum = 0.0f;

        for (int tile_start = 0; tile_start < k; tile_start += kVecTileK_v2) {
            const int tile_elems = min(kVecTileK_v2, k - tile_start);

            // 128-bit vectorized load of x into shared memory
            const int vec8_limit_load = tile_elems & ~7;
            for (int idx = threadIdx.x * 8; idx < vec8_limit_load; idx += kRow4Threads * 8) {
                *reinterpret_cast<uint4*>(&x_sh[idx]) =
                    *reinterpret_cast<const uint4*>(&x_half[tile_start + idx]);
            }
            for (int idx = vec8_limit_load + threadIdx.x; idx < tile_elems; idx += kRow4Threads) {
                x_sh[idx] = x_half[tile_start + idx];
            }
            __syncthreads();

            if (row < n) {
                // hfma2 accumulation with per-tile flush
                __half2 h2_acc = __float2half2_rn(0.0f);
                const int compute_vec_limit = tile_elems & ~7;
                for (int idx = lane * 8; idx < compute_vec_limit; idx += 32 * 8) {
                    const uint4 x_vec = *reinterpret_cast<const uint4*>(&x_sh[idx]);
                    const uint4 w_vec = *reinterpret_cast<const uint4*>(
                        &w_half[row * stride_wn + tile_start + idx]);

                    const __half2* x2 = reinterpret_cast<const __half2*>(&x_vec);
                    const __half2* w2 = reinterpret_cast<const __half2*>(&w_vec);

                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        h2_acc = __hfma2(x2[i], w2[i], h2_acc);
                    }
                }
                // flush half2 accumulator to float
                const float2 f2 = __half22float2(h2_acc);
                sum += f2.x + f2.y;

                // scalar tail (stays in float)
                for (int idx = compute_vec_limit + lane; idx < tile_elems; idx += 32) {
                    sum += __half2float(x_sh[idx]) * __half2float(w_half[row * stride_wn + tile_start + idx]);
                }
            }
            __syncthreads();
        }

        if (row < n) {
            sum = warp_sum(sum);
            if (lane == 0) {
                float result = sum;
                if (has_bias) {
                    result += __half2float(reinterpret_cast<const __half*>(bias)[row]);
                }
                reinterpret_cast<__half*>(out)[row] = __float2half(result);
            }
        }
    }
}

void launch_nosplit_variant(
    const at::Half* x,
    const at::Half* weight,
    const at::Half* bias,
    at::Half* out,
    int64_t stride_wn,
    int n,
    int k,
    bool has_bias,
    int64_t variant,
    int num_sms,
    cudaStream_t stream) {
    const bool auto_use_row2_k2048 = (k == kNosplitSpecializedK2048 && n <= 2048);
    const int row2_grid_blocks = (n + 1) / 2;
    const int row4_grid_blocks = (n + 3) / 4;

    switch (variant) {
        case kNosplitVariantAuto:
            if (auto_use_row2_k2048) {
                nosplit_matvec_k2048_row2_hfma2_fwd_kernel<false><<<row2_grid_blocks, kRow2Threads, 0, stream>>>(
                    x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            } else if (k == kNosplitSpecializedK2048) {
                nosplit_matvec_k2048_hfma2_fwd_kernel<false><<<row4_grid_blocks, kRow4Threads, 0, stream>>>(
                    x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            } else if (k == 6144) {
                nosplit_matvec_k6144_hfma2_fwd_kernel<false><<<row4_grid_blocks, kRow4Threads, 0, stream>>>(
                    x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            } else {
                persistent_matvec_half_fwd_kernel<<<num_sms, kRow4Threads, 0, stream>>>(
                    x, weight, bias, out, stride_wn, n, k, has_bias);
            }
            break;
        case kNosplitVariantK2048Row2:
            TORCH_CHECK(k == kNosplitSpecializedK2048, "k2048_row2 variant requires k=2048");
            nosplit_matvec_k2048_row2_hfma2_fwd_kernel<false><<<row2_grid_blocks, kRow2Threads, 0, stream>>>(
                x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            break;
        case kNosplitVariantK2048Row4:
            TORCH_CHECK(k == kNosplitSpecializedK2048, "k2048_row4 variant requires k=2048");
            nosplit_matvec_k2048_hfma2_fwd_kernel<false><<<row4_grid_blocks, kRow4Threads, 0, stream>>>(
                x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            break;
        case kNosplitVariantK6144Row4:
            TORCH_CHECK(k == 6144, "k6144_row4 variant requires k=6144");
            nosplit_matvec_k6144_hfma2_fwd_kernel<false><<<row4_grid_blocks, kRow4Threads, 0, stream>>>(
                x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            break;
        case kNosplitVariantK6144WeightAsync:
            TORCH_CHECK(k == 6144, "k6144_weight_async variant requires k=6144");
            nosplit_matvec_k6144_weight_async_fwd_kernel<false><<<row4_grid_blocks, kRow4Threads, 0, stream>>>(
                x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            break;
        case kNosplitVariantK6144N8K1024Async: {
            TORCH_CHECK(k == 6144, "k6144_n8k1024_async variant requires k=6144");
            const int row8_grid_blocks = (n + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;
            nosplit_matvec_k6144_n8k1024_async_fwd_kernel<false><<<row8_grid_blocks, kRow8Threads, 0, stream>>>(
                x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            break;
        }
        case kNosplitVariantK6144N8K1024XWAsync: {
            TORCH_CHECK(k == 6144, "k6144_n8k1024_xw_async variant requires k=6144");
            const int row8_grid_blocks = (n + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;
            nosplit_matvec_k6144_n8k1024_xw_async_fwd_kernel<false><<<row8_grid_blocks, kRow8Threads, 0, stream>>>(
                x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            break;
        }
        case kNosplitVariantK6144N8K512S3XWAsync: {
            TORCH_CHECK(k == 6144, "k6144_n8k512_s3_xw_async variant requires k=6144");
            const int row8_grid_blocks = (n + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;
            nosplit_matvec_k6144_n8_xw_async_multi_stage_fwd_kernel<kNosplitAsyncTileK6144Small, 3, false>
                <<<row8_grid_blocks, kRow8Threads, 0, stream>>>(
                    x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            break;
        }
        case kNosplitVariantK6144PersistentXFullW2Async:
            TORCH_CHECK(k == 6144, "k6144_persistent_xfull_w2_async variant requires k=6144");
            {
                int cc_major = 0;
                int cc_minor = 0;
                cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, at::cuda::current_device());
                cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, at::cuda::current_device());
                TORCH_CHECK(
                    (cc_major > 8) || (cc_major == 8 && cc_minor >= 6),
                    "k6144_persistent_xfull_w2_async variant requires sm86+ due to static shared-memory footprint");
            }
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                nosplit_matvec_k6144_persistent_xfull_w2_async_fwd_kernel<false>,
                cudaFuncAttributePreferredSharedMemoryCarveout,
                100));
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                nosplit_matvec_k6144_persistent_xfull_w2_async_fwd_kernel<false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                0));
            nosplit_matvec_k6144_persistent_xfull_w2_async_fwd_kernel<false>
                <<<num_sms, kRow4Threads, 0, stream>>>(
                    x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            break;
        case kNosplitVariantK6144PersistentXFullRow1Warp8Async:
            TORCH_CHECK(k == 6144, "k6144_persistent_xfull_row1_warp8_async variant requires k=6144");
            nosplit_matvec_k6144_persistent_xfull_row1_warp8_async_fwd_kernel<false>
                <<<num_sms, kRow8Threads, 0, stream>>>(
                    x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            break;
        case kNosplitVariantK6144PersistentXFullN8K1024WAsync:
            TORCH_CHECK(k == 6144, "k6144_persistent_xfull_n8k1024_w_async variant requires k=6144");
            nosplit_matvec_k6144_persistent_xfull_n8k1024_w_async_fwd_kernel<false>
                <<<num_sms, kRow8Threads, 0, stream>>>(
                    x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            break;
        case kNosplitVariantK6144N8K1024XWarpWAsync: {
            TORCH_CHECK(k == 6144, "k6144_n8k1024_xwarp_w_async variant requires k=6144");
            const int row8_grid_blocks = (n + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;
            nosplit_matvec_k6144_n8k1024_xwarp_w_async_fwd_kernel<false><<<row8_grid_blocks, kRow8Threads, 0, stream>>>(
                x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            break;
        }
        case kNosplitVariantK6144N8K1024XWarpWAsyncSms2: {
            TORCH_CHECK(k == 6144, "k6144_n8k1024_xwarp_w_async_sms2 variant requires k=6144");
            const int row8_grid_blocks = (n + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;
            const int launch_grid_blocks = std::min(row8_grid_blocks, std::max(1, num_sms * 2));
            nosplit_matvec_k6144_n8k1024_xwarp_w_async_sms2_fwd_kernel<false>
                <<<launch_grid_blocks, kRow8Threads, 0, stream>>>(
                    x, weight, bias, out, nullptr, stride_wn, n, has_bias);
            break;
        }
        case kNosplitVariantK6144N8K1024XWarpWAsyncSms2Balanced: {
            TORCH_CHECK(k == 6144, "k6144_n8k1024_xwarp_w_async_sms2_balanced variant requires k=6144");
            const int row8_grid_blocks = (n + kRowsPerBlockAsync - 1) / kRowsPerBlockAsync;
            const int launch_grid_blocks = std::min(row8_grid_blocks, std::max(1, num_sms * 2));
            nosplit_matvec_k6144_n8k1024_xwarp_w_async_sms2_balanced_fwd_kernel
                <<<launch_grid_blocks, kRow8Threads, 0, stream>>>(
                    x, weight, bias, out, stride_wn, n, has_bias);
            break;
        }
        case kNosplitVariantPersistent:
            persistent_matvec_half_fwd_kernel<<<num_sms, kRow4Threads, 0, stream>>>(
                x, weight, bias, out, stride_wn, n, k, has_bias);
            break;
        case kNosplitVariantPersistentHfma2:
            persistent_matvec_hfma2_fwd_kernel<<<num_sms, kRow4Threads, 0, stream>>>(
                x, weight, bias, out, stride_wn, n, k, has_bias);
            break;
        default:
            TORCH_CHECK(false, "unknown nosplit variant: ", variant);
    }
}

}  // namespace

