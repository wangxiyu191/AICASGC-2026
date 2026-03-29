#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

NCU_BIN=/usr/local/cuda-12.6/bin/ncu
NCU_OUT_DIR=results/ncu
mkdir -p "${NCU_OUT_DIR}"

# Default path: use explicit one-pass metrics.
# This avoids Nsight Compute kernel replay on the current machine, which
# fails with LaunchFailed during backing-store optimization.
TQDM_DISABLE=1 sudo "${NCU_BIN}" -f \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*small_decode_attention_grouped_prefix_cachepos_dot_fwd_kernel.*' \
  --launch-skip 3 \
  --launch-count 1 \
  --metrics \
sm__cycles_elapsed.avg,smsp__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed \
  --export "${NCU_OUT_DIR}/ncu_attention_dot_layer14_step16_metrics" \
  ./venv/bin/python ./profile_attention_ncu.py \
    --sample-idx 0 \
    --layer-idx 14 \
    --decode-step 16 \
    --variant triton_grouped_cachepos_prefix_dot \
    --warmup 3 \
    --iters 1 \
    --metadata-output "${NCU_OUT_DIR}/attention_ncu_layer14_step16_meta.json"

# Richer section profiling workaround on this machine:
# use application replay instead of the default kernel replay.
# It is slower, but avoids the LaunchFailed path caused by kernel replay.
#
# TQDM_DISABLE=1 sudo "${NCU_BIN}" -f \
#   --target-processes all \
#   --kernel-name-base demangled \
#   --kernel-name 'regex:.*small_decode_attention_grouped_prefix_cachepos_dot_fwd_kernel.*' \
#   --launch-skip 3 \
#   --launch-count 1 \
#   --replay-mode application \
#   --app-replay-mode relaxed \
#   --section Occupancy \
#   --section SchedulerStats \
#   --section WarpStateStats \
#   --export "${NCU_OUT_DIR}/ncu_attention_dot_layer14_step16_appreplay" \
#   ./venv/bin/python ./profile_attention_ncu.py \
#     --sample-idx 0 \
#     --layer-idx 14 \
#     --decode-step 16 \
#     --variant triton_grouped_cachepos_prefix_dot \
#     --warmup 3 \
#     --iters 1 \
#     --metadata-output "${NCU_OUT_DIR}/attention_ncu_layer14_step16_meta.json"
