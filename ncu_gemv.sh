#!/usr/bin/env bash
# NCU profiling for decode-phase GEMV operators (o_proj, down_proj, lm_head).
#
# Usage:
#   bash ncu_gemv.sh                   # Profile all 3 targets with one-pass metrics
#   bash ncu_gemv.sh o_proj            # Profile only o_proj
#   bash ncu_gemv.sh down_proj deep    # Profile down_proj with application-replay sections
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

NCU_BIN=/usr/local/cuda-12.6/bin/ncu
NCU_OUT_DIR=results/ncu
PYTHON=./venv/bin/python
SCRIPT=profile_gemv_ncu.py
mkdir -p "${NCU_OUT_DIR}"

TARGET="${1:-all}"
MODE="${2:-metrics}"

# Common capture params
SAMPLE_IDX=0
LAYER_IDX=14
DECODE_STEP=16
VARIANT=f_linear
WARMUP=5
ITERS=1

# One-pass explicit metrics (safe: avoids kernel-replay LaunchFailed)
METRICS="sm__cycles_elapsed.avg"
METRICS="${METRICS},sm__cycles_elapsed.avg.per_second"
METRICS="${METRICS},dram__throughput.avg.pct_of_peak_sustained_elapsed"
METRICS="${METRICS},smsp__warps_active.avg.pct_of_peak_sustained_active"
METRICS="${METRICS},l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum"
METRICS="${METRICS},l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum"
METRICS="${METRICS},smsp__inst_executed.sum"

run_ncu_metrics() {
    local target=$1
    echo "=== [NCU-GEMV] Profiling ${target} (variant=${VARIANT}, mode=metrics) ==="
    # --profile-from-start off: only profile kernels between cudaProfilerStart/Stop
    # The Python script calls cudaProfilerStart() right before the profiling loop.
    TQDM_DISABLE=1 sudo "${NCU_BIN}" -f \
        --target-processes all \
        --kernel-name-base demangled \
        --profile-from-start off \
        --metrics "${METRICS}" \
        --print-summary per-kernel \
        --export "${NCU_OUT_DIR}/gemv_${target}_${VARIANT}_metrics" \
        "${PYTHON}" "${SCRIPT}" \
            --target "${target}" \
            --variant "${VARIANT}" \
            --sample-idx "${SAMPLE_IDX}" \
            --layer-idx "${LAYER_IDX}" \
            --decode-step "${DECODE_STEP}" \
            --warmup "${WARMUP}" \
            --iters "${ITERS}" \
            --metadata-output "${NCU_OUT_DIR}/gemv_${target}_${VARIANT}_meta.json"
    echo "=== [NCU-GEMV] ${target} metrics done: ${NCU_OUT_DIR}/gemv_${target}_${VARIANT}_metrics ==="
}

run_ncu_deep() {
    local target=$1
    echo "=== [NCU-GEMV] Profiling ${target} (variant=${VARIANT}, mode=deep / application-replay) ==="
    TQDM_DISABLE=1 sudo "${NCU_BIN}" -f \
        --target-processes all \
        --kernel-name-base demangled \
        --profile-from-start off \
        --replay-mode application \
        --app-replay-mode relaxed \
        --section MemoryWorkloadAnalysis \
        --section Occupancy \
        --section WarpStateStats \
        --section SchedulerStats \
        --section LaunchStats \
        --print-summary per-kernel \
        --export "${NCU_OUT_DIR}/gemv_${target}_${VARIANT}_deep" \
        "${PYTHON}" "${SCRIPT}" \
            --target "${target}" \
            --variant "${VARIANT}" \
            --sample-idx "${SAMPLE_IDX}" \
            --layer-idx "${LAYER_IDX}" \
            --decode-step "${DECODE_STEP}" \
            --warmup "${WARMUP}" \
            --iters "${ITERS}" \
            --metadata-output "${NCU_OUT_DIR}/gemv_${target}_${VARIANT}_meta.json"
    echo "=== [NCU-GEMV] ${target} deep done: ${NCU_OUT_DIR}/gemv_${target}_${VARIANT}_deep ==="
}

run_target() {
    local target=$1
    if [ "${MODE}" = "deep" ]; then
        run_ncu_deep "${target}"
    else
        run_ncu_metrics "${target}"
    fi
}

if [ "${TARGET}" = "all" ]; then
    run_target o_proj
    run_target down_proj
    run_target lm_head
else
    run_target "${TARGET}"
fi

echo "=== [NCU-GEMV] All done. Results in ${NCU_OUT_DIR}/ ==="
