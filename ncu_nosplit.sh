#!/usr/bin/env bash
# NCU profiling for current CUDA nosplit GEMV targets.
#
# Usage:
#   bash ncu_nosplit.sh
#   bash ncu_nosplit.sh qkv
#   bash ncu_nosplit.sh down_proj cuda_nosplit_hfma2_matvec
#   bash ncu_nosplit.sh o_proj cuda_nosplit_matvec deep
#   bash ncu_nosplit.sh down_proj cuda_nosplit_matvec full
#   bash ncu_nosplit.sh all cuda_nosplit_matvec kernel_deep
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

NCU_BIN=/usr/local/cuda-12.6/bin/ncu
NCU_OUT_DIR=results/ncu
PYTHON=./venv/bin/python
SCRIPT=profile_gemv_ncu.py
mkdir -p "${NCU_OUT_DIR}"

TARGET="${1:-all}"
VARIANT="${2:-cuda_nosplit_matvec}"
MODE="${3:-metrics}"

SAMPLE_IDX=0
LAYER_IDX=14
DECODE_STEP=16
WARMUP=5
ITERS=1

METRICS="sm__cycles_elapsed.avg"
METRICS="${METRICS},sm__cycles_elapsed.avg.per_second"
METRICS="${METRICS},dram__throughput.avg.pct_of_peak_sustained_elapsed"
METRICS="${METRICS},smsp__warps_active.avg.pct_of_peak_sustained_active"
METRICS="${METRICS},l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum"
METRICS="${METRICS},l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum"
METRICS="${METRICS},smsp__inst_executed.sum"

run_target() {
    local target=$1
    local suffix
    local extra_args=()
    case "${MODE}" in
        metrics)
            suffix="metrics"
            extra_args=(
                --metrics "${METRICS}"
            )
            ;;
        deep)
            suffix="deep"
            extra_args=(
                --replay-mode application
                --app-replay-mode relaxed
                --section ComputeWorkloadAnalysis
                --section LaunchStats
                --section MemoryWorkloadAnalysis
                --section MemoryWorkloadAnalysis_Chart
                --section MemoryWorkloadAnalysis_Tables
                --section Occupancy
                --section SchedulerStats
                --section SourceCounters
                --section SpeedOfLight
                --section WarpStateStats
                --import-source yes
                --source-folders "./cuda_extensions,."
            )
            ;;
        full)
            suffix="full"
            extra_args=(
                --replay-mode application
                --app-replay-mode relaxed
                --set full
                --import-source yes
                --source-folders "./cuda_extensions,."
            )
            ;;
        kernel_deep)
            suffix="kernel_deep"
            extra_args=(
                -k "regex:.*(persistent_matvec_half_fwd_kernel|nosplit_matvec_.*fwd_kernel).*"
                -c 1
                --section ComputeWorkloadAnalysis
                --section LaunchStats
                --section MemoryWorkloadAnalysis
                --section MemoryWorkloadAnalysis_Chart
                --section MemoryWorkloadAnalysis_Tables
                --section Occupancy
                --section SchedulerStats
                --section SourceCounters
                --section SpeedOfLight
                --section WarpStateStats
                --import-source yes
                --source-folders "./cuda_extensions,."
            )
            ;;
        *)
            echo "Unsupported MODE=${MODE}. Use metrics|deep|full|kernel_deep" >&2
            exit 2
            ;;
    esac

    echo "=== [NCU-NOSPLIT] Profiling ${target} (variant=${VARIANT}, mode=${MODE}) ==="
    TQDM_DISABLE=1 sudo "${NCU_BIN}" -f \
        --target-processes all \
        --kernel-name-base demangled \
        --profile-from-start off \
        --print-summary per-kernel \
        "${extra_args[@]}" \
        --export "${NCU_OUT_DIR}/nosplit_${target}_${VARIANT}_${suffix}" \
        "${PYTHON}" "${SCRIPT}" \
            --target "${target}" \
            --variant "${VARIANT}" \
            --sample-idx "${SAMPLE_IDX}" \
            --layer-idx "${LAYER_IDX}" \
            --decode-step "${DECODE_STEP}" \
            --warmup "${WARMUP}" \
            --iters "${ITERS}" \
            --metadata-output "${NCU_OUT_DIR}/nosplit_${target}_${VARIANT}_meta.json"
    echo "=== [NCU-NOSPLIT] ${target} done: ${NCU_OUT_DIR}/nosplit_${target}_${VARIANT}_${suffix} ==="
}

if [ "${TARGET}" = "all" ]; then
    run_target qkv
    run_target gate_up
    run_target o_proj
    run_target down_proj
else
    run_target "${TARGET}"
fi

echo "=== [NCU-NOSPLIT] All done. Results in ${NCU_OUT_DIR}/ ==="
