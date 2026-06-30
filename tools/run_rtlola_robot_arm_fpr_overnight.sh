#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${PZR_OUT_DIR:-$ROOT_DIR/results/rtlola-arm-fpr-overnight}"
PYTHON="${PZR_PYTHON:-$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm/bin/python}"
ENV_PREFIX="${PZR_ENV_PREFIX:-$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm}"
LENGTH="${PZR_LENGTH:-2340}"
BUDGETS="${PZR_BUDGETS:-40,80,120,180}"
EVAL_TRACES="${PZR_EVAL_TRACES:-figure8_violated,square_violated}"
TRAIN_TRACES="${PZR_TRAIN_TRACES:-figure8,square}"
METHODS="${PZR_METHODS:-girard,scott,interval_hull,pca,combastel,mpc_beam}"
HORIZON="${PZR_HORIZON:-4}"
BEAM_WIDTH="${PZR_BEAM_WIDTH:-4}"
SEEDS="${PZR_SEEDS:-1}"
REGRET_ITERATIONS="${PZR_REGRET_ITERATIONS:-1}"
REGRET_EPOCHS="${PZR_REGRET_EPOCHS:-50}"
REGRET_TRAIN_SEEDS="${PZR_REGRET_TRAIN_SEEDS:-1}"
REGRET_EVAL_SEEDS="${PZR_REGRET_EVAL_SEEDS:-1}"
MAX_SECONDS="${PZR_MAX_SECONDS:-27000}"
SKIP_LEARNING="${PZR_SKIP_LEARNING:-0}"
START_SECONDS="$(date +%s)"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUT_DIR/matplotlib-cache}"
if [[ -f "$ENV_PREFIX/lib/libopenblas.so" ]]; then
    export LD_PRELOAD="$ENV_PREFIX/lib/libopenblas.so${LD_PRELOAD:+:$LD_PRELOAD}"
fi

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/references" "$OUT_DIR/runs"

remaining_seconds() {
    local elapsed
    elapsed=$(( $(date +%s) - START_SECONDS ))
    echo $(( MAX_SECONDS - elapsed ))
}

run_stage() {
    local stage="$1"
    shift
    local marker="$OUT_DIR/.${stage}.complete"
    local log="$OUT_DIR/logs/${stage}.log"
    if [[ -f "$marker" ]]; then
        echo "skip completed stage: $stage"
        return 0
    fi
    local remaining
    remaining="$(remaining_seconds)"
    if (( remaining <= 0 )); then
        echo "time budget exhausted before stage: $stage"
        return 124
    fi
    echo "start stage: $stage (remaining ${remaining}s)"
    timeout --signal=TERM "${remaining}s" "$@" >"$log" 2>&1
    local status=$?
    if (( status != 0 )); then
        echo "stage failed: $stage (status $status, log $log)"
        return "$status"
    fi
    touch "$marker"
    echo "complete stage: $stage"
}

IFS=',' read -r -a budget_values <<< "$BUDGETS"
IFS=',' read -r -a eval_trace_values <<< "$EVAL_TRACES"

for trace_kind in "${eval_trace_values[@]}"; do
    for budget in "${budget_values[@]}"; do
        stage="${trace_kind}_budget_${budget}"
        cell_dir="$OUT_DIR/runs/$trace_kind/budget_$budget"
        run_stage "$stage" "$PYTHON" -m pzr.rtlola.cli \
            --profile paper \
            --scenario robot_arm \
            --trace-kind "$trace_kind" \
            --length "$LENGTH" \
            --seeds "$SEEDS" \
            --budget "$budget" \
            --horizon "$HORIZON" \
            --beam-width "$BEAM_WIDTH" \
            --methods "$METHODS" \
            --reference-mode verdict \
            --reference-cache "$OUT_DIR/references/${trace_kind}.seed_0.json" \
            --no-progress \
            --output "$cell_dir" || exit $?
    done
done

if [[ "$SKIP_LEARNING" != "1" ]]; then
    first_budget="${budget_values[0]}"
    first_trace="${eval_trace_values[0]}"
    run_stage "pooled_learning" "$PYTHON" -m pzr.rtlola.cli \
        --profile paper \
        --scenario robot_arm \
        --trace-kind "$first_trace" \
        --length "$LENGTH" \
        --seeds 1 \
        --budget "$first_budget" \
        --horizon "$HORIZON" \
        --beam-width "$BEAM_WIDTH" \
        --methods girard \
        --reference-mode verdict \
        --reference-cache "$OUT_DIR/references/${first_trace}.seed_0.json" \
        --learned-mode regret \
        --regret-iterations "$REGRET_ITERATIONS" \
        --regret-epochs "$REGRET_EPOCHS" \
        --regret-train-seeds "$REGRET_TRAIN_SEEDS" \
        --regret-eval-seeds "$REGRET_EVAL_SEEDS" \
        --regret-budgets "$BUDGETS" \
        --regret-train-traces "$TRAIN_TRACES" \
        --regret-eval-traces "$EVAL_TRACES" \
        --no-progress \
        --output "$OUT_DIR/learning_stage" || exit $?
fi

"$PYTHON" -m pzr.rtlola.sweep_report --root "$OUT_DIR"
echo "RTLola FPR overnight sweep complete: $OUT_DIR"
