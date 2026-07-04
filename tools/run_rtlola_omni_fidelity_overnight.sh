#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${PZR_OUT_DIR:-$ROOT_DIR/results/rtlola-omni-a143dd6-release}"
PYTHON="${PZR_PYTHON:-$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm/bin/python}"
ENV_PREFIX="${PZR_ENV_PREFIX:-$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm}"
LENGTH="${PZR_LENGTH:-250}"
BUDGETS="${PZR_BUDGETS:-8,12,16,20}"
TRACES="${PZR_TRACES:-canonical,safe,x_violated,y_violated}"
METHODS="${PZR_METHODS:-girard,scott,interval_hull,pca,combastel,mpc_beam}"
HORIZON="${PZR_HORIZON:-4}"
HORIZON_SCAN="${PZR_HORIZON_SCAN:-1,2,4,8}"
HORIZON_BUDGET="${PZR_HORIZON_BUDGET:-8}"
BEAM_WIDTH="${PZR_BEAM_WIDTH:-4}"
SEEDS="${PZR_SEEDS:-10}"
REGRET_ITERATIONS="${PZR_REGRET_ITERATIONS:-1}"
REGRET_EPOCHS="${PZR_REGRET_EPOCHS:-50}"
REGRET_TRAIN_SEEDS="${PZR_REGRET_TRAIN_SEEDS:-10}"
REGRET_EVAL_SEEDS="${PZR_REGRET_EVAL_SEEDS:-10}"
REGRET_TRAIN_SEED_START="${PZR_REGRET_TRAIN_SEED_START:-10000}"
REGRET_EVAL_SEED_START="${PZR_REGRET_EVAL_SEED_START:-0}"
MAX_SECONDS="${PZR_MAX_SECONDS:-43200}"
SKIP_HORIZON="${PZR_SKIP_HORIZON:-0}"
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

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/runs"

RUN_PROVENANCE="$(
    "$PYTHON" -c '
import hashlib
from pzr.rtlola.binding import (
    BINDING_BUILD_PROFILE,
    BINDING_REVISION,
    INTERPRETER_REVISION,
)
from pzr.rtlola.actions import default_action_catalog
from pzr.rtlola.scenarios import scenario_by_name

scenario = scenario_by_name("omni_robot")
mpc_candidates = ",".join(default_action_catalog().mpc_candidate_names)
print(
    ";".join(
        (
            f"binding={BINDING_REVISION}",
            f"interpreter={INTERPRETER_REVISION}",
            f"profile={BINDING_BUILD_PROFILE}",
            f"spec={hashlib.sha256(scenario.spec.encode()).hexdigest()}",
            f"source={scenario.source_revision}",
            f"mpc_candidates={mpc_candidates}",
        )
    )
)
'
)"
if [[ -z "$RUN_PROVENANCE" ]]; then
    echo "failed to determine RTLola run provenance" >&2
    exit 1
fi

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
    local fingerprint
    fingerprint="$(
        {
            printf '%s\0' "$RUN_PROVENANCE"
            printf '%s\0' "$@"
        } | sha256sum | awk '{print $1}'
    )"
    if [[ -f "$marker" ]] && [[ "$(cat "$marker")" == "$fingerprint" ]]; then
        echo "skip completed stage: $stage"
        return 0
    fi
    if [[ -f "$marker" ]]; then
        echo "rerun stale stage: $stage"
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
    printf '%s\n' "$fingerprint" >"$marker.tmp"
    mv "$marker.tmp" "$marker"
    echo "complete stage: $stage"
}

IFS=',' read -r -a budget_values <<< "$BUDGETS"
IFS=',' read -r -a trace_values <<< "$TRACES"
IFS=',' read -r -a horizon_values <<< "$HORIZON_SCAN"

for trace_kind in "${trace_values[@]}"; do
    for budget in "${budget_values[@]}"; do
        stage="primary_${trace_kind}_budget_${budget}"
        cell_dir="$OUT_DIR/runs/$trace_kind/budget_$budget"
        run_stage "$stage" "$PYTHON" -m pzr.rtlola.cli \
            --profile paper \
            --scenario omni_robot \
            --trace-kind "$trace_kind" \
            --length "$LENGTH" \
            --seeds "$SEEDS" \
            --budget "$budget" \
            --horizon "$HORIZON" \
            --beam-width "$BEAM_WIDTH" \
            --methods "$METHODS" \
            --reference-mode exact \
            --no-progress \
            --output "$cell_dir" || exit $?
    done
done

if [[ "$SKIP_HORIZON" != "1" ]]; then
    for horizon in "${horizon_values[@]}"; do
        horizon_root="$OUT_DIR/horizon_scan/h$horizon"
        for trace_kind in "${trace_values[@]}"; do
            stage="horizon_${horizon}_${trace_kind}_budget_${HORIZON_BUDGET}"
            cell_dir="$horizon_root/runs/$trace_kind/budget_$HORIZON_BUDGET"
            run_stage "$stage" "$PYTHON" -m pzr.rtlola.cli \
                --profile paper \
                --scenario omni_robot \
                --trace-kind "$trace_kind" \
                --length "$LENGTH" \
                --seeds "$SEEDS" \
                --budget "$HORIZON_BUDGET" \
                --horizon "$horizon" \
                --beam-width "$BEAM_WIDTH" \
                --methods mpc_beam \
                --reference-mode exact \
                --no-progress \
                --output "$cell_dir" || exit $?
        done
        "$PYTHON" -m pzr.rtlola.sweep_report \
            --root "$horizon_root" \
            --scenario omni_robot
    done
fi

if [[ "$SKIP_LEARNING" != "1" ]]; then
    first_budget="${budget_values[0]}"
    first_trace="${trace_values[0]}"
    run_stage "native_pooled_learning" "$PYTHON" -m pzr.rtlola.cli \
        --profile paper \
        --scenario omni_robot \
        --trace-kind "$first_trace" \
        --length "$LENGTH" \
        --seeds 1 \
        --budget "$first_budget" \
        --horizon "$HORIZON" \
        --beam-width "$BEAM_WIDTH" \
        --methods girard \
        --reference-mode exact \
        --learned-mode regret \
        --regret-iterations "$REGRET_ITERATIONS" \
        --regret-epochs "$REGRET_EPOCHS" \
        --regret-train-seeds "$REGRET_TRAIN_SEEDS" \
        --regret-eval-seeds "$REGRET_EVAL_SEEDS" \
        --regret-train-seed-start "$REGRET_TRAIN_SEED_START" \
        --regret-eval-seed-start "$REGRET_EVAL_SEED_START" \
        --regret-budgets "$BUDGETS" \
        --regret-train-traces "$TRACES" \
        --regret-eval-traces "$TRACES" \
        --no-progress \
        --output "$OUT_DIR/learning_stage" || exit $?
fi

"$PYTHON" -m pzr.rtlola.sweep_report \
    --root "$OUT_DIR" \
    --scenario omni_robot
echo "RTLola Omni fidelity pilot complete: $OUT_DIR"
