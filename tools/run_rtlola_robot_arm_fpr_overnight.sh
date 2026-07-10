#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${PZR_OUT_DIR:-$ROOT_DIR/results/rtlola-arm-mpc-variants-b4cfbf4-e6ecd0b-exact-metrics}"
REFERENCE_DIR="${PZR_REFERENCE_DIR:-$OUT_DIR/references}"
PYTHON="${PZR_PYTHON:-$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm/bin/python}"
ENV_PREFIX="${PZR_ENV_PREFIX:-$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm}"
LENGTH_OVERRIDE="${PZR_LENGTH:-}"
BUDGETS="${PZR_BUDGETS:-40,80,120,180}"
EVAL_TRACES="${PZR_EVAL_TRACES:-figure8,figure8_drift,random,random_drift,square,square_drift}"
TRAIN_TRACES="${PZR_TRAIN_TRACES:-figure8,random,square}"
METHODS="${PZR_METHODS:-girard,scott,interval_hull,pca,combastel,mpc_terminal_beam,mpc_terminal_girard_tail,mpc_cumulative_girard_tail,mpc_one_step_girard_rollout}"
HORIZON="${PZR_HORIZON:-4}"
BEAM_WIDTH="${PZR_BEAM_WIDTH:-4}"
MPC_TAIL_HORIZON="${PZR_MPC_TAIL_HORIZON:-8}"
MPC_ROOT_BEAM_WIDTH="${PZR_MPC_ROOT_BEAM_WIDTH:-1}"
MPC_CANDIDATES="${PZR_MPC_CANDIDATES:-}"
SEEDS="${PZR_SEEDS:-1}"
REGRET_ITERATIONS="${PZR_REGRET_ITERATIONS:-1}"
REGRET_EPOCHS="${PZR_REGRET_EPOCHS:-50}"
REGRET_TRAIN_SEEDS="${PZR_REGRET_TRAIN_SEEDS:-1}"
REGRET_EVAL_SEEDS="${PZR_REGRET_EVAL_SEEDS:-1}"
MAX_SECONDS="${PZR_MAX_SECONDS:-259200}"
SKIP_LEARNING="${PZR_SKIP_LEARNING:-1}"
JOBS="${PZR_JOBS:-1}"
START_SECONDS="$(date +%s)"

if [[ ! "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "PZR_JOBS must be a positive integer, got: $JOBS" >&2
    exit 2
fi

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUT_DIR/matplotlib-cache}"
if [[ -f "$ENV_PREFIX/lib/libopenblas.so" ]]; then
    export LD_PRELOAD="$ENV_PREFIX/lib/libopenblas.so${LD_PRELOAD:+:$LD_PRELOAD}"
fi

mkdir -p "$OUT_DIR/logs" "$REFERENCE_DIR" "$OUT_DIR/runs"

PZR_SOURCE_SHA256="$(
    find "$ROOT_DIR/src/pzr/rtlola" -type f \
        \( -name '*.py' -o -name '*.lola' \) -print0 \
        | sort -z \
        | xargs -0 sha256sum \
        | sha256sum \
        | awk '{print $1}'
)"
RUN_PROVENANCE="$(
    "$PYTHON" -c '
import hashlib
import sys
from pzr.rtlola.binding import (
    BINDING_BUILD_PROFILE,
    BINDING_REVISION,
    INTERPRETER_REVISION,
)
from pzr.rtlola.actions import default_action_catalog
from pzr.rtlola.scenarios import scenario_by_name

scenario = scenario_by_name("robot_arm")
requested = tuple(part for part in sys.argv[1].split(",") if part)
mpc_candidates = ",".join(
    default_action_catalog(requested or None).mpc_candidate_names
)
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
' "$MPC_CANDIDATES"
)"
if [[ -z "$RUN_PROVENANCE" ]]; then
    echo "failed to determine RTLola run provenance" >&2
    exit 1
fi
RUN_PROVENANCE="$RUN_PROVENANCE;pzr_source=$PZR_SOURCE_SHA256"
CALCULATED_REFERENCE_NAMESPACE="$(
    printf '%s' "$RUN_PROVENANCE" | sha256sum | cut -c1-16
)"
REFERENCE_NAMESPACE="${PZR_REFERENCE_NAMESPACE:-$CALCULATED_REFERENCE_NAMESPACE}"
if [[ ! "$REFERENCE_NAMESPACE" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "invalid PZR_REFERENCE_NAMESPACE: $REFERENCE_NAMESPACE" >&2
    exit 2
fi

remaining_seconds() {
    local elapsed
    elapsed=$(( $(date +%s) - START_SECONDS ))
    echo $(( MAX_SECONDS - elapsed ))
}

trace_length_for() {
    local trace_kind="$1"
    if [[ -n "$LENGTH_OVERRIDE" ]]; then
        printf '%s\n' "$LENGTH_OVERRIDE"
        return
    fi
    "$PYTHON" -c '
import sys
from pzr.rtlola.robot_arm import ROBOT_ARM_TRACE_ROWS

print(ROBOT_ARM_TRACE_ROWS[sys.argv[1]])
' "$trace_kind"
}

validate_stage_output() {
    local cell_dir="$1"
    local method="$2"
    local expected_length="$3"
    local expected_seeds="$4"
    "$PYTHON" -c '
import pathlib
import sys

import pandas as pd

cell = pathlib.Path(sys.argv[1]) / "robot_arm"
method = sys.argv[2]
length = int(sys.argv[3])
seeds = int(sys.argv[4])

def read_csv(name):
    path = cell / name
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

timeseries = read_csv("timeseries.csv")
summary = read_csv("summary.csv")
failures = read_csv("run_failures.csv")
required_summary_columns = {
    "method",
    "seed",
    "false_positive_count",
    "false_negative_count",
    "reference_positive_count",
    "reference_negative_count",
    "fpr",
    "fnr",
    "mean_approx_loss",
    "final_approx_loss",
    "max_approx_loss",
    "sum_approx_loss",
    "mean_state_width",
    "max_state_width",
    "total_time_ms",
}
for seed in range(seeds):
    completed = (
        not timeseries.empty
        and {"method", "seed"} <= set(timeseries.columns)
        and len(timeseries[
            (timeseries["method"] == method)
            & (timeseries["seed"] == seed)
        ]) == length
    )
    recorded_failure = (
        not failures.empty
        and {"method", "seed"} <= set(failures.columns)
        and bool((
            (failures["method"] == method)
            & (failures["seed"] == seed)
        ).any())
    )
    if not completed and not recorded_failure:
        raise SystemExit(
            f"incomplete artifact for method={method}, seed={seed}: "
            f"expected {length} rows or an explicit run failure"
        )
    if completed:
        if summary.empty or not required_summary_columns <= set(summary.columns):
            missing = sorted(required_summary_columns - set(summary.columns))
            raise SystemExit(
                f"summary schema is incomplete for method={method}, seed={seed}: "
                f"missing {missing}"
            )
        rows = summary[
            (summary["method"] == method)
            & (summary["seed"] == seed)
        ]
        if len(rows) != 1:
            raise SystemExit(
                f"expected one summary row for method={method}, seed={seed}, "
                f"got {len(rows)}"
            )
        native_metrics = rows[
            [
                "mean_approx_loss",
                "final_approx_loss",
                "max_approx_loss",
                "sum_approx_loss",
            ]
        ]
        if native_metrics.isna().to_numpy().any():
            raise SystemExit(
                f"exact-reference native loss is missing for "
                f"method={method}, seed={seed}"
            )
' "$cell_dir" "$method" "$expected_length" "$expected_seeds"
}

run_stage() {
    local stage="$1"
    local cell_dir="$2"
    local method="$3"
    local expected_length="$4"
    local expected_seeds="$5"
    shift 5
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
    if ! validate_stage_output \
        "$cell_dir" "$method" "$expected_length" "$expected_seeds"; then
        echo "stage produced incomplete artifacts: $stage (log $log)"
        return 1
    fi
    printf '%s\n' "$fingerprint" >"$marker.tmp"
    mv "$marker.tmp" "$marker"
    echo "complete stage: $stage"
}

run_reference_stage() {
    local stage="$1"
    local cache_path="$2"
    shift 2
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
    if [[ ! -s "$cache_path" ]]; then
        echo "reference stage produced no cache: $stage (log $log)"
        return 1
    fi
    printf '%s\n' "$fingerprint" >"$marker.tmp"
    mv "$marker.tmp" "$marker"
    echo "complete stage: $stage"
}

declare -a stage_pids=()
declare -a stage_names=()

wait_for_stage_batch() {
    local index
    local status
    local failed=0
    for index in "${!stage_pids[@]}"; do
        if wait "${stage_pids[$index]}"; then
            continue
        else
            status=$?
        fi
        echo "parallel stage failed: ${stage_names[$index]} (status $status)" >&2
        if (( failed == 0 )); then
            failed="$status"
        fi
    done
    stage_pids=()
    stage_names=()
    return "$failed"
}

IFS=',' read -r -a budget_values <<< "$BUDGETS"
IFS=',' read -r -a eval_trace_values <<< "$EVAL_TRACES"
IFS=',' read -r -a method_values <<< "$METHODS"
declare -a mpc_candidate_args=()
if [[ -n "$MPC_CANDIDATES" ]]; then
    mpc_candidate_args=(--mpc-candidates "$MPC_CANDIDATES")
fi
declare -A trace_lengths
for trace_kind in "${eval_trace_values[@]}"; do
    if ! trace_length="$(trace_length_for "$trace_kind")"; then
        echo "failed to resolve packaged length for trace: $trace_kind" >&2
        exit 1
    fi
    if [[ ! "$trace_length" =~ ^[1-9][0-9]*$ ]]; then
        echo "invalid length for trace $trace_kind: $trace_length" >&2
        exit 1
    fi
    trace_lengths["$trace_kind"]="$trace_length"
done

for trace_kind in "${eval_trace_values[@]}"; do
    trace_length="${trace_lengths[$trace_kind]}"
    reference_cache="$REFERENCE_DIR/${trace_kind}.seed_0.${REFERENCE_NAMESPACE}.json"
    expected_reference_cache="$reference_cache"
    if (( SEEDS > 1 )); then
        expected_reference_cache="${reference_cache%.json}.seed_0.json"
    fi
    run_reference_stage \
        "reference_${trace_kind}" "$expected_reference_cache" \
        "$PYTHON" -m pzr.rtlola.cli \
        --profile paper \
        --scenario robot_arm \
        --trace-kind "$trace_kind" \
        --length "$trace_length" \
        --seeds "$SEEDS" \
        --reference-mode exact \
        --reference-cache "$reference_cache" \
        --reference-only \
        "${mpc_candidate_args[@]}" \
        --no-progress \
        --output "$OUT_DIR/reference_stage/$trace_kind" || exit $?
done

for trace_kind in "${eval_trace_values[@]}"; do
    trace_length="${trace_lengths[$trace_kind]}"
    for budget in "${budget_values[@]}"; do
        for method in "${method_values[@]}"; do
            stage="${trace_kind}_budget_${budget}_${method}"
            cell_dir="$OUT_DIR/runs/$trace_kind/budget_$budget/$method"
            run_stage \
                "$stage" "$cell_dir" "$method" "$trace_length" "$SEEDS" \
                "$PYTHON" -m pzr.rtlola.cli \
                --profile paper \
                --scenario robot_arm \
                --trace-kind "$trace_kind" \
                --length "$trace_length" \
                --seeds "$SEEDS" \
                --budget "$budget" \
                --horizon "$HORIZON" \
                --beam-width "$BEAM_WIDTH" \
                --mpc-tail-horizon "$MPC_TAIL_HORIZON" \
                --mpc-root-beam-width "$MPC_ROOT_BEAM_WIDTH" \
                "${mpc_candidate_args[@]}" \
                --methods "$method" \
                --reference-mode exact \
                --reference-cache "$REFERENCE_DIR/${trace_kind}.seed_0.${REFERENCE_NAMESPACE}.json" \
                --no-progress \
                --output "$cell_dir" &
            stage_pids+=("$!")
            stage_names+=("$stage")
            if (( ${#stage_pids[@]} >= JOBS )); then
                wait_for_stage_batch || exit $?
            fi
        done
    done
done
wait_for_stage_batch || exit $?

if [[ "$SKIP_LEARNING" != "1" ]]; then
    first_budget="${budget_values[0]}"
    first_trace="${eval_trace_values[0]}"
    learning_dir="$OUT_DIR/learning_stage"
    run_stage \
        "pooled_learning" "$learning_dir" "girard" \
        "${trace_lengths[$first_trace]}" 1 \
        "$PYTHON" -m pzr.rtlola.cli \
        --profile paper \
        --scenario robot_arm \
        --trace-kind "$first_trace" \
        --length "${trace_lengths[$first_trace]}" \
        --seeds 1 \
        --budget "$first_budget" \
        --horizon "$HORIZON" \
        --beam-width "$BEAM_WIDTH" \
        --mpc-tail-horizon "$MPC_TAIL_HORIZON" \
        --mpc-root-beam-width "$MPC_ROOT_BEAM_WIDTH" \
        "${mpc_candidate_args[@]}" \
        --methods girard \
        --reference-mode exact \
        --reference-cache "$REFERENCE_DIR/${first_trace}.seed_0.${REFERENCE_NAMESPACE}.json" \
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
