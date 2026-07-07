#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${PZR_OUT_DIR:-$ROOT_DIR/results/rtlola-arm-mpc-variants}"
PYTHON="${PZR_PYTHON:-$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm/bin/python}"
ENV_PREFIX="${PZR_ENV_PREFIX:-$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm}"
TRACES="${PZR_EVAL_TRACES:-figure8,figure8_drift}"
BUDGETS="${PZR_BUDGETS:-80,120}"
TAIL_HORIZONS="${PZR_TAIL_HORIZONS:-0,4,8,16}"
LENGTH="${PZR_LENGTH:-300}"
HORIZON="${PZR_HORIZON:-4}"
BEAM_WIDTH="${PZR_BEAM_WIDTH:-4}"
ROOT_BEAM_WIDTH="${PZR_ROOT_BEAM_WIDTH:-1}"
METHODS="${PZR_METHODS:-girard,scott,mpc_terminal_beam,mpc_terminal_girard_tail,mpc_cumulative_girard_tail,mpc_one_step_girard_rollout}"
MAX_SECONDS="${PZR_MAX_SECONDS:-86400}"
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
from pzr.rtlola.actions import default_action_catalog
from pzr.rtlola.binding import (
    BINDING_BUILD_PROFILE,
    BINDING_REVISION,
    INTERPRETER_REVISION,
)
from pzr.rtlola.scenarios import scenario_by_name

scenario = scenario_by_name("robot_arm")
mpc_candidates = ",".join(default_action_catalog().mpc_candidate_names)
print(";".join((
    f"binding={BINDING_REVISION}",
    f"interpreter={INTERPRETER_REVISION}",
    f"profile={BINDING_BUILD_PROFILE}",
    f"spec={hashlib.sha256(scenario.spec.encode()).hexdigest()}",
    f"source={scenario.source_revision}",
    f"mpc_candidates={mpc_candidates}",
)))
'
)"
if [[ -z "$RUN_PROVENANCE" ]]; then
    echo "failed to determine RTLola run provenance" >&2
    exit 1
fi
RUN_PROVENANCE="$RUN_PROVENANCE;pzr_source=$PZR_SOURCE_SHA256"

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

IFS=',' read -r -a trace_values <<< "$TRACES"
IFS=',' read -r -a budget_values <<< "$BUDGETS"
IFS=',' read -r -a tail_values <<< "$TAIL_HORIZONS"

for trace_kind in "${trace_values[@]}"; do
    for budget in "${budget_values[@]}"; do
        for tail_horizon in "${tail_values[@]}"; do
            stage="${trace_kind}_budget_${budget}_tail_${tail_horizon}"
            cell_dir="$OUT_DIR/runs/$trace_kind/budget_$budget/tail_$tail_horizon"
            run_stage "$stage" "$PYTHON" -m pzr.rtlola.cli \
                --profile paper \
                --scenario robot_arm \
                --trace-kind "$trace_kind" \
                --length "$LENGTH" \
                --seeds 1 \
                --budget "$budget" \
                --horizon "$HORIZON" \
                --beam-width "$BEAM_WIDTH" \
                --mpc-tail-horizon "$tail_horizon" \
                --mpc-root-beam-width "$ROOT_BEAM_WIDTH" \
                --methods "$METHODS" \
                --reference-mode exact \
                --no-progress \
                --output "$cell_dir" || exit $?
        done
    done
done

"$PYTHON" -m pzr.rtlola.sweep_report --root "$OUT_DIR"
echo "RTLola MPC variant study complete: $OUT_DIR"
