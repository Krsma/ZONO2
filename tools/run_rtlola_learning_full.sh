#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${PZR_OUT_DIR:-$ROOT_DIR/results/rtlola-learning-geometry15-drift4cm-7371495-b4cfbf4-e6ecd0b}"
PYTHON="${PZR_PYTHON:-$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm/bin/python}"
ENV_PREFIX="${PZR_ENV_PREFIX:-$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm}"
EVENT_COUNT="${PZR_EVENT_COUNT:-2000}"
BUDGETS="${PZR_BUDGETS:-40,80,120,180}"
CANDIDATES="${PZR_CANDIDATES:-girard,scott,pca,combastel}"
CONDITIONS="${PZR_CONDITIONS:-random_waypoint,random_waypoint_drift,random_waypoint_geofence,random_waypoint_drift_geofence}"
WAYPOINT_DRIFT_Z="${PZR_WAYPOINT_DRIFT_Z:-0.04}"
TRACE_KINDS="${PZR_TRACE_KINDS:-figure8,figure8_drift,random,random_drift,square,square_drift}"
EPOCHS="${PZR_EPOCHS:-100}"
BATCH_SIZE="${PZR_BATCH_SIZE:-256}"
PATIENCE="${PZR_PATIENCE:-10}"
TRAINING_SEED="${PZR_TRAINING_SEED:-42}"
EVAL_LENGTH="${PZR_EVAL_LENGTH:-}"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUT_DIR/matplotlib-cache}"
if [[ -f "$ENV_PREFIX/lib/libopenblas.so" ]]; then
    export LD_PRELOAD="$ENV_PREFIX/lib/libopenblas.so${LD_PRELOAD:+:$LD_PRELOAD}"
fi

mkdir -p "$OUT_DIR/logs" "$MPLCONFIGDIR"

run_logged() {
    local name="$1"
    shift
    echo "start stage: $name"
    "$@" 2>&1 | tee "$OUT_DIR/logs/$name.log"
    echo "complete stage: $name"
}

run_logged collect_base \
    "$PYTHON" -m pzr.learning.cli collect \
    --output "$OUT_DIR/base" \
    --event-count "$EVENT_COUNT" \
    --budgets "$BUDGETS" \
    --candidates "$CANDIDATES" \
    --conditions "$CONDITIONS" \
    --waypoint-drift-z "$WAYPOINT_DRIFT_Z" \
    --train-seeds 3 \
    --validation-seeds 1 \
    --test-seeds 0 \
    --seed-start 0

run_logged train_base \
    "$PYTHON" -m pzr.learning.cli train \
    --dataset "$OUT_DIR/base/dataset" \
    --output "$OUT_DIR/model-base" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --patience "$PATIENCE" \
    --seed "$TRAINING_SEED"

run_logged collect_dagger_1 \
    "$PYTHON" -m pzr.learning.cli collect \
    --output "$OUT_DIR/dagger-1" \
    --event-count "$EVENT_COUNT" \
    --budgets "$BUDGETS" \
    --candidates "$CANDIDATES" \
    --conditions "$CONDITIONS" \
    --waypoint-drift-z "$WAYPOINT_DRIFT_Z" \
    --train-seeds 3 \
    --validation-seeds 0 \
    --test-seeds 0 \
    --seed-start 4 \
    --behavior-model "$OUT_DIR/model-base"

run_logged train_dagger_1 \
    "$PYTHON" -m pzr.learning.cli train \
    --dataset "$OUT_DIR/base/dataset" \
    --dataset "$OUT_DIR/dagger-1/dataset" \
    --output "$OUT_DIR/model-dagger-1" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --patience "$PATIENCE" \
    --seed "$TRAINING_SEED"

run_logged collect_dagger_2 \
    "$PYTHON" -m pzr.learning.cli collect \
    --output "$OUT_DIR/dagger-2" \
    --event-count "$EVENT_COUNT" \
    --budgets "$BUDGETS" \
    --candidates "$CANDIDATES" \
    --conditions "$CONDITIONS" \
    --waypoint-drift-z "$WAYPOINT_DRIFT_Z" \
    --train-seeds 3 \
    --validation-seeds 0 \
    --test-seeds 0 \
    --seed-start 7 \
    --behavior-model "$OUT_DIR/model-dagger-1"

run_logged train_final \
    "$PYTHON" -m pzr.learning.cli train \
    --dataset "$OUT_DIR/base/dataset" \
    --dataset "$OUT_DIR/dagger-1/dataset" \
    --dataset "$OUT_DIR/dagger-2/dataset" \
    --output "$OUT_DIR/model-final" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --patience "$PATIENCE" \
    --seed "$TRAINING_SEED"

declare -a length_args=()
if [[ -n "$EVAL_LENGTH" ]]; then
    length_args=(--length "$EVAL_LENGTH")
fi
run_logged evaluate_fixed \
    "$PYTHON" -m pzr.learning.cli evaluate \
    --model "$OUT_DIR/model-final" \
    --model-name learned_geometry15 \
    --output "$OUT_DIR/evaluation" \
    --budgets "$BUDGETS" \
    --candidates "$CANDIDATES" \
    --trace-kinds "$TRACE_KINDS" \
    --baselines girard,mpc_terminal_full_width \
    "${length_args[@]}"

echo "RTLola geometry15 learning experiment complete: $OUT_DIR"
