#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${PZR_OUT_DIR:-$ROOT_DIR/results/rtlola-learning-geometry15-random500-7371495-b4cfbf4-e6ecd0b}"
TRACE_STORE="${PZR_TRACE_STORE:-$OUT_DIR/traces}"
PYTHON="${PZR_PYTHON:-$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm/bin/python}"
ENV_PREFIX="${PZR_ENV_PREFIX:-$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm}"
EVENT_COUNT="${PZR_EVENT_COUNT:-500}"
BUDGETS="${PZR_BUDGETS:-40,80,120,180}"
CANDIDATES="${PZR_CANDIDATES:-girard,scott,pca,combastel}"
CONDITIONS="${PZR_CONDITIONS:-random_waypoint}"
TRACE_KINDS="${PZR_TRACE_KINDS:-figure8,figure8_drift,random,random_drift,square,square_drift}"
EPOCHS="${PZR_EPOCHS:-100}"
BATCH_SIZE="${PZR_BATCH_SIZE:-256}"
PATIENCE="${PZR_PATIENCE:-10}"
TRAINING_SEED="${PZR_TRAINING_SEED:-42}"
EVAL_LENGTH="${PZR_EVAL_LENGTH:-}"
WORKERS="${PZR_WORKERS:-8}"
SEED_START="${PZR_SEED_START:-0}"
BASE_TRAIN_SEEDS="${PZR_BASE_TRAIN_SEEDS:-12}"
BASE_VALIDATION_SEEDS="${PZR_BASE_VALIDATION_SEEDS:-4}"
DAGGER_SEEDS="${PZR_DAGGER_SEEDS:-12}"
DAGGER_1_SEED_START=$((SEED_START + BASE_TRAIN_SEEDS + BASE_VALIDATION_SEEDS))
DAGGER_2_SEED_START=$((DAGGER_1_SEED_START + DAGGER_SEEDS))
TRACE_SEED_COUNT=$((BASE_TRAIN_SEEDS + BASE_VALIDATION_SEEDS + 2 * DAGGER_SEEDS))

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

run_logged generate_traces \
    "$PYTHON" -m pzr.learning.cli generate \
    --output "$TRACE_STORE" \
    --event-count "$EVENT_COUNT" \
    --conditions "$CONDITIONS" \
    --seed-start "$SEED_START" \
    --seed-count "$TRACE_SEED_COUNT"

run_logged collect_base \
    "$PYTHON" -m pzr.learning.cli collect \
    --output "$OUT_DIR/base" \
    --trace-store "$TRACE_STORE" \
    --budgets "$BUDGETS" \
    --candidates "$CANDIDATES" \
    --train-seeds "$BASE_TRAIN_SEEDS" \
    --validation-seeds "$BASE_VALIDATION_SEEDS" \
    --test-seeds 0 \
    --workers "$WORKERS" \
    --seed-start "$SEED_START"

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
    --trace-store "$TRACE_STORE" \
    --budgets "$BUDGETS" \
    --candidates "$CANDIDATES" \
    --train-seeds "$DAGGER_SEEDS" \
    --validation-seeds 0 \
    --test-seeds 0 \
    --workers "$WORKERS" \
    --seed-start "$DAGGER_1_SEED_START" \
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
    --trace-store "$TRACE_STORE" \
    --budgets "$BUDGETS" \
    --candidates "$CANDIDATES" \
    --train-seeds "$DAGGER_SEEDS" \
    --validation-seeds 0 \
    --test-seeds 0 \
    --workers "$WORKERS" \
    --seed-start "$DAGGER_2_SEED_START" \
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
    --workers "$WORKERS" \
    "${length_args[@]}"

echo "RTLola geometry15 learning experiment complete: $OUT_DIR"
