#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${PZR_OUT_DIR:-$ROOT_DIR/results/rtlola-learning-pairwise-ranking-policy-v2-7371495-b4cfbf4-e6ecd0b}"
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
COLLECTION_WORKERS="${PZR_COLLECTION_WORKERS:-10}"
EVALUATION_WORKERS="${PZR_EVALUATION_WORKERS:-4}"
SEED_START="${PZR_SEED_START:-0}"
CLEAN_TRAIN_SEEDS="${PZR_CLEAN_TRAIN_SEEDS:-20}"
CLEAN_VALIDATION_SEEDS="${PZR_CLEAN_VALIDATION_SEEDS:-6}"
CLEAN_SEED_COUNT=$((CLEAN_TRAIN_SEEDS + CLEAN_VALIDATION_SEEDS))

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
    --seed-count "$CLEAN_SEED_COUNT"

run_logged collect_clean \
    "$PYTHON" -m pzr.learning.cli collect \
    --output "$OUT_DIR/clean" \
    --trace-store "$TRACE_STORE" \
    --budgets "$BUDGETS" \
    --candidates "$CANDIDATES" \
    --train-seeds "$CLEAN_TRAIN_SEEDS" \
    --validation-seeds "$CLEAN_VALIDATION_SEEDS" \
    --test-seeds 0 \
    --workers "$COLLECTION_WORKERS" \
    --seed-start "$SEED_START" \
    --collection-mode teacher

run_logged train_pairwise_ranking_policy \
    "$PYTHON" -m pzr.learning.cli train \
    --dataset "clean=$OUT_DIR/clean/dataset" \
    --output "$OUT_DIR/model-pairwise-ranking-policy" \
    --objective pairwise \
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
    --model "pairwise_ranking_policy=$OUT_DIR/model-pairwise-ranking-policy" \
    --output "$OUT_DIR/evaluation" \
    --budgets "$BUDGETS" \
    --candidates "$CANDIDATES" \
    --trace-kinds "$TRACE_KINDS" \
    --benchmark-methods girard,scott,pca,combastel,mpc_terminal_full_width \
    --horizon 1 \
    --expected-cell-count 144 \
    --workers "$EVALUATION_WORKERS" \
    "${length_args[@]}"

echo "RTLola Pairwise Ranking Policy primary experiment complete: $OUT_DIR"
