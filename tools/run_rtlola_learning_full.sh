#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${PZR_OUT_DIR:-$ROOT_DIR/results/rtlola-learning-geometry15-random500-soft-dart-v3-7371495-b4cfbf4-e6ecd0b}"
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
DISTURBANCE_SEED="${PZR_DISTURBANCE_SEED:-20260717}"
SOFT_TEMPERATURES="${PZR_SOFT_TEMPERATURES:-0.05,0.1,0.2,0.5}"
FEASIBILITY_PENALTY="${PZR_FEASIBILITY_PENALTY:-1.0}"
EVAL_LENGTH="${PZR_EVAL_LENGTH:-}"
COLLECTION_WORKERS="${PZR_COLLECTION_WORKERS:-10}"
EVALUATION_WORKERS="${PZR_EVALUATION_WORKERS:-4}"
SEED_START="${PZR_SEED_START:-0}"
CLEAN_TRAIN_SEEDS="${PZR_CLEAN_TRAIN_SEEDS:-20}"
CLEAN_VALIDATION_SEEDS="${PZR_CLEAN_VALIDATION_SEEDS:-6}"
DART_TRAIN_SEEDS="${PZR_DART_TRAIN_SEEDS:-16}"
DART_VALIDATION_SEEDS="${PZR_DART_VALIDATION_SEEDS:-6}"
CLEAN_SEED_COUNT=$((CLEAN_TRAIN_SEEDS + CLEAN_VALIDATION_SEEDS))
DART_SEED_START=$((SEED_START + CLEAN_SEED_COUNT))
DART_SEED_COUNT=$((DART_TRAIN_SEEDS + DART_VALIDATION_SEEDS))
TRACE_SEED_COUNT=$((CLEAN_SEED_COUNT + DART_SEED_COUNT))

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

run_logged train_pairwise_clean \
    "$PYTHON" -m pzr.learning.cli train \
    --dataset "clean=$OUT_DIR/clean/dataset" \
    --output "$OUT_DIR/model-pairwise-clean" \
    --objective pairwise \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --patience "$PATIENCE" \
    --seed "$TRAINING_SEED"

run_logged train_soft_clean \
    "$PYTHON" -m pzr.learning.cli train \
    --dataset "clean=$OUT_DIR/clean/dataset" \
    --output "$OUT_DIR/model-soft-clean" \
    --objective soft-kl \
    --temperature-grid "$SOFT_TEMPERATURES" \
    --feasibility-penalty "$FEASIBILITY_PENALTY" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --patience "$PATIENCE" \
    --seed "$TRAINING_SEED"

run_logged calibrate_dart \
    "$PYTHON" -m pzr.learning.cli calibrate-dart \
    --model "$OUT_DIR/model-soft-clean" \
    --dataset "clean=$OUT_DIR/clean/dataset" \
    --split validation \
    --output "$OUT_DIR/dart-calibration"

run_logged collect_dart \
    "$PYTHON" -m pzr.learning.cli collect \
    --output "$OUT_DIR/dart" \
    --trace-store "$TRACE_STORE" \
    --budgets "$BUDGETS" \
    --candidates "$CANDIDATES" \
    --train-seeds "$DART_TRAIN_SEEDS" \
    --validation-seeds "$DART_VALIDATION_SEEDS" \
    --test-seeds 0 \
    --workers "$COLLECTION_WORKERS" \
    --seed-start "$DART_SEED_START" \
    --collection-mode dart \
    --dart-calibration "$OUT_DIR/dart-calibration" \
    --disturbance-seed "$DISTURBANCE_SEED"

run_logged train_soft_dart \
    "$PYTHON" -m pzr.learning.cli train \
    --dataset "clean=$OUT_DIR/clean/dataset" \
    --dataset "dart=$OUT_DIR/dart/dataset" \
    --output "$OUT_DIR/model-soft-dart" \
    --objective soft-kl \
    --temperature-from "$OUT_DIR/model-soft-clean" \
    --feasibility-penalty "$FEASIBILITY_PENALTY" \
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
    --model "learned_pairwise_clean=$OUT_DIR/model-pairwise-clean" \
    --model "learned_soft_clean=$OUT_DIR/model-soft-clean" \
    --model "learned_soft_dart=$OUT_DIR/model-soft-dart" \
    --output "$OUT_DIR/evaluation" \
    --budgets "$BUDGETS" \
    --candidates "$CANDIDATES" \
    --trace-kinds "$TRACE_KINDS" \
    --baselines girard,scott,pca,combastel,mpc_terminal_full_width \
    --workers "$EVALUATION_WORKERS" \
    "${length_args[@]}"

echo "RTLola Geometry15 soft-DART experiment complete: $OUT_DIR"
