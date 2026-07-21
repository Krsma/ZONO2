#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRIMARY_DIR="${PZR_PRIMARY_DIR:-$ROOT_DIR/results/rtlola-learning-pairwise-ranking-policy-v2-01c92a2-2724b05-2257d07}"
OUT_DIR="${PZR_OUT_DIR:-$ROOT_DIR/results/rtlola-learning-bounded-exploration-v1-01c92a2-2724b05-2257d07}"
TRACE_STORE="${PZR_EXTRA_TRACE_STORE:-$OUT_DIR/traces-extra-clean16}"
PYTHON="${PZR_PYTHON:-$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm/bin/python}"
ENV_PREFIX="${PZR_ENV_PREFIX:-$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm}"
EVENT_COUNT="${PZR_EVENT_COUNT:-500}"
BUDGETS="${PZR_BUDGETS:-40,80,120,180}"
CANDIDATES="${PZR_CANDIDATES:-girard,scott,pca,combastel}"
CONDITIONS="${PZR_CONDITIONS:-random_waypoint}"
SCREEN_TRACE_KINDS="${PZR_SCREEN_TRACE_KINDS:-figure8,figure8_drift,figure8_geofence,figure8_drift_geofence,random,random_drift,random_geofence,random_drift_geofence,square,square_drift,square_geofence,square_drift_geofence}"
FULL_TRACE_KINDS="${PZR_FULL_TRACE_KINDS:-figure8,figure8_drift,figure8_geofence,figure8_drift_geofence,random,random_drift,random_geofence,random_drift_geofence,square,square_drift,square_geofence,square_drift_geofence}"
EPOCHS="${PZR_EPOCHS:-100}"
BATCH_SIZE="${PZR_BATCH_SIZE:-256}"
PATIENCE="${PZR_PATIENCE:-10}"
TRAINING_SEED="${PZR_TRAINING_SEED:-42}"
DISTURBANCE_SEED="${PZR_DISTURBANCE_SEED:-20260717}"
DART_REGRET_CAP_QUANTILE="${PZR_DART_REGRET_CAP_QUANTILE:-0.9}"
DART_DIRECTION_PSEUDOCOUNT="${PZR_DART_DIRECTION_PSEUDOCOUNT:-1.0}"
DART_RECOVERY_DECISIONS="${PZR_DART_RECOVERY_DECISIONS:-1}"
EVAL_LENGTH="${PZR_EVAL_LENGTH:-}"
COLLECTION_WORKERS="${PZR_COLLECTION_WORKERS:-10}"
EVALUATION_WORKERS="${PZR_EVALUATION_WORKERS:-4}"
EXTRA_SEED_START="${PZR_EXTRA_SEED_START:-26}"
EXTRA_TRAIN_SEEDS="${PZR_EXTRA_TRAIN_SEEDS:-16}"

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

if [[ ! -f "$PRIMARY_DIR/model-pairwise-ranking-policy/model.json" ]]; then
    echo "Missing frozen primary Pairwise Ranking Policy model: $PRIMARY_DIR/model-pairwise-ranking-policy" >&2
    exit 1
fi
if [[ ! -f "$PRIMARY_DIR/clean/dataset/manifest.json" ]]; then
    echo "Missing primary clean dataset: $PRIMARY_DIR/clean/dataset" >&2
    exit 1
fi

run_logged generate_extra_traces \
    "$PYTHON" -m pzr.learning.cli generate \
    --output "$TRACE_STORE" \
    --event-count "$EVENT_COUNT" \
    --conditions "$CONDITIONS" \
    --seed-start "$EXTRA_SEED_START" \
    --seed-count "$EXTRA_TRAIN_SEEDS"

run_logged collect_extra_clean \
    "$PYTHON" -m pzr.learning.cli collect \
    --output "$OUT_DIR/extra-clean" \
    --trace-store "$TRACE_STORE" \
    --budgets "$BUDGETS" \
    --candidates "$CANDIDATES" \
    --train-seeds "$EXTRA_TRAIN_SEEDS" \
    --validation-seeds 0 \
    --test-seeds 0 \
    --workers "$COLLECTION_WORKERS" \
    --seed-start "$EXTRA_SEED_START" \
    --collection-mode teacher

run_logged calibrate_dart \
    "$PYTHON" -m pzr.learning.cli calibrate-dart \
    --model "$PRIMARY_DIR/model-pairwise-ranking-policy" \
    --dataset "clean=$PRIMARY_DIR/clean/dataset" \
    --split validation \
    --regret-cap-quantile "$DART_REGRET_CAP_QUANTILE" \
    --direction-pseudocount "$DART_DIRECTION_PSEUDOCOUNT" \
    --recovery-decisions "$DART_RECOVERY_DECISIONS" \
    --output "$OUT_DIR/dart-calibration"

run_logged collect_extra_dart \
    "$PYTHON" -m pzr.learning.cli collect \
    --output "$OUT_DIR/extra-dart" \
    --trace-store "$TRACE_STORE" \
    --budgets "$BUDGETS" \
    --candidates "$CANDIDATES" \
    --train-seeds "$EXTRA_TRAIN_SEEDS" \
    --validation-seeds 0 \
    --test-seeds 0 \
    --workers "$COLLECTION_WORKERS" \
    --seed-start "$EXTRA_SEED_START" \
    --collection-mode dart \
    --dart-calibration "$OUT_DIR/dart-calibration" \
    --disturbance-seed "$DISTURBANCE_SEED"

run_logged train_pairwise_ranking_policy_clean36 \
    "$PYTHON" -m pzr.learning.cli train \
    --dataset "primary_clean20=$PRIMARY_DIR/clean/dataset" \
    --dataset "extra_clean16=$OUT_DIR/extra-clean/dataset" \
    --output "$OUT_DIR/model-pairwise-ranking-policy-clean36" \
    --objective pairwise \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --patience "$PATIENCE" \
    --seed "$TRAINING_SEED"

run_logged train_pairwise_ranking_policy_dart36 \
    "$PYTHON" -m pzr.learning.cli train \
    --dataset "primary_clean20=$PRIMARY_DIR/clean/dataset" \
    --dataset "extra_dart16=$OUT_DIR/extra-dart/dataset" \
    --output "$OUT_DIR/model-pairwise-ranking-policy-dart36" \
    --objective pairwise \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --patience "$PATIENCE" \
    --seed "$TRAINING_SEED"

run_logged train_expected_regret_clean20 \
    "$PYTHON" -m pzr.learning.cli train \
    --dataset "primary_clean20=$PRIMARY_DIR/clean/dataset" \
    --output "$OUT_DIR/model-expected-regret-clean20" \
    --objective expected-regret \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --patience "$PATIENCE" \
    --seed "$TRAINING_SEED"

declare -a length_args=()
if [[ -n "$EVAL_LENGTH" ]]; then
    length_args=(--length "$EVAL_LENGTH")
fi

run_logged evaluate_screen \
    "$PYTHON" -m pzr.learning.cli evaluate \
    --model "pairwise_ranking_policy_clean20=$PRIMARY_DIR/model-pairwise-ranking-policy" \
    --model "pairwise_ranking_policy_clean36=$OUT_DIR/model-pairwise-ranking-policy-clean36" \
    --model "pairwise_ranking_policy_dart36=$OUT_DIR/model-pairwise-ranking-policy-dart36" \
    --model "expected_regret_clean20=$OUT_DIR/model-expected-regret-clean20" \
    --comparison "data_scale=pairwise_ranking_policy_clean36:pairwise_ranking_policy_clean20" \
    --comparison "dart_effect=pairwise_ranking_policy_dart36:pairwise_ranking_policy_clean36" \
    --comparison "objective=expected_regret_clean20:pairwise_ranking_policy_clean20" \
    --output "$OUT_DIR/screen-evaluation" \
    --budgets "$BUDGETS" \
    --candidates "$CANDIDATES" \
    --trace-kinds "$SCREEN_TRACE_KINDS" \
    --benchmark-methods girard \
    --expected-cell-count 240 \
    --workers "$EVALUATION_WORKERS" \
    "${length_args[@]}"

run_logged select_challenger \
    "$PYTHON" -m pzr.learning.cli select-challenger \
    --evaluation "$OUT_DIR/screen-evaluation" \
    --model "pairwise_ranking_policy_clean20=$PRIMARY_DIR/model-pairwise-ranking-policy" \
    --model "pairwise_ranking_policy_clean36=$OUT_DIR/model-pairwise-ranking-policy-clean36" \
    --model "pairwise_ranking_policy_dart36=$OUT_DIR/model-pairwise-ranking-policy-dart36" \
    --model "expected_regret_clean20=$OUT_DIR/model-expected-regret-clean20" \
    --output "$OUT_DIR/selection" \
    --expected-cell-count 240

WINNER="$("$PYTHON" -c 'import json, sys; value=json.load(open(sys.argv[1]))["winner"]; print("" if value is None else value["challenger"])' "$OUT_DIR/selection/selection.json")"
if [[ -z "$WINNER" ]]; then
    echo "Bounded exploration complete with no promoted challenger: $OUT_DIR"
    exit 0
fi
REFERENCE="$("$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["winner"]["reference"])' "$OUT_DIR/selection/selection.json")"

model_path() {
    case "$1" in
        pairwise_ranking_policy_clean20) echo "$PRIMARY_DIR/model-pairwise-ranking-policy" ;;
        pairwise_ranking_policy_clean36) echo "$OUT_DIR/model-pairwise-ranking-policy-clean36" ;;
        pairwise_ranking_policy_dart36) echo "$OUT_DIR/model-pairwise-ranking-policy-dart36" ;;
        expected_regret_clean20) echo "$OUT_DIR/model-expected-regret-clean20" ;;
        *) echo "Unknown selected model: $1" >&2; return 1 ;;
    esac
}

WINNER_PATH="$(model_path "$WINNER")"
REFERENCE_PATH="$(model_path "$REFERENCE")"
run_logged evaluate_promoted \
    "$PYTHON" -m pzr.learning.cli evaluate \
    --model "$WINNER=$WINNER_PATH" \
    --model "$REFERENCE=$REFERENCE_PATH" \
    --comparison "promotion=$WINNER:$REFERENCE" \
    --output "$OUT_DIR/promoted-evaluation" \
    --budgets "$BUDGETS" \
    --candidates "$CANDIDATES" \
    --trace-kinds "$FULL_TRACE_KINDS" \
    --benchmark-methods girard \
    --expected-cell-count 144 \
    --workers "$EVALUATION_WORKERS" \
    "${length_args[@]}"

echo "Bounded exploration promoted $WINNER against $REFERENCE: $OUT_DIR"
