#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRIMARY_DIR="${PZR_PRIMARY_DIR:-$ROOT_DIR/results/rtlola-learning-pairwise-ranking-policy-v2-01c92a2-2724b05-2257d07}"
OUT_DIR="${PZR_OUT_DIR:-$ROOT_DIR/results/rtlola-learning-online-mpc-addon-v1-01c92a2-2724b05-2257d07}"
MODEL_DIR="${PZR_MODEL_DIR:-$PRIMARY_DIR/model-pairwise-ranking-policy}"
PYTHON="${PZR_PYTHON:-$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm/bin/python}"
ENV_PREFIX="${PZR_ENV_PREFIX:-$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm}"
BUDGETS="${PZR_BUDGETS:-40,80,120,180}"
CANDIDATES="${PZR_CANDIDATES:-girard,scott,pca,combastel}"
TRACE_KINDS="${PZR_TRACE_KINDS:-figure8,figure8_drift,figure8_geofence,figure8_drift_geofence,random,random_drift,random_geofence,random_drift_geofence,square,square_drift,square_geofence,square_drift_geofence}"
EVAL_LENGTH="${PZR_EVAL_LENGTH:-}"
EVALUATION_WORKERS="${PZR_EVALUATION_WORKERS:-4}"
PREDICTION_STEP_SECONDS="${PZR_PREDICTION_STEP_SECONDS:-0.1}"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUT_DIR/matplotlib-cache}"
if [[ -f "$ENV_PREFIX/lib/libopenblas.so" ]]; then
    export LD_PRELOAD="$ENV_PREFIX/lib/libopenblas.so${LD_PRELOAD:+:$LD_PRELOAD}"
fi

if [[ ! -f "$MODEL_DIR/model.json" || ! -f "$MODEL_DIR/weights.pt" ]]; then
    echo "Frozen primary model is missing: $MODEL_DIR" >&2
    exit 1
fi

mkdir -p "$OUT_DIR/logs" "$MPLCONFIGDIR"
declare -a length_args=()
if [[ -n "$EVAL_LENGTH" ]]; then
    length_args=(--length "$EVAL_LENGTH")
fi

"$PYTHON" -m pzr.learning.cli evaluate \
    --model "pairwise_ranking_policy=$MODEL_DIR" \
    --output "$OUT_DIR/evaluation" \
    --budgets "$BUDGETS" \
    --candidates "$CANDIDATES" \
    --trace-kinds "$TRACE_KINDS" \
    --benchmark-methods girard,mpc_terminal_beam,mpc_cumulative_beam,mpc_terminal_full_width,mpc_terminal_beam_predictive_hold,mpc_terminal_beam_predictive_linear,mpc_terminal_beam_predictive_quadratic \
    --horizon 3 \
    --beam-width 4 \
    --prediction-step-seconds "$PREDICTION_STEP_SECONDS" \
    --expected-cell-count 384 \
    --workers "$EVALUATION_WORKERS" \
    "${length_args[@]}" \
    2>&1 | tee "$OUT_DIR/logs/evaluate_mpc_addon.log"

echo "RTLola online-MPC add-on complete: $OUT_DIR"
