#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="corl-main-realistic-$(date +%Y%m%d-%H%M%S)"
OUT_DIR="$ROOT_DIR/results/$RUN_ID"
LOG_PATH="$ROOT_DIR/results/logs/$RUN_ID.log"

export PZR_SAFE_CONTROL_GYM_ROOT="${PZR_SAFE_CONTROL_GYM_ROOT:-$ROOT_DIR/external/safe-control-gym}"
export PZR_SAFE_CONTROL_PYTHON="${PZR_SAFE_CONTROL_PYTHON:-$ROOT_DIR/external/miniconda3/envs/pzr-safe-control-fw/bin/python}"
export PZR_SAFE_CONTROL_CONFIG="${PZR_SAFE_CONTROL_CONFIG:-competition/level1.yaml}"
PZR_CORL_PROFILE="${PZR_CORL_PROFILE:-overnight}"
PZR_CORL_METHOD_SET="${PZR_CORL_METHOD_SET:-core}"
PZR_CORL_LEARNED_MODE="${PZR_CORL_LEARNED_MODE:-none}"
PZR_CORL_BUDGET="${PZR_CORL_BUDGET:-8}"
PZR_CORL_HORIZON="${PZR_CORL_HORIZON:-6}"
PZR_CORL_MAX_STEPS="${PZR_CORL_MAX_STEPS:-1000}"
PZR_CORL_TRAIN_SEEDS="${PZR_CORL_TRAIN_SEEDS:-20}"
PZR_CORL_EVAL_SEEDS="${PZR_CORL_EVAL_SEEDS:-50}"
PZR_CORL_DAGGER_ITERATIONS="${PZR_CORL_DAGGER_ITERATIONS:-3}"
PZR_CORL_DAGGER_EXPERT="${PZR_CORL_DAGGER_EXPERT:-mpc_wide_fixed_girard}"
PZR_CORL_BOOTSTRAP_SAMPLES="${PZR_CORL_BOOTSTRAP_SAMPLES:-5000}"
FORCE_ARGS=()
if [[ "${PZR_CORL_FORCE:-0}" == "1" ]]; then
  FORCE_ARGS=(--force)
fi

mkdir -p "$ROOT_DIR/results/logs"

{
  echo "CoRL headline output: $OUT_DIR"
  echo "CoRL headline log: $LOG_PATH"
  echo "Running CoRL firmware preflight..."

  pzr-run-corl \
    --preflight \
    --profile "$PZR_CORL_PROFILE" \
    --safe-control-gym-root "$PZR_SAFE_CONTROL_GYM_ROOT" \
    --safe-control-python "$PZR_SAFE_CONTROL_PYTHON" \
    --safe-control-config "$PZR_SAFE_CONTROL_CONFIG" \
    --safe-control-controller-mode firmware

  echo "Starting CoRL headline run..."

  pzr-run-corl \
    --profile "$PZR_CORL_PROFILE" \
    --safe-control-gym-root "$PZR_SAFE_CONTROL_GYM_ROOT" \
    --safe-control-python "$PZR_SAFE_CONTROL_PYTHON" \
    --safe-control-config "$PZR_SAFE_CONTROL_CONFIG" \
    --safe-control-controller-mode firmware \
    --out "$OUT_DIR" \
    "${FORCE_ARGS[@]}" \
    --method-set "$PZR_CORL_METHOD_SET" \
    --learned-mode "$PZR_CORL_LEARNED_MODE" \
    --budget "$PZR_CORL_BUDGET" \
    --horizon "$PZR_CORL_HORIZON" \
    --max-steps "$PZR_CORL_MAX_STEPS" \
    --train-seeds "$PZR_CORL_TRAIN_SEEDS" \
    --eval-seeds "$PZR_CORL_EVAL_SEEDS" \
    --dagger-iterations "$PZR_CORL_DAGGER_ITERATIONS" \
    --dagger-expert "$PZR_CORL_DAGGER_EXPERT" \
    --bootstrap-samples "$PZR_CORL_BOOTSTRAP_SAMPLES" \
    --fail-on-unusable
} 2>&1 | tee "$LOG_PATH"
