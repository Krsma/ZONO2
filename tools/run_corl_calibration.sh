#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="corl-calibration-$(date +%Y%m%d-%H%M%S)"
OUT_DIR="$ROOT_DIR/results/$RUN_ID"
LOG_PATH="$ROOT_DIR/results/logs/$RUN_ID.log"

export PZR_SAFE_CONTROL_GYM_ROOT="${PZR_SAFE_CONTROL_GYM_ROOT:-$ROOT_DIR/external/safe-control-gym}"
export PZR_SAFE_CONTROL_PYTHON="${PZR_SAFE_CONTROL_PYTHON:-$ROOT_DIR/external/miniconda3/envs/pzr-safe-control-fw/bin/python}"
PZR_CORL_PROFILE="${PZR_CORL_PROFILE:-overnight}"
PZR_CORL_CALIBRATION_SEEDS="${PZR_CORL_CALIBRATION_SEEDS:-5}"
PZR_CORL_CALIBRATION_MAX_STEPS="${PZR_CORL_CALIBRATION_MAX_STEPS:-1000}"
FORCE_ARGS=()
if [[ "${PZR_CORL_FORCE:-0}" == "1" ]]; then
  FORCE_ARGS=(--force)
fi

mkdir -p "$ROOT_DIR/results/logs"

{
  echo "CoRL calibration output: $OUT_DIR"
  echo "CoRL calibration log: $LOG_PATH"

  pzr-run-corl \
    --profile "$PZR_CORL_PROFILE" \
    --calibration \
    --safe-control-gym-root "$PZR_SAFE_CONTROL_GYM_ROOT" \
    --safe-control-python "$PZR_SAFE_CONTROL_PYTHON" \
    --safe-control-controller-mode firmware \
    --out "$OUT_DIR" \
    "${FORCE_ARGS[@]}" \
    --no-archive \
    --calibration-seeds "$PZR_CORL_CALIBRATION_SEEDS" \
    --calibration-max-steps "$PZR_CORL_CALIBRATION_MAX_STEPS"
} 2>&1 | tee "$LOG_PATH"
