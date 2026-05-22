#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DATE="$(date +%Y%m%d)"

export PZR_SAFE_CONTROL_GYM_ROOT="${PZR_SAFE_CONTROL_GYM_ROOT:-$ROOT_DIR/external/safe-control-gym}"
export PZR_SAFE_CONTROL_PYTHON="${PZR_SAFE_CONTROL_PYTHON:-$ROOT_DIR/external/miniconda3/envs/pzr-safe-control/bin/python}"

mkdir -p "$ROOT_DIR/results/logs"

pzr-run-corl \
  --profile overnight \
  --safe-control-gym-root "$PZR_SAFE_CONTROL_GYM_ROOT" \
  --safe-control-python "$PZR_SAFE_CONTROL_PYTHON" \
  --safe-control-config competition/level3.yaml \
  --out "$ROOT_DIR/results/corl-main-$RUN_DATE" \
  --force \
  --budget 8 \
  --horizon 6 \
  --max-steps 1000 \
  --train-seeds 20 \
  --eval-seeds 50 \
  --dagger-iterations 3 \
  --bootstrap-samples 5000 \
  2>&1 | tee "$ROOT_DIR/results/logs/corl-main-$RUN_DATE.log"
