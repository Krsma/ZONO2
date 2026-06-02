#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${PZR_RUN_ID:-smoke-parallel-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${PZR_OUT_DIR:-results/$RUN_ID}"
LOG_PATH="${PZR_LOG_PATH:-results/logs/$RUN_ID.log}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/results/matplotlib-cache}"

SCENARIO="${PZR_SCENARIO:-robot_arm}"
METHOD_SET="${PZR_METHOD_SET:-all}"
BUDGET="${PZR_BUDGET:-10}"
HORIZON="${PZR_HORIZON:-3}"
SEEDS="${PZR_SEEDS:-4}"
JOBS="${PZR_JOBS:-2}"

mkdir -p "$(dirname "$LOG_PATH")" "$MPLCONFIGDIR"

{
  echo "Starting PZR parallel smoke benchmark"
  echo "  output: $OUT_DIR"
  echo "  log: $LOG_PATH"
  echo "  scenario: $SCENARIO"
  echo "  method set: $METHOD_SET"
  echo "  seeds: $SEEDS"
  echo "  jobs: $JOBS"
  echo

  pzr-benchmark \
    --profile smoke \
    --scenario "$SCENARIO" \
    --method-set "$METHOD_SET" \
    --budget "$BUDGET" \
    --horizon "$HORIZON" \
    --seeds "$SEEDS" \
    --jobs "$JOBS" \
    --no-dagger \
    --no-progress \
    --output "$OUT_DIR"
} 2>&1 | tee "$LOG_PATH"
