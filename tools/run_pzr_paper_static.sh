#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${PZR_RUN_ID:-paper-static-$(date +%Y%m%d-%H%M%S)}"
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

PROFILE="${PZR_PROFILE:-paper}"
SCENARIO="${PZR_SCENARIO:-all}"
BUDGET="${PZR_BUDGET:-10}"
HORIZON="${PZR_HORIZON:-4}"
SEEDS="${PZR_SEEDS:-30}"
JOBS="${PZR_JOBS:-4}"

mkdir -p "$(dirname "$LOG_PATH")" "$MPLCONFIGDIR"

{
  echo "Starting PZR static-baseline paper benchmark"
  echo "  output: $OUT_DIR"
  echo "  log: $LOG_PATH"
  echo "  scenario: $SCENARIO"
  echo "  seeds: $SEEDS"
  echo "  jobs: $JOBS"
  echo

  pzr-benchmark \
    --profile "$PROFILE" \
    --scenario "$SCENARIO" \
    --method-set static \
    --budget "$BUDGET" \
    --horizon "$HORIZON" \
    --seeds "$SEEDS" \
    --jobs "$JOBS" \
    --output "$OUT_DIR"
} 2>&1 | tee "$LOG_PATH"
