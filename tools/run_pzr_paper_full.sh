#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${PZR_RUN_ID:-paper-full-$(date +%Y%m%d-%H%M%S)}"
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
METHOD_SET="${PZR_METHOD_SET:-all}"
BUDGET="${PZR_BUDGET:-10}"
HORIZON="${PZR_HORIZON:-4}"
SEEDS="${PZR_SEEDS:-30}"
JOBS="${PZR_JOBS:-4}"

LEARNED_ARGS=(--learned-mode none)
if [[ "${PZR_WITH_REGRET:-0}" == "1" ]]; then
  LEARNED_ARGS=(
    --learned-mode regret
    --regret-oracle "${PZR_REGRET_ORACLE:-beam3}"
    --regret-iterations "${PZR_REGRET_ITERATIONS:-3}"
    --regret-epochs "${PZR_REGRET_EPOCHS:-100}"
  )
fi

mkdir -p "$(dirname "$LOG_PATH")" "$MPLCONFIGDIR"

{
  echo "Starting PZR full paper benchmark"
  echo "  output: $OUT_DIR"
  echo "  log: $LOG_PATH"
  echo "  profile: $PROFILE"
  echo "  scenario: $SCENARIO"
  echo "  method set: $METHOD_SET"
  echo "  budget: $BUDGET"
  echo "  horizon: $HORIZON"
  echo "  seeds: $SEEDS"
  echo "  jobs: $JOBS"
  echo "  learned: ${LEARNED_ARGS[*]}"
  echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
  echo "  OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
  echo "  MKL_NUM_THREADS=$MKL_NUM_THREADS"
  echo "  NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"
  echo "  VECLIB_MAXIMUM_THREADS=$VECLIB_MAXIMUM_THREADS"
  echo "  BLIS_NUM_THREADS=$BLIS_NUM_THREADS"
  echo

  pzr-benchmark \
    --profile "$PROFILE" \
    --scenario "$SCENARIO" \
    --method-set "$METHOD_SET" \
    --budget "$BUDGET" \
    --horizon "$HORIZON" \
    --seeds "$SEEDS" \
    --jobs "$JOBS" \
    "${LEARNED_ARGS[@]}" \
    --output "$OUT_DIR"
} 2>&1 | tee "$LOG_PATH"
