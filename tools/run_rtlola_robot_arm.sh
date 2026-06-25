#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${PZR_CONDA:-$ROOT_DIR/external/miniconda3/bin/conda}"
ENV_NAME="${PZR_ROBOT_ARM_ENV:-pzr-robot-arm}"
ENV_PREFIX="$ROOT_DIR/external/miniconda3/envs/$ENV_NAME"

if [ ! -x "$CONDA" ]; then
  echo "Conda executable not found: $CONDA" >&2
  exit 1
fi

if [ ! -d "$ENV_PREFIX" ]; then
  echo "Conda environment not found: $ENV_NAME" >&2
  echo "Create it first with: tools/setup_robot_arm_env.sh" >&2
  exit 1
fi

mkdir -p "${MPLCONFIGDIR:-/tmp/pzr-matplotlib-cache}"

ARGS=(
  --profile "${PZR_RTLOLA_PROFILE:-smoke}"
  --scenario robot_arm
  --trace-kind "${PZR_ROBOT_ARM_TRACE_KIND:-figure8_violated}"
  --budget "${PZR_ROBOT_ARM_BUDGET:-80}"
  --length "${PZR_ROBOT_ARM_LENGTH:-200}"
  --seeds "${PZR_ROBOT_ARM_SEEDS:-3}"
  --method-set "${PZR_ROBOT_ARM_METHOD_SET:-all}"
  --output "${PZR_ROBOT_ARM_OUTPUT:-/tmp/pzr-rtlola-arm}"
  --no-progress
)

cd "$ROOT_DIR"

LD_PRELOAD="$ENV_PREFIX/lib/libopenblas.so${LD_PRELOAD:+ $LD_PRELOAD}" \
LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pzr-matplotlib-cache}" \
CONDA_NO_PLUGINS=true "$CONDA" run -n "$ENV_NAME" \
  python -m pzr.rtlola.cli "${ARGS[@]}" "$@"
