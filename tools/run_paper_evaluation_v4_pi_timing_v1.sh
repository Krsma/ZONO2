#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$HOME/.cargo/env" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi
PYTHON_BIN="${PZR_PI_TIMING_PYTHON:-$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm/bin/python}"
CONFIG="${PZR_PI_TIMING_CONFIG:-$ROOT_DIR/experiments/paper_evaluation_v4_pi_timing_v1.yaml}"
COMMAND="${1:?usage: tools/run_paper_evaluation_v4_pi_timing_v1.sh COMMAND [OPTIONS]}"
shift

export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR/tools${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TORCH_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pzr-pi-timing-v1-matplotlib}"

if [[ -f "$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm/lib/libopenblas.so" ]]; then
    export LD_PRELOAD="$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm/lib/libopenblas.so${LD_PRELOAD:+:$LD_PRELOAD}"
fi

cd "$ROOT_DIR"
exec "$PYTHON_BIN" tools/paper_evaluation_v4_pi_timing_v1.py \
    "$COMMAND" --config "$CONFIG" "$@"
