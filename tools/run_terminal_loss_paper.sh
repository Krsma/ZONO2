#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PZR_PYTHON:-$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm/bin/python}"
CONFIG="${PZR_PAPER_CONFIG:-$ROOT_DIR/experiments/terminal_loss_paper_v1.yaml}"
STAGE="${1:?usage: tools/run_terminal_loss_paper.sh STAGE [pzr-paper options]}"
shift

export PYTHONPATH="${PYTHONPATH:-$ROOT_DIR/src}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pzr-matplotlib}"

if [[ -f "$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm/lib/libopenblas.so" ]]; then
    export LD_PRELOAD="$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm/lib/libopenblas.so${LD_PRELOAD:+:$LD_PRELOAD}"
fi

exec "$PYTHON_BIN" -m pzr.rtlola.paper_pipeline "$STAGE" --config "$CONFIG" "$@"
