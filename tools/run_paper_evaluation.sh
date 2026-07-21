#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PZR_PYTHON:-$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm/bin/python}"
CONFIG="${PZR_PAPER_CONFIG:-$ROOT_DIR/experiments/paper_evaluation_v1.yaml}"
STAGE="${1:?usage: tools/run_paper_evaluation.sh COMMAND [pzr-paper options]}"
shift

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pzr-matplotlib}"

if [[ -f "$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm/lib/libopenblas.so" ]]; then
    export LD_PRELOAD="$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm/lib/libopenblas.so${LD_PRELOAD:+:$LD_PRELOAD}"
fi

cd "$ROOT_DIR"
exec "$PYTHON_BIN" -m pzr.rtlola.paper_pipeline "$STAGE" --config "$CONFIG" "$@"
