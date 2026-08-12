#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PZR_PYTHON:-$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm/bin/python}"
COMMAND="${1:?usage: tools/run_prp_v4_exploratory.sh COMMAND [OPTIONS]}"
shift

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pzr-prp-v4-matplotlib}"

if [[ -f "$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm/lib/libopenblas.so" ]]; then
    export LD_PRELOAD="$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm/lib/libopenblas.so${LD_PRELOAD:+:$LD_PRELOAD}"
fi

cd "$ROOT_DIR"
exec "$PYTHON_BIN" tools/prp_v4_exploratory.py "$COMMAND" "$@"
