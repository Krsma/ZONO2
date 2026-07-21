#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="${PZR_PAPER_DIR:-$ROOT_DIR/results/rtlola-learning-paper-v2-01c92a2-2724b05-2257d07}"
PRIMARY_DIR="${PZR_PRIMARY_DIR:-$PAPER_DIR/primary}"
MPC_ADDON_DIR="${PZR_MPC_ADDON_DIR:-$PAPER_DIR/mpc-addon}"
PYTHON="${PZR_PYTHON:-$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm/bin/python}"

PZR_OUT_DIR="$PRIMARY_DIR" tools/run_rtlola_learning_primary.sh
PZR_PRIMARY_DIR="$PRIMARY_DIR" PZR_OUT_DIR="$MPC_ADDON_DIR" \
    tools/run_rtlola_learning_mpc_addon.sh

PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" -m pzr.learning.cli report-paper \
    --primary "$PRIMARY_DIR/evaluation" \
    --mpc-addon "$MPC_ADDON_DIR/evaluation" \
    --output "$PAPER_DIR/paper-reports"

echo "RTLola primary, online-MPC add-on, and paper reports complete: $PAPER_DIR"
