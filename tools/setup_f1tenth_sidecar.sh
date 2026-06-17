#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOOTSTRAP_PYTHON="${PZR_F1TENTH_BOOTSTRAP_PYTHON:-$ROOT_DIR/external/miniconda3/envs/pzr-safe-control-fw/bin/python}"
VENV_DIR="${PZR_F1TENTH_VENV:-$ROOT_DIR/external/f1tenth-py38-venv}"

"$BOOTSTRAP_PYTHON" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade "pip<24.1" "setuptools==65.5.0" "wheel<0.40"
"$VENV_DIR/bin/python" -m pip install "git+https://github.com/f1tenth/f1tenth_gym.git"

cat <<EOF
F1TENTH sidecar ready:
  $VENV_DIR/bin/python

Use:
  python -m pzr.experiments.robotics_probe \\
    --candidate f1tenth \\
    --trace-source live \\
    --f1tenth-sidecar-python "$VENV_DIR/bin/python" \\
    --output /tmp/pzr-f1tenth-live
EOF
