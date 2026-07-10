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

cd "$ROOT_DIR"

echo "Setting up RTLola binding environment '$ENV_NAME'..."
PZR_RTLOLA_ENV="$ENV_NAME" PZR_CONDA="$CONDA" "$ROOT_DIR/tools/setup_rtlola_binding.sh"

echo "Installing robot-arm MuJoCO dependencies..."
CONDA_NO_PLUGINS=true "$CONDA" run -n "$ENV_NAME" \
  python -m pip install --upgrade "numpy>=1.26,<2.7" "mujoco>=3.0"

echo "Checking installed package constraints..."
CONDA_NO_PLUGINS=true "$CONDA" run -n "$ENV_NAME" python -m pip check

echo "Verifying robot-arm imports..."
LD_PRELOAD="$ENV_PREFIX/lib/libopenblas.so${LD_PRELOAD:+ $LD_PRELOAD}" \
LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
CONDA_NO_PLUGINS=true "$CONDA" run -n "$ENV_NAME" python -c '
import matplotlib
import mujoco
import numpy
import pandas
import torch
from rlola_python_binding import RLolaMonitor, ZonotopeConfig

print(f"numpy {numpy.__version__}")
print(f"pandas {pandas.__version__}")
print(f"matplotlib {matplotlib.__version__}")
print(f"mujoco {mujoco.__version__}")
print(f"torch {torch.__version__}")
print("rtlola binding ok")
'

cat <<EOF
Robot-arm environment ready.

Use:
  tools/run_rtlola_robot_arm.sh --output /tmp/pzr-rtlola-arm

This environment contains only the RTLola benchmark and optional MuJoCo
dependencies; retired robotics simulator stacks are intentionally excluded.
EOF
