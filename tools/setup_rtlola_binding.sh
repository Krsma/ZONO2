#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${PZR_CONDA:-$ROOT_DIR/external/miniconda3/bin/conda}"
ENV_NAME="${PZR_RTLOLA_ENV:-pzr-rtlola}"
BINDING_REMOTE="${PZR_RTLOLA_BINDING_REMOTE:-git@projects.cispa.saarland:group-finkbeiner/tools/RTLola/rlolapythonbinding.git}"
BINDING_REV="${PZR_RTLOLA_BINDING_REV:-72622a3}"
if [ -n "${PZR_RTLOLA_BINDING_DIR:-}" ]; then
  BINDING_DIR="$PZR_RTLOLA_BINDING_DIR"
elif [ -e "$ROOT_DIR/vendor/rlola-python-binding/.git" ]; then
  BINDING_DIR="$ROOT_DIR/vendor/rlola-python-binding"
elif [ -d "$ROOT_DIR/rlolapythonbinding" ]; then
  BINDING_DIR="$ROOT_DIR/rlolapythonbinding"
else
  BINDING_DIR="$ROOT_DIR/vendor/rlola-python-binding"
fi

if ! command -v cargo >/dev/null 2>&1; then
  if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
  fi
fi

if ! command -v cargo >/dev/null 2>&1 || ! command -v rustc >/dev/null 2>&1; then
  cat >&2 <<EOF
Rust is required to build rlola_python_binding.
Install it with:
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  source "\$HOME/.cargo/env"
EOF
  exit 1
fi

if [ ! -x "$CONDA" ]; then
  echo "Conda executable not found: $CONDA" >&2
  exit 1
fi

cd "$ROOT_DIR"

if ! "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  CONDA_NO_PLUGINS=true "$CONDA" create --solver classic -y -n "$ENV_NAME" python=3.11
fi

CONDA_NO_PLUGINS=true "$CONDA" install --solver classic -y -n "$ENV_NAME" \
  fontconfig pkg-config openblas libopenblas
CONDA_NO_PLUGINS=true "$CONDA" run -n "$ENV_NAME" python -m pip install --upgrade pip
CONDA_NO_PLUGINS=true "$CONDA" run -n "$ENV_NAME" \
  python -m pip install -e ".[dev,learning]" maturin numpy

if [ ! -e "$BINDING_DIR/.git" ]; then
  mkdir -p "$(dirname "$BINDING_DIR")"
  git submodule add "$BINDING_REMOTE" "$BINDING_DIR"
fi

git -C "$BINDING_DIR" fetch --all --tags
git -C "$BINDING_DIR" checkout "$BINDING_REV"

ENV_PREFIX="$ROOT_DIR/external/miniconda3/envs/$ENV_NAME"
CARGO_NET_GIT_FETCH_WITH_CLI=true \
PKG_CONFIG_PATH="$ENV_PREFIX/lib/pkgconfig:$ENV_PREFIX/share/pkgconfig:${PKG_CONFIG_PATH:-}" \
LD_PRELOAD="$ENV_PREFIX/lib/libopenblas.so${LD_PRELOAD:+ $LD_PRELOAD}" \
LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
CONDA_NO_PLUGINS=true "$CONDA" run -n "$ENV_NAME" \
  python -m maturin develop --release --manifest-path "$BINDING_DIR/Cargo.toml"

LD_PRELOAD="$ENV_PREFIX/lib/libopenblas.so${LD_PRELOAD:+ $LD_PRELOAD}" \
LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
CONDA_NO_PLUGINS=true "$CONDA" run -n "$ENV_NAME" python - <<'PY'
from rlola_python_binding import RLolaMonitor, ZonotopeConfig
print("rtlola binding ok")
PY

cat <<EOF
RTLola binding environment ready.

Use:
  LD_PRELOAD="$ENV_PREFIX/lib/libopenblas.so" LD_LIBRARY_PATH="$ENV_PREFIX/lib:\${LD_LIBRARY_PATH:-}" CONDA_NO_PLUGINS=true $CONDA run -n $ENV_NAME python -m pzr.rtlola.cli --profile smoke --scenario omni_robot --output /tmp/pzr-rtlola-smoke
EOF
