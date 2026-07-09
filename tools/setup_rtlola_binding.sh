#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${PZR_CONDA:-$ROOT_DIR/external/miniconda3/bin/conda}"
ENV_NAME="${PZR_RTLOLA_ENV:-pzr-rtlola}"
BINDING_REV="dbef0fb52b66f38da763f694f857dfa6f1e40975"
INTERPRETER_REV="a143dd6a1500d54c1eabe9e83e5b54271734d6b2"
BINDING_PROFILE="release"
BINDING_DIR="$ROOT_DIR/rlolapythonbinding"

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
  python -m pip install -e ".[dev]" maturin numpy

if [ ! -e "$BINDING_DIR/.git" ]; then
  echo "RTLola binding submodule is not initialized." >&2
  echo "Run: git submodule update --init --recursive" >&2
  exit 1
fi

ACTUAL_REV="$(git -C "$BINDING_DIR" rev-parse HEAD)"
if [ "$ACTUAL_REV" != "$BINDING_REV" ]; then
  echo "Unexpected RTLola binding revision: $ACTUAL_REV" >&2
  echo "Expected the superproject pin: $BINDING_REV" >&2
  echo "Run: git submodule update --init --recursive" >&2
  exit 1
fi

LOCKED_INTERPRETER_COUNT="$(
  grep -c "rtlola-interpreter.git?branch=slack-vars#$INTERPRETER_REV" \
    "$BINDING_DIR/Cargo.lock" || true
)"
if [ "$LOCKED_INTERPRETER_COUNT" -ne 2 ]; then
  echo "Binding Cargo.lock does not pin both RTLola crates to $INTERPRETER_REV" >&2
  exit 1
fi

ENV_PREFIX="$ROOT_DIR/external/miniconda3/envs/$ENV_NAME"
WHEEL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pzr-rtlola-wheel.XXXXXX")"
trap 'rm -rf "$WHEEL_DIR"' EXIT
CARGO_NET_GIT_FETCH_WITH_CLI=true \
RUSTC_BOOTSTRAP=kmeans \
PKG_CONFIG_PATH="$ENV_PREFIX/lib/pkgconfig:$ENV_PREFIX/share/pkgconfig:${PKG_CONFIG_PATH:-}" \
LD_PRELOAD="$ENV_PREFIX/lib/libopenblas.so${LD_PRELOAD:+ $LD_PRELOAD}" \
LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
CONDA_NO_PLUGINS=true "$CONDA" run -n "$ENV_NAME" \
  python -m maturin build \
    --release \
    --locked \
    --interpreter "$ENV_PREFIX/bin/python" \
    --manifest-path "$BINDING_DIR/Cargo.toml" \
    --out "$WHEEL_DIR"

mapfile -t WHEELS < <(find "$WHEEL_DIR" -maxdepth 1 -type f -name '*.whl' -print)
if [ "${#WHEELS[@]}" -ne 1 ]; then
  echo "Expected exactly one RTLola binding wheel, found ${#WHEELS[@]}" >&2
  exit 1
fi
CONDA_NO_PLUGINS=true "$CONDA" run -n "$ENV_NAME" \
  python -m pip install --force-reinstall --no-deps "${WHEELS[0]}"

CONDA_NO_PLUGINS=true "$CONDA" run -n "$ENV_NAME" \
  python -c '
import pathlib
import sys
import sysconfig

binding_revision, interpreter_revision, build_profile = sys.argv[1:]
site_packages = pathlib.Path(sysconfig.get_paths()["purelib"])
marker = site_packages / "rlola_python_binding_pzr_provenance.py"
marker.write_text(
    "\n".join((
        f"BINDING_REVISION = {binding_revision!r}",
        f"INTERPRETER_REVISION = {interpreter_revision!r}",
        f"BINDING_BUILD_PROFILE = {build_profile!r}",
        "",
    )),
)
print(f"wrote RTLola binding provenance marker: {marker}")
' "$BINDING_REV" "$INTERPRETER_REV" "$BINDING_PROFILE"

LD_PRELOAD="$ENV_PREFIX/lib/libopenblas.so${LD_PRELOAD:+ $LD_PRELOAD}" \
LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}" \
PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
CONDA_NO_PLUGINS=true "$CONDA" run -n "$ENV_NAME" python -c '
from pzr.rtlola.binding import (
    BINDING_BUILD_PROFILE,
    BINDING_REVISION,
    INTERPRETER_REVISION,
    require_binding,
)

require_binding()
print(
    f"rtlola binding ok ({BINDING_BUILD_PROFILE}, "
    f"binding {BINDING_REVISION}, interpreter {INTERPRETER_REVISION})"
)
'

cat <<EOF
RTLola binding environment ready.

Use:
  LD_PRELOAD="$ENV_PREFIX/lib/libopenblas.so" LD_LIBRARY_PATH="$ENV_PREFIX/lib:\${LD_LIBRARY_PATH:-}" CONDA_NO_PLUGINS=true $CONDA run -n $ENV_NAME python -m pzr.rtlola.cli --profile smoke --scenario omni_robot --output /tmp/pzr-rtlola-smoke
EOF
