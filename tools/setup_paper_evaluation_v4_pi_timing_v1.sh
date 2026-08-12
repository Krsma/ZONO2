#!/usr/bin/env bash
set -euo pipefail

COMMAND="${1:?usage: setup_paper_evaluation_v4_pi_timing_v1.sh setup|host-controls BUNDLE_ROOT}"
BUNDLE_ROOT="${2:?bundle root is required}"
BUNDLE_ROOT="$(cd "$BUNDLE_ROOT" && pwd)"
ENV_ROOT="${PZR_PI_TIMING_ENV_ROOT:-$BUNDLE_ROOT/../pzr-pi-timing-env-v1}"
mkdir -p "$ENV_ROOT"
ENV_ROOT="$(cd "$ENV_ROOT" && pwd)"
MINIFORGE_ROOT="$ENV_ROOT/miniforge"
RUNTIME_PREFIX="$ENV_ROOT/runtime"

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "Pi timing requires aarch64 Linux" >&2
    exit 1
fi
if ! tr -d '\0' </proc/device-tree/model 2>/dev/null | grep -q "Raspberry Pi 5"; then
    echo "Pi timing requires a Raspberry Pi 5" >&2
    exit 1
fi

case "$COMMAND" in
host-controls)
    if [[ "$EUID" -ne 0 ]]; then
        echo "host-controls must be run with sudo" >&2
        exit 1
    fi
    for governor in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        [[ -e "$governor" ]] || continue
        printf '%s\n' performance >"$governor"
    done
    echo "performance governor requested on all available cores"
    ;;
setup)
    sudo apt-get update
    sudo apt-get install -y \
        build-essential curl git libopenblas-dev pkg-config openssh-client

    INSTALLER="$ENV_ROOT/Miniforge3-26.3.2-2-Linux-aarch64.sh"
    if [[ ! -f "$INSTALLER" ]]; then
        curl -fL \
            "https://github.com/conda-forge/miniforge/releases/download/26.3.2-2/Miniforge3-26.3.2-2-Linux-aarch64.sh" \
            -o "$INSTALLER"
    fi
    printf '%s  %s\n' \
        535144deb6908e2a8ef8c60306a9a6c4fdbbe85034f056a98776ec3dcb9e9c14 \
        "$INSTALLER" | sha256sum --check --status
    if [[ ! -x "$MINIFORGE_ROOT/bin/conda" ]]; then
        bash "$INSTALLER" -b -p "$MINIFORGE_ROOT"
    fi
    if [[ ! -x "$RUNTIME_PREFIX/bin/python" ]]; then
        CONDA_NO_PLUGINS=true "$MINIFORGE_ROOT/bin/conda" create \
            --solver classic -y -p "$RUNTIME_PREFIX" python=3.11.15 pip
    fi
    "$RUNTIME_PREFIX/bin/python" -m pip install --upgrade pip
    "$RUNTIME_PREFIX/bin/python" -m pip install \
        numpy==2.4.6 pandas==3.0.3 pyyaml==6.0.3 \
        maturin==1.14.1 pytest==9.1.1
    "$RUNTIME_PREFIX/bin/python" -m pip install \
        'torch==2.12.1+cpu' --index-url https://download.pytorch.org/whl/cpu

    if ! command -v rustup >/dev/null 2>&1; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
            sh -s -- -y --profile minimal --default-toolchain 1.96.0
    fi
    if [[ -f "$HOME/.cargo/env" ]]; then
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
    fi
    rustup toolchain install 1.96.0 --profile minimal
    (
        cd "$BUNDLE_ROOT/rlolapythonbinding"
        rustup override set 1.96.0
    )

    WHEEL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pzr-pi-binding.XXXXXX")"
    trap 'rm -rf "$WHEEL_DIR"' EXIT
    CARGO_NET_GIT_FETCH_WITH_CLI=true \
    CARGO_TARGET_DIR="$ENV_ROOT/cargo-target" \
    RUSTC_BOOTSTRAP=kmeans \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    "$RUNTIME_PREFIX/bin/python" -m maturin build \
        --release --locked \
        --interpreter "$RUNTIME_PREFIX/bin/python" \
        --manifest-path "$BUNDLE_ROOT/rlolapythonbinding/Cargo.toml" \
        --out "$WHEEL_DIR"
    mapfile -t WHEELS < <(find "$WHEEL_DIR" -maxdepth 1 -name '*.whl' -type f)
    if [[ "${#WHEELS[@]}" -ne 1 ]]; then
        echo "expected exactly one binding wheel" >&2
        exit 1
    fi
    "$RUNTIME_PREFIX/bin/python" -m pip install --force-reinstall --no-deps "${WHEELS[0]}"
    "$RUNTIME_PREFIX/bin/python" - "$RUNTIME_PREFIX" <<'PY'
import pathlib
import sys
import sysconfig

prefix = pathlib.Path(sys.argv[1])
if pathlib.Path(sys.prefix).resolve() != prefix.resolve():
    raise SystemExit("binding provenance marker is running in the wrong environment")
site_packages = pathlib.Path(sysconfig.get_paths()["purelib"])
marker = site_packages / "rlola_python_binding_pzr_provenance.py"
marker.write_text(
    "BINDING_REVISION = '01c92a2bfac58755e3b832bb0094816f3f36e1d1'\n"
    "INTERPRETER_REVISION = '2724b05ae6c62ed0df14f1401ed8db89472725a6'\n"
    "BINDING_BUILD_PROFILE = 'release'\n"
)
PY
    echo "Pi environment ready: $RUNTIME_PREFIX"
    ;;
*)
    echo "unknown command: $COMMAND" >&2
    exit 1
    ;;
esac
