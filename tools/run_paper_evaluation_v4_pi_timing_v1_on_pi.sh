#!/usr/bin/env bash
# Convenience launcher for the isolated Raspberry Pi 5 timing add-on.
#
# Run this only from an extracted, hash-verified Pi timing bundle.  It never
# writes into that bundle: the runtime environment and all results are siblings
# of it.  Run `setup`, reboot the Pi, then use `smoke` and `measure`.
set -euo pipefail

COMMAND="${1:?usage: $0 setup|smoke|measure [BUNDLE_ROOT] [SMOKE_BUNDLE_ROOT]}"
BUNDLE_ROOT="${2:-$PWD}"
BUNDLE_ROOT="$(cd "$BUNDLE_ROOT" && pwd)"
SMOKE_BUNDLE_ROOT="${3:-}"
ENV_ROOT="${PZR_PI_TIMING_ENV_ROOT:-$BUNDLE_ROOT/../pzr-pi-timing-env-v1}"
PYTHON_BIN="${PZR_PI_TIMING_PYTHON:-$ENV_ROOT/runtime/bin/python}"
RUNS_ROOT="${PZR_PI_TIMING_RUNS_ROOT:-$BUNDLE_ROOT/../pzr-pi-timing-runs-v1}"
RUNNER="$BUNDLE_ROOT/tools/run_paper_evaluation_v4_pi_timing_v1.sh"
SETUP="$BUNDLE_ROOT/tools/setup_paper_evaluation_v4_pi_timing_v1.sh"

if [[ ! -x "$RUNNER" || ! -x "$SETUP" ]]; then
    echo "BUNDLE_ROOT is not an extracted Pi timing bundle: $BUNDLE_ROOT" >&2
    exit 1
fi

require_runtime() {
    if [[ ! -x "$PYTHON_BIN" ]]; then
        echo "Pi runtime is missing: $PYTHON_BIN" >&2
        echo "Run '$0 setup $BUNDLE_ROOT', then reboot before timing." >&2
        exit 1
    fi
}

run_bundle() {
    PZR_PI_TIMING_PYTHON="$PYTHON_BIN" "$RUNNER" "$@"
}

wait_for_idle() {
    local timeout_seconds="${PZR_PI_TIMING_IDLE_TIMEOUT_SECONDS:-600}"
    if [[ ! "$timeout_seconds" =~ ^[1-9][0-9]*$ ]]; then
        echo "PZR_PI_TIMING_IDLE_TIMEOUT_SECONDS must be a positive integer" >&2
        exit 1
    fi
    local deadline=$((SECONDS + timeout_seconds))
    local load_average
    while true; do
        read -r load_average _ </proc/loadavg
        if awk -v load="$load_average" 'BEGIN { exit !(load < 0.5) }'; then
            echo "Pi is idle enough for timing (one-minute load average: $load_average)"
            return
        fi
        if ((SECONDS >= deadline)); then
            echo "Pi did not become idle within ${timeout_seconds}s (load: $load_average)" >&2
            exit 1
        fi
        echo "Waiting for contract-test load to decay below 0.5 (current: $load_average)"
        sleep 10
    done
}

case "$COMMAND" in
setup)
    "$SETUP" setup "$BUNDLE_ROOT"
    sudo "$SETUP" host-controls "$BUNDLE_ROOT"
    echo
    echo "Setup is complete. Reboot the Pi now; after reboot run:"
    echo "  $0 smoke $BUNDLE_ROOT /path/to/extracted-smoke-bundle"
    echo "  $0 measure $BUNDLE_ROOT"
    ;;
smoke)
    require_runtime
    if [[ -z "$SMOKE_BUNDLE_ROOT" || ! -d "$SMOKE_BUNDLE_ROOT" ]]; then
        echo "smoke requires the extracted smoke bundle as its third argument" >&2
        exit 1
    fi
    SMOKE_BUNDLE_ROOT="$(cd "$SMOKE_BUNDLE_ROOT" && pwd)"
    # Reboot restores the governor on Raspberry Pi OS; preflight requires it.
    sudo "$SETUP" host-controls "$BUNDLE_ROOT"
    mkdir -p "$RUNS_ROOT/smoke"
    PZR_PI_TIMING_CONFIG="$SMOKE_BUNDLE_ROOT/experiments/paper_evaluation_v4_pi_timing_v1.yaml" \
        PZR_PI_TIMING_PYTHON="$PYTHON_BIN" "$RUNNER" smoke \
        --bundle "$SMOKE_BUNDLE_ROOT" \
        --run-output "$RUNS_ROOT/smoke/paper-evaluation-v4-pi-timing-v1"
    ;;
measure)
    require_runtime
    # Reapply this after the required reboot, immediately before measuring.
    sudo "$SETUP" host-controls "$BUNDLE_ROOT"
    mkdir -p "$RUNS_ROOT/full"
    RUN_OUTPUT="$RUNS_ROOT/full/paper-evaluation-v4-pi-timing-v1"
    ARCHIVE="$RUNS_ROOT/paper-evaluation-v4-pi-timing-v1-results.tar.gz"
    run_bundle preflight --bundle "$BUNDLE_ROOT"
    run_bundle contract-tests --bundle "$BUNDLE_ROOT"
    wait_for_idle
    run_bundle run --bundle "$BUNDLE_ROOT" --run-output "$RUN_OUTPUT"
    run_bundle profile --bundle "$BUNDLE_ROOT" --run-output "$RUN_OUTPUT"
    run_bundle pack --bundle "$BUNDLE_ROOT" --run-output "$RUN_OUTPUT" --archive "$ARCHIVE"
    echo "Pi result archive: $ARCHIVE"
    ;;
*)
    echo "unknown command: $COMMAND (expected setup, smoke, or measure)" >&2
    exit 1
    ;;
esac
