#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_ID="${PZR_CORL_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${PZR_CORL_OUT_DIR:-results/corl-level0-firmware-replay-${RUN_ID}}"
LOG_PATH="${PZR_CORL_LOG_PATH:-results/corl-level0-firmware-replay-${RUN_ID}.log}"

SAFE_CONTROL_GYM_ROOT="${PZR_SAFE_CONTROL_GYM_ROOT:-/home/vlkr/Faks/phd/ZONO2/external/safe-control-gym}"
SAFE_CONTROL_PYTHON="${PZR_SAFE_CONTROL_PYTHON:-/home/vlkr/Faks/phd/ZONO2/external/miniconda3/envs/pzr-safe-control-fw/bin/python}"
SAFE_CONTROL_CONFIG="${PZR_SAFE_CONTROL_CONFIG:-competition/level0.yaml}"

SEEDS="${PZR_CORL_REPLAY_SEEDS:-50}"
SEED_START="${PZR_CORL_REPLAY_SEED_START:-0}"
LENGTH="${PZR_CORL_REPLAY_LENGTH:-800}"
BUDGET="${PZR_CORL_REPLAY_BUDGET:-8}"
HORIZON="${PZR_CORL_REPLAY_HORIZON:-6}"
METHOD_SET="${PZR_CORL_REPLAY_METHOD_SET:-paper_plus_mpc_ablation}"
BOOTSTRAP_SAMPLES="${PZR_CORL_REPLAY_BOOTSTRAP_SAMPLES:-5000}"
SENSOR_BIAS_BOUND="${PZR_CORL_REPLAY_SENSOR_BIAS_BOUND:-0.015}"
SENSOR_NOISE_BOUND="${PZR_CORL_REPLAY_SENSOR_NOISE_BOUND:-0.03}"
POSE_CORRECTION_GAIN="${PZR_CORL_REPLAY_POSE_CORRECTION_GAIN:-0.35}"
VELOCITY_CORRECTION_GAIN="${PZR_CORL_REPLAY_VELOCITY_CORRECTION_GAIN:-0.3}"
MONITOR_OVERLAP="${PZR_CORL_REPLAY_MONITOR_OVERLAP:-0.0}"

if [[ "${SAFE_CONTROL_CONFIG}" != "competition/level0.yaml" ]]; then
  echo "Refusing to run: this disposable CoRL run is locked to competition/level0.yaml" >&2
  exit 2
fi

mkdir -p "$(dirname "${LOG_PATH}")"

FORCE_ARGS=()
if [[ "${PZR_CORL_FORCE:-0}" == "1" ]]; then
  FORCE_ARGS=(--force)
fi

echo "Running CoRL Level0 firmware replay experiment"
echo "  out: ${OUT_DIR}"
echo "  log: ${LOG_PATH}"
echo "  safe-control-gym: ${SAFE_CONTROL_GYM_ROOT}"
echo "  sidecar python: ${SAFE_CONTROL_PYTHON}"
echo "  config: ${SAFE_CONTROL_CONFIG}"
echo "  seeds: ${SEEDS} from ${SEED_START}, length: ${LENGTH}, budget: ${BUDGET}, horizon: ${HORIZON}"

python -m pzr.experiments.corl_firmware_replay \
  --safe-control-gym-root "${SAFE_CONTROL_GYM_ROOT}" \
  --safe-control-python "${SAFE_CONTROL_PYTHON}" \
  --safe-control-config "${SAFE_CONTROL_CONFIG}" \
  --safe-control-controller-mode firmware \
  --seeds "${SEEDS}" \
  --seed-start "${SEED_START}" \
  --length "${LENGTH}" \
  --budget "${BUDGET}" \
  --horizon "${HORIZON}" \
  --method-set "${METHOD_SET}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --sensor-bias-bound "${SENSOR_BIAS_BOUND}" \
  --sensor-noise-bound "${SENSOR_NOISE_BOUND}" \
  --pose-correction-gain "${POSE_CORRECTION_GAIN}" \
  --velocity-correction-gain "${VELOCITY_CORRECTION_GAIN}" \
  --monitor-overlap "${MONITOR_OVERLAP}" \
  --out "${OUT_DIR}" \
  "${FORCE_ARGS[@]}" \
  2>&1 | tee "${LOG_PATH}"

python - "${OUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

out = Path(sys.argv[1])
manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
checks = manifest["preflight"]["checks"]

def require(condition, message):
    if not condition:
        raise SystemExit(message)

require(checks.get("sidecar_controller_firmware"), "preflight did not use firmware controller")
require(checks.get("pycffirmware_available"), "pycffirmware was not available")
require(checks.get("firmware_wrapper_available"), "firmware wrapper was not available")
require(checks.get("firmware_reset") and checks.get("firmware_step"), "firmware reset/step preflight failed")
require(not checks.get("fake_env_reset", False), "fake environment was used")

raw = pd.read_csv(out / "raw_runs.csv")
decisions = pd.read_csv(out / "decision_features.csv")
traces = pd.read_csv(out / "firmware_traces.csv")
require(not raw.empty, "raw_runs.csv is empty")
require(not decisions.empty, "decision_features.csv is empty")
require(not traces.empty, "firmware_traces.csv is empty")

non_reference = raw[raw["method"] != "reference"]
for column in ("budget_violation_count", "unsound_certificate_count", "reduction_failure_count"):
    require(int(non_reference[column].sum()) == 0, f"{column} is nonzero")

summary = pd.read_csv(out / "summary.csv")
false_alarm = summary[summary["metric"].eq("false_alarm_rate")][["method", "mean"]]
print("\nFalse-alarm-rate means:")
print(false_alarm.sort_values("mean").to_string(index=False))

means = dict(zip(false_alarm["method"], false_alarm["mean"]))
best_mpc = min(
    (means[name], name)
    for name in means
    if name.startswith("mpc_") or name == "mpc"
) if any(name.startswith("mpc_") or name == "mpc" for name in means) else None
if best_mpc is not None and "box" in means:
    if best_mpc[0] < means["box"]:
        print(f"\nPZR signal present: {best_mpc[1]} beats box on false_alarm_rate ({best_mpc[0]:.6g} < {means['box']:.6g}).")
    else:
        print(f"\nWarning: best MPC false_alarm_rate did not beat box ({best_mpc[0]:.6g} >= {means['box']:.6g}).")

print(f"\nValidated firmware replay artifacts in {out}")
PY
