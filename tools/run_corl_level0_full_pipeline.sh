#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${PZR_CORL_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="$ROOT_DIR/results/logs"
PREFLIGHT_JSON="$LOG_DIR/corl-level0-preflight-$RUN_ID.json"
LOG_PATH="$LOG_DIR/corl-level0-full-pipeline-$RUN_ID.log"

export PZR_SAFE_CONTROL_GYM_ROOT="${PZR_SAFE_CONTROL_GYM_ROOT:-$ROOT_DIR/external/safe-control-gym}"
export PZR_SAFE_CONTROL_PYTHON="${PZR_SAFE_CONTROL_PYTHON:-$ROOT_DIR/external/miniconda3/envs/pzr-safe-control-fw/bin/python}"
export PZR_SAFE_CONTROL_CONFIG="${PZR_SAFE_CONTROL_CONFIG:-competition/level0.yaml}"

PZR_CORL_PROFILE="${PZR_CORL_PROFILE:-overnight}"
PZR_CORL_METHOD_SET="${PZR_CORL_METHOD_SET:-core}"
PZR_CORL_BUDGET="${PZR_CORL_BUDGET:-8}"
PZR_CORL_HORIZON="${PZR_CORL_HORIZON:-6}"
PZR_CORL_MAX_STEPS="${PZR_CORL_MAX_STEPS:-1000}"
PZR_CORL_CALIBRATION_SEEDS="${PZR_CORL_CALIBRATION_SEEDS:-10}"
PZR_CORL_CALIBRATION_MAX_STEPS="${PZR_CORL_CALIBRATION_MAX_STEPS:-1000}"
PZR_CORL_TRAIN_SEEDS="${PZR_CORL_TRAIN_SEEDS:-20}"
PZR_CORL_EVAL_SEEDS="${PZR_CORL_EVAL_SEEDS:-50}"
PZR_CORL_PAPER_EVAL_SEEDS="${PZR_CORL_PAPER_EVAL_SEEDS:-100}"
PZR_CORL_BOOTSTRAP_SAMPLES="${PZR_CORL_BOOTSTRAP_SAMPLES:-5000}"
PZR_CORL_MONITOR_OVERLAP="${PZR_CORL_MONITOR_OVERLAP:-}"
PZR_CORL_GENERATOR_MEMORY_DECAY="${PZR_CORL_GENERATOR_MEMORY_DECAY:-}"

CALIBRATION_OUT="$ROOT_DIR/results/corl-level0-calibration-monitor-first-$RUN_ID"
HELDOUT_OUT="$ROOT_DIR/results/corl-level0-core-heldout-$RUN_ID"
PAPER_OUT="$ROOT_DIR/results/corl-level0-core-paper-$RUN_ID"
REGRET_OUT="$ROOT_DIR/results/corl-level0-regret-$RUN_ID"

FORCE_ARGS=()
if [[ "${PZR_CORL_FORCE:-0}" == "1" ]]; then
  FORCE_ARGS=(--force)
fi

MONITOR_ARGS=()
if [[ -n "$PZR_CORL_MONITOR_OVERLAP" ]]; then
  MONITOR_ARGS+=(--monitor-overlap "$PZR_CORL_MONITOR_OVERLAP")
fi
if [[ -n "$PZR_CORL_GENERATOR_MEMORY_DECAY" ]]; then
  MONITOR_ARGS+=(--generator-memory-decay "$PZR_CORL_GENERATOR_MEMORY_DECAY")
fi

COMMON_ENV_ARGS=(
  --safe-control-gym-root "$PZR_SAFE_CONTROL_GYM_ROOT"
  --safe-control-python "$PZR_SAFE_CONTROL_PYTHON"
  --safe-control-config "$PZR_SAFE_CONTROL_CONFIG"
  --safe-control-controller-mode firmware
)

COMMON_RUN_ARGS=(
  --profile "$PZR_CORL_PROFILE"
  "${COMMON_ENV_ARGS[@]}"
  --method-set "$PZR_CORL_METHOD_SET"
  --budget "$PZR_CORL_BUDGET"
  --horizon "$PZR_CORL_HORIZON"
  --max-steps "$PZR_CORL_MAX_STEPS"
  "${MONITOR_ARGS[@]}"
  --bootstrap-samples "$PZR_CORL_BOOTSTRAP_SAMPLES"
)

mkdir -p "$LOG_DIR"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "Missing $label: $path" >&2
    exit 2
  fi
}

validate_preflight() {
  python - "$PREFLIGHT_JSON" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)
checks = data.get("checks", {})
required = (
    "safe_control_python_exists",
    "safe_control_gym_root_exists",
    "pycffirmware_available",
    "firmware_wrapper_available",
    "firmware_reset",
    "firmware_step",
    "sidecar_reset",
    "sidecar_step",
)
missing = [name for name in required if not checks.get(name, False)]
if not data.get("ok", False):
    missing.append("ok")
if "fake_env_reset" in checks:
    missing.append("fake_env_reset_present")
if missing:
    print("Firmware preflight failed or fake environment was selected:", ", ".join(missing), file=sys.stderr)
    print(json.dumps(data, indent=2, sort_keys=True), file=sys.stderr)
    sys.exit(2)
PY
}

validate_calibration() {
  python - "$CALIBRATION_OUT/calibration_recommendations.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)
candidates = data.get("paper_candidate_config_ids", [])
if not candidates:
    print("Calibration produced no paper_candidate_config_ids.", file=sys.stderr)
    print(json.dumps(data, indent=2, sort_keys=True), file=sys.stderr)
    sys.exit(2)
print("Calibration candidate config IDs:", ", ".join(candidates))
print("Recommended config ID:", data.get("recommended_config_id"))
PY
}

validate_headline_run() {
  local out_dir="$1"
  local label="$2"
  python - "$out_dir" "$label" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
label = sys.argv[2]
notes_path = out_dir / "analysis_notes.json"
failures_path = out_dir / "failure_events.csv"
decisions_path = out_dir / "decision_features.csv"
manifest_path = out_dir / "manifest.json"
config_path = out_dir / "config.json"

with notes_path.open("r", encoding="utf-8") as handle:
    notes = json.load(handle)
if not notes.get("paper_usable", False):
    print(f"{label} is not paper usable:", notes.get("paper_usable_reasons", []), file=sys.stderr)
    sys.exit(2)

with failures_path.open("r", encoding="utf-8", newline="") as handle:
    failure_rows = list(csv.DictReader(handle))
if failure_rows:
    print(f"{label} failure_events.csv is nonempty: {len(failure_rows)} rows", file=sys.stderr)
    sys.exit(2)

with decisions_path.open("r", encoding="utf-8", newline="") as handle:
    decision_rows = list(csv.DictReader(handle))
if not decision_rows:
    print(f"{label} decision_features.csv is empty", file=sys.stderr)
    sys.exit(2)

with manifest_path.open("r", encoding="utf-8") as handle:
    manifest = json.load(handle)
preflight_checks = manifest.get("preflight", {}).get("checks", {})
if "fake_env_reset" in preflight_checks:
    print(f"{label} manifest indicates fake environment preflight", file=sys.stderr)
    sys.exit(2)

with config_path.open("r", encoding="utf-8") as handle:
    config = json.load(handle)
args = config.get("args", {})
if args.get("safe_control_controller_mode") != "firmware":
    print(f"{label} did not run with firmware controller mode", file=sys.stderr)
    sys.exit(2)
if args.get("safe_control_config") != "competition/level0.yaml":
    print(f"{label} did not use competition/level0.yaml", file=sys.stderr)
    sys.exit(2)

print(f"{label} passed headline checks.")
PY
}

report_regret_notes() {
  python - "$REGRET_OUT/analysis_notes.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    notes = json.load(handle)
learning = {
    key: value for key, value in notes.items()
    if "learning" in key or "regret" in key
}
print(json.dumps(learning, indent=2, sort_keys=True))
PY
}

require_file "$PZR_SAFE_CONTROL_GYM_ROOT" "safe-control-gym root"
require_file "$PZR_SAFE_CONTROL_PYTHON" "safe-control sidecar Python"
if [[ "$PZR_SAFE_CONTROL_CONFIG" != "competition/level0.yaml" ]]; then
  echo "This script is locked to firmware Level0; set PZR_SAFE_CONTROL_CONFIG=competition/level0.yaml." >&2
  echo "Current PZR_SAFE_CONTROL_CONFIG=$PZR_SAFE_CONTROL_CONFIG" >&2
  exit 2
fi

{
  echo "CoRL Level0 full pipeline run id: $RUN_ID"
  echo "Log: $LOG_PATH"
  echo "safe-control-gym root: $PZR_SAFE_CONTROL_GYM_ROOT"
  echo "safe-control Python: $PZR_SAFE_CONTROL_PYTHON"
  echo "safe-control config: $PZR_SAFE_CONTROL_CONFIG"
  echo

  echo "Running firmware preflight..."
  pzr-run-corl \
    --preflight \
    --profile "$PZR_CORL_PROFILE" \
    "${COMMON_ENV_ARGS[@]}" \
    | tee "$PREFLIGHT_JSON"
  validate_preflight
  if [[ "${PZR_CORL_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "PZR_CORL_PREFLIGHT_ONLY=1, stopping after successful firmware preflight."
    exit 0
  fi

  echo
  echo "Running Level0 calibration: $CALIBRATION_OUT"
  pzr-run-corl \
    --profile "$PZR_CORL_PROFILE" \
    --calibration \
    "${COMMON_ENV_ARGS[@]}" \
    --out "$CALIBRATION_OUT" \
    "${FORCE_ARGS[@]}" \
    "${MONITOR_ARGS[@]}" \
    --no-archive \
    --calibration-seeds "$PZR_CORL_CALIBRATION_SEEDS" \
    --calibration-max-steps "$PZR_CORL_CALIBRATION_MAX_STEPS"
  validate_calibration

  echo
  echo "Running Level0 core held-out evaluation: $HELDOUT_OUT"
  pzr-run-corl \
    "${COMMON_RUN_ARGS[@]}" \
    --learned-mode none \
    --train-seeds "$PZR_CORL_TRAIN_SEEDS" \
    --eval-seeds "$PZR_CORL_EVAL_SEEDS" \
    --out "$HELDOUT_OUT" \
    "${FORCE_ARGS[@]}" \
    --fail-on-unusable
  validate_headline_run "$HELDOUT_OUT" "core held-out"

  echo
  echo "Running Level0 core paper-scale evaluation: $PAPER_OUT"
  pzr-run-corl \
    "${COMMON_RUN_ARGS[@]}" \
    --learned-mode none \
    --train-seeds "$PZR_CORL_TRAIN_SEEDS" \
    --eval-seeds "$PZR_CORL_PAPER_EVAL_SEEDS" \
    --out "$PAPER_OUT" \
    "${FORCE_ARGS[@]}" \
    --fail-on-unusable
  validate_headline_run "$PAPER_OUT" "core paper-scale"

  echo
  echo "Running Level0 regret learned-policy ablation: $REGRET_OUT"
  pzr-run-corl \
    "${COMMON_RUN_ARGS[@]}" \
    --learned-mode regret \
    --train-seeds "$PZR_CORL_TRAIN_SEEDS" \
    --eval-seeds "$PZR_CORL_EVAL_SEEDS" \
    --out "$REGRET_OUT" \
    "${FORCE_ARGS[@]}" \
    --fail-on-unusable
  validate_headline_run "$REGRET_OUT" "regret ablation"
  report_regret_notes

  echo
  echo "CoRL Level0 full pipeline completed."
  echo "Calibration: $CALIBRATION_OUT"
  echo "Core held-out: $HELDOUT_OUT"
  echo "Core paper-scale: $PAPER_OUT"
  echo "Regret ablation: $REGRET_OUT"
} 2>&1 | tee "$LOG_PATH"
