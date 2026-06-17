#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${PZR_RUN_ID:-paper-live-headline-len200-k8-30-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${PZR_OUT_DIR:-results/$RUN_ID}"
LOG_PATH="${PZR_LOG_PATH:-results/logs/$RUN_ID.log}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/results/matplotlib-cache}"

LENGTH="${PZR_LENGTH:-200}"
SEEDS="${PZR_SEEDS:-10}"
BUDGETS="${PZR_BUDGETS:-8,10,12,16,20,24,30}"
HORIZON="${PZR_HORIZON:-4}"
BEAM_WIDTH="${PZR_BEAM_WIDTH:-4}"
JOBS="${PZR_JOBS:-4}"
DRONE_CONTROLLER="${PZR_DRONE_CONTROLLER:-sim}"
F1TENTH_PYTHON="${PZR_F1TENTH_PYTHON:-external/f1tenth-py38-venv/bin/python}"
F1TENTH_MAP="${PZR_F1TENTH_MAP:-paper_chicane}"
WITH_REGRET="${PZR_WITH_REGRET:-1}"
REGRET_ORACLE="${PZR_REGRET_ORACLE:-beam3}"
REGRET_ITERATIONS="${PZR_REGRET_ITERATIONS:-3}"
REGRET_EPOCHS="${PZR_REGRET_EPOCHS:-100}"
REGRET_TRAIN_SEEDS="${PZR_REGRET_TRAIN_SEEDS:-$SEEDS}"
REGRET_EVAL_SEEDS="${PZR_REGRET_EVAL_SEEDS:-$SEEDS}"
REGRET_ARGS=(--learned-mode none)
if [[ "$WITH_REGRET" == "1" ]]; then
  REGRET_ARGS=(
    --learned-mode regret
    --regret-oracle "$REGRET_ORACLE"
    --regret-iterations "$REGRET_ITERATIONS"
    --regret-epochs "$REGRET_EPOCHS"
    --regret-train-seeds "$REGRET_TRAIN_SEEDS"
    --regret-eval-seeds "$REGRET_EVAL_SEEDS"
  )
fi

mkdir -p "$(dirname "$LOG_PATH")" "$OUT_DIR" "$MPLCONFIGDIR"

{
  echo "Starting PZR live headline paper evaluation"
  echo "  output: $OUT_DIR"
  echo "  log: $LOG_PATH"
  echo "  length: $LENGTH"
  echo "  seeds: $SEEDS"
  echo "  budgets: $BUDGETS"
  echo "  horizon: $HORIZON"
  echo "  beam width: $BEAM_WIDTH"
  echo "  jobs: $JOBS"
  echo "  drone controller: $DRONE_CONTROLLER"
  echo "  f1tenth python: $F1TENTH_PYTHON"
  echo "  f1tenth map: $F1TENTH_MAP"
  echo "  learned: ${REGRET_ARGS[*]}"
  echo

  python - "$F1TENTH_PYTHON" <<'PY'
import sys
from pathlib import Path

from pzr.experiments.robotics_probe import _f1tenth_status, _sidecar_status

f1_python = Path(sys.argv[1])
checks = {
    "safe_control": _sidecar_status(),
    "f1tenth": _f1tenth_status(f1_python),
}
for name, status in checks.items():
    print(f"{name} sidecar:")
    for key, value in status.items():
        if key in {"stdout", "stderr"} and isinstance(value, str) and len(value) > 400:
            value = value[-400:]
        print(f"  {key}: {value}")
    if not status.get("available", False):
        raise SystemExit(f"{name} sidecar is unavailable")
PY

  cat > "$OUT_DIR/manifest.txt" <<EOF
run_id=$RUN_ID
out_dir=$OUT_DIR
length=$LENGTH
seeds=$SEEDS
budgets=$BUDGETS
horizon=$HORIZON
beam_width=$BEAM_WIDTH
jobs=$JOBS
drone_controller=$DRONE_CONTROLLER
f1tenth_python=$F1TENTH_PYTHON
f1tenth_map=$F1TENTH_MAP
method_set=headline
top3=girard,methA,scott
learned=${REGRET_ARGS[*]}
EOF

  echo
  echo "Running live robotics preflight"
  python -m pzr.experiments.robotics_replay sweep \
    --candidate all \
    --trace-source live \
    --monitor physical \
    --length 40 \
    --seed 0 \
    --seeds 2 \
    --budgets 8,30 \
    --horizon "$HORIZON" \
    --beam-width "$BEAM_WIDTH" \
    --method-set headline \
    --drone-controller "$DRONE_CONTROLLER" \
    --f1tenth-sidecar-python "$F1TENTH_PYTHON" \
    --f1tenth-map "$F1TENTH_MAP" \
    --output "$OUT_DIR/preflight-robotics"

  echo
  echo "Running live robotics headline K sweep"
  python -m pzr.experiments.robotics_replay sweep \
    --candidate all \
    --trace-source live \
    --monitor physical \
    --length "$LENGTH" \
    --seed 0 \
    --seeds "$SEEDS" \
    --budgets "$BUDGETS" \
    --horizon "$HORIZON" \
    --beam-width "$BEAM_WIDTH" \
    --method-set headline \
    --drone-controller "$DRONE_CONTROLLER" \
    --f1tenth-sidecar-python "$F1TENTH_PYTHON" \
    --f1tenth-map "$F1TENTH_MAP" \
    "${REGRET_ARGS[@]}" \
    --output "$OUT_DIR/robotics"

  echo
  echo "Running omni headline K sweep"
  python -m pzr.cli \
    --profile standard \
    --scenario omni_robot \
    --method-set headline \
    --budget-sweep "$BUDGETS" \
    --seeds "$SEEDS" \
    --horizon "$HORIZON" \
    --beam-width "$BEAM_WIDTH" \
    --jobs "$JOBS" \
    "${REGRET_ARGS[@]}" \
    --no-progress \
    --output "$OUT_DIR/omni"

  echo
  echo "Finished PZR live headline paper evaluation"
  echo "  robotics gain table: $OUT_DIR/robotics/budget_policy_gain.csv"
  echo "  robotics renders: $OUT_DIR/robotics/render_best_static and $OUT_DIR/robotics/render_scott"
  echo "  omni sweep: $OUT_DIR/omni/budget_sweep"
  echo "  learned robotics artifacts: $OUT_DIR/robotics/budget_*/learning"
  echo "  learned omni artifacts: $OUT_DIR/omni/budget_*/learning"
  echo "  log: $LOG_PATH"
} 2>&1 | tee "$LOG_PATH"
