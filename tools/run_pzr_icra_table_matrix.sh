#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${PZR_RUN_ID:-icra-table-matrix-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${PZR_OUT_DIR:-results/$RUN_ID}"
LOG_PATH="${PZR_LOG_PATH:-results/logs/$RUN_ID.log}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/results/matplotlib-cache}"

LENGTH="${PZR_LENGTH:-250}"
SEEDS="${PZR_SEEDS:-10}"
BUDGETS="${PZR_BUDGETS:-8,10,12,16,20,24,30}"
HORIZONS="${PZR_HORIZONS:-1,2,4,8,12}"
HORIZON_BUDGETS="${PZR_HORIZON_BUDGETS:-8,16,30}"
PRIMARY_HORIZON="${PZR_PRIMARY_HORIZON:-4}"
BEAM_WIDTH="${PZR_BEAM_WIDTH:-4}"
JOBS="${PZR_JOBS:-4}"

PRIMARY_METHOD_SET="${PZR_PRIMARY_METHOD_SET:-paper_core}"
HORIZON_METHOD_SET="${PZR_HORIZON_METHOD_SET:-paper_core}"
RESUME="${PZR_RESUME:-1}"

WITH_REGRET="${PZR_WITH_REGRET:-0}"
REGRET_ORACLE="${PZR_REGRET_ORACLE:-beam3}"
REGRET_ITERATIONS="${PZR_REGRET_ITERATIONS:-2}"
REGRET_EPOCHS="${PZR_REGRET_EPOCHS:-80}"
REGRET_LOSS="${PZR_REGRET_LOSS:-pairwise}"
REGRET_TRAIN_SEEDS="${PZR_REGRET_TRAIN_SEEDS:-$SEEDS}"
REGRET_EVAL_SEEDS="${PZR_REGRET_EVAL_SEEDS:-$SEEDS}"
REGRET_BUDGETS="${PZR_REGRET_BUDGETS:-8,16,30}"

INCLUDE_SEQUENCE_AUDIT="${PZR_INCLUDE_SEQUENCE_AUDIT:-0}"
SEQUENCE_AUDIT_BUDGETS="${PZR_SEQUENCE_AUDIT_BUDGETS:-8,16,30}"
SEQUENCE_AUDIT_SEEDS="${PZR_SEQUENCE_AUDIT_SEEDS:-3}"

INCLUDE_LIVE_SMOKE="${PZR_INCLUDE_LIVE_SMOKE:-0}"
F1TENTH_PYTHON="${PZR_F1TENTH_PYTHON:-external/f1tenth-py38-venv/bin/python}"

mkdir -p "$(dirname "$LOG_PATH")" "$OUT_DIR" "$MPLCONFIGDIR"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*"
}

complete_marker() {
  local cell_dir="$1"
  echo "$cell_dir/.complete"
}

run_cell() {
  local label="$1"
  local cell_dir="$2"
  shift 2

  local marker
  marker="$(complete_marker "$cell_dir")"
  if [[ "$RESUME" == "1" && -f "$marker" ]]; then
    log "SKIP $label"
    return 0
  fi

  mkdir -p "$cell_dir"
  log "START $label"
  "$@"
  touch "$marker"
  log "DONE  $label"
}

run_cell_or_legacy() {
  local label="$1"
  local cell_dir="$2"
  local legacy_file="$3"
  shift 3

  local marker
  marker="$(complete_marker "$cell_dir")"
  if [[ "$RESUME" == "1" && -f "$legacy_file" && ! -f "$marker" ]]; then
    mkdir -p "$cell_dir"
    touch "$marker"
    log "SKIP $label (legacy artifact found: $legacy_file)"
    return 0
  fi

  run_cell "$label" "$cell_dir" "$@"
}

csv_to_array() {
  local csv="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a out_ref <<< "$csv"
}

robotics_sweep_cell() {
  local output="$1"
  local budget="$2"
  local horizon="$3"
  local method_set="$4"
  shift 4
  python -m pzr.experiments.robotics_replay sweep \
    --candidate all \
    --trace-source procedural \
    --monitor physical \
    --scenario-family stress \
    --length "$LENGTH" \
    --seed 0 \
    --seeds "$SEEDS" \
    --budgets "$budget" \
    --horizon "$horizon" \
    --beam-width "$BEAM_WIDTH" \
    --method-set "$method_set" \
    "$@" \
    --no-render \
    --output "$output"
}

omni_sweep_cell() {
  local output="$1"
  local budget="$2"
  local horizon="$3"
  local method_set="$4"
  shift 4
  python -m pzr.cli \
    --profile standard \
    --scenario omni_robot \
    --method-set "$method_set" \
    --length "$LENGTH" \
    --budget-sweep "$budget" \
    --seeds "$SEEDS" \
    --horizon "$horizon" \
    --beam-width "$BEAM_WIDTH" \
    --jobs "$JOBS" \
    "$@" \
    --no-progress \
    --output "$output"
}

{
  log "Starting PZR ICRA table matrix"
  echo "  output: $OUT_DIR"
  echo "  length: $LENGTH"
  echo "  seeds: $SEEDS"
  echo "  budgets: $BUDGETS"
  echo "  primary horizon: $PRIMARY_HORIZON"
  echo "  horizon sweep: $HORIZONS at budgets $HORIZON_BUDGETS"
  echo "  beam width: $BEAM_WIDTH"
  echo "  primary method set: $PRIMARY_METHOD_SET"
  echo "  horizon method set: $HORIZON_METHOD_SET"
  echo "  resume: $RESUME"
  echo "  learned stage enabled: $WITH_REGRET"
  echo "  exact sequence audit enabled: $INCLUDE_SEQUENCE_AUDIT"

  cat > "$OUT_DIR/manifest.txt" <<EOF
run_id=$RUN_ID
out_dir=$OUT_DIR
length=$LENGTH
seeds=$SEEDS
budgets=$BUDGETS
horizons=$HORIZONS
horizon_budgets=$HORIZON_BUDGETS
primary_horizon=$PRIMARY_HORIZON
beam_width=$BEAM_WIDTH
jobs=$JOBS
primary_method_set=$PRIMARY_METHOD_SET
horizon_method_set=$HORIZON_METHOD_SET
resume=$RESUME
learned_enabled=$WITH_REGRET
regret_oracle=$REGRET_ORACLE
regret_iterations=$REGRET_ITERATIONS
regret_epochs=$REGRET_EPOCHS
regret_loss=$REGRET_LOSS
regret_train_seeds=$REGRET_TRAIN_SEEDS
regret_eval_seeds=$REGRET_EVAL_SEEDS
regret_budgets=$REGRET_BUDGETS
sequence_audit_enabled=$INCLUDE_SEQUENCE_AUDIT
sequence_audit_budgets=$SEQUENCE_AUDIT_BUDGETS
sequence_audit_seeds=$SEQUENCE_AUDIT_SEEDS
EOF

  csv_to_array "$BUDGETS" BUDGET_LIST
  csv_to_array "$HORIZONS" HORIZON_LIST
  csv_to_array "$HORIZON_BUDGETS" HORIZON_BUDGET_LIST

  echo
  log "Primary h=$PRIMARY_HORIZON budget sweep: robotics procedural stress"
  for budget in "${BUDGET_LIST[@]}"; do
    run_cell_or_legacy \
      "robotics primary h=$PRIMARY_HORIZON k=$budget method_set=$PRIMARY_METHOD_SET" \
      "$OUT_DIR/robotics_k_sweep_h${PRIMARY_HORIZON}/k${budget}" \
      "$OUT_DIR/robotics_k_sweep_h${PRIMARY_HORIZON}/budget_${budget}/summary.csv" \
      robotics_sweep_cell \
      "$OUT_DIR/robotics_k_sweep_h${PRIMARY_HORIZON}/k${budget}" \
      "$budget" \
      "$PRIMARY_HORIZON" \
      "$PRIMARY_METHOD_SET" \
      --learned-mode none
  done

  echo
  log "Primary h=$PRIMARY_HORIZON budget sweep: omni_robot"
  for budget in "${BUDGET_LIST[@]}"; do
    run_cell_or_legacy \
      "omni primary h=$PRIMARY_HORIZON k=$budget method_set=$PRIMARY_METHOD_SET" \
      "$OUT_DIR/omni_k_sweep_h${PRIMARY_HORIZON}/k${budget}" \
      "$OUT_DIR/omni_k_sweep_h${PRIMARY_HORIZON}/budget_${budget}/omni_robot/aggregate.csv" \
      omni_sweep_cell \
      "$OUT_DIR/omni_k_sweep_h${PRIMARY_HORIZON}/k${budget}" \
      "$budget" \
      "$PRIMARY_HORIZON" \
      "$PRIMARY_METHOD_SET" \
      --learned-mode none
  done

  echo
  log "Prediction horizon sweep"
  for horizon in "${HORIZON_LIST[@]}"; do
    for budget in "${HORIZON_BUDGET_LIST[@]}"; do
      run_cell_or_legacy \
        "robotics horizon h=$horizon k=$budget method_set=$HORIZON_METHOD_SET" \
        "$OUT_DIR/horizon_scan/h${horizon}/robotics/k${budget}" \
        "$OUT_DIR/horizon_scan/h${horizon}/robotics/budget_${budget}/summary.csv" \
        robotics_sweep_cell \
        "$OUT_DIR/horizon_scan/h${horizon}/robotics/k${budget}" \
        "$budget" \
        "$horizon" \
        "$HORIZON_METHOD_SET" \
        --learned-mode none

      run_cell_or_legacy \
        "omni horizon h=$horizon k=$budget method_set=$HORIZON_METHOD_SET" \
        "$OUT_DIR/horizon_scan/h${horizon}/omni/k${budget}" \
        "$OUT_DIR/horizon_scan/h${horizon}/omni/budget_${budget}/omni_robot/aggregate.csv" \
        omni_sweep_cell \
        "$OUT_DIR/horizon_scan/h${horizon}/omni/k${budget}" \
        "$budget" \
        "$horizon" \
        "$HORIZON_METHOD_SET" \
        --learned-mode none
    done
  done

  if [[ "$INCLUDE_SEQUENCE_AUDIT" == "1" ]]; then
    echo
    log "Optional exact mpc_sequence3 audit"
    csv_to_array "$SEQUENCE_AUDIT_BUDGETS" SEQUENCE_AUDIT_BUDGET_LIST
    for budget in "${SEQUENCE_AUDIT_BUDGET_LIST[@]}"; do
      run_cell \
        "robotics sequence audit h=$PRIMARY_HORIZON k=$budget seeds=$SEQUENCE_AUDIT_SEEDS" \
        "$OUT_DIR/sequence3_audit/robotics/k${budget}" \
        python -m pzr.experiments.robotics_replay sweep \
          --candidate all \
          --trace-source procedural \
          --monitor physical \
          --scenario-family stress \
          --length "$LENGTH" \
          --seed 0 \
          --seeds "$SEQUENCE_AUDIT_SEEDS" \
          --budgets "$budget" \
          --horizon "$PRIMARY_HORIZON" \
          --beam-width "$BEAM_WIDTH" \
          --method-set headline \
          --learned-mode none \
          --no-render \
          --output "$OUT_DIR/sequence3_audit/robotics/k${budget}"

      run_cell \
        "omni sequence audit h=$PRIMARY_HORIZON k=$budget seeds=$SEQUENCE_AUDIT_SEEDS" \
        "$OUT_DIR/sequence3_audit/omni/k${budget}" \
        python -m pzr.cli \
          --profile standard \
          --scenario omni_robot \
          --method-set headline \
          --length "$LENGTH" \
          --budget-sweep "$budget" \
          --seeds "$SEQUENCE_AUDIT_SEEDS" \
          --horizon "$PRIMARY_HORIZON" \
          --beam-width "$BEAM_WIDTH" \
          --jobs "$JOBS" \
          --learned-mode none \
          --no-progress \
          --output "$OUT_DIR/sequence3_audit/omni/k${budget}"
    done
  fi

  if [[ "$WITH_REGRET" == "1" ]]; then
    echo
    log "Optional regret/ranking distillation stage"
    csv_to_array "$REGRET_BUDGETS" REGRET_BUDGET_LIST
    REGRET_ARGS=(
      --learned-mode regret
      --regret-oracle "$REGRET_ORACLE"
      --regret-iterations "$REGRET_ITERATIONS"
      --regret-epochs "$REGRET_EPOCHS"
      --regret-loss "$REGRET_LOSS"
      --regret-train-seeds "$REGRET_TRAIN_SEEDS"
      --regret-eval-seeds "$REGRET_EVAL_SEEDS"
    )
    for budget in "${REGRET_BUDGET_LIST[@]}"; do
      run_cell \
        "robotics regret h=$PRIMARY_HORIZON k=$budget oracle=$REGRET_ORACLE" \
        "$OUT_DIR/regret_stage/robotics/k${budget}" \
        robotics_sweep_cell \
        "$OUT_DIR/regret_stage/robotics/k${budget}" \
        "$budget" \
        "$PRIMARY_HORIZON" \
        "$PRIMARY_METHOD_SET" \
        "${REGRET_ARGS[@]}"

      run_cell \
        "omni regret h=$PRIMARY_HORIZON k=$budget oracle=$REGRET_ORACLE" \
        "$OUT_DIR/regret_stage/omni/k${budget}" \
        omni_sweep_cell \
        "$OUT_DIR/regret_stage/omni/k${budget}" \
        "$budget" \
        "$PRIMARY_HORIZON" \
        "$PRIMARY_METHOD_SET" \
        "${REGRET_ARGS[@]}"
    done
  fi

  if [[ "$INCLUDE_LIVE_SMOKE" == "1" ]]; then
    echo
    run_cell \
      "optional live sidecar smoke" \
      "$OUT_DIR/live_smoke" \
      python -m pzr.experiments.robotics_replay sweep \
        --candidate all \
        --trace-source live \
        --monitor physical \
        --scenario-family stress \
        --length 40 \
        --seed 0 \
        --seeds 2 \
        --budgets 8 \
        --horizon "$PRIMARY_HORIZON" \
        --beam-width "$BEAM_WIDTH" \
        --method-set paper_core \
        --f1tenth-sidecar-python "$F1TENTH_PYTHON" \
        --learned-mode none \
        --no-render \
        --output "$OUT_DIR/live_smoke"
  fi

  echo
  log "Building LaTeX table bundle"
  python -m pzr.experiments.paper_tables \
    --input "$OUT_DIR/robotics_k_sweep_h${PRIMARY_HORIZON}" \
            "$OUT_DIR/omni_k_sweep_h${PRIMARY_HORIZON}" \
            "$OUT_DIR/horizon_scan" \
            "$OUT_DIR/regret_stage" \
    --output "$OUT_DIR/tables"

  log "Finished PZR ICRA table matrix"
  echo "  combined table: $OUT_DIR/tables/combined_summary.csv"
  echo "  LaTeX overview: $OUT_DIR/tables/overview.tex"
  echo "  log: $LOG_PATH"
} 2>&1 | tee "$LOG_PATH"
