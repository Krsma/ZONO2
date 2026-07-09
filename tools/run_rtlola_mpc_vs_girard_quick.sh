#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${PZR_OUT_DIR:-$ROOT_DIR/results/rtlola-arm-mpc-vs-girard-four-actions-b60-80-100-20260706}"
REFERENCE_DIR="${PZR_REFERENCE_DIR:-$ROOT_DIR/results/rtlola-arm-mpc-variants-a143dd6-e6ecd0b-exact-metrics/references}"
PYTHON="${PZR_PYTHON:-$ROOT_DIR/external/miniconda3/envs/pzr-robot-arm/bin/python}"

if [[ -z "${PZR_REFERENCE_NAMESPACE:-}" ]]; then
    shopt -s nullglob
    reference_candidates=("$REFERENCE_DIR"/figure8.seed_0.*.json)
    shopt -u nullglob
    if (( ${#reference_candidates[@]} == 1 )); then
        reference_name="$(basename "${reference_candidates[0]}")"
        PZR_REFERENCE_NAMESPACE="${reference_name#figure8.seed_0.}"
        PZR_REFERENCE_NAMESPACE="${PZR_REFERENCE_NAMESPACE%.json}"
        export PZR_REFERENCE_NAMESPACE
    elif (( ${#reference_candidates[@]} > 1 )); then
        echo "multiple reusable reference namespaces found in $REFERENCE_DIR" >&2
        echo "set PZR_REFERENCE_NAMESPACE explicitly" >&2
        exit 2
    fi
fi

export PZR_OUT_DIR="$OUT_DIR"
export PZR_REFERENCE_DIR="$REFERENCE_DIR"
export PZR_BUDGETS="${PZR_BUDGETS:-60,80,100}"
export PZR_METHODS="${PZR_METHODS:-girard,mpc_terminal_beam}"
export PZR_MPC_CANDIDATES="${PZR_MPC_CANDIDATES:-scott,girard,combastel,pca}"
export PZR_EVAL_TRACES="${PZR_EVAL_TRACES:-figure8,figure8_drift,random,random_drift,square,square_drift}"
export PZR_SEEDS="${PZR_SEEDS:-1}"
export PZR_HORIZON="${PZR_HORIZON:-4}"
export PZR_BEAM_WIDTH="${PZR_BEAM_WIDTH:-4}"
export PZR_JOBS="${PZR_JOBS:-4}"
export PZR_SKIP_LEARNING=1

"$ROOT_DIR/tools/run_rtlola_robot_arm_fpr_overnight.sh"

"$PYTHON" - "$OUT_DIR" <<'PY'
from pathlib import Path
import sys

import numpy as np
import pandas as pd

output = Path(sys.argv[1])
summary = pd.read_csv(output / "combined_summary.csv")
rows = []
for (trace_kind, budget), frame in summary.groupby(
    ["trace_kind", "budget"],
    sort=True,
):
    indexed = frame.set_index("method")
    if not {"girard", "mpc_terminal_beam"} <= set(indexed.index):
        continue
    girard = indexed.loc["girard"]
    mpc = indexed.loc["mpc_terminal_beam"]
    row = {
        "trace_kind": trace_kind,
        "budget": int(budget),
        "reference_positive_count": int(girard["reference_positive_count"]),
        "reference_negative_count": int(girard["reference_negative_count"]),
        "girard_false_positive_count": int(girard["false_positive_count"]),
        "mpc_false_positive_count": int(mpc["false_positive_count"]),
        "girard_fpr": float(girard["fpr"]),
        "mpc_fpr": float(mpc["fpr"]),
        "fpr_absolute_improvement": float(girard["fpr"] - mpc["fpr"]),
        "girard_false_negative_count": int(girard["false_negative_count"]),
        "mpc_false_negative_count": int(mpc["false_negative_count"]),
        "girard_fnr": float(girard["fnr"]),
        "mpc_fnr": float(mpc["fnr"]),
    }
    for metric in (
        "mean_approx_loss",
        "final_approx_loss",
        "max_approx_loss",
        "sum_approx_loss",
        "mean_state_width",
        "max_state_width",
    ):
        girard_value = float(girard[metric])
        mpc_value = float(mpc[metric])
        improvement = girard_value - mpc_value
        row[f"girard_{metric}"] = girard_value
        row[f"mpc_{metric}"] = mpc_value
        row[f"{metric}_absolute_improvement"] = improvement
        row[f"{metric}_relative_improvement"] = (
            improvement / girard_value
            if np.isfinite(girard_value) and girard_value != 0.0
            else float("nan")
        )
    row["native_loss_absolute_improvement"] = row[
        "mean_approx_loss_absolute_improvement"
    ]
    row["native_loss_relative_improvement"] = row[
        "mean_approx_loss_relative_improvement"
    ]
    rows.append(row)

comparison = pd.DataFrame(rows).sort_values(["trace_kind", "budget"])
path = output / "fpr_native_loss_comparison.csv"
comparison.to_csv(path, index=False)
print(f"Focused comparison: {path}")
print(comparison.to_string(index=False))
PY
