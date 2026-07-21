from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pzr.rtlola.paper_reporting import (
    ADDON_METHODS,
    PRIMARY_METHODS,
    PaperReportConfig,
    write_paper_reports,
)
from pzr.rtlola.policy_evaluation import POLICY_EVALUATION_SCHEMA


def _write_evaluation(root: Path, methods, *, anchor_offset: float = 0.0):
    benchmark = tuple(method for method in methods if method != "pairwise_ranking_policy")
    manifest = {
        "schema": POLICY_EVALUATION_SCHEMA,
        "failure_count": 0,
        "trace_kinds": ["figure8"],
        "budgets": [40],
        "candidate_names": ["girard", "scott", "pca", "combastel"],
        "benchmark_methods": list(benchmark),
        "models": {"pairwise_ranking_policy": {"sha256": "model-hash"}},
        "pzr_source_sha256": "source-hash",
        "binding_revision": "binding",
        "interpreter_revision": "interpreter",
        "binding_build_profile": "release",
        "experiment_fingerprint": root.name,
        "cell_count": len(methods),
    }
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps(manifest))
    summaries = []
    timeseries = []
    for index, method in enumerate(methods):
        anchor = method in {"girard", "pairwise_ranking_policy"}
        value = float(index if not anchor else (0 if method == "girard" else 1))
        value += anchor_offset if anchor else 0.0
        summary = {
            "trace_kind": "figure8", "budget": 40, "method": method,
            "fpr": 0.0, "fnr": value, "false_negative_count": int(value),
            "reference_positive_count": 2, "mean_approx_loss": value,
            "final_approx_loss": value, "max_approx_loss": value,
            "sum_approx_loss": value, "mean_state_width": value,
            "max_state_width": value, "fallback_count": 0,
            "reducer_failure_count": 0, "infeasible_candidate_count": 0,
            "total_time_ms": float(index + 10),
        }
        row = {
            "trace_kind": "figure8", "budget": 40, "method": method,
            "step": 0, "approx_loss": value, "state_width": value,
            "fallback_used": False, "infeasible_candidate_count": 0,
            "decision_time_ms": float(index + 10), "binding_runtime_ns": 1.0,
        }
        summaries.append(summary)
        timeseries.append(row)
        cell = root / "cells" / "figure8" / "budget-40" / method
        cell.mkdir(parents=True)
        identity = {
            "trace_kind": "figure8", "length": 1, "budget": 40,
            "method": method,
            "candidate_names": ["girard", "scott", "pca", "combastel"],
            "reference_mode": "exact",
            "exact_reference_contract": (
                "trigger_booleans_and_logical_row_center_dynamic_total_radius_v2"
            ),
            "model_sha256": "model-hash" if method == "pairwise_ranking_policy" else None,
            "pzr_source_sha256": "source-hash",
            "exact_reference_sha256": "reference-hash",
            "binding_revision": "binding", "interpreter_revision": "interpreter",
            "binding_build_profile": "release",
        }
        (cell / "manifest.json").write_text(json.dumps(identity))
        pd.DataFrame([summary]).to_csv(cell / "summary.csv", index=False)
        pd.DataFrame([row]).to_csv(cell / "timeseries.csv", index=False)
    pd.DataFrame(summaries).to_csv(root / "summary.csv", index=False)
    pd.DataFrame(timeseries).to_csv(root / "timeseries.csv", index=False)
    if tuple(methods) == ADDON_METHODS:
        prediction = pd.DataFrame([{
            "trace_kind": "figure8", "predictor": "linear", "lead": 1,
            "channel_index": 1, "channel_name": "a1m", "error": 0.1,
        }])
        prediction.to_csv(root / "input_prediction_errors.csv", index=False)
        pd.DataFrame([{"trace_kind": "figure8", "mae": 0.1}]).to_csv(
            root / "input_prediction_error_summary.csv", index=False,
        )


def test_paper_reporting_joins_unique_methods_and_checks_anchors(tmp_path):
    primary = tmp_path / "primary"
    addon = tmp_path / "addon"
    output = tmp_path / "reports"
    _write_evaluation(primary, PRIMARY_METHODS)
    _write_evaluation(addon, ADDON_METHODS)

    joined = write_paper_reports(PaperReportConfig(primary, addon, output))

    assert len(joined) == 11
    assert len(pd.read_csv(output / "main_table.csv")) == 9
    assert len(pd.read_csv(output / "prediction_ablation.csv")) == 3
    assert len(pd.read_csv(output / "safety_accounting.csv")) == 11
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["cell_count"] == 11
    assert manifest["headline_online_mpc"] == "mpc_terminal_beam_predictive_linear"


def test_paper_reporting_rejects_anchor_differences(tmp_path):
    primary = tmp_path / "primary"
    addon = tmp_path / "addon"
    _write_evaluation(primary, PRIMARY_METHODS)
    _write_evaluation(addon, ADDON_METHODS, anchor_offset=1.0)

    with pytest.raises(ValueError, match="anchor scientific outputs differ"):
        write_paper_reports(PaperReportConfig(primary, addon, tmp_path / "reports"))
