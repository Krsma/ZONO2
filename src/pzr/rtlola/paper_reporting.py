"""Validated joined reporting for primary PRP and online-MPC artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from pzr.artifact_io import write_csv_atomic, write_json_atomic
from pzr.rtlola.policy_evaluation import POLICY_EVALUATION_SCHEMA


STATIC_METHODS = ("girard", "scott", "pca", "combastel")
LEARNED_METHOD = "pairwise_ranking_policy"
PRIMARY_TEACHER = "mpc_terminal_full_width"
ORACLE_METHODS = ("mpc_terminal_beam", "mpc_terminal_full_width")
PREDICTIVE_METHODS = (
    "mpc_terminal_beam_predictive_hold",
    "mpc_terminal_beam_predictive_linear",
    "mpc_terminal_beam_predictive_quadratic",
)
PRIMARY_METHODS = (*STATIC_METHODS, PRIMARY_TEACHER, LEARNED_METHOD)
ADDON_METHODS = ("girard", LEARNED_METHOD, *ORACLE_METHODS, *PREDICTIVE_METHODS)
JOINED_METHODS = (*STATIC_METHODS, *ORACLE_METHODS, *PREDICTIVE_METHODS, LEARNED_METHOD)
HEADLINE_METHODS = (
    *STATIC_METHODS,
    *ORACLE_METHODS,
    "mpc_terminal_beam_predictive_linear",
    LEARNED_METHOD,
)
SCIENTIFIC_RUNTIME_COLUMNS = {
    "decision_time_ms", "binding_runtime_ns", "total_time_ms",
}


@dataclass(frozen=True)
class PaperReportConfig:
    primary: Path
    mpc_addon: Path
    output: Path


def write_paper_reports(config: PaperReportConfig) -> pd.DataFrame:
    """Validate both evaluations and write table-ready joined CSV artifacts."""
    primary_manifest, primary_summary, primary_timeseries = _load_evaluation(
        config.primary, PRIMARY_METHODS,
    )
    addon_manifest, addon_summary, addon_timeseries = _load_evaluation(
        config.mpc_addon, ADDON_METHODS,
    )
    _validate_cross_artifact_identity(primary_manifest, addon_manifest)
    _validate_anchor_cells(
        config.primary,
        config.mpc_addon,
        primary_manifest,
        methods=("girard", LEARNED_METHOD),
    )

    primary_keep = primary_summary[primary_summary["method"].isin(
        (*STATIC_METHODS, LEARNED_METHOD),
    )].copy()
    addon_keep = addon_summary[addon_summary["method"].isin(
        (*ORACLE_METHODS, *PREDICTIVE_METHODS),
    )].copy()
    primary_keep.insert(0, "source_artifact", "primary")
    addon_keep.insert(0, "source_artifact", "mpc_addon")
    joined = pd.concat([primary_keep, addon_keep], ignore_index=True)
    _validate_method_matrix(
        joined,
        JOINED_METHODS,
        tuple(primary_manifest["trace_kinds"]),
        tuple(int(value) for value in primary_manifest["budgets"]),
    )

    config.output.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(joined, config.output / "paper_dataset.csv")
    write_csv_atomic(
        joined[joined["method"].isin(HEADLINE_METHODS)].copy(),
        config.output / "main_table.csv",
    )
    write_csv_atomic(
        joined[joined["method"].isin(PREDICTIVE_METHODS)].copy(),
        config.output / "prediction_ablation.csv",
    )
    safety_columns = [
        column for column in (
            "source_artifact", "trace_kind", "budget", "method", "fnr",
            "false_negative_count", "reference_positive_count", "fallback_count",
            "reducer_failure_count", "infeasible_candidate_count",
        ) if column in joined
    ]
    write_csv_atomic(joined[safety_columns], config.output / "safety_accounting.csv")
    _write_prediction_reports(config.mpc_addon, config.output)
    write_json_atomic({
        "schema": "pzr.paper-policy-mpc-report.v1",
        "policy_evaluation_schema": POLICY_EVALUATION_SCHEMA,
        "primary_experiment_fingerprint": primary_manifest["experiment_fingerprint"],
        "mpc_addon_experiment_fingerprint": addon_manifest["experiment_fingerprint"],
        "trace_kinds": primary_manifest["trace_kinds"],
        "budgets": primary_manifest["budgets"],
        "methods": list(JOINED_METHODS),
        "headline_online_mpc": "mpc_terminal_beam_predictive_linear",
        "cell_count": len(joined),
        "expected_cell_count": (
            len(primary_manifest["trace_kinds"])
            * len(primary_manifest["budgets"])
            * len(JOINED_METHODS)
        ),
        "anchor_methods": ["girard", LEARNED_METHOD],
        "anchor_equality": "exact_excluding_runtime_columns",
    }, config.output / "manifest.json")
    return joined


def _load_evaluation(
    root: Path,
    expected_methods: Sequence[str],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"policy evaluation manifest is missing: {root}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != POLICY_EVALUATION_SCHEMA:
        raise ValueError(f"unsupported policy evaluation schema: {root}")
    if manifest.get("failure_count") != 0:
        raise ValueError(f"policy evaluation contains failures: {root}")
    declared_methods = (*manifest.get("benchmark_methods", ()), *manifest.get("models", {}))
    if set(declared_methods) != set(expected_methods):
        raise ValueError(f"policy evaluation methods differ: {root}")
    summary_path = root / "summary.csv"
    timeseries_path = root / "timeseries.csv"
    if not summary_path.is_file() or not timeseries_path.is_file():
        raise ValueError(f"policy evaluation tables are incomplete: {root}")
    summary = pd.read_csv(summary_path)
    timeseries = pd.read_csv(timeseries_path)
    traces = tuple(manifest.get("trace_kinds", ()))
    budgets = tuple(int(value) for value in manifest.get("budgets", ()))
    _validate_method_matrix(summary, expected_methods, traces, budgets)
    expected_cells = len(traces) * len(budgets) * len(expected_methods)
    if manifest.get("cell_count") != expected_cells or len(summary) != expected_cells:
        raise ValueError(f"policy evaluation cell count is incomplete: {root}")
    for trace in traces:
        for budget in budgets:
            for method in expected_methods:
                if not (root / "cells" / trace / f"budget-{budget}" / method / "manifest.json").is_file():
                    raise ValueError(f"policy evaluation cell manifest is missing: {root}")
    return manifest, summary, timeseries


def _validate_method_matrix(
    frame: pd.DataFrame,
    methods: Sequence[str],
    traces: Sequence[str],
    budgets: Sequence[int],
) -> None:
    required = {(trace, budget, method) for trace in traces for budget in budgets for method in methods}
    actual = {
        (str(row.trace_kind), int(row.budget), str(row.method))
        for row in frame[["trace_kind", "budget", "method"]].itertuples(index=False)
    }
    if actual != required or len(frame) != len(required):
        raise ValueError("evaluation trace/budget/method cells do not form the declared matrix")


def _validate_cross_artifact_identity(
    primary: dict[str, object],
    addon: dict[str, object],
) -> None:
    for key in (
        "trace_kinds", "budgets", "candidate_names", "pzr_source_sha256",
        "binding_revision", "interpreter_revision", "binding_build_profile",
    ):
        if primary.get(key) != addon.get(key):
            raise ValueError(f"primary and MPC add-on differ in {key}")
    primary_model = primary.get("models", {}).get(LEARNED_METHOD, {})
    addon_model = addon.get("models", {}).get(LEARNED_METHOD, {})
    if primary_model.get("sha256") != addon_model.get("sha256"):
        raise ValueError("primary and MPC add-on model hashes differ")


def _validate_anchor_cells(
    primary: Path,
    addon: Path,
    manifest: dict[str, object],
    *,
    methods: Sequence[str],
) -> None:
    for trace in manifest["trace_kinds"]:
        for budget in manifest["budgets"]:
            for method in methods:
                primary_cell = primary / "cells" / trace / f"budget-{budget}" / method
                addon_cell = addon / "cells" / trace / f"budget-{budget}" / method
                left_identity = json.loads((primary_cell / "manifest.json").read_text())
                right_identity = json.loads((addon_cell / "manifest.json").read_text())
                for key in (
                    "trace_kind", "length", "budget", "method", "candidate_names",
                    "reference_mode", "exact_reference_contract", "model_sha256",
                    "pzr_source_sha256", "exact_reference_sha256", "binding_revision",
                    "interpreter_revision", "binding_build_profile",
                ):
                    if left_identity.get(key) != right_identity.get(key):
                        raise ValueError(f"anchor {method} differs in {key}")
                _assert_scientific_equal(
                    pd.read_csv(primary_cell / "timeseries.csv"),
                    pd.read_csv(addon_cell / "timeseries.csv"),
                    label=f"{trace}/budget-{budget}/{method} timeseries",
                )
                _assert_scientific_equal(
                    pd.read_csv(primary_cell / "summary.csv"),
                    pd.read_csv(addon_cell / "summary.csv"),
                    label=f"{trace}/budget-{budget}/{method} summary",
                )


def _assert_scientific_equal(left: pd.DataFrame, right: pd.DataFrame, *, label: str) -> None:
    columns = sorted((set(left.columns) & set(right.columns)) - SCIENTIFIC_RUNTIME_COLUMNS)
    if set(left.columns) != set(right.columns):
        raise ValueError(f"anchor columns differ: {label}")
    try:
        pd.testing.assert_frame_equal(
            left[columns].reset_index(drop=True),
            right[columns].reset_index(drop=True),
            check_exact=True,
        )
    except AssertionError as exc:
        raise ValueError(f"anchor scientific outputs differ: {label}") from exc


def _write_prediction_reports(addon: Path, output: Path) -> None:
    raw_path = addon / "input_prediction_errors.csv"
    aggregate_path = addon / "input_prediction_error_summary.csv"
    if not raw_path.is_file() or not aggregate_path.is_file():
        raise ValueError("MPC add-on prediction diagnostics are incomplete")
    raw = pd.read_csv(raw_path)
    aggregate = pd.read_csv(aggregate_path)
    if raw.empty or aggregate.empty:
        raise ValueError("MPC add-on prediction diagnostics are empty")
    write_csv_atomic(raw, output / "input_prediction_errors.csv")
    write_csv_atomic(aggregate, output / "input_prediction_error_summary.csv")
