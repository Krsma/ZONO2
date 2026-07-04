"""Consolidate resumable RTLola sweep artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pzr.rtlola.benchmark import trigger_confusion
from pzr.rtlola.scenarios import scenario_by_name


def consolidate_sweep(root: Path, scenario: str = "robot_arm") -> None:
    """Write compact fidelity and trigger tables from completed cells."""
    summaries = _read_tables(root / "runs", "summary.csv")
    learning_summary = _read_optional(
        root / "learning_stage" / "learning" / scenario
        / "regret_eval_summary.csv",
    )
    if learning_summary is not None:
        summaries.append(learning_summary)
    combined = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    if not combined.empty:
        identity = ["method", "budget", "trace_kind", "seed"]
        combined = combined.drop_duplicates(
            subset=identity,
            keep="last",
        ).sort_values([
            column
            for column in (
                "trace_kind",
                "budget",
                "method",
                "seed",
            )
            if column in combined
        ])
    combined.to_csv(root / "combined_summary.csv", index=False)

    confusion = _read_tables(root / "runs", "trigger_confusion.csv")
    learning_timeseries = _read_optional(
        root / "learning_stage" / "learning" / scenario
        / "regret_eval_timeseries.csv",
    )
    if learning_timeseries is not None:
        confusion.append(trigger_confusion(
            learning_timeseries,
            scenario_by_name(scenario).trigger_keys,
        ))
    combined_confusion = (
        pd.concat(confusion, ignore_index=True)
        if confusion else pd.DataFrame()
    )
    if not combined_confusion.empty:
        combined_confusion = combined_confusion.sort_values(
            [
                column
                for column in (
                    "trace_kind",
                    "budget",
                    "method",
                    "trigger_key",
                )
                if column in combined_confusion
            ],
        )
    combined_confusion.to_csv(root / "combined_trigger_confusion.csv", index=False)

    failures = _read_tables(root / "runs", "run_failures.csv")
    combined_failures = (
        pd.concat(failures, ignore_index=True)
        if failures else pd.DataFrame()
    )
    combined_failures.to_csv(root / "combined_run_failures.csv", index=False)

    reducer_counts = _reducer_counts(root, scenario)
    reducer_counts.to_csv(root / "combined_reducer_counts.csv", index=False)
    _method_comparison(combined).to_csv(
        root / "method_comparison.csv",
        index=False,
    )
    _mpc_action_composition(reducer_counts).to_csv(
        root / "mpc_action_composition.csv",
        index=False,
    )
    _mpc_comparison(combined).to_csv(
        root / "mpc_vs_static_fpr.csv",
        index=False,
    )
    _fidelity_comparison(combined).to_csv(
        root / "mpc_vs_static_fidelity.csv",
        index=False,
    )
    _learned_comparison(combined).to_csv(
        root / "learned_vs_mpc_fidelity.csv",
        index=False,
    )


def _read_tables(root: Path, filename: str) -> list[pd.DataFrame]:
    if not root.exists():
        return []
    frames: list[pd.DataFrame] = []
    for path in sorted(root.rglob(filename)):
        if path.stat().st_size == 0:
            continue
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frames.append(frame)
    return frames


def _read_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path)


def _reducer_counts(root: Path, scenario: str) -> pd.DataFrame:
    frames = _read_tables(root / "runs", "timeseries.csv")
    learned = _read_optional(
        root / "learning_stage" / "learning" / scenario
        / "regret_eval_timeseries.csv",
    )
    if learned is not None:
        frames.append(learned)
    if not frames:
        return pd.DataFrame()
    timeseries = pd.concat(frames, ignore_index=True)
    columns = [
        column
        for column in (
            "trace_kind",
            "budget",
            "method",
            "reducer_used",
        )
        if column in timeseries
    ]
    return (
        timeseries.groupby(columns, dropna=False)
        .size()
        .rename("step_count")
        .reset_index()
        .sort_values(columns)
    )


def _method_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """Return one compact performance row per trace, budget, and method."""
    group_columns = [
        column
        for column in ("trace_kind", "budget", "method")
        if column in summary
    ]
    metric_columns = [
        column
        for column in (
            "false_positive_rate",
            "false_negative_rate",
            "trigger_positive_rate",
            "mean_approx_loss",
            "mean_state_zonotope_approx_error",
            "mean_state_zonotope_width",
            "max_state_zonotope_width",
            "mean_generator_count",
            "mean_active_dynamic_generator_count",
            "total_reductions",
            "total_time_ms",
            "fallback_count",
            "fallback_rate",
            "reducer_failure_count",
            "infeasible_candidate_count",
        )
        if column in summary
    ]
    if len(group_columns) != 3 or not metric_columns:
        return pd.DataFrame(columns=(*group_columns, *metric_columns))
    return (
        summary.groupby(group_columns, as_index=False, dropna=False)[metric_columns]
        .mean()
        .sort_values(group_columns)
    )


def _mpc_action_composition(reducer_counts: pd.DataFrame) -> pd.DataFrame:
    """Return committed MPC action shares, including and excluding no-op steps."""
    columns = [
        "trace_kind",
        "budget",
        "reducer_used",
        "step_count",
        "total_step_count",
        "reduction_step_count",
        "step_share",
        "reduction_share",
    ]
    required = {
        "trace_kind",
        "budget",
        "method",
        "reducer_used",
        "step_count",
    }
    if reducer_counts.empty or not required <= set(reducer_counts.columns):
        return pd.DataFrame(columns=columns)
    mpc = reducer_counts[reducer_counts["method"] == "mpc_beam"].copy()
    if mpc.empty:
        return pd.DataFrame(columns=columns)
    group_columns = ["trace_kind", "budget"]
    mpc["total_step_count"] = mpc.groupby(group_columns)["step_count"].transform("sum")
    reduced = mpc["reducer_used"] != "none"
    reduction_totals = (
        mpc.loc[reduced]
        .groupby(group_columns)["step_count"]
        .sum()
        .rename("reduction_step_count")
    )
    mpc = mpc.join(reduction_totals, on=group_columns)
    mpc["reduction_step_count"] = mpc["reduction_step_count"].fillna(0).astype(int)
    mpc["step_share"] = mpc["step_count"] / mpc["total_step_count"]
    mpc["reduction_share"] = np.where(
        reduced & (mpc["reduction_step_count"] > 0),
        mpc["step_count"] / mpc["reduction_step_count"],
        np.nan,
    )
    return mpc[columns].sort_values([*group_columns, "reducer_used"])


def _mpc_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_columns = [
        column
        for column in ("trace_kind", "budget")
        if column in summary
    ]
    for group_key, frame in summary.groupby(group_columns, dropna=False):
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        group_values = dict(zip(group_columns, values))
        numeric = [
            "false_positive_rate",
            "false_negative_rate",
            "total_time_ms",
            "fallback_count",
            "reducer_failure_count",
        ]
        by_method = (
            frame.groupby("method", as_index=False)[numeric]
            .mean()
        )
        mpc = by_method[by_method["method"] == "mpc_beam"]
        static = by_method[
            ~by_method["method"].isin(["none", "mpc_beam", "learned_direct"])
        ]
        if mpc.empty or static.empty:
            continue
        mpc_row = mpc.iloc[0]
        finite_static = static[np.isfinite(static["false_positive_rate"])]
        if finite_static.empty:
            continue
        static_row = finite_static.sort_values(
            ["false_positive_rate", "false_negative_rate", "total_time_ms"],
            na_position="last",
        ).iloc[0]
        mpc_fpr = float(mpc_row["false_positive_rate"])
        static_fpr = float(static_row["false_positive_rate"])
        rows.append({
            **group_values,
            "mpc_fpr": mpc_fpr,
            "best_static_method": static_row["method"],
            "best_static_fpr": static_fpr,
            "absolute_fpr_reduction": static_fpr - mpc_fpr,
            "relative_fpr_reduction": (
                (static_fpr - mpc_fpr) / static_fpr
                if static_fpr > 0.0 else float("nan")
            ),
            "mpc_fnr": mpc_row["false_negative_rate"],
            "best_static_fnr": static_row["false_negative_rate"],
            "mpc_fallback_count": mpc_row["fallback_count"],
            "mpc_reducer_failure_count": mpc_row["reducer_failure_count"],
            "mpc_runtime_ms": mpc_row["total_time_ms"],
            "best_static_runtime_ms": static_row["total_time_ms"],
        })
    return pd.DataFrame(rows)


def _fidelity_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    required = {"method", "mean_approx_loss"}
    if summary.empty or not required <= set(summary.columns):
        return pd.DataFrame()
    group_columns = [
        column
        for column in ("trace_kind", "budget")
        if column in summary
    ]
    metric_columns = [
        column
        for column in (
            "mean_approx_loss",
            "mean_state_zonotope_approx_error",
            "mean_state_zonotope_width",
            "false_positive_rate",
            "false_negative_rate",
            "total_time_ms",
        )
        if column in summary
    ]
    rows: list[dict[str, object]] = []
    for group_key, frame in summary.groupby(group_columns, dropna=False):
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        group_values = dict(zip(group_columns, values))
        by_method = frame.groupby("method", as_index=False)[metric_columns].mean()
        mpc = by_method[by_method["method"] == "mpc_beam"]
        static = by_method[
            ~by_method["method"].isin(["none", "mpc_beam", "learned_direct"])
        ]
        finite_static = static[np.isfinite(static["mean_approx_loss"])]
        if mpc.empty or finite_static.empty:
            continue
        static_row = finite_static.sort_values(
            [
                column
                for column in (
                    "mean_approx_loss",
                    "mean_state_zonotope_approx_error",
                    "total_time_ms",
                )
                if column in finite_static
            ],
            na_position="last",
        ).iloc[0]
        mpc_row = mpc.iloc[0]
        row: dict[str, object] = {
            **group_values,
            "best_static_method": static_row["method"],
        }
        for metric in metric_columns:
            mpc_value = float(mpc_row[metric])
            static_value = float(static_row[metric])
            row[f"mpc_{metric}"] = mpc_value
            row[f"best_static_{metric}"] = static_value
            row[f"absolute_{metric}_change"] = mpc_value - static_value
            row[f"relative_{metric}_change"] = (
                mpc_value / static_value - 1.0
                if np.isfinite(static_value) and static_value != 0.0
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _learned_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or "mean_approx_loss" not in summary:
        return pd.DataFrame()
    group_columns = [
        column
        for column in ("trace_kind", "budget")
        if column in summary
    ]
    metric_columns = [
        column
        for column in (
            "mean_approx_loss",
            "mean_state_zonotope_approx_error",
            "mean_state_zonotope_width",
            "false_positive_rate",
            "false_negative_rate",
            "total_time_ms",
        )
        if column in summary
    ]
    rows: list[dict[str, object]] = []
    for group_key, frame in summary.groupby(group_columns, dropna=False):
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        group_values = dict(zip(group_columns, values))
        by_method = frame.groupby("method", as_index=False)[metric_columns].mean()
        learned = by_method[by_method["method"] == "learned_direct"]
        mpc = by_method[by_method["method"] == "mpc_beam"]
        if learned.empty or mpc.empty:
            continue
        learned_row = learned.iloc[0]
        mpc_row = mpc.iloc[0]
        row: dict[str, object] = dict(group_values)
        for metric in metric_columns:
            learned_value = float(learned_row[metric])
            mpc_value = float(mpc_row[metric])
            row[f"learned_{metric}"] = learned_value
            row[f"mpc_{metric}"] = mpc_value
            row[f"absolute_{metric}_change"] = learned_value - mpc_value
            row[f"relative_{metric}_change"] = (
                learned_value / mpc_value - 1.0
                if np.isfinite(mpc_value) and mpc_value != 0.0
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate RTLola sweep artifacts",
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scenario", default="robot_arm")
    args = parser.parse_args(argv)
    consolidate_sweep(args.root, args.scenario)


if __name__ == "__main__":
    main()
