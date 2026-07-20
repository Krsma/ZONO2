"""Consolidate resumable RTLola sweep artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pzr.artifact_io import write_csv_atomic
from pzr.rtlola.plots import write_sweep_figures
from pzr.rtlola.tables import (
    budget_sensitivity,
    method_comparison,
    mpc_static_improvement_by_budget,
    mpc_action_composition,
    mpc_girard_deferral,
    mpc_metric_comparison,
    mpc_plan_followthrough,
    primary_metrics,
    reference_balance_by_trace,
    runtime_summary,
    timeseries_metric_summary,
)


def consolidate_sweep(root: Path, scenario: str = "robot_arm") -> None:
    """Write compact metric and trigger tables from completed cells."""
    summaries = _read_tables(root / "runs", "summary.csv")
    combined = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    timeseries = _combined_timeseries(root)
    if not combined.empty:
        identity = [
            column
            for column in (
                "method",
                "budget",
                "trace_kind",
                "seed",
                "optimized_horizon",
                "configured_tail_horizon",
                "root_beam_width",
            )
            if column in combined
        ]
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
    write_csv_atomic(combined, root / "combined_summary.csv")
    primary = primary_metrics(combined)
    primary_path = root / "primary_metrics.csv"
    write_csv_atomic(primary, primary_path)
    print(f"Primary metrics: {primary_path}")
    if not primary.empty:
        print(primary.to_string(index=False))

    confusion = _read_tables(root / "runs", "trigger_confusion.csv")
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
    write_csv_atomic(combined_confusion, root / "combined_trigger_confusion.csv")

    failures = _read_tables(root / "runs", "run_failures.csv")
    combined_failures = (
        pd.concat(failures, ignore_index=True)
        if failures else pd.DataFrame()
    )
    write_csv_atomic(combined_failures, root / "combined_run_failures.csv")

    root_evaluations = _read_tables(root / "runs", "mpc_root_evaluations.csv")
    combined_root_evaluations = (
        pd.concat(root_evaluations, ignore_index=True)
        if root_evaluations else pd.DataFrame()
    )
    write_csv_atomic(
        combined_root_evaluations, root / "combined_mpc_root_evaluations.csv",
    )

    reducer_counts = _reducer_counts(root)
    write_csv_atomic(reducer_counts, root / "combined_reducer_counts.csv")
    write_csv_atomic(
        method_comparison(combined), root / "method_comparison.csv",
    )
    write_csv_atomic(
        mpc_action_composition(reducer_counts), root / "mpc_action_composition.csv",
    )
    write_csv_atomic(
        mpc_plan_followthrough(timeseries), root / "mpc_plan_followthrough.csv",
    )
    write_csv_atomic(
        mpc_girard_deferral(timeseries), root / "mpc_girard_deferral.csv",
    )
    mpc_comparison = mpc_metric_comparison(combined)
    write_csv_atomic(mpc_comparison, root / "mpc_vs_static_metrics.csv")
    budget_table = budget_sensitivity(combined)
    write_csv_atomic(budget_table, root / "budget_sensitivity.csv")
    runtime_table = runtime_summary(combined, timeseries)
    write_csv_atomic(runtime_table, root / "runtime_summary.csv")
    write_csv_atomic(
        reference_balance_by_trace(combined), root / "reference_balance_by_trace.csv",
    )
    timeseries_table = timeseries_metric_summary(timeseries)
    write_csv_atomic(timeseries_table, root / "timeseries_metric_summary.csv")
    write_csv_atomic(
        mpc_static_improvement_by_budget(mpc_comparison),
        root / "mpc_static_improvement_by_budget.csv",
    )
    written = write_sweep_figures(
        root,
        budget_table,
        timeseries_table,
        combined_confusion,
        runtime_table,
    )
    if written:
        print(f"Figures: {root / 'figures'} ({len(written)} files)")


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


def _reducer_counts(root: Path) -> pd.DataFrame:
    frames = _read_tables(root / "runs", "timeseries.csv")
    if not frames:
        return pd.DataFrame()
    timeseries = pd.concat(frames, ignore_index=True)
    columns = [
        column
        for column in (
            "trace_kind",
            "budget",
            "method",
            "optimized_horizon",
            "configured_tail_horizon",
            "root_beam_width",
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


def _combined_timeseries(root: Path) -> pd.DataFrame:
    frames = _read_tables(root / "runs", "timeseries.csv")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
