"""Consolidate resumable RTLola sweep artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pzr.rtlola.scenarios import scenario_by_name
from pzr.rtlola.tables import (
    learned_comparison,
    method_comparison,
    mpc_action_composition,
    mpc_girard_deferral,
    mpc_metric_comparison,
    mpc_plan_followthrough,
    primary_metrics,
    trigger_confusion,
)


def consolidate_sweep(root: Path, scenario: str = "robot_arm") -> None:
    """Write compact metric and trigger tables from completed cells."""
    summaries = _read_tables(root / "runs", "summary.csv")
    learning_summary = _read_optional(
        root / "learning_stage" / "learning" / scenario
        / "regret_eval_summary.csv",
    )
    if learning_summary is not None:
        summaries.append(learning_summary)
    combined = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    timeseries = _combined_timeseries(root, scenario)
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
    combined.to_csv(root / "combined_summary.csv", index=False)
    primary = primary_metrics(combined)
    primary_path = root / "primary_metrics.csv"
    primary.to_csv(primary_path, index=False)
    print(f"Primary metrics: {primary_path}")
    if not primary.empty:
        print(primary.to_string(index=False))

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

    root_evaluations = _read_tables(root / "runs", "mpc_root_evaluations.csv")
    combined_root_evaluations = (
        pd.concat(root_evaluations, ignore_index=True)
        if root_evaluations else pd.DataFrame()
    )
    combined_root_evaluations.to_csv(
        root / "combined_mpc_root_evaluations.csv",
        index=False,
    )

    reducer_counts = _reducer_counts(root, scenario)
    reducer_counts.to_csv(root / "combined_reducer_counts.csv", index=False)
    method_comparison(combined).to_csv(
        root / "method_comparison.csv",
        index=False,
    )
    mpc_action_composition(reducer_counts).to_csv(
        root / "mpc_action_composition.csv",
        index=False,
    )
    mpc_plan_followthrough(timeseries).to_csv(
        root / "mpc_plan_followthrough.csv",
        index=False,
    )
    mpc_girard_deferral(timeseries).to_csv(
        root / "mpc_girard_deferral.csv",
        index=False,
    )
    mpc_metric_comparison(combined).to_csv(
        root / "mpc_vs_static_metrics.csv",
        index=False,
    )
    learned_comparison(combined).to_csv(
        root / "learned_vs_mpc_metrics.csv",
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


def _combined_timeseries(root: Path, scenario: str) -> pd.DataFrame:
    frames = _read_tables(root / "runs", "timeseries.csv")
    learned = _read_optional(
        root / "learning_stage" / "learning" / scenario
        / "regret_eval_timeseries.csv",
    )
    if learned is not None:
        frames.append(learned)
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
