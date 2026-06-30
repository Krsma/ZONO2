"""Consolidate resumable RTLola robot-arm sweep artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pzr.rtlola.benchmark import trigger_confusion
from pzr.rtlola.scenarios import scenario_by_name


def consolidate_sweep(root: Path) -> None:
    """Write compact FPR-first tables from completed benchmark cells."""
    summaries = _read_tables(root / "runs", "summary.csv")
    learning_summary = _read_optional(
        root / "learning_stage" / "learning" / "robot_arm"
        / "regret_eval_summary.csv",
    )
    if learning_summary is not None:
        summaries.append(learning_summary)
    combined = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    if not combined.empty:
        combined = combined.drop_duplicates(
            subset=["method", "budget", "trace_kind", "seed"],
            keep="last",
        ).sort_values(["trace_kind", "budget", "method", "seed"])
    combined.to_csv(root / "combined_summary.csv", index=False)

    confusion = _read_tables(root / "runs", "trigger_confusion.csv")
    learning_timeseries = _read_optional(
        root / "learning_stage" / "learning" / "robot_arm"
        / "regret_eval_timeseries.csv",
    )
    if learning_timeseries is not None:
        confusion.append(trigger_confusion(
            learning_timeseries,
            scenario_by_name("robot_arm").trigger_keys,
        ))
    combined_confusion = (
        pd.concat(confusion, ignore_index=True)
        if confusion else pd.DataFrame()
    )
    if not combined_confusion.empty:
        combined_confusion = combined_confusion.sort_values(
            ["trace_kind", "budget", "method", "trigger_key"],
        )
    combined_confusion.to_csv(root / "combined_trigger_confusion.csv", index=False)

    reducer_counts = _reducer_counts(root)
    reducer_counts.to_csv(root / "combined_reducer_counts.csv", index=False)
    _mpc_comparison(combined).to_csv(
        root / "mpc_vs_static_fpr.csv",
        index=False,
    )


def _read_tables(root: Path, filename: str) -> list[pd.DataFrame]:
    if not root.exists():
        return []
    return [
        pd.read_csv(path)
        for path in sorted(root.rglob(filename))
        if path.stat().st_size > 0
    ]


def _read_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path)


def _reducer_counts(root: Path) -> pd.DataFrame:
    frames = _read_tables(root / "runs", "timeseries.csv")
    learned = _read_optional(
        root / "learning_stage" / "learning" / "robot_arm"
        / "regret_eval_timeseries.csv",
    )
    if learned is not None:
        frames.append(learned)
    if not frames:
        return pd.DataFrame()
    timeseries = pd.concat(frames, ignore_index=True)
    columns = ["trace_kind", "budget", "method", "reducer_used"]
    return (
        timeseries.groupby(columns, dropna=False)
        .size()
        .rename("step_count")
        .reset_index()
        .sort_values(columns)
    )


def _mpc_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (trace_kind, budget), frame in summary.groupby(
        ["trace_kind", "budget"],
        dropna=False,
    ):
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
            ~by_method["method"].isin(["mpc_beam", "learned_direct"])
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
            "trace_kind": trace_kind,
            "budget": budget,
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate RTLola FPR sweep artifacts",
    )
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    consolidate_sweep(args.root)


if __name__ == "__main__":
    main()
