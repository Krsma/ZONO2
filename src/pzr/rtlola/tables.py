"""Schema-sensitive RTLola reporting table builders."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


PRIMARY_METRIC_COLUMNS = (
    "trace_kind",
    "budget",
    "method",
    "seed",
    "false_positive_count",
    "false_negative_count",
    "reference_positive_count",
    "reference_negative_count",
    "fpr",
    "fnr",
    "mean_approx_loss",
    "final_approx_loss",
    "sum_approx_loss",
    "mean_state_width",
    "max_state_width",
    "total_time_ms",
)


def trigger_confusion(timeseries: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    if timeseries.empty:
        return pd.DataFrame(columns=(
            "method",
            "budget",
            "trace_kind",
            "trigger_key",
            "false_positive_steps",
            "false_negative_steps",
            "reference_positive_steps",
            "reference_negative_steps",
            "trigger_positive_steps",
            "steps",
            "fpr",
            "fnr",
            "trigger_positive_rate",
        ))
    rows = []
    group_columns = [
        column for column in ("method", "budget", "trace_kind")
        if column in timeseries
    ]
    for group_key, frame in timeseries.groupby(group_columns, dropna=False):
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        group_values = dict(zip(group_columns, values))
        for key in ("__any__", *keys):
            predicted_column = "trigger_positive" if key == "__any__" else key
            exact_column = (
                "exact_trigger_positive"
                if key == "__any__" else f"exact_{key}"
            )
            predicted = _boolean_series(frame, predicted_column)
            exact = _boolean_series(frame, exact_column)
            valid = exact.notna()
            predicted_valid = predicted[valid].astype(bool)
            exact_valid = exact[valid].astype(bool)
            fp = int((predicted_valid & ~exact_valid).sum())
            fn = int((~predicted_valid & exact_valid).sum())
            positives = int(exact_valid.sum())
            negatives = int((~exact_valid).sum())
            rows.append({
                **group_values,
                "trigger_key": key,
                "false_positive_steps": fp,
                "false_negative_steps": fn,
                "reference_positive_steps": positives,
                "reference_negative_steps": negatives,
                "trigger_positive_steps": int(predicted.fillna(False).sum()),
                "steps": int(len(frame)),
                "fpr": fp / negatives if negatives else float("nan"),
                "fnr": fn / positives if positives else float("nan"),
                "trigger_positive_rate": float(predicted.mean()),
            })
    return pd.DataFrame(rows)


def primary_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the concise end-of-sweep metric contract."""
    if summary.empty:
        return pd.DataFrame(columns=PRIMARY_METRIC_COLUMNS)
    primary = summary.reindex(columns=PRIMARY_METRIC_COLUMNS)
    return primary.sort_values(
        ["trace_kind", "budget", "method", "seed"],
    ).reset_index(drop=True)


def method_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """Return one compact performance row per trace, budget, and method."""
    group_columns = [
        column
        for column in (
            "trace_kind",
            "budget",
            "method",
            "optimized_horizon",
            "configured_tail_horizon",
            "root_beam_width",
        )
        if column in summary
    ]
    metric_columns = [
        column
        for column in (
            "fpr",
            "fnr",
            "trigger_positive_rate",
            "mean_approx_loss",
            "final_approx_loss",
            "max_approx_loss",
            "sum_approx_loss",
            "mean_state_width",
            "max_state_width",
            "mean_generator_count",
            "mean_active_dynamic_generator_count",
            "total_reductions",
            "total_time_ms",
            "fallback_count",
            "fallback_rate",
            "reducer_failure_count",
            "infeasible_candidate_count",
            "tail_fallback_count",
        )
        if column in summary
    ]
    if not {"trace_kind", "budget", "method"} <= set(group_columns) or not metric_columns:
        return pd.DataFrame(columns=(*group_columns, *metric_columns))
    return (
        summary.groupby(group_columns, as_index=False, dropna=False)[metric_columns]
        .mean()
        .sort_values(group_columns)
    )


def mpc_action_composition(reducer_counts: pd.DataFrame) -> pd.DataFrame:
    """Return committed MPC action shares, including and excluding no-op steps."""
    metadata_columns = [
        column
        for column in (
            "optimized_horizon",
            "configured_tail_horizon",
            "root_beam_width",
        )
        if column in reducer_counts
    ]
    columns = [
        "trace_kind",
        "budget",
        "method",
        *metadata_columns,
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
    mpc = reducer_counts[
        reducer_counts["method"].astype(str).str.startswith("mpc_")
    ].copy()
    if mpc.empty:
        return pd.DataFrame(columns=columns)
    group_columns = [
        column
        for column in (
            "trace_kind",
            "budget",
            "method",
            "optimized_horizon",
            "configured_tail_horizon",
            "root_beam_width",
        )
        if column in mpc
    ]
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


def mpc_plan_followthrough(timeseries: pd.DataFrame) -> pd.DataFrame:
    """Compare each scheduled MPC action with the action later committed."""
    metadata_columns = [
        column
        for column in (
            "optimized_horizon",
            "configured_tail_horizon",
            "root_beam_width",
        )
        if column in timeseries
    ]
    columns = [
        "trace_kind",
        "budget",
        "seed",
        "method",
        *metadata_columns,
        "position",
        "predicted_action",
        "prediction_count",
        "position_prediction_count",
        "position_action_share",
        "realized_count",
        "realized_match_count",
        "realization_rate",
        "realized_scott_count",
    ]
    required = {
        "trace_kind",
        "budget",
        "seed",
        "method",
        "step",
        "reducer_used",
        "predicted_sequence",
    }
    if timeseries.empty or not required <= set(timeseries.columns):
        return pd.DataFrame(columns=columns)
    mpc = timeseries[
        timeseries["method"].astype(str).str.startswith("mpc_")
    ].copy()
    if mpc.empty:
        return pd.DataFrame(columns=columns)
    group_columns = [
        "trace_kind",
        "budget",
        "seed",
        "method",
        *metadata_columns,
    ]
    rows: list[dict[str, object]] = []
    for group_key, frame in mpc.groupby(group_columns, dropna=False):
        group_values = dict(zip(group_columns, group_key))
        committed = {
            int(row.step): str(row.reducer_used)
            for row in frame.itertuples()
        }
        counts: dict[tuple[int, str], dict[str, int]] = {}
        position_totals: dict[int, int] = {}
        for row in frame.itertuples():
            raw_sequence = getattr(row, "predicted_sequence")
            if not isinstance(raw_sequence, str) or not raw_sequence:
                continue
            sequence = tuple(
                action for action in raw_sequence.split(",") if action
            )
            for position, predicted_action in enumerate(sequence):
                key = (position, predicted_action)
                values = counts.setdefault(
                    key,
                    {
                        "prediction_count": 0,
                        "realized_count": 0,
                        "realized_match_count": 0,
                        "realized_scott_count": 0,
                    },
                )
                values["prediction_count"] += 1
                position_totals[position] = position_totals.get(position, 0) + 1
                realized = committed.get(int(row.step) + position)
                if realized is None:
                    continue
                values["realized_count"] += 1
                values["realized_match_count"] += int(realized == predicted_action)
                values["realized_scott_count"] += int(realized == "scott")
        for (position, predicted_action), values in sorted(counts.items()):
            total = position_totals[position]
            realized_count = values["realized_count"]
            rows.append({
                **group_values,
                "position": position,
                "predicted_action": predicted_action,
                **values,
                "position_prediction_count": total,
                "position_action_share": values["prediction_count"] / total,
                "realization_rate": (
                    values["realized_match_count"] / realized_count
                    if realized_count else float("nan")
                ),
            })
    return pd.DataFrame(rows, columns=columns)


def mpc_girard_deferral(timeseries: pd.DataFrame) -> pd.DataFrame:
    """Measure Scott-first plans whose terminal Girard is later replaced."""
    metadata_columns = [
        column
        for column in (
            "optimized_horizon",
            "configured_tail_horizon",
            "root_beam_width",
        )
        if column in timeseries
    ]
    columns = [
        "trace_kind",
        "budget",
        "seed",
        "method",
        *metadata_columns,
        "scott_first_terminal_girard_count",
        "realized_terminal_count",
        "realized_girard_count",
        "realized_scott_count",
        "girard_realization_rate",
        "girard_deferral_rate",
    ]
    required = {
        "trace_kind",
        "budget",
        "seed",
        "method",
        "step",
        "reducer_used",
        "predicted_sequence",
    }
    if timeseries.empty or not required <= set(timeseries.columns):
        return pd.DataFrame(columns=columns)
    mpc = timeseries[
        timeseries["method"].astype(str).str.startswith("mpc_")
    ].copy()
    group_columns = [
        "trace_kind",
        "budget",
        "seed",
        "method",
        *metadata_columns,
    ]
    rows: list[dict[str, object]] = []
    for group_key, frame in mpc.groupby(group_columns, dropna=False):
        group_values = dict(zip(group_columns, group_key))
        committed = {
            int(row.step): str(row.reducer_used)
            for row in frame.itertuples()
        }
        planned = realized = girard = scott = 0
        for row in frame.itertuples():
            raw_sequence = getattr(row, "predicted_sequence")
            if not isinstance(raw_sequence, str) or not raw_sequence:
                continue
            sequence = tuple(
                action for action in raw_sequence.split(",") if action
            )
            if len(sequence) < 2 or sequence[0] != "scott" or sequence[-1] != "girard":
                continue
            planned += 1
            action = committed.get(int(row.step) + len(sequence) - 1)
            if action is None:
                continue
            realized += 1
            girard += int(action == "girard")
            scott += int(action == "scott")
        rows.append({
            **group_values,
            "scott_first_terminal_girard_count": planned,
            "realized_terminal_count": realized,
            "realized_girard_count": girard,
            "realized_scott_count": scott,
            "girard_realization_rate": girard / realized if realized else float("nan"),
            "girard_deferral_rate": scott / realized if realized else float("nan"),
        })
    return pd.DataFrame(rows, columns=columns)


def mpc_metric_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """Compare MPC with the independently best static method for each metric."""
    columns = [
        "trace_kind",
        "budget",
        "mpc_method",
        "optimized_horizon",
        "configured_tail_horizon",
        "root_beam_width",
        "metric",
        "mpc_value",
        "best_static_method",
        "best_static_value",
        "absolute_improvement",
        "relative_improvement",
    ]
    required = {
        "method",
        "trace_kind",
        "budget",
        "fpr",
        "fnr",
        "mean_approx_loss",
        "mean_state_width",
    }
    if summary.empty or not required <= set(summary.columns):
        return pd.DataFrame(columns=columns)
    group_columns = [
        column
        for column in ("trace_kind", "budget")
        if column in summary
    ]
    metric_columns = [
        "fpr",
        "fnr",
        "mean_approx_loss",
        "final_approx_loss",
        "sum_approx_loss",
        "mean_state_width",
    ]
    rows: list[dict[str, object]] = []
    for group_key, frame in summary.groupby(group_columns, dropna=False):
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        group_values = dict(zip(group_columns, values))
        is_mpc = frame["method"].astype(str).str.startswith("mpc_")
        metadata = [
            column
            for column in (
                "optimized_horizon",
                "configured_tail_horizon",
                "root_beam_width",
            )
            if column in frame
        ]
        mpc = (
            frame.loc[is_mpc]
            .groupby(["method", *metadata], as_index=False, dropna=False)[metric_columns]
            .mean()
        )
        static = (
            frame.loc[
                ~is_mpc & ~frame["method"].isin(["none", "learned_direct"])
            ]
            .groupby("method", as_index=False)[metric_columns]
            .mean()
        )
        if mpc.empty or static.empty:
            continue
        for metric in metric_columns:
            finite_static = static[np.isfinite(static[metric])]
            if finite_static.empty:
                continue
            static_row = finite_static.sort_values([metric, "method"]).iloc[0]
            static_value = float(static_row[metric])
            finite_mpc = mpc[np.isfinite(mpc[metric])]
            for _, mpc_row in finite_mpc.iterrows():
                mpc_value = float(mpc_row[metric])
                improvement = static_value - mpc_value
                rows.append({
                    **group_values,
                    "mpc_method": mpc_row["method"],
                    **{column: mpc_row[column] for column in metadata},
                    "metric": metric,
                    "mpc_value": mpc_value,
                    "best_static_method": static_row["method"],
                    "best_static_value": static_value,
                    "absolute_improvement": improvement,
                    "relative_improvement": (
                        improvement / static_value
                        if static_value != 0.0 else float("nan")
                    ),
                })
    return pd.DataFrame(rows, columns=columns)


def learned_comparison(summary: pd.DataFrame) -> pd.DataFrame:
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
            "final_approx_loss",
            "sum_approx_loss",
            "mean_state_width",
            "fpr",
            "fnr",
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
        mpc = by_method[by_method["method"] == "mpc_terminal_beam"]
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


def _boolean_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=object)
    return frame[column].map(
        lambda value: np.nan if pd.isna(value) else bool(value),
    )
