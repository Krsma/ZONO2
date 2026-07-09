"""Reproducible RTLola sweep plots built from consolidated artifacts."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


BUDGET_METRICS = (
    "fpr",
    "fnr",
    "mean_approx_loss",
    "final_approx_loss",
    "sum_approx_loss",
    "mean_state_width",
    "total_time_ms",
)

TIMESERIES_METRICS = (
    "approx_loss",
    "state_width",
)

PLOT_METADATA_COLUMNS = (
    "optimized_horizon",
    "configured_tail_horizon",
    "root_beam_width",
)


def write_sweep_figures(
    root: Path,
    budget_table: pd.DataFrame,
    timeseries_table: pd.DataFrame,
    confusion: pd.DataFrame,
    runtime_table: pd.DataFrame,
) -> tuple[Path, ...]:
    """Write deterministic sweep figures and return the generated paths."""
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for metric in BUDGET_METRICS:
        written.extend(_plot_budget_metric(
            budget_table,
            metric,
            figures_dir / f"budget_sensitivity_{_safe_name(metric)}",
        ))
    for metric in TIMESERIES_METRICS:
        written.extend(_plot_timeseries_metric(timeseries_table, metric, figures_dir))
    written.extend(_plot_trigger_confusion(
        confusion,
        figures_dir / "trigger_fpr_fnr_by_trace_budget",
    ))
    written.extend(_plot_runtime_vs_loss(
        runtime_table,
        budget_table,
        figures_dir / "runtime_vs_loss_by_trace",
    ))
    return tuple(written)


def _plot_budget_metric(
    budget_table: pd.DataFrame,
    metric: str,
    stem: Path,
) -> tuple[Path, ...]:
    required = {"trace_kind", "budget", "method", "metric", "mean"}
    if budget_table.empty or not required <= set(budget_table.columns):
        return ()
    data = budget_table[budget_table["metric"] == metric].copy()
    if data.empty:
        return ()
    data["budget"] = pd.to_numeric(data["budget"], errors="coerce")
    data["mean"] = pd.to_numeric(data["mean"], errors="coerce")
    data = data[np.isfinite(data["budget"]) & np.isfinite(data["mean"])]
    if data.empty:
        return ()

    plt = _pyplot()
    trace_kinds = _sorted_values(data["trace_kind"])
    fig, axes = plt.subplots(
        len(trace_kinds),
        1,
        figsize=(7.0, max(3.0, 2.4 * len(trace_kinds))),
        squeeze=False,
        sharex=False,
    )
    for ax, trace_kind in zip(axes.ravel(), trace_kinds):
        frame = data[data["trace_kind"] == trace_kind]
        for group_key, method_frame in frame.groupby(
            _series_group_columns(frame),
            dropna=False,
        ):
            ordered = method_frame.sort_values("budget")
            ax.plot(
                ordered["budget"],
                ordered["mean"],
                marker="o",
                linewidth=1.3,
                markersize=3.5,
                label=_series_label(method_frame, group_key),
            )
        ax.set_title(str(trace_kind), fontsize=10)
        ax.set_xlabel("Budget")
        ax.set_ylabel(_label(metric))
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    return _save(fig, stem, plt)


def _plot_timeseries_metric(
    timeseries_table: pd.DataFrame,
    metric: str,
    figures_dir: Path,
) -> tuple[Path, ...]:
    required = {"trace_kind", "budget", "method", "step", "metric", "mean"}
    if timeseries_table.empty or not required <= set(timeseries_table.columns):
        return ()
    data = timeseries_table[timeseries_table["metric"] == metric].copy()
    if data.empty:
        return ()
    data["step"] = pd.to_numeric(data["step"], errors="coerce")
    data["mean"] = pd.to_numeric(data["mean"], errors="coerce")
    data = data[np.isfinite(data["step"]) & np.isfinite(data["mean"])]
    if data.empty:
        return ()

    plt = _pyplot()
    written: list[Path] = []
    for (trace_kind, budget), frame in data.groupby(
        ["trace_kind", "budget"],
        dropna=False,
    ):
        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        for group_key, method_frame in frame.groupby(
            _series_group_columns(frame),
            dropna=False,
        ):
            ordered = method_frame.sort_values("step")
            ax.plot(
                ordered["step"],
                ordered["mean"],
                linewidth=1.1,
                label=_series_label(method_frame, group_key),
            )
        ax.set_title(f"{trace_kind} / budget {budget}", fontsize=10)
        ax.set_xlabel("Step")
        ax.set_ylabel(_label(metric))
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        written.extend(_save(
            fig,
            figures_dir / (
                f"timeseries_{_safe_name(metric)}_"
                f"{_safe_name(trace_kind)}_budget_{_safe_name(budget)}"
            ),
            plt,
        ))
    return tuple(written)


def _plot_trigger_confusion(
    confusion: pd.DataFrame,
    stem: Path,
) -> tuple[Path, ...]:
    required = {"trace_kind", "budget", "method", "trigger_key", "fpr", "fnr"}
    if confusion.empty or not required <= set(confusion.columns):
        return ()
    data = confusion[confusion["trigger_key"] == "__any__"].copy()
    if data.empty:
        return ()
    data["budget"] = pd.to_numeric(data["budget"], errors="coerce")
    for column in ("fpr", "fnr"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[["fpr", "fnr"]].dropna(how="all").empty:
        return ()

    plt = _pyplot()
    trace_kinds = _sorted_values(data["trace_kind"])
    fig, axes = plt.subplots(
        len(trace_kinds),
        1,
        figsize=(7.0, max(3.0, 2.4 * len(trace_kinds))),
        squeeze=False,
        sharex=False,
    )
    for ax, trace_kind in zip(axes.ravel(), trace_kinds):
        frame = data[data["trace_kind"] == trace_kind]
        for method, method_frame in frame.groupby("method", dropna=False):
            ordered = method_frame.sort_values("budget")
            for metric, linestyle in (("fpr", "-"), ("fnr", "--")):
                values = ordered[metric]
                if values.notna().any():
                    ax.plot(
                        ordered["budget"],
                        values,
                        marker="o",
                        linestyle=linestyle,
                        linewidth=1.2,
                        markersize=3.5,
                        label=f"{method} {metric}",
                    )
        ax.set_title(str(trace_kind), fontsize=10)
        ax.set_xlabel("Budget")
        ax.set_ylabel("Rate")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    return _save(fig, stem, plt)


def _plot_runtime_vs_loss(
    runtime_table: pd.DataFrame,
    budget_table: pd.DataFrame,
    stem: Path,
) -> tuple[Path, ...]:
    runtime_required = {
        "trace_kind",
        "budget",
        "method",
        "total_time_mean_ms",
    }
    budget_required = {"trace_kind", "budget", "method", "metric", "mean"}
    if (
        runtime_table.empty
        or budget_table.empty
        or not runtime_required <= set(runtime_table.columns)
        or not budget_required <= set(budget_table.columns)
    ):
        return ()
    loss = budget_table[
        budget_table["metric"].isin(["mean_approx_loss", "mean_state_width"])
    ].copy()
    if loss.empty:
        return ()
    merge_keys = [
        column
        for column in ("trace_kind", "budget", "method", *PLOT_METADATA_COLUMNS)
        if column in runtime_table and column in loss
    ]
    loss = loss.pivot_table(
        index=merge_keys,
        columns="metric",
        values="mean",
        aggfunc="mean",
    ).reset_index()
    data = runtime_table.merge(
        loss,
        on=merge_keys,
        how="inner",
    )
    if data.empty:
        return ()
    data["total_time_mean_ms"] = pd.to_numeric(
        data["total_time_mean_ms"],
        errors="coerce",
    )
    for column in ("mean_approx_loss", "mean_state_width"):
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    y_column = (
        "mean_state_width"
        if "mean_approx_loss" not in data or data["mean_approx_loss"].isna().all()
        else "mean_approx_loss"
    )
    data = data[
        np.isfinite(data["total_time_mean_ms"])
        & np.isfinite(data[y_column])
    ]
    if data.empty:
        return ()

    plt = _pyplot()
    trace_kinds = _sorted_values(data["trace_kind"])
    fig, axes = plt.subplots(
        len(trace_kinds),
        1,
        figsize=(7.0, max(3.0, 2.6 * len(trace_kinds))),
        squeeze=False,
        sharex=False,
    )
    for ax, trace_kind in zip(axes.ravel(), trace_kinds):
        frame = data[data["trace_kind"] == trace_kind]
        for group_key, method_frame in frame.groupby(
            _series_group_columns(frame),
            dropna=False,
        ):
            ax.scatter(
                method_frame["total_time_mean_ms"],
                method_frame[y_column],
                s=24,
                label=_series_label(method_frame, group_key),
            )
            for row in method_frame.itertuples(index=False):
                ax.annotate(
                    str(row.budget),
                    (row.total_time_mean_ms, getattr(row, y_column)),
                    fontsize=7,
                )
        ax.set_title(str(trace_kind), fontsize=10)
        ax.set_xlabel("Runtime [ms]")
        ax.set_ylabel(_label(y_column))
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    return _save(fig, stem, plt)


def _save(fig: object, stem: Path, plt: object) -> tuple[Path, Path]:
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=160)
    plt.close(fig)
    return pdf, png


def _pyplot() -> object:
    cache_dir = Path(tempfile.gettempdir()) / "pzr-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _safe_name(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "value"


def _sorted_values(series: pd.Series) -> list[object]:
    return sorted(series.dropna().unique().tolist(), key=str)


def _series_group_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in ("method", *PLOT_METADATA_COLUMNS)
        if column in frame
    ]


def _series_label(frame: pd.DataFrame, group_key: object) -> str:
    values = group_key if isinstance(group_key, tuple) else (group_key,)
    group_values = dict(zip(_series_group_columns(frame), values))
    label = str(group_values.get("method", "method"))
    metadata = []
    for column, short in (
        ("optimized_horizon", "h"),
        ("configured_tail_horizon", "tail"),
        ("root_beam_width", "beam"),
    ):
        value = group_values.get(column)
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if not pd.isna(numeric) and numeric > 0:
            metadata.append(f"{short}={int(numeric)}")
    return f"{label} ({', '.join(metadata)})" if metadata else label


def _label(metric: str) -> str:
    labels = {
        "fpr": "False-positive rate",
        "fnr": "False-negative rate",
        "mean_approx_loss": "Mean approximation loss",
        "final_approx_loss": "Final approximation loss",
        "sum_approx_loss": "Summed approximation loss",
        "mean_state_width": "Mean state width",
        "total_time_ms": "Runtime [ms]",
        "approx_loss": "Approximation loss",
        "state_width": "State width",
    }
    return labels.get(metric, metric.replace("_", " ").title())
