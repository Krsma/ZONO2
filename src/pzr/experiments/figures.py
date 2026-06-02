"""Figure generation for benchmark results.

Produces publication-quality matplotlib figures for:
- Trigger width time series with confidence bands
- Generator count time series with confidence bands
- Method comparison bar charts with confidence intervals
- Budget sweep plots
- Reducer selection frequency charts
- DAgger learning curves
- Inference time comparison
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_COLORS: dict[str, str] = {
    "girard": "#1f77b4",
    "combastel": "#2ca02c",
    "pca": "#17becf",
    "methA": "#bcbd22",
    "scott": "#9467bd",
    "box": "#7f7f7f",
    "mpc_rollout": "#d62728",
    "mpc_rollout_methA": "#8c564b",
    "mpc_rollout_scott": "#c5b0d5",
    "mpc_pair_rollout3": "#e377c2",
    "mpc_sequence": "#ff7f0e",
    "mpc_sequence3": "#ffbb78",
    "mpc_beam3": "#2f4b7c",
    "learned_dagger": "#f7b6d2",
}


def _color(method: str) -> str:
    return METHOD_COLORS.get(method, "#333333")


def plot_trigger_width_timeseries(
    timeseries: pd.DataFrame,
    methods: Sequence[str] | None = None,
    title: str = "Trigger Width Over Time",
    out_path: Path | None = None,
) -> plt.Figure:
    """Plot per-step trigger width with mean ± std bands."""
    if methods is None:
        methods = sorted(timeseries["method"].unique())

    fig, ax = plt.subplots(figsize=(10, 5))
    for method in methods:
        df = timeseries[timeseries["method"] == method]
        grouped = df.groupby("step")["trigger_width_sum"]
        mean = grouped.mean()
        std = grouped.std().fillna(0)
        ax.plot(mean.index, mean.values, label=method, linewidth=1.5, color=_color(method))
        ax.fill_between(mean.index, (mean - std).values, (mean + std).values,
                        alpha=0.15, color=_color(method))

    ax.set_xlabel("Step")
    ax.set_ylabel("Trigger Width Sum")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path:
        _save_fig(fig, out_path)
    return fig


def plot_generator_count_timeseries(
    timeseries: pd.DataFrame,
    methods: Sequence[str] | None = None,
    title: str = "Generator Count Over Time",
    out_path: Path | None = None,
) -> plt.Figure:
    """Plot per-step generator count with mean ± std bands."""
    if methods is None:
        methods = sorted(timeseries["method"].unique())

    fig, ax = plt.subplots(figsize=(10, 5))
    for method in methods:
        df = timeseries[timeseries["method"] == method]
        grouped = df.groupby("step")["generator_count"]
        mean = grouped.mean()
        std = grouped.std().fillna(0)
        ax.plot(mean.index, mean.values, label=method, linewidth=1.5, color=_color(method))
        ax.fill_between(mean.index, (mean - std).values, (mean + std).values,
                        alpha=0.15, color=_color(method))

    ax.set_xlabel("Step")
    ax.set_ylabel("Generator Count")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path:
        _save_fig(fig, out_path)
    return fig


def plot_combined_timeseries(
    timeseries: pd.DataFrame,
    methods: Sequence[str] | None = None,
    title: str = "",
    out_path: Path | None = None,
) -> plt.Figure:
    """Two-panel plot: trigger width and generator count side by side."""
    if methods is None:
        methods = sorted(timeseries["method"].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for method in methods:
        df = timeseries[timeseries["method"] == method]
        c = _color(method)

        tw = df.groupby("step")["trigger_width_sum"]
        tw_mean, tw_std = tw.mean(), tw.std().fillna(0)
        ax1.plot(tw_mean.index, tw_mean.values, label=method, linewidth=1.5, color=c)
        ax1.fill_between(tw_mean.index, (tw_mean - tw_std).values,
                         (tw_mean + tw_std).values, alpha=0.15, color=c)

        gc = df.groupby("step")["generator_count"]
        gc_mean, gc_std = gc.mean(), gc.std().fillna(0)
        ax2.plot(gc_mean.index, gc_mean.values, label=method, linewidth=1.5, color=c)
        ax2.fill_between(gc_mean.index, (gc_mean - gc_std).values,
                         (gc_mean + gc_std).values, alpha=0.15, color=c)

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Trigger Width Sum")
    ax1.set_title("Trigger Width")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Generator Count")
    ax2.set_title("Generator Count")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if out_path:
        _save_fig(fig, out_path)
    return fig


def plot_approximation_error_timeseries(
    timeseries: pd.DataFrame,
    methods: Sequence[str] | None = None,
    title: str = "Approximation Error Over Time",
    out_path: Path | None = None,
) -> plt.Figure:
    """Per-step approximation error vs unreduced zonotope, with mean ± std bands.

    Mirrors Figure 5 of arxiv:2601.11358 (Cutting Corners on Uncertainty).
    """
    if methods is None:
        methods = sorted(timeseries["method"].unique())

    fig, ax = plt.subplots(figsize=(10, 5))
    for method in methods:
        df = timeseries[timeseries["method"] == method]
        grouped = df.groupby("step")["approx_error_sum"]
        mean = grouped.mean()
        std = grouped.std().fillna(0)
        ax.plot(mean.index, mean.values, label=method, linewidth=1.5, color=_color(method))
        ax.fill_between(mean.index, (mean - std).values, (mean + std).values,
                        alpha=0.15, color=_color(method))

    ax.set_xlabel("Step")
    ax.set_ylabel("Approximation Error (|approx − exact| summed over trigger axes)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path:
        _save_fig(fig, out_path)
    return fig


def plot_fig4_panel(
    aggregate: pd.DataFrame,
    title: str = "False Positive Rate & Absolute Error Range",
    out_path: Path | None = None,
) -> plt.Figure:
    """Two-panel chart mirroring arxiv:2601.11358 Figure 4.

    Left: False Positive Rate per method (with CI). Right: Absolute Error Range.
    """
    if "false_positive_rate_mean" not in aggregate.columns:
        raise ValueError("aggregate missing FPR columns; re-run benchmark with ground truth")

    df = aggregate.sort_values("false_positive_rate_mean")
    methods = df["method"].values
    colors = [_color(m) for m in methods]
    x = np.arange(len(methods))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    fpr = df["false_positive_rate_mean"].values
    fpr_lo = df["false_positive_rate_ci95_lo"].values
    fpr_hi = df["false_positive_rate_ci95_hi"].values
    ax1.bar(x, fpr, yerr=[fpr - fpr_lo, fpr_hi - fpr], capsize=4, color=colors, alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("False Positive Rate")
    ax1.set_title("False Positive Rate")
    ax1.grid(True, axis="y", alpha=0.3)

    aer = df["abs_error_range_mean"].values
    aer_lo = df["abs_error_range_ci95_lo"].values
    aer_hi = df["abs_error_range_ci95_hi"].values
    ax2.bar(x, aer, yerr=[aer - aer_lo, aer_hi - aer], capsize=4, color=colors, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Absolute Error Range (max − min)")
    ax2.set_title("Absolute Error Range")
    ax2.grid(True, axis="y", alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if out_path:
        _save_fig(fig, out_path)
    return fig


def plot_method_comparison_bars(
    aggregate: pd.DataFrame,
    metric: str = "mean_trigger_width",
    title: str = "Method Comparison",
    out_path: Path | None = None,
) -> plt.Figure:
    """Bar chart comparing methods on a metric with CI error bars."""
    mean_col = f"{metric}_mean"
    lo_col = f"{metric}_ci95_lo"
    hi_col = f"{metric}_ci95_hi"

    if mean_col not in aggregate.columns:
        raise ValueError(f"metric {metric} not found in aggregate")

    df = aggregate.sort_values(mean_col)
    methods = df["method"].values
    means = df[mean_col].values
    lo = df[lo_col].values if lo_col in df.columns else means
    hi = df[hi_col].values if hi_col in df.columns else means
    err_lo = means - lo
    err_hi = hi - means
    colors = [_color(m) for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=[err_lo, err_hi], capsize=4, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if out_path:
        _save_fig(fig, out_path)
    return fig


def plot_reducer_selection_bars(
    timeseries: pd.DataFrame,
    methods: Sequence[str] | None = None,
    title: str = "Reducer Selection Frequency",
    out_path: Path | None = None,
) -> plt.Figure:
    """Stacked bar chart of reducer selection frequency per method."""
    reduced = timeseries[timeseries["reduced"] == True].copy()
    if methods is None:
        methods = sorted(reduced["method"].unique())

    reduced = reduced[reduced["method"].isin(methods)]
    if reduced.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title(title)
        if out_path:
            _save_fig(fig, out_path)
        return fig

    counts = reduced.groupby(["method", "reducer_used"]).size().unstack(fill_value=0)
    totals = counts.sum(axis=1)
    fractions = counts.div(totals, axis=0)

    reducers = fractions.columns.tolist()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(fractions))
    bottom = np.zeros(len(fractions))

    for reducer in reducers:
        vals = fractions[reducer].values
        ax.bar(x, vals, bottom=bottom, label=reducer, color=_color(reducer), alpha=0.85)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(fractions.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of Reductions")
    ax.set_title(title)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    if out_path:
        _save_fig(fig, out_path)
    return fig


def plot_budget_sweep(
    results_by_budget: dict[int, pd.DataFrame],
    metric: str = "mean_trigger_width",
    methods: Sequence[str] | None = None,
    title: str = "Budget Sweep",
    out_path: Path | None = None,
) -> plt.Figure:
    """Plot metric vs budget for each method."""
    mean_col = f"{metric}_mean"
    budgets = sorted(results_by_budget.keys())

    if methods is None:
        methods = sorted(results_by_budget[budgets[0]]["method"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        values = []
        for b in budgets:
            agg = results_by_budget[b]
            row = agg[agg["method"] == method]
            if len(row) > 0 and mean_col in row.columns:
                values.append(float(row[mean_col].values[0]))
            else:
                values.append(np.nan)
        ax.plot(budgets, values, marker="o", label=method, linewidth=1.5, color=_color(method))

    ax.set_xlabel("Generator Budget")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path:
        _save_fig(fig, out_path)
    return fig


def plot_dagger_learning_curve(
    train_accuracies: Sequence[float],
    val_accuracies: Sequence[float] | None = None,
    title: str = "DAgger Learning Curve",
    out_path: Path | None = None,
) -> plt.Figure:
    """Plot train (and optionally val) accuracy per DAgger iteration."""
    fig, ax = plt.subplots(figsize=(7, 4))
    iterations = list(range(1, len(train_accuracies) + 1))
    ax.plot(iterations, train_accuracies, "o-", label="Train", linewidth=1.5,
            color=METHOD_COLORS["mpc_rollout"])
    if val_accuracies:
        ax.plot(iterations, val_accuracies, "s--", label="Validation", linewidth=1.5,
                color=METHOD_COLORS["learned_dagger"])
    ax.set_xlabel("DAgger Iteration")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    if out_path:
        _save_fig(fig, out_path)
    return fig


def plot_inference_time_comparison(
    method_times: dict[str, float],
    title: str = "Inference Time Comparison",
    out_path: Path | None = None,
) -> plt.Figure:
    """Bar chart of per-decision inference times."""
    fig, ax = plt.subplots(figsize=(7, 4))
    methods = list(method_times.keys())
    times = list(method_times.values())
    x = np.arange(len(methods))
    colors = [_color(m) for m in methods]
    ax.bar(x, times, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Time per Decision (ms)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if out_path:
        _save_fig(fig, out_path)
    return fig


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    stem = path.stem
    suffix = path.suffix
    if suffix == ".pdf":
        fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    elif suffix == ".png":
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
