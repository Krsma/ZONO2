"""Diagnostic and paper-facing plots for learned reducer evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def write_learning_plots(
    timeseries: pd.DataFrame,
    summary: pd.DataFrame,
    output: Path,
) -> None:
    if timeseries.empty or summary.empty:
        raise ValueError("learning plots require non-empty evaluation artifacts")
    output.mkdir(parents=True, exist_ok=True)
    _metric_budget_plot(summary, output / "metrics_vs_budget.png")
    _trace_generalization_plot(summary, output / "generalization_by_trace.png")
    _candidate_selection_plot(timeseries, output / "candidate_selection.png")
    _loss_over_time_plot(timeseries, output / "learned_loss_over_time.png")


def _metric_budget_plot(summary: pd.DataFrame, path: Path) -> None:
    metrics = ("mean_approx_loss", "mean_state_width", "fpr", "fnr")
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    averaged = summary.groupby(["method", "budget"], as_index=False)[list(metrics)].mean()
    for axis, metric in zip(axes.flat, metrics):
        for method, rows in averaged.groupby("method"):
            rows = rows.sort_values("budget")
            axis.plot(rows["budget"], rows[metric], marker="o", label=method)
        axis.set_title(metric.replace("_", " "))
        axis.set_xlabel("binding transform bound")
        axis.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=7)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _trace_generalization_plot(summary: pd.DataFrame, path: Path) -> None:
    grouped = summary.groupby(["trace_kind", "method"], as_index=False)[
        "mean_approx_loss"
    ].mean()
    pivot = grouped.pivot(index="trace_kind", columns="method", values="mean_approx_loss")
    axis = pivot.plot(kind="bar", figsize=(11, 5))
    axis.set_ylabel("mean binding-native approximation loss")
    axis.set_xlabel("fixed trace kind")
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(path, dpi=180)
    plt.close(axis.figure)


def _candidate_selection_plot(timeseries: pd.DataFrame, path: Path) -> None:
    learned = timeseries[timeseries["method"] == "learned_direct"]
    counts = learned.groupby(["trace_kind", "reducer_used"]).size()
    fractions = counts.groupby(level=0).transform(lambda values: values / values.sum())
    pivot = fractions.unstack(fill_value=0.0)
    axis = pivot.plot(kind="bar", stacked=True, figsize=(11, 5))
    axis.set_ylabel("selection fraction")
    axis.set_xlabel("fixed trace kind")
    axis.set_ylim(0.0, 1.0)
    axis.figure.tight_layout()
    axis.figure.savefig(path, dpi=180)
    plt.close(axis.figure)


def _loss_over_time_plot(timeseries: pd.DataFrame, path: Path) -> None:
    learned = timeseries[timeseries["method"] == "learned_direct"]
    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for (trace_kind, budget), rows in learned.groupby(["trace_kind", "budget"]):
        axis.plot(
            rows["step"], rows["approx_loss"], alpha=0.75,
            label=f"{trace_kind}, b={budget}",
        )
    axis.set_xlabel("event")
    axis.set_ylabel("binding-native approximation loss")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    figure.savefig(path, dpi=180)
    plt.close(figure)
