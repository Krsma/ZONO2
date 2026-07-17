"""Diagnostic and paper-facing plots for learned reducer evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pzr.learning.ranker import ReducerTrainingResult


def write_training_plots(
    temperatures: pd.DataFrame,
    result: ReducerTrainingResult,
    output: Path,
) -> None:
    """Write selected-objective histories and temperature diagnostics."""
    output.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.plot(result.train_loss_history, label="train total")
    axis.plot(result.val_loss_history, label="validation total")
    if result.objective == "soft-kl":
        axis.plot(result.train_kl_history, linestyle="--", label="train KL")
        axis.plot(result.val_kl_history, linestyle="--", label="validation KL")
        axis.plot(
            result.val_feasibility_history,
            linestyle=":",
            label="validation infeasible mass",
        )
    axis.set_xlabel("epoch")
    axis.set_ylabel("state-balanced objective")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(output / "training_history.png", dpi=180)
    plt.close(figure)

    if temperatures["temperature"].notna().any():
        ordered = temperatures.dropna(subset=["temperature"]).sort_values("temperature")
        figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        axis.plot(
            ordered["temperature"], ordered["mean_selected_normalized_regret"],
            marker="o", label="mean selected regret",
        )
        axis.plot(
            ordered["temperature"], ordered["max_selected_normalized_regret"],
            marker="o", label="max selected regret",
        )
        selected = ordered[ordered["selected"]]
        if not selected.empty:
            axis.axvline(float(selected.iloc[0]["temperature"]), color="black", linestyle="--", label="selected")
        axis.set_xscale("log")
        axis.set_xlabel("soft-target temperature")
        axis.set_ylabel("normalized regret")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        figure.savefig(output / "temperature_selection.png", dpi=180)
        plt.close(figure)


def write_dart_calibration_plot(diagnostics: pd.DataFrame, path: Path) -> None:
    """Plot held-out exact disagreement by teacher action and budget."""
    if diagnostics.empty:
        raise ValueError("DART calibration plot requires diagnostics")
    pivot = diagnostics.pivot(
        index="teacher_action", columns="budget", values="disagreement_rate",
    )
    axis = pivot.plot(kind="bar", figsize=(9, 5))
    axis.set_ylabel("held-out action disagreement")
    axis.set_xlabel("teacher action")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(path, dpi=180)
    plt.close(axis.figure)


def write_learning_plots(
    timeseries: pd.DataFrame,
    summary: pd.DataFrame,
    output: Path,
    *,
    learned_methods: tuple[str, ...] = ("learned_direct",),
) -> None:
    if timeseries.empty or summary.empty:
        raise ValueError("learning plots require non-empty evaluation artifacts")
    output.mkdir(parents=True, exist_ok=True)
    _metric_budget_plot(summary, output / "metrics_vs_budget.png")
    _trace_generalization_plot(summary, output / "generalization_by_trace.png")
    _objective_data_ablation_plot(
        summary, output / "objective_data_ablation.png", learned_methods,
    )
    for learned_method in learned_methods:
        _candidate_selection_plot(
            timeseries,
            output / f"candidate_composition_{learned_method}.png",
            learned_method,
        )
        _loss_over_time_plot(
            timeseries,
            output / f"loss_over_time_{learned_method}.png",
            learned_method,
        )


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


def _candidate_selection_plot(
    timeseries: pd.DataFrame,
    path: Path,
    learned_method: str,
) -> None:
    learned = timeseries[timeseries["method"] == learned_method]
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


def _loss_over_time_plot(
    timeseries: pd.DataFrame,
    path: Path,
    learned_method: str,
) -> None:
    learned = timeseries[timeseries["method"] == learned_method]
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


def _objective_data_ablation_plot(
    summary: pd.DataFrame,
    path: Path,
    learned_methods: tuple[str, ...],
) -> None:
    learned = summary[summary["method"].isin(learned_methods)]
    grouped = learned.groupby(["method", "budget"], as_index=False)[
        "mean_approx_loss"
    ].mean()
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for method in learned_methods:
        rows = grouped[grouped["method"] == method].sort_values("budget")
        axis.plot(rows["budget"], rows["mean_approx_loss"], marker="o", label=method)
    axis.set_xlabel("binding transform bound")
    axis.set_ylabel("mean binding-native approximation loss")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    figure.savefig(path, dpi=180)
    plt.close(figure)
