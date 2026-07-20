"""Diagnostic and paper-facing plots for learned reducer evaluation."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from pzr.learning.ranker import ReducerTrainingResult

def write_training_plots(
    temperatures: pd.DataFrame,
    result: ReducerTrainingResult,
    output: Path,
) -> None:
    """Write selected-objective histories and temperature diagnostics."""
    plt = _pyplot()
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


def write_dart_calibration_plot(
    budget_diagnostics: pd.DataFrame,
    direction_diagnostics: pd.DataFrame,
    path: Path,
) -> None:
    """Plot calibrated disturbance magnitude, eligibility, and regret radius."""
    plt = _pyplot()
    if budget_diagnostics.empty or direction_diagnostics.empty:
        raise ValueError("DART calibration plot requires diagnostics")
    ordered = budget_diagnostics.sort_values("budget")
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    for column, label in (
        ("target_disturbance_rate", "target novice error"),
        ("expected_disturbance_rate", "expected disturbed"),
        ("eligible_fraction", "eligible states"),
        ("injection_probability", "injection coin"),
    ):
        axes[0].plot(ordered["budget"], ordered[column], marker="o", label=label)
    axes[0].set_ylabel("fraction / probability")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].bar(ordered["budget"].astype(str), ordered["regret_cap"])
    axes[1].set_xlabel("binding transform bound")
    axes[1].set_ylabel("Q90 normalized-regret cap")
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
