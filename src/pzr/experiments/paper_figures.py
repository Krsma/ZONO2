"""Generate paper-style benchmark figures and plotting CSVs."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from pzr.experiments.benchmark import (
    BenchmarkConfig,
    BenchmarkReport,
    default_methods,
    learned_distilled_method,
    paper_baseline_methods,
    run_benchmark,
)
from pzr.experiments.scenarios import SCENARIOS

PAPER_METHODS = ("box", "girard", "combastel", "methA", "scott", "pca", "adaptive")
METHOD_LABELS = {
    "box": "Box",
    "girard": "Girard",
    "combastel": "Combastel",
    "methA": "MethA",
    "scott": "Scott",
    "pca": "PCA",
    "adaptive": "Adaptive",
    "mpc_rollout_girard": "MPC rollout (ours)",
    "mpc_rollout_wide": "MPC rollout wide (ours)",
    "learned_distilled": "Learned distilled",
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)
    out_dir = Path(args.out)
    data_dir = out_dir / "data"
    figure_dir = out_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    methods = _selected_methods(args.method_set)
    if args.learned_policy:
        methods = (*methods, learned_distilled_method(args.learned_policy))
    plot_methods = tuple(method.name for method in methods if method.kind != "reference")
    seeds = tuple(range(args.seeds))
    budgets = _parse_ints(args.budgets)
    formats = tuple(item.strip() for item in args.formats.split(",") if item.strip())
    base_config = BenchmarkConfig(
        length=args.length,
        budget=args.budget,
        horizon=args.horizon,
        seeds=seeds,
        predictor_mode="online",
        include_reference=True,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )

    simple_time = _run_named(
        "fig3a_simple_time",
        SCENARIOS["robot_simple"](),
        base_config,
        methods,
        data_dir,
    )
    simple_budget = _run_budget_sweep(
        "fig3b_simple_budget",
        SCENARIOS["robot_simple"](),
        base_config,
        budgets,
        methods,
        data_dir,
    )
    omni_time = _run_named(
        "fig5a_omni_time",
        SCENARIOS["robot"](),
        base_config,
        methods,
        data_dir,
    )
    omni_budget = _run_budget_sweep(
        "fig5b_omni_budget",
        SCENARIOS["robot"](),
        base_config,
        budgets,
        methods,
        data_dir,
    )
    fig4_trace = _run_named(
        "fig4a_omni_trace",
        SCENARIOS["robot"](),
        replace(
            base_config,
            length=args.fig4_length,
            seeds=(args.fig4_seed,),
        ),
        methods,
        data_dir,
    )
    fig4_fpr = _run_named(
        "fig4b_omni_false_alarm",
        SCENARIOS["robot"](),
        replace(base_config, length=args.fpr_length),
        methods,
        data_dir,
    )
    thermostat_diag = _run_named(
        "diagnostic_thermostat_selection",
        SCENARIOS["thermostat"](),
        base_config,
        methods,
        data_dir,
    )

    summaries = [
        _write_time_summary(
            "fig3a_simple_error_over_time",
            simple_time.timeseries,
            data_dir / "fig3a_simple_error_over_time.csv",
            plot_methods,
        ),
        _write_budget_summary(
            "fig3b_simple_error_by_budget",
            simple_budget,
            data_dir / "fig3b_simple_error_by_budget.csv",
            plot_methods,
        ),
        _write_time_summary(
            "fig5a_omni_error_over_time",
            omni_time.timeseries,
            data_dir / "fig5a_omni_error_over_time.csv",
            plot_methods,
        ),
        _write_budget_summary(
            "fig5b_omni_error_by_budget",
            omni_budget,
            data_dir / "fig5b_omni_error_by_budget.csv",
            plot_methods,
        ),
        _write_false_alarm_summary(
            "fig4b_omni_false_alarm_rates",
            fig4_fpr.raw_runs,
            data_dir / "fig4b_omni_false_alarm_rates.csv",
            plot_methods,
        ),
    ]
    pd.concat(summaries, ignore_index=True).to_csv(
        data_dir / "figure_summaries.csv",
        index=False,
    )
    fig4_bounds = fig4_trace.bounds_timeseries
    fig4_bounds.to_csv(data_dir / "fig4a_omni_position_x_trace.csv", index=False)
    selection_summary, predicted_sequence_summary = _write_selection_diagnostics(
        (simple_time, omni_time, thermostat_diag),
        data_dir,
    )

    _plot_time(
        summaries[0],
        "interval_hull_mse",
        figure_dir / "fig3a_simple_error_over_time",
        formats,
        plot_methods,
    )
    _plot_budget(
        summaries[1],
        figure_dir / "fig3b_simple_error_by_budget",
        formats,
        plot_methods,
    )
    _plot_bounds(
        fig4_bounds,
        figure_dir / "fig4a_omni_position_x_trace",
        formats,
        plot_methods,
    )
    _plot_false_alarm(
        summaries[4],
        figure_dir / "fig4b_omni_false_alarm_rates",
        formats,
        plot_methods,
    )
    _plot_time(
        summaries[2],
        "interval_hull_mse",
        figure_dir / "fig5a_omni_error_over_time",
        formats,
        plot_methods,
    )
    _plot_budget(
        summaries[3],
        figure_dir / "fig5b_omni_error_by_budget",
        formats,
        plot_methods,
    )
    _plot_selection_summaries(
        selection_summary,
        figure_dir,
        formats,
        plot_methods,
    )
    _plot_fallback_box_usage(
        predicted_sequence_summary,
        figure_dir / "fallback_box_usage",
        formats,
        plot_methods,
    )
    return 0


def _run_named(
    name: str,
    scenario,
    config: BenchmarkConfig,
    methods,
    data_dir: Path,
) -> BenchmarkReport:
    report = run_benchmark(scenario, config, methods=methods)
    report.write_artifacts(data_dir / "runs" / name)
    return report


def _run_budget_sweep(
    name: str,
    scenario,
    config: BenchmarkConfig,
    budgets: tuple[int, ...],
    methods,
    data_dir: Path,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for budget in budgets:
        report = _run_named(
            f"{name}_budget_{budget}",
            scenario,
            replace(config, budget=budget),
            methods,
            data_dir,
        )
        frames.append(report.timeseries)
    return pd.concat(frames, ignore_index=True)


def _write_time_summary(
    name: str,
    timeseries: pd.DataFrame,
    path: Path,
    methods: Sequence[str],
) -> pd.DataFrame:
    data = timeseries[timeseries["method"].isin(methods)]
    summary = _mean_ci(
        data,
        ["scenario", "budget", "method", "step"],
        "interval_hull_mse",
    )
    summary.insert(0, "figure", name)
    summary.to_csv(path, index=False)
    return summary


def _write_budget_summary(
    name: str,
    timeseries: pd.DataFrame,
    path: Path,
    methods: Sequence[str],
) -> pd.DataFrame:
    data = timeseries[timeseries["method"].isin(methods)]
    seed_means = (
        data.groupby(["scenario", "budget", "method", "seed"], sort=True)[
            "interval_hull_mse"
        ]
        .mean()
        .reset_index()
    )
    summary = _mean_ci(seed_means, ["scenario", "budget", "method"], "interval_hull_mse")
    summary.insert(0, "figure", name)
    summary.to_csv(path, index=False)
    return summary


def _write_false_alarm_summary(
    name: str,
    raw_runs: pd.DataFrame,
    path: Path,
    methods: Sequence[str],
) -> pd.DataFrame:
    data = raw_runs[raw_runs["method"].isin(methods)]
    summary = _mean_ci(data, ["scenario", "budget", "method"], "false_alarm_rate")
    summary.insert(0, "figure", name)
    summary.to_csv(path, index=False)
    return summary


def _write_selection_diagnostics(
    reports: Sequence[BenchmarkReport],
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection = _concat_report_frames(reports, "selection_summary")
    predicted = _concat_report_frames(reports, "predicted_sequence_summary")
    selection.to_csv(data_dir / "selection_summary.csv", index=False)
    predicted.to_csv(data_dir / "predicted_sequence_summary.csv", index=False)
    fallback_columns = [
        "scenario",
        "predictor_mode",
        "method",
        "decision_count",
        "first_action_box_count",
        "future_box_count",
        "future_box_fraction",
    ]
    fallback = (
        predicted[fallback_columns]
        if not predicted.empty
        else pd.DataFrame(columns=fallback_columns)
    )
    fallback.to_csv(data_dir / "fallback_box_usage.csv", index=False)
    _write_analysis_notes(data_dir, reports, predicted)
    return selection, predicted


def _concat_report_frames(
    reports: Sequence[BenchmarkReport],
    attr: str,
) -> pd.DataFrame:
    frames = [getattr(report, attr) for report in reports if not getattr(report, attr).empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _write_analysis_notes(
    data_dir: Path,
    reports: Sequence[BenchmarkReport],
    predicted: pd.DataFrame,
) -> None:
    raw = _concat_report_frames(reports, "raw_runs")
    summary = _concat_report_frames(reports, "summary")
    notes = {
        "top_winners": _top_winners(summary),
        "soundness_checks": {
            "budget_violation_count": _sum_column(raw, "budget_violation_count"),
            "unsound_certificate_count": _sum_column(raw, "unsound_certificate_count"),
            "reduction_failure_count": _sum_column(raw, "reduction_failure_count"),
            "no_op_count": _sum_column(raw, "no_op_count"),
        },
        "warning_flags": _warning_flags(raw, predicted),
    }
    (data_dir / "analysis_notes.json").write_text(
        json.dumps(notes, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _top_winners(summary: pd.DataFrame) -> list[dict[str, object]]:
    if summary.empty:
        return []
    metrics = {
        "inconclusive_rate",
        "extra_inconclusive_count",
        "unsafe_disagreement_count",
        "mean_trigger_width",
        "mean_width_inflation",
    }
    rows: list[dict[str, object]] = []
    for key, group in summary[summary["metric"].isin(metrics)].groupby(
        ["scenario", "predictor_mode", "metric"],
        sort=True,
    ):
        scenario, predictor_mode, metric = key
        winner = group.sort_values(["mean", "method"], ascending=[True, True]).iloc[0]
        rows.append(
            {
                "scenario": scenario,
                "predictor_mode": predictor_mode,
                "metric": metric,
                "method": winner["method"],
                "mean": float(winner["mean"]),
            }
        )
    return rows


def _warning_flags(raw: pd.DataFrame, predicted: pd.DataFrame) -> list[str]:
    flags: list[str] = []
    for column in (
        "budget_violation_count",
        "unsound_certificate_count",
        "reduction_failure_count",
        "no_op_count",
    ):
        value = _sum_column(raw, column)
        if value:
            flags.append(f"{column}={value}")
    if not predicted.empty:
        wide = predicted[predicted["method"] == "mpc_rollout_wide"]
        first_box = _sum_column(wide, "first_action_box_count")
        if first_box:
            flags.append(f"mpc_rollout_wide_first_action_box_count={first_box}")
    return flags


def _sum_column(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    return int(frame[column].fillna(0).sum())


def _mean_ci(data: pd.DataFrame, keys: list[str], metric: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for values_key, group in data.groupby(keys, sort=True):
        values = group[metric].astype(float).to_numpy()
        mean = float(np.mean(values)) if values.size else 0.0
        if values.size > 1:
            ci = 1.96 * float(np.std(values, ddof=1)) / float(np.sqrt(values.size))
        else:
            ci = 0.0
        if not isinstance(values_key, tuple):
            values_key = (values_key,)
        row = {key: value for key, value in zip(keys, values_key)}
        row.update(
            {
                "metric": metric,
                "n": int(values.size),
                "mean": mean,
                "ci95_low": max(0.0, mean - ci),
                "ci95_high": mean + ci,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_time(
    summary: pd.DataFrame,
    metric: str,
    path: Path,
    formats: Sequence[str],
    methods: Sequence[str],
) -> None:
    _ = metric
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    for method in methods:
        data = summary[summary["method"] == method].sort_values("step")
        if data.empty:
            continue
        x = data["step"].to_numpy(dtype=float)
        y = np.maximum(data["mean"].to_numpy(dtype=float), 1e-16)
        low = np.maximum(data["ci95_low"].to_numpy(dtype=float), 1e-16)
        high = np.maximum(data["ci95_high"].to_numpy(dtype=float), 1e-16)
        ax.plot(x, y, label=METHOD_LABELS.get(method, method), linewidth=1.7)
        ax.fill_between(x, low, high, alpha=0.16)
    ax.set_xlabel("Step")
    ax.set_ylabel("Interval-hull MSE")
    ax.set_yscale("log")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    ax.legend(ncol=2, fontsize=8)
    _save(fig, path, formats)


def _plot_budget(
    summary: pd.DataFrame,
    path: Path,
    formats: Sequence[str],
    methods: Sequence[str],
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    for method in methods:
        data = summary[summary["method"] == method].sort_values("budget")
        if data.empty:
            continue
        x = data["budget"].to_numpy(dtype=float)
        y = np.maximum(data["mean"].to_numpy(dtype=float), 1e-16)
        low = np.maximum(data["ci95_low"].to_numpy(dtype=float), 1e-16)
        high = np.maximum(data["ci95_high"].to_numpy(dtype=float), 1e-16)
        ax.plot(x, y, marker="o", label=METHOD_LABELS.get(method, method), linewidth=1.7)
        ax.fill_between(x, low, high, alpha=0.16)
    ax.set_xlabel("Generator budget")
    ax.set_ylabel("Mean interval-hull MSE")
    ax.set_yscale("log")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    ax.legend(ncol=2, fontsize=8)
    _save(fig, path, formats)


def _plot_bounds(
    bounds: pd.DataFrame,
    path: Path,
    formats: Sequence[str],
    methods: Sequence[str],
) -> None:
    data = bounds[bounds["state_name"] == "position_x"].copy()
    if data.empty:
        data = bounds[bounds["state_index"] == 0].copy()
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    reference = data[data["method"] == "reference"].sort_values("step")
    if not reference.empty:
        x = reference["step"].to_numpy(dtype=float)
        center_column = (
            "reference_center"
            if "reference_center" in reference.columns
            else "center"
        )
        ax.plot(
            x,
            reference[center_column].to_numpy(dtype=float),
            color="black",
            linewidth=1.5,
            label="Reference center",
        )
        ax.fill_between(
            x,
            reference["lower"].to_numpy(dtype=float),
            reference["upper"].to_numpy(dtype=float),
            color="0.75",
            alpha=0.65,
            label="Reference band",
        )
    for method in methods:
        method_data = data[data["method"] == method].sort_values("step")
        if method_data.empty:
            continue
        x = method_data["step"].to_numpy(dtype=float)
        color = ax.plot(
            x,
            method_data["upper"].to_numpy(dtype=float),
            label=METHOD_LABELS.get(method, method),
            linewidth=1.3,
        )[0].get_color()
        ax.plot(
            x,
            method_data["lower"].to_numpy(dtype=float),
            color=color,
            linewidth=1.0,
            linestyle="--",
            alpha=0.8,
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Position x bounds")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(ncol=2, fontsize=8)
    _save(fig, path, formats)


def _plot_false_alarm(
    summary: pd.DataFrame,
    path: Path,
    formats: Sequence[str],
    methods: Sequence[str],
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    data = summary.set_index("method").reindex(methods).dropna(subset=["mean"])
    x = np.arange(len(data))
    means = data["mean"].to_numpy(dtype=float)
    errors = np.vstack(
        [
            means - data["ci95_low"].to_numpy(dtype=float),
            data["ci95_high"].to_numpy(dtype=float) - means,
        ]
    )
    ax.bar(x, means, yerr=errors, capsize=3.0)
    ax.set_xticks(x, [METHOD_LABELS.get(method, method) for method in data.index], rotation=25)
    ax.set_ylabel("False alarm rate")
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)
    _save(fig, path, formats)


def _plot_selection_summaries(
    selection: pd.DataFrame,
    figure_dir: Path,
    formats: Sequence[str],
    methods: Sequence[str],
) -> None:
    if selection.empty:
        for scenario in ("robot_simple", "robot", "thermostat"):
            _plot_empty(figure_dir / f"selection_{scenario}", formats, "No selections")
        return
    for scenario in ("robot_simple", "robot", "thermostat"):
        data = selection[
            (selection["scenario"] == scenario) & selection["method"].isin(methods)
        ].copy()
        path = figure_dir / f"selection_{scenario}"
        if data.empty:
            _plot_empty(path, formats, "No selections")
            continue
        pivot = data.pivot_table(
            index="method",
            columns="selected_reducer",
            values="selection_fraction",
            aggfunc="sum",
            fill_value=0.0,
        ).reindex([method for method in methods if method in set(data["method"])])
        reducers = sorted(pivot.columns)
        fig, ax = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
        x = np.arange(len(pivot.index))
        bottom = np.zeros(len(pivot.index), dtype=float)
        for reducer in reducers:
            values = pivot[reducer].to_numpy(dtype=float)
            ax.bar(x, values, bottom=bottom, label=reducer)
            bottom += values
        ax.set_xticks(
            x,
            [METHOD_LABELS.get(method, method) for method in pivot.index],
            rotation=25,
            ha="right",
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Selection fraction")
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)
        ax.legend(ncol=2, fontsize=8)
        _save(fig, path, formats)


def _plot_fallback_box_usage(
    predicted: pd.DataFrame,
    path: Path,
    formats: Sequence[str],
    methods: Sequence[str],
) -> None:
    if predicted.empty:
        _plot_empty(path, formats, "No MPC sequences")
        return
    data = predicted[predicted["method"].isin(methods)].copy()
    if data.empty:
        _plot_empty(path, formats, "No MPC sequences")
        return
    labels = []
    first_values = []
    future_values = []
    for _, row in data.sort_values(["scenario", "method"]).iterrows():
        labels.append(
            f"{row['scenario']}\n{METHOD_LABELS.get(str(row['method']), row['method'])}"
        )
        first_values.append(float(row["first_action_box_fraction"]))
        future_values.append(float(row["future_box_fraction"]))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    ax.bar(x, first_values, label="First action")
    ax.bar(x, future_values, bottom=first_values, label="Future fallback")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Box usage fraction")
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)
    ax.legend(fontsize=8)
    _save(fig, path, formats)


def _plot_empty(path: Path, formats: Sequence[str], message: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_axis_off()
    _save(fig, path, formats)


def _save(fig, path: Path, formats: Sequence[str]) -> None:
    for fmt in formats:
        fig.savefig(path.with_suffix(f".{fmt}"), dpi=180)
    plt.close(fig)


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pzr-paper-figures",
        description="Generate Figures 3-5 style plots from benchmark runs.",
    )
    parser.add_argument("--out", type=str, default="results/paper-figures")
    parser.add_argument(
        "--method-set",
        choices=("paper", "paper_plus_ours", "paper_plus_wide", "extended"),
        default="paper",
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--budgets", type=str, default="6,8,10,12,16,20")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--fig4-length", type=int, default=20)
    parser.add_argument("--fig4-seed", type=int, default=0)
    parser.add_argument("--fpr-length", type=int, default=1000)
    parser.add_argument("--formats", type=str, default="png,pdf")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--learned-policy", type=str, default=None)
    return parser


def _selected_methods(method_set: str):
    if method_set == "paper":
        return paper_baseline_methods()
    if method_set == "paper_plus_ours":
        ours = tuple(
            method for method in default_methods() if method.name == "mpc_rollout_girard"
        )
        return (*paper_baseline_methods(), *ours)
    if method_set == "paper_plus_wide":
        ours = tuple(
            method
            for method in default_methods()
            if method.name in {"mpc_rollout_girard", "mpc_rollout_wide"}
        )
        return (*paper_baseline_methods(), *ours)
    if method_set == "extended":
        return default_methods()
    raise ValueError(f"unknown method set: {method_set}")


if __name__ == "__main__":
    raise SystemExit(main())
