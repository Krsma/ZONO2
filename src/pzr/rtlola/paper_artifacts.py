"""Publication tables and figures for the paper evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from pzr.artifact_io import write_csv_atomic, write_json_atomic, write_text_atomic
from pzr.rtlola.paper_experiment import (
    COMPOSITION_METHODS,
    HEADLINE_METHODS,
    ORDINARY_REDUCERS,
    PaperExperimentConfig,
    RunState,
    aggregate_trace_metrics,
    artifact_hash_manifest,
    reducer_composition,
    trace_level_metrics,
)


METHOD_LABELS = {
    "girard": "Girard",
    "scott": "Scott",
    "pca": "PCA",
    "combastel": "Combastel",
    "mpc_terminal_beam": "Terminal beam (offline)",
    "mpc_terminal_beam_predictive_linear": "Terminal beam (linear)",
    "mpc_terminal_full_width": "Full-width teacher",
    "mpc_cumulative_beam": "Cumulative beam (offline)",
    "pairwise_ranking_policy": "Pairwise ranking policy",
    "pairwise_ranking_policy_budget80": "Budget-80 policy",
}
COLORS = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
    "#56B4E9", "#000000", "#999999", "#F0E442",
)
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "h")
LINESTYLES = ("-", "--", "-.", ":", "-", "--", "-.", ":", "-")


def write_paper_evaluation_reports(
    config: PaperExperimentConfig,
    *,
    headline_summary: pd.DataFrame,
    generalization_summary: pd.DataFrame,
    objective_summary: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    timing_summary: pd.DataFrame,
    composition_timeseries: pd.DataFrame,
    pilot_projection: Mapping[str, object],
    output: Path | None = None,
) -> Path:
    """Write compact source tables, TeX, and deterministic PDF/PNG figures."""
    destination = config.paper_artifact_dir if output is None else output
    destination.mkdir(parents=True, exist_ok=True)
    headline = aggregate_trace_metrics(headline_summary)
    generalization = aggregate_trace_metrics(generalization_summary)
    objective = objective_comparison_table(objective_summary)
    long_table = headline_long_table(
        headline_summary, timing_summary, config.budgets, config.figure8_conditions,
    )
    fallback = fallback_diagnostics(headline_summary)
    extrapolation = budget80_extrapolation(generalization)
    composition = reducer_composition(composition_timeseries)
    ablation = ablation_table(ablation_summary)

    tables = {
        "headline_aggregates.csv": headline,
        "generalization_aggregates.csv": generalization,
        "objective_comparison.csv": objective,
        "headline_long_table.csv": long_table,
        "fallback_diagnostics.csv": fallback,
        "budget80_extrapolation.csv": extrapolation,
        "reducer_composition.csv": composition,
        "ablation_heatmaps.csv": ablation,
        "timing_summary.csv": timing_summary,
    }
    for name, frame in tables.items():
        write_csv_atomic(frame, destination / name)
    write_text_atomic(_headline_tex(long_table), destination / "headline_long_table.tex")
    write_text_atomic(_fallback_tex(fallback), destination / "fallback_diagnostics.tex")
    write_text_atomic(
        _extrapolation_tex(extrapolation),
        destination / "budget80_extrapolation.tex",
    )
    write_text_atomic(
        _objective_tex(objective),
        destination / "objective_comparison.tex",
    )
    write_json_atomic(dict(pilot_projection), destination / "pilot_projection.json")

    _plot_budget_facets(headline, destination / "headline_budget_curves")
    _plot_composition(composition, destination / "reducer_composition")
    _plot_ablation(ablation, destination / "ablation_heatmaps")
    write_json_atomic({
        "schema": "pzr.paper-evaluation-report.v1",
        "config_sha256": config.config_sha256,
        "bootstrap": {
            "replicates": 10_000,
            "interval": "paired seed-level percentile 95% CI",
            "aggregation_unit": "trace",
        },
        "missing_points": "unavailable if any contributing run failed",
        "loss_scale": _loss_scale(headline),
        "ordinary_composition_exclusions": ["none", "fallback", "infeasible_event"],
    }, destination / "report_manifest.json")
    write_json_atomic(
        artifact_hash_manifest(destination),
        destination / "artifact_hashes.json",
    )
    return destination


def headline_long_table(
    summary: pd.DataFrame,
    timing: pd.DataFrame,
    budgets: Sequence[int],
    conditions: Sequence[str],
) -> pd.DataFrame:
    """Build seven budget blocks with eight methods and grouped condition metrics."""
    data = trace_level_metrics(summary)
    rows: list[dict[str, object]] = []
    for budget in budgets:
        for method in HEADLINE_METHODS:
            row: dict[str, object] = {"budget": int(budget), "method": method}
            for condition in conditions:
                cell = data[
                    (data["budget"] == budget)
                    & (data["method"] == method)
                    & (data["condition"] == condition)
                ]
                if len(cell) != 1:
                    raise ValueError(
                        f"headline cell is missing or duplicated: {condition}/{budget}/{method}"
                    )
                completed = cell.iloc[0]["status"] == RunState.COMPLETED.value
                prefix = _short_condition(condition)
                row[f"{prefix}_fpr"] = float(cell.iloc[0]["fpr"]) if completed else np.nan
                row[f"{prefix}_loss"] = (
                    float(cell.iloc[0]["mean_approx_loss"]) if completed else np.nan
                )
                timing_cell = timing[
                    (timing["budget"] == budget)
                    & (timing["method"] == method)
                    & (timing["condition"] == condition)
                ]
                row[f"{prefix}_throughput"] = (
                    float(timing_cell["median_throughput_events_per_second"].iloc[0])
                    if completed and len(timing_cell) == 1 else np.nan
                )
            rows.append(row)
    result = pd.DataFrame(rows)
    expected = len(budgets) * len(HEADLINE_METHODS)
    if len(result) != expected:
        raise AssertionError(f"headline table has {len(result)} rows, expected {expected}")
    return result


def objective_comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the matched terminal and cumulative offline beam comparison."""
    expected = {"mpc_terminal_beam", "mpc_cumulative_beam"}
    if set(summary["method"].astype(str)) != expected:
        raise ValueError("objective comparison method identities differ")
    result = aggregate_trace_metrics(summary)
    counts = result.groupby(["condition", "budget"])["method"].nunique()
    if bool((counts != 2).any()):
        raise ValueError("objective comparison methods do not align")
    return result.sort_values(["condition", "budget", "method"]).reset_index(drop=True)


def fallback_diagnostics(summary: pd.DataFrame) -> pd.DataFrame:
    """Keep fallback progress diagnostics adjacent to unavailable headline values."""
    columns = [
        "condition", "seed", "budget", "method", "status", "first_fallback_event",
        "completed_fraction", "pre_fallback_mean_loss",
        "pre_fallback_throughput_events_per_second",
    ]
    data = summary.copy()
    for name in columns:
        if name not in data:
            data[name] = np.nan
    return data.loc[data["status"] != RunState.COMPLETED.value, columns].reset_index(
        drop=True,
    )


def budget80_extrapolation(generalization: pd.DataFrame) -> pd.DataFrame:
    """Compare the budget-80-only policy with the all-budget policy everywhere."""
    methods = ("pairwise_ranking_policy", "pairwise_ranking_policy_budget80")
    selected = generalization[generalization["method"].isin(methods)].copy()
    counts = selected.groupby(["condition", "budget"])["method"].nunique()
    if not counts.empty and bool((counts != 2).any()):
        raise ValueError("budget-80 extrapolation policies do not align")
    return selected.sort_values(["condition", "budget", "method"]).reset_index(drop=True)


def ablation_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the H/W grid without admitting invalid runs into main values."""
    required = {"condition", "horizon", "beam_width", "status"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"ablation summary lacks columns: {sorted(missing)}")
    data = trace_level_metrics(summary)
    rows = []
    for keys, frame in data.groupby(["condition", "horizon", "beam_width"], sort=True):
        condition, horizon, width = keys
        completed = frame[frame["status"] == RunState.COMPLETED.value]
        available = len(completed) == len(frame)
        rows.append({
            "condition": condition,
            "horizon": int(horizon),
            "beam_width": int(width),
            "available": available,
            "valid_count": len(completed),
            "failed_count": len(frame) - len(completed),
            "mean_loss": (
                float(completed["mean_approx_loss"].mean()) if available else np.nan
            ),
            "macro_fpr": float(completed["fpr"].mean()) if available else np.nan,
            "median_throughput_events_per_second": (
                float(completed["throughput_events_per_second"].median())
                if available else np.nan
            ),
            "highlight_default": int(horizon) == 4 and int(width) == 4,
        })
    return pd.DataFrame(rows)


def _plot_budget_facets(data: pd.DataFrame, stem: Path) -> None:
    plt = _pyplot()
    conditions = tuple(dict.fromkeys(data["condition"].astype(str)))
    methods = tuple(method for method in HEADLINE_METHODS if method in set(data["method"]))
    figure, axes = plt.subplots(
        3, len(conditions), figsize=(7.05, 5.7), sharex=True,
        constrained_layout=True, squeeze=False,
    )
    for column, condition in enumerate(conditions):
        for method_index, method in enumerate(methods):
            frame = data[
                (data["condition"] == condition) & (data["method"] == method)
            ].sort_values("budget")
            style = {
                "color": COLORS[method_index],
                "marker": MARKERS[method_index],
                "linestyle": LINESTYLES[method_index],
                "linewidth": 1.0,
                "markersize": 3.5,
            }
            axes[0, column].plot(frame["budget"], frame["macro_fpr"], **style)
            axes[0, column].fill_between(
                frame["budget"], frame["macro_fpr_ci_low"], frame["macro_fpr_ci_high"],
                color=COLORS[method_index], alpha=0.12, linewidth=0,
            )
            axes[1, column].plot(
                frame["budget"], frame["macro_mean_approx_loss"], **style,
            )
            axes[2, column].plot(frame["budget"], frame["fallback_rate"], **style)
        axes[0, column].set_title(_condition_label(condition), fontsize=8)
        axes[2, column].set_xscale("log")
        axes[2, column].set_xlabel("Transform bound")
        for row in range(3):
            axes[row, column].grid(True, color="#dddddd", linewidth=0.4)
    axes[0, 0].set_ylabel("Macro FPR")
    axes[1, 0].set_ylabel("Mean native loss")
    axes[2, 0].set_ylabel("Fallback rate")
    if _loss_scale(data) == "log":
        for axis in axes[1]:
            axis.set_yscale("log")
    handles = [
        plt.Line2D(
            [], [], color=COLORS[index], marker=MARKERS[index],
            linestyle=LINESTYLES[index], label=METHOD_LABELS.get(method, method),
        )
        for index, method in enumerate(methods)
    ]
    figure.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=6)
    _save_figure(figure, stem)
    plt.close(figure)


def _plot_composition(data: pd.DataFrame, stem: Path) -> None:
    plt = _pyplot()
    methods = tuple(method for method in COMPOSITION_METHODS if method in set(data["method"]))
    conditions = tuple(dict.fromkeys(data["condition"].astype(str)))
    if not methods or not conditions:
        figure, axis = plt.subplots(figsize=(3.35, 1.8), constrained_layout=True)
        axis.text(0.5, 0.5, "No ordinary reducer selections", ha="center", va="center")
        axis.set_axis_off()
        _save_figure(figure, stem)
        plt.close(figure)
        return
    figure, axes = plt.subplots(
        len(methods), len(conditions), figsize=(7.05, 1.65 * len(methods)),
        sharex=True, sharey=True, constrained_layout=True, squeeze=False,
    )
    for row, method in enumerate(methods):
        for column, condition in enumerate(conditions):
            axis = axes[row, column]
            frame = data[(data["method"] == method) & (data["condition"] == condition)]
            budgets = sorted(frame["budget"].unique())
            bottom = np.zeros(len(budgets))
            for reducer_index, reducer in enumerate(ORDINARY_REDUCERS):
                values = np.asarray([
                    frame.loc[
                        (frame["budget"] == budget) & (frame["reducer_used"] == reducer),
                        "percentage",
                    ].sum()
                    for budget in budgets
                ])
                axis.bar(
                    budgets, values, bottom=bottom, width=np.asarray(budgets) * 0.20,
                    color=COLORS[reducer_index], label=METHOD_LABELS[reducer],
                )
                bottom += values
            axis.set_xscale("log")
            axis.set_ylim(0, 100)
            axis.grid(True, axis="y", color="#dddddd", linewidth=0.4)
            if row == 0:
                axis.set_title(_condition_label(condition), fontsize=8)
            if column == 0:
                axis.set_ylabel(f"{METHOD_LABELS[method]}\nshare (%)")
            if row == len(methods) - 1:
                axis.set_xlabel("Transform bound")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=7)
    _save_figure(figure, stem)
    plt.close(figure)


def _plot_ablation(data: pd.DataFrame, stem: Path) -> None:
    plt = _pyplot()
    conditions = tuple(dict.fromkeys(data["condition"].astype(str)))
    metrics = (
        ("mean_loss", "Native loss"),
        ("macro_fpr", "Macro FPR"),
        ("median_throughput_events_per_second", "Events/s"),
    )
    figure, axes = plt.subplots(
        len(metrics), len(conditions), figsize=(7.05, 5.0),
        constrained_layout=True, squeeze=False,
    )
    for row, (metric, label) in enumerate(metrics):
        finite = data[metric].to_numpy(dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        vmin = float(np.min(finite)) if len(finite) else 0.0
        vmax = float(np.max(finite)) if len(finite) else 1.0
        image_artist = None
        for column, condition in enumerate(conditions):
            axis = axes[row, column]
            frame = data[data["condition"] == condition]
            horizons = sorted(frame["horizon"].unique())
            widths = sorted(frame["beam_width"].unique())
            matrix = np.full((len(horizons), len(widths)), np.nan)
            for i, horizon in enumerate(horizons):
                for j, width in enumerate(widths):
                    cell = frame[
                        (frame["horizon"] == horizon) & (frame["beam_width"] == width)
                    ]
                    if len(cell) == 1:
                        matrix[i, j] = float(cell.iloc[0][metric])
            image_artist = axis.imshow(
                matrix, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto",
            )
            axis.set_xticks(range(len(widths)), widths)
            axis.set_yticks(range(len(horizons)), horizons)
            for i, horizon in enumerate(horizons):
                for j, width in enumerate(widths):
                    if not np.isfinite(matrix[i, j]):
                        axis.text(j, i, "×", ha="center", va="center", color="white")
                    if horizon == 4 and width == 4:
                        axis.add_patch(plt.Rectangle(
                            (j - 0.48, i - 0.48), 0.96, 0.96,
                            fill=False, edgecolor="white", linewidth=1.2,
                        ))
            if row == 0:
                axis.set_title(_condition_label(condition), fontsize=8)
            if column == 0:
                axis.set_ylabel(f"{label}\nHorizon")
            if row == len(metrics) - 1:
                axis.set_xlabel("Beam width")
        if image_artist is not None:
            figure.colorbar(
                image_artist, ax=axes[row, :].tolist(), label=label,
                fraction=0.025, pad=0.02,
            )
    _save_figure(figure, stem)
    plt.close(figure)


def _headline_tex(frame: pd.DataFrame) -> str:
    condition_prefixes = [
        column.removesuffix("_fpr") for column in frame if column.endswith("_fpr")
    ]
    columns = "ll" + "rrr" * len(condition_prefixes)
    lines = [
        "% Generated by pzr-paper; do not edit.",
        f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        "Budget & Method & " + " & ".join(
            f"\\multicolumn{{3}}{{c}}{{{_tex(prefix)}}}" for prefix in condition_prefixes
        ) + " \\\\",
        " &  & " + " & ".join("FPR & Loss & Events/s" for _ in condition_prefixes) + " \\\\",
        "\\midrule",
    ]
    previous_budget = None
    for row in frame.itertuples(index=False):
        budget = str(row.budget) if row.budget != previous_budget else ""
        values = []
        for prefix in condition_prefixes:
            values.extend((
                _format_number(getattr(row, f"{prefix}_fpr")),
                _format_number(getattr(row, f"{prefix}_loss")),
                _format_number(getattr(row, f"{prefix}_throughput")),
            ))
        lines.append(
            f"{budget} & {_tex(METHOD_LABELS.get(row.method, row.method))} & "
            + " & ".join(values) + " \\\\"
        )
        previous_budget = row.budget
    lines.extend(("\\bottomrule", "\\end{tabular}", ""))
    return "\n".join(lines)


def _fallback_tex(frame: pd.DataFrame) -> str:
    lines = [
        "% Generated by pzr-paper; do not edit.",
        "\\begin{tabular}{llrlrr}",
        "\\toprule",
        "Condition & Method & Budget & State & First event & Completed \\\\",
        "\\midrule",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"{_tex(str(row.condition))} & {_tex(METHOD_LABELS.get(row.method, row.method))} "
            f"& {row.budget} & {_tex(str(row.status))} & "
            f"{_format_number(row.first_fallback_event)} & "
            f"{_format_number(row.completed_fraction)} \\\\"
        )
    lines.extend(("\\bottomrule", "\\end{tabular}", ""))
    return "\n".join(lines)


def _extrapolation_tex(frame: pd.DataFrame) -> str:
    columns = ["condition", "budget", "method", "macro_fpr", "macro_mean_approx_loss"]
    compact = frame[columns] if set(columns) <= set(frame.columns) else pd.DataFrame(columns=columns)
    lines = [
        "% Generated by pzr-paper; do not edit.",
        "\\begin{tabular}{lrlll}",
        "\\toprule",
        "Condition & Budget & Policy & Macro FPR & Mean loss \\\\",
        "\\midrule",
    ]
    for row in compact.itertuples(index=False):
        lines.append(
            f"{_tex(str(row.condition))} & {row.budget} & "
            f"{_tex(METHOD_LABELS.get(row.method, row.method))} & "
            f"{_format_number(row.macro_fpr)} & "
            f"{_format_number(row.macro_mean_approx_loss)} \\\\"
        )
    lines.extend(("\\bottomrule", "\\end{tabular}", ""))
    return "\n".join(lines)


def _objective_tex(frame: pd.DataFrame) -> str:
    lines = [
        "% Generated by pzr-paper; do not edit.",
        "\\begin{tabular}{lrllll}",
        "\\toprule",
        "Condition & Budget & Objective & Available & Macro FPR & Mean loss \\\\",
        "\\midrule",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"{_tex(str(row.condition))} & {row.budget} & "
            f"{_tex(METHOD_LABELS.get(row.method, row.method))} & "
            f"{'yes' if bool(row.available) else 'no'} & "
            f"{_format_number(row.macro_fpr)} & "
            f"{_format_number(row.macro_mean_approx_loss)} \\\\"
        )
    lines.extend(("\\bottomrule", "\\end{tabular}", ""))
    return "\n".join(lines)


def _loss_scale(data: pd.DataFrame) -> str:
    values = data["macro_mean_approx_loss"].to_numpy(dtype=np.float64)
    displayed = values[np.isfinite(values)]
    return "log" if len(displayed) and bool(np.all(displayed > 0.0)) else "linear"


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return plt


def _save_figure(figure: object, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    figure.savefig(
        stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02,
    )


def _short_condition(condition: str) -> str:
    return condition.removeprefix("figure8_").replace("figure8", "nominal")


def _condition_label(condition: str) -> str:
    return _short_condition(condition).replace("_", " ").title()


def _format_number(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(number):
        return "--"
    return f"{number:.3g}"


def _tex(value: str) -> str:
    return value.replace("_", "\\_").replace("%", "\\%")
