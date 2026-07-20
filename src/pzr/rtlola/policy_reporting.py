"""Tables and diagnostic plots for fixed-trace policy evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pzr.artifact_io import write_csv_atomic
from pzr.rtlola.benchmark import summarize_prediction_errors

if TYPE_CHECKING:
    from pzr.rtlola.policy_evaluation import FixedPolicyEvaluationConfig, PolicyComparison


COMPARISON_METRICS = (
    "fpr",
    "fnr",
    "mean_approx_loss",
    "final_approx_loss",
    "max_approx_loss",
    "sum_approx_loss",
    "mean_state_width",
    "max_state_width",
    "total_time_ms",
)


def write_policy_reports(
    config: FixedPolicyEvaluationConfig,
    timeseries: pd.DataFrame,
    summary: pd.DataFrame,
    prediction_errors: pd.DataFrame | None = None,
) -> None:
    """Write all aggregate policy-evaluation tables and plots."""
    write_csv_atomic(timeseries, config.output / "timeseries.csv")
    write_csv_atomic(summary, config.output / "summary.csv")
    predictions = prediction_errors if prediction_errors is not None else pd.DataFrame()
    write_csv_atomic(predictions, config.output / "input_prediction_errors.csv")
    write_csv_atomic(
        summarize_prediction_errors(predictions),
        config.output / "input_prediction_error_summary.csv",
    )
    write_csv_atomic(candidate_selection(timeseries), config.output / "candidate_selection.csv")
    write_csv_atomic(decision_accounting(timeseries), config.output / "decision_accounting.csv")
    macro = macro_metrics(summary)
    write_csv_atomic(macro, config.output / "macro_metrics.csv")
    write_csv_atomic(
        macro[[
            "method", "budget", "aggregation", "mean_approx_loss",
            "final_approx_loss", "max_approx_loss", "sum_approx_loss",
        ]],
        config.output / "macro_loss_metrics.csv",
    )
    write_csv_atomic(
        macro[["method", "budget", "aggregation", "mean_state_width", "max_state_width"]],
        config.output / "macro_width_metrics.csv",
    )
    write_csv_atomic(
        macro[["method", "budget", "aggregation", "total_time_ms"]],
        config.output / "macro_runtime_metrics.csv",
    )
    write_csv_atomic(micro_trigger_metrics(summary), config.output / "micro_trigger_metrics.csv")
    comparisons = pd.concat([
        comparison_to_reference(summary, policy_method, benchmark_method)
        for policy_method in config.model_names
        for benchmark_method in config.benchmark_methods
    ], ignore_index=True)
    write_csv_atomic(comparisons, config.output / "method_comparisons.csv")
    static_methods = tuple(
        method for method in config.benchmark_methods
        if method in {"girard", "scott", "pca", "combastel"}
    )
    write_csv_atomic(
        best_static_metrics(summary, static_methods),
        config.output / "best_static_metrics.csv",
    )
    if config.comparisons:
        write_csv_atomic(
            explicit_policy_comparisons(summary, config.comparisons),
            config.output / "policy_comparisons.csv",
        )
    write_policy_plots(
        timeseries,
        summary,
        config.output / "plots",
        policy_methods=config.model_names,
        comparisons=config.comparisons,
    )


def candidate_selection(timeseries: pd.DataFrame) -> pd.DataFrame:
    data = timeseries.copy()
    data["reduction_required"] = data["pre_generator_count"] > data["budget"]
    groups = ["trace_kind", "budget", "method", "reduction_required", "reducer_used"]
    result = data.groupby(groups, dropna=False).size().rename("count").reset_index()
    totals = result.groupby(groups[:-1], dropna=False)["count"].transform("sum")
    result["fraction"] = result["count"] / totals
    return result


def decision_accounting(timeseries: pd.DataFrame) -> pd.DataFrame:
    data = timeseries.copy()
    data["reduction_required"] = data["pre_generator_count"] > data["budget"]
    data["automatic_none"] = ~data["reduction_required"] & (data["reducer_used"] == "none")
    data["infeasible_step"] = data["infeasible_candidate_count"] > 0
    rows = []
    for keys, frame in data.groupby(["trace_kind", "budget", "method"], dropna=False):
        trace_kind, budget, method = keys
        required = frame["reduction_required"]
        rows.append({
            "trace_kind": trace_kind,
            "budget": budget,
            "method": method,
            "step_count": len(frame),
            "reduction_required_count": int(required.sum()),
            "reduction_required_rate": float(required.mean()),
            "automatic_none_count": int(frame["automatic_none"].sum()),
            "automatic_none_rate": float(frame["automatic_none"].mean()),
            "fallback_count": int(frame["fallback_used"].sum()),
            "fallback_rate_on_reductions": _conditional_rate(frame["fallback_used"], required),
            "infeasible_candidate_count": int(frame["infeasible_candidate_count"].sum()),
            "infeasible_step_count": int(frame["infeasible_step"].sum()),
            "infeasible_step_rate_on_reductions": _conditional_rate(
                frame["infeasible_step"], required,
            ),
        })
    return pd.DataFrame(rows)


def _conditional_rate(values: pd.Series, condition: pd.Series) -> float:
    return float(values[condition].mean()) if bool(condition.any()) else 0.0


def macro_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_approx_loss", "final_approx_loss", "max_approx_loss",
        "sum_approx_loss", "mean_state_width", "max_state_width", "total_time_ms",
    ]
    result = summary.groupby(["method", "budget"], as_index=False)[metrics].mean()
    result.insert(2, "aggregation", "macro_trace_mean")
    return result


def micro_trigger_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "false_positive_count", "false_negative_count",
        "reference_positive_count", "reference_negative_count",
    ]
    result = summary.groupby(["method", "budget"], as_index=False)[columns].sum()
    result["fpr"] = result["false_positive_count"] / result[
        "reference_negative_count"
    ].replace(0, np.nan)
    result["fnr"] = result["false_negative_count"] / result[
        "reference_positive_count"
    ].replace(0, np.nan)
    result.insert(2, "aggregation", "micro_trigger_counts")
    return result


def comparison_to_reference(
    summary: pd.DataFrame,
    policy_method: str,
    reference_method: str,
) -> pd.DataFrame:
    keys = ["trace_kind", "budget"]
    policy = summary[summary["method"] == policy_method].set_index(keys)
    reference = summary[summary["method"] == reference_method].set_index(keys)
    if set(policy.index) != set(reference.index):
        raise ValueError(f"policy and {reference_method} evaluation cells do not align")
    rows = []
    for key in sorted(policy.index):
        trace_kind, budget = key
        for metric in COMPARISON_METRICS:
            policy_value = float(policy.loc[key, metric])
            reference_value = float(reference.loc[key, metric])
            rows.append({
                "trace_kind": trace_kind,
                "budget": budget,
                "policy_method": policy_method,
                "reference_method": reference_method,
                "metric": metric,
                "policy_value": policy_value,
                "reference_value": reference_value,
                "difference": policy_value - reference_value,
                "ratio": policy_value / reference_value if reference_value != 0.0 else float("nan"),
            })
    return pd.DataFrame(rows)


def best_static_metrics(
    summary: pd.DataFrame,
    static_methods: tuple[str, ...],
) -> pd.DataFrame:
    if not static_methods:
        raise ValueError("best-static reporting requires a static reducer")
    rows = []
    static = summary[summary["method"].isin(static_methods)]
    for (trace_kind, budget), frame in static.groupby(["trace_kind", "budget"]):
        ordered = frame.assign(
            _method_order=frame["method"].map({
                method: index for index, method in enumerate(static_methods)
            }),
        ).sort_values("_method_order")
        if set(ordered["method"]) != set(static_methods):
            raise ValueError("static evaluation cells do not align")
        for metric in COMPARISON_METRICS:
            values = ordered[metric].to_numpy(dtype=np.float64)
            finite = np.isfinite(values)
            if "approx_loss" in metric and not finite.all():
                raise ValueError("best-static native loss contains non-finite values")
            best_index = int(np.argmin(np.where(finite, values, np.inf))) if finite.any() else None
            rows.append({
                "trace_kind": trace_kind,
                "budget": int(budget),
                "metric": metric,
                "defined_static_count": int(np.count_nonzero(finite)),
                "best_static_method": (
                    str(ordered.iloc[best_index]["method"]) if best_index is not None else None
                ),
                "best_static_value": (
                    float(values[best_index]) if best_index is not None else float("nan")
                ),
            })
    return pd.DataFrame(rows)


def explicit_policy_comparisons(
    summary: pd.DataFrame,
    comparisons: tuple[PolicyComparison, ...],
) -> pd.DataFrame:
    rows = []
    for requested in comparisons:
        comparison = comparison_to_reference(
            summary, requested.challenger, requested.reference,
        ).rename(columns={
            "policy_method": "challenger",
            "reference_method": "reference",
            "policy_value": "challenger_value",
            "reference_value": "reference_value",
        })
        comparison.insert(2, "comparison", requested.name)
        rows.append(comparison)
    columns = [
        "trace_kind", "budget", "comparison", "challenger", "reference",
        "metric", "challenger_value", "reference_value", "difference", "ratio",
    ]
    if not rows:
        raise ValueError("explicit policy comparisons must not be empty")
    return pd.concat(rows, ignore_index=True)[columns]


def write_policy_plots(
    timeseries: pd.DataFrame,
    summary: pd.DataFrame,
    output: Path,
    *,
    policy_methods: tuple[str, ...],
    comparisons: tuple[PolicyComparison, ...] = (),
) -> None:
    if timeseries.empty or summary.empty:
        raise ValueError("policy plots require non-empty evaluation artifacts")
    plt = _pyplot()
    output.mkdir(parents=True, exist_ok=True)
    _metric_budget_plot(summary, output / "metrics_vs_budget.png", plt)
    _trace_generalization_plot(summary, output / "generalization_by_trace.png", plt)
    if comparisons:
        compared_methods = tuple(dict.fromkeys(
            method
            for comparison in comparisons
            for method in (comparison.challenger, comparison.reference)
        ))
        _policy_comparison_plot(
            summary, output / "policy_comparisons.png", compared_methods, plt,
        )
    for method in policy_methods:
        _candidate_selection_plot(
            timeseries, output / f"candidate_composition_{method}.png", method, plt,
        )
        _loss_over_time_plot(
            timeseries, output / f"loss_over_time_{method}.png", method, plt,
        )


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _metric_budget_plot(summary: pd.DataFrame, path: Path, plt: object) -> None:
    metrics = ("mean_approx_loss", "mean_state_width", "fpr", "fnr")
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    averaged = summary.groupby(["method", "budget"], as_index=False)[list(metrics)].mean()
    for axis, metric in zip(axes.flat, metrics):
        for method, rows in averaged.groupby("method"):
            rows = rows.sort_values("budget")
            axis.plot(rows["budget"], rows[metric], marker="o", label=method)
        axis.set_title(metric.replace("_", " "))
        axis.set_xlabel("binding transform bound")
        if "approx_loss" in metric:
            axis.set_yscale("symlog", linthresh=1e-12)
        axis.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=7)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _trace_generalization_plot(summary: pd.DataFrame, path: Path, plt: object) -> None:
    grouped = summary.groupby(["trace_kind", "method"], as_index=False)["mean_approx_loss"].mean()
    pivot = grouped.pivot(index="trace_kind", columns="method", values="mean_approx_loss")
    axis = pivot.plot(kind="bar", figsize=(11, 5))
    axis.set_ylabel("mean binding-native approximation loss")
    axis.set_xlabel("fixed trace kind")
    axis.set_yscale("symlog", linthresh=1e-12)
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(path, dpi=180)
    plt.close(axis.figure)


def _candidate_selection_plot(
    timeseries: pd.DataFrame, path: Path, method: str, plt: object,
) -> None:
    selected = timeseries[timeseries["method"] == method]
    counts = selected.groupby(["trace_kind", "reducer_used"]).size()
    fractions = counts.groupby(level=0).transform(lambda values: values / values.sum())
    axis = fractions.unstack(fill_value=0.0).plot(kind="bar", stacked=True, figsize=(11, 5))
    axis.set_ylabel("selection fraction")
    axis.set_xlabel("fixed trace kind")
    axis.set_ylim(0.0, 1.0)
    axis.figure.tight_layout()
    axis.figure.savefig(path, dpi=180)
    plt.close(axis.figure)


def _loss_over_time_plot(
    timeseries: pd.DataFrame, path: Path, method: str, plt: object,
) -> None:
    selected = timeseries[timeseries["method"] == method]
    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for (trace_kind, budget), rows in selected.groupby(["trace_kind", "budget"]):
        axis.plot(rows["step"], rows["approx_loss"], alpha=0.75, label=f"{trace_kind}, b={budget}")
    axis.set_xlabel("event")
    axis.set_ylabel("binding-native approximation loss")
    axis.set_yscale("symlog", linthresh=1e-12)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _policy_comparison_plot(
    summary: pd.DataFrame,
    path: Path,
    methods: tuple[str, ...],
    plt: object,
) -> None:
    selected = summary[summary["method"].isin(methods)]
    grouped = selected.groupby(["method", "budget"], as_index=False)["mean_approx_loss"].mean()
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for method in methods:
        rows = grouped[grouped["method"] == method].sort_values("budget")
        axis.plot(rows["budget"], rows["mean_approx_loss"], marker="o", label=method)
    axis.set_xlabel("binding transform bound")
    axis.set_ylabel("mean binding-native approximation loss")
    axis.set_yscale("symlog", linthresh=1e-12)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    figure.savefig(path, dpi=180)
    plt.close(figure)
