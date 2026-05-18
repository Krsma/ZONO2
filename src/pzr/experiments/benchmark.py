"""Reusable benchmark harness for reduction-method comparisons."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Generic, Literal, Sequence, TypeVar

import numpy as np
import pandas as pd
from scipy import stats

from pzr.control.costs import CostWeights, WeightedZonotopeCost
from pzr.control.policies import (
    MPCPolicy,
    RolloutMPCPolicy,
    SequenceMPCPolicy,
    StaticReductionPolicy,
)
from pzr.learning.features import (
    DECISION_FEATURE_NAMES,
    DECISION_FEATURE_SCHEMA_VERSION,
    decision_feature_values,
)
from pzr.monitoring.base import (
    MonitorAdapter,
    MonitorState,
    Verdict,
    evaluate_triggers,
    trigger_straddles_threshold,
)
from pzr.reduction.base import Reducer
from pzr.reduction.paper_reducers import (
    AdaptiveReducer,
    CombastelReducer,
    GirardReducer,
    MethAReducer,
    PcaReducer,
    ScottReducer,
)
from pzr.reduction.reducers import (
    BoxReducer,
    ProtectedReducer,
    ScoredKeepReducer,
    TargetBudgetReducer,
)

InputT = TypeVar("InputT")
MethodKind = Literal[
    "reference",
    "static",
    "mpc",
    "mpc_sequence",
    "mpc_rollout",
    "learned",
]
PredictorMode = Literal["online", "oracle", "both"]

KEY_METRICS = (
    "inconclusive_rate",
    "extra_inconclusive_count",
    "verdict_disagreement_count",
    "unsafe_disagreement_count",
    "mean_trigger_width",
    "max_trigger_width",
    "mean_width_inflation",
    "max_width_inflation",
    "mean_generators",
    "max_generators",
    "reduction_count",
    "no_op_count",
    "total_seconds",
)

_TIMESERIES_COLUMNS = (
    "scenario",
    "method",
    "method_kind",
    "seed",
    "length",
    "budget",
    "horizon",
    "predictor_mode",
    "step",
    "interval_hull_mse",
    "trigger_interval_hull_mse",
    "width_inflation",
    "max_width_inflation",
    "generator_count",
    "safe_count",
    "violation_count",
    "inconclusive_count",
    "reference_safe_count",
    "reference_violation_count",
    "reference_inconclusive_count",
    "verdict_disagreement_count",
    "unsafe_disagreement_count",
    "false_violation_count",
    "false_violation_rate",
    "false_alarm_count",
    "false_alarm_rate",
    "reduction_applied",
    "no_op_selected",
    "reducer_name",
    "reduction_seconds",
    "unsound_certificate",
    "reduction_failed",
    "predicted_cost",
    "predicted_sequence",
    "evaluated_sequence_count",
    "pruned_sequence_count",
)

_BOUNDS_TIMESERIES_COLUMNS = (
    "scenario",
    "method",
    "method_kind",
    "seed",
    "length",
    "budget",
    "horizon",
    "predictor_mode",
    "step",
    "state_index",
    "state_name",
    "lower",
    "upper",
    "center",
    "width",
    "reference_lower",
    "reference_upper",
    "reference_center",
    "reference_width",
)

_DECISION_FEATURE_METADATA_COLUMNS = (
    "feature_schema_version",
    "scenario",
    "method",
    "method_kind",
    "seed",
    "length",
    "budget",
    "horizon",
    "predictor_mode",
    "step",
    "chosen_reducer_label",
    "predicted_cost",
    "predicted_sequence",
    "evaluated_sequence_count",
    "pruned_sequence_count",
    "candidate_reducer_names",
    "no_op_selected",
)

_SELECTION_SUMMARY_COLUMNS = (
    "scenario",
    "predictor_mode",
    "method",
    "selected_reducer",
    "selection_count",
    "selection_fraction",
    "decision_count",
    "reduction_count",
    "reduction_failure_count",
    "evaluated_sequence_count",
    "pruned_sequence_count",
)

_PREDICTED_SEQUENCE_SUMMARY_COLUMNS = (
    "scenario",
    "predictor_mode",
    "method",
    "decision_count",
    "sequence_with_box_count",
    "sequence_with_box_fraction",
    "first_action_box_count",
    "first_action_box_fraction",
    "future_box_count",
    "future_box_fraction",
    "mean_sequence_length",
    "evaluated_sequence_count",
    "pruned_sequence_count",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration shared across methods for one benchmark run."""

    length: int = 200
    budget: int = 8
    horizon: int = 4
    seeds: tuple[int, ...] = tuple(range(30))
    predictor_mode: PredictorMode = "online"
    include_reference: bool = True
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 0


@dataclass(frozen=True)
class BenchmarkScenario(Generic[InputT]):
    """A benchmark scenario that can be evaluated by the generic harness."""

    name: str
    make_monitor: Callable[[], MonitorAdapter[InputT]]
    generate_trace: Callable[[int, int], tuple[InputT, ...]]
    predict_inputs: Callable[[Sequence[InputT], int], tuple[InputT, ...]]
    state_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class MethodSpec:
    """A reduction method included in a benchmark suite."""

    name: str
    kind: MethodKind
    reducer_factory: Callable[[], Reducer] | None = None
    mpc_reducer_factories: tuple[Callable[[], Reducer], ...] = ()
    mpc_base_reducer_factory: Callable[[], Reducer] | None = None
    mpc_fallback_reducer_factory: Callable[[], Reducer] | None = None
    learned_policy_path: str | None = None

    @staticmethod
    def reference() -> "MethodSpec":
        return MethodSpec("reference", "reference")

    @staticmethod
    def static(name: str, reducer_factory: Callable[[], Reducer]) -> "MethodSpec":
        return MethodSpec(name, "static", reducer_factory=reducer_factory)

    @staticmethod
    def mpc(
        name: str,
        reducer_factories: tuple[Callable[[], Reducer], ...],
    ) -> "MethodSpec":
        return MethodSpec(name, "mpc", mpc_reducer_factories=reducer_factories)

    @staticmethod
    def sequence_mpc(
        name: str,
        reducer_factories: tuple[Callable[[], Reducer], ...],
    ) -> "MethodSpec":
        return MethodSpec(name, "mpc_sequence", mpc_reducer_factories=reducer_factories)

    @staticmethod
    def rollout_mpc(
        name: str,
        reducer_factories: tuple[Callable[[], Reducer], ...],
        base_reducer_factory: Callable[[], Reducer],
        fallback_reducer_factory: Callable[[], Reducer] | None = None,
    ) -> "MethodSpec":
        return MethodSpec(
            name,
            "mpc_rollout",
            mpc_reducer_factories=reducer_factories,
            mpc_base_reducer_factory=base_reducer_factory,
            mpc_fallback_reducer_factory=fallback_reducer_factory,
        )

    @staticmethod
    def learned(
        policy_path: str | Path,
        reducer_factories: tuple[Callable[[], Reducer], ...],
    ) -> "MethodSpec":
        return MethodSpec(
            "learned_distilled",
            "learned",
            mpc_reducer_factories=reducer_factories,
            mpc_fallback_reducer_factory=_protected(BoxReducer),
            learned_policy_path=str(policy_path),
        )


@dataclass(frozen=True)
class RunRecord:
    """One benchmark result row for one scenario/method/seed."""

    scenario: str
    method: str
    method_kind: MethodKind
    seed: int
    length: int
    budget: int
    horizon: int
    predictor_mode: str
    metrics: dict[str, int | float | str | bool]

    def to_row(self) -> dict[str, int | float | str | bool]:
        return {
            "scenario": self.scenario,
            "method": self.method,
            "method_kind": self.method_kind,
            "seed": self.seed,
            "length": self.length,
            "budget": self.budget,
            "horizon": self.horizon,
            "predictor_mode": self.predictor_mode,
            **self.metrics,
        }


@dataclass(frozen=True)
class ReferenceTrace:
    """Reference verdicts and widths from the unreduced monitor."""

    statuses: tuple[tuple[str, ...], ...]
    widths: np.ndarray
    inconclusive: np.ndarray
    center: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class BenchmarkReport:
    """Raw and aggregated benchmark results."""

    config: BenchmarkConfig
    raw_runs: pd.DataFrame
    summary: pd.DataFrame
    comparisons: pd.DataFrame
    predictor_comparisons: pd.DataFrame
    timeseries: pd.DataFrame
    bounds_timeseries: pd.DataFrame
    decision_features: pd.DataFrame
    selection_summary: pd.DataFrame
    predicted_sequence_summary: pd.DataFrame

    def write_artifacts(self, out_dir: str | Path) -> None:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.raw_runs.to_csv(path / "raw_runs.csv", index=False)
        self.summary.to_csv(path / "summary.csv", index=False)
        self.comparisons.to_csv(path / "comparisons.csv", index=False)
        self.predictor_comparisons.to_csv(path / "predictor_comparisons.csv", index=False)
        self.timeseries.to_csv(path / "timeseries.csv", index=False)
        self.bounds_timeseries.to_csv(path / "bounds_timeseries.csv", index=False)
        self.decision_features.to_csv(path / "decision_features.csv", index=False)
        self.selection_summary.to_csv(path / "selection_summary.csv", index=False)
        self.predicted_sequence_summary.to_csv(
            path / "predicted_sequence_summary.csv",
            index=False,
        )
        (path / "config.json").write_text(
            json.dumps(_json_safe(asdict(self.config)), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (path / "report.json").write_text(
            json.dumps(
                _json_safe(
                    {
                        "config": asdict(self.config),
                        "raw_runs": self.raw_runs.to_dict("records"),
                        "summary": self.summary.to_dict("records"),
                        "comparisons": self.comparisons.to_dict("records"),
                        "predictor_comparisons": self.predictor_comparisons.to_dict(
                            "records"
                        ),
                        "decision_features": self.decision_features.to_dict("records"),
                        "selection_summary": self.selection_summary.to_dict("records"),
                        "predicted_sequence_summary": (
                            self.predicted_sequence_summary.to_dict("records")
                        ),
                    }
                ),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


def default_methods() -> tuple[MethodSpec, ...]:
    """Extended method suite, including predictive and keep-based reducers."""

    mpc_candidates = (
        _protected(BoxReducer),
        _protected(GirardReducer),
        _protected(ScoredKeepReducer.by_norm),
        _protected(ScoredKeepReducer.calibration_aware),
    )
    rollout_candidates = (
        _protected(GirardReducer),
        _protected(ScoredKeepReducer.by_norm),
        _protected(ScoredKeepReducer.calibration_aware),
    )
    wide_rollout_candidates = (
        _protected(GirardReducer),
        _protected(CombastelReducer),
        _protected(MethAReducer),
        _protected(ScottReducer),
        _protected(PcaReducer),
        _protected(AdaptiveReducer),
        _protected(ScoredKeepReducer.by_norm),
        _protected(ScoredKeepReducer.calibration_aware),
    )
    return (
        MethodSpec.static("box", _protected(BoxReducer)),
        MethodSpec.static("girard", _protected(GirardReducer)),
        MethodSpec.static(
            "girard7",
            _target_budget(_protected(GirardReducer), 7, "girard7"),
        ),
        MethodSpec.static("combastel", _protected(CombastelReducer)),
        MethodSpec.static("methA", _protected(MethAReducer)),
        MethodSpec.static("scott", _protected(ScottReducer)),
        MethodSpec.static("pca", _protected(PcaReducer)),
        MethodSpec.static("adaptive", _protected(AdaptiveReducer)),
        MethodSpec.static("keep_norm", _protected(ScoredKeepReducer.by_norm)),
        MethodSpec.static(
            "keep_calibration_aware",
            _protected(ScoredKeepReducer.calibration_aware),
        ),
        MethodSpec.mpc("mpc", mpc_candidates),
        MethodSpec.sequence_mpc("mpc_sequence", mpc_candidates),
        MethodSpec.rollout_mpc(
            "mpc_rollout_girard",
            rollout_candidates,
            _protected(GirardReducer),
            _protected(BoxReducer),
        ),
        MethodSpec.rollout_mpc(
            "mpc_rollout_wide",
            wide_rollout_candidates,
            _protected(GirardReducer),
            _protected(BoxReducer),
        ),
    )


def wide_rollout_reducer_factories() -> tuple[Callable[[], Reducer], ...]:
    """Protected precision candidates used by wide rollout and learned policies."""

    return (
        _protected(GirardReducer),
        _protected(CombastelReducer),
        _protected(MethAReducer),
        _protected(ScottReducer),
        _protected(PcaReducer),
        _protected(AdaptiveReducer),
        _protected(ScoredKeepReducer.by_norm),
        _protected(ScoredKeepReducer.calibration_aware),
    )


def learned_distilled_method(policy_path: str | Path) -> MethodSpec:
    """Create the benchmark method spec for a distilled learned policy."""

    return MethodSpec.learned(policy_path, wide_rollout_reducer_factories())


def paper_baseline_methods() -> tuple[MethodSpec, ...]:
    """Static reducers used in the paper replica plots."""

    return (
        MethodSpec.static("box", BoxReducer),
        MethodSpec.static("girard", GirardReducer),
        MethodSpec.static("combastel", CombastelReducer),
        MethodSpec.static("methA", MethAReducer),
        MethodSpec.static("scott", ScottReducer),
        MethodSpec.static("pca", PcaReducer),
        MethodSpec.static("adaptive", AdaptiveReducer),
    )


def _protected(factory: Callable[[], Reducer]) -> Callable[[], Reducer]:
    def make() -> Reducer:
        return ProtectedReducer(factory())

    return make


def _target_budget(
    factory: Callable[[], Reducer],
    target_budget: int,
    name: str,
) -> Callable[[], Reducer]:
    def make() -> Reducer:
        return TargetBudgetReducer(factory(), target_budget, name=name)

    return make


def run_benchmark(
    scenario: BenchmarkScenario[InputT],
    config: BenchmarkConfig,
    methods: Sequence[MethodSpec] | None = None,
) -> BenchmarkReport:
    """Run all methods on all seeds and return raw and aggregate results."""

    selected_methods = tuple(methods or default_methods())
    if config.include_reference:
        selected_methods = (MethodSpec.reference(), *selected_methods)

    rows: list[dict[str, int | float | str | bool]] = []
    timeseries_rows: list[dict[str, int | float | str | bool]] = []
    bounds_rows: list[dict[str, int | float | str | bool]] = []
    decision_feature_rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        trace = scenario.generate_trace(config.length, seed)
        reference = (
            _run_reference_trace(scenario, config, seed, trace)
            if config.include_reference
            else None
        )
        for method in selected_methods:
            if method.kind == "reference":
                if reference is None:
                    continue
                record, method_timeseries, method_bounds = _run_reference_method(
                    scenario,
                    config,
                    method,
                    seed,
                    trace,
                    reference,
                )
            else:
                (
                    record,
                    method_timeseries,
                    method_bounds,
                    method_decision_features,
                ) = _run_budgeted_method(
                    scenario,
                    config,
                    method,
                    seed,
                    trace,
                    reference if config.include_reference else None,
                )
                decision_feature_rows.extend(method_decision_features)
            rows.append(record.to_row())
            timeseries_rows.extend(method_timeseries)
            bounds_rows.extend(method_bounds)

    raw = pd.DataFrame(rows)
    timeseries = pd.DataFrame(timeseries_rows, columns=_TIMESERIES_COLUMNS)
    bounds_timeseries = pd.DataFrame(bounds_rows, columns=_BOUNDS_TIMESERIES_COLUMNS)
    decision_features = pd.DataFrame(
        decision_feature_rows,
        columns=tuple(
            dict.fromkeys((*_DECISION_FEATURE_METADATA_COLUMNS, *DECISION_FEATURE_NAMES))
        ),
    )
    summary = summarize_runs(raw, config)
    comparisons = compare_against_mpc(raw)
    predictor_comparisons = compare_predictor_modes(raw)
    selection_summary = summarize_selection(decision_features, raw)
    predicted_sequence_summary = summarize_predicted_sequences(decision_features, raw)
    return BenchmarkReport(
        config,
        raw,
        summary,
        comparisons,
        predictor_comparisons,
        timeseries,
        bounds_timeseries,
        decision_features,
        selection_summary,
        predicted_sequence_summary,
    )


def combine_reports(config: BenchmarkConfig, reports: Sequence[BenchmarkReport]) -> BenchmarkReport:
    """Combine separately run predictor-mode reports into one report."""

    raw = pd.concat([report.raw_runs for report in reports], ignore_index=True)
    timeseries = pd.concat([report.timeseries for report in reports], ignore_index=True)
    bounds_timeseries = pd.concat(
        [report.bounds_timeseries for report in reports],
        ignore_index=True,
    )
    decision_features = pd.concat(
        [report.decision_features for report in reports],
        ignore_index=True,
    )
    summary = summarize_runs(raw, config)
    comparisons = compare_against_mpc(raw)
    predictor_comparisons = compare_predictor_modes(raw)
    selection_summary = summarize_selection(decision_features, raw)
    predicted_sequence_summary = summarize_predicted_sequences(decision_features, raw)
    return BenchmarkReport(
        config,
        raw,
        summary,
        comparisons,
        predictor_comparisons,
        timeseries,
        bounds_timeseries,
        decision_features,
        selection_summary,
        predicted_sequence_summary,
    )


def summarize_runs(raw: pd.DataFrame, config: BenchmarkConfig) -> pd.DataFrame:
    """Create one summary row per scenario/method/metric."""

    metric_columns = [
        col
        for col in raw.columns
        if col
        not in {
            "scenario",
            "method",
            "method_kind",
            "seed",
            "length",
            "budget",
            "horizon",
            "predictor_mode",
        }
        and pd.api.types.is_numeric_dtype(raw[col])
    ]
    rows: list[dict[str, Any]] = []
    for (scenario, predictor_mode, method), group in raw.groupby(
        ["scenario", "predictor_mode", "method"],
        sort=True,
    ):
        for metric in metric_columns:
            values = group[metric].dropna().astype(float).to_numpy()
            if values.size == 0:
                continue
            ci_low, ci_high = _bootstrap_mean_ci(
                values,
                samples=config.bootstrap_samples,
                seed=config.bootstrap_seed,
            )
            rows.append(
                {
                    "scenario": scenario,
                    "predictor_mode": predictor_mode,
                    "method": method,
                    "metric": metric,
                    "n": int(values.size),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def compare_against_mpc(raw: pd.DataFrame) -> pd.DataFrame:
    """Create paired method-vs-MPC comparisons for numeric metrics."""

    metric_columns = [
        col
        for col in raw.columns
        if col
        not in {
            "scenario",
            "method",
            "method_kind",
            "seed",
            "length",
            "budget",
            "horizon",
            "predictor_mode",
        }
        and pd.api.types.is_numeric_dtype(raw[col])
    ]
    rows: list[dict[str, Any]] = []
    for (scenario, predictor_mode), scenario_df in raw.groupby(
        ["scenario", "predictor_mode"],
        sort=True,
    ):
        method_names = set(scenario_df["method"])
        baseline = next(
            (
                candidate
                for candidate in (
                    "mpc_rollout_wide",
                    "mpc_rollout_girard",
                    "mpc_sequence",
                    "mpc",
                )
                if candidate in method_names
            ),
            "",
        )
        if baseline not in set(scenario_df["method"]):
            continue
        methods = sorted(set(scenario_df["method"]) - {baseline, "reference"})
        for method in methods:
            for metric in metric_columns:
                pair = scenario_df[scenario_df["method"].isin({method, baseline})]
                pivot = pair.pivot(index="seed", columns="method", values=metric).dropna()
                if method not in pivot or baseline not in pivot or pivot.empty:
                    continue
                delta = (
                    pivot[method].astype(float).to_numpy()
                    - pivot[baseline].astype(float).to_numpy()
                )
                p_value = _wilcoxon_p_value(delta)
                rows.append(
                    {
                        "scenario": scenario,
                        "predictor_mode": predictor_mode,
                        "method": method,
                        "baseline": baseline,
                        "metric": metric,
                        "n": int(delta.size),
                        "mean_delta_method_minus_mpc": float(np.mean(delta)),
                        "median_delta_method_minus_mpc": float(np.median(delta)),
                        "effect_size_dz": _paired_effect_size(delta),
                        "wilcoxon_p_value": p_value,
                    }
                )
    return pd.DataFrame(rows)


def compare_predictor_modes(raw: pd.DataFrame) -> pd.DataFrame:
    """Create paired online-vs-oracle comparisons for matching methods."""

    columns = [
        "scenario",
        "method",
        "metric",
        "n",
        "mean_delta_online_minus_oracle",
        "median_delta_online_minus_oracle",
        "effect_size_dz",
        "wilcoxon_p_value",
    ]
    if not {"online", "oracle"} <= set(raw["predictor_mode"]):
        return pd.DataFrame(columns=columns)
    metric_columns = [
        col
        for col in raw.columns
        if col
        not in {
            "scenario",
            "method",
            "method_kind",
            "seed",
            "length",
            "budget",
            "horizon",
            "predictor_mode",
        }
        and pd.api.types.is_numeric_dtype(raw[col])
    ]
    rows: list[dict[str, Any]] = []
    for (scenario, method), group in raw.groupby(["scenario", "method"], sort=True):
        for metric in metric_columns:
            pivot = group.pivot(index="seed", columns="predictor_mode", values=metric).dropna()
            if "online" not in pivot or "oracle" not in pivot or pivot.empty:
                continue
            delta = pivot["online"].astype(float).to_numpy() - pivot["oracle"].astype(float).to_numpy()
            rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "metric": metric,
                    "n": int(delta.size),
                    "mean_delta_online_minus_oracle": float(np.mean(delta)),
                    "median_delta_online_minus_oracle": float(np.median(delta)),
                    "effect_size_dz": _paired_effect_size(delta),
                    "wilcoxon_p_value": _wilcoxon_p_value(delta),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def summarize_selection(decision_features: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Summarize first reducer choices at actual over-budget decisions."""

    if decision_features.empty:
        return pd.DataFrame(columns=_SELECTION_SUMMARY_COLUMNS)

    data = decision_features[
        decision_features["chosen_reducer_label"].notna()
        & (decision_features["chosen_reducer_label"].astype(str) != "")
    ].copy()
    if data.empty:
        return pd.DataFrame(columns=_SELECTION_SUMMARY_COLUMNS)

    totals = _raw_reduction_totals(raw)
    decision_counts = (
        data.groupby(["scenario", "predictor_mode", "method"], sort=True)
        .size()
        .rename("decision_count")
    )
    rows: list[dict[str, Any]] = []
    for key, selected in data.groupby(
        ["scenario", "predictor_mode", "method", "chosen_reducer_label"],
        sort=True,
    ):
        scenario, predictor_mode, method, selected_reducer = key
        group_key = (scenario, predictor_mode, method)
        decision_count = int(decision_counts[group_key])
        raw_totals = totals.get(group_key, {})
        rows.append(
            {
                "scenario": scenario,
                "predictor_mode": predictor_mode,
                "method": method,
                "selected_reducer": selected_reducer,
                "selection_count": int(selected.shape[0]),
                "selection_fraction": (
                    float(selected.shape[0] / decision_count) if decision_count else 0.0
                ),
                "decision_count": decision_count,
                "reduction_count": int(raw_totals.get("reduction_count", 0)),
                "reduction_failure_count": int(
                    raw_totals.get("reduction_failure_count", 0)
                ),
                "evaluated_sequence_count": int(
                    raw_totals.get("evaluated_sequence_count", 0)
                ),
                "pruned_sequence_count": int(raw_totals.get("pruned_sequence_count", 0)),
            }
        )
    return pd.DataFrame(rows, columns=_SELECTION_SUMMARY_COLUMNS)


def summarize_predicted_sequences(
    decision_features: pd.DataFrame,
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize predicted MPC reducer sequences and fallback-box usage."""

    if decision_features.empty:
        return pd.DataFrame(columns=_PREDICTED_SEQUENCE_SUMMARY_COLUMNS)

    data = decision_features[
        decision_features["method_kind"].isin({"mpc", "mpc_sequence", "mpc_rollout"})
    ].copy()
    if data.empty:
        return pd.DataFrame(columns=_PREDICTED_SEQUENCE_SUMMARY_COLUMNS)

    data["_sequence"] = data["predicted_sequence"].map(_parse_sequence)
    data["_sequence_length"] = data["_sequence"].map(len)
    data["_sequence_has_box"] = data["_sequence"].map(lambda seq: "box" in seq)
    data["_first_action_box"] = data["_sequence"].map(
        lambda seq: bool(seq) and seq[0] == "box"
    )
    data["_future_box"] = data["_sequence"].map(lambda seq: "box" in seq[1:])

    totals = _raw_reduction_totals(raw)
    rows: list[dict[str, Any]] = []
    for key, group in data.groupby(["scenario", "predictor_mode", "method"], sort=True):
        scenario, predictor_mode, method = key
        decision_count = int(group.shape[0])
        raw_totals = totals.get(key, {})
        sequence_with_box = int(group["_sequence_has_box"].sum())
        first_action_box = int(group["_first_action_box"].sum())
        future_box = int(group["_future_box"].sum())
        rows.append(
            {
                "scenario": scenario,
                "predictor_mode": predictor_mode,
                "method": method,
                "decision_count": decision_count,
                "sequence_with_box_count": sequence_with_box,
                "sequence_with_box_fraction": (
                    float(sequence_with_box / decision_count) if decision_count else 0.0
                ),
                "first_action_box_count": first_action_box,
                "first_action_box_fraction": (
                    float(first_action_box / decision_count) if decision_count else 0.0
                ),
                "future_box_count": future_box,
                "future_box_fraction": (
                    float(future_box / decision_count) if decision_count else 0.0
                ),
                "mean_sequence_length": float(group["_sequence_length"].mean())
                if decision_count
                else 0.0,
                "evaluated_sequence_count": int(
                    raw_totals.get("evaluated_sequence_count", 0)
                ),
                "pruned_sequence_count": int(raw_totals.get("pruned_sequence_count", 0)),
            }
        )
    return pd.DataFrame(rows, columns=_PREDICTED_SEQUENCE_SUMMARY_COLUMNS)


def load_benchmark_artifacts(path: str | Path) -> dict[str, pd.DataFrame]:
    """Load benchmark CSV artifacts from one report or suite aggregate directory."""

    root = Path(path)
    candidates = [root]
    if (root / "aggregate").is_dir():
        candidates.insert(0, root / "aggregate")
    artifact_dir = next((item for item in candidates if (item / "raw_runs.csv").exists()), root)
    frames: dict[str, pd.DataFrame] = {}
    for name in (
        "raw_runs",
        "summary",
        "comparisons",
        "predictor_comparisons",
        "timeseries",
        "bounds_timeseries",
        "decision_features",
        "selection_summary",
        "predicted_sequence_summary",
    ):
        csv_path = artifact_dir / f"{name}.csv"
        if csv_path.exists():
            frames[name] = pd.read_csv(csv_path)
    return frames


def format_terminal_summary(report: BenchmarkReport) -> str:
    """Compact human-readable summary for CLI output."""

    summary = report.summary[report.summary["metric"].isin(KEY_METRICS)]
    if summary.empty:
        return "No benchmark metrics were produced."
    pivot = summary.pivot_table(
        index=["scenario", "predictor_mode", "method"],
        columns="metric",
        values="mean",
        aggfunc="first",
    ).reset_index()
    available = [
        "scenario",
        "predictor_mode",
        "method",
        *[metric for metric in KEY_METRICS if metric in pivot],
    ]
    return pivot[available].to_string(index=False, float_format=lambda value: f"{value:.6g}")


def _run_reference_trace(
    scenario: BenchmarkScenario[InputT],
    config: BenchmarkConfig,
    seed: int,
    trace: Sequence[InputT],
) -> ReferenceTrace:
    record, statuses, widths, inconclusive, center, lower, upper, _, _ = _run_without_reduction(
        scenario,
        config,
        MethodSpec.reference(),
        seed,
        trace,
        collect_reference=True,
        reference=None,
    )
    _ = record
    return ReferenceTrace(
        tuple(statuses),
        np.asarray(widths, dtype=float),
        np.asarray(inconclusive),
        np.asarray(center, dtype=float),
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
    )


def _run_reference_method(
    scenario: BenchmarkScenario[InputT],
    config: BenchmarkConfig,
    method: MethodSpec,
    seed: int,
    trace: Sequence[InputT],
    reference: ReferenceTrace,
) -> tuple[RunRecord, list[dict[str, Any]], list[dict[str, Any]]]:
    record, _, _, _, _, _, _, timeseries, bounds = _run_without_reduction(
        scenario,
        config,
        method,
        seed,
        trace,
        collect_reference=False,
        reference=reference,
    )
    return record, timeseries, bounds


def _run_without_reduction(
    scenario: BenchmarkScenario[InputT],
    config: BenchmarkConfig,
    method: MethodSpec,
    seed: int,
    trace: Sequence[InputT],
    *,
    collect_reference: bool,
    reference: ReferenceTrace | None,
) -> tuple[
    RunRecord,
    list[tuple[str, ...]],
    list[list[float]],
    list[list[bool]],
    list[list[float]],
    list[list[float]],
    list[list[float]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    monitor = scenario.make_monitor()
    state = monitor.initial_state()
    accumulator = _MetricAccumulator(monitor.triggers)
    statuses: list[tuple[str, ...]] = []
    widths: list[list[float]] = []
    inconclusive: list[list[bool]] = []
    centers: list[list[float]] = []
    lower_bounds: list[list[float]] = []
    upper_bounds: list[list[float]] = []
    timeseries: list[dict[str, Any]] = []
    bounds_timeseries: list[dict[str, Any]] = []
    start = perf_counter()

    for index, measurement in enumerate(trace):
        tick_start = perf_counter()
        result = monitor.step(state, measurement)
        state = result.state
        verdicts = evaluate_triggers(state.zonotope, monitor.triggers)
        tick_seconds = perf_counter() - tick_start
        reference_index = index if reference is not None else None
        accumulator.add_step(
            state,
            verdicts,
            tick_seconds=tick_seconds,
            reference=reference,
            reference_index=reference_index,
        )
        step_lower, step_upper = state.zonotope.interval_bounds()
        if collect_reference:
            statuses.append(tuple(verdict.status for verdict in verdicts))
            widths.append([verdict.upper - verdict.lower for verdict in verdicts])
            inconclusive.append([verdict.status == "inconclusive" for verdict in verdicts])
            centers.append([float(value) for value in state.zonotope.center])
            lower_bounds.append([float(value) for value in step_lower])
            upper_bounds.append([float(value) for value in step_upper])
        timeseries.append(
            _timeseries_row(
                scenario,
                config,
                method,
                seed,
                index,
                state,
                verdicts,
                reference,
                reduction_applied=False,
            )
        )
        bounds_timeseries.extend(
            _bounds_rows(
                scenario,
                config,
                method,
                seed,
                index,
                state,
                reference,
            )
        )

    metrics = accumulator.finish(
        total_seconds=perf_counter() - start,
        final_state=state,
        config=config,
        method_kind="reference",
    )
    return (
        RunRecord(
            scenario=scenario.name,
            method=method.name,
            method_kind=method.kind,
            seed=seed,
            length=config.length,
            budget=config.budget,
            horizon=config.horizon,
            predictor_mode=config.predictor_mode,
            metrics=metrics,
        ),
        statuses,
        widths,
        inconclusive,
        centers,
        lower_bounds,
        upper_bounds,
        timeseries,
        bounds_timeseries,
    )


def _run_budgeted_method(
    scenario: BenchmarkScenario[InputT],
    config: BenchmarkConfig,
    method: MethodSpec,
    seed: int,
    trace: Sequence[InputT],
    reference: ReferenceTrace | None,
) -> tuple[RunRecord, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    monitor = scenario.make_monitor()
    state = monitor.initial_state()
    accumulator = _MetricAccumulator(monitor.triggers)
    policy = _make_policy(method, monitor, config)
    history: list[InputT] = []
    timeseries: list[dict[str, Any]] = []
    bounds_timeseries: list[dict[str, Any]] = []
    decision_features: list[dict[str, Any]] = []
    candidate_names = _method_candidate_names(method)
    start = perf_counter()

    for index, measurement in enumerate(trace):
        history.append(measurement)
        tick_start = perf_counter()
        result = monitor.step(state, measurement)
        state = result.state
        reduction_applied = False
        no_op_selected = False
        reducer_name = ""
        reduction_seconds = 0.0
        unsound_certificate = False
        reduction_failed = False
        predicted_cost = 0.0
        predicted_sequence = ()
        evaluated_sequences = 0
        pruned_sequences = 0

        if state.zonotope.generator_count > config.budget:
            decision_feature_values_before = decision_feature_values(
                monitor,
                state,
                budget=config.budget,
                horizon=config.horizon,
            )
            reduction_start = perf_counter()
            try:
                if method.kind == "static":
                    decision = policy.reduce_state(monitor, state)
                elif method.kind in {"mpc", "mpc_sequence", "mpc_rollout"}:
                    predicted = _predicted_inputs(scenario, config, history, trace, index)
                    decision = policy.reduce_state(monitor, state, predicted)
                elif method.kind == "learned":
                    decision = policy.reduce_state(monitor, state)
                else:
                    raise ValueError(f"unsupported budgeted method kind: {method.kind}")
                if decision.is_no_op and state.zonotope.generator_count > config.budget:
                    raise ValueError("no-op decision cannot satisfy an over-budget state")
            except Exception:
                accumulator.reduction_failure_count += 1
                reduction_failed = True
            else:
                decision_seconds = perf_counter() - reduction_start
                no_op_selected = decision.is_no_op
                reducer_name = decision.reducer_name
                if no_op_selected:
                    accumulator.no_op_count += 1
                else:
                    accumulator.reduction_count += 1
                    reduction_applied = True
                    reduction_seconds = decision_seconds
                    accumulator.reduction_seconds.append(reduction_seconds)
                unsound_certificate = not decision.result.certificate.is_sound
                accumulator.unsound_certificate_count += int(unsound_certificate)
                accumulator.chosen_reducers.update([decision.reducer_name])
                if math.isfinite(decision.predicted_cost):
                    accumulator.predicted_costs.append(decision.predicted_cost)
                    predicted_cost = decision.predicted_cost
                predicted_sequence = decision.predicted_sequence
                accumulator.evaluated_sequences += decision.evaluated_sequences
                accumulator.pruned_sequences += decision.pruned_sequences
                evaluated_sequences = decision.evaluated_sequences
                pruned_sequences = decision.pruned_sequences
                state = decision.state
                decision_features.append(
                    {
                        "feature_schema_version": DECISION_FEATURE_SCHEMA_VERSION,
                        "scenario": scenario.name,
                        "method": method.name,
                        "method_kind": method.kind,
                        "seed": seed,
                        "length": config.length,
                        "budget": config.budget,
                        "horizon": config.horizon,
                        "predictor_mode": config.predictor_mode,
                        "step": index + 1,
                        "chosen_reducer_label": decision.reducer_name,
                        "predicted_cost": predicted_cost,
                        "predicted_sequence": json.dumps(list(predicted_sequence)),
                        "evaluated_sequence_count": evaluated_sequences,
                        "pruned_sequence_count": pruned_sequences,
                        "candidate_reducer_names": json.dumps(list(candidate_names)),
                        "no_op_selected": no_op_selected,
                        **decision_feature_values_before,
                    }
                )

        verdicts = evaluate_triggers(state.zonotope, monitor.triggers)
        reference_index = index if reference is not None else None
        accumulator.add_step(
            state,
            verdicts,
            tick_seconds=perf_counter() - tick_start,
            reference=reference,
            reference_index=reference_index,
        )
        timeseries.append(
            _timeseries_row(
                scenario,
                config,
                method,
                seed,
                index,
                state,
                verdicts,
                reference,
                reduction_applied=reduction_applied,
                no_op_selected=no_op_selected,
                reducer_name=reducer_name,
                reduction_seconds=reduction_seconds,
                unsound_certificate=unsound_certificate,
                reduction_failed=reduction_failed,
                predicted_cost=predicted_cost,
                predicted_sequence=predicted_sequence,
                evaluated_sequences=evaluated_sequences,
                pruned_sequences=pruned_sequences,
            )
        )
        bounds_timeseries.extend(
            _bounds_rows(
                scenario,
                config,
                method,
                seed,
                index,
                state,
                reference,
            )
        )

    metrics = accumulator.finish(
        total_seconds=perf_counter() - start,
        final_state=state,
        config=config,
        method_kind=method.kind,
    )
    return (
        RunRecord(
            scenario=scenario.name,
            method=method.name,
            method_kind=method.kind,
            seed=seed,
            length=config.length,
            budget=config.budget,
            horizon=config.horizon,
            predictor_mode=config.predictor_mode,
            metrics=metrics,
        ),
        timeseries,
        bounds_timeseries,
        decision_features,
    )


def _timeseries_row(
    scenario: BenchmarkScenario[InputT],
    config: BenchmarkConfig,
    method: MethodSpec,
    seed: int,
    index: int,
    state: MonitorState,
    verdicts: Sequence[Verdict],
    reference: ReferenceTrace | None,
    *,
    reduction_applied: bool,
    no_op_selected: bool = False,
    reducer_name: str = "",
    reduction_seconds: float = 0.0,
    unsound_certificate: bool = False,
    reduction_failed: bool = False,
    predicted_cost: float = 0.0,
    predicted_sequence: Sequence[str] = (),
    evaluated_sequences: int = 0,
    pruned_sequences: int = 0,
) -> dict[str, Any]:
    lower, upper = state.zonotope.interval_bounds()
    ref_lower = lower if reference is None else reference.lower[index]
    ref_upper = upper if reference is None else reference.upper[index]
    interval_mse = _interval_hull_mse(lower, upper, ref_lower, ref_upper)
    trigger_indices = _trigger_indices(verdicts)
    trigger_mse = _interval_hull_mse(
        lower[trigger_indices],
        upper[trigger_indices],
        ref_lower[trigger_indices],
        ref_upper[trigger_indices],
    )
    width_gap = (upper - lower) - (ref_upper - ref_lower)

    statuses = [verdict.status for verdict in verdicts]
    if reference is None:
        ref_statuses = statuses
    else:
        ref_statuses = list(reference.statuses[index])
    trigger_count = max(1, len(statuses))
    false_violation_count = sum(
        status == "violation" and ref_status != "violation"
        for status, ref_status in zip(statuses, ref_statuses)
    )
    false_alarm_count = sum(
        status != "safe" and ref_status == "safe"
        for status, ref_status in zip(statuses, ref_statuses)
    )

    return {
        "scenario": scenario.name,
        "method": method.name,
        "method_kind": method.kind,
        "seed": seed,
        "length": config.length,
        "budget": config.budget,
        "horizon": config.horizon,
        "predictor_mode": config.predictor_mode,
        "step": index + 1,
        "interval_hull_mse": interval_mse,
        "trigger_interval_hull_mse": trigger_mse,
        "width_inflation": float(np.mean(width_gap)) if width_gap.size else 0.0,
        "max_width_inflation": float(np.max(width_gap)) if width_gap.size else 0.0,
        "generator_count": state.zonotope.generator_count,
        "safe_count": statuses.count("safe"),
        "violation_count": statuses.count("violation"),
        "inconclusive_count": statuses.count("inconclusive"),
        "reference_safe_count": ref_statuses.count("safe"),
        "reference_violation_count": ref_statuses.count("violation"),
        "reference_inconclusive_count": ref_statuses.count("inconclusive"),
        "verdict_disagreement_count": sum(
            status != ref_status for status, ref_status in zip(statuses, ref_statuses)
        ),
        "unsafe_disagreement_count": sum(
            status != "inconclusive" and status != ref_status
            for status, ref_status in zip(statuses, ref_statuses)
        ),
        "false_violation_count": false_violation_count,
        "false_violation_rate": false_violation_count / trigger_count,
        "false_alarm_count": false_alarm_count,
        "false_alarm_rate": false_alarm_count / trigger_count,
        "reduction_applied": reduction_applied,
        "no_op_selected": no_op_selected,
        "reducer_name": reducer_name,
        "reduction_seconds": reduction_seconds,
        "unsound_certificate": unsound_certificate,
        "reduction_failed": reduction_failed,
        "predicted_cost": predicted_cost,
        "predicted_sequence": json.dumps(list(predicted_sequence)),
        "evaluated_sequence_count": evaluated_sequences,
        "pruned_sequence_count": pruned_sequences,
    }


def _bounds_rows(
    scenario: BenchmarkScenario[InputT],
    config: BenchmarkConfig,
    method: MethodSpec,
    seed: int,
    index: int,
    state: MonitorState,
    reference: ReferenceTrace | None,
) -> list[dict[str, Any]]:
    lower, upper = state.zonotope.interval_bounds()
    center = state.zonotope.center
    ref_lower = lower if reference is None else reference.lower[index]
    ref_upper = upper if reference is None else reference.upper[index]
    ref_center = center if reference is None else reference.center[index]
    names = scenario.state_names or tuple(f"x{axis}" for axis in range(state.zonotope.dimension))
    rows: list[dict[str, Any]] = []
    for axis in range(state.zonotope.dimension):
        rows.append(
            {
                "scenario": scenario.name,
                "method": method.name,
                "method_kind": method.kind,
                "seed": seed,
                "length": config.length,
                "budget": config.budget,
                "horizon": config.horizon,
                "predictor_mode": config.predictor_mode,
                "step": index + 1,
                "state_index": axis,
                "state_name": names[axis] if axis < len(names) else f"x{axis}",
                "lower": float(lower[axis]),
                "upper": float(upper[axis]),
                "center": float(center[axis]),
                "width": float(upper[axis] - lower[axis]),
                "reference_lower": float(ref_lower[axis]),
                "reference_upper": float(ref_upper[axis]),
                "reference_center": float(ref_center[axis]),
                "reference_width": float(ref_upper[axis] - ref_lower[axis]),
            }
        )
    return rows


def _interval_hull_mse(
    lower: np.ndarray,
    upper: np.ndarray,
    ref_lower: np.ndarray,
    ref_upper: np.ndarray,
) -> float:
    if lower.size == 0:
        return 0.0
    errors = ((lower - ref_lower) ** 2 + (upper - ref_upper) ** 2) / 2.0
    return float(np.mean(errors))


def _trigger_indices(verdicts: Sequence[Verdict]) -> np.ndarray:
    indices = sorted({verdict.trigger.state_index for verdict in verdicts})
    return np.asarray(indices, dtype=int)


def _make_policy(
    method: MethodSpec,
    monitor: MonitorAdapter[InputT],
    config: BenchmarkConfig,
) -> (
    StaticReductionPolicy
    | MPCPolicy[InputT]
    | SequenceMPCPolicy[InputT]
    | RolloutMPCPolicy[InputT]
    | Any
):
    if method.kind == "static":
        if method.reducer_factory is None:
            raise ValueError(f"static method {method.name} has no reducer factory")
        return StaticReductionPolicy(method.reducer_factory(), config.budget)
    if method.kind == "mpc":
        cost = WeightedZonotopeCost(
            CostWeights(trigger_width=1.0, straddling=20.0, generator_count=0.01),
            triggers=monitor.triggers,
        )
        return MPCPolicy(
            reducers=tuple(factory() for factory in method.mpc_reducer_factories),
            budget=config.budget,
            horizon=config.horizon,
            cost=cost,
        )
    if method.kind == "mpc_sequence":
        cost = WeightedZonotopeCost(
            CostWeights(trigger_width=1.0, straddling=20.0, generator_count=0.01),
            triggers=monitor.triggers,
        )
        return SequenceMPCPolicy(
            reducers=tuple(factory() for factory in method.mpc_reducer_factories),
            budget=config.budget,
            horizon=config.horizon,
            cost=cost,
        )
    if method.kind == "mpc_rollout":
        if method.mpc_base_reducer_factory is None:
            raise ValueError(f"rollout method {method.name} has no base reducer factory")
        cost = WeightedZonotopeCost(
            CostWeights(
                trigger_width=1.0,
                straddling=20.0,
                generator_count=0.0,
            ),
            triggers=monitor.triggers,
        )
        return RolloutMPCPolicy(
            reducers=tuple(factory() for factory in method.mpc_reducer_factories),
            base_reducer=method.mpc_base_reducer_factory(),
            fallback_reducer=(
                method.mpc_fallback_reducer_factory()
                if method.mpc_fallback_reducer_factory is not None
                else None
            ),
            budget=config.budget,
            horizon=config.horizon,
            cost=cost,
            terminal_cost_multiplier=0.0,
        )
    if method.kind == "learned":
        if method.learned_policy_path is None:
            raise ValueError(f"learned method {method.name} has no policy path")
        from pzr.learning.policy import LearnedReductionPolicy

        return LearnedReductionPolicy(
            checkpoint_path=method.learned_policy_path,
            reducers=tuple(factory() for factory in method.mpc_reducer_factories),
            fallback_reducer=(
                method.mpc_fallback_reducer_factory()
                if method.mpc_fallback_reducer_factory is not None
                else None
            ),
            budget=config.budget,
            horizon=config.horizon,
        )
    raise ValueError(f"method {method.name} is not a budgeted policy")


def _method_candidate_names(method: MethodSpec) -> tuple[str, ...]:
    if method.kind == "static":
        return (method.reducer_factory().name,) if method.reducer_factory is not None else ()
    names: list[str] = []
    for factory in method.mpc_reducer_factories:
        names.append(factory().name)
    if method.mpc_base_reducer_factory is not None:
        names.append(method.mpc_base_reducer_factory().name)
    if method.mpc_fallback_reducer_factory is not None:
        names.append(method.mpc_fallback_reducer_factory().name)
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return tuple(ordered)


def _predicted_inputs(
    scenario: BenchmarkScenario[InputT],
    config: BenchmarkConfig,
    history: Sequence[InputT],
    trace: Sequence[InputT],
    index: int,
) -> tuple[InputT, ...]:
    if config.predictor_mode == "oracle":
        return tuple(trace[index + 1 : index + 1 + config.horizon])
    return tuple(scenario.predict_inputs(history, config.horizon))


class _MetricAccumulator:
    def __init__(self, triggers: Sequence[Any]) -> None:
        self.triggers = tuple(triggers)
        self.steps = 0
        self.inconclusive_count = 0
        self.trigger_straddle_count = 0
        self.width_sum = 0.0
        self.width_max = 0.0
        self.per_trigger_width_sum = np.zeros(len(self.triggers), dtype=float)
        self.per_trigger_width_max = np.zeros(len(self.triggers), dtype=float)
        self.extra_inconclusive_count = 0
        self.verdict_disagreement_count = 0
        self.unsafe_disagreement_count = 0
        self.false_violation_count = 0
        self.false_alarm_count = 0
        self.width_gap_sum = 0.0
        self.width_gap_max = 0.0
        self.interval_hull_mse_sum = 0.0
        self.trigger_interval_hull_mse_sum = 0.0
        self.generators: list[int] = []
        self.tick_seconds: list[float] = []
        self.reduction_seconds: list[float] = []
        self.predicted_costs: list[float] = []
        self.chosen_reducers: Counter[str] = Counter()
        self.reduction_count = 0
        self.no_op_count = 0
        self.unsound_certificate_count = 0
        self.reduction_failure_count = 0
        self.evaluated_sequences = 0
        self.pruned_sequences = 0

    def add_step(
        self,
        state: MonitorState,
        verdicts: Sequence[Verdict],
        *,
        tick_seconds: float,
        reference: ReferenceTrace | None = None,
        reference_index: int | None = None,
    ) -> None:
        self.steps += 1
        self.tick_seconds.append(tick_seconds)
        self.generators.append(state.zonotope.generator_count)

        widths = np.asarray([verdict.upper - verdict.lower for verdict in verdicts], dtype=float)
        self.width_sum += float(np.sum(widths))
        self.width_max = max(self.width_max, float(np.max(widths)) if widths.size else 0.0)
        if widths.size:
            self.per_trigger_width_sum += widths
            self.per_trigger_width_max = np.maximum(self.per_trigger_width_max, widths)

        for verdict in verdicts:
            is_inconclusive = verdict.status == "inconclusive"
            self.inconclusive_count += int(is_inconclusive)
            self.trigger_straddle_count += int(
                trigger_straddles_threshold(verdict.lower, verdict.upper, verdict.trigger)
            )

        if reference is not None and reference_index is not None:
            ref_statuses = reference.statuses[reference_index]
            ref_widths = reference.widths[reference_index]
            ref_inconclusive = reference.inconclusive[reference_index]
            lower, upper = state.zonotope.interval_bounds()
            ref_lower = reference.lower[reference_index]
            ref_upper = reference.upper[reference_index]
            self.interval_hull_mse_sum += _interval_hull_mse(
                lower,
                upper,
                ref_lower,
                ref_upper,
            )
            trigger_indices = _trigger_indices(verdicts)
            self.trigger_interval_hull_mse_sum += _interval_hull_mse(
                lower[trigger_indices],
                upper[trigger_indices],
                ref_lower[trigger_indices],
                ref_upper[trigger_indices],
            )
            for verdict_index, verdict in enumerate(verdicts):
                status = verdict.status
                ref_status = ref_statuses[verdict_index]
                self.extra_inconclusive_count += int(
                    status == "inconclusive" and not ref_inconclusive[verdict_index]
                )
                self.verdict_disagreement_count += int(status != ref_status)
                self.unsafe_disagreement_count += int(
                    status != "inconclusive" and status != ref_status
                )
                self.false_violation_count += int(
                    status == "violation" and ref_status != "violation"
                )
                self.false_alarm_count += int(status != "safe" and ref_status == "safe")
                gap = (verdict.upper - verdict.lower) - ref_widths[verdict_index]
                self.width_gap_sum += float(gap)
                self.width_gap_max = max(self.width_gap_max, float(gap))

    def finish(
        self,
        *,
        total_seconds: float,
        final_state: MonitorState,
        config: BenchmarkConfig,
        method_kind: MethodKind,
    ) -> dict[str, int | float | str | bool]:
        trigger_count = max(1, len(self.triggers))
        sample_count = max(1, self.steps * trigger_count)
        metrics: dict[str, int | float | str | bool] = {
            "steps": self.steps,
            "inconclusive_count": self.inconclusive_count,
            "inconclusive_rate": self.inconclusive_count / sample_count,
            "trigger_straddle_count": self.trigger_straddle_count,
            "mean_trigger_width": self.width_sum / sample_count,
            "max_trigger_width": self.width_max,
            "sum_trigger_width": self.width_sum,
            "extra_inconclusive_count": self.extra_inconclusive_count,
            "verdict_disagreement_count": self.verdict_disagreement_count,
            "unsafe_disagreement_count": self.unsafe_disagreement_count,
            "false_violation_count": self.false_violation_count,
            "false_violation_rate": self.false_violation_count / sample_count,
            "false_alarm_count": self.false_alarm_count,
            "false_alarm_rate": self.false_alarm_count / sample_count,
            "mean_interval_hull_mse": self.interval_hull_mse_sum / max(1, self.steps),
            "mean_trigger_interval_hull_mse": (
                self.trigger_interval_hull_mse_sum / max(1, self.steps)
            ),
            "mean_width_inflation": self.width_gap_sum / sample_count,
            "max_width_inflation": self.width_gap_max,
            "sum_width_gap": self.width_gap_sum,
            "mean_generators": float(np.mean(self.generators)) if self.generators else 0.0,
            "max_generators": max(self.generators) if self.generators else 0,
            "final_generators": final_state.zonotope.generator_count,
            "budget_violation_count": (
                0
                if method_kind == "reference"
                else sum(count > config.budget for count in self.generators)
            ),
            "reduction_count": self.reduction_count,
            "no_op_count": self.no_op_count,
            "mean_tick_seconds": float(np.mean(self.tick_seconds)) if self.tick_seconds else 0.0,
            "total_seconds": total_seconds,
            "mean_reduction_seconds": (
                float(np.mean(self.reduction_seconds)) if self.reduction_seconds else 0.0
            ),
            "total_reduction_seconds": float(np.sum(self.reduction_seconds)),
            "unsound_certificate_count": self.unsound_certificate_count,
            "reduction_failure_count": self.reduction_failure_count,
            "chosen_box_count": self.chosen_reducers["box"],
            "chosen_girard_count": self.chosen_reducers["girard"],
            "chosen_combastel_count": self.chosen_reducers["combastel"],
            "chosen_methA_count": self.chosen_reducers["methA"],
            "chosen_scott_count": self.chosen_reducers["scott"],
            "chosen_pca_count": self.chosen_reducers["pca"],
            "chosen_adaptive_count": self.chosen_reducers["adaptive"],
            "chosen_keep_norm_count": self.chosen_reducers["keep_norm"],
            "chosen_keep_calibration_aware_count": self.chosen_reducers[
                "keep_calibration_aware"
            ],
            "chosen_no_reduction_count": self.chosen_reducers["no_reduction"],
            "chosen_other_count": sum(
                count
                for name, count in self.chosen_reducers.items()
                if name
                not in {
                    "box",
                    "girard",
                    "combastel",
                    "methA",
                    "scott",
                    "pca",
                    "adaptive",
                    "keep_norm",
                    "keep_calibration_aware",
                    "no_reduction",
                }
            ),
            "mean_predicted_cost": (
                float(np.mean(self.predicted_costs)) if self.predicted_costs else 0.0
            ),
            "rollout_steps": (
                self.reduction_count * config.horizon
                if method_kind in {"mpc", "mpc_sequence", "mpc_rollout"}
                else 0
            ),
            "evaluated_sequence_count": self.evaluated_sequences,
            "pruned_sequence_count": self.pruned_sequences,
        }
        for index, trigger in enumerate(self.triggers):
            safe_name = _safe_metric_name(trigger.name)
            metrics[f"trigger_mean_width__{safe_name}"] = (
                float(self.per_trigger_width_sum[index] / max(1, self.steps))
            )
            metrics[f"trigger_max_width__{safe_name}"] = float(
                self.per_trigger_width_max[index]
            )
        return metrics


def _bootstrap_mean_ci(values: np.ndarray, *, samples: int, seed: int) -> tuple[float, float]:
    if values.size == 1 or samples <= 0:
        mean = float(np.mean(values))
        return mean, mean
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, values.size), replace=True)
    means = np.mean(draws, axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _wilcoxon_p_value(delta: np.ndarray) -> float:
    if delta.size < 2 or np.allclose(delta, 0.0):
        return 1.0
    try:
        return float(stats.wilcoxon(delta).pvalue)
    except ValueError:
        return float("nan")


def _paired_effect_size(delta: np.ndarray) -> float:
    if delta.size < 2:
        return 0.0
    std = float(np.std(delta, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(delta) / std)


def _safe_metric_name(name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in name).strip("_")


def _raw_reduction_totals(raw: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, int]]:
    columns = (
        "reduction_count",
        "reduction_failure_count",
        "evaluated_sequence_count",
        "pruned_sequence_count",
    )
    if raw.empty or not {"scenario", "predictor_mode", "method"} <= set(raw.columns):
        return {}
    present = [column for column in columns if column in raw.columns]
    totals: dict[tuple[str, str, str], dict[str, int]] = {}
    for key, group in raw.groupby(["scenario", "predictor_mode", "method"], sort=True):
        totals[key] = {
            column: int(group[column].fillna(0).astype(int).sum()) for column in present
        }
    return totals


def _parse_sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    if not isinstance(value, str) or not value:
        return ()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return ()
    if not isinstance(parsed, list):
        return ()
    return tuple(str(item) for item in parsed)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value
