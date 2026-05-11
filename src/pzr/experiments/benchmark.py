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
from pzr.monitoring.base import MonitorAdapter, MonitorState, Verdict, evaluate_triggers
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
MethodKind = Literal["reference", "static", "mpc", "mpc_sequence", "mpc_rollout"]
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
    "total_seconds",
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


@dataclass(frozen=True)
class MethodSpec:
    """A reduction method included in a benchmark suite."""

    name: str
    kind: MethodKind
    reducer_factory: Callable[[], Reducer] | None = None
    mpc_reducer_factories: tuple[Callable[[], Reducer], ...] = ()
    mpc_base_reducer_factory: Callable[[], Reducer] | None = None
    mpc_fallback_reducer_factory: Callable[[], Reducer] | None = None

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


@dataclass(frozen=True)
class BenchmarkReport:
    """Raw and aggregated benchmark results."""

    config: BenchmarkConfig
    raw_runs: pd.DataFrame
    summary: pd.DataFrame
    comparisons: pd.DataFrame
    predictor_comparisons: pd.DataFrame

    def write_artifacts(self, out_dir: str | Path) -> None:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.raw_runs.to_csv(path / "raw_runs.csv", index=False)
        self.summary.to_csv(path / "summary.csv", index=False)
        self.comparisons.to_csv(path / "comparisons.csv", index=False)
        self.predictor_comparisons.to_csv(path / "predictor_comparisons.csv", index=False)
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
                    }
                ),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


def default_methods() -> tuple[MethodSpec, ...]:
    """Methods used by the paper-style benchmark by default."""

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
                record = _run_reference_method(scenario, config, method, seed, trace, reference)
            else:
                record = _run_budgeted_method(
                    scenario,
                    config,
                    method,
                    seed,
                    trace,
                    reference if config.include_reference else None,
                )
            rows.append(record.to_row())

    raw = pd.DataFrame(rows)
    summary = summarize_runs(raw, config)
    comparisons = compare_against_mpc(raw)
    predictor_comparisons = compare_predictor_modes(raw)
    return BenchmarkReport(config, raw, summary, comparisons, predictor_comparisons)


def combine_reports(config: BenchmarkConfig, reports: Sequence[BenchmarkReport]) -> BenchmarkReport:
    """Combine separately run predictor-mode reports into one report."""

    raw = pd.concat([report.raw_runs for report in reports], ignore_index=True)
    summary = summarize_runs(raw, config)
    comparisons = compare_against_mpc(raw)
    predictor_comparisons = compare_predictor_modes(raw)
    return BenchmarkReport(config, raw, summary, comparisons, predictor_comparisons)


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
                for candidate in ("mpc_rollout_girard", "mpc_sequence", "mpc")
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
    record, statuses, widths, inconclusive = _run_without_reduction(
        scenario,
        config,
        MethodSpec.reference(),
        seed,
        trace,
        collect_reference=True,
    )
    _ = record
    return ReferenceTrace(tuple(statuses), np.asarray(widths, dtype=float), np.asarray(inconclusive))


def _run_reference_method(
    scenario: BenchmarkScenario[InputT],
    config: BenchmarkConfig,
    method: MethodSpec,
    seed: int,
    trace: Sequence[InputT],
    reference: ReferenceTrace,
) -> RunRecord:
    record, _, _, _ = _run_without_reduction(
        scenario,
        config,
        method,
        seed,
        trace,
        collect_reference=False,
    )
    return record


def _run_without_reduction(
    scenario: BenchmarkScenario[InputT],
    config: BenchmarkConfig,
    method: MethodSpec,
    seed: int,
    trace: Sequence[InputT],
    *,
    collect_reference: bool,
) -> tuple[RunRecord, list[tuple[str, ...]], list[list[float]], list[list[bool]]]:
    monitor = scenario.make_monitor()
    state = monitor.initial_state()
    accumulator = _MetricAccumulator(monitor.triggers)
    statuses: list[tuple[str, ...]] = []
    widths: list[list[float]] = []
    inconclusive: list[list[bool]] = []
    start = perf_counter()

    for measurement in trace:
        tick_start = perf_counter()
        result = monitor.step(state, measurement)
        state = result.state
        verdicts = evaluate_triggers(state.zonotope, monitor.triggers)
        accumulator.add_step(state, verdicts, tick_seconds=perf_counter() - tick_start)
        if collect_reference:
            statuses.append(tuple(verdict.status for verdict in verdicts))
            widths.append([verdict.upper - verdict.lower for verdict in verdicts])
            inconclusive.append([verdict.status == "inconclusive" for verdict in verdicts])

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
    )


def _run_budgeted_method(
    scenario: BenchmarkScenario[InputT],
    config: BenchmarkConfig,
    method: MethodSpec,
    seed: int,
    trace: Sequence[InputT],
    reference: ReferenceTrace | None,
) -> RunRecord:
    monitor = scenario.make_monitor()
    state = monitor.initial_state()
    accumulator = _MetricAccumulator(monitor.triggers)
    policy = _make_policy(method, monitor, config)
    history: list[InputT] = []
    start = perf_counter()

    for index, measurement in enumerate(trace):
        history.append(measurement)
        tick_start = perf_counter()
        result = monitor.step(state, measurement)
        state = result.state

        if state.zonotope.generator_count > config.budget:
            reduction_start = perf_counter()
            try:
                if method.kind == "static":
                    decision = policy.reduce_state(monitor, state)
                elif method.kind in {"mpc", "mpc_sequence", "mpc_rollout"}:
                    predicted = _predicted_inputs(scenario, config, history, trace, index)
                    decision = policy.reduce_state(monitor, state, predicted)
                else:
                    raise ValueError(f"unsupported budgeted method kind: {method.kind}")
            except Exception:
                accumulator.reduction_failure_count += 1
            else:
                accumulator.reduction_count += 1
                accumulator.reduction_seconds.append(perf_counter() - reduction_start)
                accumulator.unsound_certificate_count += int(
                    not decision.result.certificate.is_sound
                )
                accumulator.chosen_reducers.update([decision.reducer_name])
                if math.isfinite(decision.predicted_cost):
                    accumulator.predicted_costs.append(decision.predicted_cost)
                accumulator.evaluated_sequences += decision.evaluated_sequences
                accumulator.pruned_sequences += decision.pruned_sequences
                state = decision.state

        verdicts = evaluate_triggers(state.zonotope, monitor.triggers)
        reference_index = index if reference is not None else None
        accumulator.add_step(
            state,
            verdicts,
            tick_seconds=perf_counter() - tick_start,
            reference=reference,
            reference_index=reference_index,
        )

    metrics = accumulator.finish(
        total_seconds=perf_counter() - start,
        final_state=state,
        config=config,
        method_kind=method.kind,
    )
    return RunRecord(
        scenario=scenario.name,
        method=method.name,
        method_kind=method.kind,
        seed=seed,
        length=config.length,
        budget=config.budget,
        horizon=config.horizon,
        predictor_mode=config.predictor_mode,
        metrics=metrics,
    )


def _make_policy(
    method: MethodSpec,
    monitor: MonitorAdapter[InputT],
    config: BenchmarkConfig,
) -> (
    StaticReductionPolicy
    | MPCPolicy[InputT]
    | SequenceMPCPolicy[InputT]
    | RolloutMPCPolicy[InputT]
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
    raise ValueError(f"method {method.name} is not a budgeted policy")


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
        self.width_gap_sum = 0.0
        self.width_gap_max = 0.0
        self.generators: list[int] = []
        self.tick_seconds: list[float] = []
        self.reduction_seconds: list[float] = []
        self.predicted_costs: list[float] = []
        self.chosen_reducers: Counter[str] = Counter()
        self.reduction_count = 0
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
                verdict.lower <= verdict.trigger.threshold <= verdict.upper
            )

        if reference is not None and reference_index is not None:
            ref_statuses = reference.statuses[reference_index]
            ref_widths = reference.widths[reference_index]
            ref_inconclusive = reference.inconclusive[reference_index]
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
        reductions = max(1, self.reduction_count)
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
        _ = reductions
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
