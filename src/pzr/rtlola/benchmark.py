"""RTLola-native benchmark execution, aggregation, and artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
from pathlib import Path
import time
from typing import Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
import yaml

from pzr.artifact_io import write_csv_atomic, write_text_atomic
from pzr.rtlola.actions import (
    CORE_STATIC_ACTION_NAMES,
    EXACT_BASELINE_ACTION_NAME,
    EXPLICIT_ACTION_METHOD_NAMES,
    MPC_ACTION_NAMES,
    RtlolaAction,
    STATIC_ACTION_METHOD_NAMES,
    default_action_catalog,
)
from pzr.rtlola.binding import (
    BINDING_BUILD_PROFILE,
    BINDING_REVISION,
    INTERPRETER_REVISION,
)
from pzr.rtlola.engine import (
    RtlolaApproximationReference,
    RtlolaBindingError,
    RtlolaEngine,
    RtlolaEvent,
    RtlolaStateRef,
    RtlolaStepResult,
)
from pzr.rtlola.input_prediction import (
    InputPredictionDiagnostic,
    prediction_diagnostics,
    predict_future_events,
)
from pzr.rtlola.reference import (
    REFERENCE_CACHE_SCHEMA,
    RtlolaReferenceStep,
    load_or_compute_reference,
    reference_cache_path,
)
from pzr.rtlola.scenarios import RtlolaScenario, scenario_by_name
from pzr.rtlola.search import (
    MPC_VARIANTS,
    MpcRootEvaluation,
    RtlolaNoFeasibleAction,
    RtlolaSearchResult,
    beam_search,
    choose_static_action,
    full_width_terminal_search,
    search_mpc_variant,
)
from pzr.rtlola.tables import trigger_confusion


METHOD_SET_CHOICES = ("core", "static", "mpc", "all")
CORE_STATIC_METHODS = CORE_STATIC_ACTION_NAMES
STATIC_METHODS = STATIC_ACTION_METHOD_NAMES
BASELINE_MPC_METHODS = ("mpc_terminal_beam",)
FULL_WIDTH_MPC_METHOD = "mpc_terminal_full_width"
PREDICTIVE_MPC_METHODS = {
    "mpc_terminal_beam_predictive_hold": "hold",
    "mpc_terminal_beam_predictive_linear": "linear",
    "mpc_terminal_beam_predictive_quadratic": "quadratic",
}
MPC_METHODS = (
    *tuple(MPC_VARIANTS), FULL_WIDTH_MPC_METHOD, *tuple(PREDICTIVE_MPC_METHODS),
)
ALL_METHODS = (*STATIC_METHODS, *MPC_METHODS)
CORE_METHODS = (*CORE_STATIC_METHODS, *BASELINE_MPC_METHODS)
TERMINAL_BINDING_APPROX_LOSS = "terminal_binding_approx_loss"
RTLOLA_AGGREGATE_METRICS = [
    "mean_state_width",
    "max_state_width",
    "mean_generator_count",
    "mean_active_dynamic_generator_count",
    "mean_zero_dynamic_generator_count",
    "total_reductions",
    "total_time_ms",
    "event_loop_time_ms",
    "mean_approx_loss",
    "final_approx_loss",
    "max_approx_loss",
    "sum_approx_loss",
    "fpr",
    "fnr",
    "false_positive_count",
    "false_negative_count",
    "reference_positive_count",
    "reference_negative_count",
    "trigger_positive_rate",
    "post_event_over_bound_count",
    "post_event_over_bound_rate",
    "fallback_count",
    "fallback_rate",
    "reducer_failure_count",
    "infeasible_candidate_count",
    "tail_fallback_count",
]


class DirectRtlolaPolicy(Protocol):
    def choose(
        self,
        engine: RtlolaEngine,
        state: RtlolaStateRef,
        event: RtlolaEvent,
        budget: int,
    ) -> RtlolaSearchResult: ...


@dataclass(frozen=True)
class RtlolaBenchmarkConfig:
    scenario: str = "omni_robot"
    trace_kind: str = "default"
    length: int = 30
    budget: int = 10
    horizon: int = 2
    beam_width: int = 4
    prediction_step_seconds: float = 0.1
    mpc_tail_horizon: int = 8
    mpc_root_beam_width: int = 1
    seeds: int = 3
    method_set: str = "core"
    methods: list[str] | None = None
    reference_mode: str = "exact"
    mpc_reference: str = "rollout"
    reference_cache: str | None = None
    output_dir: str = "results/rtlola"
    mpc_objective: str = field(
        init=False,
        default=TERMINAL_BINDING_APPROX_LOSS,
    )
    binding_revision: str = field(init=False, default=BINDING_REVISION)
    interpreter_revision: str = field(init=False, default=INTERPRETER_REVISION)
    binding_build_profile: str = field(init=False, default=BINDING_BUILD_PROFILE)
    mpc_candidate_names: list[str] = field(
        default_factory=lambda: list(MPC_ACTION_NAMES),
    )


@dataclass(frozen=True)
class RtlolaStepRecord:
    seed: int
    method: str
    step: int
    pre_generator_count: int
    generator_count: int
    total_generator_count: int
    active_dynamic_generator_count: int
    active_total_generator_count: int
    zero_dynamic_generator_count: int
    zero_total_generator_count: int
    logical_dynamic_dimension: int
    reduced: bool
    reducer_used: str
    state_width: float
    approx_loss: float
    false_positive: bool | float
    false_negative: bool | float
    trigger_positive: bool
    exact_trigger_positive: bool | float
    trigger_verdicts: dict[str, bool]
    exact_trigger_verdicts: dict[str, bool]
    decision_time_ms: float
    binding_runtime_ns: float
    predicted_cost: float = 0.0
    predicted_sequence: tuple[str, ...] = ()
    evaluated_leaves: int = 0
    pruned_branches: int = 0
    post_event_over_bound: bool = False
    fallback_used: bool = False
    reducer_failure_count: int = 0
    infeasible_candidate_count: int = 0
    mpc_variant: str = ""
    mpc_objective: str = ""
    root_strategy: str = ""
    optimized_horizon: int = 0
    realized_optimized_horizon: int = 0
    configured_tail_horizon: int = 0
    realized_tail_steps: int = 0
    root_beam_width: int = 0
    explicit_path_loss: float = float("nan")
    explicit_terminal_loss: float = float("nan")
    tail_path_loss: float = float("nan")
    tail_terminal_loss: float = float("nan")
    tail_fallback_count: int = 0
    input_predictor: str = ""
    prediction_step_seconds: float = float("nan")
    root_evaluations: tuple[MpcRootEvaluation, ...] = ()


@dataclass(frozen=True)
class RtlolaExecutedStep:
    """Selected action and committed native result before metric row creation."""

    pre_generator_count: int
    committed: RtlolaStepResult
    decision: RtlolaSearchResult
    decision_time_ms: float


@dataclass(frozen=True)
class RtlolaRunResult:
    method: str
    seed: int
    steps: tuple[RtlolaStepRecord, ...]
    budget: int | None = None
    trace_kind: str = "default"
    prediction_diagnostics: tuple[InputPredictionDiagnostic, ...] = ()
    event_loop_time_ms: float = float("nan")

    @property
    def total_reductions(self) -> int:
        return sum(1 for step in self.steps if step.reduced)

    @property
    def total_time_ms(self) -> float:
        return float(sum(step.decision_time_ms for step in self.steps))


@dataclass(frozen=True)
class RtlolaRunFailure:
    scenario: str
    trace_kind: str
    method: str
    seed: int
    budget: int
    step: int
    time: float
    phase: str
    failure_type: str
    message: str


@dataclass(frozen=True)
class RtlolaFailedRun:
    """A native failure plus every scientifically useful pre-failure step."""

    failure: RtlolaRunFailure
    partial: RtlolaRunResult


@dataclass
class RtlolaBenchmarkResult:
    config: RtlolaBenchmarkConfig
    raw_results: tuple[RtlolaRunResult, ...]
    timeseries: pd.DataFrame
    summary: pd.DataFrame
    aggregate: pd.DataFrame
    prediction_diagnostics: pd.DataFrame
    prediction_error_summary: pd.DataFrame
    failures: tuple[RtlolaRunFailure, ...] = ()
    failed_timeseries: pd.DataFrame = field(default_factory=pd.DataFrame)


def methods_for_config(config: RtlolaBenchmarkConfig) -> tuple[str, ...]:
    if config.methods is not None:
        available = {*EXPLICIT_ACTION_METHOD_NAMES, *MPC_METHODS}
        unknown = [method for method in config.methods if method not in available]
        if unknown:
            valid = ", ".join(sorted(available))
            bad = ", ".join(unknown)
            raise ValueError(f"unknown RTLola method(s): {bad}; valid methods: {valid}")
        return tuple(config.methods)
    if config.method_set == "core":
        return CORE_METHODS
    if config.method_set == "static":
        return STATIC_METHODS
    if config.method_set == "mpc":
        return MPC_METHODS
    if config.method_set == "all":
        return (*STATIC_METHODS, *MPC_METHODS)
    valid_sets = ", ".join(METHOD_SET_CHOICES)
    raise ValueError(f"method_set must be one of: {valid_sets}")


def bootstrap_ci(
    values: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return a deterministic bootstrap mean and confidence interval."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.asarray([
        np.mean(rng.choice(finite, size=finite.size, replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.mean(finite)),
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


def aggregate_summary(
    summary: pd.DataFrame,
    *,
    metric_columns: Sequence[str] = RTLOLA_AGGREGATE_METRICS,
) -> pd.DataFrame:
    """Aggregate seed-level RTLola metrics by method and experiment cell."""
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_columns = [
        column for column in ("method", "budget", "trace_kind")
        if column in summary
    ]
    for group_key, group in summary.groupby(group_columns, dropna=False):
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        row: dict[str, object] = dict(zip(group_columns, keys))
        for column in metric_columns:
            if column not in group:
                continue
            mean, lo, hi = bootstrap_ci(group[column].to_numpy(dtype=np.float64))
            row[f"{column}_mean"] = mean
            row[f"{column}_ci95_lo"] = lo
            row[f"{column}_ci95_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def run_benchmark(config: RtlolaBenchmarkConfig) -> RtlolaBenchmarkResult:
    """Evaluate configured built-in methods through the common validated loop."""
    return _run_benchmark_loop(
        config,
        methods=methods_for_config(config),
        direct_policies={},
    )


def run_direct_policy_benchmark(
    config: RtlolaBenchmarkConfig,
    policy: DirectRtlolaPolicy,
    *,
    method: str,
) -> RtlolaBenchmarkResult:
    """Evaluate one direct policy through the common validated loop."""
    return _run_benchmark_loop(
        config,
        methods=(method,),
        direct_policies={method: policy},
    )


def run_event_trace_benchmark(
    config: RtlolaBenchmarkConfig,
    events: Sequence[RtlolaEvent],
    *,
    trace_kind: str,
    seed: int,
    method: str,
    policy: DirectRtlolaPolicy | None = None,
    reference_steps: Sequence[RtlolaReferenceStep] | None = None,
) -> RtlolaBenchmarkResult:
    """Evaluate one supplied trace while retaining normal native accounting."""
    methods = (method,)
    direct_policies = {method: policy} if policy is not None else {}
    _validate_benchmark_config(config, methods, direct_policies)
    if len(events) != config.length:
        raise ValueError(
            f"supplied trace has {len(events)} events, expected {config.length}"
        )
    scenario = scenario_by_name(config.scenario)
    catalog = default_action_catalog(tuple(config.mpc_candidate_names))
    if reference_steps is not None:
        if config.reference_mode == "off":
            raise ValueError("supplied exact reference cannot be used with reference_mode=off")
        if len(reference_steps) != len(events):
            raise ValueError("supplied exact reference length differs from trace")
        reference = tuple(reference_steps)
    else:
        reference = (
            load_or_compute_reference(
                events,
                scenario=scenario,
                trace_kind=trace_kind,
                seed=seed,
                cache_path=reference_cache_path(config.reference_cache, seed, 1),
                include_approximation=config.reference_mode == "exact",
            )
            if config.reference_mode != "off" else None
        )
    outcome = _run_single(
        config,
        scenario,
        events,
        method,
        catalog.mpc_candidates,
        catalog.by_name,
        catalog.fallback,
        seed,
        trace_kind,
        reference,
        direct_policy=policy,
    )
    raw = () if isinstance(outcome, RtlolaFailedRun) else (outcome,)
    failures = (outcome.failure,) if isinstance(outcome, RtlolaFailedRun) else ()
    partial = (outcome.partial,) if isinstance(outcome, RtlolaFailedRun) else ()
    return _build_benchmark_result(config, raw, failures, partial)


def _run_benchmark_loop(
    config: RtlolaBenchmarkConfig,
    *,
    methods: Sequence[str],
    direct_policies: Mapping[str, DirectRtlolaPolicy],
) -> RtlolaBenchmarkResult:
    """Run built-in and direct policies with identical trace/reference accounting."""
    _validate_benchmark_config(config, methods, direct_policies)
    scenario = scenario_by_name(config.scenario)
    catalog = default_action_catalog(tuple(config.mpc_candidate_names))
    by_name = catalog.by_name
    fallback = catalog.fallback
    mpc_candidates = catalog.mpc_candidates
    raw: list[RtlolaRunResult] = []
    failures: list[RtlolaRunFailure] = []
    partial: list[RtlolaRunResult] = []
    for seed in range(config.seeds):
        generated = scenario.generate_trace(
            config.length,
            seed,
            trace_kind=config.trace_kind,
        )
        trace = generated.events
        reference = (
            load_or_compute_reference(
                trace,
                scenario=scenario,
                trace_kind=generated.trace_kind,
                seed=seed,
                cache_path=reference_cache_path(
                    config.reference_cache,
                    seed,
                    config.seeds,
                ),
                include_approximation=config.reference_mode == "exact",
            )
            if config.reference_mode != "off" else None
        )
        for method in methods:
            outcome = _run_single(
                config,
                scenario,
                trace,
                method,
                mpc_candidates,
                by_name,
                fallback,
                seed,
                generated.trace_kind,
                reference,
                direct_policy=direct_policies.get(method),
            )
            if isinstance(outcome, RtlolaFailedRun):
                failures.append(outcome.failure)
                partial.append(outcome.partial)
            else:
                raw.append(outcome)
    return _build_benchmark_result(
        config, tuple(raw), tuple(failures), tuple(partial),
    )


def _validate_benchmark_config(
    config: RtlolaBenchmarkConfig,
    methods: Sequence[str],
    direct_policies: Mapping[str, DirectRtlolaPolicy],
) -> None:
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("benchmark methods must be non-empty and unique")
    if not set(direct_policies) <= set(methods):
        raise ValueError("direct policies must correspond to requested methods")
    if config.reference_mode not in {"exact", "verdict", "off"}:
        raise ValueError("reference_mode must be one of: exact, verdict, off")
    if config.mpc_reference not in {"rollout", "cache"}:
        raise ValueError("mpc_reference must be one of: rollout, cache")
    if config.mpc_reference == "cache" and config.reference_mode != "exact":
        raise ValueError("mpc_reference=cache requires reference_mode=exact")
    if config.mpc_reference == "cache" and set(methods) & set(PREDICTIVE_MPC_METHODS):
        raise ValueError(
            "predictive MPC cannot use exact-cache references for predicted inputs"
        )
    if config.length < 1:
        raise ValueError("length must be >= 1")
    if config.seeds < 1:
        raise ValueError("seeds must be >= 1")
    if config.budget < 0:
        raise ValueError("budget must be non-negative")
    if config.horizon < 0:
        raise ValueError("horizon must be non-negative")
    if config.beam_width < 1:
        raise ValueError("beam_width must be >= 1")
    if not np.isfinite(config.prediction_step_seconds) or config.prediction_step_seconds <= 0.0:
        raise ValueError("prediction_step_seconds must be positive and finite")
    if config.mpc_tail_horizon < 0:
        raise ValueError("mpc_tail_horizon must be non-negative")
    if config.mpc_root_beam_width < 1:
        raise ValueError("mpc_root_beam_width must be >= 1")


def _build_benchmark_result(
    config: RtlolaBenchmarkConfig,
    raw: tuple[RtlolaRunResult, ...],
    failures: tuple[RtlolaRunFailure, ...],
    partial: tuple[RtlolaRunResult, ...] = (),
) -> RtlolaBenchmarkResult:
    """Build aligned dataframes for generated and caller-supplied traces."""
    timeseries = results_to_dataframe(raw)
    summary = summarize_results(raw)
    aggregate = aggregate_summary(summary)
    diagnostic_rows = [
        {
            **asdict(row),
            "method": run.method,
            "seed": run.seed,
            "budget": run.budget,
            "trace_kind": run.trace_kind,
        }
        for run in raw
        for row in run.prediction_diagnostics
    ]
    prediction_rows = pd.DataFrame(diagnostic_rows)
    prediction_summary = summarize_prediction_errors(prediction_rows)
    if timeseries.empty:
        timeseries = pd.DataFrame(
            columns=("seed", "method", "budget", "trace_kind"),
        )
    if summary.empty:
        summary = pd.DataFrame(
            columns=("method", "seed", "budget", "trace_kind"),
        )
    if aggregate.empty:
        aggregate = pd.DataFrame(
            columns=("method", "budget", "trace_kind"),
        )
    return RtlolaBenchmarkResult(
        config=config,
        raw_results=tuple(raw),
        timeseries=timeseries,
        summary=summary,
        aggregate=aggregate,
        prediction_diagnostics=prediction_rows,
        prediction_error_summary=prediction_summary,
        failures=failures,
        failed_timeseries=results_to_dataframe(partial),
    )


def _run_single(
    config: RtlolaBenchmarkConfig,
    scenario: RtlolaScenario,
    trace: Sequence[RtlolaEvent],
    method: str,
    mpc_candidates: tuple[RtlolaAction, ...],
    by_name: dict[str, RtlolaAction],
    fallback: RtlolaAction,
    seed: int,
    trace_kind: str,
    reference: Sequence[RtlolaReferenceStep] | None,
    direct_policy: DirectRtlolaPolicy | None = None,
) -> RtlolaRunResult | RtlolaFailedRun:
    engine = RtlolaEngine(
        scenario.spec,
        event_arity=scenario.event_arity,
        expected_verdict_keys=scenario.expected_verdict_keys,
    )
    steps: list[RtlolaStepRecord] = []
    run_prediction_diagnostics: list[InputPredictionDiagnostic] = []
    run_start = time.perf_counter()
    for index, event in enumerate(trace):
        state = engine.snapshot(step=index, time=event.time)
        try:
            pre_metrics = engine.metrics(state)
        except RtlolaBindingError as exc:
            return _run_failure(
                config,
                scenario,
                trace_kind,
                method,
                seed,
                index,
                event,
                "inspect",
                exc,
                steps=steps,
                prediction_diagnostics=run_prediction_diagnostics,
                event_loop_time_ms=(time.perf_counter() - run_start) * 1000.0,
            )
        predictor = PREDICTIVE_MPC_METHODS.get(method)
        if predictor is None:
            future = tuple(trace[index + 1:index + 1 + config.horizon])
            prediction = None
        else:
            prediction = predict_future_events(
                trace[:index + 1],
                predictor=predictor,
                horizon=config.horizon,
                step_seconds=config.prediction_step_seconds,
                timestamp_channel_indices=scenario.timestamp_channel_indices,
            )
            future = prediction.events
        start = time.perf_counter()
        try:
            decision = direct_policy.choose(
                engine, state, event, config.budget,
            ) if direct_policy is not None else _select_method_decision(
                config=config,
                engine=engine,
                trace=trace,
                state=state,
                event=event,
                step=index,
                method=method,
                future=future,
                mpc_candidates=mpc_candidates,
                by_name=by_name,
                fallback=fallback,
                reference=reference,
            )
        except (RtlolaBindingError, RtlolaNoFeasibleAction) as exc:
            return _run_failure(
                config,
                scenario,
                trace_kind,
                method,
                seed,
                index,
                event,
                "select",
                exc,
                steps=steps,
                prediction_diagnostics=run_prediction_diagnostics,
                event_loop_time_ms=(time.perf_counter() - run_start) * 1000.0,
            )

        if prediction is not None:
            actual_future = tuple(trace[index + 1:index + 1 + config.horizon])
            run_prediction_diagnostics.extend(prediction_diagnostics(
                prediction,
                actual_future,
                predictor=predictor,
                decision_step=index,
                channel_names=scenario.input_channel_names,
            ))

        try:
            committed = _commit_decision(
                engine,
                event,
                decision,
                step=index + 1,
            )
        except RtlolaBindingError as exc:
            return _run_failure(
                config,
                scenario,
                trace_kind,
                method,
                seed,
                index,
                event,
                "commit",
                exc,
                steps=steps,
                prediction_diagnostics=run_prediction_diagnostics,
                event_loop_time_ms=(time.perf_counter() - run_start) * 1000.0,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        executed = RtlolaExecutedStep(
            pre_generator_count=pre_metrics.dynamic_generator_count,
            committed=committed,
            decision=decision,
            decision_time_ms=elapsed_ms,
        )
        try:
            steps.append(make_step_record(
                engine=engine,
                scenario=scenario,
                seed=seed,
                method=method,
                step=index,
                budget=config.budget,
                executed=executed,
                reference=reference[index] if reference is not None else None,
            ))
        except RtlolaBindingError as exc:
            return _run_failure(
                config,
                scenario,
                trace_kind,
                method,
                seed,
                index,
                event,
                "measure",
                exc,
                steps=steps,
                prediction_diagnostics=run_prediction_diagnostics,
                event_loop_time_ms=(time.perf_counter() - run_start) * 1000.0,
            )
    return RtlolaRunResult(
        method=method,
        seed=seed,
        steps=tuple(steps),
        budget=config.budget,
        trace_kind=trace_kind,
        prediction_diagnostics=tuple(run_prediction_diagnostics),
        event_loop_time_ms=(time.perf_counter() - run_start) * 1000.0,
    )


def _select_method_decision(
    *,
    config: RtlolaBenchmarkConfig,
    engine: RtlolaEngine,
    trace: Sequence[RtlolaEvent],
    state: RtlolaStateRef,
    event: RtlolaEvent,
    step: int,
    method: str,
    future: Sequence[RtlolaEvent],
    mpc_candidates: tuple[RtlolaAction, ...],
    by_name: dict[str, RtlolaAction],
    fallback: RtlolaAction,
    reference: Sequence[RtlolaReferenceStep] | None,
) -> RtlolaSearchResult:
    """Select a native action for one benchmark event without committing it."""
    if method == EXACT_BASELINE_ACTION_NAME:
        first = by_name[EXACT_BASELINE_ACTION_NAME]
        first_step = engine.branch_step(state, event, first, config.budget)
        return RtlolaSearchResult(
            first_action=first,
            first_action_budget=config.budget,
            first_step=first_step,
            predicted_cost=first_step.metrics.cost(),
            predicted_sequence=(EXACT_BASELINE_ACTION_NAME,),
            evaluated_leaves=1,
            pruned_branches=0,
        )
    if method == "mpc_terminal_beam" or method in PREDICTIVE_MPC_METHODS:
        explicit_reference = _mpc_reference_steps(
            config,
            reference,
            start=step,
            count=1 + len(future),
        )
        decision = beam_search(
            engine,
            state,
            event,
            future,
            mpc_candidates,
            config.budget,
            config.beam_width,
            fallback=fallback,
            none_action=by_name[EXACT_BASELINE_ACTION_NAME],
            use_reference_loss=True,
            configured_horizon=config.horizon,
            reference_steps=explicit_reference,
        )
        if method in PREDICTIVE_MPC_METHODS:
            return replace(
                decision,
                mpc_variant=method,
                input_predictor=PREDICTIVE_MPC_METHODS[method],
                prediction_step_seconds=config.prediction_step_seconds,
            )
        return decision
    if method == FULL_WIDTH_MPC_METHOD:
        explicit_reference = _mpc_reference_steps(
            config,
            reference,
            start=step,
            count=1 + len(future),
        )
        return full_width_terminal_search(
            engine,
            state,
            event,
            future,
            mpc_candidates,
            config.budget,
            fallback=fallback,
            none_action=by_name[EXACT_BASELINE_ACTION_NAME],
            configured_horizon=config.horizon,
            reference_steps=explicit_reference,
        )
    if method in MPC_VARIANTS:
        variant = MPC_VARIANTS[method]
        optimized_future_count = (
            config.horizon if variant.uses_configured_horizon else 0
        )
        optimized_future = tuple(
            trace[step + 1:step + 1 + optimized_future_count]
        )
        tail_start = step + 1 + optimized_future_count
        tail = (
            tuple(trace[tail_start:tail_start + config.mpc_tail_horizon])
            if variant.uses_tail else ()
        )
        explicit_reference = _mpc_reference_steps(
            config,
            reference,
            start=step,
            count=1 + len(optimized_future) + len(tail),
        )
        return search_mpc_variant(
            engine,
            state,
            event,
            optimized_future,
            tail,
            mpc_candidates,
            config.budget,
            config.beam_width,
            variant=variant,
            root_beam_width=config.mpc_root_beam_width,
            fallback=fallback,
            none_action=by_name[EXACT_BASELINE_ACTION_NAME],
            tail_action=by_name["girard"],
            configured_horizon=config.horizon,
            configured_tail_horizon=config.mpc_tail_horizon,
            reference_steps=explicit_reference,
        )
    return choose_static_action(
        engine,
        state,
        event,
        by_name[method],
        config.budget,
        fallback=fallback,
        none_action=by_name[EXACT_BASELINE_ACTION_NAME],
    )


def _mpc_reference_steps(
    config: RtlolaBenchmarkConfig,
    reference: Sequence[RtlolaReferenceStep] | None,
    *,
    start: int,
    count: int,
) -> tuple[RtlolaApproximationReference, ...] | None:
    """Select absolute exact rows for cache-referenced MPC scoring."""
    if config.mpc_reference == "rollout":
        return None
    if reference is None:
        raise ValueError("exact MPC references are unavailable")
    selected = tuple(reference[start:start + count])
    if len(selected) != count:
        raise ValueError("exact MPC reference slice is shorter than the rollout")
    compact = tuple(step.approximation for step in selected)
    if any(item is None for item in compact):
        raise ValueError("exact MPC reference lacks approximation intervals")
    return compact  # type: ignore[return-value]


def _commit_decision(
    engine: RtlolaEngine,
    event: RtlolaEvent,
    decision: RtlolaSearchResult,
    *,
    step: int,
) -> RtlolaStepResult:
    """Apply the selected binding-native transform to the live monitor."""
    return engine.live_step(
        event,
        decision.first_action,
        decision.first_action_budget,
        step=step,
    )


def _run_failure(
    config: RtlolaBenchmarkConfig,
    scenario: RtlolaScenario,
    trace_kind: str,
    method: str,
    seed: int,
    step: int,
    event: RtlolaEvent,
    phase: str,
    error: Exception,
    *,
    steps: Sequence[RtlolaStepRecord],
    prediction_diagnostics: Sequence[InputPredictionDiagnostic],
    event_loop_time_ms: float,
) -> RtlolaFailedRun:
    failure = RtlolaRunFailure(
        scenario=scenario.name, trace_kind=trace_kind, method=method, seed=seed,
        budget=config.budget, step=step, time=event.time, phase=phase,
        failure_type=type(error).__name__, message=str(error),
    )
    return RtlolaFailedRun(
        failure=failure,
        partial=RtlolaRunResult(
            method=method,
            seed=seed,
            steps=tuple(steps),
            budget=config.budget,
            trace_kind=trace_kind,
            prediction_diagnostics=tuple(prediction_diagnostics),
            event_loop_time_ms=event_loop_time_ms,
        ),
    )


def make_step_record(
    *,
    engine: RtlolaEngine,
    scenario: RtlolaScenario,
    seed: int,
    method: str,
    step: int,
    budget: int,
    executed: RtlolaExecutedStep,
    reference: RtlolaReferenceStep | None,
) -> RtlolaStepRecord:
    """Create one benchmark row for any static, predictive, or learned policy."""
    committed = executed.committed
    decision = executed.decision
    approx_loss = (
        engine.approx_loss_reference(reference.approximation, committed.state)
        if reference is not None and reference.approximation is not None
        else float("nan")
    )
    predicted_triggers = {
        key: bool(committed.verdict.get(key, False))
        for key in scenario.trigger_keys
    }
    exact_triggers = (
        dict(reference.verdicts)
        if reference is not None else {}
    )
    predicted_positive = any(predicted_triggers.values())
    if exact_triggers:
        exact_positive: bool | float = any(exact_triggers.values())
        false_positive: bool | float = predicted_positive and not exact_positive
        false_negative: bool | float = not predicted_positive and exact_positive
    else:
        exact_positive = float("nan")
        false_positive = float("nan")
        false_negative = float("nan")
    return RtlolaStepRecord(
        seed=seed,
        method=method,
        step=step,
        pre_generator_count=executed.pre_generator_count,
        generator_count=committed.metrics.dynamic_generator_count,
        total_generator_count=committed.metrics.total_generator_count,
        active_dynamic_generator_count=committed.metrics.active_dynamic_generator_count,
        active_total_generator_count=committed.metrics.active_total_generator_count,
        zero_dynamic_generator_count=committed.metrics.zero_dynamic_generator_count,
        zero_total_generator_count=committed.metrics.zero_total_generator_count,
        logical_dynamic_dimension=committed.metrics.logical_dynamic_dimension,
        reduced=decision.first_action.name != EXACT_BASELINE_ACTION_NAME,
        reducer_used=decision.first_action.name,
        state_width=committed.metrics.state_width,
        approx_loss=approx_loss,
        false_positive=false_positive,
        false_negative=false_negative,
        trigger_positive=predicted_positive,
        exact_trigger_positive=exact_positive,
        trigger_verdicts=predicted_triggers,
        exact_trigger_verdicts=exact_triggers,
        decision_time_ms=executed.decision_time_ms,
        binding_runtime_ns=_binding_runtime_ns(committed.verdict),
        predicted_cost=decision.predicted_cost,
        predicted_sequence=decision.predicted_sequence,
        evaluated_leaves=decision.evaluated_leaves,
        pruned_branches=decision.pruned_branches,
        post_event_over_bound=committed.metrics.dynamic_generator_count > budget,
        fallback_used=decision.fallback_used,
        reducer_failure_count=decision.reducer_failure_count,
        infeasible_candidate_count=decision.infeasible_candidate_count,
        mpc_variant=decision.mpc_variant,
        mpc_objective=decision.mpc_objective,
        root_strategy=decision.root_strategy,
        optimized_horizon=decision.optimized_horizon,
        realized_optimized_horizon=decision.realized_optimized_horizon,
        configured_tail_horizon=decision.configured_tail_horizon,
        realized_tail_steps=decision.realized_tail_steps,
        root_beam_width=decision.root_beam_width,
        explicit_path_loss=decision.explicit_path_loss,
        explicit_terminal_loss=decision.explicit_terminal_loss,
        tail_path_loss=decision.tail_path_loss,
        tail_terminal_loss=decision.tail_terminal_loss,
        tail_fallback_count=decision.tail_fallback_count,
        input_predictor=decision.input_predictor,
        prediction_step_seconds=decision.prediction_step_seconds,
        root_evaluations=decision.root_evaluations,
    )


def prepare_reference_cache(config: RtlolaBenchmarkConfig) -> tuple[Path, ...]:
    """Generate or validate configured exact-reference caches without a run."""
    if config.reference_mode == "off":
        raise ValueError("reference-only preparation requires exact or verdict mode")
    if config.reference_cache is None:
        raise ValueError("reference-only preparation requires --reference-cache")
    scenario = scenario_by_name(config.scenario)
    paths: list[Path] = []
    for seed in range(config.seeds):
        generated = scenario.generate_trace(
            config.length,
            seed,
            trace_kind=config.trace_kind,
        )
        path = reference_cache_path(config.reference_cache, seed, config.seeds)
        assert path is not None
        load_or_compute_reference(
            generated.events,
            scenario=scenario,
            trace_kind=generated.trace_kind,
            seed=seed,
            cache_path=path,
            include_approximation=config.reference_mode == "exact",
        )
        paths.append(path)
    return tuple(paths)


def results_to_dataframe(results: Sequence[RtlolaRunResult]) -> pd.DataFrame:
    rows = []
    for run in results:
        for step in run.steps:
            row = {
                "seed": step.seed,
                "method": step.method,
                "budget": run.budget,
                "trace_kind": run.trace_kind,
                "step": step.step,
                "pre_generator_count": step.pre_generator_count,
                "generator_count": step.generator_count,
                "total_generator_count": step.total_generator_count,
                "active_dynamic_generator_count": step.active_dynamic_generator_count,
                "active_total_generator_count": step.active_total_generator_count,
                "zero_dynamic_generator_count": step.zero_dynamic_generator_count,
                "zero_total_generator_count": step.zero_total_generator_count,
                "logical_dynamic_dimension": step.logical_dynamic_dimension,
                "reduced": step.reduced,
                "reducer_used": step.reducer_used,
                "state_width": step.state_width,
                "approx_loss": step.approx_loss,
                "false_positive": step.false_positive,
                "false_negative": step.false_negative,
                "trigger_positive": step.trigger_positive,
                "exact_trigger_positive": step.exact_trigger_positive,
                "decision_time_ms": step.decision_time_ms,
                "binding_runtime_ns": step.binding_runtime_ns,
                "predicted_cost": step.predicted_cost,
                "predicted_sequence": ",".join(step.predicted_sequence),
                "evaluated_leaves": step.evaluated_leaves,
                "pruned_branches": step.pruned_branches,
                "post_event_over_bound": step.post_event_over_bound,
                "fallback_used": step.fallback_used,
                "reducer_failure_count": step.reducer_failure_count,
                "infeasible_candidate_count": step.infeasible_candidate_count,
                "mpc_variant": step.mpc_variant,
                "mpc_objective": step.mpc_objective,
                "root_strategy": step.root_strategy,
                "optimized_horizon": step.optimized_horizon,
                "realized_optimized_horizon": step.realized_optimized_horizon,
                "configured_tail_horizon": step.configured_tail_horizon,
                "realized_tail_steps": step.realized_tail_steps,
                "root_beam_width": step.root_beam_width,
                "explicit_path_loss": step.explicit_path_loss,
                "explicit_terminal_loss": step.explicit_terminal_loss,
                "tail_path_loss": step.tail_path_loss,
                "tail_terminal_loss": step.tail_terminal_loss,
                "tail_fallback_count": step.tail_fallback_count,
                "input_predictor": step.input_predictor,
                "prediction_step_seconds": step.prediction_step_seconds,
            }
            row.update(step.trigger_verdicts)
            row.update({
                f"exact_{key}": value
                for key, value in step.exact_trigger_verdicts.items()
            })
            rows.append(row)
    return pd.DataFrame(rows)


def root_evaluations_to_dataframe(
    results: Sequence[RtlolaRunResult],
) -> pd.DataFrame:
    """Return one diagnostic row per evaluated MPC first action."""
    columns = (
        "seed",
        "method",
        "budget",
        "trace_kind",
        "step",
        "root_action",
        "selected",
        "feasible",
        "complete",
        "predicted_cost",
        "predicted_sequence",
        "explicit_path_loss",
        "explicit_terminal_loss",
        "tail_path_loss",
        "tail_terminal_loss",
        "realized_tail_steps",
        "failure_count",
        "mpc_objective",
        "root_strategy",
        "optimized_horizon",
        "realized_optimized_horizon",
        "configured_tail_horizon",
        "root_beam_width",
        "input_predictor",
        "prediction_step_seconds",
    )
    rows: list[dict[str, object]] = []
    for run in results:
        for step in run.steps:
            for evaluation in step.root_evaluations:
                rows.append({
                    "seed": step.seed,
                    "method": step.method,
                    "budget": run.budget,
                    "trace_kind": run.trace_kind,
                    "step": step.step,
                    "root_action": evaluation.root_action,
                    "selected": evaluation.root_action == step.reducer_used,
                    "feasible": evaluation.feasible,
                    "complete": evaluation.complete,
                    "predicted_cost": evaluation.predicted_cost,
                    "predicted_sequence": ",".join(evaluation.predicted_sequence),
                    "explicit_path_loss": evaluation.explicit_path_loss,
                    "explicit_terminal_loss": evaluation.explicit_terminal_loss,
                    "tail_path_loss": evaluation.tail_path_loss,
                    "tail_terminal_loss": evaluation.tail_terminal_loss,
                    "realized_tail_steps": evaluation.realized_tail_steps,
                    "failure_count": evaluation.failure_count,
                    "mpc_objective": step.mpc_objective,
                    "root_strategy": step.root_strategy,
                    "optimized_horizon": step.optimized_horizon,
                    "realized_optimized_horizon": step.realized_optimized_horizon,
                    "configured_tail_horizon": step.configured_tail_horizon,
                    "root_beam_width": step.root_beam_width,
                    "input_predictor": step.input_predictor,
                    "prediction_step_seconds": step.prediction_step_seconds,
                })
    return pd.DataFrame(rows, columns=columns)


def failures_to_dataframe(
    failures: Sequence[RtlolaRunFailure],
) -> pd.DataFrame:
    columns = tuple(RtlolaRunFailure.__dataclass_fields__)
    return pd.DataFrame(
        [asdict(failure) for failure in failures],
        columns=columns,
    )


def summarize_results(results: Sequence[RtlolaRunResult]) -> pd.DataFrame:
    rows = []
    for run in results:
        widths = np.asarray([step.state_width for step in run.steps], dtype=np.float64)
        gens = np.asarray([step.generator_count for step in run.steps], dtype=np.float64)
        active_gens = np.asarray(
            [step.active_dynamic_generator_count for step in run.steps],
            dtype=np.float64,
        )
        zero_gens = np.asarray(
            [step.zero_dynamic_generator_count for step in run.steps],
            dtype=np.float64,
        )
        logical_dims = np.asarray(
            [step.logical_dynamic_dimension for step in run.steps],
            dtype=np.float64,
        )
        approx_losses = np.asarray([step.approx_loss for step in run.steps], dtype=np.float64)
        fps = np.asarray([step.false_positive for step in run.steps], dtype=np.float64)
        fns = np.asarray([step.false_negative for step in run.steps], dtype=np.float64)
        exact_positives = np.asarray(
            [step.exact_trigger_positive for step in run.steps],
            dtype=np.float64,
        )
        trigger_positives = np.asarray([step.trigger_positive for step in run.steps], dtype=np.float64)
        has_reference = np.isfinite(exact_positives)
        reference_positive_count = int(np.sum(exact_positives[has_reference] == 1.0))
        reference_negative_count = int(np.sum(exact_positives[has_reference] == 0.0))
        false_positive_count = int(np.nansum(fps))
        false_negative_count = int(np.nansum(fns))
        post_event_over_bound_count = sum(1 for step in run.steps if step.post_event_over_bound)
        fallback_count = sum(1 for step in run.steps if step.fallback_used)
        reducer_failure_count = sum(step.reducer_failure_count for step in run.steps)
        infeasible_candidate_count = sum(step.infeasible_candidate_count for step in run.steps)
        tail_fallback_count = sum(step.tail_fallback_count for step in run.steps)
        evaluated_leaves = np.asarray(
            [step.evaluated_leaves for step in run.steps], dtype=np.float64,
        )
        pruned_branches = np.asarray(
            [step.pruned_branches for step in run.steps], dtype=np.float64,
        )
        realized_horizons = np.asarray(
            [step.realized_optimized_horizon for step in run.steps], dtype=np.float64,
        )
        first_step = run.steps[0] if run.steps else None
        rows.append({
            "method": run.method,
            "seed": run.seed,
            "budget": run.budget,
            "trace_kind": run.trace_kind,
            "mpc_variant": first_step.mpc_variant if first_step is not None else "",
            "mpc_objective": first_step.mpc_objective if first_step is not None else "",
            "root_strategy": first_step.root_strategy if first_step is not None else "",
            "optimized_horizon": first_step.optimized_horizon if first_step is not None else 0,
            "mean_realized_optimized_horizon": float(np.mean(realized_horizons)),
            "min_realized_optimized_horizon": int(np.min(realized_horizons)),
            "max_realized_optimized_horizon": int(np.max(realized_horizons)),
            "configured_tail_horizon": (
                first_step.configured_tail_horizon if first_step is not None else 0
            ),
            "root_beam_width": first_step.root_beam_width if first_step is not None else 0,
            "input_predictor": first_step.input_predictor if first_step is not None else "",
            "prediction_step_seconds": (
                first_step.prediction_step_seconds if first_step is not None else float("nan")
            ),
            "mean_state_width": float(np.mean(widths)),
            "max_state_width": float(np.max(widths)),
            "mean_generator_count": float(np.mean(gens)),
            "max_generator_count": int(np.max(gens)),
            "mean_active_dynamic_generator_count": float(np.mean(active_gens)),
            "max_active_dynamic_generator_count": int(np.max(active_gens)),
            "mean_zero_dynamic_generator_count": float(np.mean(zero_gens)),
            "max_zero_dynamic_generator_count": int(np.max(zero_gens)),
            "mean_logical_dynamic_dimension": float(np.mean(logical_dims)),
            "max_logical_dynamic_dimension": int(np.max(logical_dims)),
            "total_reductions": run.total_reductions,
            "total_time_ms": run.total_time_ms,
            "event_loop_time_ms": run.event_loop_time_ms,
            "mean_approx_loss": _nanmean(approx_losses),
            "final_approx_loss": _nanfinal(approx_losses),
            "max_approx_loss": _nanmax(approx_losses),
            "sum_approx_loss": _nansum(approx_losses),
            "false_positive_count": false_positive_count,
            "false_negative_count": false_negative_count,
            "reference_positive_count": reference_positive_count,
            "reference_negative_count": reference_negative_count,
            "fpr": (
                false_positive_count / reference_negative_count
                if reference_negative_count else float("nan")
            ),
            "fnr": (
                false_negative_count / reference_positive_count
                if reference_positive_count else float("nan")
            ),
            "trigger_positive_rate": _nanmean(trigger_positives),
            "post_event_over_bound_count": post_event_over_bound_count,
            "post_event_over_bound_rate": post_event_over_bound_count / len(run.steps) if run.steps else 0.0,
            "fallback_count": fallback_count,
            "fallback_rate": fallback_count / len(run.steps) if run.steps else 0.0,
            "reducer_failure_count": reducer_failure_count,
            "infeasible_candidate_count": infeasible_candidate_count,
            "tail_fallback_count": tail_fallback_count,
            "mean_evaluated_leaves": float(np.mean(evaluated_leaves)),
            "max_evaluated_leaves": int(np.max(evaluated_leaves)),
            "total_evaluated_leaves": int(np.sum(evaluated_leaves)),
            "mean_pruned_branches": float(np.mean(pruned_branches)),
            "max_pruned_branches": int(np.max(pruned_branches)),
            "total_pruned_branches": int(np.sum(pruned_branches)),
        })
    return pd.DataFrame(rows)


def summarize_prediction_errors(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate causal forecast errors by trace, predictor, lead, and channel."""
    columns = (
        "trace_kind", "predictor", "lead", "channel_index", "channel_name",
        "prediction_count", "mean_error", "mae", "rmse", "max_absolute_error",
    )
    if diagnostics.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    group_columns = [
        "trace_kind", "predictor", "lead", "channel_index", "channel_name",
    ]
    for keys, frame in diagnostics.groupby(group_columns, dropna=False, sort=True):
        errors = frame["error"].to_numpy(dtype=np.float64)
        absolute = np.abs(errors)
        rows.append({
            **dict(zip(group_columns, keys)),
            "prediction_count": len(frame),
            "mean_error": float(np.mean(errors)),
            "mae": float(np.mean(absolute)),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "max_absolute_error": float(np.max(absolute)),
        })
    return pd.DataFrame(rows, columns=columns)


def _nanmean(values: np.ndarray) -> float:
    if values.size == 0 or np.isnan(values).all():
        return float("nan")
    return float(np.nanmean(values))


def _nanmax(values: np.ndarray) -> float:
    if values.size == 0 or np.isnan(values).all():
        return float("nan")
    return float(np.nanmax(values))


def _nanfinal(values: np.ndarray) -> float:
    if values.size == 0 or not np.isfinite(values[-1]):
        return float("nan")
    return float(values[-1])


def _nansum(values: np.ndarray) -> float:
    if values.size == 0 or np.isnan(values).all():
        return float("nan")
    return float(np.nansum(values))


def _binding_runtime_ns(verdict: dict[str, object]) -> float:
    value = verdict.get("runtime_ns", float("nan"))
    try:
        runtime = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return runtime if np.isfinite(runtime) else float("nan")


def save_benchmark_results(result: RtlolaBenchmarkResult, output_dir: Path) -> None:
    scenario_dir = output_dir / result.config.scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(result.timeseries, scenario_dir / "timeseries.csv")
    write_csv_atomic(result.summary, scenario_dir / "summary.csv")
    write_csv_atomic(result.aggregate, scenario_dir / "aggregate.csv")
    write_csv_atomic(
        result.prediction_diagnostics,
        scenario_dir / "input_prediction_errors.csv",
    )
    write_csv_atomic(
        result.prediction_error_summary,
        scenario_dir / "input_prediction_error_summary.csv",
    )
    write_csv_atomic(
        root_evaluations_to_dataframe(result.raw_results),
        scenario_dir / "mpc_root_evaluations.csv",
    )
    write_csv_atomic(
        failures_to_dataframe(result.failures), scenario_dir / "run_failures.csv",
    )
    _write_dashboard_artifacts(result, scenario_dir, output_dir)
    scenario = scenario_by_name(result.config.scenario)
    config_payload = {
        **asdict(result.config),
        "mpc_variants": {
            name: {
                "objective": variant.objective.value,
                "root_strategy": variant.root_strategy.value,
                "uses_configured_horizon": variant.uses_configured_horizon,
                "uses_tail": variant.uses_tail,
            }
            for name, variant in MPC_VARIANTS.items()
        },
        "spec_sha256": hashlib.sha256(
            scenario.spec.encode("utf-8"),
        ).hexdigest(),
        "source_revision": scenario.source_revision,
        "reference_cache_schema": REFERENCE_CACHE_SCHEMA,
        "trigger_labels": dict(
            zip(scenario.trigger_keys, scenario.trigger_labels)
        ),
    }
    write_text_atomic(
        yaml.safe_dump(config_payload, sort_keys=False), output_dir / "config.yaml",
    )


def _write_dashboard_artifacts(
    result: RtlolaBenchmarkResult,
    scenario_dir: Path,
    output_dir: Path,
) -> None:
    scenario = scenario_by_name(result.config.scenario)
    write_csv_atomic(
        trigger_confusion(result.timeseries, scenario.trigger_keys),
        scenario_dir / "trigger_confusion.csv",
    )
    pareto_columns = [
        "method",
        "seed",
        "total_time_ms",
        "mean_approx_loss",
        "final_approx_loss",
        "max_approx_loss",
        "sum_approx_loss",
        "mean_state_width",
        "max_state_width",
    ]
    pareto = (
        result.summary[pareto_columns].copy()
        if not result.summary.empty
        else pd.DataFrame(columns=pareto_columns)
    )
    write_csv_atomic(pareto, scenario_dir / "pareto_runtime_vs_loss.csv")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_pareto(pareto, figures_dir / f"{result.config.scenario}_pareto_runtime_vs_loss")


def _plot_pareto(pareto: pd.DataFrame, stem: Path) -> None:
    if pareto.empty:
        return
    import matplotlib.pyplot as plt

    grouped = pareto.groupby("method", as_index=False).agg({
        "total_time_ms": "mean",
        "mean_approx_loss": "mean",
        "mean_state_width": "mean",
    })
    y_col = (
        "mean_state_width"
        if grouped["mean_approx_loss"].isna().all() else "mean_approx_loss"
    )
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.scatter(grouped["total_time_ms"], grouped[y_col])
    for row in grouped.itertuples(index=False):
        ax.annotate(row.method, (row.total_time_ms, getattr(row, y_col)), fontsize=8)
    ax.set_xlabel("Runtime [ms]")
    ax.set_ylabel(
        "Mean state width" if y_col == "mean_state_width"
        else "Mean approximation loss"
    )
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=160)
    plt.close(fig)
