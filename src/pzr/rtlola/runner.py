"""Benchmark runner for RTLola-native monitors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import time
from typing import Sequence

import numpy as np
import pandas as pd
import yaml

from pzr.experiments.evaluation import aggregate_summary
from pzr.rtlola.actions import RtlolaAction, action_by_name, default_actions
from pzr.rtlola.engine import RtlolaEngine, RtlolaEvent, RtlolaStateRef
from pzr.rtlola.metrics import selected_row_width_sum
from pzr.rtlola.scenarios import RtlolaScenarioSpec, scenario_by_name
from pzr.rtlola.search import RtlolaSearchResult, beam_search, choose_static_action


STATIC_METHODS = (
    "none",
    "girard",
    "scott",
    "interval_hull",
    "pca",
    "althoff_a",
    "colinear_scale",
    "colinear",
)
MPC_METHODS = ("mpc_beam",)
ALL_METHODS = (*STATIC_METHODS, *MPC_METHODS)
MPC_ACTION_NAMES = ("girard", "scott", "interval_hull", "pca")
RTLOLA_AGGREGATE_METRICS = [
    "mean_state_zonotope_width",
    "max_state_zonotope_width",
    "mean_generator_count",
    "mean_active_dynamic_generator_count",
    "mean_zero_dynamic_generator_count",
    "total_reductions",
    "total_time_ms",
    "mean_state_zonotope_approx_error",
    "max_state_zonotope_approx_error",
    "state_zonotope_abs_error_range",
    "mean_relevant_state_width",
    "mean_relevant_state_approx_error",
    "mean_approx_loss",
    "max_approx_loss",
    "false_positive_rate",
    "false_negative_rate",
    "trigger_positive_rate",
    "post_event_over_bound_count",
    "post_event_over_bound_rate",
    "fallback_count",
    "fallback_rate",
    "reducer_failure_count",
    "infeasible_candidate_count",
]


@dataclass(frozen=True)
class RtlolaBenchmarkConfig:
    scenario: str = "omni_robot"
    trace_kind: str = "default"
    length: int = 30
    budget: int = 10
    horizon: int = 2
    beam_width: int = 4
    seeds: int = 3
    method_set: str = "all"
    methods: list[str] | None = None
    reference_mode: str = "exact"
    output_dir: str = "results/rtlola"
    learned_mode: str = "none"
    regret_iterations: int = 3
    regret_epochs: int = 100
    regret_train_seeds: int | None = None
    regret_eval_seeds: int | None = None
    regret_loss: str = "pairwise"


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
    reduced: bool
    reducer_used: str
    state_zonotope_width_sum: float
    exact_state_zonotope_width_sum: float
    state_zonotope_approx_error_sum: float
    relevant_state_width_sum: float
    exact_relevant_state_width_sum: float
    relevant_state_approx_error_sum: float
    approx_loss: float
    false_positive: bool | float
    false_negative: bool | float
    trigger_positive: bool
    verdicts: dict[str, object]
    public_bounds: dict[str, tuple[float, float]]
    reduction_time_ms: float
    predicted_cost: float = 0.0
    predicted_sequence: tuple[str, ...] = ()
    evaluated_leaves: int = 0
    pruned_branches: int = 0
    post_event_over_bound: bool = False
    budget_violation: bool = False
    fallback_used: bool = False
    reducer_failure_count: int = 0
    infeasible_candidate_count: int = 0


@dataclass(frozen=True)
class RtlolaGroundTruthStep:
    """Unreduced RTLola state-zonotope bounds and public verdicts."""

    lower: np.ndarray
    upper: np.ndarray
    dynamic_matrix: np.ndarray
    state: RtlolaStateRef
    relevant_width_sum: float
    width_sum: float
    verdicts: dict[str, object]
    public_bounds: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class RtlolaRunResult:
    method: str
    seed: int
    steps: tuple[RtlolaStepRecord, ...]

    @property
    def total_reductions(self) -> int:
        return sum(1 for step in self.steps if step.reduced)

    @property
    def total_time_ms(self) -> float:
        return float(sum(step.reduction_time_ms for step in self.steps))

    @property
    def budget_violations(self) -> int:
        return sum(1 for step in self.steps if step.budget_violation)


@dataclass
class RtlolaBenchmarkResult:
    config: RtlolaBenchmarkConfig
    raw_results: tuple[RtlolaRunResult, ...]
    timeseries: pd.DataFrame
    summary: pd.DataFrame
    aggregate: pd.DataFrame


def methods_for_config(config: RtlolaBenchmarkConfig) -> tuple[str, ...]:
    if config.methods is not None:
        unknown = [method for method in config.methods if method not in ALL_METHODS]
        if unknown:
            valid = ", ".join(ALL_METHODS)
            bad = ", ".join(unknown)
            raise ValueError(f"unknown RTLola method(s): {bad}; valid methods: {valid}")
        return tuple(config.methods)
    if config.method_set == "static":
        return STATIC_METHODS
    if config.method_set == "mpc":
        return MPC_METHODS
    if config.method_set == "all":
        return (*STATIC_METHODS, *MPC_METHODS)
    raise ValueError("method_set must be one of: static, mpc, all")


def mpc_actions(by_name: dict[str, RtlolaAction]) -> tuple[RtlolaAction, ...]:
    """Return budgeted, apples-to-apples actions used by RTLola MPC."""
    return tuple(by_name[name] for name in MPC_ACTION_NAMES)


def run_benchmark(config: RtlolaBenchmarkConfig) -> RtlolaBenchmarkResult:
    if config.reference_mode not in {"exact", "off"}:
        raise ValueError("reference_mode must be one of: exact, off")
    scenario = scenario_by_name(config.scenario)
    actions = default_actions()
    by_name = action_by_name(actions)
    fallback = by_name["interval"]
    mpc_candidates = mpc_actions(by_name)
    raw: list[RtlolaRunResult] = []
    for seed in range(config.seeds):
        trace = scenario.generate_events(config.length, seed, trace_kind=config.trace_kind)
        ground_truth = (
            compute_ground_truth(trace, scenario=scenario)
            if config.reference_mode == "exact" else None
        )
        for method in methods_for_config(config):
            raw.append(_run_single(
                config,
                scenario,
                trace,
                method,
                mpc_candidates,
                by_name,
                fallback,
                seed,
                ground_truth,
            ))
    timeseries = results_to_dataframe(raw)
    summary = summarize_results(raw)
    aggregate = aggregate_summary(summary, metric_columns=RTLOLA_AGGREGATE_METRICS)
    return RtlolaBenchmarkResult(
        config=config,
        raw_results=tuple(raw),
        timeseries=timeseries,
        summary=summary,
        aggregate=aggregate,
    )


def _run_single(
    config: RtlolaBenchmarkConfig,
    scenario: RtlolaScenarioSpec,
    trace: Sequence[RtlolaEvent],
    method: str,
    mpc_candidates: tuple[RtlolaAction, ...],
    by_name: dict[str, RtlolaAction],
    fallback: RtlolaAction,
    seed: int,
    ground_truth: Sequence[RtlolaGroundTruthStep] | None,
) -> RtlolaRunResult:
    engine = RtlolaEngine(
        scenario.spec,
        event_arity=scenario.event_arity,
        expected_verdict_keys=(*scenario.expected_verdict_keys, *scenario.public_stream_keys),
    )
    steps: list[RtlolaStepRecord] = []
    for index, event in enumerate(trace):
        state = engine.snapshot(step=index, time=event.time)
        pre_metrics = engine.metrics(state)
        future = tuple(trace[index + 1:index + 1 + config.horizon])
        start = time.perf_counter()
        if method == "none":
            first = by_name["none"]
            first_step = engine.branch_step(state, event, first, config.budget)
            decision = RtlolaSearchResult(
                first_action=first,
                first_action_budget=config.budget,
                first_step=first_step,
                predicted_cost=scenario.cost(engine, first_step),
                predicted_sequence=("none",),
                evaluated_leaves=1,
                pruned_branches=0,
            )
        elif method == "mpc_beam":
            decision = beam_search(
                engine,
                state,
                event,
                future,
                mpc_candidates,
                config.budget,
                config.beam_width,
                fallback=fallback,
                none_action=by_name["none"],
                cost_fn=scenario.cost,
                use_reference_loss=True,
            )
        else:
            decision = choose_static_action(
                engine,
                state,
                event,
                by_name[method],
                config.budget,
                fallback=fallback,
                none_action=by_name["none"],
                cost_fn=scenario.cost,
            )

        committed = engine.live_step(
            event,
            decision.first_action,
            decision.first_action_budget,
            step=index + 1,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        dynamic_matrix = engine.matrices(committed.state)[0]
        lower, upper = _state_interval_bounds(dynamic_matrix)
        relevant_width = selected_row_width_sum(dynamic_matrix, scenario.relevant_rows)
        if ground_truth is not None:
            gt = ground_truth[index]
            if lower.shape != gt.lower.shape:
                raise RuntimeError(
                    "RTLola reduced and exact state-zonotope dimensions differ "
                    f"(method={method}, seed={seed}, step={index}, "
                    f"reduced_dim={lower.shape[0]}, exact_dim={gt.lower.shape[0]})"
                )
            approx_error = float(np.sum(np.abs(lower - gt.lower) + np.abs(upper - gt.upper)))
            relevant_error = _selected_interval_error(
                lower,
                upper,
                gt.lower,
                gt.upper,
                scenario.relevant_rows,
            )
            approx_loss = engine.approx_loss(gt.state, committed.state)
            false_positive = _false_positive(
                committed.verdict, gt.verdicts, scenario.expected_verdict_keys,
            )
            false_negative = _false_negative(
                committed.verdict, gt.verdicts, scenario.expected_verdict_keys,
            )
            exact_width = gt.width_sum
            exact_relevant_width = gt.relevant_width_sum
        else:
            approx_error = float("nan")
            relevant_error = float("nan")
            approx_loss = float("nan")
            false_positive = float("nan")
            false_negative = float("nan")
            exact_width = float("nan")
            exact_relevant_width = float("nan")
        public_bounds = _public_bounds(committed.verdict, scenario.public_stream_keys)
        trigger_positive = _trigger_positive(committed.verdict, scenario.trigger_keys)
        post_event_over_bound = committed.metrics.dynamic_generator_count > config.budget
        violation = False
        steps.append(RtlolaStepRecord(
            seed=seed,
            method=method,
            step=index,
            pre_generator_count=pre_metrics.dynamic_generator_count,
            generator_count=committed.metrics.dynamic_generator_count,
            total_generator_count=committed.metrics.total_generator_count,
            active_dynamic_generator_count=committed.metrics.active_dynamic_generator_count,
            active_total_generator_count=committed.metrics.active_total_generator_count,
            zero_dynamic_generator_count=committed.metrics.zero_dynamic_generator_count,
            zero_total_generator_count=committed.metrics.zero_total_generator_count,
            reduced=decision.first_action.name != "none",
            reducer_used=decision.first_action.name,
            state_zonotope_width_sum=committed.metrics.full_width_sum,
            exact_state_zonotope_width_sum=exact_width,
            state_zonotope_approx_error_sum=approx_error,
            relevant_state_width_sum=relevant_width,
            exact_relevant_state_width_sum=exact_relevant_width,
            relevant_state_approx_error_sum=relevant_error,
            approx_loss=approx_loss,
            false_positive=false_positive,
            false_negative=false_negative,
            trigger_positive=trigger_positive,
            verdicts=committed.verdict,
            public_bounds=public_bounds,
            reduction_time_ms=elapsed_ms,
            predicted_cost=decision.predicted_cost,
            predicted_sequence=decision.predicted_sequence,
            evaluated_leaves=decision.evaluated_leaves,
            pruned_branches=decision.pruned_branches,
            post_event_over_bound=post_event_over_bound,
            budget_violation=violation,
            fallback_used=decision.fallback_used,
            reducer_failure_count=decision.reducer_failure_count,
            infeasible_candidate_count=decision.infeasible_candidate_count,
        ))
        if violation:
            raise RuntimeError(
                "committed RTLola step exceeded dynamic budget "
                f"(method={method}, seed={seed}, step={index}, "
                f"count={committed.metrics.dynamic_generator_count}, budget={config.budget})"
            )
    return RtlolaRunResult(method=method, seed=seed, steps=tuple(steps))


def infer_fresh_generator_reserve(
    scenario: RtlolaScenarioSpec,
    trace: Sequence[RtlolaEvent],
    by_name: dict[str, RtlolaAction],
    *,
    sample_steps: int = 5,
) -> int | None:
    """Infer fixed per-event generator growth for diagnostics.

    RTLola applies the zonotope transform before accepting an event. The
    benchmark budget is the RTLola transform bound, so this reserve is not used
    to choose reducer targets. It remains useful for auditing per-cycle slack
    growth.
    """
    if not trace:
        return 0
    engine = RtlolaEngine(
        scenario.spec,
        event_arity=scenario.event_arity,
        expected_verdict_keys=(*scenario.expected_verdict_keys, *scenario.public_stream_keys),
    )
    none = by_name["none"]
    deltas: list[int] = []
    for index, event in enumerate(trace[:max(1, sample_steps)]):
        state = engine.snapshot(step=index, time=event.time)
        before = engine.metrics(state).dynamic_generator_count
        committed = engine.live_step(event, none, budget=0, step=index + 1)
        after = committed.metrics.dynamic_generator_count
        delta = after - before
        if delta < 0:
            return None
        deltas.append(delta)
    first = deltas[0]
    if all(delta == first for delta in deltas):
        return first
    return None


def compute_ground_truth(
    trace: Sequence[RtlolaEvent],
    *,
    scenario: RtlolaScenarioSpec | None = None,
) -> tuple[RtlolaGroundTruthStep, ...]:
    """Run the RTLola monitor without reductions for exact state-zonotope metrics."""
    scenario = scenario or scenario_by_name("omni_robot")
    actions = action_by_name(default_actions())
    engine = RtlolaEngine(
        scenario.spec,
        event_arity=scenario.event_arity,
        expected_verdict_keys=(*scenario.expected_verdict_keys, *scenario.public_stream_keys),
    )
    out: list[RtlolaGroundTruthStep] = []
    for step, event in enumerate(trace):
        committed = engine.live_step(event, actions["none"], budget=0, step=step + 1)
        verdict = committed.verdict
        for key in (*scenario.expected_verdict_keys, *scenario.public_stream_keys):
            if key not in verdict:
                raise RuntimeError(f"RTLola ground truth verdict missing key at step {step}: {key}")
        zono = engine.matrices(committed.state)[0]
        lower, upper = _state_interval_bounds(zono)
        out.append(RtlolaGroundTruthStep(
            lower=lower,
            upper=upper,
            dynamic_matrix=zono.copy(),
            state=committed.state,
            relevant_width_sum=selected_row_width_sum(zono, scenario.relevant_rows),
            width_sum=float(np.sum(upper - lower)),
            verdicts=dict(verdict),
            public_bounds=_public_bounds(verdict, scenario.public_stream_keys),
        ))
    return tuple(out)


def results_to_dataframe(results: Sequence[RtlolaRunResult]) -> pd.DataFrame:
    rows = []
    for run in results:
        for step in run.steps:
            row = {
                "seed": step.seed,
                "method": step.method,
                "step": step.step,
                "pre_generator_count": step.pre_generator_count,
                "generator_count": step.generator_count,
                "total_generator_count": step.total_generator_count,
                "active_dynamic_generator_count": step.active_dynamic_generator_count,
                "active_total_generator_count": step.active_total_generator_count,
                "zero_dynamic_generator_count": step.zero_dynamic_generator_count,
                "zero_total_generator_count": step.zero_total_generator_count,
                "reduced": step.reduced,
                "reducer_used": step.reducer_used,
                "state_zonotope_width_sum": step.state_zonotope_width_sum,
                "exact_state_zonotope_width_sum": step.exact_state_zonotope_width_sum,
                "state_zonotope_approx_error_sum": step.state_zonotope_approx_error_sum,
                "relevant_state_width_sum": step.relevant_state_width_sum,
                "exact_relevant_state_width_sum": step.exact_relevant_state_width_sum,
                "relevant_state_approx_error_sum": step.relevant_state_approx_error_sum,
                "approx_loss": step.approx_loss,
                "false_positive": step.false_positive,
                "false_negative": step.false_negative,
                "trigger_positive": step.trigger_positive,
                "reduction_time_ms": step.reduction_time_ms,
                "predicted_cost": step.predicted_cost,
                "predicted_sequence": ",".join(step.predicted_sequence),
                "evaluated_leaves": step.evaluated_leaves,
                "pruned_branches": step.pruned_branches,
                "post_event_over_bound": step.post_event_over_bound,
                "budget_violation": step.budget_violation,
                "fallback_used": step.fallback_used,
                "reducer_failure_count": step.reducer_failure_count,
                "infeasible_candidate_count": step.infeasible_candidate_count,
            }
            row.update(step.verdicts)
            for key, bounds in step.public_bounds.items():
                row[f"{key}_lower"] = bounds[0]
                row[f"{key}_upper"] = bounds[1]
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_results(results: Sequence[RtlolaRunResult]) -> pd.DataFrame:
    rows = []
    for run in results:
        widths = np.asarray([step.state_zonotope_width_sum for step in run.steps], dtype=np.float64)
        gens = np.asarray([step.generator_count for step in run.steps], dtype=np.float64)
        active_gens = np.asarray(
            [step.active_dynamic_generator_count for step in run.steps],
            dtype=np.float64,
        )
        zero_gens = np.asarray(
            [step.zero_dynamic_generator_count for step in run.steps],
            dtype=np.float64,
        )
        approx_errors = np.asarray(
            [step.state_zonotope_approx_error_sum for step in run.steps],
            dtype=np.float64,
        )
        relevant_widths = np.asarray(
            [step.relevant_state_width_sum for step in run.steps],
            dtype=np.float64,
        )
        relevant_errors = np.asarray(
            [step.relevant_state_approx_error_sum for step in run.steps],
            dtype=np.float64,
        )
        approx_losses = np.asarray([step.approx_loss for step in run.steps], dtype=np.float64)
        fps = np.asarray([step.false_positive for step in run.steps], dtype=np.float64)
        fns = np.asarray([step.false_negative for step in run.steps], dtype=np.float64)
        trigger_positives = np.asarray([step.trigger_positive for step in run.steps], dtype=np.float64)
        post_event_over_bound_count = sum(1 for step in run.steps if step.post_event_over_bound)
        fallback_count = sum(1 for step in run.steps if step.fallback_used)
        reducer_failure_count = sum(step.reducer_failure_count for step in run.steps)
        infeasible_candidate_count = sum(step.infeasible_candidate_count for step in run.steps)
        rows.append({
            "method": run.method,
            "seed": run.seed,
            "mean_state_zonotope_width": float(np.mean(widths)),
            "max_state_zonotope_width": float(np.max(widths)),
            "mean_generator_count": float(np.mean(gens)),
            "max_generator_count": int(np.max(gens)),
            "mean_active_dynamic_generator_count": float(np.mean(active_gens)),
            "max_active_dynamic_generator_count": int(np.max(active_gens)),
            "mean_zero_dynamic_generator_count": float(np.mean(zero_gens)),
            "max_zero_dynamic_generator_count": int(np.max(zero_gens)),
            "total_reductions": run.total_reductions,
            "total_time_ms": run.total_time_ms,
            "budget_violations": run.budget_violations,
            "unsound_certificates": 0,
            "mean_state_zonotope_approx_error": float(np.mean(approx_errors)),
            "max_state_zonotope_approx_error": float(np.max(approx_errors)),
            "state_zonotope_abs_error_range": float(np.max(approx_errors) - np.min(approx_errors)),
            "mean_relevant_state_width": float(np.mean(relevant_widths)),
            "max_relevant_state_width": float(np.max(relevant_widths)),
            "mean_relevant_state_approx_error": float(np.mean(relevant_errors)),
            "max_relevant_state_approx_error": float(np.max(relevant_errors)),
            "mean_approx_loss": float(np.mean(approx_losses)),
            "max_approx_loss": float(np.max(approx_losses)),
            "false_positive_rate": _nanmean(fps),
            "false_negative_rate": _nanmean(fns),
            "trigger_positive_rate": _nanmean(trigger_positives),
            "post_event_over_bound_count": post_event_over_bound_count,
            "post_event_over_bound_rate": post_event_over_bound_count / len(run.steps) if run.steps else 0.0,
            "fallback_count": fallback_count,
            "fallback_rate": fallback_count / len(run.steps) if run.steps else 0.0,
            "reducer_failure_count": reducer_failure_count,
            "infeasible_candidate_count": infeasible_candidate_count,
        })
    return pd.DataFrame(rows)


def _nanmean(values: np.ndarray) -> float:
    if values.size == 0 or np.isnan(values).all():
        return float("nan")
    return float(np.nanmean(values))


def _state_interval_bounds(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(matrix, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] < 1:
        raise ValueError(f"expected 2D state-zonotope matrix, got {z.shape}")
    center = z[:, 0]
    radius = np.abs(z[:, 1:]).sum(axis=1) if z.shape[1] > 1 else np.zeros(z.shape[0])
    return center - radius, center + radius


def _false_positive(
    reduced_verdict: dict[str, object],
    exact_verdict: dict[str, object],
    keys: Sequence[str],
) -> bool:
    for key in keys:
        if bool(reduced_verdict.get(key, False)) and not bool(exact_verdict.get(key, False)):
            return True
    return False


def _false_negative(
    reduced_verdict: dict[str, object],
    exact_verdict: dict[str, object],
    keys: Sequence[str],
) -> bool:
    for key in keys:
        if not bool(reduced_verdict.get(key, False)) and bool(exact_verdict.get(key, False)):
            return True
    return False


def _trigger_positive(verdict: dict[str, object], keys: Sequence[str]) -> bool:
    return any(bool(verdict.get(key, False)) for key in keys)


def _selected_interval_error(
    lower: np.ndarray,
    upper: np.ndarray,
    exact_lower: np.ndarray,
    exact_upper: np.ndarray,
    rows: Sequence[int],
) -> float:
    if not rows:
        return float(np.sum(np.abs(lower - exact_lower) + np.abs(upper - exact_upper)))
    idx = np.asarray(tuple(rows), dtype=np.int64)
    return float(np.sum(np.abs(lower[idx] - exact_lower[idx]) + np.abs(upper[idx] - exact_upper[idx])))


def _public_bounds(verdict: dict[str, object], keys: Sequence[str]) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for key in keys:
        if key not in verdict:
            continue
        parsed = _value_bounds(verdict[key])
        if parsed is not None:
            bounds[key] = parsed
    return bounds


def _value_bounds(value: object) -> tuple[float, float] | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        scalar = float(value)
        return (scalar, scalar) if np.isfinite(scalar) else None
    for left, right in (("lower", "upper"), ("lo", "hi"), ("lb", "ub")):
        if hasattr(value, left) and hasattr(value, right):
            lo_value = getattr(value, left)
            hi_value = getattr(value, right)
            if callable(lo_value) or callable(hi_value):
                continue
            lo = float(lo_value)
            hi = float(hi_value)
            return (lo, hi) if np.isfinite(lo) and np.isfinite(hi) else None
    text = str(value)
    affine_coeffs = [
        float(v) for v in re.findall(
            r"([-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*\*\s*s\d+",
            text,
        )
    ]
    nums = [float(v) for v in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", text)]
    if affine_coeffs and nums:
        center = nums[0]
        radius = float(np.sum(np.abs(np.asarray(affine_coeffs, dtype=np.float64))))
        return (center - radius, center + radius)
    if len(nums) >= 2:
        lo, hi = nums[0], nums[1]
        return (lo, hi) if np.isfinite(lo) and np.isfinite(hi) else None
    if len(nums) == 1:
        scalar = nums[0]
        return (scalar, scalar) if np.isfinite(scalar) else None
    return None


def save_benchmark_results(result: RtlolaBenchmarkResult, output_dir: Path) -> None:
    scenario_dir = output_dir / result.config.scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    result.timeseries.to_csv(scenario_dir / "timeseries.csv", index=False)
    result.summary.to_csv(scenario_dir / "summary.csv", index=False)
    result.aggregate.to_csv(scenario_dir / "aggregate.csv", index=False)
    _write_dashboard_artifacts(result, scenario_dir, output_dir)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(asdict(result.config), f, sort_keys=False)


def _write_dashboard_artifacts(
    result: RtlolaBenchmarkResult,
    scenario_dir: Path,
    output_dir: Path,
) -> None:
    scenario = scenario_by_name(result.config.scenario)
    _trigger_confusion(result.timeseries, scenario.trigger_keys).to_csv(
        scenario_dir / "trigger_confusion.csv", index=False,
    )
    pareto = result.summary[[
        "method",
        "seed",
        "total_time_ms",
        "mean_approx_loss",
        "max_approx_loss",
        "mean_relevant_state_width",
    ]].copy()
    pareto.to_csv(scenario_dir / "pareto_runtime_vs_loss.csv", index=False)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_pareto(pareto, figures_dir / f"{result.config.scenario}_pareto_runtime_vs_loss")
    for stream in scenario.public_stream_keys:
        _plot_public_range(
            result.timeseries,
            stream,
            figures_dir / f"{result.config.scenario}_{stream}_range",
        )


def _trigger_confusion(timeseries: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    rows = []
    for method, frame in timeseries.groupby("method"):
        rows.append({
            "method": method,
            "trigger_keys": ",".join(keys),
            "false_positive_steps": _series_sum(frame, "false_positive"),
            "false_negative_steps": _series_sum(frame, "false_negative"),
            "trigger_positive_steps": int(frame["trigger_positive"].sum()) if "trigger_positive" in frame else 0,
            "steps": int(len(frame)),
            "false_positive_rate": _series_mean(frame, "false_positive"),
            "false_negative_rate": _series_mean(frame, "false_negative"),
            "trigger_positive_rate": _series_mean(frame, "trigger_positive"),
        })
    return pd.DataFrame(rows)


def _series_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame or frame.empty:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().all():
        return float("nan")
    return float(values.mean(skipna=True))


def _series_sum(frame: pd.DataFrame, column: str) -> float:
    if column not in frame or frame.empty:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().all():
        return float("nan")
    return float(values.sum(skipna=True))


def _plot_pareto(pareto: pd.DataFrame, stem: Path) -> None:
    if pareto.empty:
        return
    import matplotlib.pyplot as plt

    grouped = pareto.groupby("method", as_index=False).agg({
        "total_time_ms": "mean",
        "mean_approx_loss": "mean",
        "mean_relevant_state_width": "mean",
    })
    y_col = (
        "mean_relevant_state_width"
        if grouped["mean_approx_loss"].isna().all() else "mean_approx_loss"
    )
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.scatter(grouped["total_time_ms"], grouped[y_col])
    for row in grouped.itertuples(index=False):
        ax.annotate(row.method, (row.total_time_ms, getattr(row, y_col)), fontsize=8)
    ax.set_xlabel("Runtime [ms]")
    ax.set_ylabel(
        "Mean relevant state width" if y_col == "mean_relevant_state_width"
        else "Mean approximation loss"
    )
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=160)
    plt.close(fig)


def _plot_public_range(timeseries: pd.DataFrame, stream: str, stem: Path) -> None:
    lower_col = f"{stream}_lower"
    upper_col = f"{stream}_upper"
    if lower_col not in timeseries or upper_col not in timeseries:
        return
    import matplotlib.pyplot as plt

    frame = timeseries[timeseries["seed"] == timeseries["seed"].min()]
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    for method, method_frame in frame.groupby("method"):
        ordered = method_frame.sort_values("step")
        x = ordered["step"].to_numpy(dtype=np.float64)
        lo = ordered[lower_col].to_numpy(dtype=np.float64)
        hi = ordered[upper_col].to_numpy(dtype=np.float64)
        ax.plot(x, lo, linewidth=1.0, label=method)
        ax.plot(x, hi, linewidth=1.0)
        ax.fill_between(x, lo, hi, alpha=0.12)
    ax.set_xlabel("Step")
    ax.set_ylabel(stream)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=160)
    plt.close(fig)
