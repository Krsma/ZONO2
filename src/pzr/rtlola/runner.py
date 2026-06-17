"""Benchmark runner for RTLola-native monitors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml

from pzr.experiments.evaluation import aggregate_summary
from pzr.rtlola.binding import require_binding
from pzr.rtlola.actions import RtlolaAction, action_by_name, default_actions
from pzr.rtlola.engine import RtlolaEngine, RtlolaEvent
from pzr.rtlola.omni import OMNI_EXPECTED_VERDICT_KEYS, OMNI_SPEC, generate_omni_events
from pzr.rtlola.search import beam_search, choose_static_action


STATIC_METHODS = ("none", "girard", "scott", "interval_hull", "colinear_scale", "colinear")
MPC_METHODS = ("mpc_beam",)
RTLOLA_AGGREGATE_METRICS = [
    "mean_state_zonotope_width",
    "max_state_zonotope_width",
    "mean_generator_count",
    "total_reductions",
    "total_time_ms",
    "mean_state_zonotope_approx_error",
    "max_state_zonotope_approx_error",
    "state_zonotope_abs_error_range",
    "false_positive_rate",
]


@dataclass(frozen=True)
class RtlolaBenchmarkConfig:
    scenario: str = "omni_robot"
    length: int = 30
    budget: int = 10
    horizon: int = 2
    beam_width: int = 4
    seeds: int = 3
    method_set: str = "all"
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
    generator_count: int
    total_generator_count: int
    reduced: bool
    reducer_used: str
    state_zonotope_width_sum: float
    exact_state_zonotope_width_sum: float
    state_zonotope_approx_error_sum: float
    false_positive: bool
    verdicts: dict[str, object]
    reduction_time_ms: float
    predicted_cost: float = 0.0
    predicted_sequence: tuple[str, ...] = ()
    evaluated_leaves: int = 0
    pruned_branches: int = 0
    budget_violation: bool = False


@dataclass(frozen=True)
class RtlolaGroundTruthStep:
    """Unreduced RTLola state-zonotope bounds and public verdicts."""

    lower: np.ndarray
    upper: np.ndarray
    width_sum: float
    verdicts: dict[str, object]


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
    if config.method_set == "static":
        return STATIC_METHODS
    if config.method_set == "mpc":
        return MPC_METHODS
    if config.method_set == "all":
        return (*STATIC_METHODS, *MPC_METHODS)
    raise ValueError("method_set must be one of: static, mpc, all")


def run_benchmark(config: RtlolaBenchmarkConfig) -> RtlolaBenchmarkResult:
    if config.scenario != "omni_robot":
        raise ValueError("only scenario='omni_robot' is supported in RTLola v1")
    actions = default_actions()
    by_name = action_by_name(actions)
    fallback = by_name["interval"]
    raw: list[RtlolaRunResult] = []
    for seed in range(config.seeds):
        trace = generate_omni_events(config.length, seed=seed)
        ground_truth = compute_ground_truth(trace)
        for method in methods_for_config(config):
            raw.append(_run_single(
                config, trace, method, actions, by_name, fallback, seed, ground_truth,
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
    trace: Sequence[RtlolaEvent],
    method: str,
    actions: tuple[RtlolaAction, ...],
    by_name: dict[str, RtlolaAction],
    fallback: RtlolaAction,
    seed: int,
    ground_truth: Sequence[RtlolaGroundTruthStep],
) -> RtlolaRunResult:
    engine = RtlolaEngine(
        OMNI_SPEC,
        event_arity=3,
        expected_verdict_keys=OMNI_EXPECTED_VERDICT_KEYS,
    )
    steps: list[RtlolaStepRecord] = []
    for index, event in enumerate(trace):
        state = engine.snapshot(step=index, time=event.time)
        future = tuple(trace[index + 1:index + 1 + config.horizon])
        if method == "mpc_beam":
            decision = beam_search(
                engine,
                state,
                event,
                future,
                actions,
                config.budget,
                config.beam_width,
                fallback=fallback,
            )
        else:
            decision = choose_static_action(
                engine,
                state,
                event,
                by_name[method],
                config.budget,
                fallback=fallback,
            )

        committed = engine.live_step(
            event,
            decision.first_action,
            config.budget,
            step=index + 1,
        )
        gt = ground_truth[index]
        lower, upper = _state_interval_bounds(engine.matrices(committed.state)[0])
        if lower.shape != gt.lower.shape:
            raise RuntimeError(
                "RTLola reduced and exact state-zonotope dimensions differ "
                f"(method={method}, seed={seed}, step={index}, "
                f"reduced_dim={lower.shape[0]}, exact_dim={gt.lower.shape[0]})"
            )
        approx_error = float(np.sum(np.abs(lower - gt.lower) + np.abs(upper - gt.upper)))
        false_positive = _false_positive(committed.verdict, gt.verdicts)
        violation = committed.metrics.dynamic_generator_count > config.budget
        steps.append(RtlolaStepRecord(
            seed=seed,
            method=method,
            step=index,
            generator_count=committed.metrics.dynamic_generator_count,
            total_generator_count=committed.metrics.total_generator_count,
            reduced=decision.first_action.name != "none",
            reducer_used=decision.first_action.name,
            state_zonotope_width_sum=committed.metrics.full_width_sum,
            exact_state_zonotope_width_sum=gt.width_sum,
            state_zonotope_approx_error_sum=approx_error,
            false_positive=false_positive,
            verdicts=committed.verdict,
            reduction_time_ms=0.0,
            predicted_cost=decision.predicted_cost,
            predicted_sequence=decision.predicted_sequence,
            evaluated_leaves=decision.evaluated_leaves,
            pruned_branches=decision.pruned_branches,
            budget_violation=violation,
        ))
        if violation:
            raise RuntimeError(
                "committed RTLola step exceeded dynamic budget "
                f"(method={method}, seed={seed}, step={index}, "
                f"count={committed.metrics.dynamic_generator_count}, budget={config.budget})"
            )
    return RtlolaRunResult(method=method, seed=seed, steps=tuple(steps))


def compute_ground_truth(trace: Sequence[RtlolaEvent]) -> tuple[RtlolaGroundTruthStep, ...]:
    """Run the RTLola monitor without reductions for exact state-zonotope metrics."""
    _, RLolaMonitor, ZonotopeConfig = require_binding()
    monitor = RLolaMonitor(OMNI_SPEC)
    out: list[RtlolaGroundTruthStep] = []
    for step, event in enumerate(trace):
        verdict = monitor.accept_event(list(event.values), float(event.time), ZonotopeConfig.none())
        for key in OMNI_EXPECTED_VERDICT_KEYS:
            if key not in verdict:
                raise RuntimeError(f"RTLola ground truth verdict missing key at step {step}: {key}")
        zono = np.asarray(monitor.current_zonotope(False), dtype=np.float64)
        lower, upper = _state_interval_bounds(zono)
        out.append(RtlolaGroundTruthStep(
            lower=lower,
            upper=upper,
            width_sum=float(np.sum(upper - lower)),
            verdicts=dict(verdict),
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
                "generator_count": step.generator_count,
                "total_generator_count": step.total_generator_count,
                "reduced": step.reduced,
                "reducer_used": step.reducer_used,
                "state_zonotope_width_sum": step.state_zonotope_width_sum,
                "exact_state_zonotope_width_sum": step.exact_state_zonotope_width_sum,
                "state_zonotope_approx_error_sum": step.state_zonotope_approx_error_sum,
                "false_positive": step.false_positive,
                "reduction_time_ms": step.reduction_time_ms,
                "predicted_cost": step.predicted_cost,
                "predicted_sequence": ",".join(step.predicted_sequence),
                "evaluated_leaves": step.evaluated_leaves,
                "pruned_branches": step.pruned_branches,
                "budget_violation": step.budget_violation,
            }
            row.update(step.verdicts)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_results(results: Sequence[RtlolaRunResult]) -> pd.DataFrame:
    rows = []
    for run in results:
        widths = np.asarray([step.state_zonotope_width_sum for step in run.steps], dtype=np.float64)
        gens = np.asarray([step.generator_count for step in run.steps], dtype=np.float64)
        approx_errors = np.asarray(
            [step.state_zonotope_approx_error_sum for step in run.steps],
            dtype=np.float64,
        )
        fps = np.asarray([step.false_positive for step in run.steps], dtype=np.float64)
        rows.append({
            "method": run.method,
            "seed": run.seed,
            "mean_state_zonotope_width": float(np.mean(widths)),
            "max_state_zonotope_width": float(np.max(widths)),
            "mean_generator_count": float(np.mean(gens)),
            "max_generator_count": int(np.max(gens)),
            "total_reductions": run.total_reductions,
            "total_time_ms": run.total_time_ms,
            "budget_violations": run.budget_violations,
            "unsound_certificates": 0,
            "mean_state_zonotope_approx_error": float(np.mean(approx_errors)),
            "max_state_zonotope_approx_error": float(np.max(approx_errors)),
            "state_zonotope_abs_error_range": float(np.max(approx_errors) - np.min(approx_errors)),
            "false_positive_rate": float(np.mean(fps)),
        })
    return pd.DataFrame(rows)


def _state_interval_bounds(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(matrix, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] < 1:
        raise ValueError(f"expected 2D state-zonotope matrix, got {z.shape}")
    center = z[:, 0]
    radius = np.abs(z[:, 1:]).sum(axis=1) if z.shape[1] > 1 else np.zeros(z.shape[0])
    return center - radius, center + radius


def _false_positive(reduced_verdict: dict[str, object], exact_verdict: dict[str, object]) -> bool:
    for key in OMNI_EXPECTED_VERDICT_KEYS:
        if bool(reduced_verdict.get(key, False)) and not bool(exact_verdict.get(key, False)):
            return True
    return False


def save_benchmark_results(result: RtlolaBenchmarkResult, output_dir: Path) -> None:
    scenario_dir = output_dir / result.config.scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    result.timeseries.to_csv(scenario_dir / "timeseries.csv", index=False)
    result.summary.to_csv(scenario_dir / "summary.csv", index=False)
    result.aggregate.to_csv(scenario_dir / "aggregate.csv", index=False)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(asdict(result.config), f, sort_keys=False)
