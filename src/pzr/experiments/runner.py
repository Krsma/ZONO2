"""Experiment runner: orchestrate baseline and MPC benchmark runs.

Runs a monitor scenario with multiple reduction methods across multiple
seeds, collecting per-step metrics and reduction decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence, TypeVar

import numpy as np
import pandas as pd

from pzr.imitation.features import extract_features
from pzr.imitation.traces import ReductionTrace, TraceCollector
from pzr.monitoring.base import MonitorAdapter, MonitorState
from pzr.monitoring.triggers import evaluate_triggers
from pzr.mpc.objectives import CostWeights, WeightedZonotopeCost
from pzr.mpc.policies import (
    BeamMPCPolicy,
    MPCPolicy,
    PairRolloutMPCPolicy,
    ReductionDecision,
    RolloutMPCPolicy,
)
from pzr.mpc.prediction import ConstantPredictor
from pzr.utils.timing import timed
from pzr.zonotope.protected import ProtectedReducer, reduce_with_protection
from pzr.zonotope.reduction import Reducer, ReductionResult, _cert

InputT = TypeVar("InputT")


class ReductionPolicy(Protocol):
    """Uniform interface for all reduction strategies."""

    name: str

    def decide(
        self,
        monitor: MonitorAdapter,
        state: MonitorState,
        history: Sequence,
        budget: int,
    ) -> ReductionDecision: ...


@dataclass(frozen=True)
class StaticReductionPolicy:
    """Wrap a certified reducer (optionally with protection) as a policy."""

    reducer: Reducer | ProtectedReducer
    _name: str = ""

    @property
    def name(self) -> str:
        return self._name or getattr(self.reducer, "name", "static")

    def decide(
        self,
        monitor: MonitorAdapter,
        state: MonitorState,
        history: Sequence,
        budget: int,
    ) -> ReductionDecision:
        cal = state.calibration_indices
        result = reduce_with_protection(
            self.reducer, state.zonotope, budget,
            protected_indices=cal,
        )
        new_cal = tuple(range(len(cal))) if cal else ()
        return ReductionDecision(
            state=state.with_zonotope(result.reduced, calibration_indices=new_cal),
            result=result,
            reducer_name=self.name,
        )


@dataclass(frozen=True)
class MPCReductionPolicy:
    """Wrap an MPC or RolloutMPC policy with a predictor."""

    policy: MPCPolicy | BeamMPCPolicy | RolloutMPCPolicy | PairRolloutMPCPolicy
    _name: str = ""
    horizon: int = 4

    @property
    def name(self) -> str:
        return self._name or "mpc"

    def decide(
        self,
        monitor: MonitorAdapter,
        state: MonitorState,
        history: Sequence,
        budget: int,
    ) -> ReductionDecision:
        predictor = ConstantPredictor()
        predicted = predictor.predict(history, self.horizon)
        decision = self.policy.select(monitor, state, predicted)
        return ReductionDecision(
            state=decision.state,
            result=decision.result,
            reducer_name=decision.reducer_name,
            predicted_cost=decision.predicted_cost,
            predicted_sequence=decision.predicted_sequence,
            evaluated_leaves=decision.evaluated_leaves,
            pruned_branches=decision.pruned_branches,
        )


@dataclass(frozen=True)
class GroundTruth:
    """Per-step exact (unreduced) trigger-zonotope bounds and verdicts."""

    lower: np.ndarray
    upper: np.ndarray
    width_sum: float
    verdicts: dict[str, str]


@dataclass
class StepRecord:
    seed: int
    method: str
    step: int
    generator_count: int
    reduced: bool
    reducer_used: str
    trigger_width_sum: float
    verdicts: dict[str, str]
    reduction_time_ms: float
    approx_error_sum: float = 0.0
    false_positive: bool = False
    exact_trigger_width_sum: float = 0.0


@dataclass
class RunResult:
    method: str
    seed: int
    steps: list[StepRecord]
    total_reductions: int = 0
    total_time_ms: float = 0.0
    budget_violations: int = 0
    unsound_certificates: int = 0


def _trigger_metrics(
    monitor: MonitorAdapter,
    state: MonitorState,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, str]]:
    """Return (lower, upper, width_sum, verdict_dict) for the trigger zonotope."""
    tz = monitor.trigger_zonotope(state)
    lower, upper = tz.interval_bounds()
    verdicts = evaluate_triggers(tz, monitor.triggers)
    verdict_dict = {v.trigger.name: v.status for v in verdicts}
    width_sum = sum(
        float(upper[t.state_index] - lower[t.state_index]) for t in monitor.triggers
    )
    return lower, upper, width_sum, verdict_dict


def compute_ground_truth(
    monitor: MonitorAdapter,
    trace: Sequence,
) -> list[GroundTruth]:
    """Run the monitor with no reduction; record exact trigger-zonotope bounds per step."""
    state = monitor.initial_state()
    out: list[GroundTruth] = []
    for measurement in trace:
        result = monitor.step(state, measurement)
        state = result.state
        lower, upper, width_sum, verdict_dict = _trigger_metrics(monitor, state)
        out.append(GroundTruth(
            lower=lower, upper=upper, width_sum=width_sum, verdicts=verdict_dict,
        ))
    return out


def run_single(
    monitor: MonitorAdapter,
    trace: Sequence,
    policy: ReductionPolicy,
    budget: int,
    seed: int,
    trace_collector: TraceCollector | None = None,
    ground_truth: Sequence[GroundTruth] | None = None,
) -> RunResult:
    """Run a single policy on a single trace."""
    state = monitor.initial_state()
    steps: list[StepRecord] = []
    total_reductions = 0
    total_time_ms = 0.0
    budget_violations = 0
    unsound_certs = 0
    history: list = []

    for i, measurement in enumerate(trace):
        result = monitor.step(state, measurement)
        state = result.state
        history.append(measurement)

        reduced = False
        reducer_used = ""
        reduction_time = 0.0

        if state.zonotope.generator_count > budget:
            decision_features = None
            if trace_collector is not None:
                decision_features = extract_features(
                    state, budget, monitor.triggers,
                    trigger_zonotope=monitor.trigger_zonotope,
                )
            with timed() as t:
                decision = policy.decide(monitor, state, history, budget)

            reduction_time = t.elapsed_ms
            state = decision.state
            reduced = True
            reducer_used = decision.reducer_name
            total_reductions += 1
            total_time_ms += reduction_time

            if decision.result.reduced.generator_count > budget:
                budget_violations += 1
            if not decision.result.certificate.is_sound:
                unsound_certs += 1

            if trace_collector is not None:
                trace_collector.record(ReductionTrace(
                    features=decision_features,
                    action=reducer_used,
                    cost=decision.predicted_cost,
                    step=i,
                    episode_id=seed,
                ))

        lower, upper, width_sum, verdict_dict = _trigger_metrics(monitor, state)

        approx_error_sum = 0.0
        false_positive = False
        exact_width_sum = 0.0
        if ground_truth is not None:
            gt = ground_truth[i]
            approx_error_sum = float(sum(
                abs(lower[t.state_index] - gt.lower[t.state_index])
                + abs(upper[t.state_index] - gt.upper[t.state_index])
                for t in monitor.triggers
            ))
            exact_width_sum = gt.width_sum
            for trigger in monitor.triggers:
                if verdict_dict[trigger.name] == "violation" and gt.verdicts[trigger.name] == "safe":
                    false_positive = True
                    break

        steps.append(StepRecord(
            seed=seed,
            method=policy.name,
            step=i,
            generator_count=state.zonotope.generator_count,
            reduced=reduced,
            reducer_used=reducer_used,
            trigger_width_sum=width_sum,
            verdicts=verdict_dict,
            reduction_time_ms=reduction_time,
            approx_error_sum=approx_error_sum,
            false_positive=false_positive,
            exact_trigger_width_sum=exact_width_sum,
        ))

    return RunResult(
        method=policy.name,
        seed=seed,
        steps=steps,
        total_reductions=total_reductions,
        total_time_ms=total_time_ms,
        budget_violations=budget_violations,
        unsound_certificates=unsound_certs,
    )


def results_to_dataframe(results: list[RunResult]) -> pd.DataFrame:
    """Convert run results to a pandas DataFrame."""
    rows = []
    for r in results:
        for s in r.steps:
            row = {
                "seed": s.seed,
                "method": s.method,
                "step": s.step,
                "generator_count": s.generator_count,
                "reduced": s.reduced,
                "reducer_used": s.reducer_used,
                "trigger_width_sum": s.trigger_width_sum,
                "exact_trigger_width_sum": s.exact_trigger_width_sum,
                "approx_error_sum": s.approx_error_sum,
                "false_positive": s.false_positive,
                "reduction_time_ms": s.reduction_time_ms,
            }
            row.update(s.verdicts)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_results(results: list[RunResult]) -> pd.DataFrame:
    """Aggregate metrics per method per seed."""
    rows = []
    for r in results:
        trigger_widths = [s.trigger_width_sum for s in r.steps]
        gen_counts = [s.generator_count for s in r.steps]
        approx_errors = [s.approx_error_sum for s in r.steps]
        fps = [s.false_positive for s in r.steps]
        rows.append({
            "method": r.method,
            "seed": r.seed,
            "mean_trigger_width": float(np.mean(trigger_widths)),
            "max_trigger_width": float(np.max(trigger_widths)),
            "mean_generator_count": float(np.mean(gen_counts)),
            "max_generator_count": int(np.max(gen_counts)),
            "total_reductions": r.total_reductions,
            "total_time_ms": r.total_time_ms,
            "budget_violations": r.budget_violations,
            "unsound_certificates": r.unsound_certificates,
            "mean_approx_error": float(np.mean(approx_errors)),
            "max_approx_error": float(np.max(approx_errors)),
            "abs_error_range": float(np.max(approx_errors) - np.min(approx_errors)),
            "false_positive_rate": float(np.mean(fps)),
        })
    return pd.DataFrame(rows)
