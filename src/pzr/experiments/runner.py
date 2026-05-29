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
from pzr.mpc.policies import MPCPolicy, ReductionDecision, RolloutMPCPolicy
from pzr.mpc.prediction import ConstantPredictor
from pzr.utils.timing import timed
from pzr.zonotope.protected import ProtectedReducer
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
        if isinstance(self.reducer, ProtectedReducer) and cal:
            result = self.reducer.reduce(state.zonotope, budget, protected_indices=cal)
        else:
            result = self.reducer.reduce(state.zonotope, budget)
        new_cal = tuple(range(len(cal))) if cal else ()
        return ReductionDecision(
            state=state.with_zonotope(result.reduced, calibration_indices=new_cal),
            result=result,
            reducer_name=self.name,
        )


@dataclass(frozen=True)
class MPCReductionPolicy:
    """Wrap an MPC or RolloutMPC policy with a predictor."""

    policy: MPCPolicy | RolloutMPCPolicy
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


@dataclass
class RunResult:
    method: str
    seed: int
    steps: list[StepRecord]
    total_reductions: int = 0
    total_time_ms: float = 0.0
    budget_violations: int = 0
    unsound_certificates: int = 0


def run_single(
    monitor: MonitorAdapter,
    trace: Sequence,
    policy: ReductionPolicy,
    budget: int,
    seed: int,
    trace_collector: TraceCollector | None = None,
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
                features = extract_features(state, budget, monitor.triggers)
                trace_collector.record(ReductionTrace(
                    features=features,
                    action=reducer_used,
                    cost=decision.predicted_cost,
                    step=i,
                    episode_id=seed,
                ))

        verdicts = evaluate_triggers(state.zonotope, monitor.triggers)
        verdict_dict = {v.trigger.name: v.status for v in verdicts}
        widths = state.zonotope.widths()
        trigger_width_sum = sum(
            float(widths[t.state_index]) for t in monitor.triggers
        )

        steps.append(StepRecord(
            seed=seed,
            method=policy.name,
            step=i,
            generator_count=state.zonotope.generator_count,
            reduced=reduced,
            reducer_used=reducer_used,
            trigger_width_sum=trigger_width_sum,
            verdicts=verdict_dict,
            reduction_time_ms=reduction_time,
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
        })
    return pd.DataFrame(rows)
