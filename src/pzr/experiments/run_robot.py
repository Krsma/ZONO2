"""Small reproducible robot experiment used by tests and examples."""

from __future__ import annotations

from dataclasses import dataclass

from pzr.benchmarks.robot import (
    OmnidirectionalRobotMonitor,
    generate_robot_trace,
    predict_robot_inputs,
)
from pzr.control.costs import CostWeights, WeightedZonotopeCost
from pzr.control.policies import MPCPolicy, StaticReductionPolicy
from pzr.monitoring.base import evaluate_triggers
from pzr.reduction.reducers import BoxReducer, ScoredKeepReducer


@dataclass(frozen=True)
class RobotRunMetrics:
    """Compact metrics for comparing reduction policies."""

    steps: int
    max_generators: int
    max_trigger_width: float
    inconclusive_verdicts: int
    reductions: int
    final_generators: int


def _run_static(length: int, budget: int, seed: int) -> RobotRunMetrics:
    monitor = OmnidirectionalRobotMonitor()
    policy = StaticReductionPolicy(ScoredKeepReducer.calibration_aware(), budget)
    state = monitor.initial_state()
    trace = generate_robot_trace(length, seed=seed)
    reductions = 0
    max_generators = 0
    max_trigger_width = 0.0
    inconclusive = 0

    for measurement in trace:
        result = monitor.step(state, measurement)
        state = result.state
        if state.zonotope.generator_count > budget:
            state = policy.reduce_state(monitor, state).state
            reductions += 1
        verdicts = evaluate_triggers(state.zonotope, monitor.triggers)
        max_generators = max(max_generators, state.zonotope.generator_count)
        for verdict in verdicts:
            max_trigger_width = max(max_trigger_width, verdict.upper - verdict.lower)
            inconclusive += int(verdict.status == "inconclusive")

    return RobotRunMetrics(
        steps=length,
        max_generators=max_generators,
        max_trigger_width=max_trigger_width,
        inconclusive_verdicts=inconclusive,
        reductions=reductions,
        final_generators=state.zonotope.generator_count,
    )


def run_robot_experiment(
    *,
    length: int = 40,
    budget: int = 8,
    horizon: int = 4,
    seed: int = 0,
) -> dict[str, RobotRunMetrics]:
    """Run static and MPC-guided reducers on the same generated trace."""

    monitor = OmnidirectionalRobotMonitor()
    trace = generate_robot_trace(length, seed=seed)
    cost = WeightedZonotopeCost(
        CostWeights(trigger_width=1.0, straddling=20.0, generator_count=0.01),
        triggers=monitor.triggers,
    )
    mpc_policy = MPCPolicy(
        reducers=(
            ScoredKeepReducer.by_norm(),
            ScoredKeepReducer.calibration_aware(),
            BoxReducer(),
        ),
        budget=budget,
        horizon=horizon,
        cost=cost,
    )

    state = monitor.initial_state()
    reductions = 0
    max_generators = 0
    max_trigger_width = 0.0
    inconclusive = 0
    history = []

    for index, measurement in enumerate(trace):
        history.append(measurement)
        result = monitor.step(state, measurement)
        state = result.state
        if state.zonotope.generator_count > budget:
            observed_future = trace[index + 1 : index + 1 + horizon]
            predicted = observed_future or predict_robot_inputs(history, horizon)
            state = mpc_policy.reduce_state(monitor, state, predicted).state
            reductions += 1
        verdicts = evaluate_triggers(state.zonotope, monitor.triggers)
        max_generators = max(max_generators, state.zonotope.generator_count)
        for verdict in verdicts:
            max_trigger_width = max(max_trigger_width, verdict.upper - verdict.lower)
            inconclusive += int(verdict.status == "inconclusive")

    mpc_metrics = RobotRunMetrics(
        steps=length,
        max_generators=max_generators,
        max_trigger_width=max_trigger_width,
        inconclusive_verdicts=inconclusive,
        reductions=reductions,
        final_generators=state.zonotope.generator_count,
    )
    return {
        "static_calibration_aware": _run_static(length, budget, seed),
        "mpc": mpc_metrics,
    }
