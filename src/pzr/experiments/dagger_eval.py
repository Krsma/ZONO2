"""DAgger training and evaluation integrated with the benchmark pipeline.

Collects MPC expert traces, trains a learned policy via DAgger, and evaluates
it alongside static and MPC baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from tqdm.auto import tqdm

from pzr.experiments.runner import (
    MPCReductionPolicy,
    ReductionPolicy,
    RunResult,
    StaticReductionPolicy,
    compute_ground_truth,
    run_single,
)
from pzr.imitation.dataset import build_dataset
from pzr.imitation.features import extract_features
from pzr.imitation.policy import LearnedPolicy, TrainingResult, train_policy
from pzr.imitation.traces import ReductionTrace, TraceCollector
from pzr.monitoring.base import MonitorAdapter, MonitorState
from pzr.mpc.objectives import CostWeights, WeightedZonotopeCost
from pzr.mpc.policies import ReductionDecision, RolloutMPCPolicy
from pzr.utils.timing import timed
from pzr.zonotope.protected import ProtectedReducer
from pzr.zonotope.reduction import ALL_REDUCERS, BoxReducer, GirardReducer, Reducer


@dataclass
class LearnedReductionPolicy:
    """Wraps a learned policy as a ReductionPolicy."""

    learned: LearnedPolicy
    candidates: dict[str, Reducer | ProtectedReducer]
    _name: str = "learned"

    @property
    def name(self) -> str:
        return self._name

    def decide(
        self,
        monitor: MonitorAdapter,
        state: MonitorState,
        history: Sequence,
        budget: int,
    ) -> ReductionDecision:
        features = extract_features(
            state, budget, monitor.triggers,
            trigger_zonotope=monitor.trigger_zonotope,
        )
        cal = state.calibration_indices
        result = self.learned.select_reducer(
            features, self.candidates, state.zonotope, budget,
            protected_indices=cal,
        )
        if result is None:
            fallback = ProtectedReducer(base=BoxReducer())
            red = fallback.reduce(state.zonotope, budget, protected_indices=cal)
            new_cal = tuple(range(len(cal)))
            return ReductionDecision(
                state=state.with_zonotope(red.reduced, calibration_indices=new_cal),
                result=red,
                reducer_name="box_fallback",
            )
        name, red_result = result
        new_cal = tuple(range(len(cal)))
        return ReductionDecision(
            state=state.with_zonotope(red_result.reduced, calibration_indices=new_cal),
            result=red_result,
            reducer_name=name,
        )


@dataclass
class DAggerEvalResult:
    policy: LearnedPolicy
    training_results: list[TrainingResult]
    total_traces: int
    eval_results: list[RunResult]
    inference_time_ms: float


def collect_expert_traces(
    monitor: MonitorAdapter,
    trace_fn: Callable[[int, int], Sequence],
    expert_policy: ReductionPolicy,
    budget: int,
    seeds: range,
    length: int,
) -> TraceCollector:
    """Run the MPC expert and collect reduction traces."""
    collector = TraceCollector()
    for seed in seeds:
        trace = trace_fn(length, seed)
        run_single(monitor, trace, expert_policy, budget, seed, trace_collector=collector)
    return collector


def train_and_evaluate_dagger(
    monitor: MonitorAdapter,
    trace_fn: Callable[[int, int], Sequence],
    expert_policy: ReductionPolicy,
    budget: int,
    train_seeds: range,
    eval_seeds: range,
    length: int,
    dagger_iterations: int = 3,
    epochs_per_iteration: int = 100,
    hidden_sizes: tuple[int, ...] = (64, 64),
    seed: int = 42,
    candidate_names: tuple[str, ...] | None = None,
    show_progress: bool = True,
) -> DAggerEvalResult:
    """Full DAgger pipeline: collect → train → evaluate."""
    all_collectors: list[TraceCollector] = []
    policy: LearnedPolicy | None = None
    training_results: list[TrainingResult] = []

    iter_bar = tqdm(
        range(dagger_iterations), desc="dagger iters",
        disable=not show_progress, unit="iter", leave=True,
    )
    for iteration in iter_bar:
        collector = TraceCollector()

        train_seed_iter = tqdm(
            list(train_seeds), desc=f"iter {iteration} · collect",
            disable=not show_progress, unit="seed", leave=False,
        )
        for ep_seed in train_seed_iter:
            trace = trace_fn(length, ep_seed + iteration * 1000)
            if policy is None:
                run_single(monitor, trace, expert_policy, budget, ep_seed, trace_collector=collector)
            else:
                candidates = _candidate_reducers(candidate_names)
                learned_policy = LearnedReductionPolicy(policy, candidates, _name="dagger_learner")
                state = monitor.initial_state()
                history: list = []
                for i, measurement in enumerate(trace):
                    result = monitor.step(state, measurement)
                    state = result.state
                    history.append(measurement)
                    if state.zonotope.generator_count > budget:
                        expert_decision = expert_policy.decide(monitor, state, history, budget)
                        features = extract_features(
                            state, budget, monitor.triggers,
                            trigger_zonotope=monitor.trigger_zonotope,
                        )
                        collector.record(ReductionTrace(
                            features=features,
                            action=expert_decision.reducer_name,
                            cost=expert_decision.predicted_cost,
                            step=i,
                            episode_id=ep_seed,
                        ))
                        learner_decision = learned_policy.decide(monitor, state, history, budget)
                        state = learner_decision.state

        all_collectors.append(collector)

        combined = TraceCollector()
        for c in all_collectors:
            for t in c.traces:
                combined.record(t)

        if len(combined) == 0:
            continue
        dataset = build_dataset(combined)
        if dataset.num_classes < 2:
            continue

        policy, result = train_policy(
            dataset, hidden_sizes=hidden_sizes,
            epochs=epochs_per_iteration, seed=seed + iteration,
            show_progress=show_progress,
        )
        training_results.append(result)

    if policy is None:
        raise ValueError("DAgger produced no policy")

    candidates = _candidate_reducers(candidate_names)
    learned_pol = LearnedReductionPolicy(policy, candidates, _name="learned_dagger")

    eval_results: list[RunResult] = []
    inference_times: list[float] = []
    eval_iter = tqdm(
        list(eval_seeds), desc="dagger eval",
        disable=not show_progress, unit="seed", leave=False,
    )
    for ep_seed in eval_iter:
        trace = trace_fn(length, ep_seed)
        gt = compute_ground_truth(monitor, trace)
        r = run_single(monitor, trace, learned_pol, budget, ep_seed, ground_truth=gt)
        eval_results.append(r)
        if r.total_reductions > 0:
            inference_times.append(r.total_time_ms / r.total_reductions)

    total_traces = sum(len(c) for c in all_collectors)
    avg_inference = float(np.mean(inference_times)) if inference_times else 0.0

    return DAggerEvalResult(
        policy=policy,
        training_results=training_results,
        total_traces=total_traces,
        eval_results=eval_results,
        inference_time_ms=avg_inference,
    )


def _candidate_reducers(
    candidate_names: tuple[str, ...] | None,
) -> dict[str, Reducer | ProtectedReducer]:
    names = candidate_names or tuple(name for name in ALL_REDUCERS if name != "identity")
    return {
        name: ProtectedReducer(base=ALL_REDUCERS[name])
        for name in names
        if name != "identity"
    }
