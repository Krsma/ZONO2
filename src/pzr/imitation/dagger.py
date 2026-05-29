"""DAgger: Dataset Aggregation for imitation learning.

Iteratively:
1. Roll out the learned policy on training episodes
2. At each reduction point, query the MPC expert for its action
3. Aggregate new (features, expert_action) pairs with previous data
4. Retrain the policy on accumulated data
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

import numpy as np
from numpy.typing import NDArray

from pzr.imitation.dataset import ReductionDataset, build_dataset
from pzr.imitation.features import extract_features
from pzr.imitation.policy import LearnedPolicy, TrainingResult, train_policy
from pzr.imitation.traces import ReductionTrace, TraceCollector
from pzr.monitoring.base import MonitorAdapter, MonitorState
from pzr.zonotope.reduction import Reducer

InputT = TypeVar("InputT")


@dataclass
class DAggerResult:
    policy: LearnedPolicy
    training_results: list[TrainingResult]
    total_traces: int
    iterations: int


ExpertQueryFn = Callable[[MonitorAdapter, MonitorState, int], tuple[str, float]]


def run_dagger(
    monitor: MonitorAdapter,
    traces_per_episode: Callable[[MonitorAdapter, LearnedPolicy | None, int, int, TraceCollector, ExpertQueryFn], None],
    expert_query: ExpertQueryFn,
    budget: int,
    num_episodes: int,
    num_iterations: int = 3,
    epochs_per_iteration: int = 100,
    hidden_sizes: tuple[int, ...] = (64, 64),
    seed: int = 42,
) -> DAggerResult:
    """Run the DAgger loop.

    Args:
        monitor: The monitor adapter.
        traces_per_episode: Function that runs one episode, collecting traces.
            Signature: (monitor, policy_or_None, episode_id, budget, collector, expert_query)
        expert_query: Function that queries the MPC expert for an action.
            Signature: (monitor, state, budget) -> (action_name, cost)
        budget: Generator budget.
        num_episodes: Episodes per iteration.
        num_iterations: Number of DAgger rounds.
        epochs_per_iteration: Training epochs per round.
        hidden_sizes: MLP architecture.
        seed: Random seed.
    """
    all_collectors: list[TraceCollector] = []
    policy: LearnedPolicy | None = None
    training_results: list[TrainingResult] = []

    for iteration in range(num_iterations):
        collector = TraceCollector()

        for ep in range(num_episodes):
            traces_per_episode(
                monitor, policy, ep + iteration * num_episodes,
                budget, collector, expert_query,
            )

        all_collectors.append(collector)

        # Aggregate all traces
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
            dataset,
            hidden_sizes=hidden_sizes,
            epochs=epochs_per_iteration,
            seed=seed + iteration,
        )
        training_results.append(result)

    total = sum(len(c) for c in all_collectors)
    if policy is None:
        raise ValueError("DAgger produced no policy — insufficient training data")

    return DAggerResult(
        policy=policy,
        training_results=training_results,
        total_traces=total,
        iterations=num_iterations,
    )
