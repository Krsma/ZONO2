"""RTLola teacher collection and ranking-dataset assembly."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from pzr.learning.artifacts import write_ranking_dataset
from pzr.learning.dataset import RankingDataset
from pzr.learning.targets import tolerant_best_mask
from pzr.rtlola.actions import default_action_catalog
from pzr.rtlola.engine import RtlolaEngine, RtlolaEvent, RtlolaStateRef
from pzr.rtlola.features import (
    RTL_RANKING_FEATURE_NAMES,
    extract_ranking_features,
)
from pzr.rtlola.scenarios import RtlolaScenario
from pzr.rtlola.search import RtlolaSearchResult, full_width_terminal_search


class BehaviorPolicy(Protocol):
    """Minimal direct-policy interface used for state aggregation."""

    def choose(
        self,
        engine: RtlolaEngine,
        state: RtlolaStateRef,
        event: RtlolaEvent,
        budget: int,
    ) -> RtlolaSearchResult: ...


@dataclass(frozen=True)
class CollectedRankingSample:
    sample_id: str
    trace_id: str
    split: str
    condition: str
    seed: int
    budget: int
    step: int
    features: np.ndarray
    candidate_names: tuple[str, ...]
    teacher_costs: tuple[float, ...]
    feasible: tuple[bool, ...]
    tie_mask: tuple[bool, ...]
    teacher_action: str
    teacher_sequence: tuple[str, ...]
    behavior: str
    behavior_action: str
    evaluated_leaves: int
    teacher_reducer_failure_count: int
    teacher_infeasible_candidate_count: int
    behavior_reducer_failure_count: int
    behavior_infeasible_candidate_count: int
    behavior_fallback_used: bool


def collect_teacher_episode(
    *,
    scenario: RtlolaScenario,
    events: Sequence[RtlolaEvent],
    trace_id: str,
    split: str,
    condition: str,
    seed: int,
    budget: int,
    candidate_names: tuple[str, ...],
    behavior_policy: BehaviorPolicy | None = None,
) -> tuple[CollectedRankingSample, ...]:
    """Label every over-bound state while following teacher or learned behavior."""
    if len(events) < 2:
        raise ValueError("two-event teacher collection requires at least two events")
    catalog = default_action_catalog(candidate_names)
    engine = RtlolaEngine(
        scenario.spec,
        event_arity=scenario.event_arity,
        expected_verdict_keys=scenario.expected_verdict_keys,
    )
    samples = []
    for step, event in enumerate(events[:-1]):
        state = engine.snapshot(step=step, time=event.time)
        metrics = engine.metrics(state)
        if metrics.dynamic_generator_count <= budget:
            action = catalog.no_op
        else:
            decision = full_width_terminal_search(
                engine,
                state,
                event,
                events[step + 1],
                catalog.mpc_candidates,
                budget,
                fallback=catalog.fallback,
                none_action=catalog.no_op,
            )
            costs, feasible = _aligned_root_costs(
                decision.root_evaluations, candidate_names,
            )
            tie_mask = tolerant_best_mask(costs, feasible)
            behavior_name = "teacher" if behavior_policy is None else "learned"
            sample_id = (
                f"{trace_id}:{behavior_name}:budget-{budget}:step-{step}"
            )
            behavior_decision = (
                decision
                if behavior_policy is None
                else behavior_policy.choose(engine, state, event, budget)
            )
            samples.append(CollectedRankingSample(
                sample_id=sample_id,
                trace_id=trace_id,
                split=split,
                condition=condition,
                seed=seed,
                budget=budget,
                step=step,
                features=extract_ranking_features(engine, state, budget),
                candidate_names=candidate_names,
                teacher_costs=tuple(float(value) for value in costs),
                feasible=tuple(bool(value) for value in feasible),
                tie_mask=tuple(bool(value) for value in tie_mask),
                teacher_action=decision.first_action.name,
                teacher_sequence=decision.predicted_sequence,
                behavior=behavior_name,
                behavior_action=behavior_decision.first_action.name,
                evaluated_leaves=decision.evaluated_leaves,
                teacher_reducer_failure_count=decision.reducer_failure_count,
                teacher_infeasible_candidate_count=decision.infeasible_candidate_count,
                behavior_reducer_failure_count=behavior_decision.reducer_failure_count,
                behavior_infeasible_candidate_count=(
                    behavior_decision.infeasible_candidate_count
                ),
                behavior_fallback_used=behavior_decision.fallback_used,
            ))
            action = behavior_decision.first_action
        engine.live_step(event, action, budget, step=step + 1)
    return tuple(samples)


def build_ranking_dataset(
    samples: Sequence[CollectedRankingSample],
    *,
    candidate_names: tuple[str, ...] | None = None,
) -> tuple[RankingDataset, pd.DataFrame]:
    if not samples:
        if candidate_names is None:
            raise ValueError("empty collection requires an explicit candidate catalog")
        return _empty_ranking_dataset(candidate_names)
    sample_candidate_names = samples[0].candidate_names
    if candidate_names is not None and candidate_names != sample_candidate_names:
        raise ValueError("explicit candidate catalog differs from collected samples")
    candidate_names = sample_candidate_names
    for sample in samples:
        if sample.candidate_names != candidate_names:
            raise ValueError("collected samples use different candidate catalogs")
    dataset = RankingDataset(
        features=np.stack([sample.features for sample in samples]).astype(np.float32),
        teacher_costs=np.asarray(
            [sample.teacher_costs for sample in samples], dtype=np.float64,
        ),
        feasible=np.asarray(
            [sample.feasible for sample in samples], dtype=np.bool_,
        ),
        tie_mask=np.asarray(
            [sample.tie_mask for sample in samples], dtype=np.bool_,
        ),
        candidate_names=candidate_names,
        feature_names=RTL_RANKING_FEATURE_NAMES,
        splits=tuple(sample.split for sample in samples),
        sample_ids=tuple(sample.sample_id for sample in samples),
    )
    metadata = pd.DataFrame([
        {
            "sample_id": sample.sample_id,
            "trace_id": sample.trace_id,
            "split": sample.split,
            "condition": sample.condition,
            "seed": sample.seed,
            "budget": sample.budget,
            "step": sample.step,
            "teacher_action": sample.teacher_action,
            "teacher_sequence": json.dumps(sample.teacher_sequence),
            "behavior": sample.behavior,
            "behavior_action": sample.behavior_action,
            "evaluated_leaves": sample.evaluated_leaves,
            "teacher_reducer_failure_count": sample.teacher_reducer_failure_count,
            "teacher_infeasible_candidate_count": (
                sample.teacher_infeasible_candidate_count
            ),
            "behavior_reducer_failure_count": sample.behavior_reducer_failure_count,
            "behavior_infeasible_candidate_count": (
                sample.behavior_infeasible_candidate_count
            ),
            "behavior_fallback_used": sample.behavior_fallback_used,
        }
        for sample in samples
    ])
    return dataset, metadata


def _empty_ranking_dataset(
    candidate_names: tuple[str, ...],
) -> tuple[RankingDataset, pd.DataFrame]:
    candidate_count = len(candidate_names)
    dataset = RankingDataset(
        features=np.empty((0, len(RTL_RANKING_FEATURE_NAMES)), dtype=np.float32),
        teacher_costs=np.empty((0, candidate_count), dtype=np.float64),
        feasible=np.empty((0, candidate_count), dtype=np.bool_),
        tie_mask=np.empty((0, candidate_count), dtype=np.bool_),
        candidate_names=candidate_names,
        feature_names=RTL_RANKING_FEATURE_NAMES,
        splits=(),
        sample_ids=(),
    )
    return dataset, pd.DataFrame(columns=(
        "sample_id", "trace_id", "split", "condition", "seed", "budget",
        "step", "teacher_action", "teacher_sequence", "behavior",
        "behavior_action", "evaluated_leaves",
        "teacher_reducer_failure_count", "teacher_infeasible_candidate_count",
        "behavior_reducer_failure_count", "behavior_infeasible_candidate_count",
        "behavior_fallback_used",
    ))


def write_collected_dataset(
    samples: Sequence[CollectedRankingSample],
    directory: Path,
    metadata: Mapping[str, object],
    *,
    candidate_names: tuple[str, ...] | None = None,
) -> RankingDataset:
    dataset, sample_metadata = build_ranking_dataset(
        samples, candidate_names=candidate_names,
    )
    write_ranking_dataset(dataset, directory, sample_metadata, metadata)
    return dataset


def _aligned_root_costs(
    root_evaluations: Sequence[object],
    candidate_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    by_name = {str(row.root_action): row for row in root_evaluations}
    if set(by_name) != set(candidate_names):
        raise ValueError("teacher root evaluations do not match candidate catalog")
    feasible = np.asarray([
        bool(by_name[name].feasible and by_name[name].complete)
        for name in candidate_names
    ])
    costs = np.asarray([
        float(by_name[name].predicted_cost) if feasible[index] else float("nan")
        for index, name in enumerate(candidate_names)
    ])
    return costs, feasible
