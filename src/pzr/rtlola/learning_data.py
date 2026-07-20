"""Teacher and one-step discrete-DART reducer-cost collection."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from pzr.learning.artifacts import write_reducer_cost_dataset
from pzr.learning.dart import DartCalibration
from pzr.learning.dataset import ReducerCostDataset
from pzr.learning.objectives import normalized_regrets
from pzr.rtlola.actions import default_action_catalog
from pzr.rtlola.engine import RtlolaEngine, RtlolaEvent
from pzr.rtlola.features import RTL_RANKING_FEATURE_NAMES, extract_ranking_features
from pzr.rtlola.scenarios import RtlolaScenario
from pzr.rtlola.search import full_width_terminal_search


CollectionMode = Literal["teacher", "dart"]


@dataclass(frozen=True)
class DartDecisionMetadata:
    """Optional disturbance state emitted only by guarded-DART collection."""

    executed_action: str
    disturbed: bool
    disturbance_eligible: bool
    disturbance_attempted: bool
    recovery_forced: bool
    target_disturbance_rate: float
    injection_probability: float
    disturbance_probability: float
    regret_cap: float
    selected_direction_probability: float
    sampled_normalized_regret: float
    calibration_sha256: str | None


@dataclass(frozen=True)
class CollectedReducerCostSample:
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
    teacher_action: str
    teacher_sequence: tuple[str, ...]
    evaluated_leaves: int
    teacher_reducer_failure_count: int
    teacher_infeasible_candidate_count: int
    execution_fallback_used: bool
    dart: DartDecisionMetadata | None = None


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
    collection_mode: CollectionMode = "teacher",
    dart_calibration: DartCalibration | None = None,
    dart_calibration_sha256: str | None = None,
    disturbance_seed: int = 0,
) -> tuple[CollectedReducerCostSample, ...]:
    """Label over-bound states and return control to the teacher after each action."""
    if len(events) < 2:
        raise ValueError("two-event teacher collection requires at least two events")
    if collection_mode not in ("teacher", "dart"):
        raise ValueError(f"unsupported collection mode: {collection_mode}")
    if collection_mode == "dart" and dart_calibration is None:
        raise ValueError("DART collection requires a calibration")
    if collection_mode == "teacher" and dart_calibration is not None:
        raise ValueError("teacher collection does not accept a DART calibration")
    if disturbance_seed < 0:
        raise ValueError("disturbance seed must be non-negative")
    catalog = default_action_catalog(candidate_names)
    if dart_calibration is not None and dart_calibration.candidate_names != candidate_names:
        raise ValueError("DART calibration candidate catalog differs")
    engine = RtlolaEngine(
        scenario.spec,
        event_arity=scenario.event_arity,
        expected_verdict_keys=scenario.expected_verdict_keys,
    )
    samples = []
    recovery_remaining = 0
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
                (events[step + 1],),
                catalog.mpc_candidates,
                budget,
                fallback=catalog.fallback,
                none_action=catalog.no_op,
                configured_horizon=1,
            )
            costs, feasible = _aligned_root_costs(decision.root_evaluations, candidate_names)
            teacher_action = decision.first_action.name
            executed_action = teacher_action
            disturbed = False
            disturbance_eligible = False
            disturbance_attempted = False
            recovery_forced = False
            target_disturbance_rate = 0.0
            injection_probability = 0.0
            disturbance_probability = 0.0
            regret_cap = float("nan")
            selected_direction_probability = float("nan")
            sampled_regret = float("nan")
            if collection_mode == "dart" and dart_calibration is not None:
                budget_index = dart_calibration.budget_index(budget)
                target_disturbance_rate = float(
                    dart_calibration.target_disturbance_rates[budget_index]
                )
                injection_probability = float(
                    dart_calibration.injection_probabilities[budget_index]
                )
                regret_cap = float(dart_calibration.regret_caps[budget_index])
            if recovery_remaining > 0:
                recovery_forced = True
                recovery_remaining -= 1
            elif (
                collection_mode == "dart"
                and dart_calibration is not None
                and teacher_action in candidate_names
                and np.any(feasible)
            ):
                teacher_index = candidate_names.index(teacher_action)
                regrets = normalized_regrets(costs, feasible)
                distribution = dart_calibration.alternative_distribution(
                    budget, teacher_action, feasible, regrets,
                )
                disturbance_eligible = bool(np.sum(distribution) > 0.0)
                if disturbance_eligible:
                    disturbance_probability = injection_probability
                    rng = np.random.default_rng(np.random.SeedSequence([
                        disturbance_seed, seed, budget, step,
                    ]))
                    disturbance_attempted = bool(rng.random() < injection_probability)
                    if disturbance_attempted:
                        selected = int(rng.choice(len(candidate_names), p=distribution))
                        executed_action = candidate_names[selected]
                        disturbed = selected != teacher_index
                        sampled_regret = float(regrets[selected])
                        selected_direction_probability = float(distribution[selected])
                        if sampled_regret > regret_cap + 1e-15:
                            raise AssertionError("DART sampled an action beyond its regret cap")
                        recovery_remaining = dart_calibration.config.recovery_decisions
            action = (
                catalog.by_name[executed_action]
                if executed_action in candidate_names
                else decision.first_action
            )
            sample_id = f"{trace_id}:{collection_mode}:budget-{budget}:step-{step}"
            dart_metadata = (
                DartDecisionMetadata(
                    executed_action=executed_action,
                    disturbed=disturbed,
                    disturbance_eligible=disturbance_eligible,
                    disturbance_attempted=disturbance_attempted,
                    recovery_forced=recovery_forced,
                    target_disturbance_rate=target_disturbance_rate,
                    injection_probability=injection_probability,
                    disturbance_probability=disturbance_probability,
                    regret_cap=regret_cap,
                    selected_direction_probability=selected_direction_probability,
                    sampled_normalized_regret=sampled_regret,
                    calibration_sha256=dart_calibration_sha256,
                )
                if collection_mode == "dart" else None
            )
            samples.append(CollectedReducerCostSample(
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
                teacher_action=teacher_action,
                teacher_sequence=decision.predicted_sequence,
                evaluated_leaves=decision.evaluated_leaves,
                teacher_reducer_failure_count=decision.reducer_failure_count,
                teacher_infeasible_candidate_count=decision.infeasible_candidate_count,
                execution_fallback_used=executed_action not in candidate_names,
                dart=dart_metadata,
            ))
        engine.live_step(event, action, budget, step=step + 1)
    return tuple(samples)


def build_reducer_cost_dataset(
    samples: Sequence[CollectedReducerCostSample],
    *,
    candidate_names: tuple[str, ...] | None = None,
) -> tuple[ReducerCostDataset, pd.DataFrame]:
    if not samples:
        if candidate_names is None:
            raise ValueError("empty collection requires an explicit candidate catalog")
        return _empty_reducer_cost_dataset(candidate_names)
    sample_candidate_names = samples[0].candidate_names
    if candidate_names is not None and candidate_names != sample_candidate_names:
        raise ValueError("explicit candidate catalog differs from collected samples")
    candidate_names = sample_candidate_names
    dart_presence = {sample.dart is not None for sample in samples}
    if len(dart_presence) != 1:
        raise ValueError("clean and DART decision metadata must not be mixed in one shard")
    for sample in samples:
        if sample.candidate_names != candidate_names:
            raise ValueError("collected samples use different candidate catalogs")
    dataset = ReducerCostDataset(
        features=np.stack([sample.features for sample in samples]).astype(np.float32),
        teacher_costs=np.asarray([sample.teacher_costs for sample in samples], dtype=np.float64),
        feasible=np.asarray([sample.feasible for sample in samples], dtype=np.bool_),
        candidate_names=candidate_names,
        feature_names=RTL_RANKING_FEATURE_NAMES,
        splits=tuple(sample.split for sample in samples),
        sample_ids=tuple(sample.sample_id for sample in samples),
    )
    metadata = pd.DataFrame([_sample_metadata(sample) for sample in samples])
    return dataset, metadata


def _sample_metadata(sample: CollectedReducerCostSample) -> dict[str, object]:
    metadata: dict[str, object] = {
        "sample_id": sample.sample_id,
        "trace_id": sample.trace_id,
        "split": sample.split,
        "condition": sample.condition,
        "seed": sample.seed,
        "budget": sample.budget,
        "step": sample.step,
        "teacher_action": sample.teacher_action,
        "teacher_sequence": json.dumps(sample.teacher_sequence),
        "evaluated_leaves": sample.evaluated_leaves,
        "teacher_reducer_failure_count": sample.teacher_reducer_failure_count,
        "teacher_infeasible_candidate_count": sample.teacher_infeasible_candidate_count,
        "execution_fallback_used": sample.execution_fallback_used,
    }
    if sample.dart is not None:
        metadata.update({
            "executed_action": sample.dart.executed_action,
            "disturbed": sample.dart.disturbed,
            "disturbance_eligible": sample.dart.disturbance_eligible,
            "disturbance_attempted": sample.dart.disturbance_attempted,
            "recovery_forced": sample.dart.recovery_forced,
            "target_disturbance_rate": sample.dart.target_disturbance_rate,
            "injection_probability": sample.dart.injection_probability,
            "disturbance_probability": sample.dart.disturbance_probability,
            "regret_cap": sample.dart.regret_cap,
            "selected_direction_probability": sample.dart.selected_direction_probability,
            "sampled_normalized_regret": sample.dart.sampled_normalized_regret,
            "dart_calibration_sha256": sample.dart.calibration_sha256,
        })
    return metadata


def _empty_reducer_cost_dataset(
    candidate_names: tuple[str, ...],
) -> tuple[ReducerCostDataset, pd.DataFrame]:
    candidate_count = len(candidate_names)
    dataset = ReducerCostDataset(
        features=np.empty((0, len(RTL_RANKING_FEATURE_NAMES)), dtype=np.float32),
        teacher_costs=np.empty((0, candidate_count), dtype=np.float64),
        feasible=np.empty((0, candidate_count), dtype=np.bool_),
        candidate_names=candidate_names,
        feature_names=RTL_RANKING_FEATURE_NAMES,
        splits=(),
        sample_ids=(),
    )
    return dataset, pd.DataFrame(columns=tuple(_sample_metadata_columns()))


def _sample_metadata_columns() -> tuple[str, ...]:
    return (
        "sample_id", "trace_id", "split", "condition", "seed", "budget",
        "step", "teacher_action", "teacher_sequence", "evaluated_leaves",
        "teacher_reducer_failure_count", "teacher_infeasible_candidate_count",
        "execution_fallback_used",
    )


def write_collected_dataset(
    samples: Sequence[CollectedReducerCostSample],
    directory: Path,
    metadata: Mapping[str, object],
    *,
    candidate_names: tuple[str, ...] | None = None,
) -> ReducerCostDataset:
    dataset, sample_metadata = build_reducer_cost_dataset(samples, candidate_names=candidate_names)
    write_reducer_cost_dataset(dataset, directory, sample_metadata, metadata)
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
