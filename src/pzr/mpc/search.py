"""Discrete tree search for MPC reducer selection.

Searches over sequences of certified reducer choices at predicted overflow
points. Uses branch-and-bound pruning: branches whose accumulated cost
exceeds the best complete sequence are discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar

from pzr.monitoring.base import MonitorAdapter, MonitorState
from pzr.mpc.objectives import WeightedZonotopeCost
from pzr.zonotope.protected import ProtectedReducer, reduce_with_protection
from pzr.zonotope.reduction import Reducer, ReductionResult

InputT = TypeVar("InputT")


@dataclass(frozen=True)
class SearchResult:
    """Result of the tree search."""

    best_reducer: str
    best_result: ReductionResult
    best_state: MonitorState
    predicted_cost: float
    predicted_sequence: tuple[str, ...]
    evaluated_leaves: int
    pruned_branches: int


@dataclass(frozen=True)
class BeamItem:
    """Partial beam-search trajectory."""

    first_reducer: str
    first_result: ReductionResult
    first_state: MonitorState
    rollout_state: MonitorState
    predicted_cost: float
    predicted_sequence: tuple[str, ...]


def try_certified_reduce(
    monitor: MonitorAdapter,
    state: MonitorState,
    reducer: Reducer | ProtectedReducer,
    budget: int,
) -> tuple[MonitorState, ReductionResult] | None:
    """Reduce a monitor state while preserving protected generator metadata."""
    try:
        result = reduce_with_protection(
            reducer, state.zonotope, budget,
            protected_indices=state.calibration_indices,
        )
    except ValueError:
        return None
    if not result.certificate.is_sound:
        return None
    reduced_state = monitor.replace_zonotope(state, result.reduced)
    if state.calibration_indices:
        new_cal = tuple(range(len(state.calibration_indices)))
        reduced_state = reduced_state.with_zonotope(
            reduced_state.zonotope, calibration_indices=new_cal,
        )
    return reduced_state, result


def tree_search(
    monitor: MonitorAdapter,
    state: MonitorState,
    candidates: tuple[Reducer | ProtectedReducer, ...],
    budget: int,
    horizon: int,
    cost_fn: WeightedZonotopeCost,
    predicted_inputs: Sequence,
    fallback: Reducer | ProtectedReducer | None = None,
) -> SearchResult:
    """Exhaustive tree search with cost pruning over reducer sequences."""
    inputs = tuple(predicted_inputs)[:horizon]
    best: dict = {"result": None, "evaluated": 0, "pruned": 0}

    def rollout(
        index: int,
        rollout_state: MonitorState,
        total_cost: float,
        sequence: tuple[str, ...],
        first_state: MonitorState,
        first_result: ReductionResult,
        first_reducer: str,
    ) -> None:
        if best["result"] is not None and total_cost >= best["result"].predicted_cost:
            best["pruned"] += 1
            return

        if index >= len(inputs):
            best["evaluated"] += 1
            if (
                best["result"] is None
                or total_cost < best["result"].predicted_cost
                or (
                    total_cost == best["result"].predicted_cost
                    and sequence < best["result"].predicted_sequence
                )
            ):
                best["result"] = SearchResult(
                    best_reducer=first_reducer,
                    best_result=first_result,
                    best_state=first_state,
                    predicted_cost=total_cost,
                    predicted_sequence=sequence,
                    evaluated_leaves=0,
                    pruned_branches=0,
                )
            return

        step_result = monitor.step(rollout_state, inputs[index])
        next_state = step_result.state

        if next_state.zonotope.generator_count <= budget:
            rollout(
                index + 1,
                next_state,
                total_cost + cost_fn(next_state, step_result.verdicts),
                sequence,
                first_state,
                first_result,
                first_reducer,
            )
            return

        any_child = False
        for reducer in candidates:
            reduced = try_certified_reduce(monitor, next_state, reducer, budget)
            if reduced is None:
                continue
            any_child = True
            reduced_state, _ = reduced
            rollout(
                index + 1,
                reduced_state,
                total_cost + cost_fn(reduced_state, step_result.verdicts),
                (*sequence, reducer.name),
                first_state,
                first_result,
                first_reducer,
            )

        if not any_child and fallback is not None:
            reduced = try_certified_reduce(monitor, next_state, fallback, budget)
            if reduced is not None:
                reduced_state, _ = reduced
                rollout(
                    index + 1,
                    reduced_state,
                    total_cost + cost_fn(reduced_state, step_result.verdicts),
                    (*sequence, fallback.name),
                    first_state,
                    first_result,
                    first_reducer,
                )

    # Try each candidate as first action
    first_success = False
    for reducer in candidates:
        reduced = try_certified_reduce(monitor, state, reducer, budget)
        if reduced is None:
            continue
        first_success = True
        first_state, first_result = reduced
        rollout(
            0,
            first_state,
            cost_fn(first_state),
            (reducer.name,),
            first_state,
            first_result,
            reducer.name,
        )

    if not first_success and fallback is not None:
        reduced = try_certified_reduce(monitor, state, fallback, budget)
        if reduced is not None:
            first_state, first_result = reduced
            rollout(
                0,
                first_state,
                cost_fn(first_state),
                (fallback.name,),
                first_state,
                first_result,
                fallback.name,
            )

    if best["result"] is None:
        raise ValueError("no candidate reducer sequence found")

    return SearchResult(
        best_reducer=best["result"].best_reducer,
        best_result=best["result"].best_result,
        best_state=best["result"].best_state,
        predicted_cost=best["result"].predicted_cost,
        predicted_sequence=best["result"].predicted_sequence,
        evaluated_leaves=best["evaluated"],
        pruned_branches=best["pruned"],
    )


def beam_search(
    monitor: MonitorAdapter,
    state: MonitorState,
    candidates: tuple[Reducer | ProtectedReducer, ...],
    budget: int,
    horizon: int,
    cost_fn: WeightedZonotopeCost,
    predicted_inputs: Sequence,
    beam_width: int,
    fallback: Reducer | ProtectedReducer | None = None,
) -> SearchResult:
    """Bounded-width deterministic search over reducer sequences."""
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")

    inputs = tuple(predicted_inputs)[:horizon]
    beam: list[BeamItem] = []

    for reducer in candidates:
        reduced = try_certified_reduce(monitor, state, reducer, budget)
        if reduced is None:
            continue
        first_state, first_result = reduced
        beam.append(BeamItem(
            first_reducer=reducer.name,
            first_result=first_result,
            first_state=first_state,
            rollout_state=first_state,
            predicted_cost=cost_fn(first_state),
            predicted_sequence=(reducer.name,),
        ))

    if not beam and fallback is not None:
        reduced = try_certified_reduce(monitor, state, fallback, budget)
        if reduced is not None:
            first_state, first_result = reduced
            beam.append(BeamItem(
                first_reducer=fallback.name,
                first_result=first_result,
                first_state=first_state,
                rollout_state=first_state,
                predicted_cost=cost_fn(first_state),
                predicted_sequence=(fallback.name,),
            ))

    if not beam:
        raise ValueError("no candidate reducer sequence found")

    beam.sort(key=_beam_sort_key)
    pruned = max(0, len(beam) - beam_width)
    beam = beam[:beam_width]

    for measurement in inputs:
        expanded: list[BeamItem] = []
        for item in beam:
            step_result = monitor.step(item.rollout_state, measurement)
            next_state = step_result.state

            if next_state.zonotope.generator_count <= budget:
                expanded.append(BeamItem(
                    first_reducer=item.first_reducer,
                    first_result=item.first_result,
                    first_state=item.first_state,
                    rollout_state=next_state,
                    predicted_cost=item.predicted_cost
                    + cost_fn(next_state, step_result.verdicts),
                    predicted_sequence=item.predicted_sequence,
                ))
                continue

            children: list[BeamItem] = []
            for reducer in candidates:
                reduced = try_certified_reduce(monitor, next_state, reducer, budget)
                if reduced is None:
                    continue
                reduced_state, _ = reduced
                children.append(BeamItem(
                    first_reducer=item.first_reducer,
                    first_result=item.first_result,
                    first_state=item.first_state,
                    rollout_state=reduced_state,
                    predicted_cost=item.predicted_cost
                    + cost_fn(reduced_state, step_result.verdicts),
                    predicted_sequence=(*item.predicted_sequence, reducer.name),
                ))

            if not children and fallback is not None:
                reduced = try_certified_reduce(monitor, next_state, fallback, budget)
                if reduced is not None:
                    reduced_state, _ = reduced
                    children.append(BeamItem(
                        first_reducer=item.first_reducer,
                        first_result=item.first_result,
                        first_state=item.first_state,
                        rollout_state=reduced_state,
                        predicted_cost=item.predicted_cost
                        + cost_fn(reduced_state, step_result.verdicts),
                        predicted_sequence=(*item.predicted_sequence, fallback.name),
                    ))
            expanded.extend(children)

        if not expanded:
            raise ValueError("no candidate reducer sequence found")

        expanded.sort(key=_beam_sort_key)
        pruned += max(0, len(expanded) - beam_width)
        beam = expanded[:beam_width]

    best = min(beam, key=_beam_sort_key)
    return SearchResult(
        best_reducer=best.first_reducer,
        best_result=best.first_result,
        best_state=best.first_state,
        predicted_cost=best.predicted_cost,
        predicted_sequence=best.predicted_sequence,
        evaluated_leaves=len(beam),
        pruned_branches=pruned,
    )


def _beam_sort_key(item: BeamItem) -> tuple[float, tuple[str, ...]]:
    return item.predicted_cost, item.predicted_sequence
