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


def tree_search(
    monitor: MonitorAdapter,
    state: MonitorState,
    candidates: tuple[Reducer, ...],
    budget: int,
    horizon: int,
    cost_fn: WeightedZonotopeCost,
    predicted_inputs: Sequence,
    fallback: Reducer | None = None,
) -> SearchResult:
    """Exhaustive tree search with cost pruning over reducer sequences."""
    inputs = tuple(predicted_inputs)[:horizon]
    best: dict = {"result": None, "evaluated": 0, "pruned": 0}

    def try_reduce(reducer: Reducer, s: MonitorState) -> tuple[MonitorState, ReductionResult] | None:
        try:
            result = reducer.reduce(s.zonotope, budget)
        except ValueError:
            return None
        if not result.certificate.is_sound:
            return None
        return monitor.replace_zonotope(s, result.reduced), result

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
            if best["result"] is None or total_cost < best["result"].predicted_cost:
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
            reduced = try_reduce(reducer, next_state)
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
            reduced = try_reduce(fallback, next_state)
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
        reduced = try_reduce(reducer, state)
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
        reduced = try_reduce(fallback, state)
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
