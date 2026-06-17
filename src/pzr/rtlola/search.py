"""RTLola-native action search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pzr.rtlola.actions import RtlolaAction
from pzr.rtlola.engine import RtlolaEngine, RtlolaEvent, RtlolaStateRef, RtlolaStepResult


@dataclass(frozen=True)
class RtlolaSearchResult:
    first_action: RtlolaAction
    first_step: RtlolaStepResult
    predicted_cost: float
    predicted_sequence: tuple[str, ...]
    evaluated_leaves: int
    pruned_branches: int


@dataclass(frozen=True)
class BeamItem:
    first_action: RtlolaAction
    first_step: RtlolaStepResult
    rollout_state: RtlolaStateRef
    predicted_cost: float
    predicted_sequence: tuple[str, ...]


def choose_static_action(
    engine: RtlolaEngine,
    state: RtlolaStateRef,
    event: RtlolaEvent,
    action: RtlolaAction,
    budget: int,
    *,
    fallback: RtlolaAction,
) -> RtlolaSearchResult:
    """Evaluate one static action with fallback if it exceeds budget."""
    result = _try_action(engine, state, event, action, budget)
    selected_action = action
    if result is None and action.name != fallback.name:
        result = _try_action(engine, state, event, fallback, budget)
        selected_action = fallback
    if result is None:
        raise ValueError(
            f"no static RTLola action fits budget={budget}; "
            f"tried {action.name} and fallback {fallback.name}"
        )
    return RtlolaSearchResult(
        first_action=selected_action,
        first_step=result,
        predicted_cost=result.metrics.cost(),
        predicted_sequence=(selected_action.name,),
        evaluated_leaves=1,
        pruned_branches=0,
    )


def beam_search(
    engine: RtlolaEngine,
    state: RtlolaStateRef,
    current_event: RtlolaEvent,
    future_events: Sequence[RtlolaEvent],
    actions: tuple[RtlolaAction, ...],
    budget: int,
    beam_width: int,
    *,
    fallback: RtlolaAction,
) -> RtlolaSearchResult:
    """Bounded-width deterministic search over RTLola action sequences."""
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")
    if not actions:
        raise ValueError("at least one action is required")

    beam: list[BeamItem] = []
    for action in actions:
        step = _try_action(engine, state, current_event, action, budget)
        if step is None:
            continue
        beam.append(BeamItem(
            first_action=action,
            first_step=step,
            rollout_state=step.state,
            predicted_cost=step.metrics.cost(),
            predicted_sequence=(action.name,),
        ))

    if not beam:
        step = _try_action(engine, state, current_event, fallback, budget)
        if step is not None:
            beam.append(BeamItem(
                first_action=fallback,
                first_step=step,
                rollout_state=step.state,
                predicted_cost=step.metrics.cost(),
                predicted_sequence=(fallback.name,),
            ))

    if not beam:
        raise ValueError(f"no RTLola first action fits budget={budget}")

    beam.sort(key=_sort_key)
    pruned = max(0, len(beam) - beam_width)
    beam = beam[:beam_width]

    for event in tuple(future_events):
        expanded: list[BeamItem] = []
        for item in beam:
            children: list[BeamItem] = []
            for action in actions:
                step = _try_action(engine, item.rollout_state, event, action, budget)
                if step is None:
                    continue
                children.append(BeamItem(
                    first_action=item.first_action,
                    first_step=item.first_step,
                    rollout_state=step.state,
                    predicted_cost=item.predicted_cost + step.metrics.cost(),
                    predicted_sequence=(*item.predicted_sequence, action.name),
                ))
            if not children:
                step = _try_action(engine, item.rollout_state, event, fallback, budget)
                if step is not None:
                    children.append(BeamItem(
                        first_action=item.first_action,
                        first_step=item.first_step,
                        rollout_state=step.state,
                        predicted_cost=item.predicted_cost + step.metrics.cost(),
                        predicted_sequence=(*item.predicted_sequence, fallback.name),
                    ))
            expanded.extend(children)

        if not expanded:
            raise ValueError(f"no RTLola branch fits budget={budget}")
        expanded.sort(key=_sort_key)
        pruned += max(0, len(expanded) - beam_width)
        beam = expanded[:beam_width]

    best = min(beam, key=_sort_key)
    return RtlolaSearchResult(
        first_action=best.first_action,
        first_step=best.first_step,
        predicted_cost=best.predicted_cost,
        predicted_sequence=best.predicted_sequence,
        evaluated_leaves=len(beam),
        pruned_branches=pruned,
    )


def _try_action(
    engine: RtlolaEngine,
    state: RtlolaStateRef,
    event: RtlolaEvent,
    action: RtlolaAction,
    budget: int,
) -> RtlolaStepResult | None:
    if action.explicit_budget and budget < engine.metrics(state).dimension:
        return None
    step = engine.branch_step(state, event, action, budget)
    if step.metrics.dynamic_generator_count > budget:
        return None
    return step


def _sort_key(item: BeamItem) -> tuple[float, tuple[str, ...]]:
    return item.predicted_cost, item.predicted_sequence
