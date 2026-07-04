"""RTLola-native action search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from pzr.rtlola.actions import RtlolaAction
from pzr.rtlola.engine import (
    RtlolaBindingError,
    RtlolaEngine,
    RtlolaEvent,
    RtlolaStateRef,
    RtlolaStepResult,
)

CostFunction = Callable[[RtlolaEngine, RtlolaStepResult], float]


class RtlolaNoFeasibleAction(RuntimeError):
    """No binding transform could produce a sound successor state."""


@dataclass(frozen=True)
class RtlolaSearchResult:
    first_action: RtlolaAction
    first_action_budget: int
    first_step: RtlolaStepResult
    predicted_cost: float
    predicted_sequence: tuple[str, ...]
    evaluated_leaves: int
    pruned_branches: int
    fallback_used: bool = False
    reducer_failure_count: int = 0
    infeasible_candidate_count: int = 0


@dataclass(frozen=True)
class BeamItem:
    first_action: RtlolaAction
    first_action_budget: int
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
    none_action: RtlolaAction | None = None,
    cost_fn: CostFunction | None = None,
) -> RtlolaSearchResult:
    """Evaluate one static action only when the pre-event state exceeds the bound."""
    cost = cost_fn or _default_cost
    pre_metrics = engine.metrics(state)
    if none_action is not None:
        if pre_metrics.dynamic_generator_count <= budget:
            none_step = _branch_none(engine, state, event, none_action, budget)
            return RtlolaSearchResult(
                first_action=none_action,
                first_action_budget=budget,
                first_step=none_step,
                predicted_cost=cost(engine, none_step),
                predicted_sequence=(none_action.name,),
                evaluated_leaves=1,
                pruned_branches=0,
            )

    action_budget = int(budget)
    result = _try_action(engine, state, event, action, action_budget)
    selected_action = action
    selected_budget = action_budget
    failures = 0
    fallback_used = False
    if result is None:
        failures = 1
    if result is None and action.name != fallback.name:
        result = _try_action(engine, state, event, fallback, budget)
        selected_action = fallback
        selected_budget = budget
        fallback_used = result is not None
    if result is None:
        raise RtlolaNoFeasibleAction(
            f"no static RTLola action ran with bound={budget}; "
            f"tried {action.name} and fallback {fallback.name}"
        )
    return RtlolaSearchResult(
        first_action=selected_action,
        first_action_budget=selected_budget,
        first_step=result,
        predicted_cost=cost(engine, result),
        predicted_sequence=(selected_action.name,),
        evaluated_leaves=1,
        pruned_branches=0,
        fallback_used=fallback_used,
        reducer_failure_count=failures,
        infeasible_candidate_count=failures,
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
    none_action: RtlolaAction | None = None,
    cost_fn: CostFunction | None = None,
    use_reference_loss: bool = False,
    forced_first_action: RtlolaAction | None = None,
) -> RtlolaSearchResult:
    """Bounded-width deterministic search over RTLola action sequences."""
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")
    if not actions:
        raise ValueError("at least one action is required")
    if use_reference_loss and none_action is None:
        raise ValueError("reference-loss search requires a none_action")
    cost = cost_fn or _default_cost
    pre_metrics = engine.metrics(state)

    if none_action is not None:
        if pre_metrics.dynamic_generator_count <= budget:
            none_step = _branch_none(engine, state, current_event, none_action, budget)
            return RtlolaSearchResult(
                first_action=none_action,
                first_action_budget=budget,
                first_step=none_step,
                predicted_cost=0.0 if use_reference_loss else cost(engine, none_step),
                predicted_sequence=(none_action.name,),
                evaluated_leaves=1,
                pruned_branches=0,
            )

    events = (current_event, *tuple(future_events))
    reference_states = (
        _reference_rollout(engine, state, events, none_action, budget)
        if use_reference_loss and none_action is not None else ()
    )

    action_budget = int(budget)
    beam: list[BeamItem] = []
    failures = 0
    root_actions = (forced_first_action,) if forced_first_action is not None else actions
    for action in root_actions:
        if none_action is not None and action.name == none_action.name:
            continue
        step = _try_action(engine, state, current_event, action, action_budget)
        if step is None:
            failures += 1
            continue
        beam.append(BeamItem(
            first_action=action,
            first_action_budget=action_budget,
            first_step=step,
            rollout_state=step.state,
            predicted_cost=_score_step(
                engine,
                step,
                cost,
                reference_states=reference_states,
                depth=0,
            ),
            predicted_sequence=(action.name,),
        ))

    first_fallback_used = False
    if not beam:
        step = _try_action(engine, state, current_event, fallback, budget)
        if step is not None:
            first_fallback_used = True
            beam.append(BeamItem(
                first_action=fallback,
                first_action_budget=budget,
                first_step=step,
                rollout_state=step.state,
                predicted_cost=_score_step(
                    engine,
                    step,
                    cost,
                    reference_states=reference_states,
                    depth=0,
                ),
                predicted_sequence=(fallback.name,),
            ))

    if not beam:
        raise RtlolaNoFeasibleAction(
            f"no RTLola first action ran with bound={budget}"
        )

    beam.sort(key=_sort_key)
    pruned = max(0, len(beam) - beam_width)
    beam = beam[:beam_width]

    for depth, event in enumerate(tuple(future_events), start=1):
        expanded: list[BeamItem] = []
        for item in beam:
            if none_action is not None:
                rollout_metrics = engine.metrics(item.rollout_state)
                if rollout_metrics.dynamic_generator_count <= budget:
                    none_step = _branch_none(engine, item.rollout_state, event, none_action, budget)
                    expanded.append(BeamItem(
                        first_action=item.first_action,
                        first_action_budget=item.first_action_budget,
                        first_step=item.first_step,
                        rollout_state=none_step.state,
                        predicted_cost=_score_step(
                            engine,
                            none_step,
                            cost,
                            reference_states=reference_states,
                            depth=depth,
                        ),
                        predicted_sequence=(*item.predicted_sequence, none_action.name),
                    ))
                    continue

            action_budget = int(budget)
            children: list[BeamItem] = []
            for action in actions:
                if none_action is not None and action.name == none_action.name:
                    continue
                step = _try_action(engine, item.rollout_state, event, action, action_budget)
                if step is None:
                    failures += 1
                    continue
                children.append(BeamItem(
                    first_action=item.first_action,
                    first_action_budget=item.first_action_budget,
                    first_step=item.first_step,
                    rollout_state=step.state,
                    predicted_cost=_score_step(
                        engine,
                        step,
                        cost,
                        reference_states=reference_states,
                        depth=depth,
                    ),
                    predicted_sequence=(*item.predicted_sequence, action.name),
                ))
            if not children:
                step = _try_action(engine, item.rollout_state, event, fallback, budget)
                if step is not None:
                    children.append(BeamItem(
                        first_action=item.first_action,
                        first_action_budget=item.first_action_budget,
                        first_step=item.first_step,
                        rollout_state=step.state,
                        predicted_cost=_score_step(
                        engine,
                        step,
                        cost,
                        reference_states=reference_states,
                        depth=depth,
                        ),
                        predicted_sequence=(*item.predicted_sequence, fallback.name),
                    ))
            expanded.extend(children)

        if not expanded:
            raise RtlolaNoFeasibleAction(
                f"no RTLola branch ran with bound={budget}"
            )
        expanded.sort(key=_sort_key)
        pruned += max(0, len(expanded) - beam_width)
        beam = expanded[:beam_width]

    best = min(beam, key=_sort_key)
    return RtlolaSearchResult(
        first_action=best.first_action,
        first_action_budget=best.first_action_budget,
        first_step=best.first_step,
        predicted_cost=best.predicted_cost,
        predicted_sequence=best.predicted_sequence,
        evaluated_leaves=len(beam),
        pruned_branches=pruned,
        fallback_used=first_fallback_used and best.first_action.name == fallback.name,
        reducer_failure_count=failures,
        infeasible_candidate_count=failures,
    )


def _try_action(
    engine: RtlolaEngine,
    state: RtlolaStateRef,
    event: RtlolaEvent,
    action: RtlolaAction,
    action_budget: int,
) -> RtlolaStepResult | None:
    if action.explicit_budget and action_budget < engine.metrics(state).dimension:
        return None
    config_budget = action_budget if action.explicit_budget else max(0, action_budget)
    try:
        return engine.branch_step(state, event, action, config_budget)
    except RtlolaBindingError:
        return None


def _branch_none(
    engine: RtlolaEngine,
    state: RtlolaStateRef,
    event: RtlolaEvent,
    none_action: RtlolaAction,
    budget: int,
) -> RtlolaStepResult:
    return engine.branch_step(state, event, none_action, budget)


def _default_cost(_engine: RtlolaEngine, step: RtlolaStepResult) -> float:
    return step.metrics.cost()


def _reference_rollout(
    engine: RtlolaEngine,
    state: RtlolaStateRef,
    events: Sequence[RtlolaEvent],
    none_action: RtlolaAction,
    budget: int,
) -> tuple[RtlolaStateRef, ...]:
    references: list[RtlolaStateRef] = []
    rollout_state = state
    for event in events:
        step = _branch_none(engine, rollout_state, event, none_action, budget)
        references.append(step.state)
        rollout_state = step.state
    return tuple(references)


def _score_step(
    engine: RtlolaEngine,
    step: RtlolaStepResult,
    cost: CostFunction,
    *,
    reference_states: tuple[RtlolaStateRef, ...],
    depth: int,
) -> float:
    if reference_states:
        return engine.approx_loss(reference_states[depth], step.state)
    return cost(engine, step)


def _sort_key(item: BeamItem) -> tuple[float, tuple[str, ...]]:
    return item.predicted_cost, item.predicted_sequence
