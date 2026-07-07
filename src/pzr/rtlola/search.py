"""RTLola-native action search."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
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


class MpcObjective(str, Enum):
    """Binding-native objective used to rank complete MPC rollouts."""

    TERMINAL = "terminal_binding_approx_loss"
    EXTENDED_ENDPOINT = "extended_terminal_binding_approx_loss"
    INTEGRATED_TAIL = "integrated_binding_approx_loss_with_girard_tail"


class MpcRootStrategy(str, Enum):
    """How beam capacity is shared between first-action lineages."""

    GLOBAL = "global"
    STRATIFIED = "stratified"
    ROOT_ONLY = "root_only"


@dataclass(frozen=True)
class MpcVariant:
    """Named, reproducible MPC search semantics."""

    method: str
    objective: MpcObjective
    root_strategy: MpcRootStrategy
    uses_configured_horizon: bool
    uses_tail: bool


MPC_VARIANTS = {
    variant.method: variant
    for variant in (
        MpcVariant(
            "mpc_terminal_beam",
            MpcObjective.TERMINAL,
            MpcRootStrategy.GLOBAL,
            uses_configured_horizon=True,
            uses_tail=False,
        ),
        MpcVariant(
            "mpc_terminal_girard_tail",
            MpcObjective.EXTENDED_ENDPOINT,
            MpcRootStrategy.STRATIFIED,
            uses_configured_horizon=True,
            uses_tail=True,
        ),
        MpcVariant(
            "mpc_cumulative_girard_tail",
            MpcObjective.INTEGRATED_TAIL,
            MpcRootStrategy.STRATIFIED,
            uses_configured_horizon=True,
            uses_tail=True,
        ),
        MpcVariant(
            "mpc_one_step_girard_rollout",
            MpcObjective.INTEGRATED_TAIL,
            MpcRootStrategy.ROOT_ONLY,
            uses_configured_horizon=False,
            uses_tail=True,
        ),
    )
}


@dataclass(frozen=True)
class MpcRootEvaluation:
    """Best complete rollout retained for one first action."""

    root_action: str
    feasible: bool
    complete: bool
    predicted_cost: float = float("nan")
    predicted_sequence: tuple[str, ...] = ()
    explicit_path_loss: float = float("nan")
    explicit_terminal_loss: float = float("nan")
    tail_path_loss: float = float("nan")
    tail_terminal_loss: float = float("nan")
    realized_tail_steps: int = 0
    failure_count: int = 0


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
    mpc_variant: str = ""
    mpc_objective: str = ""
    root_strategy: str = ""
    optimized_horizon: int = 0
    realized_optimized_horizon: int = 0
    configured_tail_horizon: int = 0
    realized_tail_steps: int = 0
    root_beam_width: int = 0
    explicit_path_loss: float = float("nan")
    explicit_terminal_loss: float = float("nan")
    tail_path_loss: float = float("nan")
    tail_terminal_loss: float = float("nan")
    tail_fallback_count: int = 0
    root_evaluations: tuple[MpcRootEvaluation, ...] = ()


@dataclass(frozen=True)
class BeamItem:
    first_action: RtlolaAction
    first_action_budget: int
    first_step: RtlolaStepResult
    rollout_state: RtlolaStateRef
    predicted_cost: float
    predicted_sequence: tuple[str, ...]
    explicit_path_loss: float = 0.0
    explicit_terminal_loss: float = 0.0
    tail_path_loss: float = 0.0
    tail_terminal_loss: float = float("nan")
    realized_tail_steps: int = 0
    tail_fallback_count: int = 0


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
    configured_horizon: int | None = None,
) -> RtlolaSearchResult:
    """Run the historical terminal-loss beam search."""
    return _search_mpc(
        engine,
        state,
        current_event,
        future_events,
        (),
        actions,
        budget,
        beam_width,
        variant=MPC_VARIANTS["mpc_terminal_beam"],
        root_beam_width=beam_width,
        fallback=fallback,
        none_action=none_action,
        tail_action=None,
        cost_fn=cost_fn,
        use_reference_loss=use_reference_loss,
        forced_first_action=forced_first_action,
        configured_horizon=configured_horizon,
        configured_tail_horizon=0,
    )


def search_mpc_variant(
    engine: RtlolaEngine,
    state: RtlolaStateRef,
    current_event: RtlolaEvent,
    future_events: Sequence[RtlolaEvent],
    tail_events: Sequence[RtlolaEvent],
    actions: tuple[RtlolaAction, ...],
    budget: int,
    beam_width: int,
    *,
    variant: MpcVariant,
    root_beam_width: int,
    fallback: RtlolaAction,
    none_action: RtlolaAction,
    tail_action: RtlolaAction,
    configured_horizon: int | None = None,
    configured_tail_horizon: int | None = None,
) -> RtlolaSearchResult:
    """Run one named MPC variant with binding-native reference losses."""
    optimized_future = tuple(future_events) if variant.uses_configured_horizon else ()
    evaluated_tail = tuple(tail_events) if variant.uses_tail else ()
    return _search_mpc(
        engine,
        state,
        current_event,
        optimized_future,
        evaluated_tail,
        actions,
        budget,
        beam_width,
        variant=variant,
        root_beam_width=root_beam_width,
        fallback=fallback,
        none_action=none_action,
        tail_action=tail_action,
        use_reference_loss=True,
        configured_horizon=(
            configured_horizon if variant.uses_configured_horizon else 0
        ),
        configured_tail_horizon=configured_tail_horizon,
    )


def _search_mpc(
    engine: RtlolaEngine,
    state: RtlolaStateRef,
    current_event: RtlolaEvent,
    future_events: Sequence[RtlolaEvent],
    tail_events: Sequence[RtlolaEvent],
    actions: tuple[RtlolaAction, ...],
    budget: int,
    beam_width: int,
    *,
    variant: MpcVariant,
    root_beam_width: int,
    fallback: RtlolaAction,
    none_action: RtlolaAction | None,
    tail_action: RtlolaAction | None,
    cost_fn: CostFunction | None = None,
    use_reference_loss: bool = False,
    forced_first_action: RtlolaAction | None = None,
    configured_horizon: int | None = None,
    configured_tail_horizon: int | None = None,
) -> RtlolaSearchResult:
    """Bounded-width deterministic search over RTLola action sequences."""
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")
    if root_beam_width < 1:
        raise ValueError("root_beam_width must be >= 1")
    if not actions:
        raise ValueError("at least one action is required")
    if use_reference_loss and none_action is None:
        raise ValueError("reference-loss search requires a none_action")
    if variant.uses_tail and tail_action is None:
        raise ValueError("tail MPC variant requires a tail action")
    cost = cost_fn or _default_cost
    recorded_horizon = (
        len(future_events) if configured_horizon is None else configured_horizon
    )
    recorded_tail_horizon = (
        len(tail_events)
        if configured_tail_horizon is None else configured_tail_horizon
    )
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
                mpc_variant=variant.method,
                mpc_objective=variant.objective.value,
                root_strategy=variant.root_strategy.value,
                optimized_horizon=recorded_horizon,
                realized_optimized_horizon=len(future_events),
                configured_tail_horizon=recorded_tail_horizon,
                root_beam_width=(
                    root_beam_width
                    if variant.root_strategy is not MpcRootStrategy.GLOBAL else 0
                ),
            )

    explicit_events = (current_event, *tuple(future_events))
    events = (*explicit_events, *tuple(tail_events))
    reference_states = (
        _reference_rollout(engine, state, events, none_action, budget)
        if use_reference_loss and none_action is not None else ()
    )

    action_budget = int(budget)
    beam: list[BeamItem] = []
    failures = 0
    root_failures = {action.name: 0 for action in actions}
    feasible_roots: set[str] = set()
    root_actions = (forced_first_action,) if forced_first_action is not None else actions
    for action in root_actions:
        if none_action is not None and action.name == none_action.name:
            continue
        step = _try_action(engine, state, current_event, action, action_budget)
        if step is None:
            failures += 1
            root_failures[action.name] = root_failures.get(action.name, 0) + 1
            continue
        step_loss = _score_step(
            engine,
            step,
            cost,
            reference_states=reference_states,
            depth=0,
        )
        feasible_roots.add(action.name)
        beam.append(BeamItem(
            first_action=action,
            first_action_budget=action_budget,
            first_step=step,
            rollout_state=step.state,
            predicted_cost=step_loss,
            predicted_sequence=(action.name,),
            explicit_path_loss=step_loss,
            explicit_terminal_loss=step_loss,
        ))

    first_fallback_used = False
    if not beam:
        step = _try_action(engine, state, current_event, fallback, budget)
        if step is not None:
            first_fallback_used = True
            feasible_roots.add(fallback.name)
            step_loss = _score_step(
                engine,
                step,
                cost,
                reference_states=reference_states,
                depth=0,
            )
            beam.append(BeamItem(
                first_action=fallback,
                first_action_budget=budget,
                first_step=step,
                rollout_state=step.state,
                predicted_cost=step_loss,
                predicted_sequence=(fallback.name,),
                explicit_path_loss=step_loss,
                explicit_terminal_loss=step_loss,
            ))

    if not beam:
        raise RtlolaNoFeasibleAction(
            f"no RTLola first action ran with bound={budget}"
        )

    beam, pruned = _prune_beam(
        beam,
        variant=variant,
        beam_width=beam_width,
        root_beam_width=root_beam_width,
    )

    for depth, event in enumerate(tuple(future_events), start=1):
        expanded: list[BeamItem] = []
        for item in beam:
            if none_action is not None:
                rollout_metrics = engine.metrics(item.rollout_state)
                if rollout_metrics.dynamic_generator_count <= budget:
                    none_step = _branch_none(engine, item.rollout_state, event, none_action, budget)
                    step_loss = _score_step(
                        engine,
                        none_step,
                        cost,
                        reference_states=reference_states,
                        depth=depth,
                    )
                    expanded.append(BeamItem(
                        first_action=item.first_action,
                        first_action_budget=item.first_action_budget,
                        first_step=item.first_step,
                        rollout_state=none_step.state,
                        predicted_cost=_prefix_cost(
                            variant,
                            item.explicit_path_loss + step_loss,
                            step_loss,
                        ),
                        predicted_sequence=(*item.predicted_sequence, none_action.name),
                        explicit_path_loss=item.explicit_path_loss + step_loss,
                        explicit_terminal_loss=step_loss,
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
                    root_name = item.first_action.name
                    root_failures[root_name] = root_failures.get(root_name, 0) + 1
                    continue
                step_loss = _score_step(
                    engine,
                    step,
                    cost,
                    reference_states=reference_states,
                    depth=depth,
                )
                children.append(BeamItem(
                    first_action=item.first_action,
                    first_action_budget=item.first_action_budget,
                    first_step=item.first_step,
                    rollout_state=step.state,
                    predicted_cost=_prefix_cost(
                        variant,
                        item.explicit_path_loss + step_loss,
                        step_loss,
                    ),
                    predicted_sequence=(*item.predicted_sequence, action.name),
                    explicit_path_loss=item.explicit_path_loss + step_loss,
                    explicit_terminal_loss=step_loss,
                ))
            if not children:
                step = _try_action(engine, item.rollout_state, event, fallback, budget)
                if step is not None:
                    step_loss = _score_step(
                        engine,
                        step,
                        cost,
                        reference_states=reference_states,
                        depth=depth,
                    )
                    children.append(BeamItem(
                        first_action=item.first_action,
                        first_action_budget=item.first_action_budget,
                        first_step=item.first_step,
                        rollout_state=step.state,
                        predicted_cost=_prefix_cost(
                            variant,
                            item.explicit_path_loss + step_loss,
                            step_loss,
                        ),
                        predicted_sequence=(*item.predicted_sequence, fallback.name),
                        explicit_path_loss=item.explicit_path_loss + step_loss,
                        explicit_terminal_loss=step_loss,
                    ))
            expanded.extend(children)

        if not expanded:
            raise RtlolaNoFeasibleAction(
                f"no RTLola branch ran with bound={budget}"
            )
        beam, newly_pruned = _prune_beam(
            expanded,
            variant=variant,
            beam_width=beam_width,
            root_beam_width=root_beam_width,
        )
        pruned += newly_pruned

    completed: list[BeamItem] = []
    for item in beam:
        evaluated, tail_failures = _evaluate_tail(
            engine,
            item,
            tuple(tail_events),
            reference_states,
            explicit_depth=len(explicit_events),
            budget=budget,
            none_action=none_action,
            tail_action=tail_action,
            fallback=fallback,
            variant=variant,
        )
        failures += tail_failures
        root_name = item.first_action.name
        root_failures[root_name] = root_failures.get(root_name, 0) + tail_failures
        if evaluated is not None:
            completed.append(evaluated)
    if not completed:
        raise RtlolaNoFeasibleAction(
            f"no RTLola branch completed the tail with bound={budget}"
        )

    best = min(completed, key=_sort_key)
    best_by_root: dict[str, BeamItem] = {}
    for item in completed:
        root_name = item.first_action.name
        retained = best_by_root.get(root_name)
        if retained is None or _sort_key(item) < _sort_key(retained):
            best_by_root[root_name] = item
    root_names = [action.name for action in root_actions]
    if first_fallback_used and fallback.name not in root_names:
        root_names.append(fallback.name)
    root_evaluations = tuple(
        _root_evaluation(
            name,
            best_by_root.get(name),
            name in feasible_roots,
            root_failures.get(name, 0),
        )
        for name in root_names
    )
    return RtlolaSearchResult(
        first_action=best.first_action,
        first_action_budget=best.first_action_budget,
        first_step=best.first_step,
        predicted_cost=best.predicted_cost,
        predicted_sequence=best.predicted_sequence,
        evaluated_leaves=len(completed),
        pruned_branches=pruned,
        fallback_used=first_fallback_used and best.first_action.name == fallback.name,
        reducer_failure_count=failures,
        infeasible_candidate_count=failures,
        mpc_variant=variant.method,
        mpc_objective=variant.objective.value,
        root_strategy=variant.root_strategy.value,
        optimized_horizon=recorded_horizon,
        realized_optimized_horizon=len(future_events),
        configured_tail_horizon=recorded_tail_horizon,
        realized_tail_steps=best.realized_tail_steps,
        root_beam_width=(
            root_beam_width
            if variant.root_strategy is not MpcRootStrategy.GLOBAL else 0
        ),
        explicit_path_loss=best.explicit_path_loss,
        explicit_terminal_loss=best.explicit_terminal_loss,
        tail_path_loss=best.tail_path_loss,
        tail_terminal_loss=best.tail_terminal_loss,
        tail_fallback_count=best.tail_fallback_count,
        root_evaluations=root_evaluations,
    )


def _prefix_cost(
    variant: MpcVariant,
    explicit_path_loss: float,
    explicit_terminal_loss: float,
) -> float:
    if variant.objective is MpcObjective.INTEGRATED_TAIL:
        return explicit_path_loss
    return explicit_terminal_loss


def _prune_beam(
    items: Sequence[BeamItem],
    *,
    variant: MpcVariant,
    beam_width: int,
    root_beam_width: int,
) -> tuple[list[BeamItem], int]:
    if variant.root_strategy is MpcRootStrategy.GLOBAL:
        ordered = sorted(items, key=_sort_key)
        return ordered[:beam_width], max(0, len(ordered) - beam_width)
    groups: dict[str, list[BeamItem]] = {}
    for item in items:
        groups.setdefault(item.first_action.name, []).append(item)
    retained: list[BeamItem] = []
    for name in sorted(groups):
        retained.extend(sorted(groups[name], key=_sort_key)[:root_beam_width])
    return retained, len(items) - len(retained)


def _evaluate_tail(
    engine: RtlolaEngine,
    item: BeamItem,
    tail_events: tuple[RtlolaEvent, ...],
    reference_states: tuple[RtlolaStateRef, ...],
    *,
    explicit_depth: int,
    budget: int,
    none_action: RtlolaAction | None,
    tail_action: RtlolaAction | None,
    fallback: RtlolaAction,
    variant: MpcVariant,
) -> tuple[BeamItem | None, int]:
    if not tail_events:
        return replace(
            item,
            predicted_cost=_complete_cost(
                variant,
                item.explicit_path_loss,
                item.explicit_terminal_loss,
                0.0,
                float("nan"),
            ),
        ), 0
    if none_action is None or tail_action is None:
        raise ValueError("tail evaluation requires none and tail actions")
    state = item.rollout_state
    path_loss = 0.0
    terminal_loss = float("nan")
    failures = 0
    fallback_count = 0
    realized_steps = 0
    for offset, event in enumerate(tail_events):
        metrics = engine.metrics(state)
        if metrics.dynamic_generator_count <= budget:
            step = _try_action(engine, state, event, none_action, budget)
        else:
            step = _try_action(engine, state, event, tail_action, budget)
            if step is None:
                failures += 1
                step = _try_action(engine, state, event, fallback, budget)
                if step is not None:
                    fallback_count += 1
        if step is None:
            failures += 1
            return None, failures
        reference = reference_states[explicit_depth + offset]
        terminal_loss = engine.approx_loss(reference, step.state)
        path_loss += terminal_loss
        state = step.state
        realized_steps += 1
    return replace(
        item,
        rollout_state=state,
        predicted_cost=_complete_cost(
            variant,
            item.explicit_path_loss,
            item.explicit_terminal_loss,
            path_loss,
            terminal_loss,
        ),
        tail_path_loss=path_loss,
        tail_terminal_loss=terminal_loss,
        realized_tail_steps=realized_steps,
        tail_fallback_count=fallback_count,
    ), failures


def _complete_cost(
    variant: MpcVariant,
    explicit_path_loss: float,
    explicit_terminal_loss: float,
    tail_path_loss: float,
    tail_terminal_loss: float,
) -> float:
    if variant.objective is MpcObjective.TERMINAL:
        return explicit_terminal_loss
    if variant.objective is MpcObjective.EXTENDED_ENDPOINT:
        return (
            tail_terminal_loss
            if tail_terminal_loss == tail_terminal_loss else explicit_terminal_loss
        )
    return explicit_path_loss + tail_path_loss


def _root_evaluation(
    root_action: str,
    item: BeamItem | None,
    feasible: bool,
    failure_count: int,
) -> MpcRootEvaluation:
    if item is None:
        return MpcRootEvaluation(
            root_action=root_action,
            feasible=feasible,
            complete=False,
            failure_count=failure_count,
        )
    return MpcRootEvaluation(
        root_action=root_action,
        feasible=True,
        complete=True,
        predicted_cost=item.predicted_cost,
        predicted_sequence=item.predicted_sequence,
        explicit_path_loss=item.explicit_path_loss,
        explicit_terminal_loss=item.explicit_terminal_loss,
        tail_path_loss=item.tail_path_loss,
        tail_terminal_loss=item.tail_terminal_loss,
        realized_tail_steps=item.realized_tail_steps,
        failure_count=failure_count,
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
