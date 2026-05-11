"""Static and receding-horizon policies for certified reductions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar

from pzr.control.costs import WeightedZonotopeCost
from pzr.core.certificates import ReductionResult
from pzr.monitoring.base import MonitorAdapter, MonitorState
from pzr.reduction.base import Reducer, ReductionContext

InputT = TypeVar("InputT")


@dataclass(frozen=True)
class ReductionDecision:
    """Chosen reduction and resulting monitor state."""

    state: MonitorState
    result: ReductionResult
    reducer_name: str
    predicted_cost: float = 0.0
    predicted_sequence: tuple[str, ...] = ()
    evaluated_sequences: int = 0
    pruned_sequences: int = 0


@dataclass(frozen=True)
class StaticReductionPolicy:
    """Always apply the same certified reducer."""

    reducer: Reducer
    budget: int

    def reduce_state(
        self,
        monitor: MonitorAdapter[InputT],
        state: MonitorState,
        context: ReductionContext | None = None,
    ) -> ReductionDecision:
        ctx = context or _reduction_context(monitor, state)
        result = self.reducer.reduce(state.zonotope, self.budget, ctx)
        if not result.certificate.is_sound:
            raise ValueError(f"reducer {self.reducer.name} returned an unsound certificate")
        reduced_state = monitor.replace_zonotope(state, result.reduced)
        return ReductionDecision(
            state=reduced_state,
            result=result,
            reducer_name=self.reducer.name,
        )


@dataclass(frozen=True)
class MPCPolicy(Generic[InputT]):
    """Receding-horizon reducer selection over certified candidate reducers."""

    reducers: tuple[Reducer, ...]
    budget: int
    horizon: int
    cost: WeightedZonotopeCost

    def reduce_state(
        self,
        monitor: MonitorAdapter[InputT],
        state: MonitorState,
        predicted_inputs: Sequence[InputT],
        context: ReductionContext | None = None,
    ) -> ReductionDecision:
        ctx = context or _reduction_context(monitor, state)
        best: ReductionDecision | None = None

        for reducer in self.reducers:
            try:
                first_result = reducer.reduce(state.zonotope, self.budget, ctx)
            except ValueError:
                continue
            if not first_result.certificate.is_sound:
                continue

            rollout_state = monitor.replace_zonotope(monitor.clone_state(state), first_result.reduced)
            total_cost = self.cost(rollout_state)

            for measurement in tuple(predicted_inputs)[: self.horizon]:
                step_result = monitor.step(rollout_state, measurement)
                rollout_state = step_result.state
                step_ctx = _reduction_context(monitor, rollout_state)
                if rollout_state.zonotope.generator_count > self.budget:
                    try:
                        rollout_reduction = reducer.reduce(
                            rollout_state.zonotope,
                            self.budget,
                            step_ctx,
                        )
                    except ValueError:
                        total_cost = float("inf")
                        break
                    if not rollout_reduction.certificate.is_sound:
                        total_cost = float("inf")
                        break
                    rollout_state = monitor.replace_zonotope(
                        rollout_state,
                        rollout_reduction.reduced,
                    )
                total_cost += self.cost(rollout_state, step_result.verdicts)

            decision = ReductionDecision(
                state=monitor.replace_zonotope(state, first_result.reduced),
                result=first_result,
                reducer_name=reducer.name,
                predicted_cost=float(total_cost),
                predicted_sequence=(reducer.name,),
                evaluated_sequences=1,
            )
            if best is None or decision.predicted_cost < best.predicted_cost:
                best = decision

        if best is None:
            raise ValueError("no candidate reducer could produce a certified budgeted state")
        return best


@dataclass(frozen=True)
class SequenceMPCPolicy(Generic[InputT]):
    """Receding-horizon search over reducer sequences at predicted overflows."""

    reducers: tuple[Reducer, ...]
    budget: int
    horizon: int
    cost: WeightedZonotopeCost

    def reduce_state(
        self,
        monitor: MonitorAdapter[InputT],
        state: MonitorState,
        predicted_inputs: Sequence[InputT],
        context: ReductionContext | None = None,
    ) -> ReductionDecision:
        ctx = context or _reduction_context(monitor, state)
        best: ReductionDecision | None = None
        evaluated_sequences = 0
        pruned_sequences = 0
        inputs = tuple(predicted_inputs)[: self.horizon]

        def try_reduce(
            reducer: Reducer,
            candidate_state: MonitorState,
            step_context: ReductionContext | None = None,
        ) -> tuple[MonitorState, ReductionResult] | None:
            step_ctx = step_context or _reduction_context(monitor, candidate_state)
            try:
                result = reducer.reduce(candidate_state.zonotope, self.budget, step_ctx)
            except ValueError:
                return None
            if not result.certificate.is_sound:
                return None
            return monitor.replace_zonotope(candidate_state, result.reduced), result

        def consider_leaf(
            first_state: MonitorState,
            first_result: ReductionResult,
            first_reducer: str,
            total_cost: float,
            sequence: tuple[str, ...],
        ) -> None:
            nonlocal best, evaluated_sequences
            evaluated_sequences += 1
            decision = ReductionDecision(
                state=monitor.replace_zonotope(state, first_result.reduced),
                result=first_result,
                reducer_name=first_reducer,
                predicted_cost=float(total_cost),
                predicted_sequence=sequence,
            )
            _ = first_state
            if best is None or decision.predicted_cost < best.predicted_cost:
                best = decision

        def rollout(
            index: int,
            rollout_state: MonitorState,
            total_cost: float,
            sequence: tuple[str, ...],
            first_state: MonitorState,
            first_result: ReductionResult,
            first_reducer: str,
        ) -> None:
            nonlocal pruned_sequences
            if best is not None and total_cost >= best.predicted_cost:
                pruned_sequences += 1
                return
            if index >= len(inputs):
                consider_leaf(
                    first_state,
                    first_result,
                    first_reducer,
                    total_cost,
                    sequence,
                )
                return

            step_result = monitor.step(rollout_state, inputs[index])
            next_state = step_result.state
            if next_state.zonotope.generator_count <= self.budget:
                rollout(
                    index + 1,
                    next_state,
                    total_cost + self.cost(next_state, step_result.verdicts),
                    sequence,
                    first_state,
                    first_result,
                    first_reducer,
                )
                return

            any_child = False
            for reducer in self.reducers:
                reduced = try_reduce(reducer, next_state)
                if reduced is None:
                    continue
                reduced_state, _ = reduced
                any_child = True
                rollout(
                    index + 1,
                    reduced_state,
                    total_cost + self.cost(reduced_state, step_result.verdicts),
                    (*sequence, reducer.name),
                    first_state,
                    first_result,
                    first_reducer,
                )
            if not any_child:
                pruned_sequences += 1

        for reducer in self.reducers:
            reduced = try_reduce(reducer, state, ctx)
            if reduced is None:
                continue
            first_state, first_result = reduced
            rollout(
                0,
                first_state,
                self.cost(first_state),
                (reducer.name,),
                first_state,
                first_result,
                reducer.name,
            )

        if best is None:
            raise ValueError("no candidate reducer sequence could produce a certified budgeted state")
        return ReductionDecision(
            state=best.state,
            result=best.result,
            reducer_name=best.reducer_name,
            predicted_cost=best.predicted_cost,
            predicted_sequence=best.predicted_sequence,
            evaluated_sequences=evaluated_sequences,
            pruned_sequences=pruned_sequences,
        )


@dataclass(frozen=True)
class RolloutMPCPolicy(Generic[InputT]):
    """One-step reducer lookahead with a fixed certified rollout base policy."""

    reducers: tuple[Reducer, ...]
    base_reducer: Reducer
    budget: int
    horizon: int
    cost: WeightedZonotopeCost
    fallback_reducer: Reducer | None = None
    terminal_cost_multiplier: float = 0.0

    def reduce_state(
        self,
        monitor: MonitorAdapter[InputT],
        state: MonitorState,
        predicted_inputs: Sequence[InputT],
        context: ReductionContext | None = None,
    ) -> ReductionDecision:
        ctx = context or _reduction_context(monitor, state)
        inputs = tuple(predicted_inputs)[: self.horizon]
        best = self._best_from_candidates(
            monitor,
            state,
            inputs,
            ctx,
            self.reducers,
        )
        if best is None and self.fallback_reducer is not None:
            best = self._best_from_candidates(
                monitor,
                state,
                inputs,
                ctx,
                (self.fallback_reducer,),
            )
        if best is None:
            raise ValueError("no rollout candidate reducer could produce a certified budgeted state")
        return best

    def _best_from_candidates(
        self,
        monitor: MonitorAdapter[InputT],
        state: MonitorState,
        predicted_inputs: tuple[InputT, ...],
        context: ReductionContext,
        reducers: tuple[Reducer, ...],
    ) -> ReductionDecision | None:
        best: ReductionDecision | None = None
        evaluated = 0
        pruned = 0
        for reducer in reducers:
            first = _try_reduce(monitor, reducer, state, self.budget, context)
            if first is None:
                pruned += 1
                continue
            first_state, first_result = first
            evaluated += 1
            total_cost = self.cost(first_state)
            rollout_state = first_state
            sequence = [reducer.name]
            failed = False

            for measurement in predicted_inputs:
                step_result = monitor.step(rollout_state, measurement)
                rollout_state = step_result.state
                if rollout_state.zonotope.generator_count > self.budget:
                    applied_reducer = self.base_reducer
                    reduced = _try_reduce(
                        monitor,
                        applied_reducer,
                        rollout_state,
                        self.budget,
                    )
                    if reduced is None and self.fallback_reducer is not None:
                        applied_reducer = self.fallback_reducer
                        reduced = _try_reduce(
                            monitor,
                            applied_reducer,
                            rollout_state,
                            self.budget,
                        )
                    if reduced is None:
                        pruned += 1
                        failed = True
                        break
                    rollout_state, _ = reduced
                    sequence.append(applied_reducer.name)
                total_cost += self.cost(rollout_state, step_result.verdicts)

            if failed:
                continue
            if self.terminal_cost_multiplier:
                total_cost += self.terminal_cost_multiplier * self.cost(rollout_state)
            decision = ReductionDecision(
                state=monitor.replace_zonotope(state, first_result.reduced),
                result=first_result,
                reducer_name=reducer.name,
                predicted_cost=float(total_cost),
                predicted_sequence=tuple(sequence),
                evaluated_sequences=evaluated,
                pruned_sequences=pruned,
            )
            if best is None or decision.predicted_cost < best.predicted_cost:
                best = decision

        if best is None:
            return None
        return ReductionDecision(
            state=best.state,
            result=best.result,
            reducer_name=best.reducer_name,
            predicted_cost=best.predicted_cost,
            predicted_sequence=best.predicted_sequence,
            evaluated_sequences=evaluated,
            pruned_sequences=pruned,
        )


def _try_reduce(
    monitor: MonitorAdapter[InputT],
    reducer: Reducer,
    state: MonitorState,
    budget: int,
    context: ReductionContext | None = None,
) -> tuple[MonitorState, ReductionResult] | None:
    ctx = context or _reduction_context(monitor, state)
    try:
        result = reducer.reduce(state.zonotope, budget, ctx)
    except ValueError:
        return None
    if not result.certificate.is_sound:
        return None
    return monitor.replace_zonotope(state, result.reduced), result


def _reduction_context(
    monitor: MonitorAdapter[InputT],
    state: MonitorState,
) -> ReductionContext:
    requirements = monitor.required_generator_metadata(state)
    return ReductionContext(
        step=state.step,
        triggers=monitor.triggers,
        required_generators=tuple(requirements),
    )
