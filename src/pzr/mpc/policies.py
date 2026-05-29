"""Reduction policies: static and MPC-based.

All policies select among certified reducers. The soundness guarantee is
policy-independent: any selector over certified candidates inherits Z ⊆ Z'.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar

from pzr.monitoring.base import MonitorAdapter, MonitorState
from pzr.mpc.objectives import WeightedZonotopeCost
from pzr.mpc.prediction import InputPredictor
from pzr.mpc.search import SearchResult, tree_search
from pzr.zonotope.reduction import Reducer, ReductionResult

InputT = TypeVar("InputT")


@dataclass(frozen=True)
class ReductionDecision:
    """Result of a policy's reducer selection."""

    state: MonitorState
    result: ReductionResult
    reducer_name: str
    predicted_cost: float = 0.0
    predicted_sequence: tuple[str, ...] = ()
    evaluated_leaves: int = 0
    pruned_branches: int = 0


@dataclass(frozen=True)
class StaticPolicy:
    """Always apply the same reducer."""

    reducer: Reducer
    budget: int

    def select(
        self,
        monitor: MonitorAdapter,
        state: MonitorState,
    ) -> ReductionDecision:
        result = self.reducer.reduce(state.zonotope, self.budget)
        if not result.certificate.is_sound:
            raise ValueError(f"reducer {self.reducer.name} returned unsound certificate")
        reduced_state = monitor.replace_zonotope(state, result.reduced)
        return ReductionDecision(
            state=reduced_state,
            result=result,
            reducer_name=self.reducer.name,
        )


@dataclass(frozen=True)
class MPCPolicy(Generic[InputT]):
    """MPC reducer selection via tree search over a finite horizon."""

    candidates: tuple[Reducer, ...]
    budget: int
    horizon: int
    cost: WeightedZonotopeCost
    fallback: Reducer | None = None

    def select(
        self,
        monitor: MonitorAdapter[InputT],
        state: MonitorState,
        predicted_inputs: Sequence[InputT],
    ) -> ReductionDecision:
        search = tree_search(
            monitor=monitor,
            state=state,
            candidates=self.candidates,
            budget=self.budget,
            horizon=self.horizon,
            cost_fn=self.cost,
            predicted_inputs=predicted_inputs,
            fallback=self.fallback,
        )
        return ReductionDecision(
            state=search.best_state,
            result=search.best_result,
            reducer_name=search.best_reducer,
            predicted_cost=search.predicted_cost,
            predicted_sequence=search.predicted_sequence,
            evaluated_leaves=search.evaluated_leaves,
            pruned_branches=search.pruned_branches,
        )


@dataclass(frozen=True)
class RolloutMPCPolicy(Generic[InputT]):
    """Broad first-action search, fixed base reducer for future overflows.

    Evaluates each candidate as the first action, then rolls forward using
    a fixed base reducer for any predicted future overflows. This gives broad
    first-action coverage with cheaper future simulation.
    """

    candidates: tuple[Reducer, ...]
    base_reducer: Reducer
    budget: int
    horizon: int
    cost: WeightedZonotopeCost
    fallback: Reducer | None = None

    def select(
        self,
        monitor: MonitorAdapter[InputT],
        state: MonitorState,
        predicted_inputs: Sequence[InputT],
    ) -> ReductionDecision:
        inputs = tuple(predicted_inputs)[:self.horizon]
        best: ReductionDecision | None = None

        for reducer in self.candidates:
            result = self._try_first_action(monitor, state, reducer, inputs)
            if result is None:
                continue
            if best is None or result.predicted_cost < best.predicted_cost:
                best = result

        if best is None and self.fallback is not None:
            best = self._try_first_action(monitor, state, self.fallback, inputs)

        if best is None:
            raise ValueError("no candidate reducer could produce a certified budgeted state")
        return best

    def _try_first_action(
        self,
        monitor: MonitorAdapter[InputT],
        state: MonitorState,
        reducer: Reducer,
        inputs: tuple[InputT, ...],
    ) -> ReductionDecision | None:
        try:
            first_result = reducer.reduce(state.zonotope, self.budget)
        except ValueError:
            return None
        if not first_result.certificate.is_sound:
            return None

        first_state = monitor.replace_zonotope(state, first_result.reduced)
        total_cost = self.cost(first_state)
        rollout_state = first_state
        sequence = [reducer.name]

        for measurement in inputs:
            step_result = monitor.step(rollout_state, measurement)
            rollout_state = step_result.state

            if rollout_state.zonotope.generator_count > self.budget:
                reduced = self._try_reduce(monitor, self.base_reducer, rollout_state)
                if reduced is None and self.fallback is not None:
                    reduced = self._try_reduce(monitor, self.fallback, rollout_state)
                if reduced is None:
                    return None
                rollout_state, _ = reduced
                sequence.append(self.base_reducer.name)

            total_cost += self.cost(rollout_state, step_result.verdicts)

        return ReductionDecision(
            state=first_state,
            result=first_result,
            reducer_name=reducer.name,
            predicted_cost=total_cost,
            predicted_sequence=tuple(sequence),
        )

    def _try_reduce(
        self,
        monitor: MonitorAdapter,
        reducer: Reducer,
        state: MonitorState,
    ) -> tuple[MonitorState, ReductionResult] | None:
        try:
            result = reducer.reduce(state.zonotope, self.budget)
        except ValueError:
            return None
        if not result.certificate.is_sound:
            return None
        return monitor.replace_zonotope(state, result.reduced), result
