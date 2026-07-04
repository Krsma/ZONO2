"""Guarded live/planner wrapper around RLolaMonitor."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from pzr.rtlola.actions import RtlolaAction
from pzr.rtlola.binding import require_binding
from pzr.rtlola.metrics import RtlolaMatrixMetrics, matrix_metrics


class RtlolaBindingError(RuntimeError):
    """The native RTLola binding failed while evaluating a state."""


@dataclass(frozen=True)
class RtlolaEvent:
    """One event for an RTLola monitor."""

    time: float
    values: tuple[Any, ...]


@dataclass(frozen=True)
class RtlolaStateRef:
    """Evaluator state tagged with the spec and logical step."""

    state: Any
    spec_id: str
    step: int
    time: float


@dataclass(frozen=True)
class RtlolaStepResult:
    verdict: dict[str, Any]
    state: RtlolaStateRef
    action_name: str
    metrics: RtlolaMatrixMetrics


class RtlolaEngine:
    """Owns live and planner RTLola monitors for safe branching."""

    def __init__(
        self,
        spec: str,
        *,
        event_arity: int,
        expected_verdict_keys: Iterable[str] = (),
    ) -> None:
        _, RLolaMonitor, _ = require_binding()
        self.spec = spec
        self.spec_id = hashlib.sha256(spec.encode("utf-8")).hexdigest()
        self.event_arity = int(event_arity)
        self.expected_verdict_keys = tuple(expected_verdict_keys)
        self.live = RLolaMonitor(spec)
        self.planner = RLolaMonitor(spec)
        self._last_live_time = -np.inf

    def snapshot(self, *, step: int, time: float) -> RtlolaStateRef:
        return RtlolaStateRef(self.live.state(), self.spec_id, int(step), float(time))

    def branch_step(
        self,
        state: RtlolaStateRef,
        event: RtlolaEvent,
        action: RtlolaAction,
        budget: int,
    ) -> RtlolaStepResult:
        self._validate_state(state)
        self._validate_event(event)
        if event.time < state.time:
            raise ValueError(
                f"branch event time {event.time} is earlier than snapshot time {state.time}"
            )
        self._validate_budget_for_state(state, action, budget)
        config = action.make_config(budget)
        try:
            verdict, new_state = self.planner.accept_event_from_state(
                state.state,
                list(event.values),
                float(event.time),
                config,
            )
        except BaseException as exc:  # PyO3 PanicException is not an Exception.
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise RtlolaBindingError(
                "RTLola planner branch failed "
                f"(step={state.step + 1}, time={event.time}, action={action.name})"
            ) from exc
        result_ref = RtlolaStateRef(new_state, self.spec_id, state.step + 1, float(event.time))
        metrics = self.metrics(result_ref)
        return RtlolaStepResult(dict(verdict), result_ref, action.name, metrics)

    def live_step(
        self,
        event: RtlolaEvent,
        action: RtlolaAction,
        budget: int,
        *,
        step: int,
    ) -> RtlolaStepResult:
        self._validate_event(event)
        if event.time <= self._last_live_time:
            raise ValueError(
                f"live event time must be strictly increasing "
                f"(last={self._last_live_time}, current={event.time})"
            )
        self._validate_live_budget(action, budget)
        config = action.make_config(budget)
        try:
            verdict = self.live.accept_event(list(event.values), float(event.time), config)
        except BaseException as exc:  # PyO3 PanicException is not an Exception.
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise RtlolaBindingError(
                "RTLola live step failed "
                f"(step={step}, time={event.time}, action={action.name})"
            ) from exc
        self._validate_verdict(verdict, step=step)
        self._last_live_time = float(event.time)
        result_ref = RtlolaStateRef(self.live.state(), self.spec_id, int(step), float(event.time))
        metrics = self.metrics(result_ref)
        return RtlolaStepResult(dict(verdict), result_ref, action.name, metrics)

    def matrices(self, state: RtlolaStateRef) -> tuple[np.ndarray, np.ndarray]:
        self._validate_state(state)
        try:
            dyn = np.asarray(self.planner.state_zonotope(state.state, False), dtype=np.float64)
            total = np.asarray(self.planner.state_zonotope(state.state, True), dtype=np.float64)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise RtlolaBindingError(
                f"failed to extract RTLola zonotope matrices at step {state.step}"
            ) from exc
        return dyn, total

    def metrics(self, state: RtlolaStateRef) -> RtlolaMatrixMetrics:
        dyn, total = self.matrices(state)
        return matrix_metrics(dyn, total)

    def approx_loss(self, reference: RtlolaStateRef, candidate: RtlolaStateRef) -> float:
        """Return the binding-native approximation loss from reference to candidate."""
        self._validate_state(reference)
        self._validate_state(candidate)
        previous = self.planner.state()
        try:
            self.planner.apply_state(reference.state)
            loss = float(self.planner.approx_loss_state(candidate.state))
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise RtlolaBindingError(
                "failed to compute RTLola binding approximation loss "
                f"(reference_step={reference.step}, candidate_step={candidate.step})"
            ) from exc
        finally:
            self.planner.apply_state(previous)
        if not np.isfinite(loss):
            raise RtlolaBindingError(
                "RTLola binding approximation loss was non-finite"
            )
        return loss

    def _validate_state(self, state: RtlolaStateRef) -> None:
        if state.spec_id != self.spec_id:
            raise ValueError("RTLola state belongs to a different specification")

    def _validate_event(self, event: RtlolaEvent) -> None:
        if len(event.values) != self.event_arity:
            raise ValueError(f"expected {self.event_arity} event values, got {len(event.values)}")
        if not np.isfinite(event.time):
            raise ValueError("event time must be finite")
        for value in event.values:
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError("event contains non-finite float value")

    def _validate_budget_for_state(
        self,
        state: RtlolaStateRef,
        action: RtlolaAction,
        budget: int,
    ) -> None:
        if action.explicit_budget and budget < self.metrics(state).dimension:
            raise ValueError(
                "RTLola budget is below the current state-zonotope dimension "
                f"(action={action.name}, budget={budget}, "
                f"dimension={self.metrics(state).dimension})"
            )

    def _validate_live_budget(self, action: RtlolaAction, budget: int) -> None:
        if not action.explicit_budget:
            return
        dyn = np.asarray(self.live.current_zonotope(False), dtype=np.float64)
        dimension = int(dyn.shape[0]) if dyn.ndim == 2 else 0
        if budget < dimension:
            raise ValueError(
                "RTLola budget is below the current live state-zonotope dimension "
                f"(action={action.name}, budget={budget}, dimension={dimension})"
            )

    def _validate_verdict(self, verdict: dict[str, Any], *, step: int) -> None:
        missing = [key for key in self.expected_verdict_keys if key not in verdict]
        if missing:
            raise RuntimeError(
                f"RTLola verdict missing expected trigger keys at step {step}: {missing}"
            )
