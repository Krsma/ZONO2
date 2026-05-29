"""Input prediction for MPC lookahead.

Predictors generate future input sequences for the MPC controller's
internal simulation. Prediction quality affects precision (not soundness).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, Sequence, TypeVar

InputT = TypeVar("InputT")


class InputPredictor(Protocol[InputT]):
    def predict(self, history: Sequence[InputT], horizon: int) -> tuple[InputT, ...]: ...


@dataclass(frozen=True)
class ConstantPredictor:
    """Repeat the last observed input for the entire horizon."""

    def predict(self, history: Sequence[InputT], horizon: int) -> tuple[InputT, ...]:
        if not history or horizon <= 0:
            return ()
        last = history[-1]
        if len(history) >= 2:
            dt = getattr(history[-1], "time", 1.0) - getattr(history[-2], "time", 0.0)
        else:
            dt = 1.0
        dt = max(dt, 1e-6)

        results = []
        base_time = getattr(last, "time", 0.0)
        for i in range(horizon):
            predicted = _with_time(last, base_time + dt * (i + 1))
            results.append(predicted)
        return tuple(results)


@dataclass(frozen=True)
class OraclePredictor:
    """Use the actual future inputs (for ablation only, not deployable)."""

    future: tuple

    def predict(self, history: Sequence[InputT], horizon: int) -> tuple[InputT, ...]:
        start = len(history)
        return tuple(self.future[start : start + horizon])


def _with_time(measurement, time: float):
    """Create a copy of the measurement with updated time."""
    try:
        return replace(measurement, time=time)
    except TypeError:
        return measurement
