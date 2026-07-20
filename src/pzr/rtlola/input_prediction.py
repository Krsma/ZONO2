"""Causal, specification-independent input prediction for online MPC."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Sequence

import numpy as np

from pzr.rtlola.engine import RtlolaEvent


INPUT_PREDICTION_SCHEMA = "pzr.input-prediction.v1"
PREDICTOR_NAMES = ("hold", "linear", "quadratic")


@dataclass(frozen=True)
class InputPrediction:
    """One scheduled causal forecast and its per-channel realized predictor."""

    events: tuple[RtlolaEvent, ...]
    realized_predictors: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class InputPredictionDiagnostic:
    """Known-future diagnostic computed after, and outside, action selection."""

    schema: str
    predictor: str
    realized_predictor: str
    decision_step: int
    lead: int
    scheduled_time: float
    channel_index: int
    channel_name: str
    predicted_value: float
    actual_value: float
    error: float
    absolute_error: float
    squared_error: float


def predict_future_events(
    history: Sequence[RtlolaEvent],
    *,
    predictor: str,
    horizon: int,
    step_seconds: float,
    timestamp_channel_indices: Sequence[int] = (),
) -> InputPrediction:
    """Forecast ``horizon`` events from contiguous causal channel histories.

    Numeric channels use their actual observation times. Missing current inputs
    remain missing, nonnumeric inputs are held, and polynomial predictors degrade
    deterministically when too few contiguous numeric observations are available.
    """
    if predictor not in PREDICTOR_NAMES:
        raise ValueError(f"predictor must be one of {PREDICTOR_NAMES}")
    if not history:
        raise ValueError("input prediction requires the arrived current event")
    if horizon < 0:
        raise ValueError("prediction horizon must be non-negative")
    if not np.isfinite(step_seconds) or step_seconds <= 0.0:
        raise ValueError("prediction step seconds must be positive and finite")
    arity = len(history[-1].values)
    if any(len(event.values) != arity for event in history):
        raise ValueError("input prediction history has inconsistent event arity")
    timestamp_indices = tuple(int(index) for index in timestamp_channel_indices)
    if len(set(timestamp_indices)) != len(timestamp_indices) or any(
        index < 0 or index >= arity for index in timestamp_indices
    ):
        raise ValueError("timestamp channel indices must be unique and in range")

    channel_predictions = [
        _predict_channel(history, channel, predictor)
        for channel in range(arity)
    ]
    events: list[RtlolaEvent] = []
    realized_rows: list[tuple[str, ...]] = []
    current_time = float(history[-1].time)
    for lead in range(1, horizon + 1):
        scheduled_time = current_time + lead * step_seconds
        values: list[Any] = []
        realized: list[str] = []
        for channel, (predict, realized_predictor) in enumerate(channel_predictions):
            if channel in timestamp_indices:
                values.append(scheduled_time)
                realized.append("timestamp")
            else:
                values.append(predict(scheduled_time))
                realized.append(realized_predictor)
        events.append(RtlolaEvent(scheduled_time, tuple(values)))
        realized_rows.append(tuple(realized))
    return InputPrediction(tuple(events), tuple(realized_rows))


def prediction_diagnostics(
    prediction: InputPrediction,
    actual_future: Sequence[RtlolaEvent],
    *,
    predictor: str,
    decision_step: int,
    channel_names: Sequence[str],
) -> tuple[InputPredictionDiagnostic, ...]:
    """Compare forecasts to recorded futures without feeding them into search."""
    rows: list[InputPredictionDiagnostic] = []
    for lead, (predicted, actual, realized) in enumerate(
        zip(prediction.events, actual_future, prediction.realized_predictors),
        start=1,
    ):
        for channel, (predicted_value, actual_value) in enumerate(
            zip(predicted.values, actual.values)
        ):
            if not _is_numeric(predicted_value) or not _is_numeric(actual_value):
                continue
            predicted_float = float(predicted_value)
            actual_float = float(actual_value)
            error = predicted_float - actual_float
            rows.append(InputPredictionDiagnostic(
                schema=INPUT_PREDICTION_SCHEMA,
                predictor=predictor,
                realized_predictor=realized[channel],
                decision_step=int(decision_step),
                lead=lead,
                scheduled_time=predicted.time,
                channel_index=channel,
                channel_name=(
                    str(channel_names[channel])
                    if channel < len(channel_names) else f"input_{channel}"
                ),
                predicted_value=predicted_float,
                actual_value=actual_float,
                error=error,
                absolute_error=abs(error),
                squared_error=error * error,
            ))
    return tuple(rows)


def _predict_channel(
    history: Sequence[RtlolaEvent],
    channel: int,
    requested: str,
):
    current = history[-1].values[channel]
    if current is None:
        return (lambda _time: None), "absent"
    if not _is_numeric(current):
        return (lambda _time, value=current: value), "hold"

    contiguous: list[tuple[float, float]] = []
    for event in reversed(history):
        value = event.values[channel]
        if not _is_numeric(value):
            break
        contiguous.append((float(event.time), float(value)))
        if len(contiguous) == 3:
            break
    contiguous.reverse()
    distinct = _distinct_time_suffix(contiguous)
    degree = min(PREDICTOR_NAMES.index(requested), len(distinct) - 1)
    realized = PREDICTOR_NAMES[degree]
    points = distinct[-(degree + 1):]
    if degree == 0:
        value = points[-1][1]
        return (lambda _time, held=value: held), realized
    if degree == 1:
        (t0, y0), (t1, y1) = points
        slope = (y1 - y0) / (t1 - t0)
        return (lambda time, y=y1, t=t1, m=slope: y + m * (time - t)), realized
    (t0, y0), (t1, y1), (t2, y2) = points
    first_01 = (y1 - y0) / (t1 - t0)
    first_12 = (y2 - y1) / (t2 - t1)
    second = (first_12 - first_01) / (t2 - t0)
    return (
        lambda time, a=y0, b=first_01, c=second: (
            a + b * (time - t0) + c * (time - t0) * (time - t1)
        )
    ), realized


def _distinct_time_suffix(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    distinct: list[tuple[float, float]] = []
    for point in points:
        if distinct and point[0] <= distinct[-1][0]:
            distinct = [point]
        else:
            distinct.append(point)
    return distinct


def _is_numeric(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and np.isfinite(value)
