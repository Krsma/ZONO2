from __future__ import annotations

import pytest

from pzr.rtlola.engine import RtlolaEvent
from pzr.rtlola.input_prediction import (
    prediction_diagnostics,
    predict_future_events,
)


def test_polynomial_prediction_uses_irregular_observation_times():
    history = (
        RtlolaEvent(0.0, (0.0, 1.0, "mode-a", None)),
        RtlolaEvent(0.4, (0.4, 1.16, "mode-b", 5.0)),
        RtlolaEvent(1.0, (1.0, 2.0, "mode-c", None)),
    )

    predicted = predict_future_events(
        history,
        predictor="quadratic",
        horizon=3,
        step_seconds=0.1,
        timestamp_channel_indices=(0,),
    )

    assert [event.time for event in predicted.events] == pytest.approx([1.1, 1.2, 1.3])
    assert [event.values[0] for event in predicted.events] == pytest.approx([1.1, 1.2, 1.3])
    assert [event.values[1] for event in predicted.events] == pytest.approx([
        1.0 + 1.1**2, 1.0 + 1.2**2, 1.0 + 1.3**2,
    ])
    assert [event.values[2] for event in predicted.events] == ["mode-c"] * 3
    assert [event.values[3] for event in predicted.events] == [None] * 3
    assert predicted.realized_predictors[0] == (
        "timestamp", "quadratic", "hold", "absent",
    )


def test_prediction_degrades_and_requires_contiguous_numeric_history():
    startup = predict_future_events(
        (RtlolaEvent(2.0, (3.0,)),),
        predictor="quadratic",
        horizon=1,
        step_seconds=0.5,
    )
    assert startup.events[0].values == (3.0,)
    assert startup.realized_predictors[0] == ("hold",)

    broken = predict_future_events(
        (
            RtlolaEvent(0.0, (1.0,)),
            RtlolaEvent(1.0, (None,)),
            RtlolaEvent(2.0, (5.0,)),
        ),
        predictor="linear",
        horizon=1,
        step_seconds=1.0,
    )
    assert broken.events[0].values == (5.0,)
    assert broken.realized_predictors[0] == ("hold",)


def test_prediction_is_deterministic_and_diagnostics_do_not_affect_forecast():
    history = (RtlolaEvent(0.0, (0.0,)), RtlolaEvent(1.0, (2.0,)))
    first = predict_future_events(
        history, predictor="linear", horizon=2, step_seconds=1.0,
    )
    second = predict_future_events(
        history, predictor="linear", horizon=2, step_seconds=1.0,
    )
    assert first == second

    original = prediction_diagnostics(
        first,
        (RtlolaEvent(2.0, (4.0,)), RtlolaEvent(3.0, (6.0,))),
        predictor="linear",
        decision_step=1,
        channel_names=("x",),
    )
    altered = prediction_diagnostics(
        first,
        (RtlolaEvent(2.0, (400.0,)), RtlolaEvent(3.0, (-6.0,))),
        predictor="linear",
        decision_step=1,
        channel_names=("x",),
    )
    assert first == second
    assert original != altered
