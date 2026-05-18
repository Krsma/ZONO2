import pytest

from pzr.core.zonotope import Zonotope
from pzr.monitoring.base import (
    TriggerSpec,
    evaluate_triggers,
    trigger_satisfaction_fraction,
    trigger_straddles_threshold,
)


def _interval_zonotope(lower: float, upper: float) -> Zonotope:
    center = (lower + upper) / 2.0
    radius = (upper - lower) / 2.0
    return Zonotope([center], [[radius]])


def _status(lower: float, upper: float, trigger: TriggerSpec) -> str:
    verdict, = evaluate_triggers(_interval_zonotope(lower, upper), (trigger,))
    return verdict.status


def test_above_trigger_uses_strict_overlap_fraction() -> None:
    trigger = TriggerSpec("above", 0, 1.0, direction="above", overlap=0.25)

    assert _status(1.5, 2.0, trigger) == "violation"
    assert _status(0.0, 1.0, trigger) == "safe"
    assert _status(0.0, 2.0, trigger) == "violation"
    assert trigger_satisfaction_fraction(0.0, 2.0, trigger) == 0.5


def test_below_trigger_uses_strict_overlap_fraction() -> None:
    trigger = TriggerSpec("below", 0, 1.0, direction="below", overlap=0.25)

    assert _status(0.0, 0.5, trigger) == "violation"
    assert _status(1.0, 2.0, trigger) == "safe"
    assert _status(0.0, 2.0, trigger) == "violation"
    assert trigger_satisfaction_fraction(0.0, 2.0, trigger) == 0.5


def test_fraction_equal_to_overlap_is_safe() -> None:
    above = TriggerSpec("above", 0, 1.0, direction="above", overlap=0.5)
    below = TriggerSpec("below", 0, 1.0, direction="below", overlap=0.5)

    assert _status(0.0, 2.0, above) == "safe"
    assert _status(0.0, 2.0, below) == "safe"


def test_zero_width_intervals_use_strict_point_semantics() -> None:
    above = TriggerSpec("above", 0, 1.0, direction="above", overlap=1.0)
    below = TriggerSpec("below", 0, 1.0, direction="below", overlap=1.0)

    assert _status(0.5, 0.5, above) == "safe"
    assert _status(1.0, 1.0, above) == "safe"
    assert _status(1.5, 1.5, above) == "violation"
    assert _status(0.5, 0.5, below) == "violation"
    assert _status(1.0, 1.0, below) == "safe"
    assert _status(1.5, 1.5, below) == "safe"


def test_invalid_overlap_values_are_rejected() -> None:
    for overlap in (-0.01, 1.01):
        trigger = TriggerSpec("bad", 0, 1.0, overlap=overlap)
        with pytest.raises(ValueError, match="overlap"):
            evaluate_triggers(_interval_zonotope(0.0, 2.0), (trigger,))


def test_threshold_straddling_is_separate_from_trigger_semantics() -> None:
    trigger = TriggerSpec("above", 0, 1.0, direction="above", overlap=0.5)

    assert trigger_straddles_threshold(0.0, 2.0, trigger)
    assert _status(0.0, 2.0, trigger) == "safe"
