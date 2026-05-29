"""Tests for monitoring protocol and trigger evaluation."""

import pytest

from pzr.monitoring.base import TriggerSpec
from pzr.monitoring.triggers import (
    evaluate_triggers,
    trigger_predicate_holds,
    trigger_satisfaction_fraction,
    trigger_straddles_threshold,
)
from pzr.zonotope.core import Zonotope


class TestTriggerSatisfaction:
    def test_above_fully_satisfied(self):
        t = TriggerSpec("test", 0, threshold=1.0, direction="above")
        assert trigger_satisfaction_fraction(2.0, 3.0, t) == pytest.approx(1.0)

    def test_above_not_satisfied(self):
        t = TriggerSpec("test", 0, threshold=5.0, direction="above")
        assert trigger_satisfaction_fraction(2.0, 3.0, t) == pytest.approx(0.0)

    def test_above_partial(self):
        t = TriggerSpec("test", 0, threshold=2.5, direction="above")
        assert trigger_satisfaction_fraction(2.0, 3.0, t) == pytest.approx(0.5)

    def test_below_fully_satisfied(self):
        t = TriggerSpec("test", 0, threshold=5.0, direction="below")
        assert trigger_satisfaction_fraction(2.0, 3.0, t) == pytest.approx(1.0)

    def test_below_partial(self):
        t = TriggerSpec("test", 0, threshold=2.5, direction="below")
        assert trigger_satisfaction_fraction(2.0, 3.0, t) == pytest.approx(0.5)

    def test_degenerate_above_satisfied(self):
        t = TriggerSpec("test", 0, threshold=1.0, direction="above")
        assert trigger_satisfaction_fraction(2.0, 2.0, t) == 1.0

    def test_degenerate_above_not_satisfied(self):
        t = TriggerSpec("test", 0, threshold=3.0, direction="above")
        assert trigger_satisfaction_fraction(2.0, 2.0, t) == 0.0

    def test_degenerate_below_satisfied(self):
        t = TriggerSpec("test", 0, threshold=3.0, direction="below")
        assert trigger_satisfaction_fraction(2.0, 2.0, t) == 1.0


class TestTriggerPredicate:
    def test_above_with_overlap_zero(self):
        t = TriggerSpec("test", 0, threshold=2.5, direction="above", overlap=0.0)
        assert trigger_predicate_holds(2.0, 3.0, t) is True

    def test_above_with_high_overlap(self):
        t = TriggerSpec("test", 0, threshold=2.5, direction="above", overlap=0.6)
        # fraction is 0.5, overlap is 0.6 -> not a violation
        assert trigger_predicate_holds(2.0, 3.0, t) is False

    def test_degenerate_violation(self):
        t = TriggerSpec("test", 0, threshold=1.0, direction="above")
        assert trigger_predicate_holds(2.0, 2.0, t) is True

    def test_degenerate_safe(self):
        t = TriggerSpec("test", 0, threshold=3.0, direction="above")
        assert trigger_predicate_holds(2.0, 2.0, t) is False


class TestStraddling:
    def test_straddles(self):
        t = TriggerSpec("test", 0, threshold=2.5)
        assert trigger_straddles_threshold(2.0, 3.0, t) is True

    def test_not_straddles_above(self):
        t = TriggerSpec("test", 0, threshold=4.0)
        assert trigger_straddles_threshold(2.0, 3.0, t) is False

    def test_boundary(self):
        t = TriggerSpec("test", 0, threshold=3.0)
        assert trigger_straddles_threshold(2.0, 3.0, t) is True


class TestEvaluateTriggers:
    def test_single_safe(self):
        z = Zonotope([0.0, 0.0], [[0.5, 0.0], [0.0, 0.5]])
        triggers = (TriggerSpec("x_above", 0, 2.0, "above"),)
        verdicts = evaluate_triggers(z, triggers)
        assert len(verdicts) == 1
        assert verdicts[0].status == "safe"

    def test_single_violation(self):
        z = Zonotope([3.0, 0.0], [[0.1, 0.0], [0.0, 0.1]])
        triggers = (TriggerSpec("x_above", 0, 2.0, "above"),)
        verdicts = evaluate_triggers(z, triggers)
        assert verdicts[0].status == "violation"

    def test_multiple_triggers(self):
        z = Zonotope([3.0, 0.0], [[0.1, 0.0], [0.0, 0.1]])
        triggers = (
            TriggerSpec("x_above", 0, 2.0, "above"),
            TriggerSpec("y_above", 1, 2.0, "above"),
        )
        verdicts = evaluate_triggers(z, triggers)
        assert verdicts[0].status == "violation"
        assert verdicts[1].status == "safe"
