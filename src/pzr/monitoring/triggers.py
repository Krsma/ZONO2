"""Trigger evaluation: overlap predicates and straddling checks.

Implements the paper's ternary trigger semantics (Theorem 3.1):
For an interval [l, u] and threshold c with overlap p:
  - above trigger: violation if (u - c) / (u - l) > p
  - below trigger: violation if (c - l) / (u - l) > p
  - degenerate (l == u): strict point comparison
"""

from __future__ import annotations

from pzr.monitoring.base import TriggerSpec, Verdict, VerdictStatus
from pzr.zonotope.core import Zonotope


def trigger_satisfaction_fraction(lower: float, upper: float, trigger: TriggerSpec) -> float:
    """Fraction of interval satisfying the trigger condition."""
    if upper < lower:
        raise ValueError("upper bound must be >= lower bound")
    if upper == lower:
        if trigger.direction == "above":
            return 1.0 if lower > trigger.threshold else 0.0
        return 1.0 if lower < trigger.threshold else 0.0

    width = upper - lower
    if trigger.direction == "above":
        fraction = (upper - trigger.threshold) / width
    else:
        fraction = (trigger.threshold - lower) / width
    return min(1.0, max(0.0, fraction))


def trigger_predicate_holds(lower: float, upper: float, trigger: TriggerSpec) -> bool:
    """Whether the paper's strict overlap trigger predicate holds."""
    fraction = trigger_satisfaction_fraction(lower, upper, trigger)
    if upper == lower:
        return fraction == 1.0
    return fraction > trigger.overlap


def trigger_straddles_threshold(lower: float, upper: float, trigger: TriggerSpec) -> bool:
    """Whether the interval geometrically straddles the trigger threshold."""
    return lower <= trigger.threshold <= upper


def evaluate_triggers(z: Zonotope, triggers: tuple[TriggerSpec, ...]) -> tuple[Verdict, ...]:
    """Evaluate all triggers over the interval hull."""
    lower, upper = z.interval_bounds()
    verdicts: list[Verdict] = []
    for trigger in triggers:
        lo = float(lower[trigger.state_index])
        hi = float(upper[trigger.state_index])
        status: VerdictStatus = "violation" if trigger_predicate_holds(lo, hi, trigger) else "safe"
        verdicts.append(Verdict(trigger, status, lo, hi))
    return tuple(verdicts)
