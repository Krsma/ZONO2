"""Black-box monitor adapter protocol and trigger evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol, TypeVar

from pzr.core.zonotope import GeneratorRequirement, Zonotope

Direction = Literal["above", "below"]
VerdictStatus = Literal["safe", "violation", "inconclusive"]

InputT = TypeVar("InputT")


@dataclass(frozen=True)
class TriggerSpec:
    """Axis-aligned trigger over one stream coordinate."""

    name: str
    state_index: int
    threshold: float
    direction: Direction = "above"
    overlap: float = 0.0


@dataclass(frozen=True)
class Verdict:
    """Three-valued verdict for one trigger at one monitor step."""

    trigger: TriggerSpec
    status: VerdictStatus
    lower: float
    upper: float


@dataclass(frozen=True)
class MonitorState:
    """Opaque monitor state with a replaceable zonotope component."""

    zonotope: Zonotope
    step: int = 0
    payload: Any = None

    def with_zonotope(self, zonotope: Zonotope) -> "MonitorState":
        return replace(self, zonotope=zonotope)


@dataclass(frozen=True)
class MonitorResult:
    """State and verdicts produced by one monitor step."""

    state: MonitorState
    verdicts: tuple[Verdict, ...]


class MonitorAdapter(Protocol[InputT]):
    """Black-box monitor adapter.

    Controllers may step, clone, and replace the zonotope part of the state,
    but should not inspect monitor equations.
    """

    @property
    def triggers(self) -> tuple[TriggerSpec, ...]:
        ...

    def initial_state(self) -> MonitorState:
        ...

    def step(self, state: MonitorState, measurement: InputT) -> MonitorResult:
        ...

    def clone_state(self, state: MonitorState) -> MonitorState:
        ...

    def replace_zonotope(self, state: MonitorState, zonotope: Zonotope) -> MonitorState:
        ...

    def required_generator_metadata(
        self,
        state: MonitorState,
    ) -> tuple[GeneratorRequirement, ...]:
        ...


def evaluate_triggers(zonotope: Zonotope, triggers: tuple[TriggerSpec, ...]) -> tuple[Verdict, ...]:
    """Evaluate paper-style overlap-aware triggers over the interval hull."""

    lower, upper = zonotope.interval_bounds()
    verdicts: list[Verdict] = []
    for trigger in triggers:
        lo = float(lower[trigger.state_index])
        hi = float(upper[trigger.state_index])
        status: VerdictStatus = (
            "violation" if trigger_predicate_holds(lo, hi, trigger) else "safe"
        )
        verdicts.append(Verdict(trigger, status, lo, hi))
    return tuple(verdicts)


def trigger_satisfaction_fraction(lower: float, upper: float, trigger: TriggerSpec) -> float:
    """Fraction of an interval satisfying an axis-aligned trigger condition."""

    _validate_trigger_overlap(trigger)
    lo = float(lower)
    hi = float(upper)
    if hi < lo:
        raise ValueError("interval upper bound must be greater than or equal to lower bound")

    threshold = trigger.threshold
    if hi == lo:
        if trigger.direction == "above":
            return 1.0 if lo > threshold else 0.0
        return 1.0 if lo < threshold else 0.0

    width = hi - lo
    if trigger.direction == "above":
        fraction = (hi - threshold) / width
    else:
        fraction = (threshold - lo) / width
    return float(min(1.0, max(0.0, fraction)))


def trigger_predicate_holds(lower: float, upper: float, trigger: TriggerSpec) -> bool:
    """Return whether the paper's strict overlap trigger predicate holds."""

    fraction = trigger_satisfaction_fraction(lower, upper, trigger)
    if float(upper) == float(lower):
        return fraction == 1.0
    return fraction > trigger.overlap


def trigger_straddles_threshold(lower: float, upper: float, trigger: TriggerSpec) -> bool:
    """Return whether the interval geometrically straddles the trigger threshold."""

    lo = float(lower)
    hi = float(upper)
    if hi < lo:
        raise ValueError("interval upper bound must be greater than or equal to lower bound")
    return lo <= trigger.threshold <= hi


def _validate_trigger_overlap(trigger: TriggerSpec) -> None:
    if not 0.0 <= trigger.overlap <= 1.0:
        raise ValueError("trigger overlap must be in [0, 1]")
