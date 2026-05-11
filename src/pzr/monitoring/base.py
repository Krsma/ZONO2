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
    """Evaluate axis-aligned triggers over the zonotope interval hull."""

    lower, upper = zonotope.interval_bounds()
    verdicts: list[Verdict] = []
    for trigger in triggers:
        lo = float(lower[trigger.state_index])
        hi = float(upper[trigger.state_index])
        if trigger.direction == "above":
            if lo > trigger.threshold:
                status: VerdictStatus = "violation"
            elif hi <= trigger.threshold:
                status = "safe"
            else:
                status = "inconclusive"
        else:
            if hi < trigger.threshold:
                status = "violation"
            elif lo >= trigger.threshold:
                status = "safe"
            else:
                status = "inconclusive"
        verdicts.append(Verdict(trigger, status, lo, hi))
    return tuple(verdicts)
