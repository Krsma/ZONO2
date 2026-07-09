"""Scenario descriptors for RTLola-native benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pzr.rtlola.engine import RtlolaEvent
from pzr.rtlola.omni import (
    OMNI_DEFAULT_TRACE_KIND,
    OMNI_EXPECTED_VERDICT_KEYS,
    OMNI_PUBLIC_STREAM_KEYS,
    OMNI_SPEC,
    OMNI_TRACE_KINDS,
    generate_omni_events,
)
from pzr.rtlola.robot_arm import (
    ARM_PUBLIC_STREAM_KEYS,
    ARM_SPEC,
    ARM_TRIGGER_KEYS,
    ARM_TRIGGER_LABELS,
    DEFAULT_TRACE_KIND,
    RLOLAEVAL_REVISION,
    TRACE_KINDS,
    generate_robot_arm_events,
)


TraceFactory = Callable[[int, int, str], tuple[RtlolaEvent, ...]]


@dataclass(frozen=True)
class RtlolaTrace:
    """Immutable events and provenance for one scenario trace."""

    scenario: str
    trace_kind: str
    seed: int
    events: tuple[RtlolaEvent, ...]


@dataclass(frozen=True)
class RtlolaScenario:
    """Registered RTLola specification and trace adapter."""

    name: str
    spec: str
    event_arity: int
    trace_kinds: tuple[str, ...]
    default_trace_kind: str
    expected_verdict_keys: tuple[str, ...]
    public_stream_keys: tuple[str, ...]
    trigger_keys: tuple[str, ...]
    trace_factory: TraceFactory
    source_revision: str | None = None
    trigger_labels: tuple[str, ...] = ()

    def generate_trace(
        self,
        length: int,
        seed: int,
        trace_kind: str = "default",
    ) -> RtlolaTrace:
        selected = self.default_trace_kind if trace_kind == "default" else trace_kind
        if selected not in self.trace_kinds:
            raise ValueError(
                f"trace_kind for scenario {self.name!r} must be one of "
                f"{self.trace_kinds}, got {selected!r}"
            )
        return RtlolaTrace(
            scenario=self.name,
            trace_kind=selected,
            seed=int(seed),
            events=self.trace_factory(length, seed, selected),
        )

    def generate_events(
        self,
        length: int,
        seed: int,
        trace_kind: str = "default",
    ) -> tuple[RtlolaEvent, ...]:
        return self.generate_trace(length, seed, trace_kind).events


def _omni_trace_factory(
    length: int,
    seed: int,
    trace_kind: str,
) -> tuple[RtlolaEvent, ...]:
    return generate_omni_events(length, seed=seed, trace_kind=trace_kind)


def _arm_trace_factory(length: int, seed: int, trace_kind: str) -> tuple[RtlolaEvent, ...]:
    return generate_robot_arm_events(length, seed=seed, trace_kind=trace_kind)


def registered_scenarios() -> tuple[RtlolaScenario, ...]:
    return (
        RtlolaScenario(
            name="omni_robot",
            spec=OMNI_SPEC,
            event_arity=3,
            trace_kinds=OMNI_TRACE_KINDS,
            default_trace_kind=OMNI_DEFAULT_TRACE_KIND,
            expected_verdict_keys=(
                *OMNI_EXPECTED_VERDICT_KEYS,
                *OMNI_PUBLIC_STREAM_KEYS,
            ),
            public_stream_keys=OMNI_PUBLIC_STREAM_KEYS,
            trigger_keys=OMNI_EXPECTED_VERDICT_KEYS,
            trace_factory=_omni_trace_factory,
            trigger_labels=OMNI_EXPECTED_VERDICT_KEYS,
        ),
        RtlolaScenario(
            name="robot_arm",
            spec=ARM_SPEC,
            event_arity=13,
            trace_kinds=TRACE_KINDS,
            default_trace_kind=DEFAULT_TRACE_KIND,
            expected_verdict_keys=ARM_PUBLIC_STREAM_KEYS,
            public_stream_keys=ARM_PUBLIC_STREAM_KEYS,
            trigger_keys=ARM_TRIGGER_KEYS,
            trace_factory=_arm_trace_factory,
            source_revision=RLOLAEVAL_REVISION,
            trigger_labels=tuple(
                ARM_TRIGGER_LABELS[key] for key in ARM_TRIGGER_KEYS
            ),
        ),
    )


def scenario_by_name(name: str) -> RtlolaScenario:
    for scenario in registered_scenarios():
        if scenario.name == name:
            return scenario
    known = ", ".join(s.name for s in registered_scenarios())
    raise ValueError(f"unknown RTLola scenario {name!r}; expected one of: {known}")
