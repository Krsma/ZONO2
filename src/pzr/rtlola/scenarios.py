"""Scenario descriptors for RTLola-native benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from pzr.rtlola.engine import RtlolaEngine, RtlolaEvent, RtlolaStepResult
from pzr.rtlola.metrics import relevant_row_cost
from pzr.rtlola.omni import OMNI_EXPECTED_VERDICT_KEYS, OMNI_SPEC, generate_omni_events
from pzr.rtlola.robot_arm import (
    ARM_EXPECTED_VERDICT_KEYS,
    ARM_PUBLIC_STREAM_KEYS,
    ARM_RELEVANT_ROWS,
    ARM_SPEC,
    DEFAULT_TRACE_KIND,
    TRACE_KINDS,
    generate_robot_arm_events,
)


TraceFactory = Callable[[int, int, str], tuple[RtlolaEvent, ...]]
CostFactory = Callable[[RtlolaEngine, RtlolaStepResult], float]


@dataclass(frozen=True)
class RtlolaScenarioSpec:
    """Registered RTLola scenario metadata."""

    name: str
    spec: str
    event_arity: int
    trace_kinds: tuple[str, ...]
    default_trace_kind: str
    expected_verdict_keys: tuple[str, ...]
    public_stream_keys: tuple[str, ...]
    trigger_keys: tuple[str, ...]
    relevant_rows: tuple[int, ...]
    trace_factory: TraceFactory
    cost: CostFactory

    def generate_events(
        self,
        length: int,
        seed: int,
        trace_kind: str = "default",
    ) -> tuple[RtlolaEvent, ...]:
        selected = self.default_trace_kind if trace_kind == "default" else trace_kind
        if selected not in self.trace_kinds:
            raise ValueError(
                f"trace_kind for scenario {self.name!r} must be one of "
                f"{self.trace_kinds}, got {selected!r}"
            )
        return self.trace_factory(length, seed, selected)


def _default_cost(_engine: RtlolaEngine, step: RtlolaStepResult) -> float:
    return step.metrics.cost()


def _relevant_rows_cost(rows: Sequence[int]) -> CostFactory:
    def cost(engine: RtlolaEngine, step: RtlolaStepResult) -> float:
        matrix = engine.matrices(step.state)[0]
        return relevant_row_cost(matrix, rows)

    return cost


def _omni_trace_factory(length: int, seed: int, _trace_kind: str) -> tuple[RtlolaEvent, ...]:
    return generate_omni_events(length, seed=seed)


def _arm_trace_factory(length: int, seed: int, trace_kind: str) -> tuple[RtlolaEvent, ...]:
    return generate_robot_arm_events(length, seed=seed, trace_kind=trace_kind)


def registered_scenarios() -> tuple[RtlolaScenarioSpec, ...]:
    return (
        RtlolaScenarioSpec(
            name="omni_robot",
            spec=OMNI_SPEC,
            event_arity=3,
            trace_kinds=("default",),
            default_trace_kind="default",
            expected_verdict_keys=OMNI_EXPECTED_VERDICT_KEYS,
            public_stream_keys=OMNI_EXPECTED_VERDICT_KEYS,
            trigger_keys=OMNI_EXPECTED_VERDICT_KEYS,
            relevant_rows=(),
            trace_factory=_omni_trace_factory,
            cost=_default_cost,
        ),
        RtlolaScenarioSpec(
            name="robot_arm",
            spec=ARM_SPEC,
            event_arity=6,
            trace_kinds=TRACE_KINDS,
            default_trace_kind=DEFAULT_TRACE_KIND,
            expected_verdict_keys=ARM_EXPECTED_VERDICT_KEYS,
            public_stream_keys=ARM_PUBLIC_STREAM_KEYS,
            trigger_keys=ARM_EXPECTED_VERDICT_KEYS,
            relevant_rows=ARM_RELEVANT_ROWS,
            trace_factory=_arm_trace_factory,
            cost=_relevant_rows_cost(ARM_RELEVANT_ROWS),
        ),
    )


def scenario_by_name(name: str) -> RtlolaScenarioSpec:
    for scenario in registered_scenarios():
        if scenario.name == name:
            return scenario
    known = ", ".join(s.name for s in registered_scenarios())
    raise ValueError(f"unknown RTLola scenario {name!r}; expected one of: {known}")
