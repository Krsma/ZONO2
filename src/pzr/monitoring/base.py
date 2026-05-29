"""Monitor adapter protocol and trigger evaluation.

The monitor protocol defines a black-box interface for stepping a zonotope-
based uncertainty representation through system dynamics. Controllers can
step, clone, and replace the zonotope, but cannot inspect monitor equations.

Trigger semantics follow the paper's overlap-aware RLola predicates
(Finkbeiner et al., arXiv:2601.11358v1, Theorem 3.1).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol, TypeVar

from pzr.zonotope.core import Zonotope

Direction = Literal["above", "below"]
VerdictStatus = Literal["safe", "violation"]

InputT = TypeVar("InputT")


@dataclass(frozen=True)
class TriggerSpec:
    """Axis-aligned trigger on one state dimension."""

    name: str
    state_index: int
    threshold: float
    direction: Direction = "above"
    overlap: float = 0.0


@dataclass(frozen=True)
class Verdict:
    trigger: TriggerSpec
    status: VerdictStatus
    lower: float
    upper: float


@dataclass(frozen=True)
class MonitorState:
    """Monitor state with explicit calibration generator tracking."""

    zonotope: Zonotope
    step: int = 0
    calibration_indices: tuple[int, ...] = ()
    payload: Any = None

    def with_zonotope(
        self,
        zonotope: Zonotope,
        calibration_indices: tuple[int, ...] | None = None,
    ) -> MonitorState:
        return replace(
            self,
            zonotope=zonotope,
            calibration_indices=calibration_indices if calibration_indices is not None else self.calibration_indices,
        )


@dataclass(frozen=True)
class MonitorResult:
    state: MonitorState
    verdicts: tuple[Verdict, ...]


class MonitorAdapter(Protocol[InputT]):
    """Black-box monitor adapter protocol."""

    @property
    def triggers(self) -> tuple[TriggerSpec, ...]: ...

    def initial_state(self) -> MonitorState: ...

    def step(self, state: MonitorState, measurement: InputT) -> MonitorResult: ...

    def clone_state(self, state: MonitorState) -> MonitorState: ...

    def replace_zonotope(self, state: MonitorState, zonotope: Zonotope) -> MonitorState: ...

    @property
    def num_calibration_generators(self) -> int: ...
