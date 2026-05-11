"""Reducer interfaces shared by static and predictive policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pzr.core.certificates import ReductionResult
from pzr.core.zonotope import GeneratorRequirement, Zonotope
from pzr.monitoring.base import TriggerSpec


@dataclass(frozen=True)
class ReductionContext:
    """Optional monitor-aware information available to reducers."""

    step: int = 0
    triggers: tuple[TriggerSpec, ...] = ()
    preserve_calibration: bool = True
    required_generators: tuple[GeneratorRequirement, ...] = ()


class Reducer(Protocol):
    """A certified zonotope reduction strategy."""

    name: str

    def reduce(
        self,
        zonotope: Zonotope,
        budget: int,
        context: ReductionContext | None = None,
    ) -> ReductionResult:
        """Return a sound over-approximation with at most ``budget`` generators."""
