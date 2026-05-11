"""Certificates returned by certified zonotope reductions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ReductionCertificate:
    """A lightweight record of why a reduction is admissible.

    The project treats certificates as executable bookkeeping rather than
    proof objects. A reducer may be selected by any policy only if it returns a
    certificate with ``is_sound=True``.
    """

    reducer: str
    original_generators: int
    reduced_generators: int
    is_sound: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReductionResult:
    """Result of applying a certified reducer."""

    original: Any
    reduced: Any
    certificate: ReductionCertificate
    stats: dict[str, Any] = field(default_factory=dict)
