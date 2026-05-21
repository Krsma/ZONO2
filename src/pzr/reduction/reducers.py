"""Certified reducer implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from pzr.core.certificates import ReductionCertificate, ReductionResult
from pzr.core.zonotope import (
    GeneratorKind,
    GeneratorMetadata,
    GeneratorRequirement,
    Zonotope,
)
from pzr.reduction.base import Reducer, ReductionContext
from pzr.reduction.scoring import (
    calibration_aware_scores,
    norm_scores,
    trigger_influence_scores,
)

ScoreFunction = Callable[[Zonotope, ReductionContext | None], NDArray[np.float64]]


def _certificate(
    name: str,
    original: Zonotope,
    reduced: Zonotope,
    reason: str,
    details: dict[str, object] | None = None,
) -> ReductionCertificate:
    return ReductionCertificate(
        reducer=name,
        original_generators=original.generator_count,
        reduced_generators=reduced.generator_count,
        is_sound=True,
        reason=reason,
        details={} if details is None else details,
    )


def _axis_box_generators(
    radius: NDArray[np.float64],
    *,
    tol: float = 1e-12,
) -> tuple[NDArray[np.float64], tuple[GeneratorMetadata, ...]]:
    active_axes = [index for index, value in enumerate(radius) if abs(value) > tol]
    generators = np.zeros((radius.size, len(active_axes)), dtype=float)
    metadata: list[GeneratorMetadata] = []
    for column, axis in enumerate(active_axes):
        generators[axis, column] = radius[axis]
        metadata.append(
            GeneratorMetadata(
                kind=GeneratorKind.SYNTHETIC,
                source=f"box_axis_{axis}",
                age=0,
            )
        )
    return generators, tuple(metadata)


@dataclass(frozen=True)
class IdentityReducer:
    """Return the original zonotope when it already satisfies the budget."""

    name: str = "no_reduction"

    def reduce(
        self,
        zonotope: Zonotope,
        budget: int,
        context: ReductionContext | None = None,
    ) -> ReductionResult:
        _ = context
        if budget < 0:
            raise ValueError("budget must be non-negative")
        if zonotope.generator_count > budget:
            raise ValueError(
                f"no-op reducer cannot reduce {zonotope.generator_count} generators to {budget}"
            )
        certificate = _certificate(
            self.name,
            zonotope,
            zonotope,
            "No reduction was needed because the state already satisfies the budget.",
        )
        return ReductionResult(zonotope, zonotope, certificate)


@dataclass(frozen=True)
class ProtectedReducer:
    """Reducer wrapper that keeps monitor-required generators exact."""

    base: Reducer
    require_existing: bool = True
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(self, "name", self.base.name)

    def reduce(
        self,
        zonotope: Zonotope,
        budget: int,
        context: ReductionContext | None = None,
    ) -> ReductionResult:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        ctx = context or ReductionContext()
        requirements = tuple(ctx.required_generators)
        if not requirements:
            return self.base.reduce(zonotope, budget, ctx)

        protected = _required_generator_indices(
            zonotope.metadata,
            requirements,
            require_existing=self.require_existing,
        )
        if len(protected) > budget:
            raise ValueError(
                f"{self.base.name} cannot preserve {len(protected)} required generators "
                f"within budget {budget}"
            )
        if zonotope.generator_count <= budget:
            certificate = _certificate(
                self.name,
                zonotope,
                zonotope,
                "No reduction was needed because the state already satisfies the budget.",
                {"protected_indices": protected, "base_reducer": self.base.name},
            )
            return ReductionResult(
                zonotope,
                zonotope,
                certificate,
                {"protected": len(protected), "base": self.base.name},
            )

        residual = tuple(
            index for index in range(zonotope.generator_count) if index not in protected
        )
        residual_budget = budget - len(protected)
        residual_zonotope = Zonotope(
            np.zeros(zonotope.dimension, dtype=float),
            zonotope.generators[:, residual]
            if residual
            else np.zeros((zonotope.dimension, 0), dtype=float),
            tuple(zonotope.metadata[index] for index in residual),
        )
        residual_result = self.base.reduce(residual_zonotope, residual_budget, ctx)
        if not residual_result.certificate.is_sound:
            raise ValueError(f"wrapped reducer {self.base.name} returned an unsound certificate")
        protected_generators = (
            zonotope.generators[:, protected]
            if protected
            else np.zeros((zonotope.dimension, 0), dtype=float)
        )
        reduced_generators = (
            np.hstack([protected_generators, residual_result.reduced.generators])
            if residual_result.reduced.generator_count
            else protected_generators
        )
        reduced_metadata = (
            tuple(zonotope.metadata[index] for index in protected)
            + residual_result.reduced.metadata
        )
        reduced = Zonotope(zonotope.center, reduced_generators, reduced_metadata)
        certificate = _certificate(
            self.name,
            zonotope,
            reduced,
            "Required generators were preserved exactly; the residual zonotope "
            "was reduced by the wrapped certified reducer.",
            {
                "base_reducer": self.base.name,
                "protected_indices": protected,
                "residual_generators": len(residual),
                "residual_reduced_generators": residual_result.reduced.generator_count,
                "base_certificate": residual_result.certificate.details,
            },
        )
        return ReductionResult(
            zonotope,
            reduced,
            certificate,
            {
                "protected": len(protected),
                "base": self.base.name,
                "base_stats": residual_result.stats,
            },
        )


@dataclass(frozen=True)
class TargetBudgetReducer:
    """Reducer wrapper that spends at most a fixed target budget."""

    base: Reducer
    target_budget: int
    name: str = ""

    def __post_init__(self) -> None:
        if self.target_budget < 0:
            raise ValueError("target_budget must be non-negative")
        if not self.name:
            object.__setattr__(self, "name", f"{self.base.name}{self.target_budget}")

    def reduce(
        self,
        zonotope: Zonotope,
        budget: int,
        context: ReductionContext | None = None,
    ) -> ReductionResult:
        return self.base.reduce(zonotope, min(budget, self.target_budget), context)


@dataclass(frozen=True)
class BudgetSlackReducer:
    """Reducer wrapper that deliberately leaves generator budget headroom."""

    base: Reducer
    slack: int
    name: str = ""

    def __post_init__(self) -> None:
        if self.slack < 0:
            raise ValueError("slack must be non-negative")
        if not self.name:
            object.__setattr__(self, "name", f"{self.base.name}_slack{self.slack}")

    def reduce(
        self,
        zonotope: Zonotope,
        budget: int,
        context: ReductionContext | None = None,
    ) -> ReductionResult:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        effective_budget = max(0, budget - self.slack)
        result = self.base.reduce(zonotope, effective_budget, context)
        certificate = _certificate(
            self.name,
            zonotope,
            result.reduced,
            "Applied wrapped reducer with deliberate generator budget slack.",
            {
                "base_reducer": self.base.name,
                "requested_budget": budget,
                "effective_budget": effective_budget,
                "slack": self.slack,
                "base_certificate": result.certificate.details,
            },
        )
        return ReductionResult(
            zonotope,
            result.reduced,
            certificate,
            {"base": self.base.name, "slack": self.slack, "base_stats": result.stats},
        )


@dataclass(frozen=True)
class BoxReducer:
    """Reduce a zonotope to its interval hull represented as an axis-aligned box."""

    tol: float = 1e-12
    name: str = "box"

    def reduce(
        self,
        zonotope: Zonotope,
        budget: int,
        context: ReductionContext | None = None,
    ) -> ReductionResult:
        _ = context
        if budget < 0:
            raise ValueError("budget must be non-negative")
        generators, metadata = _axis_box_generators(zonotope.interval_radius(), tol=self.tol)
        if generators.shape[1] > budget:
            raise ValueError(
                f"box reducer needs {generators.shape[1]} generators for this state, budget is {budget}"
            )
        reduced = Zonotope(zonotope.center, generators, metadata)
        certificate = _certificate(
            self.name,
            zonotope,
            reduced,
            "The interval hull contains the original zonotope component-wise.",
            {"active_axes": generators.shape[1]},
        )
        return ReductionResult(zonotope, reduced, certificate)


@dataclass(frozen=True)
class ScoredKeepReducer:
    """Keep high-scoring generators and box-merge the discarded generators."""

    score: ScoreFunction = calibration_aware_scores
    tol: float = 1e-12
    name: str = "scored_keep"

    @classmethod
    def by_norm(cls) -> "ScoredKeepReducer":
        return cls(score=norm_scores, name="keep_norm")

    @classmethod
    def calibration_aware(cls) -> "ScoredKeepReducer":
        return cls(score=calibration_aware_scores, name="keep_calibration_aware")

    @classmethod
    def trigger_influence(cls) -> "ScoredKeepReducer":
        return cls(score=trigger_influence_scores, name="keep_trigger")

    def reduce(
        self,
        zonotope: Zonotope,
        budget: int,
        context: ReductionContext | None = None,
    ) -> ReductionResult:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        if zonotope.generator_count <= budget:
            certificate = _certificate(
                self.name,
                zonotope,
                zonotope,
                "No reduction was needed because the state already satisfies the budget.",
            )
            return ReductionResult(zonotope, zonotope, certificate, {"kept": zonotope.generator_count})

        scores = np.asarray(self.score(zonotope, context), dtype=float)
        if scores.shape != (zonotope.generator_count,):
            raise ValueError("score function returned the wrong shape")

        order = tuple(int(i) for i in np.argsort(-scores, kind="mergesort"))
        best: tuple[Zonotope, tuple[int, ...], int] | None = None

        for keep_count in range(min(budget, zonotope.generator_count), -1, -1):
            keep = order[:keep_count]
            discard = order[keep_count:]
            discarded_generators = zonotope.generators[:, discard]
            discarded_radius = (
                np.sum(np.abs(discarded_generators), axis=1)
                if discard
                else np.zeros(zonotope.dimension)
            )
            box_generators, box_metadata = _axis_box_generators(discarded_radius, tol=self.tol)
            total_generators = keep_count + box_generators.shape[1]
            if total_generators <= budget:
                kept_generators = (
                    zonotope.generators[:, keep]
                    if keep
                    else np.zeros((zonotope.dimension, 0), dtype=float)
                )
                reduced_generators = (
                    np.hstack([kept_generators, box_generators])
                    if box_generators.shape[1]
                    else kept_generators
                )
                reduced_metadata = tuple(zonotope.metadata[i] for i in keep) + box_metadata
                best = (Zonotope(zonotope.center, reduced_generators, reduced_metadata), keep, len(discard))
                break

        if best is None:
            raise ValueError(
                f"{self.name} cannot soundly reduce this {zonotope.dimension}D state to budget {budget}"
            )

        reduced, kept, discarded_count = best
        certificate = _certificate(
            self.name,
            zonotope,
            reduced,
            "Kept selected generators exactly and enclosed discarded generators by their interval hull.",
            {
                "kept_indices": kept,
                "discarded_count": discarded_count,
                "score_min": float(np.min(scores)) if scores.size else 0.0,
                "score_max": float(np.max(scores)) if scores.size else 0.0,
            },
        )
        return ReductionResult(
            zonotope,
            reduced,
            certificate,
            {"kept": len(kept), "discarded": discarded_count},
        )


def _required_generator_indices(
    metadata: tuple[GeneratorMetadata, ...],
    requirements: tuple[GeneratorRequirement, ...],
    *,
    require_existing: bool,
) -> tuple[int, ...]:
    protected: list[int] = []
    for requirement in requirements:
        matches = [
            index for index, meta in enumerate(metadata) if requirement.matches(meta)
        ]
        if require_existing and not matches:
            raise ValueError(f"required generator metadata not present: {requirement}")
        protected.extend(matches)
    return tuple(sorted(set(protected)))
