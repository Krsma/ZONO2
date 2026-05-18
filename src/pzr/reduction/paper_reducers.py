"""CORA-style zonotope reducers used as paper baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import qr

from pzr.core.certificates import ReductionResult
from pzr.core.zonotope import GeneratorKind, GeneratorMetadata, Zonotope
from pzr.monitoring.base import trigger_straddles_threshold
from pzr.reduction.base import Reducer, ReductionContext
from pzr.reduction.reducers import BoxReducer, _axis_box_generators, _certificate


def girard_scores(
    zonotope: Zonotope,
    context: ReductionContext | None = None,
) -> NDArray[np.float64]:
    """Generator scores used by Girard-style reduction."""

    _ = context
    if zonotope.generator_count == 0:
        return np.zeros(0)
    l1 = np.sum(np.abs(zonotope.generators), axis=0)
    linf = np.max(np.abs(zonotope.generators), axis=0)
    return l1 - linf


def l2_scores(
    zonotope: Zonotope,
    context: ReductionContext | None = None,
) -> NDArray[np.float64]:
    """Generator scores used by Combastel-style reduction."""

    _ = context
    if zonotope.generator_count == 0:
        return np.zeros(0)
    return np.linalg.norm(zonotope.generators, axis=0)


ScoreFunction = Callable[[Zonotope, ReductionContext | None], NDArray[np.float64]]
BasisFunction = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class GirardReducer:
    """CORA-style Girard reduction: keep high metric generators, box the rest."""

    tol: float = 1e-12
    name: str = "girard"

    def reduce(
        self,
        zonotope: Zonotope,
        budget: int,
        context: ReductionContext | None = None,
    ) -> ReductionResult:
        return _reduce_keep_and_box(
            zonotope,
            budget,
            context,
            score=girard_scores,
            name=self.name,
            tol=self.tol,
            reason="Kept generators selected by Girard's l1-minus-linf metric and boxed the rest.",
        )


@dataclass(frozen=True)
class CombastelReducer:
    """CORA-style Combastel reduction using L2 generator ordering."""

    tol: float = 1e-12
    name: str = "combastel"

    def reduce(
        self,
        zonotope: Zonotope,
        budget: int,
        context: ReductionContext | None = None,
    ) -> ReductionResult:
        return _reduce_keep_and_box(
            zonotope,
            budget,
            context,
            score=l2_scores,
            name=self.name,
            tol=self.tol,
            reason="Kept generators selected by L2 norm and boxed the rest.",
        )


@dataclass(frozen=True)
class PcaReducer:
    """PCA-basis interval-hull reduction."""

    tol: float = 1e-12
    name: str = "pca"

    def reduce(
        self,
        zonotope: Zonotope,
        budget: int,
        context: ReductionContext | None = None,
    ) -> ReductionResult:
        return _reduce_keep_and_transform(
            zonotope,
            budget,
            context,
            basis=_pca_basis,
            name=self.name,
            tol=self.tol,
            reason="Reduced discarded generators through a PCA-basis interval hull.",
        )


@dataclass(frozen=True)
class MethAReducer:
    """Method-A-style transform reduction with a long-generator basis."""

    tol: float = 1e-12
    max_condition: float = 1e10
    name: str = "methA"

    def reduce(
        self,
        zonotope: Zonotope,
        budget: int,
        context: ReductionContext | None = None,
    ) -> ReductionResult:
        return _reduce_keep_and_transform(
            zonotope,
            budget,
            context,
            basis=lambda generators: _long_generator_basis(
                generators,
                tol=self.tol,
                max_condition=self.max_condition,
            ),
            name=self.name,
            tol=self.tol,
            reason="Reduced discarded generators through a stable long-generator transform basis.",
        )


@dataclass(frozen=True)
class ScottReducer:
    """Scott-style transform reduction using pivoted independent directions."""

    tol: float = 1e-12
    max_condition: float = 1e10
    name: str = "scott"

    def reduce(
        self,
        zonotope: Zonotope,
        budget: int,
        context: ReductionContext | None = None,
    ) -> ReductionResult:
        return _reduce_keep_and_transform(
            zonotope,
            budget,
            context,
            basis=lambda generators: _pivot_basis(
                generators,
                tol=self.tol,
                max_condition=self.max_condition,
            ),
            name=self.name,
            tol=self.tol,
            reason="Reduced discarded generators through a pivoted independent transform basis.",
        )


@dataclass(frozen=True)
class AdaptiveReducer:
    """Choose the lowest current monitor-aware cost among paper baseline reducers."""

    candidates: tuple[Callable[[], Reducer], ...] = field(
        default_factory=lambda: (
            GirardReducer,
            CombastelReducer,
            PcaReducer,
            MethAReducer,
            ScottReducer,
            BoxReducer,
        )
    )
    trigger_width_weight: float = 1.0
    straddling_weight: float = 20.0
    generator_weight: float = 0.01
    name: str = "adaptive"

    def reduce(
        self,
        zonotope: Zonotope,
        budget: int,
        context: ReductionContext | None = None,
    ) -> ReductionResult:
        ctx = context or ReductionContext()
        best_result: ReductionResult | None = None
        best_name = ""
        best_cost = float("inf")
        failures: list[str] = []
        for factory in self.candidates:
            reducer = factory()
            try:
                result = reducer.reduce(zonotope, budget, ctx)
            except ValueError as exc:
                failures.append(f"{reducer.name}: {exc}")
                continue
            if not result.certificate.is_sound:
                failures.append(f"{reducer.name}: unsound certificate")
                continue
            cost = _current_cost(
                result.reduced,
                ctx,
                trigger_width_weight=self.trigger_width_weight,
                straddling_weight=self.straddling_weight,
                generator_weight=self.generator_weight,
            )
            if cost < best_cost:
                best_cost = cost
                best_name = reducer.name
                best_result = result
        if best_result is None:
            raise ValueError(
                "adaptive reducer found no certified candidate"
                + (f": {'; '.join(failures)}" if failures else "")
            )
        certificate = _certificate(
            self.name,
            zonotope,
            best_result.reduced,
            "Selected the certified candidate with the lowest current monitor-aware cost.",
            {
                "chosen_reducer": best_name,
                "chosen_cost": best_cost,
                "failures": failures,
            },
        )
        return ReductionResult(
            zonotope,
            best_result.reduced,
            certificate,
            {
                "chosen": best_name,
                "chosen_cost": best_cost,
                "candidate_stats": best_result.stats,
            },
        )


def _reduce_keep_and_box(
    zonotope: Zonotope,
    budget: int,
    context: ReductionContext | None,
    *,
    score: ScoreFunction,
    name: str,
    tol: float,
    reason: str,
) -> ReductionResult:
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if zonotope.generator_count <= budget:
        certificate = _certificate(
            name,
            zonotope,
            zonotope,
            "No reduction was needed because the state already satisfies the budget.",
        )
        return ReductionResult(zonotope, zonotope, certificate, {"kept": zonotope.generator_count})

    scores = np.asarray(score(zonotope, context), dtype=float)
    if scores.shape != (zonotope.generator_count,):
        raise ValueError("score function returned the wrong shape")
    order = tuple(int(index) for index in np.argsort(-scores, kind="mergesort"))

    for keep_count in range(min(budget, zonotope.generator_count), -1, -1):
        keep = order[:keep_count]
        discard = order[keep_count:]
        discarded_generators = zonotope.generators[:, discard]
        discarded_radius = (
            np.sum(np.abs(discarded_generators), axis=1)
            if discard
            else np.zeros(zonotope.dimension)
        )
        box_generators, box_metadata = _axis_box_generators(discarded_radius, tol=tol)
        if keep_count + box_generators.shape[1] > budget:
            continue
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
        reduced_metadata = tuple(zonotope.metadata[index] for index in keep) + box_metadata
        reduced = Zonotope(zonotope.center, reduced_generators, reduced_metadata)
        certificate = _certificate(
            name,
            zonotope,
            reduced,
            reason,
            {
                "kept_indices": keep,
                "discarded_count": len(discard),
                "score_min": float(np.min(scores)) if scores.size else 0.0,
                "score_max": float(np.max(scores)) if scores.size else 0.0,
            },
        )
        return ReductionResult(
            zonotope,
            reduced,
            certificate,
            {"kept": len(keep), "discarded": len(discard)},
        )

    raise ValueError(f"{name} cannot soundly reduce this state to budget {budget}")


def _reduce_keep_and_transform(
    zonotope: Zonotope,
    budget: int,
    context: ReductionContext | None,
    *,
    basis: BasisFunction,
    name: str,
    tol: float,
    reason: str,
) -> ReductionResult:
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if zonotope.generator_count <= budget:
        certificate = _certificate(
            name,
            zonotope,
            zonotope,
            "No reduction was needed because the state already satisfies the budget.",
        )
        return ReductionResult(zonotope, zonotope, certificate, {"kept": zonotope.generator_count})

    scores = l2_scores(zonotope, context)
    order = tuple(int(index) for index in np.argsort(-scores, kind="mergesort"))
    failures: list[str] = []

    for keep_count in range(min(budget, zonotope.generator_count), -1, -1):
        keep = order[:keep_count]
        discard = order[keep_count:]
        if not discard:
            continue
        discarded_generators = zonotope.generators[:, discard]
        try:
            transform_basis = basis(discarded_generators)
            transformed_generators, transformed_metadata = _transform_interval_generators(
                discarded_generators,
                transform_basis,
                tol=tol,
                source_prefix=name,
            )
        except ValueError as exc:
            failures.append(str(exc))
            continue
        if keep_count + transformed_generators.shape[1] > budget:
            continue
        kept_generators = (
            zonotope.generators[:, keep]
            if keep
            else np.zeros((zonotope.dimension, 0), dtype=float)
        )
        reduced_generators = (
            np.hstack([kept_generators, transformed_generators])
            if transformed_generators.shape[1]
            else kept_generators
        )
        reduced_metadata = (
            tuple(zonotope.metadata[index] for index in keep) + transformed_metadata
        )
        reduced = Zonotope(zonotope.center, reduced_generators, reduced_metadata)
        certificate = _certificate(
            name,
            zonotope,
            reduced,
            reason,
            {
                "kept_indices": keep,
                "discarded_count": len(discard),
                "transformed_generators": transformed_generators.shape[1],
                "failures": failures,
            },
        )
        return ReductionResult(
            zonotope,
            reduced,
            certificate,
            {
                "kept": len(keep),
                "discarded": len(discard),
                "transformed": transformed_generators.shape[1],
            },
        )

    raise ValueError(f"{name} cannot soundly reduce this state to budget {budget}")


def _transform_interval_generators(
    generators: NDArray[np.float64],
    basis: NDArray[np.float64],
    *,
    tol: float,
    source_prefix: str,
) -> tuple[NDArray[np.float64], tuple[GeneratorMetadata, ...]]:
    dimension = generators.shape[0]
    if generators.shape[1] == 0:
        return np.zeros((dimension, 0), dtype=float), ()
    if basis.shape != (dimension, dimension):
        raise ValueError("transform basis has the wrong shape")
    try:
        transformed = np.linalg.solve(basis, generators)
    except np.linalg.LinAlgError as exc:
        raise ValueError("transform basis is singular") from exc
    radius = np.sum(np.abs(transformed), axis=1)
    active = [index for index, value in enumerate(radius) if abs(value) > tol]
    reduced = np.zeros((dimension, len(active)), dtype=float)
    metadata: list[GeneratorMetadata] = []
    for column, axis in enumerate(active):
        reduced[:, column] = basis[:, axis] * radius[axis]
        metadata.append(
            GeneratorMetadata(
                kind=GeneratorKind.SYNTHETIC,
                source=f"{source_prefix}_axis_{axis}",
                age=0,
            )
        )
    return reduced, tuple(metadata)


def _pca_basis(generators: NDArray[np.float64]) -> NDArray[np.float64]:
    dimension = generators.shape[0]
    if generators.shape[1] == 0:
        return np.eye(dimension)
    u, _, _ = np.linalg.svd(generators, full_matrices=True)
    return u[:, :dimension]


def _long_generator_basis(
    generators: NDArray[np.float64],
    *,
    tol: float,
    max_condition: float,
) -> NDArray[np.float64]:
    dimension = generators.shape[0]
    if generators.shape[1] == 0:
        return np.eye(dimension)
    norms = np.linalg.norm(generators, axis=0)
    order = tuple(int(index) for index in np.argsort(-norms, kind="mergesort"))
    selected = _independent_columns(generators[:, order], dimension, tol=tol)
    basis = _complete_basis(selected, dimension, tol=tol)
    if np.linalg.cond(basis) > max_condition:
        return _pca_basis(generators)
    return _orthonormal_basis(basis)


def _pivot_basis(
    generators: NDArray[np.float64],
    *,
    tol: float,
    max_condition: float,
) -> NDArray[np.float64]:
    dimension = generators.shape[0]
    if generators.shape[1] == 0:
        return np.eye(dimension)
    _, _, pivots = qr(generators, pivoting=True, mode="economic")
    selected = _independent_columns(generators[:, tuple(int(i) for i in pivots)], dimension, tol=tol)
    basis = _complete_basis(selected, dimension, tol=tol)
    if np.linalg.cond(basis) > max_condition:
        return _pca_basis(generators)
    return _orthonormal_basis(basis)


def _independent_columns(
    generators: NDArray[np.float64],
    dimension: int,
    *,
    tol: float,
) -> list[NDArray[np.float64]]:
    selected: list[NDArray[np.float64]] = []
    current = np.zeros((dimension, 0), dtype=float)
    for column in range(generators.shape[1]):
        candidate = generators[:, column]
        if np.linalg.norm(candidate) <= tol:
            continue
        candidate_matrix = np.column_stack([current, candidate])
        if np.linalg.matrix_rank(candidate_matrix, tol=tol) > current.shape[1]:
            selected.append(candidate / np.linalg.norm(candidate))
            current = candidate_matrix
        if len(selected) == dimension:
            break
    return selected


def _complete_basis(
    selected: list[NDArray[np.float64]],
    dimension: int,
    *,
    tol: float,
) -> NDArray[np.float64]:
    columns = list(selected)
    current = (
        np.column_stack(columns)
        if columns
        else np.zeros((dimension, 0), dtype=float)
    )
    for axis in range(dimension):
        unit = np.eye(dimension)[:, axis]
        candidate_matrix = np.column_stack([current, unit])
        if np.linalg.matrix_rank(candidate_matrix, tol=tol) > current.shape[1]:
            columns.append(unit)
            current = candidate_matrix
        if len(columns) == dimension:
            break
    if len(columns) != dimension:
        return np.eye(dimension)
    return np.column_stack(columns)


def _orthonormal_basis(basis: NDArray[np.float64]) -> NDArray[np.float64]:
    q, _ = np.linalg.qr(basis)
    return q


def _current_cost(
    zonotope: Zonotope,
    context: ReductionContext,
    *,
    trigger_width_weight: float,
    straddling_weight: float,
    generator_weight: float,
) -> float:
    lower, upper = zonotope.interval_bounds()
    widths = upper - lower
    total = generator_weight * zonotope.generator_count
    if context.triggers:
        for trigger in context.triggers:
            width = float(widths[trigger.state_index])
            total += trigger_width_weight * width
            if trigger_straddles_threshold(
                lower[trigger.state_index],
                upper[trigger.state_index],
                trigger,
            ):
                total += straddling_weight
    else:
        total += trigger_width_weight * float(np.sum(widths))
    return float(total)
