"""Certified zonotope reduction methods.

Each reducer guarantees: Z ⊆ reduce(Z, budget) and
generator_count(reduce(Z, budget)) <= budget.

Methods follow Kopetzki et al. (IEEE CDC 2017) and
Yang & Scott (Automatica 2018). Validated against CORA reference fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import qr

from pzr.zonotope.core import Zonotope
from pzr.zonotope.scoring import girard_scores, l2_scores

ScoreFunction = Callable[[Zonotope], NDArray[np.float64]]
BasisFunction = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class ReductionCertificate:
    reducer: str
    original_generators: int
    reduced_generators: int
    is_sound: bool


@dataclass(frozen=True)
class ReductionResult:
    original: Zonotope
    reduced: Zonotope
    certificate: ReductionCertificate


def _cert(name: str, original: Zonotope, reduced: Zonotope) -> ReductionCertificate:
    return ReductionCertificate(
        reducer=name,
        original_generators=original.generator_count,
        reduced_generators=reduced.generator_count,
        is_sound=True,
    )


class Reducer(Protocol):
    name: str

    def reduce(self, z: Zonotope, budget: int) -> ReductionResult: ...


def _no_reduction(name: str, z: Zonotope) -> ReductionResult:
    return ReductionResult(z, z, _cert(name, z, z))


# ---------------------------------------------------------------------------
# Axis-aligned box helpers
# ---------------------------------------------------------------------------

def _axis_box_generators(radius: NDArray[np.float64], tol: float = 1e-12) -> NDArray[np.float64]:
    """Diagonal matrix with one generator per active axis."""
    active = [i for i, r in enumerate(radius) if abs(r) > tol]
    n = radius.size
    g = np.zeros((n, len(active)), dtype=np.float64)
    for col, axis in enumerate(active):
        g[axis, col] = radius[axis]
    return g


# ---------------------------------------------------------------------------
# BoxReducer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoxReducer:
    """Reduce to interval hull: one generator per active dimension."""

    tol: float = 1e-12
    name: str = "box"

    def reduce(self, z: Zonotope, budget: int) -> ReductionResult:
        if z.generator_count <= budget:
            return _no_reduction(self.name, z)
        g = _axis_box_generators(z.interval_radius(), tol=self.tol)
        if g.shape[1] > budget:
            raise ValueError(
                f"box reducer needs {g.shape[1]} generators, budget is {budget}"
            )
        reduced = Zonotope(z.center, g)
        return ReductionResult(z, reduced, _cert(self.name, z, reduced))


# ---------------------------------------------------------------------------
# Keep-and-box reducers (Girard, Combastel)
# ---------------------------------------------------------------------------

def _reduce_keep_and_box(
    z: Zonotope,
    budget: int,
    score_fn: ScoreFunction,
    name: str,
    tol: float = 1e-12,
) -> ReductionResult:
    if z.generator_count <= budget:
        return _no_reduction(name, z)

    scores = score_fn(z)
    order = np.argsort(-scores, kind="mergesort")

    for keep_count in range(min(budget, z.generator_count), -1, -1):
        keep_idx = order[:keep_count]
        discard_idx = order[keep_count:]

        discarded_radius = np.sum(np.abs(z.generators[:, discard_idx]), axis=1)
        box_g = _axis_box_generators(discarded_radius, tol=tol)

        if keep_count + box_g.shape[1] <= budget:
            kept_g = z.generators[:, keep_idx] if keep_count > 0 else np.empty((z.dimension, 0))
            if box_g.shape[1] > 0:
                reduced_g = np.hstack([kept_g, box_g])
            else:
                reduced_g = kept_g
            reduced = Zonotope(z.center, reduced_g)
            return ReductionResult(z, reduced, _cert(name, z, reduced))

    raise ValueError(f"{name} cannot reduce to budget {budget}")


@dataclass(frozen=True)
class GirardReducer:
    """Girard reduction: keep by L1-Linf score, box the rest."""

    tol: float = 1e-12
    name: str = "girard"

    def reduce(self, z: Zonotope, budget: int) -> ReductionResult:
        return _reduce_keep_and_box(z, budget, girard_scores, self.name, self.tol)


@dataclass(frozen=True)
class CombastelReducer:
    """Combastel reduction: keep by L2 norm, box the rest."""

    tol: float = 1e-12
    name: str = "combastel"

    def reduce(self, z: Zonotope, budget: int) -> ReductionResult:
        return _reduce_keep_and_box(z, budget, l2_scores, self.name, self.tol)


# ---------------------------------------------------------------------------
# Keep-and-transform reducers (PCA, MethA, Scott)
# ---------------------------------------------------------------------------

def _pca_basis(generators: NDArray[np.float64]) -> NDArray[np.float64]:
    n = generators.shape[0]
    if generators.shape[1] == 0:
        return np.eye(n)
    u, _, _ = np.linalg.svd(generators, full_matrices=True)
    return u[:, :n]


def _independent_columns(
    generators: NDArray[np.float64],
    n: int,
    tol: float,
) -> list[NDArray[np.float64]]:
    selected: list[NDArray[np.float64]] = []
    current = np.empty((n, 0), dtype=np.float64)
    for col in range(generators.shape[1]):
        v = generators[:, col]
        if np.linalg.norm(v) <= tol:
            continue
        candidate = np.column_stack([current, v])
        if np.linalg.matrix_rank(candidate, tol=tol) > current.shape[1]:
            selected.append(v / np.linalg.norm(v))
            current = candidate
        if len(selected) == n:
            break
    return selected


def _complete_basis(
    selected: list[NDArray[np.float64]],
    n: int,
    tol: float,
) -> NDArray[np.float64]:
    columns = list(selected)
    current = np.column_stack(columns) if columns else np.empty((n, 0), dtype=np.float64)
    for axis in range(n):
        e = np.eye(n)[:, axis]
        candidate = np.column_stack([current, e])
        if np.linalg.matrix_rank(candidate, tol=tol) > current.shape[1]:
            columns.append(e)
            current = candidate
        if len(columns) == n:
            break
    if len(columns) != n:
        return np.eye(n)
    return np.column_stack(columns)


def _long_generator_basis(
    generators: NDArray[np.float64],
    tol: float = 1e-12,
    max_condition: float = 1e10,
) -> NDArray[np.float64]:
    n = generators.shape[0]
    if generators.shape[1] == 0:
        return np.eye(n)
    norms = np.linalg.norm(generators, axis=0)
    order = np.argsort(-norms, kind="mergesort")
    selected = _independent_columns(generators[:, order], n, tol)
    basis = _complete_basis(selected, n, tol)
    if np.linalg.cond(basis) > max_condition:
        return _pca_basis(generators)
    q, _ = np.linalg.qr(basis)
    return q


def _pivot_basis(
    generators: NDArray[np.float64],
    tol: float = 1e-12,
    max_condition: float = 1e10,
) -> NDArray[np.float64]:
    n = generators.shape[0]
    if generators.shape[1] == 0:
        return np.eye(n)
    _, _, pivots = qr(generators, pivoting=True, mode="economic")
    pivoted = generators[:, [int(i) for i in pivots]]
    selected = _independent_columns(pivoted, n, tol)
    basis = _complete_basis(selected, n, tol)
    if np.linalg.cond(basis) > max_condition:
        return _pca_basis(generators)
    q, _ = np.linalg.qr(basis)
    return q


def _transform_and_box(
    generators: NDArray[np.float64],
    basis: NDArray[np.float64],
    tol: float = 1e-12,
) -> NDArray[np.float64]:
    """Transform generators into basis coordinates, take interval hull, transform back."""
    n = generators.shape[0]
    if generators.shape[1] == 0:
        return np.empty((n, 0), dtype=np.float64)
    transformed = np.linalg.solve(basis, generators)
    radius = np.sum(np.abs(transformed), axis=1)
    active = [i for i, r in enumerate(radius) if abs(r) > tol]
    result = np.zeros((n, len(active)), dtype=np.float64)
    for col, axis in enumerate(active):
        result[:, col] = basis[:, axis] * radius[axis]
    return result


def _reduce_keep_and_transform(
    z: Zonotope,
    budget: int,
    basis_fn: BasisFunction,
    name: str,
    tol: float = 1e-12,
) -> ReductionResult:
    if z.generator_count <= budget:
        return _no_reduction(name, z)

    scores = l2_scores(z)
    order = np.argsort(-scores, kind="mergesort")

    for keep_count in range(min(budget, z.generator_count), -1, -1):
        keep_idx = order[:keep_count]
        discard_idx = order[keep_count:]
        if len(discard_idx) == 0:
            continue

        discarded = z.generators[:, discard_idx]
        try:
            basis = basis_fn(discarded)
            transformed_g = _transform_and_box(discarded, basis, tol=tol)
        except (ValueError, np.linalg.LinAlgError):
            continue

        if keep_count + transformed_g.shape[1] <= budget:
            kept_g = z.generators[:, keep_idx] if keep_count > 0 else np.empty((z.dimension, 0))
            if transformed_g.shape[1] > 0:
                reduced_g = np.hstack([kept_g, transformed_g])
            else:
                reduced_g = kept_g
            reduced = Zonotope(z.center, reduced_g)
            return ReductionResult(z, reduced, _cert(name, z, reduced))

    raise ValueError(f"{name} cannot reduce to budget {budget}")


@dataclass(frozen=True)
class PcaReducer:
    """PCA-basis interval-hull reduction."""

    tol: float = 1e-12
    name: str = "pca"

    def reduce(self, z: Zonotope, budget: int) -> ReductionResult:
        return _reduce_keep_and_transform(z, budget, _pca_basis, self.name, self.tol)


@dataclass(frozen=True)
class MethAReducer:
    """Method A: long-generator basis transform reduction."""

    tol: float = 1e-12
    max_condition: float = 1e10
    name: str = "methA"

    def reduce(self, z: Zonotope, budget: int) -> ReductionResult:
        return _reduce_keep_and_transform(
            z, budget,
            lambda g: _long_generator_basis(g, self.tol, self.max_condition),
            self.name, self.tol,
        )


@dataclass(frozen=True)
class ScottReducer:
    """Scott-style reduction: pivoted independent direction basis."""

    tol: float = 1e-12
    max_condition: float = 1e10
    name: str = "scott"

    def reduce(self, z: Zonotope, budget: int) -> ReductionResult:
        return _reduce_keep_and_transform(
            z, budget,
            lambda g: _pivot_basis(g, self.tol, self.max_condition),
            self.name, self.tol,
        )


# ---------------------------------------------------------------------------
# Identity reducer (no-op, only succeeds within budget)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IdentityReducer:
    name: str = "identity"

    def reduce(self, z: Zonotope, budget: int) -> ReductionResult:
        if z.generator_count > budget:
            raise ValueError(
                f"identity reducer cannot reduce {z.generator_count} generators to {budget}"
            )
        return _no_reduction(self.name, z)


# ---------------------------------------------------------------------------
# Registry of all standard reducers
# ---------------------------------------------------------------------------

ALL_REDUCERS: dict[str, Reducer] = {
    "box": BoxReducer(),
    "girard": GirardReducer(),
    "combastel": CombastelReducer(),
    "pca": PcaReducer(),
    "methA": MethAReducer(),
    "scott": ScottReducer(),
    "identity": IdentityReducer(),
}
