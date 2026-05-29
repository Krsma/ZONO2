"""Zonotope: center-generator representation Z = c + G[-1,1]^m."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class Zonotope:
    """Immutable zonotope in center-generator form.

    A zonotope Z(c, G) = {c + G @ xi : xi in [-1, 1]^m}
    where c in R^n is the center and G in R^{n x m} is the generator matrix.
    """

    center: NDArray[np.float64]
    generators: NDArray[np.float64]

    def __init__(
        self,
        center: ArrayLike,
        generators: ArrayLike | None = None,
    ) -> None:
        c = np.asarray(center, dtype=np.float64).ravel()
        if generators is None:
            g = np.empty((c.size, 0), dtype=np.float64)
        else:
            g = np.asarray(generators, dtype=np.float64)
            if g.ndim == 1:
                g = g.reshape(c.size, 1) if g.size > 0 else np.empty((c.size, 0), dtype=np.float64)
            if g.ndim != 2 or g.shape[0] != c.size:
                raise ValueError(
                    f"generators shape {g.shape} incompatible with center dimension {c.size}"
                )

        c = c.copy()
        g = g.copy()
        c.setflags(write=False)
        g.setflags(write=False)
        object.__setattr__(self, "center", c)
        object.__setattr__(self, "generators", g)

    @property
    def dimension(self) -> int:
        return int(self.center.size)

    @property
    def generator_count(self) -> int:
        return int(self.generators.shape[1])

    @property
    def order(self) -> float:
        if self.dimension == 0:
            return 0.0
        return self.generator_count / self.dimension

    def interval_radius(self) -> NDArray[np.float64]:
        return np.sum(np.abs(self.generators), axis=1)

    def interval_bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r = self.interval_radius()
        return self.center - r, self.center + r

    def widths(self) -> NDArray[np.float64]:
        return 2.0 * self.interval_radius()

    def affine_map(self, matrix: ArrayLike, bias: ArrayLike | None = None) -> Zonotope:
        a = np.asarray(matrix, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != self.dimension:
            raise ValueError(f"matrix shape {a.shape} incompatible with dimension {self.dimension}")
        new_center = a @ self.center
        if bias is not None:
            new_center = new_center + np.asarray(bias, dtype=np.float64).ravel()
        return Zonotope(new_center, a @ self.generators)

    def minkowski_sum(self, other: Zonotope) -> Zonotope:
        if self.dimension != other.dimension:
            raise ValueError("dimension mismatch")
        return Zonotope(
            self.center + other.center,
            np.hstack([self.generators, other.generators]),
        )

    def append_generators(self, generators: ArrayLike) -> Zonotope:
        g = np.asarray(generators, dtype=np.float64)
        if g.ndim == 1:
            g = g.reshape(self.dimension, 1) if g.size > 0 else np.empty((self.dimension, 0), dtype=np.float64)
        if g.shape[0] != self.dimension:
            raise ValueError("appended generators must match zonotope dimension")
        if self.generator_count == 0:
            return Zonotope(self.center, g)
        return Zonotope(self.center, np.hstack([self.generators, g]))

    def take_generators(self, indices: Sequence[int]) -> Zonotope:
        idx = list(indices)
        if not idx:
            return Zonotope(self.center)
        return Zonotope(self.center, self.generators[:, idx])

    def with_center(self, center: ArrayLike) -> Zonotope:
        return Zonotope(center, self.generators)

    def with_generators(self, generators: ArrayLike) -> Zonotope:
        return Zonotope(self.center, generators)

    def sample(self, coefficients: ArrayLike) -> NDArray[np.float64]:
        xi = np.asarray(coefficients, dtype=np.float64).ravel()
        if xi.shape != (self.generator_count,):
            raise ValueError(f"expected {self.generator_count} coefficients, got {xi.size}")
        if np.any(np.abs(xi) > 1.0 + 1e-12):
            raise ValueError("coefficients must lie in [-1, 1]")
        return self.center + self.generators @ xi

    def contains_in_interval_hull(self, point: ArrayLike, atol: float = 1e-10) -> bool:
        p = np.asarray(point, dtype=np.float64).ravel()
        if p.size != self.dimension:
            raise ValueError("point dimension mismatch")
        lo, hi = self.interval_bounds()
        return bool(np.all(p >= lo - atol) and np.all(p <= hi + atol))

    def volume_proxy(self) -> float:
        w = self.widths()
        return float(np.prod(w))

    def __repr__(self) -> str:
        return f"Zonotope(dim={self.dimension}, generators={self.generator_count})"
