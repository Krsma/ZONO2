"""Zonotope primitives used throughout the project."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


class GeneratorKind(str, Enum):
    """Semantic role of a generator in a monitor state."""

    CALIBRATION = "calibration"
    MEASUREMENT = "measurement"
    SYNTHETIC = "synthetic"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GeneratorMetadata:
    """Optional semantic metadata for a zonotope generator."""

    kind: GeneratorKind = GeneratorKind.UNKNOWN
    source: str | None = None
    age: int = 0

    def aged(self, steps: int = 1) -> "GeneratorMetadata":
        return replace(self, age=self.age + steps)


@dataclass(frozen=True)
class GeneratorRequirement:
    """Metadata pattern for generators that a monitor requires exactly."""

    kind: GeneratorKind | None = None
    source: str | None = None

    def matches(self, metadata: GeneratorMetadata) -> bool:
        if self.kind is not None and metadata.kind != self.kind:
            return False
        if self.source is not None and metadata.source != self.source:
            return False
        return True


@dataclass(frozen=True)
class Zonotope:
    """A center-generator representation ``c + G[-1, 1]^m``."""

    center: NDArray[np.float64]
    generators: NDArray[np.float64]
    metadata: tuple[GeneratorMetadata, ...] = ()

    def __init__(
        self,
        center: ArrayLike,
        generators: ArrayLike | None = None,
        metadata: Sequence[GeneratorMetadata] | None = None,
    ) -> None:
        c = np.asarray(center, dtype=float).reshape(-1)
        if generators is None:
            g = np.zeros((c.size, 0), dtype=float)
        else:
            g = np.asarray(generators, dtype=float)
            if g.ndim == 1:
                if g.size == 0:
                    g = np.zeros((c.size, 0), dtype=float)
                else:
                    g = g.reshape(c.size, 1)
            if g.ndim != 2:
                raise ValueError("generators must be a 2D array")
            if g.shape[0] != c.size:
                raise ValueError(
                    f"generator row count {g.shape[0]} does not match center dimension {c.size}"
                )

        if metadata is None:
            meta = tuple(GeneratorMetadata() for _ in range(g.shape[1]))
        else:
            meta = tuple(metadata)
            if len(meta) != g.shape[1]:
                raise ValueError(
                    f"metadata length {len(meta)} does not match generator count {g.shape[1]}"
                )

        c = c.copy()
        g = g.copy()
        c.setflags(write=False)
        g.setflags(write=False)
        object.__setattr__(self, "center", c)
        object.__setattr__(self, "generators", g)
        object.__setattr__(self, "metadata", meta)

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

    @property
    def is_empty_generator_set(self) -> bool:
        return self.generator_count == 0

    def interval_radius(self) -> NDArray[np.float64]:
        return np.sum(np.abs(self.generators), axis=1)

    def interval_bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        radius = self.interval_radius()
        return self.center - radius, self.center + radius

    def widths(self) -> NDArray[np.float64]:
        lower, upper = self.interval_bounds()
        return upper - lower

    def affine_map(self, matrix: ArrayLike, bias: ArrayLike | None = None) -> "Zonotope":
        a = np.asarray(matrix, dtype=float)
        if a.ndim != 2 or a.shape[1] != self.dimension:
            raise ValueError("matrix must have shape (out_dimension, zonotope.dimension)")
        b = np.zeros(a.shape[0], dtype=float) if bias is None else np.asarray(bias, dtype=float)
        if b.shape != (a.shape[0],):
            raise ValueError("bias must have shape (out_dimension,)")
        return Zonotope(a @ self.center + b, a @ self.generators, self.metadata)

    def with_generators(
        self,
        generators: ArrayLike,
        metadata: Sequence[GeneratorMetadata] | None = None,
    ) -> "Zonotope":
        return Zonotope(self.center, generators, metadata)

    def with_center(self, center: ArrayLike) -> "Zonotope":
        return Zonotope(center, self.generators, self.metadata)

    def append_generators(
        self,
        generators: ArrayLike,
        metadata: Sequence[GeneratorMetadata],
    ) -> "Zonotope":
        new_g = np.asarray(generators, dtype=float)
        if new_g.ndim == 1:
            if new_g.size == 0:
                new_g = np.zeros((self.dimension, 0), dtype=float)
            else:
                new_g = new_g.reshape(self.dimension, 1)
        if new_g.shape[0] != self.dimension:
            raise ValueError("new generators must match zonotope dimension")
        if len(metadata) != new_g.shape[1]:
            raise ValueError("metadata length must match appended generator count")
        if self.generator_count == 0:
            return Zonotope(self.center, new_g, metadata)
        return Zonotope(
            self.center,
            np.hstack([self.generators, new_g]),
            (*self.metadata, *metadata),
        )

    def take_generators(self, indices: Iterable[int]) -> "Zonotope":
        idx = tuple(indices)
        return Zonotope(
            self.center,
            self.generators[:, idx] if idx else np.zeros((self.dimension, 0)),
            tuple(self.metadata[i] for i in idx),
        )

    def age_generators(self, steps: int = 1) -> "Zonotope":
        return Zonotope(
            self.center,
            self.generators,
            tuple(meta.aged(steps) for meta in self.metadata),
        )

    def sample(self, coefficients: ArrayLike) -> NDArray[np.float64]:
        coeffs = np.asarray(coefficients, dtype=float).reshape(-1)
        if coeffs.shape != (self.generator_count,):
            raise ValueError("coefficients must match generator count")
        if np.any(np.abs(coeffs) > 1.0):
            raise ValueError("zonotope sample coefficients must lie in [-1, 1]")
        return self.center + self.generators @ coeffs

    def contains_in_interval_hull(self, point: ArrayLike, atol: float = 1e-10) -> bool:
        p = np.asarray(point, dtype=float).reshape(-1)
        if p.shape != (self.dimension,):
            raise ValueError("point dimension does not match zonotope")
        lower, upper = self.interval_bounds()
        return bool(np.all(p >= lower - atol) and np.all(p <= upper + atol))
