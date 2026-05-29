"""Zonotope metrics: distance, containment checks, error measures."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pzr.zonotope.core import Zonotope


def interval_hull_mse(z1: Zonotope, z2: Zonotope) -> float:
    """Mean squared error between interval hulls of two zonotopes."""
    lo1, hi1 = z1.interval_bounds()
    lo2, hi2 = z2.interval_bounds()
    return float(0.5 * (np.mean((lo1 - lo2) ** 2) + np.mean((hi1 - hi2) ** 2)))


def interval_hull_max_error(z1: Zonotope, z2: Zonotope) -> float:
    """Maximum absolute difference between interval hulls."""
    lo1, hi1 = z1.interval_bounds()
    lo2, hi2 = z2.interval_bounds()
    return float(max(np.max(np.abs(lo1 - lo2)), np.max(np.abs(hi1 - hi2))))


def width_inflation(original: Zonotope, reduced: Zonotope) -> NDArray[np.float64]:
    """Per-dimension ratio of reduced widths to original widths."""
    w_orig = original.widths()
    w_red = reduced.widths()
    safe = np.where(w_orig > 0, w_orig, 1.0)
    return w_red / safe


def containment_check(
    original: Zonotope,
    reduced: Zonotope,
    n_samples: int = 1000,
    rng: np.random.Generator | None = None,
) -> bool:
    """Check that sampled points from original lie in reduced's interval hull."""
    if rng is None:
        rng = np.random.default_rng(42)
    for _ in range(n_samples):
        xi = rng.uniform(-1.0, 1.0, size=original.generator_count)
        point = original.sample(xi)
        if not reduced.contains_in_interval_hull(point):
            return False
    return True
