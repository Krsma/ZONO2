"""Generator scoring functions for keep-and-reduce strategies.

Scoring determines which generators are most important to preserve during
reduction. Higher scores mean the generator is more valuable and should
be kept.

Sources:
  - Girard metric: Girard 2005 / Kopetzki et al. 2017
  - L2 (Combastel): Combastel 2003
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pzr.zonotope.core import Zonotope


def girard_scores(z: Zonotope) -> NDArray[np.float64]:
    """L1-norm minus L-infinity norm per generator (Girard metric).

    Generators with high L1 but small max-component create large interval-hull
    artifacts when discarded; keeping them reduces overapproximation.
    """
    if z.generator_count == 0:
        return np.empty(0, dtype=np.float64)
    G = z.generators
    l1 = np.sum(np.abs(G), axis=0)
    linf = np.max(np.abs(G), axis=0)
    return l1 - linf


def l2_scores(z: Zonotope) -> NDArray[np.float64]:
    """Euclidean norm per generator (Combastel metric)."""
    if z.generator_count == 0:
        return np.empty(0, dtype=np.float64)
    return np.linalg.norm(z.generators, axis=0)


def norm_scores(z: Zonotope) -> NDArray[np.float64]:
    """Alias for l2_scores."""
    return l2_scores(z)
