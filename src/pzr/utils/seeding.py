"""Reproducible RNG management."""

from __future__ import annotations

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def derive_seed(base_seed: int, index: int) -> int:
    """Derive a deterministic child seed from a base seed and index."""
    rng = np.random.default_rng(base_seed)
    seeds = rng.integers(0, 2**31, size=index + 1)
    return int(seeds[index])
