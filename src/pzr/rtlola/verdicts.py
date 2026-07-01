"""Typed access to numeric RTLola verdict intervals."""

from __future__ import annotations

from numbers import Real

import numpy as np


def interval_bounds(value: object) -> tuple[float, float]:
    """Return finite interval bounds from a scalar or binding affine value."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("Boolean verdicts do not have numeric interval bounds")
    if isinstance(value, Real):
        scalar = float(value)
        if not np.isfinite(scalar):
            raise ValueError("RTLola verdict scalar is non-finite")
        return scalar, scalar
    if isinstance(value, (str, bytes)):
        raise TypeError(
            "numeric RTLola verdict must use the binding AffineValue, "
            f"got {type(value).__name__}"
        )
    if hasattr(value, "lower") and hasattr(value, "upper"):
        lower = getattr(value, "lower")
        upper = getattr(value, "upper")
        if callable(lower) or callable(upper):
            raise TypeError("RTLola affine bounds must be numeric properties")
        lo = float(lower)
        hi = float(upper)
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError("RTLola affine bounds are non-finite")
        if hi < lo:
            raise ValueError(f"RTLola affine bounds are reversed: [{lo}, {hi}]")
        return lo, hi
    raise TypeError(
        "numeric RTLola verdict must be a scalar or binding AffineValue, "
        f"got {type(value).__name__}"
    )
