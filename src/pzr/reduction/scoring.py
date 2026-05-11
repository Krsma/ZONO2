"""Generator scoring functions for reducer policies."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pzr.core.zonotope import GeneratorKind, Zonotope
from pzr.reduction.base import ReductionContext


def norm_scores(zonotope: Zonotope, context: ReductionContext | None = None) -> NDArray[np.float64]:
    """Score generators by Euclidean norm."""

    _ = context
    if zonotope.generator_count == 0:
        return np.zeros(0)
    return np.linalg.norm(zonotope.generators, axis=0)


def threshold_risk_scores(
    zonotope: Zonotope,
    context: ReductionContext | None = None,
    *,
    eps: float = 1e-9,
) -> NDArray[np.float64]:
    """Score generators by influence on near-threshold trigger dimensions."""

    if zonotope.generator_count == 0:
        return np.zeros(0)
    if context is None or not context.triggers:
        return np.zeros(zonotope.generator_count)

    scores = np.zeros(zonotope.generator_count)
    for trigger in context.triggers:
        distance = abs(zonotope.center[trigger.state_index] - trigger.threshold)
        influence = np.abs(zonotope.generators[trigger.state_index, :])
        scores += influence / (distance + eps)
    return scores


def calibration_aware_scores(
    zonotope: Zonotope,
    context: ReductionContext | None = None,
    *,
    norm_weight: float = 1.0,
    threshold_weight: float = 1.0,
    calibration_bonus: float = 1e6,
    age_weight: float = 0.0,
) -> NDArray[np.float64]:
    """Default monitor-aware score used by metadata-preserving baselines."""

    scores = norm_weight * norm_scores(zonotope, context)
    scores = scores + threshold_weight * threshold_risk_scores(zonotope, context)
    if scores.size == 0:
        return scores

    for index, meta in enumerate(zonotope.metadata):
        if meta.kind == GeneratorKind.CALIBRATION and (
            context is None or context.preserve_calibration
        ):
            scores[index] += calibration_bonus
        scores[index] += age_weight * meta.age
    return scores
