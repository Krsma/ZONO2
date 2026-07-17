"""Versioned target semantics for cost-sensitive reducer ranking."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


TARGET_SCHEMA = "pzr.reducer-ranking-target.v2"
ABSOLUTE_TOLERANCE = 1e-15
RELATIVE_TOLERANCE = 1e-9
TARGET_CONTRACT: dict[str, object] = {
    "schema": TARGET_SCHEMA,
    "loss": "weighted_pairwise_softplus",
    "reduction": "per_state_pair_weight_sum_then_rankable_state_mean",
    "absolute_tolerance": ABSOLUTE_TOLERANCE,
    "relative_tolerance": RELATIVE_TOLERANCE,
    "feasible_pair_weighting": "gap_over_largest_meaningful_feasible_gap_in_state",
    "infeasible_semantics": "every_feasible_candidate_ranks_above_every_infeasible_candidate",
    "infeasible_pair_weight": 1.0,
    "score_semantics": "lower_is_better_uncalibrated_ranking_score",
}


def pair_tolerance(
    left: NDArray[np.floating] | float,
    right: NDArray[np.floating] | float,
) -> NDArray[np.float64] | np.float64:
    """Return ``max(1e-15, 1e-9 max(|left|, |right|))`` elementwise."""
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    return np.maximum(
        ABSOLUTE_TOLERANCE,
        RELATIVE_TOLERANCE * np.maximum(np.abs(left_values), np.abs(right_values)),
    )


def tolerant_best_mask(
    costs: NDArray[np.float64],
    feasible: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Mark feasible candidates tied with the minimum under the target tolerance."""
    values = np.asarray(costs, dtype=np.float64)
    mask = np.asarray(feasible, dtype=np.bool_)
    if values.shape != mask.shape or values.ndim not in (1, 2):
        raise ValueError("costs and feasibility must be aligned vectors or matrices")
    squeeze = values.ndim == 1
    if squeeze:
        values = values[None, :]
        mask = mask[None, :]
    if np.any(~np.any(mask, axis=1)):
        raise ValueError("every state needs at least one feasible candidate")
    safe = np.where(mask, values, np.inf)
    best = np.min(safe, axis=1, keepdims=True)
    result = mask & ((values - best) <= pair_tolerance(values, best))
    return result[0] if squeeze else result


def ranking_pair_weights(
    costs: NDArray[np.float64],
    feasible: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Return directed target weights with shape ``(state, better, worse)``."""
    values = np.asarray(costs, dtype=np.float64)
    mask = np.asarray(feasible, dtype=np.bool_)
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("cost and feasibility matrices must align")
    safe = np.where(mask, values, 0.0)
    cost_i = safe[:, :, None]
    cost_j = safe[:, None, :]
    feasible_i = mask[:, :, None]
    feasible_j = mask[:, None, :]
    gap = cost_j - cost_i
    tolerance = pair_tolerance(cost_i, cost_j)
    meaningful = feasible_i & feasible_j & (gap > tolerance)
    meaningful_gaps = np.where(meaningful, gap, 0.0)
    largest = np.max(meaningful_gaps, axis=(1, 2), keepdims=True)
    normalized = np.divide(
        meaningful_gaps,
        largest,
        out=np.zeros_like(meaningful_gaps),
        where=largest > 0.0,
    )
    return np.where(feasible_i & ~feasible_j, 1.0, normalized)


def rankable_state_mask(
    costs: NDArray[np.float64],
    feasible: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Return states that contribute at least one target ordering."""
    return ranking_pair_weights(costs, feasible).sum(axis=(1, 2)) > 0.0
