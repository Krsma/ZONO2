"""Versioned cost and objective semantics for reducer learning."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


COST_SCHEMA = "pzr.reducer-cost-target.v3"
ABSOLUTE_TOLERANCE = 1e-15
RELATIVE_TOLERANCE = 1e-9
COST_CONTRACT: dict[str, object] = {
    "schema": COST_SCHEMA,
    "teacher_cost": "binding_native_two_event_full_width_terminal_cost",
    "absolute_tolerance": ABSOLUTE_TOLERANCE,
    "relative_tolerance": RELATIVE_TOLERANCE,
    "infeasible_cost": "nan_with_explicit_false_feasibility",
}
PAIRWISE_OBJECTIVE_CONTRACT: dict[str, object] = {
    "schema": "pzr.reducer-objective.pairwise-v2",
    "loss": "weighted_pairwise_softplus",
    "reduction": "per_state_pair_weight_sum_then_rankable_state_mean",
    "absolute_tolerance": ABSOLUTE_TOLERANCE,
    "relative_tolerance": RELATIVE_TOLERANCE,
    "feasible_pair_weighting": "gap_over_largest_meaningful_feasible_gap_in_state",
    "infeasible_semantics": "every_feasible_candidate_ranks_above_every_infeasible_candidate",
    "infeasible_pair_weight": 1.0,
    "score_semantics": "lower_is_better_uncalibrated_score",
}


def soft_objective_contract(
    temperature: float,
    feasibility_penalty: float,
) -> dict[str, object]:
    """Return the complete soft action-value distillation contract."""
    if temperature <= 0.0 or not np.isfinite(temperature):
        raise ValueError("soft-target temperature must be finite and positive")
    if feasibility_penalty < 0.0 or not np.isfinite(feasibility_penalty):
        raise ValueError("feasibility penalty must be finite and non-negative")
    return {
        "schema": "pzr.reducer-objective.soft-kl-v1",
        "loss": "state_balanced_soft_action_value_kl",
        "reduction": "valid_state_mean",
        "temperature": float(temperature),
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "regret": "tolerance_zeroed_gap_over_largest_meaningful_feasible_gap",
        "tie_semantics": "uniform_over_feasible_candidates_when_no_meaningful_gap",
        "infeasible_target_probability": 0.0,
        "feasibility_penalty": float(feasibility_penalty),
        "feasibility_penalty_semantics": "sum_student_probability_on_infeasible_candidates",
        "score_semantics": "lower_is_better_uncalibrated_score",
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
    """Mark feasible candidates tied with the minimum under the cost tolerance."""
    values, mask, squeeze = _aligned_costs(costs, feasible)
    safe = np.where(mask, values, np.inf)
    best = np.min(safe, axis=1, keepdims=True)
    result = mask & ((values - best) <= pair_tolerance(values, best))
    return result[0] if squeeze else result


def normalized_regrets(
    costs: NDArray[np.float64],
    feasible: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Return tolerance-aware feasible regrets in ``[0, 1]`` and NaN otherwise."""
    values, mask, squeeze = _aligned_costs(costs, feasible)
    safe = np.where(mask, values, np.inf)
    best = np.min(safe, axis=1, keepdims=True)
    gaps = values - best
    meaningful = mask & (gaps > pair_tolerance(values, best))
    gaps = np.where(meaningful, gaps, 0.0)
    span = np.max(gaps, axis=1, keepdims=True)
    regrets = np.divide(gaps, span, out=np.zeros_like(gaps), where=span > 0.0)
    regrets = np.where(mask, regrets, np.nan)
    return regrets[0] if squeeze else regrets


def soft_teacher_distribution(
    costs: NDArray[np.float64],
    feasible: NDArray[np.bool_],
    temperature: float,
) -> NDArray[np.float64]:
    """Return ``softmax(-normalized_regret / temperature)`` on feasible actions."""
    if temperature <= 0.0 or not np.isfinite(temperature):
        raise ValueError("soft-target temperature must be finite and positive")
    regrets = normalized_regrets(costs, feasible)
    mask = np.asarray(feasible, dtype=np.bool_)
    squeeze = regrets.ndim == 1
    if squeeze:
        regrets = regrets[None, :]
        mask = mask[None, :]
    logits = np.where(mask, -regrets / temperature, -np.inf)
    maximum = np.max(logits, axis=1, keepdims=True)
    valid = np.any(mask, axis=1, keepdims=True)
    shifted = np.where(valid, logits - maximum, -np.inf)
    weights = np.where(mask, np.exp(shifted), 0.0)
    total = np.sum(weights, axis=1, keepdims=True)
    result = np.divide(weights, total, out=np.zeros_like(weights), where=total > 0.0)
    return result[0] if squeeze else result


def ranking_pair_weights(
    costs: NDArray[np.float64],
    feasible: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Return directed target weights with shape ``(state, better, worse)``."""
    values, mask, squeeze = _aligned_costs(costs, feasible)
    safe = np.where(mask, values, 0.0)
    cost_i = safe[:, :, None]
    cost_j = safe[:, None, :]
    feasible_i = mask[:, :, None]
    feasible_j = mask[:, None, :]
    gap = cost_j - cost_i
    meaningful = feasible_i & feasible_j & (gap > pair_tolerance(cost_i, cost_j))
    meaningful_gaps = np.where(meaningful, gap, 0.0)
    largest = np.max(meaningful_gaps, axis=(1, 2), keepdims=True)
    normalized = np.divide(
        meaningful_gaps,
        largest,
        out=np.zeros_like(meaningful_gaps),
        where=largest > 0.0,
    )
    result = np.where(feasible_i & ~feasible_j, 1.0, normalized)
    return result[0] if squeeze else result


def rankable_state_mask(
    costs: NDArray[np.float64],
    feasible: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Return states that contribute at least one pairwise ordering."""
    weights = ranking_pair_weights(costs, feasible)
    if weights.ndim == 2:
        return np.asarray(np.sum(weights) > 0.0)
    return weights.sum(axis=(1, 2)) > 0.0


def _aligned_costs(
    costs: NDArray[np.float64],
    feasible: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.bool_], bool]:
    values = np.asarray(costs, dtype=np.float64)
    mask = np.asarray(feasible, dtype=np.bool_)
    if values.shape != mask.shape or values.ndim not in (1, 2):
        raise ValueError("costs and feasibility must be aligned vectors or matrices")
    if np.any(mask & ~np.isfinite(values)):
        raise ValueError("feasible costs must be finite")
    squeeze = values.ndim == 1
    if squeeze:
        values = values[None, :]
        mask = mask[None, :]
    return values, mask, squeeze
