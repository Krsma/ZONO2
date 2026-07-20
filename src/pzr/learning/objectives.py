"""Cost contracts, targets, and losses for reducer-learning objectives."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from torch.nn import functional as F


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
EXPECTED_REGRET_INFEASIBLE_TARGET = 2.0
EXPECTED_REGRET_OBJECTIVE_CONTRACT: dict[str, object] = {
    "schema": "pzr.reducer-objective.expected-regret-v1",
    "normalization": "tolerance_zeroed_gap_over_largest_meaningful_feasible_gap_in_state",
    "absolute_tolerance": ABSOLUTE_TOLERANCE,
    "relative_tolerance": RELATIVE_TOLERANCE,
    "feasible_target_range": [0.0, 1.0],
    "infeasible_target": EXPECTED_REGRET_INFEASIBLE_TARGET,
    "all_infeasible_states": "skipped_and_reported",
    "loss": "mean_squared_error",
    "reduction": "candidate_mean_within_state_then_valid_state_mean",
    "score_semantics": "lower_is_better_raw_conditional_mean_penalized_regret",
    "prediction_clamping": "none",
}
ObjectiveName = Literal["pairwise", "soft-kl", "expected-regret"]


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


def objective_contract(
    objective: ObjectiveName,
    *,
    temperature: float | None,
    feasibility_penalty: float,
) -> dict[str, object]:
    """Return the validated serialized contract for one training objective."""
    if objective == "pairwise":
        if temperature is not None:
            raise ValueError("pairwise training does not accept a temperature")
        return dict(PAIRWISE_OBJECTIVE_CONTRACT)
    if objective == "soft-kl":
        if temperature is None:
            raise ValueError("soft-KL training requires a temperature")
        return soft_objective_contract(temperature, feasibility_penalty)
    if objective == "expected-regret":
        if temperature is not None:
            raise ValueError("expected-regret training does not accept a temperature")
        return dict(EXPECTED_REGRET_OBJECTIVE_CONTRACT)
    raise ValueError(f"unsupported training objective: {objective}")


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
    best = np.where(np.any(mask, axis=1, keepdims=True), best, 0.0)
    gaps = values - best
    meaningful = mask & (gaps > pair_tolerance(values, best))
    gaps = np.where(meaningful, gaps, 0.0)
    span = np.max(gaps, axis=1, keepdims=True)
    regrets = np.divide(gaps, span, out=np.zeros_like(gaps), where=span > 0.0)
    regrets = np.where(mask, regrets, np.nan)
    return regrets[0] if squeeze else regrets


def expected_regret_targets(
    costs: NDArray[np.float64],
    feasible: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Return feasible normalized regrets and the fixed infeasible target ``2``."""
    regrets = normalized_regrets(costs, feasible)
    mask = np.asarray(feasible, dtype=np.bool_)
    return np.where(mask, regrets, EXPECTED_REGRET_INFEASIBLE_TARGET)


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


def cost_sensitive_pairwise_loss(
    scores: Tensor,
    teacher_costs: Tensor,
    feasible: Tensor,
) -> Tensor:
    """Average independently normalized pairwise softplus losses over states."""
    if scores.shape != teacher_costs.shape or feasible.shape != scores.shape:
        raise ValueError("score, cost, and feasibility tensors must align")
    safe_costs = torch.where(feasible, teacher_costs, torch.zeros_like(teacher_costs))
    cost_i = safe_costs.unsqueeze(2)
    cost_j = safe_costs.unsqueeze(1)
    feasible_i = feasible.unsqueeze(2)
    feasible_j = feasible.unsqueeze(1)
    gap = cost_j - cost_i
    tolerance = torch.maximum(
        torch.full_like(gap, ABSOLUTE_TOLERANCE),
        RELATIVE_TOLERANCE * torch.maximum(cost_i.abs(), cost_j.abs()),
    )
    ranked = feasible_i & feasible_j & (gap > tolerance)
    meaningful_gaps = torch.where(ranked, gap, torch.zeros_like(gap))
    largest_gap = meaningful_gaps.amax(dim=(1, 2), keepdim=True)
    weights = torch.where(
        largest_gap > 0.0,
        meaningful_gaps / torch.clamp_min(largest_gap, ABSOLUTE_TOLERANCE),
        torch.zeros_like(gap),
    )
    weights = torch.where(feasible_i & ~feasible_j, torch.ones_like(weights), weights)
    score_margin = scores.unsqueeze(2) - scores.unsqueeze(1)
    state_weight = weights.sum(dim=(1, 2))
    rankable = state_weight > 0.0
    if not bool(torch.any(rankable)):
        return scores.sum() * 0.0
    state_loss = (weights * F.softplus(score_margin)).sum(dim=(1, 2))
    return (state_loss[rankable] / state_weight[rankable]).mean()


def soft_distillation_loss(
    scores: Tensor,
    teacher_probabilities: Tensor,
    feasible: Tensor,
    *,
    feasibility_penalty: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return state-balanced total, KL, and infeasible-mass losses."""
    if scores.shape != teacher_probabilities.shape or feasible.shape != scores.shape:
        raise ValueError("score, target-probability, and feasibility tensors must align")
    valid = feasible.any(dim=1)
    if not bool(torch.any(valid)):
        zero = scores.sum() * 0.0
        return zero, zero, zero
    log_probability = F.log_softmax(-scores, dim=1)
    probability = torch.softmax(-scores, dim=1)
    positive = teacher_probabilities > 0.0
    log_teacher = torch.where(
        positive,
        torch.log(torch.clamp_min(teacher_probabilities, torch.finfo(scores.dtype).tiny)),
        torch.zeros_like(teacher_probabilities),
    )
    state_kl = torch.where(
        positive,
        teacher_probabilities * (log_teacher - log_probability),
        torch.zeros_like(scores),
    ).sum(dim=1)
    state_infeasible = torch.where(
        ~feasible, probability, torch.zeros_like(probability),
    ).sum(dim=1)
    kl = state_kl[valid].mean()
    infeasible_mass = state_infeasible[valid].mean()
    return kl + feasibility_penalty * infeasible_mass, kl, infeasible_mass


def expected_regret_loss(
    scores: Tensor,
    targets: Tensor,
    feasible: Tensor,
) -> Tensor:
    """Average candidate-mean squared error equally over valid states."""
    if scores.shape != targets.shape or feasible.shape != scores.shape:
        raise ValueError("score, target, and feasibility tensors must align")
    valid = feasible.any(dim=1)
    if not bool(torch.any(valid)):
        return scores.sum() * 0.0
    state_mse = torch.square(scores - targets).mean(dim=1)
    return state_mse[valid].mean()


def validate_objective_contract(contract: dict[str, object]) -> None:
    """Reject incomplete or altered serialized objective contracts."""
    schema = contract.get("schema")
    if schema == PAIRWISE_OBJECTIVE_CONTRACT["schema"]:
        if contract != PAIRWISE_OBJECTIVE_CONTRACT:
            raise ValueError("pairwise objective contract differs")
        return
    if schema == "pzr.reducer-objective.soft-kl-v1":
        expected = soft_objective_contract(
            float(contract["temperature"]),
            float(contract["feasibility_penalty"]),
        )
        if contract != expected:
            raise ValueError("soft-KL objective contract differs")
        return
    if schema == EXPECTED_REGRET_OBJECTIVE_CONTRACT["schema"]:
        if contract != EXPECTED_REGRET_OBJECTIVE_CONTRACT:
            raise ValueError("expected-regret objective contract differs")
        return
    raise ValueError("unsupported reducer objective contract")
