"""Inspectable diagnostics for reducer-cost learning objectives."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from pzr.learning.dataset import ReducerCostDataset
from pzr.learning.ranker import ReducerPolicy, evaluate_reducer
from pzr.learning.objectives import (
    EXPECTED_REGRET_OBJECTIVE_CONTRACT,
    expected_regret_targets,
    normalized_regrets,
    rankable_state_mask,
    soft_teacher_distribution,
    tolerant_best_mask,
)


def validation_metrics(
    policy: ReducerPolicy,
    dataset: ReducerCostDataset,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Return validation metrics grouped by named dataset label and budget."""
    _validate_labels(dataset, metadata)
    rows = []
    validation = metadata[metadata["split"] == "validation"]
    if validation.empty:
        raise ValueError("training diagnostics require validation samples")
    for (label, budget), indices in validation.groupby(["dataset_label", "budget"], dropna=False).groups.items():
        selected = np.asarray(list(indices), dtype=np.int64)
        rows.append({
            "dataset_label": str(label),
            "budget": int(budget),
            "sample_count": len(selected),
            **asdict(evaluate_reducer(policy, dataset.subset(selected))),
        })
    return pd.DataFrame(rows).sort_values(["dataset_label", "budget"], ignore_index=True)


def dataset_diagnostics(
    dataset: ReducerCostDataset,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize objective contribution and collection behavior by named input."""
    _validate_labels(dataset, metadata)
    valid = np.any(dataset.feasible, axis=1)
    rankable = rankable_state_mask(dataset.teacher_costs, dataset.feasible)
    safe_costs = np.where(dataset.feasible, dataset.teacher_costs, np.nan)
    spans = np.full(dataset.num_samples, np.nan, dtype=np.float64)
    spans[valid] = np.nanmax(safe_costs[valid], axis=1) - np.nanmin(safe_costs[valid], axis=1)
    total_valid_by_split = {
        split: int(np.count_nonzero(valid[metadata["split"].to_numpy() == split]))
        for split in sorted(set(metadata["split"]))
    }
    total_rankable_by_split = {
        split: int(np.count_nonzero(rankable[metadata["split"].to_numpy() == split]))
        for split in sorted(set(metadata["split"]))
    }
    rows = []
    for (label, split, budget), indices in metadata.groupby(
        ["dataset_label", "split", "budget"], dropna=False,
    ).groups.items():
        selected = np.asarray(list(indices), dtype=np.int64)
        selected_valid = valid[selected]
        selected_spans = spans[selected][selected_valid]
        teacher = metadata.loc[selected, "teacher_action"].astype(str).to_numpy()
        executed = (
            metadata.loc[selected, "executed_action"].astype(str).to_numpy()
            if "executed_action" in metadata else teacher
        )
        split_total = total_valid_by_split[str(split)]
        split_rankable = total_rankable_by_split[str(split)]
        rows.append({
            "dataset_label": str(label),
            "split": str(split),
            "budget": int(budget),
            "samples": len(selected),
            "valid_states": int(np.count_nonzero(selected_valid)),
            "all_infeasible_states": int(np.count_nonzero(~selected_valid)),
            "rankable_states": int(np.count_nonzero(rankable[selected])),
            "skipped_tie_states": int(np.count_nonzero(selected_valid & ~rankable[selected])),
            "soft_objective_fraction": (
                float(np.count_nonzero(selected_valid) / split_total) if split_total else 0.0
            ),
            "expected_regret_objective_fraction": (
                float(np.count_nonzero(selected_valid) / split_total) if split_total else 0.0
            ),
            "pairwise_objective_fraction": (
                float(np.count_nonzero(rankable[selected]) / split_rankable)
                if split_rankable else 0.0
            ),
            "cost_span_q25": _quantile(selected_spans, 0.25),
            "cost_span_q50": _quantile(selected_spans, 0.50),
            "cost_span_q75": _quantile(selected_spans, 0.75),
            "cost_span_q95": _quantile(selected_spans, 0.95),
            "infeasible_candidate_count": int(np.count_nonzero(~dataset.feasible[selected])),
            "infeasible_state_count": int(np.count_nonzero(np.any(~dataset.feasible[selected], axis=1))),
            "executed_teacher_agreement": float(np.mean(teacher == executed)),
            "disturbed_fraction": (
                float(np.mean(metadata.loc[selected, "disturbed"].astype(bool)))
                if "disturbed" in metadata else 0.0
            ),
            "mean_disturbance_probability": (
                float(np.mean(metadata.loc[selected, "disturbance_probability"].astype(float)))
                if "disturbance_probability" in metadata else 0.0
            ),
            "disturbance_eligible_fraction": (
                float(np.mean(metadata.loc[selected, "disturbance_eligible"].astype(bool)))
                if "disturbance_eligible" in metadata else 0.0
            ),
            "recovery_forced_fraction": (
                float(np.mean(metadata.loc[selected, "recovery_forced"].astype(bool)))
                if "recovery_forced" in metadata else 0.0
            ),
            "mean_target_disturbance_rate": (
                float(np.mean(metadata.loc[selected, "target_disturbance_rate"].astype(float)))
                if "target_disturbance_rate" in metadata else 0.0
            ),
        })
    return pd.DataFrame(rows).sort_values(["dataset_label", "split", "budget"], ignore_index=True)


def candidate_diagnostics(
    policy: ReducerPolicy,
    dataset: ReducerCostDataset,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize best labels, soft mass, selections, and regret per candidate."""
    _validate_labels(dataset, metadata)
    scores = np.asarray(policy.predict_scores(dataset.features), dtype=np.float64)
    predicted_probabilities = np.asarray(
        policy.predict_probabilities(dataset.features), dtype=np.float64,
    )
    selected_candidate = np.argmin(scores, axis=1)
    tolerant_best = tolerant_best_mask(dataset.teacher_costs, dataset.feasible)
    regrets = normalized_regrets(dataset.teacher_costs, dataset.feasible)
    safe = np.where(dataset.feasible, dataset.teacher_costs, np.inf)
    exact_best = np.argmin(safe, axis=1)
    valid = np.any(dataset.feasible, axis=1)
    teacher_probabilities = np.full_like(predicted_probabilities, np.nan)
    regression_targets = np.full_like(scores, np.nan)
    regression_objective = False
    if policy.objective_contract.get("schema") == "pzr.reducer-objective.soft-kl-v1":
        teacher_probabilities = soft_teacher_distribution(
            dataset.teacher_costs,
            dataset.feasible,
            float(policy.objective_contract["temperature"]),
        )
    elif policy.objective_contract.get("schema") == EXPECTED_REGRET_OBJECTIVE_CONTRACT["schema"]:
        regression_objective = True
        regression_targets = expected_regret_targets(
            dataset.teacher_costs, dataset.feasible,
        )
    rows = []
    validation = metadata[metadata["split"] == "validation"]
    for (label, budget), indices in validation.groupby(["dataset_label", "budget"], dropna=False).groups.items():
        selected = np.asarray(list(indices), dtype=np.int64)
        for candidate_index, candidate in enumerate(dataset.candidate_names):
            feasible = dataset.feasible[selected, candidate_index]
            candidate_regrets = regrets[selected, candidate_index][feasible]
            regression_rows = valid[selected] if regression_objective else np.zeros(
                len(selected), dtype=np.bool_,
            )
            candidate_errors = (
                scores[selected, candidate_index][regression_rows]
                - regression_targets[selected, candidate_index][regression_rows]
            )
            finite_errors = candidate_errors[np.isfinite(candidate_errors)]
            candidate_scores = scores[selected, candidate_index][regression_rows]
            rows.append({
                "dataset_label": str(label),
                "budget": int(budget),
                "candidate": candidate,
                "sample_count": len(selected),
                "teacher_best_count": int(np.count_nonzero(valid[selected] & (exact_best[selected] == candidate_index))),
                "tolerant_best_count": int(np.count_nonzero(tolerant_best[selected, candidate_index])),
                "predicted_selection_count": int(np.count_nonzero(selected_candidate[selected] == candidate_index)),
                "mean_teacher_probability": (
                    float(np.mean(teacher_probabilities[selected, candidate_index]))
                    if np.all(np.isfinite(teacher_probabilities[selected, candidate_index]))
                    else float("nan")
                ),
                "mean_predicted_probability": float(np.mean(
                    predicted_probabilities[selected, candidate_index]
                )),
                "infeasible_count": int(np.count_nonzero(~feasible)),
                "mean_normalized_regret": float(np.mean(candidate_regrets)) if candidate_regrets.size else float("nan"),
                "median_normalized_regret": float(np.median(candidate_regrets)) if candidate_regrets.size else float("nan"),
                "max_normalized_regret": float(np.max(candidate_regrets)) if candidate_regrets.size else float("nan"),
                "regression_rmse": (
                    float(np.sqrt(np.mean(np.square(finite_errors))))
                    if finite_errors.size else float("nan")
                ),
                "regression_mae": (
                    float(np.mean(np.abs(finite_errors)))
                    if finite_errors.size else float("nan")
                ),
                "prediction_below_zero_count": int(np.count_nonzero(candidate_scores < 0.0)),
                "prediction_above_two_count": int(np.count_nonzero(candidate_scores > 2.0)),
                "prediction_outside_target_range_count": int(np.count_nonzero(
                    (candidate_scores < 0.0) | (candidate_scores > 2.0)
                )),
            })
    return pd.DataFrame(rows).sort_values(["dataset_label", "budget", "candidate"], ignore_index=True)


def _quantile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile)) if values.size else float("nan")


def _validate_labels(dataset: ReducerCostDataset, metadata: pd.DataFrame) -> None:
    required = {"dataset_label", "split", "budget", "teacher_action"}
    if not required <= set(metadata.columns):
        raise ValueError(f"diagnostic metadata lacks columns: {sorted(required - set(metadata))}")
    if len(metadata) != dataset.num_samples or not np.array_equal(
        metadata.index.to_numpy(), np.arange(dataset.num_samples),
    ):
        raise ValueError("diagnostic metadata must align positionally with the dataset")
