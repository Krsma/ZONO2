"""Inspectable diagnostics for version-2 reducer-ranking targets."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from pzr.learning.dataset import RankingDataset
from pzr.learning.ranker import RankingPolicy, evaluate_ranking
from pzr.learning.targets import rankable_state_mask, tolerant_best_mask


def validation_metrics(
    policy: RankingPolicy,
    dataset: RankingDataset,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Return validation metrics grouped by named dataset label and budget."""
    _validate_labels(dataset, metadata)
    rows = []
    validation = metadata[metadata["split"] == "validation"]
    if validation.empty:
        raise ValueError("training diagnostics require validation samples")
    for (label, budget), indices in validation.groupby(
        ["dataset_label", "budget"], dropna=False,
    ).groups.items():
        selected = np.asarray(list(indices), dtype=np.int64)
        rows.append({
            "dataset_label": str(label),
            "budget": int(budget),
            "sample_count": len(selected),
            **asdict(evaluate_ranking(policy, dataset.subset(selected))),
        })
    return pd.DataFrame(rows).sort_values(
        ["dataset_label", "budget"], ignore_index=True,
    )


def dataset_diagnostics(
    dataset: RankingDataset,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize objective contribution and collection behavior by input stage."""
    _validate_labels(dataset, metadata)
    rankable = rankable_state_mask(dataset.teacher_costs, dataset.feasible)
    safe_costs = np.where(dataset.feasible, dataset.teacher_costs, np.nan)
    spans = np.nanmax(safe_costs, axis=1) - np.nanmin(safe_costs, axis=1)
    total_rankable_by_split = {
        split: int(np.count_nonzero(rankable[metadata["split"].to_numpy() == split]))
        for split in sorted(set(metadata["split"]))
    }
    rows = []
    for (label, split, budget), indices in metadata.groupby(
        ["dataset_label", "split", "budget"], dropna=False,
    ).groups.items():
        selected = np.asarray(list(indices), dtype=np.int64)
        selected_rankable = rankable[selected]
        selected_spans = spans[selected]
        teacher_action = metadata.loc[selected, "teacher_action"] if (
            "teacher_action" in metadata
        ) else None
        behavior_action = metadata.loc[selected, "behavior_action"] if (
            "behavior_action" in metadata
        ) else None
        agreement = (
            float(np.mean(teacher_action.to_numpy() == behavior_action.to_numpy()))
            if teacher_action is not None and behavior_action is not None
            else float("nan")
        )
        split_total = total_rankable_by_split[str(split)]
        rows.append({
            "dataset_label": str(label),
            "split": str(split),
            "budget": int(budget),
            "samples": len(selected),
            "rankable_states": int(np.count_nonzero(selected_rankable)),
            "skipped_tie_states": int(np.count_nonzero(~selected_rankable)),
            "objective_fraction": (
                float(np.count_nonzero(selected_rankable) / split_total)
                if split_total else 0.0
            ),
            "cost_span_q25": float(np.quantile(selected_spans, 0.25)),
            "cost_span_q50": float(np.quantile(selected_spans, 0.50)),
            "cost_span_q75": float(np.quantile(selected_spans, 0.75)),
            "cost_span_q95": float(np.quantile(selected_spans, 0.95)),
            "infeasible_candidate_count": int(
                np.count_nonzero(~dataset.feasible[selected])
            ),
            "infeasible_state_count": int(np.count_nonzero(
                np.any(~dataset.feasible[selected], axis=1)
            )),
            "behavior_teacher_agreement": agreement,
        })
    return pd.DataFrame(rows).sort_values(
        ["dataset_label", "split", "budget"], ignore_index=True,
    )


def candidate_diagnostics(
    policy: RankingPolicy,
    dataset: RankingDataset,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize best labels, selections, and native regret per candidate."""
    _validate_labels(dataset, metadata)
    scores = np.asarray(policy.predict_scores(dataset.features), dtype=np.float64)
    selected_candidate = np.argmin(scores, axis=1)
    tolerant_best = tolerant_best_mask(dataset.teacher_costs, dataset.feasible)
    safe = np.where(dataset.feasible, dataset.teacher_costs, np.inf)
    exact_best = np.argmin(safe, axis=1)
    best_cost = np.min(safe, axis=1)
    rows = []
    validation = metadata[metadata["split"] == "validation"]
    for (label, budget), indices in validation.groupby(
        ["dataset_label", "budget"], dropna=False,
    ).groups.items():
        selected = np.asarray(list(indices), dtype=np.int64)
        for candidate_index, candidate in enumerate(dataset.candidate_names):
            feasible = dataset.feasible[selected, candidate_index]
            regrets = (
                dataset.teacher_costs[selected, candidate_index][feasible]
                - best_cost[selected][feasible]
            )
            rows.append({
                "dataset_label": str(label),
                "budget": int(budget),
                "candidate": candidate,
                "sample_count": len(selected),
                "teacher_best_count": int(np.count_nonzero(
                    exact_best[selected] == candidate_index
                )),
                "tolerant_best_count": int(np.count_nonzero(
                    tolerant_best[selected, candidate_index]
                )),
                "predicted_selection_count": int(np.count_nonzero(
                    selected_candidate[selected] == candidate_index
                )),
                "infeasible_count": int(np.count_nonzero(~feasible)),
                "mean_regret": float(np.mean(regrets)) if regrets.size else float("nan"),
                "median_regret": (
                    float(np.median(regrets)) if regrets.size else float("nan")
                ),
                "max_regret": float(np.max(regrets)) if regrets.size else float("nan"),
            })
    return pd.DataFrame(rows).sort_values(
        ["dataset_label", "budget", "candidate"], ignore_index=True,
    )


def _validate_labels(dataset: RankingDataset, metadata: pd.DataFrame) -> None:
    required = {"dataset_label", "split", "budget"}
    if not required <= set(metadata.columns):
        raise ValueError(f"diagnostic metadata lacks columns: {sorted(required - set(metadata))}")
    if len(metadata) != dataset.num_samples or not np.array_equal(
        metadata.index.to_numpy(), np.arange(dataset.num_samples),
    ):
        raise ValueError("diagnostic metadata must align positionally with the dataset")
