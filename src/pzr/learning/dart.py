"""Discrete DART calibration for one-step disturbed teacher collection."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from pzr.learning.dataset import ReducerCostDataset
from pzr.learning.ranker import ReducerPolicy
from pzr.learning.targets import normalized_regrets, tolerant_best_mask


DART_CALIBRATION_SCHEMA = "pzr.dart-calibration.v1"


@dataclass(frozen=True)
class DartCalibration:
    """Teacher-conditioned categorical novice-confusion distributions."""

    candidate_names: tuple[str, ...]
    budgets: tuple[int, ...]
    probabilities: NDArray[np.float64]
    row_counts: NDArray[np.int64]
    context: Mapping[str, object]

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.probabilities, dtype=np.float64).copy()
        row_counts = np.asarray(self.row_counts, dtype=np.int64).copy()
        expected = (len(self.budgets), len(self.candidate_names), len(self.candidate_names))
        if probabilities.shape != expected:
            raise ValueError("DART probabilities do not match budget/candidate axes")
        if row_counts.shape != expected[:2]:
            raise ValueError("DART row counts do not match budget/teacher axes")
        if len(set(self.budgets)) != len(self.budgets) or tuple(sorted(self.budgets)) != self.budgets:
            raise ValueError("DART budgets must be sorted and unique")
        if not self.candidate_names or len(set(self.candidate_names)) != len(self.candidate_names):
            raise ValueError("DART candidates must be non-empty and unique")
        if np.any(probabilities < 0.0) or not np.all(np.isfinite(probabilities)):
            raise ValueError("DART probabilities must be finite and non-negative")
        if not np.allclose(np.sum(probabilities, axis=2), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("every DART confusion row must sum to one")
        if np.any(row_counts < 0):
            raise ValueError("DART row counts must be non-negative")
        probabilities.setflags(write=False)
        row_counts.setflags(write=False)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "row_counts", row_counts)
        object.__setattr__(self, "context", dict(self.context))

    def collection_distribution(
        self,
        budget: int,
        teacher_action: str,
        feasible: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Mask infeasible disturbance mass back onto the teacher action."""
        try:
            budget_index = self.budgets.index(int(budget))
            teacher_index = self.candidate_names.index(teacher_action)
        except ValueError as exc:
            raise ValueError("DART calibration does not cover this budget/action") from exc
        mask = np.asarray(feasible, dtype=np.bool_)
        if mask.shape != (len(self.candidate_names),):
            raise ValueError("DART feasibility vector does not match candidates")
        if not mask[teacher_index]:
            raise ValueError("teacher action must be feasible")
        distribution = self.probabilities[budget_index, teacher_index].copy()
        redirected = float(np.sum(distribution[~mask]))
        distribution[~mask] = 0.0
        distribution[teacher_index] += redirected
        distribution /= np.sum(distribution)
        return distribution

    def save(self, directory: Path, diagnostics: pd.DataFrame) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": DART_CALIBRATION_SCHEMA,
            "candidate_names": list(self.candidate_names),
            "budgets": list(self.budgets),
            "probabilities": self.probabilities.tolist(),
            "row_counts": self.row_counts.tolist(),
            "context": dict(self.context),
        }
        _write_text_atomic(json.dumps(payload, indent=2, sort_keys=True), directory / "calibration.json")
        _write_csv_atomic(diagnostics, directory / "dart_calibration.csv")

    @classmethod
    def load(cls, directory: Path) -> "DartCalibration":
        payload = json.loads((directory / "calibration.json").read_text())
        if payload.get("schema") != DART_CALIBRATION_SCHEMA:
            raise ValueError("unsupported DART calibration schema")
        return cls(
            candidate_names=tuple(payload["candidate_names"]),
            budgets=tuple(int(value) for value in payload["budgets"]),
            probabilities=np.asarray(payload["probabilities"], dtype=np.float64),
            row_counts=np.asarray(payload["row_counts"], dtype=np.int64),
            context=dict(payload["context"]),
        )


def calibrate_dart(
    policy: ReducerPolicy,
    dataset: ReducerCostDataset,
    metadata: pd.DataFrame,
    *,
    split: str,
    context: Mapping[str, object],
) -> tuple[DartCalibration, pd.DataFrame]:
    """Fit the categorical MLE of novice actions conditioned on teacher actions."""
    required = {"split", "budget", "teacher_action"}
    if not required <= set(metadata.columns):
        raise ValueError(f"DART metadata lacks columns: {sorted(required - set(metadata))}")
    if len(metadata) != dataset.num_samples:
        raise ValueError("DART metadata and dataset are not aligned")
    if policy.candidate_names != dataset.candidate_names:
        raise ValueError("DART model and dataset candidate catalogs differ")
    if policy.feature_schema.feature_names != dataset.feature_names:
        raise ValueError("DART model and dataset feature schemas differ")
    split_rows = np.flatnonzero(metadata["split"].astype(str).to_numpy() == split)
    if split_rows.size == 0:
        raise ValueError("DART calibration split is empty")
    candidate_names = dataset.candidate_names
    candidate_index = {name: index for index, name in enumerate(candidate_names)}
    split_teacher = metadata.iloc[split_rows]["teacher_action"].astype(str).to_numpy()
    usable = np.any(dataset.feasible[split_rows], axis=1) & np.isin(
        split_teacher, candidate_names,
    )
    selected_rows = split_rows[usable]
    if selected_rows.size == 0:
        raise ValueError("DART calibration split has no state with a feasible catalog teacher action")
    excluded_state_count = int(split_rows.size - selected_rows.size)
    budgets = tuple(sorted(int(value) for value in metadata.iloc[selected_rows]["budget"].unique()))
    scores = np.asarray(policy.predict_scores(dataset.features[selected_rows]), dtype=np.float64)
    novice = np.argmin(scores, axis=1)
    tie_mask = tolerant_best_mask(
        dataset.teacher_costs[selected_rows], dataset.feasible[selected_rows],
    )
    regrets = normalized_regrets(
        dataset.teacher_costs[selected_rows], dataset.feasible[selected_rows],
    )
    counts = np.zeros((len(budgets), len(candidate_names), len(candidate_names)), dtype=np.int64)
    row_counts = np.zeros((len(budgets), len(candidate_names)), dtype=np.int64)
    budget_index = {budget: index for index, budget in enumerate(budgets)}
    for local, dataset_row in enumerate(selected_rows):
        teacher_name = str(metadata.iloc[dataset_row]["teacher_action"])
        if teacher_name not in candidate_index:
            raise ValueError("DART teacher action is outside the candidate catalog")
        b_index = budget_index[int(metadata.iloc[dataset_row]["budget"])]
        t_index = candidate_index[teacher_name]
        counts[b_index, t_index, novice[local]] += 1
        row_counts[b_index, t_index] += 1
    probabilities = np.zeros_like(counts, dtype=np.float64)
    for b_index in range(len(budgets)):
        for teacher in range(len(candidate_names)):
            if row_counts[b_index, teacher]:
                probabilities[b_index, teacher] = counts[b_index, teacher] / row_counts[b_index, teacher]
            else:
                probabilities[b_index, teacher, teacher] = 1.0
    calibration = DartCalibration(
        candidate_names=candidate_names,
        budgets=budgets,
        probabilities=probabilities,
        row_counts=row_counts,
        context=context,
    )
    rows = []
    metadata_selected = metadata.iloc[selected_rows].reset_index(drop=True)
    for b_index, budget in enumerate(budgets):
        budget_mask = metadata_selected["budget"].to_numpy() == budget
        for teacher_index, teacher_name in enumerate(candidate_names):
            teacher_mask = metadata_selected["teacher_action"].astype(str).to_numpy() == teacher_name
            local = np.flatnonzero(budget_mask & teacher_mask)
            predicted = novice[local]
            exact_agreement = float(np.mean(predicted == teacher_index)) if local.size else float("nan")
            tolerant_agreement = float(np.mean(tie_mask[local, predicted])) if local.size else float("nan")
            predicted_feasible = dataset.feasible[selected_rows[local], predicted] if local.size else np.asarray([], dtype=bool)
            chosen_regret = regrets[local, predicted] if local.size else np.asarray([], dtype=float)
            rows.append({
                "budget": budget,
                "teacher_action": teacher_name,
                "row_count": int(row_counts[b_index, teacher_index]),
                "excluded_split_state_count": excluded_state_count,
                "exact_agreement": exact_agreement,
                "tolerant_agreement": tolerant_agreement,
                "disagreement_rate": 1.0 - exact_agreement if local.size else float("nan"),
                "predicted_infeasible_count": int(np.count_nonzero(~predicted_feasible)),
                "mean_predicted_normalized_regret": float(np.nanmean(chosen_regret)) if local.size else float("nan"),
                **{
                    f"probability_{name}": float(probabilities[b_index, teacher_index, action])
                    for action, name in enumerate(candidate_names)
                },
            })
    return calibration, pd.DataFrame(rows).sort_values(["budget", "teacher_action"], ignore_index=True)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_text_atomic(value: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value)
    temporary.replace(path)
