"""Validated, scenario-neutral datasets for reducer ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from pzr.learning.targets import tolerant_best_mask


@dataclass(frozen=True)
class RankingDataset:
    """Current-state features and masked native teacher costs."""

    features: NDArray[np.float32]
    teacher_costs: NDArray[np.float64]
    feasible: NDArray[np.bool_]
    tie_mask: NDArray[np.bool_]
    candidate_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    splits: tuple[str, ...]
    sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float32).copy()
        costs = np.asarray(self.teacher_costs, dtype=np.float64).copy()
        feasible = np.asarray(self.feasible, dtype=np.bool_).copy()
        tie_mask = np.asarray(self.tie_mask, dtype=np.bool_).copy()
        if features.ndim != 2:
            raise ValueError(f"features must be two-dimensional, got {features.shape}")
        if costs.ndim != 2 or feasible.shape != costs.shape:
            raise ValueError("teacher costs and feasibility mask shapes differ")
        if tie_mask.shape != costs.shape:
            raise ValueError("teacher costs and tie mask shapes differ")
        if features.shape[0] != costs.shape[0]:
            raise ValueError("feature and teacher-cost sample counts differ")
        if costs.shape[1] != len(self.candidate_names):
            raise ValueError("candidate names do not match teacher-cost columns")
        if features.shape[1] != len(self.feature_names):
            raise ValueError("feature names do not match feature columns")
        if len(self.splits) != features.shape[0] or len(self.sample_ids) != features.shape[0]:
            raise ValueError("split or sample identifiers do not match sample count")
        if not self.candidate_names or len(set(self.candidate_names)) != len(self.candidate_names):
            raise ValueError("candidate names must be non-empty and unique")
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature names must be non-empty and unique")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample identifiers must be unique")
        if not np.all(np.isfinite(features)):
            raise ValueError("features contain non-finite values")
        if np.any(~feasible & ~np.isnan(costs)):
            raise ValueError("infeasible candidate costs must be NaN")
        if np.any(feasible & ~np.isfinite(costs)):
            raise ValueError("feasible candidate costs must be finite")
        if costs.shape[0] and np.any(~np.any(feasible, axis=1)):
            raise ValueError("every sample must have at least one feasible candidate")
        if np.any(tie_mask & ~feasible) or (
            costs.shape[0] and np.any(~np.any(tie_mask, axis=1))
        ):
            raise ValueError("tie mask must select at least one feasible candidate")
        if costs.shape[0] and not np.array_equal(
            tie_mask, tolerant_best_mask(costs, feasible),
        ):
            raise ValueError("tie mask does not follow the ranking target tolerance")
        features.setflags(write=False)
        costs.setflags(write=False)
        feasible.setflags(write=False)
        tie_mask.setflags(write=False)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "teacher_costs", costs)
        object.__setattr__(self, "feasible", feasible)
        object.__setattr__(self, "tie_mask", tie_mask)

    @property
    def num_samples(self) -> int:
        return int(self.features.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.features.shape[1])

    @property
    def num_candidates(self) -> int:
        return int(self.teacher_costs.shape[1])

    def indices_for_split(self, split: str) -> NDArray[np.int64]:
        return np.asarray(
            [index for index, value in enumerate(self.splits) if value == split],
            dtype=np.int64,
        )

    def subset(self, indices: Sequence[int]) -> "RankingDataset":
        selected = np.asarray(indices, dtype=np.int64)
        return RankingDataset(
            features=self.features[selected],
            teacher_costs=self.teacher_costs[selected],
            feasible=self.feasible[selected],
            tie_mask=self.tie_mask[selected],
            candidate_names=self.candidate_names,
            feature_names=self.feature_names,
            splits=tuple(self.splits[index] for index in selected),
            sample_ids=tuple(self.sample_ids[index] for index in selected),
        )

    @classmethod
    def concatenate(cls, datasets: Sequence["RankingDataset"]) -> "RankingDataset":
        if not datasets:
            raise ValueError("at least one ranking dataset is required")
        first = datasets[0]
        for dataset in datasets[1:]:
            if dataset.candidate_names != first.candidate_names:
                raise ValueError("cannot combine datasets with different candidates")
            if dataset.feature_names != first.feature_names:
                raise ValueError("cannot combine datasets with different features")
        return cls(
            features=np.concatenate([dataset.features for dataset in datasets]),
            teacher_costs=np.concatenate([dataset.teacher_costs for dataset in datasets]),
            feasible=np.concatenate([dataset.feasible for dataset in datasets]),
            tie_mask=np.concatenate([dataset.tie_mask for dataset in datasets]),
            candidate_names=first.candidate_names,
            feature_names=first.feature_names,
            splits=tuple(value for dataset in datasets for value in dataset.splits),
            sample_ids=tuple(value for dataset in datasets for value in dataset.sample_ids),
        )
