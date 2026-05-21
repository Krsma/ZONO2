"""DAgger dataset utilities for reducer-selection learning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from pzr.learning.features import DECISION_FEATURE_SCHEMA_VERSION


@dataclass(frozen=True)
class DAggerIteration:
    """One batch of learner-induced states labeled by an expert policy."""

    iteration: int
    rows: pd.DataFrame
    learner_policy: str = ""
    expert_policy: str = "mpc_focused_sequence"


def load_dagger_iterations(
    paths: Sequence[str | Path],
    *,
    expert_method: str = "mpc_focused_sequence",
    predictor_mode: str = "online",
) -> tuple[DAggerIteration, ...]:
    """Load decision-feature CSVs as ordered DAgger iterations."""

    iterations: list[DAggerIteration] = []
    for index, path in enumerate(paths):
        frame = pd.read_csv(path)
        rows = _filter_expert_rows(
            frame,
            expert_method=expert_method,
            predictor_mode=predictor_mode,
        )
        rows = rows.copy()
        rows["dagger_iteration"] = index
        rows["dagger_source"] = str(path)
        iterations.append(
            DAggerIteration(
                iteration=index,
                rows=rows,
                expert_policy=expert_method,
            )
        )
    return tuple(iterations)


def aggregate_dagger_rows(iterations: Iterable[DAggerIteration]) -> pd.DataFrame:
    """Aggregate examples across DAgger iterations with provenance columns."""

    frames = []
    for iteration in iterations:
        rows = iteration.rows.copy()
        rows["dagger_iteration"] = iteration.iteration
        rows["dagger_learner_policy"] = iteration.learner_policy
        rows["dagger_expert_policy"] = iteration.expert_policy
        frames.append(rows)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def class_balanced_indices(labels: Sequence[str], seed: int = 0) -> np.ndarray:
    """Return row indices oversampled to the largest class count."""

    label_array = np.asarray([str(label) for label in labels], dtype=object)
    if label_array.size == 0:
        return np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(label_array, return_counts=True)
    target = int(np.max(counts))
    sampled: list[np.ndarray] = []
    for cls in classes:
        cls_indices = np.flatnonzero(label_array == cls)
        replace = cls_indices.size < target
        sampled.append(rng.choice(cls_indices, size=target, replace=replace))
    result = np.concatenate(sampled)
    rng.shuffle(result)
    return result.astype(np.int64)


def _filter_expert_rows(
    frame: pd.DataFrame,
    *,
    expert_method: str,
    predictor_mode: str,
) -> pd.DataFrame:
    required = {
        "feature_schema_version",
        "method",
        "predictor_mode",
        "chosen_reducer_label",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"DAgger decision data is missing columns: {missing}")
    rows = frame[
        (frame["feature_schema_version"] == DECISION_FEATURE_SCHEMA_VERSION)
        & (frame["method"] == expert_method)
        & (frame["predictor_mode"] == predictor_mode)
        & frame["chosen_reducer_label"].notna()
        & (frame["chosen_reducer_label"].astype(str) != "")
    ].copy()
    if rows.empty:
        raise ValueError(
            f"no DAgger rows found for method={expert_method!r}, "
            f"predictor_mode={predictor_mode!r}"
        )
    return rows
