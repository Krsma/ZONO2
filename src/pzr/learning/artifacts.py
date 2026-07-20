"""Versioned, inspectable reducer-cost dataset artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from pzr.artifact_io import write_csv_atomic, write_json_atomic
from pzr.learning.dataset import ReducerCostDataset
from pzr.learning.objectives import COST_CONTRACT


REDUCER_COST_DATASET_SCHEMA = "pzr.reducer-cost-dataset.v5"
CANDIDATE_COST_COLUMNS = (
    "sample_id", "candidate", "candidate_index", "feasible", "teacher_cost",
)


def write_reducer_cost_dataset(
    dataset: ReducerCostDataset,
    directory: Path,
    sample_metadata: pd.DataFrame,
    metadata: Mapping[str, object],
) -> None:
    """Write aligned arrays, provenance, and long-form teacher costs."""
    _validate_sample_metadata(dataset, sample_metadata)
    directory.mkdir(parents=True, exist_ok=True)
    arrays_path = directory / "samples.npz"
    temporary_arrays = directory / ".samples.npz.tmp"
    with temporary_arrays.open("wb") as handle:
        np.savez_compressed(
            handle,
            features=dataset.features,
            teacher_costs=dataset.teacher_costs,
            feasible=dataset.feasible,
            splits=np.asarray(dataset.splits),
            sample_ids=np.asarray(dataset.sample_ids),
        )
    temporary_arrays.replace(arrays_path)
    write_csv_atomic(sample_metadata, directory / "samples.csv")
    write_csv_atomic(_candidate_cost_frame(dataset), directory / "candidate_costs.csv")
    manifest = {
        "schema": REDUCER_COST_DATASET_SCHEMA,
        "num_samples": dataset.num_samples,
        "candidate_names": list(dataset.candidate_names),
        "feature_names": list(dataset.feature_names),
        "cost_contract": COST_CONTRACT,
        "splits": {
            split: dataset.splits.count(split)
            for split in sorted(set(dataset.splits))
        },
        **dict(metadata),
    }
    write_json_atomic(manifest, directory / "manifest.json")


def load_reducer_cost_dataset(
    directory: Path,
) -> tuple[ReducerCostDataset, pd.DataFrame, dict[str, object]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != REDUCER_COST_DATASET_SCHEMA:
        raise ValueError("unsupported reducer-cost dataset schema")
    if manifest.get("cost_contract") != COST_CONTRACT:
        raise ValueError("reducer-cost dataset contract differs")
    with np.load(directory / "samples.npz", allow_pickle=False) as arrays:
        dataset = ReducerCostDataset(
            features=arrays["features"],
            teacher_costs=arrays["teacher_costs"],
            feasible=arrays["feasible"],
            candidate_names=tuple(manifest["candidate_names"]),
            feature_names=tuple(manifest["feature_names"]),
            splits=tuple(str(value) for value in arrays["splits"]),
            sample_ids=tuple(str(value) for value in arrays["sample_ids"]),
        )
    if dataset.num_samples != int(manifest["num_samples"]):
        raise ValueError("reducer-cost dataset manifest sample count differs")
    sample_metadata = pd.read_csv(directory / "samples.csv")
    _validate_sample_metadata(dataset, sample_metadata)
    expected_costs = _candidate_cost_frame(dataset)
    actual_costs = pd.read_csv(directory / "candidate_costs.csv")
    if list(actual_costs.columns) != list(expected_costs.columns):
        raise ValueError("candidate-cost artifact columns differ")
    if len(actual_costs) != len(expected_costs):
        raise ValueError("candidate-cost artifact row count differs")
    return dataset, sample_metadata, manifest


def _candidate_cost_frame(dataset: ReducerCostDataset) -> pd.DataFrame:
    rows = []
    for sample, sample_id in enumerate(dataset.sample_ids):
        for candidate, name in enumerate(dataset.candidate_names):
            rows.append({
                "sample_id": sample_id,
                "candidate": name,
                "candidate_index": candidate,
                "feasible": bool(dataset.feasible[sample, candidate]),
                "teacher_cost": dataset.teacher_costs[sample, candidate],
            })
    return pd.DataFrame(rows, columns=CANDIDATE_COST_COLUMNS)


def _validate_sample_metadata(dataset: ReducerCostDataset, metadata: pd.DataFrame) -> None:
    required = {"sample_id", "split", "trace_id", "budget", "step"}
    if not required <= set(metadata.columns):
        raise ValueError(f"sample metadata lacks columns: {sorted(required - set(metadata))}")
    if len(metadata) != dataset.num_samples:
        raise ValueError("sample metadata row count differs from dataset")
    if tuple(metadata["sample_id"].astype(str)) != dataset.sample_ids:
        raise ValueError("sample metadata identifiers are not aligned")
    if tuple(metadata["split"].astype(str)) != dataset.splits:
        raise ValueError("sample metadata splits are not aligned")
