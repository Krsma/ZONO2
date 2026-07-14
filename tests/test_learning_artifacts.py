import numpy as np
import pandas as pd
import pytest

from pzr.learning.artifacts import load_ranking_dataset, write_ranking_dataset
from pzr.learning.dataset import RankingDataset


def _dataset() -> RankingDataset:
    return RankingDataset(
        features=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        teacher_costs=np.asarray([[0.0, 1.0], [2.0, np.nan]]),
        feasible=np.asarray([[True, True], [True, False]]),
        tie_mask=np.asarray([[True, False], [True, False]]),
        candidate_names=("girard", "scott"),
        feature_names=("count", "width"),
        splits=("train", "validation"),
        sample_ids=("trace-a:0", "trace-b:0"),
    )


def _metadata() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "sample_id": "trace-a:0", "split": "train",
            "trace_id": "trace-a", "budget": 10, "step": 0,
        },
        {
            "sample_id": "trace-b:0", "split": "validation",
            "trace_id": "trace-b", "budget": 10, "step": 0,
        },
    ])


def test_ranking_dataset_artifact_round_trip_is_non_empty(tmp_path):
    write_ranking_dataset(_dataset(), tmp_path, _metadata(), {"teacher": "full_width"})
    loaded, metadata, manifest = load_ranking_dataset(tmp_path)

    np.testing.assert_array_equal(loaded.features, _dataset().features)
    np.testing.assert_allclose(
        loaded.teacher_costs, _dataset().teacher_costs, equal_nan=True,
    )
    assert tuple(metadata["sample_id"]) == loaded.sample_ids
    assert manifest["teacher"] == "full_width"
    for name in ("samples.npz", "samples.csv", "candidate_costs.csv", "manifest.json"):
        assert (tmp_path / name).stat().st_size > 0


def test_ranking_dataset_artifact_rejects_misaligned_metadata(tmp_path):
    metadata = _metadata().iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="identifiers"):
        write_ranking_dataset(_dataset(), tmp_path, metadata, {})


def test_empty_ranking_shard_round_trip_keeps_explicit_schema(tmp_path):
    dataset = RankingDataset(
        features=np.empty((0, 2), dtype=np.float32),
        teacher_costs=np.empty((0, 2), dtype=np.float64),
        feasible=np.empty((0, 2), dtype=np.bool_),
        tie_mask=np.empty((0, 2), dtype=np.bool_),
        candidate_names=("girard", "scott"),
        feature_names=("count", "width"),
        splits=(),
        sample_ids=(),
    )
    metadata = pd.DataFrame(columns=(
        "sample_id", "split", "trace_id", "budget", "step",
    ))

    write_ranking_dataset(dataset, tmp_path, metadata, {"collection_shard": True})
    loaded, loaded_metadata, manifest = load_ranking_dataset(tmp_path)

    assert loaded.num_samples == 0
    assert loaded.features.shape == (0, 2)
    assert loaded_metadata.empty
    assert manifest["collection_shard"] is True
