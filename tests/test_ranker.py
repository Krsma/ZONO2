import numpy as np
import pytest
import torch

from pzr.learning.dataset import RankingDataset
from pzr.learning.ranker import (
    FeatureSchema,
    RankingPolicy,
    ReducerRanker,
    cost_sensitive_pairwise_loss,
    train_ranking_policy,
)


SCHEMA = FeatureSchema(
    name="test",
    version=1,
    feature_names=("left", "right"),
    log1p_features=("left",),
)


def _dataset() -> RankingDataset:
    return RankingDataset(
        features=np.asarray([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ], dtype=np.float32),
        teacher_costs=np.asarray([
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, np.nan],
            [1.0, 0.0],
        ]),
        feasible=np.asarray([
            [True, True],
            [True, True],
            [True, True],
            [True, True],
            [True, False],
            [True, True],
        ]),
        tie_mask=np.asarray([
            [True, False],
            [True, False],
            [False, True],
            [False, True],
            [True, False],
            [False, True],
        ]),
        candidate_names=("girard", "scott"),
        feature_names=SCHEMA.feature_names,
        splits=("train", "train", "train", "train", "validation", "validation"),
        sample_ids=tuple(f"sample-{index}" for index in range(6)),
    )


def test_ranking_dataset_rejects_silent_candidate_misalignment():
    with pytest.raises(ValueError, match="candidate names"):
        RankingDataset(
            features=np.zeros((1, 2), dtype=np.float32),
            teacher_costs=np.zeros((1, 2)),
            feasible=np.ones((1, 2), dtype=bool),
            tie_mask=np.asarray([[True, False]]),
            candidate_names=("girard",),
            feature_names=("left", "right"),
            splits=("train",),
            sample_ids=("sample",),
        )


def test_pairwise_loss_respects_cost_gaps_and_infeasibility():
    costs = torch.tensor([[0.0, 2.0, float("nan")]], dtype=torch.float64)
    feasible = torch.tensor([[True, True, False]])
    good = cost_sensitive_pairwise_loss(
        torch.tensor([[-2.0, 0.0, 2.0]], dtype=torch.float64), costs, feasible,
    )
    bad = cost_sensitive_pairwise_loss(
        torch.tensor([[2.0, 0.0, -2.0]], dtype=torch.float64), costs, feasible,
    )
    assert good < bad


def test_ranker_trains_deterministically_and_round_trips(tmp_path):
    left, left_result = train_ranking_policy(
        _dataset(), SCHEMA, epochs=8, patience=8, seed=7,
    )
    right, right_result = train_ranking_policy(
        _dataset(), SCHEMA, epochs=8, patience=8, seed=7,
    )
    raw = np.asarray([1.0, 0.5], dtype=np.float32)
    np.testing.assert_allclose(left.predict_scores(raw), right.predict_scores(raw))
    assert left_result.train_loss_history == right_result.train_loss_history
    left.save(tmp_path)
    loaded = RankingPolicy.load(tmp_path)
    np.testing.assert_allclose(loaded.predict_scores(raw), left.predict_scores(raw))
    assert loaded.candidate_names == ("girard", "scott")


def test_model_output_dimension_is_fixed_by_candidates():
    model = ReducerRanker(SCHEMA, ("girard", "scott", "pca"))
    assert model(torch.zeros((4, 2))).shape == (4, 3)
