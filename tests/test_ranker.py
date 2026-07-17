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
from pzr.learning.targets import rankable_state_mask, tolerant_best_mask


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


def test_pairwise_loss_is_invariant_to_positive_cost_rescaling():
    scores = torch.tensor([[0.3, -0.2, 0.8]], dtype=torch.float64)
    costs = torch.tensor([[2.0, 5.0, 11.0]], dtype=torch.float64)
    feasible = torch.ones_like(costs, dtype=torch.bool)

    original = cost_sensitive_pairwise_loss(scores, costs, feasible)
    rescaled = cost_sensitive_pairwise_loss(scores, costs * 1e6, feasible)

    torch.testing.assert_close(original, rescaled)


def test_pairwise_loss_gives_states_equal_weight_despite_catastrophic_gap():
    scores = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    costs = torch.tensor([[0.0, 1.0], [0.0, 1e30]], dtype=torch.float64)
    feasible = torch.ones_like(costs, dtype=torch.bool)

    combined = cost_sensitive_pairwise_loss(scores, costs, feasible)
    separate = torch.stack([
        cost_sensitive_pairwise_loss(scores[index:index + 1], costs[index:index + 1], feasible[index:index + 1])
        for index in range(2)
    ]).mean()

    torch.testing.assert_close(combined, separate)


def test_tight_scale_aware_ties_are_skipped_consistently():
    costs = np.asarray([[1e6, 1e6 + 5e-4], [0.0, 2e-15]])
    feasible = np.ones_like(costs, dtype=np.bool_)

    np.testing.assert_array_equal(tolerant_best_mask(costs, feasible), [[True, True], [True, False]])
    np.testing.assert_array_equal(rankable_state_mask(costs, feasible), [False, True])


def test_inference_ties_use_candidate_catalog_order():
    model = ReducerRanker(SCHEMA, ("girard", "scott", "pca"))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    policy = RankingPolicy(
        model,
        normalizer=type("Normalizer", (), {
            "mean": np.zeros(2, dtype=np.float32),
            "std": np.ones(2, dtype=np.float32),
            "transform": lambda self, values: np.asarray(values, dtype=np.float32),
        })(),
    )

    assert policy.rank_candidates(np.asarray([0.0, 0.0])) == ["girard", "scott", "pca"]


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
