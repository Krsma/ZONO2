import numpy as np
import pytest
import torch

from pzr.learning.dataset import ReducerCostDataset
from pzr.learning.ranker import (
    FeatureNormalizer,
    FeatureSchema,
    ReducerPolicy,
    ReducerScorer,
    cost_sensitive_pairwise_loss,
    soft_distillation_loss,
    train_reducer_policy,
)
from pzr.learning.targets import (
    PAIRWISE_OBJECTIVE_CONTRACT,
    normalized_regrets,
    rankable_state_mask,
    soft_objective_contract,
    soft_teacher_distribution,
    tolerant_best_mask,
)


SCHEMA = FeatureSchema(
    name="test", version=1, feature_names=("left", "right"), log1p_features=("left",),
)


def _dataset() -> ReducerCostDataset:
    return ReducerCostDataset(
        features=np.asarray([
            [0.0, 0.0], [0.0, 1.0], [1.0, 0.0],
            [1.0, 1.0], [2.0, 0.0], [2.0, 1.0],
        ], dtype=np.float32),
        teacher_costs=np.asarray([
            [0.0, 1.0], [0.0, 1.0], [1.0, 0.0],
            [1.0, 0.0], [1.0, np.nan], [1.0, 0.0],
        ]),
        feasible=np.asarray([
            [True, True], [True, True], [True, True],
            [True, True], [True, False], [True, True],
        ]),
        candidate_names=("girard", "scott"),
        feature_names=SCHEMA.feature_names,
        splits=("train", "train", "train", "train", "validation", "validation"),
        sample_ids=tuple(f"sample-{index}" for index in range(6)),
    )


def test_cost_dataset_rejects_silent_candidate_misalignment():
    with pytest.raises(ValueError, match="candidate names"):
        ReducerCostDataset(
            features=np.zeros((1, 2), dtype=np.float32),
            teacher_costs=np.zeros((1, 2)),
            feasible=np.ones((1, 2), dtype=bool),
            candidate_names=("girard",),
            feature_names=("left", "right"),
            splits=("train",),
            sample_ids=("sample",),
        )


def test_pairwise_loss_respects_cost_gaps_and_infeasibility():
    costs = torch.tensor([[0.0, 2.0, float("nan")]], dtype=torch.float64)
    feasible = torch.tensor([[True, True, False]])
    good = cost_sensitive_pairwise_loss(torch.tensor([[-2.0, 0.0, 2.0]], dtype=torch.float64), costs, feasible)
    bad = cost_sensitive_pairwise_loss(torch.tensor([[2.0, 0.0, -2.0]], dtype=torch.float64), costs, feasible)
    assert good < bad


def test_soft_targets_are_invariant_to_positive_cost_rescaling():
    costs = np.asarray([[2.0, 5.0, 11.0]])
    feasible = np.ones_like(costs, dtype=np.bool_)
    np.testing.assert_allclose(
        soft_teacher_distribution(costs, feasible, 0.2),
        soft_teacher_distribution(costs * 1e6, feasible, 0.2),
    )


def test_soft_loss_gives_states_equal_weight_despite_catastrophic_gap():
    costs = np.asarray([[0.0, 1.0], [0.0, 1e30]])
    feasible = np.ones_like(costs, dtype=np.bool_)
    targets = torch.tensor(soft_teacher_distribution(costs, feasible, 0.2))
    scores = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    feasible_tensor = torch.tensor(feasible)
    combined = soft_distillation_loss(
        scores, targets, feasible_tensor, feasibility_penalty=1.0,
    )[0]
    separate = torch.stack([
        soft_distillation_loss(
            scores[index:index + 1], targets[index:index + 1],
            feasible_tensor[index:index + 1], feasibility_penalty=1.0,
        )[0]
        for index in range(2)
    ]).mean()
    torch.testing.assert_close(combined, separate)


def test_soft_target_zeroes_infeasible_probability_and_penalizes_it():
    costs = np.asarray([[0.0, np.nan]])
    feasible = np.asarray([[True, False]])
    target = soft_teacher_distribution(costs, feasible, 0.2)
    np.testing.assert_array_equal(target, [[1.0, 0.0]])
    good = soft_distillation_loss(
        torch.tensor([[-2.0, 2.0]], dtype=torch.float64), torch.tensor(target),
        torch.tensor(feasible), feasibility_penalty=1.0,
    )[0]
    bad = soft_distillation_loss(
        torch.tensor([[2.0, -2.0]], dtype=torch.float64), torch.tensor(target),
        torch.tensor(feasible), feasibility_penalty=1.0,
    )[0]
    assert good < bad


def test_tight_scale_aware_ties_are_uniform_and_skipped_consistently():
    costs = np.asarray([[1e6, 1e6 + 5e-4], [0.0, 2e-15]])
    feasible = np.ones_like(costs, dtype=np.bool_)
    np.testing.assert_array_equal(tolerant_best_mask(costs, feasible), [[True, True], [True, False]])
    np.testing.assert_array_equal(rankable_state_mask(costs, feasible), [False, True])
    np.testing.assert_allclose(normalized_regrets(costs, feasible)[0], [0.0, 0.0])
    np.testing.assert_allclose(soft_teacher_distribution(costs, feasible, 0.1)[0], [0.5, 0.5])


def test_all_infeasible_state_is_skipped_by_soft_loss():
    scores = torch.tensor([[0.0, 1.0]], dtype=torch.float64, requires_grad=True)
    total, kl, penalty = soft_distillation_loss(
        scores, torch.zeros_like(scores), torch.zeros_like(scores, dtype=torch.bool),
        feasibility_penalty=1.0,
    )
    assert float(total.detach()) == float(kl.detach()) == float(penalty.detach()) == 0.0


def test_inference_ties_use_candidate_catalog_order():
    model = ReducerScorer(SCHEMA, ("girard", "scott", "pca"))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    policy = ReducerPolicy(
        model,
        FeatureNormalizer(np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)),
        PAIRWISE_OBJECTIVE_CONTRACT,
    )
    assert policy.rank_candidates(np.asarray([0.0, 0.0])) == ["girard", "scott", "pca"]


@pytest.mark.parametrize("objective,temperature", [("pairwise", None), ("soft-kl", 0.2)])
def test_scorer_trains_deterministically_and_round_trips(tmp_path, objective, temperature):
    left, left_result = train_reducer_policy(
        _dataset(), SCHEMA, objective=objective, temperature=temperature,
        epochs=8, patience=8, seed=7,
    )
    right, right_result = train_reducer_policy(
        _dataset(), SCHEMA, objective=objective, temperature=temperature,
        epochs=8, patience=8, seed=7,
    )
    raw = np.asarray([1.0, 0.5], dtype=np.float32)
    np.testing.assert_allclose(left.predict_scores(raw), right.predict_scores(raw))
    assert left_result.train_loss_history == right_result.train_loss_history
    left.save(tmp_path)
    loaded = ReducerPolicy.load(tmp_path)
    np.testing.assert_allclose(loaded.predict_scores(raw), left.predict_scores(raw))
    assert loaded.objective_contract == (
        PAIRWISE_OBJECTIVE_CONTRACT if objective == "pairwise" else soft_objective_contract(0.2, 1.0)
    )


def test_model_output_dimension_is_fixed_by_candidates():
    model = ReducerScorer(SCHEMA, ("girard", "scott", "pca"))
    assert model(torch.zeros((4, 2))).shape == (4, 3)
