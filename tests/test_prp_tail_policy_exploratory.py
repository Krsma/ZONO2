from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


TOOLS = Path(__file__).parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from prp_tail_policy_exploratory import (  # noqa: E402
    BUDGETS,
    CANDIDATES,
    CONFIRMATION_SEEDS,
    DAGGER_TRAIN_SEEDS,
    DAGGER_VALIDATION_SEEDS,
    MIXTURE_SHARES,
    OPTIMIZER_SEEDS,
    SCHEMA,
    SELECTION_SEEDS,
    ExploreConfig,
    RankEnsemblePolicy,
    _load_stage,
    _source_snapshot,
    build_mixture_dataset,
    learner_count_for_share,
    new_methods,
    regret_resample_plan,
    smoke_config,
    stable_rank_ensemble,
)
from pzr.learning.dataset import ReducerCostDataset  # noqa: E402
from pzr.rtlola.engine import RtlolaEvent  # noqa: E402


def _learner_metadata(*, split: str, seeds: tuple[int, ...], rows_per_seed: int = 4) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        for step in range(rows_per_seed):
            rows.append({
                "sample_id": f"learner-{split}-{seed}-{step}",
                "seed": seed,
                "trace_id": f"trace-{seed}",
                "budget": 40,
                "step": step,
                "split": split,
                "selected_normalized_regret": float(step),
            })
    return pd.DataFrame(rows)


def _dataset(metadata: pd.DataFrame, *, clean: bool = False) -> ReducerCostDataset:
    count = len(metadata)
    costs = np.arange(count * 4, dtype=np.float64).reshape(count, 4)
    feasible = np.ones((count, 4), dtype=np.bool_)
    costs[:, 0] = 0.0
    return ReducerCostDataset(
        features=np.arange(count * 15, dtype=np.float32).reshape(count, 15),
        teacher_costs=costs,
        feasible=feasible,
        candidate_names=CANDIDATES,
        feature_names=tuple(f"f{index}" for index in range(15)),
        splits=tuple(metadata["split"].astype(str)),
        sample_ids=tuple(metadata["sample_id"].astype(str)),
    )


def test_seed_roles_and_canonical_cell_counts_are_exact_and_disjoint():
    config = ExploreConfig()
    groups = tuple(map(set, (
        DAGGER_TRAIN_SEEDS,
        DAGGER_VALIDATION_SEEDS,
        SELECTION_SEEDS,
        CONFIRMATION_SEEDS,
    )))
    assert not any(left & right for index, left in enumerate(groups) for right in groups[index + 1:])
    assert config.expected_new_cells == 392
    assert config.expected_reference_cells == 112
    assert config.expected_report_cells == 504
    assert len(new_methods()) == 7
    assert BUDGETS == (40, 80, 120, 150, 200, 250, 500)

    with pytest.raises(ValueError, match="must be disjoint"):
        ExploreConfig(selection_seeds=(318,))


@pytest.mark.parametrize("share", MIXTURE_SHARES)
def test_mixture_share_is_exact_within_one_row_and_trace_quotas_are_balanced(share: float):
    metadata = _learner_metadata(split="train", seeds=(312, 313, 314))
    plan = regret_resample_plan(
        101,
        metadata,
        target_share=share,
        optimizer_seed=42,
        budget=40,
        split="train",
    )
    expected = learner_count_for_share(101, share)
    assert len(plan) == expected
    assert abs(len(plan) / (101 + len(plan)) - share) <= 1.0 / (101 + len(plan))
    quotas = plan.groupby("trace_id").size()
    assert int(quotas.max() - quotas.min()) <= 1


def test_regret_sampling_is_deterministic_weighted_and_seed_separated():
    metadata = _learner_metadata(split="train", seeds=(312,), rows_per_seed=2)
    metadata.loc[:, "selected_normalized_regret"] = (0.0, 1.0)
    first = regret_resample_plan(
        1000, metadata, target_share=0.20, optimizer_seed=42, budget=40, split="train",
    )
    second = regret_resample_plan(
        1000, metadata, target_share=0.20, optimizer_seed=42, budget=40, split="train",
    )
    other = regret_resample_plan(
        1000, metadata, target_share=0.20, optimizer_seed=1042, budget=40, split="train",
    )
    pd.testing.assert_frame_equal(first, second)
    assert tuple(first["source_sample_id"]) != tuple(other["source_sample_id"])
    counts = first["source_sample_id"].value_counts()
    assert counts["learner-train-312-1"] > 10 * counts["learner-train-312-0"]
    assert set(first["optimizer_seed"]) == {42}
    assert set(other["optimizer_seed"]) == {1042}


def test_mixture_preserves_alignment_candidate_symmetry_and_split_disjointness():
    clean_metadata = pd.concat((
        _learner_metadata(split="train", seeds=(0,), rows_per_seed=8),
        _learner_metadata(split="validation", seeds=(20,), rows_per_seed=8),
    ), ignore_index=True)
    clean_metadata["sample_id"] = "clean-" + clean_metadata["sample_id"]
    dagger_metadata = pd.concat((
        _learner_metadata(split="train", seeds=(312,), rows_per_seed=4),
        _learner_metadata(split="validation", seeds=(318,), rows_per_seed=4),
    ), ignore_index=True)
    clean = _dataset(clean_metadata, clean=True)
    dagger = _dataset(dagger_metadata)
    plans = []
    for split, clean_count in (("train", 8), ("validation", 8)):
        plans.append(regret_resample_plan(
            clean_count,
            dagger_metadata,
            target_share=0.20,
            optimizer_seed=42,
            budget=40,
            split=split,
        ))
    aggregate, metadata = build_mixture_dataset(
        clean, clean_metadata, dagger, dagger_metadata, pd.concat(plans, ignore_index=True),
    )

    assert aggregate.candidate_names == CANDIDATES
    assert aggregate.teacher_costs.shape == aggregate.feasible.shape == (len(metadata), 4)
    assert tuple(metadata["sample_id"].astype(str)) == aggregate.sample_ids
    assert set(metadata.loc[
        metadata["training_source"] == "learner_visited_dagger_train", "seed"
    ]) == {312}
    assert set(metadata.loc[
        metadata["training_source"] == "learner_visited_dagger_validation", "seed"
    ]) == {318}
    # Every resampled row carries all four costs from exactly one source row.
    learner = metadata[metadata["training_source"].str.startswith("learner_")]
    source = {sample_id: dagger.teacher_costs[index] for index, sample_id in enumerate(dagger.sample_ids)}
    aggregate_index = {sample_id: index for index, sample_id in enumerate(aggregate.sample_ids)}
    for _, row in learner.iterrows():
        np.testing.assert_array_equal(
            aggregate.teacher_costs[aggregate_index[str(row["sample_id"])]],
            source[str(row["source_sample_id"])],
        )


def test_rank_ensemble_is_scale_invariant_and_single_member_equivalent():
    scores = np.asarray([
        [0.0, 2.0, 1.0, 3.0],
        [4.0, 1.0, 2.0, 3.0],
        [0.5, 0.0, 4.0, 2.0],
    ])
    order, ranks, standardized, _ = stable_rank_ensemble(scores)
    scaled = scores * np.asarray([[10.0], [0.1], [3.0]]) + np.asarray([[9.0], [-2.0], [100.0]])
    scaled_order, scaled_ranks, scaled_standardized, _ = stable_rank_ensemble(scaled)
    np.testing.assert_array_equal(order, scaled_order)
    np.testing.assert_array_equal(ranks, scaled_ranks)
    np.testing.assert_allclose(standardized, scaled_standardized)

    one = np.asarray([[2.0, -1.0, 4.0, 0.0]])
    single_order, _, _, _ = stable_rank_ensemble(one)
    np.testing.assert_array_equal(single_order, np.argsort(one[0], kind="stable"))


def test_rank_ties_use_standardized_scores_then_catalog_order():
    # Candidates 0 and 1 have equal mean rank; standardized score favors 1.
    scores = np.asarray([[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 2.0, 3.0]])
    order, mean_ranks, mean_standardized, _ = stable_rank_ensemble(scores)
    assert mean_ranks[0] == mean_ranks[1]
    assert mean_standardized[0] == mean_standardized[1]
    assert tuple(order[:2]) == (0, 1)  # exact secondary tie reaches catalog order

    scores = np.asarray([[0.0, 3.0, 1.0, 2.0], [2.0, 0.0, 1.0, 3.0]])
    order, mean_ranks, mean_standardized, _ = stable_rank_ensemble(scores)
    tied = [index for index in range(4) if mean_ranks[index] == mean_ranks[0]]
    if len(tied) > 1:
        expected = min(tied, key=lambda index: (mean_standardized[index], index))
        assert int(order[0]) == expected


class _Policy:
    candidate_names = CANDIDATES
    feature_schema = None

    def __init__(self, scores: tuple[float, ...]) -> None:
        from pzr.rtlola.features import RTL_RANKING_FEATURE_SCHEMA

        self.feature_schema = RTL_RANKING_FEATURE_SCHEMA
        self.scores = np.asarray(scores, dtype=np.float32)

    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        assert features.shape == (15,)
        return self.scores


class _Metrics:
    dynamic_generator_count = 41
    dimension = 1


class _Engine:
    def __init__(self) -> None:
        self.branch_calls = 0

    def metrics(self, state: object) -> _Metrics:
        return _Metrics()

    def branch_step(self, state: object, event: object, action: object, budget: int) -> object:
        self.branch_calls += 1
        return object()


def test_direct_ensemble_inference_evaluates_exactly_one_leaf(monkeypatch: pytest.MonkeyPatch):
    import prp_tail_policy_exploratory as module

    monkeypatch.setattr(module, "extract_ranking_features", lambda engine, state, budget: np.zeros(15, dtype=np.float32))
    event = RtlolaEvent(0.0, tuple(float(index) for index in range(6)))
    policy = RankEnsemblePolicy((
        _Policy((0.0, 1.0, 2.0, 3.0)),
        _Policy((1.0, 0.0, 2.0, 3.0)),
        _Policy((0.0, 2.0, 1.0, 3.0)),
    ), (event,))
    engine = _Engine()
    decision = policy.choose(engine, object(), event, 40)
    assert decision.evaluated_leaves == 1
    assert engine.branch_calls == 1
    assert not decision.fallback_used


def test_stale_manifest_is_rejected(tmp_path: Path):
    config = smoke_config(tmp_path)
    stage = tmp_path / "prepare"
    stage.mkdir(parents=True)
    (stage / "manifest.json").write_text(json.dumps({
        "schema": SCHEMA,
        "experiment_fingerprint": "stale",
        "stage": "prepare",
        "status": "completed",
    }))
    with pytest.raises(ValueError, match="stale tail-policy manifest"):
        _load_stage(config, "prepare")


def test_parent_v3_v4_snapshot_is_repeatable_and_hash_verified(tmp_path: Path):
    config = smoke_config(tmp_path)
    first = _source_snapshot(config)
    second = _source_snapshot(config)
    assert first == second
    assert "v4_trace_store" in first
    assert "clean_model:42:40" in first
    assert "clean_model:1042:40" in first
    assert "dagger:312:40" in first
    assert "dagger:318:40" in first
