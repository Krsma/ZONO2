from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pzr.learning.artifacts import load_ranking_dataset, write_ranking_dataset
import pzr.learning.cli as learning_cli
from pzr.learning.cli import _split_seeds, build_parser, run_collect, run_train
from pzr.learning.dataset import RankingDataset
from pzr.learning.ranker import RankingPolicy
from pzr.rtlola.features import RTL_RANKING_FEATURE_NAMES
from pzr.rtlola.learning_data import CollectedRankingSample


def test_split_seeds_are_trajectory_disjoint_and_deterministic():
    args = Namespace(
        seed_start=10,
        train_seeds=2,
        validation_seeds=1,
        test_seeds=2,
    )

    assert _split_seeds(args) == (
        ("train", 10),
        ("train", 11),
        ("validation", 12),
        ("test", 13),
        ("test", 14),
    )


def test_staged_train_writes_explicit_pytorch_artifacts(tmp_path):
    feature_count = len(RTL_RANKING_FEATURE_NAMES)
    features = np.arange(4 * feature_count, dtype=np.float32).reshape(
        4, feature_count,
    )
    dataset = RankingDataset(
        features=features,
        teacher_costs=np.asarray([[0.0, 1.0], [1.0, 0.0]] * 2),
        feasible=np.ones((4, 2), dtype=np.bool_),
        tie_mask=np.asarray([[True, False], [False, True]] * 2),
        candidate_names=("girard", "scott"),
        feature_names=RTL_RANKING_FEATURE_NAMES,
        splits=("train", "train", "validation", "test"),
        sample_ids=("a", "b", "c", "d"),
    )
    dataset_dir = tmp_path / "dataset"
    write_ranking_dataset(
        dataset,
        dataset_dir,
        pd.DataFrame({
            "sample_id": dataset.sample_ids,
            "split": dataset.splits,
            "trace_id": ("ta", "tb", "tc", "td"),
            "budget": (10, 10, 10, 10),
            "step": (1, 1, 1, 1),
        }),
        metadata={},
    )
    output = tmp_path / "model"

    run_train(Namespace(
        dataset=[dataset_dir],
        output=output,
        epochs=2,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=2,
        seed=7,
    ))

    policy = RankingPolicy.load(output)
    assert policy.candidate_names == ("girard", "scott")
    assert (output / "weights.pt").stat().st_size > 0
    assert (output / "model.json").stat().st_size > 0
    assert (output / "training.json").stat().st_size > 0
    assert (output / "validation_metrics.csv").stat().st_size > 0
    assert not (output / "model.onnx").exists()


def test_evaluate_command_defaults_to_all_fixed_traces_and_exact_lengths(tmp_path):
    args = build_parser().parse_args([
        "evaluate",
        "--model", str(tmp_path / "model"),
        "--output", str(tmp_path / "evaluation"),
        "--budgets", "40,80",
    ])

    assert args.trace_kinds == (
        "figure8", "figure8_drift", "random", "random_drift",
        "square", "square_drift",
    )
    assert args.budgets == (40, 80)
    assert args.candidates == ("girard", "scott", "pca", "combastel")
    assert args.baselines == ("girard", "scott", "pca", "combastel")


def test_collection_reuses_validated_trace_budget_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(learning_cli, "default_action_catalog", lambda _names: object())
    trace = SimpleNamespace(
        events=(object(), object()),
        metadata=SimpleNamespace(trace_sha256="trace-hash"),
    )
    monkeypatch.setattr(learning_cli, "_load_or_generate_trace", lambda *_args: trace)
    calls = []

    def collect(**kwargs):
        calls.append((kwargs["condition"], kwargs["budget"]))
        return (_collected_sample(
            f"{kwargs['trace_id']}:teacher:budget-{kwargs['budget']}:step-0",
            kwargs["split"], kwargs["condition"], kwargs["seed"], kwargs["budget"],
        ),)

    monkeypatch.setattr(learning_cli, "collect_teacher_episode", collect)
    args = Namespace(
        output=tmp_path,
        event_count=2,
        budgets=(40,),
        candidates=("girard", "scott"),
        conditions=("random_waypoint",),
        train_seeds=1,
        validation_seeds=0,
        test_seeds=0,
        seed_start=0,
        behavior_model=None,
    )

    run_collect(args)
    assert calls == [("random_waypoint", 40)]
    calls.clear()
    run_collect(args)
    assert calls == []
    dataset, _, manifest = load_ranking_dataset(tmp_path / "dataset")
    assert dataset.num_samples == 1
    assert manifest["shard_count"] == 1


def _collected_sample(
    sample_id: str,
    split: str,
    condition: str,
    seed: int,
    budget: int,
) -> CollectedRankingSample:
    return CollectedRankingSample(
        sample_id=sample_id,
        trace_id=f"{condition}:seed-{seed}",
        split=split,
        condition=condition,
        seed=seed,
        budget=budget,
        step=0,
        features=np.arange(len(RTL_RANKING_FEATURE_NAMES), dtype=np.float32),
        candidate_names=("girard", "scott"),
        teacher_costs=(0.0, 1.0),
        feasible=(True, True),
        tie_mask=(True, False),
        teacher_action="girard",
        teacher_sequence=("girard", "none"),
        behavior="teacher",
        behavior_action="girard",
        evaluated_leaves=4,
        teacher_reducer_failure_count=0,
        teacher_infeasible_candidate_count=0,
        behavior_reducer_failure_count=0,
        behavior_infeasible_candidate_count=0,
        behavior_fallback_used=False,
    )
