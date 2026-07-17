from argparse import Namespace
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pzr.learning.artifacts import load_ranking_dataset, write_ranking_dataset
import pzr.learning.cli as learning_cli
from pzr.learning.cli import (
    NamedPath,
    _split_seeds,
    _validate_named_datasets,
    build_parser,
    run_collect,
    run_train,
)
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
        dataset=[NamedPath("base", dataset_dir)],
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
    assert (output / "dataset_diagnostics.csv").stat().st_size > 0
    assert (output / "candidate_diagnostics.csv").stat().st_size > 0
    assert not (output / "model.onnx").exists()
    training = json.loads((output / "training.json").read_text())
    assert training["datasets"][0]["name"] == "base"
    assert len(training["datasets"][0]["sha256"]) == 64
    assert training["target_contract"]["schema"] == "pzr.reducer-ranking-target.v2"
    assert len(training["pzr_source_sha256"]) == 64


def test_evaluate_command_defaults_to_all_fixed_traces_and_exact_lengths(tmp_path):
    args = build_parser().parse_args([
        "evaluate",
        "--model", f"learned_base={tmp_path / 'model'}",
        "--output", str(tmp_path / "evaluation"),
        "--budgets", "40,80",
    ])

    assert args.trace_kinds == (
        "figure8", "figure8_drift", "random", "random_drift",
        "square", "square_drift",
    )
    assert args.budgets == (40, 80)
    assert args.candidates == ("girard", "scott", "pca", "combastel")
    assert args.baselines == (
        "girard", "scott", "pca", "combastel", "mpc_terminal_full_width",
    )
    assert args.model == [NamedPath("learned_base", tmp_path / "model")]
    assert args.length is None
    assert args.horizon == 1
    assert args.workers == 1


def test_generate_command_defaults_to_nominal_random_waypoints(tmp_path):
    args = build_parser().parse_args([
        "generate",
        "--output", str(tmp_path / "traces"),
        "--event-count", "500",
        "--seed-count", "40",
    ])

    assert args.conditions == ("random_waypoint",)
    assert args.event_count == 500
    assert args.seed_start == 0
    assert args.seed_count == 40


def test_collection_parallelizes_and_reuses_validated_trace_budget_shards(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(learning_cli, "default_action_catalog", lambda _names: object())
    trace = SimpleNamespace(
        events=(object(), object()),
        metadata=SimpleNamespace(trace_sha256="trace-hash"),
    )
    stored_trace = SimpleNamespace(
        trace_id="random_waypoint:seed-0",
        condition="random_waypoint",
        seed=0,
        relative_path="random_waypoint:seed-0",
        trace=trace,
    )
    trace_store = SimpleNamespace(
        root=tmp_path / "traces",
        event_count=2,
        conditions=("random_waypoint",),
        manifest_sha256="store-hash",
        traces_for_seed=lambda seed: (stored_trace,) if seed == 0 else (),
    )
    monkeypatch.setattr(
        learning_cli,
        "load_random_waypoint_trace_store",
        lambda _path: trace_store,
    )
    calls = []
    pool_sizes = []

    class ImmediatePool:
        def __init__(self, *, max_workers, mp_context):
            del mp_context
            pool_sizes.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def map(self, function, jobs):
            return tuple(function(job) for job in jobs)

    monkeypatch.setattr(learning_cli, "ProcessPoolExecutor", ImmediatePool)

    def collect(**kwargs):
        calls.append((kwargs["condition"], kwargs["budget"]))
        return (_collected_sample(
            f"{kwargs['trace_id']}:teacher:budget-{kwargs['budget']}:step-0",
            kwargs["split"], kwargs["condition"], kwargs["seed"], kwargs["budget"],
        ),)

    monkeypatch.setattr(learning_cli, "collect_teacher_episode", collect)
    args = Namespace(
        output=tmp_path,
        trace_store=tmp_path / "traces",
        budgets=(40,),
        candidates=("girard", "scott"),
        train_seeds=1,
        validation_seeds=0,
        test_seeds=0,
        seed_start=0,
        workers=2,
        behavior_model=None,
    )

    run_collect(args)
    assert calls == [("random_waypoint", 40)]
    assert pool_sizes == [2]
    calls.clear()
    run_collect(args)
    assert calls == []
    assert pool_sizes == [2]
    dataset, _, manifest = load_ranking_dataset(tmp_path / "dataset")
    assert dataset.num_samples == 1
    assert manifest["shard_count"] == 1
    assert manifest["trace_store_manifest_sha256"] == "store-hash"


def test_learned_behavior_collection_keeps_validation_seeds_held_out(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(learning_cli, "default_action_catalog", lambda _names: object())
    monkeypatch.setattr(learning_cli, "_load_policy", lambda _path: object())
    monkeypatch.setattr(
        learning_cli, "RtlolaRankingPolicy", lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(learning_cli, "model_sha256", lambda _path: "model-hash")
    stored = {}
    for seed in (0, 1):
        trace = SimpleNamespace(
            events=(object(), object()),
            metadata=SimpleNamespace(trace_sha256=f"trace-hash-{seed}"),
        )
        stored[seed] = SimpleNamespace(
            trace_id=f"random_waypoint:seed-{seed}",
            condition="random_waypoint",
            relative_path=f"seed-{seed}.json",
            trace=trace,
        )
    trace_store = SimpleNamespace(
        root=tmp_path / "traces",
        event_count=2,
        conditions=("random_waypoint",),
        manifest_sha256="store-hash",
        traces_for_seed=lambda seed: (stored[seed],),
    )
    monkeypatch.setattr(
        learning_cli, "load_random_waypoint_trace_store", lambda _path: trace_store,
    )

    def collect(**kwargs):
        sample = _collected_sample(
            f"{kwargs['trace_id']}:learned:budget-{kwargs['budget']}:step-0",
            kwargs["split"], kwargs["condition"], kwargs["seed"], kwargs["budget"],
        )
        return (replace(sample, behavior="learned"),)

    monkeypatch.setattr(learning_cli, "collect_teacher_episode", collect)
    run_collect(Namespace(
        output=tmp_path / "dagger",
        trace_store=tmp_path / "traces",
        budgets=(40,),
        candidates=("girard", "scott"),
        train_seeds=1,
        validation_seeds=1,
        test_seeds=0,
        seed_start=0,
        workers=1,
        behavior_model=tmp_path / "model",
    ))

    dataset, metadata, manifest = load_ranking_dataset(tmp_path / "dagger" / "dataset")
    assert dataset.splits == ("train", "validation")
    assert set(metadata.loc[metadata["split"] == "train", "seed"]) == {0}
    assert set(metadata.loc[metadata["split"] == "validation", "seed"]) == {1}
    assert manifest["behavior_model_sha256"] == "model-hash"


def test_named_dataset_alignment_rejects_candidate_mismatch():
    first = RankingDataset(
        features=np.zeros((2, 1), dtype=np.float32),
        teacher_costs=np.asarray([[0.0, 1.0], [0.0, 1.0]]),
        feasible=np.ones((2, 2), dtype=np.bool_),
        tie_mask=np.asarray([[True, False], [True, False]]),
        candidate_names=("girard", "scott"),
        feature_names=("feature",),
        splits=("train", "validation"),
        sample_ids=("a", "b"),
    )
    second = RankingDataset(
        features=first.features,
        teacher_costs=first.teacher_costs,
        feasible=first.feasible,
        tie_mask=first.tie_mask,
        candidate_names=("girard", "pca"),
        feature_names=first.feature_names,
        splits=first.splits,
        sample_ids=("c", "d"),
    )
    manifest = {"target_contract": {"schema": "same"}}

    with np.testing.assert_raises_regex(ValueError, "candidate catalog"):
        _validate_named_datasets(
            (NamedPath("base", SimpleNamespace()), NamedPath("dagger1", SimpleNamespace())),
            ((first, pd.DataFrame(), manifest), (second, pd.DataFrame(), manifest)),
        )


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
