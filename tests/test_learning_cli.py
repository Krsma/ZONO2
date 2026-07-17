from argparse import Namespace
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pzr.learning.artifacts import load_reducer_cost_dataset, write_reducer_cost_dataset
import pzr.learning.cli as learning_cli
from pzr.learning.cli import NamedPath, _split_seeds, _validate_named_datasets, build_parser, run_collect, run_train
from pzr.learning.dataset import ReducerCostDataset
from pzr.learning.ranker import ReducerPolicy
from pzr.rtlola.features import RTL_RANKING_FEATURE_NAMES
from pzr.rtlola.learning_data import CollectedReducerCostSample


def test_split_seeds_are_trajectory_disjoint_and_deterministic():
    args = Namespace(seed_start=10, train_seeds=2, validation_seeds=1, test_seeds=2)
    assert _split_seeds(args) == (
        ("train", 10), ("train", 11), ("validation", 12), ("test", 13), ("test", 14),
    )


def _training_dataset() -> ReducerCostDataset:
    feature_count = len(RTL_RANKING_FEATURE_NAMES)
    return ReducerCostDataset(
        features=np.arange(4 * feature_count, dtype=np.float32).reshape(4, feature_count),
        teacher_costs=np.asarray([[0.0, 1.0], [1.0, 0.0]] * 2),
        feasible=np.ones((4, 2), dtype=np.bool_),
        candidate_names=("girard", "scott"), feature_names=RTL_RANKING_FEATURE_NAMES,
        splits=("train", "train", "validation", "validation"),
        sample_ids=("a", "b", "c", "d"),
    )


def _write_training_dataset(path, dataset=None):
    dataset = dataset or _training_dataset()
    write_reducer_cost_dataset(
        dataset, path,
        pd.DataFrame({
            "sample_id": dataset.sample_ids, "split": dataset.splits,
            "trace_id": ("ta", "tb", "tc", "td"), "budget": (10,) * 4,
            "step": (1,) * 4, "teacher_action": ("girard", "scott", "girard", "scott"),
            "collection_mode": ("teacher",) * 4, "executed_action": ("girard", "scott", "girard", "scott"),
            "disturbed": (False,) * 4, "disturbance_probability": (0.0,) * 4,
        }), {},
    )


def test_staged_soft_train_selects_temperature_and_writes_artifacts(tmp_path):
    dataset_dir = tmp_path / "dataset"
    _write_training_dataset(dataset_dir)
    output = tmp_path / "model"
    run_train(Namespace(
        dataset=[NamedPath("clean", dataset_dir)], output=output, objective="soft-kl",
        temperature_grid=(0.1, 0.2), temperature_from=None, feasibility_penalty=1.0,
        epochs=2, batch_size=2, learning_rate=1e-3, weight_decay=1e-4,
        patience=2, seed=7,
    ))
    policy = ReducerPolicy.load(output)
    assert policy.candidate_names == ("girard", "scott")
    for name in (
        "weights.pt", "model.json", "training.json", "temperature_selection.csv",
        "validation_metrics.csv", "dataset_diagnostics.csv", "candidate_diagnostics.csv",
    ):
        assert (output / name).stat().st_size > 0
    training = json.loads((output / "training.json").read_text())
    assert training["datasets"][0]["name"] == "clean"
    assert training["objective_contract"]["schema"] == "pzr.reducer-objective.soft-kl-v1"
    assert training["selected_temperature"] in (0.1, 0.2)


def test_soft_dart_training_reuses_source_temperature(tmp_path):
    dataset_dir = tmp_path / "dataset"
    _write_training_dataset(dataset_dir)
    clean_model = tmp_path / "clean-model"
    common = dict(
        dataset=[NamedPath("clean", dataset_dir)], objective="soft-kl",
        feasibility_penalty=1.0, epochs=1, batch_size=2, learning_rate=1e-3,
        weight_decay=1e-4, patience=1, seed=7,
    )
    run_train(Namespace(output=clean_model, temperature_grid=(0.2,), temperature_from=None, **common))
    dart_model = tmp_path / "dart-model"
    run_train(Namespace(output=dart_model, temperature_grid=None, temperature_from=clean_model, **common))
    assert ReducerPolicy.load(dart_model).objective_contract["temperature"] == 0.2


def test_cli_removes_behavior_model_and_exposes_dart_contract(tmp_path):
    args = build_parser().parse_args([
        "collect", "--output", str(tmp_path / "out"), "--trace-store", str(tmp_path / "traces"),
        "--budgets", "40", "--collection-mode", "dart", "--dart-calibration", str(tmp_path / "cal"),
    ])
    assert args.collection_mode == "dart"
    assert not hasattr(args, "behavior_model")
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "collect", "--output", str(tmp_path / "out"), "--trace-store", str(tmp_path / "traces"),
            "--budgets", "40", "--behavior-model", str(tmp_path / "model"),
        ])


def test_evaluate_defaults_to_all_fixed_traces_and_baselines(tmp_path):
    args = build_parser().parse_args([
        "evaluate", "--model", f"learned_soft_dart={tmp_path / 'model'}",
        "--output", str(tmp_path / "evaluation"), "--budgets", "40,80",
    ])
    assert args.trace_kinds == (
        "figure8", "figure8_drift", "random", "random_drift", "square", "square_drift",
    )
    assert args.baselines == ("girard", "scott", "pca", "combastel", "mpc_terminal_full_width")
    assert args.length is None


def test_collection_parallelizes_and_resumes_validated_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(learning_cli, "default_action_catalog", lambda _names: object())
    trace = SimpleNamespace(events=(object(), object()), metadata=SimpleNamespace(trace_sha256="trace-hash"))
    stored_trace = SimpleNamespace(
        trace_id="random_waypoint:seed-0", condition="random_waypoint", seed=0,
        relative_path="seed-0", trace=trace,
    )
    trace_store = SimpleNamespace(
        root=tmp_path / "traces", event_count=2, conditions=("random_waypoint",),
        manifest_sha256="store-hash", traces_for_seed=lambda seed: (stored_trace,) if seed == 0 else (),
    )
    monkeypatch.setattr(learning_cli, "load_random_waypoint_trace_store", lambda _path: trace_store)
    calls = []
    pool_sizes = []

    class ImmediatePool:
        def __init__(self, *, max_workers, mp_context):
            del mp_context
            pool_sizes.append(max_workers)
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def map(self, function, jobs): return tuple(function(job) for job in jobs)

    monkeypatch.setattr(learning_cli, "ProcessPoolExecutor", ImmediatePool)
    monkeypatch.setattr(
        learning_cli, "collect_teacher_episode",
        lambda **kwargs: (calls.append((kwargs["collection_mode"], kwargs["budget"])) or _collected_sample(
            f"{kwargs['trace_id']}:teacher:budget-{kwargs['budget']}:step-0",
            kwargs["split"], kwargs["condition"], kwargs["seed"], kwargs["budget"],
        ),),
    )
    args = Namespace(
        output=tmp_path, trace_store=tmp_path / "traces", budgets=(40,),
        candidates=("girard", "scott"), train_seeds=1, validation_seeds=0,
        test_seeds=0, seed_start=0, workers=2, collection_mode="teacher",
        dart_calibration=None, disturbance_seed=3,
    )
    run_collect(args)
    assert calls == [("teacher", 40)]
    calls.clear()
    run_collect(args)
    assert calls == []
    assert pool_sizes == [2]
    dataset, _, manifest = load_reducer_cost_dataset(tmp_path / "dataset")
    assert dataset.num_samples == 1
    assert manifest["collection_mode"] == "teacher"


def test_named_dataset_alignment_rejects_candidate_mismatch():
    first = ReducerCostDataset(
        features=np.zeros((2, 1), dtype=np.float32), teacher_costs=np.asarray([[0.0, 1.0]] * 2),
        feasible=np.ones((2, 2), dtype=np.bool_), candidate_names=("girard", "scott"),
        feature_names=("feature",), splits=("train", "validation"), sample_ids=("a", "b"),
    )
    second = ReducerCostDataset(
        features=first.features, teacher_costs=first.teacher_costs, feasible=first.feasible,
        candidate_names=("girard", "pca"), feature_names=first.feature_names,
        splits=first.splits, sample_ids=("c", "d"),
    )
    manifest = {"cost_contract": {"schema": "same"}}
    with pytest.raises(ValueError, match="candidate catalog"):
        _validate_named_datasets(
            (NamedPath("clean", SimpleNamespace()), NamedPath("dart", SimpleNamespace())),
            ((first, pd.DataFrame(), manifest), (second, pd.DataFrame(), manifest)),
        )


def _collected_sample(sample_id, split, condition, seed, budget):
    return CollectedReducerCostSample(
        sample_id=sample_id, trace_id=f"{condition}:seed-{seed}", split=split,
        condition=condition, seed=seed, budget=budget, step=0,
        features=np.arange(len(RTL_RANKING_FEATURE_NAMES), dtype=np.float32),
        candidate_names=("girard", "scott"), teacher_costs=(0.0, 1.0),
        feasible=(True, True), teacher_action="girard", teacher_sequence=("girard", "none"),
        collection_mode="teacher", executed_action="girard", disturbed=False,
        disturbance_probability=0.0, infeasible_probability_redirected=0.0,
        sampled_normalized_regret=float("nan"), dart_calibration_sha256=None,
        evaluated_leaves=4, teacher_reducer_failure_count=0,
        teacher_infeasible_candidate_count=0, execution_fallback_used=False,
    )
