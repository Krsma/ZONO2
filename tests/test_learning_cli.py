from argparse import Namespace

import numpy as np
import pandas as pd

from pzr.learning.artifacts import write_ranking_dataset
from pzr.learning.cli import _split_seeds, build_parser, run_train
from pzr.learning.dataset import RankingDataset
from pzr.learning.ranker import RankingPolicy
from pzr.rtlola.features import RTL_RANKING_FEATURE_NAMES


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
    features = np.arange(48, dtype=np.float32).reshape(4, 12)
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
