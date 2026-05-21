import json

import numpy as np
import pandas as pd
import pytest

from pzr.benchmarks.robot import OmnidirectionalRobotMonitor, generate_robot_trace
from pzr.experiments.cli import main as benchmark_main
from pzr.learning.distill_cli import main as distill_main
from pzr.learning.dagger import aggregate_dagger_rows, class_balanced_indices, load_dagger_iterations
from pzr.learning.features import (
    DECISION_FEATURE_NAMES,
    DECISION_FEATURE_SCHEMA_VERSION,
    decision_feature_values,
)
from pzr.learning.policy import LearnedReductionPolicy
from pzr.reduction.reducers import BoxReducer, ProtectedReducer

torch = pytest.importorskip("torch")


class FailingReducer:
    name = "girard"

    def reduce(self, zonotope, budget, context=None):
        _ = zonotope, budget, context
        raise ValueError("intentional failure")


def test_decision_feature_extraction_is_finite_and_covers_metadata() -> None:
    monitor = OmnidirectionalRobotMonitor()
    state = monitor.initial_state()
    for measurement in generate_robot_trace(8, seed=1):
        state = monitor.step(state, measurement).state

    features = decision_feature_values(monitor, state, budget=6, horizon=2)

    assert tuple(features) == DECISION_FEATURE_NAMES
    assert all(np.isfinite(value) for value in features.values())
    assert features["generator_count"] == state.zonotope.generator_count
    assert features["metadata_calibration_count"] >= 1
    assert features["trigger_count"] == len(monitor.triggers)
    assert features["required_generator_rule_count"] >= 1


def test_benchmark_writes_non_empty_decision_features(tmp_path) -> None:
    exit_code = benchmark_main(
        [
            "robot",
            "--length",
            "8",
            "--budget",
            "6",
            "--horizon",
            "2",
            "--seeds",
            "1",
            "--bootstrap-samples",
            "10",
            "--out",
            str(tmp_path),
            "--quiet",
        ]
    )

    assert exit_code == 0
    decisions = pd.read_csv(tmp_path / "decision_features.csv")
    assert not decisions.empty
    assert set(decisions["feature_schema_version"]) == {DECISION_FEATURE_SCHEMA_VERSION}
    assert {"chosen_reducer_label", "predicted_sequence", *DECISION_FEATURE_NAMES} <= set(
        decisions.columns
    )
    assert (decisions["method"] == "mpc_focused_sequence").any()
    assert "no_reduction" not in set(decisions["chosen_reducer_label"])
    assert not (decisions["no_op_selected"] == True).any()
    assert (decisions["generator_count"] > decisions["budget"]).all()

    selection = pd.read_csv(tmp_path / "selection_summary.csv")
    assert not selection.empty
    sequences = pd.read_csv(tmp_path / "predicted_sequence_summary.csv")
    wide_sequences = sequences[sequences["method"] == "mpc_wide_fixed_girard"]
    assert not wide_sequences.empty
    assert (wide_sequences["first_action_box_count"] == 0).all()


def test_distill_cli_trains_tiny_policy_from_decision_rows(tmp_path) -> None:
    data = tmp_path / "decisions.csv"
    _write_synthetic_decisions(data)
    checkpoint = tmp_path / "distilled.pt"

    exit_code = distill_main(
        [
            "train",
            "--data",
            str(data),
            "--out",
            str(checkpoint),
            "--epochs",
            "5",
            "--batch-size",
            "4",
        ]
    )

    assert exit_code == 0
    assert checkpoint.exists()
    metrics = json.loads(checkpoint.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert metrics["row_count"] == 6
    assert set(metrics["class_counts"]) == {"box", "girard"}
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["schema_version"] == DECISION_FEATURE_SCHEMA_VERSION
    assert saved["feature_names"] == list(DECISION_FEATURE_NAMES)


def test_dagger_aggregates_iterations_and_trains_balanced_checkpoint(tmp_path) -> None:
    first = tmp_path / "iter0.csv"
    second = tmp_path / "iter1.csv"
    _write_synthetic_decisions(first)
    _write_synthetic_decisions(second)
    iterations = load_dagger_iterations((first, second))
    aggregate = aggregate_dagger_rows(iterations)

    assert set(aggregate["dagger_iteration"]) == {0, 1}
    sampled = class_balanced_indices(["box", "girard", "girard"], seed=0)
    assert sampled.shape == (4,)

    checkpoint = tmp_path / "dagger.pt"
    exit_code = distill_main(
        [
            "dagger",
            "--data",
            str(first),
            str(second),
            "--out",
            str(checkpoint),
            "--epochs",
            "5",
            "--batch-size",
            "4",
        ]
    )

    assert exit_code == 0
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["training_config"]["training_mode"] == "dagger"
    assert saved["training_config"]["class_balanced"] is True
    assert saved["dagger"]["iteration_count"] == 2
    assert checkpoint.with_suffix(".dagger_dataset.csv").exists()


def test_learned_policy_returns_certified_budgeted_state_and_falls_back(tmp_path) -> None:
    checkpoint = tmp_path / "prefer_girard.pt"
    _write_prefer_girard_checkpoint(checkpoint)
    monitor = OmnidirectionalRobotMonitor()
    state = monitor.initial_state()
    for measurement in generate_robot_trace(8, seed=2):
        state = monitor.step(state, measurement).state

    policy = LearnedReductionPolicy(
        checkpoint,
        reducers=(FailingReducer(), ProtectedReducer(BoxReducer())),
        fallback_reducer=ProtectedReducer(BoxReducer()),
        budget=6,
        horizon=2,
    )
    decision = policy.reduce_state(monitor, state)

    assert decision.reducer_name == "box"
    assert decision.result.certificate.is_sound
    assert decision.state.zonotope.generator_count <= 6
    assert decision.pruned_sequences >= 1
    assert decision.predicted_sequence[0] == "girard"


def test_benchmark_cli_can_evaluate_learned_policy(tmp_path) -> None:
    data = tmp_path / "decisions.csv"
    _write_synthetic_decisions(data)
    checkpoint = tmp_path / "distilled.pt"
    distill_main(
        [
            "train",
            "--data",
            str(data),
            "--out",
            str(checkpoint),
            "--epochs",
            "5",
        ]
    )

    exit_code = benchmark_main(
        [
            "robot",
            "--method-set",
            "paper_plus_mpc_ablation",
            "--learned-policy",
            str(checkpoint),
            "--length",
            "8",
            "--budget",
            "6",
            "--horizon",
            "2",
            "--seeds",
            "1",
            "--bootstrap-samples",
            "10",
            "--out",
            str(tmp_path / "eval"),
            "--quiet",
        ]
    )

    assert exit_code == 0
    raw = pd.read_csv(tmp_path / "eval" / "raw_runs.csv")
    learned = raw[raw["method"] == "learned_distilled"]
    assert not learned.empty
    assert (learned["budget_violation_count"] == 0).all()
    chosen_total = (
        learned["chosen_box_count"]
        + learned["chosen_girard_count"]
        + learned["chosen_girard_slack1_count"]
        + learned["chosen_combastel_count"]
        + learned["chosen_methA_count"]
        + learned["chosen_scott_count"]
        + learned["chosen_pca_count"]
        + learned["chosen_adaptive_count"]
        + learned["chosen_keep_trigger_count"]
        + learned["chosen_keep_norm_count"]
        + learned["chosen_keep_calibration_aware_count"]
        + learned["chosen_other_count"]
    )
    assert (chosen_total == learned["reduction_count"]).all()


def _write_synthetic_decisions(path) -> None:
    rows = []
    for seed in range(2):
        for index, label in enumerate(("girard", "box", "girard")):
            features = {name: float(index + seed) for name in DECISION_FEATURE_NAMES}
            features["budget"] = 6.0
            features["horizon"] = 2.0
            features["generator_count"] = 7.0 + index
            rows.append(
                {
                    **features,
                    "feature_schema_version": DECISION_FEATURE_SCHEMA_VERSION,
                    "scenario": "synthetic",
                    "method": "mpc_focused_sequence",
                    "method_kind": "mpc_sequence",
                    "seed": seed,
                    "length": 3,
                    "budget": 6,
                    "horizon": 2,
                    "predictor_mode": "online",
                    "step": index + 1,
                    "chosen_reducer_label": label,
                    "predicted_cost": 0.0,
                    "predicted_sequence": json.dumps([label]),
                    "evaluated_sequence_count": 1,
                    "pruned_sequence_count": 0,
                    "candidate_reducer_names": json.dumps(["box", "girard"]),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_prefer_girard_checkpoint(path) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(len(DECISION_FEATURE_NAMES), 2))
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias[:] = torch.tensor([10.0, -10.0])
    checkpoint = {
        "schema_version": DECISION_FEATURE_SCHEMA_VERSION,
        "model_state": model.state_dict(),
        "feature_names": list(DECISION_FEATURE_NAMES),
        "class_names": ["girard", "box"],
        "candidate_reducer_names": ["girard", "box"],
        "normalizer_mean": [0.0] * len(DECISION_FEATURE_NAMES),
        "normalizer_std": [1.0] * len(DECISION_FEATURE_NAMES),
        "hidden_sizes": [],
        "training_config": {},
    }
    torch.save(checkpoint, path)
