import json
import tarfile

import pandas as pd
import pytest

from pzr.experiments.paper_figures import main as paper_figures_main
from pzr.experiments.suite import main as suite_main
from pzr.learning.features import DECISION_FEATURE_NAMES, DECISION_FEATURE_SCHEMA_VERSION

torch = pytest.importorskip("torch")


def test_experiment_suite_smoke_writes_package_ready_artifacts(tmp_path) -> None:
    out = tmp_path / "suite"

    exit_code = suite_main(
        [
            "--profile",
            "smoke",
            "--out",
            str(out),
            "--formats",
            "png,pdf",
        ]
    )

    assert exit_code == 0
    for scenario in ("robot", "robot_simple", "thermostat"):
        assert (out / "runs" / scenario / "baseline" / "raw_runs.csv").exists()
        assert (out / "runs" / scenario / "learned" / "raw_runs.csv").exists()

    checkpoint = out / "learning" / "learned_distilled.pt"
    assert checkpoint.exists()
    assert checkpoint.with_suffix(".metrics.json").exists()
    assert (out / "learning" / "training_decision_features.csv").exists()

    learned_raw = pd.concat(
        [
            pd.read_csv(out / "runs" / scenario / "learned" / "raw_runs.csv")
            for scenario in ("robot", "robot_simple", "thermostat")
        ],
        ignore_index=True,
    )
    assert "learned_distilled" in set(learned_raw["method"])

    aggregate_raw = pd.read_csv(out / "aggregate" / "raw_runs.csv")
    assert {"robot", "robot_simple", "thermostat"} <= set(aggregate_raw["scenario"])
    assert "learned_distilled" in set(aggregate_raw["method"])
    budgeted = aggregate_raw[aggregate_raw["method"] != "reference"]
    assert (budgeted["budget_violation_count"] == 0).all()
    assert (budgeted["unsound_certificate_count"] == 0).all()
    assert (budgeted["reduction_failure_count"] == 0).all()

    aggregate_summary = pd.read_csv(out / "aggregate" / "summary.csv")
    assert not aggregate_summary.empty
    aggregate_decisions = pd.read_csv(out / "aggregate" / "decision_features.csv")
    assert not aggregate_decisions.empty
    aggregate_selection = pd.read_csv(out / "aggregate" / "selection_summary.csv")
    assert not aggregate_selection.empty
    aggregate_sequences = pd.read_csv(out / "aggregate" / "predicted_sequence_summary.csv")
    assert not aggregate_sequences.empty
    notes = json.loads((out / "aggregate" / "analysis_notes.json").read_text(encoding="utf-8"))
    assert notes["soundness_checks"]["no_op_count"] == 0
    aggregate_counts = aggregate_raw.groupby(["scenario", "predictor_mode", "method"]).size()
    assert (aggregate_counts <= 1).all()

    for name in (
        "fig3a_simple_error_over_time",
        "fig3b_simple_error_by_budget",
        "fig4a_omni_position_x_trace",
        "fig4b_omni_false_alarm_rates",
        "fig5a_omni_error_over_time",
        "fig5b_omni_error_by_budget",
        "selection_robot_simple",
        "selection_robot",
        "selection_thermostat",
        "fallback_box_usage",
    ):
        assert (out / "figures" / "figures" / f"{name}.png").stat().st_size > 0
        assert (out / "figures" / "figures" / f"{name}.pdf").stat().st_size > 0

    figure_summary = pd.read_csv(out / "figures" / "data" / "figure_summaries.csv")
    assert "learned_distilled" in set(figure_summary["method"])

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["profile"] == "smoke"
    assert manifest["scenarios"] == ["robot", "robot_simple", "thermostat"]
    assert {step["kind"] for step in manifest["steps"]} >= {
        "baseline_benchmark",
        "learned_policy_distillation",
        "learned_evaluation",
        "paper_figures",
        "aggregate_outputs",
    }

    index = pd.read_csv(out / "artifact_index.csv")
    assert not index.empty
    for relative in index["path"]:
        assert (out / relative).exists()
    assert {"csv", "json", "figure_png", "figure_pdf", "checkpoint"} <= set(index["kind"])

    archive = out.with_suffix(".tar.gz")
    assert archive.exists()
    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
    assert f"{out.name}/manifest.json" in names
    assert f"{out.name}/artifact_index.csv" in names


def test_paper_figures_can_include_learned_policy(tmp_path) -> None:
    checkpoint = tmp_path / "learned.pt"
    _write_box_checkpoint(checkpoint)

    exit_code = paper_figures_main(
        [
            "--out",
            str(tmp_path / "figures"),
            "--method-set",
            "paper_plus_mpc_ablation",
            "--learned-policy",
            str(checkpoint),
            "--seeds",
            "1",
            "--length",
            "8",
            "--budget",
            "6",
            "--budgets",
            "6,8",
            "--fig4-length",
            "4",
            "--fig4-seed",
            "0",
            "--fpr-length",
            "8",
            "--formats",
            "png",
            "--bootstrap-samples",
            "10",
        ]
    )

    assert exit_code == 0
    summary = pd.read_csv(tmp_path / "figures" / "data" / "figure_summaries.csv")
    assert "learned_distilled" in set(summary["method"])
    assert (tmp_path / "figures" / "figures" / "fig3a_simple_error_over_time.png").exists()


def _write_box_checkpoint(path) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(len(DECISION_FEATURE_NAMES), 1))
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()
    checkpoint = {
        "schema_version": DECISION_FEATURE_SCHEMA_VERSION,
        "model_state": model.state_dict(),
        "feature_names": list(DECISION_FEATURE_NAMES),
        "class_names": ["box"],
        "candidate_reducer_names": ["box"],
        "normalizer_mean": [0.0] * len(DECISION_FEATURE_NAMES),
        "normalizer_std": [1.0] * len(DECISION_FEATURE_NAMES),
        "hidden_sizes": [],
        "training_config": {},
    }
    torch.save(checkpoint, path)
