import json
import tarfile
import builtins
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pzr.experiments.corl_suite import main as corl_main
from pzr.robotics.safe_control_gym import SidecarSafeControlGymClient, preflight_safe_control_gym


def test_corl_preflight_smoke_uses_fake_environment() -> None:
    result = preflight_safe_control_gym(
        profile="smoke",
        safe_control_gym_root=None,
        safe_control_python=None,
    )

    assert result.ok
    assert result.checks["fake_env_reset"]
    assert result.checks["torch"]


def test_corl_preflight_smoke_does_not_require_torch(monkeypatch) -> None:
    real_import = builtins.__import__

    def import_without_torch(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise ImportError("torch intentionally hidden")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_torch)

    result = preflight_safe_control_gym(
        profile="smoke",
        safe_control_gym_root=None,
        safe_control_python=None,
    )

    assert result.ok
    assert not result.checks["torch"]


def test_corl_preflight_fails_without_safe_control_gym_for_overnight() -> None:
    result = preflight_safe_control_gym(
        profile="overnight",
        safe_control_gym_root=None,
        safe_control_python=None,
    )

    assert not result.ok
    assert not result.checks["root_exists"]
    assert result.messages


def test_corl_smoke_suite_writes_headline_artifacts(tmp_path) -> None:
    out = tmp_path / "corl"

    exit_code = corl_main(
        [
            "--profile",
            "smoke",
            "--out",
            str(out),
            "--force",
            "--learned-mode",
            "none",
            "--method-set",
            "core",
        ]
    )

    assert exit_code == 0
    expected = (
        "raw_episodes.csv",
        "intervention_timeseries.csv",
        "monitor_timeseries.csv",
        "decision_features.csv",
        "selection_summary.csv",
        "predicted_sequence_summary.csv",
        "headline_table.csv",
        "headline_table.md",
        "headline_quality.md",
        "analysis_notes.json",
        "failure_events.csv",
        "progress.jsonl",
        "manifest.json",
        "artifact_index.csv",
    )
    for name in expected:
        assert (out / name).stat().st_size > 0

    raw = pd.read_csv(out / "raw_episodes.csv")
    assert {"reference_unbounded", "nominal_no_monitor", "girard"} <= set(raw["method"])
    assert "learned_dagger" not in set(raw["method"])
    assert set(raw["seed"]) == {1}
    assert (raw[raw["method"] != "reference_unbounded"]["budget_violation_count"] == 0).all()

    headline = pd.read_csv(out / "headline_table.csv")
    assert {
        "task_completion_rate",
        "spurious_intervention_rate",
        "missed_violation_rate",
        "mean_reducer_latency_ms",
        "budget_violation_count",
        "unsound_certificate_count",
    } <= set(headline.columns)
    assert "learned_dagger" not in set(headline["method"])

    decisions = pd.read_csv(out / "decision_features.csv")
    assert set(decisions.columns)
    assert decisions["candidate_reducer_names"].map(json.loads).map(bool).all()
    interventions = pd.read_csv(out / "intervention_timeseries.csv")
    assert {
        "monitor_trigger_names",
        "oracle_trigger_names",
        "pose_z",
        "stream_safety_margin",
        "controller_mode",
        "episode_len_sec",
        "simulator_time",
    } <= set(interventions.columns)

    notes = json.loads((out / "analysis_notes.json").read_text(encoding="utf-8"))
    assert notes["soundness_checks"]["budget_violation_count"] == 0
    assert "paper_usable" in notes
    assert "learning_label_quality" in notes
    assert isinstance(notes["warning_flags"], list)

    label_summary = pd.read_csv(out / "learning" / "dagger_label_summary.csv")
    assert label_summary.empty
    failures = pd.read_csv(out / "failure_events.csv")
    assert failures.empty
    progress = (out / "progress.jsonl").read_text(encoding="utf-8")
    assert "evaluation_seed_complete" in progress
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["learned_mode"] == "none"
    assert manifest["dagger_expert"] == "mpc_wide_fixed_girard"

    archive = out.with_suffix(".tar.gz")
    assert archive.exists()
    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
    assert f"{out.name}/headline_table.csv" in names


def test_corl_smoke_suite_can_run_dagger_and_include_learned(tmp_path) -> None:
    pytest.importorskip("torch")
    out = tmp_path / "corl-dagger"

    exit_code = corl_main(
        [
            "--profile",
            "smoke",
            "--out",
            str(out),
            "--force",
            "--learned-mode",
            "dagger",
            "--include-failed-learned",
        ]
    )

    assert exit_code == 0
    assert (out / "dagger_dataset.csv").stat().st_size > 0
    raw = pd.read_csv(out / "raw_episodes.csv")
    assert "learned_dagger" in set(raw["method"])
    label_summary = pd.read_csv(out / "learning" / "dagger_label_summary.csv")
    assert not label_summary.empty


def test_corl_fail_on_unusable_raises_after_writing_artifacts(tmp_path) -> None:
    out = tmp_path / "corl-unusable"

    with pytest.raises(RuntimeError, match="not usable as headline evidence"):
        corl_main(
            [
                "--profile",
                "smoke",
                "--out",
                str(out),
                "--force",
                "--learned-mode",
                "none",
                "--method-set",
                "core",
                "--no-archive",
                "--fail-on-unusable",
            ]
        )

    assert (out / "headline_quality.md").stat().st_size > 0
    notes = json.loads((out / "analysis_notes.json").read_text(encoding="utf-8"))
    assert not notes["paper_usable"]
    assert "paper_usable=false" in notes["warning_flags"]


def test_controller_validation_smoke_writes_nominal_artifacts(tmp_path) -> None:
    out = tmp_path / "controller-validation"

    exit_code = corl_main(
        [
            "--profile",
            "smoke",
            "--controller-validation",
            "--eval-seeds",
            "2",
            "--out",
            str(out),
            "--force",
            "--no-archive",
        ]
    )

    assert exit_code == 0
    expected = (
        "raw_episodes.csv",
        "intervention_timeseries.csv",
        "monitor_timeseries.csv",
        "controller_validation_summary.csv",
        "analysis_notes.json",
        "failure_events.csv",
        "manifest.json",
        "artifact_index.csv",
    )
    for name in expected:
        assert (out / name).stat().st_size > 0

    raw = pd.read_csv(out / "raw_episodes.csv")
    assert set(raw["method"]) == {"nominal_no_monitor"}
    assert set(raw["seed"]) == {0, 1}

    summary = pd.read_csv(out / "controller_validation_summary.csv")
    assert int(summary.iloc[0]["episode_count"]) == 2
    assert "pass_gate" in set(summary.columns)

    notes = json.loads((out / "analysis_notes.json").read_text(encoding="utf-8"))
    assert notes["success_gate"].startswith("at least 8/10")


def test_corl_calibration_smoke_writes_recommendations(tmp_path) -> None:
    out = tmp_path / "calibration"

    exit_code = corl_main(
        [
            "--profile",
            "smoke",
            "--calibration",
            "--calibration-seeds",
            "1",
            "--calibration-max-steps",
            "12",
            "--out",
            str(out),
            "--force",
            "--no-archive",
        ]
    )

    assert exit_code == 0
    for name in (
        "calibration_runs.csv",
        "calibration_summary.csv",
        "calibration_recommendations.json",
        "failure_events.csv",
        "analysis_notes.json",
        "artifact_index.csv",
    ):
        assert (out / name).stat().st_size > 0
    summary = pd.read_csv(out / "calibration_summary.csv")
    assert {"config_id", "paper_candidate", "rejection_reasons"} <= set(summary.columns)
    recommendations = json.loads((out / "calibration_recommendations.json").read_text(encoding="utf-8"))
    assert "recommended_config_id" in recommendations


def test_sidecar_payload_infers_completion_and_yaml_bounds(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "pybullet",
        types.SimpleNamespace(getBasePositionAndOrientation=lambda *args, **kwargs: ((0.0, 0.0, 0.0), None)),
    )
    from pzr.robotics.safe_control_worker import _scenario_payload, _snapshot_payload

    class Env:
        NUM_GATES = 4
        GATES = []
        EFFECTIVE_GATES_POSITIONS = []
        OBSTACLES_IDS = []
        OBSTACLES = []
        COLLISION_R = 0.0

    config = {
        "quadrotor_config": {
            "constraints": [
                {
                    "constraint_form": "bounded_constraint",
                    "constrained_variable": "state",
                    "active_dims": [0, 2, 4],
                    "lower_bounds": [-3, -3, -0.1],
                    "upper_bounds": [3, 3, 2],
                }
            ]
        }
    }
    info = {"current_target_gate_id": -1}
    obs = np.asarray([0.0, 0.0, 0.0, 0.0, 2.2, 0.0], dtype=float)

    snapshot = _snapshot_payload(Env(), obs, info, done=True, time=1.5, config=config)
    scenario = _scenario_payload(Env(), {}, config)

    assert snapshot["task_completed"]
    assert snapshot["gates_passed"] == 4
    assert snapshot["constraint_violation"]
    assert scenario["altitude_min"] == -0.1
    assert scenario["altitude_max"] == 2.0
    assert scenario["corridor_radius"] == 3.0


def test_safe_control_sidecar_nominal_climbs_when_local_checkout_exists() -> None:
    root = Path("external/safe-control-gym")
    python = Path("external/miniconda3/envs/pzr-safe-control-fw/bin/python")
    if not root.exists() or not python.exists():
        pytest.skip("local firmware safe-control-gym sidecar environment is not installed")
    client = SidecarSafeControlGymClient(python, root, "competition/level0.yaml", controller_mode="firmware")
    try:
        try:
            status = client.status()
        except RuntimeError as exc:
            pytest.skip(f"local sidecar environment is incomplete: {exc}")
        if not status.get("pycffirmware_available", False):
            pytest.skip("local sidecar environment does not provide pycffirmware")
        snapshot = client.reset(20)
        for _ in range(90):
            snapshot = client.step(client.nominal_command(snapshot))
            if snapshot.done:
                break
        assert snapshot.pose[2] > 0.5
        assert not snapshot.collision
        assert not snapshot.constraint_violation
    finally:
        client.close()
