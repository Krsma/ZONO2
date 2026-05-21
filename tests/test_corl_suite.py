import json
import tarfile

import pandas as pd
import pytest

from pzr.experiments.corl_suite import main as corl_main
from pzr.robotics.safe_control_gym import preflight_safe_control_gym

torch = pytest.importorskip("torch")


def test_corl_preflight_smoke_uses_fake_environment() -> None:
    result = preflight_safe_control_gym(
        profile="smoke",
        safe_control_gym_root=None,
        safe_control_python=None,
    )

    assert result.ok
    assert result.checks["fake_env_reset"]
    assert result.checks["torch"]


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
        ]
    )

    assert exit_code == 0
    expected = (
        "raw_episodes.csv",
        "intervention_timeseries.csv",
        "monitor_timeseries.csv",
        "decision_features.csv",
        "dagger_dataset.csv",
        "selection_summary.csv",
        "predicted_sequence_summary.csv",
        "headline_table.csv",
        "headline_table.md",
        "analysis_notes.json",
        "manifest.json",
        "artifact_index.csv",
    )
    for name in expected:
        assert (out / name).stat().st_size > 0

    raw = pd.read_csv(out / "raw_episodes.csv")
    assert {"reference_unbounded", "nominal_no_monitor", "learned_dagger"} <= set(raw["method"])
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
    assert "learned_dagger" in set(headline["method"])

    decisions = pd.read_csv(out / "decision_features.csv")
    assert not decisions.empty
    assert (decisions["method"] == "mpc_focused_sequence").any()

    notes = json.loads((out / "analysis_notes.json").read_text(encoding="utf-8"))
    assert notes["soundness_checks"]["budget_violation_count"] == 0
    assert notes["warning_flags"] == []

    archive = out.with_suffix(".tar.gz")
    assert archive.exists()
    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
    assert f"{out.name}/headline_table.csv" in names
