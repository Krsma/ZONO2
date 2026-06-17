"""Tests for the robotics candidate probe path."""

from __future__ import annotations

import pandas as pd

from pzr.experiments import robotics_probe


def test_interesting_synthetic_trace_is_promotable():
    bundle = robotics_probe.make_synthetic_probe_bundle(
        "drone", length=60, seed=0,
    )
    result = robotics_probe.run_bundle(bundle, budget=10, seed=0)

    report = result["report"]
    assert report["recommendation"] == "promote"
    assert report["relative_width_spread"] > 0.05
    assert report["differentiated_methods"] >= 2
    assert report["budget_violations"] == 0
    assert report["unsound_certificates"] == 0


def test_degenerate_trace_is_rejected():
    bundle = robotics_probe.make_synthetic_probe_bundle(
        "degenerate", length=40, seed=1,
    )
    result = robotics_probe.run_bundle(bundle, budget=10, seed=1)

    report = result["report"]
    assert report["recommendation"] == "reject"
    assert report["differentiated_methods"] == 0
    assert report["near_threshold_fraction"] == 0.0


def test_probe_outputs_are_written_for_available_candidate(tmp_path, monkeypatch):
    def fake_candidate_bundle(name: str, **kwargs):
        length = kwargs["length"]
        seed = kwargs["seed"]
        bundle = robotics_probe.make_synthetic_probe_bundle(
            "drone", length=length, seed=seed,
        )
        return bundle, bundle.metadata

    monkeypatch.setattr(robotics_probe, "_candidate_bundle", fake_candidate_bundle)

    result = robotics_probe.run_probe(
        candidates=("drone",),
        length=30,
        seed=2,
        budget=10,
        output=tmp_path,
    )

    assert (tmp_path / "probe_metadata.json").stat().st_size > 0
    assert (tmp_path / "method_scores.csv").stat().st_size > 0
    assert (tmp_path / "candidate_scores.csv").stat().st_size > 0
    assert (tmp_path / "method_score_summary.csv").stat().st_size > 0
    assert (tmp_path / "trace_summary.csv").stat().st_size > 0
    assert (tmp_path / "candidate_report.md").stat().st_size > 0
    assert (tmp_path / "drone_timeseries.csv").stat().st_size > 0
    assert (tmp_path / "drone_derived_streams.csv").stat().st_size > 0
    assert not result["method_scores"].empty
    assert set(result["method_scores"]["candidate"]) == {"drone"}


def test_probe_multi_seed_writes_seed_subdirs_and_aggregate_scores(tmp_path, monkeypatch):
    def fake_candidate_bundle(name: str, **kwargs):
        length = kwargs["length"]
        seed = kwargs["seed"]
        bundle = robotics_probe.make_synthetic_probe_bundle(
            "drone", length=length, seed=seed,
        )
        return bundle, bundle.metadata

    monkeypatch.setattr(robotics_probe, "_candidate_bundle", fake_candidate_bundle)

    result = robotics_probe.run_probe(
        candidates=("drone",),
        length=20,
        seed=3,
        seeds=2,
        budget=10,
        output=tmp_path,
    )

    assert (tmp_path / "seed_3" / "drone_timeseries.csv").stat().st_size > 0
    assert (tmp_path / "seed_4" / "drone_timeseries.csv").stat().st_size > 0
    assert set(result["method_scores"]["seed"]) == {3, 4}
    assert set(pd.read_csv(tmp_path / "candidate_scores.csv")["seed"]) == {3, 4}
    assert not pd.read_csv(tmp_path / "candidate_score_summary.csv").empty


def test_warmup_trims_trace_before_scoring(tmp_path, monkeypatch):
    def fake_candidate_bundle(name: str, **kwargs):
        length = kwargs["length"]
        seed = kwargs["seed"]
        bundle = robotics_probe.make_synthetic_probe_bundle(
            "drone", length=length, seed=seed,
        )
        return bundle, bundle.metadata

    monkeypatch.setattr(robotics_probe, "_candidate_bundle", fake_candidate_bundle)

    result = robotics_probe.run_probe(
        candidates=("drone",),
        length=12,
        seed=0,
        warmup_steps=5,
        budget=10,
        output=tmp_path,
    )

    streams = pd.read_csv(tmp_path / "drone_derived_streams.csv")
    assert len(streams) == 7
    assert result["reports"][0]["length"] == 7


def test_unavailable_f1tenth_records_metadata_without_crashing(tmp_path, monkeypatch):
    def fake_status(sidecar_python=None):
        return {
            "available": False,
            "package": "f110_gym",
            "reason": "missing in test",
        }

    monkeypatch.setattr(robotics_probe, "_f1tenth_status", fake_status)

    result = robotics_probe.run_probe(
        candidates=("f1tenth",),
        length=10,
        seed=0,
        budget=10,
        output=tmp_path,
    )

    metadata = result["metadata"]
    assert metadata["candidates"]["f1tenth"]["status"] == "unavailable"
    assert result["method_scores"].empty
    assert pd.read_csv(tmp_path / "method_scores.csv").empty
    assert (tmp_path / "candidate_report.md").stat().st_size > 0


def test_drone_record_to_safety_streams():
    record = {
        "step": 0,
        "time": 0.0,
        "obs": [-0.9, 0.0, -2.9, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "gates": [
            [0.5, -2.5, 0, 0, 0, -1.57, 0],
            [2.0, -1.5, 0, 0, 0, 0, 1],
        ],
        "obstacles": [[1.5, -2.5, 0, 0, 0, 0]],
        "info": {
            "collision": [None, False],
            "constraint_violation": False,
            "current_target_gate_id": 0,
            "current_target_gate_type": 0,
            "current_target_gate_pos": [0.5, -2.5, 0, 0, 0, -1.57],
        },
        "done": False,
    }

    values = robotics_probe._drone_true_safety_streams(record)

    assert values.shape == (6,)
    assert values[3] == 0.4
    assert values[4] == 1.5
    assert values[5] == 1.75


def test_drone_far_gate_alignment_is_neutral():
    record = {
        "obs": [-0.9, 0.0, -2.9, 0.0, 0.5, 0.0],
        "gates": [[3.0, -2.9, 0, 0, 0, 0.0, 0]],
        "obstacles": [],
        "info": {
            "current_target_gate_id": 0,
            "current_target_gate_type": 0,
            "current_target_gate_pos": [3.0, -2.9, 0, 0, 0, 0.0],
            "current_target_gate_in_range": False,
        },
    }

    values = robotics_probe._drone_true_safety_streams(record)

    assert values[1] == 0.35


def test_f1tenth_observation_to_safety_streams():
    scan = [2.0] * 1080
    obs = {
        "scans": [scan],
        "linear_vels_x": [1.0],
        "ang_vels_z": [0.2],
        "poses_theta": [0.1],
        "collisions": [False],
    }

    values = robotics_probe._f1tenth_true_safety_streams(obs)

    assert values.shape == (6,)
    assert values[0] > 0.0
    assert values[1] > 0.0
    assert values[2] > 0.0


def test_f1tenth_heading_wraps_before_margin():
    obs = {
        "scans": [[2.0] * 1080],
        "linear_vels_x": [0.6],
        "ang_vels_z": [0.1],
        "poses_theta": [4.0 * 3.141592653589793],
        "collisions": [False],
    }

    values = robotics_probe._f1tenth_true_safety_streams(obs)

    assert values[4] > 0.7


def test_live_source_unavailable_does_not_fall_back_for_drone(tmp_path, monkeypatch):
    def fake_live_drone_bundle(**kwargs):
        return None, {
            "candidate": "drone",
            "status": "unavailable",
            "reason": "sim failed in test",
            "trace_source": "safe_control_gym_live_rollout",
        }

    monkeypatch.setattr(
        robotics_probe,
        "_sidecar_status",
        lambda **kwargs: {"available": True},
    )
    monkeypatch.setattr(robotics_probe, "_make_live_drone_probe_bundle", fake_live_drone_bundle)

    result = robotics_probe.run_probe(
        candidates=("drone",),
        length=10,
        seed=0,
        budget=10,
        output=tmp_path,
        trace_source="live",
    )

    assert result["method_scores"].empty
    assert result["metadata"]["candidates"]["drone"]["reason"] == "sim failed in test"
