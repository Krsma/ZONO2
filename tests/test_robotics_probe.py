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
    def fake_candidate_bundle(name: str, *, length: int, seed: int):
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
    assert (tmp_path / "trace_summary.csv").stat().st_size > 0
    assert (tmp_path / "candidate_report.md").stat().st_size > 0
    assert (tmp_path / "drone_timeseries.csv").stat().st_size > 0
    assert not result["method_scores"].empty
    assert set(result["method_scores"]["candidate"]) == {"drone"}


def test_unavailable_f1tenth_records_metadata_without_crashing(tmp_path, monkeypatch):
    def fake_status():
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
