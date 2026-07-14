import json

import pytest

from pzr.rtlola.learning_traces import (
    RANDOM_WAYPOINT_TRACE_STORE_SCHEMA,
    RandomWaypointTraceStoreConfig,
    generate_random_waypoint_trace_store,
    load_random_waypoint_trace_store,
)


def _config(tmp_path, *, event_count=3, seed_count=1):
    return RandomWaypointTraceStoreConfig(
        output=tmp_path,
        event_count=event_count,
        conditions=("random_waypoint",),
        seed_start=0,
        seed_count=seed_count,
    )


def test_random_waypoint_trace_store_is_non_empty_resumable_and_aligned(
    tmp_path, monkeypatch,
):
    pytest.importorskip("mujoco")
    first = generate_random_waypoint_trace_store(_config(tmp_path, seed_count=2))
    manifest = json.loads((tmp_path / "manifest.json").read_text())

    assert manifest["schema"] == RANDOM_WAYPOINT_TRACE_STORE_SCHEMA
    assert manifest["seeds"] == [0, 1]
    assert len(first.traces) == 2
    assert all(item.trace.events for item in first.traces)
    assert first.traces_for_seed(0)[0].trace_id == "random_waypoint:seed-0"
    with pytest.raises(ValueError, match="every condition"):
        first.traces_for_seed(2)

    def unexpected_generation(_config):
        raise AssertionError("validated trace should have been reused")

    monkeypatch.setattr(
        "pzr.rtlola.learning_traces.generate_random_waypoint_trace",
        unexpected_generation,
    )
    resumed = generate_random_waypoint_trace_store(_config(tmp_path, seed_count=2))
    assert resumed.manifest_sha256 == first.manifest_sha256


def test_random_waypoint_trace_store_rejects_tampering(tmp_path):
    pytest.importorskip("mujoco")
    generate_random_waypoint_trace_store(_config(tmp_path))
    trace_path = tmp_path / "random_waypoint:seed-0" / "trace.csv"
    trace_path.write_text(trace_path.read_text().replace("time,", "changed,", 1))

    with pytest.raises(ValueError, match="CSV schema"):
        load_random_waypoint_trace_store(tmp_path)


def test_random_waypoint_trace_store_rejects_incomplete_and_incompatible_artifacts(
    tmp_path,
):
    pytest.importorskip("mujoco")
    generate_random_waypoint_trace_store(_config(tmp_path))
    metadata_path = tmp_path / "random_waypoint:seed-0" / "metadata.json"
    metadata_path.unlink()
    (tmp_path / "manifest.json").unlink()

    with pytest.raises(ValueError, match="incomplete"):
        generate_random_waypoint_trace_store(_config(tmp_path))

    other = tmp_path / "other"
    generate_random_waypoint_trace_store(_config(other))
    with pytest.raises(ValueError, match="identity differs"):
        generate_random_waypoint_trace_store(_config(other, event_count=4))
