from dataclasses import replace

import numpy as np
import pytest

from pzr.rtlola.robot_arm_random import (
    RANDOM_WAYPOINT_CONDITIONS,
    RandomWaypointConfig,
    _make_waypoint_path,
    _nearest_neighbor_sort,
    generate_random_waypoint_trace,
    load_random_waypoint_trace,
    write_random_waypoint_trace,
)


def test_random_waypoint_path_is_closed_and_arc_length_parameterized():
    waypoints = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    path, perimeter = _make_waypoint_path(waypoints)

    assert perimeter == pytest.approx(2.0 + np.sqrt(2.0))
    np.testing.assert_allclose(path(0.5), [0.5, 0.0, 0.0])
    np.testing.assert_allclose(path(perimeter), path(0.0))


def test_nearest_neighbor_ordering_is_deterministic_on_ties():
    waypoints = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    ordered = _nearest_neighbor_sort(waypoints)
    np.testing.assert_array_equal(ordered, waypoints)


def test_random_waypoint_conditions_have_expected_fault_flags():
    for condition in RANDOM_WAYPOINT_CONDITIONS:
        config = RandomWaypointConfig(seed=1, condition=condition, event_count=2)
        assert config.has_drift == ("drift" in condition)
        assert config.has_geofence_fault == ("geofence" in condition)


def test_random_waypoint_config_rejects_negative_drift():
    with pytest.raises(ValueError, match="drift must be non-negative"):
        RandomWaypointConfig(
            seed=1,
            condition="random_waypoint_drift",
            event_count=2,
            drift_z=-0.01,
        )


def test_random_waypoint_generation_is_deterministic_and_persistable(tmp_path):
    pytest.importorskip("mujoco")
    config = RandomWaypointConfig(
        seed=23,
        condition="random_waypoint",
        event_count=3,
        n_waypoints=4,
        n_candidates=40,
        max_retries=20,
        max_tracking_error=0.05,
        sv_threshold=100.0,
    )
    first = generate_random_waypoint_trace(config)
    second = generate_random_waypoint_trace(replace(config))

    assert first.metadata.trace_sha256 == second.metadata.trace_sha256
    assert len(first.events) == 3
    assert all(len(event.values) == 13 for event in first.events)
    assert all(
        left.time < right.time
        for left, right in zip(first.events, first.events[1:])
    )
    assert first.metadata.max_tracking_error <= config.max_tracking_error
    assert first.rows[0].expected_center[0] is not None
    assert first.rows[1].expected_center == (None, None, None)

    write_random_waypoint_trace(first, tmp_path)
    assert (tmp_path / "trace.csv").stat().st_size > 0
    assert (tmp_path / "metadata.json").stat().st_size > 0
    loaded = load_random_waypoint_trace(tmp_path)
    assert loaded.metadata == first.metadata
    assert loaded.rows == first.rows
    assert loaded.events == first.events
