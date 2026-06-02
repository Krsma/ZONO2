"""Tests for robot-arm visualization helpers."""

import json

import numpy as np
import pytest

from pzr.envs.robot_arm import LINK_LENGTHS, NUM_JOINTS
from pzr.experiments.robot_arm_animation import (
    _physical_view_limits,
    _trace_quality,
    _trigger_view_limits,
    build_parser,
    interval_rectangle,
    joint_positions,
    render_robot_arm_animation,
    replay_robot_arm_visualization,
    zonotope_vertices_2d,
)
from pzr.zonotope.core import Zonotope


def _has_mujoco():
    try:
        import mujoco
        return True
    except ImportError:
        return False


def test_joint_positions_zero_angles():
    positions = joint_positions(np.zeros(NUM_JOINTS))
    expected = np.array([
        [0.0, 0.0],
        [LINK_LENGTHS[0], 0.0],
        [LINK_LENGTHS[0] + LINK_LENGTHS[1], 0.0],
        [sum(LINK_LENGTHS), 0.0],
    ])
    np.testing.assert_allclose(positions, expected, atol=1e-12)


def test_zonotope_vertices_2d_matches_box_bounds():
    z = Zonotope(
        center=np.array([1.0, 2.0]),
        generators=np.array([[0.5, 0.0], [0.0, 1.0]]),
    )
    vertices = zonotope_vertices_2d(z)
    np.testing.assert_allclose(vertices[0], vertices[-1])
    np.testing.assert_allclose(np.min(vertices, axis=0), [0.5, 1.0])
    np.testing.assert_allclose(np.max(vertices, axis=0), [1.5, 3.0])


def test_interval_rectangle_is_closed():
    rect = interval_rectangle(np.array([0.0, -1.0]), np.array([2.0, 3.0]))
    assert rect.shape == (5, 2)
    np.testing.assert_allclose(rect[0], rect[-1])
    np.testing.assert_allclose(np.min(rect, axis=0), [0.0, -1.0])
    np.testing.assert_allclose(np.max(rect, axis=0), [2.0, 3.0])


def test_parser_rejects_removed_ghost_options():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--method", "scott", "--ghost-samples", "4"])


def test_parser_requires_method():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


@pytest.mark.skipif(
    not _has_mujoco(), reason="MuJoCo not available",
)
def test_replay_robot_arm_visualization_frames():
    frames = replay_robot_arm_visualization(
        method="scott",
        trace="benchmark",
        seed=0,
        length=8,
        budget=10,
        horizon=2,
    )
    assert len(frames) == 8
    assert max(frame.generator_count for frame in frames) <= 10
    for frame in frames:
        assert frame.trigger_zonotope.dimension == 2
        assert frame.record.true_state.shape == (2 * NUM_JOINTS,)
        assert np.all(np.isfinite(frame.trigger_lower))
        assert np.all(np.isfinite(frame.trigger_upper))


@pytest.mark.skipif(
    not _has_mujoco(), reason="MuJoCo not available",
)
def test_benchmark_and_paper_traces_are_distinct():
    benchmark_frames = replay_robot_arm_visualization(
        method="scott",
        trace="benchmark",
        seed=0,
        length=10,
        budget=10,
        horizon=2,
    )
    paper_frames = replay_robot_arm_visualization(
        method="scott",
        trace="paper",
        seed=0,
        length=10,
        budget=10,
        horizon=2,
    )

    benchmark_ee = np.array([frame.record.ee_pos for frame in benchmark_frames])
    paper_ee = np.array([frame.record.ee_pos for frame in paper_frames])
    assert benchmark_ee.shape == paper_ee.shape
    assert not np.allclose(benchmark_ee, paper_ee)
    assert all(frame.record.episode_id == 0 for frame in paper_frames)


@pytest.mark.skipif(
    not _has_mujoco(), reason="MuJoCo not available",
)
def test_paper_trace_has_visible_motion_without_action_saturation():
    frames = replay_robot_arm_visualization(
        method="scott",
        trace="paper",
        seed=0,
        length=80,
        budget=10,
        horizon=2,
    )
    quality = _trace_quality(frames, trace="paper")
    assert quality.trace_model == "scripted_kinematic_explanatory"
    assert quality.ee_path_length > 0.2
    assert quality.action_saturation_fraction == 0.0
    assert quality.waypoints_reached >= 4


@pytest.mark.skipif(
    not _has_mujoco(), reason="MuJoCo not available",
)
def test_physical_view_is_not_scaled_by_large_trigger_hull():
    frames = replay_robot_arm_visualization(
        method="scott",
        trace="paper",
        seed=0,
        length=120,
        budget=10,
        horizon=2,
    )
    final_width = float(np.sum(frames[-1].trigger_upper - frames[-1].trigger_lower))
    physical_limits = _physical_view_limits(frames)
    trigger_limits = _trigger_view_limits()
    physical_span = physical_limits[1] - physical_limits[0]
    trigger_span = trigger_limits[1] - trigger_limits[0]

    assert final_width > physical_span
    assert physical_span < 1.2
    assert trigger_span < 1.0


@pytest.mark.skipif(
    not _has_mujoco(), reason="MuJoCo not available",
)
def test_render_robot_arm_animation_smoke(tmp_path):
    artifacts = render_robot_arm_animation(
        output=tmp_path,
        method="scott",
        trace="paper",
        seed=0,
        length=6,
        budget=10,
        horizon=2,
        fps=3,
        stride=2,
        dpi=60,
    )
    assert "gif" in artifacts
    for path in artifacts.values():
        assert path.exists()
        assert path.stat().st_size > 0

    assert artifacts["storyboard_pdf"].name == "robot_arm_paper_scott_seed0_storyboard.pdf"
    metadata = json.loads(artifacts["metadata"].read_text(encoding="utf-8"))
    assert metadata["trace"] == "paper"
    assert metadata["method"] == "scott"
    assert "ghost_samples" not in metadata
    assert metadata["trace_description"] == "explanatory visualization trace"
    assert metadata["trace_model"] == "scripted_kinematic_explanatory"
    assert metadata["ee_path_length"] > 0.0
    assert metadata["action_saturation_fraction"] == 0.0
    assert metadata["physical_view_limits"][1] - metadata["physical_view_limits"][0] < 1.2
    assert metadata["trigger_view_limits"][1] - metadata["trigger_view_limits"][0] < 1.0
    assert "final_trigger_width" in metadata
    assert "end_effector_trigger_region" in metadata
