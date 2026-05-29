"""Tests for the 3-joint planar robot arm scenario."""

import numpy as np
import pytest

from pzr.envs.robot_arm import (
    LINK_LENGTHS,
    NUM_JOINTS,
    STATE_DIM,
    fk_jacobian,
    forward_kinematics,
)
from pzr.envs.robot_arm_monitor import (
    RobotArmMeasurement,
    RobotArmMonitor,
    generate_robot_arm_trace,
)
from pzr.envs.base import NoisySensorModel
from pzr.zonotope.core import Zonotope


def _has_mujoco():
    try:
        import mujoco
        return True
    except ImportError:
        return False


class TestForwardKinematics:
    def test_zero_angles(self):
        angles = np.zeros(3)
        ee = forward_kinematics(angles)
        expected_x = sum(LINK_LENGTHS)
        np.testing.assert_allclose(ee, [expected_x, 0.0], atol=1e-12)

    def test_first_joint_90(self):
        angles = np.array([np.pi / 2, 0.0, 0.0])
        ee = forward_kinematics(angles)
        expected_y = sum(LINK_LENGTHS)
        np.testing.assert_allclose(ee, [0.0, expected_y], atol=1e-12)

    def test_folded_arm(self):
        angles = np.array([0.0, np.pi, 0.0])
        ee = forward_kinematics(angles)
        expected_x = LINK_LENGTHS[0] - LINK_LENGTHS[1] - LINK_LENGTHS[2]
        np.testing.assert_allclose(ee, [expected_x, 0.0], atol=1e-12)

    def test_custom_link_lengths(self):
        lengths = (1.0, 1.0, 1.0)
        angles = np.zeros(3)
        ee = forward_kinematics(angles, lengths)
        np.testing.assert_allclose(ee, [3.0, 0.0], atol=1e-12)


class TestJacobian:
    def test_shape(self):
        angles = np.zeros(3)
        J = fk_jacobian(angles)
        assert J.shape == (2, 3)

    def test_numerical_agreement(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            angles = rng.uniform(-2, 2, 3)
            J_analytical = fk_jacobian(angles)
            J_numerical = np.zeros((2, 3))
            eps = 1e-7
            for j in range(3):
                a_plus = angles.copy()
                a_plus[j] += eps
                a_minus = angles.copy()
                a_minus[j] -= eps
                J_numerical[:, j] = (
                    forward_kinematics(a_plus) - forward_kinematics(a_minus)
                ) / (2 * eps)
            np.testing.assert_allclose(J_analytical, J_numerical, atol=1e-5)

    def test_singular_at_extension(self):
        angles = np.zeros(3)
        J = fk_jacobian(angles)
        sv = np.linalg.svd(J, compute_uv=False)
        assert sv[0] > 10 * sv[1], "Extended arm should be near-singular"


class TestRobotArmMonitor:
    def _make_monitor(self):
        return RobotArmMonitor(
            noise_model=NoisySensorModel(
                bias_bound=np.array([0.02, 0.02, 0.02, 0.01, 0.01, 0.01]),
                noise_bound=np.array([0.01, 0.01, 0.01, 0.005, 0.005, 0.005]),
            ),
        )

    def test_initial_state(self):
        mon = self._make_monitor()
        state = mon.initial_state()
        assert state.zonotope.dimension == STATE_DIM
        assert state.zonotope.generator_count == NUM_JOINTS
        assert len(state.calibration_indices) == NUM_JOINTS

    def test_step_grows_generators(self):
        mon = self._make_monitor()
        state = mon.initial_state()
        m = RobotArmMeasurement(0.0, (0.1, 0.2, -0.1), (0.0, 0.0, 0.0))
        result = mon.step(state, m)
        assert result.state.zonotope.generator_count == NUM_JOINTS + NUM_JOINTS

        result2 = mon.step(result.state, RobotArmMeasurement(1.0, (0.15, 0.25, -0.05), (0.0, 0.0, 0.0)))
        assert result2.state.zonotope.generator_count == 2 * NUM_JOINTS + NUM_JOINTS

    def test_triggers_count(self):
        mon = self._make_monitor()
        assert len(mon.triggers) == 4

    def test_cartesian_zonotope_center(self):
        mon = self._make_monitor()
        state = mon.initial_state()
        angles = np.array([0.3, 0.5, -0.2])
        m = RobotArmMeasurement(0.0, tuple(angles), (0.0, 0.0, 0.0))
        result = mon.step(state, m)

        cart_z = mon._cartesian_zonotope(result.state.zonotope)
        expected_ee = forward_kinematics(angles)
        np.testing.assert_allclose(cart_z.center, expected_ee, atol=1e-10)

    def test_cartesian_zonotope_generators_shape(self):
        mon = self._make_monitor()
        state = mon.initial_state()
        m = RobotArmMeasurement(0.0, (0.1, 0.2, -0.1), (0.0, 0.0, 0.0))
        result = mon.step(state, m)

        cart_z = mon._cartesian_zonotope(result.state.zonotope)
        assert cart_z.dimension == 2
        assert cart_z.generator_count == result.state.zonotope.generator_count

    def test_verdicts_returned(self):
        mon = self._make_monitor()
        state = mon.initial_state()
        m = RobotArmMeasurement(0.0, (0.1, 0.2, -0.1), (0.0, 0.0, 0.0))
        result = mon.step(state, m)
        assert len(result.verdicts) == len(mon.triggers)

    def test_clone_independence(self):
        mon = self._make_monitor()
        state = mon.initial_state()
        m = RobotArmMeasurement(0.0, (0.1, 0.2, -0.1), (0.0, 0.0, 0.0))
        result = mon.step(state, m)
        cloned = mon.clone_state(result.state)
        assert cloned.zonotope is not result.state.zonotope
        np.testing.assert_allclose(
            cloned.zonotope.generators, result.state.zonotope.generators,
        )


@pytest.mark.skipif(
    not _has_mujoco(), reason="MuJoCo not available",
)
class TestTraceGeneration:
    def test_trace_length(self):
        trace = generate_robot_arm_trace(50, seed=0)
        assert len(trace) == 50

    def test_trace_measurements_finite(self):
        trace = generate_robot_arm_trace(30, seed=1)
        for m in trace:
            assert np.all(np.isfinite(m.joint_angles))
            assert np.all(np.isfinite(m.joint_velocities))

    def test_monitor_processes_trace(self):
        mon = RobotArmMonitor(
            noise_model=NoisySensorModel(
                bias_bound=np.array([0.02, 0.02, 0.02, 0.01, 0.01, 0.01]),
                noise_bound=np.array([0.01, 0.01, 0.01, 0.005, 0.005, 0.005]),
            ),
        )
        trace = generate_robot_arm_trace(20, seed=0)
        state = mon.initial_state()
        for m in trace:
            result = mon.step(state, m)
            state = result.state
            assert state.zonotope.dimension == STATE_DIM
            assert len(result.verdicts) == len(mon.triggers)
