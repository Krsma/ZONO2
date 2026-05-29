"""Tests for system dynamics and monitors."""

import numpy as np
import pytest

from pzr.systems.omni_robot import (
    OmniRobotMeasurement,
    OmniRobotMonitor,
    generate_omni_robot_trace,
)
from pzr.systems.simple_robot import (
    SimpleRobotMeasurement,
    SimpleRobotMonitor,
    generate_simple_robot_trace,
)
from pzr.zonotope.reduction import GirardReducer
from pzr.zonotope.protected import ProtectedReducer


class TestOmniRobotMonitor:
    def test_initial_state(self):
        monitor = OmniRobotMonitor()
        state = monitor.initial_state()
        assert state.zonotope.dimension == 5
        assert state.zonotope.generator_count == 1
        assert state.calibration_indices == (0,)

    def test_generators_grow(self):
        monitor = OmniRobotMonitor()
        state = monitor.initial_state()
        trace = generate_omni_robot_trace(10, seed=42)
        for m in trace:
            result = monitor.step(state, m)
            state = result.state
        assert state.zonotope.generator_count == 11  # 1 calibration + 10 measurements

    def test_calibration_index_preserved(self):
        monitor = OmniRobotMonitor()
        state = monitor.initial_state()
        trace = generate_omni_robot_trace(5, seed=0)
        for m in trace:
            result = monitor.step(state, m)
            state = result.state
        assert state.calibration_indices == (0,)

    def test_center_evolves(self):
        monitor = OmniRobotMonitor()
        state = monitor.initial_state()
        m = OmniRobotMeasurement(time=0.0, direction=0.0, acceleration=1.0)
        result = monitor.step(state, m)
        assert result.state.zonotope.center[0] != 0.0  # a_filter updated

    def test_triggers_safe_initially(self):
        monitor = OmniRobotMonitor()
        state = monitor.initial_state()
        m = OmniRobotMeasurement(time=0.0, direction=0.0, acceleration=0.0)
        result = monitor.step(state, m)
        assert all(v.status == "safe" for v in result.verdicts)

    def test_clone_independence(self):
        monitor = OmniRobotMonitor()
        state = monitor.initial_state()
        m = OmniRobotMeasurement(time=0.0, direction=0.0, acceleration=1.0)
        result = monitor.step(state, m)
        clone = monitor.clone_state(result.state)
        # Mutating clone should not affect original
        assert np.array_equal(clone.zonotope.center, result.state.zonotope.center)

    def test_reduction_preserves_containment(self):
        monitor = OmniRobotMonitor()
        state = monitor.initial_state()
        trace = generate_omni_robot_trace(10, seed=123)
        for m in trace:
            result = monitor.step(state, m)
            state = result.state

        budget = 8
        protected = ProtectedReducer(base=GirardReducer())
        red_result = protected.reduce(
            state.zonotope, budget, protected_indices=state.calibration_indices,
        )
        assert red_result.reduced.generator_count <= budget
        # Verify interval hull containment
        lo_orig, hi_orig = state.zonotope.interval_bounds()
        lo_red, hi_red = red_result.reduced.interval_bounds()
        assert np.all(lo_red <= lo_orig + 1e-10)
        assert np.all(hi_red >= hi_orig - 1e-10)


class TestSimpleRobotMonitor:
    def test_initial_state(self):
        monitor = SimpleRobotMonitor()
        state = monitor.initial_state()
        assert state.zonotope.dimension == 6
        assert state.zonotope.generator_count == 2
        assert state.calibration_indices == (0, 1)

    def test_generators_grow_by_two(self):
        monitor = SimpleRobotMonitor()
        state = monitor.initial_state()
        trace = generate_simple_robot_trace(5, seed=42)
        for m in trace:
            result = monitor.step(state, m)
            state = result.state
        assert state.zonotope.generator_count == 12  # 2 calibration + 5 * 2 measurements

    def test_endstop_resets_position(self):
        monitor = SimpleRobotMonitor()
        state = monitor.initial_state()
        m = SimpleRobotMeasurement(time=0.0, bump_x=True, vel_x=0.5, bump_y=False, vel_y=0.0)
        result = monitor.step(state, m)
        assert result.state.zonotope.center[2] == 0.0  # position_x reset

    def test_no_endstop_accumulates(self):
        monitor = SimpleRobotMonitor()
        state = monitor.initial_state()
        m1 = SimpleRobotMeasurement(time=0.0, bump_x=False, vel_x=1.0, bump_y=False, vel_y=0.0)
        result = monitor.step(state, m1)
        m2 = SimpleRobotMeasurement(time=1.0, bump_x=False, vel_x=1.0, bump_y=False, vel_y=0.0)
        result2 = monitor.step(result.state, m2)
        assert result2.state.zonotope.center[2] > 0.0  # position_x accumulated

    def test_calibration_indices_stable(self):
        monitor = SimpleRobotMonitor()
        state = monitor.initial_state()
        trace = generate_simple_robot_trace(3, seed=0)
        for m in trace:
            result = monitor.step(state, m)
            state = result.state
        assert state.calibration_indices == (0, 1)

    def test_reduction_and_continue(self):
        """Run monitor, reduce, then continue stepping."""
        monitor = SimpleRobotMonitor()
        state = monitor.initial_state()
        trace = generate_simple_robot_trace(8, seed=99)
        for m in trace[:5]:
            result = monitor.step(state, m)
            state = result.state

        budget = 8
        protected = ProtectedReducer(base=GirardReducer())
        red_result = protected.reduce(
            state.zonotope, budget, protected_indices=state.calibration_indices,
        )
        # After reduction, calibration generators are at positions 0..1
        state = state.with_zonotope(
            red_result.reduced,
            calibration_indices=(0, 1),
        )
        assert state.zonotope.generator_count <= budget

        # Continue stepping
        for m in trace[5:]:
            result = monitor.step(state, m)
            state = result.state
        assert state.step == 8


class TestTraceGeneration:
    def test_omni_deterministic(self):
        t1 = generate_omni_robot_trace(10, seed=42)
        t2 = generate_omni_robot_trace(10, seed=42)
        for a, b in zip(t1, t2):
            assert a.time == b.time
            assert a.acceleration == b.acceleration

    def test_simple_deterministic(self):
        t1 = generate_simple_robot_trace(10, seed=42)
        t2 = generate_simple_robot_trace(10, seed=42)
        for a, b in zip(t1, t2):
            assert a.vel_x == b.vel_x

    def test_omni_correct_length(self):
        assert len(generate_omni_robot_trace(20)) == 20

    def test_simple_correct_length(self):
        assert len(generate_simple_robot_trace(20)) == 20
