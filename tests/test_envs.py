"""Tests for MuJoCo point-mass environment and monitor."""

import numpy as np
import pytest

from pzr.envs.base import NoisySensorModel, InterventionStats
from pzr.envs.point_mass import PointMassEnv, PointMassConfig, simple_pd_controller
from pzr.envs.point_mass_monitor import (
    PointMassMonitor,
    PointMassMeasurement,
    make_measurement,
)
from pzr.zonotope.protected import ProtectedReducer
from pzr.zonotope.reduction import GirardReducer


class TestPointMassEnv:
    def test_reset_returns_state(self):
        env = PointMassEnv()
        state = env.reset(seed=0)
        assert state.shape == (4,)
        assert np.all(np.isfinite(state))
        env.close()

    def test_step_returns_correct_shape(self):
        env = PointMassEnv()
        env.reset(seed=0)
        state, reward, done, info = env.step(np.array([0.5, 0.5]))
        assert state.shape == (4,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "dist_to_goal" in info
        env.close()

    def test_pd_controller_moves_toward_goal(self):
        env = PointMassEnv()
        state = env.reset(seed=0)
        initial_dist = info_dist = None
        for i in range(50):
            action = simple_pd_controller(state)
            state, _, done, info = env.step(action)
            if i == 0:
                initial_dist = info["dist_to_goal"]
            if done:
                break
        final_dist = info["dist_to_goal"]
        assert final_dist < initial_dist
        env.close()

    def test_episode_terminates(self):
        env = PointMassEnv(config=PointMassConfig(max_steps=50))
        env.reset(seed=0)
        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(np.array([0.0, 0.0]))
            steps += 1
        assert steps <= 50
        env.close()

    def test_true_state_matches_obs(self):
        env = PointMassEnv()
        env.reset(seed=0)
        env.step(np.array([0.5, 0.3]))
        state = env.true_state()
        assert state.shape == (4,)
        assert not np.allclose(state[:2], 0.0)
        env.close()


class TestNoisySensorModel:
    def test_bias_within_bounds(self):
        model = NoisySensorModel(
            bias_bound=np.array([0.1, 0.1, 0.05, 0.05]),
            noise_bound=np.array([0.02, 0.02, 0.01, 0.01]),
        )
        rng = np.random.default_rng(42)
        model.reset(rng)
        assert np.all(np.abs(model.bias) <= 0.1 + 1e-10)

    def test_observe_adds_noise(self):
        model = NoisySensorModel(
            bias_bound=np.array([0.1, 0.1, 0.05, 0.05]),
            noise_bound=np.array([0.02, 0.02, 0.01, 0.01]),
        )
        rng = np.random.default_rng(42)
        model.reset(rng)
        true_state = np.array([1.0, 2.0, 0.5, -0.3])
        observed = model.observe(true_state, rng)
        assert not np.allclose(observed, true_state)
        assert observed.shape == (4,)

    def test_bounded_error(self):
        model = NoisySensorModel(
            bias_bound=np.array([0.1, 0.1, 0.05, 0.05]),
            noise_bound=np.array([0.02, 0.02, 0.01, 0.01]),
        )
        rng = np.random.default_rng(42)
        model.reset(rng)
        true_state = np.zeros(4)
        for _ in range(100):
            obs = model.observe(true_state, rng)
            assert np.all(np.abs(obs) <= 0.12 + 1e-10)


class TestPointMassMonitor:
    def _make_monitor(self):
        noise = NoisySensorModel(
            bias_bound=np.array([0.05, 0.05, 0.02, 0.02]),
            noise_bound=np.array([0.02, 0.02, 0.01, 0.01]),
        )
        return PointMassMonitor(noise_model=noise)

    def test_initial_state(self):
        monitor = self._make_monitor()
        state = monitor.initial_state()
        assert state.zonotope.dimension == 4
        assert state.zonotope.generator_count == 2
        assert state.calibration_indices == (0, 1)

    def test_generators_grow(self):
        monitor = self._make_monitor()
        state = monitor.initial_state()
        for t in range(10):
            m = PointMassMeasurement(time=float(t), position_x=0.0, position_y=0.0, velocity_x=0.0, velocity_y=0.0)
            result = monitor.step(state, m)
            state = result.state
        assert state.zonotope.generator_count == 22  # 2 cal + 10*2 meas

    def test_triggers_safe_at_origin(self):
        monitor = self._make_monitor()
        state = monitor.initial_state()
        m = PointMassMeasurement(time=0.0, position_x=0.0, position_y=0.0, velocity_x=0.0, velocity_y=0.0)
        result = monitor.step(state, m)
        assert all(v.status == "safe" for v in result.verdicts)

    def test_triggers_fire_near_boundary(self):
        monitor = self._make_monitor()
        state = monitor.initial_state()
        m = PointMassMeasurement(time=0.0, position_x=2.8, position_y=0.0, velocity_x=0.0, velocity_y=0.0)
        result = monitor.step(state, m)
        statuses = {v.trigger.name: v.status for v in result.verdicts}
        assert statuses["boundary_x_high"] == "violation"

    def test_reduction_preserves_containment(self):
        monitor = self._make_monitor()
        state = monitor.initial_state()
        rng = np.random.default_rng(42)
        for t in range(15):
            m = PointMassMeasurement(
                time=float(t),
                position_x=float(rng.uniform(-1, 1)),
                position_y=float(rng.uniform(-1, 1)),
                velocity_x=float(rng.uniform(-0.5, 0.5)),
                velocity_y=float(rng.uniform(-0.5, 0.5)),
            )
            result = monitor.step(state, m)
            state = result.state

        budget = 8
        protected = ProtectedReducer(base=GirardReducer())
        red = protected.reduce(state.zonotope, budget, protected_indices=state.calibration_indices)
        assert red.reduced.generator_count <= budget

        lo_orig, hi_orig = state.zonotope.interval_bounds()
        lo_red, hi_red = red.reduced.interval_bounds()
        assert np.all(lo_red <= lo_orig + 1e-10)
        assert np.all(hi_red >= hi_orig - 1e-10)


class TestEndToEndMuJoCo:
    def test_env_with_monitor(self):
        """Run the environment with noise injection and zonotope monitoring."""
        env = PointMassEnv(config=PointMassConfig(max_steps=30))
        noise = NoisySensorModel(
            bias_bound=np.array([0.05, 0.05, 0.02, 0.02]),
            noise_bound=np.array([0.02, 0.02, 0.01, 0.01]),
        )
        monitor = PointMassMonitor(noise_model=noise)
        rng = np.random.default_rng(42)

        true_state = env.reset(seed=42)
        noise.reset(rng)
        mon_state = monitor.initial_state()
        budget = 8

        reductions = 0
        protected = ProtectedReducer(base=GirardReducer())

        for t in range(30):
            action = simple_pd_controller(true_state)
            true_state, _, done, info = env.step(action)

            measurement = make_measurement(true_state, noise, rng, float(t))
            mon_result = monitor.step(mon_state, measurement)
            mon_state = mon_result.state

            if mon_state.zonotope.generator_count > budget:
                cal = mon_state.calibration_indices
                red = protected.reduce(mon_state.zonotope, budget, protected_indices=cal)
                new_cal = tuple(range(len(cal)))
                mon_state = mon_state.with_zonotope(red.reduced, calibration_indices=new_cal)
                reductions += 1
                assert mon_state.zonotope.generator_count <= budget

            if done:
                break

        env.close()
        assert reductions > 0
        assert mon_state.step > 0


class TestInterventionStats:
    def test_rates(self):
        stats = InterventionStats(
            total_steps=100,
            spurious=5,
            justified=3,
            missed=2,
            interventions=8,
        )
        assert stats.spurious_rate == 0.05
        assert stats.missed_rate == 0.02
        assert stats.intervention_rate == 0.08
