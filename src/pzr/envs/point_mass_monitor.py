"""Monitor adapter for the MuJoCo point-mass environment.

State: [position_x, position_y, velocity_x, velocity_y] (4D)
Generators: 2 persistent calibration (bias_x, bias_y) + 2 fresh measurement per step
Triggers:
  - obstacle_clearance_violation: min clearance to obstacle < threshold
  - boundary_x_violation: |position_x| > arena_limit
  - boundary_y_violation: |position_y| > arena_limit
  - speed_violation: speed > max_speed
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from pzr.envs.base import NoisySensorModel
from pzr.envs.point_mass import ARENA_HALF_SIZE, PointMassConfig
from pzr.monitoring.base import MonitorResult, MonitorState, TriggerSpec
from pzr.monitoring.triggers import evaluate_triggers
from pzr.zonotope.core import Zonotope

POS_X = 0
POS_Y = 1
VEL_X = 2
VEL_Y = 3
STATE_DIM = 4


@dataclass(frozen=True)
class PointMassMeasurement:
    """Noisy observation from the point-mass environment."""

    time: float
    position_x: float
    position_y: float
    velocity_x: float
    velocity_y: float


@dataclass(frozen=True)
class PointMassMonitor:
    """Zonotope monitor for the point-mass environment."""

    noise_model: NoisySensorModel
    config: PointMassConfig = field(default_factory=PointMassConfig)
    boundary_limit: float = 2.0

    @property
    def triggers(self) -> tuple[TriggerSpec, ...]:
        return (
            TriggerSpec("boundary_x_high", POS_X, self.boundary_limit, "above", overlap=0.05),
            TriggerSpec("boundary_x_low", POS_X, -self.boundary_limit, "below", overlap=0.05),
            TriggerSpec("boundary_y_high", POS_Y, self.boundary_limit, "above", overlap=0.05),
            TriggerSpec("boundary_y_low", POS_Y, -self.boundary_limit, "below", overlap=0.05),
        )

    @property
    def num_calibration_generators(self) -> int:
        return 2

    def initial_state(self) -> MonitorState:
        center = np.zeros(STATE_DIM, dtype=np.float64)
        generators = np.zeros((STATE_DIM, 2), dtype=np.float64)
        return MonitorState(
            zonotope=Zonotope(center, generators),
            step=0,
            calibration_indices=(0, 1),
        )

    def clone_state(self, state: MonitorState) -> MonitorState:
        z = state.zonotope
        return MonitorState(
            zonotope=Zonotope(z.center.copy(), z.generators.copy()),
            step=state.step,
            calibration_indices=state.calibration_indices,
            payload=state.payload,
        )

    def replace_zonotope(self, state: MonitorState, zonotope: Zonotope) -> MonitorState:
        return state.with_zonotope(zonotope)

    def trigger_zonotope(self, state: MonitorState) -> Zonotope:
        return state.zonotope

    def step(self, state: MonitorState, measurement: PointMassMeasurement) -> MonitorResult:
        old_z = state.zonotope
        old_g = old_z.generators
        n_existing = old_z.generator_count
        cal = state.calibration_indices

        new_center = np.array([
            measurement.position_x,
            measurement.position_y,
            measurement.velocity_x,
            measurement.velocity_y,
        ], dtype=np.float64)

        # Calibration bias contributes to position observations
        cal_scale_x = float(self.noise_model.bias_bound[0]) if len(self.noise_model.bias_bound) > 0 else 0.0
        cal_scale_y = float(self.noise_model.bias_bound[1]) if len(self.noise_model.bias_bound) > 1 else 0.0
        noise_scale_x = float(self.noise_model.noise_bound[0]) if len(self.noise_model.noise_bound) > 0 else 0.0
        noise_scale_y = float(self.noise_model.noise_bound[1]) if len(self.noise_model.noise_bound) > 1 else 0.0

        # Build new generator columns: [existing] + [2 new measurement noise]
        n_new = n_existing + 2
        new_g = np.zeros((STATE_DIM, n_new), dtype=np.float64)

        # Copy existing generators (they propagate)
        if n_existing > 0:
            new_g[:, :n_existing] = old_g

        # Calibration generators affect position dims
        if len(cal) >= 1 and cal[0] < n_existing:
            new_g[POS_X, cal[0]] = cal_scale_x
        if len(cal) >= 2 and cal[1] < n_existing:
            new_g[POS_Y, cal[1]] = cal_scale_y

        # Fresh measurement noise
        new_g[POS_X, n_existing] = noise_scale_x
        new_g[VEL_X, n_existing] = noise_scale_x * 0.5
        new_g[POS_Y, n_existing + 1] = noise_scale_y
        new_g[VEL_Y, n_existing + 1] = noise_scale_y * 0.5

        new_z = Zonotope(new_center, new_g)
        new_state = MonitorState(
            zonotope=new_z,
            step=state.step + 1,
            calibration_indices=cal,
            payload=measurement.time,
        )
        return MonitorResult(new_state, evaluate_triggers(new_z, self.triggers))


def generate_point_mass_trace(
    length: int,
    *,
    seed: int = 0,
    bias_bound: NDArray[np.float64] | None = None,
    noise_bound: NDArray[np.float64] | None = None,
) -> tuple[PointMassMeasurement, ...]:
    """Generate a trace of noisy measurements from a MuJoCo point-mass episode.

    Uses multi-waypoint navigation to produce diverse trajectories that
    approach obstacles and boundaries, creating interesting zonotope dynamics.
    """
    from pzr.envs.point_mass import PointMassEnv, simple_pd_controller

    if bias_bound is None:
        bias_bound = np.array([0.15, 0.15, 0.08, 0.08])
    if noise_bound is None:
        noise_bound = np.array([0.08, 0.08, 0.04, 0.04])

    noise_model = NoisySensorModel(bias_bound=bias_bound, noise_bound=noise_bound)
    rng = np.random.default_rng(seed)
    noise_model.reset(rng)

    num_waypoints = max(3, length // 25)
    waypoints = rng.uniform(-2.2, 2.2, (num_waypoints, 2))
    waypoint_idx = 0

    env = PointMassEnv(config=PointMassConfig(max_steps=length + 10))
    true_state = env.reset(seed=seed)
    trace: list[PointMassMeasurement] = []

    for t in range(length):
        goal = waypoints[waypoint_idx]
        action = simple_pd_controller(true_state, goal=goal)
        true_state, _, done, _ = env.step(action)
        trace.append(make_measurement(true_state, noise_model, rng, float(t)))

        dist = float(np.linalg.norm(true_state[:2] - goal))
        if dist < 0.5 and waypoint_idx < num_waypoints - 1:
            waypoint_idx += 1

        if done and t < length - 1:
            env = PointMassEnv(config=PointMassConfig(max_steps=length + 10))
            true_state = env.reset(seed=seed + t + 1)

    env.close()
    return tuple(trace)


def make_measurement(
    true_state: NDArray[np.float64],
    noise_model: NoisySensorModel,
    rng: np.random.Generator,
    time: float,
) -> PointMassMeasurement:
    """Create a noisy measurement from the true state."""
    noisy = noise_model.observe(true_state, rng)
    return PointMassMeasurement(
        time=time,
        position_x=float(noisy[0]),
        position_y=float(noisy[1]),
        velocity_x=float(noisy[2]),
        velocity_y=float(noisy[3]),
    )
