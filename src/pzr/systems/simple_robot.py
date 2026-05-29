"""Simple two-axis robot from the paper's motivating example.

State: [vx, vx_filter, position_x, vy, vy_filter, position_y]
Dynamics:
  vx := measured velocity x (with calibration + noise)
  vx_filter := gain * vx + (1-gain) * vx_filter.prev
  position_x := position_x.prev + vx_filter * dt  (reset to 0 on bump)
  (symmetric for y)

Generators: 2 persistent calibration (delta_x, delta_y) + 2 fresh measurement per step.
Triggers: position_x > geofence, position_y > geofence.
Endstop: bump resets position and its generator row to zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pzr.monitoring.base import MonitorResult, MonitorState, TriggerSpec
from pzr.monitoring.triggers import evaluate_triggers
from pzr.zonotope.core import Zonotope

VX = 0
VX_FILTER = 1
POS_X = 2
VY = 3
VY_FILTER = 4
POS_Y = 5
STATE_DIM = 6


@dataclass(frozen=True)
class SimpleRobotMeasurement:
    time: float
    bump_x: bool
    vel_x: float
    bump_y: bool
    vel_y: float


@dataclass(frozen=True)
class SimpleRobotMonitor:
    filter_gain: float = 0.8
    measurement_noise_scale: float = 0.1
    calibration_error_scale: float = 0.05
    geofence_threshold: float = 4.0

    @property
    def triggers(self) -> tuple[TriggerSpec, ...]:
        return (
            TriggerSpec("position_x_above_geofence", POS_X, self.geofence_threshold, overlap=0.01),
            TriggerSpec("position_y_above_geofence", POS_Y, self.geofence_threshold, overlap=0.01),
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

    def step(self, state: MonitorState, measurement: SimpleRobotMeasurement) -> MonitorResult:
        old_z = state.zonotope
        old_c = old_z.center
        old_g = old_z.generators
        n_existing = old_z.generator_count
        cal_indices = state.calibration_indices

        prev_time = state.payload
        dt = 0.0 if prev_time is None else measurement.time - prev_time

        gain = self.filter_gain
        mem = 1.0 - gain

        # Input coefficient vectors: x uses delta_x + epsilon_x, y uses delta_y + epsilon_y
        x_input = np.zeros(n_existing + 2, dtype=np.float64)
        y_input = np.zeros(n_existing + 2, dtype=np.float64)
        if len(cal_indices) >= 1 and cal_indices[0] < n_existing:
            x_input[cal_indices[0]] = self.calibration_error_scale
        if len(cal_indices) >= 2 and cal_indices[1] < n_existing:
            y_input[cal_indices[1]] = self.calibration_error_scale
        x_input[-2] = self.measurement_noise_scale
        y_input[-1] = self.measurement_noise_scale

        prev_vxf = np.append(old_g[VX_FILTER, :], [0.0, 0.0])
        prev_px = np.append(old_g[POS_X, :], [0.0, 0.0])
        prev_vyf = np.append(old_g[VY_FILTER, :], [0.0, 0.0])
        prev_py = np.append(old_g[POS_Y, :], [0.0, 0.0])

        vx_c = measurement.vel_x
        vxf_c = gain * vx_c + mem * old_c[VX_FILTER]
        px_c = 0.0 if measurement.bump_x else old_c[POS_X] + vxf_c * dt
        vy_c = measurement.vel_y
        vyf_c = gain * vy_c + mem * old_c[VY_FILTER]
        py_c = 0.0 if measurement.bump_y else old_c[POS_Y] + vyf_c * dt

        new_center = np.array([vx_c, vxf_c, px_c, vy_c, vyf_c, py_c], dtype=np.float64)

        vx_g = x_input
        vxf_g = gain * vx_g + mem * prev_vxf
        px_g = np.zeros(n_existing + 2, dtype=np.float64) if measurement.bump_x else prev_px + vxf_g * dt
        vy_g = y_input
        vyf_g = gain * vy_g + mem * prev_vyf
        py_g = np.zeros(n_existing + 2, dtype=np.float64) if measurement.bump_y else prev_py + vyf_g * dt

        new_generators = np.vstack([vx_g, vxf_g, px_g, vy_g, vyf_g, py_g])
        new_z = Zonotope(new_center, new_generators)

        new_state = MonitorState(
            zonotope=new_z,
            step=state.step + 1,
            calibration_indices=cal_indices,
            payload=measurement.time,
        )
        return MonitorResult(new_state, evaluate_triggers(new_z, self.triggers))


def generate_simple_robot_trace(
    length: int,
    *,
    seed: int = 0,
    dt: float = 1.0,
) -> tuple[SimpleRobotMeasurement, ...]:
    rng = np.random.default_rng(seed)
    cal_x = float(rng.uniform(-0.05, 0.05))
    cal_y = float(rng.uniform(-0.05, 0.05))
    pos_x = pos_y = 0.0
    vel_x = vel_y = 0.0
    trace: list[SimpleRobotMeasurement] = []
    for i in range(length):
        vel_x = 0.78 * vel_x + 0.20 * np.sin(i / 15.0) + float(rng.normal(0.0, 0.08))
        vel_y = 0.78 * vel_y + 0.20 * np.cos(i / 17.0) + float(rng.normal(0.0, 0.08))
        pos_x += vel_x * dt
        pos_y += vel_y * dt
        bump_x = pos_x <= 0.0
        bump_y = pos_y <= 0.0
        if bump_x:
            pos_x = 0.0
            vel_x = abs(vel_x) * 0.25 + float(rng.uniform(0.0, 0.04))
        if bump_y:
            pos_y = 0.0
            vel_y = abs(vel_y) * 0.25 + float(rng.uniform(0.0, 0.04))
        measured_x = vel_x + cal_x + float(rng.uniform(-0.1, 0.1))
        measured_y = vel_y + cal_y + float(rng.uniform(-0.1, 0.1))
        trace.append(SimpleRobotMeasurement(time=i * dt, bump_x=bump_x, vel_x=measured_x, bump_y=bump_y, vel_y=measured_y))
    return tuple(trace)
