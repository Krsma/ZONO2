"""Omnidirectional robot monitor from the paper.

State: [a_filter, velocity, distance, position_x, position_y]
Dynamics:
  a_filter := gain * acceleration + (1-gain) * a_filter.prev
  velocity := velocity.prev + a_filter * dt
  distance := 0.5 * a_filter * dt^2 + velocity.prev * dt
  position_x := position_x.prev + cos(direction) * distance
  position_y := position_y.prev + sin(direction) * distance

Generators: 1 persistent calibration (delta) + 1 fresh measurement per step.
Triggers: position_x > geofence, position_y > geofence.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

import numpy as np

from pzr.monitoring.base import MonitorResult, MonitorState, TriggerSpec
from pzr.monitoring.triggers import evaluate_triggers
from pzr.zonotope.core import Zonotope

A_FILTER = 0
VELOCITY = 1
DISTANCE = 2
POSITION_X = 3
POSITION_Y = 4
STATE_DIM = 5


@dataclass(frozen=True)
class OmniRobotMeasurement:
    time: float
    direction: float
    acceleration: float


@dataclass(frozen=True)
class OmniRobotMonitor:
    filter_gain: float = 0.7
    measurement_noise_scale: float = 0.01
    calibration_error_scale: float = 0.005
    geofence_threshold: float = 4.0

    @property
    def triggers(self) -> tuple[TriggerSpec, ...]:
        return (
            TriggerSpec("position_x_above_geofence", POSITION_X, self.geofence_threshold, overlap=0.01),
            TriggerSpec("position_y_above_geofence", POSITION_Y, self.geofence_threshold, overlap=0.01),
        )

    @property
    def num_calibration_generators(self) -> int:
        return 1

    def initial_state(self) -> MonitorState:
        center = np.zeros(STATE_DIM, dtype=np.float64)
        generators = np.zeros((STATE_DIM, 1), dtype=np.float64)
        return MonitorState(
            zonotope=Zonotope(center, generators),
            step=0,
            calibration_indices=(0,),
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

    def step(self, state: MonitorState, measurement: OmniRobotMeasurement) -> MonitorResult:
        old_z = state.zonotope
        old_c = old_z.center
        old_g = old_z.generators
        n_existing = old_z.generator_count

        prev_time = state.payload
        dt = 0.0 if prev_time is None else measurement.time - prev_time

        gain = self.filter_gain
        mem = 1.0 - gain
        dir_cos = cos(measurement.direction)
        dir_sin = sin(measurement.direction)

        a_c = measurement.acceleration
        af_c = gain * a_c + mem * old_c[A_FILTER]
        v_c = old_c[VELOCITY] + af_c * dt
        d_c = 0.5 * af_c * dt ** 2 + old_c[VELOCITY] * dt
        px_c = old_c[POSITION_X] + dir_cos * d_c
        py_c = old_c[POSITION_Y] + dir_sin * d_c
        new_center = np.array([af_c, v_c, d_c, px_c, py_c], dtype=np.float64)

        # Build input coefficient vector: calibration + measurement
        input_coeffs = np.zeros(n_existing + 1, dtype=np.float64)
        for idx in state.calibration_indices:
            if idx < n_existing:
                input_coeffs[idx] = self.calibration_error_scale
        input_coeffs[-1] = self.measurement_noise_scale

        prev_af = np.append(old_g[A_FILTER, :], 0.0)
        prev_v = np.append(old_g[VELOCITY, :], 0.0)
        prev_px = np.append(old_g[POSITION_X, :], 0.0)
        prev_py = np.append(old_g[POSITION_Y, :], 0.0)

        af_g = gain * input_coeffs + mem * prev_af
        v_g = prev_v + af_g * dt
        d_g = 0.5 * af_g * dt ** 2 + prev_v * dt
        px_g = prev_px + dir_cos * d_g
        py_g = prev_py + dir_sin * d_g

        new_generators = np.vstack([af_g, v_g, d_g, px_g, py_g])
        new_z = Zonotope(new_center, new_generators)

        new_state = MonitorState(
            zonotope=new_z,
            step=state.step + 1,
            calibration_indices=state.calibration_indices,
            payload=measurement.time,
        )
        return MonitorResult(new_state, evaluate_triggers(new_z, self.triggers))


def generate_omni_robot_trace(
    length: int,
    *,
    seed: int = 0,
    dt: float = 1.0,
) -> tuple[OmniRobotMeasurement, ...]:
    rng = np.random.default_rng(seed)
    direction = 0.0
    trace: list[OmniRobotMeasurement] = []
    for i in range(length):
        direction += float(rng.normal(0.0, 0.18))
        acceleration = 0.18 * np.sin(i / 5.0) + float(rng.normal(0.0, 0.04))
        trace.append(OmniRobotMeasurement(time=i * dt, direction=direction, acceleration=acceleration))
    return tuple(trace)
