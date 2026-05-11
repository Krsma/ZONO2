"""Robot benchmark inspired by the paper's omnidirectional RLola example."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from typing import Sequence

import numpy as np

from pzr.core.zonotope import (
    GeneratorKind,
    GeneratorMetadata,
    GeneratorRequirement,
    Zonotope,
)
from pzr.monitoring.base import (
    MonitorResult,
    MonitorState,
    TriggerSpec,
    evaluate_triggers,
)

A_FILTER = 0
VELOCITY = 1
DISTANCE = 2
POSITION_X = 3
POSITION_Y = 4

STATE_NAMES = ("a_filter", "velocity", "distance", "position_x", "position_y")


@dataclass(frozen=True)
class RobotMeasurement:
    """One measured robot input event."""

    time: float
    direction: float
    acceleration: float


@dataclass(frozen=True)
class RobotPayload:
    """Private state for the black-box robot adapter."""

    previous_time: float | None = None


@dataclass(frozen=True)
class OmnidirectionalRobotMonitor:
    """Black-box monitor for the paper's harder omnidirectional robot example."""

    filter_gain: float = 0.7
    measurement_noise_scale: float = 0.01
    calibration_error_scale: float = 0.005
    geofence_threshold: float = 4.0

    @property
    def triggers(self) -> tuple[TriggerSpec, ...]:
        return (
            TriggerSpec("position_x_above_geofence", POSITION_X, self.geofence_threshold),
            TriggerSpec("position_y_above_geofence", POSITION_Y, self.geofence_threshold),
        )

    def initial_state(self) -> MonitorState:
        center = np.zeros(len(STATE_NAMES), dtype=float)
        generators = np.zeros((len(STATE_NAMES), 1), dtype=float)
        metadata = (
            GeneratorMetadata(GeneratorKind.CALIBRATION, source="delta", age=0),
        )
        return MonitorState(
            Zonotope(center, generators, metadata),
            step=0,
            payload=RobotPayload(previous_time=None),
        )

    def clone_state(self, state: MonitorState) -> MonitorState:
        return MonitorState(
            Zonotope(state.zonotope.center, state.zonotope.generators, state.zonotope.metadata),
            step=state.step,
            payload=state.payload,
        )

    def replace_zonotope(self, state: MonitorState, zonotope: Zonotope) -> MonitorState:
        return state.with_zonotope(zonotope)

    def required_generator_metadata(
        self,
        state: MonitorState,
    ) -> tuple[GeneratorRequirement, ...]:
        _ = state
        return (
            GeneratorRequirement(
                kind=GeneratorKind.CALIBRATION,
                source="delta",
            ),
        )

    def step(self, state: MonitorState, measurement: RobotMeasurement) -> MonitorResult:
        payload = state.payload
        if not isinstance(payload, RobotPayload):
            raise TypeError("robot monitor state payload has the wrong type")

        if payload.previous_time is None:
            dt = 0.0
        else:
            dt = measurement.time - payload.previous_time
            if dt < 0:
                raise ValueError("robot measurements must have nondecreasing time")

        old = state.zonotope.age_generators()
        c = old.center
        gain = self.filter_gain
        memory = 1.0 - gain
        direction_cos = cos(measurement.direction)
        direction_sin = sin(measurement.direction)

        a_center = measurement.acceleration
        a_filter_center = gain * a_center + memory * c[A_FILTER]
        velocity_center = c[VELOCITY] + a_filter_center * dt
        distance_center = 0.5 * a_filter_center * dt * dt + c[VELOCITY] * dt
        position_x_center = c[POSITION_X] + direction_cos * distance_center
        position_y_center = c[POSITION_Y] + direction_sin * distance_center
        new_center = np.array(
            [
                a_filter_center,
                velocity_center,
                distance_center,
                position_x_center,
                position_y_center,
            ],
            dtype=float,
        )

        old_g = old.generators
        existing_count = old.generator_count
        input_coeffs = np.zeros(existing_count + 1, dtype=float)
        for index, meta in enumerate(old.metadata):
            if meta.kind == GeneratorKind.CALIBRATION and meta.source == "delta":
                input_coeffs[index] = self.calibration_error_scale
        input_coeffs[-1] = self.measurement_noise_scale

        previous_a_filter = np.append(old_g[A_FILTER, :], 0.0)
        previous_velocity = np.append(old_g[VELOCITY, :], 0.0)
        previous_position_x = np.append(old_g[POSITION_X, :], 0.0)
        previous_position_y = np.append(old_g[POSITION_Y, :], 0.0)

        a_filter_g = gain * input_coeffs + memory * previous_a_filter
        velocity_g = previous_velocity + a_filter_g * dt
        distance_g = 0.5 * a_filter_g * dt * dt + previous_velocity * dt
        position_x_g = previous_position_x + direction_cos * distance_g
        position_y_g = previous_position_y + direction_sin * distance_g
        new_generators = np.vstack(
            [
                a_filter_g,
                velocity_g,
                distance_g,
                position_x_g,
                position_y_g,
            ]
        )

        new_metadata = (
            *old.metadata,
            GeneratorMetadata(
                GeneratorKind.MEASUREMENT,
                source=f"epsilon@{state.step + 1}",
                age=0,
            ),
        )
        new_state = MonitorState(
            Zonotope(new_center, new_generators, new_metadata),
            step=state.step + 1,
            payload=RobotPayload(previous_time=measurement.time),
        )
        return MonitorResult(new_state, evaluate_triggers(new_state.zonotope, self.triggers))


def generate_robot_trace(
    length: int,
    *,
    seed: int = 0,
    dt: float = 1.0,
) -> tuple[RobotMeasurement, ...]:
    """Generate a reproducible robot measurement trace."""

    rng = np.random.default_rng(seed)
    direction = 0.0
    trace: list[RobotMeasurement] = []
    for index in range(length):
        direction += float(rng.normal(0.0, 0.18))
        acceleration = 0.18 * np.sin(index / 5.0) + float(rng.normal(0.0, 0.04))
        trace.append(
            RobotMeasurement(
                time=index * dt,
                direction=direction,
                acceleration=acceleration,
            )
        )
    return tuple(trace)


def predict_robot_inputs(
    history: Sequence[RobotMeasurement],
    horizon: int,
) -> tuple[RobotMeasurement, ...]:
    """Constant-input short-horizon predictor for the robot benchmark."""

    if not history or horizon <= 0:
        return ()
    last = history[-1]
    if len(history) >= 2:
        step = history[-1].time - history[-2].time
    else:
        step = 1.0
    if step <= 0:
        step = 1.0
    return tuple(
        RobotMeasurement(
            time=last.time + step * (offset + 1),
            direction=last.direction,
            acceleration=last.acceleration,
        )
        for offset in range(horizon)
    )
