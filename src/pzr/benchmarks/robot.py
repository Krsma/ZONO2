"""Robot benchmarks inspired by the paper's RLola examples."""

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

VX = 0
VX_FILTER = 1
MOTIVATING_POSITION_X = 2
VY = 3
VY_FILTER = 4
MOTIVATING_POSITION_Y = 5

MOTIVATING_STATE_NAMES = (
    "vx",
    "vx_filter",
    "position_x",
    "vy",
    "vy_filter",
    "position_y",
)


@dataclass(frozen=True)
class RobotMeasurement:
    """One measured robot input event."""

    time: float
    direction: float
    acceleration: float


@dataclass(frozen=True)
class VelocityRobotMeasurement:
    """One measured event for the paper's velocity/endstop robot."""

    time: float
    bump_x: bool
    vel_x: float
    bump_y: bool
    vel_y: float


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
            TriggerSpec(
                "position_x_above_geofence",
                POSITION_X,
                self.geofence_threshold,
                overlap=0.01,
            ),
            TriggerSpec(
                "position_y_above_geofence",
                POSITION_Y,
                self.geofence_threshold,
                overlap=0.01,
            ),
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

        old_g, old_metadata = _ensure_calibration_generators(
            old,
            ("delta",),
        )
        existing_count = old_g.shape[1]
        input_coeffs = np.zeros(existing_count + 1, dtype=float)
        for index, meta in enumerate(old_metadata):
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
            *old_metadata,
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


@dataclass(frozen=True)
class MotivatingRobotMonitor:
    """Two-axis velocity/filter/endstop robot from the paper's motivating example."""

    filter_gain: float = 0.8
    measurement_noise_scale: float = 0.1
    calibration_error_scale: float = 0.05
    geofence_threshold: float = 4.0

    @property
    def triggers(self) -> tuple[TriggerSpec, ...]:
        return (
            TriggerSpec(
                "position_x_above_geofence",
                MOTIVATING_POSITION_X,
                self.geofence_threshold,
                overlap=0.01,
            ),
            TriggerSpec(
                "position_y_above_geofence",
                MOTIVATING_POSITION_Y,
                self.geofence_threshold,
                overlap=0.01,
            ),
        )

    def initial_state(self) -> MonitorState:
        center = np.zeros(len(MOTIVATING_STATE_NAMES), dtype=float)
        generators = np.zeros((len(MOTIVATING_STATE_NAMES), 2), dtype=float)
        metadata = (
            GeneratorMetadata(GeneratorKind.CALIBRATION, source="delta_x", age=0),
            GeneratorMetadata(GeneratorKind.CALIBRATION, source="delta_y", age=0),
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
                source="delta_x",
            ),
            GeneratorRequirement(
                kind=GeneratorKind.CALIBRATION,
                source="delta_y",
            ),
        )

    def step(self, state: MonitorState, measurement: VelocityRobotMeasurement) -> MonitorResult:
        payload = state.payload
        if not isinstance(payload, RobotPayload):
            raise TypeError("motivating robot monitor state payload has the wrong type")

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

        old_g, old_metadata = _ensure_calibration_generators(
            old,
            ("delta_x", "delta_y"),
        )
        existing_count = old_g.shape[1]
        x_input_coeffs = np.zeros(existing_count + 2, dtype=float)
        y_input_coeffs = np.zeros(existing_count + 2, dtype=float)
        for index, meta in enumerate(old_metadata):
            if meta.kind == GeneratorKind.CALIBRATION and meta.source == "delta_x":
                x_input_coeffs[index] = self.calibration_error_scale
            elif meta.kind == GeneratorKind.CALIBRATION and meta.source == "delta_y":
                y_input_coeffs[index] = self.calibration_error_scale
        x_input_coeffs[-2] = self.measurement_noise_scale
        y_input_coeffs[-1] = self.measurement_noise_scale

        previous_vx_filter = np.append(old_g[VX_FILTER, :], [0.0, 0.0])
        previous_position_x = np.append(old_g[MOTIVATING_POSITION_X, :], [0.0, 0.0])
        previous_vy_filter = np.append(old_g[VY_FILTER, :], [0.0, 0.0])
        previous_position_y = np.append(old_g[MOTIVATING_POSITION_Y, :], [0.0, 0.0])

        vx_center = measurement.vel_x
        vx_filter_center = gain * vx_center + memory * c[VX_FILTER]
        position_x_center = (
            0.0
            if measurement.bump_x
            else c[MOTIVATING_POSITION_X] + vx_filter_center * dt
        )
        vy_center = measurement.vel_y
        vy_filter_center = gain * vy_center + memory * c[VY_FILTER]
        position_y_center = (
            0.0
            if measurement.bump_y
            else c[MOTIVATING_POSITION_Y] + vy_filter_center * dt
        )

        new_center = np.array(
            [
                vx_center,
                vx_filter_center,
                position_x_center,
                vy_center,
                vy_filter_center,
                position_y_center,
            ],
            dtype=float,
        )
        vx_g = x_input_coeffs
        vx_filter_g = gain * vx_g + memory * previous_vx_filter
        position_x_g = (
            np.zeros(existing_count + 2, dtype=float)
            if measurement.bump_x
            else previous_position_x + vx_filter_g * dt
        )
        vy_g = y_input_coeffs
        vy_filter_g = gain * vy_g + memory * previous_vy_filter
        position_y_g = (
            np.zeros(existing_count + 2, dtype=float)
            if measurement.bump_y
            else previous_position_y + vy_filter_g * dt
        )
        new_generators = np.vstack(
            [vx_g, vx_filter_g, position_x_g, vy_g, vy_filter_g, position_y_g]
        )
        new_metadata = (
            *old_metadata,
            GeneratorMetadata(
                GeneratorKind.MEASUREMENT,
                source=f"epsilon@{state.step + 1}",
                age=0,
            ),
            GeneratorMetadata(
                GeneratorKind.MEASUREMENT,
                source=f"tau@{state.step + 1}",
                age=0,
            ),
        )

        new_state = MonitorState(
            Zonotope(new_center, new_generators, new_metadata),
            step=state.step + 1,
            payload=RobotPayload(previous_time=measurement.time),
        )
        return MonitorResult(new_state, evaluate_triggers(new_state.zonotope, self.triggers))


SimpleRobotMonitor = MotivatingRobotMonitor
SIMPLE_STATE_NAMES = MOTIVATING_STATE_NAMES


def _ensure_calibration_generators(
    zonotope: Zonotope,
    sources: tuple[str, ...],
) -> tuple[np.ndarray, tuple[GeneratorMetadata, ...]]:
    generators = zonotope.generators
    metadata = list(zonotope.metadata)
    missing = [
        source
        for source in sources
        if not any(
            meta.kind == GeneratorKind.CALIBRATION and meta.source == source
            for meta in metadata
        )
    ]
    if not missing:
        return generators, tuple(metadata)
    appended = np.zeros((zonotope.dimension, len(missing)), dtype=float)
    generators = np.hstack([generators, appended])
    metadata.extend(
        GeneratorMetadata(GeneratorKind.CALIBRATION, source=source, age=0)
        for source in missing
    )
    return generators, tuple(metadata)


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


def generate_simple_robot_trace(
    length: int,
    *,
    seed: int = 0,
    dt: float = 1.0,
) -> tuple[VelocityRobotMeasurement, ...]:
    """Generate a reproducible trace for the paper's motivating robot."""

    rng = np.random.default_rng(seed)
    calibration_x = float(rng.uniform(-0.05, 0.05))
    calibration_y = float(rng.uniform(-0.05, 0.05))
    position_x = 0.0
    position_y = 0.0
    velocity_x = 0.0
    velocity_y = 0.0
    trace: list[VelocityRobotMeasurement] = []
    for index in range(length):
        velocity_x = (
            0.78 * velocity_x
            + 0.20 * np.sin(index / 15.0)
            + float(rng.normal(0.0, 0.08))
        )
        velocity_y = (
            0.78 * velocity_y
            + 0.20 * np.cos(index / 17.0)
            + float(rng.normal(0.0, 0.08))
        )
        position_x += velocity_x * dt
        position_y += velocity_y * dt
        bump_x = position_x <= 0.0
        bump_y = position_y <= 0.0
        if bump_x:
            position_x = 0.0
            velocity_x = abs(velocity_x) * 0.25 + float(rng.uniform(0.0, 0.04))
        if bump_y:
            position_y = 0.0
            velocity_y = abs(velocity_y) * 0.25 + float(rng.uniform(0.0, 0.04))
        measured_x = (
            velocity_x
            + calibration_x
            + float(rng.uniform(-0.1, 0.1))
        )
        measured_y = (
            velocity_y
            + calibration_y
            + float(rng.uniform(-0.1, 0.1))
        )
        trace.append(
            VelocityRobotMeasurement(
                time=index * dt,
                bump_x=bump_x,
                vel_x=measured_x,
                bump_y=bump_y,
                vel_y=measured_y,
            )
        )
    return tuple(trace)


def predict_robot_inputs(
    history: Sequence[RobotMeasurement | VelocityRobotMeasurement],
    horizon: int,
) -> tuple[RobotMeasurement | VelocityRobotMeasurement, ...]:
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
        _prediction_with_time(last, last.time + step * (offset + 1))
        for offset in range(horizon)
    )


def _prediction_with_time(
    measurement: RobotMeasurement | VelocityRobotMeasurement,
    time: float,
) -> RobotMeasurement | VelocityRobotMeasurement:
    if isinstance(measurement, VelocityRobotMeasurement):
        return VelocityRobotMeasurement(
            time=time,
            bump_x=False,
            vel_x=measurement.vel_x,
            bump_y=False,
            vel_y=measurement.vel_y,
        )
    return RobotMeasurement(
        time=time,
        direction=measurement.direction,
        acceleration=measurement.acceleration,
    )
