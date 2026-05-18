"""Thermostat benchmark with persistent calibration and fresh sensor noise."""

from __future__ import annotations

from dataclasses import dataclass
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

TEMPERATURE = 0
FILTERED_TEMPERATURE = 1
HVAC_EFFORT = 2
COMFORT_DEVIATION = 3

THERMOSTAT_STATE_NAMES = (
    "temperature",
    "filtered_temperature",
    "hvac_effort",
    "comfort_deviation",
)


@dataclass(frozen=True)
class ThermostatMeasurement:
    """One thermostat input event."""

    time: float
    ambient_temperature: float
    hvac_effort: float


@dataclass(frozen=True)
class ThermostatPayload:
    """Private state for the thermostat adapter."""

    previous_time: float | None = None


@dataclass(frozen=True)
class ThermostatMonitor:
    """Black-box thermostat monitor with comfort and safety triggers."""

    setpoint: float = 21.0
    filter_gain: float = 0.35
    thermal_leak: float = 0.08
    hvac_gain: float = 0.45
    measurement_noise_scale: float = 0.04
    calibration_error_scale: float = 0.025
    comfort_band: float = 1.0
    safety_low: float = 15.0
    safety_high: float = 28.0

    @property
    def triggers(self) -> tuple[TriggerSpec, ...]:
        return (
            TriggerSpec(
                "comfort_deviation_above_band",
                COMFORT_DEVIATION,
                self.comfort_band,
                direction="above",
                overlap=0.01,
            ),
            TriggerSpec(
                "comfort_deviation_below_band",
                COMFORT_DEVIATION,
                -self.comfort_band,
                direction="below",
                overlap=0.01,
            ),
            TriggerSpec(
                "temperature_above_safety",
                TEMPERATURE,
                self.safety_high,
                direction="above",
                overlap=0.01,
            ),
            TriggerSpec(
                "temperature_below_safety",
                TEMPERATURE,
                self.safety_low,
                direction="below",
                overlap=0.01,
            ),
        )

    def initial_state(self) -> MonitorState:
        center = np.array([self.setpoint, self.setpoint, 0.0, 0.0], dtype=float)
        generators = np.zeros((len(THERMOSTAT_STATE_NAMES), 1), dtype=float)
        metadata = (
            GeneratorMetadata(
                GeneratorKind.CALIBRATION,
                source="thermal_bias",
                age=0,
            ),
        )
        return MonitorState(
            Zonotope(center, generators, metadata),
            step=0,
            payload=ThermostatPayload(previous_time=None),
        )

    def clone_state(self, state: MonitorState) -> MonitorState:
        return MonitorState(
            Zonotope(
                state.zonotope.center,
                state.zonotope.generators,
                state.zonotope.metadata,
            ),
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
                source="thermal_bias",
            ),
        )

    def step(
        self,
        state: MonitorState,
        measurement: ThermostatMeasurement,
    ) -> MonitorResult:
        payload = state.payload
        if not isinstance(payload, ThermostatPayload):
            raise TypeError("thermostat monitor state payload has the wrong type")

        if payload.previous_time is None:
            dt = 0.0
        else:
            dt = measurement.time - payload.previous_time
            if dt < 0:
                raise ValueError("thermostat measurements must have nondecreasing time")

        old = state.zonotope.age_generators()
        c = old.center
        old_g, old_metadata = _ensure_calibration_generator(old)
        existing_count = old_g.shape[1]

        input_coeffs = np.zeros(existing_count + 1, dtype=float)
        for index, meta in enumerate(old_metadata):
            if meta.kind == GeneratorKind.CALIBRATION and meta.source == "thermal_bias":
                input_coeffs[index] = self.calibration_error_scale
        input_coeffs[-1] = self.measurement_noise_scale

        previous_temperature = np.append(old_g[TEMPERATURE, :], 0.0)
        previous_filtered = np.append(old_g[FILTERED_TEMPERATURE, :], 0.0)

        leak_term = self.thermal_leak * (measurement.ambient_temperature - c[TEMPERATURE])
        hvac_term = self.hvac_gain * measurement.hvac_effort
        temperature_center = c[TEMPERATURE] + dt * (leak_term + hvac_term)
        filtered_center = (
            self.filter_gain * temperature_center
            + (1.0 - self.filter_gain) * c[FILTERED_TEMPERATURE]
        )
        effort_center = measurement.hvac_effort
        deviation_center = filtered_center - self.setpoint
        new_center = np.array(
            [
                temperature_center,
                filtered_center,
                effort_center,
                deviation_center,
            ],
            dtype=float,
        )

        temperature_g = previous_temperature + dt * (
            -self.thermal_leak * previous_temperature + input_coeffs
        )
        filtered_g = (
            self.filter_gain * temperature_g
            + (1.0 - self.filter_gain) * previous_filtered
        )
        effort_g = np.zeros(existing_count + 1, dtype=float)
        deviation_g = filtered_g
        new_generators = np.vstack(
            [temperature_g, filtered_g, effort_g, deviation_g]
        )
        new_metadata = (
            *old_metadata,
            GeneratorMetadata(
                GeneratorKind.MEASUREMENT,
                source=f"temperature_noise@{state.step + 1}",
                age=0,
            ),
        )
        new_state = MonitorState(
            Zonotope(new_center, new_generators, new_metadata),
            step=state.step + 1,
            payload=ThermostatPayload(previous_time=measurement.time),
        )
        return MonitorResult(new_state, evaluate_triggers(new_state.zonotope, self.triggers))


def generate_thermostat_trace(
    length: int,
    *,
    seed: int = 0,
    dt: float = 1.0,
    setpoint: float = 21.0,
) -> tuple[ThermostatMeasurement, ...]:
    """Generate a deterministic thermostat trace for one seed."""

    phase = 0.37 * seed
    temperature = setpoint
    effort = 0.0
    trace: list[ThermostatMeasurement] = []
    for index in range(length):
        ambient = (
            setpoint
            + 6.0 * np.sin(index / 18.0 + phase)
            + 2.0 * np.sin(index / 7.0 + 0.5 * phase)
        )
        if temperature < setpoint - 0.45:
            effort = 1.0
        elif temperature > setpoint + 0.45:
            effort = -1.0
        else:
            effort *= 0.5
        trace.append(
            ThermostatMeasurement(
                time=index * dt,
                ambient_temperature=float(ambient),
                hvac_effort=float(effort),
            )
        )
        temperature = temperature + dt * (
            0.08 * (float(ambient) - temperature) + 0.45 * effort
        )
    return tuple(trace)


def predict_thermostat_inputs(
    history: Sequence[ThermostatMeasurement],
    horizon: int,
) -> tuple[ThermostatMeasurement, ...]:
    """Constant-input short-horizon predictor for the thermostat benchmark."""

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
        ThermostatMeasurement(
            time=last.time + step * (offset + 1),
            ambient_temperature=last.ambient_temperature,
            hvac_effort=last.hvac_effort,
        )
        for offset in range(horizon)
    )


def _ensure_calibration_generator(
    zonotope: Zonotope,
) -> tuple[np.ndarray, tuple[GeneratorMetadata, ...]]:
    if any(
        meta.kind == GeneratorKind.CALIBRATION and meta.source == "thermal_bias"
        for meta in zonotope.metadata
    ):
        return zonotope.generators, zonotope.metadata
    generators = np.hstack(
        [zonotope.generators, np.zeros((zonotope.dimension, 1), dtype=float)]
    )
    metadata = (
        *zonotope.metadata,
        GeneratorMetadata(GeneratorKind.CALIBRATION, "thermal_bias", age=0),
    )
    return generators, metadata
