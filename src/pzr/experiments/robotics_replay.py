"""Replay evaluation and visualization for robotics candidate scenarios.

This module is intentionally separate from the default benchmark registry.  It
uses the regular reducer policies and runner, but keeps live sidecar collection
and paper-facing visualization in an explicit robotics-only path.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pzr.experiments.benchmark import TOP3_REDUCER_NAMES
from pzr.experiments.evaluation import aggregate_summary
from pzr.experiments.runner import (
    MPCReductionPolicy,
    StaticReductionPolicy,
    compute_ground_truth,
    results_to_dataframe,
    run_single,
    summarize_results,
)
from pzr.experiments.regret_eval import (
    REGRET_ORACLE_MODES,
    RegretOracleConfig,
    train_and_evaluate_regret_on_traces,
    write_regret_artifacts,
)
from pzr.experiments.robotics_probe import (
    DroneController,
    ProbeBundle,
    SafetyStreamMeasurement,
    SafetyStreamMonitor,
    TraceSource,
    _candidate_bundle,
    _load_level0_geometry,
    _trim_bundle,
    drone_stream_profile,
    f1tenth_stream_profile,
    trace_summary,
)
from pzr.monitoring.base import MonitorResult, MonitorState, TriggerSpec
from pzr.monitoring.triggers import evaluate_triggers
from pzr.mpc.objectives import CostWeights, WeightedZonotopeCost
from pzr.mpc.policies import (
    BeamMPCPolicy,
    MPCPolicy,
    PairRolloutMPCPolicy,
    RolloutMPCPolicy,
)
from pzr.utils.serialization import save_json
from pzr.zonotope.core import Zonotope
from pzr.zonotope.protected import ProtectedReducer
from pzr.zonotope.reduction import (
    BoxReducer,
    CombastelReducer,
    GirardReducer,
    MethAReducer,
    PcaReducer,
    ScottReducer,
)


STATIC_METHODS = ("girard", "combastel", "pca", "methA", "scott", "box")
FOCUSED_STATIC_METHODS = ("girard", "combastel", "methA", "scott", "box")
FOCUSED_MPC_METHODS = ("mpc_rollout_scott", "mpc_beam3", "mpc_sequence3")
FOCUSED_METHODS = FOCUSED_STATIC_METHODS + FOCUSED_MPC_METHODS
SWEEP_MPC_METHODS = ("mpc_beam3",)
SWEEP_METHODS = FOCUSED_STATIC_METHODS + SWEEP_MPC_METHODS
HEADLINE_MPC_METHODS = ("mpc_rollout", "mpc_pair_rollout3", "mpc_beam3", "mpc_sequence3")
HEADLINE_METHODS = STATIC_METHODS + HEADLINE_MPC_METHODS
PAPER_CORE_MPC_METHODS = ("mpc_rollout", "mpc_pair_rollout3", "mpc_beam3")
PAPER_CORE_METHODS = STATIC_METHODS + PAPER_CORE_MPC_METHODS
REPLAY_TRACE_SOURCES = ("procedural", "proxy", "live", "auto")
REPLAY_MONITORS = ("stream", "physical")
REPLAY_SCENARIO_FAMILIES = ("legacy", "stress")
ROBOTICS_MPC_CANDIDATE_NAMES = TOP3_REDUCER_NAMES


def _candidate_monitor_model(
    candidate: str,
    monitor: str,
    scenario_family: str,
) -> str:
    if monitor == "physical" and scenario_family == "stress":
        if candidate == "f1tenth":
            return "dynamics_physical_v3"
        if candidate == "drone":
            return "dynamics_physical_v2"
    return monitor


def _aggregate_monitor_model(
    candidates: Sequence[str],
    monitor: str,
    scenario_family: str,
) -> str:
    models = sorted({
        _candidate_monitor_model(candidate, monitor, scenario_family)
        for candidate in candidates
    })
    return models[0] if len(models) == 1 else ",".join(models)


@dataclass(frozen=True)
class ReplayProfile:
    """Small profile object shared by replay monitors and artifact writers."""

    stream_names: tuple[str, ...]
    dimension: int
    near_threshold: float = 0.25


@dataclass(frozen=True)
class F1TenthTrackGeometry:
    """Procedural centerline and track-width parameters for physical replay."""

    amp1: float
    freq1: float
    phase1: float
    amp2: float
    freq2: float
    phase2: float
    half_width: float
    width_wave: float
    width_freq: float
    width_phase: float
    bottleneck_x: float
    bottleneck_depth: float
    bottleneck_sigma: float
    front_phase: float

    def center_y(self, x: float) -> float:
        return float(
            self.amp1 * np.sin(self.freq1 * x + self.phase1)
            + self.amp2 * np.sin(self.freq2 * x + self.phase2)
        )

    def center_dy(self, x: float) -> float:
        return float(
            self.amp1 * self.freq1 * np.cos(self.freq1 * x + self.phase1)
            + self.amp2 * self.freq2 * np.cos(self.freq2 * x + self.phase2)
        )

    def center_ddy(self, x: float) -> float:
        return float(
            -self.amp1 * self.freq1**2 * np.sin(self.freq1 * x + self.phase1)
            - self.amp2 * self.freq2**2 * np.sin(self.freq2 * x + self.phase2)
        )

    def tangent(self, x: float) -> float:
        return float(np.arctan(self.center_dy(x)))

    def curvature(self, x: float) -> float:
        dy = self.center_dy(x)
        return float(self.center_ddy(x) / max((1.0 + dy * dy) ** 1.5, 1e-9))

    def width(self, x: float) -> float:
        bottleneck = self.bottleneck_depth * np.exp(
            -0.5 * ((x - self.bottleneck_x) / self.bottleneck_sigma) ** 2
        )
        wave = self.width_wave * np.sin(self.width_freq * x + self.width_phase)
        return float(max(0.38, self.half_width + wave - bottleneck))

    def width_dx(self, x: float) -> float:
        e = np.exp(-0.5 * ((x - self.bottleneck_x) / self.bottleneck_sigma) ** 2)
        bottleneck_dx = (
            self.bottleneck_depth * e * (x - self.bottleneck_x)
            / max(self.bottleneck_sigma**2, 1e-9)
        )
        wave_dx = self.width_wave * self.width_freq * np.cos(self.width_freq * x + self.width_phase)
        return float(wave_dx + bottleneck_dx)

    def front_clearance(self, x: float, lateral: float) -> float:
        upcoming = 1.55 + 0.55 * np.sin(0.55 * x + self.front_phase)
        bottleneck = 0.42 * np.exp(
            -0.5 * ((x - self.bottleneck_x) / max(self.bottleneck_sigma * 1.25, 1e-9)) ** 2
        )
        return float(max(0.25, upcoming - bottleneck - 0.22 * abs(lateral)))

    def front_dx(self, x: float) -> float:
        sine_dx = 0.55 * 0.55 * np.cos(0.55 * x + self.front_phase)
        sigma = max(self.bottleneck_sigma * 1.25, 1e-9)
        e = np.exp(-0.5 * ((x - self.bottleneck_x) / sigma) ** 2)
        bottleneck_dx = 0.42 * e * (x - self.bottleneck_x) / (sigma * sigma)
        return float(sine_dx + bottleneck_dx)

    def to_payload(self) -> dict[str, float]:
        return {
            "amp1": self.amp1,
            "freq1": self.freq1,
            "phase1": self.phase1,
            "amp2": self.amp2,
            "freq2": self.freq2,
            "phase2": self.phase2,
            "half_width": self.half_width,
            "width_wave": self.width_wave,
            "width_freq": self.width_freq,
            "width_phase": self.width_phase,
            "bottleneck_x": self.bottleneck_x,
            "bottleneck_depth": self.bottleneck_depth,
            "bottleneck_sigma": self.bottleneck_sigma,
            "front_phase": self.front_phase,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> F1TenthTrackGeometry:
        return cls(**{field: float(payload[field]) for field in cls.__dataclass_fields__})


@dataclass(frozen=True)
class DroneGateGeometry:
    """Procedural gate, obstacle, and corridor parameters for drone replay."""

    gates: tuple[tuple[float, float, float], ...]
    obstacles: tuple[tuple[float, float], ...]
    obstacle_radius: float
    corridor_radius: float
    gate_lateral_radius: float
    gate_vertical_radius: float
    altitude_floor: float
    altitude_ceiling: float
    speed_limit: float

    def gate(self, index: int) -> np.ndarray:
        idx = int(np.clip(index, 0, len(self.gates) - 1))
        return np.asarray(self.gates[idx], dtype=np.float64)

    def previous_gate(self, index: int) -> np.ndarray:
        if index <= 0:
            first = self.gate(0)
            return first + np.array([-0.9, -0.8, -0.15], dtype=np.float64)
        return self.gate(index - 1)

    def to_payload(self) -> dict[str, Any]:
        return {
            "gates": [[float(v) for v in gate] for gate in self.gates],
            "obstacles": [[float(v) for v in obstacle] for obstacle in self.obstacles],
            "obstacle_radius": self.obstacle_radius,
            "corridor_radius": self.corridor_radius,
            "gate_lateral_radius": self.gate_lateral_radius,
            "gate_vertical_radius": self.gate_vertical_radius,
            "altitude_floor": self.altitude_floor,
            "altitude_ceiling": self.altitude_ceiling,
            "speed_limit": self.speed_limit,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> DroneGateGeometry:
        return cls(
            gates=tuple(tuple(float(v) for v in gate[:3]) for gate in payload["gates"]),
            obstacles=tuple(tuple(float(v) for v in obstacle[:2]) for obstacle in payload["obstacles"]),
            obstacle_radius=float(payload["obstacle_radius"]),
            corridor_radius=float(payload["corridor_radius"]),
            gate_lateral_radius=float(payload["gate_lateral_radius"]),
            gate_vertical_radius=float(payload["gate_vertical_radius"]),
            altitude_floor=float(payload["altitude_floor"]),
            altitude_ceiling=float(payload["altitude_ceiling"]),
            speed_limit=float(payload["speed_limit"]),
        )


DRONE_PHYSICAL_STATE_NAMES = ("x", "y", "z", "vx", "vy", "vz")
DRONE_PHYSICAL_TRIGGER_NAMES = (
    "obstacle_clearance_margin",
    "gate_alignment_margin",
    "corridor_margin",
    "altitude_low_margin",
    "altitude_high_margin",
    "speed_margin",
)
F1TENTH_PHYSICAL_STATE_NAMES = ("x", "y", "theta", "speed", "yaw_rate")
F1TENTH_PHYSICAL_TRIGGER_NAMES = (
    "left_boundary_margin",
    "right_boundary_margin",
    "heading_margin",
    "time_to_collision_margin",
    "curvature_speed_margin",
    "yaw_rate_margin",
)


def _drone_physical_calibration_generators() -> np.ndarray:
    base = np.array([
        [0.05, 0.00],
        [0.00, 0.05],
        [0.02, 0.03],
        [0.02, -0.01],
        [-0.01, 0.02],
        [0.01, 0.01],
    ], dtype=np.float64)
    return 0.55 * base


def _drone_physical_fresh_generators() -> np.ndarray:
    base = np.array([
        [0.050, 0.000, 0.020, -0.010, 0.000, 0.030],
        [0.000, 0.050, -0.015, 0.020, 0.030, 0.000],
        [0.010, -0.010, 0.045, 0.000, 0.000, 0.025],
        [0.035, 0.012, 0.000, 0.040, -0.020, 0.000],
        [-0.012, 0.035, 0.000, -0.020, 0.040, 0.000],
        [0.000, 0.000, 0.022, 0.000, 0.000, 0.035],
    ], dtype=np.float64)
    return 0.18 * base


@dataclass(frozen=True)
class DronePhysicalMonitor:
    """Drone gate-flying monitor with physical state and projected margins."""

    calibration_generators: np.ndarray
    fresh_generators: np.ndarray
    remainder_scale: float = 1.0

    @property
    def profile(self) -> ReplayProfile:
        return ReplayProfile(
            stream_names=DRONE_PHYSICAL_TRIGGER_NAMES,
            dimension=len(DRONE_PHYSICAL_STATE_NAMES),
            near_threshold=0.25,
        )

    @property
    def state_names(self) -> tuple[str, ...]:
        return DRONE_PHYSICAL_STATE_NAMES

    @property
    def triggers(self) -> tuple[TriggerSpec, ...]:
        return tuple(
            TriggerSpec(f"{name}_violation", i, 0.0, "below", overlap=0.05)
            for i, name in enumerate(DRONE_PHYSICAL_TRIGGER_NAMES)
        )

    @property
    def num_calibration_generators(self) -> int:
        return int(self.calibration_generators.shape[1])

    def initial_state(self) -> MonitorState:
        return MonitorState(
            zonotope=Zonotope(np.zeros(len(DRONE_PHYSICAL_STATE_NAMES)), self.calibration_generators),
            step=0,
            calibration_indices=tuple(range(self.num_calibration_generators)),
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

    def step(self, state: MonitorState, measurement: SafetyStreamMeasurement) -> MonitorResult:
        values = np.asarray(measurement.values, dtype=np.float64)
        if values.shape != (len(DRONE_PHYSICAL_STATE_NAMES),):
            raise ValueError(
                f"expected {len(DRONE_PHYSICAL_STATE_NAMES)} physical values, got {values.size}"
            )

        geometry = _drone_geometry_from_payload(measurement.payload)
        gate_id = _drone_gate_id_from_payload(measurement.payload)
        transition = _drone_transition_matrix(state, measurement, geometry, gate_id)
        fresh_generators = self.fresh_generators_for(measurement)
        old_g = state.zonotope.generators
        n_existing = state.zonotope.generator_count
        n_fresh = fresh_generators.shape[1]
        generators = np.zeros((values.size, n_existing + n_fresh), dtype=np.float64)
        if n_existing:
            generators[:, :n_existing] = transition @ old_g
        generators[:, n_existing:] = fresh_generators
        margins, jacobian = _drone_physical_margin_and_jacobian(values, geometry, gate_id)
        remainder = _drone_projection_remainder(
            Zonotope(values, generators), geometry,
        ) * self.remainder_scale
        diagnostics = _dynamics_diagnostics(
            propagated=generators[:, :n_existing],
            fresh=fresh_generators,
            jacobian=jacobian,
            remainder=remainder,
            transition=transition,
        )
        payload = _measurement_payload_with_diagnostics(
            measurement.payload,
            diagnostics,
            model="dynamics_physical_v2",
        )
        payload_measurement = SafetyStreamMeasurement(
            time=measurement.time,
            values=measurement.values,
            true_values=measurement.true_values,
            oracle_violation=measurement.oracle_violation,
            payload=payload,
        )

        new_state = MonitorState(
            zonotope=Zonotope(values, generators),
            step=state.step + 1,
            calibration_indices=state.calibration_indices,
            payload=payload_measurement,
        )
        projected_g = jacobian @ generators
        trigger_z = Zonotope(margins, np.hstack([projected_g, np.diag(remainder)]))
        return MonitorResult(new_state, evaluate_triggers(trigger_z, self.triggers))

    def trigger_zonotope(self, state: MonitorState) -> Zonotope:
        geometry = _drone_geometry_from_payload(getattr(state.payload, "payload", None))
        gate_id = _drone_gate_id_from_payload(getattr(state.payload, "payload", None))
        center = state.zonotope.center
        margins, jacobian = _drone_physical_margin_and_jacobian(center, geometry, gate_id)
        projected_g = jacobian @ state.zonotope.generators
        remainder = _drone_projection_remainder(state.zonotope, geometry) * self.remainder_scale
        generators = np.hstack([projected_g, np.diag(remainder)])
        return Zonotope(margins, generators)

    def fresh_generators_for(self, measurement: SafetyStreamMeasurement) -> np.ndarray:
        values = np.asarray(measurement.values, dtype=np.float64)
        geometry = _drone_geometry_from_payload(measurement.payload)
        gate_id = _drone_gate_id_from_payload(measurement.payload)
        margins = _drone_physical_margins(values, geometry, gate_id)
        speed = float(np.linalg.norm(values[3:6]))
        risk = np.clip((0.35 - margins) / 0.55, 0.0, 1.8)
        speed_pressure = float(np.clip((speed - 0.65) / 0.55, 0.0, 1.8))
        phase = 0.5 + 0.5 * np.sin(0.9 * values[0] - 0.7 * values[1] + 0.4 * gate_id)
        scales = np.array([
            1.00 + 1.80 * risk[0] + 0.30 * phase,
            1.00 + 1.65 * risk[1] + 0.45 * (1.0 - phase),
            1.00 + 1.45 * risk[2] + 0.45 * phase,
            1.00 + 1.25 * risk[3] + 0.30 * (1.0 - phase),
            1.00 + 1.25 * risk[4] + 0.35 * phase,
            1.00 + 1.80 * risk[5] + 0.75 * speed_pressure,
        ], dtype=np.float64)
        return self.fresh_generators * scales[np.newaxis, :]


def make_drone_physical_monitor() -> DronePhysicalMonitor:
    return DronePhysicalMonitor(
        calibration_generators=_drone_physical_calibration_generators(),
        fresh_generators=_drone_physical_fresh_generators(),
    )


def _f1tenth_physical_calibration_generators() -> np.ndarray:
    base = np.array([
        [0.08, 0.00, 0.00],
        [0.00, 0.08, 0.00],
        [0.00, 0.00, 0.025],
        [0.02, 0.01, -0.02],
        [0.01, -0.01, 0.015],
    ], dtype=np.float64)
    return 0.55 * base


def _f1tenth_physical_fresh_generators() -> np.ndarray:
    base = np.array([
        [-0.12, 0.02, 0.06, 0.04, -0.16413973, -0.00052033],
        [0.02, 0.12, 0.06, 0.08, 0.02353809, 0.15756260],
        [0.01, 0.01, -0.06, -0.05, -0.07662582, 0.04407207],
        [0.00, 0.00, -0.05, -0.07, 0.02661324, -0.00770411],
        [0.00, 0.00, 0.00, -0.02, 0.02327616, -0.06354180],
    ], dtype=np.float64)
    return 0.14 * base


@dataclass(frozen=True)
class F1TenthPhysicalMonitor:
    """F1TENTH monitor with physical state and projected trigger margins."""

    calibration_generators: np.ndarray
    fresh_generators: np.ndarray
    remainder_scale: float = 1.0

    @property
    def profile(self) -> ReplayProfile:
        return ReplayProfile(
            stream_names=F1TENTH_PHYSICAL_TRIGGER_NAMES,
            dimension=len(F1TENTH_PHYSICAL_STATE_NAMES),
            near_threshold=0.25,
        )

    @property
    def state_names(self) -> tuple[str, ...]:
        return F1TENTH_PHYSICAL_STATE_NAMES

    @property
    def triggers(self) -> tuple[TriggerSpec, ...]:
        return tuple(
            TriggerSpec(f"{name}_violation", i, 0.0, "below", overlap=0.05)
            for i, name in enumerate(F1TENTH_PHYSICAL_TRIGGER_NAMES)
        )

    @property
    def num_calibration_generators(self) -> int:
        return int(self.calibration_generators.shape[1])

    def initial_state(self) -> MonitorState:
        return MonitorState(
            zonotope=Zonotope(np.zeros(len(F1TENTH_PHYSICAL_STATE_NAMES)), self.calibration_generators),
            step=0,
            calibration_indices=tuple(range(self.num_calibration_generators)),
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

    def step(self, state: MonitorState, measurement: SafetyStreamMeasurement) -> MonitorResult:
        values = np.asarray(measurement.values, dtype=np.float64)
        if values.shape != (len(F1TENTH_PHYSICAL_STATE_NAMES),):
            raise ValueError(
                f"expected {len(F1TENTH_PHYSICAL_STATE_NAMES)} physical values, got {values.size}"
            )

        geometry = _geometry_from_payload(measurement.payload)
        transition = _f1tenth_transition_matrix(state, measurement, geometry)
        fresh_generators = self.fresh_generators_for(measurement)
        old_g = state.zonotope.generators
        n_existing = state.zonotope.generator_count
        n_fresh = fresh_generators.shape[1]
        generators = np.zeros((values.size, n_existing + n_fresh), dtype=np.float64)
        effective_transition = _f1tenth_observer_contraction() @ transition
        if n_existing:
            generators[:, :n_existing] = effective_transition @ old_g
        generators[:, n_existing:] = fresh_generators
        margins, jacobian = _f1tenth_physical_margin_and_jacobian(values, geometry)
        remainder = _f1tenth_projection_remainder(
            Zonotope(values, generators), geometry,
        ) * self.remainder_scale
        diagnostics = _dynamics_diagnostics(
            propagated=generators[:, :n_existing],
            fresh=fresh_generators,
            jacobian=jacobian,
            remainder=remainder,
            transition=effective_transition,
        )
        payload = _measurement_payload_with_diagnostics(
            measurement.payload,
            diagnostics,
            model="dynamics_physical_v3",
        )
        payload_measurement = SafetyStreamMeasurement(
            time=measurement.time,
            values=measurement.values,
            true_values=measurement.true_values,
            oracle_violation=measurement.oracle_violation,
            payload=payload,
        )

        new_state = MonitorState(
            zonotope=Zonotope(values, generators),
            step=state.step + 1,
            calibration_indices=state.calibration_indices,
            payload=payload_measurement,
        )
        projected_g = jacobian @ generators
        generators_out = np.hstack([projected_g, np.diag(remainder)]) if remainder.size else projected_g
        trigger_z = Zonotope(margins, generators_out)
        return MonitorResult(new_state, evaluate_triggers(trigger_z, self.triggers))

    def trigger_zonotope(self, state: MonitorState) -> Zonotope:
        geometry = _geometry_from_payload(getattr(state.payload, "payload", None))
        center = state.zonotope.center
        margins, jacobian = _f1tenth_physical_margin_and_jacobian(center, geometry)
        projected_g = jacobian @ state.zonotope.generators
        remainder = _f1tenth_projection_remainder(state.zonotope, geometry) * self.remainder_scale
        if remainder.size:
            generators = np.hstack([projected_g, np.diag(remainder)])
        else:
            generators = projected_g
        return Zonotope(margins, generators)

    def fresh_generators_for(self, measurement: SafetyStreamMeasurement) -> np.ndarray:
        """Scale the fresh basis by local racing phase.

        F1TENTH sensing/model uncertainty is not uniform over a lap: boundary
        proximity, curvature, speed, and front-clearance phases emphasize
        different coupled state directions.  Keeping this deterministic and
        trace-local gives the replay a real reducer-selection problem while
        staying outside the quantitative benchmark defaults.
        """
        values = np.asarray(measurement.values, dtype=np.float64)
        geometry = _geometry_from_payload(measurement.payload)
        margins = _f1tenth_physical_margins(values, geometry)
        x, y, theta, speed, yaw_rate = values
        lateral = float(y - geometry.center_y(float(x)))
        tangent_error = abs(_wrap_angle(float(theta - geometry.tangent(float(x)))))
        curvature = abs(geometry.curvature(float(x)))
        boundary_margin = min(float(margins[0]), float(margins[1]))
        ttc_margin = float(margins[3])
        boundary_pressure = float(np.clip((0.56 - boundary_margin) / 0.56, 0.0, 1.8))
        curvature_pressure = float(np.clip(curvature / 0.30, 0.0, 1.8))
        speed_pressure = float(np.clip((abs(float(speed)) - 0.82) / 0.50, 0.0, 1.8))
        ttc_pressure = float(np.clip((0.45 - ttc_margin) / 0.80, 0.0, 1.8))
        heading_pressure = float(np.clip(tangent_error / 0.22, 0.0, 1.8))
        yaw_pressure = float(np.clip(abs(float(yaw_rate)) / 0.20, 0.0, 1.8))
        side = 1.0 if lateral >= 0.0 else -1.0
        phase_mix = 0.5 + 0.5 * np.sin(1.25 * float(x) + 0.7 * side)
        scales = np.array([
            1.00 + 1.90 * boundary_pressure + 0.35 * phase_mix,
            1.00 + 1.20 * boundary_pressure + 0.50 * (1.0 - phase_mix),
            1.00 + 1.85 * heading_pressure + 0.90 * curvature_pressure,
            1.00 + 1.90 * ttc_pressure + 0.65 * speed_pressure,
            1.00 + 2.15 * curvature_pressure + 0.55 * phase_mix,
            1.00 + 1.25 * yaw_pressure + 0.80 * (1.0 - phase_mix),
        ], dtype=np.float64)
        return self.fresh_generators * scales[np.newaxis, :]


def make_f1tenth_physical_monitor() -> F1TenthPhysicalMonitor:
    return F1TenthPhysicalMonitor(
        calibration_generators=_f1tenth_physical_calibration_generators(),
        fresh_generators=_f1tenth_physical_fresh_generators(),
    )


def _effective_monitor_dt(state: MonitorState, measurement: SafetyStreamMeasurement) -> float:
    previous = state.payload
    if isinstance(previous, SafetyStreamMeasurement):
        raw_dt = float(measurement.time - previous.time)
    else:
        raw_dt = 0.0
    if raw_dt <= 0.0:
        return 0.0
    return float(np.clip(raw_dt, 0.15, 0.35))


def _drone_transition_matrix(
    state: MonitorState,
    measurement: SafetyStreamMeasurement,
    geometry: DroneGateGeometry,
    gate_id: int,
) -> np.ndarray:
    values = np.asarray(measurement.values, dtype=np.float64)
    dt = _effective_monitor_dt(state, measurement)
    transition = np.eye(len(DRONE_PHYSICAL_STATE_NAMES), dtype=np.float64)
    if dt <= 0.0:
        return transition

    transition[0, 3] = 0.42 * dt
    transition[1, 4] = 0.42 * dt
    transition[2, 5] = 0.32 * dt
    transition[3, 3] = 0.72
    transition[4, 4] = 0.72
    transition[5, 5] = 0.66

    gate = geometry.gate(gate_id)
    prev_gate = geometry.previous_gate(gate_id)
    segment_xy = gate[:2] - prev_gate[:2]
    segment_norm = max(float(np.linalg.norm(segment_xy)), 1e-9)
    tangent = segment_xy / segment_norm
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    rel_xy = values[:2] - prev_gate[:2]
    tau = float(np.clip(np.dot(rel_xy, segment_xy) / max(segment_norm**2, 1e-9), 0.0, 1.0))
    gate_pressure = float(np.exp(-0.5 * ((tau - 0.78) / 0.20) ** 2))

    obstacle_pressure = 0.0
    obstacle_dir = normal
    obstacles = np.asarray(geometry.obstacles, dtype=np.float64)
    if obstacles.size:
        deltas = values[:2] - obstacles[:, :2]
        distances = np.linalg.norm(deltas, axis=1)
        nearest = int(np.argmin(distances))
        dist = max(float(distances[nearest]), 1e-9)
        obstacle_dir = deltas[nearest] / dist
        clearance = dist - geometry.obstacle_radius
        obstacle_pressure = float(np.clip((0.48 - clearance) / 0.70, 0.0, 1.0))

    lateral_control = -0.045 * gate_pressure * np.outer(normal, normal)
    obstacle_control = 0.035 * obstacle_pressure * np.outer(obstacle_dir, obstacle_dir)
    transition[3:5, 0:2] += dt * (lateral_control + obstacle_control)
    transition[0:2, 0:2] += 0.020 * gate_pressure * np.outer(normal, tangent)

    low_pressure = float(np.clip((geometry.altitude_floor + 0.24 - values[2]) / 0.42, 0.0, 1.0))
    high_pressure = float(np.clip((values[2] - geometry.altitude_ceiling + 0.24) / 0.42, 0.0, 1.0))
    transition[5, 2] += dt * (0.035 * low_pressure - 0.035 * high_pressure)
    speed = max(float(np.linalg.norm(values[3:6])), 1e-9)
    speed_pressure = float(np.clip((speed - 0.72) / 0.50, 0.0, 1.0))
    vel_dir = values[3:6] / speed
    transition[3:6, 3:6] += -0.030 * speed_pressure * np.outer(vel_dir, vel_dir)
    return transition


def _f1tenth_transition_matrix(
    state: MonitorState,
    measurement: SafetyStreamMeasurement,
    geometry: F1TenthTrackGeometry,
) -> np.ndarray:
    values = np.asarray(measurement.values, dtype=np.float64)
    dt = _effective_monitor_dt(state, measurement)
    transition = np.eye(len(F1TENTH_PHYSICAL_STATE_NAMES), dtype=np.float64)
    if dt <= 0.0:
        return transition

    x, y, theta, speed, yaw_rate = values
    tangent = geometry.tangent(float(x))
    lateral = float(y - geometry.center_y(float(x)))
    heading_error = _wrap_angle(float(theta - tangent))
    curvature = geometry.curvature(float(x))
    curvature_abs = abs(curvature)
    curvature_pressure = float(np.clip(curvature_abs / 0.30, 0.0, 1.0))
    bottleneck_pressure = float(np.exp(
        -0.5 * ((float(x) - geometry.bottleneck_x) / max(geometry.bottleneck_sigma, 1e-9)) ** 2
    ))

    transition[0, 2] = -float(speed) * np.sin(theta) * dt
    transition[0, 3] = np.cos(theta) * dt
    transition[1, 2] = float(speed) * np.cos(theta) * dt
    transition[1, 3] = np.sin(theta) * dt
    transition[2, 4] = dt
    transition[3, 3] = 0.78
    transition[4, 4] = 0.66

    dy = geometry.center_dy(float(x))
    ddy = geometry.center_ddy(float(x))
    tangent_dx = ddy / max(1.0 + dy * dy, 1e-9)
    transition[1, 0] += dy * 0.045 * dt
    transition[2, 0] += tangent_dx * (0.12 + 0.20 * curvature_pressure) * dt
    transition[2, 1] += -0.035 * np.sign(lateral if lateral != 0.0 else heading_error) * dt
    transition[3, 0] += -0.035 * bottleneck_pressure * geometry.width_dx(float(x)) * dt
    transition[4, 0] += 0.070 * curvature * dt
    transition[4, 1] += -0.030 * np.sign(lateral) * (0.3 + curvature_pressure) * dt
    transition[4, 2] += -0.025 * np.sign(heading_error) * dt
    return transition


def _f1tenth_observer_contraction() -> np.ndarray:
    """Measurement-update contraction for F1TENTH physical replay uncertainty."""
    return np.diag([0.72, 0.68, 0.62, 0.55, 0.50]).astype(np.float64)


def _dynamics_diagnostics(
    *,
    propagated: np.ndarray,
    fresh: np.ndarray,
    jacobian: np.ndarray,
    remainder: np.ndarray,
    transition: np.ndarray,
) -> dict[str, float]:
    projected_propagated = jacobian @ propagated if propagated.size else np.zeros((jacobian.shape[0], 0))
    projected_fresh = jacobian @ fresh if fresh.size else np.zeros((jacobian.shape[0], 0))
    propagated_radius = float(np.sum(np.abs(projected_propagated)))
    fresh_radius = float(np.sum(np.abs(projected_fresh)))
    remainder_radius = float(np.sum(np.abs(remainder)))
    total = max(propagated_radius + fresh_radius + remainder_radius, 1e-12)
    return {
        "propagated_trigger_radius": propagated_radius,
        "fresh_trigger_radius": fresh_radius,
        "projection_remainder_radius": remainder_radius,
        "propagated_width_fraction": propagated_radius / total,
        "fresh_width_fraction": fresh_radius / total,
        "projection_remainder_fraction": remainder_radius / total,
        "transition_variation_score": float(np.linalg.norm(transition - np.eye(transition.shape[0]), ord="fro")),
    }


def _measurement_payload_with_diagnostics(
    payload: dict[str, Any] | None,
    diagnostics: dict[str, float],
    *,
    model: str,
) -> dict[str, Any]:
    base = dict(payload or {})
    base["monitor_model"] = model
    base["monitor_diagnostics"] = diagnostics
    return base


@dataclass(frozen=True)
class F1TenthPhysicalReplayCost:
    """Robotics-only MPC cost guard for long F1TENTH physical replays."""

    base: WeightedZonotopeCost
    remainder_weight: float = 0.04
    state_radius_weight: float = 0.015

    def __call__(
        self,
        state: MonitorState,
        verdicts: tuple[Any, ...] | None = None,
    ) -> float:
        payload = getattr(state.payload, "payload", None)
        geometry = _geometry_from_payload(payload)
        remainder = _f1tenth_projection_remainder(state.zonotope, geometry)
        radius = state.zonotope.interval_radius()
        return float(
            self.base(state, verdicts)
            + self.remainder_weight * float(np.sum(remainder))
            + self.state_radius_weight * float(np.sum(radius))
        )


@dataclass(frozen=True)
class SafetyStreamTrendPredictor:
    """History-only linear predictor for safety-stream measurements."""

    max_slope: float = 0.08

    def predict(
        self,
        history: Sequence[SafetyStreamMeasurement],
        horizon: int,
    ) -> tuple[SafetyStreamMeasurement, ...]:
        if not history or horizon <= 0:
            return ()
        last = history[-1]
        if len(history) >= 2:
            prev = history[-2]
            dt = max(float(last.time - prev.time), 1e-6)
            slope = (
                np.asarray(last.values, dtype=np.float64)
                - np.asarray(prev.values, dtype=np.float64)
            ) / dt
            slope = np.clip(slope, -self.max_slope, self.max_slope)
        else:
            dt = 1.0
            slope = np.zeros(len(last.values), dtype=np.float64)

        base = np.asarray(last.values, dtype=np.float64)
        predicted: list[SafetyStreamMeasurement] = []
        for i in range(horizon):
            values = base + slope * dt * float(i + 1)
            predicted.append(SafetyStreamMeasurement(
                time=float(last.time + dt * float(i + 1)),
                values=tuple(float(v) for v in values),
                true_values=tuple(float(v) for v in values),
                oracle_violation=bool(np.any(values < 0.0)),
                payload={
                    **(last.payload or {}),
                    "trace_source": "history_trend_prediction",
                },
            ))
        return tuple(predicted)


@dataclass(frozen=True)
class DronePhysicalPredictor:
    """Compact kinematic predictor for drone physical replay measurements."""

    def predict(
        self,
        history: Sequence[SafetyStreamMeasurement],
        horizon: int,
    ) -> tuple[SafetyStreamMeasurement, ...]:
        if not history or horizon <= 0:
            return ()
        last = history[-1]
        values = np.asarray(last.values, dtype=np.float64).copy()
        geometry = _drone_geometry_from_payload(last.payload)
        gate_id = _drone_gate_id_from_payload(last.payload)
        dt = _history_dt(history)
        predicted: list[SafetyStreamMeasurement] = []
        for i in range(horizon):
            values[:3] = values[:3] + values[3:6] * dt
            gate_id = _predicted_drone_gate_id(values, geometry, gate_id)
            true_values = _drone_physical_margins(values, geometry, gate_id)
            predicted.append(SafetyStreamMeasurement(
                time=float(last.time + dt * float(i + 1)),
                values=tuple(float(v) for v in values),
                true_values=tuple(float(v) for v in true_values),
                oracle_violation=bool(np.any(true_values < 0.0)),
                payload=_predicted_drone_payload(last.payload, geometry, gate_id),
            ))
        return tuple(predicted)


@dataclass(frozen=True)
class F1TenthPhysicalPredictor:
    """Compact kinematic predictor for F1TENTH physical replay measurements."""

    def predict(
        self,
        history: Sequence[SafetyStreamMeasurement],
        horizon: int,
    ) -> tuple[SafetyStreamMeasurement, ...]:
        if not history or horizon <= 0:
            return ()
        last = history[-1]
        values = np.asarray(last.values, dtype=np.float64).copy()
        geometry = _geometry_from_payload(last.payload)
        dt = _history_dt(history)
        predicted: list[SafetyStreamMeasurement] = []
        for i in range(horizon):
            x, y, theta, speed, yaw_rate = values
            tangent = geometry.tangent(float(x))
            heading_error = _wrap_angle(float(theta - tangent))
            values[0] = x + speed * np.cos(theta) * dt
            values[1] = y + speed * np.sin(theta) * dt
            values[2] = _wrap_angle(theta + yaw_rate * dt - 0.12 * heading_error * dt)
            values[3] = 0.96 * speed
            values[4] = 0.82 * yaw_rate + 0.22 * geometry.curvature(float(values[0])) * speed
            true_values = _f1tenth_physical_margins(values, geometry)
            predicted.append(SafetyStreamMeasurement(
                time=float(last.time + dt * float(i + 1)),
                values=tuple(float(v) for v in values),
                true_values=tuple(float(v) for v in true_values),
                oracle_violation=bool(np.any(true_values < 0.0)),
                payload=_predicted_f1tenth_payload(last.payload, geometry, values),
            ))
        return tuple(predicted)


def _history_dt(history: Sequence[SafetyStreamMeasurement]) -> float:
    if len(history) >= 2:
        raw_dt = float(history[-1].time - history[-2].time)
    else:
        raw_dt = 1.0
    if raw_dt <= 0.0:
        raw_dt = 1.0
    return float(np.clip(raw_dt, 0.15, 0.35))


def _predicted_drone_gate_id(
    values: np.ndarray,
    geometry: DroneGateGeometry,
    gate_id: int,
) -> int:
    gate = geometry.gate(gate_id)
    previous = geometry.previous_gate(gate_id)
    segment = gate[:2] - previous[:2]
    rel = values[:2] - previous[:2]
    tau = float(np.dot(rel, segment) / max(float(np.dot(segment, segment)), 1e-9))
    if tau > 0.92 and gate_id + 1 < len(geometry.gates):
        return gate_id + 1
    return gate_id


def _predicted_drone_payload(
    payload: dict[str, Any] | None,
    geometry: DroneGateGeometry,
    gate_id: int,
) -> dict[str, Any]:
    base = dict(payload or {})
    raw = dict(base.get("raw_record", {}) if isinstance(base.get("raw_record"), dict) else {})
    info = dict(raw.get("info", {}) if isinstance(raw.get("info"), dict) else {})
    info["current_target_gate_id"] = int(gate_id)
    raw["info"] = info
    raw["geometry"] = geometry.to_payload()
    base["geometry"] = geometry.to_payload()
    base["gate_id"] = int(gate_id)
    base["raw_record"] = raw
    base["trace_source"] = "physical_kinematic_prediction"
    return base


def _predicted_f1tenth_payload(
    payload: dict[str, Any] | None,
    geometry: F1TenthTrackGeometry,
    values: np.ndarray,
) -> dict[str, Any]:
    base = dict(payload or {})
    raw = dict(base.get("raw_record", {}) if isinstance(base.get("raw_record"), dict) else {})
    raw["obs"] = {
        "poses_x": [float(values[0])],
        "poses_y": [float(values[1])],
        "poses_theta": [float(values[2])],
        "linear_vels_x": [float(values[3])],
        "ang_vels_z": [float(values[4])],
    }
    raw["geometry"] = geometry.to_payload()
    base["geometry"] = geometry.to_payload()
    base["raw_record"] = raw
    base["trace_source"] = "physical_kinematic_prediction"
    return base


def run_replay_eval(
    *,
    candidates: Sequence[str],
    length: int,
    seed: int,
    seeds: int,
    warmup_steps: int,
    budget: int,
    horizon: int,
    beam_width: int,
    output: Path,
    trace_source: str = "procedural",
    monitor: str = "physical",
    scenario_family: str = "stress",
    method_set: str = "focused",
    drone_controller: DroneController = "sim",
    drone_sidecar_python: Path | None = None,
    f1tenth_sidecar_python: Path | None = None,
    f1tenth_map: str = "vegas",
    cached_bundles: dict[tuple[str, int], tuple[ProbeBundle | None, dict[str, Any]]] | None = None,
    learned_mode: str = "none",
    regret_oracle: str = "beam3",
    regret_iterations: int = 3,
    regret_epochs: int = 100,
    regret_train_seeds: int | None = None,
    regret_eval_seeds: int | None = None,
    regret_loss: str = "pairwise",
) -> dict[str, Any]:
    """Run focused static/MPC replay evaluation for robotics candidates."""
    eval_t0 = time.perf_counter()
    print(
        "robotics_replay eval: "
        f"budget={budget} horizon={horizon} length={length} seeds={seeds} "
        f"method_set={method_set} learned={learned_mode}",
        flush=True,
    )
    if trace_source not in REPLAY_TRACE_SOURCES:
        raise ValueError(f"unknown trace_source {trace_source!r}")
    if monitor not in REPLAY_MONITORS:
        raise ValueError(f"unknown monitor {monitor!r}")
    if scenario_family not in REPLAY_SCENARIO_FAMILIES:
        raise ValueError(f"unknown scenario_family {scenario_family!r}")
    if seeds < 1:
        raise ValueError("seeds must be at least 1")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if learned_mode not in {"none", "regret"}:
        raise ValueError("learned_mode must be none|regret")
    if regret_oracle not in REGRET_ORACLE_MODES:
        raise ValueError(f"unknown regret_oracle {regret_oracle!r}")
    output.mkdir(parents=True, exist_ok=True)

    requested = ("drone", "f1tenth") if "all" in candidates else tuple(candidates)
    seed_values = tuple(range(seed, seed + seeds))
    summaries: list[pd.DataFrame] = []
    timeseries_rows: list[pd.DataFrame] = []
    trace_summaries: list[pd.DataFrame] = []
    scenario_summaries: list[dict[str, Any]] = []
    intervention_summaries: list[pd.DataFrame] = []
    metadata: dict[str, Any] = {
        "kind": "robotics_replay_eval",
        "length": length,
        "seed": seed,
        "seeds": seeds,
        "seed_values": list(seed_values),
        "warmup_steps": warmup_steps,
        "budget": budget,
        "horizon": horizon,
        "beam_width": beam_width,
        "trace_source": trace_source,
        "monitor": monitor,
        "scenario_family": scenario_family,
        "monitor_model": _aggregate_monitor_model(requested, monitor, scenario_family),
        "method_set": method_set,
        "methods": list(_method_names(method_set)),
        "mpc_candidate_reducers": list(ROBOTICS_MPC_CANDIDATE_NAMES),
        "learned_mode": learned_mode,
        "regret_oracle": regret_oracle if learned_mode == "regret" else "",
        "regret_loss": regret_loss if learned_mode == "regret" else "",
        "requested_candidates": list(requested),
        "candidates": {},
    }
    if learned_mode == "regret":
        metadata["methods"].append(f"learned_regret_{regret_oracle}")

    bundles_by_candidate: dict[str, dict[int, ProbeBundle]] = {
        name: {} for name in requested
    }

    for current_seed in seed_values:
        seed_output = output / f"seed_{current_seed}"
        seed_output.mkdir(parents=True, exist_ok=True)
        for candidate in requested:
            print(
                f"  eval seed={current_seed} candidate={candidate}",
                flush=True,
            )
            cached = (cached_bundles or {}).get((candidate, current_seed))
            if cached is None:
                bundle, candidate_metadata = _load_replay_bundle(
                    candidate=candidate,
                    length=length,
                    seed=current_seed,
                    trace_source=trace_source,
                    monitor=monitor,
                    scenario_family=scenario_family,
                    output=seed_output,
                    drone_controller=drone_controller,
                    drone_sidecar_python=drone_sidecar_python,
                    f1tenth_sidecar_python=f1tenth_sidecar_python,
                    f1tenth_map=f1tenth_map,
                )
            else:
                bundle, candidate_metadata = cached
            candidate_metadata = {**candidate_metadata, "seed": current_seed}
            aggregate_metadata = metadata["candidates"].setdefault(
                candidate,
                {"candidate": candidate, "per_seed": []},
            )
            aggregate_metadata["per_seed"].append(candidate_metadata)
            if candidate_metadata.get("status", "available") == "available":
                aggregate_metadata["status"] = "available"
            else:
                aggregate_metadata.setdefault(
                    "status", candidate_metadata.get("status", "unavailable"),
                )
            if bundle is None:
                if trace_source == "live":
                    reason = candidate_metadata.get("reason", "live trace is unavailable")
                    raise RuntimeError(
                        f"live trace unavailable for {candidate} seed {current_seed}: {reason}"
                    )
                continue

            bundle = _trim_bundle(bundle, warmup_steps)
            if not bundle.trace:
                if trace_source == "live":
                    raise RuntimeError(
                        f"live trace for {candidate} seed {current_seed} is empty "
                        "after warmup trimming"
                    )
                aggregate_metadata["per_seed"][-1] = {
                    **bundle.metadata,
                    "status": "unavailable",
                    "reason": "trace is empty after warmup trimming",
                    "seed": current_seed,
                }
                continue
            bundles_by_candidate[candidate][current_seed] = bundle

            result = _run_replay_bundle(
                bundle,
                budget=budget,
                seed=current_seed,
                horizon=horizon,
                beam_width=beam_width,
                method_set=method_set,
            )
            ts = result["timeseries"].copy()
            ts.insert(0, "candidate", candidate)
            summary = result["summary"].copy()
            summary.insert(0, "candidate", candidate)
            summaries.append(summary)
            timeseries_rows.append(ts)
            trace_summaries.append(trace_summary(bundle, seed=current_seed))
            scenario_summaries.append(_scenario_summary(bundle, seed=current_seed))
            intervention_summaries.append(_intervention_summary(
                candidate=candidate,
                seed=current_seed,
                bundle=bundle,
                timeseries=ts,
            ))
            ts.to_csv(seed_output / f"{candidate}_timeseries.csv", index=False)
            summary.to_csv(seed_output / f"{candidate}_summary.csv", index=False)
            _write_replay_trace_csv(bundle, seed_output / f"{candidate}_derived_streams.csv")
            _write_payload_jsonl(bundle, seed_output / f"{candidate}_payload.jsonl")

    if learned_mode == "regret":
        print(
            "  training regret/ranking selector "
            f"oracle={regret_oracle} iterations={regret_iterations} "
            f"epochs={regret_epochs} loss={regret_loss}",
            flush=True,
        )
        learned = _run_replay_regret_learning(
            bundles_by_candidate=bundles_by_candidate,
            budget=budget,
            horizon=horizon,
            beam_width=beam_width,
            regret_oracle=regret_oracle,
            regret_iterations=regret_iterations,
            regret_epochs=regret_epochs,
            regret_train_seeds=regret_train_seeds or seeds,
            regret_eval_seeds=regret_eval_seeds or seeds,
            regret_loss=regret_loss,
            output=output,
        )
        summaries.extend(learned["summaries"])
        timeseries_rows.extend(learned["timeseries_rows"])
        intervention_summaries.extend(learned["intervention_summaries"])
        metadata["learning"] = learned["metadata"]

    timeseries = (
        pd.concat(timeseries_rows, ignore_index=True)
        if timeseries_rows else pd.DataFrame()
    )
    summary = (
        pd.concat(summaries, ignore_index=True)
        if summaries else pd.DataFrame()
    )
    aggregate = _aggregate_replay_summary(summary)
    trace_scores = (
        pd.concat(trace_summaries, ignore_index=True)
        if trace_summaries else pd.DataFrame()
    )
    scenario_summary = pd.DataFrame(scenario_summaries)
    intervention_summary = (
        pd.concat(intervention_summaries, ignore_index=True)
        if intervention_summaries else pd.DataFrame()
    )
    policy_gain = _policy_gain(summary)
    winner_by_step = _winner_by_step(timeseries)

    timeseries.to_csv(output / "timeseries.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    aggregate.to_csv(output / "aggregate.csv", index=False)
    trace_scores.to_csv(output / "trace_summary.csv", index=False)
    scenario_summary.to_csv(output / "scenario_summary.csv", index=False)
    intervention_summary.to_csv(output / "intervention_summary.csv", index=False)
    policy_gain.to_csv(output / "policy_gain.csv", index=False)
    winner_by_step.to_csv(output / "winner_by_step.csv", index=False)
    save_json(metadata, output / "trace_metadata.json")
    _write_eval_report(
        output / "replay_report.md",
        metadata=metadata,
        aggregate=aggregate,
        policy_gain=policy_gain,
    )
    print(
        f"robotics_replay eval complete: budget={budget} "
        f"elapsed={time.perf_counter() - eval_t0:.1f}s output={output}",
        flush=True,
    )
    return {
        "metadata": metadata,
        "timeseries": timeseries,
        "summary": summary,
        "aggregate": aggregate,
        "trace_summary": trace_scores,
        "scenario_summary": scenario_summary,
        "intervention_summary": intervention_summary,
        "policy_gain": policy_gain,
        "winner_by_step": winner_by_step,
    }


def _run_replay_regret_learning(
    *,
    bundles_by_candidate: dict[str, dict[int, ProbeBundle]],
    budget: int,
    horizon: int,
    beam_width: int,
    regret_oracle: str,
    regret_iterations: int,
    regret_epochs: int,
    regret_train_seeds: int,
    regret_eval_seeds: int,
    regret_loss: str,
    output: Path,
) -> dict[str, Any]:
    summaries: list[pd.DataFrame] = []
    timeseries_rows: list[pd.DataFrame] = []
    intervention_summaries: list[pd.DataFrame] = []
    metadata: dict[str, Any] = {
        "regret_oracle": regret_oracle,
        "regret_iterations": regret_iterations,
        "regret_epochs": regret_epochs,
        "regret_train_seeds": regret_train_seeds,
        "regret_eval_seeds": regret_eval_seeds,
        "regret_loss": regret_loss,
        "candidates": {},
    }

    for candidate, by_seed in bundles_by_candidate.items():
        available_seeds = tuple(sorted(by_seed))
        if not available_seeds:
            metadata["candidates"][candidate] = {
                "status": "skipped",
                "reason": "no available traces",
            }
            continue
        if regret_train_seeds > len(available_seeds) or regret_eval_seeds > len(available_seeds):
            raise ValueError(
                "robotics regret seed counts must be <= replay --seeds because "
                "learning reuses the cached replay traces"
            )
        train_seed_values = available_seeds[:regret_train_seeds]
        eval_seed_values = available_seeds[:regret_eval_seeds]
        train_traces = tuple(
            (seed, by_seed[seed].trace) for seed in train_seed_values
        )
        eval_traces = tuple(
            (seed, by_seed[seed].trace) for seed in eval_seed_values
        )
        monitor = by_seed[available_seeds[0]].monitor
        oracle_config = RegretOracleConfig(
            mode=regret_oracle,  # type: ignore[arg-type]
            horizon=horizon,
            beam_width=beam_width,
            predictor=_replay_predictor(monitor),
        )
        result = train_and_evaluate_regret_on_traces(
            monitor=monitor,
            train_traces=train_traces,
            eval_traces=eval_traces,
            budget=budget,
            oracle_config=oracle_config,
            iterations=regret_iterations,
            epochs_per_iteration=regret_epochs,
            regret_loss=regret_loss,
            show_progress=False,
        )
        ts = results_to_dataframe(result.eval_results)
        ts.insert(0, "candidate", candidate)
        summary = summarize_results(result.eval_results)
        summary.insert(0, "candidate", candidate)
        timeseries_rows.append(ts)
        summaries.append(summary)
        for run in result.eval_results:
            bundle = by_seed[run.seed]
            run_ts = results_to_dataframe([run])
            intervention_summaries.append(_intervention_summary(
                candidate=candidate,
                seed=run.seed,
                bundle=bundle,
                timeseries=run_ts,
            ))

        artifact_dir = output / "learning" / candidate
        write_regret_artifacts(
            result,
            artifact_dir,
            metadata={
                "candidate": candidate,
                "budget": budget,
                "horizon": horizon,
                "beam_width": beam_width,
                "train_seeds": list(train_seed_values),
                "eval_seeds": list(eval_seed_values),
                "candidate_names": list(oracle_config.candidate_names),
                "regret_loss": regret_loss,
                "predictor": type(_replay_predictor(monitor)).__name__,
            },
        )
        metadata["candidates"][candidate] = {
            "status": "available",
            "total_traces": result.total_traces,
            "eval_runs": len(result.eval_results),
            "artifact_dir": str(artifact_dir),
        }

    return {
        "summaries": summaries,
        "timeseries_rows": timeseries_rows,
        "intervention_summaries": intervention_summaries,
        "metadata": metadata,
    }


def run_replay_budget_sweep(
    *,
    candidate: str,
    budgets: Sequence[int],
    length: int,
    seed: int,
    seeds: int,
    warmup_steps: int,
    horizon: int,
    beam_width: int,
    output: Path,
    trace_source: str = "procedural",
    monitor: str = "physical",
    scenario_family: str = "stress",
    drone_controller: DroneController = "sim",
    drone_sidecar_python: Path | None = None,
    f1tenth_sidecar_python: Path | None = None,
    f1tenth_map: str = "vegas",
    method_set: str = "sweep",
    render_selected: bool = True,
    learned_mode: str = "none",
    regret_oracle: str = "beam3",
    regret_iterations: int = 3,
    regret_epochs: int = 100,
    regret_train_seeds: int | None = None,
    regret_eval_seeds: int | None = None,
    regret_loss: str = "pairwise",
) -> dict[str, Any]:
    """Run replay evaluation across generator budgets and aggregate paper metrics."""
    sweep_t0 = time.perf_counter()
    requested = ("drone", "f1tenth") if candidate == "all" else (candidate,)
    if any(name not in {"drone", "f1tenth"} for name in requested):
        raise ValueError("robotics replay budget sweep supports candidate drone|f1tenth|all")
    if scenario_family not in REPLAY_SCENARIO_FAMILIES:
        raise ValueError(f"unknown scenario_family {scenario_family!r}")
    clean_budgets = tuple(int(b) for b in budgets)
    if not clean_budgets:
        raise ValueError("at least one budget is required")
    if any(b < 1 for b in clean_budgets):
        raise ValueError("budgets must be positive")
    if len(set(clean_budgets)) != len(clean_budgets):
        raise ValueError("budgets must be unique")

    output.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "kind": "robotics_replay_budget_sweep",
        "candidate": candidate,
        "requested_candidates": list(requested),
        "budgets": list(clean_budgets),
        "length": length,
        "seed": seed,
        "seeds": seeds,
        "seed_values": list(range(seed, seed + seeds)),
        "warmup_steps": warmup_steps,
        "horizon": horizon,
        "beam_width": beam_width,
        "trace_source": trace_source,
        "monitor": monitor,
        "scenario_family": scenario_family,
        "monitor_model": _aggregate_monitor_model(requested, monitor, scenario_family),
        "method_set": method_set,
        "methods": list(_method_names(method_set)),
        "mpc_candidate_reducers": list(ROBOTICS_MPC_CANDIDATE_NAMES),
        "learned_mode": learned_mode,
        "regret_oracle": regret_oracle if learned_mode == "regret" else "",
        "regret_iterations": regret_iterations if learned_mode == "regret" else 0,
        "regret_epochs": regret_epochs if learned_mode == "regret" else 0,
        "regret_train_seeds": (
            regret_train_seeds if regret_train_seeds is not None else seeds
        ) if learned_mode == "regret" else 0,
        "regret_eval_seeds": (
            regret_eval_seeds if regret_eval_seeds is not None else seeds
        ) if learned_mode == "regret" else 0,
        "regret_loss": regret_loss if learned_mode == "regret" else "",
        "render_selected": render_selected,
        "budget_dirs": {},
    }
    if learned_mode == "regret":
        metadata["methods"].append(f"learned_regret_{regret_oracle}")

    cached_bundles: dict[tuple[str, int], tuple[ProbeBundle | None, dict[str, Any]]] = {}
    cache_dir = output / "trace_cache"
    print(
        "robotics_replay sweep: "
        f"candidate={candidate} budgets={list(clean_budgets)} horizon={horizon} "
        f"length={length} seeds={seeds} method_set={method_set} "
        f"learned={learned_mode}",
        flush=True,
    )
    for current_seed in range(seed, seed + seeds):
        seed_cache_dir = cache_dir / f"seed_{current_seed}"
        seed_cache_dir.mkdir(parents=True, exist_ok=True)
        for requested_candidate in requested:
            print(
                f"  cache trace seed={current_seed} candidate={requested_candidate}",
                flush=True,
            )
            cached = _load_replay_bundle(
                candidate=requested_candidate,
                length=length,
                seed=current_seed,
                trace_source=trace_source,
                monitor=monitor,
                scenario_family=scenario_family,
                output=seed_cache_dir,
                drone_controller=drone_controller,
                drone_sidecar_python=drone_sidecar_python,
                f1tenth_sidecar_python=f1tenth_sidecar_python,
                f1tenth_map=f1tenth_map,
            )
            if trace_source == "live" and (cached[0] is None or not cached[0].trace):
                reason = cached[1].get("reason", "live trace is unavailable")
                raise RuntimeError(
                    f"live trace unavailable for {requested_candidate} "
                    f"seed {current_seed}: {reason}"
                )
            cached_bundles[(requested_candidate, current_seed)] = cached

    budget_results: dict[int, dict[str, Any]] = {}
    selected: dict[str, Any] = {}
    artifacts: dict[str, Path] = {}
    for budget in clean_budgets:
        budget_t0 = time.perf_counter()
        print(f"robotics_replay sweep budget start: k={budget}", flush=True)
        budget_output = output / f"budget_{budget}"
        result = run_replay_eval(
            candidates=requested,
            length=length,
            seed=seed,
            seeds=seeds,
            warmup_steps=warmup_steps,
            budget=budget,
            horizon=horizon,
            beam_width=beam_width,
            output=budget_output,
            trace_source=trace_source,
            monitor=monitor,
            scenario_family=scenario_family,
            method_set=method_set,
            drone_controller=drone_controller,
            drone_sidecar_python=drone_sidecar_python,
            f1tenth_sidecar_python=f1tenth_sidecar_python,
            f1tenth_map=f1tenth_map,
            cached_bundles=cached_bundles,
            learned_mode=learned_mode,
            regret_oracle=regret_oracle,
            regret_iterations=regret_iterations,
            regret_epochs=regret_epochs,
            regret_train_seeds=regret_train_seeds if regret_train_seeds is not None else seeds,
            regret_eval_seeds=regret_eval_seeds if regret_eval_seeds is not None else seeds,
            regret_loss=regret_loss,
        )
        budget_results[budget] = result
        metadata["budget_dirs"][str(budget)] = str(budget_output)
        selected = _select_sweep_budget(budget_results)
        _write_budget_sweep_outputs(
            output,
            budget_results=budget_results,
            metadata={**metadata, "selected_budget": selected},
        )
        print(
            f"robotics_replay sweep budget complete: k={budget} "
            f"elapsed={time.perf_counter() - budget_t0:.1f}s",
            flush=True,
        )

    if render_selected and selected:
        selected_budget = int(selected["budget"])
        eval_dir = output / f"budget_{selected_budget}"
        artifacts.update(render_replay(
            eval_dir=eval_dir,
            output=output / "render_best_static",
            candidates=(candidate,),
            methods=("best_static", "mpc_beam3"),
            save_gif=False,
        ))
        artifacts.update({
            f"scott_{name}": path
            for name, path in render_replay(
                eval_dir=eval_dir,
                output=output / "render_scott",
                candidates=(candidate,),
                methods=("scott", "mpc_beam3"),
                save_gif=False,
            ).items()
        })
        save_json(
            {name: str(path) for name, path in artifacts.items()},
            output / "selected_render_artifacts.json",
        )

    metadata = {**metadata, "selected_budget": selected}
    save_json(metadata, output / "budget_sweep_metadata.json")
    print(
        f"robotics_replay sweep complete: elapsed={time.perf_counter() - sweep_t0:.1f}s "
        f"output={output}",
        flush=True,
    )
    return {
        "metadata": metadata,
        "budget_results": budget_results,
        "budget_sweep_summary": pd.read_csv(output / "budget_sweep_summary.csv"),
        "budget_policy_gain": pd.read_csv(output / "budget_policy_gain.csv"),
        "budget_reducer_counts": pd.read_csv(output / "budget_reducer_counts.csv"),
        "budget_runtime": pd.read_csv(output / "budget_runtime.csv"),
        "budget_scenario_summary": pd.read_csv(output / "budget_scenario_summary.csv"),
        "budget_intervention_summary": pd.read_csv(output / "budget_intervention_summary.csv"),
        "budget_degeneracy_summary": pd.read_csv(output / "budget_degeneracy_summary.csv"),
        "selected_budget": selected,
        "render_artifacts": artifacts,
    }


def render_replay(
    *,
    eval_dir: Path,
    output: Path,
    candidates: Sequence[str],
    methods: tuple[str, str] = ("scott", "mpc_beam3"),
    seed: int | None = None,
    fps: int = 10,
    stride: int = 3,
    dpi: int = 140,
    save_gif: bool = True,
) -> dict[str, Path]:
    """Render focused static-vs-MPC replay visualizations from eval artifacts."""
    output.mkdir(parents=True, exist_ok=True)
    requested = ("drone", "f1tenth") if "all" in candidates else tuple(candidates)
    artifacts: dict[str, Path] = {}
    for candidate in requested:
        chosen_seed = seed if seed is not None else _select_render_seed(eval_dir, candidate, methods[1])
        candidate_artifacts = _render_candidate(
            eval_dir=eval_dir,
            output=output,
            candidate=candidate,
            seed=chosen_seed,
            methods=methods,
            fps=fps,
            stride=stride,
            dpi=dpi,
            save_gif=save_gif,
        )
        artifacts.update({
            f"{candidate}_{name}": path
            for name, path in candidate_artifacts.items()
        })
    return artifacts


def _load_replay_bundle(
    *,
    candidate: str,
    length: int,
    seed: int,
    trace_source: str,
    monitor: str,
    scenario_family: str,
    output: Path,
    drone_controller: DroneController,
    drone_sidecar_python: Path | None,
    f1tenth_sidecar_python: Path | None,
    f1tenth_map: str,
) -> tuple[ProbeBundle | None, dict[str, Any]]:
    if trace_source == "procedural":
        selected_monitor = monitor if scenario_family == "stress" else (
            monitor if candidate == "f1tenth" else "stream"
        )
        bundle = make_procedural_replay_bundle(
            candidate,
            length=length,
            seed=seed,
            monitor=selected_monitor,
            scenario_family=scenario_family,
        )
        return bundle, bundle.metadata
    bundle, metadata = _candidate_bundle(
        candidate,
        length=length,
        seed=seed,
        trace_source=trace_source,  # type: ignore[arg-type]
        output=output,
        drone_controller=drone_controller,
        drone_sidecar_python=drone_sidecar_python,
        f1tenth_sidecar_python=f1tenth_sidecar_python,
        f1tenth_map=f1tenth_map,
        stress_randomize=trace_source == "live" and scenario_family == "stress",
    )
    if monitor == "physical" and trace_source == "live" and bundle is not None:
        try:
            bundle = _live_bundle_to_physical(
                candidate=candidate,
                bundle=bundle,
                seed=seed,
                f1tenth_map=f1tenth_map,
            )
            metadata = bundle.metadata
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            return None, {
                **metadata,
                "status": "unavailable",
                "reason": f"live physical conversion failed: {exc}",
                "monitor": "physical",
            }
    return bundle, metadata


def _live_bundle_to_physical(
    *,
    candidate: str,
    bundle: ProbeBundle,
    seed: int,
    f1tenth_map: str,
) -> ProbeBundle:
    if candidate == "drone":
        return _live_drone_bundle_to_physical(bundle, seed=seed)
    if candidate == "f1tenth":
        return _live_f1tenth_bundle_to_physical(
            bundle, seed=seed, map_name=f1tenth_map,
        )
    raise ValueError(f"unknown live physical candidate: {candidate}")


def _level0_gate_height(gate_type: float) -> float:
    return 1.0 if int(gate_type) == 0 else 0.525


def _drone_level0_geometry() -> DroneGateGeometry:
    level0 = _load_level0_geometry()
    gates = level0.get("gates", []) if level0.get("available", False) else []
    obstacles = level0.get("obstacles", []) if level0.get("available", False) else []
    if not gates:
        return _drone_default_geometry()
    return DroneGateGeometry(
        gates=tuple(
            (
                float(gate[0]),
                float(gate[1]),
                _level0_gate_height(float(gate[6] if len(gate) > 6 else 0.0)),
            )
            for gate in gates
        ),
        obstacles=tuple((float(ob[0]), float(ob[1])) for ob in obstacles),
        obstacle_radius=0.44,
        corridor_radius=0.50,
        gate_lateral_radius=0.50,
        gate_vertical_radius=0.36,
        altitude_floor=0.16,
        altitude_ceiling=1.75,
        speed_limit=1.06,
    )


def _live_drone_geometry_from_record(raw: dict[str, Any]) -> DroneGateGeometry:
    geometry_payload = raw.get("geometry")
    if isinstance(geometry_payload, dict):
        return DroneGateGeometry.from_payload(geometry_payload)

    fallback = _drone_level0_geometry()
    gates_raw = raw.get("gates", [])
    gates: list[tuple[float, float, float]] = []
    for gate in gates_raw if isinstance(gates_raw, list) else []:
        if not isinstance(gate, (list, tuple)) or len(gate) < 2:
            continue
        gate_type = float(gate[6]) if len(gate) > 6 else 0.0
        height = _level0_gate_height(gate_type)
        gates.append((float(gate[0]), float(gate[1]), height))
    obstacles_raw = raw.get("obstacles", [])
    obstacles: list[tuple[float, float]] = []
    for obstacle in obstacles_raw if isinstance(obstacles_raw, list) else []:
        if not isinstance(obstacle, (list, tuple)) or len(obstacle) < 2:
            continue
        obstacles.append((float(obstacle[0]), float(obstacle[1])))

    if not gates:
        return fallback
    return DroneGateGeometry(
        gates=tuple(gates),
        obstacles=tuple(obstacles),
        obstacle_radius=fallback.obstacle_radius,
        corridor_radius=fallback.corridor_radius,
        gate_lateral_radius=fallback.gate_lateral_radius,
        gate_vertical_radius=fallback.gate_vertical_radius,
        altitude_floor=fallback.altitude_floor,
        altitude_ceiling=fallback.altitude_ceiling,
        speed_limit=fallback.speed_limit,
    )


def _live_drone_bundle_to_physical(bundle: ProbeBundle, *, seed: int) -> ProbeBundle:
    monitor = make_drone_physical_monitor()
    rng = np.random.default_rng(seed)
    trace: list[SafetyStreamMeasurement] = []
    for measurement in bundle.trace:
        payload = dict(measurement.payload or {})
        raw = dict(payload.get("raw_record", payload))
        geometry = _live_drone_geometry_from_record(raw)
        obs = np.asarray(raw.get("obs", []), dtype=np.float64).ravel()
        if obs.size < 6:
            raise ValueError("drone live record observation must contain at least 6 values")
        values = np.array([obs[0], obs[2], obs[4], obs[1], obs[3], obs[5]], dtype=np.float64)
        gate_id = _drone_gate_id_from_payload({"raw_record": raw, **payload})
        true_values = _drone_physical_margins(values, geometry, gate_id)
        observed = values + rng.normal(
            0.0,
            np.array([0.020, 0.020, 0.018, 0.025, 0.025, 0.020], dtype=np.float64),
        )
        raw["geometry"] = geometry.to_payload()
        payload.update({
            "trace_source": "safe_control_gym_live_physical_replay",
            "geometry": geometry.to_payload(),
            "gate_id": gate_id,
            "raw_record": raw,
        })
        trace.append(SafetyStreamMeasurement(
            time=measurement.time,
            values=tuple(float(v) for v in observed),
            true_values=tuple(float(v) for v in true_values),
            oracle_violation=bool(np.any(true_values < 0.0) or measurement.oracle_violation),
            payload=payload,
        ))
    return ProbeBundle(
        candidate=bundle.candidate,
        monitor=monitor,
        trace=tuple(trace),
        metadata={
            **bundle.metadata,
            "status": "available",
            "trace_source": "safe_control_gym_live_physical_replay",
            "monitor": "physical",
            "monitor_model": "dynamics_physical_v2",
            "source_monitor": "stream",
            "live_physical_conversion": True,
        },
    )


def _live_f1tenth_geometry(map_name: str) -> F1TenthTrackGeometry:
    return F1TenthTrackGeometry(
        amp1=0.28,
        freq1=0.45,
        phase1=0.0,
        amp2=0.0,
        freq2=0.90,
        phase2=0.0,
        half_width=0.82,
        width_wave=0.10,
        width_freq=0.90,
        width_phase=0.40,
        bottleneck_x=0.0,
        bottleneck_depth=0.0,
        bottleneck_sigma=1.0,
        front_phase=0.0 if map_name else 0.0,
    )


def _live_f1tenth_geometry_from_record(
    raw: dict[str, Any],
    map_name: str,
) -> F1TenthTrackGeometry:
    geometry_payload = raw.get("map_geometry")
    if not isinstance(geometry_payload, dict) or not geometry_payload:
        geometry_payload = raw.get("geometry")
    if isinstance(geometry_payload, dict) and geometry_payload:
        return F1TenthTrackGeometry.from_payload(geometry_payload)
    return _live_f1tenth_geometry(map_name)


def _live_f1tenth_bundle_to_physical(
    bundle: ProbeBundle,
    *,
    seed: int,
    map_name: str,
) -> ProbeBundle:
    monitor = make_f1tenth_physical_monitor()
    rng = np.random.default_rng(seed)
    trace: list[SafetyStreamMeasurement] = []
    for measurement in bundle.trace:
        payload = dict(measurement.payload or {})
        raw = dict(payload.get("raw_record", payload))
        geometry = _live_f1tenth_geometry_from_record(raw, map_name)
        obs = raw.get("obs", {})
        obs_data = obs[0] if isinstance(obs, tuple) else obs
        if not isinstance(obs_data, dict):
            raise ValueError("F1TENTH live record observation must be a dictionary")
        values = np.array([
            _first_array_scalar(obs_data.get("poses_x"), np.nan),
            _first_array_scalar(obs_data.get("poses_y"), np.nan),
            _first_array_scalar(obs_data.get("poses_theta"), 0.0),
            _first_array_scalar(obs_data.get("linear_vels_x"), 0.0),
            _first_array_scalar(obs_data.get("ang_vels_z"), 0.0),
        ], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("F1TENTH live record has non-finite physical state")
        true_values = _f1tenth_physical_margins(values, geometry)
        observed = values + rng.normal(
            0.0,
            np.array([0.025, 0.025, 0.010, 0.020, 0.010], dtype=np.float64),
        )
        centerline = [
            [float(x), geometry.center_y(float(x))]
            for x in np.linspace(-8.5, 8.5, 120)
        ]
        width_profile = [
            [float(x), geometry.width(float(x))]
            for x in np.linspace(-8.5, 8.5, 120)
        ]
        raw["geometry"] = geometry.to_payload()
        raw["centerline"] = centerline
        raw["width_profile"] = width_profile
        raw["corridor_width"] = float(np.mean([w for _, w in width_profile]))
        payload.update({
            "trace_source": "f1tenth_live_physical_replay",
            "geometry": geometry.to_payload(),
            "raw_record": raw,
        })
        trace.append(SafetyStreamMeasurement(
            time=measurement.time,
            values=tuple(float(v) for v in observed),
            true_values=tuple(float(v) for v in true_values),
            oracle_violation=bool(np.any(true_values < 0.0) or measurement.oracle_violation),
            payload=payload,
        ))
    return ProbeBundle(
        candidate=bundle.candidate,
        monitor=monitor,
        trace=tuple(trace),
        metadata={
            **bundle.metadata,
            "status": "available",
            "trace_source": "f1tenth_live_physical_replay",
            "monitor": "physical",
            "monitor_model": "dynamics_physical_v3",
            "source_monitor": "stream",
            "live_physical_conversion": True,
            "map": map_name,
        },
    )


def make_procedural_replay_bundle(
    candidate: str,
    *,
    length: int,
    seed: int,
    monitor: str = "stream",
    scenario_family: str = "stress",
) -> ProbeBundle:
    """Create deterministic seed-varying robotics replay traces."""
    if scenario_family not in REPLAY_SCENARIO_FAMILIES:
        raise ValueError(f"unknown scenario_family {scenario_family!r}")
    rng = np.random.default_rng(seed)
    if candidate == "drone":
        if monitor == "physical" and scenario_family == "stress":
            monitor_adapter = make_drone_physical_monitor()
            trace = _stress_drone_physical_trace(length=length, seed=seed, rng=rng)
        elif monitor == "stream":
            profile = drone_stream_profile()
            trace = _procedural_drone_trace(length=length, seed=seed, rng=rng)
            monitor_adapter = SafetyStreamMonitor(profile)
        else:
            raise ValueError(f"unknown drone replay monitor/family: {monitor}/{scenario_family}")
    elif candidate == "f1tenth":
        if monitor == "physical":
            monitor_adapter = make_f1tenth_physical_monitor()
            trace = _procedural_f1tenth_physical_trace(
                length=length,
                seed=seed,
                rng=rng,
                scenario_family=scenario_family,
            )
        elif monitor == "stream":
            profile = f1tenth_stream_profile()
            monitor_adapter = SafetyStreamMonitor(profile)
            trace = _procedural_f1tenth_trace(length=length, seed=seed, rng=rng)
        else:
            raise ValueError(f"unknown F1TENTH replay monitor: {monitor}")
    else:
        raise ValueError(f"unknown robotics replay candidate: {candidate}")
    return ProbeBundle(
        candidate=candidate,
        monitor=monitor_adapter,
        trace=tuple(trace),
        metadata={
            "candidate": candidate,
            "status": "available",
            "trace_source": "procedural_replay",
            "monitor": monitor,
            "scenario_family": scenario_family,
            "monitor_model": _candidate_monitor_model(candidate, monitor, scenario_family),
            "length": length,
            "seed": seed,
            "note": (
                "Deterministic seed-varying replay trace for evaluating and "
                "visualizing reducer policies; not a live simulator rollout."
            ),
        },
    )


def _procedural_drone_trace(
    *,
    length: int,
    seed: int,
    rng: np.random.Generator,
) -> list[SafetyStreamMeasurement]:
    gates = np.array([
        [0.4, -2.4, 1.0],
        [1.7, -1.3, 0.55],
        [0.2, 0.3, 1.0],
        [-0.6, 2.4, 0.75],
    ], dtype=np.float64)
    gates[:, :2] += rng.normal(0.0, 0.12, size=(4, 2))
    obstacle_center = np.array([0.7, -0.8], dtype=np.float64) + rng.normal(0.0, 0.2, 2)
    trace: list[SafetyStreamMeasurement] = []
    for t in range(length):
        u = t / max(length - 1, 1)
        phase = 2.0 * np.pi * u
        segment = min(int(u * len(gates)), len(gates) - 1)
        start = np.array([-0.9, -2.9, 0.08], dtype=np.float64)
        waypoint = gates[segment]
        next_waypoint = gates[min(segment + 1, len(gates) - 1)]
        local = (u * len(gates)) % 1.0
        pos = (1.0 - local) * waypoint + local * next_waypoint
        if segment == 0:
            pos = (1.0 - local) * start + local * next_waypoint
        pos[:2] += 0.18 * np.array([np.sin(4.0 * phase + seed), np.cos(3.0 * phase)])
        pos[2] += 0.09 * np.sin(5.0 * phase + 0.3 * seed)
        vel_norm = 0.55 + 0.25 * abs(np.cos(2.2 * phase))
        obstacle_margin = float(np.linalg.norm(pos[:2] - obstacle_center) - 0.55)
        gate_alignment = float(
            min(0.35 - abs(pos[1] - waypoint[1]), 0.40 - abs(pos[2] - waypoint[2]))
            + 0.32 * (1.0 - local)
        )
        corridor_margin = float(0.28 + 0.24 * np.sin(1.6 * phase + 0.4))
        altitude_low = float(pos[2] - 0.10)
        altitude_high = float(1.9 - pos[2])
        speed_margin = float(1.1 - vel_norm)
        true_values = np.array([
            obstacle_margin,
            gate_alignment,
            corridor_margin,
            altitude_low,
            altitude_high,
            speed_margin,
        ], dtype=np.float64)
        observed = true_values + rng.normal(0.0, 0.025, true_values.size)
        trace.append(SafetyStreamMeasurement(
            time=float(t),
            values=tuple(float(v) for v in observed),
            true_values=tuple(float(v) for v in true_values),
            oracle_violation=bool(np.any(true_values < 0.0)),
            payload={
                "trace_source": "procedural_replay",
                "raw_record": {
                    "step": t,
                    "obs": [pos[0], 0.0, pos[1], 0.0, pos[2], 0.0],
                    "gates": [[float(g[0]), float(g[1]), 0, 0, 0, 0, 0] for g in gates],
                    "obstacles": [[float(obstacle_center[0]), float(obstacle_center[1]), 0, 0, 0, 0]],
                    "info": {"current_target_gate_id": int(segment)},
                },
            },
        ))
    return trace


def _stress_drone_physical_trace(
    *,
    length: int,
    seed: int,
    rng: np.random.Generator,
) -> list[SafetyStreamMeasurement]:
    geometry = _procedural_drone_geometry(seed, rng)
    gates = np.asarray(geometry.gates, dtype=np.float64)
    start = geometry.previous_gate(0)
    waypoints = np.vstack([start, gates])
    positions: list[np.ndarray] = []
    gate_ids: list[int] = []
    for t in range(length):
        u = t / max(length - 1, 1)
        progress = u * (len(waypoints) - 1)
        segment = min(int(progress), len(waypoints) - 2)
        local = progress - segment
        smooth = local * local * (3.0 - 2.0 * local)
        p0 = waypoints[segment]
        p1 = waypoints[segment + 1]
        base = (1.0 - smooth) * p0 + smooth * p1
        segment_vec = p1 - p0
        segment_xy = segment_vec[:2]
        norm = max(float(np.linalg.norm(segment_xy)), 1e-9)
        normal = np.array([-segment_xy[1], segment_xy[0]], dtype=np.float64) / norm
        phase = 2.0 * np.pi * u
        gate_pressure = np.exp(-0.5 * ((local - 0.78) / 0.18) ** 2)
        lateral = (
            0.18 * np.sin(3.2 * phase + 0.45 * seed)
            + 0.16 * gate_pressure * np.sin(7.0 * local + 0.6 * seed)
        )
        vertical = (
            0.08 * np.sin(4.8 * phase + 0.2 * seed)
            - 0.10 * gate_pressure * np.cos(5.0 * local + 0.3 * seed)
        )
        pos = base.copy()
        pos[:2] += lateral * normal
        pos[2] += vertical
        positions.append(pos)
        gate_ids.append(int(np.clip(segment, 0, len(gates) - 1)))

    if len(positions) >= 2:
        velocities = np.gradient(np.asarray(positions, dtype=np.float64), axis=0)
    else:
        velocities = np.zeros((len(positions), 3), dtype=np.float64)

    trace: list[SafetyStreamMeasurement] = []
    centerline = [[float(p[0]), float(p[1]), float(p[2])] for p in waypoints]
    for t, (pos, vel, gate_id) in enumerate(zip(positions, velocities, gate_ids)):
        u = t / max(length - 1, 1)
        speed_scale = 0.70 + 0.25 * abs(np.sin(2.6 * np.pi * u + 0.2 * seed))
        if float(np.linalg.norm(vel)) > 1e-9:
            vel = vel / float(np.linalg.norm(vel)) * speed_scale
        else:
            vel = np.array([speed_scale, 0.0, 0.0], dtype=np.float64)
        values = np.array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]], dtype=np.float64)
        true_values = _drone_physical_margins(values, geometry, gate_id)
        observed = values + rng.normal(
            0.0,
            np.array([0.020, 0.020, 0.018, 0.025, 0.025, 0.020], dtype=np.float64),
        )
        raw_gates = [
            [float(g[0]), float(g[1]), 0.0, 0.0, 0.0, 0.0, 0.0]
            for g in gates
        ]
        raw_obstacles = [
            [float(o[0]), float(o[1]), 0.0, 0.0, 0.0, 0.0]
            for o in np.asarray(geometry.obstacles, dtype=np.float64)
        ]
        gate = geometry.gate(gate_id)
        trace.append(SafetyStreamMeasurement(
            time=float(t),
            values=tuple(float(v) for v in observed),
            true_values=tuple(float(v) for v in true_values),
            oracle_violation=bool(np.any(true_values < 0.0)),
            payload={
                "trace_source": "procedural_physical_replay",
                "geometry": geometry.to_payload(),
                "gate_id": gate_id,
                "raw_record": {
                    "step": t,
                    "obs": [
                        float(values[0]), float(values[3]),
                        float(values[1]), float(values[4]),
                        float(values[2]), float(values[5]),
                    ],
                    "gates": raw_gates,
                    "obstacles": raw_obstacles,
                    "centerline": centerline,
                    "geometry": geometry.to_payload(),
                    "info": {
                        "current_target_gate_id": gate_id,
                        "current_target_gate_pos": [float(gate[0]), float(gate[1]), float(gate[2]), 0, 0, 0],
                        "current_target_gate_in_range": True,
                    },
                },
            },
        ))
    return trace


def _procedural_f1tenth_trace(
    *,
    length: int,
    seed: int,
    rng: np.random.Generator,
) -> list[SafetyStreamMeasurement]:
    trace: list[SafetyStreamMeasurement] = []
    curvature = 0.35 + 0.08 * rng.normal()
    corridor_width = 0.86 + 0.08 * rng.normal()
    for t in range(length):
        u = t / max(length - 1, 1)
        x = -7.5 + 15.0 * u
        y = 0.55 * np.sin(curvature * x) + 0.12 * np.sin(1.7 * x + seed)
        theta = np.arctan2(0.55 * curvature * np.cos(curvature * x), 1.0)
        speed = 0.85 + 0.30 * np.cos(2.0 * np.pi * u)
        yaw_rate = abs(0.16 * np.sin(3.4 * np.pi * u + 0.2 * seed))
        side_clearance = corridor_width - abs(y - 0.55 * np.sin(curvature * x))
        front_clearance = 1.65 + 0.40 * np.sin(2.6 * np.pi * u + 0.3)
        ttc = front_clearance / max(abs(speed), 0.2)
        heading_error = abs(theta - 0.08 * np.sin(2.0 * np.pi * u))
        true_values = np.array([
            front_clearance - 1.2,
            side_clearance - 0.55,
            ttc - 1.0,
            side_clearance - 0.65,
            0.75 - heading_error,
            1.8 - abs(speed) - 0.5 * yaw_rate,
        ], dtype=np.float64)
        observed = true_values + rng.normal(0.0, 0.025, true_values.size)
        trace.append(SafetyStreamMeasurement(
            time=float(t),
            values=tuple(float(v) for v in observed),
            true_values=tuple(float(v) for v in true_values),
            oracle_violation=bool(np.any(true_values < 0.0)),
            payload={
                "trace_source": "procedural_replay",
                "raw_record": {
                    "step": t,
                    "obs": {
                        "poses_x": [float(x)],
                        "poses_y": [float(y)],
                        "poses_theta": [float(theta)],
                        "linear_vels_x": [float(speed)],
                        "ang_vels_z": [float(yaw_rate)],
                    },
                    "map": "procedural_chicane",
                    "centerline": [
                        [float(xx), float(0.55 * np.sin(curvature * xx))]
                        for xx in np.linspace(-8.5, 8.5, 80)
                    ],
                    "corridor_width": float(corridor_width),
                },
            },
        ))
    return trace


def _procedural_f1tenth_physical_trace(
    *,
    length: int,
    seed: int,
    rng: np.random.Generator,
    scenario_family: str = "stress",
) -> list[SafetyStreamMeasurement]:
    geometry = _procedural_f1tenth_geometry(seed, rng, scenario_family=scenario_family)
    trace: list[SafetyStreamMeasurement] = []
    for t in range(length):
        u = t / max(length - 1, 1)
        x = -7.6 + 15.2 * u
        center_y = geometry.center_y(x)
        if scenario_family == "stress":
            bottleneck_pressure = np.exp(
                -0.5 * ((x - geometry.bottleneck_x) / max(geometry.bottleneck_sigma, 1e-9)) ** 2
            )
            lateral = (
                0.22 * np.sin(2.8 * np.pi * u + 0.7 * seed)
                + 0.10 * bottleneck_pressure * np.sin(8.0 * u + 0.3 * seed)
            )
            heading_amp = 0.14
            speed = (
                0.88
                + 0.24 * np.cos(2.0 * np.pi * u + 0.2)
                + 0.06 * bottleneck_pressure
            )
            yaw_rate = (
                0.11 * np.sin(4.8 * np.pi * u + 0.3 * seed)
                + 0.05 * bottleneck_pressure * np.cos(7.0 * u)
            )
        else:
            lateral = 0.24 * np.sin(2.6 * np.pi * u + 0.7 * seed)
            heading_amp = 0.16
            speed = 0.95 + 0.34 * np.cos(2.0 * np.pi * u + 0.2)
            yaw_rate = 0.12 * np.sin(4.5 * np.pi * u + 0.3 * seed)
        y = center_y + lateral
        tangent = geometry.tangent(x)
        theta = tangent + heading_amp * np.sin(3.1 * np.pi * u + 0.4 * seed)
        values = np.array([x, y, theta, speed, yaw_rate], dtype=np.float64)
        true_values = _f1tenth_physical_margins(values, geometry)
        observed = values + rng.normal(
            0.0,
            np.array([0.025, 0.025, 0.010, 0.020, 0.010], dtype=np.float64),
        )
        centerline = [
            [float(xx), geometry.center_y(float(xx))]
            for xx in np.linspace(-8.5, 8.5, 120)
        ]
        width_profile = [
            [float(xx), geometry.width(float(xx))]
            for xx in np.linspace(-8.5, 8.5, 120)
        ]
        trace.append(SafetyStreamMeasurement(
            time=float(t),
            values=tuple(float(v) for v in observed),
            true_values=tuple(float(v) for v in true_values),
            oracle_violation=bool(np.any(true_values < 0.0)),
            payload={
                "trace_source": "procedural_physical_replay",
                "geometry": geometry.to_payload(),
                "raw_record": {
                    "step": t,
                    "obs": {
                        "poses_x": [float(values[0])],
                        "poses_y": [float(values[1])],
                        "poses_theta": [float(values[2])],
                        "linear_vels_x": [float(values[3])],
                        "ang_vels_z": [float(values[4])],
                    },
                    "map": "procedural_physical_chicane",
                    "centerline": centerline,
                    "width_profile": width_profile,
                    "corridor_width": float(np.mean([w for _, w in width_profile])),
                    "geometry": geometry.to_payload(),
                },
            },
        ))
    return trace


def _procedural_f1tenth_geometry(
    seed: int,
    rng: np.random.Generator,
    scenario_family: str = "stress",
) -> F1TenthTrackGeometry:
    stress = scenario_family == "stress"
    return F1TenthTrackGeometry(
        amp1=(0.60 if stress else 0.58) + 0.08 * rng.normal(),
        freq1=(0.40 if stress else 0.38) + 0.03 * rng.normal(),
        phase1=0.35 * seed,
        amp2=(0.20 if stress else 0.18) + 0.04 * rng.normal(),
        freq2=(0.96 if stress else 0.90) + 0.07 * rng.normal(),
        phase2=0.7 + 0.2 * seed,
        half_width=(0.75 if stress else 0.74) + 0.04 * rng.normal(),
        width_wave=(0.12 if stress else 0.11) + 0.02 * rng.random(),
        width_freq=(0.76 if stress else 0.72) + 0.05 * rng.normal(),
        width_phase=0.4 * seed,
        bottleneck_x=-0.8 + 1.6 * rng.random(),
        bottleneck_depth=(0.26 if stress else 0.23) + 0.06 * rng.random(),
        bottleneck_sigma=(0.96 if stress else 1.1) + 0.25 * rng.random(),
        front_phase=0.8 + 0.25 * seed,
    )


def _geometry_from_payload(payload: dict[str, Any] | None) -> F1TenthTrackGeometry:
    if payload:
        geometry_payload = payload.get("geometry")
        if geometry_payload is None and isinstance(payload.get("raw_record"), dict):
            geometry_payload = payload["raw_record"].get("geometry")
        if isinstance(geometry_payload, dict):
            return F1TenthTrackGeometry.from_payload(geometry_payload)
    return F1TenthTrackGeometry(
        amp1=0.58,
        freq1=0.38,
        phase1=0.0,
        amp2=0.18,
        freq2=0.90,
        phase2=0.7,
        half_width=0.74,
        width_wave=0.11,
        width_freq=0.72,
        width_phase=0.0,
        bottleneck_x=0.0,
        bottleneck_depth=0.25,
        bottleneck_sigma=1.2,
        front_phase=0.8,
    )


def _f1tenth_physical_margins(
    state: np.ndarray,
    geometry: F1TenthTrackGeometry,
) -> np.ndarray:
    x, y, theta, speed, yaw_rate = np.asarray(state, dtype=np.float64).ravel()
    center_y = geometry.center_y(float(x))
    lateral = float(y - center_y)
    width = geometry.width(float(x))
    tangent = geometry.tangent(float(x))
    heading_error = _wrap_angle(float(theta - tangent))
    curvature = abs(geometry.curvature(float(x)))
    front = geometry.front_clearance(float(x), lateral)
    ttc = front / max(abs(float(speed)), 0.2)
    speed_limit = max(0.45, 1.34 - 1.85 * curvature)
    return np.array([
        width - lateral,
        width + lateral,
        0.34 - abs(heading_error),
        ttc - 0.78,
        speed_limit - abs(speed),
        0.42 - abs(yaw_rate),
    ], dtype=np.float64)


def _f1tenth_physical_margin_and_jacobian(
    state: np.ndarray,
    geometry: F1TenthTrackGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(state, dtype=np.float64).ravel()
    x, y, theta, speed, yaw_rate = state
    margins = _f1tenth_physical_margins(state, geometry)
    center_dy = geometry.center_dy(float(x))
    lateral = float(y - geometry.center_y(float(x)))
    width_dx = geometry.width_dx(float(x))
    tangent = geometry.tangent(float(x))
    dy = geometry.center_dy(float(x))
    ddy = geometry.center_ddy(float(x))
    tangent_dx = ddy / max(1.0 + dy * dy, 1e-9)
    heading_error = _wrap_angle(float(theta - tangent))
    heading_sign = 1.0 if heading_error >= 0.0 else -1.0
    front = geometry.front_clearance(float(x), lateral)
    front_dx = geometry.front_dx(float(x)) + 0.22 * np.sign(lateral) * center_dy
    front_dy = -0.22 * np.sign(lateral)
    speed_abs = max(abs(float(speed)), 0.2)
    speed_sign = 1.0 if speed >= 0.0 else -1.0
    curvature = geometry.curvature(float(x))
    curvature_sign = 1.0 if curvature >= 0.0 else -1.0
    # Finite difference keeps this derivative consistent with the procedural geometry.
    eps = 1e-4
    curvature_dx = (
        geometry.curvature(float(x + eps)) - geometry.curvature(float(x - eps))
    ) / (2.0 * eps)
    yaw_sign = 1.0 if yaw_rate >= 0.0 else -1.0

    j = np.zeros((len(F1TENTH_PHYSICAL_TRIGGER_NAMES), len(F1TENTH_PHYSICAL_STATE_NAMES)), dtype=np.float64)
    # left = width - (y - center_y)
    j[0, 0] = width_dx + center_dy
    j[0, 1] = -1.0
    # right = width + (y - center_y)
    j[1, 0] = width_dx - center_dy
    j[1, 1] = 1.0
    # heading = limit - |theta - tangent|
    j[2, 0] = heading_sign * tangent_dx
    j[2, 2] = -heading_sign
    # ttc = front / |speed| - threshold
    j[3, 0] = front_dx / speed_abs
    j[3, 1] = front_dy / speed_abs
    j[3, 3] = -front * speed_sign / max(speed_abs * speed_abs, 1e-9)
    # curvature-speed = speed_limit(curvature) - |speed|
    j[4, 0] = -1.85 * curvature_sign * curvature_dx
    j[4, 3] = -speed_sign
    # yaw-rate
    j[5, 4] = -yaw_sign
    return margins, j


def _f1tenth_projection_remainder(
    zonotope: Zonotope,
    geometry: F1TenthTrackGeometry,
) -> np.ndarray:
    r = zonotope.interval_radius()
    x_radius = float(r[0])
    y_radius = float(r[1])
    theta_radius = float(r[2])
    speed_radius = float(r[3])
    yaw_radius = float(r[4])
    return np.array([
        0.015 + 0.035 * x_radius**2,
        0.015 + 0.035 * x_radius**2,
        0.012 + 0.060 * x_radius**2 + 0.050 * theta_radius**2,
        0.025 + 0.080 * x_radius**2 + 0.040 * y_radius**2 + 0.120 * speed_radius**2,
        0.020 + 0.100 * x_radius**2 + 0.040 * speed_radius**2,
        0.008 + 0.030 * yaw_radius**2,
    ], dtype=np.float64)


def _drone_default_geometry() -> DroneGateGeometry:
    return DroneGateGeometry(
        gates=(
            (0.2, -2.2, 0.95),
            (1.6, -1.1, 0.62),
            (0.5, 0.5, 1.12),
            (-0.8, 1.9, 0.78),
        ),
        obstacles=((0.7, -0.7), (-0.1, 1.0)),
        obstacle_radius=0.44,
        corridor_radius=0.46,
        gate_lateral_radius=0.46,
        gate_vertical_radius=0.34,
        altitude_floor=0.18,
        altitude_ceiling=1.75,
        speed_limit=1.05,
    )


def _drone_geometry_from_payload(payload: dict[str, Any] | None) -> DroneGateGeometry:
    if payload:
        geometry_payload = payload.get("geometry")
        if geometry_payload is None and isinstance(payload.get("raw_record"), dict):
            geometry_payload = payload["raw_record"].get("geometry")
        if isinstance(geometry_payload, dict):
            return DroneGateGeometry.from_payload(geometry_payload)
    return _drone_default_geometry()


def _drone_gate_id_from_payload(payload: dict[str, Any] | None) -> int:
    if not payload:
        return 0
    raw = payload.get("raw_record", payload)
    info = raw.get("info", {}) if isinstance(raw, dict) else {}
    return int(info.get("current_target_gate_id", payload.get("gate_id", 0)))


def _procedural_drone_geometry(seed: int, rng: np.random.Generator) -> DroneGateGeometry:
    base = np.array([
        [0.0, -2.35, 0.92],
        [1.45, -1.15, 0.64],
        [0.55, 0.35, 1.10],
        [-0.95, 1.55, 0.74],
        [0.35, 2.55, 1.05],
    ], dtype=np.float64)
    base[:, :2] += rng.normal(0.0, 0.16, size=(base.shape[0], 2))
    base[:, 2] += rng.normal(0.0, 0.05, size=base.shape[0])
    obstacles = np.array([
        [0.78, -0.68],
        [-0.24, 0.78],
        [0.15, 1.82],
    ], dtype=np.float64)
    obstacles += rng.normal(0.0, 0.12, size=obstacles.shape)
    return DroneGateGeometry(
        gates=tuple(tuple(float(v) for v in row) for row in base),
        obstacles=tuple(tuple(float(v) for v in row) for row in obstacles),
        obstacle_radius=float(0.39 + 0.04 * rng.random()),
        corridor_radius=float(0.50 + 0.04 * rng.random()),
        gate_lateral_radius=float(0.50 + 0.04 * rng.random()),
        gate_vertical_radius=float(0.36 + 0.03 * rng.random()),
        altitude_floor=float(0.16 + 0.02 * rng.random()),
        altitude_ceiling=float(1.72 + 0.06 * rng.random()),
        speed_limit=float(1.06 + 0.08 * rng.random()),
    )


def _drone_physical_margins(
    state: np.ndarray,
    geometry: DroneGateGeometry,
    gate_id: int,
) -> np.ndarray:
    x, y, z, vx, vy, vz = np.asarray(state, dtype=np.float64).ravel()
    pos = np.array([x, y, z], dtype=np.float64)
    vel = np.array([vx, vy, vz], dtype=np.float64)
    gate = geometry.gate(gate_id)
    prev_gate = geometry.previous_gate(gate_id)
    segment = gate - prev_gate
    segment_xy = segment[:2]
    rel_xy = pos[:2] - prev_gate[:2]
    denom = max(float(np.dot(segment_xy, segment_xy)), 1e-9)
    tau = float(np.clip(np.dot(rel_xy, segment_xy) / denom, 0.0, 1.0))
    nearest = prev_gate + tau * segment
    corridor_distance = float(np.linalg.norm(pos[:2] - nearest[:2]))

    obstacles = np.asarray(geometry.obstacles, dtype=np.float64)
    if obstacles.size:
        obstacle_clearance = float(
            np.min(np.linalg.norm(obstacles[:, :2] - pos[:2], axis=1))
            - geometry.obstacle_radius
        )
    else:
        obstacle_clearance = 10.0

    if tau < 0.62:
        gate_alignment = 0.34
    else:
        gate_lateral = float(np.linalg.norm(pos[:2] - gate[:2]))
        gate_vertical = abs(float(z - gate[2]))
        gate_alignment = min(
            geometry.gate_lateral_radius - gate_lateral,
            geometry.gate_vertical_radius - gate_vertical,
        )
    speed = float(np.linalg.norm(vel))
    return np.array([
        obstacle_clearance,
        gate_alignment,
        geometry.corridor_radius - corridor_distance,
        z - geometry.altitude_floor,
        geometry.altitude_ceiling - z,
        geometry.speed_limit - speed,
    ], dtype=np.float64)


def _drone_physical_margin_and_jacobian(
    state: np.ndarray,
    geometry: DroneGateGeometry,
    gate_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(state, dtype=np.float64).ravel()
    margins = _drone_physical_margins(state, geometry, gate_id)
    eps = 1e-4
    jacobian = np.zeros((len(DRONE_PHYSICAL_TRIGGER_NAMES), len(DRONE_PHYSICAL_STATE_NAMES)), dtype=np.float64)
    for axis in range(state.size):
        delta = np.zeros_like(state)
        delta[axis] = eps
        hi = _drone_physical_margins(state + delta, geometry, gate_id)
        lo = _drone_physical_margins(state - delta, geometry, gate_id)
        jacobian[:, axis] = (hi - lo) / (2.0 * eps)
    return margins, jacobian


def _drone_projection_remainder(
    zonotope: Zonotope,
    geometry: DroneGateGeometry,
) -> np.ndarray:
    r = zonotope.interval_radius()
    pos_radius = float(np.linalg.norm(r[:3]))
    xy_radius = float(np.linalg.norm(r[:2]))
    z_radius = float(r[2])
    vel_radius = float(np.linalg.norm(r[3:6]))
    geometry_scale = max(
        geometry.obstacle_radius,
        geometry.corridor_radius,
        geometry.gate_lateral_radius,
        geometry.gate_vertical_radius,
    )
    return np.array([
        0.025 + 0.12 * xy_radius**2 + 0.02 * geometry_scale,
        0.035 + 0.18 * pos_radius**2,
        0.030 + 0.14 * xy_radius**2,
        0.010 + 0.04 * z_radius**2,
        0.010 + 0.04 * z_radius**2,
        0.025 + 0.18 * vel_radius**2,
    ], dtype=np.float64)


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _run_replay_bundle(
    bundle: ProbeBundle,
    *,
    budget: int,
    seed: int,
    horizon: int,
    beam_width: int,
    method_set: str,
) -> dict[str, pd.DataFrame]:
    methods = _replay_methods(
        bundle.monitor,
        budget=budget,
        horizon=horizon,
        beam_width=beam_width,
        method_set=method_set,
    )
    ground_truth = compute_ground_truth(bundle.monitor, bundle.trace)
    results = [
        run_single(
            monitor=bundle.monitor,
            trace=bundle.trace,
            policy=method,
            budget=budget,
            seed=seed,
            ground_truth=ground_truth,
        )
        for method in methods
    ]
    return {
        "timeseries": results_to_dataframe(results),
        "summary": summarize_results(results),
    }


def _replay_methods(
    monitor: SafetyStreamMonitor,
    *,
    budget: int,
    horizon: int,
    beam_width: int,
    method_set: str,
) -> list[StaticReductionPolicy | MPCReductionPolicy]:
    reducer_by_name = {
        "girard": GirardReducer(),
        "combastel": CombastelReducer(),
        "pca": PcaReducer(),
        "methA": MethAReducer(),
        "scott": ScottReducer(),
        "box": BoxReducer(),
    }
    static_names = STATIC_METHODS if method_set in {"headline", "static"} else FOCUSED_STATIC_METHODS
    static = [
        StaticReductionPolicy(
            ProtectedReducer(base=reducer_by_name[name]),
            _name=name,
        )
        for name in static_names
    ]
    if method_set == "static":
        return static
    if method_set not in {"focused", "sweep", "headline", "paper_core"}:
        raise ValueError(
            "robotics replay currently supports method_set focused|static|sweep|headline|paper_core"
        )

    cost = WeightedZonotopeCost(
        weights=CostWeights(),
        triggers=monitor.triggers,
        trigger_zonotope=monitor.trigger_zonotope,
    )
    if isinstance(monitor, F1TenthPhysicalMonitor):
        cost = F1TenthPhysicalReplayCost(base=cost)
    mpc_candidate_names = _mpc_candidate_names_for_monitor(monitor)
    top3 = tuple(
        ProtectedReducer(base=reducer_by_name[name])
        for name in mpc_candidate_names
    )
    broad = tuple(
        ProtectedReducer(base=reducer_by_name[name])
        for name in ("girard", "combastel", "pca", "methA", "scott")
    )
    fallback = ProtectedReducer(base=BoxReducer())
    predictor = _replay_predictor(monitor)
    rollout_scott = RolloutMPCPolicy(
        candidates=top3,
        base_reducer=ProtectedReducer(base=ScottReducer()),
        budget=budget,
        horizon=horizon,
        cost=cost,
        fallback=fallback,
    )
    beam = BeamMPCPolicy(
        candidates=top3,
        budget=budget,
        horizon=horizon,
        beam_width=beam_width,
        cost=cost,
        fallback=fallback,
    )
    if method_set == "sweep":
        return [
            *static,
            MPCReductionPolicy(beam, _name="mpc_beam3", horizon=horizon, predictor=predictor),
        ]

    sequence = MPCPolicy(
        candidates=top3,
        budget=budget,
        horizon=horizon,
        cost=cost,
        fallback=fallback,
    )
    if method_set in {"headline", "paper_core"}:
        rollout = RolloutMPCPolicy(
            candidates=broad,
            base_reducer=ProtectedReducer(base=GirardReducer()),
            budget=budget,
            horizon=horizon,
            cost=cost,
            fallback=fallback,
        )
        pair_rollout = PairRolloutMPCPolicy(
            first_candidates=top3,
            base_candidates=top3,
            budget=budget,
            horizon=horizon,
            cost=cost,
            fallback=fallback,
        )
        methods = [
            *static,
            MPCReductionPolicy(rollout, _name="mpc_rollout", horizon=horizon, predictor=predictor),
            MPCReductionPolicy(
                pair_rollout, _name="mpc_pair_rollout3", horizon=horizon,
                predictor=predictor,
            ),
            MPCReductionPolicy(beam, _name="mpc_beam3", horizon=horizon, predictor=predictor),
        ]
        if method_set == "headline":
            methods.append(
                MPCReductionPolicy(
                    sequence, _name="mpc_sequence3", horizon=horizon,
                    predictor=predictor,
                )
            )
        return methods

    return [
        *static,
        MPCReductionPolicy(
            rollout_scott, _name="mpc_rollout_scott", horizon=horizon,
            predictor=predictor,
        ),
        MPCReductionPolicy(beam, _name="mpc_beam3", horizon=horizon, predictor=predictor),
        MPCReductionPolicy(sequence, _name="mpc_sequence3", horizon=horizon, predictor=predictor),
    ]


def _replay_predictor(monitor: Any) -> SafetyStreamTrendPredictor | DronePhysicalPredictor | F1TenthPhysicalPredictor:
    if isinstance(monitor, DronePhysicalMonitor):
        return DronePhysicalPredictor()
    if isinstance(monitor, F1TenthPhysicalMonitor):
        return F1TenthPhysicalPredictor()
    return SafetyStreamTrendPredictor()


def _mpc_candidate_names_for_monitor(monitor: Any) -> tuple[str, ...]:
    if isinstance(monitor, F1TenthPhysicalMonitor):
        return ("girard", "methA", "scott")
    return ROBOTICS_MPC_CANDIDATE_NAMES


def _method_names(method_set: str) -> tuple[str, ...]:
    if method_set == "static":
        return STATIC_METHODS
    if method_set == "sweep":
        return SWEEP_METHODS
    if method_set == "focused":
        return FOCUSED_METHODS
    if method_set == "headline":
        return HEADLINE_METHODS
    if method_set == "paper_core":
        return PAPER_CORE_METHODS
    raise ValueError(
        "robotics replay currently supports method_set focused|static|sweep|headline|paper_core"
    )


def _scenario_summary(bundle: ProbeBundle, *, seed: int) -> dict[str, Any]:
    true_values = np.asarray([m.true_values for m in bundle.trace], dtype=np.float64)
    min_abs = np.min(np.abs(true_values), axis=1) if true_values.size else np.array([])
    payload = bundle.trace[0].payload if bundle.trace else {}
    raw = (payload or {}).get("raw_record", {}) if isinstance(payload, dict) else {}
    geometry = (payload or {}).get("geometry", raw.get("geometry", {})) if isinstance(raw, dict) else {}
    dynamics = _monitor_dynamics_summary(bundle)
    return {
        "candidate": bundle.candidate,
        "seed": seed,
        "monitor": bundle.metadata.get("monitor", ""),
        "scenario_family": bundle.metadata.get("scenario_family", ""),
        "monitor_model": bundle.metadata.get("monitor_model", ""),
        "length": len(bundle.trace),
        "near_threshold_fraction": float(
            np.mean(min_abs <= bundle.monitor.profile.near_threshold)
        ) if min_abs.size else 0.0,
        "oracle_violation_fraction": float(
            np.mean([m.oracle_violation for m in bundle.trace])
        ) if bundle.trace else 0.0,
        "min_true_margin": float(np.min(true_values)) if true_values.size else np.nan,
        "mean_min_abs_margin": float(np.mean(min_abs)) if min_abs.size else np.nan,
        "geometry_kind": "physical" if geometry else "stream",
        "gate_count": int(len(geometry.get("gates", []))) if isinstance(geometry, dict) else 0,
        "obstacle_count": int(len(geometry.get("obstacles", []))) if isinstance(geometry, dict) else 0,
        **dynamics,
    }


def _monitor_dynamics_summary(bundle: ProbeBundle) -> dict[str, float]:
    keys = (
        "propagated_width_fraction",
        "fresh_width_fraction",
        "projection_remainder_fraction",
        "transition_variation_score",
    )
    values: dict[str, list[float]] = {key: [] for key in keys}
    state_radii: list[np.ndarray] = []
    remainder_radii: list[float] = []
    exact_widths: list[float] = []
    state = bundle.monitor.initial_state()
    for measurement in bundle.trace:
        result = bundle.monitor.step(state, measurement)
        state = result.state
        payload = getattr(state.payload, "payload", None)
        diagnostics = payload.get("monitor_diagnostics", {}) if isinstance(payload, dict) else {}
        for key in keys:
            if key in diagnostics:
                values[key].append(float(diagnostics[key]))
        state_radii.append(state.zonotope.interval_radius())
        if "projection_remainder_radius" in diagnostics:
            remainder_radii.append(float(diagnostics["projection_remainder_radius"]))
        trigger_z = bundle.monitor.trigger_zonotope(state)
        lower, upper = trigger_z.interval_bounds()
        exact_widths.append(float(np.sum(upper - lower)))
    summary = {
        f"mean_{key}": float(np.mean(items)) if items else np.nan
        for key, items in values.items()
    }
    names = getattr(bundle.monitor, "state_names", ())
    if state_radii and names:
        radius_arr = np.vstack(state_radii)
        for idx, name in enumerate(names):
            summary[f"mean_state_radius_{name}"] = float(np.mean(radius_arr[:, idx]))
    summary["max_projection_remainder_radius"] = (
        float(np.max(remainder_radii)) if remainder_radii else np.nan
    )
    summary["max_exact_trigger_width"] = (
        float(np.max(exact_widths)) if exact_widths else np.nan
    )
    summary["max_trigger_width_exact_ratio"] = 1.0 if exact_widths else np.nan
    return summary


def _intervention_summary(
    *,
    candidate: str,
    seed: int,
    bundle: ProbeBundle,
    timeseries: pd.DataFrame,
    fallback_hold_steps: int = 2,
) -> pd.DataFrame:
    verdict_cols = [
        col for col in timeseries.columns
        if col.endswith("_violation") and col not in {"oracle_violation"}
    ]
    oracle = np.asarray([m.oracle_violation for m in bundle.trace], dtype=bool)
    rows = []
    for method, group in timeseries.groupby("method"):
        ordered = group.sort_values("step")
        if verdict_cols:
            triggered = ordered[verdict_cols].ne("safe").any(axis=1).to_numpy(dtype=bool)
        else:
            triggered = np.zeros(len(ordered), dtype=bool)
        aligned_oracle = oracle[:len(triggered)]
        fallback = np.zeros(len(triggered), dtype=bool)
        for idx, active in enumerate(triggered):
            if active:
                fallback[idx:min(idx + fallback_hold_steps, len(fallback))] = True
        rows.append({
            "candidate": candidate,
            "seed": seed,
            "method": method,
            "trigger_steps": int(np.sum(triggered)),
            "oracle_violation_steps": int(np.sum(aligned_oracle)),
            "spurious_interventions": int(np.sum(triggered & ~aligned_oracle)),
            "justified_interventions": int(np.sum(triggered & aligned_oracle)),
            "missed_violations": int(np.sum(~triggered & aligned_oracle)),
            "fallback_steps": int(np.sum(fallback)),
            "fallback_hold_steps": fallback_hold_steps,
        })
    return pd.DataFrame(rows)


def _aggregate_replay_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    aggregates = []
    for candidate, df in summary.groupby("candidate"):
        agg = aggregate_summary(df.drop(columns=["candidate"]))
        agg.insert(0, "candidate", candidate)
        aggregates.append(agg)
    return pd.concat(aggregates, ignore_index=True) if aggregates else pd.DataFrame()


def _policy_gain(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=[
            "candidate", "seed", "method", "baseline_method",
            "mean_trigger_width_gain", "false_positive_rate_gain",
            "mean_approx_error_gain", "scott_mean_trigger_width_gain",
            "scott_false_positive_rate_gain",
        ])
    rows = []
    static_names = set(STATIC_METHODS)
    for (candidate, seed), group in summary.groupby(["candidate", "seed"]):
        static = group[group["method"].isin(static_names)]
        if static.empty:
            continue
        baseline = static.sort_values(
            ["mean_trigger_width", "false_positive_rate", "mean_approx_error"],
        ).iloc[0]
        scott = static[static["method"] == "scott"]
        scott_row = scott.iloc[0] if not scott.empty else None
        for _, row in group.iterrows():
            method_name = str(row["method"])
            if not _is_policy_method(method_name):
                continue
            gain_row = {
                "candidate": candidate,
                "seed": int(seed),
                "method": row["method"],
                "baseline_method": baseline["method"],
                "baseline_mean_trigger_width": float(baseline["mean_trigger_width"]),
                "method_mean_trigger_width": float(row["mean_trigger_width"]),
                "mean_trigger_width_gain": float(
                    baseline["mean_trigger_width"] - row["mean_trigger_width"]
                ),
                "false_positive_rate_gain": float(
                    baseline["false_positive_rate"] - row["false_positive_rate"]
                ),
                "mean_approx_error_gain": float(
                    baseline["mean_approx_error"] - row["mean_approx_error"]
                ),
            }
            gain_row["visualization_ready"] = bool(
                gain_row["mean_trigger_width_gain"] > 1e-9
                and gain_row["false_positive_rate_gain"] >= -1e-12
            )
            if scott_row is not None:
                gain_row["scott_mean_trigger_width_gain"] = float(
                    scott_row["mean_trigger_width"] - row["mean_trigger_width"]
                )
                gain_row["scott_false_positive_rate_gain"] = float(
                    scott_row["false_positive_rate"] - row["false_positive_rate"]
                )
            rows.append(gain_row)
    return pd.DataFrame(rows)


def _winner_by_step(timeseries: pd.DataFrame) -> pd.DataFrame:
    if timeseries.empty:
        return pd.DataFrame(columns=[
            "candidate", "seed", "step", "winning_static_method",
            "winning_static_width", "mpc_beam3_width", "mpc_beam3_gain",
        ])
    rows = []
    static_names = set(STATIC_METHODS)
    for (candidate, seed, step), group in timeseries.groupby(["candidate", "seed", "step"]):
        static = group[group["method"].isin(static_names)]
        if static.empty:
            continue
        best = static.sort_values("trigger_width_sum").iloc[0]
        mpc = group[group["method"] == "mpc_beam3"]
        mpc_width = float(mpc.iloc[0]["trigger_width_sum"]) if not mpc.empty else np.nan
        rows.append({
            "candidate": candidate,
            "seed": int(seed),
            "step": int(step),
            "winning_static_method": best["method"],
            "winning_static_width": float(best["trigger_width_sum"]),
            "mpc_beam3_width": mpc_width,
            "mpc_beam3_gain": float(best["trigger_width_sum"] - mpc_width)
            if np.isfinite(mpc_width) else np.nan,
        })
    return pd.DataFrame(rows)


def _is_policy_method(method: str) -> bool:
    return method.startswith("mpc") or method.startswith("learned_regret")


def _write_budget_sweep_outputs(
    output: Path,
    *,
    budget_results: dict[int, dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    summary = _budget_sweep_summary(budget_results)
    policy_gain = _budget_policy_gain(budget_results)
    reducer_counts = _budget_reducer_counts(budget_results)
    runtime = _budget_runtime(budget_results)
    scenario_summary = _budget_scenario_summary(budget_results)
    intervention_summary = _budget_intervention_summary(budget_results)
    degeneracy_summary = _budget_degeneracy_summary(
        summary=summary,
        policy_gain=policy_gain,
        reducer_counts=reducer_counts,
        scenario_summary=scenario_summary,
    )
    summary.to_csv(output / "budget_sweep_summary.csv", index=False)
    policy_gain.to_csv(output / "budget_policy_gain.csv", index=False)
    reducer_counts.to_csv(output / "budget_reducer_counts.csv", index=False)
    runtime.to_csv(output / "budget_runtime.csv", index=False)
    scenario_summary.to_csv(output / "budget_scenario_summary.csv", index=False)
    intervention_summary.to_csv(output / "budget_intervention_summary.csv", index=False)
    degeneracy_summary.to_csv(output / "budget_degeneracy_summary.csv", index=False)
    save_json(metadata, output / "budget_sweep_metadata.json")
    _write_budget_sweep_report(
        output / "budget_sweep_report.md",
        metadata=metadata,
        summary=summary,
        policy_gain=policy_gain,
        runtime=runtime,
        degeneracy_summary=degeneracy_summary,
    )
    _write_budget_sweep_plots(
        output,
        policy_gain=policy_gain,
        reducer_counts=reducer_counts,
        runtime=runtime,
    )


def _budget_sweep_summary(budget_results: dict[int, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for budget, result in sorted(budget_results.items()):
        summary = result["summary"]
        if summary.empty:
            continue
        for _, row in summary.iterrows():
            rows.append({
                "budget": budget,
                "candidate": row["candidate"],
                "seed": int(row["seed"]),
                "method": row["method"],
                "mean_trigger_width": float(row["mean_trigger_width"]),
                "max_trigger_width": float(row["max_trigger_width"]),
                "false_positive_rate": float(row["false_positive_rate"]),
                "mean_approx_error": float(row["mean_approx_error"]),
                "total_reductions": int(row["total_reductions"]),
                "mean_generator_count": float(row["mean_generator_count"]),
                "total_time_ms": float(row["total_time_ms"]),
                "budget_violations": int(row["budget_violations"]),
                "unsound_certificates": int(row["unsound_certificates"]),
            })
    return pd.DataFrame(rows)


def _budget_policy_gain(budget_results: dict[int, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for budget, result in sorted(budget_results.items()):
        gain = result["policy_gain"]
        if gain.empty:
            continue
        for _, row in gain.iterrows():
            row_dict = row.to_dict()
            row_dict["budget"] = budget
            rows.append(row_dict)
    if not rows:
        return pd.DataFrame(columns=[
            "budget", "candidate", "seed", "method", "baseline_method",
            "mean_trigger_width_gain", "false_positive_rate_gain",
            "mean_approx_error_gain", "visualization_ready",
        ])
    cols = ["budget"] + [c for c in pd.DataFrame(rows).columns if c != "budget"]
    return pd.DataFrame(rows)[cols]


def _budget_reducer_counts(budget_results: dict[int, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for budget, result in sorted(budget_results.items()):
        timeseries = result["timeseries"]
        if timeseries.empty:
            continue
        reductions = timeseries[
            timeseries["method"].astype(str).map(_is_policy_method)
        ].copy()
        if reductions.empty:
            continue
        reductions["reducer_used"] = reductions["reducer_used"].fillna("")
        grouped = reductions.groupby(
            ["candidate", "seed", "method", "reducer_used"],
            dropna=False,
        ).size().reset_index(name="steps")
        for _, row in grouped.iterrows():
            rows.append({
                "budget": budget,
                "candidate": row["candidate"],
                "seed": int(row["seed"]),
                "method": row["method"],
                "reducer_used": row["reducer_used"],
                "steps": int(row["steps"]),
            })
    return pd.DataFrame(rows)


def _budget_runtime(budget_results: dict[int, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for budget, result in sorted(budget_results.items()):
        summary = result["summary"]
        if summary.empty:
            continue
        for _, row in summary.iterrows():
            rows.append({
                "budget": budget,
                "candidate": row["candidate"],
                "seed": int(row["seed"]),
                "method": row["method"],
                "total_time_ms": float(row["total_time_ms"]),
                "total_reductions": int(row["total_reductions"]),
                "mean_reduction_time_ms": (
                    float(row["total_time_ms"]) / max(int(row["total_reductions"]), 1)
                ),
                "mean_generator_count": float(row["mean_generator_count"]),
            })
    return pd.DataFrame(rows)


def _budget_scenario_summary(budget_results: dict[int, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for budget, result in sorted(budget_results.items()):
        scenario = result.get("scenario_summary", pd.DataFrame())
        if scenario.empty:
            continue
        for _, row in scenario.iterrows():
            data = row.to_dict()
            data["budget"] = budget
            rows.append(data)
    if not rows:
        return pd.DataFrame(columns=[
            "budget", "candidate", "seed", "monitor", "scenario_family",
            "monitor_model",
            "length", "near_threshold_fraction", "oracle_violation_fraction",
            "min_true_margin", "mean_min_abs_margin", "geometry_kind",
            "gate_count", "obstacle_count", "mean_propagated_width_fraction",
            "mean_fresh_width_fraction", "mean_projection_remainder_fraction",
            "mean_transition_variation_score", "max_projection_remainder_radius",
            "max_exact_trigger_width", "max_trigger_width_exact_ratio",
        ])
    cols = ["budget"] + [c for c in pd.DataFrame(rows).columns if c != "budget"]
    return pd.DataFrame(rows)[cols]


def _budget_intervention_summary(budget_results: dict[int, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for budget, result in sorted(budget_results.items()):
        intervention = result.get("intervention_summary", pd.DataFrame())
        if intervention.empty:
            continue
        for _, row in intervention.iterrows():
            data = row.to_dict()
            data["budget"] = budget
            rows.append(data)
    if not rows:
        return pd.DataFrame(columns=[
            "budget", "candidate", "seed", "method", "trigger_steps",
            "oracle_violation_steps", "spurious_interventions",
            "justified_interventions", "missed_violations", "fallback_steps",
            "fallback_hold_steps",
        ])
    cols = ["budget"] + [c for c in pd.DataFrame(rows).columns if c != "budget"]
    return pd.DataFrame(rows)[cols]


def _budget_degeneracy_summary(
    *,
    summary: pd.DataFrame,
    policy_gain: pd.DataFrame,
    reducer_counts: pd.DataFrame,
    scenario_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if summary.empty:
        return pd.DataFrame(columns=[
            "budget", "candidate", "seed_count", "unique_scenario_rows",
            "scenario_diverse", "best_static_method", "best_static_methods",
            "best_static_width", "best_static_approx_fraction",
            "mpc_width", "mpc_approx_fraction", "mpc_gain",
            "mpc_gain_percent", "mpc_non_girard_fraction",
            "mean_propagated_width_fraction", "mean_fresh_width_fraction",
            "mean_projection_remainder_fraction", "mean_transition_variation_score",
            "max_projection_remainder_radius", "max_exact_trigger_width",
            "mpc_max_trigger_width_exact_ratio",
        ])

    static_names = set(STATIC_METHODS)
    for (budget, candidate), group in summary.groupby(["budget", "candidate"]):
        static = group[group["method"].isin(static_names)]
        if static.empty:
            continue
        best_rows = []
        for _, seed_group in static.groupby("seed"):
            best_rows.append(seed_group.sort_values(
                ["mean_trigger_width", "false_positive_rate", "mean_approx_error"],
            ).iloc[0])
        best_static = pd.DataFrame(best_rows)
        seed_count = int(best_static["seed"].nunique())
        best_static_width = float(best_static["mean_trigger_width"].mean())
        best_static_error = float(best_static["mean_approx_error"].mean())
        best_static_max_width = float(best_static["max_trigger_width"].max())

        mpc = group[group["method"] == "mpc_beam3"]
        mpc_width = float(mpc["mean_trigger_width"].mean()) if not mpc.empty else np.nan
        mpc_error = float(mpc["mean_approx_error"].mean()) if not mpc.empty else np.nan
        mpc_max_width = float(mpc["max_trigger_width"].max()) if not mpc.empty else np.nan
        gain_rows = (
            policy_gain[
                (policy_gain["budget"] == budget)
                & (policy_gain["candidate"] == candidate)
                & (policy_gain["method"] == "mpc_beam3")
            ]
            if not policy_gain.empty else pd.DataFrame()
        )
        mpc_gain = float(gain_rows["mean_trigger_width_gain"].mean()) if not gain_rows.empty else np.nan
        counts = (
            reducer_counts[
                (reducer_counts["budget"] == budget)
                & (reducer_counts["candidate"] == candidate)
                & (reducer_counts["method"] == "mpc_beam3")
                & (reducer_counts["reducer_used"] != "")
            ]
            if not reducer_counts.empty else pd.DataFrame()
        )
        total_steps = int(counts["steps"].sum()) if not counts.empty else 0
        non_girard_steps = int(
            counts[counts["reducer_used"] != "girard"]["steps"].sum()
        ) if total_steps else 0
        scenario = (
            scenario_summary[
                (scenario_summary["budget"] == budget)
                & (scenario_summary["candidate"] == candidate)
            ]
            if not scenario_summary.empty else pd.DataFrame()
        )
        scenario_cols = [
            c for c in (
                "monitor", "scenario_family", "length", "near_threshold_fraction",
                "oracle_violation_fraction", "min_true_margin",
                "mean_min_abs_margin", "geometry_kind", "gate_count",
                "obstacle_count",
            )
            if c in scenario.columns
        ]
        unique_scenario_rows = int(
            scenario[scenario_cols].drop_duplicates().shape[0]
        ) if scenario_cols else 0
        best_methods = sorted(str(v) for v in best_static["method"].unique())
        exact_max_width = (
            float(scenario["max_exact_trigger_width"].max())
            if "max_exact_trigger_width" in scenario else np.nan
        )
        rows.append({
            "budget": int(budget),
            "candidate": candidate,
            "seed_count": seed_count,
            "unique_scenario_rows": unique_scenario_rows,
            "scenario_diverse": bool(seed_count > 1 and unique_scenario_rows > 1),
            "best_static_method": str(best_static["method"].mode().iloc[0]),
            "best_static_methods": ",".join(best_methods),
            "best_static_width": best_static_width,
            "best_static_max_trigger_width_exact_ratio": (
                best_static_max_width / exact_max_width
                if np.isfinite(exact_max_width) and exact_max_width > 0.0 else np.nan
            ),
            "best_static_approx_fraction": (
                best_static_error / best_static_width if best_static_width > 0.0 else np.nan
            ),
            "mpc_width": mpc_width,
            "mpc_max_trigger_width_exact_ratio": (
                mpc_max_width / exact_max_width
                if np.isfinite(mpc_max_width)
                and np.isfinite(exact_max_width)
                and exact_max_width > 0.0 else np.nan
            ),
            "mpc_approx_fraction": (
                mpc_error / mpc_width if np.isfinite(mpc_width) and mpc_width > 0.0 else np.nan
            ),
            "mpc_gain": mpc_gain,
            "mpc_gain_percent": (
                100.0 * mpc_gain / best_static_width
                if np.isfinite(mpc_gain) and best_static_width > 0.0 else np.nan
            ),
            "mpc_non_girard_fraction": (
                non_girard_steps / total_steps if total_steps else 0.0
            ),
            "mean_propagated_width_fraction": (
                float(scenario["mean_propagated_width_fraction"].mean())
                if "mean_propagated_width_fraction" in scenario else np.nan
            ),
            "mean_fresh_width_fraction": (
                float(scenario["mean_fresh_width_fraction"].mean())
                if "mean_fresh_width_fraction" in scenario else np.nan
            ),
            "mean_projection_remainder_fraction": (
                float(scenario["mean_projection_remainder_fraction"].mean())
                if "mean_projection_remainder_fraction" in scenario else np.nan
            ),
            "mean_transition_variation_score": (
                float(scenario["mean_transition_variation_score"].mean())
                if "mean_transition_variation_score" in scenario else np.nan
            ),
            "max_projection_remainder_radius": (
                float(scenario["max_projection_remainder_radius"].max())
                if "max_projection_remainder_radius" in scenario else np.nan
            ),
            "max_exact_trigger_width": exact_max_width,
        })
    return pd.DataFrame(rows)


def _select_sweep_budget(budget_results: dict[int, dict[str, Any]]) -> dict[str, Any]:
    gain = _budget_policy_gain(budget_results)
    runtime = _budget_runtime(budget_results)
    counts = _budget_reducer_counts(budget_results)
    if gain.empty:
        return {}
    beam = gain[gain["method"] == "mpc_beam3"].copy()
    if beam.empty:
        return {}
    candidates = []
    for budget, group in beam.groupby("budget"):
        budget_runtime = runtime[
            (runtime["budget"] == budget) & (runtime["method"] == "mpc_beam3")
        ]
        reducer_counts = counts[
            (counts["budget"] == budget)
            & (counts["method"] == "mpc_beam3")
            & (counts["reducer_used"] != "")
        ]
        switch_count = int(
            reducer_counts[reducer_counts["reducer_used"] != "girard"]["steps"].sum()
        ) if not reducer_counts.empty else 0
        candidates.append({
            "budget": int(budget),
            "min_best_static_gain": float(group["mean_trigger_width_gain"].min()),
            "mean_best_static_gain": float(group["mean_trigger_width_gain"].mean()),
            "ready_seeds": int(group["visualization_ready"].sum()),
            "seed_count": int(len(group)),
            "mpc_switch_steps": switch_count,
            "mean_mpc_time_ms": float(budget_runtime["total_time_ms"].mean())
            if not budget_runtime.empty else float("inf"),
        })
    selected = sorted(
        candidates,
        key=lambda row: (
            row["min_best_static_gain"],
            row["mean_best_static_gain"],
            row["mpc_switch_steps"],
            -row["mean_mpc_time_ms"],
        ),
        reverse=True,
    )[0]
    return selected


def _write_budget_sweep_report(
    path: Path,
    *,
    metadata: dict[str, Any],
    summary: pd.DataFrame,
    policy_gain: pd.DataFrame,
    runtime: pd.DataFrame,
    degeneracy_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Robotics Replay Budget Sweep",
        "",
        f"- candidate: {metadata['candidate']}",
        f"- requested_candidates: {', '.join(metadata.get('requested_candidates', [metadata['candidate']]))}",
        f"- budgets: {', '.join(str(b) for b in metadata['budgets'])}",
        f"- monitor: {metadata['monitor']}",
        f"- scenario_family: {metadata.get('scenario_family', '')}",
        f"- length: {metadata['length']}",
        f"- seeds: {metadata['seeds']}",
        f"- horizon: {metadata['horizon']}",
        f"- beam_width: {metadata['beam_width']}",
        f"- methods: {', '.join(metadata['methods'])}",
        f"- mpc_candidate_reducers: {', '.join(metadata.get('mpc_candidate_reducers', []))}",
    ]
    selected = metadata.get("selected_budget") or {}
    if selected:
        lines.extend([
            "",
            "## Selected Budget",
            "",
            f"- budget: {selected['budget']}",
            f"- min_best_static_gain: {selected['min_best_static_gain']:.6g}",
            f"- mean_best_static_gain: {selected['mean_best_static_gain']:.6g}",
            f"- mpc_switch_steps: {selected['mpc_switch_steps']}",
        ])
    if not policy_gain.empty:
        beam = policy_gain[policy_gain["method"] == "mpc_beam3"]
        agg = beam.groupby("budget").agg(
            mean_best_static_gain=("mean_trigger_width_gain", "mean"),
            min_best_static_gain=("mean_trigger_width_gain", "min"),
            ready_seeds=("visualization_ready", "sum"),
            mean_scott_gain=("scott_mean_trigger_width_gain", "mean"),
        ).reset_index()
        lines.extend(["", "## Beam Gain By Budget", ""])
        lines.extend(_markdown_table(agg))
    if not runtime.empty:
        beam_runtime = runtime[runtime["method"] == "mpc_beam3"].groupby("budget").agg(
            mean_total_time_ms=("total_time_ms", "mean"),
            mean_reduction_time_ms=("mean_reduction_time_ms", "mean"),
            mean_reductions=("total_reductions", "mean"),
        ).reset_index()
        lines.extend(["", "## Beam Runtime By Budget", ""])
        lines.extend(_markdown_table(beam_runtime))
    if not degeneracy_summary.empty:
        display_cols = [
            "budget", "candidate", "seed_count", "scenario_diverse",
            "best_static_method", "best_static_approx_fraction",
            "mpc_gain_percent", "mpc_non_girard_fraction",
            "mean_propagated_width_fraction",
            "mean_projection_remainder_fraction",
            "mpc_max_trigger_width_exact_ratio",
            "mean_transition_variation_score",
        ]
        lines.extend(["", "## Degeneracy Diagnostics", ""])
        lines.extend(_markdown_table(degeneracy_summary[display_cols]))
    path.write_text("\n".join(lines) + "\n")


def _write_budget_sweep_plots(
    output: Path,
    *,
    policy_gain: pd.DataFrame,
    reducer_counts: pd.DataFrame,
    runtime: pd.DataFrame,
) -> None:
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    if not policy_gain.empty:
        beam = policy_gain[policy_gain["method"] == "mpc_beam3"]
        if not beam.empty:
            agg = beam.groupby("budget").agg(
                mean_gain=("mean_trigger_width_gain", "mean"),
                min_gain=("mean_trigger_width_gain", "min"),
                max_gain=("mean_trigger_width_gain", "max"),
            ).reset_index()
            fig, ax = plt.subplots(figsize=(6.4, 3.8))
            ax.plot(agg["budget"], agg["mean_gain"], marker="o", color="#2563eb", label="mean")
            ax.fill_between(
                agg["budget"], agg["min_gain"], agg["max_gain"],
                color="#93c5fd", alpha=0.35, label="seed range",
            )
            ax.axhline(0.0, color="#111827", linewidth=0.9)
            ax.set_xlabel("generator budget K")
            ax.set_ylabel("gain vs best static")
            ax.grid(color="#e5e7eb", linewidth=0.7)
            ax.legend(fontsize=8)
            _save_render_figure(fig, figure_dir / "budget_gain_vs_k", dpi=150)
    if not reducer_counts.empty:
        beam_counts = reducer_counts[reducer_counts["method"] == "mpc_beam3"].copy()
        beam_counts = beam_counts[beam_counts["reducer_used"] != ""]
        if not beam_counts.empty:
            pivot = beam_counts.groupby(["budget", "reducer_used"])["steps"].sum().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(6.4, 3.8))
            bottom = np.zeros(len(pivot), dtype=float)
            colors = {
                "girard": "#2563eb",
                "combastel": "#16a34a",
                "methA": "#8b5cf6",
                "scott": "#f59e0b",
                "box": "#64748b",
            }
            for reducer in pivot.columns:
                values = pivot[reducer].to_numpy(dtype=float)
                ax.bar(
                    pivot.index.astype(str), values, bottom=bottom,
                    label=str(reducer), color=colors.get(str(reducer), None),
                )
                bottom += values
            ax.set_xlabel("generator budget K")
            ax.set_ylabel("MPC selected steps")
            ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
            ax.legend(fontsize=8)
            _save_render_figure(fig, figure_dir / "budget_reducer_counts", dpi=150)
    if not runtime.empty:
        beam_runtime = runtime[runtime["method"] == "mpc_beam3"].groupby("budget").agg(
            mean_total_time_ms=("total_time_ms", "mean"),
        ).reset_index()
        if not beam_runtime.empty:
            fig, ax = plt.subplots(figsize=(6.4, 3.8))
            ax.plot(
                beam_runtime["budget"], beam_runtime["mean_total_time_ms"],
                marker="o", color="#7c3aed",
            )
            ax.set_xlabel("generator budget K")
            ax.set_ylabel("mean MPC reduction time (ms)")
            ax.grid(color="#e5e7eb", linewidth=0.7)
            _save_render_figure(fig, figure_dir / "budget_runtime_vs_k", dpi=150)


def _write_payload_jsonl(bundle: ProbeBundle, path: Path) -> None:
    with open(path, "w") as f:
        for measurement in bundle.trace:
            payload = measurement.payload or {}
            raw = payload.get("raw_record", payload)
            f.write(json.dumps(raw) + "\n")


def _write_replay_trace_csv(bundle: ProbeBundle, path: Path) -> None:
    stream_names = tuple(bundle.monitor.profile.stream_names)
    state_names = tuple(getattr(bundle.monitor, "state_names", stream_names))
    rows = []
    for measurement in bundle.trace:
        row: dict[str, Any] = {
            "time": measurement.time,
            "oracle_violation": measurement.oracle_violation,
        }
        for i, value in enumerate(measurement.values):
            name = state_names[i] if i < len(state_names) else f"state_{i}"
            row[name] = value
        for i, true_value in enumerate(measurement.true_values):
            name = stream_names[i] if i < len(stream_names) else f"trigger_{i}"
            row[f"true_{name}"] = true_value
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_eval_report(
    path: Path,
    *,
    metadata: dict[str, Any],
    aggregate: pd.DataFrame,
    policy_gain: pd.DataFrame,
) -> None:
    lines = [
        "# Robotics Replay Report",
        "",
        f"- trace_source: {metadata['trace_source']}",
        f"- monitor: {metadata['monitor']}",
        f"- scenario_family: {metadata.get('scenario_family', '')}",
        f"- length: {metadata['length']}",
        f"- seeds: {metadata['seeds']}",
        f"- warmup_steps: {metadata['warmup_steps']}",
        f"- budget: {metadata['budget']}",
        f"- horizon: {metadata['horizon']}",
        f"- methods: {', '.join(metadata['methods'])}",
        f"- mpc_candidate_reducers: {', '.join(metadata.get('mpc_candidate_reducers', []))}",
    ]
    if not policy_gain.empty:
        lines.extend(["", "## MPC Gains", ""])
        display_cols = [
            "candidate", "seed", "method", "baseline_method",
            "mean_trigger_width_gain", "false_positive_rate_gain",
        ]
        if "visualization_ready" in policy_gain.columns:
            display_cols.append("visualization_ready")
        if "scott_mean_trigger_width_gain" in policy_gain.columns:
            display_cols.append("scott_mean_trigger_width_gain")
        display = policy_gain[display_cols]
        lines.extend(_markdown_table(display))
    if not aggregate.empty:
        lines.extend(["", "## Aggregate", ""])
        cols = [
            c for c in (
                "candidate", "method", "mean_trigger_width_mean",
                "false_positive_rate_mean", "mean_approx_error_mean",
                "total_time_ms_mean",
            )
            if c in aggregate.columns
        ]
        lines.extend(_markdown_table(aggregate[cols]))
    path.write_text("\n".join(lines) + "\n")


def _markdown_table(df: pd.DataFrame) -> list[str]:
    rows = ["| " + " | ".join(str(c) for c in df.columns) + " |"]
    rows.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
    for _, row in df.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return rows


def _select_render_seed(eval_dir: Path, candidate: str, mpc_method: str) -> int:
    path = eval_dir / "policy_gain.csv"
    if not path.exists():
        return 0
    df = pd.read_csv(path)
    subset = df[(df["candidate"] == candidate) & (df["method"] == mpc_method)]
    if subset.empty:
        return int(df[df["candidate"] == candidate]["seed"].min()) if not df.empty else 0
    sort_col = (
        "scott_mean_trigger_width_gain"
        if "scott_mean_trigger_width_gain" in subset.columns
        else "mean_trigger_width_gain"
    )
    subset = subset.sort_values(sort_col, ascending=False)
    return int(subset.iloc[0]["seed"])


def _render_candidate(
    *,
    eval_dir: Path,
    output: Path,
    candidate: str,
    seed: int,
    methods: tuple[str, str],
    fps: int,
    stride: int,
    dpi: int,
    save_gif: bool,
) -> dict[str, Path]:
    seed_dir = eval_dir / f"seed_{seed}"
    streams = pd.read_csv(seed_dir / f"{candidate}_derived_streams.csv")
    payload = _load_payload(seed_dir / f"{candidate}_payload.jsonl")
    methods = tuple(
        _resolve_render_method(eval_dir, candidate, seed, method)
        for method in methods
    )  # type: ignore[assignment]
    timeseries = pd.read_csv(eval_dir / "timeseries.csv")
    timeseries = timeseries[
        (timeseries["candidate"] == candidate)
        & (timeseries["seed"] == seed)
        & (timeseries["method"].isin(methods))
    ].copy()
    if streams.empty or timeseries.empty:
        raise ValueError(f"no renderable data for {candidate} seed {seed}")

    prefix = f"robotics_{candidate}_{methods[0]}_vs_{methods[1]}_seed{seed}"
    frame_indices = list(range(0, len(streams), max(int(stride), 1)))
    if frame_indices[-1] != len(streams) - 1:
        frame_indices.append(len(streams) - 1)
    artifacts: dict[str, Path] = {}

    for label, idx in {
        "first": 0,
        "middle": len(streams) // 2,
        "last": len(streams) - 1,
    }.items():
        base = output / f"{prefix}_{label}"
        fig = _make_render_figure(
            candidate=candidate,
            streams=streams,
            payload=payload,
            timeseries=timeseries,
            frame_index=int(idx),
            methods=methods,
            seed=seed,
        )
        _save_render_figure(fig, base, dpi=dpi)
        artifacts[f"{label}_png"] = base.with_suffix(".png")
        artifacts[f"{label}_pdf"] = base.with_suffix(".pdf")

    storyboard_base = output / f"{prefix}_storyboard"
    _save_storyboard(
        storyboard_base,
        candidate=candidate,
        streams=streams,
        payload=payload,
        timeseries=timeseries,
        methods=methods,
        seed=seed,
        dpi=dpi,
    )
    artifacts["storyboard_png"] = storyboard_base.with_suffix(".png")
    artifacts["storyboard_pdf"] = storyboard_base.with_suffix(".pdf")

    if save_gif:
        gif_path = output / f"{prefix}.gif"
        fig = plt.figure(figsize=(10.8, 7.2))

        def update(frame_index: int):
            fig.clear()
            _draw_render_layout(
                fig,
                candidate=candidate,
                streams=streams,
                payload=payload,
                timeseries=timeseries,
                frame_index=frame_index,
                methods=methods,
                seed=seed,
            )
            return []

        animation = mpl_animation.FuncAnimation(
            fig,
            update,
            frames=frame_indices,
            interval=1000 / max(int(fps), 1),
            blit=False,
        )
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        animation.save(gif_path, writer=mpl_animation.PillowWriter(fps=max(int(fps), 1)), dpi=dpi)
        plt.close(fig)
        artifacts["gif"] = gif_path

    metadata_path = output / f"{prefix}_metadata.json"
    save_json({
        "candidate": candidate,
        "seed": seed,
        "methods": list(methods),
        "frame_count": int(len(streams)),
        "rendered_frame_count": int(len(frame_indices)),
        "eval_dir": str(eval_dir),
    }, metadata_path)
    artifacts["metadata"] = metadata_path
    return artifacts


def _resolve_render_method(
    eval_dir: Path,
    candidate: str,
    seed: int,
    method: str,
) -> str:
    if method != "best_static":
        return method
    summary_path = eval_dir / "summary.csv"
    if not summary_path.exists():
        raise ValueError("best_static render method requires summary.csv")
    summary = pd.read_csv(summary_path)
    static = summary[
        (summary["candidate"] == candidate)
        & (summary["seed"] == seed)
        & (summary["method"].isin(STATIC_METHODS))
    ]
    if static.empty:
        raise ValueError(f"no static methods found for {candidate} seed {seed}")
    best = static.sort_values(
        ["mean_trigger_width", "false_positive_rate", "mean_approx_error"],
    ).iloc[0]
    return str(best["method"])


def _load_payload(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _make_render_figure(
    *,
    candidate: str,
    streams: pd.DataFrame,
    payload: list[dict[str, Any]],
    timeseries: pd.DataFrame,
    frame_index: int,
    methods: tuple[str, str],
    seed: int,
) -> plt.Figure:
    fig = plt.figure(figsize=(10.8, 7.2))
    _draw_render_layout(
        fig,
        candidate=candidate,
        streams=streams,
        payload=payload,
        timeseries=timeseries,
        frame_index=frame_index,
        methods=methods,
        seed=seed,
    )
    return fig


def _draw_render_layout(
    fig: plt.Figure,
    *,
    candidate: str,
    streams: pd.DataFrame,
    payload: list[dict[str, Any]],
    timeseries: pd.DataFrame,
    frame_index: int,
    methods: tuple[str, str],
    seed: int,
) -> None:
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 0.88, 0.72])
    domain_axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    bar_axes = [fig.add_subplot(gs[1, i]) for i in range(2)]
    timeline_ax = fig.add_subplot(gs[2, :])
    for ax, method in zip(domain_axes, methods):
        _draw_domain_panel(ax, candidate, payload, frame_index, method=method, seed=seed)
    for ax, method in zip(bar_axes, methods):
        _draw_margin_bars(ax, streams, timeseries, frame_index, method)
    _draw_comparison_timeline(timeline_ax, timeseries, methods, frame_index)
    fig.suptitle(
        f"{candidate} replay: {methods[0]} vs {methods[1]} (seed {seed})",
        fontsize=12,
    )
    if hasattr(fig, "tight_layout"):
        fig.tight_layout()


def _draw_domain_panel(
    ax: plt.Axes,
    candidate: str,
    payload: list[dict[str, Any]],
    frame_index: int,
    *,
    method: str,
    seed: int,
) -> None:
    if candidate == "drone":
        points = []
        gates = []
        obstacles = []
        for record in payload:
            obs = np.asarray(record.get("obs", []), dtype=float).ravel()
            if obs.size >= 5:
                points.append([obs[0], obs[2], obs[4]])
            gates = record.get("gates", gates)
            obstacles = record.get("obstacles", obstacles)
        pts = np.asarray(points, dtype=float) if points else np.zeros((0, 3))
        if len(pts):
            ax.plot(pts[:, 0], pts[:, 1], color="#64748b", linewidth=1.3)
            ax.scatter(pts[:frame_index + 1, 0], pts[:frame_index + 1, 1], s=5, color="#94a3b8")
            idx = min(frame_index, len(pts) - 1)
            ax.scatter([pts[idx, 0]], [pts[idx, 1]], s=38, color="#2563eb", zorder=5)
        if gates:
            g = np.asarray(gates, dtype=float)
            ax.scatter(g[:, 0], g[:, 1], marker="s", s=42, color="#16a34a", label="gates")
        if obstacles:
            o = np.asarray(obstacles, dtype=float)
            ax.scatter(o[:, 0], o[:, 1], marker="x", s=48, color="#dc2626", label="obstacles")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        points = []
        centerline = []
        width = None
        width_profile = []
        for record in payload:
            obs = record.get("obs", {})
            if isinstance(obs, dict):
                x = _first_array_scalar(obs.get("poses_x"), np.nan)
                y = _first_array_scalar(obs.get("poses_y"), np.nan)
                theta = _first_array_scalar(obs.get("poses_theta"), 0.0)
                if np.isfinite(x) and np.isfinite(y):
                    points.append([x, y, theta])
            centerline = record.get("centerline", centerline)
            width = record.get("corridor_width", width)
            width_profile = record.get("width_profile", width_profile)
        pts = np.asarray(points, dtype=float) if points else np.zeros((0, 3))
        if centerline:
            c = np.asarray(centerline, dtype=float)
            ax.plot(c[:, 0], c[:, 1], color="#16a34a", linewidth=1.1, label="centerline")
            if width_profile:
                w = np.asarray(width_profile, dtype=float)
                if w.shape[0] == c.shape[0]:
                    ax.fill_between(c[:, 0], c[:, 1] - w[:, 1], c[:, 1] + w[:, 1], color="#dcfce7", alpha=0.55)
            elif width is not None:
                ax.fill_between(c[:, 0], c[:, 1] - float(width), c[:, 1] + float(width), color="#dcfce7", alpha=0.55)
        if len(pts):
            ax.plot(pts[:, 0], pts[:, 1], color="#64748b", linewidth=1.3)
            idx = min(frame_index, len(pts) - 1)
            ax.scatter([pts[idx, 0]], [pts[idx, 1]], s=38, color="#2563eb", zorder=5)
            ax.arrow(
                pts[idx, 0], pts[idx, 1],
                0.35 * np.cos(pts[idx, 2]), 0.35 * np.sin(pts[idx, 2]),
                color="#1d4ed8", width=0.015, length_includes_head=True,
            )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    ax.set_title(method, fontsize=10)
    ax.grid(color="#e5e7eb", linewidth=0.7)


def _draw_margin_bars(
    ax: plt.Axes,
    streams: pd.DataFrame,
    timeseries: pd.DataFrame,
    frame_index: int,
    method: str,
) -> None:
    row = streams.iloc[min(frame_index, len(streams) - 1)]
    names = [c for c in streams.columns if c.startswith("true_")]
    labels = [_short_stream_label(n.removeprefix("true_")) for n in names]
    values = np.array([float(row[n]) for n in names], dtype=float)
    colors = np.where(values < 0.0, "#dc2626", np.where(np.abs(values) < 0.25, "#f59e0b", "#2563eb"))
    y = np.arange(len(values))
    ax.barh(y, values, color=colors)
    ax.axvline(0.0, color="#111827", linewidth=1.0)
    method_ts = timeseries[timeseries["method"] == method]
    step = min(frame_index, len(method_ts) - 1)
    if len(method_ts):
        ts_row = method_ts.sort_values("step").iloc[step]
        ax.set_title(
            f"width={ts_row['trigger_width_sum']:.1f}, generators={int(ts_row['generator_count'])}",
            fontsize=9,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.7)


def _short_stream_label(name: str) -> str:
    return {
        "obstacle_clearance_margin": "obstacle",
        "gate_alignment_margin": "gate",
        "corridor_margin": "corridor",
        "altitude_low_margin": "alt low",
        "altitude_high_margin": "alt high",
        "speed_margin": "speed",
        "front_clearance_margin": "front",
        "side_clearance_margin": "side",
        "time_to_collision_margin": "ttc",
        "heading_margin": "heading",
        "curvature_speed_margin": "curve speed",
        "left_boundary_margin": "left",
        "right_boundary_margin": "right",
        "yaw_rate_margin": "yaw rate",
    }.get(name, name.replace("_margin", "").replace("_", " "))


def _draw_comparison_timeline(
    ax: plt.Axes,
    timeseries: pd.DataFrame,
    methods: tuple[str, str],
    frame_index: int,
) -> None:
    colors = {methods[0]: "#6b7280", methods[1]: "#2563eb"}
    for method in methods:
        df = timeseries[timeseries["method"] == method].sort_values("step")
        if df.empty:
            continue
        ax.plot(df["step"], df["trigger_width_sum"], label=method, color=colors[method], linewidth=1.5)
        reductions = df[df["reduced"] == True]
        if not reductions.empty:
            ax.scatter(reductions["step"], reductions["trigger_width_sum"], s=9, color=colors[method], alpha=0.45)
    ax.axvline(frame_index, color="#111827", linewidth=0.9, linestyle=":")
    ax.set_xlabel("step")
    ax.set_ylabel("trigger width")
    ax.grid(color="#e5e7eb", linewidth=0.7)
    ax.legend(loc="upper left", fontsize=8)


def _save_storyboard(
    base: Path,
    *,
    candidate: str,
    streams: pd.DataFrame,
    payload: list[dict[str, Any]],
    timeseries: pd.DataFrame,
    methods: tuple[str, str],
    seed: int,
    dpi: int,
) -> None:
    count = min(4, len(streams))
    indices = np.linspace(0, len(streams) - 1, count, dtype=int)
    fig = plt.figure(figsize=(10.8, 5.2 * count))
    gs = fig.add_gridspec(count, 1)
    for i, idx in enumerate(indices):
        subfig = fig.add_subfigure(gs[i, 0])
        _draw_render_layout(
            subfig,
            candidate=candidate,
            streams=streams,
            payload=payload,
            timeseries=timeseries,
            frame_index=int(idx),
            methods=methods,
            seed=seed,
        )
    _save_render_figure(fig, base, dpi=dpi)


def _save_render_figure(fig: plt.Figure, base: Path, *, dpi: int) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=dpi)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def _first_array_scalar(value: Any, default: float) -> float:
    if value is None:
        return default
    arr = np.asarray(value, dtype=float).ravel()
    if arr.size == 0:
        return default
    return float(arr[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate and render robotics replay scenarios.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("--candidate", choices=("drone", "f1tenth", "all"), default="all")
    eval_parser.add_argument("--trace-source", choices=REPLAY_TRACE_SOURCES, default="procedural")
    eval_parser.add_argument("--monitor", choices=REPLAY_MONITORS, default="physical")
    eval_parser.add_argument("--scenario-family", choices=REPLAY_SCENARIO_FAMILIES, default="stress")
    eval_parser.add_argument("--length", type=int, default=160)
    eval_parser.add_argument("--seed", type=int, default=0)
    eval_parser.add_argument("--seeds", type=int, default=3)
    eval_parser.add_argument("--warmup-steps", type=int, default=0)
    eval_parser.add_argument("--budget", type=int, default=12)
    eval_parser.add_argument("--horizon", type=int, default=4)
    eval_parser.add_argument("--beam-width", type=int, default=4)
    eval_parser.add_argument(
        "--method-set",
        choices=("focused", "static", "sweep", "headline", "paper_core"),
        default="focused",
    )
    eval_parser.add_argument("--drone-controller", choices=("sim", "firmware"), default="sim")
    eval_parser.add_argument("--drone-sidecar-python", type=Path, default=None)
    eval_parser.add_argument("--f1tenth-sidecar-python", type=Path, default=None)
    eval_parser.add_argument("--f1tenth-map", default="vegas")
    eval_parser.add_argument("--learned-mode", choices=("none", "regret"), default="none")
    eval_parser.add_argument("--regret-oracle", choices=REGRET_ORACLE_MODES, default="beam3")
    eval_parser.add_argument("--regret-iterations", type=int, default=3)
    eval_parser.add_argument("--regret-epochs", type=int, default=100)
    eval_parser.add_argument("--regret-train-seeds", type=int, default=None)
    eval_parser.add_argument("--regret-eval-seeds", type=int, default=None)
    eval_parser.add_argument("--regret-loss", choices=("pairwise", "mse"), default="pairwise")
    eval_parser.add_argument("--output", type=Path, default=Path("results/robotics-replay-eval"))

    render_parser = sub.add_parser("render")
    render_parser.add_argument("--eval-dir", type=Path, required=True)
    render_parser.add_argument("--candidate", choices=("drone", "f1tenth", "all"), default="all")
    render_parser.add_argument("--methods", default="scott,mpc_beam3")
    render_parser.add_argument("--seed", type=int, default=None)
    render_parser.add_argument("--fps", type=int, default=10)
    render_parser.add_argument("--stride", type=int, default=3)
    render_parser.add_argument("--dpi", type=int, default=140)
    render_parser.add_argument("--no-gif", action="store_true")
    render_parser.add_argument("--output", type=Path, default=Path("results/robotics-replay-viz"))

    sweep_parser = sub.add_parser("sweep")
    sweep_parser.add_argument("--candidate", choices=("drone", "f1tenth", "all"), default="all")
    sweep_parser.add_argument("--trace-source", choices=REPLAY_TRACE_SOURCES, default="procedural")
    sweep_parser.add_argument("--monitor", choices=REPLAY_MONITORS, default="physical")
    sweep_parser.add_argument("--scenario-family", choices=REPLAY_SCENARIO_FAMILIES, default="stress")
    sweep_parser.add_argument("--budgets", nargs="+", default=("8,10,12,16,20,24",))
    sweep_parser.add_argument("--length", type=int, default=80)
    sweep_parser.add_argument("--seed", type=int, default=0)
    sweep_parser.add_argument("--seeds", type=int, default=2)
    sweep_parser.add_argument("--warmup-steps", type=int, default=0)
    sweep_parser.add_argument("--horizon", type=int, default=4)
    sweep_parser.add_argument("--beam-width", type=int, default=4)
    sweep_parser.add_argument(
        "--method-set",
        choices=("focused", "static", "sweep", "headline", "paper_core"),
        default="sweep",
    )
    sweep_parser.add_argument("--drone-controller", choices=("sim", "firmware"), default="sim")
    sweep_parser.add_argument("--drone-sidecar-python", type=Path, default=None)
    sweep_parser.add_argument("--f1tenth-sidecar-python", type=Path, default=None)
    sweep_parser.add_argument("--f1tenth-map", default="vegas")
    sweep_parser.add_argument("--learned-mode", choices=("none", "regret"), default="none")
    sweep_parser.add_argument("--regret-oracle", choices=REGRET_ORACLE_MODES, default="beam3")
    sweep_parser.add_argument("--regret-iterations", type=int, default=3)
    sweep_parser.add_argument("--regret-epochs", type=int, default=100)
    sweep_parser.add_argument("--regret-train-seeds", type=int, default=None)
    sweep_parser.add_argument("--regret-eval-seeds", type=int, default=None)
    sweep_parser.add_argument("--regret-loss", choices=("pairwise", "mse"), default="pairwise")
    sweep_parser.add_argument("--no-render", action="store_true")
    sweep_parser.add_argument("--output", type=Path, default=Path("results/robotics-replay-sweep"))
    return parser


def _parse_budget_list(values: Sequence[str]) -> tuple[int, ...]:
    budgets: list[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                budgets.append(int(part))
    return tuple(budgets)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "eval":
        result = run_replay_eval(
            candidates=(args.candidate,),
            length=args.length,
            seed=args.seed,
            seeds=args.seeds,
            warmup_steps=args.warmup_steps,
            budget=args.budget,
            horizon=args.horizon,
            beam_width=args.beam_width,
            output=args.output,
            trace_source=args.trace_source,
            monitor=args.monitor,
            scenario_family=args.scenario_family,
            method_set=args.method_set,
            drone_controller=args.drone_controller,
            drone_sidecar_python=args.drone_sidecar_python,
            f1tenth_sidecar_python=args.f1tenth_sidecar_python,
            f1tenth_map=args.f1tenth_map,
            learned_mode=args.learned_mode,
            regret_oracle=args.regret_oracle,
            regret_iterations=args.regret_iterations,
            regret_epochs=args.regret_epochs,
            regret_train_seeds=args.regret_train_seeds,
            regret_eval_seeds=args.regret_eval_seeds,
            regret_loss=args.regret_loss,
        )
        if result["summary"].empty:
            print("No robotics replay candidates were evaluated.")
        else:
            for _, row in result["policy_gain"].iterrows():
                print(
                    f"{row['candidate']} seed {int(row['seed'])} {row['method']}: "
                    f"width gain {row['mean_trigger_width_gain']:.3f} "
                    f"vs {row['baseline_method']}"
                )
        print(f"Artifacts written to {args.output}")
        return

    if args.command == "sweep":
        result = run_replay_budget_sweep(
            candidate=args.candidate,
            budgets=_parse_budget_list(args.budgets),
            length=args.length,
            seed=args.seed,
            seeds=args.seeds,
            warmup_steps=args.warmup_steps,
            horizon=args.horizon,
            beam_width=args.beam_width,
            output=args.output,
            trace_source=args.trace_source,
            monitor=args.monitor,
            scenario_family=args.scenario_family,
            drone_controller=args.drone_controller,
            drone_sidecar_python=args.drone_sidecar_python,
            f1tenth_sidecar_python=args.f1tenth_sidecar_python,
            f1tenth_map=args.f1tenth_map,
            method_set=args.method_set,
            render_selected=not args.no_render,
            learned_mode=args.learned_mode,
            regret_oracle=args.regret_oracle,
            regret_iterations=args.regret_iterations,
            regret_epochs=args.regret_epochs,
            regret_train_seeds=args.regret_train_seeds,
            regret_eval_seeds=args.regret_eval_seeds,
            regret_loss=args.regret_loss,
        )
        selected = result["selected_budget"]
        if selected:
            print(
                "Selected budget "
                f"{selected['budget']} with min gain "
                f"{selected['min_best_static_gain']:.3f} "
                f"and mean gain {selected['mean_best_static_gain']:.3f}"
            )
        else:
            print("No selected budget; no MPC gain rows were produced.")
        print(f"Artifacts written to {args.output}")
        return

    method_parts = tuple(part.strip() for part in args.methods.split(",") if part.strip())
    if len(method_parts) != 2:
        raise ValueError("--methods must contain exactly two comma-separated method names")
    artifacts = render_replay(
        eval_dir=args.eval_dir,
        output=args.output,
        candidates=(args.candidate,),
        methods=(method_parts[0], method_parts[1]),
        seed=args.seed,
        fps=args.fps,
        stride=args.stride,
        dpi=args.dpi,
        save_gif=not args.no_gif,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
