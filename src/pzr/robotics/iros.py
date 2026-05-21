"""Python RTLola-equivalent monitor utilities for IROS gate flying."""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pzr.core.zonotope import GeneratorKind, GeneratorMetadata, GeneratorRequirement, Zonotope
from pzr.monitoring.base import (
    MonitorResult,
    MonitorState,
    TriggerSpec,
    Verdict,
    evaluate_triggers,
)

OBSTACLE_CLEARANCE = 0
GATE_DEVIATION = 1
CORRIDOR_DEVIATION = 2
ALTITUDE_LOW_MARGIN = 3
ALTITUDE_HIGH_MARGIN = 4
SPEED = 5
SAFETY_MARGIN = 6

IROS_STREAM_NAMES = (
    "obstacle_clearance",
    "gate_deviation",
    "corridor_deviation",
    "altitude_low_margin",
    "altitude_high_margin",
    "speed",
    "safety_margin",
)


@dataclass(frozen=True)
class Gate:
    """Axis-aligned gate target used by the simulated gate-flying monitor."""

    center: NDArray[np.float64]
    width: float
    height: float

    def __init__(self, center: ArrayLike, width: float, height: float) -> None:
        c = np.asarray(center, dtype=float).reshape(3)
        object.__setattr__(self, "center", c)
        object.__setattr__(self, "width", float(width))
        object.__setattr__(self, "height", float(height))


@dataclass(frozen=True)
class Obstacle:
    """Spherical obstacle for oracle and monitor clearance checks."""

    center: NDArray[np.float64]
    radius: float

    def __init__(self, center: ArrayLike, radius: float) -> None:
        c = np.asarray(center, dtype=float).reshape(3)
        object.__setattr__(self, "center", c)
        object.__setattr__(self, "radius", float(radius))


@dataclass(frozen=True)
class IrosScenario:
    """Gate, obstacle, and envelope parameters for the gate-flying task."""

    gates: tuple[Gate, ...]
    obstacles: tuple[Obstacle, ...] = ()
    corridor_radius: float = 1.5
    min_obstacle_clearance: float = 0.1
    collision_radius: float = 0.0
    altitude_min: float = 0.2
    altitude_max: float = 3.0
    speed_max: float = 4.0
    gate_pass_radius: float = 0.35

    def gate(self, index: int) -> Gate:
        if not self.gates:
            raise ValueError("IROS scenario must contain at least one gate")
        return self.gates[min(max(int(index), 0), len(self.gates) - 1)]


@dataclass(frozen=True)
class IrosObservation:
    """One bounded observation consumed by the simulated RTLola monitor."""

    pose: NDArray[np.float64]
    velocity: NDArray[np.float64]
    target_gate_index: int = 0
    command: NDArray[np.float64] | None = None
    reference_state: NDArray[np.float64] | None = None
    noise_radius: NDArray[np.float64] | None = None
    time: float = 0.0

    def __init__(
        self,
        pose: ArrayLike,
        velocity: ArrayLike,
        target_gate_index: int = 0,
        command: ArrayLike | None = None,
        reference_state: ArrayLike | None = None,
        noise_radius: ArrayLike | None = None,
        time: float = 0.0,
    ) -> None:
        object.__setattr__(self, "pose", np.asarray(pose, dtype=float).reshape(3))
        object.__setattr__(self, "velocity", np.asarray(velocity, dtype=float).reshape(3))
        object.__setattr__(self, "target_gate_index", int(target_gate_index))
        object.__setattr__(
            self,
            "command",
            None if command is None else np.asarray(command, dtype=float).reshape(-1),
        )
        object.__setattr__(
            self,
            "reference_state",
            None
            if reference_state is None
            else np.asarray(reference_state, dtype=float).reshape(-1),
        )
        radius = None if noise_radius is None else np.asarray(noise_radius, dtype=float).reshape(6)
        object.__setattr__(self, "noise_radius", radius)
        object.__setattr__(self, "time", float(time))


@dataclass
class NoisySensorModel:
    """Persistent-bias plus fresh bounded-noise observation model."""

    bias_bound: float | Sequence[float] = 0.0
    noise_bound: float | Sequence[float] = 0.0
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)
    _bias: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        bias_bound = _six_vector(self.bias_bound)
        self._bias = self._rng.uniform(-bias_bound, bias_bound)

    @property
    def bias(self) -> NDArray[np.float64]:
        return self._bias.copy()

    @property
    def radius(self) -> NDArray[np.float64]:
        return np.abs(_six_vector(self.bias_bound)) + np.abs(_six_vector(self.noise_bound))

    def observe(
        self,
        pose: ArrayLike,
        velocity: ArrayLike,
        *,
        target_gate_index: int = 0,
        command: ArrayLike | None = None,
        reference_state: ArrayLike | None = None,
        time: float = 0.0,
    ) -> IrosObservation:
        true = np.concatenate(
            [np.asarray(pose, dtype=float).reshape(3), np.asarray(velocity, dtype=float).reshape(3)]
        )
        noise_bound = _six_vector(self.noise_bound)
        fresh_noise = self._rng.uniform(-noise_bound, noise_bound)
        measured = true + self._bias + fresh_noise
        return IrosObservation(
            measured[:3],
            measured[3:],
            target_gate_index=target_gate_index,
            command=command,
            reference_state=reference_state,
            noise_radius=self.radius,
            time=time,
        )


@dataclass(frozen=True)
class IrosGatePayload:
    """Private payload for gate monitor progress and observation time."""

    previous_time: float | None = None
    gates_passed: int = 0


@dataclass(frozen=True)
class IrosGateMonitor:
    """Python monitor with deterministic RTLola-equivalent stream semantics."""

    scenario: IrosScenario
    measurement_noise_scale: float = 0.0
    overlap: float = 0.0
    generator_memory_decay: float = 0.0

    @property
    def triggers(self) -> tuple[TriggerSpec, ...]:
        return (
            TriggerSpec("collision_risk", SAFETY_MARGIN, 0.0, direction="below", overlap=self.overlap),
            TriggerSpec(
                "obstacle_clearance_violation",
                OBSTACLE_CLEARANCE,
                self.scenario.min_obstacle_clearance,
                direction="below",
                overlap=self.overlap,
            ),
            TriggerSpec(
                "corridor_violation",
                CORRIDOR_DEVIATION,
                self.scenario.corridor_radius,
                direction="above",
                overlap=self.overlap,
            ),
            TriggerSpec("altitude_low_violation", ALTITUDE_LOW_MARGIN, 0.0, direction="below", overlap=self.overlap),
            TriggerSpec("altitude_high_violation", ALTITUDE_HIGH_MARGIN, 0.0, direction="below", overlap=self.overlap),
            TriggerSpec("speed_envelope_violation", SPEED, self.scenario.speed_max, direction="above", overlap=self.overlap),
        )

    def initial_state(self) -> MonitorState:
        center = np.zeros(len(IROS_STREAM_NAMES), dtype=float)
        generators = np.zeros((len(IROS_STREAM_NAMES), 1), dtype=float)
        metadata = (GeneratorMetadata(GeneratorKind.CALIBRATION, "iros_sensor_bias", 0),)
        return MonitorState(
            Zonotope(center, generators, metadata),
            step=0,
            payload=IrosGatePayload(),
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
        return (GeneratorRequirement(GeneratorKind.CALIBRATION, "iros_sensor_bias"),)

    def step(self, state: MonitorState, measurement: IrosObservation) -> MonitorResult:
        payload = state.payload
        if not isinstance(payload, IrosGatePayload):
            raise TypeError("IROS gate monitor payload has the wrong type")
        center = iros_stream_values(self.scenario, measurement)
        radius = self._stream_radius(measurement)
        generators, metadata = _axis_generators(
            radius,
            source_prefix=f"iros_noise@{state.step + 1}",
            include_bias=True,
        )
        if self.generator_memory_decay > 0.0 and state.zonotope.generator_count:
            old = state.zonotope.age_generators()
            memory = float(self.generator_memory_decay)
            generators = np.hstack([memory * old.generators, generators])
            metadata = (*tuple(_memory_metadata(meta) for meta in old.metadata), *metadata)
        gates_passed = payload.gates_passed
        gate = self.scenario.gate(measurement.target_gate_index)
        if np.linalg.norm(measurement.pose - gate.center) <= self.scenario.gate_pass_radius:
            gates_passed = max(gates_passed, min(measurement.target_gate_index + 1, len(self.scenario.gates)))
        new_state = MonitorState(
            Zonotope(center, generators, metadata),
            step=state.step + 1,
            payload=IrosGatePayload(previous_time=measurement.time, gates_passed=gates_passed),
        )
        return MonitorResult(new_state, evaluate_triggers(new_state.zonotope, self.triggers))

    def oracle_verdicts(self, pose: ArrayLike, velocity: ArrayLike, target_gate_index: int = 0) -> tuple[Verdict, ...]:
        observation = IrosObservation(pose, velocity, target_gate_index=target_gate_index)
        point = Zonotope(iros_stream_values(self.scenario, observation))
        return evaluate_triggers(point, self.triggers)

    def _stream_radius(self, measurement: IrosObservation) -> NDArray[np.float64]:
        source_radius = (
            np.full(6, self.measurement_noise_scale, dtype=float)
            if measurement.noise_radius is None
            else np.asarray(measurement.noise_radius, dtype=float).reshape(6)
        )
        pose_r = source_radius[:3]
        velocity_r = source_radius[3:]
        position_radius = float(np.linalg.norm(pose_r))
        speed_radius = float(np.linalg.norm(velocity_r))
        altitude_radius = float(abs(pose_r[2]))
        return np.asarray(
            [
                position_radius,
                position_radius,
                position_radius,
                altitude_radius,
                altitude_radius,
                speed_radius,
                position_radius + speed_radius,
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class InterventionMetrics:
    """Operational robotics metrics for monitor-triggered fallback control."""

    steps: int = 0
    fallback_activation_count: int = 0
    fallback_duration: int = 0
    spurious_intervention_count: int = 0
    justified_intervention_count: int = 0
    missed_violation_count: int = 0
    collision_count: int = 0
    constraint_violation_count: int = 0
    gates_passed: int = 0
    task_completed: bool = False
    time_to_target: float | None = None
    reducer_latency_seconds: float = 0.0
    budget_violation_count: int = 0
    unsound_certificate_count: int = 0
    reducer_choices: Mapping[str, int] = field(default_factory=dict)

    @property
    def spurious_intervention_rate(self) -> float:
        return self.spurious_intervention_count / self.steps if self.steps else 0.0

    @property
    def missed_violation_rate(self) -> float:
        return self.missed_violation_count / self.steps if self.steps else 0.0


@dataclass
class InterventionManager:
    """Switch nominal gate-following commands to fallback on monitor trigger."""

    fallback_command: NDArray[np.float64]
    fallback_hold_steps: int = 1
    expected_gate_count: int | None = None
    _remaining_fallback_steps: int = 0
    _previous_fallback: bool = False
    _metrics: InterventionMetrics = field(default_factory=InterventionMetrics)
    _reducer_choices: dict[str, int] = field(default_factory=dict)

    def __init__(
        self,
        fallback_command: ArrayLike,
        fallback_hold_steps: int = 1,
        expected_gate_count: int | None = None,
    ) -> None:
        self.fallback_command = np.asarray(fallback_command, dtype=float).reshape(-1)
        self.fallback_hold_steps = int(fallback_hold_steps)
        self.expected_gate_count = expected_gate_count
        self._remaining_fallback_steps = 0
        self._previous_fallback = False
        self._metrics = InterventionMetrics()
        self._reducer_choices = {}

    @property
    def metrics(self) -> InterventionMetrics:
        return replace(self._metrics, reducer_choices=dict(self._reducer_choices))

    def choose_command(
        self,
        nominal_command: ArrayLike,
        monitor_verdicts: Sequence[Verdict],
        oracle_verdicts: Sequence[Verdict],
        *,
        gates_passed: int = 0,
        time: float | None = None,
        reducer_name: str | None = None,
        reducer_latency_seconds: float = 0.0,
        budget_violation: bool = False,
        unsound_certificate: bool = False,
    ) -> NDArray[np.float64]:
        monitor_triggered = any(verdict.status == "violation" for verdict in monitor_verdicts)
        oracle_violated = any(verdict.status == "violation" for verdict in oracle_verdicts)
        if monitor_triggered:
            self._remaining_fallback_steps = max(1, self.fallback_hold_steps)
        use_fallback = self._remaining_fallback_steps > 0
        if use_fallback:
            self._remaining_fallback_steps -= 1
        if reducer_name:
            self._reducer_choices[reducer_name] = self._reducer_choices.get(reducer_name, 0) + 1

        activation = int(use_fallback and not self._previous_fallback)
        self._previous_fallback = use_fallback
        task_completed = self._metrics.task_completed or (
            self.expected_gate_count is not None
            and gates_passed >= self.expected_gate_count
        )
        time_to_target = self._metrics.time_to_target
        if task_completed and time_to_target is None and time is not None:
            time_to_target = float(time)

        self._metrics = InterventionMetrics(
            steps=self._metrics.steps + 1,
            fallback_activation_count=self._metrics.fallback_activation_count + activation,
            fallback_duration=self._metrics.fallback_duration + int(use_fallback),
            spurious_intervention_count=(
                self._metrics.spurious_intervention_count
                + int(monitor_triggered and not oracle_violated)
            ),
            justified_intervention_count=(
                self._metrics.justified_intervention_count
                + int(monitor_triggered and oracle_violated)
            ),
            missed_violation_count=(
                self._metrics.missed_violation_count
                + int((not monitor_triggered) and oracle_violated)
            ),
            collision_count=(
                self._metrics.collision_count
                + int(_verdict_named_violation(oracle_verdicts, "collision_risk"))
            ),
            constraint_violation_count=(
                self._metrics.constraint_violation_count + int(oracle_violated)
            ),
            gates_passed=max(self._metrics.gates_passed, int(gates_passed)),
            task_completed=task_completed,
            time_to_target=time_to_target,
            reducer_latency_seconds=(
                self._metrics.reducer_latency_seconds + float(reducer_latency_seconds)
            ),
            budget_violation_count=(
                self._metrics.budget_violation_count + int(budget_violation)
            ),
            unsound_certificate_count=(
                self._metrics.unsound_certificate_count + int(unsound_certificate)
            ),
            reducer_choices=dict(self._reducer_choices),
        )
        return self.fallback_command.copy() if use_fallback else np.asarray(nominal_command, dtype=float).reshape(-1)


def iros_stream_values(
    scenario: IrosScenario,
    observation: IrosObservation,
) -> NDArray[np.float64]:
    """Compute deterministic monitor streams from one observation."""

    pose = observation.pose
    velocity = observation.velocity
    gate = scenario.gate(observation.target_gate_index)
    obstacle_clearance = _minimum_obstacle_clearance(
        pose,
        scenario.obstacles,
        scenario.collision_radius,
    )
    gate_delta = pose - gate.center
    gate_deviation = float(np.linalg.norm(gate_delta[[1, 2]]))
    corridor_deviation = float(np.linalg.norm(gate_delta[:2]))
    altitude_low_margin = float(pose[2] - scenario.altitude_min)
    altitude_high_margin = float(scenario.altitude_max - pose[2])
    speed = float(np.linalg.norm(velocity))
    safety_margin = min(
        obstacle_clearance - scenario.min_obstacle_clearance,
        scenario.corridor_radius - corridor_deviation,
        altitude_low_margin,
        altitude_high_margin,
        scenario.speed_max - speed,
    )
    return np.asarray(
        [
            obstacle_clearance,
            gate_deviation,
            corridor_deviation,
            altitude_low_margin,
            altitude_high_margin,
            speed,
            safety_margin,
        ],
        dtype=float,
    )


def load_safe_control_gym_iros(root: str | Path | None = None) -> Any:
    """Load an optional safe-control-gym checkout for the IROS competition task."""

    configured_root = root or os.environ.get("PZR_SAFE_CONTROL_GYM_ROOT")
    if not configured_root:
        raise ImportError(
            "Set PZR_SAFE_CONTROL_GYM_ROOT to a safe-control-gym checkout to use the IROS adapter."
        )
    path = Path(configured_root).expanduser().resolve()
    if not path.exists():
        raise ImportError(f"safe-control-gym root does not exist: {path}")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    if importlib.util.find_spec("safe_control_gym") is None:
        raise ImportError(f"safe_control_gym package was not found under {path}")
    import safe_control_gym  # type: ignore[import-not-found]

    return safe_control_gym


def _minimum_obstacle_clearance(
    pose: NDArray[np.float64],
    obstacles: tuple[Obstacle, ...],
    collision_radius: float,
) -> float:
    if not obstacles:
        return float("inf")
    return float(
        min(
            np.linalg.norm(pose - obstacle.center) - obstacle.radius - collision_radius
            for obstacle in obstacles
        )
    )


def _axis_generators(
    radius: NDArray[np.float64],
    *,
    source_prefix: str,
    include_bias: bool,
) -> tuple[NDArray[np.float64], tuple[GeneratorMetadata, ...]]:
    active = [index for index, value in enumerate(radius) if abs(value) > 1e-12]
    generators = np.zeros((radius.size, len(active) + int(include_bias)), dtype=float)
    metadata: list[GeneratorMetadata] = []
    if include_bias:
        generators[:, 0] = radius
        metadata.append(GeneratorMetadata(GeneratorKind.CALIBRATION, "iros_sensor_bias", 0))
        offset = 1
    else:
        offset = 0
    for column, axis in enumerate(active, start=offset):
        generators[axis, column] = radius[axis]
        metadata.append(GeneratorMetadata(GeneratorKind.MEASUREMENT, f"{source_prefix}_axis_{axis}", 0))
    return generators, tuple(metadata)


def _six_vector(value: float | Sequence[float]) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(6, float(array), dtype=float)
    return array.reshape(6)


def _verdict_named_violation(verdicts: Sequence[Verdict], name: str) -> bool:
    return any(
        verdict.trigger.name == name and verdict.status == "violation"
        for verdict in verdicts
    )


def _memory_metadata(metadata: GeneratorMetadata) -> GeneratorMetadata:
    if metadata.kind == GeneratorKind.CALIBRATION:
        return GeneratorMetadata(
            GeneratorKind.UNKNOWN,
            f"memory:{metadata.source}" if metadata.source else "memory",
            metadata.age,
        )
    return metadata
