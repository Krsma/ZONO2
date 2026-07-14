"""Deterministic random-waypoint traces for robot-arm learning.

The trajectory construction follows RLolaEval revision e6ecd0b2f60263e0a4270bd76a71cd9c90e685e5.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from pzr.rtlola.engine import RtlolaEvent
from pzr.rtlola.robot_arm import RobotArmTraceRow, SCENE_PATH


RANDOM_WAYPOINT_SOURCE_REVISION = "e6ecd0b2f60263e0a4270bd76a71cd9c90e685e5"
RANDOM_WAYPOINT_CONDITIONS = (
    "random_waypoint",
    "random_waypoint_drift",
    "random_waypoint_geofence",
    "random_waypoint_drift_geofence",
)
SENSOR_ERROR = 0.0107


@dataclass(frozen=True)
class RandomWaypointConfig:
    seed: int
    condition: str
    event_count: int
    n_waypoints: int = 10
    n_candidates: int = 500
    speed: float = 0.10
    sample_rate: float = 10.0
    z_min: float = 0.03
    xy_min: float = 0.05
    wall_margin: float = 0.025
    drift_z: float = 0.08
    fault_rotation: float = 0.3
    fault_onset_fraction: float = 0.5
    fault_full_fraction: float = 0.8
    max_tracking_error: float = 0.02
    max_retries: int = 100
    sv_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.condition not in RANDOM_WAYPOINT_CONDITIONS:
            raise ValueError(
                f"condition must be one of {RANDOM_WAYPOINT_CONDITIONS}, "
                f"got {self.condition!r}"
            )
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.event_count < 2:
            raise ValueError("event_count must be at least two")
        if self.n_waypoints < 2 or self.n_candidates < self.n_waypoints:
            raise ValueError("candidate count must cover at least two waypoints")
        if self.speed <= 0.0 or self.sample_rate <= 0.0:
            raise ValueError("speed and sample rate must be positive")
        if self.drift_z < 0.0:
            raise ValueError("drift must be non-negative")
        if self.max_retries < 1 or self.max_tracking_error <= 0.0:
            raise ValueError("retry and tracking-error bounds must be positive")
        if not 0.0 <= self.fault_onset_fraction < self.fault_full_fraction <= 1.0:
            raise ValueError("fault fractions must satisfy 0 <= onset < full <= 1")

    @property
    def has_drift(self) -> bool:
        return "drift" in self.condition

    @property
    def has_geofence_fault(self) -> bool:
        return "geofence" in self.condition

    @property
    def effective_sv_threshold(self) -> float:
        if self.sv_threshold is not None:
            return float(self.sv_threshold)
        return 3.0 if self.has_geofence_fault else 1.7


@dataclass(frozen=True)
class RandomWaypointMetadata:
    source_revision: str
    condition: str
    seed: int
    event_count: int
    attempts: int
    waypoint_center: tuple[float, float, float]
    perimeter: float
    traveled_distance: float
    completed_lap_fraction: float
    sv_spread: float
    singular_values: tuple[float, float, float]
    max_tracking_error: float
    geofence: tuple[float, float, float, float]
    mujoco_version: str
    trace_sha256: str
    generator_config: dict[str, object]


@dataclass(frozen=True)
class RandomWaypointTrace:
    rows: tuple[RobotArmTraceRow, ...]
    events: tuple[RtlolaEvent, ...]
    metadata: RandomWaypointMetadata


@dataclass(frozen=True)
class _SimulationResult:
    times: NDArray[np.float64]
    qposes: NDArray[np.float64]
    tcp: NDArray[np.float64]
    max_tracking_error: float
    completed: bool


PathFunction = Callable[[float], NDArray[np.float64]]


def generate_random_waypoint_trace(config: RandomWaypointConfig) -> RandomWaypointTrace:
    """Generate one benchmark-ready trace without writing repository assets."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    joint_low = model.jnt_range[:, 0].copy()
    joint_high = model.jnt_range[:, 1].copy()
    rng = np.random.default_rng(config.seed)
    duration = config.event_count / config.sample_rate
    accepted: tuple[
        NDArray[np.float64],
        float,
        NDArray[np.float64],
        _SimulationResult,
    ] | None = None
    for attempt in range(1, config.max_retries + 1):
        candidates = _sample_reachable_waypoints(
            model,
            data,
            site_id,
            joint_low,
            joint_high,
            config,
            rng,
        )
        indices = rng.choice(
            len(candidates), size=config.n_waypoints, replace=False,
        )
        waypoints = candidates[indices]
        spread, singular_values = _compute_sv_spread(
            model, data, site_id, waypoints, joint_low, joint_high,
        )
        if spread > config.effective_sv_threshold:
            continue
        ordered = _nearest_neighbor_sort(waypoints)
        path, perimeter = _make_waypoint_path(ordered)
        simulation = _simulate(
            model,
            data,
            site_id,
            joint_low,
            joint_high,
            path,
            config,
            duration,
        )
        if simulation.completed:
            accepted = ordered, perimeter, singular_values, simulation
            accepted_attempt = attempt
            break
    if accepted is None:
        raise RuntimeError(
            "could not find a valid random-waypoint trace after "
            f"{config.max_retries} attempts"
        )
    waypoints, perimeter, singular_values, simulation = accepted
    center = np.mean(waypoints, axis=0)
    path, _ = _make_waypoint_path(waypoints)
    walls = _compute_geofence(path, perimeter, config.wall_margin)
    rows = tuple(
        RobotArmTraceRow(
            time=float(time),
            angles=tuple(float(value) for value in qpose[:5]),  # type: ignore[arg-type]
            tcp=tuple(float(value) for value in tcp),  # type: ignore[arg-type]
            expected_center=(
                tuple(float(value) for value in center)  # type: ignore[arg-type]
                if index == 0 else (None, None, None)
            ),
            geofence=(
                walls if index == 0 else (None, None, None, None)
            ),
        )
        for index, (time, qpose, tcp) in enumerate(zip(
            simulation.times, simulation.qposes, simulation.tcp,
        ))
    )
    events = tuple(_row_to_event(row) for row in rows)
    traveled_distance = (
        float(np.linalg.norm(np.diff(simulation.tcp, axis=0), axis=1).sum())
        if len(simulation.tcp) > 1 else 0.0
    )
    trace_hash = _trace_sha256(rows)
    metadata = RandomWaypointMetadata(
        source_revision=RANDOM_WAYPOINT_SOURCE_REVISION,
        condition=config.condition,
        seed=config.seed,
        event_count=len(events),
        attempts=accepted_attempt,
        waypoint_center=tuple(float(value) for value in center),  # type: ignore[arg-type]
        perimeter=perimeter,
        traveled_distance=traveled_distance,
        completed_lap_fraction=(config.speed * duration) / perimeter,
        sv_spread=float(singular_values[0] / singular_values[2]),
        singular_values=tuple(float(value) for value in singular_values),  # type: ignore[arg-type]
        max_tracking_error=simulation.max_tracking_error,
        geofence=walls,
        mujoco_version=mujoco.__version__,
        trace_sha256=trace_hash,
        generator_config=asdict(config),
    )
    return RandomWaypointTrace(rows, events, metadata)


def write_random_waypoint_trace(trace: RandomWaypointTrace, directory: Path) -> None:
    """Persist a generated trace and provenance as an explicit artifact."""
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "trace.csv"
    temporary_csv = directory / ".trace.csv.tmp"
    with temporary_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "time", "a1m", "a2m", "a3m", "a4m", "a5m",
            "x", "y", "z", "cx", "cy", "cz",
            "x_min", "x_max", "y_min", "y_max",
        ))
        for row in trace.rows:
            writer.writerow((
                row.time,
                *row.angles,
                *row.tcp,
                *(_sparse_value(value) for value in row.expected_center),
                *(_sparse_value(value) for value in row.geofence),
            ))
    temporary_csv.replace(csv_path)
    temporary_metadata = directory / ".metadata.json.tmp"
    temporary_metadata.write_text(
        json.dumps(asdict(trace.metadata), indent=2, sort_keys=True),
    )
    temporary_metadata.replace(directory / "metadata.json")


def load_random_waypoint_trace(directory: Path) -> RandomWaypointTrace:
    """Load and validate one persisted generated trace."""
    metadata_payload = json.loads((directory / "metadata.json").read_text())
    with (directory / "trace.csv").open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        expected_header = [
            "time", "a1m", "a2m", "a3m", "a4m", "a5m",
            "x", "y", "z", "cx", "cy", "cz",
            "x_min", "x_max", "y_min", "y_max",
        ]
        if header != expected_header:
            raise ValueError("random-waypoint trace CSV schema differs")
        rows = tuple(_parse_trace_row(values) for values in reader)
    metadata = RandomWaypointMetadata(
        source_revision=str(metadata_payload["source_revision"]),
        condition=str(metadata_payload["condition"]),
        seed=int(metadata_payload["seed"]),
        event_count=int(metadata_payload["event_count"]),
        attempts=int(metadata_payload["attempts"]),
        waypoint_center=tuple(metadata_payload["waypoint_center"]),  # type: ignore[arg-type]
        perimeter=float(metadata_payload["perimeter"]),
        traveled_distance=float(metadata_payload["traveled_distance"]),
        completed_lap_fraction=float(metadata_payload["completed_lap_fraction"]),
        sv_spread=float(metadata_payload["sv_spread"]),
        singular_values=tuple(metadata_payload["singular_values"]),  # type: ignore[arg-type]
        max_tracking_error=float(metadata_payload["max_tracking_error"]),
        geofence=tuple(metadata_payload["geofence"]),  # type: ignore[arg-type]
        mujoco_version=str(metadata_payload["mujoco_version"]),
        trace_sha256=str(metadata_payload["trace_sha256"]),
        generator_config=dict(metadata_payload["generator_config"]),
    )
    if len(rows) != metadata.event_count:
        raise ValueError("random-waypoint trace length differs from metadata")
    if _trace_sha256(rows) != metadata.trace_sha256:
        raise ValueError("random-waypoint trace hash differs from metadata")
    return RandomWaypointTrace(
        rows=rows,
        events=tuple(_row_to_event(row) for row in rows),
        metadata=metadata,
    )


def _parse_trace_row(values: list[str]) -> RobotArmTraceRow:
    if len(values) != 16:
        raise ValueError(f"random-waypoint trace row has {len(values)} columns")
    return RobotArmTraceRow(
        time=float(values[0]),
        angles=tuple(float(value) for value in values[1:6]),  # type: ignore[arg-type]
        tcp=tuple(float(value) for value in values[6:9]),  # type: ignore[arg-type]
        expected_center=tuple(_parse_sparse_value(value) for value in values[9:12]),  # type: ignore[arg-type]
        geofence=tuple(_parse_sparse_value(value) for value in values[12:16]),  # type: ignore[arg-type]
    )


def _parse_sparse_value(value: str) -> float | None:
    return None if value == "#" else float(value)


def _sample_reachable_waypoints(
    model: object,
    data: object,
    site_id: int,
    joint_low: NDArray[np.float64],
    joint_high: NDArray[np.float64],
    config: RandomWaypointConfig,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    candidates = []
    attempts = 0
    limit = max(config.n_candidates * 10_000, 10_000)
    while len(candidates) < config.n_candidates and attempts < limit:
        attempts += 1
        q = np.zeros(model.nq, dtype=np.float64)  # type: ignore[attr-defined]
        q[:5] = rng.uniform(joint_low[:5], joint_high[:5])
        position = _forward_kinematics(model, data, site_id, q)
        if position[2] < config.z_min:
            continue
        # Drift feasibility is checked on the complete faulted simulation. A
        # drift-only z prefilter otherwise changes the base path distribution.
        if np.linalg.norm(position[:2]) < config.xy_min:
            continue
        if config.has_geofence_fault:
            angle = float(np.arctan2(position[1], position[0]))
            if angle < 1.0 or angle > 2.2:
                continue
        candidates.append(position)
    if len(candidates) < config.n_candidates:
        raise RuntimeError("could not sample enough reachable robot-arm waypoints")
    return np.asarray(candidates, dtype=np.float64)


def _forward_kinematics(
    model: object,
    data: object,
    site_id: int,
    q: NDArray[np.float64],
) -> NDArray[np.float64]:
    import mujoco

    data.qpos[:] = q  # type: ignore[attr-defined]
    mujoco.mj_forward(model, data)
    return np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()  # type: ignore[attr-defined]


def _inverse_kinematics(
    model: object,
    site_id: int,
    joint_low: NDArray[np.float64],
    joint_high: NDArray[np.float64],
    target: NDArray[np.float64],
    initial: NDArray[np.float64],
    ik_data: object,
    *,
    iterations: int = 60,
    damping: float = 0.1,
) -> NDArray[np.float64]:
    import mujoco

    ik_data.qpos[:] = initial  # type: ignore[attr-defined]
    for _ in range(iterations):
        mujoco.mj_forward(model, ik_data)
        error = target - ik_data.site_xpos[site_id]  # type: ignore[attr-defined]
        if np.linalg.norm(error) < 1e-5:
            break
        jacobian = np.zeros((3, model.nv), dtype=np.float64)  # type: ignore[attr-defined]
        mujoco.mj_jacSite(model, ik_data, jacobian, None, site_id)
        active = jacobian[:, :5]
        delta = active.T @ np.linalg.solve(
            active @ active.T + damping**2 * np.eye(3), error,
        )
        ik_data.qpos[:5] = np.clip(  # type: ignore[attr-defined]
            ik_data.qpos[:5] + delta, joint_low[:5], joint_high[:5],  # type: ignore[attr-defined]
        )
    return np.asarray(ik_data.qpos, dtype=np.float64).copy()  # type: ignore[attr-defined]


def _compute_sv_spread(
    model: object,
    data: object,
    site_id: int,
    waypoints: NDArray[np.float64],
    joint_low: NDArray[np.float64],
    joint_high: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    import mujoco

    ik_data = mujoco.MjData(model)
    previous = np.zeros(model.nq, dtype=np.float64)  # type: ignore[attr-defined]
    previous[:5] = (0.0, -0.5, 0.4, 0.0, 0.0)
    columns = []
    for waypoint in waypoints:
        q = _inverse_kinematics(
            model, site_id, joint_low, joint_high, waypoint, previous, ik_data,
        )
        data.qpos[:] = q  # type: ignore[attr-defined]
        mujoco.mj_forward(model, data)
        jacobian = np.zeros((3, model.nv), dtype=np.float64)  # type: ignore[attr-defined]
        mujoco.mj_jacSite(model, data, jacobian, None, site_id)
        columns.append(jacobian[:, :5] * SENSOR_ERROR)
        previous = q
    singular_values = np.linalg.svd(np.hstack(columns), compute_uv=False)
    spread = (
        float(singular_values[0] / singular_values[2])
        if singular_values[2] > 1e-12 else float("inf")
    )
    return spread, singular_values


def _nearest_neighbor_sort(waypoints: NDArray[np.float64]) -> NDArray[np.float64]:
    remaining = list(range(len(waypoints)))
    order = [remaining.pop(0)]
    while remaining:
        distances = np.linalg.norm(
            waypoints[remaining] - waypoints[order[-1]], axis=1,
        )
        order.append(remaining.pop(int(np.argmin(distances))))
    return waypoints[order]


def _make_waypoint_path(
    waypoints: NDArray[np.float64],
) -> tuple[PathFunction, float]:
    segments = np.diff(np.vstack((waypoints, waypoints[:1])), axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    perimeter = float(np.sum(lengths))
    if perimeter <= 0.0:
        raise ValueError("random-waypoint path has zero perimeter")
    cumulative = np.concatenate((np.zeros(1), np.cumsum(lengths)))

    def path(distance: float) -> NDArray[np.float64]:
        wrapped = distance % perimeter
        index = min(
            int(np.searchsorted(cumulative, wrapped, side="right")) - 1,
            len(waypoints) - 1,
        )
        fraction = (
            (wrapped - cumulative[index]) / lengths[index]
            if lengths[index] > 0.0 else 0.0
        )
        return waypoints[index] + fraction * (
            waypoints[(index + 1) % len(waypoints)] - waypoints[index]
        )

    return path, perimeter


def _compute_geofence(
    path: PathFunction,
    perimeter: float,
    margin: float,
) -> tuple[float, float, float, float]:
    points = np.asarray([
        path(distance)
        for distance in np.linspace(0.0, perimeter, 500, endpoint=False)
    ])
    return (
        float(np.min(points[:, 0]) - margin),
        float(np.max(points[:, 0]) + margin),
        float(np.min(points[:, 1]) - margin),
        float(np.max(points[:, 1]) + margin),
    )


def _simulate(
    model: object,
    data: object,
    site_id: int,
    joint_low: NDArray[np.float64],
    joint_high: NDArray[np.float64],
    path: PathFunction,
    config: RandomWaypointConfig,
    duration: float,
) -> _SimulationResult:
    import mujoco

    ik_data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    initial = np.zeros(model.nq, dtype=np.float64)  # type: ignore[attr-defined]
    initial[:5] = (0.0, -0.5, 0.4, 0.0, 0.0)
    data.qpos[:] = _inverse_kinematics(  # type: ignore[attr-defined]
        model, site_id, joint_low, joint_high, path(0.0), initial, ik_data,
    )
    data.ctrl[:5] = data.qpos[:5]  # type: ignore[attr-defined]
    times: list[float] = []
    qposes: list[NDArray[np.float64]] = []
    tcp_positions: list[NDArray[np.float64]] = []
    last_sample = 0.0
    max_error = 0.0
    onset = config.fault_onset_fraction * duration
    full = config.fault_full_fraction * duration
    while len(times) < config.event_count:
        raw = path(config.speed * data.time)  # type: ignore[attr-defined]
        target = raw.copy()
        if config.has_geofence_fault and data.time > onset:  # type: ignore[attr-defined]
            progress = min(1.0, (data.time - onset) / (full - onset))  # type: ignore[attr-defined]
            angle = config.fault_rotation * progress
            cosine, sine = np.cos(angle), np.sin(angle)
            target[:2] = (
                raw[0] * cosine - raw[1] * sine,
                raw[0] * sine + raw[1] * cosine,
            )
        if config.has_drift:
            target[2] += (data.time / duration) * config.drift_z  # type: ignore[attr-defined]
        controlled = _inverse_kinematics(
            model,
            site_id,
            joint_low,
            joint_high,
            target,
            np.asarray(data.qpos, dtype=np.float64),  # type: ignore[attr-defined]
            ik_data,
        )
        data.ctrl[:5] = controlled[:5]  # type: ignore[attr-defined]
        mujoco.mj_step(model, data)
        current_tcp = np.asarray(data.site_xpos[site_id], dtype=np.float64)  # type: ignore[attr-defined]
        max_error = max(max_error, float(np.linalg.norm(current_tcp - target)))
        if max_error > config.max_tracking_error:
            return _SimulationResult(
                np.asarray(times), np.asarray(qposes), np.asarray(tcp_positions),
                max_error, False,
            )
        if data.time - last_sample >= 1.0 / config.sample_rate:  # type: ignore[attr-defined]
            times.append(float(data.time))  # type: ignore[attr-defined]
            qposes.append(np.asarray(data.qpos[:5], dtype=np.float64).copy())  # type: ignore[attr-defined]
            tcp_positions.append(current_tcp.copy())
            last_sample = float(data.time)  # type: ignore[attr-defined]
    return _SimulationResult(
        np.asarray(times, dtype=np.float64),
        np.asarray(qposes, dtype=np.float64),
        np.asarray(tcp_positions, dtype=np.float64),
        max_error,
        True,
    )


def _row_to_event(row: RobotArmTraceRow) -> RtlolaEvent:
    return RtlolaEvent(
        time=row.time,
        values=(row.time, *row.angles, *row.expected_center, *row.geofence),
    )


def _trace_sha256(rows: tuple[RobotArmTraceRow, ...]) -> str:
    payload = json.dumps(
        [asdict(row) for row in rows], sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sparse_value(value: float | None) -> float | str:
    return "#" if value is None else value
