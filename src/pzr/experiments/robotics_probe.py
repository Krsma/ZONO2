"""Audit probes for candidate robotics evaluation environments.

This module is intentionally outside the default benchmark registry.  It is a
small diagnostic path for answering one question before adding a new scenario:
does the candidate produce non-degenerate zonotope-reduction behavior?
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from pzr.experiments.runner import (
    StaticReductionPolicy,
    compute_ground_truth,
    results_to_dataframe,
    run_single,
    summarize_results,
)
from pzr.monitoring.base import MonitorResult, MonitorState, TriggerSpec
from pzr.monitoring.triggers import evaluate_triggers
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


STATIC_REDUCERS = (
    ("girard", GirardReducer()),
    ("combastel", CombastelReducer()),
    ("pca", PcaReducer()),
    ("methA", MethAReducer()),
    ("scott", ScottReducer()),
    ("box", BoxReducer()),
)
METHOD_SCORE_COLUMNS = [
    "candidate",
    "method",
    "seed",
    "mean_trigger_width",
    "max_trigger_width",
    "mean_generator_count",
    "total_reductions",
    "budget_violations",
    "unsound_certificates",
    "false_positive_rate",
]
TRACE_SUMMARY_COLUMNS = [
    "candidate",
    "seed",
    "stream",
    "mean",
    "min",
    "max",
    "near_threshold_fraction",
]
REPORT_COLUMNS = [
    "candidate",
    "seed",
    "recommendation",
    "budget",
    "length",
    "relative_width_spread",
    "differentiated_methods",
    "reduction_rate",
    "near_threshold_fraction",
    "oracle_violation_fraction",
    "budget_violations",
    "unsound_certificates",
]
TraceSource = Literal["proxy", "live", "auto"]
DroneController = Literal["sim", "firmware"]


@dataclass(frozen=True)
class SafetyStreamMeasurement:
    """Low-dimensional measurement of derived safety margins."""

    time: float
    values: tuple[float, ...]
    true_values: tuple[float, ...]
    oracle_violation: bool = False
    payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class SafetyStreamProfile:
    """Static monitor configuration for a candidate environment probe."""

    stream_names: tuple[str, ...]
    calibration_generators: NDArray[np.float64]
    fresh_generators: NDArray[np.float64]
    near_threshold: float = 0.25

    @property
    def dimension(self) -> int:
        return len(self.stream_names)


@dataclass(frozen=True)
class SafetyStreamMonitor:
    """Monitor over derived safety margins.

    All streams are encoded as positive-is-safe margins.  A trigger fires when
    the corresponding margin can go below zero.
    """

    profile: SafetyStreamProfile

    @property
    def triggers(self) -> tuple[TriggerSpec, ...]:
        return tuple(
            TriggerSpec(f"{name}_violation", i, 0.0, "below", overlap=0.05)
            for i, name in enumerate(self.profile.stream_names)
        )

    @property
    def num_calibration_generators(self) -> int:
        return self.profile.calibration_generators.shape[1]

    def initial_state(self) -> MonitorState:
        center = np.zeros(self.profile.dimension, dtype=np.float64)
        generators = self.profile.calibration_generators
        return MonitorState(
            zonotope=Zonotope(center, generators),
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

    def trigger_zonotope(self, state: MonitorState) -> Zonotope:
        return state.zonotope

    def step(self, state: MonitorState, measurement: SafetyStreamMeasurement) -> MonitorResult:
        values = np.asarray(measurement.values, dtype=np.float64)
        if values.shape != (self.profile.dimension,):
            raise ValueError(
                f"expected {self.profile.dimension} safety streams, got {values.size}"
            )

        old_g = state.zonotope.generators
        n_existing = state.zonotope.generator_count
        n_fresh = self.profile.fresh_generators.shape[1]
        generators = np.zeros(
            (self.profile.dimension, n_existing + n_fresh),
            dtype=np.float64,
        )
        if n_existing:
            generators[:, :n_existing] = old_g
        for local_idx, col in enumerate(state.calibration_indices):
            if col < n_existing:
                generators[:, col] = self.profile.calibration_generators[:, local_idx]
        generators[:, n_existing:] = self.profile.fresh_generators

        zonotope = Zonotope(values, generators)
        new_state = MonitorState(
            zonotope=zonotope,
            step=state.step + 1,
            calibration_indices=state.calibration_indices,
            payload=measurement,
        )
        return MonitorResult(new_state, evaluate_triggers(zonotope, self.triggers))


@dataclass(frozen=True)
class ProbeBundle:
    """A candidate trace and its monitor configuration."""

    candidate: str
    monitor: SafetyStreamMonitor
    trace: tuple[SafetyStreamMeasurement, ...]
    metadata: dict[str, Any]


def _trim_bundle(bundle: ProbeBundle, warmup_steps: int) -> ProbeBundle:
    if warmup_steps <= 0:
        return bundle
    trace = tuple(bundle.trace[warmup_steps:])
    metadata = {
        **bundle.metadata,
        "raw_length": len(bundle.trace),
        "length": len(trace),
        "warmup_steps": warmup_steps,
    }
    return ProbeBundle(
        candidate=bundle.candidate,
        monitor=bundle.monitor,
        trace=trace,
        metadata=metadata,
    )


def drone_stream_profile() -> SafetyStreamProfile:
    """Profile for Crazyflie gate-flying derived streams."""
    stream_names = (
        "obstacle_clearance_margin",
        "gate_alignment_margin",
        "corridor_margin",
        "altitude_low_margin",
        "altitude_high_margin",
        "speed_margin",
    )
    calibration = np.array([
        [0.08, -0.02, 0.04],
        [0.05, 0.07, -0.01],
        [0.06, -0.04, 0.03],
        [-0.03, 0.01, 0.05],
        [0.02, -0.01, 0.04],
        [0.04, 0.03, -0.05],
    ], dtype=np.float64)
    fresh = np.array([
        [0.05, 0.00, 0.02, -0.01],
        [0.03, 0.04, -0.01, 0.00],
        [0.04, -0.02, 0.03, 0.01],
        [0.01, 0.00, 0.03, -0.02],
        [0.02, -0.01, 0.02, 0.03],
        [0.03, 0.02, -0.02, 0.04],
    ], dtype=np.float64)
    return SafetyStreamProfile(stream_names, calibration, fresh, near_threshold=0.25)


def f1tenth_stream_profile() -> SafetyStreamProfile:
    """Profile for F1TENTH map/LiDAR-derived streams."""
    stream_names = (
        "front_clearance_margin",
        "side_clearance_margin",
        "time_to_collision_margin",
        "corridor_margin",
        "heading_margin",
        "curvature_speed_margin",
    )
    calibration = np.array([
        [0.10, -0.03, 0.02],
        [0.06, 0.05, -0.03],
        [0.08, 0.03, 0.05],
        [0.05, -0.06, 0.02],
        [-0.03, 0.04, 0.05],
        [0.07, 0.02, -0.04],
    ], dtype=np.float64)
    fresh = np.array([
        [0.06, 0.02, 0.00, -0.02],
        [0.04, -0.03, 0.03, 0.01],
        [0.05, 0.03, -0.02, 0.04],
        [0.03, 0.05, 0.02, -0.01],
        [0.02, -0.02, 0.04, 0.03],
        [0.05, 0.01, -0.03, 0.04],
    ], dtype=np.float64)
    return SafetyStreamProfile(stream_names, calibration, fresh, near_threshold=0.25)


def degenerate_stream_profile() -> SafetyStreamProfile:
    """Tiny profile used by regression tests for boring traces."""
    stream_names = ("clearance_margin", "speed_margin", "corridor_margin")
    calibration = np.eye(3, dtype=np.float64) * 0.01
    fresh = np.eye(3, dtype=np.float64) * 0.005
    return SafetyStreamProfile(stream_names, calibration, fresh, near_threshold=0.1)


def make_synthetic_probe_bundle(
    candidate: str,
    *,
    length: int,
    seed: int,
    degenerate: bool = False,
) -> ProbeBundle:
    """Create deterministic derived-stream traces for tests and dry probes."""
    if candidate == "f1tenth":
        profile = f1tenth_stream_profile()
    elif candidate == "degenerate":
        profile = degenerate_stream_profile()
        degenerate = True
    else:
        profile = drone_stream_profile()

    rng = np.random.default_rng(seed)
    trace: list[SafetyStreamMeasurement] = []
    for t in range(length):
        if degenerate:
            true_values = np.full(profile.dimension, 3.0, dtype=np.float64)
            observed = true_values + rng.normal(0.0, 0.001, profile.dimension)
        else:
            phase = 2.0 * np.pi * t / max(length, 1)
            true_values = np.array([
                0.16 + 0.18 * np.sin(phase),
                0.15 + 0.14 * np.cos(1.7 * phase + 0.2),
                0.12 + 0.16 * np.sin(1.3 * phase + 1.0),
                0.20 + 0.10 * np.cos(0.9 * phase - 0.5),
                0.18 + 0.12 * np.sin(1.9 * phase + 0.6),
                0.14 + 0.17 * np.cos(1.1 * phase + 1.3),
            ], dtype=np.float64)[:profile.dimension]
            observed = true_values + rng.normal(0.0, 0.02, profile.dimension)
        trace.append(SafetyStreamMeasurement(
            time=float(t),
            values=tuple(float(v) for v in observed),
            true_values=tuple(float(v) for v in true_values),
            oracle_violation=bool(np.any(true_values < 0.0)),
            payload={"trace_source": "synthetic"},
        ))

    return ProbeBundle(
        candidate=candidate,
        monitor=SafetyStreamMonitor(profile),
        trace=tuple(trace),
        metadata={
            "candidate": candidate,
            "status": "available",
            "trace_source": "synthetic_degenerate" if degenerate else "synthetic_derived_streams",
            "length": length,
            "seed": seed,
        },
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_trace_csv(bundle: ProbeBundle, path: Path) -> None:
    rows = []
    for measurement in bundle.trace:
        row: dict[str, Any] = {
            "time": measurement.time,
            "oracle_violation": measurement.oracle_violation,
        }
        for name, value, true_value in zip(
            bundle.monitor.profile.stream_names,
            measurement.values,
            measurement.true_values,
        ):
            row[name] = value
            row[f"true_{name}"] = true_value
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _sidecar_status(
    root: Path | None = None,
    python_path: Path | None = None,
) -> dict[str, Any]:
    repo = _repo_root() if root is None else root
    python_path = python_path or (
        repo / "external" / "miniconda3" / "envs" / "pzr-safe-control-fw" / "bin" / "python"
    )
    gym_root = repo / "external" / "safe-control-gym"
    status: dict[str, Any] = {
        "python": str(python_path),
        "safe_control_gym_root": str(gym_root),
        "available": False,
    }
    if not python_path.exists() or not gym_root.exists():
        status["reason"] = "safe-control-gym sidecar path is missing"
        return status

    env = os.environ.copy()
    env["PYTHONPATH"] = str(gym_root) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        proc = subprocess.run(
            [
                str(python_path),
                "-c",
                "import safe_control_gym, pybullet; print('ok')",
            ],
            cwd=gym_root,
            env=env,
            check=False,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        status["reason"] = str(exc)
        return status

    status["returncode"] = proc.returncode
    status["stdout"] = proc.stdout.strip()
    status["stderr"] = proc.stderr.strip()
    status["available"] = proc.returncode == 0
    if proc.returncode != 0:
        status["reason"] = "sidecar import check failed"
    return status


def _load_level0_geometry(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root
    path = repo / "external" / "safe-control-gym" / "competition" / "level0.yaml"
    if not path.exists():
        return {"path": str(path), "available": False}
    with open(path) as f:
        config = yaml.safe_load(f)
    quad = config.get("quadrotor_config", {})
    return {
        "path": str(path),
        "available": True,
        "init_state": quad.get("init_state", {}),
        "gates": quad.get("gates", []),
        "obstacles": quad.get("obstacles", []),
        "stabilization_goal": quad.get("task_info", {}).get("stabilization_goal", []),
        "ctrl_freq": quad.get("ctrl_freq"),
        "episode_len_sec": quad.get("episode_len_sec"),
    }


def make_drone_probe_bundle(
    *,
    length: int,
    seed: int,
    trace_source: TraceSource = "auto",
    output: Path | None = None,
    controller_mode: DroneController = "sim",
    sidecar_python: Path | None = None,
    stress_randomize: bool = False,
) -> ProbeBundle | None:
    """Build a drone candidate from live safe-control-gym or proxy metadata."""
    status = _sidecar_status(python_path=sidecar_python)
    if not status.get("available", False):
        return None

    if trace_source in ("live", "auto"):
        live_bundle, live_metadata = _make_live_drone_probe_bundle(
            length=length,
            seed=seed,
            output=output,
            controller_mode=controller_mode,
            sidecar_python=sidecar_python,
            sidecar_status=status,
            stress_randomize=stress_randomize,
        )
        if live_bundle is not None:
            return live_bundle
        if trace_source == "live":
            return ProbeBundle(
                candidate="drone",
                monitor=SafetyStreamMonitor(drone_stream_profile()),
                trace=(),
                metadata=live_metadata,
            )

    geometry = _load_level0_geometry()
    bundle = make_synthetic_probe_bundle("drone", length=length, seed=seed)
    return ProbeBundle(
        candidate="drone",
        monitor=bundle.monitor,
        trace=tuple(_attach_geometry_payload(bundle.trace, geometry)),
        metadata={
            **bundle.metadata,
            "trace_source": "safe_control_gym_level0_geometry_proxy",
            "sidecar": status,
            "geometry": geometry,
            "stress_randomize": stress_randomize,
            "live_attempt": None if trace_source == "proxy" else live_metadata,
            "note": (
                "This is a derived-stream audit trace seeded from Level0 geometry, "
                "not a closed-loop safe-control-gym benchmark."
            ),
        },
    )


def _make_live_drone_probe_bundle(
    *,
    length: int,
    seed: int,
    output: Path | None,
    controller_mode: DroneController,
    sidecar_python: Path | None,
    sidecar_status: dict[str, Any],
    stress_randomize: bool,
) -> tuple[ProbeBundle | None, dict[str, Any]]:
    repo = _repo_root()
    python_path = sidecar_python or Path(sidecar_status["python"])
    safe_control_root = Path(sidecar_status["safe_control_gym_root"])
    config_path = safe_control_root / "competition" / "level0.yaml"
    raw_path = (output or Path("/tmp")) / "drone_raw_trace.jsonl"
    collector = repo / "tools" / "collect_safe_control_drone_trace.py"
    command = [
        str(python_path),
        str(collector),
        "--safe-control-root", str(safe_control_root),
        "--config", str(config_path),
        "--output", str(raw_path),
        "--length", str(length),
        "--seed", str(seed),
        "--controller-mode", controller_mode,
    ]
    if stress_randomize:
        command.append("--stress-randomize")
    metadata: dict[str, Any] = {
        "candidate": "drone",
        "status": "unavailable",
        "trace_source": "safe_control_gym_live_rollout",
        "sidecar": sidecar_status,
        "collector_command": command,
        "controller_mode": controller_mode,
        "stress_randomize": stress_randomize,
    }
    try:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            command,
            cwd=repo,
            check=False,
            text=True,
            capture_output=True,
            timeout=max(45, length * 3),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        metadata["reason"] = str(exc)
        return None, metadata

    metadata["returncode"] = proc.returncode
    metadata["stdout"] = proc.stdout[-4000:]
    metadata["stderr"] = proc.stderr[-4000:]
    metadata["raw_trace_path"] = str(raw_path)
    if proc.returncode != 0:
        metadata["reason"] = "safe-control-gym live collector failed"
        return None, metadata
    raw_records = _load_jsonl(raw_path)
    if not raw_records:
        metadata["reason"] = "safe-control-gym live collector produced no records"
        return None, metadata

    geometry = _load_level0_geometry()
    trace = tuple(_drone_records_to_measurements(raw_records, seed=seed))
    metadata.update({
        "status": "available",
        "length": len(trace),
        "seed": seed,
        "geometry": geometry,
        "stress_randomize": stress_randomize,
        "note": "Closed-loop safe-control-gym rollout converted to derived safety streams.",
    })
    return ProbeBundle(
        candidate="drone",
        monitor=SafetyStreamMonitor(drone_stream_profile()),
        trace=trace,
        metadata=metadata,
    ), metadata


def _attach_geometry_payload(
    trace: Sequence[SafetyStreamMeasurement],
    geometry: dict[str, Any],
) -> Iterable[SafetyStreamMeasurement]:
    for measurement in trace:
        payload = dict(measurement.payload or {})
        payload["geometry"] = {
            "gates": geometry.get("gates", []),
            "obstacles": geometry.get("obstacles", []),
        }
        yield SafetyStreamMeasurement(
            time=measurement.time,
            values=measurement.values,
            true_values=measurement.true_values,
            oracle_violation=measurement.oracle_violation,
            payload=payload,
        )


def _drone_records_to_measurements(
    records: Sequence[dict[str, Any]],
    *,
    seed: int,
) -> Iterable[SafetyStreamMeasurement]:
    rng = np.random.default_rng(seed)
    for record in records:
        true_values = _drone_true_safety_streams(record)
        observed = true_values + rng.normal(0.0, 0.02, true_values.size)
        info = record.get("info", {})
        collision = bool((info.get("collision") or [None, False])[1])
        constraint_violation = bool(info.get("constraint_violation", False))
        yield SafetyStreamMeasurement(
            time=float(record.get("time", record.get("step", 0.0))),
            values=tuple(float(v) for v in observed),
            true_values=tuple(float(v) for v in true_values),
            oracle_violation=bool(
                collision or constraint_violation or np.any(true_values < 0.0)
            ),
            payload={
                "trace_source": "safe_control_gym_live_rollout",
                "raw_step": int(record.get("step", 0)),
                "done": bool(record.get("done", False)),
                "raw_record": record,
            },
        )


def _drone_true_safety_streams(record: dict[str, Any]) -> NDArray[np.float64]:
    obs = np.asarray(record.get("obs", []), dtype=np.float64).ravel()
    if obs.size < 6:
        raise ValueError("drone record observation must contain at least 6 state values")
    pos = np.array([obs[0], obs[2], obs[4]], dtype=np.float64)
    vel = np.array([obs[1], obs[3], obs[5]], dtype=np.float64)
    gates = np.asarray(record.get("gates", []), dtype=np.float64)
    obstacles = np.asarray(record.get("obstacles", []), dtype=np.float64)
    info = record.get("info", {})

    obstacle_margin = _min_xy_distance_margin(pos, obstacles, threshold=0.45)
    gate_pos = np.asarray(info.get("current_target_gate_pos", []), dtype=np.float64).ravel()
    gate_yaw = 0.0
    if gate_pos.size >= 2:
        gate_xyz = np.array([
            gate_pos[0],
            gate_pos[1],
            _gate_height(float(info.get("current_target_gate_type", 0))),
        ], dtype=np.float64)
        if gate_pos.size >= 6:
            gate_yaw = float(gate_pos[5])
    elif gates.size:
        idx = int(info.get("current_target_gate_id", 0))
        idx = int(np.clip(idx, 0, len(gates) - 1))
        gate_xyz = np.array([gates[idx, 0], gates[idx, 1], _gate_height(gates[idx, 6])])
        if gates.shape[-1] >= 6:
            gate_yaw = float(gates[idx, 5])
    else:
        gate_xyz = pos

    gate_alignment_margin = _gate_alignment_margin(
        pos,
        gate_xyz,
        gate_yaw=gate_yaw,
        in_range=bool(info.get("current_target_gate_in_range", False)),
    )
    corridor_margin = _corridor_margin(pos, gates)
    altitude_low_margin = float(pos[2] - 0.10)
    altitude_high_margin = float(2.0 - pos[2])
    speed_margin = float(1.75 - np.linalg.norm(vel))
    return np.array([
        obstacle_margin,
        gate_alignment_margin,
        corridor_margin,
        altitude_low_margin,
        altitude_high_margin,
        speed_margin,
    ], dtype=np.float64)


def _gate_alignment_margin(
    pos: NDArray[np.float64],
    gate_xyz: NDArray[np.float64],
    *,
    gate_yaw: float,
    in_range: bool,
) -> float:
    rel_xy = pos[:2] - gate_xyz[:2]
    forward = np.array([np.cos(gate_yaw), np.sin(gate_yaw)], dtype=np.float64)
    lateral_axis = np.array([-np.sin(gate_yaw), np.cos(gate_yaw)], dtype=np.float64)
    lateral = abs(float(np.dot(rel_xy, lateral_axis)))
    vertical = abs(float(pos[2] - gate_xyz[2]))
    if not in_range and float(np.linalg.norm(rel_xy)) > 1.5:
        return 0.35
    return float(min(1.20 - lateral, 0.75 - vertical))


def _gate_height(gate_type: float) -> float:
    return 1.0 if int(gate_type) == 0 else 0.525


def _min_xy_distance_margin(
    pos: NDArray[np.float64],
    objects: NDArray[np.float64],
    *,
    threshold: float,
) -> float:
    if objects.size == 0:
        return 10.0
    obj = np.asarray(objects, dtype=np.float64).reshape((-1, objects.shape[-1]))
    dists = np.linalg.norm(obj[:, :2] - pos[:2], axis=1)
    return float(np.min(dists) - threshold)


def _corridor_margin(pos: NDArray[np.float64], gates: NDArray[np.float64]) -> float:
    if gates.size == 0:
        return 1.0
    gate_points = np.asarray(gates, dtype=np.float64).reshape((-1, gates.shape[-1]))[:, :2]
    points = np.vstack([np.array([-0.9, -2.9], dtype=np.float64), gate_points])
    point = pos[:2]
    best = min(_point_segment_distance(point, a, b) for a, b in zip(points[:-1], points[1:]))
    return float(0.65 - best)


def _point_segment_distance(
    point: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + t * ab)))


def _f1tenth_status(sidecar_python: Path | None = None) -> dict[str, Any]:
    if sidecar_python is None:
        available = importlib.util.find_spec("f110_gym") is not None
        status: dict[str, Any] = {
            "available": available,
            "package": "f110_gym",
            "python": "current",
        }
    else:
        status = {
            "available": False,
            "package": "f110_gym",
            "python": str(sidecar_python),
        }
        if sidecar_python.exists():
            proc = subprocess.run(
                [
                    str(sidecar_python),
                    "-c",
                    "import f110_gym, gym; print('ok')",
                ],
                check=False,
                text=True,
                capture_output=True,
                timeout=10,
            )
            status["returncode"] = proc.returncode
            status["stdout"] = proc.stdout.strip()
            status["stderr"] = proc.stderr.strip()
            status["available"] = proc.returncode == 0
        else:
            status["reason"] = "F1TENTH sidecar Python does not exist"
    if not status["available"] and "reason" not in status:
        status["reason"] = (
            "f110_gym is not installed in the selected Python environment"
        )
    return status


def make_f1tenth_probe_bundle(
    *,
    length: int,
    seed: int,
    trace_source: TraceSource = "auto",
    output: Path | None = None,
    sidecar_python: Path | None = None,
    map_name: str = "vegas",
    stress_randomize: bool = False,
) -> ProbeBundle | None:
    """Build an F1TENTH candidate if the optional package is installed."""
    status = _f1tenth_status(sidecar_python=sidecar_python)
    if not status.get("available", False) and trace_source == "live":
        return ProbeBundle(
            candidate="f1tenth",
            monitor=SafetyStreamMonitor(f1tenth_stream_profile()),
            trace=(),
            metadata={
                "candidate": "f1tenth",
                "status": "unavailable",
                "trace_source": "f1tenth_live_gym_rollout",
                "dependency": status,
                "stress_randomize": stress_randomize,
            },
        )
    if not status.get("available", False) and trace_source == "auto":
        return None
    live_error: dict[str, Any] | None = None
    if trace_source in ("live", "auto") and status.get("available", False):
        live_bundle = _make_live_f1tenth_bundle(
            length=length,
            seed=seed,
            output=output,
            status=status,
            sidecar_python=sidecar_python,
            map_name=map_name,
            stress_randomize=stress_randomize,
        )
        if live_bundle is not None:
            return live_bundle
        live_error = dict(status)
        if trace_source == "live":
            return ProbeBundle(
                candidate="f1tenth",
                monitor=SafetyStreamMonitor(f1tenth_stream_profile()),
                trace=(),
                metadata={
                    "candidate": "f1tenth",
                    "status": "unavailable",
                    "trace_source": "f1tenth_live_gym_rollout",
                    "dependency": status,
                    "stress_randomize": stress_randomize,
                },
            )
    if trace_source == "live":
        return None
    bundle = make_synthetic_probe_bundle("f1tenth", length=length, seed=seed)
    return ProbeBundle(
        candidate="f1tenth",
        monitor=bundle.monitor,
        trace=bundle.trace,
        metadata={
            **bundle.metadata,
            "trace_source": "f1tenth_derived_stream_proxy",
            "dependency": status,
            "live_attempt": live_error,
            "stress_randomize": stress_randomize,
            "note": (
                "The current probe checks the F1TENTH-derived stream design. "
                "The local package was present, but live Gym rollout capture "
                "did not initialize with the default map/API assumptions."
            ),
        },
    )


def _make_live_f1tenth_bundle(
    *,
    length: int,
    seed: int,
    output: Path | None,
    status: dict[str, Any],
    sidecar_python: Path | None,
    map_name: str,
    stress_randomize: bool,
) -> ProbeBundle | None:
    """Run the standalone F1TENTH collector and convert raw records."""
    repo = _repo_root()
    python_path = sidecar_python or Path(status.get("python", "python"))
    raw_path = (output or Path("/tmp")) / "f1tenth_raw_trace.jsonl"
    collector = repo / "tools" / "collect_f1tenth_trace.py"
    command = [
        str(python_path),
        str(collector),
        "--output", str(raw_path),
        "--length", str(length),
        "--seed", str(seed),
        "--map", map_name,
    ]
    if stress_randomize:
        command.append("--stress-randomize")
    try:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            command,
            cwd=repo,
            check=False,
            text=True,
            capture_output=True,
            timeout=max(45, length * 2),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        status["live_rollout_error"] = str(exc)
        return None
    status["collector_command"] = command
    status["collector_returncode"] = proc.returncode
    status["collector_stdout"] = proc.stdout[-4000:]
    status["collector_stderr"] = proc.stderr[-4000:]
    status["raw_trace_path"] = str(raw_path)
    if proc.returncode != 0:
        status["live_rollout_error"] = "F1TENTH collector failed"
        return None

    raw_records = _load_jsonl(raw_path)
    if not raw_records:
        status["live_rollout_error"] = "live rollout produced no observations"
        return None
    rng = np.random.default_rng(seed)
    trace = tuple(
        _f1tenth_measurement_from_record(record, float(i), rng)
        for i, record in enumerate(raw_records)
    )
    return ProbeBundle(
        candidate="f1tenth",
        monitor=SafetyStreamMonitor(f1tenth_stream_profile()),
        trace=tuple(trace),
        metadata={
            "candidate": "f1tenth",
            "status": "available",
            "trace_source": "f1tenth_live_gym_rollout",
            "dependency": status,
            "length": len(trace),
            "seed": seed,
            "map": map_name,
            "stress_randomize": stress_randomize,
        },
    )


def _f1tenth_measurement_from_record(
    record: dict[str, Any],
    time: float,
    rng: np.random.Generator,
) -> SafetyStreamMeasurement:
    obs = record.get("obs", {})
    true_values = _f1tenth_true_safety_streams(obs)
    observed = true_values + rng.normal(0.0, 0.02, true_values.size)
    collision = bool(np.asarray(obs.get("collisions", [False])).ravel()[0])
    return SafetyStreamMeasurement(
        time=float(record.get("time", time)),
        values=tuple(float(v) for v in observed),
        true_values=tuple(float(v) for v in true_values),
        oracle_violation=bool(collision or np.any(true_values < 0.0)),
        payload={
            "trace_source": "f1tenth_live_gym_rollout",
            "raw_step": int(record.get("step", 0)),
            "done": bool(record.get("done", False)),
            "raw_record": record,
        },
    )


def _f1tenth_true_safety_streams(obs: Any) -> NDArray[np.float64]:
    data = obs[0] if isinstance(obs, tuple) else obs
    if not isinstance(data, dict):
        raise ValueError("expected F1TENTH observation dictionary")

    scans = np.asarray(data.get("scans", []), dtype=np.float64)
    scan = scans[0] if scans.ndim == 2 else scans
    if scan.size == 0:
        raise ValueError("F1TENTH observation did not include LiDAR scans")
    finite_mask = np.isfinite(scan)
    if not np.any(finite_mask):
        raise ValueError("F1TENTH LiDAR scan contains no finite ranges")
    finite_scan = np.where(finite_mask, scan, np.nanmax(scan[finite_mask]))
    n = finite_scan.size
    front = finite_scan[n // 2 - max(n // 18, 1): n // 2 + max(n // 18, 1)]
    left = finite_scan[2 * n // 3: 5 * n // 6]
    right = finite_scan[n // 6: n // 3]

    speed = _obs_scalar(data, "linear_vels_x", default=0.0)
    yaw_rate = abs(_obs_scalar(data, "ang_vels_z", default=0.0))
    heading = abs(_wrap_angle(_obs_scalar(data, "poses_theta", default=0.0)))
    front_clearance = float(np.nanmin(front))
    side_clearance = float(min(np.nanmin(left), np.nanmin(right)))
    ttc = front_clearance / max(abs(speed), 0.2)

    return np.array([
        front_clearance - 1.2,
        side_clearance - 0.55,
        ttc - 1.0,
        side_clearance - 0.65,
        0.75 - heading,
        1.8 - abs(speed) - 0.5 * yaw_rate,
    ], dtype=np.float64)


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _obs_scalar(data: dict[str, Any], key: str, *, default: float) -> float:
    value = data.get(key)
    if value is None:
        return default
    arr = np.asarray(value, dtype=np.float64).ravel()
    if arr.size == 0:
        return default
    return float(arr[0])


def run_bundle(bundle: ProbeBundle, *, budget: int, seed: int) -> dict[str, Any]:
    """Run static reducers on a candidate bundle and score degeneracy."""
    methods = [
        StaticReductionPolicy(ProtectedReducer(base=reducer), _name=name)
        for name, reducer in STATIC_REDUCERS
    ]
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
    timeseries = results_to_dataframe(results)
    summary = summarize_results(results)
    report = score_candidate(bundle, summary, budget=budget)
    return {
        "bundle": bundle,
        "timeseries": timeseries,
        "summary": summary,
        "report": report,
    }


def score_candidate(
    bundle: ProbeBundle,
    summary: pd.DataFrame,
    *,
    budget: int,
) -> dict[str, Any]:
    """Classify whether a candidate is worth promoting to benchmark work."""
    widths = summary["mean_trigger_width"].to_numpy(dtype=np.float64)
    best = float(np.min(widths)) if widths.size else 0.0
    width_spread = float(np.max(widths) - np.min(widths)) if widths.size else 0.0
    relative_width_spread = width_spread / max(float(np.mean(widths)), 1e-12)
    reductions = summary["total_reductions"].to_numpy(dtype=np.float64)
    reduction_rate = float(np.mean(reductions) / max(len(bundle.trace), 1))
    differentiated_methods = int(np.sum(widths > best * 1.03 + 1e-9))

    true_values = np.array([m.true_values for m in bundle.trace], dtype=np.float64)
    min_abs_margin = np.min(np.abs(true_values), axis=1) if true_values.size else np.array([])
    near_threshold_fraction = float(
        np.mean(min_abs_margin <= bundle.monitor.profile.near_threshold)
    ) if min_abs_margin.size else 0.0
    oracle_violation_fraction = float(
        np.mean([m.oracle_violation for m in bundle.trace])
    ) if bundle.trace else 0.0
    budget_violations = int(summary["budget_violations"].sum())
    unsound_certificates = int(summary["unsound_certificates"].sum())
    boring = (
        relative_width_spread < 0.05
        or differentiated_methods < 2
        or reduction_rate < 0.2
        or near_threshold_fraction < 0.05
    )
    saturated = oracle_violation_fraction > 0.85

    if budget_violations or unsound_certificates:
        recommendation = "reject"
    elif boring:
        recommendation = "reject"
    elif saturated:
        recommendation = "revise"
    else:
        recommendation = "promote"

    return {
        "candidate": bundle.candidate,
        "recommendation": recommendation,
        "budget": budget,
        "length": len(bundle.trace),
        "width_spread": width_spread,
        "relative_width_spread": relative_width_spread,
        "differentiated_methods": differentiated_methods,
        "reduction_rate": reduction_rate,
        "near_threshold_fraction": near_threshold_fraction,
        "oracle_violation_fraction": oracle_violation_fraction,
        "budget_violations": budget_violations,
        "unsound_certificates": unsound_certificates,
    }


def trace_summary(bundle: ProbeBundle, *, seed: int) -> pd.DataFrame:
    values = np.array([m.true_values for m in bundle.trace], dtype=np.float64)
    if values.size == 0:
        return pd.DataFrame(columns=TRACE_SUMMARY_COLUMNS)
    rows = []
    for i, name in enumerate(bundle.monitor.profile.stream_names):
        stream = values[:, i]
        rows.append({
            "candidate": bundle.candidate,
            "seed": seed,
            "stream": name,
            "mean": float(np.mean(stream)),
            "min": float(np.min(stream)),
            "max": float(np.max(stream)),
            "near_threshold_fraction": float(
                np.mean(np.abs(stream) <= bundle.monitor.profile.near_threshold)
            ),
        })
    return pd.DataFrame(rows, columns=TRACE_SUMMARY_COLUMNS)


def _candidate_bundle(
    name: str,
    *,
    length: int,
    seed: int,
    trace_source: TraceSource,
    output: Path,
    drone_controller: DroneController,
    drone_sidecar_python: Path | None,
    f1tenth_sidecar_python: Path | None,
    f1tenth_map: str,
    stress_randomize: bool = False,
) -> tuple[ProbeBundle | None, dict[str, Any]]:
    if name == "drone":
        bundle = make_drone_probe_bundle(
            length=length,
            seed=seed,
            trace_source=trace_source,
            output=output,
            controller_mode=drone_controller,
            sidecar_python=drone_sidecar_python,
            stress_randomize=stress_randomize,
        )
        if bundle is None or not bundle.trace:
            return None, {
                "candidate": name,
                "status": "unavailable",
                "sidecar": _sidecar_status(python_path=drone_sidecar_python),
                "trace_source": trace_source,
            } if bundle is None else bundle.metadata
        return bundle, bundle.metadata
    if name == "f1tenth":
        bundle = make_f1tenth_probe_bundle(
            length=length,
            seed=seed,
            trace_source=trace_source,
            output=output,
            sidecar_python=f1tenth_sidecar_python,
            map_name=f1tenth_map,
            stress_randomize=stress_randomize,
        )
        if bundle is None or not bundle.trace:
            return None, {
                "candidate": name,
                "status": "unavailable",
                "dependency": _f1tenth_status(sidecar_python=f1tenth_sidecar_python),
                "trace_source": trace_source,
            } if bundle is None else bundle.metadata
        return bundle, bundle.metadata
    raise ValueError(f"unknown robotics probe candidate: {name}")


def run_probe(
    *,
    candidates: Sequence[str],
    length: int,
    seed: int,
    seeds: int = 1,
    warmup_steps: int = 0,
    budget: int,
    output: Path,
    trace_source: TraceSource = "auto",
    drone_controller: DroneController = "sim",
    drone_sidecar_python: Path | None = None,
    f1tenth_sidecar_python: Path | None = None,
    f1tenth_map: str = "vegas",
) -> dict[str, Any]:
    """Run candidate probes and write all artifacts."""
    if seeds < 1:
        raise ValueError("seeds must be at least 1")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    output.mkdir(parents=True, exist_ok=True)
    requested = ("drone", "f1tenth") if "all" in candidates else tuple(candidates)
    seed_values = tuple(range(seed, seed + seeds))

    metadata: dict[str, Any] = {
        "length": length,
        "seed": seed,
        "seeds": seeds,
        "seed_values": list(seed_values),
        "warmup_steps": warmup_steps,
        "budget": budget,
        "trace_source": trace_source,
        "drone_controller": drone_controller,
        "f1tenth_map": f1tenth_map,
        "requested_candidates": list(requested),
        "candidates": {},
    }
    summaries: list[pd.DataFrame] = []
    trace_summaries: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []

    for current_seed in seed_values:
        seed_output = output if seeds == 1 else output / f"seed_{current_seed}"
        seed_output.mkdir(parents=True, exist_ok=True)
        for candidate in requested:
            bundle, candidate_metadata = _candidate_bundle(
                candidate,
                length=length,
                seed=current_seed,
                trace_source=trace_source,
                output=seed_output,
                drone_controller=drone_controller,
                drone_sidecar_python=drone_sidecar_python,
                f1tenth_sidecar_python=f1tenth_sidecar_python,
                f1tenth_map=f1tenth_map,
            )
            candidate_metadata = {**candidate_metadata, "seed": current_seed}
            if seeds == 1:
                metadata["candidates"][candidate] = candidate_metadata
            else:
                aggregate_metadata = metadata["candidates"].setdefault(
                    candidate,
                    {"candidate": candidate, "per_seed": []},
                )
                aggregate_metadata["per_seed"].append(candidate_metadata)
                if candidate_metadata.get("status", "available") == "available":
                    aggregate_metadata["status"] = "available"
                else:
                    aggregate_metadata.setdefault("status", candidate_metadata.get("status", "unavailable"))
            if bundle is None:
                continue

            bundle = _trim_bundle(bundle, warmup_steps)
            if not bundle.trace:
                empty_metadata = {
                    **bundle.metadata,
                    "status": "unavailable",
                    "reason": "trace is empty after warmup trimming",
                    "seed": current_seed,
                }
                if seeds == 1:
                    metadata["candidates"][candidate] = empty_metadata
                else:
                    metadata["candidates"][candidate]["per_seed"][-1] = empty_metadata
                continue

            result = run_bundle(bundle, budget=budget, seed=current_seed)
            summary = result["summary"].copy()
            summary.insert(0, "candidate", candidate)
            if "seed" not in summary.columns:
                summary.insert(2, "seed", current_seed)
            summaries.append(summary)
            trace_summaries.append(trace_summary(bundle, seed=current_seed))
            report = {**result["report"], "seed": current_seed}
            reports.append(report)
            result["timeseries"].to_csv(seed_output / f"{candidate}_timeseries.csv", index=False)
            _write_trace_csv(bundle, seed_output / f"{candidate}_derived_streams.csv")

    method_scores = (
        pd.concat(summaries, ignore_index=True)
        if summaries else pd.DataFrame(columns=METHOD_SCORE_COLUMNS)
    )
    trace_scores = (
        pd.concat(trace_summaries, ignore_index=True)
        if trace_summaries else pd.DataFrame(columns=TRACE_SUMMARY_COLUMNS)
    )
    candidate_scores = (
        pd.DataFrame(reports)
        if reports else pd.DataFrame(columns=REPORT_COLUMNS)
    )
    method_score_summary = _aggregate_method_scores(method_scores)
    candidate_score_summary = _aggregate_candidate_scores(candidate_scores)
    method_scores.to_csv(output / "method_scores.csv", index=False)
    trace_scores.to_csv(output / "trace_summary.csv", index=False)
    candidate_scores.to_csv(output / "candidate_scores.csv", index=False)
    method_score_summary.to_csv(output / "method_score_summary.csv", index=False)
    candidate_score_summary.to_csv(output / "candidate_score_summary.csv", index=False)

    metadata["reports"] = reports
    save_json(metadata, output / "probe_metadata.json")
    _write_report_md(
        metadata,
        method_scores,
        candidate_scores,
        method_score_summary,
        output / "candidate_report.md",
    )
    return {
        "metadata": metadata,
        "method_scores": method_scores,
        "method_score_summary": method_score_summary,
        "candidate_scores": candidate_scores,
        "candidate_score_summary": candidate_score_summary,
        "trace_summary": trace_scores,
        "reports": reports,
    }


def _aggregate_method_scores(method_scores: pd.DataFrame) -> pd.DataFrame:
    if method_scores.empty:
        return pd.DataFrame()
    numeric = [
        "mean_trigger_width",
        "max_trigger_width",
        "mean_generator_count",
        "total_reductions",
        "budget_violations",
        "unsound_certificates",
        "false_positive_rate",
    ]
    present = [c for c in numeric if c in method_scores.columns]
    return (
        method_scores
        .groupby(["candidate", "method"])[present]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .pipe(_flatten_columns)
    )


def _aggregate_candidate_scores(candidate_scores: pd.DataFrame) -> pd.DataFrame:
    if candidate_scores.empty:
        return pd.DataFrame()
    numeric = [
        "relative_width_spread",
        "differentiated_methods",
        "reduction_rate",
        "near_threshold_fraction",
        "oracle_violation_fraction",
        "budget_violations",
        "unsound_certificates",
    ]
    present = [c for c in numeric if c in candidate_scores.columns]
    aggregate = (
        candidate_scores
        .groupby("candidate")[present]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .pipe(_flatten_columns)
    )
    recommendations = (
        candidate_scores
        .groupby(["candidate", "recommendation"], as_index=False)
        .size()
        .rename(columns={"size": "recommendation_count"})
    )
    return aggregate.merge(recommendations, on="candidate", how="left")


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple) else str(col)
        for col in df.columns
    ]
    return df


def _write_report_md(
    metadata: dict[str, Any],
    method_scores: pd.DataFrame,
    candidate_scores: pd.DataFrame,
    method_score_summary: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# Robotics Probe Report",
        "",
        f"- length: {metadata['length']}",
        f"- seed: {metadata['seed']}",
        f"- seeds: {metadata.get('seeds', 1)}",
        f"- warmup_steps: {metadata.get('warmup_steps', 0)}",
        f"- budget: {metadata['budget']}",
        "",
        "## Candidate Status",
    ]
    for name, candidate_metadata in metadata["candidates"].items():
        status = candidate_metadata.get("status", "available")
        lines.append(f"- {name}: {status}")
        note = candidate_metadata.get("note")
        if note:
            lines.append(f"  - {note}")
        if "per_seed" in candidate_metadata:
            lines.append(f"  - per-seed traces: {len(candidate_metadata['per_seed'])}")

    if metadata.get("reports"):
        lines.extend(["", "## Recommendations"])
        for report in metadata["reports"]:
            lines.append(
                f"- {report['candidate']} seed {report.get('seed', metadata['seed'])}: "
                f"{report['recommendation']} "
                f"(relative width spread={report['relative_width_spread']:.3f}, "
                f"reduction rate={report['reduction_rate']:.3f}, "
                f"oracle violations={report['oracle_violation_fraction']:.3f})"
            )

    if not candidate_scores.empty:
        lines.extend(["", "## Candidate Scores", ""])
        display_cols = [
            "candidate",
            "seed",
            "recommendation",
            "relative_width_spread",
            "near_threshold_fraction",
            "oracle_violation_fraction",
        ]
        lines.extend(_markdown_table(candidate_scores[display_cols]))

    if not method_score_summary.empty:
        lines.extend(["", "## Method Score Summary", ""])
        display_cols = [
            "candidate",
            "method",
            "mean_trigger_width_mean",
            "mean_trigger_width_std",
            "false_positive_rate_mean",
        ]
        lines.extend(_markdown_table(method_score_summary[display_cols]))

    if not method_scores.empty:
        lines.extend(["", "## Method Scores", ""])
        display_cols = [
            "candidate",
            "method",
            "seed",
            "mean_trigger_width",
            "total_reductions",
            "budget_violations",
            "unsound_certificates",
        ]
        lines.extend(_markdown_table(method_scores[display_cols]))

    path.write_text("\n".join(lines) + "\n")


def _markdown_table(df: pd.DataFrame) -> list[str]:
    headers = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(headers) + " |"]
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return rows


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit candidate robotics environments before benchmark promotion.",
    )
    parser.add_argument(
        "--candidate",
        choices=("drone", "f1tenth", "all"),
        default="all",
        help="Candidate environment to probe.",
    )
    parser.add_argument("--length", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of consecutive seeds to run, starting at --seed.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Drop this many initial measurements before scoring each trace.",
    )
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument(
        "--trace-source",
        choices=("proxy", "live", "auto"),
        default="auto",
        help="Use proxy traces, require live simulator traces, or try live then fall back.",
    )
    parser.add_argument(
        "--drone-controller",
        choices=("sim", "firmware"),
        default="sim",
        help="safe-control-gym controller path for live drone traces.",
    )
    parser.add_argument(
        "--drone-sidecar-python",
        type=Path,
        default=None,
        help="Python interpreter for the safe-control-gym sidecar.",
    )
    parser.add_argument(
        "--f1tenth-sidecar-python",
        type=Path,
        default=None,
        help="Python interpreter for the isolated F1TENTH sidecar.",
    )
    parser.add_argument("--f1tenth-map", default="vegas")
    parser.add_argument("--output", type=Path, default=Path("results/robotics-probe"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = run_probe(
        candidates=(args.candidate,),
        length=args.length,
        seed=args.seed,
        seeds=args.seeds,
        warmup_steps=args.warmup_steps,
        budget=args.budget,
        output=args.output,
        trace_source=args.trace_source,
        drone_controller=args.drone_controller,
        drone_sidecar_python=args.drone_sidecar_python,
        f1tenth_sidecar_python=args.f1tenth_sidecar_python,
        f1tenth_map=args.f1tenth_map,
    )
    reports = result["reports"]
    if reports:
        for report in reports:
            print(
                f"{report['candidate']} seed {report.get('seed', args.seed)}: "
                f"{report['recommendation']} "
                f"(relative width spread={report['relative_width_spread']:.3f})"
            )
    else:
        print("No available robotics candidates were probed.")
    print(f"Artifacts written to {args.output}")


if __name__ == "__main__":
    main()
