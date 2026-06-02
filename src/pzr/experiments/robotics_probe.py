"""Audit probes for candidate robotics evaluation environments.

This module is intentionally outside the default benchmark registry.  It is a
small diagnostic path for answering one question before adding a new scenario:
does the candidate produce non-degenerate zonotope-reduction behavior?
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

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
    "stream",
    "mean",
    "min",
    "max",
    "near_threshold_fraction",
]


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


def _sidecar_status(root: Path | None = None) -> dict[str, Any]:
    repo = _repo_root() if root is None else root
    python_path = repo / "external" / "miniconda3" / "envs" / "pzr-safe-control-fw" / "bin" / "python"
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


def make_drone_probe_bundle(*, length: int, seed: int) -> ProbeBundle | None:
    """Build a drone candidate from safe-control-gym Level0 geometry metadata."""
    status = _sidecar_status()
    if not status.get("available", False):
        return None

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
            "note": (
                "This is a derived-stream audit trace seeded from Level0 geometry, "
                "not a closed-loop safe-control-gym benchmark."
            ),
        },
    )


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


def _f1tenth_status() -> dict[str, Any]:
    available = importlib.util.find_spec("f110_gym") is not None
    status: dict[str, Any] = {"available": available, "package": "f110_gym"}
    if not available:
        status["reason"] = "f110_gym is not installed in the active Python environment"
    return status


def make_f1tenth_probe_bundle(*, length: int, seed: int) -> ProbeBundle | None:
    """Build an F1TENTH candidate if the optional package is installed."""
    status = _f1tenth_status()
    if not status.get("available", False):
        return None
    live_bundle = _make_live_f1tenth_bundle(length=length, seed=seed, status=status)
    if live_bundle is not None:
        return live_bundle
    bundle = make_synthetic_probe_bundle("f1tenth", length=length, seed=seed)
    return ProbeBundle(
        candidate="f1tenth",
        monitor=bundle.monitor,
        trace=bundle.trace,
        metadata={
            **bundle.metadata,
            "trace_source": "f1tenth_derived_stream_proxy",
            "dependency": status,
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
    status: dict[str, Any],
) -> ProbeBundle | None:
    """Best-effort live F1TENTH rollout for environments using the Gym API."""
    try:
        gym_module = importlib.import_module("gymnasium")
    except ImportError:
        try:
            gym_module = importlib.import_module("gym")
        except ImportError:
            status["live_rollout_error"] = "neither gymnasium nor gym is installed"
            return None

    try:
        env = gym_module.make("f110_gym:f110-v0", map="vegas", num_agents=1)
        reset_pose = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        reset_result = env.reset(poses=reset_pose)
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        rng = np.random.default_rng(seed)
        trace: list[SafetyStreamMeasurement] = []
        for t in range(length):
            measurement = _f1tenth_measurement_from_obs(obs, float(t), rng)
            trace.append(measurement)
            action = np.array([[0.0, 1.0]], dtype=np.float64)
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, _, terminated, truncated, _ = step_result
                done = bool(terminated or truncated)
            else:
                obs, _, done, _ = step_result
            if done:
                break
        close = getattr(env, "close", None)
        if callable(close):
            close()
    except Exception as exc:  # pragma: no cover - depends on optional package API.
        status["live_rollout_error"] = str(exc)
        return None

    if not trace:
        status["live_rollout_error"] = "live rollout produced no observations"
        return None
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
        },
    )


def _f1tenth_measurement_from_obs(
    obs: Any,
    time: float,
    rng: np.random.Generator,
) -> SafetyStreamMeasurement:
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
    heading = abs(_obs_scalar(data, "poses_theta", default=0.0))
    front_clearance = float(np.nanmin(front))
    side_clearance = float(min(np.nanmin(left), np.nanmin(right)))
    ttc = front_clearance / max(abs(speed), 0.2)

    true_values = np.array([
        front_clearance - 0.6,
        side_clearance - 0.35,
        ttc - 0.8,
        side_clearance - 0.45,
        0.35 - heading,
        1.5 - abs(speed) - 0.4 * yaw_rate,
    ], dtype=np.float64)
    observed = true_values + rng.normal(0.0, 0.02, true_values.size)
    collision = bool(np.asarray(data.get("collisions", [False])).ravel()[0])
    return SafetyStreamMeasurement(
        time=time,
        values=tuple(float(v) for v in observed),
        true_values=tuple(float(v) for v in true_values),
        oracle_violation=bool(collision or np.any(true_values < 0.0)),
        payload={"trace_source": "f1tenth_live_gym_rollout"},
    )


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


def trace_summary(bundle: ProbeBundle) -> pd.DataFrame:
    values = np.array([m.true_values for m in bundle.trace], dtype=np.float64)
    if values.size == 0:
        return pd.DataFrame(columns=TRACE_SUMMARY_COLUMNS)
    rows = []
    for i, name in enumerate(bundle.monitor.profile.stream_names):
        stream = values[:, i]
        rows.append({
            "candidate": bundle.candidate,
            "stream": name,
            "mean": float(np.mean(stream)),
            "min": float(np.min(stream)),
            "max": float(np.max(stream)),
            "near_threshold_fraction": float(
                np.mean(np.abs(stream) <= bundle.monitor.profile.near_threshold)
            ),
        })
    return pd.DataFrame(rows, columns=TRACE_SUMMARY_COLUMNS)


def _candidate_bundle(name: str, *, length: int, seed: int) -> tuple[ProbeBundle | None, dict[str, Any]]:
    if name == "drone":
        bundle = make_drone_probe_bundle(length=length, seed=seed)
        if bundle is None:
            return None, {
                "candidate": name,
                "status": "unavailable",
                "sidecar": _sidecar_status(),
            }
        return bundle, bundle.metadata
    if name == "f1tenth":
        bundle = make_f1tenth_probe_bundle(length=length, seed=seed)
        if bundle is None:
            return None, {
                "candidate": name,
                "status": "unavailable",
                "dependency": _f1tenth_status(),
            }
        return bundle, bundle.metadata
    raise ValueError(f"unknown robotics probe candidate: {name}")


def run_probe(
    *,
    candidates: Sequence[str],
    length: int,
    seed: int,
    budget: int,
    output: Path,
) -> dict[str, Any]:
    """Run candidate probes and write all artifacts."""
    output.mkdir(parents=True, exist_ok=True)
    requested = ("drone", "f1tenth") if "all" in candidates else tuple(candidates)

    metadata: dict[str, Any] = {
        "length": length,
        "seed": seed,
        "budget": budget,
        "requested_candidates": list(requested),
        "candidates": {},
    }
    summaries: list[pd.DataFrame] = []
    trace_summaries: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []

    for candidate in requested:
        bundle, candidate_metadata = _candidate_bundle(candidate, length=length, seed=seed)
        metadata["candidates"][candidate] = candidate_metadata
        if bundle is None:
            continue

        result = run_bundle(bundle, budget=budget, seed=seed)
        summary = result["summary"].copy()
        summary.insert(0, "candidate", candidate)
        summaries.append(summary)
        trace_summaries.append(trace_summary(bundle))
        reports.append(result["report"])
        result["timeseries"].to_csv(output / f"{candidate}_timeseries.csv", index=False)

    method_scores = (
        pd.concat(summaries, ignore_index=True)
        if summaries else pd.DataFrame(columns=METHOD_SCORE_COLUMNS)
    )
    trace_scores = (
        pd.concat(trace_summaries, ignore_index=True)
        if trace_summaries else pd.DataFrame(columns=TRACE_SUMMARY_COLUMNS)
    )
    method_scores.to_csv(output / "method_scores.csv", index=False)
    trace_scores.to_csv(output / "trace_summary.csv", index=False)

    metadata["reports"] = reports
    save_json(metadata, output / "probe_metadata.json")
    _write_report_md(metadata, method_scores, output / "candidate_report.md")
    return {
        "metadata": metadata,
        "method_scores": method_scores,
        "trace_summary": trace_scores,
        "reports": reports,
    }


def _write_report_md(metadata: dict[str, Any], method_scores: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Robotics Probe Report",
        "",
        f"- length: {metadata['length']}",
        f"- seed: {metadata['seed']}",
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

    if metadata.get("reports"):
        lines.extend(["", "## Recommendations"])
        for report in metadata["reports"]:
            lines.append(
                f"- {report['candidate']}: {report['recommendation']} "
                f"(relative width spread={report['relative_width_spread']:.3f}, "
                f"reduction rate={report['reduction_rate']:.3f})"
            )

    if not method_scores.empty:
        lines.extend(["", "## Method Scores", ""])
        display_cols = [
            "candidate",
            "method",
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
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results/robotics-probe"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = run_probe(
        candidates=(args.candidate,),
        length=args.length,
        seed=args.seed,
        budget=args.budget,
        output=args.output,
    )
    reports = result["reports"]
    if reports:
        for report in reports:
            print(
                f"{report['candidate']}: {report['recommendation']} "
                f"(relative width spread={report['relative_width_spread']:.3f})"
            )
    else:
        print("No available robotics candidates were probed.")
    print(f"Artifacts written to {args.output}")


if __name__ == "__main__":
    main()
