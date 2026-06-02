"""2D Matplotlib visualization for MuJoCo robot-arm monitor traces."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pzr-matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from numpy.typing import NDArray

from pzr.envs.base import NoisySensorModel
from pzr.envs.robot_arm import (
    FORBIDDEN_ZONE_CENTER,
    FORBIDDEN_ZONE_HALF,
    LINK_LENGTHS,
    NUM_JOINTS,
    TOTAL_REACH,
    forward_kinematics,
)
from pzr.envs.robot_arm_monitor import (
    RobotArmMeasurement,
    RobotArmMonitor,
    RobotArmTraceRecord,
    generate_robot_arm_trace_records,
)
from pzr.experiments.benchmark import default_methods
from pzr.monitoring.base import MonitorState, Verdict
from pzr.monitoring.triggers import evaluate_triggers
from pzr.zonotope.core import Zonotope


DEFAULT_BIAS_BOUND = np.array([0.02, 0.02, 0.02, 0.01, 0.01, 0.01], dtype=np.float64)
DEFAULT_NOISE_BOUND = np.array([0.01, 0.01, 0.01, 0.005, 0.005, 0.005], dtype=np.float64)
TRACE_KINDS = ("benchmark", "paper")
TRIGGER_VIEW_MARGIN = 0.22
PAPER_TRACE_JOINT_WAYPOINTS = np.array([
    [0.02, 0.72, -0.95],
    [-0.04, 0.66, -1.10],
    [-0.12, 0.70, -1.28],
    [-0.20, 0.82, -1.42],
    [-0.14, 0.92, -1.60],
    [-0.02, 0.82, -1.52],
    [0.08, 0.66, -1.30],
], dtype=np.float64)


@dataclass(frozen=True)
class VisualizationFrame:
    """One replay frame after monitor update and optional reduction."""

    record: RobotArmTraceRecord
    monitor_state: MonitorState
    trigger_zonotope: Zonotope
    trigger_lower: NDArray[np.float64]
    trigger_upper: NDArray[np.float64]
    verdicts: tuple[Verdict, ...]
    reduced: bool
    reducer_used: str
    generator_count: int


@dataclass(frozen=True)
class TraceQuality:
    """Diagnostics describing the physical trace used for visualization."""

    trace_model: str
    ee_path_length: float
    action_saturation_fraction: float
    waypoints_reached: int
    mean_target_error: float
    max_target_error: float


def joint_positions(
    angles: NDArray[np.float64],
    link_lengths: tuple[float, ...] = LINK_LENGTHS,
) -> NDArray[np.float64]:
    """Return base, joint, and end-effector positions for a planar arm."""
    a = np.asarray(angles, dtype=np.float64).ravel()
    if a.size != len(link_lengths):
        raise ValueError(f"expected {len(link_lengths)} joint angles, got {a.size}")

    positions = np.zeros((len(link_lengths) + 1, 2), dtype=np.float64)
    cumulative = np.cumsum(a)
    for i, length in enumerate(link_lengths):
        positions[i + 1, 0] = positions[i, 0] + length * np.cos(cumulative[i])
        positions[i + 1, 1] = positions[i, 1] + length * np.sin(cumulative[i])
    return positions


def zonotope_vertices_2d(zonotope: Zonotope) -> NDArray[np.float64]:
    """Return exact boundary vertices for a 2D zonotope."""
    if zonotope.dimension != 2:
        raise ValueError(f"expected a 2D zonotope, got dimension {zonotope.dimension}")

    generators = zonotope.generators.T
    norms = np.linalg.norm(generators, axis=1)
    generators = generators[norms > 1e-14]
    if generators.size == 0:
        return zonotope.center.reshape(1, 2).copy()

    oriented = generators.copy()
    flip = (oriented[:, 1] < 0.0) | (
        (np.abs(oriented[:, 1]) <= 1e-14) & (oriented[:, 0] < 0.0)
    )
    oriented[flip] *= -1.0
    angles = np.arctan2(oriented[:, 1], oriented[:, 0])
    order = np.argsort(angles, kind="mergesort")
    sorted_generators = oriented[order]

    vertex = zonotope.center - np.sum(sorted_generators, axis=0)
    vertices = [vertex.copy()]
    for generator in sorted_generators:
        vertex = vertex + 2.0 * generator
        vertices.append(vertex.copy())
    for generator in sorted_generators:
        vertex = vertex - 2.0 * generator
        vertices.append(vertex.copy())
    return np.array(vertices, dtype=np.float64)


def interval_rectangle(
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return closed rectangle vertices for 2D interval bounds."""
    lo = np.asarray(lower, dtype=np.float64).ravel()
    hi = np.asarray(upper, dtype=np.float64).ravel()
    if lo.size != 2 or hi.size != 2:
        raise ValueError("interval rectangle expects 2D bounds")
    return np.array([
        [lo[0], lo[1]],
        [hi[0], lo[1]],
        [hi[0], hi[1]],
        [lo[0], hi[1]],
        [lo[0], lo[1]],
    ], dtype=np.float64)


def replay_robot_arm_visualization(
    *,
    method: str,
    trace: str = "benchmark",
    seed: int = 0,
    length: int = 200,
    budget: int = 10,
    horizon: int = 4,
    beam_width: int = 4,
) -> list[VisualizationFrame]:
    """Replay a robot-arm trace and collect frame state for visualization."""
    if trace not in TRACE_KINDS:
        raise ValueError(f"unknown trace {trace!r}; expected one of {', '.join(TRACE_KINDS)}")
    monitor = _make_monitor()
    policy = _policy_by_name(monitor, method, budget, horizon, beam_width)
    records = _trace_records(trace, length=length, seed=seed)

    state = monitor.initial_state()
    history = []
    frames: list[VisualizationFrame] = []

    for record in records:
        result = monitor.step(state, record.measurement)
        state = result.state
        history.append(record.measurement)

        reduced = False
        reducer_used = ""
        verdicts = result.verdicts
        if policy is not None and state.zonotope.generator_count > budget:
            decision = policy.decide(monitor, state, history, budget)
            state = decision.state
            reduced = True
            reducer_used = decision.reducer_name
            verdicts = evaluate_triggers(monitor.trigger_zonotope(state), monitor.triggers)

        trigger_z = monitor.trigger_zonotope(state)
        lower, upper = trigger_z.interval_bounds()
        frames.append(VisualizationFrame(
            record=record,
            monitor_state=state,
            trigger_zonotope=trigger_z,
            trigger_lower=lower,
            trigger_upper=upper,
            verdicts=verdicts,
            reduced=reduced,
            reducer_used=reducer_used,
            generator_count=state.zonotope.generator_count,
        ))

    return frames


def render_robot_arm_animation(
    *,
    output: Path,
    method: str,
    trace: str = "benchmark",
    seed: int = 0,
    length: int = 200,
    budget: int = 10,
    horizon: int = 4,
    beam_width: int = 4,
    fps: int = 12,
    stride: int = 2,
    dpi: int = 140,
    save_gif: bool = True,
) -> dict[str, Path]:
    """Render a GIF plus paper-friendly stills/storyboard for the robot arm."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frames = replay_robot_arm_visualization(
        method=method,
        trace=trace,
        seed=seed,
        length=length,
        budget=budget,
        horizon=horizon,
        beam_width=beam_width,
    )
    if not frames:
        raise ValueError("cannot render an empty robot-arm trace")
    quality = _trace_quality(frames, trace=trace)

    stride = max(int(stride), 1)
    fps = max(int(fps), 1)
    frame_indices = list(range(0, len(frames), stride))
    if frame_indices[-1] != len(frames) - 1:
        frame_indices.append(len(frames) - 1)

    prefix = f"robot_arm_{trace}_{method}_seed{seed}"
    artifacts: dict[str, Path] = {}
    physical_limits = _physical_view_limits(frames)
    trigger_limits = _trigger_view_limits()

    still_indices = {
        "first": 0,
        "middle": len(frames) // 2,
        "last": len(frames) - 1,
    }
    for label, index in still_indices.items():
        base = output / f"{prefix}_{label}"
        _save_static_frame(
            frames,
            index,
            base,
            method=method,
            trace=trace,
            seed=seed,
            physical_limits=physical_limits,
            trigger_limits=trigger_limits,
            budget=budget,
            dpi=dpi,
        )
        artifacts[f"{label}_png"] = base.with_suffix(".png")
        artifacts[f"{label}_pdf"] = base.with_suffix(".pdf")

    storyboard_base = output / f"{prefix}_storyboard"
    _save_storyboard(
        frames,
        storyboard_base,
        method=method,
        trace=trace,
        seed=seed,
        physical_limits=physical_limits,
        trigger_limits=trigger_limits,
        budget=budget,
        dpi=dpi,
    )
    artifacts["storyboard_png"] = storyboard_base.with_suffix(".png")
    artifacts["storyboard_pdf"] = storyboard_base.with_suffix(".pdf")

    if save_gif:
        gif_path = output / f"{prefix}.gif"
        _save_gif(
            frames,
            frame_indices,
            gif_path,
            method=method,
            trace=trace,
            seed=seed,
            fps=fps,
            physical_limits=physical_limits,
            trigger_limits=trigger_limits,
            budget=budget,
            dpi=dpi,
        )
        artifacts["gif"] = gif_path

    metadata_path = output / f"{prefix}_metadata.json"
    metadata = _metadata(
        frames,
        method=method,
        trace=trace,
        seed=seed,
        length=length,
        budget=budget,
        horizon=horizon,
        beam_width=beam_width,
        fps=fps,
        stride=stride,
        rendered_frame_count=len(frame_indices),
        quality=quality,
        physical_limits=physical_limits,
        trigger_limits=trigger_limits,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    artifacts["metadata"] = metadata_path
    return artifacts


def _make_monitor() -> RobotArmMonitor:
    return RobotArmMonitor(
        noise_model=NoisySensorModel(
            bias_bound=DEFAULT_BIAS_BOUND.copy(),
            noise_bound=DEFAULT_NOISE_BOUND.copy(),
        ),
    )


def _trace_records(
    trace: str,
    *,
    length: int,
    seed: int,
) -> tuple[RobotArmTraceRecord, ...]:
    if trace == "benchmark":
        return generate_robot_arm_trace_records(length, seed=seed)
    if trace == "paper":
        return _generate_paper_trace_records(length, seed=seed)
    raise ValueError(f"unknown trace {trace!r}; expected one of {', '.join(TRACE_KINDS)}")


def _generate_paper_trace_records(
    length: int,
    *,
    seed: int = 0,
) -> tuple[RobotArmTraceRecord, ...]:
    """Generate a deterministic scripted trace near the EE trigger region."""
    if length <= 0:
        return ()

    rng = np.random.default_rng(seed)
    noise_model = NoisySensorModel(
        bias_bound=DEFAULT_BIAS_BOUND.copy(),
        noise_bound=DEFAULT_NOISE_BOUND.copy(),
    )
    noise_model.reset(rng)

    records: list[RobotArmTraceRecord] = []
    previous_angles: NDArray[np.float64] | None = None
    previous_velocities = np.zeros(NUM_JOINTS, dtype=np.float64)

    for t in range(length):
        angles, target = _scripted_paper_angles(t, length)
        if previous_angles is None:
            velocities = np.zeros(NUM_JOINTS, dtype=np.float64)
        else:
            velocities = angles - previous_angles
        true_state = np.concatenate([angles, velocities]).astype(np.float64)
        measurement = _measurement_from_state(true_state, noise_model, rng, float(t))
        ee_pos = forward_kinematics(angles)
        action = velocities - previous_velocities
        records.append(RobotArmTraceRecord(
            time=float(t),
            true_state=true_state.copy(),
            measurement=measurement,
            target_angles=target.copy(),
            action=action.copy(),
            ee_pos=ee_pos.copy(),
            in_forbidden_zone=bool(_point_in_trigger_region(ee_pos)),
            done=False,
            episode_id=0,
        ))
        previous_angles = angles
        previous_velocities = velocities

    return tuple(records)


def _scripted_paper_angles(
    step: int,
    length: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if length <= 1:
        return PAPER_TRACE_JOINT_WAYPOINTS[0].copy(), PAPER_TRACE_JOINT_WAYPOINTS[0].copy()
    progress = step / float(length - 1)
    segment_position = progress * (len(PAPER_TRACE_JOINT_WAYPOINTS) - 1)
    segment = min(int(np.floor(segment_position)), len(PAPER_TRACE_JOINT_WAYPOINTS) - 2)
    local = segment_position - segment
    smooth = local * local * (3.0 - 2.0 * local)
    start = PAPER_TRACE_JOINT_WAYPOINTS[segment]
    target = PAPER_TRACE_JOINT_WAYPOINTS[segment + 1]
    angles = (1.0 - smooth) * start + smooth * target
    return angles.astype(np.float64), target.copy()


def _point_in_trigger_region(point: NDArray[np.float64]) -> bool:
    lower = FORBIDDEN_ZONE_CENTER - FORBIDDEN_ZONE_HALF
    upper = FORBIDDEN_ZONE_CENTER + FORBIDDEN_ZONE_HALF
    return bool(np.all(point >= lower) and np.all(point <= upper))


def _measurement_from_state(
    true_state: NDArray[np.float64],
    noise_model: NoisySensorModel,
    rng: np.random.Generator,
    time: float,
) -> RobotArmMeasurement:
    noisy = noise_model.observe(true_state, rng)
    return RobotArmMeasurement(
        time=time,
        joint_angles=(float(noisy[0]), float(noisy[1]), float(noisy[2])),
        joint_velocities=(float(noisy[3]), float(noisy[4]), float(noisy[5])),
    )


def _policy_by_name(
    monitor: RobotArmMonitor,
    method: str,
    budget: int,
    horizon: int,
    beam_width: int,
):
    if method == "none":
        return None
    methods = default_methods(
        monitor,
        budget=budget,
        horizon=horizon,
        beam_width=beam_width,
    )
    for spec in methods:
        if spec.name == method:
            return spec.policy
    valid = ["none"] + [spec.name for spec in methods]
    raise ValueError(f"unknown method {method!r}; expected one of {', '.join(valid)}")


def _save_gif(
    frames: Sequence[VisualizationFrame],
    frame_indices: Sequence[int],
    path: Path,
    *,
    method: str,
    trace: str,
    seed: int,
    fps: int,
    physical_limits: tuple[float, float, float, float],
    trigger_limits: tuple[float, float, float, float],
    budget: int,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(9.4, 5.8))

    def draw(render_index: int):
        frame_index = frame_indices[render_index]
        _draw_frame_layout(
            fig,
            frames,
            frame_index,
            method=method,
            trace=trace,
            seed=seed,
            physical_limits=physical_limits,
            trigger_limits=trigger_limits,
            budget=budget,
            compact=False,
        )
        return []

    animation = mpl_animation.FuncAnimation(
        fig,
        draw,
        frames=len(frame_indices),
        interval=1000.0 / fps,
        blit=False,
    )
    writer = mpl_animation.PillowWriter(fps=fps)
    animation.save(path, writer=writer, dpi=dpi)
    plt.close(fig)


def _save_static_frame(
    frames: Sequence[VisualizationFrame],
    frame_index: int,
    base: Path,
    *,
    method: str,
    trace: str,
    seed: int,
    physical_limits: tuple[float, float, float, float],
    trigger_limits: tuple[float, float, float, float],
    budget: int,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(9.4, 5.8))
    _draw_frame_layout(
        fig,
        frames,
        frame_index,
        method=method,
        trace=trace,
        seed=seed,
        physical_limits=physical_limits,
        trigger_limits=trigger_limits,
        budget=budget,
        compact=False,
    )
    _save_figure(fig, base, dpi=dpi)


def _save_storyboard(
    frames: Sequence[VisualizationFrame],
    base: Path,
    *,
    method: str,
    trace: str,
    seed: int,
    physical_limits: tuple[float, float, float, float],
    trigger_limits: tuple[float, float, float, float],
    budget: int,
    dpi: int,
) -> None:
    count = min(6, len(frames))
    indices = np.linspace(0, len(frames) - 1, count, dtype=int)
    fig = plt.figure(figsize=(10.8, 2.9 * count + 1.5))
    gs = fig.add_gridspec(count + 1, 2, height_ratios=[1.0] * count + [0.42])
    for row, frame_index in enumerate(indices):
        physical_ax = fig.add_subplot(gs[row, 0])
        trigger_ax = fig.add_subplot(gs[row, 1])
        _draw_physical_panel(
            physical_ax,
            frames,
            int(frame_index),
            method=method,
            trace=trace,
            seed=seed,
            physical_limits=physical_limits,
            compact=True,
        )
        _draw_trigger_panel(
            trigger_ax,
            frames[int(frame_index)],
            trigger_limits=trigger_limits,
            compact=True,
        )
    timeline_ax = fig.add_subplot(gs[count, :])
    _draw_storyboard_timeline(timeline_ax, frames, budget=budget)
    fig.tight_layout()
    _save_figure(fig, base, dpi=dpi)


def _save_figure(fig: Figure, base: Path, *, dpi: int) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=dpi)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def _trace_quality(
    frames: Sequence[VisualizationFrame],
    *,
    trace: str,
) -> TraceQuality:
    ee = np.array([frame.record.ee_pos for frame in frames], dtype=np.float64)
    path_length = 0.0
    if len(ee) > 1:
        path_length = float(np.sum(np.linalg.norm(np.diff(ee, axis=0), axis=1)))
    actions = np.array([frame.record.action for frame in frames], dtype=np.float64)
    if actions.size:
        action_saturation = float(np.mean(np.any(np.isclose(np.abs(actions), 1.0), axis=1)))
    else:
        action_saturation = 0.0
    target_errors = np.array([
        np.linalg.norm(frame.record.true_state[:NUM_JOINTS] - frame.record.target_angles)
        for frame in frames
    ], dtype=np.float64)
    waypoints_reached = 0
    if trace == "paper":
        reached = []
        for waypoint in PAPER_TRACE_JOINT_WAYPOINTS[1:]:
            reached.append(np.min([
                np.linalg.norm(frame.record.true_state[:NUM_JOINTS] - waypoint)
                for frame in frames
            ]) < 0.08)
        waypoints_reached = int(sum(reached))
    return TraceQuality(
        trace_model=(
            "scripted_kinematic_explanatory"
            if trace == "paper" else "mujoco_benchmark"
        ),
        ee_path_length=path_length,
        action_saturation_fraction=action_saturation,
        waypoints_reached=waypoints_reached,
        mean_target_error=float(np.mean(target_errors)) if target_errors.size else 0.0,
        max_target_error=float(np.max(target_errors)) if target_errors.size else 0.0,
    )


def _physical_view_limits(frames: Sequence[VisualizationFrame]) -> tuple[float, float, float, float]:
    points: list[NDArray[np.float64]] = [
        np.array([[0.0, 0.0], [TOTAL_REACH, 0.0]], dtype=np.float64),
        interval_rectangle(
            FORBIDDEN_ZONE_CENTER - FORBIDDEN_ZONE_HALF,
            FORBIDDEN_ZONE_CENTER + FORBIDDEN_ZONE_HALF,
        ),
    ]
    for frame in frames:
        record = frame.record
        points.append(joint_positions(record.true_state[:NUM_JOINTS]))
        measured_angles = np.array(record.measurement.joint_angles, dtype=np.float64)
        points.append(joint_positions(measured_angles))
        points.append(record.ee_pos.reshape(1, 2))

    all_points = np.vstack(points)
    lo = np.min(all_points, axis=0)
    hi = np.max(all_points, axis=0)
    center = 0.5 * (lo + hi)
    span = np.maximum(hi - lo, 0.35)
    width = float(max(span[0], span[1]) + 0.16)
    return (
        float(center[0] - 0.5 * width),
        float(center[0] + 0.5 * width),
        float(center[1] - 0.5 * width),
        float(center[1] + 0.5 * width),
    )


def _trigger_view_limits() -> tuple[float, float, float, float]:
    lower = FORBIDDEN_ZONE_CENTER - FORBIDDEN_ZONE_HALF - TRIGGER_VIEW_MARGIN
    upper = FORBIDDEN_ZONE_CENTER + FORBIDDEN_ZONE_HALF + TRIGGER_VIEW_MARGIN
    center = 0.5 * (lower + upper)
    span = upper - lower
    width = float(max(span[0], span[1]))
    return (
        float(center[0] - 0.5 * width),
        float(center[0] + 0.5 * width),
        float(center[1] - 0.5 * width),
        float(center[1] + 0.5 * width),
    )


def _draw_storyboard_timeline(
    ax: Axes,
    frames: Sequence[VisualizationFrame],
    *,
    budget: int,
) -> None:
    steps = np.array([frame.record.time for frame in frames], dtype=np.float64)
    widths = np.array([
        np.sum(frame.trigger_upper - frame.trigger_lower) for frame in frames
    ], dtype=np.float64)
    generators = np.array([frame.generator_count for frame in frames], dtype=np.float64)
    ax.plot(steps, widths, color="#b45309", linewidth=1.4, label="trigger width")
    ax.set_ylabel("width", color="#b45309", fontsize=8)
    ax.tick_params(axis="y", labelcolor="#b45309", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("step", fontsize=8)
    ax.grid(color="#e5e7eb", linewidth=0.7)

    ax2 = ax.twinx()
    ax2.plot(steps, generators, color="#111827", linewidth=1.2, label="generators")
    ax2.axhline(budget, color="#111827", linewidth=0.9, linestyle=":", alpha=0.75)
    ax2.set_ylabel("generators", color="#111827", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="#111827", labelsize=8)

    reduction_steps = [frame.record.time for frame in frames if frame.reduced]
    if reduction_steps:
        ymin, ymax = ax.get_ylim()
        ax.vlines(
            reduction_steps,
            ymin=ymin,
            ymax=ymax,
            colors="#2563eb",
            linewidth=0.7,
            alpha=0.25,
        )
        ax.set_ylim(ymin, ymax)
    ax.set_title("Trigger width, generator budget, and reduction points", fontsize=9)


def _draw_frame_layout(
    fig: Figure,
    frames: Sequence[VisualizationFrame],
    frame_index: int,
    *,
    method: str,
    trace: str,
    seed: int,
    physical_limits: tuple[float, float, float, float],
    trigger_limits: tuple[float, float, float, float],
    budget: int,
    compact: bool,
) -> None:
    fig.clear()
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.28], width_ratios=[1.18, 1.0])
    physical_ax = fig.add_subplot(gs[0, 0])
    trigger_ax = fig.add_subplot(gs[0, 1])
    timeline_ax = fig.add_subplot(gs[1, :])
    _draw_physical_panel(
        physical_ax,
        frames,
        frame_index,
        method=method,
        trace=trace,
        seed=seed,
        physical_limits=physical_limits,
        compact=compact,
    )
    _draw_trigger_panel(
        trigger_ax,
        frames[frame_index],
        trigger_limits=trigger_limits,
        compact=compact,
    )
    _draw_storyboard_timeline(timeline_ax, frames[:frame_index + 1], budget=budget)
    fig.tight_layout()


def _draw_physical_panel(
    ax: Axes,
    frames: Sequence[VisualizationFrame],
    frame_index: int,
    *,
    method: str,
    trace: str,
    seed: int,
    physical_limits: tuple[float, float, float, float],
    compact: bool,
) -> None:
    frame = frames[frame_index]
    record = frame.record
    ax.clear()
    ax.set_aspect("equal", adjustable="box")
    xmin, xmax, ymin, ymax = physical_limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(color="#e5e7eb", linewidth=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    _draw_trigger_region(ax)
    trail = np.array([f.record.ee_pos for f in frames[:frame_index + 1]], dtype=np.float64)
    if trail.size:
        ax.plot(
            trail[:, 0], trail[:, 1],
            color="#334155",
            alpha=0.45,
            linewidth=1.5 if not compact else 1.1,
            zorder=2,
        )

    true_positions = joint_positions(record.true_state[:NUM_JOINTS])
    measured_angles = np.array(record.measurement.joint_angles, dtype=np.float64)
    measured_positions = joint_positions(measured_angles)

    _plot_arm(ax, measured_positions, color="#2563eb", linewidth=1.8, alpha=0.75, linestyle="--")
    _plot_arm(ax, true_positions, color="#111827", linewidth=2.8, alpha=1.0, linestyle="-")
    ax.scatter(
        [record.ee_pos[0]],
        [record.ee_pos[1]],
        s=38,
        color="#111827",
        edgecolor="white",
        linewidth=0.8,
        zorder=8,
    )

    if compact:
        ax.set_title(f"arm t={int(record.time)}  g={frame.generator_count}", fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
    else:
        title = f"Robot arm monitor replay: {trace} trace, {method}"
        ax.set_title(title)
        reducer = frame.reducer_used if frame.reducer_used else "-"
        statuses = ", ".join(f"{v.trigger.name}:{v.status}" for v in frame.verdicts)
        annotation = (
            f"seed {seed} | step {int(record.time)} | generators {frame.generator_count}\n"
            f"reduced {frame.reduced} | reducer {reducer}\n"
            f"{statuses}"
        )
        ax.text(
            0.02,
            0.98,
            annotation,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": "#d1d5db",
            },
        )
        _draw_legend(ax)


def _draw_trigger_panel(
    ax: Axes,
    frame: VisualizationFrame,
    *,
    trigger_limits: tuple[float, float, float, float],
    compact: bool,
) -> None:
    ax.clear()
    ax.set_aspect("equal", adjustable="box")
    xmin, xmax, ymin, ymax = trigger_limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(color="#e5e7eb", linewidth=0.7)
    ax.set_xlabel("trigger x")
    ax.set_ylabel("trigger y")
    _draw_trigger_region(ax)
    _draw_trigger_zonotope(ax, frame)
    ax.scatter(
        [frame.record.ee_pos[0]],
        [frame.record.ee_pos[1]],
        s=28,
        color="#111827",
        edgecolor="white",
        linewidth=0.7,
        zorder=8,
    )
    full_width = float(np.sum(frame.trigger_upper - frame.trigger_lower))
    rect = interval_rectangle(frame.trigger_lower, frame.trigger_upper)
    clipped = bool(
        np.any(rect[:, 0] < xmin)
        or np.any(rect[:, 0] > xmax)
        or np.any(rect[:, 1] < ymin)
        or np.any(rect[:, 1] > ymax)
    )
    title = "trigger-space uncertainty"
    if compact:
        title = f"trigger width {full_width:.2f}"
    ax.set_title(title, fontsize=9 if compact else 10)
    if clipped:
        ax.text(
            0.98,
            0.02,
            f"full hull width {full_width:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox={
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": "#d1d5db",
            },
        )
    if compact:
        ax.set_xlabel("")
        ax.set_ylabel("")


def _draw_trigger_region(ax: Axes) -> None:
    lower_left = FORBIDDEN_ZONE_CENTER - FORBIDDEN_ZONE_HALF
    size = 2.0 * FORBIDDEN_ZONE_HALF
    ax.add_patch(Rectangle(
        lower_left,
        size[0],
        size[1],
        facecolor="#fecaca",
        edgecolor="#b91c1c",
        linewidth=1.3,
        alpha=0.55,
        zorder=0,
    ))
    ax.axvline(0.7, color="#ef4444", linewidth=1.0, linestyle="--", alpha=0.7, zorder=1)
    ax.axhline(-0.1, color="#ef4444", linewidth=1.0, linestyle="--", alpha=0.7, zorder=1)


def _draw_trigger_zonotope(ax: Axes, frame: VisualizationFrame) -> None:
    vertices = zonotope_vertices_2d(frame.trigger_zonotope)
    if vertices.shape[0] >= 3:
        closed = np.vstack([vertices, vertices[0]])
        ax.fill(
            closed[:, 0],
            closed[:, 1],
            facecolor="#f59e0b",
            edgecolor="#b45309",
            linewidth=1.3,
            alpha=0.28,
            zorder=4,
        )
    elif vertices.shape[0] == 2:
        ax.plot(vertices[:, 0], vertices[:, 1], color="#b45309", linewidth=1.3, zorder=4)
    else:
        ax.scatter(vertices[:, 0], vertices[:, 1], color="#b45309", s=24, zorder=4)

    rect = interval_rectangle(frame.trigger_lower, frame.trigger_upper)
    ax.plot(
        rect[:, 0],
        rect[:, 1],
        color="#7c2d12",
        linewidth=1.0,
        linestyle="--",
        alpha=0.75,
        zorder=3,
    )


def _plot_arm(
    ax: Axes,
    positions: NDArray[np.float64],
    *,
    color: str,
    linewidth: float,
    alpha: float,
    linestyle: str,
) -> None:
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        linestyle=linestyle,
        marker="o",
        markersize=3.5,
        zorder=6,
    )


def _draw_legend(ax: Axes) -> None:
    handles = [
        Line2D([0], [0], color="#111827", linewidth=2.8, label="true arm"),
        Line2D([0], [0], color="#2563eb", linewidth=1.8, linestyle="--", label="measured arm"),
        Line2D([0], [0], color="#334155", linewidth=1.5, alpha=0.45, label="EE trail"),
        Rectangle((0, 0), 1, 1, facecolor="#f59e0b", edgecolor="#b45309", alpha=0.28, label="EE zonotope"),
        Line2D([0], [0], color="#7c2d12", linewidth=1.0, linestyle="--", label="interval hull"),
        Rectangle((0, 0), 1, 1, facecolor="#fecaca", edgecolor="#b91c1c", alpha=0.55, label="EE trigger region"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)


def _metadata(
    frames: Sequence[VisualizationFrame],
    *,
    method: str,
    trace: str,
    seed: int,
    length: int,
    budget: int,
    horizon: int,
    beam_width: int,
    fps: int,
    stride: int,
    rendered_frame_count: int,
    quality: TraceQuality,
    physical_limits: tuple[float, float, float, float],
    trigger_limits: tuple[float, float, float, float],
) -> dict[str, object]:
    reducer_counts = Counter(frame.reducer_used for frame in frames if frame.reducer_used)
    trigger_lower = FORBIDDEN_ZONE_CENTER - FORBIDDEN_ZONE_HALF
    trigger_upper = FORBIDDEN_ZONE_CENTER + FORBIDDEN_ZONE_HALF
    return {
        "method": method,
        "trace": trace,
        "trace_description": (
            "explanatory visualization trace"
            if trace == "paper" else "benchmark random waypoint trace"
        ),
        "trace_model": quality.trace_model,
        "seed": seed,
        "length": length,
        "budget": budget,
        "horizon": horizon,
        "beam_width": beam_width,
        "fps": fps,
        "stride": stride,
        "frame_count": len(frames),
        "rendered_frame_count": rendered_frame_count,
        "ee_path_length": quality.ee_path_length,
        "action_saturation_fraction": quality.action_saturation_fraction,
        "waypoints_reached": quality.waypoints_reached,
        "mean_target_error": quality.mean_target_error,
        "max_target_error": quality.max_target_error,
        "physical_view_limits": [float(v) for v in physical_limits],
        "trigger_view_limits": [float(v) for v in trigger_limits],
        "reductions": sum(1 for frame in frames if frame.reduced),
        "reducer_counts": dict(sorted(reducer_counts.items())),
        "max_generator_count": max(frame.generator_count for frame in frames),
        "mean_trigger_width": float(np.mean([
            np.sum(frame.trigger_upper - frame.trigger_lower)
            for frame in frames
        ])),
        "final_trigger_width": float(np.sum(frames[-1].trigger_upper - frames[-1].trigger_lower)),
        "end_effector_trigger_region": {
            "lower": [float(v) for v in trigger_lower],
            "upper": [float(v) for v in trigger_upper],
            "wall_x": 0.7,
            "floor_y": -0.1,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a 2D MuJoCo robot-arm zonotope animation.",
    )
    parser.add_argument("--output", type=Path, default=Path("results/robot-arm-animation"))
    parser.add_argument("--method", required=True)
    parser.add_argument("--trace", choices=TRACE_KINDS, default="benchmark")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--no-gif", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    artifacts = render_robot_arm_animation(
        output=args.output,
        method=args.method,
        trace=args.trace,
        seed=args.seed,
        length=args.length,
        budget=args.budget,
        horizon=args.horizon,
        beam_width=args.beam_width,
        fps=args.fps,
        stride=args.stride,
        dpi=args.dpi,
        save_gif=not args.no_gif,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
