"""RTLola 5-DOF low-cost robot arm scenario."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pzr.rtlola.engine import RtlolaEvent


TRACE_KINDS = ("figure8_violated", "figure8", "square_violated", "square")
DEFAULT_TRACE_KIND = "figure8_violated"

MODEL_DIR = Path(__file__).parents[1] / "envs" / "mujoco_models" / "low_cost_robot_arm"
MODEL_PATH = MODEL_DIR / "low_cost_robot_arm.xml"
SCENE_PATH = MODEL_DIR / "scene.xml"
TRACE_DIR = Path(__file__).parent / "traces" / "robot_arm"

ARM_EXPECTED_VERDICT_KEYS = (
    "dist_to_expected_exceeded",
    "tpl_exceeded",
)
ARM_PUBLIC_STREAM_KEYS = (
    "dist_to_expected",
    "tpl",
)

# Binding state-zonotope rows observed for this spec:
# tcp position, cumulative tcp sums, and cumulative Euclidean toolpath.
ARM_RELEVANT_ROWS = (0, 1, 2, 3, 4, 5, 6)

EXPECTED_CENTER = np.array([-0.18, 0.0, 0.05], dtype=np.float64)

Q = 0.000767
J = 0.001309
I = 0.008727
H = 0.003491


ARM_SPEC = """
import math

input time: Float
input a1m: Float64
input a2m: Float64
input a3m: Float64
input a4m: Float64
input a5m: Float64

constant Q: Float64 := 0.000767
constant J: Float64 := 0.001309
constant I: Float64 := 0.008727
constant H: Float64 := 0.003491

output a1R: Variable @a1m
constant a1H: Variable
output a1 := a1m + (Q + J + I) * a1R + H * a1H

output a2R: Variable @a2m
constant a2H: Variable
output a2 := a2m + (Q + J + I) * a2R + H * a2H

output a3R: Variable @a3m
constant a3H: Variable
output a3 := a3m + (Q + J + I) * a3R + H * a3H

output a4R: Variable @a4m
constant a4H: Variable
output a4 := a4m + (Q + J + I) * a4R + H * a4H

output a5R: Variable @a5m
constant a5H: Variable
output a5 := a5m + (Q + J + I) * a5R + H * a5H

output s1 := sin(a1)
output s2 := sin(a2)
output s3 := sin(a3)
output s4 := sin(a4)
output s5 := sin(a5)

output c1 := cos(a1)
output c2 := cos(a2)
output c3 := cos(a3)
output c4 := cos(a4)
output c5 := cos(a5)

output w5y := 0.013097 + 0.0105 * c5 - 0.00045 * s5
constant w5x: Float64 := -0.1118
output w5z := 0.0105 * s5 + 0.00045 * c5

output w4x := -0.10048 + w5x * c4 + w5z * s4
output w4y := 0.00005 + w5y
output w4z := 0.0026999 - w5x * s4 + w5z * c4

output w3x := -0.0148 + w4x * c3 - w4z * s3
output w3y := 0.0065 + w4y
output w3z := 0.1083 + w4x * s3 + w4z * c3

output w2x := w3x * c2 + w3z * s2
output w2y := -0.0209 + w3y
output w2z := 0.0154 - w3x * s2 + w3z * c2

output px := 0.012 + w2x * c1 + w2y * s1
output py := -w2x * s1 + w2y * c1
output pz := 0.0409 + w2z

output count: Float64 @a1m := count.offset(by: -1).defaults(to: 0.0) + 1.0
output sum_x := sum_x.offset(by: -1).defaults(to: 0.0) + px
output sum_y := sum_y.offset(by: -1).defaults(to: 0.0) + py
output sum_z := sum_z.offset(by: -1).defaults(to: 0.0) + pz

output avg_x := sum_x / count
output avg_y := sum_y / count
output avg_z := sum_z / count

constant expected_x: Float64 := -0.18
constant expected_y: Float64 := 0.0
constant expected_z: Float64 := 0.05

output dx := avg_x - expected_x
output dy := avg_y - expected_y
output dz := avg_z - expected_z
#[public]
output dist_to_expected := sqrt((dx * dx) + (dy * dy) + (dz * dz))

constant max_dist: Float64 := 0.05
constant threshold: Float64 := 0.0
#[public]
output dist_to_expected_exceeded := pAbove(dist_to_expected, max_dist) > threshold && time > 4.0
trigger pAbove(dist_to_expected, max_dist) > threshold && time > 4.0 "Toolhead drift detected"

output dpx := px - px.offset(by: -1).defaults(to: expected_x)
output dpy := py - py.offset(by: -1).defaults(to: expected_y)
output dpz := pz - pz.offset(by: -1).defaults(to: expected_z)
output step_len := sqrt((dpx * dpx) + (dpy * dpy) + (dpz * dpz))
#[public]
output tpl := tpl.offset(by: -1).defaults(to: 0.0) + step_len

constant max_toolpath_length: Float64 := 1000.0
#[public]
output tpl_exceeded := pAbove(tpl, max_toolpath_length) > threshold
trigger pAbove(tpl, max_toolpath_length) > threshold "Toolhead travelled 1km: Service now!"
"""


@dataclass(frozen=True)
class RobotArmTraceRow:
    """One low-cost arm trace row."""

    time: float
    angles: tuple[float, float, float, float, float]
    tcp: tuple[float, float, float]


def forward_kinematics_5dof(angles: NDArray[np.float64]) -> NDArray[np.float64]:
    """Analytic FK for the MuJoCo `tcp` site used by the RTLola spec."""
    q = np.asarray(angles, dtype=np.float64)
    if q.shape != (5,):
        raise ValueError(f"expected 5 joint angles, got {q.shape}")
    s1, s2, s3, s4, s5 = np.sin(q)
    c1, c2, c3, c4, c5 = np.cos(q)

    w5y = 0.013097 + 0.0105 * c5 - 0.00045 * s5
    w5x = -0.1118
    w5z = 0.0105 * s5 + 0.00045 * c5

    w4x = -0.10048 + w5x * c4 + w5z * s4
    w4y = 0.00005 + w5y
    w4z = 0.0026999 - w5x * s4 + w5z * c4

    w3x = -0.0148 + w4x * c3 - w4z * s3
    w3y = 0.0065 + w4y
    w3z = 0.1083 + w4x * s3 + w4z * c3

    w2x = w3x * c2 + w3z * s2
    w2y = -0.0209 + w3y
    w2z = 0.0154 - w3x * s2 + w3z * c2

    return np.array([
        0.012 + w2x * c1 + w2y * s1,
        -w2x * s1 + w2y * c1,
        0.0409 + w2z,
    ], dtype=np.float64)


def trace_path(trace_kind: str) -> Path:
    if trace_kind not in TRACE_KINDS:
        raise ValueError(f"trace_kind must be one of {TRACE_KINDS}, got {trace_kind!r}")
    return TRACE_DIR / f"{trace_kind}.csv"


def load_robot_arm_trace(trace_kind: str = DEFAULT_TRACE_KIND) -> tuple[RobotArmTraceRow, ...]:
    path = trace_path(trace_kind)
    df = pd.read_csv(path)
    rows = []
    for record in df.itertuples(index=False):
        rows.append(RobotArmTraceRow(
            time=float(record.time),
            angles=(
                float(record.a1m),
                float(record.a2m),
                float(record.a3m),
                float(record.a4m),
                float(record.a5m),
            ),
            tcp=(float(record.x), float(record.y), float(record.z)),
        ))
    return tuple(rows)


def generate_robot_arm_events(
    length: int,
    seed: int = 0,
    *,
    trace_kind: str = DEFAULT_TRACE_KIND,
) -> tuple[RtlolaEvent, ...]:
    """Load a deterministic low-cost arm trace as RTLola events."""
    _ = seed
    rows = load_robot_arm_trace(trace_kind)
    if length > 0:
        rows = rows[:length]
    return tuple(
        RtlolaEvent(time=row.time, values=(row.time, *row.angles))
        for row in rows
    )


def validate_trace_tcp_against_fk(
    trace_kind: str = DEFAULT_TRACE_KIND,
    *,
    atol: float = 3e-4,
    max_rows: int | None = 25,
) -> float:
    """Return max CSV-vs-FK TCP error for a trace subset."""
    rows = load_robot_arm_trace(trace_kind)
    if max_rows is not None:
        rows = rows[:max_rows]
    max_err = 0.0
    for row in rows:
        expected = np.asarray(row.tcp, dtype=np.float64)
        actual = forward_kinematics_5dof(np.asarray(row.angles, dtype=np.float64))
        max_err = max(max_err, float(np.max(np.abs(actual - expected))))
    if max_err > atol:
        raise AssertionError(
            f"{trace_kind} TCP FK mismatch exceeds {atol}: {max_err}"
        )
    return max_err


def mujoco_tcp_position(angles: NDArray[np.float64]) -> NDArray[np.float64]:
    """Load the vendored MuJoCo model and return its `tcp` site position."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    q = np.asarray(angles, dtype=np.float64)
    if q.shape != (5,):
        raise ValueError(f"expected 5 joint angles, got {q.shape}")
    data.qpos[:5] = q
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    return np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
