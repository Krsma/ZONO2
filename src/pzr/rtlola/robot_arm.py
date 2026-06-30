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
ARM_SPEC_PATH = Path(__file__).parent / "specs" / "robot_arm.lola"

ARM_EXPECTED_VERDICT_KEYS = (
    "dist_to_expected_exceeded",
    "tpl_exceeded",
)
ARM_PUBLIC_STREAM_KEYS = (
    "dist_to_expected",
    "tpl",
)

EXPECTED_CENTER = np.array([-0.18, 0.0, 0.05], dtype=np.float64)

Q = 0.000767
J = 0.001309
I = 0.008727
H = 0.003491


ARM_SPEC = ARM_SPEC_PATH.read_text()


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
