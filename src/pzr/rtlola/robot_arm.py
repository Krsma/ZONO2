"""RTLola 5-DOF low-cost robot arm scenario."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pzr.rtlola.engine import RtlolaEvent


TRACE_KINDS = (
    "figure8",
    "figure8_drift",
    "figure8_geofence",
    "figure8_drift_geofence",
    "random",
    "random_drift",
    "random_geofence",
    "random_drift_geofence",
    "square",
    "square_drift",
    "square_geofence",
    "square_drift_geofence",
)
DEFAULT_TRACE_KIND = "figure8_drift"
RLOLAEVAL_REVISION = "2257d074173a6dd475c042ef9a82cd8755a81ac3"
ROBOT_ARM_SPEC_SHA256 = (
    "aab5b768d872bc4f5b6dc11b96805c2d451cc5c91eb573225f6b0e246cee6acc"
)
ROBOT_ARM_TRACE_SHA256 = {
    "figure8": "fa07293b1a30c409ede95162f359f087f8c2e77e0df07a333d0045978150f309",
    "figure8_drift": "1a9def5a128a236f0e246f9d7403869ec676e62617a3cf4edf9c904e841362f7",
    "figure8_geofence": "f2d68199baadd956ccfa5b43688cbb973956d7367aa2dc986123f6bb25d28ede",
    "figure8_drift_geofence": "e055f7faeade23f9f30952f7c0871bf3803bb87a09c3a16ba3ef09f2bba9cd03",
    "random": "1c93cebfd7b2b3169d8e8eba7922bce742b68eeef17f53231c38d65be80198ff",
    "random_drift": "9a71cfdb8caa9da63c5fb9dd0f8be45f75e9cee2889b10d0efe4b60447823175",
    "random_geofence": "b4d6bb57a3cf39975b5db287c0df69abc7611be71ac289de5a9017dac0295879",
    "random_drift_geofence": "c010201e0985b950912377c08e882942546ff5add73d9e3614389b01d23a8936",
    "square": "c53282d3cbb5cbe25752215e5a5bc329ccd94e290aa449b932dc776a7ca3dc5c",
    "square_drift": "335c3c2a40e43ca87fca9fe1a68fe6d9a62dfbaee6af863e227a40d8a8766356",
    "square_geofence": "85bfbe0419ead65ccfd7746cf38997f76dc1903911121bdaa6f6229c7a818be2",
    "square_drift_geofence": "783840ab31f9469b9b49868f08ebbbbfc2ac184e2cba3f5d851dd4cb604f5285",
}
ROBOT_ARM_TRACE_ROWS = {
    "figure8": 2340,
    "figure8_drift": 2340,
    "figure8_geofence": 2340,
    "figure8_drift_geofence": 2340,
    "random": 1495,
    "random_drift": 1433,
    "random_geofence": 1063,
    "random_drift_geofence": 1105,
    "square": 1983,
    "square_drift": 1983,
    "square_geofence": 1983,
    "square_drift_geofence": 1983,
}

MODEL_DIR = Path(__file__).parents[1] / "envs" / "mujoco_models" / "low_cost_robot_arm"
MODEL_PATH = MODEL_DIR / "low_cost_robot_arm.xml"
SCENE_PATH = MODEL_DIR / "scene.xml"
TRACE_DIR = Path(__file__).parent / "traces" / "robot_arm"
ARM_SPEC_PATH = Path(__file__).parent / "specs" / "robot_arm.lola"

ARM_PUBLIC_STREAM_KEYS = (
    "dist_to_expected",
    "dxb",
    "dyb",
)
ARM_TRIGGER_KEYS = tuple(f"Trigger#{index}" for index in range(5))
ARM_TRIGGER_LABELS = {
    "Trigger#0": "Toolhead drift detected",
    "Trigger#1": "Cannot stop before +X boundary",
    "Trigger#2": "Cannot stop before -X boundary",
    "Trigger#3": "Cannot stop before +Y boundary",
    "Trigger#4": "Cannot stop before -Y boundary",
}

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
    expected_center: tuple[float | None, float | None, float | None]
    geofence: tuple[float | None, float | None, float | None, float | None]


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
    df = pd.read_csv(path, na_values=["#"])
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
            expected_center=(
                _optional_float(record.cx),
                _optional_float(record.cy),
                _optional_float(record.cz),
            ),
            geofence=(
                _optional_float(record.x_min),
                _optional_float(record.x_max),
                _optional_float(record.y_min),
                _optional_float(record.y_max),
            ),
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
        RtlolaEvent(
            time=row.time,
            values=(row.time, *row.angles, *row.expected_center, *row.geofence),
        )
        for row in rows
    )


def _optional_float(value: object) -> float | None:
    return None if pd.isna(value) else float(value)


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
