"""RTLola Omni robot specification and deterministic trace generation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from pzr.rtlola.engine import RtlolaEvent

OMNI_TRACE_KINDS = (
    "canonical",
    "safe",
    "x_violated",
    "y_violated",
)
OMNI_DEFAULT_TRACE_KIND = "canonical"

OMNI_EXPECTED_VERDICT_KEYS = (
    "position_x_above_geofence",
    "position_y_above_geofence",
)
OMNI_PUBLIC_STREAM_KEYS = (
    "position_x",
    "position_y",
)

OMNI_SPEC_PATH = Path(__file__).parent / "specs" / "omni_robot.lola"
OMNI_SPEC = OMNI_SPEC_PATH.read_text()

_BALANCED_DT = 0.075
_BALANCED_DIRECTION_AMPLITUDE = 0.65
_BALANCED_DIRECTION_PERIOD = 17.0
_BALANCED_DIRECTION_JITTER_MEMORY = 0.85
_BALANCED_DIRECTION_JITTER_SCALE = 0.015
_BALANCED_ACCELERATION_AMPLITUDE = 0.006
_BALANCED_ACCELERATION_PERIOD = 13.0
_BALANCED_ACCELERATION_NOISE_SCALE = 0.003
_VIOLATED_ACCELERATION_BASE = 0.05


def generate_omni_events(
    length: int,
    seed: int = 0,
    *,
    trace_kind: str = OMNI_DEFAULT_TRACE_KIND,
    dt: float | None = None,
) -> tuple[RtlolaEvent, ...]:
    """Generate one canonical or calibrated Omni input trace."""
    if trace_kind not in OMNI_TRACE_KINDS:
        raise ValueError(
            f"trace_kind must be one of {OMNI_TRACE_KINDS}, got {trace_kind!r}"
        )
    if trace_kind == "canonical":
        return _generate_canonical_events(
            length,
            seed,
            dt=1.0 if dt is None else float(dt),
        )
    if dt is not None:
        raise ValueError("dt override is supported only for the canonical Omni trace")
    return _generate_balanced_events(length, seed, trace_kind)


def _generate_canonical_events(
    length: int,
    seed: int,
    *,
    dt: float,
) -> tuple[RtlolaEvent, ...]:
    """Reproduce the established stochastic Omni input trace exactly."""
    rng = np.random.default_rng(seed)
    direction = 0.0
    events: list[RtlolaEvent] = []
    for index in range(length):
        direction += float(rng.normal(0.0, 0.18))
        acceleration = (
            0.18 * np.sin(index / 5.0)
            + float(rng.normal(0.0, 0.04))
        )
        time = float(index * dt)
        events.append(RtlolaEvent(
            time=time,
            values=(time, direction, acceleration),
        ))
    return tuple(events)


def _generate_balanced_events(
    length: int,
    seed: int,
    trace_kind: str,
) -> tuple[RtlolaEvent, ...]:
    rng = np.random.default_rng(seed)
    direction_phase = float(rng.uniform(0.0, 2.0 * math.pi))
    acceleration_phase = float(rng.uniform(0.0, 2.0 * math.pi))
    axis = math.pi / 2.0 if trace_kind == "y_violated" else 0.0
    acceleration_base = (
        0.0 if trace_kind == "safe" else _VIOLATED_ACCELERATION_BASE
    )
    direction_jitter = 0.0
    events: list[RtlolaEvent] = []
    for index in range(length):
        direction_jitter = (
            _BALANCED_DIRECTION_JITTER_MEMORY * direction_jitter
            + float(rng.normal(0.0, _BALANCED_DIRECTION_JITTER_SCALE))
        )
        direction = (
            axis
            + _BALANCED_DIRECTION_AMPLITUDE
            * math.sin(index / _BALANCED_DIRECTION_PERIOD + direction_phase)
            + direction_jitter
        )
        acceleration = (
            acceleration_base
            + _BALANCED_ACCELERATION_AMPLITUDE
            * math.sin(index / _BALANCED_ACCELERATION_PERIOD + acceleration_phase)
            + float(rng.normal(0.0, _BALANCED_ACCELERATION_NOISE_SCALE))
        )
        time = float(index * _BALANCED_DT)
        events.append(RtlolaEvent(
            time=time,
            values=(time, direction, acceleration),
        ))
    return tuple(events)
