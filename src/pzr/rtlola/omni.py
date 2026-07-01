"""RTLola Omni robot specification and deterministic trace generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pzr.rtlola.engine import RtlolaEvent

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


def generate_omni_events(
    length: int,
    seed: int = 0,
    *,
    dt: float = 1.0,
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
