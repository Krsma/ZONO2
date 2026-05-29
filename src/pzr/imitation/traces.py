"""Expert trace recording for imitation learning.

Traces store (features, action) pairs collected at reduction decision points.
Both MPC expert demonstrations and DAgger on-policy collections use this
format.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass
class ReductionTrace:
    features: NDArray[np.float64]
    action: str
    cost: float
    step: int
    episode_id: int


class TraceCollector:
    """Collects reduction traces during rollouts."""

    def __init__(self) -> None:
        self._traces: list[ReductionTrace] = []

    def record(self, trace: ReductionTrace) -> None:
        self._traces.append(trace)

    @property
    def traces(self) -> list[ReductionTrace]:
        return list(self._traces)

    def __len__(self) -> int:
        return len(self._traces)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for t in self._traces:
            records.append({
                "features": t.features.tolist(),
                "action": t.action,
                "cost": t.cost,
                "step": t.step,
                "episode_id": t.episode_id,
            })
        with open(path, "w") as f:
            json.dump(records, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TraceCollector":
        with open(path) as f:
            records = json.load(f)
        collector = cls()
        for r in records:
            collector.record(ReductionTrace(
                features=np.array(r["features"], dtype=np.float64),
                action=r["action"],
                cost=r["cost"],
                step=r["step"],
                episode_id=r["episode_id"],
            ))
        return collector
