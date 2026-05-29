"""Environment protocol and monitored environment wrapper.

The environment protocol defines a simple reset/step/close interface.
MonitoredEnvironment wraps an environment with noise injection, zonotope
monitoring, and an intervention mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


class Environment(Protocol):
    """Minimal environment interface."""

    def reset(self, seed: int = 0) -> NDArray[np.float64]: ...
    def step(self, action: NDArray[np.float64]) -> tuple[NDArray[np.float64], float, bool, dict[str, Any]]: ...
    def close(self) -> None: ...
    def true_state(self) -> NDArray[np.float64]: ...


@dataclass
class NoisySensorModel:
    """ISO 5725 noise model: persistent bias + per-step bounded noise."""

    bias_bound: NDArray[np.float64]
    noise_bound: NDArray[np.float64]
    _bias: NDArray[np.float64] = field(default_factory=lambda: np.empty(0), repr=False)

    def reset(self, rng: np.random.Generator) -> None:
        self._bias = rng.uniform(-self.bias_bound, self.bias_bound)

    @property
    def bias(self) -> NDArray[np.float64]:
        return self._bias

    def observe(self, true_state: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.float64]:
        noise = rng.uniform(-self.noise_bound, self.noise_bound)
        return true_state + self._bias + noise


@dataclass
class InterventionStats:
    """Tracks intervention quality per episode."""

    total_steps: int = 0
    monitor_triggers: int = 0
    oracle_triggers: int = 0
    spurious: int = 0
    justified: int = 0
    missed: int = 0
    interventions: int = 0

    @property
    def spurious_rate(self) -> float:
        return self.spurious / max(self.total_steps, 1)

    @property
    def missed_rate(self) -> float:
        return self.missed / max(self.total_steps, 1)

    @property
    def intervention_rate(self) -> float:
        return self.interventions / max(self.total_steps, 1)
