"""Base classes for system dynamics and noise models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class NoiseModel:
    """ISO 5725 error model: persistent calibration bias + per-step noise.

    measured = true + bias + noise
    where bias is sampled once and noise is fresh each step.
    """

    calibration_bound: float
    measurement_bound: float

    def sample_bias(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(-self.calibration_bound, self.calibration_bound))

    def sample_noise(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(-self.measurement_bound, self.measurement_bound))
