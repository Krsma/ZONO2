"""ProtectedReducer: preserves designated generator columns during reduction.

Used by monitors that declare specific generators (e.g., calibration bias)
which must survive reduction exactly. The monitor provides the column indices
to protect; the wrapper splits them off, reduces the remainder, and
recombines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pzr.zonotope.core import Zonotope
from pzr.zonotope.reduction import Reducer, ReductionResult, _cert


@dataclass(frozen=True)
class ProtectedReducer:
    """Wrapper that preserves specified generator columns before reducing the rest."""

    base: Reducer
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(self, "name", self.base.name)

    def reduce(
        self,
        z: Zonotope,
        budget: int,
        protected_indices: tuple[int, ...] = (),
    ) -> ReductionResult:
        if not protected_indices:
            return self.base.reduce(z, budget)

        if len(protected_indices) > budget:
            raise ValueError(
                f"cannot preserve {len(protected_indices)} generators within budget {budget}"
            )

        if z.generator_count <= budget:
            return ReductionResult(z, z, _cert(self.name, z, z))

        protected_set = set(protected_indices)
        residual_idx = [i for i in range(z.generator_count) if i not in protected_set]
        residual_budget = budget - len(protected_indices)

        residual_z = Zonotope(
            np.zeros(z.dimension, dtype=np.float64),
            z.generators[:, residual_idx] if residual_idx else np.empty((z.dimension, 0)),
        )
        residual_result = self.base.reduce(residual_z, residual_budget)
        if not residual_result.certificate.is_sound:
            raise ValueError(f"wrapped reducer {self.base.name} returned unsound certificate")

        protected_g = z.generators[:, list(protected_indices)]
        if residual_result.reduced.generator_count > 0:
            reduced_g = np.hstack([protected_g, residual_result.reduced.generators])
        else:
            reduced_g = protected_g
        reduced = Zonotope(z.center, reduced_g)
        return ReductionResult(z, reduced, _cert(self.name, z, reduced))
