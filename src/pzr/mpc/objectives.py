"""Cost functions for MPC reduction selection.

The cost function evaluates the quality of a monitor state after reduction.
Lower cost means better precision. The MPC controller minimizes total cost
over a finite horizon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pzr.monitoring.base import MonitorState, TriggerSpec, Verdict
from pzr.monitoring.triggers import trigger_straddles_threshold


@dataclass(frozen=True)
class CostWeights:
    trigger_width: float = 1.0
    straddling: float = 10.0
    generator_count: float = 0.01
    total_width: float = 0.0


@dataclass(frozen=True)
class WeightedZonotopeCost:
    """Weighted cost over zonotope state quality metrics."""

    weights: CostWeights = CostWeights()
    triggers: tuple[TriggerSpec, ...] = ()

    def __call__(
        self,
        state: MonitorState,
        verdicts: tuple[Verdict, ...] | None = None,
    ) -> float:
        z = state.zonotope
        widths = z.widths()
        total = self.weights.generator_count * z.generator_count
        total += self.weights.total_width * float(np.sum(widths))

        triggers = self.triggers
        if verdicts is not None:
            triggers = tuple(v.trigger for v in verdicts)

        if triggers:
            lower, upper = z.interval_bounds()
            for trigger in triggers:
                w = float(widths[trigger.state_index])
                total += self.weights.trigger_width * w
                lo = float(lower[trigger.state_index])
                hi = float(upper[trigger.state_index])
                if trigger_straddles_threshold(lo, hi, trigger):
                    total += self.weights.straddling
        return float(total)
