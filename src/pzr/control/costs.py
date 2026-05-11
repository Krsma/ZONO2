"""Monitor-aware rollout costs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pzr.monitoring.base import MonitorState, TriggerSpec, Verdict


@dataclass(frozen=True)
class CostWeights:
    """Weights for a monitor-aware precision cost."""

    trigger_width: float = 1.0
    straddling: float = 10.0
    generator_count: float = 0.01
    total_width: float = 0.0


@dataclass(frozen=True)
class WeightedZonotopeCost:
    """Cost function for short-horizon reduction rollouts."""

    weights: CostWeights = CostWeights()
    triggers: tuple[TriggerSpec, ...] = ()

    def __call__(
        self,
        state: MonitorState,
        verdicts: tuple[Verdict, ...] | None = None,
    ) -> float:
        zonotope = state.zonotope
        widths = zonotope.widths()
        total = self.weights.generator_count * zonotope.generator_count
        total += self.weights.total_width * float(np.sum(widths))

        triggers = self.triggers
        if verdicts is not None:
            triggers = tuple(verdict.trigger for verdict in verdicts)

        if triggers:
            for trigger in triggers:
                width = float(widths[trigger.state_index])
                total += self.weights.trigger_width * width
                lower, upper = zonotope.interval_bounds()
                lo = float(lower[trigger.state_index])
                hi = float(upper[trigger.state_index])
                if lo <= trigger.threshold <= hi:
                    total += self.weights.straddling
        return float(total)
