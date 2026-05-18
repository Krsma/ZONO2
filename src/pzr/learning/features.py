"""Stable feature extraction for reducer-selection decisions."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np

from pzr.core.zonotope import GeneratorKind, GeneratorRequirement
from pzr.monitoring.base import (
    MonitorAdapter,
    MonitorState,
    trigger_satisfaction_fraction,
    trigger_straddles_threshold,
)

DECISION_FEATURE_SCHEMA_VERSION = "decision_features_v1"

DECISION_FEATURE_NAMES = (
    "state_step",
    "budget",
    "horizon",
    "dimension",
    "generator_count",
    "generator_over_budget",
    "zonotope_order",
    "center_l2",
    "center_linf",
    "generator_abs_sum",
    "generator_abs_mean",
    "generator_abs_max",
    "generator_l2_sum",
    "generator_l2_mean",
    "generator_l2_max",
    "interval_width_sum",
    "interval_width_mean",
    "interval_width_max",
    "interval_radius_mean",
    "interval_radius_max",
    "trigger_count",
    "trigger_width_sum",
    "trigger_width_mean",
    "trigger_width_max",
    "trigger_straddle_count",
    "trigger_satisfaction_mean",
    "trigger_satisfaction_max",
    "trigger_threshold_margin_abs_min",
    "trigger_threshold_margin_abs_mean",
    "metadata_calibration_count",
    "metadata_measurement_count",
    "metadata_synthetic_count",
    "metadata_unknown_count",
    "metadata_age_mean",
    "metadata_age_max",
    "required_generator_rule_count",
    "required_generator_match_count",
)


def decision_feature_values(
    monitor: MonitorAdapter[Any],
    state: MonitorState,
    *,
    budget: int,
    horizon: int,
    required_generators: Sequence[GeneratorRequirement] | None = None,
) -> dict[str, float]:
    """Return finite numeric features for the pre-reduction monitor state."""

    zonotope = state.zonotope
    generators = np.asarray(zonotope.generators, dtype=float)
    center = np.asarray(zonotope.center, dtype=float)
    lower, upper = zonotope.interval_bounds()
    widths = np.asarray(upper - lower, dtype=float)
    radii = widths / 2.0
    generator_norms = (
        np.linalg.norm(generators, axis=0)
        if zonotope.generator_count
        else np.zeros(0, dtype=float)
    )
    abs_generators = np.abs(generators)

    verdict_widths: list[float] = []
    satisfactions: list[float] = []
    straddles = 0
    margins: list[float] = []
    for trigger in monitor.triggers:
        lo = float(lower[trigger.state_index])
        hi = float(upper[trigger.state_index])
        verdict_widths.append(hi - lo)
        satisfactions.append(trigger_satisfaction_fraction(lo, hi, trigger))
        straddles += int(trigger_straddles_threshold(lo, hi, trigger))
        if trigger.direction == "above":
            margins.append(abs(float(center[trigger.state_index]) - trigger.threshold))
        else:
            margins.append(abs(trigger.threshold - float(center[trigger.state_index])))

    metadata_counts = Counter(meta.kind for meta in zonotope.metadata)
    ages = np.asarray([meta.age for meta in zonotope.metadata], dtype=float)
    requirements = (
        tuple(required_generators)
        if required_generators is not None
        else tuple(monitor.required_generator_metadata(state))
    )
    required_matches = sum(
        any(requirement.matches(metadata) for requirement in requirements)
        for metadata in zonotope.metadata
    )

    features = {
        "state_step": float(state.step),
        "budget": float(budget),
        "horizon": float(horizon),
        "dimension": float(zonotope.dimension),
        "generator_count": float(zonotope.generator_count),
        "generator_over_budget": float(max(0, zonotope.generator_count - budget)),
        "zonotope_order": float(zonotope.order),
        "center_l2": _safe_stat(np.linalg.norm(center)),
        "center_linf": _safe_stat(np.max(np.abs(center)) if center.size else 0.0),
        "generator_abs_sum": _safe_stat(np.sum(abs_generators)),
        "generator_abs_mean": _safe_stat(
            np.mean(abs_generators) if abs_generators.size else 0.0
        ),
        "generator_abs_max": _safe_stat(
            np.max(abs_generators) if abs_generators.size else 0.0
        ),
        "generator_l2_sum": _safe_stat(np.sum(generator_norms)),
        "generator_l2_mean": _safe_stat(
            np.mean(generator_norms) if generator_norms.size else 0.0
        ),
        "generator_l2_max": _safe_stat(
            np.max(generator_norms) if generator_norms.size else 0.0
        ),
        "interval_width_sum": _safe_stat(np.sum(widths)),
        "interval_width_mean": _safe_stat(np.mean(widths) if widths.size else 0.0),
        "interval_width_max": _safe_stat(np.max(widths) if widths.size else 0.0),
        "interval_radius_mean": _safe_stat(np.mean(radii) if radii.size else 0.0),
        "interval_radius_max": _safe_stat(np.max(radii) if radii.size else 0.0),
        "trigger_count": float(len(monitor.triggers)),
        "trigger_width_sum": _safe_stat(
            np.sum(verdict_widths) if verdict_widths else 0.0
        ),
        "trigger_width_mean": _safe_stat(
            np.mean(verdict_widths) if verdict_widths else 0.0
        ),
        "trigger_width_max": _safe_stat(
            np.max(verdict_widths) if verdict_widths else 0.0
        ),
        "trigger_straddle_count": float(straddles),
        "trigger_satisfaction_mean": _safe_stat(
            np.mean(satisfactions) if satisfactions else 0.0
        ),
        "trigger_satisfaction_max": _safe_stat(
            np.max(satisfactions) if satisfactions else 0.0
        ),
        "trigger_threshold_margin_abs_min": _safe_stat(
            np.min(margins) if margins else 0.0
        ),
        "trigger_threshold_margin_abs_mean": _safe_stat(
            np.mean(margins) if margins else 0.0
        ),
        "metadata_calibration_count": float(metadata_counts[GeneratorKind.CALIBRATION]),
        "metadata_measurement_count": float(metadata_counts[GeneratorKind.MEASUREMENT]),
        "metadata_synthetic_count": float(metadata_counts[GeneratorKind.SYNTHETIC]),
        "metadata_unknown_count": float(metadata_counts[GeneratorKind.UNKNOWN]),
        "metadata_age_mean": _safe_stat(np.mean(ages) if ages.size else 0.0),
        "metadata_age_max": _safe_stat(np.max(ages) if ages.size else 0.0),
        "required_generator_rule_count": float(len(requirements)),
        "required_generator_match_count": float(required_matches),
    }
    return {name: _safe_stat(features[name]) for name in DECISION_FEATURE_NAMES}


def _safe_stat(value: float | np.floating[Any]) -> float:
    result = float(value)
    if not np.isfinite(result):
        return 0.0
    return result
