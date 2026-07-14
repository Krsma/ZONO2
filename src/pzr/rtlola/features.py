"""Current-state, specification-neutral RTLola ranking features."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pzr.learning.ranker import FeatureSchema
from pzr.rtlola.engine import RtlolaEngine, RtlolaStateRef
from pzr.rtlola.metrics import matrix_metrics


RTL_RANKING_FEATURE_NAMES = (
    "budget",
    "dynamic_generator_count",
    "active_dynamic_generator_count",
    "compact_dimension",
    "logical_dynamic_dimension",
    "generator_overflow_ratio",
    "zero_dynamic_fraction",
    "state_width",
    "max_row_width",
    "mean_active_generator_norm",
    "max_to_mean_generator_norm",
    "generator_coupling",
    "row_width_concentration",
    "active_generator_norm_cv",
    "mean_generator_off_axis_fraction",
)

RTL_RANKING_FEATURE_SCHEMA = FeatureSchema(
    name="rtlola.current-zonotope",
    version=2,
    feature_names=RTL_RANKING_FEATURE_NAMES,
    log1p_features=(
        "budget",
        "dynamic_generator_count",
        "active_dynamic_generator_count",
        "compact_dimension",
        "logical_dynamic_dimension",
        "state_width",
        "max_row_width",
        "mean_active_generator_norm",
        "active_generator_norm_cv",
    ),
)


def extract_ranking_features(
    engine: RtlolaEngine,
    state: RtlolaStateRef,
    budget: int,
) -> NDArray[np.float32]:
    dynamic, total = engine.matrices(state)
    return ranking_features_from_matrices(dynamic, total, budget)


def ranking_features_from_matrices(
    dynamic_matrix: NDArray[np.float64],
    total_matrix: NDArray[np.float64],
    budget: int,
    *,
    atol: float = 1e-12,
) -> NDArray[np.float32]:
    if budget < 0:
        raise ValueError("budget must be non-negative")
    dynamic = np.asarray(dynamic_matrix, dtype=np.float64)
    metrics = matrix_metrics(dynamic, np.asarray(total_matrix, dtype=np.float64))
    generators = dynamic[:, 1:]
    norms = (
        np.linalg.norm(generators, axis=0)
        if generators.size else np.zeros(0, dtype=np.float64)
    )
    active = norms > atol
    active_norms = norms[active]
    mean_norm = float(np.mean(active_norms)) if active_norms.size else 0.0
    max_to_mean = (
        float(np.max(active_norms) / mean_norm)
        if mean_norm > atol else 0.0
    )
    norm_cv = (
        float(np.std(active_norms) / mean_norm)
        if mean_norm > atol else 0.0
    )
    coupling = _active_generator_coupling(generators[:, active], active_norms)
    row_width_concentration = (
        float(metrics.width_max / metrics.state_width)
        if metrics.state_width > atol else 0.0
    )
    off_axis_fraction = _mean_generator_off_axis_fraction(
        generators[:, active], atol=atol,
    )
    dense_count = metrics.dynamic_generator_count
    features = np.asarray([
        budget,
        dense_count,
        metrics.active_dynamic_generator_count,
        metrics.dimension,
        metrics.logical_dynamic_dimension,
        dense_count / max(budget, 1),
        metrics.zero_dynamic_generator_count / max(dense_count, 1),
        metrics.state_width,
        metrics.width_max,
        mean_norm,
        max_to_mean,
        coupling,
        row_width_concentration,
        norm_cv,
        off_axis_fraction,
    ], dtype=np.float32)
    if not np.all(np.isfinite(features)):
        raise ValueError("RTLola ranking features contain non-finite values")
    return features


def _active_generator_coupling(
    generators: NDArray[np.float64],
    norms: NDArray[np.float64],
) -> float:
    if generators.shape[1] < 2:
        return 0.0
    normalized = generators / norms[np.newaxis, :]
    gram = np.abs(normalized.T @ normalized)
    indices = np.triu_indices(gram.shape[0], k=1)
    return float(np.mean(gram[indices]))


def _mean_generator_off_axis_fraction(
    generators: NDArray[np.float64],
    *,
    atol: float,
) -> float:
    if generators.shape[1] == 0:
        return 0.0
    absolute = np.abs(generators)
    l1 = np.sum(absolute, axis=0)
    linf = np.max(absolute, axis=0)
    valid = l1 > atol
    if not np.any(valid):
        return 0.0
    return float(np.mean((l1[valid] - linf[valid]) / l1[valid]))
