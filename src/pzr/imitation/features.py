"""Feature extraction at reduction decision points.

Features summarize the current zonotope state for the learned policy.
All features must be finite (no NaN/inf).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pzr.monitoring.base import MonitorState, TriggerSpec
from pzr.monitoring.triggers import trigger_straddles_threshold

FEATURE_NAMES = (
    "generator_count",
    "dimension",
    "budget_headroom",
    "order",
    "gen_norm_mean",
    "gen_norm_max",
    "gen_norm_median",
    "width_sum",
    "width_mean",
    "width_max",
    "radius_mean",
    "radius_max",
    "trigger_width_sum",
    "trigger_width_mean",
    "trigger_straddle_count",
    "num_calibration",
    "num_non_calibration",
    "center_l2",
    "center_linf",
    "gen_sparsity",
    "gen_coupling",
    "gen_pca_explained",
    "gen_norm_std",
    "gen_condition",
)


def _safe(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(x)


def extract_features(
    state: MonitorState,
    budget: int,
    triggers: tuple[TriggerSpec, ...] = (),
) -> NDArray[np.float64]:
    """Extract feature vector for the learned policy."""
    z = state.zonotope
    n_cal = len(state.calibration_indices)
    n_gen = z.generator_count
    n_dim = z.dimension

    gen_norms = np.linalg.norm(z.generators, axis=0) if n_gen > 0 else np.array([0.0])
    widths = z.widths()
    radius = z.interval_radius()

    trigger_width_sum = 0.0
    trigger_width_mean = 0.0
    trigger_straddle_count = 0.0
    if triggers:
        lower, upper = z.interval_bounds()
        t_widths = []
        for t in triggers:
            w = float(widths[t.state_index])
            t_widths.append(w)
            if trigger_straddles_threshold(float(lower[t.state_index]), float(upper[t.state_index]), t):
                trigger_straddle_count += 1
        trigger_width_sum = sum(t_widths)
        trigger_width_mean = trigger_width_sum / len(triggers)

    G = z.generators
    sparsity = float(np.mean(np.abs(G) < 1e-10)) if G.size > 0 else 1.0
    coupling = 0.0
    if n_gen > 0 and n_dim > 1:
        col_norms = np.linalg.norm(G, axis=0, keepdims=True)
        safe_norms = np.maximum(col_norms, 1e-12)
        G_normed = G / safe_norms
        gram = G_normed.T @ G_normed
        np.fill_diagonal(gram, 0.0)
        coupling = float(np.mean(np.abs(gram)))

    pca_explained = 0.0
    if n_gen > 0 and n_dim > 1:
        sv = np.linalg.svd(G, compute_uv=False)
        total_var = float(np.sum(sv ** 2))
        if total_var > 1e-12:
            pca_explained = float(sv[0] ** 2 / total_var)

    gen_norm_std = float(np.std(gen_norms)) if len(gen_norms) > 1 else 0.0
    gen_condition = 0.0
    if n_gen >= n_dim and n_dim > 0:
        sv = np.linalg.svd(G, compute_uv=False)
        if sv[-1] > 1e-12:
            gen_condition = float(sv[0] / sv[-1])

    features = np.array([
        _safe(n_gen),
        _safe(n_dim),
        _safe(budget - n_gen),
        _safe(n_gen / max(n_dim, 1)),
        _safe(float(np.mean(gen_norms))),
        _safe(float(np.max(gen_norms))),
        _safe(float(np.median(gen_norms))),
        _safe(float(np.sum(widths))),
        _safe(float(np.mean(widths))),
        _safe(float(np.max(widths))),
        _safe(float(np.mean(radius))),
        _safe(float(np.max(radius))),
        _safe(trigger_width_sum),
        _safe(trigger_width_mean),
        _safe(trigger_straddle_count),
        _safe(n_cal),
        _safe(n_gen - n_cal),
        _safe(float(np.linalg.norm(z.center))),
        _safe(float(np.max(np.abs(z.center)))),
        _safe(sparsity),
        _safe(coupling),
        _safe(pca_explained),
        _safe(gen_norm_std),
        _safe(gen_condition),
    ], dtype=np.float64)
    return features
