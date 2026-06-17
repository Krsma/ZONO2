"""Matrix metrics for RTLola evaluator states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RtlolaMatrixMetrics:
    """Budget and width metrics extracted from RTLola zonotope matrices."""

    dynamic_generator_count: int
    total_generator_count: int
    full_width_sum: float
    width_mean: float
    width_max: float
    dimension: int
    center_l2: float
    center_linf: float
    gen_norm_mean: float
    gen_norm_max: float
    gen_norm_std: float
    gen_sparsity: float
    gen_coupling: float
    gen_pca_explained: float

    def cost(self, generator_weight: float = 0.01) -> float:
        return float(self.full_width_sum + generator_weight * self.dynamic_generator_count)


def generator_count(matrix: NDArray[np.float64]) -> int:
    return max(int(matrix.shape[1]) - 1, 0)


def matrix_metrics(
    dynamic_matrix: NDArray[np.float64],
    total_matrix: NDArray[np.float64] | None = None,
) -> RtlolaMatrixMetrics:
    """Compute finite, deterministic metrics from an RTLola zonotope matrix."""
    z = np.asarray(dynamic_matrix, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] < 1:
        raise ValueError(f"expected 2D zonotope matrix with center column, got {z.shape}")
    if not np.all(np.isfinite(z)):
        raise ValueError("zonotope matrix contains non-finite values")

    total = np.asarray(total_matrix, dtype=np.float64) if total_matrix is not None else z
    center = z[:, 0]
    generators = z[:, 1:]
    widths = 2.0 * np.abs(generators).sum(axis=1) if generators.size else np.zeros(z.shape[0])
    gen_norms = (
        np.linalg.norm(generators, axis=0)
        if generators.size else np.asarray([0.0], dtype=np.float64)
    )

    coupling = 0.0
    pca_explained = 0.0
    if generators.shape[1] > 0 and generators.shape[0] > 1:
        norms = np.maximum(np.linalg.norm(generators, axis=0, keepdims=True), 1e-12)
        normalized = generators / norms
        gram = normalized.T @ normalized
        np.fill_diagonal(gram, 0.0)
        coupling = _safe(float(np.mean(np.abs(gram))))
        sv = np.linalg.svd(generators, compute_uv=False)
        total_var = float(np.sum(sv * sv))
        if total_var > 1e-12:
            pca_explained = _safe(float((sv[0] * sv[0]) / total_var))

    return RtlolaMatrixMetrics(
        dynamic_generator_count=generator_count(z),
        total_generator_count=generator_count(total),
        full_width_sum=_safe(float(np.sum(widths))),
        width_mean=_safe(float(np.mean(widths))) if widths.size else 0.0,
        width_max=_safe(float(np.max(widths))) if widths.size else 0.0,
        dimension=int(z.shape[0]),
        center_l2=_safe(float(np.linalg.norm(center))),
        center_linf=_safe(float(np.max(np.abs(center)))) if center.size else 0.0,
        gen_norm_mean=_safe(float(np.mean(gen_norms))),
        gen_norm_max=_safe(float(np.max(gen_norms))),
        gen_norm_std=_safe(float(np.std(gen_norms))) if len(gen_norms) > 1 else 0.0,
        gen_sparsity=_safe(float(np.mean(np.abs(generators) < 1e-10))) if generators.size else 1.0,
        gen_coupling=coupling,
        gen_pca_explained=pca_explained,
    )


def _safe(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0
