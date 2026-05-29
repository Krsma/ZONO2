"""Evaluation metrics and comparison tables."""

from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap mean with confidence interval."""
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(float(np.mean(sample)))
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = means[int(alpha * n_bootstrap)]
    hi = means[int((1.0 - alpha) * n_bootstrap)]
    return float(np.mean(values)), lo, hi


def aggregate_summary(
    summary_df: pd.DataFrame,
    metric_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate per-seed summaries into mean ± CI per method."""
    if metric_columns is None:
        metric_columns = [
            "mean_trigger_width",
            "max_trigger_width",
            "mean_generator_count",
            "total_reductions",
            "total_time_ms",
        ]
    rows = []
    for method, group in summary_df.groupby("method"):
        row = {"method": method}
        for col in metric_columns:
            if col in group.columns:
                values = group[col].values
                mean, lo, hi = bootstrap_ci(values)
                row[f"{col}_mean"] = mean
                row[f"{col}_ci95_lo"] = lo
                row[f"{col}_ci95_hi"] = hi
        row["budget_violations"] = int(group["budget_violations"].sum())
        row["unsound_certificates"] = int(group["unsound_certificates"].sum())
        rows.append(row)
    return pd.DataFrame(rows)
