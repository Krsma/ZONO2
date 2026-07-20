"""Bounded challenger screening for learned RTLola reducer policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from pzr.artifact_io import write_csv_atomic, write_json_atomic
from pzr.learning.provenance import model_sha256
from pzr.learning.training import NamedDataset
from pzr.rtlola.policy_evaluation import PolicyComparison


EXPLORATION_SELECTION_SCHEMA = "pzr.policy-exploration-selection.v1"
SAFETY_COUNT_COLUMNS = (
    "false_positive_count",
    "false_negative_count",
    "infeasible_candidate_count",
    "fallback_count",
)
PREFERRED_CHALLENGERS = (
    "pairwise_ranking_policy_clean36",
    "expected_regret_clean20",
    "pairwise_ranking_policy_dart36",
)


@dataclass(frozen=True)
class ChallengerSelectionConfig:
    evaluation: Path
    models: tuple[NamedDataset, ...]
    output: Path
    expected_cell_count: int = 60


def run_challenger_selection(config: ChallengerSelectionConfig) -> dict[str, object]:
    """Validate an exploratory evaluation and write its bounded promotion decision."""
    if config.expected_cell_count < 1:
        raise ValueError("expected exploratory cell count must be positive")
    manifest = json.loads((config.evaluation / "manifest.json").read_text())
    if manifest.get("failure_count") != 0:
        raise ValueError("exploratory evaluation contains native failures")
    if manifest.get("cell_count") != config.expected_cell_count:
        raise ValueError(
            f"exploratory evaluation has {manifest.get('cell_count')} cells, "
            f"expected {config.expected_cell_count}"
        )
    comparisons = tuple(
        PolicyComparison(
            str(item["name"]), str(item["challenger"]), str(item["reference"]),
        )
        for item in manifest.get("comparisons", ())
    )
    if not comparisons:
        raise ValueError("exploratory evaluation manifest has no comparisons")
    if len({item.name for item in config.models}) != len(config.models):
        raise ValueError("challenger selection model names must be unique")
    required_models = {
        method
        for comparison in comparisons
        for method in (comparison.challenger, comparison.reference)
    }
    if {item.name for item in config.models} != required_models:
        raise ValueError("challenger selection models do not match comparisons")
    manifest_models = manifest.get("models", {})
    for model in config.models:
        if manifest_models.get(model.name, {}).get("sha256") != model_sha256(model.path):
            raise ValueError(f"evaluation model hash differs for {model.name!r}")
    validation_regret = {
        model.name: _validation_mean_selected_regret(model.path)
        for model in config.models
    }
    summary = pd.read_csv(config.evaluation / "summary.csv")
    assessments, selection = screen_challengers(summary, comparisons, validation_regret)
    selection.update({
        "evaluation": str(config.evaluation),
        "evaluation_fingerprint": manifest["experiment_fingerprint"],
        "evaluation_cell_count": int(manifest["cell_count"]),
        "models": {
            model.name: {
                "path": str(model.path),
                "sha256": model_sha256(model.path),
                "validation_mean_selected_normalized_regret": validation_regret[model.name],
            }
            for model in config.models
        },
    })
    write_csv_atomic(assessments, config.output / "challenger_assessments.csv")
    write_json_atomic(selection, config.output / "selection.json")
    return selection


def _validation_mean_selected_regret(model: Path) -> float:
    metrics = pd.read_csv(model / "validation_metrics.csv")
    required = {"sample_count", "mean_chosen_normalized_regret"}
    if metrics.empty or not required <= set(metrics.columns):
        raise ValueError(f"model {model} lacks clean-validation regret diagnostics")
    weights = metrics["sample_count"].to_numpy(dtype=float)
    values = metrics["mean_chosen_normalized_regret"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or weights.sum() <= 0.0:
        raise ValueError(f"model {model} has invalid validation regret diagnostics")
    return float(np.average(values, weights=weights))


def screen_challengers(
    summary: pd.DataFrame,
    comparisons: Sequence[PolicyComparison],
    validation_regret: Mapping[str, float],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply the predeclared safety and native-loss promotion gate."""
    required = {
        "trace_kind", "budget", "method", "mean_approx_loss", "sum_approx_loss",
        *SAFETY_COUNT_COLUMNS,
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"exploration summary lacks columns: {sorted(missing)}")
    if not comparisons:
        raise ValueError("challenger screening requires explicit comparisons")
    names = [comparison.name for comparison in comparisons]
    if len(set(names)) != len(names):
        raise ValueError("challenger comparison names must be unique")
    model_names = {
        method
        for comparison in comparisons
        for method in (comparison.challenger, comparison.reference)
    }
    if set(validation_regret) != model_names:
        raise ValueError("validation-regret models do not match comparison models")
    if not all(np.isfinite(value) for value in validation_regret.values()):
        raise ValueError("validation regret contains non-finite values")

    rows = []
    for comparison in comparisons:
        challenger, reference = _aligned_method_rows(
            summary, comparison.challenger, comparison.reference,
        )
        finite_columns = ["mean_approx_loss", "sum_approx_loss", *SAFETY_COUNT_COLUMNS]
        finite_artifacts = bool(
            np.isfinite(challenger[finite_columns].to_numpy(dtype=np.float64)).all()
            and np.isfinite(reference[finite_columns].to_numpy(dtype=np.float64)).all()
        )
        safety_deltas = {
            column: float(challenger[column].sum() - reference[column].sum())
            for column in SAFETY_COUNT_COLUMNS
        }
        no_added_safety_events = all(
            bool(np.all(
                challenger[column].to_numpy(dtype=np.float64)
                <= reference[column].to_numpy(dtype=np.float64)
            ))
            for column in SAFETY_COUNT_COLUMNS
        )
        challenger_sum = float(challenger["sum_approx_loss"].sum())
        reference_sum = float(reference["sum_approx_loss"].sum())
        summed_loss_reduction = (
            (reference_sum - challenger_sum) / reference_sum
            if reference_sum > 0.0 else 0.0
        )
        challenger_macro = float(challenger["mean_approx_loss"].mean())
        reference_macro = float(reference["mean_approx_loss"].mean())
        cell_limit = reference["sum_approx_loss"].to_numpy(dtype=np.float64) * 1.10
        cell_limit = np.where(
            reference["sum_approx_loss"].to_numpy(dtype=np.float64) == 0.0,
            0.0,
            cell_limit,
        )
        no_cell_regression = bool(np.all(
            challenger["sum_approx_loss"].to_numpy(dtype=np.float64) <= cell_limit
        ))
        challenger_regret = float(validation_regret[comparison.challenger])
        reference_regret = float(validation_regret[comparison.reference])
        validation_not_worse = challenger_regret <= reference_regret
        passed = bool(
            finite_artifacts
            and no_added_safety_events
            and summed_loss_reduction >= 0.02
            and challenger_macro < reference_macro
            and no_cell_regression
            and validation_not_worse
        )
        rows.append({
            "comparison": comparison.name,
            "challenger": comparison.challenger,
            "reference": comparison.reference,
            "finite_artifacts": finite_artifacts,
            "no_added_safety_events": no_added_safety_events,
            **{f"{column}_delta": value for column, value in safety_deltas.items()},
            "challenger_sum_approx_loss": challenger_sum,
            "reference_sum_approx_loss": reference_sum,
            "summed_loss_reduction": summed_loss_reduction,
            "challenger_macro_mean_approx_loss": challenger_macro,
            "reference_macro_mean_approx_loss": reference_macro,
            "macro_mean_loss_lower": challenger_macro < reference_macro,
            "no_cell_above_110_percent": no_cell_regression,
            "challenger_validation_mean_selected_normalized_regret": challenger_regret,
            "reference_validation_mean_selected_normalized_regret": reference_regret,
            "validation_regret_not_worse": validation_not_worse,
            "passed": passed,
        })
    assessments = pd.DataFrame(rows)
    winner = _select_winner(assessments)
    selection = {
        "schema": EXPLORATION_SELECTION_SCHEMA,
        "criteria": {
            "minimum_summed_loss_reduction": 0.02,
            "require_lower_macro_mean_loss": True,
            "maximum_cell_summed_loss_ratio": 1.10,
            "require_no_added_fp_fn_infeasible_or_fallback": True,
            "require_clean_validation_regret_not_worse": True,
            "near_tie_absolute_reduction": 0.005,
            "near_tie_preference": list(PREFERRED_CHALLENGERS),
        },
        "comparisons": [asdict(comparison) for comparison in comparisons],
        "winner": winner,
        "stop_method_expansion": winner is None,
    }
    return assessments, selection


def _aligned_method_rows(
    summary: pd.DataFrame,
    challenger: str,
    reference: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["trace_kind", "budget"]
    left = summary[summary["method"] == challenger].set_index(keys).sort_index()
    right = summary[summary["method"] == reference].set_index(keys).sort_index()
    if left.empty or right.empty or not left.index.is_unique or not right.index.is_unique:
        raise ValueError(f"comparison {challenger!r} vs {reference!r} lacks unique cells")
    if not left.index.equals(right.index):
        raise ValueError(f"comparison {challenger!r} vs {reference!r} cells do not align")
    return left, right


def _select_winner(assessments: pd.DataFrame) -> dict[str, str] | None:
    passing = assessments[assessments["passed"]].copy()
    if passing.empty:
        return None
    best_reduction = float(passing["summed_loss_reduction"].max())
    near_best = passing[
        passing["summed_loss_reduction"] >= best_reduction - 0.005
    ].copy()
    order = {name: index for index, name in enumerate(PREFERRED_CHALLENGERS)}
    near_best["_preference"] = near_best["challenger"].map(
        lambda name: order.get(str(name), len(order))
    )
    selected = near_best.sort_values(
        ["_preference", "summed_loss_reduction", "challenger"],
        ascending=[True, False, True],
    ).iloc[0]
    return {
        "comparison": str(selected["comparison"]),
        "challenger": str(selected["challenger"]),
        "reference": str(selected["reference"]),
    }
