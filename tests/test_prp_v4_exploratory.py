from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


TOOLS = Path(__file__).parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from prp_v4_exploratory import (  # noqa: E402
    BAD_DIAGNOSTIC_CELLS,
    CANDIDATES,
    CONFIRMATION_SEEDS,
    DAGGER_SEEDS,
    FeatureVariant,
    MATCHED_DIAGNOSTIC_CELLS,
    SELECTION_SEEDS,
    ExploreConfig,
    SCHEMA,
    _load_stage,
    _tail_selection_table,
    augment_features,
    balanced_dagger_dataset,
    causal_joint_features,
    challenger_root_names,
    schema_for,
    selection_metrics,
    smoke_config,
    tolerance_aware_policy_error,
)
from pzr.learning.dataset import ReducerCostDataset  # noqa: E402
from pzr.rtlola.engine import RtlolaEvent  # noqa: E402


def test_seed_roles_and_exact_selection_cell_counts_are_disjoint():
    config = ExploreConfig()

    assert not set(DAGGER_SEEDS) & set(SELECTION_SEEDS)
    assert not set(DAGGER_SEEDS) & set(CONFIRMATION_SEEDS)
    assert not set(SELECTION_SEEDS) & set(CONFIRMATION_SEEDS)
    assert config.expected_feature_cells == 224
    assert config.expected_selection_cells(FeatureVariant.G15) == 336
    assert config.expected_selection_cells(FeatureVariant.G25) == 392
    assert len(BAD_DIAGNOSTIC_CELLS) == len(MATCHED_DIAGNOSTIC_CELLS) == 5

    with pytest.raises(ValueError, match="must be disjoint"):
        ExploreConfig(selection_seeds=(312,))


def test_geometry_dimensions_and_future_invariance():
    events = (
        RtlolaEvent(0.0, (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)),
        RtlolaEvent(0.1, (0.1, 2.0, 4.0, 6.0, 8.0, 10.0)),
        RtlolaEvent(0.2, (0.2, 99.0, 99.0, 99.0, 99.0, 99.0)),
    )
    changed_future = (*events[:2], RtlolaEvent(0.2, (0.2, -99.0, -99.0, -99.0, -99.0, -99.0)))

    current, predicted = causal_joint_features(events[:2])
    other_current, other_predicted = causal_joint_features(changed_future[:2])

    np.testing.assert_allclose(current, other_current)
    np.testing.assert_allclose(predicted, other_predicted)
    np.testing.assert_allclose(predicted[0], current[0])
    np.testing.assert_allclose(predicted[1], [3.0, 6.0, 9.0, 12.0, 15.0])
    base = np.arange(15, dtype=np.float32)
    assert augment_features(base, current[1], predicted[1], FeatureVariant.G15).shape == (15,)
    assert augment_features(base, current[1], predicted[1], FeatureVariant.G20).shape == (20,)
    assert augment_features(base, current[1], predicted[1], FeatureVariant.G25).shape == (25,)
    assert len(schema_for(FeatureVariant.G25).feature_names) == 25


def _dataset(prefix: str, count: int, *, split: str) -> tuple[ReducerCostDataset, pd.DataFrame]:
    sample_ids = tuple(f"{prefix}-{index}" for index in range(count))
    dataset = ReducerCostDataset(
        features=np.arange(count * 2, dtype=np.float32).reshape(count, 2),
        teacher_costs=np.tile([[0.0, 1.0]], (count, 1)),
        feasible=np.ones((count, 2), dtype=np.bool_),
        candidate_names=("girard", "scott"),
        feature_names=("a", "b"),
        splits=tuple(split for _ in range(count)),
        sample_ids=sample_ids,
    )
    metadata = pd.DataFrame({
        "sample_id": sample_ids,
        "split": split,
        "seed": range(count),
        "budget": 40,
        "step": range(count),
    })
    return dataset, metadata


def test_balanced_dagger_training_is_deterministic_and_keeps_clean_validation():
    train, train_meta = _dataset("train", 4, split="train")
    validation, validation_meta = _dataset("validation", 2, split="validation")
    clean = ReducerCostDataset.concatenate((train, validation))
    clean_meta = pd.concat((train_meta, validation_meta), ignore_index=True)
    dagger, dagger_meta = _dataset("dagger", 2, split="train")

    first, first_meta = balanced_dagger_dataset(clean, clean_meta, dagger, dagger_meta, seed=7)
    second, second_meta = balanced_dagger_dataset(clean, clean_meta, dagger, dagger_meta, seed=7)

    np.testing.assert_array_equal(first.features, second.features)
    pd.testing.assert_frame_equal(first_meta, second_meta)
    counts = first_meta.loc[first_meta["split"] == "train", "training_source"].value_counts()
    assert counts["clean148"] == counts["learner_visited_dagger1"] == 4
    assert first.splits.count("validation") == 2


def test_tail_first_selection_prefers_tail_removal_before_mean_fpr():
    rows = []
    for seed in (1, 2):
        rows.append({"seed": seed, "budget": 40, "method": "mpc_terminal_beam_predictive_linear", "status": "completed", "mean_approx_loss": 1.0, "fpr": 0.1})
        rows.append({"seed": seed, "budget": 40, "method": "g15_clean148", "status": "completed", "mean_approx_loss": 2000.0 if seed == 2 else 0.5, "fpr": 0.0})
        rows.append({"seed": seed, "budget": 40, "method": "g20_clean148", "status": "completed", "mean_approx_loss": 2.0, "fpr": 0.2})
    table = _tail_selection_table(pd.DataFrame(rows), ("g15_clean148", "g20_clean148"))

    assert table.iloc[0]["method"] == "g20_clean148"
    assert table.iloc[0]["catastrophic_count"] == 0


def test_challenger_shortlist_and_tolerance_aware_ties_are_deterministic():
    assert challenger_root_names(("girard", "scott", "pca", "combastel")) == (
        "girard", "scott",
    )
    assert challenger_root_names(("scott", "pca", "girard", "combastel")) == (
        "scott", "pca",
    )
    costs = np.asarray([1.0, 1.0 + 5e-10, 2.0, np.nan])
    feasible = np.asarray([True, True, True, False])
    assert not tolerance_aware_policy_error(costs, feasible, 0)
    assert not tolerance_aware_policy_error(costs, feasible, 1)
    assert tolerance_aware_policy_error(costs, feasible, 2)
    assert tolerance_aware_policy_error(costs, feasible, None)


def test_smoke_contract_is_small_but_exercises_every_seed_role():
    config = smoke_config()

    assert config.budgets == (40,)
    assert config.event_count == 30
    assert config.epochs == 2
    assert config.expected_feature_cells == 4
    assert CANDIDATES == ("girard", "scott", "pca", "combastel")


def test_selection_metrics_use_both_v3_and_predictive_references():
    rows = []
    for seed in (1, 2):
        common = {
            "seed": seed,
            "budget": 40,
            "status": "completed",
            "fpr": 0.1,
            "event_count": 500,
            "event_loop_time_ms": 1000.0,
            "mean_evaluated_leaves": 1.0,
            "max_evaluated_leaves": 1,
        }
        rows.extend((
            {**common, "method": "g15_clean148", "mean_approx_loss": 2.0},
            {**common, "method": "g20_clean148", "mean_approx_loss": float(seed + 1)},
            {**common, "method": "mpc_terminal_beam_predictive_linear", "mean_approx_loss": 1.0},
        ))

    metrics = selection_metrics(pd.DataFrame(rows))

    assert set(metrics["reference_method"]) == {
        "g15_clean148",
        "mpc_terminal_beam_predictive_linear",
    }
    paired = metrics[
        (metrics["method"] == "g20_clean148")
        & (metrics["reference_method"] == "mpc_terminal_beam_predictive_linear")
    ].iloc[0]
    assert paired["worst_seed"] == 2
    assert paired["worst_loss_ratio"] == 3.0


def test_selection_metrics_separate_fixed_traces_that_share_seed_zero():
    rows = []
    for trace_id, loss in (("figure8", 2.0), ("figure8_drift", 3.0)):
        common = {
            "seed": 0,
            "budget": 40,
            "trace_id": trace_id,
            "status": "completed",
            "fpr": 0.0,
            "event_count": 500,
            "event_loop_time_ms": 1000.0,
            "mean_evaluated_leaves": 1.0,
            "max_evaluated_leaves": 1,
        }
        rows.extend((
            {**common, "method": "g15_clean148", "mean_approx_loss": 2.0},
            {**common, "method": "g20_clean148", "mean_approx_loss": loss},
            {**common, "method": "mpc_terminal_beam_predictive_linear", "mean_approx_loss": 1.0},
        ))

    metrics = selection_metrics(pd.DataFrame(rows), bootstrap=False)
    paired = metrics[
        (metrics["method"] == "g20_clean148")
        & (metrics["reference_method"] == "mpc_terminal_beam_predictive_linear")
    ].iloc[0]

    assert paired["trace_count"] == 2
    assert paired["worst_loss_ratio"] == 3.0
    assert paired["worst_trace_id"] == "figure8_drift"
    assert np.isnan(paired["mean_fpr_difference_ci_low"])
    assert np.isnan(paired["mean_fpr_difference_ci_high"])


def test_stale_stage_manifest_is_rejected(tmp_path: Path):
    config = replace(smoke_config(), output=tmp_path)
    stage = tmp_path / "prepare"
    stage.mkdir(parents=True)
    (stage / "manifest.json").write_text(json.dumps({
        "schema": SCHEMA,
        "experiment_fingerprint": "stale-fingerprint",
        "stage": "prepare",
        "status": "completed",
    }))

    with pytest.raises(ValueError, match="stale PRP v4 manifest"):
        _load_stage(config, "prepare")
