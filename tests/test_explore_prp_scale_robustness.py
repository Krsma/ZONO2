from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


TOOLS = Path(__file__).parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from explore_prp_scale_robustness import (  # noqa: E402
    BUDGETS,
    EVALUATION_SEEDS,
    FIXED_TRACE_KINDS,
    OPTIMIZER_SEEDS,
    ExploreConfig,
    ModelVariant,
    _load_manifest,
    aggregate_nominal,
    cell_identity,
    optimizer_stability,
    paired_size_effects,
    select_training_rows,
    smoke_config,
    training_seeds,
)
from pzr.learning.dataset import ReducerCostDataset  # noqa: E402
from pzr.rtlola.paper_pipeline import EvaluationTrace  # noqa: E402
from pzr.rtlola.paper_experiment import TraceSource  # noqa: E402


def test_full_scope_has_exact_nested_models_and_cell_counts():
    config = ExploreConfig()

    assert config.budgets == BUDGETS
    assert config.evaluation_seeds == EVALUATION_SEEDS
    assert config.fixed_trace_kinds == FIXED_TRACE_KINDS
    assert len(config.master_train_seeds) == 148
    assert training_seeds(config, 84)[-1] == 247
    assert training_seeds(config, 100)[-1] == 263
    assert training_seeds(config, 116)[-1] == 279
    assert training_seeds(config, 132)[-1] == 295
    assert training_seeds(config, 148)[-1] == 311
    assert len(config.variants) == 17
    assert len(config.reused_variants) == 5
    assert len(config.new_variants) == 12
    assert config.expected_new_shards == 448
    assert config.expected_new_models == 84
    assert config.expected_nominal_cells == 2380
    assert config.expected_fixed_cells == 504
    assert config.expected_imported_fixed_cells == 84
    assert config.expected_new_fixed_cells == 420
    assert config.expected_reported_cells == 2884
    assert config.expected_new_cells == 2800
    assert {item.name for item in config.variants if item.training_size == 84} == {
        "clean84",
        "clean84_opt1042",
        "clean84_opt2042",
        "clean84_opt3042",
        "clean84_opt4042",
    }


def test_seed_partitions_are_disjoint_and_invalid_overlap_is_rejected():
    config = ExploreConfig()

    assert not set(config.master_train_seeds) & set(config.validation_seeds)
    assert not set(config.master_train_seeds) & set(config.evaluation_seeds)
    assert not set(config.validation_seeds) & set(config.evaluation_seeds)

    with pytest.raises(ValueError, match="must be disjoint"):
        ExploreConfig(evaluation_seeds=(20,))


def test_smoke_scope_exercises_two_sizes_and_optimizer_seeds():
    config = smoke_config()

    assert config.event_count == 100
    assert config.workers == 1
    assert config.variants == (
        ModelVariant(1, 42),
        ModelVariant(1, 1042),
        ModelVariant(2, 42),
        ModelVariant(2, 1042),
    )
    assert config.expected_new_shards == 3
    assert config.expected_new_models == 4
    assert config.expected_nominal_cells == 4
    assert config.expected_fixed_cells == 4
    assert config.expected_reported_cells == 8
    assert config.expected_new_cells == 8


def test_training_selection_is_seed_budget_and_split_aligned():
    rows = [
        (seed, budget)
        for seed in (248, 249, 250)
        for budget in (40, 80)
    ]
    sample_ids = tuple(f"{seed}:{budget}" for seed, budget in rows)
    splits = tuple("validation" if seed == 250 else "train" for seed, _ in rows)
    dataset = ReducerCostDataset(
        features=np.ones((len(rows), 2), dtype=np.float32),
        teacher_costs=np.ones((len(rows), 2), dtype=np.float64),
        feasible=np.ones((len(rows), 2), dtype=np.bool_),
        candidate_names=("girard", "scott"),
        feature_names=("a", "b"),
        splits=splits,
        sample_ids=sample_ids,
    )
    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "split": splits,
            "seed": [seed for seed, _ in rows],
            "budget": [budget for _, budget in rows],
        }
    )

    selected, frame = select_training_rows(
        dataset,
        metadata,
        seeds=(248, 249),
        validation_seeds=(250,),
        budget=40,
    )

    assert tuple(frame["sample_id"]) == selected.sample_ids
    assert set(frame.loc[frame["split"] == "train", "seed"]) == {248, 249}
    assert set(frame.loc[frame["split"] == "validation", "seed"]) == {250}
    assert set(frame["budget"]) == {40}


def _nominal_summary(*, failed: bool = False) -> pd.DataFrame:
    rows = []
    for optimizer_seed in OPTIMIZER_SEEDS:
        for size in (84, 148):
            method = (
                f"clean{size}"
                if optimizer_seed == 42
                else f"clean{size}_opt{optimizer_seed}"
            )
            for seed in (140, 141):
                is_failed = failed and size == 148 and seed == 141
                rows.append(
                    {
                        "trace_id": f"random_waypoint:seed-{seed}",
                        "trace_sha256": f"hash-{seed}",
                        "trace_source": TraceSource.GENERATED_NOMINAL.value,
                        "trace_kind": "random_waypoint",
                        "condition": "random_waypoint",
                        "seed": seed,
                        "budget": 40,
                        "method": method,
                        "training_size": size,
                        "optimizer_seed": optimizer_seed,
                        "status": (
                            "fallback_failed" if is_failed else "completed"
                        ),
                        "false_positive_count": (
                            0 if is_failed else (10 if size == 84 else 5)
                        ),
                        "false_negative_count": 0,
                        "reference_negative_count": 0 if is_failed else 100,
                        "reference_positive_count": 20,
                        "mean_approx_loss": (
                            np.nan if is_failed else (2e-3 if size == 84 else 1e-4)
                        ),
                        "total_time_ms": np.nan if is_failed else 100.0,
                        "event_count": 500,
                    }
                )
    return pd.DataFrame(rows)


def test_nominal_aggregation_is_deterministic_and_failure_aware():
    first = aggregate_nominal(
        _nominal_summary(), bootstrap_replicates=200, bootstrap_seed=7
    )
    second = aggregate_nominal(
        _nominal_summary(), bootstrap_replicates=200, bootstrap_seed=7
    )
    pd.testing.assert_frame_equal(first, second)

    clean84 = first[
        (first["training_size"] == 84) & (first["optimizer_seed"] == 42)
    ].iloc[0]
    clean148 = first[
        (first["training_size"] == 148) & (first["optimizer_seed"] == 42)
    ].iloc[0]
    assert np.isclose(clean84["macro_fpr"], 0.1)
    assert clean84["high_loss_count"] == 2
    assert np.isclose(clean148["macro_fpr"], 0.05)
    assert clean148["high_loss_count"] == 0

    failed = aggregate_nominal(
        _nominal_summary(failed=True),
        bootstrap_replicates=20,
        bootstrap_seed=7,
    )
    failed148 = failed[failed["training_size"] == 148]
    assert not failed148["available"].any()
    assert failed148["macro_fpr"].isna().all()


def test_optimizer_summary_uses_five_training_replicates_not_trace_products():
    aggregate = aggregate_nominal(
        _nominal_summary(), bootstrap_replicates=20, bootstrap_seed=7
    )

    summary, differences = optimizer_stability(aggregate)

    assert set(summary["optimizer_count"]) == {5}
    assert len(summary) == 2 * 3
    assert len(differences) == 5
    assert set(differences["optimizer_seed"]) == set(OPTIMIZER_SEEDS)
    assert np.allclose(differences["macro_fpr_difference"], -0.05)


def test_size_effects_are_trace_paired_and_deterministic():
    first = paired_size_effects(
        _nominal_summary(),
        bootstrap_replicates=200,
        bootstrap_seed=11,
    )
    second = paired_size_effects(
        _nominal_summary(),
        bootstrap_replicates=200,
        bootstrap_seed=11,
    )
    pd.testing.assert_frame_equal(first, second)

    clean148 = first[first["training_size"] == 148].iloc[0]
    assert clean148["pair_count"] == 2
    assert clean148["valid_pair_count"] == 2
    assert clean148["available"]
    assert np.isclose(clean148["mean_fpr_difference"], -0.05)
    assert np.isclose(clean148["geometric_mean_loss_ratio"], 0.05)
    assert np.isclose(clean148["high_loss_rate_difference"], -1.0)


def test_cell_identity_records_frozen_model_training_and_trace(tmp_path):
    config = smoke_config()
    reference = tmp_path / "reference.json"
    reference.write_text("{}")
    trace = EvaluationTrace(
        trace_id="random_waypoint:seed-140",
        condition="random_waypoint",
        seed=140,
        events=(object(),) * 100,
        trace_sha256="trace-hash",
        trace_source=TraceSource.GENERATED_NOMINAL,
        trace_kind="random_waypoint",
        provenance={"generator_config_sha256": "generator-hash"},
    )
    variant = ModelVariant(2, 1042)

    identity = cell_identity(
        config,
        scope="nominal",
        trace=trace,
        budget=40,
        variant=variant,
        reference=reference,
        record={"sha256": "model-hash"},
        freeze_hash="freeze-hash",
    )

    assert identity["training_size"] == 2
    assert identity["training_seeds"] == [248, 249]
    assert identity["optimizer_seed"] == 1042
    assert identity["model_training_budget"] == 40
    assert identity["model_freeze_sha256"] == "freeze-hash"
    assert identity["trace_source"] == TraceSource.GENERATED_NOMINAL.value
    assert identity["fingerprint"]


def test_stale_stage_manifest_is_rejected(tmp_path):
    config = replace_output(smoke_config(), tmp_path)
    path = tmp_path / "check" / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema": "pzr.prp-scale-robustness-exploratory.v1",
                "experiment_fingerprint": "stale",
            }
        )
    )

    with pytest.raises(ValueError, match="stale"):
        _load_manifest(config, "check")


def replace_output(config: ExploreConfig, output: Path) -> ExploreConfig:
    return ExploreConfig(
        output=output,
        budgets=config.budgets,
        base_train_seeds=config.base_train_seeds,
        new_train_seeds=config.new_train_seeds,
        validation_seeds=config.validation_seeds,
        evaluation_seeds=config.evaluation_seeds,
        size_curve=config.size_curve,
        new_sizes=config.new_sizes,
        optimizer_seeds=config.optimizer_seeds,
        stability_sizes=config.stability_sizes,
        fixed_trace_kinds=config.fixed_trace_kinds,
        event_count=config.event_count,
        workers=config.workers,
        reuse_existing=config.reuse_existing,
        smoke=config.smoke,
    )
