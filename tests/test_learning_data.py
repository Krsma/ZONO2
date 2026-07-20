from types import SimpleNamespace

import numpy as np

from pzr.rtlola.learning_collection import write_collection_summaries
from pzr.rtlola.features import RTL_RANKING_FEATURE_NAMES
from pzr.rtlola.learning_data import (
    CollectedReducerCostSample,
    DartDecisionMetadata,
    _aligned_root_costs,
    build_reducer_cost_dataset,
)


def _sample(sample_id: str, split: str) -> CollectedReducerCostSample:
    return CollectedReducerCostSample(
        sample_id=sample_id, trace_id=sample_id.split(":")[0], split=split,
        condition="random_waypoint", seed=1, budget=40, step=3,
        features=np.arange(len(RTL_RANKING_FEATURE_NAMES), dtype=np.float32),
        candidate_names=("girard", "scott"), teacher_costs=(0.0, 2.0),
        feasible=(True, True), teacher_action="girard",
        teacher_sequence=("girard", "none"), evaluated_leaves=3,
        teacher_reducer_failure_count=1, teacher_infeasible_candidate_count=1,
        execution_fallback_used=False,
    )


def test_collected_samples_build_aligned_non_empty_dataset():
    dataset, metadata = build_reducer_cost_dataset((
        _sample("trace-a:0", "train"), _sample("trace-b:0", "validation"),
    ))
    assert dataset.num_samples == 2
    assert dataset.candidate_names == ("girard", "scott")
    assert dataset.splits == ("train", "validation")
    assert "executed_action" not in metadata
    assert "disturbed" not in metadata


def test_dart_decision_metadata_is_optional_and_schema_specific():
    sample = _sample("trace-a:0", "train")
    dart_sample = CollectedReducerCostSample(
        **{
            **sample.__dict__,
            "dart": DartDecisionMetadata(
                executed_action="scott",
                disturbed=True,
                disturbance_eligible=True,
                disturbance_attempted=True,
                recovery_forced=False,
                target_disturbance_rate=0.2,
                injection_probability=0.3,
                disturbance_probability=0.3,
                regret_cap=1.0,
                selected_direction_probability=1.0,
                sampled_normalized_regret=0.5,
                calibration_sha256="calibration",
            ),
        }
    )
    _, metadata = build_reducer_cost_dataset((dart_sample,))
    assert metadata.loc[0, "executed_action"] == "scott"
    assert bool(metadata.loc[0, "disturbed"])
    assert metadata.loc[0, "dart_calibration_sha256"] == "calibration"


def test_clean_and_dart_collection_reports_have_distinct_schemas(tmp_path):
    _, clean = build_reducer_cost_dataset((_sample("trace-a:0", "train"),))
    clean_dir = tmp_path / "clean"
    write_collection_summaries(clean, clean_dir)
    assert (clean_dir / "collection_summary.csv").is_file()
    assert (clean_dir / "teacher_action_counts.csv").is_file()
    assert not (clean_dir / "dart_collection_summary.csv").exists()
    assert not (clean_dir / "teacher_executed_confusion.csv").exists()

    sample = _sample("trace-b:0", "train")
    dart = CollectedReducerCostSample(
        **{
            **sample.__dict__,
            "dart": DartDecisionMetadata(
                executed_action="scott",
                disturbed=True,
                disturbance_eligible=True,
                disturbance_attempted=True,
                recovery_forced=False,
                target_disturbance_rate=0.2,
                injection_probability=0.3,
                disturbance_probability=0.3,
                regret_cap=1.0,
                selected_direction_probability=1.0,
                sampled_normalized_regret=0.5,
                calibration_sha256="calibration",
            ),
        }
    )
    _, dart_metadata = build_reducer_cost_dataset((dart,))
    dart_dir = tmp_path / "dart"
    write_collection_summaries(dart_metadata, dart_dir)
    assert (dart_dir / "dart_collection_summary.csv").is_file()
    assert (dart_dir / "executed_action_counts.csv").is_file()
    assert (dart_dir / "teacher_executed_confusion.csv").is_file()


def test_teacher_root_costs_align_by_name_and_mask_incomplete_roots():
    rows = (
        SimpleNamespace(root_action="scott", feasible=True, complete=False, predicted_cost=float("nan")),
        SimpleNamespace(root_action="girard", feasible=True, complete=True, predicted_cost=3.0),
    )
    costs, feasible = _aligned_root_costs(rows, ("girard", "scott"))
    np.testing.assert_array_equal(feasible, [True, False])
    assert costs[0] == 3.0
    assert np.isnan(costs[1])
