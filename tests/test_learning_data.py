from types import SimpleNamespace

import numpy as np

from pzr.rtlola.features import RTL_RANKING_FEATURE_NAMES
from pzr.rtlola.learning_data import (
    CollectedReducerCostSample,
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
        teacher_sequence=("girard", "none"), collection_mode="teacher",
        executed_action="girard", disturbed=False, disturbance_probability=0.0,
        infeasible_probability_redirected=0.0, sampled_normalized_regret=float("nan"),
        dart_calibration_sha256=None, evaluated_leaves=3,
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
    assert list(metadata["executed_action"]) == ["girard", "girard"]


def test_teacher_root_costs_align_by_name_and_mask_incomplete_roots():
    rows = (
        SimpleNamespace(root_action="scott", feasible=True, complete=False, predicted_cost=float("nan")),
        SimpleNamespace(root_action="girard", feasible=True, complete=True, predicted_cost=3.0),
    )
    costs, feasible = _aligned_root_costs(rows, ("girard", "scott"))
    np.testing.assert_array_equal(feasible, [True, False])
    assert costs[0] == 3.0
    assert np.isnan(costs[1])
