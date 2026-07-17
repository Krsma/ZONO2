import numpy as np
import pytest
import torch

pytest.importorskip("rlola_python_binding")

from pzr.learning.dart import DartCalibration
from pzr.learning.ranker import FeatureNormalizer, ReducerPolicy, ReducerScorer
from pzr.learning.targets import PAIRWISE_OBJECTIVE_CONTRACT, tolerant_best_mask
from pzr.rtlola.actions import default_action_catalog
from pzr.rtlola.benchmark import (
    RtlolaBenchmarkConfig,
    run_direct_policy_benchmark,
)
from pzr.rtlola.engine import RtlolaEngine
from pzr.rtlola.learning_data import build_reducer_cost_dataset, collect_teacher_episode
from pzr.rtlola.learned_policy import RtlolaReducerPolicy
from pzr.rtlola.features import RTL_RANKING_FEATURE_SCHEMA
from pzr.rtlola.omni import OMNI_EXPECTED_VERDICT_KEYS, OMNI_SPEC, generate_omni_events
from pzr.rtlola.scenarios import scenario_by_name
from pzr.rtlola.search import full_width_terminal_search


def _overflow_state():
    catalog = default_action_catalog()
    events = generate_omni_events(16, seed=4)
    engine = RtlolaEngine(
        OMNI_SPEC,
        event_arity=3,
        expected_verdict_keys=OMNI_EXPECTED_VERDICT_KEYS,
    )
    for step, event in enumerate(events[:12]):
        engine.live_step(event, catalog.no_op, budget=20, step=step + 1)
    return catalog, events, engine, engine.snapshot(step=12, time=events[11].time)


def _direct_policy(candidate_names=("girard", "scott")):
    model = ReducerScorer(RTL_RANKING_FEATURE_SCHEMA, candidate_names)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.network[-1].bias.copy_(torch.arange(len(candidate_names)))
    normalizer = FeatureNormalizer(
        mean=np.zeros(len(RTL_RANKING_FEATURE_SCHEMA.feature_names), dtype=np.float32),
        std=np.ones(len(RTL_RANKING_FEATURE_SCHEMA.feature_names), dtype=np.float32),
    )
    catalog = default_action_catalog(candidate_names)
    return RtlolaReducerPolicy(
        ReducerPolicy(model, normalizer, PAIRWISE_OBJECTIVE_CONTRACT), catalog,
    )


def test_full_width_teacher_scores_all_roots_without_mutating_live_state():
    _, events, engine, state = _overflow_state()
    catalog = default_action_catalog(("girard", "scott"))
    live_before = engine.snapshot(step=12, time=events[11].time)
    live_before_matrices = engine.matrices(live_before)

    decision = full_width_terminal_search(
        engine,
        state,
        events[12],
        events[13],
        catalog.mpc_candidates,
        budget=10,
        fallback=catalog.fallback,
        none_action=catalog.no_op,
    )

    assert {row.root_action for row in decision.root_evaluations} == {
        "girard", "scott",
    }
    assert decision.evaluated_leaves > 0
    assert np.isfinite(decision.predicted_cost)
    assert decision.predicted_sequence[0] == decision.first_action.name
    live_after = engine.snapshot(step=12, time=events[11].time)
    for actual, expected in zip(engine.matrices(live_after), live_before_matrices):
        np.testing.assert_allclose(actual, expected)


def test_teacher_collection_writes_aligned_binding_backed_samples():
    events = generate_omni_events(16, seed=4)
    samples = collect_teacher_episode(
        scenario=scenario_by_name("omni_robot"),
        events=events,
        trace_id="omni-seed-4",
        split="train",
        condition="omni",
        seed=4,
        budget=10,
        candidate_names=("girard", "scott"),
    )
    dataset, metadata = build_reducer_cost_dataset(samples)

    assert dataset.num_samples > 0
    assert dataset.candidate_names == ("girard", "scott")
    assert dataset.teacher_costs.shape == (dataset.num_samples, 2)
    assert np.all(np.any(dataset.feasible, axis=1))
    assert np.all(np.sum(tolerant_best_mask(dataset.teacher_costs, dataset.feasible), axis=1) >= 1)
    assert set(metadata["teacher_action"]) <= {"girard", "scott", "interval"}


def test_pytorch_policy_uses_current_state_only_for_direct_inference():
    _, events, engine, state = _overflow_state()

    decision = _direct_policy().choose(engine, state, events[12], budget=10)

    assert decision.first_action.name == "girard"
    assert decision.predicted_sequence == ("girard",)
    assert decision.evaluated_leaves == 1
    assert decision.mpc_variant == "learned_direct"


def test_direct_policy_features_do_not_depend_on_current_event_values():
    _, events, engine, state = _overflow_state()

    class RecordingPolicy:
        candidate_names = ("girard", "scott")
        feature_schema = RTL_RANKING_FEATURE_SCHEMA

        def __init__(self):
            self.features = []

        def predict_scores(self, features):
            self.features.append(np.asarray(features).copy())
            return np.asarray([0.0, 1.0], dtype=np.float32)

    ranker = RecordingPolicy()
    policy = RtlolaReducerPolicy(
        ranker, default_action_catalog(ranker.candidate_names),
    )

    first = policy.choose(engine, state, events[12], budget=10)
    second = policy.choose(engine, state, events[13], budget=10)

    assert first.first_action.name == second.first_action.name == "girard"
    np.testing.assert_array_equal(ranker.features[0], ranker.features[1])


def test_dart_collection_executes_one_step_disturbances_under_teacher_control():
    events = generate_omni_events(16, seed=4)
    calibration = DartCalibration(
        candidate_names=("girard", "scott"),
        budgets=(10,),
        probabilities=np.asarray([[[0.0, 1.0], [1.0, 0.0]]]),
        row_counts=np.asarray([[10, 10]]),
        context={},
    )
    samples = collect_teacher_episode(
        scenario=scenario_by_name("omni_robot"),
        events=events,
        trace_id="omni-dart-seed-4",
        split="train",
        condition="omni",
        seed=4,
        budget=10,
        candidate_names=("girard", "scott"),
        collection_mode="dart",
        dart_calibration=calibration,
        dart_calibration_sha256="calibration-hash",
        disturbance_seed=7,
    )

    assert samples
    assert {sample.collection_mode for sample in samples} == {"dart"}
    assert {sample.executed_action for sample in samples} <= {"girard", "scott", "interval"}
    assert any(sample.disturbed for sample in samples)


def test_direct_policy_benchmark_uses_standard_exact_metric_schema(tmp_path):
    config = RtlolaBenchmarkConfig(
        scenario="omni_robot",
        length=14,
        seeds=1,
        budget=10,
        reference_mode="exact",
        reference_cache=str(tmp_path / "reference.json"),
        mpc_candidate_names=["girard", "scott"],
    )

    result = run_direct_policy_benchmark(config, _direct_policy())

    assert not result.failures
    assert result.raw_results
    assert result.raw_results[0].method == "learned_direct"
    assert np.isfinite(result.summary["mean_approx_loss"]).all()
    assert {"fpr", "fnr", "fallback_rate"} <= set(result.summary.columns)
