import numpy as np
import pytest

pytest.importorskip("rlola_python_binding")

from pzr.learning.ranking import RegretRankingPolicy
from pzr.rtlola.actions import default_action_catalog
from pzr.rtlola.benchmark import RtlolaBenchmarkConfig, summarize_results
from pzr.rtlola.engine import RtlolaEngine
from pzr.rtlola.learning import (
    RTL_FEATURE_NAMES,
    RtlolaLearnedPolicy,
    _evaluate_candidates,
    train_and_evaluate_regret,
    write_regret_artifacts,
)
from pzr.rtlola.learning_data import build_ranking_dataset, collect_teacher_episode
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


def test_teacher_costs_force_each_root_then_use_shared_candidate_pool():
    catalog, events, engine, state = _overflow_state()

    rows = _evaluate_candidates(
        engine,
        state,
        events[12],
        events[13:15],
        catalog,
        RtlolaBenchmarkConfig(budget=10, horizon=2, beam_width=2),
    )

    assert {row.name for row in rows} == set(catalog.mpc_candidate_names)
    assert all(row.sequence[0] == row.name for row in rows)


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
    dataset, metadata = build_ranking_dataset(samples)

    assert dataset.num_samples > 0
    assert dataset.candidate_names == ("girard", "scott")
    assert dataset.teacher_costs.shape == (dataset.num_samples, 2)
    assert np.all(np.any(dataset.feasible, axis=1))
    assert np.all(np.sum(dataset.tie_mask, axis=1) >= 1)
    assert set(metadata["teacher_action"]) <= {"girard", "scott", "interval"}


def test_learned_policy_selects_one_direct_binding_action():
    catalog, events, engine, state = _overflow_state()
    num_features = len(RTL_FEATURE_NAMES)
    policy = RegretRankingPolicy(
        candidate_names=catalog.mpc_candidate_names,
        feature_mean=np.zeros(num_features),
        feature_std=np.ones(num_features),
        weights=[np.zeros((num_features, 4))],
        biases=[np.array([0.0, 1.0, 2.0, 3.0])],
        feature_names=RTL_FEATURE_NAMES,
    )

    decision = RtlolaLearnedPolicy(policy, catalog).choose(
        engine,
        state,
        events[12],
        events[13:15],
        budget=10,
    )

    assert decision.first_action.name == "girard"
    assert decision.predicted_sequence == ("girard",)
    assert decision.evaluated_leaves == 1


def test_regret_training_smoke_is_scenario_generic_and_writes_artifacts(tmp_path):
    config = RtlolaBenchmarkConfig(
        scenario="omni_robot",
        length=14,
        seeds=1,
        budget=10,
        horizon=1,
        beam_width=2,
        regret_iterations=1,
        regret_epochs=2,
        regret_train_seeds=1,
        regret_eval_seeds=1,
    )

    result = train_and_evaluate_regret(config)
    write_regret_artifacts(
        result,
        tmp_path,
        metadata={"scenario": config.scenario},
    )

    assert result.traces
    assert result.eval_results
    assert {trace.seed for trace in result.traces} == {10_000}
    assert {run.seed for run in result.eval_results} == {0}
    learned_summary = summarize_results(result.eval_results)
    assert np.isfinite(learned_summary["mean_approx_loss"]).all()
    assert result.policy.candidate_names == default_action_catalog().mpc_candidate_names
    assert (tmp_path / "learned_direct_ranker.npz").stat().st_size > 0
    assert (tmp_path / "regret_candidate_costs.csv").stat().st_size > 0
    assert (tmp_path / "regret_metadata.json").stat().st_size > 0


def test_regret_training_rejects_overlapping_seed_ranges():
    with pytest.raises(ValueError, match="seed ranges overlap"):
        train_and_evaluate_regret(RtlolaBenchmarkConfig(
            scenario="omni_robot",
            length=14,
            seeds=1,
            budget=10,
            horizon=1,
            beam_width=2,
            regret_iterations=1,
            regret_epochs=1,
            regret_train_seeds=2,
            regret_eval_seeds=2,
            regret_train_seed_start=0,
            regret_eval_seed_start=1,
        ))


def test_robot_arm_regret_training_uses_same_generic_pipeline():
    result = train_and_evaluate_regret(RtlolaBenchmarkConfig(
        scenario="robot_arm",
        trace_kind="figure8",
        length=7,
        seeds=1,
        budget=80,
        horizon=1,
        beam_width=2,
        regret_iterations=1,
        regret_epochs=1,
        regret_train_seeds=1,
        regret_eval_seeds=1,
        regret_budgets=[40, 80],
        regret_train_trace_kinds=["figure8", "square"],
        regret_eval_trace_kinds=["figure8_drift", "square_drift"],
    ))

    assert result.traces
    assert {
        (run.budget, run.trace_kind)
        for run in result.eval_results
    } == {
        (40, "figure8_drift"),
        (80, "figure8_drift"),
        (40, "square_drift"),
        (80, "square_drift"),
    }
    assert all(run.method == "learned_direct" for run in result.eval_results)
    assert {trace.budget for trace in result.traces} == {40, 80}
    assert {trace.trace_kind for trace in result.traces} == {"figure8", "square"}
