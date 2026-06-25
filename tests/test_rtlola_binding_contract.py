import csv

import numpy as np
import pytest

try:
    import rlola_python_binding as rlola
except ImportError as exc:
    pytest.skip(f"rlola_python_binding unavailable: {exc}", allow_module_level=True)

from pzr.rtlola.actions import action_by_name, default_actions
from pzr.rtlola.engine import RtlolaEngine
from pzr.rtlola.learning import _evaluate_candidates
from pzr.rtlola.omni import OMNI_EXPECTED_VERDICT_KEYS, OMNI_SPEC, generate_omni_events
from pzr.rtlola.robot_arm import (
    ARM_EXPECTED_VERDICT_KEYS,
    ARM_PUBLIC_STREAM_KEYS,
    ARM_SPEC,
    DEFAULT_TRACE_KIND,
    forward_kinematics_5dof,
    generate_robot_arm_events,
    mujoco_tcp_position,
)
from pzr.rtlola.runner import (
    RtlolaBenchmarkConfig,
    infer_fresh_generator_reserve,
    methods_for_config,
    mpc_actions,
    run_benchmark,
)
from pzr.rtlola.scenarios import scenario_by_name
from pzr.rtlola.cli import main as rtlola_main
from pzr.rtlola.search import beam_search


def test_omni_monitor_constructs_and_reports_expected_trigger_keys():
    engine = RtlolaEngine(
        OMNI_SPEC,
        event_arity=3,
        expected_verdict_keys=OMNI_EXPECTED_VERDICT_KEYS,
    )
    event = generate_omni_events(1, seed=0)[0]
    action = action_by_name(default_actions())["none"]

    result = engine.live_step(event, action, budget=20, step=1)

    for key in OMNI_EXPECTED_VERDICT_KEYS:
        assert key in result.verdict
    assert result.metrics.dynamic_generator_count >= 0
    assert result.metrics.total_generator_count >= result.metrics.dynamic_generator_count


def test_repeated_branching_from_same_snapshot_is_deterministic():
    engine = RtlolaEngine(
        OMNI_SPEC,
        event_arity=3,
        expected_verdict_keys=OMNI_EXPECTED_VERDICT_KEYS,
    )
    event = generate_omni_events(1, seed=1)[0]
    state = engine.snapshot(step=0, time=event.time)
    action = action_by_name(default_actions())["girard"]

    left = engine.branch_step(state, event, action, budget=10)
    right = engine.branch_step(state, event, action, budget=10)

    comparable_keys = set(left.verdict) | set(right.verdict)
    comparable_keys.discard("runtime_ns")
    assert {key: left.verdict[key] for key in comparable_keys} == {
        key: right.verdict[key] for key in comparable_keys
    }
    np.testing.assert_allclose(
        engine.matrices(left.state)[0],
        engine.matrices(right.state)[0],
    )


def test_beam_search_returns_budgeted_first_action():
    engine = RtlolaEngine(
        OMNI_SPEC,
        event_arity=3,
        expected_verdict_keys=OMNI_EXPECTED_VERDICT_KEYS,
    )
    events = generate_omni_events(4, seed=2)
    actions = default_actions()
    by_name = action_by_name(actions)

    result = beam_search(
        engine,
        engine.snapshot(step=0, time=events[0].time),
        events[0],
        events[1:],
        actions,
        budget=10,
        beam_width=2,
        fallback=by_name["interval"],
    )

    assert result.first_action_budget == 10
    assert result.predicted_sequence


def test_mpc_actions_exclude_opportunistic_and_fallback_actions():
    actions = action_by_name(default_actions())

    names = tuple(action.name for action in mpc_actions(actions))

    assert names == ("girard", "scott", "interval_hull", "pca")
    assert "althoff_a" not in names
    assert "colinear_scale" not in names
    assert "colinear" not in names
    assert "interval" not in names


def test_beam_search_uses_none_while_within_budget_and_reduces_on_overflow():
    engine = RtlolaEngine(
        OMNI_SPEC,
        event_arity=3,
        expected_verdict_keys=OMNI_EXPECTED_VERDICT_KEYS,
    )
    events = generate_omni_events(13, seed=2)
    actions = action_by_name(default_actions())
    candidates = mpc_actions(actions)

    under_budget = beam_search(
        engine,
        engine.snapshot(step=0, time=events[0].time),
        events[0],
        events[1:3],
        candidates,
        budget=10,
        beam_width=2,
        fallback=actions["interval"],
        none_action=actions["none"],
    )

    assert under_budget.first_action.name == "none"
    assert under_budget.predicted_sequence == ("none",)

    for step, event in enumerate(events[:11]):
        engine.live_step(event, actions["none"], budget=20, step=step + 1)
    overflow_state = engine.snapshot(step=11, time=events[10].time)
    assert engine.metrics(overflow_state).dynamic_generator_count > 10

    reduced = beam_search(
        engine,
        overflow_state,
        events[11],
        events[12:],
        candidates,
        budget=10,
        beam_width=2,
        fallback=actions["interval"],
        none_action=actions["none"],
    )

    candidate_names = {action.name for action in candidates}
    assert reduced.first_action.name != "none"
    assert reduced.first_action.name in candidate_names | {"interval"}
    assert "interval" not in candidate_names
    assert reduced.first_action_budget == 10


def test_results_use_state_zonotope_metric_names_with_real_exact_metrics():
    result = run_benchmark(RtlolaBenchmarkConfig(
        length=4,
        seeds=1,
        budget=10,
        method_set="static",
    ))

    columns = set(result.timeseries.columns)
    assert "state_zonotope_width_sum" in columns
    assert "exact_state_zonotope_width_sum" in columns
    assert "state_zonotope_approx_error_sum" in columns
    assert "trigger_width_sum" not in columns
    assert "approx_error_sum" not in columns

    none_rows = result.timeseries[result.timeseries["method"] == "none"]
    assert not none_rows.empty
    assert np.allclose(none_rows["state_zonotope_approx_error_sum"], 0.0)
    assert np.allclose(
        none_rows["state_zonotope_width_sum"],
        none_rows["exact_state_zonotope_width_sum"],
    )
    assert "mean_state_zonotope_width" in result.summary.columns
    assert "mean_trigger_width" not in result.summary.columns


def test_custom_methods_override_method_set_and_validate_names():
    config = RtlolaBenchmarkConfig(method_set="all", methods=["colinear", "mpc_beam"])
    assert methods_for_config(config) == ("colinear", "mpc_beam")

    with pytest.raises(ValueError, match="unknown RTLola method"):
        methods_for_config(RtlolaBenchmarkConfig(methods=["colinear", "missing"]))


def test_low_budget_below_state_dimension_uses_unbounded_fallback_without_panic():
    engine = RtlolaEngine(
        OMNI_SPEC,
        event_arity=3,
        expected_verdict_keys=OMNI_EXPECTED_VERDICT_KEYS,
    )
    events = generate_omni_events(6, seed=0)
    actions = action_by_name(default_actions())

    for step, event in enumerate(events[:4]):
        engine.live_step(event, actions["none"], budget=20, step=step + 1)
    state = engine.snapshot(step=4, time=events[3].time)

    with pytest.raises(ValueError, match="below the current state-zonotope dimension"):
        engine.branch_step(state, events[4], actions["girard"], budget=3)

    decision = beam_search(
        engine,
        state,
        events[4],
        (),
        (actions["girard"],),
        budget=3,
        beam_width=2,
        fallback=actions["interval"],
    )

    assert decision.first_action.name == "interval"
    assert decision.fallback_used
    assert decision.reducer_failure_count == 1


def test_live_steps_reject_repeated_event_time():
    engine = RtlolaEngine(
        OMNI_SPEC,
        event_arity=3,
        expected_verdict_keys=OMNI_EXPECTED_VERDICT_KEYS,
    )
    event = generate_omni_events(1, seed=0)[0]
    action = action_by_name(default_actions())["none"]

    engine.live_step(event, action, budget=20, step=1)
    with pytest.raises(ValueError, match="strictly increasing"):
        engine.live_step(event, action, budget=20, step=2)


def test_engine_approx_loss_matches_binding_interval_error_and_restores_planner():
    engine = RtlolaEngine(
        OMNI_SPEC,
        event_arity=3,
        expected_verdict_keys=OMNI_EXPECTED_VERDICT_KEYS,
    )
    events = generate_omni_events(7, seed=0)
    actions = action_by_name(default_actions())

    for step, event in enumerate(events[:4]):
        engine.live_step(event, actions["none"], budget=20, step=step + 1)
    anchor = engine.snapshot(step=4, time=events[3].time)
    ref = engine.branch_step(anchor, events[4], actions["none"], budget=10)
    cand = engine.branch_step(anchor, events[4], actions["girard"], budget=10)
    engine.planner.apply_state(anchor.state)
    anchor_matrix = np.asarray(engine.planner.current_zonotope(True), dtype=np.float64)

    loss = engine.approx_loss(ref.state, cand.state)

    np.testing.assert_allclose(
        np.asarray(engine.planner.current_zonotope(True), dtype=np.float64),
        anchor_matrix,
    )
    ref_matrix = engine.matrices(ref.state)[1]
    cand_matrix = engine.matrices(cand.state)[1]
    ref_radius = np.abs(ref_matrix[:, 1:]).sum(axis=1)
    cand_radius = np.abs(cand_matrix[:, 1:]).sum(axis=1)
    ref_center = ref_matrix[:, 0]
    cand_center = cand_matrix[:, 0]
    upper_error = (cand_center + cand_radius) - (ref_center + ref_radius)
    lower_error = (cand_center - cand_radius) - (ref_center - ref_radius)
    expected = (
        np.sum(upper_error * upper_error) + np.sum(lower_error * lower_error)
    ) / (2.0 * ref_matrix.shape[0])
    assert loss == pytest.approx(expected)


def test_regret_candidate_table_does_not_mislabel_fallback_first_action():
    engine = RtlolaEngine(
        OMNI_SPEC,
        event_arity=3,
        expected_verdict_keys=OMNI_EXPECTED_VERDICT_KEYS,
    )
    events = generate_omni_events(6, seed=1)
    actions = action_by_name(default_actions())
    for step, event in enumerate(events[:4]):
        engine.live_step(event, actions["none"], budget=20, step=step + 1)
    state = engine.snapshot(step=4, time=events[3].time)

    rows = _evaluate_candidates(
        engine,
        state,
        events[4],
        (),
        (actions["none"], actions["colinear"]),
        RtlolaBenchmarkConfig(budget=3, beam_width=2),
        actions["colinear"],
    )

    assert [row.name for row in rows] == ["colinear"]


def test_robot_arm_monitor_processes_none_static_and_mpc_actions():
    engine = RtlolaEngine(
        ARM_SPEC,
        event_arity=6,
        expected_verdict_keys=(*ARM_EXPECTED_VERDICT_KEYS, *ARM_PUBLIC_STREAM_KEYS),
    )
    events = generate_robot_arm_events(4, trace_kind=DEFAULT_TRACE_KIND)
    actions = default_actions()
    by_name = action_by_name(actions)

    none = engine.live_step(events[0], by_name["none"], budget=80, step=1)
    assert none.metrics.dynamic_generator_count >= 0
    for key in (*ARM_EXPECTED_VERDICT_KEYS, *ARM_PUBLIC_STREAM_KEYS):
        assert key in none.verdict

    static = engine.live_step(events[1], by_name["girard"], budget=80, step=2)
    assert static.action_name == "girard"

    decision = beam_search(
        engine,
        engine.snapshot(step=2, time=events[1].time),
        events[2],
        events[3:],
        mpc_actions(by_name),
        budget=80,
        beam_width=2,
        fallback=by_name["interval"],
        none_action=by_name["none"],
    )
    assert decision.first_action_budget == 80


def test_fresh_generator_reserve_is_inferred_without_hardcoding():
    actions = action_by_name(default_actions())

    omni_events = generate_omni_events(6, seed=0)
    omni_engine = RtlolaEngine(
        OMNI_SPEC,
        event_arity=3,
        expected_verdict_keys=OMNI_EXPECTED_VERDICT_KEYS,
    )
    omni_pre = omni_engine.metrics(omni_engine.snapshot(step=0, time=omni_events[0].time))
    omni_post = omni_engine.live_step(omni_events[0], actions["none"], budget=0, step=1)
    assert infer_fresh_generator_reserve(
        scenario_by_name("omni_robot"),
        omni_events,
        actions,
    ) == omni_post.metrics.dynamic_generator_count - omni_pre.dynamic_generator_count

    arm_events = generate_robot_arm_events(6, trace_kind=DEFAULT_TRACE_KIND)
    arm_engine = RtlolaEngine(
        ARM_SPEC,
        event_arity=6,
        expected_verdict_keys=(*ARM_EXPECTED_VERDICT_KEYS, *ARM_PUBLIC_STREAM_KEYS),
    )
    arm_pre = arm_engine.metrics(arm_engine.snapshot(step=0, time=arm_events[0].time))
    arm_post = arm_engine.live_step(arm_events[0], actions["none"], budget=0, step=1)
    assert infer_fresh_generator_reserve(
        scenario_by_name("robot_arm"),
        arm_events,
        actions,
    ) == arm_post.metrics.dynamic_generator_count - arm_pre.dynamic_generator_count


def test_robot_arm_mpc_budget_is_exact_rtlola_bound_not_post_event_cap():
    engine = RtlolaEngine(
        ARM_SPEC,
        event_arity=6,
        expected_verdict_keys=(*ARM_EXPECTED_VERDICT_KEYS, *ARM_PUBLIC_STREAM_KEYS),
    )
    events = generate_robot_arm_events(8, trace_kind=DEFAULT_TRACE_KIND)
    actions = action_by_name(default_actions())

    for step, event in enumerate(events[:4]):
        engine.live_step(event, actions["none"], budget=200, step=step + 1)
    state = engine.snapshot(step=4, time=events[3].time)
    assert engine.metrics(state).dynamic_generator_count <= 160
    none = engine.branch_step(state, events[4], actions["none"], budget=160)
    assert none.metrics.dynamic_generator_count > 160

    under_bound = beam_search(
        engine,
        state,
        events[4],
        (),
        mpc_actions(actions),
        budget=160,
        beam_width=4,
        fallback=actions["interval"],
        none_action=actions["none"],
    )

    assert under_bound.first_action.name == "none"
    assert under_bound.first_step.metrics.dynamic_generator_count > 160

    committed = engine.live_step(events[4], actions["none"], budget=160, step=5)
    assert committed.metrics.dynamic_generator_count > 160
    over_bound_state = engine.snapshot(step=5, time=events[4].time)
    assert engine.metrics(over_bound_state).dynamic_generator_count > 160

    decision = beam_search(
        engine,
        over_bound_state,
        events[5],
        (),
        mpc_actions(actions),
        budget=160,
        beam_width=4,
        fallback=actions["interval"],
        none_action=actions["none"],
    )

    assert decision.first_action.name in {
        "girard",
        "scott",
        "interval_hull",
        "pca",
    }
    assert decision.first_action.name != "interval"
    assert decision.first_action_budget == 160
    assert not decision.fallback_used


def test_robot_arm_benchmark_records_zero_none_loss_and_arm_metrics():
    result = run_benchmark(RtlolaBenchmarkConfig(
        scenario="robot_arm",
        trace_kind=DEFAULT_TRACE_KIND,
        length=4,
        seeds=1,
        budget=80,
        methods=["none", "girard"],
    ))

    none_rows = result.timeseries[result.timeseries["method"] == "none"]
    assert not none_rows.empty
    assert np.allclose(none_rows["approx_loss"], 0.0)
    assert np.allclose(none_rows["state_zonotope_approx_error_sum"], 0.0)
    girard_rows = result.timeseries[result.timeseries["method"] == "girard"]
    assert np.isfinite(girard_rows["approx_loss"]).all()
    assert not np.allclose(
        girard_rows["approx_loss"],
        girard_rows["state_zonotope_approx_error_sum"],
    )
    assert {
        "relevant_state_width_sum",
        "dist_to_expected_lower",
        "tpl_upper",
        "fallback_used",
        "reducer_failure_count",
        "infeasible_candidate_count",
        "post_event_over_bound",
        "active_dynamic_generator_count",
        "zero_dynamic_generator_count",
    } <= set(result.timeseries.columns)
    assert "mean_approx_loss" in result.summary.columns
    assert {
        "fallback_count",
        "fallback_rate",
        "reducer_failure_count",
        "infeasible_candidate_count",
        "post_event_over_bound_count",
        "mean_active_dynamic_generator_count",
        "mean_zero_dynamic_generator_count",
    } <= set(result.summary.columns)
    assert {"fallback_count_mean", "fallback_rate_mean"} <= set(result.aggregate.columns)


def test_robot_arm_no_reference_mode_marks_exact_metrics_unavailable():
    result = run_benchmark(RtlolaBenchmarkConfig(
        scenario="robot_arm",
        trace_kind=DEFAULT_TRACE_KIND,
        length=3,
        seeds=1,
        budget=80,
        methods=["colinear"],
        reference_mode="off",
    ))

    assert set(result.summary["method"]) == {"colinear"}
    assert "trigger_positive_rate" in result.summary.columns
    assert np.isnan(result.summary["mean_approx_loss"]).all()
    assert np.isnan(result.summary["false_positive_rate"]).all()
    assert np.isnan(result.summary["false_negative_rate"]).all()
    assert np.isnan(result.timeseries["exact_state_zonotope_width_sum"]).all()
    assert np.isnan(result.timeseries["state_zonotope_approx_error_sum"]).all()
    assert "trigger_positive" in result.timeseries.columns


def test_robot_arm_cli_writes_dashboard_artifacts(tmp_path):
    rtlola_main([
        "--profile", "smoke",
        "--scenario", "robot_arm",
        "--trace-kind", DEFAULT_TRACE_KIND,
        "--method-set", "static",
        "--length", "3",
        "--seeds", "1",
        "--budget", "160",
        "--output", str(tmp_path),
        "--no-progress",
    ])

    scenario_dir = tmp_path / "robot_arm"
    assert (scenario_dir / "timeseries.csv").stat().st_size > 0
    assert (scenario_dir / "summary.csv").stat().st_size > 0
    assert (scenario_dir / "aggregate.csv").stat().st_size > 0
    assert (scenario_dir / "trigger_confusion.csv").stat().st_size > 0
    assert (scenario_dir / "pareto_runtime_vs_loss.csv").stat().st_size > 0
    assert (tmp_path / "figures" / "robot_arm_pareto_runtime_vs_loss.pdf").stat().st_size > 0
    assert (tmp_path / "figures" / "robot_arm_tpl_range.pdf").stat().st_size > 0


def test_robot_arm_cli_methods_override_writes_only_requested_methods(tmp_path):
    rtlola_main([
        "--profile", "smoke",
        "--scenario", "robot_arm",
        "--trace-kind", DEFAULT_TRACE_KIND,
        "--method-set", "all",
        "--methods", "colinear,mpc_beam",
        "--reference-mode", "off",
        "--length", "2",
        "--seeds", "1",
        "--budget", "80",
        "--output", str(tmp_path),
        "--no-progress",
    ])

    with open(tmp_path / "robot_arm" / "summary.csv", newline="") as f:
        methods = {row["method"] for row in csv.DictReader(f)}
    assert methods == {"colinear", "mpc_beam"}


def test_robot_arm_mujoco_model_tcp_matches_trace_fixture():
    pytest.importorskip("mujoco")
    event = generate_robot_arm_events(1, trace_kind=DEFAULT_TRACE_KIND)[0]
    angles = np.asarray(event.values[1:], dtype=np.float64)
    tcp = mujoco_tcp_position(angles)
    np.testing.assert_allclose(tcp, forward_kinematics_5dof(angles), atol=1e-9)

    # The trace CSV stores TCP positions generated from the same site.
    from pzr.rtlola.robot_arm import load_robot_arm_trace

    expected = np.asarray(load_robot_arm_trace(DEFAULT_TRACE_KIND)[0].tcp)
    np.testing.assert_allclose(tcp, expected, atol=3e-4)
