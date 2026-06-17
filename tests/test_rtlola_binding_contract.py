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
from pzr.rtlola.runner import RtlolaBenchmarkConfig, run_benchmark
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

    assert left.verdict == right.verdict
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

    assert result.first_step.metrics.dynamic_generator_count <= 10
    assert result.predicted_sequence


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


def test_low_budget_below_state_dimension_is_branch_infeasible_without_panic():
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

    with pytest.raises(ValueError, match="no RTLola first action fits budget=3"):
        beam_search(
            engine,
            state,
            events[4],
            (),
            (actions["girard"],),
            budget=3,
            beam_width=2,
            fallback=actions["interval"],
        )


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
        RtlolaBenchmarkConfig(budget=4, beam_width=2),
        actions["colinear"],
    )

    assert [row.name for row in rows] == ["colinear"]
