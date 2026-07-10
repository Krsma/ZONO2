from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

rlola = pytest.importorskip("rlola_python_binding")

import pzr.rtlola.benchmark as benchmark_module
from pzr.rtlola.actions import default_action_catalog
from pzr.rtlola.benchmark import (
    RtlolaBenchmarkConfig,
    root_evaluations_to_dataframe,
    run_benchmark,
    save_benchmark_results,
)
from pzr.rtlola.binding import (
    BINDING_BUILD_PROFILE,
    BINDING_REVISION,
    INTERPRETER_REVISION,
)
from pzr.rtlola.engine import (
    RtlolaApproximationReference,
    RtlolaEngine,
    RtlolaEvent,
)
from pzr.rtlola.omni import (
    OMNI_EXPECTED_VERDICT_KEYS,
    OMNI_PUBLIC_STREAM_KEYS,
    OMNI_SPEC,
    generate_omni_events,
)
from pzr.rtlola.robot_arm import (
    ARM_PUBLIC_STREAM_KEYS,
    ARM_SPEC,
    ARM_TRIGGER_KEYS,
    DEFAULT_TRACE_KIND,
    generate_robot_arm_events,
)
from pzr.rtlola.search import RtlolaNoFeasibleAction


def _bounds(matrix):
    center = matrix[:, 0]
    radius = np.abs(matrix[:, 1:]).sum(axis=1)
    return center - radius, center + radius


def test_latest_binding_actions_and_mpc_candidates():
    catalog = default_action_catalog()

    from pzr.rtlola.binding import require_binding

    require_binding()
    assert {"clustering", "combastel"} <= set(catalog.by_name)
    assert catalog.mpc_candidate_names == (
        "girard",
        "scott",
        "interval_hull",
        "pca",
        "combastel",
        "clustering",
    )
    assert "none" not in catalog.mpc_candidate_names
    assert "interval" not in catalog.mpc_candidate_names


def test_binding_accepts_none_for_asynchronous_input():
    monitor = rlola.RLolaMonitor("""
        input value: Float64
        #[public]
        output held @value := value.hold(or: 0.0)
    """)

    monitor.accept_event([1.0], 0.0)
    verdict = monitor.accept_event([None], 1.0)

    assert "runtime_ns" in verdict
    assert "held" not in verdict


def test_binding_returns_symbolic_verdicts_as_strings():
    monitor = rlola.RLolaMonitor("""
        input value: Float64
        output epsilon: Variable @true
        #[public]
        output uncertain := value + 0.5 * epsilon
    """)

    uncertain = monitor.accept_event([1.0], 0.0)["uncertain"]

    assert isinstance(uncertain, str)
    assert "s" in uncertain


def test_repeated_branching_from_same_snapshot_is_deterministic():
    engine = RtlolaEngine(
        OMNI_SPEC,
        event_arity=3,
        expected_verdict_keys=OMNI_EXPECTED_VERDICT_KEYS,
    )
    event = generate_omni_events(1, seed=1)[0]
    state = engine.snapshot(step=0, time=event.time)
    action = default_action_catalog().by_name["girard"]

    left = engine.branch_step(state, event, action, budget=10)
    right = engine.branch_step(state, event, action, budget=10)

    np.testing.assert_allclose(
        engine.matrices(left.state)[0],
        engine.matrices(right.state)[0],
    )


@pytest.mark.parametrize(
    "action_name",
    [
        "girard",
        "scott",
        "interval_hull",
        "pca",
        "althoff_a",
        "clustering",
        "combastel",
        "colinear_scale",
    ],
)
def test_bounded_binding_transforms_outer_bound_exact_interval(action_name):
    catalog = default_action_catalog()
    events = generate_omni_events(14, seed=3)
    engine = RtlolaEngine(
        OMNI_SPEC,
        event_arity=3,
        expected_verdict_keys=OMNI_EXPECTED_VERDICT_KEYS,
    )
    for step, event in enumerate(events[:12]):
        engine.live_step(event, catalog.no_op, budget=20, step=step + 1)
    state = engine.snapshot(step=12, time=events[11].time)

    exact = engine.branch_step(state, events[12], catalog.no_op, budget=10)
    reduced = engine.branch_step(
        state,
        events[12],
        catalog.by_name[action_name],
        budget=10,
    )
    exact_lo, exact_hi = _bounds(engine.matrices(exact.state)[0])
    reduced_lo, reduced_hi = _bounds(engine.matrices(reduced.state)[0])

    assert np.all(reduced_lo <= exact_lo + 1e-10)
    assert np.all(reduced_hi >= exact_hi - 1e-10)


def test_robot_arm_preserves_five_constant_calibration_generators():
    catalog = default_action_catalog()
    events = generate_robot_arm_events(7, trace_kind=DEFAULT_TRACE_KIND)
    engine = RtlolaEngine(
        ARM_SPEC,
        event_arity=13,
        expected_verdict_keys=ARM_PUBLIC_STREAM_KEYS,
    )
    for step, event in enumerate(events[:5]):
        engine.live_step(event, catalog.no_op, budget=240, step=step + 1)
    state = engine.snapshot(step=5, time=events[4].time)
    exact = engine.branch_step(state, events[5], catalog.no_op, budget=160)
    reduced = engine.branch_step(
        state,
        events[5],
        catalog.by_name["girard"],
        budget=160,
    )

    exact_dynamic, exact_total = engine.matrices(exact.state)
    reduced_dynamic, reduced_total = engine.matrices(reduced.state)
    assert exact_total.shape[1] - exact_dynamic.shape[1] == 5
    assert reduced_total.shape[1] - reduced_dynamic.shape[1] == 5
    np.testing.assert_allclose(
        exact_total[:, -5:],
        reduced_total[:, -5:],
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "action_name",
    [
        "girard",
        "scott",
        "interval_hull",
        "pca",
        "althoff_a",
        "clustering",
        "combastel",
        "colinear_scale",
    ],
)
def test_omni_preserves_constant_calibration_generator(action_name):
    catalog = default_action_catalog()
    events = generate_omni_events(14, seed=5)
    engine = RtlolaEngine(
        OMNI_SPEC,
        event_arity=3,
        expected_verdict_keys=(
            *OMNI_EXPECTED_VERDICT_KEYS,
            *OMNI_PUBLIC_STREAM_KEYS,
        ),
    )
    for step, event in enumerate(events[:12]):
        engine.live_step(event, catalog.no_op, budget=20, step=step + 1)
    state = engine.snapshot(step=12, time=events[11].time)
    exact = engine.branch_step(state, events[12], catalog.no_op, budget=10)
    reduced = engine.branch_step(
        state,
        events[12],
        catalog.by_name[action_name],
        budget=10,
    )

    exact_dynamic, exact_total = engine.matrices(exact.state)
    reduced_dynamic, reduced_total = engine.matrices(reduced.state)
    assert exact_total.shape[1] - exact_dynamic.shape[1] == 1
    assert reduced_total.shape[1] - reduced_dynamic.shape[1] == 1
    np.testing.assert_allclose(exact_total[:, -1], reduced_total[:, -1], atol=1e-12)


def test_transform_bound_is_not_a_post_event_dense_cap():
    catalog = default_action_catalog()
    events = generate_robot_arm_events(6, trace_kind=DEFAULT_TRACE_KIND)
    engine = RtlolaEngine(
        ARM_SPEC,
        event_arity=13,
        expected_verdict_keys=ARM_PUBLIC_STREAM_KEYS,
    )
    for step, event in enumerate(events[:4]):
        engine.live_step(event, catalog.no_op, budget=200, step=step + 1)
    state = engine.snapshot(step=4, time=events[3].time)
    assert engine.metrics(state).dynamic_generator_count <= 160

    committed = engine.live_step(events[4], catalog.no_op, budget=160, step=5)

    assert committed.metrics.dynamic_generator_count > 160


def test_omni_transform_bound_is_not_a_post_event_dense_cap():
    catalog = default_action_catalog()
    events = generate_omni_events(8, seed=2)
    engine = RtlolaEngine(
        OMNI_SPEC,
        event_arity=3,
        expected_verdict_keys=(
            *OMNI_EXPECTED_VERDICT_KEYS,
            *OMNI_PUBLIC_STREAM_KEYS,
        ),
    )
    for step, event in enumerate(events[:5]):
        engine.live_step(event, catalog.no_op, budget=8, step=step + 1)
    state = engine.snapshot(step=5, time=events[4].time)
    assert engine.metrics(state).dynamic_generator_count <= 5

    committed = engine.live_step(events[5], catalog.no_op, budget=5, step=6)

    assert committed.metrics.dynamic_generator_count > 5


def test_benchmark_writes_rtlola_native_artifacts(tmp_path):
    result = run_benchmark(RtlolaBenchmarkConfig(
        scenario="robot_arm",
        trace_kind=DEFAULT_TRACE_KIND,
        length=3,
        seeds=1,
        budget=160,
        methods=["none", "girard"],
    ))
    save_benchmark_results(result, tmp_path)

    scenario_dir = tmp_path / "robot_arm"
    assert (scenario_dir / "timeseries.csv").stat().st_size > 0
    assert (scenario_dir / "summary.csv").stat().st_size > 0
    assert (scenario_dir / "aggregate.csv").stat().st_size > 0
    assert (scenario_dir / "run_failures.csv").stat().st_size > 0
    assert (scenario_dir / "mpc_root_evaluations.csv").stat().st_size > 0
    assert "post_event_over_bound" in result.timeseries
    assert "active_dynamic_generator_count" in result.timeseries
    assert "zero_dynamic_generator_count" in result.timeseries
    assert "logical_dynamic_dimension" in result.timeseries
    assert "mean_logical_dynamic_dimension" in result.summary
    assert "budget_violation" not in result.timeseries
    assert result.config.mpc_objective == "terminal_binding_approx_loss"
    config_text = (tmp_path / "config.yaml").read_text()
    assert "mpc_objective: terminal_binding_approx_loss" in config_text
    assert "source_revision: e6ecd0b2f60263e0a4270bd76a71cd9c90e685e5" in config_text
    assert "interpreter_revision: b4cfbf4680e6641f131a64d6d9e9ef57ec228976" in config_text
    assert "binding_build_profile: release" in config_text


def test_robot_arm_mpc_uses_binding_terminal_loss():
    result = run_benchmark(RtlolaBenchmarkConfig(
        scenario="robot_arm",
        trace_kind=DEFAULT_TRACE_KIND,
        length=8,
        seeds=1,
        budget=40,
        horizon=2,
        beam_width=2,
        methods=["mpc_terminal_beam"],
        reference_mode="exact",
    ))

    assert not result.timeseries.empty
    assert np.isfinite(result.timeseries["predicted_cost"]).all()
    assert np.isfinite(result.timeseries["approx_loss"]).all()
    assert result.config.mpc_objective == "terminal_binding_approx_loss"
    assert not any(column.endswith(("_lower", "_upper")) for column in result.timeseries)


def test_robot_arm_tail_mpc_variants_emit_root_diagnostics():
    methods = [
        "mpc_terminal_girard_tail",
        "mpc_cumulative_girard_tail",
        "mpc_one_step_girard_rollout",
    ]
    result = run_benchmark(RtlolaBenchmarkConfig(
        scenario="robot_arm",
        trace_kind=DEFAULT_TRACE_KIND,
        length=8,
        seeds=1,
        budget=40,
        horizon=1,
        beam_width=2,
        mpc_tail_horizon=2,
        mpc_root_beam_width=1,
        methods=methods,
        reference_mode="exact",
    ))

    assert not result.failures
    assert set(result.timeseries["method"]) == set(methods)
    reduced = result.timeseries[result.timeseries["reducer_used"] != "none"]
    assert set(reduced["mpc_variant"]) == set(methods)
    assert (reduced["configured_tail_horizon"] == 2).all()
    assert np.isfinite(reduced["explicit_terminal_loss"]).all()
    roots = root_evaluations_to_dataframe(result.raw_results)
    assert not roots.empty
    assert set(roots["method"]) == set(methods)
    assert {"girard", "scott", "pca", "combastel", "clustering"} <= set(
        roots["root_action"]
    )


def test_robot_arm_sparse_trigger_outputs_are_normalized():
    events = generate_robot_arm_events(80, trace_kind="random_drift")
    monitor = rlola.RLolaMonitor(ARM_SPEC)
    verdicts = [
        monitor.accept_event(
            list(event.values),
            event.time,
            rlola.ZonotopeConfig.none(),
        )
        for event in events
    ]

    assert not any(
        key in verdicts[index]
        for key in ARM_TRIGGER_KEYS
        for index in range(58)
    )
    assert verdicts[58]["Trigger#3"] == "Cannot stop before +Y boundary"

    result = run_benchmark(RtlolaBenchmarkConfig(
        scenario="robot_arm",
        trace_kind="random_drift",
        length=80,
        seeds=1,
        budget=80,
        methods=["none"],
        reference_mode="verdict",
    ))
    assert result.timeseries["Trigger#3"].sum() == 21
    assert result.timeseries["exact_Trigger#3"].sum() == 21
    assert result.timeseries.loc[58, "Trigger#3"]


@pytest.mark.parametrize(
    ("trace_kind", "expected_key"),
    [
        ("safe", None),
        ("x_violated", "position_x_above_geofence"),
        ("y_violated", "position_y_above_geofence"),
    ],
)
def test_balanced_omni_trace_exact_trigger_calibration(trace_kind, expected_key):
    for seed in range(10):
        monitor = rlola.RLolaMonitor(OMNI_SPEC)
        counts = {key: 0 for key in OMNI_EXPECTED_VERDICT_KEYS}
        first_positive = None
        for step, event in enumerate(
            generate_omni_events(250, seed=seed, trace_kind=trace_kind)
        ):
            verdict = monitor.accept_event(
                list(event.values),
                event.time,
                rlola.ZonotopeConfig.none(),
            )
            positive = False
            for key in counts:
                value = bool(verdict[key])
                counts[key] += int(value)
                positive = positive or value
            if positive and first_positive is None:
                first_positive = step

        if expected_key is None:
            assert counts == {key: 0 for key in OMNI_EXPECTED_VERDICT_KEYS}
            assert first_positive is None
        else:
            other = next(key for key in counts if key != expected_key)
            assert 90 <= counts[expected_key] <= 95
            assert counts[other] == 0
            assert first_positive is not None
            assert 155 <= first_positive <= 160


def test_verdict_reference_is_cached_and_raw_symbolic_values_are_not_saved(tmp_path):
    cache = tmp_path / "reference.json"
    config = RtlolaBenchmarkConfig(
        scenario="robot_arm",
        trace_kind="random_drift",
        length=4,
        seeds=1,
        budget=80,
        methods=["girard"],
        reference_mode="verdict",
        reference_cache=str(cache),
    )

    first = run_benchmark(config)
    second = run_benchmark(config)

    assert cache.stat().st_size > 0
    cached = json.loads(cache.read_text())
    assert cached["metadata"]["trace_sha256"]
    assert cached["metadata"]["binding_revision"] == BINDING_REVISION
    assert cached["metadata"]["interpreter_revision"] == INTERPRETER_REVISION
    assert cached["metadata"]["binding_build_profile"] == BINDING_BUILD_PROFILE
    assert cached["metadata"]["capabilities"] == ["trigger_verdicts"]
    assert all(
        isinstance(value, bool)
        for row in cached["steps"]
        for value in row["verdicts"].values()
    )
    assert first.summary.loc[0, "reference_negative_count"] >= 0
    assert first.summary.loc[0, "reference_positive_count"] >= 0
    negative_count = int(first.summary.loc[0, "reference_negative_count"])
    if negative_count:
        assert first.summary.loc[0, "fpr"] == pytest.approx(
            first.summary.loc[0, "false_positive_count"] / negative_count,
        )
    assert first.summary[
        [
            "mean_approx_loss",
            "final_approx_loss",
            "max_approx_loss",
            "sum_approx_loss",
        ]
    ].isna().all().all()
    assert "exact_trigger_positive" in first.timeseries
    assert "exact_Trigger#4" in first.timeseries
    assert "dist_to_expected" not in first.timeseries
    assert "dist_to_expected_lower" not in first.timeseries
    assert second.timeseries["exact_trigger_positive"].equals(
        first.timeseries["exact_trigger_positive"],
    )

    with pytest.raises(ValueError, match="metadata mismatch"):
        run_benchmark(replace(config, length=3))

    with pytest.raises(ValueError, match="lacks approximation data"):
        run_benchmark(replace(config, reference_mode="exact"))


def test_cached_exact_loss_matches_direct_binding_state_loss(tmp_path):
    events = generate_robot_arm_events(8, trace_kind=DEFAULT_TRACE_KIND)
    catalog = default_action_catalog().by_name
    engine = RtlolaEngine(
        ARM_SPEC,
        event_arity=13,
        expected_verdict_keys=ARM_PUBLIC_STREAM_KEYS,
    )
    exact_state = engine.snapshot(step=0, time=events[0].time)

    for index, event in enumerate(events, start=1):
        exact = engine.branch_step(exact_state, event, catalog["none"], budget=40)
        candidate = engine.live_step(event, catalog["girard"], budget=40, step=index)
        total = engine.matrices(exact.state)[1]
        reference = RtlolaApproximationReference(
            center=total[:, 0],
            radius=np.abs(total[:, 1:]).sum(axis=1),
            spec_id=engine.spec_id,
            step=index,
        )
        planner_before = np.asarray(engine.planner.current_zonotope(True)).copy()
        live_before = np.asarray(engine.live.current_zonotope(True)).copy()

        cached_loss = engine.approx_loss_reference(reference, candidate.state)
        direct_loss = engine.approx_loss(exact.state, candidate.state)

        assert cached_loss == pytest.approx(direct_loss)
        np.testing.assert_array_equal(
            engine.planner.current_zonotope(True),
            planner_before,
        )
        np.testing.assert_array_equal(
            engine.live.current_zonotope(True),
            live_before,
        )
        exact_state = exact.state

    cache = tmp_path / "exact.json"
    result = run_benchmark(RtlolaBenchmarkConfig(
        scenario="robot_arm",
        trace_kind=DEFAULT_TRACE_KIND,
        length=8,
        seeds=1,
        budget=40,
        methods=["girard"],
        reference_mode="exact",
        reference_cache=str(cache),
    ))
    payload = json.loads(cache.read_text())
    assert payload["metadata"]["capabilities"] == [
        "trigger_verdicts",
        "approx_loss",
    ]
    assert all({"verdicts", "center", "radius"} == set(row) for row in payload["steps"])
    assert np.isfinite(result.timeseries["approx_loss"]).all()
    assert {"state_width", "approx_loss"} <= set(result.timeseries)
    assert {
        "state_zonotope_width_sum",
        "exact_state_zonotope_width_sum",
        "state_zonotope_approx_error_sum",
    }.isdisjoint(result.timeseries)
    assert {
        "fpr",
        "fnr",
        "mean_state_width",
        "max_state_width",
        "mean_approx_loss",
        "final_approx_loss",
        "max_approx_loss",
        "sum_approx_loss",
    } <= set(result.summary)
    losses = result.timeseries["approx_loss"]
    widths = result.timeseries["state_width"]
    summary = result.summary.iloc[0]
    assert summary["mean_approx_loss"] == pytest.approx(losses.mean())
    assert summary["final_approx_loss"] == pytest.approx(losses.iloc[-1])
    assert summary["max_approx_loss"] == pytest.approx(losses.max())
    assert summary["sum_approx_loss"] == pytest.approx(losses.sum())
    assert summary["mean_state_width"] == pytest.approx(widths.mean())
    assert summary["max_state_width"] == pytest.approx(widths.max())
    assert {
        "final_approx_loss_mean",
        "sum_approx_loss_mean",
    } <= set(result.aggregate)


def test_logical_zero_rows_do_not_inflate_reducer_dimension_or_cached_loss():
    spec = """
        import math
        input a: Float
        constant delta: Variable
        output epsilon: Variable @a
        output corrected := a + 0.5 * epsilon + 2.0 * delta
        output sum := sum.offset(by: -1).defaults(to: 0.0) + corrected
        output zero := zero.offset(by: -1).defaults(to: 0.0) + a + 0.0 * epsilon + 0.0 * delta
    """
    catalog = default_action_catalog().by_name
    events = (
        RtlolaEvent(0.0, (12.0,)),
        RtlolaEvent(1.0, (-5.0,)),
    )
    engine = RtlolaEngine(spec, event_arity=1)
    exact_state = engine.snapshot(step=0, time=0.0)

    for index, event in enumerate(events, start=1):
        exact = engine.branch_step(exact_state, event, catalog["none"], budget=2)
        candidate = engine.live_step(event, catalog["girard"], budget=2, step=index)
        metrics = candidate.metrics
        assert metrics.logical_dynamic_dimension > metrics.dimension

        total = engine.matrices(exact.state)[1]
        reference = RtlolaApproximationReference(
            center=total[:, 0],
            radius=np.abs(total[:, 1:]).sum(axis=1),
            spec_id=engine.spec_id,
            step=index,
        )

        assert reference.center.size == metrics.logical_dynamic_dimension
        assert engine.approx_loss_reference(reference, candidate.state) == pytest.approx(
            engine.approx_loss(exact.state, candidate.state)
        )
        exact_state = exact.state


def test_failed_static_run_is_recorded_and_other_methods_continue(monkeypatch, tmp_path):
    original = benchmark_module.choose_static_action

    def fail_scott(engine, state, event, action, budget, **kwargs):
        if action.name == "scott":
            raise RtlolaNoFeasibleAction("synthetic reducer exhaustion")
        return original(engine, state, event, action, budget, **kwargs)

    monkeypatch.setattr(benchmark_module, "choose_static_action", fail_scott)
    result = run_benchmark(RtlolaBenchmarkConfig(
        scenario="robot_arm",
        length=3,
        seeds=1,
        budget=40,
        methods=["girard", "scott"],
        reference_mode="off",
    ))
    save_benchmark_results(result, tmp_path)

    assert set(result.timeseries["method"]) == {"girard"}
    assert set(result.summary["method"]) == {"girard"}
    assert len(result.failures) == 1
    assert result.failures[0].method == "scott"
    failures = (tmp_path / "robot_arm" / "run_failures.csv").read_text()
    assert "synthetic reducer exhaustion" in failures

    failed_only = run_benchmark(RtlolaBenchmarkConfig(
        scenario="robot_arm",
        length=3,
        seeds=1,
        budget=40,
        methods=["scott"],
        reference_mode="off",
    ))
    failed_dir = tmp_path / "failed_only"
    save_benchmark_results(failed_only, failed_dir)
    assert failed_only.timeseries.empty
    assert failed_only.summary.empty
    assert failed_only.aggregate.empty
    assert list(pd.read_csv(failed_dir / "robot_arm" / "timeseries.csv")) == [
        "seed",
        "method",
        "budget",
        "trace_kind",
    ]
