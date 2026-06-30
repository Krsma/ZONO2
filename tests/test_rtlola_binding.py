import numpy as np
import pytest

rlola = pytest.importorskip("rlola_python_binding")

from pzr.rtlola.actions import default_action_catalog
from pzr.rtlola.benchmark import (
    RtlolaBenchmarkConfig,
    run_benchmark,
    save_benchmark_results,
)
from pzr.rtlola.engine import RtlolaEngine, RtlolaEvent
from pzr.rtlola.omni import OMNI_EXPECTED_VERDICT_KEYS, OMNI_SPEC, generate_omni_events
from pzr.rtlola.robot_arm import (
    ARM_EXPECTED_VERDICT_KEYS,
    ARM_PUBLIC_STREAM_KEYS,
    ARM_SPEC,
    DEFAULT_TRACE_KIND,
    generate_robot_arm_events,
)


def _bounds(matrix):
    center = matrix[:, 0]
    radius = np.abs(matrix[:, 1:]).sum(axis=1)
    return center - radius, center + radius


def test_latest_binding_actions_are_exposed_but_not_mpc_candidates():
    catalog = default_action_catalog()

    assert {"clustering", "combastel"} <= set(catalog.by_name)
    assert catalog.mpc_candidate_names == (
        "girard",
        "scott",
        "interval_hull",
        "pca",
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
        event_arity=6,
        expected_verdict_keys=(*ARM_EXPECTED_VERDICT_KEYS, *ARM_PUBLIC_STREAM_KEYS),
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


def test_transform_bound_is_not_a_post_event_dense_cap():
    catalog = default_action_catalog()
    events = generate_robot_arm_events(6, trace_kind=DEFAULT_TRACE_KIND)
    engine = RtlolaEngine(
        ARM_SPEC,
        event_arity=6,
        expected_verdict_keys=(*ARM_EXPECTED_VERDICT_KEYS, *ARM_PUBLIC_STREAM_KEYS),
    )
    for step, event in enumerate(events[:4]):
        engine.live_step(event, catalog.no_op, budget=200, step=step + 1)
    state = engine.snapshot(step=4, time=events[3].time)
    assert engine.metrics(state).dynamic_generator_count <= 160

    committed = engine.live_step(events[4], catalog.no_op, budget=160, step=5)

    assert committed.metrics.dynamic_generator_count > 160


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
    assert "post_event_over_bound" in result.timeseries
    assert "active_dynamic_generator_count" in result.timeseries
    assert "zero_dynamic_generator_count" in result.timeseries
    assert "budget_violation" not in result.timeseries
