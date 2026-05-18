import numpy as np

from pzr.benchmarks.robot import (
    OmnidirectionalRobotMonitor,
    RobotMeasurement,
    SimpleRobotMonitor,
    VelocityRobotMeasurement,
    generate_robot_trace,
    generate_simple_robot_trace,
)
from pzr.control.costs import CostWeights, WeightedZonotopeCost
from pzr.control.policies import (
    MPCPolicy,
    RolloutMPCPolicy,
    SequenceMPCPolicy,
    StaticReductionPolicy,
)
from pzr.core.zonotope import GeneratorKind
from pzr.experiments.benchmark import wide_rollout_reducer_factories
from pzr.experiments.run_robot import run_robot_experiment
from pzr.reduction.reducers import (
    BoxReducer,
    IdentityReducer,
    ProtectedReducer,
    ScoredKeepReducer,
)
from pzr.reduction.paper_reducers import (
    AdaptiveReducer,
    CombastelReducer,
    GirardReducer,
    MethAReducer,
    PcaReducer,
    ScottReducer,
)


class FailingReducer:
    name = "failing"

    def reduce(self, zonotope, budget, context=None):
        _ = zonotope, budget, context
        raise ValueError("intentional failure")


def test_robot_monitor_grows_one_measurement_generator_per_step() -> None:
    monitor = OmnidirectionalRobotMonitor()
    state = monitor.initial_state()
    trace = generate_robot_trace(6, seed=3)

    for measurement in trace:
        state = monitor.step(state, measurement).state

    assert state.zonotope.generator_count == len(trace) + 1


def test_robot_monitors_use_paper_overlap_triggers() -> None:
    omni = OmnidirectionalRobotMonitor()
    simple = SimpleRobotMonitor()

    assert {trigger.overlap for trigger in omni.triggers} == {0.01}
    assert {trigger.overlap for trigger in simple.triggers} == {0.01}


def test_simple_robot_monitor_grows_one_measurement_generator_per_step() -> None:
    monitor = SimpleRobotMonitor()
    state = monitor.initial_state()
    trace = generate_simple_robot_trace(6, seed=3)

    for measurement in trace:
        state = monitor.step(state, measurement).state

    assert state.zonotope.generator_count == 2 + 2 * len(trace)


def test_simple_robot_endstop_resets_position_coordinate() -> None:
    monitor = SimpleRobotMonitor()
    state = monitor.initial_state()
    trace = (
        VelocityRobotMeasurement(
            time=0.0,
            bump_x=False,
            vel_x=1.0,
            bump_y=False,
            vel_y=0.0,
        ),
        VelocityRobotMeasurement(
            time=1.0,
            bump_x=True,
            vel_x=1.0,
            bump_y=False,
            vel_y=0.0,
        ),
    )

    for measurement in trace:
        state = monitor.step(state, measurement).state

    np.testing.assert_allclose(state.zonotope.center[2], 0.0)
    np.testing.assert_allclose(state.zonotope.generators[2, :], 0.0)


def test_simple_robot_protected_reducer_preserves_calibration_metadata() -> None:
    monitor = SimpleRobotMonitor()
    state = monitor.initial_state()
    for measurement in generate_simple_robot_trace(8, seed=11):
        state = monitor.step(state, measurement).state

    policy = StaticReductionPolicy(ProtectedReducer(BoxReducer()), budget=8)
    decision = policy.reduce_state(monitor, state)

    assert decision.result.certificate.is_sound
    assert decision.state.zonotope.generator_count <= 8
    assert any(
        meta.kind == GeneratorKind.CALIBRATION and meta.source == "delta_x"
        for meta in decision.state.zonotope.metadata
    )
    assert any(
        meta.kind == GeneratorKind.CALIBRATION and meta.source == "delta_y"
        for meta in decision.state.zonotope.metadata
    )


def test_static_policy_keeps_robot_state_within_budget() -> None:
    monitor = OmnidirectionalRobotMonitor()
    state = monitor.initial_state()
    policy = StaticReductionPolicy(ScoredKeepReducer.calibration_aware(), budget=6)

    for measurement in generate_robot_trace(15, seed=4):
        state = monitor.step(state, measurement).state
        if state.zonotope.generator_count > policy.budget:
            state = policy.reduce_state(monitor, state).state
        assert state.zonotope.generator_count <= policy.budget


def test_mpc_policy_returns_certified_budgeted_robot_state() -> None:
    monitor = OmnidirectionalRobotMonitor()
    trace = generate_robot_trace(12, seed=5)
    state = monitor.initial_state()
    for measurement in trace[:8]:
        state = monitor.step(state, measurement).state

    policy = MPCPolicy(
        reducers=(
            ProtectedReducer(ScoredKeepReducer.by_norm()),
            ProtectedReducer(ScoredKeepReducer.calibration_aware()),
            ProtectedReducer(BoxReducer()),
        ),
        budget=6,
        horizon=3,
        cost=WeightedZonotopeCost(
            CostWeights(trigger_width=1.0, straddling=20.0),
            triggers=monitor.triggers,
        ),
    )

    decision = policy.reduce_state(monitor, state, trace[8:11])

    assert decision.result.certificate.is_sound
    assert decision.state.zonotope.generator_count <= 6
    assert any(
        meta.kind == GeneratorKind.CALIBRATION and meta.source == "delta"
        for meta in decision.state.zonotope.metadata
    )


def test_mpc_policy_can_choose_no_op_only_while_budgeted() -> None:
    monitor = OmnidirectionalRobotMonitor()
    trace = generate_robot_trace(8, seed=12)
    state = monitor.initial_state()
    for measurement in trace[:2]:
        state = monitor.step(state, measurement).state

    budgeted_policy = MPCPolicy(
        reducers=(IdentityReducer(), ProtectedReducer(BoxReducer())),
        budget=state.zonotope.generator_count,
        horizon=0,
        cost=WeightedZonotopeCost(
            CostWeights(trigger_width=1.0, straddling=20.0, generator_count=0.0),
            triggers=monitor.triggers,
        ),
    )

    no_op = budgeted_policy.reduce_state(monitor, state, ())

    assert no_op.reducer_name == "no_reduction"
    assert no_op.is_no_op
    assert no_op.state.zonotope.generator_count == state.zonotope.generator_count

    for measurement in trace[2:7]:
        state = monitor.step(state, measurement).state
    over_budget_policy = MPCPolicy(
        reducers=(IdentityReducer(), ProtectedReducer(BoxReducer())),
        budget=6,
        horizon=0,
        cost=WeightedZonotopeCost(
            CostWeights(trigger_width=1.0, straddling=20.0, generator_count=0.0),
            triggers=monitor.triggers,
        ),
    )

    reduced = over_budget_policy.reduce_state(monitor, state, ())

    assert not reduced.is_no_op
    assert reduced.reducer_name == "box"
    assert reduced.state.zonotope.generator_count <= 6


def test_sequence_mpc_policy_returns_certified_metadata_safe_robot_state() -> None:
    monitor = OmnidirectionalRobotMonitor()
    trace = generate_robot_trace(12, seed=6)
    state = monitor.initial_state()
    for measurement in trace[:8]:
        state = monitor.step(state, measurement).state

    policy = SequenceMPCPolicy(
        reducers=(
            ProtectedReducer(ScoredKeepReducer.by_norm()),
            ProtectedReducer(ScoredKeepReducer.calibration_aware()),
            ProtectedReducer(BoxReducer()),
        ),
        budget=6,
        horizon=3,
        cost=WeightedZonotopeCost(
            CostWeights(trigger_width=1.0, straddling=20.0),
            triggers=monitor.triggers,
        ),
    )

    decision = policy.reduce_state(monitor, state, trace[8:11])

    assert decision.result.certificate.is_sound
    assert decision.state.zonotope.generator_count <= 6
    assert decision.evaluated_sequences > 1
    assert decision.predicted_sequence
    assert any(
        meta.kind == GeneratorKind.CALIBRATION and meta.source == "delta"
        for meta in decision.state.zonotope.metadata
    )


def test_rollout_mpc_policy_returns_certified_metadata_safe_robot_state() -> None:
    monitor = OmnidirectionalRobotMonitor()
    trace = generate_robot_trace(12, seed=8)
    state = monitor.initial_state()
    for measurement in trace[:8]:
        state = monitor.step(state, measurement).state

    policy = RolloutMPCPolicy(
        reducers=(
            ProtectedReducer(GirardReducer()),
            ProtectedReducer(ScoredKeepReducer.by_norm()),
        ),
        base_reducer=ProtectedReducer(GirardReducer()),
        fallback_reducer=ProtectedReducer(BoxReducer()),
        budget=6,
        horizon=3,
        cost=WeightedZonotopeCost(
            CostWeights(
                trigger_width=1.0,
                straddling=20.0,
                generator_count=0.0,
            ),
            triggers=monitor.triggers,
        ),
    )

    decision = policy.reduce_state(monitor, state, trace[8:11])

    assert decision.result.certificate.is_sound
    assert decision.state.zonotope.generator_count <= 6
    assert decision.reducer_name in {"girard", "keep_norm"}
    assert decision.evaluated_sequences == 2
    assert decision.predicted_sequence
    assert any(
        meta.kind == GeneratorKind.CALIBRATION and meta.source == "delta"
        for meta in decision.state.zonotope.metadata
    )


def test_wide_rollout_mpc_policy_returns_certified_metadata_safe_robot_state() -> None:
    monitor = OmnidirectionalRobotMonitor()
    trace = generate_robot_trace(12, seed=8)
    state = monitor.initial_state()
    for measurement in trace[:8]:
        state = monitor.step(state, measurement).state

    policy = RolloutMPCPolicy(
        reducers=(
            ProtectedReducer(GirardReducer()),
            ProtectedReducer(CombastelReducer()),
            ProtectedReducer(MethAReducer()),
            ProtectedReducer(ScottReducer()),
            ProtectedReducer(PcaReducer()),
            ProtectedReducer(AdaptiveReducer()),
            ProtectedReducer(ScoredKeepReducer.by_norm()),
            ProtectedReducer(ScoredKeepReducer.calibration_aware()),
        ),
        base_reducer=ProtectedReducer(GirardReducer()),
        fallback_reducer=ProtectedReducer(BoxReducer()),
        budget=6,
        horizon=3,
        cost=WeightedZonotopeCost(
            CostWeights(
                trigger_width=1.0,
                straddling=20.0,
                generator_count=0.0,
            ),
            triggers=monitor.triggers,
        ),
    )

    decision = policy.reduce_state(monitor, state, trace[8:11])

    assert decision.result.certificate.is_sound
    assert decision.state.zonotope.generator_count <= 6
    assert decision.evaluated_sequences > 3
    assert decision.reducer_name != "box"
    assert decision.predicted_sequence
    assert any(
        meta.kind == GeneratorKind.CALIBRATION and meta.source == "delta"
        for meta in decision.state.zonotope.metadata
    )


def test_rollout_mpc_policy_uses_box_fallback_only_when_active_candidates_fail() -> None:
    monitor = OmnidirectionalRobotMonitor()
    trace = generate_robot_trace(12, seed=9)
    state = monitor.initial_state()
    for measurement in trace[:8]:
        state = monitor.step(state, measurement).state

    policy = RolloutMPCPolicy(
        reducers=(FailingReducer(),),
        base_reducer=ProtectedReducer(GirardReducer()),
        fallback_reducer=ProtectedReducer(BoxReducer()),
        budget=6,
        horizon=2,
        cost=WeightedZonotopeCost(
            CostWeights(trigger_width=1.0, straddling=20.0),
            triggers=monitor.triggers,
        ),
    )

    decision = policy.reduce_state(monitor, state, trace[8:10])

    assert decision.reducer_name == "box"
    assert decision.state.zonotope.generator_count <= 6


def test_wide_rollout_default_candidates_exclude_box_first_action() -> None:
    names = [factory().name for factory in wide_rollout_reducer_factories()]

    assert "box" not in names
    assert "no_reduction" not in names
    assert {"girard", "combastel", "keep_norm"} <= set(names)


def test_rollout_mpc_policy_records_future_box_fallback() -> None:
    monitor = OmnidirectionalRobotMonitor()
    trace = generate_robot_trace(12, seed=10)
    state = monitor.initial_state()
    for measurement in trace[:8]:
        state = monitor.step(state, measurement).state

    policy = RolloutMPCPolicy(
        reducers=(ProtectedReducer(GirardReducer()),),
        base_reducer=FailingReducer(),
        fallback_reducer=ProtectedReducer(BoxReducer()),
        budget=6,
        horizon=2,
        cost=WeightedZonotopeCost(
            CostWeights(trigger_width=1.0, straddling=20.0),
            triggers=monitor.triggers,
        ),
    )

    decision = policy.reduce_state(monitor, state, trace[8:10])

    assert decision.reducer_name == "girard"
    assert "box" in decision.predicted_sequence[1:]
    assert decision.state.zonotope.generator_count <= 6


def test_robot_experiment_smoke() -> None:
    results = run_robot_experiment(length=12, budget=6, horizon=3, seed=7)

    assert set(results) == {"static_calibration_aware", "mpc"}
    assert results["mpc"].steps == 12
    assert results["mpc"].max_generators <= 6
