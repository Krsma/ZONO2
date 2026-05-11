from pzr.benchmarks.robot import OmnidirectionalRobotMonitor, generate_robot_trace
from pzr.control.costs import CostWeights, WeightedZonotopeCost
from pzr.control.policies import MPCPolicy, SequenceMPCPolicy, StaticReductionPolicy
from pzr.core.zonotope import GeneratorKind
from pzr.experiments.run_robot import run_robot_experiment
from pzr.reduction.reducers import BoxReducer, ProtectedReducer, ScoredKeepReducer


def test_robot_monitor_grows_one_measurement_generator_per_step() -> None:
    monitor = OmnidirectionalRobotMonitor()
    state = monitor.initial_state()
    trace = generate_robot_trace(6, seed=3)

    for measurement in trace:
        state = monitor.step(state, measurement).state

    assert state.zonotope.generator_count == len(trace) + 1


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


def test_robot_experiment_smoke() -> None:
    results = run_robot_experiment(length=12, budget=6, horizon=3, seed=7)

    assert set(results) == {"static_calibration_aware", "mpc"}
    assert results["mpc"].steps == 12
    assert results["mpc"].max_generators <= 6
