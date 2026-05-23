import numpy as np
import pytest

from pzr.core.zonotope import GeneratorKind
from pzr.robotics import (
    Gate,
    InterventionManager,
    IrosGateMonitor,
    IrosScenario,
    NoisySensorModel,
    Obstacle,
    load_safe_control_gym_iros,
)
from pzr.robotics.iros import IROS_STREAM_NAMES, IrosObservation, iros_stream_values


def _scenario() -> IrosScenario:
    return IrosScenario(
        gates=(Gate([1.0, 0.0, 1.0], width=1.0, height=1.0),),
        obstacles=(Obstacle([0.0, 0.0, 1.0], radius=0.25),),
        corridor_radius=0.8,
        min_obstacle_clearance=0.1,
        altitude_min=0.3,
        altitude_max=2.0,
        speed_max=2.0,
    )


def test_iros_gate_monitor_flags_obstacle_and_speed_streams() -> None:
    monitor = IrosGateMonitor(_scenario())
    state = monitor.initial_state()
    observation = NoisySensorModel(seed=1).observe(
        [0.0, 0.0, 1.0],
        [3.0, 0.0, 0.0],
        target_gate_index=0,
    )

    result = monitor.step(state, observation)
    statuses = {verdict.trigger.name: verdict.status for verdict in result.verdicts}

    assert statuses["collision_risk"] == "violation"
    assert statuses["obstacle_clearance_violation"] == "violation"
    assert statuses["speed_envelope_violation"] == "violation"
    assert result.state.zonotope.generator_count >= 1
    assert monitor.required_generator_metadata(result.state)


def test_iros_sensor_model_separates_bias_and_fresh_noise_generators() -> None:
    monitor = IrosGateMonitor(_scenario())
    state = monitor.initial_state()
    sensor = NoisySensorModel(bias_bound=0.01, noise_bound=0.02, seed=3)

    observation = sensor.observe(
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        target_gate_index=0,
    )
    result = monitor.step(state, observation)
    kinds = [meta.kind for meta in result.state.zonotope.metadata]

    assert observation.bias_radius is not None
    assert observation.noise_radius is not None
    assert np.all(observation.bias_radius == 0.01)
    assert np.all(observation.noise_radius == 0.02)
    assert kinds.count(GeneratorKind.CALIBRATION) == 1
    assert GeneratorKind.MEASUREMENT in kinds
    assert result.state.zonotope.metadata[0].source == "iros_sensor_bias"


def test_iros_stream_memory_filters_center_and_carries_history() -> None:
    scenario = _scenario()
    monitor = IrosGateMonitor(scenario, stream_memory_decay=0.5)
    state = monitor.initial_state()
    first = IrosObservation(
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        target_gate_index=0,
        bias_radius=np.full(6, 0.01),
        noise_radius=np.full(6, 0.02),
    )
    second = IrosObservation(
        [0.5, 0.1, 1.2],
        [0.2, 0.0, 0.0],
        target_gate_index=0,
        bias_radius=np.full(6, 0.01),
        noise_radius=np.full(6, 0.02),
    )

    first_result = monitor.step(state, first)
    second_result = monitor.step(first_result.state, second)
    expected = 0.5 * first_result.state.zonotope.center + 0.5 * iros_stream_values(scenario, second)
    kinds = [meta.kind for meta in second_result.state.zonotope.metadata]

    np.testing.assert_allclose(second_result.state.zonotope.center, expected)
    assert second_result.state.zonotope.generator_count > first_result.state.zonotope.generator_count
    assert kinds.count(GeneratorKind.CALIBRATION) == 1
    assert GeneratorKind.UNKNOWN in kinds


def test_intervention_manager_counts_spurious_and_missed_violations() -> None:
    monitor = IrosGateMonitor(_scenario())
    safe_oracle = monitor.oracle_verdicts([1.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    unsafe_oracle = monitor.oracle_verdicts([0.0, 0.0, 1.0], [3.0, 0.0, 0.0])
    spurious_monitor = monitor.oracle_verdicts([0.0, 0.0, 1.0], [3.0, 0.0, 0.0])
    no_monitor_trigger = monitor.oracle_verdicts([1.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    manager = InterventionManager([0.0, 0.0, 0.0], fallback_hold_steps=1, expected_gate_count=1)

    fallback = manager.choose_command(
        [1.0, 0.0, 0.0],
        spurious_monitor,
        safe_oracle,
        gates_passed=0,
        reducer_name="girard",
    )
    nominal = manager.choose_command(
        [1.0, 0.0, 0.0],
        no_monitor_trigger,
        unsafe_oracle,
        gates_passed=1,
        time=2.5,
        budget_violation=True,
    )

    np.testing.assert_allclose(fallback, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(nominal, [1.0, 0.0, 0.0])
    assert manager.metrics.spurious_intervention_count == 1
    assert manager.metrics.missed_violation_count == 1
    assert manager.metrics.fallback_activation_count == 1
    assert manager.metrics.budget_violation_count == 1
    assert manager.metrics.task_completed
    assert manager.metrics.time_to_target == 2.5
    assert manager.metrics.reducer_choices == {"girard": 1}


def test_level3_style_initial_state_is_not_a_corridor_violation() -> None:
    scenario = IrosScenario(
        gates=(Gate([0.44, -2.63, 1.0], width=0.45, height=1.0),),
        obstacles=(Obstacle([1.43, -2.51, 0.525], radius=0.05),),
        corridor_radius=10.0,
        min_obstacle_clearance=0.05,
        altitude_min=-0.1,
        altitude_max=2.0,
        speed_max=3.0,
    )
    monitor = IrosGateMonitor(scenario)
    pose = [-0.82, -2.98, 0.033]
    velocity = [0.0, 0.0, 0.0]
    verdicts = monitor.oracle_verdicts(pose, velocity)
    statuses = {verdict.trigger.name: verdict.status for verdict in verdicts}
    streams = dict(zip(IROS_STREAM_NAMES, iros_stream_values(scenario, IrosObservation(pose, velocity))))

    assert statuses["collision_risk"] == "safe"
    assert "corridor_violation" not in statuses
    assert streams["safety_margin"] > 0.0


def test_safe_control_gym_adapter_is_optional(monkeypatch) -> None:
    monkeypatch.delenv("PZR_SAFE_CONTROL_GYM_ROOT", raising=False)

    with pytest.raises(ImportError, match="PZR_SAFE_CONTROL_GYM_ROOT"):
        load_safe_control_gym_iros()
