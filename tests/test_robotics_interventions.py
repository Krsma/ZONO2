import numpy as np
import pytest

from pzr.robotics import (
    Gate,
    InterventionManager,
    IrosGateMonitor,
    IrosScenario,
    NoisySensorModel,
    Obstacle,
    load_safe_control_gym_iros,
)


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


def test_safe_control_gym_adapter_is_optional(monkeypatch) -> None:
    monkeypatch.delenv("PZR_SAFE_CONTROL_GYM_ROOT", raising=False)

    with pytest.raises(ImportError, match="PZR_SAFE_CONTROL_GYM_ROOT"):
        load_safe_control_gym_iros()
