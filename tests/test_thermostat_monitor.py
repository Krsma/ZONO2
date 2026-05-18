import pandas as pd

from pzr.benchmarks.thermostat import (
    ThermostatMonitor,
    generate_thermostat_trace,
)
from pzr.control.costs import CostWeights, WeightedZonotopeCost
from pzr.control.policies import RolloutMPCPolicy, StaticReductionPolicy
from pzr.core.zonotope import GeneratorKind
from pzr.experiments.cli import main
from pzr.reduction.paper_reducers import GirardReducer
from pzr.reduction.reducers import BoxReducer, IdentityReducer, ProtectedReducer


def test_thermostat_monitor_grows_one_measurement_generator_per_step() -> None:
    monitor = ThermostatMonitor()
    state = monitor.initial_state()

    for measurement in generate_thermostat_trace(6, seed=2):
        state = monitor.step(state, measurement).state

    assert state.zonotope.generator_count == 1 + 6
    assert any(
        meta.kind == GeneratorKind.CALIBRATION and meta.source == "thermal_bias"
        for meta in state.zonotope.metadata
    )


def test_thermostat_protected_reducer_preserves_calibration_metadata() -> None:
    monitor = ThermostatMonitor()
    state = monitor.initial_state()
    for measurement in generate_thermostat_trace(8, seed=3):
        state = monitor.step(state, measurement).state

    policy = StaticReductionPolicy(ProtectedReducer(BoxReducer()), budget=6)
    decision = policy.reduce_state(monitor, state)

    assert decision.result.certificate.is_sound
    assert decision.state.zonotope.generator_count <= 6
    assert any(
        meta.kind == GeneratorKind.CALIBRATION and meta.source == "thermal_bias"
        for meta in decision.state.zonotope.metadata
    )


def test_thermostat_rollout_mpc_returns_certified_budgeted_state() -> None:
    monitor = ThermostatMonitor()
    trace = generate_thermostat_trace(12, seed=4)
    state = monitor.initial_state()
    for measurement in trace[:8]:
        state = monitor.step(state, measurement).state

    policy = RolloutMPCPolicy(
        reducers=(IdentityReducer(), ProtectedReducer(GirardReducer())),
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
    assert not decision.is_no_op
    assert decision.state.zonotope.generator_count <= 6
    assert any(
        meta.kind == GeneratorKind.CALIBRATION and meta.source == "thermal_bias"
        for meta in decision.state.zonotope.metadata
    )


def test_thermostat_cli_writes_budgeted_artifacts(tmp_path) -> None:
    exit_code = main(
        [
            "thermostat",
            "--length",
            "8",
            "--budget",
            "6",
            "--horizon",
            "2",
            "--seeds",
            "1",
            "--bootstrap-samples",
            "10",
            "--out",
            str(tmp_path),
            "--quiet",
        ]
    )

    assert exit_code == 0
    raw = pd.read_csv(tmp_path / "raw_runs.csv")
    assert set(raw["scenario"]) == {"thermostat"}
    budgeted = raw[raw["method"] != "reference"]
    assert (budgeted["budget_violation_count"] == 0).all()
    assert (budgeted["unsound_certificate_count"] == 0).all()
    assert (budgeted["reduction_failure_count"] == 0).all()
    assert {"no_op_count", "chosen_no_reduction_count"} <= set(raw.columns)
    assert (budgeted["no_op_count"] == 0).all()
    assert (budgeted["chosen_no_reduction_count"] == 0).all()

    timeseries = pd.read_csv(tmp_path / "timeseries.csv")
    assert not (timeseries["reducer_name"] == "no_reduction").any()
    decisions = pd.read_csv(tmp_path / "decision_features.csv")
    assert "no_reduction" not in set(decisions["chosen_reducer_label"])
    assert (decisions["generator_count"] > decisions["budget"]).all()
    assert (tmp_path / "selection_summary.csv").exists()
    assert (tmp_path / "predicted_sequence_summary.csv").exists()

    bounds = pd.read_csv(tmp_path / "bounds_timeseries.csv")
    assert {
        "temperature",
        "filtered_temperature",
        "hvac_effort",
        "comfort_deviation",
    } <= set(bounds["state_name"])
