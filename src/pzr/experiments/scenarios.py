"""Registered benchmark scenarios."""

from __future__ import annotations

from pzr.benchmarks.robot import (
    OmnidirectionalRobotMonitor,
    SimpleRobotMonitor,
    SIMPLE_STATE_NAMES,
    STATE_NAMES,
    generate_simple_robot_trace,
    generate_robot_trace,
    predict_robot_inputs,
)
from pzr.benchmarks.thermostat import (
    THERMOSTAT_STATE_NAMES,
    ThermostatMonitor,
    generate_thermostat_trace,
    predict_thermostat_inputs,
)
from pzr.experiments.benchmark import BenchmarkScenario


def robot_scenario() -> BenchmarkScenario:
    return BenchmarkScenario(
        name="robot",
        make_monitor=OmnidirectionalRobotMonitor,
        generate_trace=lambda length, seed: generate_robot_trace(length, seed=seed),
        predict_inputs=predict_robot_inputs,
        state_names=STATE_NAMES,
    )


def robot_simple_scenario() -> BenchmarkScenario:
    return BenchmarkScenario(
        name="robot_simple",
        make_monitor=SimpleRobotMonitor,
        generate_trace=lambda length, seed: generate_simple_robot_trace(length, seed=seed),
        predict_inputs=predict_robot_inputs,
        state_names=SIMPLE_STATE_NAMES,
    )


def thermostat_scenario() -> BenchmarkScenario:
    return BenchmarkScenario(
        name="thermostat",
        make_monitor=ThermostatMonitor,
        generate_trace=lambda length, seed: generate_thermostat_trace(length, seed=seed),
        predict_inputs=predict_thermostat_inputs,
        state_names=THERMOSTAT_STATE_NAMES,
    )


SCENARIOS = {
    "robot": robot_scenario,
    "robot_simple": robot_simple_scenario,
    "thermostat": thermostat_scenario,
}
