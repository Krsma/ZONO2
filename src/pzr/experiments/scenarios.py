"""Registered benchmark scenarios."""

from __future__ import annotations

from pzr.benchmarks.robot import (
    OmnidirectionalRobotMonitor,
    generate_robot_trace,
    predict_robot_inputs,
)
from pzr.experiments.benchmark import BenchmarkScenario


def robot_scenario() -> BenchmarkScenario:
    return BenchmarkScenario(
        name="robot",
        make_monitor=OmnidirectionalRobotMonitor,
        generate_trace=lambda length, seed: generate_robot_trace(length, seed=seed),
        predict_inputs=predict_robot_inputs,
    )


SCENARIOS = {
    "robot": robot_scenario,
}
