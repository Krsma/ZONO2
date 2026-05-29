"""Benchmark orchestrator: run all methods across seeds and scenarios.

Ties together runner, config, evaluation, and output generation into
a single pipeline that produces paper-ready results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd
from tqdm.auto import tqdm

from pzr.experiments.config import BenchmarkConfig, save_config
from pzr.experiments.evaluation import aggregate_summary
from pzr.experiments.runner import (
    MPCReductionPolicy,
    ReductionPolicy,
    RunResult,
    StaticReductionPolicy,
    results_to_dataframe,
    run_single,
    summarize_results,
)
from pzr.imitation.traces import TraceCollector
from pzr.monitoring.base import MonitorAdapter
from pzr.mpc.objectives import CostWeights, WeightedZonotopeCost
from pzr.mpc.policies import MPCPolicy, RolloutMPCPolicy
from pzr.systems.omni_robot import OmniRobotMonitor, generate_omni_robot_trace
from pzr.systems.simple_robot import SimpleRobotMonitor, generate_simple_robot_trace
from pzr.utils.serialization import save_json

try:
    from pzr.envs.base import NoisySensorModel
    from pzr.envs.point_mass_monitor import (
        PointMassMonitor,
        generate_point_mass_trace,
    )
    from pzr.envs.robot_arm_monitor import (
        RobotArmMonitor,
        generate_robot_arm_trace,
    )

    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False
from pzr.zonotope.protected import ProtectedReducer
from pzr.zonotope.reduction import (
    BoxReducer,
    CombastelReducer,
    GirardReducer,
    MethAReducer,
    PcaReducer,
    ScottReducer,
)


@dataclass
class MethodSpec:
    """Maps a method name to a ReductionPolicy."""

    name: str
    policy: ReductionPolicy


@dataclass
class ScenarioSpec:
    """Maps a scenario name to a monitor and trace generator."""

    name: str
    monitor: MonitorAdapter
    trace_fn: Callable[[int, int], Sequence]


@dataclass
class BenchmarkResult:
    """Results from a full benchmark run."""

    config: BenchmarkConfig
    raw_results: list[RunResult]
    timeseries: pd.DataFrame
    summary: pd.DataFrame
    aggregate: pd.DataFrame


def default_scenarios() -> list[ScenarioSpec]:
    scenarios = [
        ScenarioSpec(
            name="omni_robot",
            monitor=OmniRobotMonitor(),
            trace_fn=lambda length, seed: generate_omni_robot_trace(length, seed=seed),
        ),
        ScenarioSpec(
            name="simple_robot",
            monitor=SimpleRobotMonitor(),
            trace_fn=lambda length, seed: generate_simple_robot_trace(length, seed=seed),
        ),
    ]
    if _HAS_MUJOCO:
        import numpy as np

        scenarios.append(ScenarioSpec(
            name="point_mass",
            monitor=PointMassMonitor(
                noise_model=NoisySensorModel(
                    bias_bound=np.array([0.15, 0.15, 0.08, 0.08]),
                    noise_bound=np.array([0.08, 0.08, 0.04, 0.04]),
                ),
            ),
            trace_fn=lambda length, seed: generate_point_mass_trace(length, seed=seed),
        ))
        scenarios.append(ScenarioSpec(
            name="robot_arm",
            monitor=RobotArmMonitor(
                noise_model=NoisySensorModel(
                    bias_bound=np.array([0.02, 0.02, 0.02, 0.01, 0.01, 0.01]),
                    noise_bound=np.array([0.01, 0.01, 0.01, 0.005, 0.005, 0.005]),
                ),
            ),
            trace_fn=lambda length, seed: generate_robot_arm_trace(length, seed=seed),
        ))
    return scenarios


def default_methods(
    monitor: MonitorAdapter,
    budget: int,
    horizon: int,
    cost_weights: CostWeights = CostWeights(),
) -> list[MethodSpec]:
    """Build the standard method set: static baselines + MPC variants."""
    cost = WeightedZonotopeCost(weights=cost_weights, triggers=monitor.triggers)

    static_methods = [
        ("girard", GirardReducer()),
        ("combastel", CombastelReducer()),
        ("pca", PcaReducer()),
        ("methA", MethAReducer()),
        ("scott", ScottReducer()),
        ("box", BoxReducer()),
    ]
    methods: list[MethodSpec] = []
    for name, reducer in static_methods:
        methods.append(MethodSpec(
            name=name,
            policy=StaticReductionPolicy(
                reducer=ProtectedReducer(base=reducer),
                _name=name,
            ),
        ))

    mpc_candidates = tuple(
        ProtectedReducer(base=r) for _, r in static_methods[:5]
    )

    mpc_rollout = RolloutMPCPolicy(
        candidates=mpc_candidates,
        base_reducer=ProtectedReducer(base=GirardReducer()),
        budget=budget,
        horizon=horizon,
        cost=cost,
        fallback=ProtectedReducer(base=BoxReducer()),
    )
    methods.append(MethodSpec(
        name="mpc_rollout",
        policy=MPCReductionPolicy(policy=mpc_rollout, _name="mpc_rollout", horizon=horizon),
    ))

    mpc_sequence = MPCPolicy(
        candidates=mpc_candidates,
        budget=budget,
        horizon=horizon,
        cost=cost,
        fallback=ProtectedReducer(base=BoxReducer()),
    )
    methods.append(MethodSpec(
        name="mpc_sequence",
        policy=MPCReductionPolicy(policy=mpc_sequence, _name="mpc_sequence", horizon=horizon),
    ))

    return methods


def _filter_methods(methods: list[MethodSpec], method_set: str) -> list[MethodSpec]:
    if method_set == "all":
        return methods
    if method_set == "static":
        return [m for m in methods if not m.name.startswith("mpc")]
    if method_set == "standard":
        return methods
    return methods


def run_benchmark(
    config: BenchmarkConfig,
    scenarios: list[ScenarioSpec] | None = None,
    methods: list[MethodSpec] | None = None,
    trace_collector: TraceCollector | None = None,
    show_progress: bool = True,
) -> dict[str, BenchmarkResult]:
    """Run the full benchmark across scenarios."""
    if scenarios is None:
        all_scenarios = default_scenarios()
        if config.scenario != "all":
            all_scenarios = [s for s in all_scenarios if s.name == config.scenario]
    else:
        all_scenarios = scenarios

    results_by_scenario: dict[str, BenchmarkResult] = {}

    scenario_iter = tqdm(
        all_scenarios, desc="scenarios", disable=not show_progress,
        unit="scenario", leave=True,
    )
    for scenario in scenario_iter:
        if methods is None:
            scenario_methods = default_methods(
                scenario.monitor, config.budget, config.horizon, config.cost_weights,
            )
            scenario_methods = _filter_methods(scenario_methods, config.method_set)
        else:
            scenario_methods = methods

        raw_results: list[RunResult] = []
        total_runs = len(scenario_methods) * config.seeds
        with tqdm(
            total=total_runs, desc=scenario.name, disable=not show_progress,
            unit="run", leave=False,
        ) as pbar:
            for method in scenario_methods:
                pbar.set_description_str(f"{scenario.name} · {method.name}")
                for seed in range(config.seeds):
                    trace = scenario.trace_fn(config.length, seed)
                    result = run_single(
                        monitor=scenario.monitor,
                        trace=trace,
                        policy=method.policy,
                        budget=config.budget,
                        seed=seed,
                        trace_collector=trace_collector,
                    )
                    raw_results.append(result)
                    pbar.update(1)

        timeseries = results_to_dataframe(raw_results)
        summary = summarize_results(raw_results)
        aggregate = aggregate_summary(summary)

        results_by_scenario[scenario.name] = BenchmarkResult(
            config=config,
            raw_results=raw_results,
            timeseries=timeseries,
            summary=summary,
            aggregate=aggregate,
        )

    return results_by_scenario


def save_benchmark_results(
    results: dict[str, BenchmarkResult],
    output_dir: Path,
) -> None:
    """Save benchmark results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for scenario_name, result in results.items():
        scenario_dir = output_dir / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        result.timeseries.to_csv(scenario_dir / "timeseries.csv", index=False)
        result.summary.to_csv(scenario_dir / "summary.csv", index=False)
        result.aggregate.to_csv(scenario_dir / "aggregate.csv", index=False)

    save_config(results[next(iter(results))].config, output_dir / "config.yaml")
    save_json(
        {
            "scenarios": list(results.keys()),
            "methods": sorted(set(
                r.method for br in results.values() for r in br.raw_results
            )),
        },
        output_dir / "manifest.json",
    )
