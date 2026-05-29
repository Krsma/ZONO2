"""Integration tests: end-to-end monitoring + reduction + MPC + evaluation."""

import numpy as np
import pytest

from pzr.experiments.evaluation import aggregate_summary
from pzr.experiments.runner import (
    MPCReductionPolicy,
    RunResult,
    StaticReductionPolicy,
    run_single,
    summarize_results,
)
from pzr.imitation.dataset import build_dataset
from pzr.imitation.policy import train_policy
from pzr.imitation.traces import TraceCollector
from pzr.mpc.objectives import CostWeights, WeightedZonotopeCost
from pzr.mpc.policies import RolloutMPCPolicy
from pzr.systems.omni_robot import OmniRobotMonitor, generate_omni_robot_trace
from pzr.zonotope.protected import ProtectedReducer
from pzr.zonotope.reduction import BoxReducer, CombastelReducer, GirardReducer

BUDGET = 8
LENGTH = 30


class TestEndToEnd:
    def test_static_baseline_run(self):
        monitor = OmniRobotMonitor()
        trace = generate_omni_robot_trace(LENGTH, seed=0)
        policy = StaticReductionPolicy(
            reducer=ProtectedReducer(base=GirardReducer()),
            _name="girard",
        )

        result = run_single(monitor, trace, policy, BUDGET, seed=0)
        assert result.budget_violations == 0
        assert result.unsound_certificates == 0
        assert result.total_reductions > 0
        assert len(result.steps) == LENGTH

    def test_multiple_methods_comparison(self):
        monitor = OmniRobotMonitor()
        results = []
        policies = [
            StaticReductionPolicy(ProtectedReducer(base=GirardReducer()), _name="girard"),
            StaticReductionPolicy(ProtectedReducer(base=BoxReducer()), _name="box"),
            StaticReductionPolicy(ProtectedReducer(base=CombastelReducer()), _name="combastel"),
        ]
        for policy in policies:
            for seed in range(3):
                trace = generate_omni_robot_trace(LENGTH, seed=seed)
                r = run_single(monitor, trace, policy, BUDGET, seed)
                results.append(r)

        summary = summarize_results(results)
        assert len(summary) == 9
        assert all(summary["budget_violations"] == 0)
        assert all(summary["unsound_certificates"] == 0)

        agg = aggregate_summary(summary)
        assert len(agg) == 3
        assert "girard" in agg["method"].values

    def test_mpc_rollout_via_policy(self):
        monitor = OmniRobotMonitor()
        trace = generate_omni_robot_trace(LENGTH, seed=42)
        cost = WeightedZonotopeCost(
            weights=CostWeights(trigger_width=1.0, straddling=20.0),
            triggers=monitor.triggers,
        )
        mpc = RolloutMPCPolicy(
            candidates=(
                ProtectedReducer(base=GirardReducer()),
                ProtectedReducer(base=CombastelReducer()),
            ),
            base_reducer=ProtectedReducer(base=GirardReducer()),
            budget=BUDGET,
            horizon=3,
            cost=cost,
            fallback=ProtectedReducer(base=BoxReducer()),
        )
        policy = MPCReductionPolicy(policy=mpc, _name="mpc_rollout", horizon=3)
        result = run_single(monitor, trace, policy, BUDGET, seed=42)
        assert result.budget_violations == 0
        assert result.unsound_certificates == 0
        assert result.total_reductions > 0

    def test_trace_collection_and_training(self):
        monitor = OmniRobotMonitor()
        collector = TraceCollector()

        for seed in range(5):
            trace = generate_omni_robot_trace(LENGTH, seed=seed)
            policy = StaticReductionPolicy(
                ProtectedReducer(base=GirardReducer()), _name="girard",
            )
            run_single(monitor, trace, policy, BUDGET, seed, trace_collector=collector)

        for seed in range(5, 10):
            trace = generate_omni_robot_trace(LENGTH, seed=seed)
            policy = StaticReductionPolicy(
                ProtectedReducer(base=BoxReducer()), _name="box",
            )
            run_single(monitor, trace, policy, BUDGET, seed, trace_collector=collector)

        assert len(collector) > 0
        dataset = build_dataset(collector)
        assert dataset.num_classes >= 2

        learned, result = train_policy(dataset, hidden_sizes=(32,), epochs=100)
        assert result.train_accuracy > 0.3

    def test_soundness_invariant(self):
        monitor = OmniRobotMonitor()
        policies = [
            StaticReductionPolicy(ProtectedReducer(base=GirardReducer()), _name="girard"),
            StaticReductionPolicy(ProtectedReducer(base=CombastelReducer()), _name="combastel"),
            StaticReductionPolicy(ProtectedReducer(base=BoxReducer()), _name="box"),
        ]
        for policy in policies:
            for seed in range(5):
                trace = generate_omni_robot_trace(50, seed=seed)
                r = run_single(monitor, trace, policy, BUDGET, seed)
                assert r.budget_violations == 0, f"{policy.name} seed {seed}: budget violation"
                assert r.unsound_certificates == 0, f"{policy.name} seed {seed}: unsound cert"
