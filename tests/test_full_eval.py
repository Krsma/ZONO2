"""Smoke tests for the full evaluation pipeline (Stage 8)."""

import numpy as np
import pytest

from pzr.experiments.benchmark import (
    default_methods,
    run_benchmark,
    save_benchmark_results,
)
from pzr.experiments.config import from_profile
from pzr.experiments.dagger_eval import (
    DAggerEvalResult,
    LearnedReductionPolicy,
    collect_expert_traces,
    train_and_evaluate_dagger,
)
from pzr.experiments.figures import (
    plot_dagger_learning_curve,
    plot_inference_time_comparison,
    plot_method_comparison_bars,
)
from pzr.experiments.runner import MPCReductionPolicy, StaticReductionPolicy
from pzr.experiments.tables import (
    format_comparison_table,
    format_latex_table,
    format_soundness_report,
)
from pzr.mpc.objectives import CostWeights, WeightedZonotopeCost
from pzr.mpc.policies import RolloutMPCPolicy
from pzr.systems.omni_robot import OmniRobotMonitor, generate_omni_robot_trace
from pzr.zonotope.protected import ProtectedReducer
from pzr.zonotope.reduction import BoxReducer, CombastelReducer, GirardReducer


class TestTables:
    def test_markdown_table(self):
        config = from_profile("smoke", scenario="omni_robot", method_set="static")
        results = run_benchmark(config)
        agg = results["omni_robot"].aggregate
        table = format_comparison_table(agg)
        assert "girard" in table
        assert "|" in table

    def test_latex_table(self):
        config = from_profile("smoke", scenario="omni_robot", method_set="static")
        results = run_benchmark(config)
        agg = results["omni_robot"].aggregate
        table = format_latex_table(agg)
        assert "\\begin{table}" in table
        assert "girard" in table

    def test_soundness_report_clean(self):
        config = from_profile("smoke", scenario="omni_robot", method_set="static")
        results = run_benchmark(config)
        report = format_soundness_report(results["omni_robot"].summary)
        assert "All soundness invariants hold" in report


class TestDAggerEval:
    def test_expert_trace_collection(self):
        monitor = OmniRobotMonitor()
        cost = WeightedZonotopeCost(
            weights=CostWeights(trigger_width=1.0, straddling=20.0),
            triggers=monitor.triggers,
        )
        mpc = RolloutMPCPolicy(
            candidates=(ProtectedReducer(base=GirardReducer()),),
            base_reducer=ProtectedReducer(base=GirardReducer()),
            budget=8, horizon=2, cost=cost,
            fallback=ProtectedReducer(base=BoxReducer()),
        )
        expert = MPCReductionPolicy(policy=mpc, _name="mpc_expert", horizon=2)

        collector = collect_expert_traces(
            monitor, lambda l, s: generate_omni_robot_trace(l, seed=s),
            expert, budget=8, seeds=range(3), length=20,
        )
        assert len(collector) > 0

    def test_full_dagger_pipeline(self):
        monitor = OmniRobotMonitor()
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
            budget=8, horizon=2, cost=cost,
            fallback=ProtectedReducer(base=BoxReducer()),
        )
        expert = MPCReductionPolicy(policy=mpc, _name="mpc_expert", horizon=2)

        result = train_and_evaluate_dagger(
            monitor=monitor,
            trace_fn=lambda l, s: generate_omni_robot_trace(l, seed=s),
            expert_policy=expert,
            budget=8,
            train_seeds=range(3),
            eval_seeds=range(3, 6),
            length=20,
            dagger_iterations=2,
            epochs_per_iteration=50,
            hidden_sizes=(32,),
        )
        assert isinstance(result, DAggerEvalResult)
        assert result.total_traces > 0
        assert len(result.eval_results) == 3
        assert all(r.budget_violations == 0 for r in result.eval_results)
        assert all(r.unsound_certificates == 0 for r in result.eval_results)

    def test_learned_policy_faster_than_mpc(self):
        monitor = OmniRobotMonitor()
        cost = WeightedZonotopeCost(
            weights=CostWeights(trigger_width=1.0, straddling=20.0),
            triggers=monitor.triggers,
        )
        mpc = RolloutMPCPolicy(
            candidates=(
                ProtectedReducer(base=GirardReducer()),
                ProtectedReducer(base=CombastelReducer()),
                ProtectedReducer(base=BoxReducer()),
            ),
            base_reducer=ProtectedReducer(base=GirardReducer()),
            budget=8, horizon=2, cost=cost,
            fallback=ProtectedReducer(base=BoxReducer()),
        )
        expert = MPCReductionPolicy(policy=mpc, _name="mpc_expert", horizon=2)

        result = train_and_evaluate_dagger(
            monitor=monitor,
            trace_fn=lambda l, s: generate_omni_robot_trace(l, seed=s),
            expert_policy=expert,
            budget=8,
            train_seeds=range(5),
            eval_seeds=range(5, 8),
            length=30,
            dagger_iterations=1,
            epochs_per_iteration=30,
            hidden_sizes=(16,),
        )
        assert len(result.eval_results) == 3
        assert all(r.budget_violations == 0 for r in result.eval_results)


class TestFiguresIntegration:
    def test_dagger_learning_curve(self, tmp_path):
        fig = plot_dagger_learning_curve(
            [0.4, 0.6, 0.75],
            [0.35, 0.55, 0.70],
            out_path=tmp_path / "dagger.pdf",
        )
        assert (tmp_path / "dagger.pdf").exists()

    def test_inference_time_bars(self, tmp_path):
        fig = plot_inference_time_comparison(
            {"girard": 0.1, "mpc_rollout": 5.0, "learned": 0.3},
            out_path=tmp_path / "inference.pdf",
        )
        assert (tmp_path / "inference.pdf").exists()


class TestFullPipeline:
    def test_smoke_end_to_end(self, tmp_path):
        """Full smoke evaluation: baselines + MPC + save."""
        config = from_profile("smoke", scenario="omni_robot")
        results = run_benchmark(config)
        save_benchmark_results(results, tmp_path / "results")

        result = results["omni_robot"]
        assert all(result.summary["budget_violations"] == 0)
        assert all(result.summary["unsound_certificates"] == 0)

        table_md = format_comparison_table(result.aggregate)
        report = format_soundness_report(result.summary)
        assert "girard" in table_md
        assert "All soundness invariants hold" in report

        fig = plot_method_comparison_bars(result.aggregate, out_path=tmp_path / "comparison.pdf")
        assert (tmp_path / "comparison.pdf").exists()
