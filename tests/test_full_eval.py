"""Smoke tests for the full evaluation pipeline (Stage 8)."""

import numpy as np
import pytest

from pzr.experiments.benchmark import (
    default_methods,
    run_benchmark,
    save_benchmark_results,
)
from pzr.experiments.config import from_profile
from pzr.experiments.figures import (
    plot_inference_time_comparison,
    plot_method_comparison_bars,
)
from pzr.experiments.regret_eval import (
    REGRET_ORACLE_MODES,
    RegretEvalResult,
    RegretOracleConfig,
    build_regret_dataset,
    evaluate_regret_candidates,
    train_and_evaluate_regret,
    train_and_evaluate_regret_on_traces,
)
from pzr.experiments.runner import MPCReductionPolicy, StaticReductionPolicy
from pzr.experiments.tables import (
    format_comparison_table,
    format_latex_table,
    format_soundness_report,
)
from pzr.mpc.objectives import CostWeights, WeightedZonotopeCost
from pzr.mpc.policies import MPCPolicy, RolloutMPCPolicy
from pzr.systems.omni_robot import OmniRobotMonitor, generate_omni_robot_trace
from pzr.zonotope.protected import ProtectedReducer
from pzr.zonotope.reduction import (
    BoxReducer,
    CombastelReducer,
    GirardReducer,
    MethAReducer,
    ScottReducer,
)


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


class TestRegretDistillation:
    def _overflow_state(self, length=20, seed=0):
        monitor = OmniRobotMonitor()
        state = monitor.initial_state()
        history = []
        for measurement in generate_omni_robot_trace(length, seed=seed):
            state = monitor.step(state, measurement).state
            history.append(measurement)
            if state.zonotope.generator_count > 8:
                return monitor, state, history
        raise AssertionError("trace did not overflow the generator budget")

    def test_oracle_candidate_cost_collection(self):
        monitor, state, history = self._overflow_state()

        rows = evaluate_regret_candidates(
            monitor, state, history, budget=8,
            config=RegretOracleConfig(mode="beam3", horizon=2, beam_width=3),
        )

        assert {row.name for row in rows} == {"girard", "methA", "scott"}
        assert all(np.isfinite(row.cost) for row in rows)
        assert rows == sorted(rows, key=lambda row: (row.cost, row.sequence))

    @pytest.mark.parametrize("mode", REGRET_ORACLE_MODES)
    def test_oracle_modes_return_finite_costs(self, mode):
        monitor, state, history = self._overflow_state(length=18, seed=1)

        rows = evaluate_regret_candidates(
            monitor, state, history, budget=8,
            config=RegretOracleConfig(mode=mode, horizon=1, beam_width=2),
        )

        expected = {"girard", "methA", "scott"}
        if mode in {"rollout_wide", "sequence_wide"}:
            expected = {"girard", "combastel", "pca", "methA", "scott"}
        assert {row.name for row in rows} == expected
        assert all(np.isfinite(row.cost) for row in rows)

    def test_full_regret_pipeline(self):
        monitor = OmniRobotMonitor()

        result = train_and_evaluate_regret(
            monitor=monitor,
            trace_fn=lambda l, s: generate_omni_robot_trace(l, seed=s),
            budget=8,
            train_seeds=range(3),
            eval_seeds=range(3, 6),
            length=20,
            oracle_config=RegretOracleConfig(mode="beam3", horizon=1, beam_width=2),
            iterations=2,
            epochs_per_iteration=50,
            hidden_sizes=(32,),
        )
        assert isinstance(result, RegretEvalResult)
        assert result.total_traces > 0
        dataset = build_regret_dataset(result.traces, result.policy.candidate_names)
        assert dataset.regrets.shape[1] == len(result.policy.candidate_names)
        assert np.all(dataset.regrets >= 0.0)
        assert len(result.eval_results) == 3
        assert all(r.budget_violations == 0 for r in result.eval_results)
        assert all(r.unsound_certificates == 0 for r in result.eval_results)

    def test_regret_policy_runs_without_soundness_violations(self):
        monitor = OmniRobotMonitor()

        result = train_and_evaluate_regret(
            monitor=monitor,
            trace_fn=lambda l, s: generate_omni_robot_trace(l, seed=s),
            budget=8,
            train_seeds=range(6),
            eval_seeds=range(6, 9),
            length=30,
            oracle_config=RegretOracleConfig(mode="beam3", horizon=1, beam_width=2),
            iterations=1,
            epochs_per_iteration=30,
            hidden_sizes=(16,),
        )
        assert len(result.eval_results) == 3
        assert all(r.budget_violations == 0 for r in result.eval_results)

    def test_regret_training_from_explicit_traces(self):
        monitor = OmniRobotMonitor()
        train_traces = tuple(
            (seed, generate_omni_robot_trace(20, seed=seed))
            for seed in range(2)
        )
        eval_traces = tuple(
            (seed, generate_omni_robot_trace(20, seed=seed))
            for seed in range(2, 3)
        )

        result = train_and_evaluate_regret_on_traces(
            monitor=monitor,
            train_traces=train_traces,
            eval_traces=eval_traces,
            budget=8,
            oracle_config=RegretOracleConfig(mode="beam3", horizon=1, beam_width=2),
            iterations=1,
            epochs_per_iteration=10,
            hidden_sizes=(16,),
            show_progress=False,
        )

        assert result.total_traces > 0
        assert len(result.eval_results) == 1
        assert all(r.unsound_certificates == 0 for r in result.eval_results)


class TestFiguresIntegration:
    def test_inference_time_bars(self, tmp_path):
        fig = plot_inference_time_comparison(
            {"girard": 0.1, "mpc_rollout": 5.0, "learned": 0.3},
            out_path=tmp_path / "inference.pdf",
        )
        assert (tmp_path / "inference.pdf").exists()


class TestGroundTruth:
    """Tests for the new ground-truth comparison (approx_error + FPR)."""

    def _make_runner_inputs(self, length=20, seed=0):
        from pzr.experiments.runner import compute_ground_truth
        monitor = OmniRobotMonitor()
        trace = generate_omni_robot_trace(length, seed=seed)
        gt = compute_ground_truth(monitor, trace)
        return monitor, trace, gt

    def test_compute_ground_truth_shape(self):
        monitor, trace, gt = self._make_runner_inputs(length=10)
        assert len(gt) == 10
        for entry in gt:
            assert entry.lower.shape == entry.upper.shape
            assert entry.width_sum >= 0
            assert set(entry.verdicts.keys()) == {t.name for t in monitor.triggers}

    def test_unreduced_policy_has_zero_approx_error(self):
        """A policy that never reduces (budget large enough) must match ground truth exactly."""
        from pzr.experiments.runner import run_single

        monitor, trace, gt = self._make_runner_inputs(length=8)
        huge_budget = 10_000  # never triggers reduction
        # Use any policy; with huge_budget it never decides to reduce
        policy = StaticReductionPolicy(
            reducer=ProtectedReducer(base=GirardReducer()), _name="girard",
        )
        result = run_single(monitor, trace, policy, huge_budget, 0, ground_truth=gt)
        for step in result.steps:
            assert not step.reduced, "huge budget should never reduce"
            assert step.approx_error_sum == 0.0
            assert step.false_positive is False

    def test_reduced_policy_has_nonzero_approx_error(self):
        """A policy that actually reduces should incur positive approximation error."""
        from pzr.experiments.runner import run_single

        monitor, trace, gt = self._make_runner_inputs(length=30)
        policy = StaticReductionPolicy(
            reducer=ProtectedReducer(base=BoxReducer()), _name="box",
        )
        result = run_single(monitor, trace, policy, budget=8, seed=0, ground_truth=gt)
        reductions = [s for s in result.steps if s.reduced]
        assert len(reductions) > 0
        post_reduction_errors = [s.approx_error_sum for s in result.steps if s.step >= reductions[0].step]
        assert max(post_reduction_errors) > 0.0

    def test_aggregate_has_fpr_columns(self):
        """aggregate_summary surfaces the new FPR/approximation columns."""
        config = from_profile("smoke", scenario="omni_robot", method_set="static")
        results = run_benchmark(config)
        agg = results["omni_robot"].aggregate
        for col in (
            "false_positive_rate_mean",
            "mean_approx_error_mean",
            "abs_error_range_mean",
        ):
            assert col in agg.columns


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
