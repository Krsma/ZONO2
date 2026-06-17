"""Smoke tests for the benchmark orchestrator."""

import pandas as pd
import pytest

from pzr.experiments.benchmark import (
    BenchmarkResult,
    deprecated_scenarios,
    default_methods,
    default_scenarios,
    registered_scenarios,
    run_benchmark,
    save_benchmark_results,
)
from pzr.experiments.config import BenchmarkConfig, from_profile, save_config, load_config
from pzr.cli import main as cli_main


class TestConfig:
    def test_smoke_profile(self):
        config = from_profile("smoke")
        assert config.length == 30
        assert config.seeds == 3

    def test_paper_profile(self):
        config = from_profile("paper")
        assert config.length == 200
        assert config.seeds == 30

    def test_save_load_roundtrip(self, tmp_path):
        config = from_profile("smoke", scenario="omni_robot", jobs=2, beam_width=7)
        path = tmp_path / "config.yaml"
        save_config(config, path)
        loaded = load_config(path)
        assert loaded.length == config.length
        assert loaded.seeds == config.seeds
        assert loaded.budget == config.budget
        assert loaded.jobs == config.jobs
        assert loaded.beam_width == config.beam_width

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError):
            from_profile("nonexistent")


class TestBenchmark:
    def test_smoke_single_scenario(self, tmp_path):
        config = from_profile("smoke", scenario="omni_robot", output_dir=str(tmp_path))
        results = run_benchmark(config)

        assert "omni_robot" in results
        result = results["omni_robot"]
        assert isinstance(result, BenchmarkResult)
        assert len(result.summary) > 0
        assert len(result.aggregate) > 0
        assert all(result.summary["budget_violations"] == 0)
        assert all(result.summary["unsound_certificates"] == 0)

    def test_smoke_default_scenarios_exclude_deprecated(self, tmp_path):
        config = from_profile("smoke", scenario="all", output_dir=str(tmp_path))
        results = run_benchmark(config)
        registered = {s.name for s in registered_scenarios()}
        assert "omni_robot" in results
        assert "simple_robot" not in results
        assert "point_mass" not in results
        if "robot_arm" in registered:
            assert "robot_arm" in results

    def test_registered_scenarios_keep_deprecated_explicit_options(self):
        registered = {s.name: s for s in registered_scenarios()}
        defaults = {s.name for s in default_scenarios()}
        deprecated = {s.name for s in deprecated_scenarios()}

        assert "simple_robot" in registered
        assert "simple_robot" in deprecated
        assert "simple_robot" not in defaults
        if "point_mass" in registered:
            assert "point_mass" in deprecated
            assert "point_mass" not in defaults
        assert "omni_robot" in defaults
        if "robot_arm" in registered:
            assert "robot_arm" in defaults

    def test_static_only(self, tmp_path):
        config = from_profile("smoke", scenario="omni_robot", method_set="static")
        results = run_benchmark(config)
        result = results["omni_robot"]
        methods = set(result.summary["method"].unique())
        assert not any(m.startswith("mpc") for m in methods)

    def test_method_sets_include_new_mpc_variants(self):
        scenario = next(s for s in default_scenarios() if s.name == "omni_robot")
        methods = default_methods(scenario.monitor, budget=8, horizon=2, beam_width=4)
        all_names = {m.name for m in methods}

        assert {
            "mpc_rollout",
            "mpc_rollout_methA",
            "mpc_rollout_scott",
            "mpc_pair_rollout3",
            "mpc_sequence",
            "mpc_sequence3",
            "mpc_beam3",
        } <= all_names

        standard_config = from_profile(
            "smoke", scenario="omni_robot", method_set="standard", seeds=1,
        )
        results = run_benchmark(standard_config, show_progress=False)
        standard_names = set(results["omni_robot"].summary["method"].unique())
        assert "mpc_sequence" in standard_names
        assert "mpc_beam3" not in standard_names
        assert "mpc_pair_rollout3" not in standard_names

        headline_config = from_profile(
            "smoke", scenario="omni_robot", method_set="headline", seeds=1,
        )
        headline = run_benchmark(headline_config, show_progress=False)
        headline_names = set(headline["omni_robot"].summary["method"].unique())
        assert {
            "mpc_rollout",
            "mpc_pair_rollout3",
            "mpc_beam3",
            "mpc_sequence3",
        } <= headline_names
        assert "mpc_rollout_scott" not in headline_names
        assert "mpc_rollout_methA" not in headline_names

        core_config = from_profile(
            "smoke", scenario="omni_robot", method_set="paper_core", seeds=1,
        )
        core = run_benchmark(core_config, show_progress=False)
        core_names = set(core["omni_robot"].summary["method"].unique())
        assert {
            "mpc_rollout",
            "mpc_pair_rollout3",
            "mpc_beam3",
        } <= core_names
        assert "mpc_sequence3" not in core_names

    def test_save_results(self, tmp_path):
        config = from_profile("smoke", scenario="omni_robot")
        results = run_benchmark(config)
        save_benchmark_results(results, tmp_path / "output")

        assert (tmp_path / "output" / "omni_robot" / "timeseries.csv").exists()
        assert (tmp_path / "output" / "omni_robot" / "summary.csv").exists()
        assert (tmp_path / "output" / "omni_robot" / "aggregate.csv").exists()
        assert (tmp_path / "output" / "config.yaml").exists()
        assert (tmp_path / "output" / "manifest.json").exists()

    def test_cli_default_runs_no_learned_policy(self, tmp_path):
        output = tmp_path / "default"
        cli_main([
            "--profile", "smoke",
            "--scenario", "omni_robot",
            "--method-set", "static",
            "--length", "12",
            "--seeds", "1",
            "--no-progress",
            "--output", str(output),
        ])
        summary = pd.read_csv(output / "omni_robot" / "summary.csv")
        timeseries = pd.read_csv(output / "omni_robot" / "timeseries.csv")
        assert not any(summary["method"].str.startswith("learned_regret"))
        assert int(timeseries["step"].max()) == 11

    def test_cli_regret_persists_learned_rows(self, tmp_path):
        output = tmp_path / "regret"
        cli_main([
            "--profile", "smoke",
            "--scenario", "omni_robot",
            "--method-set", "static",
            "--seeds", "1",
            "--learned-mode", "regret",
            "--regret-oracle", "beam3",
            "--regret-iterations", "1",
            "--regret-epochs", "5",
            "--regret-train-seeds", "1",
            "--regret-eval-seeds", "1",
            "--no-progress",
            "--output", str(output),
        ])
        summary = pd.read_csv(output / "omni_robot" / "summary.csv")
        timeseries = pd.read_csv(output / "omni_robot" / "timeseries.csv")
        aggregate = pd.read_csv(output / "omni_robot" / "aggregate.csv")
        assert "learned_regret_beam3" in set(summary["method"])
        assert "learned_regret_beam3" in set(timeseries["method"])
        assert "learned_regret_beam3" in set(aggregate["method"])
        assert (output / "learning" / "omni_robot" / "regret_candidate_costs.csv").exists()

    def test_cli_budget_sweep_regret_persists_learned_rows_per_budget(self, tmp_path):
        output = tmp_path / "regret_sweep"
        cli_main([
            "--profile", "smoke",
            "--scenario", "omni_robot",
            "--method-set", "static",
            "--seeds", "1",
            "--budget-sweep", "8,10",
            "--learned-mode", "regret",
            "--regret-oracle", "beam3",
            "--regret-iterations", "1",
            "--regret-epochs", "5",
            "--regret-train-seeds", "1",
            "--regret-eval-seeds", "1",
            "--no-progress",
            "--output", str(output),
        ])
        for budget in (8, 10):
            budget_dir = output / f"budget_{budget}"
            summary = pd.read_csv(budget_dir / "omni_robot" / "summary.csv")
            aggregate = pd.read_csv(budget_dir / "omni_robot" / "aggregate.csv")
            assert "learned_regret_beam3" in set(summary["method"])
            assert "learned_regret_beam3" in set(aggregate["method"])
            assert (
                budget_dir
                / "learning"
                / "omni_robot"
                / "regret_candidate_costs.csv"
            ).exists()

    def test_point_mass_scenario(self, tmp_path):
        config = from_profile("smoke", scenario="point_mass", output_dir=str(tmp_path))
        results = run_benchmark(config)
        assert "point_mass" in results
        result = results["point_mass"]
        assert len(result.summary) > 0
        assert all(result.summary["budget_violations"] == 0)
        assert all(result.summary["unsound_certificates"] == 0)

    def test_explicit_simple_robot_still_runs(self, tmp_path):
        config = from_profile("smoke", scenario="simple_robot", output_dir=str(tmp_path))
        results = run_benchmark(config)
        assert "simple_robot" in results
        result = results["simple_robot"]
        assert len(result.summary) > 0
        assert all(result.summary["budget_violations"] == 0)
        assert all(result.summary["unsound_certificates"] == 0)

    def test_all_scenarios_runs_default_headline_set(self, tmp_path):
        config = from_profile("smoke", scenario="all", output_dir=str(tmp_path))
        results = run_benchmark(config)
        registered = {s.name for s in registered_scenarios()}
        assert "omni_robot" in results
        assert "simple_robot" not in results
        assert "point_mass" not in results
        if "robot_arm" in registered:
            assert "robot_arm" in results

    def test_girard_beats_box(self, tmp_path):
        config = from_profile("smoke", scenario="omni_robot", method_set="static")
        results = run_benchmark(config)
        agg = results["omni_robot"].aggregate

        girard_tw = agg.loc[agg["method"] == "girard", "mean_trigger_width_mean"].values
        box_tw = agg.loc[agg["method"] == "box", "mean_trigger_width_mean"].values
        if len(girard_tw) > 0 and len(box_tw) > 0:
            assert girard_tw[0] <= box_tw[0]

    def test_parallel_matches_serial_on_stable_metrics(self, tmp_path):
        serial_config = from_profile(
            "smoke", scenario="omni_robot", method_set="static", seeds=2, jobs=1,
        )
        parallel_config = from_profile(
            "smoke", scenario="omni_robot", method_set="static", seeds=2, jobs=2,
        )

        serial = run_benchmark(serial_config, show_progress=False)["omni_robot"].summary
        parallel = run_benchmark(parallel_config, show_progress=False)["omni_robot"].summary

        stable_cols = [
            "method",
            "seed",
            "mean_trigger_width",
            "max_trigger_width",
            "mean_generator_count",
            "max_generator_count",
            "total_reductions",
            "budget_violations",
            "unsound_certificates",
            "mean_approx_error",
            "max_approx_error",
            "abs_error_range",
            "false_positive_rate",
        ]
        serial_stable = serial[stable_cols].sort_values(["seed", "method"]).reset_index(drop=True)
        parallel_stable = parallel[stable_cols].sort_values(["seed", "method"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(serial_stable, parallel_stable)
