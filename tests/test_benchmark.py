"""Smoke tests for the benchmark orchestrator."""

import pytest

from pzr.experiments.benchmark import (
    BenchmarkResult,
    default_methods,
    default_scenarios,
    run_benchmark,
    save_benchmark_results,
)
from pzr.experiments.config import BenchmarkConfig, from_profile, save_config, load_config


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
        config = from_profile("smoke", scenario="omni_robot")
        path = tmp_path / "config.yaml"
        save_config(config, path)
        loaded = load_config(path)
        assert loaded.length == config.length
        assert loaded.seeds == config.seeds
        assert loaded.budget == config.budget

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

    def test_smoke_both_scenarios(self, tmp_path):
        config = from_profile("smoke", scenario="all", output_dir=str(tmp_path))
        results = run_benchmark(config)
        assert "omni_robot" in results
        assert "simple_robot" in results
        assert "robot_arm" in results

    def test_static_only(self, tmp_path):
        config = from_profile("smoke", scenario="omni_robot", method_set="static")
        results = run_benchmark(config)
        result = results["omni_robot"]
        methods = set(result.summary["method"].unique())
        assert not any(m.startswith("mpc") for m in methods)

    def test_save_results(self, tmp_path):
        config = from_profile("smoke", scenario="omni_robot")
        results = run_benchmark(config)
        save_benchmark_results(results, tmp_path / "output")

        assert (tmp_path / "output" / "omni_robot" / "timeseries.csv").exists()
        assert (tmp_path / "output" / "omni_robot" / "summary.csv").exists()
        assert (tmp_path / "output" / "omni_robot" / "aggregate.csv").exists()
        assert (tmp_path / "output" / "config.yaml").exists()
        assert (tmp_path / "output" / "manifest.json").exists()

    def test_point_mass_scenario(self, tmp_path):
        config = from_profile("smoke", scenario="point_mass", output_dir=str(tmp_path))
        results = run_benchmark(config)
        assert "point_mass" in results
        result = results["point_mass"]
        assert len(result.summary) > 0
        assert all(result.summary["budget_violations"] == 0)
        assert all(result.summary["unsound_certificates"] == 0)

    def test_all_scenarios(self, tmp_path):
        config = from_profile("smoke", scenario="all", output_dir=str(tmp_path))
        results = run_benchmark(config)
        assert "omni_robot" in results
        assert "simple_robot" in results
        assert "point_mass" in results
        assert "robot_arm" in results

    def test_girard_beats_box(self, tmp_path):
        config = from_profile("smoke", scenario="omni_robot", method_set="static")
        results = run_benchmark(config)
        agg = results["omni_robot"].aggregate

        girard_tw = agg.loc[agg["method"] == "girard", "mean_trigger_width_mean"].values
        box_tw = agg.loc[agg["method"] == "box", "mean_trigger_width_mean"].values
        if len(girard_tw) > 0 and len(box_tw) > 0:
            assert girard_tw[0] <= box_tw[0]
