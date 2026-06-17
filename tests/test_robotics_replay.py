"""Tests for robotics replay evaluation and visualization."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pzr.experiments import robotics_replay


def test_procedural_replay_is_deterministic_and_seed_varying():
    a = robotics_replay.make_procedural_replay_bundle("drone", length=20, seed=0)
    b = robotics_replay.make_procedural_replay_bundle("drone", length=20, seed=0)
    c = robotics_replay.make_procedural_replay_bundle("drone", length=20, seed=1)

    assert [m.true_values for m in a.trace] == [m.true_values for m in b.trace]
    assert [m.true_values for m in a.trace] != [m.true_values for m in c.trace]
    assert a.metadata["trace_source"] == "procedural_replay"


def test_trend_predictor_uses_history_only():
    history = [
        robotics_replay.SafetyStreamMeasurement(
            time=0.0,
            values=(1.0, 2.0),
            true_values=(1.0, 2.0),
        ),
        robotics_replay.SafetyStreamMeasurement(
            time=1.0,
            values=(1.1, 1.8),
            true_values=(1.1, 1.8),
        ),
    ]

    predicted = robotics_replay.SafetyStreamTrendPredictor(max_slope=0.5).predict(
        history, horizon=2,
    )

    assert len(predicted) == 2
    assert predicted[0].time == 2.0
    assert predicted[0].values == pytest.approx((1.2, 1.6))
    assert predicted[1].values == pytest.approx((1.3, 1.4))


def test_f1tenth_physical_monitor_projects_trigger_zonotope():
    bundle = robotics_replay.make_procedural_replay_bundle(
        "f1tenth", length=12, seed=0, monitor="physical",
    )

    monitor = bundle.monitor
    state = monitor.initial_state()
    result = monitor.step(state, bundle.trace[0])
    trigger_z = monitor.trigger_zonotope(result.state)

    assert bundle.metadata["monitor"] == "physical"
    assert len(bundle.trace[0].values) == len(robotics_replay.F1TENTH_PHYSICAL_STATE_NAMES)
    assert trigger_z.dimension == len(robotics_replay.F1TENTH_PHYSICAL_TRIGGER_NAMES)
    assert len(result.verdicts) == len(robotics_replay.F1TENTH_PHYSICAL_TRIGGER_NAMES)


def test_f1tenth_physical_monitor_propagates_existing_generators():
    bundle = robotics_replay.make_procedural_replay_bundle(
        "f1tenth", length=12, seed=0, monitor="physical",
    )
    monitor = bundle.monitor
    first = monitor.step(monitor.initial_state(), bundle.trace[0]).state
    old_generators = first.zonotope.generators.copy()
    geometry = robotics_replay._geometry_from_payload(bundle.trace[1].payload)
    transition = robotics_replay._f1tenth_transition_matrix(first, bundle.trace[1], geometry)
    contraction = robotics_replay._f1tenth_observer_contraction()

    second = monitor.step(first, bundle.trace[1]).state
    propagated = second.zonotope.generators[:, :old_generators.shape[1]]

    assert np.allclose(propagated, contraction @ transition @ old_generators)
    assert not np.allclose(propagated, old_generators)
    assert second.calibration_indices == first.calibration_indices


def test_f1tenth_physical_projection_contains_sampled_trigger_points():
    bundle = robotics_replay.make_procedural_replay_bundle(
        "f1tenth", length=12, seed=1, monitor="physical",
    )
    monitor = bundle.monitor
    result = monitor.step(monitor.initial_state(), bundle.trace[4])
    trigger_z = monitor.trigger_zonotope(result.state)
    geometry = robotics_replay._geometry_from_payload(bundle.trace[4].payload)
    rng = np.random.default_rng(0)

    for _ in range(32):
        coeffs = rng.uniform(-0.35, 0.35, result.state.zonotope.generator_count)
        physical = result.state.zonotope.sample(coeffs)
        margins = robotics_replay._f1tenth_physical_margins(physical, geometry)
        assert trigger_z.contains_in_interval_hull(margins, atol=1e-8)


def test_f1tenth_physical_fresh_basis_varies_by_racing_phase():
    bundle = robotics_replay.make_procedural_replay_bundle(
        "f1tenth", length=24, seed=0, monitor="physical",
    )
    monitor = bundle.monitor

    early = monitor.fresh_generators_for(bundle.trace[2])
    late = monitor.fresh_generators_for(bundle.trace[-3])

    assert early.shape == late.shape
    assert not np.allclose(early, late)


def test_drone_physical_monitor_projects_trigger_zonotope():
    bundle = robotics_replay.make_procedural_replay_bundle(
        "drone", length=12, seed=0, monitor="physical",
    )

    monitor = bundle.monitor
    result = monitor.step(monitor.initial_state(), bundle.trace[3])
    trigger_z = monitor.trigger_zonotope(result.state)

    assert bundle.metadata["monitor"] == "physical"
    assert bundle.metadata["scenario_family"] == "stress"
    assert len(bundle.trace[0].values) == len(robotics_replay.DRONE_PHYSICAL_STATE_NAMES)
    assert trigger_z.dimension == len(robotics_replay.DRONE_PHYSICAL_TRIGGER_NAMES)
    assert len(result.verdicts) == len(robotics_replay.DRONE_PHYSICAL_TRIGGER_NAMES)


def test_drone_physical_monitor_propagates_existing_generators():
    bundle = robotics_replay.make_procedural_replay_bundle(
        "drone", length=12, seed=0, monitor="physical",
    )
    monitor = bundle.monitor
    first = monitor.step(monitor.initial_state(), bundle.trace[0]).state
    old_generators = first.zonotope.generators.copy()
    geometry = robotics_replay._drone_geometry_from_payload(bundle.trace[1].payload)
    gate_id = robotics_replay._drone_gate_id_from_payload(bundle.trace[1].payload)
    transition = robotics_replay._drone_transition_matrix(first, bundle.trace[1], geometry, gate_id)

    second = monitor.step(first, bundle.trace[1]).state
    propagated = second.zonotope.generators[:, :old_generators.shape[1]]

    assert np.allclose(propagated, transition @ old_generators)
    assert not np.allclose(propagated, old_generators)
    assert second.calibration_indices == first.calibration_indices


def test_drone_physical_projection_contains_sampled_trigger_points():
    bundle = robotics_replay.make_procedural_replay_bundle(
        "drone", length=12, seed=1, monitor="physical",
    )
    monitor = bundle.monitor
    result = monitor.step(monitor.initial_state(), bundle.trace[5])
    trigger_z = monitor.trigger_zonotope(result.state)
    geometry = robotics_replay._drone_geometry_from_payload(bundle.trace[5].payload)
    gate_id = robotics_replay._drone_gate_id_from_payload(bundle.trace[5].payload)
    rng = np.random.default_rng(0)

    for _ in range(32):
        coeffs = rng.uniform(-0.3, 0.3, result.state.zonotope.generator_count)
        physical = result.state.zonotope.sample(coeffs)
        margins = robotics_replay._drone_physical_margins(physical, geometry, gate_id)
        assert trigger_z.contains_in_interval_hull(margins, atol=1e-8)


def test_replay_eval_writes_focused_mpc_outputs(tmp_path):
    result = robotics_replay.run_replay_eval(
        candidates=("drone",),
        length=16,
        seed=0,
        seeds=1,
        warmup_steps=2,
        budget=10,
        horizon=1,
        beam_width=2,
        output=tmp_path,
        trace_source="procedural",
    )

    assert (tmp_path / "timeseries.csv").stat().st_size > 0
    assert (tmp_path / "summary.csv").stat().st_size > 0
    assert (tmp_path / "aggregate.csv").stat().st_size > 0
    assert (tmp_path / "policy_gain.csv").stat().st_size > 0
    assert (tmp_path / "winner_by_step.csv").stat().st_size > 0
    assert (tmp_path / "scenario_summary.csv").stat().st_size > 0
    assert (tmp_path / "intervention_summary.csv").stat().st_size > 0
    assert (tmp_path / "trace_metadata.json").stat().st_size > 0
    assert (tmp_path / "seed_0" / "drone_derived_streams.csv").stat().st_size > 0
    assert (tmp_path / "seed_0" / "drone_payload.jsonl").stat().st_size > 0

    methods = set(result["summary"]["method"])
    assert {"girard", "scott", "mpc_beam3", "mpc_sequence3"} <= methods
    ts = pd.read_csv(tmp_path / "timeseries.csv")
    assert {"predicted_cost", "predicted_sequence", "evaluated_leaves"} <= set(ts.columns)
    scenario = pd.read_csv(tmp_path / "scenario_summary.csv")
    assert set(scenario["scenario_family"]) == {"stress"}
    assert set(scenario["monitor_model"]) == {"dynamics_physical_v2"}
    assert scenario["mean_propagated_width_fraction"].notna().all()
    with open(tmp_path / "trace_metadata.json") as f:
        metadata = json.load(f)
    assert metadata["mpc_candidate_reducers"] == list(robotics_replay.ROBOTICS_MPC_CANDIDATE_NAMES)
    assert metadata["monitor_model"] == "dynamics_physical_v2"


def test_physical_predictors_preserve_geometry_and_advance_state():
    drone = robotics_replay.make_procedural_replay_bundle(
        "drone", length=20, seed=0, monitor="physical",
    )
    drone_predicted = robotics_replay.DronePhysicalPredictor().predict(drone.trace[:12], horizon=2)

    assert len(drone_predicted) == 2
    assert drone_predicted[0].payload["geometry"] == drone.trace[-1].payload["geometry"]
    assert drone_predicted[0].values[:3] != pytest.approx(drone.trace[11].values[:3])

    f1 = robotics_replay.make_procedural_replay_bundle(
        "f1tenth", length=20, seed=0, monitor="physical",
    )
    f1_predicted = robotics_replay.F1TenthPhysicalPredictor().predict(f1.trace[:12], horizon=2)

    assert len(f1_predicted) == 2
    assert f1_predicted[0].payload["geometry"] == f1.trace[-1].payload["geometry"]
    assert f1_predicted[0].values[0] != pytest.approx(f1.trace[11].values[0])


def test_f1tenth_physical_replay_writes_policy_gain_columns(tmp_path):
    result = robotics_replay.run_replay_eval(
        candidates=("f1tenth",),
        length=24,
        seed=0,
        seeds=1,
        warmup_steps=0,
        budget=12,
        horizon=2,
        beam_width=2,
        output=tmp_path,
        trace_source="procedural",
        monitor="physical",
    )

    gain = result["policy_gain"]
    assert {"baseline_method", "visualization_ready", "scott_mean_trigger_width_gain"} <= set(gain.columns)
    assert "combastel" in set(result["summary"]["method"])
    mpc_gain = gain[gain["method"] == "mpc_beam3"].iloc[0]
    assert mpc_gain["scott_mean_trigger_width_gain"] > 0.0
    metadata = pd.read_json(tmp_path / "trace_metadata.json", typ="series")
    assert metadata["monitor"] == "physical"


def test_f1tenth_v3_longer_replay_stays_bounded(tmp_path):
    result = robotics_replay.run_replay_eval(
        candidates=("f1tenth",),
        length=80,
        seed=0,
        seeds=1,
        warmup_steps=0,
        budget=12,
        horizon=2,
        beam_width=2,
        output=tmp_path,
        trace_source="procedural",
        monitor="physical",
        method_set="sweep",
    )

    summary = result["summary"]
    f1 = summary[summary["candidate"] == "f1tenth"]
    assert float(f1["max_trigger_width"].max()) < 1e10
    scenario = result["scenario_summary"].iloc[0]
    assert scenario["monitor_model"] == "dynamics_physical_v3"
    assert np.isfinite(float(scenario["max_projection_remainder_radius"]))
    assert float(scenario["max_projection_remainder_radius"]) < 1e9


def test_headline_method_set_uses_neutral_mpc_suite(tmp_path):
    result = robotics_replay.run_replay_eval(
        candidates=("drone",),
        length=14,
        seed=0,
        seeds=1,
        warmup_steps=0,
        budget=8,
        horizon=1,
        beam_width=2,
        output=tmp_path,
        trace_source="procedural",
        monitor="physical",
        method_set="headline",
    )

    methods = set(result["summary"]["method"])
    assert {
        "pca",
        "mpc_rollout",
        "mpc_pair_rollout3",
        "mpc_beam3",
        "mpc_sequence3",
    } <= methods
    assert "mpc_rollout_scott" not in methods
    assert result["metadata"]["mpc_candidate_reducers"] == ["girard", "methA", "scott"]


def test_paper_core_method_set_omits_exact_sequence_audit(tmp_path):
    result = robotics_replay.run_replay_eval(
        candidates=("drone",),
        length=14,
        seed=0,
        seeds=1,
        warmup_steps=0,
        budget=8,
        horizon=1,
        beam_width=2,
        output=tmp_path,
        trace_source="procedural",
        monitor="physical",
        method_set="paper_core",
    )

    methods = set(result["summary"]["method"])
    assert {"mpc_rollout", "mpc_pair_rollout3", "mpc_beam3"} <= methods
    assert "mpc_sequence3" not in methods


def test_live_drone_bundle_converts_to_physical_monitor():
    raw = {
        "step": 0,
        "time": 0.0,
        "obs": [0.5, 0.1, -2.4, 0.0, 1.0, 0.0],
        "info": {"current_target_gate_id": 0},
        "gates": [[9.5, -8.5, 0, 0, 0, -1.57, 0]],
        "obstacles": [[7.5, -6.5, 0, 0, 0, 0]],
    }
    bundle = robotics_replay.ProbeBundle(
        candidate="drone",
        monitor=robotics_replay.SafetyStreamMonitor(robotics_replay.drone_stream_profile()),
        trace=(robotics_replay.SafetyStreamMeasurement(
            time=0.0,
            values=(1.0,) * 6,
            true_values=(1.0,) * 6,
            payload={"raw_record": raw},
        ),),
        metadata={"candidate": "drone", "status": "available"},
    )

    converted = robotics_replay._live_bundle_to_physical(
        candidate="drone", bundle=bundle, seed=0, f1tenth_map="paper_chicane",
    )

    assert converted.metadata["monitor"] == "physical"
    assert converted.metadata["monitor_model"] == "dynamics_physical_v2"
    assert converted.monitor.profile.stream_names == robotics_replay.DRONE_PHYSICAL_TRIGGER_NAMES
    assert len(converted.trace[0].values) == len(robotics_replay.DRONE_PHYSICAL_STATE_NAMES)
    geometry = converted.trace[0].payload["geometry"]
    assert geometry["gates"][0][:2] == pytest.approx([9.5, -8.5])
    assert geometry["obstacles"][0] == pytest.approx([7.5, -6.5])


def test_live_f1tenth_bundle_converts_to_physical_monitor():
    geometry = robotics_replay.F1TenthTrackGeometry(
        amp1=0.51,
        freq1=0.47,
        phase1=0.11,
        amp2=0.13,
        freq2=1.02,
        phase2=-0.21,
        half_width=0.63,
        width_wave=0.18,
        width_freq=0.81,
        width_phase=0.33,
        bottleneck_x=-0.4,
        bottleneck_depth=0.28,
        bottleneck_sigma=0.9,
        front_phase=0.44,
    )
    raw = {
        "step": 0,
        "time": 0.0,
        "obs": {
            "poses_x": [-7.5],
            "poses_y": [0.0],
            "poses_theta": [0.0],
            "linear_vels_x": [0.8],
            "ang_vels_z": [0.02],
        },
        "map_geometry": geometry.to_payload(),
    }
    bundle = robotics_replay.ProbeBundle(
        candidate="f1tenth",
        monitor=robotics_replay.SafetyStreamMonitor(robotics_replay.f1tenth_stream_profile()),
        trace=(robotics_replay.SafetyStreamMeasurement(
            time=0.0,
            values=(1.0,) * 6,
            true_values=(1.0,) * 6,
            payload={"raw_record": raw},
        ),),
        metadata={"candidate": "f1tenth", "status": "available"},
    )

    converted = robotics_replay._live_bundle_to_physical(
        candidate="f1tenth", bundle=bundle, seed=0, f1tenth_map="paper_chicane",
    )

    assert converted.metadata["monitor"] == "physical"
    assert converted.metadata["monitor_model"] == "dynamics_physical_v3"
    assert converted.monitor.profile.stream_names == robotics_replay.F1TENTH_PHYSICAL_TRIGGER_NAMES
    assert len(converted.trace[0].values) == len(robotics_replay.F1TENTH_PHYSICAL_STATE_NAMES)
    assert converted.trace[0].payload["geometry"]["half_width"] == pytest.approx(0.63)
    assert converted.trace[0].payload["raw_record"]["width_profile"]


def test_physical_trigger_names_stay_stable():
    assert robotics_replay.DRONE_PHYSICAL_TRIGGER_NAMES == (
        "obstacle_clearance_margin",
        "gate_alignment_margin",
        "corridor_margin",
        "altitude_low_margin",
        "altitude_high_margin",
        "speed_margin",
    )
    assert robotics_replay.F1TENTH_PHYSICAL_TRIGGER_NAMES == (
        "left_boundary_margin",
        "right_boundary_margin",
        "heading_margin",
        "time_to_collision_margin",
        "curvature_speed_margin",
        "yaw_rate_margin",
    )


def test_f1tenth_collector_seeded_geometry_is_deterministic_and_varying():
    pytest.importorskip("PIL")
    pytest.importorskip("yaml")
    path = Path(__file__).resolve().parents[1] / "tools" / "collect_f1tenth_trace.py"
    spec = importlib.util.spec_from_file_location("collect_f1tenth_trace", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    a = module._seeded_track_geometry(0, stress_randomize=True)
    b = module._seeded_track_geometry(0, stress_randomize=True)
    c = module._seeded_track_geometry(1, stress_randomize=True)

    assert a == b
    assert a != c


def test_f1tenth_budget_sweep_writes_incremental_outputs(tmp_path):
    result = robotics_replay.run_replay_budget_sweep(
        candidate="f1tenth",
        budgets=(8, 12),
        length=14,
        seed=0,
        seeds=1,
        warmup_steps=0,
        horizon=1,
        beam_width=2,
        output=tmp_path,
        trace_source="procedural",
        monitor="physical",
        render_selected=False,
    )

    expected = {
        "budget_sweep_summary.csv",
        "budget_policy_gain.csv",
        "budget_reducer_counts.csv",
        "budget_runtime.csv",
        "budget_scenario_summary.csv",
        "budget_intervention_summary.csv",
        "budget_degeneracy_summary.csv",
        "budget_sweep_metadata.json",
        "budget_sweep_report.md",
    }
    for name in expected:
        assert (tmp_path / name).stat().st_size > 0
    for budget in (8, 12):
        budget_dir = tmp_path / f"budget_{budget}"
        assert (budget_dir / "summary.csv").stat().st_size > 0
        assert (budget_dir / "timeseries.csv").stat().st_size > 0
        assert (budget_dir / "policy_gain.csv").stat().st_size > 0

    summary = pd.read_csv(tmp_path / "budget_sweep_summary.csv")
    gain = pd.read_csv(tmp_path / "budget_policy_gain.csv")
    counts = pd.read_csv(tmp_path / "budget_reducer_counts.csv")
    runtime = pd.read_csv(tmp_path / "budget_runtime.csv")
    degeneracy = pd.read_csv(tmp_path / "budget_degeneracy_summary.csv")
    assert {"budget", "method", "mean_trigger_width", "total_time_ms"} <= set(summary.columns)
    assert {"budget", "baseline_method", "mean_trigger_width_gain", "visualization_ready"} <= set(gain.columns)
    assert {"budget", "reducer_used", "steps"} <= set(counts.columns)
    assert {"budget", "method", "mean_reduction_time_ms"} <= set(runtime.columns)
    assert {
        "budget",
        "best_static_approx_fraction",
        "mpc_non_girard_fraction",
        "mean_propagated_width_fraction",
        "mean_projection_remainder_fraction",
        "mean_transition_variation_score",
    } <= set(degeneracy.columns)
    assert degeneracy["mean_propagated_width_fraction"].notna().all()
    assert "mpc_beam3" in set(summary["method"])
    assert "mpc_sequence3" not in set(summary["method"])
    assert result["metadata"]["method_set"] == "sweep"
    assert result["selected_budget"]["budget"] in {8, 12}


def test_budget_sweep_supports_all_candidates(tmp_path):
    result = robotics_replay.run_replay_budget_sweep(
        candidate="all",
        budgets=(8,),
        length=10,
        seed=0,
        seeds=1,
        warmup_steps=0,
        horizon=1,
        beam_width=2,
        output=tmp_path,
        trace_source="procedural",
        monitor="physical",
        render_selected=False,
    )

    summary = pd.read_csv(tmp_path / "budget_sweep_summary.csv")
    scenario = pd.read_csv(tmp_path / "budget_scenario_summary.csv")
    intervention = pd.read_csv(tmp_path / "budget_intervention_summary.csv")
    assert set(summary["candidate"]) == {"drone", "f1tenth"}
    assert set(scenario["candidate"]) == {"drone", "f1tenth"}
    assert {"spurious_interventions", "missed_violations"} <= set(intervention.columns)
    assert result["metadata"]["requested_candidates"] == ["drone", "f1tenth"]


def test_budget_sweep_regret_learning_writes_rows_and_artifacts(tmp_path):
    result = robotics_replay.run_replay_budget_sweep(
        candidate="drone",
        budgets=(8,),
        length=14,
        seed=0,
        seeds=1,
        warmup_steps=0,
        horizon=1,
        beam_width=2,
        output=tmp_path,
        trace_source="procedural",
        monitor="physical",
        render_selected=False,
        learned_mode="regret",
        regret_oracle="beam3",
        regret_iterations=1,
        regret_epochs=5,
        regret_train_seeds=1,
        regret_eval_seeds=1,
    )

    summary = pd.read_csv(tmp_path / "budget_sweep_summary.csv")
    gain = pd.read_csv(tmp_path / "budget_policy_gain.csv")
    counts = pd.read_csv(tmp_path / "budget_reducer_counts.csv")
    runtime = pd.read_csv(tmp_path / "budget_runtime.csv")
    assert "learned_regret_beam3" in set(summary["method"])
    assert "learned_regret_beam3" in set(gain["method"])
    assert "learned_regret_beam3" in set(counts["method"])
    assert "learned_regret_beam3" in set(runtime["method"])
    assert (
        tmp_path
        / "budget_8"
        / "learning"
        / "drone"
        / "regret_candidate_costs.csv"
    ).stat().st_size > 0
    assert result["metadata"]["learned_mode"] == "regret"


def test_replay_render_writes_stills_storyboard_and_metadata(tmp_path):
    eval_dir = tmp_path / "eval"
    render_dir = tmp_path / "render"
    robotics_replay.run_replay_eval(
        candidates=("f1tenth",),
        length=14,
        seed=0,
        seeds=1,
        warmup_steps=0,
        budget=10,
        horizon=1,
        beam_width=2,
        output=eval_dir,
        trace_source="procedural",
        monitor="physical",
    )

    artifacts = robotics_replay.render_replay(
        eval_dir=eval_dir,
        output=render_dir,
        candidates=("f1tenth",),
        methods=("scott", "mpc_beam3"),
        seed=0,
        stride=5,
        save_gif=False,
    )

    expected = {
        "f1tenth_first_png",
        "f1tenth_first_pdf",
        "f1tenth_middle_png",
        "f1tenth_middle_pdf",
        "f1tenth_last_png",
        "f1tenth_last_pdf",
        "f1tenth_storyboard_png",
        "f1tenth_storyboard_pdf",
        "f1tenth_metadata",
    }
    assert expected <= set(artifacts)
    for path in artifacts.values():
        assert path.stat().st_size > 0
