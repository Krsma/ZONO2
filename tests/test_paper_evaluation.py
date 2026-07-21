from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pzr.learning.dataset import ReducerCostDataset
from pzr.learning.training import filter_training_budgets
from pzr.rtlola.engine import RtlolaEvent
from pzr.rtlola.paper_artifacts import (
    _plot_budget_facets,
    ablation_table,
    budget80_extrapolation,
    objective_comparison_table,
)
from pzr.rtlola.paper_experiment import (
    BOOTSTRAP_REPLICATES,
    HEADLINE_METHODS,
    PAPER_CELL_SCHEMA,
    PAPER_CONFIG_SCHEMA,
    ExecutionRegime,
    RunState,
    aggregate_trace_metrics,
    load_paper_experiment_config,
    pilot_projection,
    reducer_composition,
    validate_cell_manifest,
    validate_summary_matrix,
)
from pzr.rtlola.paper_pipeline import (
    DEFAULT_CONFIG,
    EvaluationCellJob,
    RUN_EXIT_APPROVAL_REQUIRED,
    _execute_cell_job,
    _junit_counts,
    _runtime_provenance,
    _scientific_failure_count,
    _validate_runtime_provenance,
    build_parser,
    run_complete_paper_evaluation,
    run_exploratory_bundle,
    run_paper_stage,
)
from pzr.rtlola.parity import ParityConfig


def _summary_row(
    *,
    condition: str = "random_waypoint",
    seed: int = 100,
    budget: int = 40,
    method: str = "girard",
    status: str = "completed",
    false_positives: int = 1,
    negatives: int = 10,
    loss: float = 2.0,
) -> dict[str, object]:
    return {
        "condition": condition,
        "trace_kind": condition,
        "trace_id": f"{condition}:seed-{seed}",
        "trace_sha256": f"trace-{seed}",
        "seed": seed,
        "budget": budget,
        "method": method,
        "status": status,
        "event_count": 500,
        "false_positive_count": false_positives,
        "false_negative_count": 0,
        "reference_negative_count": negatives,
        "reference_positive_count": 2,
        "mean_approx_loss": loss,
        "final_approx_loss": loss,
        "max_approx_loss": loss,
        "sum_approx_loss": loss * 500,
        "mean_state_width": 1.0,
        "max_state_width": 2.0,
        "total_time_ms": 250.0,
        "fallback_count": int(status == RunState.FALLBACK_FAILED.value),
        "infeasible_candidate_count": 0,
    }


def test_checked_config_declares_stable_methods_regimes_and_cell_counts():
    config = load_paper_experiment_config(DEFAULT_CONFIG)

    assert DEFAULT_CONFIG.name == "paper_evaluation_v1.yaml"
    assert config.schema == PAPER_CONFIG_SCHEMA == "pzr.paper-evaluation-config.v1"
    assert config.experiment_id == "paper-evaluation-v1"
    assert config.ablation_workers == 1
    assert config.expected_cells("pilot") == 216
    assert config.expected_cells("generalization") == 5_040
    assert config.expected_cells("headline") == 224
    assert config.expected_cells("objective-comparison") == 56
    assert config.expected_cells("ablation") == 320
    assert config.method_by_name["mpc_terminal_beam"].execution_regime is (
        ExecutionRegime.OFFLINE_RECORDED
    )
    assert config.method_by_name[
        "mpc_terminal_beam_predictive_linear"
    ].execution_regime is ExecutionRegime.ONLINE_PREDICTIVE
    assert config.method_by_name["mpc_terminal_full_width"].horizon == 1
    assert config.method_by_name["mpc_cumulative_beam"].objective.value == "cumulative"
    assert config.method_by_name["pairwise_ranking_policy_budget80"].horizon == 0


def test_checked_config_seed_groups_are_pairwise_disjoint():
    config = load_paper_experiment_config(DEFAULT_CONFIG)
    groups = (
        config.train_seeds, config.validation_seeds,
        config.reserved_exploration_seeds, config.pilot_seeds,
        config.generalization_seeds, config.ablation_seeds,
    )
    for index, left in enumerate(groups):
        for right in groups[index + 1:]:
            assert not set(left) & set(right)


def test_smoke_parity_event_limit_is_explicit_and_validated(tmp_path):
    config = ParityConfig(
        rlola_eval=tmp_path / "rlola-eval",
        output=tmp_path / "parity",
        trace_kinds=("figure8",),
        bounds=(15,),
        run_speed_gate=False,
        event_limit=20,
    )
    assert config.event_limit == 20
    with pytest.raises(ValueError, match="at least two"):
        replace(config, event_limit=1)


def test_runtime_provenance_rejects_stale_native_stack():
    provenance = _runtime_provenance()
    provenance["binding_revision"] = "old"
    with pytest.raises(ValueError, match="binding_revision"):
        _validate_runtime_provenance(provenance, "pilot")


def test_preflight_junit_counts_rejectable_outcomes(tmp_path):
    report = tmp_path / "pytest.xml"
    report.write_text(
        '<testsuites><testsuite tests="9" failures="1" errors="2" '
        'skipped="3"/></testsuites>'
    )

    assert _junit_counts(report) == {
        "tests": 9, "failures": 1, "errors": 2, "skipped": 3,
    }


def test_scientific_failure_count_includes_timing_failures(tmp_path):
    config = replace(
        load_paper_experiment_config(DEFAULT_CONFIG),
        output_root=tmp_path / "results",
    )
    manifest = config.output_root / "timing" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"failure_count": 4}')

    assert _scientific_failure_count(config) == 4


def test_training_budget_filter_preserves_alignment_and_both_splits():
    dataset = ReducerCostDataset(
        features=np.arange(12, dtype=np.float32).reshape(4, 3),
        teacher_costs=np.asarray([[1.0, 2.0]] * 4),
        feasible=np.ones((4, 2), dtype=bool),
        candidate_names=("girard", "scott"),
        feature_names=("a", "b", "c"),
        splits=("train", "train", "validation", "validation"),
        sample_ids=("a", "b", "c", "d"),
    )
    metadata = pd.DataFrame({
        "sample_id": dataset.sample_ids,
        "budget": [40, 80, 40, 80],
        "split": dataset.splits,
    })

    filtered, selected = filter_training_budgets(dataset, metadata, (80,))

    assert filtered.sample_ids == ("b", "d")
    assert selected["budget"].tolist() == [80, 80]
    np.testing.assert_array_equal(filtered.features, dataset.features[[1, 3]])
    with pytest.raises(ValueError, match="unavailable"):
        filter_training_budgets(dataset, metadata, (150,))


def test_macro_and_pooled_fpr_use_trace_denominators_and_bootstrap_is_deterministic():
    summary = pd.DataFrame([
        _summary_row(seed=100, method="girard", false_positives=1, negatives=10),
        _summary_row(seed=101, method="girard", false_positives=9, negatives=90),
        _summary_row(seed=100, method="scott", false_positives=2, negatives=10),
        _summary_row(seed=101, method="scott", false_positives=0, negatives=90),
    ])

    left = aggregate_trace_metrics(summary, bootstrap_replicates=200, bootstrap_seed=7)
    right = aggregate_trace_metrics(summary, bootstrap_replicates=200, bootstrap_seed=7)

    pd.testing.assert_frame_equal(left, right)
    girard = left[left["method"] == "girard"].iloc[0]
    assert girard["macro_fpr"] == pytest.approx(0.1)
    assert girard["pooled_fpr"] == pytest.approx(0.1)
    scott = left[left["method"] == "scott"].iloc[0]
    assert scott["macro_fpr"] == pytest.approx(0.1)
    assert scott["pooled_fpr"] == pytest.approx(0.02)
    assert scott["bootstrap_replicates"] == 200


def test_any_failed_run_makes_main_point_unavailable_but_retains_valid_only_values():
    summary = pd.DataFrame([
        _summary_row(seed=100),
        _summary_row(seed=101, status=RunState.FALLBACK_FAILED.value, loss=np.nan),
    ])

    point = aggregate_trace_metrics(summary, bootstrap_replicates=20).iloc[0]

    assert not bool(point["available"])
    assert np.isnan(point["macro_fpr"])
    assert point["valid_only_macro_fpr"] == pytest.approx(0.1)
    assert point["fallback_rate"] == pytest.approx(0.5)


def test_trace_misalignment_is_recorded_and_disables_paired_interval():
    summary = pd.DataFrame([
        _summary_row(seed=100, method="girard"),
        _summary_row(seed=101, method="girard"),
        _summary_row(seed=100, method="scott"),
    ])

    result = aggregate_trace_metrics(summary, bootstrap_replicates=20)

    assert not result["paired_seed_alignment"].any()
    assert result["macro_fpr_ci_low"].isna().all()


def test_reducer_composition_excludes_none_fallback_and_infeasible_events():
    timeseries = pd.DataFrame([
        {"condition": "random_waypoint", "budget": 40,
         "method": "mpc_terminal_beam", "reducer_used": "girard",
         "fallback_used": False, "infeasible_candidate_count": 0},
        {"condition": "random_waypoint", "budget": 40,
         "method": "mpc_terminal_beam", "reducer_used": "scott",
         "fallback_used": False, "infeasible_candidate_count": 0},
        {"condition": "random_waypoint", "budget": 40,
         "method": "mpc_terminal_beam", "reducer_used": "none",
         "fallback_used": False, "infeasible_candidate_count": 0},
        {"condition": "random_waypoint", "budget": 40,
         "method": "mpc_terminal_beam", "reducer_used": "interval",
         "fallback_used": True, "infeasible_candidate_count": 0},
        {"condition": "random_waypoint", "budget": 40,
         "method": "mpc_terminal_beam", "reducer_used": "pca",
         "fallback_used": False, "infeasible_candidate_count": 1},
    ])

    result = reducer_composition(timeseries)

    assert set(result["reducer_used"]) == {"girard", "scott"}
    assert result["count"].sum() == 2
    assert result["percentage"].sum() == pytest.approx(100.0)


def test_fallback_cell_is_invalidated_and_keeps_full_diagnostic_series(
    tmp_path, monkeypatch,
):
    config = load_paper_experiment_config(DEFAULT_CONFIG)
    method = config.method_by_name["girard"]
    timeseries = pd.DataFrame([
        {"step": 0, "fallback_used": False, "decision_time_ms": 2.0,
         "approx_loss": 1.0, "method": "girard"},
        {"step": 1, "fallback_used": False, "decision_time_ms": 2.0,
         "approx_loss": 3.0, "method": "girard"},
        {"step": 2, "fallback_used": True, "decision_time_ms": 2.0,
         "approx_loss": 9.0, "method": "girard"},
    ])
    summary = pd.DataFrame([_summary_row(method="girard")])
    monkeypatch.setattr(
        "pzr.rtlola.paper_pipeline.run_event_trace_benchmark",
        lambda *_args, **_kwargs: SimpleNamespace(
            failures=(), timeseries=timeseries, summary=summary,
        ),
    )
    job = EvaluationCellJob(
        stage="pilot",
        directory=tmp_path / "cell",
        trace=SimpleNamespace(
            trace_id="trace", condition="random_waypoint", seed=90,
            events=(RtlolaEvent(0.0, ()), RtlolaEvent(1.0, ()), RtlolaEvent(2.0, ())),
            trace_sha256="trace-hash",
        ),
        budget=40,
        method=method,
        runtime_method="girard",
        reference_path=tmp_path / "reference.json",
        identity={"fingerprint": "cell"},
        model_directory=None,
    )

    row = _execute_cell_job(job)

    assert row["status"] == RunState.FALLBACK_FAILED.value
    assert row["first_fallback_event"] == 2
    assert row["completed_fraction"] == pytest.approx(2 / 3)
    assert row["pre_fallback_mean_loss"] == pytest.approx(2.0)
    assert row["pre_fallback_throughput_events_per_second"] == pytest.approx(500.0)
    assert np.isnan(row["fpr"])
    assert len(pd.read_csv(job.directory / "timeseries_diagnostic.csv")) == 3


def test_stale_or_old_cell_manifest_is_rejected():
    identity = {"fingerprint": "new"}
    with pytest.raises(ValueError, match="unsupported"):
        validate_cell_manifest(
            {"schema": "pzr.old", "identity": identity, "status": "completed"},
            identity,
        )
    with pytest.raises(ValueError, match="stale"):
        validate_cell_manifest(
            {"schema": PAPER_CELL_SCHEMA, "identity": {"fingerprint": "old"},
             "status": "completed"},
            identity,
        )


def test_matrix_validation_rejects_duplicate_cells_and_wrong_count():
    config = load_paper_experiment_config(DEFAULT_CONFIG)
    row = _summary_row()
    duplicate = pd.DataFrame([row] * config.expected_cells("headline"))
    with pytest.raises(ValueError, match="duplicate"):
        validate_summary_matrix(config, "headline", duplicate)
    with pytest.raises(ValueError, match="expected 224"):
        validate_summary_matrix(config, "headline", pd.DataFrame([row]))


def test_pilot_projection_reports_scaling_disk_and_approval_gate():
    summary = pd.DataFrame([
        _summary_row(seed=90, method="girard"),
        _summary_row(seed=91, method="girard"),
    ])
    summary["total_time_ms"] = 360_000.0

    projection = pilot_projection(
        summary, target_cell_count=5_040, worker_count=4,
        disk_bytes=1_000, threshold_hours=72.0,
    )

    assert projection["projected_cpu_hours"] == pytest.approx(504.0)
    assert projection["projected_four_worker_wall_hours"] == pytest.approx(126.0)
    assert projection["projected_disk_bytes"] == 2_520_000
    assert projection["approval_required"] is True


def test_budget80_extrapolation_requires_aligned_policy_pairs():
    rows = pd.DataFrame([
        {"condition": "random_waypoint", "budget": 40,
         "method": "pairwise_ranking_policy", "macro_fpr": 0.1},
    ])
    with pytest.raises(ValueError, match="do not align"):
        budget80_extrapolation(rows)


def test_ablation_marks_failed_grid_cell_unavailable():
    rows = []
    for seed in (60, 61):
        rows.append({
            **_summary_row(seed=seed, method="mpc_terminal_beam_h4_w4"),
            "horizon": 4, "beam_width": 4,
            "status": (
                RunState.COMPLETED.value if seed == 60
                else RunState.NATIVE_FAILED.value
            ),
        })
    result = ablation_table(pd.DataFrame(rows)).iloc[0]
    assert not bool(result["available"])
    assert np.isnan(result["mean_loss"])
    assert bool(result["highlight_default"])


def test_objective_comparison_requires_aligned_terminal_and_cumulative_methods():
    rows = pd.DataFrame([
        _summary_row(condition="figure8", seed=0, method="mpc_terminal_beam"),
        _summary_row(condition="figure8", seed=0, method="mpc_cumulative_beam"),
    ])

    result = objective_comparison_table(rows)

    assert set(result["method"]) == {
        "mpc_terminal_beam", "mpc_cumulative_beam",
    }
    with pytest.raises(ValueError, match="identities differ"):
        objective_comparison_table(rows.iloc[:1])


def test_missing_budget_point_is_not_interpolated_in_exported_plot(tmp_path):
    rows = []
    for method in HEADLINE_METHODS:
        for budget in (40, 80, 150):
            rows.append({
                "condition": "figure8",
                "budget": budget,
                "method": method,
                "macro_fpr": np.nan if budget == 80 else 0.1,
                "macro_fpr_ci_low": np.nan if budget == 80 else 0.05,
                "macro_fpr_ci_high": np.nan if budget == 80 else 0.15,
                "macro_mean_approx_loss": np.nan if budget == 80 else 1.0,
                "fallback_rate": 1.0 if budget == 80 else 0.0,
            })
    _plot_budget_facets(pd.DataFrame(rows), tmp_path / "missing")
    assert (tmp_path / "missing.pdf").stat().st_size > 0
    assert (tmp_path / "missing.png").stat().st_size > 0


def test_cli_exposes_all_staged_commands_and_long_run_approval():
    parser = build_parser()
    for stage in (
        "prepare", "train", "pilot", "objective-comparison", "headline",
        "generalization", "ablation", "timing", "report", "validate",
    ):
        args = parser.parse_args([stage])
        assert args.stage == stage
    assert parser.parse_args(["run"]).stage == "run"
    assert parser.parse_args(["explore"]).stage == "explore"
    assert parser.parse_args(["status"]).stage == "status"
    assert parser.parse_args(["generalization", "--approve-long-run"]).approve_long_run
    assert BOOTSTRAP_REPLICATES == 10_000


def test_exploratory_bundle_runs_only_preflight_training_and_formal_pilot(
    tmp_path, monkeypatch,
):
    config = replace(
        load_paper_experiment_config(DEFAULT_CONFIG),
        output_root=tmp_path / "results",
        paper_artifact_dir=tmp_path / "generated",
    )
    calls = []
    preflight = []
    monkeypatch.setattr(
        "pzr.rtlola.paper_pipeline._run_provenance",
        lambda _config: {"dirty_source_paths": []},
    )
    monkeypatch.setattr(
        "pzr.rtlola.paper_pipeline._run_preflight",
        lambda *_args: preflight.append(True),
    )

    def fake_stage(_config, stage, **_kwargs):
        calls.append(stage)
        if stage == "pilot":
            path = config.output_root / "pilot" / "projection.json"
            path.parent.mkdir(parents=True)
            path.write_text('{"approval_required": false}')

    monkeypatch.setattr("pzr.rtlola.paper_pipeline._run_or_skip_stage", fake_stage)
    monkeypatch.setattr(
        "pzr.rtlola.paper_pipeline._stage_failure_count", lambda *_args: 0,
    )

    result = run_exploratory_bundle(config, smoke=True)

    assert preflight == [True]
    assert calls == ["prepare", "train", "pilot"]
    assert result.status == "exploration_completed"
    manifest = json.loads(result.manifest.read_text())
    assert manifest["included_stages"] == ["preflight", "prepare", "train", "pilot"]
    assert "parity" in manifest["excluded_stages"]
    assert "bounded-exploration" in manifest["excluded_stages"]


def test_ablation_rejects_concurrent_worker_override():
    config = load_paper_experiment_config(DEFAULT_CONFIG)
    with pytest.raises(ValueError, match="single worker"):
        run_paper_stage(config, "ablation", workers=4)


def test_complete_run_stops_at_pilot_gate_and_rejects_preapproval(
    tmp_path, monkeypatch,
):
    config = replace(
        load_paper_experiment_config(DEFAULT_CONFIG),
        output_root=tmp_path / "results",
        paper_artifact_dir=tmp_path / "generated",
    )
    calls = []
    monkeypatch.setattr(
        "pzr.rtlola.paper_pipeline._run_provenance",
        lambda _config: {"dirty_source_paths": []},
    )
    monkeypatch.setattr("pzr.rtlola.paper_pipeline._run_preflight", lambda *_: None)
    monkeypatch.setattr(
        "pzr.rtlola.paper_pipeline._run_or_resume_parity", lambda *_args, **_kwargs: None,
    )

    def fake_stage(_config, stage, **_kwargs):
        calls.append(stage)
        if stage == "pilot":
            path = config.output_root / "pilot" / "projection.json"
            path.parent.mkdir(parents=True)
            path.write_text('{"approval_required": true}')

    monkeypatch.setattr("pzr.rtlola.paper_pipeline._run_or_skip_stage", fake_stage)
    monkeypatch.setattr(
        "pzr.rtlola.paper_pipeline._scientific_failure_count", lambda _config: 0,
    )

    result = run_complete_paper_evaluation(
        config, rlola_eval=tmp_path / "rlola-eval", smoke=True,
    )

    assert result.exit_code == RUN_EXIT_APPROVAL_REQUIRED
    assert calls == ["prepare", "train", "pilot"]
    with pytest.raises(ValueError, match="only after a pilot"):
        run_complete_paper_evaluation(
            replace(config, output_root=tmp_path / "fresh"),
            rlola_eval=tmp_path / "rlola-eval",
            approve_long_run=True,
            smoke=True,
        )


def test_approved_complete_run_records_approval_and_exact_stage_order(
    tmp_path, monkeypatch,
):
    config = replace(
        load_paper_experiment_config(DEFAULT_CONFIG),
        output_root=tmp_path / "results",
        paper_artifact_dir=tmp_path / "generated",
    )
    projection = config.output_root / "pilot" / "projection.json"
    projection.parent.mkdir(parents=True)
    projection.write_text('{"approval_required": true}')
    calls = []
    monkeypatch.setattr(
        "pzr.rtlola.paper_pipeline._run_provenance",
        lambda _config: {"dirty_source_paths": []},
    )
    monkeypatch.setattr("pzr.rtlola.paper_pipeline._run_preflight", lambda *_: None)
    monkeypatch.setattr(
        "pzr.rtlola.paper_pipeline._run_or_resume_parity", lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "pzr.rtlola.paper_pipeline._run_or_skip_stage",
        lambda _config, stage, **_kwargs: calls.append(stage),
    )
    monkeypatch.setattr(
        "pzr.rtlola.paper_pipeline._scientific_failure_count", lambda _config: 0,
    )

    result = run_complete_paper_evaluation(
        config,
        rlola_eval=tmp_path / "rlola-eval",
        approve_long_run=True,
        smoke=True,
    )

    assert result.exit_code == 0
    assert calls == [
        "prepare", "train", "pilot", "objective-comparison", "headline",
        "generalization", "ablation", "timing", "report", "validate",
    ]
    approval = config.output_root / "pilot" / "approval.json"
    assert approval.is_file()
    assert json.loads(approval.read_text())["approved"] is True
