from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pzr.learning.dataset import ReducerCostDataset
from pzr.learning.provenance import pzr_source_sha256
from pzr.learning.training import filter_training_budgets
from pzr.rtlola.engine import RtlolaEvent
from pzr.rtlola.paper_artifacts import (
    _require_single_trace_source,
    _plot_budget_facets,
    ablation_table,
    objective_comparison_table,
)
from pzr.rtlola.paper_experiment import (
    BOOTSTRAP_REPLICATES,
    HEADLINE_METHODS,
    PAPER_CELL_SCHEMA,
    PAPER_CONFIG_SCHEMA,
    ExecutionRegime,
    RunState,
    TraceSource,
    aggregate_trace_metrics,
    cell_identity,
    load_paper_experiment_config,
    pilot_projection,
    reducer_composition,
    validate_cell_manifest,
    validate_summary_matrix,
)
from pzr.rtlola.paper_pipeline import (
    DEFAULT_CONFIG,
    EvaluationCellJob,
    PAPER_PREFLIGHT_MARKER_EXPRESSION,
    RUN_EXIT_APPROVAL_REQUIRED,
    RUN_EXIT_PRIMARY_READINESS_FAILED,
    _execute_cell_job,
    _fixed_figure8_traces,
    _generated_nominal_stage_traces,
    _junit_counts,
    _paper_preflight_command,
    _runtime_provenance,
    _run_objective_comparison,
    _run_prepare,
    _run_train,
    _scientific_failure_count,
    _validate_timing_stage,
    _validate_runtime_provenance,
    _validate_completed_stage,
    build_parser,
    run_complete_paper_evaluation,
    run_exploratory_bundle,
    run_paper_stage,
    run_scientific_paper_evaluation,
)


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
    trace_source = (
        TraceSource.FIXED_RLOLAEVAL.value
        if condition.startswith("figure8")
        else TraceSource.GENERATED_NOMINAL.value
    )
    return {
        "trace_source": trace_source,
        "condition": condition,
        "trace_kind": condition,
        "trace_id": f"{condition}:seed-{seed}",
        "trace_sha256": f"trace-{seed}",
        "seed": seed,
        "budget": budget,
        "method": method,
        "model_training_budget": (
            budget if method == "pairwise_ranking_policy" else np.nan
        ),
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
        "event_loop_time_ms": 250.0,
        "cell_elapsed_ms": 300.0,
        "fallback_count": int(status == RunState.FALLBACK_FAILED.value),
        "infeasible_candidate_count": 0,
    }


def _projection_payload(config, *, approval_required: bool) -> dict[str, object]:
    return {
        "schema": "pzr.paper-evaluation-pilot-projection.v3",
        "config_sha256": config.config_sha256,
        "pzr_source_sha256": pzr_source_sha256(),
        "gated_stage": "generalization",
        "trace_scope": TraceSource.GENERATED_NOMINAL.value,
        "target_cell_count": config.expected_cells("generalization"),
        "approval_required": approval_required,
    }


def test_checked_config_declares_stable_methods_regimes_and_cell_counts():
    config = load_paper_experiment_config(DEFAULT_CONFIG)

    assert DEFAULT_CONFIG.name == "paper_evaluation_v2.yaml"
    assert config.schema == PAPER_CONFIG_SCHEMA == "pzr.paper-evaluation-config.v3"
    assert config.experiment_id == "paper-evaluation-v2"
    assert config.ablation_workers == 1
    assert config.evaluation_workers == 10
    assert config.generated_nominal_trace_kind == "random_waypoint"
    assert config.fixed_figure8_trace_kinds == (
        "figure8", "figure8_drift", "figure8_geofence",
        "figure8_drift_geofence",
    )
    assert config.expected_cells("pilot") == 112
    assert config.expected_cells("generalization") == 1_120
    assert config.expected_cells("headline") == 224
    assert config.expected_cells("objective-comparison") == 56
    assert config.expected_cells("ablation") == 80
    assert config.expected_cells("timing") == 672
    assert config.expected_timing_summary_points == 224
    assert config.expected_timing_warmups == 56
    assert config.method_by_name["mpc_terminal_beam"].execution_regime is (
        ExecutionRegime.OFFLINE_RECORDED
    )
    assert config.method_by_name[
        "mpc_terminal_beam_predictive_linear"
    ].execution_regime is ExecutionRegime.ONLINE_PREDICTIVE
    assert config.method_by_name["mpc_terminal_full_width"].horizon == 1
    assert config.method_by_name["mpc_cumulative_beam"].objective.value == "cumulative"
    assert config.method_by_name["pairwise_ranking_policy"].horizon == 0
    assert config.pilot_budgets == config.budgets
    assert config.teacher_dataset_parent_sha256 == (
        "885c3dfbf70ddf614db72f564877e667e056a59966e62094d365606e0b503602"
    )


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


def test_v2_config_schema_is_rejected_instead_of_reinterpreted(tmp_path):
    stale = tmp_path / "stale.yaml"
    stale.write_text(DEFAULT_CONFIG.read_text().replace(
        "pzr.paper-evaluation-config.v3",
        "pzr.paper-evaluation-config.v2",
        1,
    ))
    with pytest.raises(ValueError, match="unsupported paper config schema"):
        load_paper_experiment_config(stale)


def test_fixed_figure8_traces_validate_pinned_hashes_lengths_and_provenance():
    config = load_paper_experiment_config(DEFAULT_CONFIG)
    traces = _fixed_figure8_traces(config)

    assert tuple(trace.trace_kind for trace in traces) == config.fixed_figure8_trace_kinds
    assert {trace.trace_source for trace in traces} == {TraceSource.FIXED_RLOLAEVAL}
    assert {len(trace.events) for trace in traces} == {2_340}
    assert all(trace.provenance["source_file_sha256"] == trace.trace_sha256 for trace in traces)
    assert all(trace.provenance["source_event_count"] == 2_340 for trace in traces)


def test_generated_evaluation_store_is_nominal_only_and_stage_owned(
    tmp_path, monkeypatch,
):
    config = replace(
        load_paper_experiment_config(DEFAULT_CONFIG),
        output_root=tmp_path,
        teacher_dataset_parent=None,
        teacher_dataset_parent_sha256=None,
        enforce_canonical_scope=False,
    )
    generated = []
    monkeypatch.setattr(
        "pzr.rtlola.paper_pipeline.generate_random_waypoint_trace_store",
        lambda trace_config: generated.append(trace_config),
    )
    monkeypatch.setattr(
        "pzr.rtlola.paper_pipeline._stored_traces", lambda *_args, **_kwargs: (),
    )

    assert _generated_nominal_stage_traces(config, "pilot", (90, 91)) == ()
    assert generated[0].conditions == ("random_waypoint",)
    assert generated[0].output == tmp_path / "pilot" / "traces" / "generated-nominal"


def test_prepare_generates_only_nominal_teacher_traces(tmp_path, monkeypatch):
    config = replace(
        load_paper_experiment_config(DEFAULT_CONFIG),
        output_root=tmp_path,
        teacher_dataset_parent=None,
        teacher_dataset_parent_sha256=None,
        enforce_canonical_scope=False,
    )
    generated = []

    def fake_generate(trace_config):
        generated.append(trace_config)
        return SimpleNamespace(root=trace_config.output, manifest_sha256="store-hash")

    monkeypatch.setattr(
        "pzr.rtlola.paper_pipeline.generate_random_waypoint_trace_store",
        fake_generate,
    )
    monkeypatch.setattr(
        "pzr.rtlola.paper_pipeline.run_learning_collection",
        lambda collection_config: collection_config.output / "dataset",
    )
    monkeypatch.setattr(
        "pzr.rtlola.paper_pipeline.dataset_sha256", lambda _path: "dataset-hash",
    )

    _run_prepare(config)

    assert len(generated) == 1
    assert generated[0].conditions == ("random_waypoint",)
    assert generated[0].seed_start == 0
    assert generated[0].seed_count == 26
    assert not (tmp_path / "prepare" / "traces" / "pilot").exists()


def test_cell_identity_records_explicit_trace_source_and_provenance(tmp_path):
    config = load_paper_experiment_config(DEFAULT_CONFIG)
    reference = tmp_path / "reference.json"
    reference.write_text("{}")
    identity = cell_identity(
        config,
        stage="pilot",
        trace_id="random_waypoint:seed-90",
        trace_sha256="trace-hash",
        trace_source=TraceSource.GENERATED_NOMINAL,
        trace_kind="random_waypoint",
        trace_provenance={
            "trace_store_manifest_sha256": "store-hash",
            "generator_config_sha256": "generator-hash",
        },
        condition="random_waypoint",
        seed=90,
        event_count=500,
        budget=40,
        method=config.method_by_name["girard"],
        reference_path=reference,
        model_sha256=None,
        source_sha256="source-hash",
    )

    assert identity["trace_source"] == TraceSource.GENERATED_NOMINAL.value
    assert identity["trace_kind"] == "random_waypoint"
    assert identity["trace_provenance"]["generator_config_sha256"] == "generator-hash"


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


def test_paper_preflight_explicitly_excludes_standalone_parity(tmp_path):
    command = _paper_preflight_command(tmp_path / "pytest.xml")

    assert command[-3:-1] == ["-m", PAPER_PREFLIGHT_MARKER_EXPRESSION]
    assert PAPER_PREFLIGHT_MARKER_EXPRESSION == "not rlola_parity"


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


def test_paper_training_builds_one_fixed_hyperparameter_model_per_budget(
    tmp_path, monkeypatch,
):
    config = replace(
        load_paper_experiment_config(DEFAULT_CONFIG),
        output_root=tmp_path,
        budgets=(40, 80),
        enforce_canonical_scope=False,
    )
    dataset_manifest = tmp_path / "prepare" / "teacher" / "dataset" / "manifest.json"
    dataset_manifest.parent.mkdir(parents=True)
    dataset_manifest.write_text("{}")
    calls = []

    def fake_train(training_config):
        calls.append(training_config)
        training_config.output.mkdir(parents=True)
        return training_config.output

    monkeypatch.setattr(
        "pzr.rtlola.paper_pipeline.run_reducer_training", fake_train,
    )
    monkeypatch.setattr(
        "pzr.rtlola.paper_pipeline.model_sha256",
        lambda path: f"hash-{path.name}",
    )

    _run_train(config)

    assert [call.budget_filter for call in calls] == [(40,), (80,)]
    assert {call.epochs for call in calls} == {config.training_epochs}
    assert {call.batch_size for call in calls} == {config.training_batch_size}
    manifest = json.loads((tmp_path / "train" / "manifest.json").read_text())
    assert set(manifest["models_by_budget"]) == {"40", "80"}


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
         "trace_source": TraceSource.GENERATED_NOMINAL.value,
         "trace_kind": "random_waypoint",
         "method": "mpc_terminal_beam", "reducer_used": "girard",
         "fallback_used": False, "infeasible_candidate_count": 0},
        {"condition": "random_waypoint", "budget": 40,
         "trace_source": TraceSource.GENERATED_NOMINAL.value,
         "trace_kind": "random_waypoint",
         "method": "mpc_terminal_beam", "reducer_used": "scott",
         "fallback_used": False, "infeasible_candidate_count": 0},
        {"condition": "random_waypoint", "budget": 40,
         "trace_source": TraceSource.GENERATED_NOMINAL.value,
         "trace_kind": "random_waypoint",
         "method": "mpc_terminal_beam", "reducer_used": "none",
         "fallback_used": False, "infeasible_candidate_count": 0},
        {"condition": "random_waypoint", "budget": 40,
         "trace_source": TraceSource.GENERATED_NOMINAL.value,
         "trace_kind": "random_waypoint",
         "method": "mpc_terminal_beam", "reducer_used": "interval",
         "fallback_used": True, "infeasible_candidate_count": 0},
        {"condition": "random_waypoint", "budget": 40,
         "trace_source": TraceSource.GENERATED_NOMINAL.value,
         "trace_kind": "random_waypoint",
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
            trace_source=TraceSource.GENERATED_NOMINAL,
            trace_kind="random_waypoint",
            provenance={},
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


def test_v1_stage_manifest_is_rejected_before_resume(tmp_path):
    config = replace(
        load_paper_experiment_config(DEFAULT_CONFIG),
        output_root=tmp_path,
    )
    manifest = tmp_path / "pilot" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({
        "schema": "pzr.paper-evaluation-stage.v1",
        "config_sha256": config.config_sha256,
        "pzr_source_sha256": pzr_source_sha256(),
    }))

    with pytest.raises(ValueError, match="unsupported pilot stage manifest schema"):
        _validate_completed_stage(config, "pilot")


def test_matrix_validation_rejects_duplicate_cells_and_wrong_count():
    config = load_paper_experiment_config(DEFAULT_CONFIG)
    row = _summary_row()
    duplicate = pd.DataFrame([row] * config.expected_cells("headline"))
    with pytest.raises(ValueError, match="duplicate"):
        validate_summary_matrix(config, "headline", duplicate)
    with pytest.raises(ValueError, match="expected 224"):
        validate_summary_matrix(config, "headline", pd.DataFrame([row]))


def test_matrix_validation_rejects_cross_budget_specialist_dispatch():
    config = replace(
        load_paper_experiment_config(DEFAULT_CONFIG),
        budgets=(40, 80),
        pilot_budgets=(40, 80),
        pilot_seeds=(90,),
        enforce_canonical_scope=False,
    )
    rows = [
        _summary_row(seed=90, budget=budget, method=method)
        for budget in config.pilot_budgets
        for method in HEADLINE_METHODS
    ]
    frame = pd.DataFrame(rows)
    validate_summary_matrix(config, "pilot", frame)
    selected = frame["method"] == "pairwise_ranking_policy"
    frame.loc[selected & (frame["budget"] == 80), "model_training_budget"] = 40
    with pytest.raises(ValueError, match="not matched"):
        validate_summary_matrix(config, "pilot", frame)


def test_pilot_projection_reports_scaling_disk_and_approval_gate():
    summary = pd.DataFrame([
        _summary_row(seed=90, method="girard"),
        _summary_row(seed=91, method="girard"),
    ])
    summary["event_loop_time_ms"] = 1_000_000.0
    summary["cell_elapsed_ms"] = 1_000_100.0

    projection = pilot_projection(
        summary, target_cell_count=20, worker_count=4,
        disk_bytes=1_000, threshold_hours=72.0,
    )

    assert projection["projected_cpu_hours"] == pytest.approx(20_000 / 3600)
    assert projection["projected_wall_hours"] == pytest.approx(5_000 / 3600)
    assert projection["projected_disk_bytes"] == 10_000
    assert projection["gated_stage"] == "generalization"
    assert projection["trace_scope"] == TraceSource.GENERATED_NOMINAL.value
    assert projection["approval_required"] is False


def test_timing_validation_asserts_existing_repetition_contract(tmp_path):
    config = replace(
        load_paper_experiment_config(DEFAULT_CONFIG),
        output_root=tmp_path,
        budgets=(40,),
        fixed_figure8_trace_kinds=("figure8",),
        timing_repetitions=2,
        enforce_canonical_scope=False,
    )
    directory = tmp_path / "timing"
    directory.mkdir()
    raw = pd.DataFrame([
        {
            "trace_source": TraceSource.FIXED_RLOLAEVAL.value,
            "trace_kind": "figure8",
            "condition": "figure8",
            "budget": 40,
            "method": method,
            "repetition": repetition,
        }
        for method in HEADLINE_METHODS
        for repetition in range(2)
    ])
    summary = raw.drop(columns="repetition").drop_duplicates()
    raw.to_csv(directory / "timing_repetitions.csv", index=False)
    summary.to_csv(directory / "summary.csv", index=False)
    manifest = {
        "cell_count": config.expected_cells("timing"),
        "warmup_count": config.expected_timing_warmups,
    }

    _validate_timing_stage(config, manifest)
    with pytest.raises(ValueError, match="manifest measured count"):
        _validate_timing_stage(config, {**manifest, "cell_count": 1})


def test_reporting_rejects_mixed_generated_and_fixed_trace_sources():
    frame = pd.DataFrame({
        "trace_source": [
            TraceSource.GENERATED_NOMINAL.value,
            TraceSource.FIXED_RLOLAEVAL.value,
        ],
    })
    with pytest.raises(ValueError, match="has trace sources"):
        _require_single_trace_source(
            frame, TraceSource.GENERATED_NOMINAL.value, "composition",
        )


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
        "generalization", "ablation", "timing", "science-report",
        "science-validate", "report", "validate",
    ):
        args = parser.parse_args([stage])
        assert args.stage == stage
    assert parser.parse_args(["run"]).stage == "run"
    assert parser.parse_args(["explore"]).stage == "explore"
    assert parser.parse_args(["evaluate"]).stage == "evaluate"
    assert parser.parse_args(["status"]).stage == "status"
    assert parser.parse_args(["generalization", "--approve-long-run"]).approve_long_run
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--rlola-eval", "/tmp/rlola-eval"])
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
            path.write_text(json.dumps(_projection_payload(
                config, approval_required=False,
            )))

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
    assert "parity" not in manifest["excluded_stages"]
    assert "bounded-exploration" in manifest["excluded_stages"]


def test_ablation_rejects_concurrent_worker_override():
    config = load_paper_experiment_config(DEFAULT_CONFIG)
    with pytest.raises(ValueError, match="single worker"):
        run_paper_stage(config, "ablation", workers=4)


@pytest.mark.parametrize(
    "prp_status,expected_exit,expected_tail",
    [
        (RunState.COMPLETED.value, 0, [
            "headline", "objective-comparison", "ablation", "generalization",
            "science-report", "science-validate",
        ]),
        (RunState.FALLBACK_FAILED.value, RUN_EXIT_PRIMARY_READINESS_FAILED, []),
    ],
)
def test_scientific_evaluate_defers_timing_and_enforces_primary_readiness(
    tmp_path, monkeypatch, prp_status, expected_exit, expected_tail,
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
        "pzr.rtlola.paper_pipeline._require_no_hard_failures", lambda *_: None,
    )
    monkeypatch.setattr(
        "pzr.rtlola.paper_pipeline._scientific_failure_count", lambda _config: 0,
    )

    def fake_stage(_config, stage, **_kwargs):
        calls.append(stage)
        if stage == "pilot":
            directory = config.output_root / "pilot"
            directory.mkdir(parents=True)
            (directory / "projection.json").write_text(json.dumps(
                _projection_payload(config, approval_required=False)
            ))
            pd.DataFrame([
                _summary_row(
                    seed=seed,
                    budget=budget,
                    method="pairwise_ranking_policy",
                    status=(
                        prp_status if (seed, budget) == (90, 40)
                        else RunState.COMPLETED.value
                    ),
                )
                for seed in config.pilot_seeds
                for budget in config.pilot_budgets
            ]).to_csv(directory / "summary.csv", index=False)

    monkeypatch.setattr("pzr.rtlola.paper_pipeline._run_or_skip_stage", fake_stage)

    result = run_scientific_paper_evaluation(config, smoke=True)

    assert result.exit_code == expected_exit
    assert calls == ["prepare", "train", "pilot", *expected_tail]
    assert "timing" not in calls


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
    def fake_stage(_config, stage, **_kwargs):
        calls.append(stage)
        if stage == "pilot":
            path = config.output_root / "pilot" / "projection.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(_projection_payload(
                config, approval_required=True,
            )))

    monkeypatch.setattr("pzr.rtlola.paper_pipeline._run_or_skip_stage", fake_stage)
    monkeypatch.setattr(
        "pzr.rtlola.paper_pipeline._scientific_failure_count", lambda _config: 0,
    )
    monkeypatch.setattr(
        "pzr.rtlola.paper_pipeline._pilot_primary_failure_count", lambda _config: 0,
    )

    result = run_complete_paper_evaluation(
        config, smoke=True,
    )

    assert result.exit_code == RUN_EXIT_APPROVAL_REQUIRED
    assert calls == ["prepare", "train", "pilot"]
    with pytest.raises(ValueError, match="only after a pilot"):
        run_complete_paper_evaluation(
            replace(config, output_root=tmp_path / "fresh"),
            approve_long_run=True,
            smoke=True,
        )


def test_scientific_gate_runs_ungated_fixed_and_ablation_stages_first(
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
        "pzr.rtlola.paper_pipeline._require_no_hard_failures", lambda *_: None,
    )
    monkeypatch.setattr(
        "pzr.rtlola.paper_pipeline._scientific_failure_count", lambda _config: 0,
    )

    def fake_stage(_config, stage, **_kwargs):
        calls.append(stage)
        if stage == "pilot":
            directory = config.output_root / "pilot"
            directory.mkdir(parents=True)
            (directory / "projection.json").write_text(json.dumps(
                _projection_payload(config, approval_required=True)
            ))
            pd.DataFrame([
                _summary_row(
                    seed=seed, budget=budget, method="pairwise_ranking_policy",
                )
                for seed in config.pilot_seeds
                for budget in config.pilot_budgets
            ]).to_csv(directory / "summary.csv", index=False)

    monkeypatch.setattr("pzr.rtlola.paper_pipeline._run_or_skip_stage", fake_stage)

    result = run_scientific_paper_evaluation(config, smoke=True)

    assert result.exit_code == RUN_EXIT_APPROVAL_REQUIRED
    assert calls == [
        "prepare", "train", "pilot", "headline", "objective-comparison", "ablation",
    ]


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
    projection.write_text(json.dumps(_projection_payload(
        config, approval_required=True,
    )))
    calls = []
    monkeypatch.setattr(
        "pzr.rtlola.paper_pipeline._run_provenance",
        lambda _config: {"dirty_source_paths": []},
    )
    monkeypatch.setattr("pzr.rtlola.paper_pipeline._run_preflight", lambda *_: None)
    monkeypatch.setattr(
        "pzr.rtlola.paper_pipeline._run_or_skip_stage",
        lambda _config, stage, **_kwargs: calls.append(stage),
    )
    monkeypatch.setattr(
        "pzr.rtlola.paper_pipeline._scientific_failure_count", lambda _config: 0,
    )
    monkeypatch.setattr(
        "pzr.rtlola.paper_pipeline._pilot_primary_failure_count", lambda _config: 0,
    )

    result = run_complete_paper_evaluation(
        config,
        approve_long_run=True,
        smoke=True,
    )

    assert result.exit_code == 0
    assert calls == [
        "prepare", "train", "pilot", "headline", "objective-comparison",
        "generalization", "ablation", "timing", "report", "validate",
    ]
    approval = config.output_root / "pilot" / "approval.json"
    assert approval.is_file()
    assert json.loads(approval.read_text())["approved"] is True


def test_objective_comparison_requires_validated_headline_cells_for_reuse(
    tmp_path, monkeypatch,
):
    config = replace(
        load_paper_experiment_config(DEFAULT_CONFIG),
        output_root=tmp_path / "results",
    )
    with pytest.raises(FileNotFoundError):
        _run_objective_comparison(config, workers=4)
