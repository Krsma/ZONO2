from types import SimpleNamespace
import json
from pathlib import Path

import pandas as pd
import pytest

import pzr.rtlola.policy_evaluation as evaluation
from pzr.rtlola.policy_evaluation import (
    FixedPolicyEvaluationConfig,
    PolicyComparison,
    run_fixed_policy_evaluation,
)
from pzr.rtlola.policy_reporting import (
    COMPARISON_METRICS,
    best_static_metrics,
    comparison_to_reference,
    explicit_policy_comparisons,
)


MODEL_NAMES = (
    "pairwise_ranking_policy", "soft_kl_secondary",
    "pairwise_ranking_policy_dart", "soft_kl_dart_secondary",
)
MODEL_HASHES = {name: f"hash-{name}" for name in MODEL_NAMES}
POLICIES = {name: object() for name in MODEL_NAMES}


def _mock_prepare_reference(config, calls=None):
    if calls is not None:
        calls.append(("reference", config.trace_kind))
    path = Path(config.reference_cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")


def _result(config, method):
    timeseries = pd.DataFrame([
        {
            "method": method,
            "budget": config.budget,
            "trace_kind": config.trace_kind,
            "step": step,
            "pre_generator_count": 50 if step else 10,
            "reducer_used": "girard" if step else "none",
            "approx_loss": float(step + 1),
            "state_width": float(step + 2),
            "fallback_used": False,
            "infeasible_candidate_count": 0,
        }
        for step in range(config.length)
    ])
    summary = pd.DataFrame([{
        "method": method,
        "budget": config.budget,
        "trace_kind": config.trace_kind,
        "fpr": 0.1,
        "fnr": 0.2,
        "mean_approx_loss": 1.5,
        "final_approx_loss": 2.0,
        "max_approx_loss": 2.0,
        "sum_approx_loss": 3.0,
        "mean_state_width": 2.5,
        "max_state_width": 3.0,
        "total_time_ms": 4.0,
        "false_positive_count": 1,
        "false_negative_count": 1,
        "reference_positive_count": 5,
        "reference_negative_count": 10,
        "fallback_count": 0,
        "infeasible_candidate_count": 0,
    }])
    return SimpleNamespace(timeseries=timeseries, summary=summary, failures=())


def test_fixed_policy_evaluation_resumes_validated_cells(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        evaluation, "prepare_reference_cache",
        lambda config: _mock_prepare_reference(config, calls),
    )

    def run_baseline(config):
        method = config.methods[0]
        calls.append(("baseline", method))
        return _result(config, method)

    def run_learned(config, _policy, *, method):
        calls.append(("learned", method))
        return _result(config, method)

    monkeypatch.setattr(evaluation, "run_benchmark", run_baseline)
    monkeypatch.setattr(evaluation, "run_direct_policy_benchmark", run_learned)
    config = FixedPolicyEvaluationConfig(
        output=tmp_path,
        model_names=MODEL_NAMES,
        trace_kinds=("figure8",),
        budgets=(40,),
        benchmark_methods=("girard", "mpc_terminal_full_width"),
        candidate_names=("girard", "scott", "pca", "combastel"),
        length=2,
    )

    run_fixed_policy_evaluation(
        config, POLICIES, model_sha256=MODEL_HASHES, source_sha256="source",
    )
    assert calls.count(("baseline", "girard")) == 1
    assert calls.count(("baseline", "mpc_terminal_full_width")) == 1
    for name in MODEL_NAMES:
        assert calls.count(("learned", name)) == 1
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["schema"] == "pzr.policy-evaluation.v2"
    assert manifest["cell_count"] == 6
    learned_cell = json.loads(
        (tmp_path / "cells/figure8/budget-40/pairwise_ranking_policy/manifest.json").read_text()
    )
    static_cell = json.loads(
        (tmp_path / "cells/figure8/budget-40/girard/manifest.json").read_text()
    )
    assert learned_cell["model_sha256"] == MODEL_HASHES["pairwise_ranking_policy"]
    assert static_cell["model_sha256"] is None
    calls.clear()
    run_fixed_policy_evaluation(
        config, POLICIES, model_sha256=MODEL_HASHES, source_sha256="source",
    )
    assert calls == [("reference", "figure8")]
    for filename in (
        "timeseries.csv", "summary.csv", "candidate_selection.csv",
        "decision_accounting.csv", "macro_metrics.csv", "macro_loss_metrics.csv",
        "macro_width_metrics.csv", "macro_runtime_metrics.csv",
        "micro_trigger_metrics.csv", "method_comparisons.csv",
        "best_static_metrics.csv", "manifest.json",
    ):
        assert (tmp_path / filename).stat().st_size > 0
    assert not (tmp_path / "policy_comparisons.csv").exists()
    assert not (tmp_path / "plots/policy_comparisons.png").exists()


def test_fixed_policy_evaluation_rejects_stale_cell(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "prepare_reference_cache", _mock_prepare_reference)
    monkeypatch.setattr(
        evaluation, "run_benchmark",
        lambda config: _result(config, config.methods[0]),
    )
    monkeypatch.setattr(
        evaluation, "run_direct_policy_benchmark",
        lambda config, _policy, *, method: _result(config, method),
    )
    config = FixedPolicyEvaluationConfig(
        output=tmp_path,
        model_names=MODEL_NAMES,
        trace_kinds=("figure8",),
        budgets=(40,),
        benchmark_methods=("girard",),
        candidate_names=("girard", "scott", "pca", "combastel"),
        length=2,
    )
    run_fixed_policy_evaluation(
        config, POLICIES, model_sha256=MODEL_HASHES, source_sha256="source",
    )

    with pytest.raises(ValueError, match="stale"):
        run_fixed_policy_evaluation(
            config, POLICIES, model_sha256=MODEL_HASHES, source_sha256="changed",
        )


def test_fixed_policy_evaluation_prepares_references_before_parallel_cells(
    tmp_path, monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        evaluation,
        "prepare_reference_cache",
        lambda config: _mock_prepare_reference(config, calls),
    )
    monkeypatch.setattr(
        evaluation,
        "run_benchmark",
        lambda config: _result(config, config.methods[0]),
    )
    monkeypatch.setattr(
        evaluation,
        "run_direct_policy_benchmark",
        lambda config, _policy, *, method: _result(config, method),
    )

    class ImmediatePool:
        def __init__(self, *, max_workers, mp_context, max_tasks_per_child):
            del mp_context
            assert max_tasks_per_child == 1
            calls.append(("pool", max_workers))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def map(self, function, jobs):
            return tuple(function(job) for job in jobs)

    monkeypatch.setattr(evaluation, "ProcessPoolExecutor", ImmediatePool)

    def run_job(job):
        policy = object() if job.method in job.learned_methods else None
        evaluation._load_or_run_cell(
            directory=job.directory,
            identity=job.identity,
            benchmark_config=job.benchmark_config,
            method=job.method,
            learned_methods=job.learned_methods,
            policy=policy,
            expected_length=job.expected_length,
        )

    monkeypatch.setattr(evaluation, "_run_evaluation_cell_job", run_job)
    config = FixedPolicyEvaluationConfig(
        output=tmp_path,
        model_names=MODEL_NAMES,
        trace_kinds=("figure8", "square"),
        budgets=(40,),
        benchmark_methods=("girard",),
        candidate_names=("girard", "scott", "pca", "combastel"),
        length=2,
    )

    _, summary = run_fixed_policy_evaluation(
        config,
        POLICIES,
        model_sha256=MODEL_HASHES,
        source_sha256="source",
        model_directories={name: tmp_path / name for name in MODEL_NAMES},
        workers=2,
    )

    assert calls[:2] == [("reference", "figure8"), ("reference", "square")]
    assert calls[2] == ("pool", 2)
    assert len(summary) == 10


def test_policy_comparison_requires_aligned_cells():
    summary = pd.DataFrame([
        {"trace_kind": "figure8", "budget": 40, "method": "learned"},
        {"trace_kind": "square", "budget": 40, "method": "girard"},
    ])
    with pytest.raises(ValueError, match="do not align"):
        comparison_to_reference(summary, "policy", "girard")


def test_best_static_is_selected_independently_for_each_metric():
    rows = []
    for method, loss, width in (("girard", 1.0, 5.0), ("scott", 2.0, 3.0)):
        row = {
            "trace_kind": "figure8", "budget": 40, "method": method,
            **{metric: 4.0 for metric in COMPARISON_METRICS},
        }
        row["mean_approx_loss"] = loss
        row["mean_state_width"] = width
        rows.append(row)

    result = best_static_metrics(pd.DataFrame(rows), ("girard", "scott"))

    loss = result[result["metric"] == "mean_approx_loss"].iloc[0]
    width = result[result["metric"] == "mean_state_width"].iloc[0]
    assert loss["best_static_method"] == "girard"
    assert width["best_static_method"] == "scott"


def test_policy_comparisons_use_repeated_explicit_pairs():
    rows = []
    for index, method in enumerate(MODEL_NAMES):
        row = {
            "trace_kind": "figure8", "budget": 40, "method": method,
            **{metric: float(index + 1) for metric in COMPARISON_METRICS},
        }
        rows.append(row)

    comparisons = (
        PolicyComparison("data_scale", "pairwise_ranking_policy_dart", "pairwise_ranking_policy"),
        PolicyComparison("objective", "soft_kl_secondary", "pairwise_ranking_policy"),
    )
    result = explicit_policy_comparisons(pd.DataFrame(rows), comparisons)

    assert set(result["comparison"]) == {"data_scale", "objective"}
    assert set(result[result["comparison"] == "data_scale"]["challenger"]) == {
        "pairwise_ranking_policy_dart"
    }


def test_best_static_reports_undefined_trigger_metrics_without_dropping_cells():
    rows = []
    for method in ("girard", "scott"):
        row = {
            "trace_kind": "figure8", "budget": 40, "method": method,
            **{metric: 1.0 for metric in COMPARISON_METRICS},
        }
        row["fpr"] = float("nan")
        rows.append(row)

    result = best_static_metrics(pd.DataFrame(rows), ("girard", "scott"))

    fpr = result[result["metric"] == "fpr"].iloc[0]
    assert fpr["defined_static_count"] == 0
    assert pd.isna(fpr["best_static_method"])
    assert pd.isna(fpr["best_static_value"])


def test_primary_evaluation_matrix_contains_144_validated_cells(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "prepare_reference_cache", _mock_prepare_reference)
    monkeypatch.setattr(
        evaluation, "run_benchmark",
        lambda config: _result(config, config.methods[0]),
    )
    monkeypatch.setattr(
        evaluation, "run_direct_policy_benchmark",
        lambda config, _policy, *, method: _result(config, method),
    )
    monkeypatch.setattr(evaluation, "write_policy_reports", lambda *_args, **_kwargs: None)
    config = FixedPolicyEvaluationConfig(
        output=tmp_path,
        model_names=("pairwise_ranking_policy",),
        trace_kinds=(
            "figure8", "figure8_drift", "random", "random_drift",
            "square", "square_drift",
        ),
        budgets=(40, 80, 120, 180),
        benchmark_methods=("girard", "scott", "pca", "combastel", "mpc_terminal_full_width"),
        candidate_names=("girard", "scott", "pca", "combastel"),
        length=1,
        expected_cell_count=144,
    )

    _, summary = run_fixed_policy_evaluation(
        config,
        {"pairwise_ranking_policy": POLICIES["pairwise_ranking_policy"]},
        model_sha256={
            "pairwise_ranking_policy": MODEL_HASHES["pairwise_ranking_policy"]
        },
        source_sha256="source",
    )

    assert len(summary) == 144


def test_exploratory_and_promoted_matrix_cell_counts_are_declared():
    screen = FixedPolicyEvaluationConfig(
        output=Path("screen"),
        model_names=("clean20", "clean36", "dart36", "expected20"),
        trace_kinds=("figure8", "random", "square_drift"),
        budgets=(40, 80, 120, 180), benchmark_methods=("girard",),
        candidate_names=("girard", "scott", "pca", "combastel"),
        comparisons=(
            PolicyComparison("data_scale", "clean36", "clean20"),
            PolicyComparison("dart_effect", "dart36", "clean36"),
            PolicyComparison("objective", "expected20", "clean20"),
        ),
        expected_cell_count=60,
    )
    assert screen.expected_cell_count == 60
    promoted = FixedPolicyEvaluationConfig(
        output=Path("promoted"), model_names=("winner", "reference"),
        trace_kinds=(
            "figure8", "figure8_drift", "random", "random_drift", "square", "square_drift",
        ),
        budgets=(40, 80, 120, 180), benchmark_methods=("girard",),
        candidate_names=("girard", "scott", "pca", "combastel"),
        comparisons=(PolicyComparison("promotion", "winner", "reference"),),
        expected_cell_count=72,
    )
    assert promoted.expected_cell_count == 72


def test_mpc_addon_matrix_declares_168_cells_and_prediction_schedule():
    config = FixedPolicyEvaluationConfig(
        output=Path("mpc-addon"),
        model_names=("pairwise_ranking_policy",),
        trace_kinds=(
            "figure8", "figure8_drift", "random", "random_drift",
            "square", "square_drift",
        ),
        budgets=(40, 80, 120, 180),
        benchmark_methods=(
            "girard", "mpc_terminal_beam", "mpc_terminal_full_width",
            "mpc_terminal_beam_predictive_hold",
            "mpc_terminal_beam_predictive_linear",
            "mpc_terminal_beam_predictive_quadratic",
        ),
        candidate_names=("girard", "scott", "pca", "combastel"),
        horizon=3,
        beam_width=4,
        prediction_step_seconds=0.1,
        expected_cell_count=168,
    )
    assert config.expected_cell_count == 168


def test_evaluation_rejects_invalid_explicit_comparisons_and_cell_counts(tmp_path):
    with pytest.raises(ValueError, match="missing learned models"):
        FixedPolicyEvaluationConfig(
            output=tmp_path, model_names=("clean",), trace_kinds=("figure8",),
            budgets=(40,), benchmark_methods=("girard",), candidate_names=("girard",),
            comparisons=(PolicyComparison("objective", "missing", "clean"),),
        )
    with pytest.raises(ValueError, match="matrix has 2 cells"):
        FixedPolicyEvaluationConfig(
            output=tmp_path, model_names=("clean",), trace_kinds=("figure8",),
            budgets=(40,), benchmark_methods=("girard",), candidate_names=("girard",),
            expected_cell_count=144,
        )
