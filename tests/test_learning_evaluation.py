from types import SimpleNamespace
import json

import pandas as pd
import pytest

import pzr.rtlola.learning_evaluation as evaluation
from pzr.rtlola.learning_evaluation import (
    FixedLearningEvaluationConfig,
    best_static_metrics,
    comparison_to_baseline,
    run_fixed_learning_evaluation,
)


MODEL_NAMES = ("learned_pairwise_clean", "learned_soft_clean", "learned_soft_dart")
MODEL_HASHES = {name: f"hash-{name}" for name in MODEL_NAMES}
POLICIES = {name: object() for name in MODEL_NAMES}


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


def test_fixed_learning_evaluation_resumes_validated_cells(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        evaluation, "prepare_reference_cache",
        lambda config: calls.append(("reference", config.trace_kind)),
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
    config = FixedLearningEvaluationConfig(
        output=tmp_path,
        model_names=MODEL_NAMES,
        trace_kinds=("figure8",),
        budgets=(40,),
        baselines=("girard", "mpc_terminal_full_width"),
        candidate_names=("girard", "scott", "pca", "combastel"),
        length=2,
    )

    run_fixed_learning_evaluation(
        config, POLICIES, model_sha256=MODEL_HASHES, source_sha256="source",
    )
    assert calls.count(("baseline", "girard")) == 1
    assert calls.count(("baseline", "mpc_terminal_full_width")) == 1
    for name in MODEL_NAMES:
        assert calls.count(("learned", name)) == 1
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["cell_count"] == 5
    learned_cell = json.loads(
        (tmp_path / "cells/figure8/budget-40/learned_pairwise_clean/manifest.json").read_text()
    )
    static_cell = json.loads(
        (tmp_path / "cells/figure8/budget-40/girard/manifest.json").read_text()
    )
    assert learned_cell["model_sha256"] == MODEL_HASHES["learned_pairwise_clean"]
    assert static_cell["model_sha256"] is None
    calls.clear()
    run_fixed_learning_evaluation(
        config, POLICIES, model_sha256=MODEL_HASHES, source_sha256="source",
    )
    assert calls == [("reference", "figure8")]
    for filename in (
        "timeseries.csv", "summary.csv", "candidate_selection.csv",
        "decision_accounting.csv", "macro_metrics.csv", "macro_loss_metrics.csv",
        "macro_width_metrics.csv", "macro_runtime_metrics.csv",
        "micro_trigger_metrics.csv", "method_comparisons.csv",
        "best_static_metrics.csv", "objective_data_ablation.csv", "manifest.json",
    ):
        assert (tmp_path / filename).stat().st_size > 0


def test_fixed_learning_evaluation_rejects_stale_cell(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "prepare_reference_cache", lambda _config: None)
    monkeypatch.setattr(
        evaluation, "run_benchmark",
        lambda config: _result(config, config.methods[0]),
    )
    monkeypatch.setattr(
        evaluation, "run_direct_policy_benchmark",
        lambda config, _policy, *, method: _result(config, method),
    )
    config = FixedLearningEvaluationConfig(
        output=tmp_path,
        model_names=MODEL_NAMES,
        trace_kinds=("figure8",),
        budgets=(40,),
        baselines=("girard",),
        candidate_names=("girard", "scott", "pca", "combastel"),
        length=2,
    )
    run_fixed_learning_evaluation(
        config, POLICIES, model_sha256=MODEL_HASHES, source_sha256="source",
    )

    with pytest.raises(ValueError, match="stale"):
        run_fixed_learning_evaluation(
            config, POLICIES, model_sha256=MODEL_HASHES, source_sha256="changed",
        )


def test_fixed_learning_evaluation_prepares_references_before_parallel_cells(
    tmp_path, monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        evaluation,
        "prepare_reference_cache",
        lambda config: calls.append(("reference", config.trace_kind)),
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
    config = FixedLearningEvaluationConfig(
        output=tmp_path,
        model_names=MODEL_NAMES,
        trace_kinds=("figure8", "square"),
        budgets=(40,),
        baselines=("girard",),
        candidate_names=("girard", "scott", "pca", "combastel"),
        length=2,
    )

    _, summary = run_fixed_learning_evaluation(
        config,
        POLICIES,
        model_sha256=MODEL_HASHES,
        source_sha256="source",
        model_directories={name: tmp_path / name for name in MODEL_NAMES},
        workers=2,
    )

    assert calls[:2] == [("reference", "figure8"), ("reference", "square")]
    assert calls[2] == ("pool", 2)
    assert len(summary) == 8


def test_learned_comparison_requires_aligned_cells():
    summary = pd.DataFrame([
        {"trace_kind": "figure8", "budget": 40, "method": "learned"},
        {"trace_kind": "square", "budget": 40, "method": "girard"},
    ])
    with pytest.raises(ValueError, match="do not align"):
        comparison_to_baseline(summary, "learned", "girard")


def test_best_static_is_selected_independently_for_each_metric():
    rows = []
    for method, loss, width in (("girard", 1.0, 5.0), ("scott", 2.0, 3.0)):
        row = {
            "trace_kind": "figure8", "budget": 40, "method": method,
            **{metric: 4.0 for metric in evaluation.COMPARISON_METRICS},
        }
        row["mean_approx_loss"] = loss
        row["mean_state_width"] = width
        rows.append(row)

    result = best_static_metrics(pd.DataFrame(rows), ("girard", "scott"))

    loss = result[result["metric"] == "mean_approx_loss"].iloc[0]
    width = result[result["metric"] == "mean_state_width"].iloc[0]
    assert loss["best_static_method"] == "girard"
    assert width["best_static_method"] == "scott"


def test_best_static_reports_undefined_trigger_metrics_without_dropping_cells():
    rows = []
    for method in ("girard", "scott"):
        row = {
            "trace_kind": "figure8", "budget": 40, "method": method,
            **{metric: 1.0 for metric in evaluation.COMPARISON_METRICS},
        }
        row["fpr"] = float("nan")
        rows.append(row)

    result = best_static_metrics(pd.DataFrame(rows), ("girard", "scott"))

    fpr = result[result["metric"] == "fpr"].iloc[0]
    assert fpr["defined_static_count"] == 0
    assert pd.isna(fpr["best_static_method"])
    assert pd.isna(fpr["best_static_value"])


def test_full_evaluation_matrix_contains_192_validated_cells(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "prepare_reference_cache", lambda _config: None)
    monkeypatch.setattr(
        evaluation, "run_benchmark",
        lambda config: _result(config, config.methods[0]),
    )
    monkeypatch.setattr(
        evaluation, "run_direct_policy_benchmark",
        lambda config, _policy, *, method: _result(config, method),
    )
    monkeypatch.setattr(evaluation, "write_learning_plots", lambda *_args, **_kwargs: None)
    config = FixedLearningEvaluationConfig(
        output=tmp_path,
        model_names=MODEL_NAMES,
        trace_kinds=(
            "figure8", "figure8_drift", "random", "random_drift",
            "square", "square_drift",
        ),
        budgets=(40, 80, 120, 180),
        baselines=("girard", "scott", "pca", "combastel", "mpc_terminal_full_width"),
        candidate_names=("girard", "scott", "pca", "combastel"),
        length=1,
    )

    _, summary = run_fixed_learning_evaluation(
        config, POLICIES, model_sha256=MODEL_HASHES, source_sha256="source",
    )

    assert len(summary) == 192
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["cell_count"] == 192
