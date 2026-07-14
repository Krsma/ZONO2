from types import SimpleNamespace

import pandas as pd
import pytest

import pzr.rtlola.learning_evaluation as evaluation
from pzr.rtlola.learning_evaluation import (
    FixedLearningEvaluationConfig,
    comparison_to_baseline,
    run_fixed_learning_evaluation,
)


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
        model_name="learned_geometry15",
        trace_kinds=("figure8",),
        budgets=(40,),
        baselines=("girard", "mpc_terminal_full_width"),
        candidate_names=("girard", "scott", "pca", "combastel"),
        length=2,
    )

    run_fixed_learning_evaluation(
        config, object(), model_sha256="model", source_sha256="source",
    )
    assert calls.count(("baseline", "girard")) == 1
    assert calls.count(("baseline", "mpc_terminal_full_width")) == 1
    assert calls.count(("learned", "learned_geometry15")) == 1
    calls.clear()
    run_fixed_learning_evaluation(
        config, object(), model_sha256="model", source_sha256="source",
    )
    assert calls == [("reference", "figure8")]
    for filename in (
        "timeseries.csv", "summary.csv", "candidate_selection.csv",
        "decision_accounting.csv", "macro_metrics.csv",
        "micro_trigger_metrics.csv", "learned_comparisons.csv", "manifest.json",
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
        model_name="learned_geometry15",
        trace_kinds=("figure8",),
        budgets=(40,),
        baselines=("girard",),
        candidate_names=("girard", "scott", "pca", "combastel"),
        length=2,
    )
    run_fixed_learning_evaluation(
        config, object(), model_sha256="model", source_sha256="source",
    )

    with pytest.raises(ValueError, match="stale"):
        run_fixed_learning_evaluation(
            config, object(), model_sha256="model", source_sha256="changed",
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
        def __init__(self, *, max_workers, mp_context):
            del mp_context
            calls.append(("pool", max_workers))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def map(self, function, jobs):
            return tuple(function(job) for job in jobs)

    monkeypatch.setattr(evaluation, "ProcessPoolExecutor", ImmediatePool)

    def run_job(job):
        policy = object() if job.method == job.learned_method else None
        evaluation._load_or_run_cell(
            directory=job.directory,
            identity=job.identity,
            benchmark_config=job.benchmark_config,
            method=job.method,
            learned_method=job.learned_method,
            policy=policy,
            expected_length=job.expected_length,
        )

    monkeypatch.setattr(evaluation, "_run_evaluation_cell_job", run_job)
    config = FixedLearningEvaluationConfig(
        output=tmp_path,
        model_name="learned_geometry15",
        trace_kinds=("figure8", "square"),
        budgets=(40,),
        baselines=("girard",),
        candidate_names=("girard", "scott", "pca", "combastel"),
        length=2,
    )

    _, summary = run_fixed_learning_evaluation(
        config,
        object(),
        model_sha256="model",
        source_sha256="source",
        model_directory=tmp_path / "model",
        workers=2,
    )

    assert calls[:2] == [("reference", "figure8"), ("reference", "square")]
    assert calls[2] == ("pool", 2)
    assert len(summary) == 4


def test_learned_comparison_requires_aligned_cells():
    summary = pd.DataFrame([
        {"trace_kind": "figure8", "budget": 40, "method": "learned"},
        {"trace_kind": "square", "budget": 40, "method": "girard"},
    ])
    with pytest.raises(ValueError, match="do not align"):
        comparison_to_baseline(summary, "learned", "girard")
