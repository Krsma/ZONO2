"""Resumable exact evaluation for direct RTLola ranking policies."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import json
from multiprocessing import get_context
from pathlib import Path
import re

import numpy as np
import pandas as pd

from pzr.learning.provenance import payload_sha256
from pzr.learning.ranker import RankingPolicy
from pzr.learning.reporting import write_learning_plots
from pzr.rtlola.actions import default_action_catalog
from pzr.rtlola.binding import (
    BINDING_BUILD_PROFILE,
    BINDING_REVISION,
    INTERPRETER_REVISION,
)
from pzr.rtlola.benchmark import (
    RtlolaBenchmarkConfig,
    prepare_reference_cache,
    run_benchmark,
    run_direct_policy_benchmark,
)
from pzr.rtlola.learned_policy import RtlolaRankingPolicy
from pzr.rtlola.robot_arm import ROBOT_ARM_TRACE_ROWS


LEARNING_EVALUATION_SCHEMA = "pzr.learning-evaluation.v2"
COMPARISON_METRICS = (
    "fpr",
    "fnr",
    "mean_approx_loss",
    "final_approx_loss",
    "max_approx_loss",
    "sum_approx_loss",
    "mean_state_width",
    "max_state_width",
    "total_time_ms",
)


@dataclass(frozen=True)
class FixedLearningEvaluationConfig:
    output: Path
    model_name: str
    trace_kinds: tuple[str, ...]
    budgets: tuple[int, ...]
    baselines: tuple[str, ...]
    candidate_names: tuple[str, ...]
    length: int | None = None
    horizon: int = 1
    beam_width: int = 4

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.model_name):
            raise ValueError("model name must be filesystem-safe")
        if self.model_name in self.baselines:
            raise ValueError("learned model name collides with a baseline")
        if not self.trace_kinds or not self.budgets or not self.baselines:
            raise ValueError("evaluation traces, budgets, and baselines must be non-empty")
        if len(set(self.baselines)) != len(self.baselines):
            raise ValueError("evaluation baselines must be unique")
        if self.length is not None and self.length < 1:
            raise ValueError("evaluation length must be positive")


def run_fixed_learning_evaluation(
    config: FixedLearningEvaluationConfig,
    policy: RtlolaRankingPolicy,
    *,
    model_sha256: str,
    source_sha256: str,
    model_directory: Path | None = None,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run or resume every exact fixed-trace evaluation cell."""
    if workers < 1:
        raise ValueError("evaluation workers must be positive")
    if workers > 1 and model_directory is None:
        raise ValueError("parallel learned evaluation requires the model directory")
    experiment_fingerprint = _experiment_fingerprint(
        config, model_sha256=model_sha256, source_sha256=source_sha256,
    )
    root_manifest_path = config.output / "manifest.json"
    if root_manifest_path.exists():
        previous = json.loads(root_manifest_path.read_text())
        if previous.get("experiment_fingerprint") != experiment_fingerprint:
            raise ValueError(f"stale learning evaluation output: {config.output}")
    config.output.mkdir(parents=True, exist_ok=True)
    method_names = (*config.baselines, config.model_name)
    reference_caches = {}
    for trace_kind in config.trace_kinds:
        length = _trace_length(config, trace_kind)
        reference_cache = (
            config.output / "references" / f"{trace_kind}-length-{length}-exact.json"
        )
        reference_config = _benchmark_config(
            config, trace_kind, length, config.baselines[0], reference_cache,
        )
        prepare_reference_cache(reference_config)
        reference_caches[trace_kind] = reference_cache

    jobs = []
    for trace_kind in config.trace_kinds:
        length = _trace_length(config, trace_kind)
        reference_cache = reference_caches[trace_kind]
        for budget in config.budgets:
            for method in method_names:
                cell_config = _benchmark_config(
                    config, trace_kind, length, method, reference_cache, budget=budget,
                )
                identity = _cell_identity(
                    config=config,
                    trace_kind=trace_kind,
                    length=length,
                    budget=budget,
                    method=method,
                    model_sha256=model_sha256,
                    source_sha256=source_sha256,
                )
                cell_dir = (
                    config.output / "cells" / trace_kind
                    / f"budget-{budget}" / method
                )
                jobs.append(_EvaluationCellJob(
                    directory=cell_dir,
                    identity=identity,
                    benchmark_config=cell_config,
                    method=method,
                    learned_method=config.model_name,
                    expected_length=length,
                    model_directory=model_directory,
                    candidate_names=config.candidate_names,
                ))
    missing = [job for job in jobs if not (job.directory / "manifest.json").is_file()]
    if not missing:
        pass
    elif workers == 1:
        for job in missing:
            _load_or_run_cell(
                directory=job.directory,
                identity=job.identity,
                benchmark_config=job.benchmark_config,
                method=job.method,
                learned_method=job.learned_method,
                policy=policy,
                expected_length=job.expected_length,
            )
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
        ) as executor:
            tuple(executor.map(_run_evaluation_cell_job, missing))

    completed = [
        _load_or_run_cell(
            directory=job.directory,
            identity=job.identity,
            benchmark_config=job.benchmark_config,
            method=job.method,
            learned_method=job.learned_method,
            policy=policy,
            expected_length=job.expected_length,
        )
        for job in jobs
    ]
    completed_timeseries = [item[0] for item in completed]
    completed_summaries = [item[1] for item in completed]
    timeseries = pd.concat(completed_timeseries, ignore_index=True)
    summary = pd.concat(completed_summaries, ignore_index=True)
    _write_evaluation_reports(config, timeseries, summary)
    manifest = {
        "schema": LEARNING_EVALUATION_SCHEMA,
        "reference_mode": "exact",
        "full_length": config.length is None,
        "length_override": config.length,
        "trace_kinds": list(config.trace_kinds),
        "budgets": list(config.budgets),
        "candidate_names": list(config.candidate_names),
        "baselines": list(config.baselines),
        "model_name": config.model_name,
        "model_sha256": model_sha256,
        "pzr_source_sha256": source_sha256,
        "experiment_fingerprint": experiment_fingerprint,
        "cell_count": len(completed_summaries),
        "failure_count": 0,
        "worker_count": workers,
        "binding_revision": BINDING_REVISION,
        "interpreter_revision": INTERPRETER_REVISION,
        "binding_build_profile": BINDING_BUILD_PROFILE,
    }
    _write_json_atomic(manifest, root_manifest_path)
    return timeseries, summary


@dataclass(frozen=True)
class _EvaluationCellJob:
    directory: Path
    identity: dict[str, object]
    benchmark_config: RtlolaBenchmarkConfig
    method: str
    learned_method: str
    expected_length: int
    model_directory: Path | None
    candidate_names: tuple[str, ...]


def _run_evaluation_cell_job(job: _EvaluationCellJob) -> None:
    policy = None
    if job.method == job.learned_method:
        if job.model_directory is None:
            raise ValueError("learned evaluation worker lacks a model directory")
        policy = RtlolaRankingPolicy(
            RankingPolicy.load(job.model_directory),
            default_action_catalog(job.candidate_names),
        )
    _load_or_run_cell(
        directory=job.directory,
        identity=job.identity,
        benchmark_config=job.benchmark_config,
        method=job.method,
        learned_method=job.learned_method,
        policy=policy,
        expected_length=job.expected_length,
    )


def _experiment_fingerprint(
    config: FixedLearningEvaluationConfig,
    *,
    model_sha256: str,
    source_sha256: str,
) -> str:
    payload = {
        "config": {
            **asdict(config),
            "output": str(config.output.resolve()),
        },
        "model_sha256": model_sha256,
        "pzr_source_sha256": source_sha256,
        "binding_revision": BINDING_REVISION,
        "interpreter_revision": INTERPRETER_REVISION,
        "binding_build_profile": BINDING_BUILD_PROFILE,
    }
    return payload_sha256(payload)


def _trace_length(config: FixedLearningEvaluationConfig, trace_kind: str) -> int:
    authoritative = ROBOT_ARM_TRACE_ROWS[trace_kind]
    if config.length is None:
        return authoritative
    if config.length > authoritative:
        raise ValueError(
            f"evaluation length {config.length} exceeds {trace_kind} length {authoritative}"
        )
    return config.length


def _benchmark_config(
    config: FixedLearningEvaluationConfig,
    trace_kind: str,
    length: int,
    method: str,
    reference_cache: Path,
    *,
    budget: int | None = None,
) -> RtlolaBenchmarkConfig:
    return RtlolaBenchmarkConfig(
        scenario="robot_arm",
        trace_kind=trace_kind,
        length=length,
        budget=config.budgets[0] if budget is None else budget,
        horizon=config.horizon,
        beam_width=config.beam_width,
        seeds=1,
        methods=[method],
        reference_mode="exact",
        reference_cache=str(reference_cache),
        mpc_candidate_names=list(config.candidate_names),
    )


def _cell_identity(
    *,
    config: FixedLearningEvaluationConfig,
    trace_kind: str,
    length: int,
    budget: int,
    method: str,
    model_sha256: str,
    source_sha256: str,
) -> dict[str, object]:
    payload = {
        "schema": LEARNING_EVALUATION_SCHEMA,
        "trace_kind": trace_kind,
        "length": length,
        "budget": budget,
        "method": method,
        "candidate_names": list(config.candidate_names),
        "horizon": config.horizon,
        "beam_width": config.beam_width,
        "reference_mode": "exact",
        "model_sha256": model_sha256 if method == config.model_name else None,
        "pzr_source_sha256": source_sha256,
        "binding_revision": BINDING_REVISION,
        "interpreter_revision": INTERPRETER_REVISION,
        "binding_build_profile": BINDING_BUILD_PROFILE,
    }
    return {**payload, "fingerprint": payload_sha256(payload)}


def _load_or_run_cell(
    *,
    directory: Path,
    identity: dict[str, object],
    benchmark_config: RtlolaBenchmarkConfig,
    method: str,
    learned_method: str,
    policy: RtlolaRankingPolicy | None,
    expected_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest != identity:
            raise ValueError(f"stale learning evaluation cell: {directory}")
        timeseries = pd.read_csv(directory / "timeseries.csv")
        summary = pd.read_csv(directory / "summary.csv")
        _validate_cell(timeseries, summary, method, expected_length)
        return timeseries, summary

    if method == learned_method:
        if policy is None:
            raise ValueError("learned evaluation cell lacks a ranking policy")
        result = run_direct_policy_benchmark(
            benchmark_config, policy, method=learned_method,
        )
    else:
        result = run_benchmark(benchmark_config)
    directory.mkdir(parents=True, exist_ok=True)
    if result.failures:
        _write_json_atomic(
            [asdict(failure) for failure in result.failures],
            directory / "failures.json",
        )
        raise RuntimeError(f"learning evaluation cell failed: {directory}")
    timeseries = result.timeseries
    summary = result.summary
    _validate_cell(timeseries, summary, method, expected_length)
    _write_csv_atomic(timeseries, directory / "timeseries.csv")
    _write_csv_atomic(summary, directory / "summary.csv")
    _write_json_atomic(identity, manifest_path)
    return timeseries, summary


def _validate_cell(
    timeseries: pd.DataFrame,
    summary: pd.DataFrame,
    method: str,
    expected_length: int,
) -> None:
    required_summary = {
        "method", "budget", "trace_kind", "fpr", "fnr",
        "mean_approx_loss", "final_approx_loss", "max_approx_loss",
        "sum_approx_loss", "mean_state_width", "max_state_width",
        "total_time_ms", "fallback_count", "infeasible_candidate_count",
    }
    if len(timeseries) != expected_length:
        raise ValueError(
            f"evaluation cell for {method} has {len(timeseries)} rows, "
            f"expected {expected_length}"
        )
    if len(summary) != 1 or set(summary["method"]) != {method}:
        raise ValueError(f"evaluation cell summary does not identify {method}")
    missing = required_summary - set(summary.columns)
    if missing:
        raise ValueError(f"evaluation cell summary lacks columns: {sorted(missing)}")
    loss_columns = [
        "mean_approx_loss", "final_approx_loss", "max_approx_loss",
        "sum_approx_loss",
    ]
    if not np.isfinite(summary[loss_columns].to_numpy(dtype=np.float64)).all():
        raise ValueError("exact evaluation cell contains non-finite native loss")


def _write_evaluation_reports(
    config: FixedLearningEvaluationConfig,
    timeseries: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    _write_csv_atomic(timeseries, config.output / "timeseries.csv")
    _write_csv_atomic(summary, config.output / "summary.csv")
    _write_csv_atomic(
        candidate_selection(timeseries), config.output / "candidate_selection.csv",
    )
    _write_csv_atomic(
        decision_accounting(timeseries), config.output / "decision_accounting.csv",
    )
    _write_csv_atomic(
        macro_metrics(summary), config.output / "macro_metrics.csv",
    )
    _write_csv_atomic(
        micro_trigger_metrics(summary), config.output / "micro_trigger_metrics.csv",
    )
    comparisons = pd.concat([
        comparison_to_baseline(summary, config.model_name, baseline)
        for baseline in config.baselines
    ], ignore_index=True)
    _write_csv_atomic(comparisons, config.output / "learned_comparisons.csv")
    write_learning_plots(
        timeseries, summary, config.output / "plots", learned_method=config.model_name,
    )


def candidate_selection(timeseries: pd.DataFrame) -> pd.DataFrame:
    data = timeseries.copy()
    data["reduction_required"] = data["pre_generator_count"] > data["budget"]
    groups = [
        "trace_kind", "budget", "method", "reduction_required", "reducer_used",
    ]
    result = data.groupby(groups, dropna=False).size().rename("count").reset_index()
    totals = result.groupby(groups[:-1], dropna=False)["count"].transform("sum")
    result["fraction"] = result["count"] / totals
    return result


def decision_accounting(timeseries: pd.DataFrame) -> pd.DataFrame:
    data = timeseries.copy()
    data["reduction_required"] = data["pre_generator_count"] > data["budget"]
    data["automatic_none"] = ~data["reduction_required"] & (
        data["reducer_used"] == "none"
    )
    data["infeasible_step"] = data["infeasible_candidate_count"] > 0
    rows = []
    for keys, frame in data.groupby(["trace_kind", "budget", "method"], dropna=False):
        trace_kind, budget, method = keys
        required = frame["reduction_required"]
        required_count = int(required.sum())
        rows.append({
            "trace_kind": trace_kind,
            "budget": budget,
            "method": method,
            "step_count": len(frame),
            "reduction_required_count": required_count,
            "reduction_required_rate": float(required.mean()),
            "automatic_none_count": int(frame["automatic_none"].sum()),
            "automatic_none_rate": float(frame["automatic_none"].mean()),
            "fallback_count": int(frame["fallback_used"].sum()),
            "fallback_rate_on_reductions": _conditional_rate(
                frame["fallback_used"], required,
            ),
            "infeasible_candidate_count": int(
                frame["infeasible_candidate_count"].sum()
            ),
            "infeasible_step_count": int(frame["infeasible_step"].sum()),
            "infeasible_step_rate_on_reductions": _conditional_rate(
                frame["infeasible_step"], required,
            ),
        })
    return pd.DataFrame(rows)


def _conditional_rate(values: pd.Series, condition: pd.Series) -> float:
    return float(values[condition].mean()) if bool(condition.any()) else 0.0


def macro_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_approx_loss", "final_approx_loss", "max_approx_loss",
        "sum_approx_loss", "mean_state_width", "max_state_width",
        "total_time_ms",
    ]
    result = summary.groupby(["method", "budget"], as_index=False)[metrics].mean()
    result.insert(2, "aggregation", "macro_trace_mean")
    return result


def micro_trigger_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "false_positive_count", "false_negative_count",
        "reference_positive_count", "reference_negative_count",
    ]
    result = summary.groupby(["method", "budget"], as_index=False)[columns].sum()
    result["fpr"] = result["false_positive_count"] / result[
        "reference_negative_count"
    ].replace(0, np.nan)
    result["fnr"] = result["false_negative_count"] / result[
        "reference_positive_count"
    ].replace(0, np.nan)
    result.insert(2, "aggregation", "micro_trigger_counts")
    return result


def comparison_to_baseline(
    summary: pd.DataFrame,
    learned_method: str,
    baseline: str,
) -> pd.DataFrame:
    keys = ["trace_kind", "budget"]
    learned = summary[summary["method"] == learned_method].set_index(keys)
    reference = summary[summary["method"] == baseline].set_index(keys)
    if set(learned.index) != set(reference.index):
        raise ValueError(f"learned and {baseline} evaluation cells do not align")
    rows = []
    for key in sorted(learned.index):
        trace_kind, budget = key
        for metric in COMPARISON_METRICS:
            learned_value = float(learned.loc[key, metric])
            baseline_value = float(reference.loc[key, metric])
            rows.append({
                "trace_kind": trace_kind,
                "budget": budget,
                "learned_method": learned_method,
                "baseline": baseline,
                "metric": metric,
                "learned_value": learned_value,
                "baseline_value": baseline_value,
                "difference": learned_value - baseline_value,
                "ratio": (
                    learned_value / baseline_value
                    if baseline_value != 0.0 else float("nan")
                ),
            })
    return pd.DataFrame(rows)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_json_atomic(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)
