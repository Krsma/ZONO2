"""Resumable exact evaluation for static, MPC, and learned reducer policies."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import json
from multiprocessing import get_context
from pathlib import Path
import re
from typing import Mapping

import numpy as np
import pandas as pd

from pzr.artifact_io import write_csv_atomic, write_json_atomic
from pzr.learning.provenance import payload_sha256, sha256_files
from pzr.learning.provenance import model_sha256 as compute_model_sha256
from pzr.learning.ranker import ReducerPolicy
from pzr.learning.training import NamedDataset
from pzr.rtlola.policy_reporting import write_policy_reports
from pzr.rtlola.actions import default_action_catalog
from pzr.rtlola.binding import (
    BINDING_BUILD_PROFILE,
    BINDING_REVISION,
    INTERPRETER_REVISION,
)
from pzr.rtlola.benchmark import (
    PREDICTIVE_MPC_METHODS,
    RtlolaBenchmarkConfig,
    prepare_reference_cache,
    run_benchmark,
    run_direct_policy_benchmark,
)
from pzr.rtlola.learned_policy import RtlolaReducerPolicy
from pzr.rtlola.robot_arm import ROBOT_ARM_TRACE_ROWS


POLICY_EVALUATION_SCHEMA = "pzr.policy-evaluation.v2"
@dataclass(frozen=True)
class PolicyComparison:
    name: str
    challenger: str
    reference: str

    def __post_init__(self) -> None:
        values = (self.name, self.challenger, self.reference)
        if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", value) for value in values):
            raise ValueError("comparison names and methods must be filesystem-safe")
        if self.challenger == self.reference:
            raise ValueError("comparison challenger and reference must differ")


@dataclass(frozen=True)
class FixedPolicyEvaluationConfig:
    output: Path
    model_names: tuple[str, ...]
    trace_kinds: tuple[str, ...]
    budgets: tuple[int, ...]
    benchmark_methods: tuple[str, ...]
    candidate_names: tuple[str, ...]
    length: int | None = None
    horizon: int = 1
    beam_width: int = 4
    prediction_step_seconds: float = 0.1
    comparisons: tuple[PolicyComparison, ...] = ()
    expected_cell_count: int | None = None

    def __post_init__(self) -> None:
        if not self.model_names or any(
            not re.fullmatch(r"[A-Za-z0-9_.-]+", name) for name in self.model_names
        ):
            raise ValueError("model names must be non-empty and filesystem-safe")
        if len(set(self.model_names)) != len(self.model_names):
            raise ValueError("learned model names must be unique")
        if set(self.model_names) & set(self.benchmark_methods):
            raise ValueError("learned model names collide with benchmark methods")
        if not self.trace_kinds or not self.budgets or not self.benchmark_methods:
            raise ValueError("evaluation traces, budgets, and benchmark methods must be non-empty")
        if len(set(self.benchmark_methods)) != len(self.benchmark_methods):
            raise ValueError("evaluation benchmark methods must be unique")
        if self.length is not None and self.length < 1:
            raise ValueError("evaluation length must be positive")
        if not np.isfinite(self.prediction_step_seconds) or self.prediction_step_seconds <= 0.0:
            raise ValueError("prediction step seconds must be positive and finite")
        comparison_names = tuple(item.name for item in self.comparisons)
        if len(set(comparison_names)) != len(comparison_names):
            raise ValueError("policy comparison names must be unique")
        available = set(self.model_names)
        for comparison in self.comparisons:
            missing = {comparison.challenger, comparison.reference} - available
            if missing:
                raise ValueError(
                    f"comparison {comparison.name!r} references missing learned models: "
                    f"{sorted(missing)}"
                )
        actual_cell_count = (
            len(self.trace_kinds)
            * len(self.budgets)
            * (len(self.benchmark_methods) + len(self.model_names))
        )
        if self.expected_cell_count is not None and self.expected_cell_count != actual_cell_count:
            raise ValueError(
                f"evaluation matrix has {actual_cell_count} cells, "
                f"expected {self.expected_cell_count}"
            )


def run_fixed_policy_evaluation(
    config: FixedPolicyEvaluationConfig,
    policies: Mapping[str, RtlolaReducerPolicy],
    *,
    model_sha256: Mapping[str, str],
    source_sha256: str,
    model_directories: Mapping[str, Path] | None = None,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run or resume every exact fixed-trace evaluation cell."""
    if workers < 1:
        raise ValueError("evaluation workers must be positive")
    if set(policies) != set(config.model_names) or set(model_sha256) != set(
        config.model_names
    ):
        raise ValueError("named learned policies, hashes, and configuration must align")
    if model_directories is not None and set(model_directories) != set(config.model_names):
        raise ValueError("named model directories and configuration must align")
    if workers > 1 and model_directories is None:
        raise ValueError("parallel learned evaluation requires model directories")
    experiment_fingerprint = _experiment_fingerprint(
        config, model_sha256=model_sha256, source_sha256=source_sha256,
    )
    root_manifest_path = config.output / "manifest.json"
    if root_manifest_path.exists():
        previous = json.loads(root_manifest_path.read_text())
        if previous.get("experiment_fingerprint") != experiment_fingerprint:
            raise ValueError(f"stale policy evaluation output: {config.output}")
    config.output.mkdir(parents=True, exist_ok=True)
    method_names = (*config.benchmark_methods, *config.model_names)
    reference_caches = {}
    for trace_kind in config.trace_kinds:
        length = _trace_length(config, trace_kind)
        reference_cache = (
            config.output / "references" / f"{trace_kind}-length-{length}-exact.json"
        )
        reference_config = _benchmark_config(
            config, trace_kind, length, config.benchmark_methods[0], reference_cache,
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
                    model_sha256=model_sha256.get(method),
                    source_sha256=source_sha256,
                    exact_reference_sha256=sha256_files((reference_cache,)),
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
                    learned_methods=config.model_names,
                    expected_length=length,
                    model_directory=(
                        model_directories[method]
                        if model_directories is not None and method in model_directories
                        else None
                    ),
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
                learned_methods=job.learned_methods,
                policy=policies.get(job.method),
                expected_length=job.expected_length,
            )
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            max_tasks_per_child=1,
        ) as executor:
            tuple(executor.map(_run_evaluation_cell_job, missing))

    completed = [
        _load_or_run_cell(
            directory=job.directory,
            identity=job.identity,
            benchmark_config=job.benchmark_config,
            method=job.method,
            learned_methods=job.learned_methods,
            policy=policies.get(job.method),
            expected_length=job.expected_length,
        )
        for job in jobs
    ]
    completed_timeseries = [item[0] for item in completed]
    completed_summaries = [item[1] for item in completed]
    completed_predictions = [item[2] for item in completed if not item[2].empty]
    timeseries = pd.concat(completed_timeseries, ignore_index=True)
    summary = pd.concat(completed_summaries, ignore_index=True)
    predictions = (
        pd.concat(completed_predictions, ignore_index=True)
        if completed_predictions else pd.DataFrame()
    )
    write_policy_reports(config, timeseries, summary, predictions)
    manifest = {
        "schema": POLICY_EVALUATION_SCHEMA,
        "reference_mode": "exact",
        "full_length": config.length is None,
        "length_override": config.length,
        "trace_kinds": list(config.trace_kinds),
        "budgets": list(config.budgets),
        "candidate_names": list(config.candidate_names),
        "horizon": config.horizon,
        "beam_width": config.beam_width,
        "prediction_step_seconds": config.prediction_step_seconds,
        "prediction_schedule": "current_event_time_plus_fixed_step_multiples",
        "predictors": {
            method: PREDICTIVE_MPC_METHODS[method]
            for method in config.benchmark_methods
            if method in PREDICTIVE_MPC_METHODS
        },
        "benchmark_methods": list(config.benchmark_methods),
        "models": {
            name: {"sha256": model_sha256[name]}
            for name in config.model_names
        },
        "comparisons": [asdict(comparison) for comparison in config.comparisons],
        "pzr_source_sha256": source_sha256,
        "experiment_fingerprint": experiment_fingerprint,
        "cell_count": len(completed_summaries),
        "expected_cell_count": config.expected_cell_count,
        "failure_count": 0,
        "worker_count": workers,
        "binding_revision": BINDING_REVISION,
        "interpreter_revision": INTERPRETER_REVISION,
        "binding_build_profile": BINDING_BUILD_PROFILE,
    }
    write_json_atomic(manifest, root_manifest_path)
    return timeseries, summary


def run_policy_evaluation_from_models(
    config: FixedPolicyEvaluationConfig,
    models: tuple[NamedDataset, ...],
    *,
    source_sha256: str,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load named model artifacts and run the common fixed-policy evaluator."""
    if tuple(model.name for model in models) != config.model_names:
        raise ValueError("named model artifacts and evaluation configuration must align")
    catalog = default_action_catalog(config.candidate_names)
    policies = {
        model.name: RtlolaReducerPolicy(ReducerPolicy.load(model.path), catalog)
        for model in models
    }
    return run_fixed_policy_evaluation(
        config,
        policies,
        model_sha256={model.name: compute_model_sha256(model.path) for model in models},
        source_sha256=source_sha256,
        model_directories={model.name: model.path for model in models},
        workers=workers,
    )


@dataclass(frozen=True)
class _EvaluationCellJob:
    directory: Path
    identity: dict[str, object]
    benchmark_config: RtlolaBenchmarkConfig
    method: str
    learned_methods: tuple[str, ...]
    expected_length: int
    model_directory: Path | None
    candidate_names: tuple[str, ...]


def _run_evaluation_cell_job(job: _EvaluationCellJob) -> None:
    policy = None
    if job.method in job.learned_methods:
        if job.model_directory is None:
            raise ValueError("learned evaluation worker lacks a model directory")
        policy = RtlolaReducerPolicy(
            ReducerPolicy.load(job.model_directory),
            default_action_catalog(job.candidate_names),
        )
    _load_or_run_cell(
        directory=job.directory,
        identity=job.identity,
        benchmark_config=job.benchmark_config,
        method=job.method,
        learned_methods=job.learned_methods,
        policy=policy,
        expected_length=job.expected_length,
    )


def _experiment_fingerprint(
    config: FixedPolicyEvaluationConfig,
    *,
    model_sha256: Mapping[str, str],
    source_sha256: str,
) -> str:
    payload = {
        "config": {
            **asdict(config),
            "output": str(config.output.resolve()),
        },
        "model_sha256": dict(sorted(model_sha256.items())),
        "pzr_source_sha256": source_sha256,
        "binding_revision": BINDING_REVISION,
        "interpreter_revision": INTERPRETER_REVISION,
        "binding_build_profile": BINDING_BUILD_PROFILE,
    }
    return payload_sha256(payload)


def _trace_length(config: FixedPolicyEvaluationConfig, trace_kind: str) -> int:
    authoritative = ROBOT_ARM_TRACE_ROWS[trace_kind]
    if config.length is None:
        return authoritative
    if config.length > authoritative:
        raise ValueError(
            f"evaluation length {config.length} exceeds {trace_kind} length {authoritative}"
        )
    return config.length


def _benchmark_config(
    config: FixedPolicyEvaluationConfig,
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
        prediction_step_seconds=config.prediction_step_seconds,
        seeds=1,
        methods=[method],
        reference_mode="exact",
        reference_cache=str(reference_cache),
        mpc_candidate_names=list(config.candidate_names),
    )


def _cell_identity(
    *,
    config: FixedPolicyEvaluationConfig,
    trace_kind: str,
    length: int,
    budget: int,
    method: str,
    model_sha256: str | None,
    source_sha256: str,
    exact_reference_sha256: str,
) -> dict[str, object]:
    payload = {
        "schema": POLICY_EVALUATION_SCHEMA,
        "trace_kind": trace_kind,
        "length": length,
        "budget": budget,
        "method": method,
        "candidate_names": list(config.candidate_names),
        "horizon": config.horizon,
        "beam_width": config.beam_width,
        "input_predictor": PREDICTIVE_MPC_METHODS.get(method),
        "prediction_step_seconds": config.prediction_step_seconds,
        "prediction_schedule": "current_event_time_plus_fixed_step_multiples",
        "reference_mode": "exact",
        "exact_reference_contract": "trigger_booleans_and_logical_row_center_radius_v1",
        "model_sha256": model_sha256 if method in config.model_names else None,
        "pzr_source_sha256": source_sha256,
        "exact_reference_sha256": exact_reference_sha256,
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
    learned_methods: tuple[str, ...],
    policy: RtlolaReducerPolicy | None,
    expected_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest != identity:
            raise ValueError(f"stale policy evaluation cell: {directory}")
        timeseries = pd.read_csv(directory / "timeseries.csv")
        summary = pd.read_csv(directory / "summary.csv")
        _validate_cell(timeseries, summary, method, expected_length)
        prediction_path = directory / "input_prediction_errors.csv"
        predictions = (
            pd.read_csv(prediction_path) if prediction_path.is_file() else pd.DataFrame()
        )
        return timeseries, summary, predictions

    if method in learned_methods:
        if policy is None:
            raise ValueError("learned evaluation cell lacks a ranking policy")
        result = run_direct_policy_benchmark(
            benchmark_config, policy, method=method,
        )
    else:
        result = run_benchmark(benchmark_config)
    directory.mkdir(parents=True, exist_ok=True)
    if result.failures:
        write_json_atomic(
            [asdict(failure) for failure in result.failures],
            directory / "failures.json",
        )
        raise RuntimeError(f"policy evaluation cell failed: {directory}")
    timeseries = result.timeseries
    summary = result.summary
    _validate_cell(timeseries, summary, method, expected_length)
    write_csv_atomic(timeseries, directory / "timeseries.csv")
    write_csv_atomic(summary, directory / "summary.csv")
    result_predictions = getattr(result, "prediction_diagnostics", pd.DataFrame())
    if not result_predictions.empty:
        write_csv_atomic(
            result_predictions,
            directory / "input_prediction_errors.csv",
        )
    write_json_atomic(identity, manifest_path)
    return timeseries, summary, result_predictions


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
