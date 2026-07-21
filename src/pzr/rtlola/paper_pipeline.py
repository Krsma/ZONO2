"""Staged execution for the versioned terminal-loss paper experiment."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from multiprocessing import get_context
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from pzr.artifact_io import write_csv_atomic, write_json_atomic
from pzr.learning.provenance import model_sha256, pzr_source_sha256, sha256_files
from pzr.learning.ranker import ReducerPolicy
from pzr.learning.training import NamedDataset, ReducerTrainingConfig, run_reducer_training
from pzr.rtlola.actions import default_action_catalog
from pzr.rtlola.benchmark import RtlolaBenchmarkConfig, run_event_trace_benchmark
from pzr.rtlola.engine import RtlolaEvent
from pzr.rtlola.learned_policy import RtlolaReducerPolicy
from pzr.rtlola.learning_collection import LearningCollectionConfig, run_learning_collection
from pzr.rtlola.learning_traces import (
    RandomWaypointTraceStoreConfig,
    generate_random_waypoint_trace_store,
    load_random_waypoint_trace_store,
)
from pzr.rtlola.paper_experiment import (
    GENERALIZATION_METHODS,
    HEADLINE_METHODS,
    OBJECTIVE_METHODS,
    PAPER_CELL_SCHEMA,
    PILOT_METHODS,
    STAGES,
    ExecutionRegime,
    MethodConfig,
    PaperExperimentConfig,
    RunState,
    cell_identity,
    load_json,
    load_paper_experiment_config,
    pilot_projection,
    stage_manifest,
    validate_cell_manifest,
    validate_summary_matrix,
)
from pzr.rtlola.reference import load_or_compute_reference
from pzr.rtlola.robot_arm import ROBOT_ARM_TRACE_SHA256
from pzr.rtlola.scenarios import scenario_by_name


DEFAULT_CONFIG = Path("experiments/terminal_loss_paper_v1.yaml")
LEARNED_METHODS = {
    "pairwise_ranking_policy": "model-all-budgets",
    "pairwise_ranking_policy_budget80": "model-budget80",
}


@dataclass(frozen=True)
class EvaluationTrace:
    trace_id: str
    condition: str
    seed: int
    events: tuple[RtlolaEvent, ...]
    trace_sha256: str


@dataclass(frozen=True)
class EvaluationCellJob:
    stage: str
    directory: Path
    trace: EvaluationTrace
    budget: int
    method: MethodConfig
    runtime_method: str
    reference_path: Path
    identity: dict[str, object]
    model_directory: Path | None


def run_paper_stage(
    config: PaperExperimentConfig,
    stage: str,
    *,
    workers: int | None = None,
    approve_long_run: bool = False,
) -> Path:
    """Run one resumable stage and return its output directory or artifact path."""
    if stage not in STAGES:
        raise ValueError(f"unknown paper stage: {stage}")
    dispatch = {
        "prepare": _run_prepare,
        "train": _run_train,
        "pilot": _run_pilot,
        "objective-comparison": _run_objective_comparison,
        "headline": _run_headline,
        "generalization": lambda cfg, *, workers: _run_generalization(
            cfg, workers=workers, approve_long_run=approve_long_run,
        ),
        "ablation": _run_ablation,
        "timing": _run_timing,
        "report": _run_report,
        "validate": _run_validate,
    }
    if stage in {"pilot", "objective-comparison", "headline", "generalization", "ablation"}:
        selected_workers = config.evaluation_workers if workers is None else workers
        return dispatch[stage](config, workers=selected_workers)  # type: ignore[call-arg]
    return dispatch[stage](config)  # type: ignore[call-arg]


def _run_prepare(config: PaperExperimentConfig) -> Path:
    stage_dir = config.output_root / "prepare"
    trace_root = stage_dir / "traces"
    generate_random_waypoint_trace_store(RandomWaypointTraceStoreConfig(
        output=trace_root / "training",
        event_count=config.event_count,
        conditions=("random_waypoint",),
        seed_start=min(config.train_seeds),
        seed_count=len(config.train_seeds) + len(config.validation_seeds),
    ))
    for name, seeds in (
        ("pilot", config.pilot_seeds),
        ("generalization", config.generalization_seeds),
        ("ablation", config.ablation_seeds),
    ):
        _require_contiguous_seeds(name, seeds)
        generate_random_waypoint_trace_store(RandomWaypointTraceStoreConfig(
            output=trace_root / name,
            event_count=config.event_count,
            conditions=config.conditions,
            seed_start=min(seeds),
            seed_count=len(seeds),
        ))
    dataset = run_learning_collection(LearningCollectionConfig(
        output=stage_dir / "teacher",
        trace_store=trace_root / "training",
        budgets=config.budgets,
        candidate_names=config.candidate_names,
        train_seeds=len(config.train_seeds),
        validation_seeds=len(config.validation_seeds),
        test_seeds=0,
        seed_start=min(config.train_seeds),
        workers=config.teacher_workers,
        collection_mode="teacher",
    ))
    write_json_atomic(stage_manifest(
        config,
        stage="prepare",
        status="completed",
        extra={
            "teacher_dataset": str(dataset),
            "teacher_budgets": list(config.budgets),
            "teacher_seed_count": len(config.train_seeds) + len(config.validation_seeds),
        },
    ), stage_dir / "manifest.json")
    return stage_dir


def _run_train(config: PaperExperimentConfig) -> Path:
    stage_dir = config.output_root / "train"
    dataset = config.output_root / "prepare" / "teacher" / "dataset"
    if not (dataset / "manifest.json").is_file():
        raise ValueError("prepare stage teacher dataset is missing")
    common = dict(
        datasets=(NamedDataset("terminal_full_width_teacher", dataset),),
        objective="pairwise",
        epochs=config.training_epochs,
        batch_size=256,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=10,
        seed=42,
    )
    all_budget = run_reducer_training(ReducerTrainingConfig(
        output=stage_dir / "model-all-budgets",
        budget_filter=None,
        **common,
    ))
    budget80 = run_reducer_training(ReducerTrainingConfig(
        output=stage_dir / "model-budget80",
        budget_filter=(80,),
        **common,
    ))
    write_json_atomic(stage_manifest(
        config,
        stage="train",
        status="completed",
        extra={
            "models": {
                "pairwise_ranking_policy": {
                    "path": str(all_budget), "sha256": model_sha256(all_budget),
                    "budget_filter": None,
                },
                "pairwise_ranking_policy_budget80": {
                    "path": str(budget80), "sha256": model_sha256(budget80),
                    "budget_filter": [80],
                },
            },
        },
    ), stage_dir / "manifest.json")
    return stage_dir


def _run_pilot(config: PaperExperimentConfig, *, workers: int) -> Path:
    traces = _stored_traces(config.output_root / "prepare" / "traces" / "pilot")
    stage_dir = _run_evaluation_matrix(
        config,
        stage="pilot",
        traces=traces,
        budgets=config.pilot_budgets,
        methods=PILOT_METHODS,
        workers=workers,
    )
    summary = pd.read_csv(stage_dir / "summary.csv")
    disk_bytes = sum(
        path.stat().st_size for path in (stage_dir / "cells").rglob("*") if path.is_file()
    )
    projection = pilot_projection(
        summary,
        target_cell_count=config.expected_cells("generalization"),
        worker_count=config.evaluation_workers,
        disk_bytes=disk_bytes,
        threshold_hours=config.maximum_projected_wall_hours,
    )
    write_json_atomic(projection, stage_dir / "projection.json")
    manifest = load_json(stage_dir / "manifest.json")
    manifest["projection"] = projection
    manifest["status"] = (
        "approval_required"
        if projection["approval_required"]
        else (
            "completed_with_failures"
            if int(manifest.get("failure_count", 0)) > 0 else "completed"
        )
    )
    write_json_atomic(manifest, stage_dir / "manifest.json")
    return stage_dir


def _run_generalization(
    config: PaperExperimentConfig,
    *,
    workers: int,
    approve_long_run: bool,
) -> Path:
    projection_path = config.output_root / "pilot" / "projection.json"
    if not projection_path.is_file():
        raise ValueError("pilot projection is required before held-out generalization")
    projection = load_json(projection_path)
    if bool(projection.get("approval_required")) and not approve_long_run:
        raise PermissionError(
            "pilot projects more than 72 four-worker hours; publish the pilot "
            "manifest and rerun with --approve-long-run"
        )
    traces = _stored_traces(
        config.output_root / "prepare" / "traces" / "generalization",
    )
    return _run_evaluation_matrix(
        config,
        stage="generalization",
        traces=traces,
        budgets=config.budgets,
        methods=GENERALIZATION_METHODS,
        workers=workers,
    )


def _run_headline(config: PaperExperimentConfig, *, workers: int) -> Path:
    return _run_evaluation_matrix(
        config,
        stage="headline",
        traces=_fixed_figure8_traces(config),
        budgets=config.budgets,
        methods=HEADLINE_METHODS,
        workers=workers,
    )


def _run_objective_comparison(
    config: PaperExperimentConfig,
    *,
    workers: int,
) -> Path:
    parity_manifest = config.output_root / "parity" / "manifest.json"
    if not parity_manifest.is_file():
        raise ValueError(
            "objective comparison requires a completed notebook-parity manifest at "
            f"{parity_manifest}"
        )
    parity = load_json(parity_manifest)
    if parity.get("status") != "complete" or not parity.get("correctness_passed"):
        raise ValueError("notebook-faithful cumulative parity is not complete")
    return _run_evaluation_matrix(
        config,
        stage="objective-comparison",
        traces=_fixed_figure8_traces(config),
        budgets=config.budgets,
        methods=OBJECTIVE_METHODS,
        workers=workers,
        extra_manifest={"parity_manifest_sha256": sha256_files((parity_manifest,))},
    )


def _run_ablation(config: PaperExperimentConfig, *, workers: int) -> Path:
    base = config.method_by_name["mpc_terminal_beam"]
    methods = tuple(
        replace(
            base,
            name=f"mpc_terminal_beam_h{horizon}_w{width}",
            horizon=horizon,
            beam_width=width,
        )
        for horizon in config.ablation_horizons
        for width in config.ablation_widths
    )
    return _run_evaluation_matrix(
        config,
        stage="ablation",
        traces=_stored_traces(
            config.output_root / "prepare" / "traces" / "ablation",
        ),
        budgets=(config.ablation_budget,),
        methods=tuple(method.name for method in methods),
        workers=workers,
        method_overrides={method.name: method for method in methods},
        runtime_overrides={method.name: "mpc_terminal_beam" for method in methods},
    )


def _run_evaluation_matrix(
    config: PaperExperimentConfig,
    *,
    stage: str,
    traces: Sequence[EvaluationTrace],
    budgets: Sequence[int],
    methods: Sequence[str],
    workers: int,
    method_overrides: Mapping[str, MethodConfig] | None = None,
    runtime_overrides: Mapping[str, str] | None = None,
    extra_manifest: Mapping[str, object] | None = None,
) -> Path:
    if workers < 1:
        raise ValueError("evaluation workers must be positive")
    stage_dir = config.output_root / stage
    references = _prepare_references(config, stage_dir, traces)
    model_paths = _model_paths(config, methods)
    model_hashes = {name: model_sha256(path) for name, path in model_paths.items()}
    source_hash = pzr_source_sha256()
    overrides = dict(method_overrides or {})
    runtime = dict(runtime_overrides or {})
    jobs = []
    for trace in traces:
        reference_path = references[trace.trace_id]
        for budget in budgets:
            for name in methods:
                method = (
                    overrides[name] if name in overrides else config.method_by_name[name]
                )
                identity = cell_identity(
                    config,
                    stage=stage,
                    trace_id=trace.trace_id,
                    trace_sha256=trace.trace_sha256,
                    condition=trace.condition,
                    seed=trace.seed,
                    event_count=len(trace.events),
                    budget=int(budget),
                    method=method,
                    reference_path=reference_path,
                    model_sha256=model_hashes.get(name),
                    source_sha256=source_hash,
                )
                jobs.append(EvaluationCellJob(
                    stage=stage,
                    directory=(
                        stage_dir / "cells" / trace.condition / f"seed-{trace.seed}"
                        / f"budget-{budget}" / name
                    ),
                    trace=trace,
                    budget=int(budget),
                    method=method,
                    runtime_method=runtime.get(name, name),
                    reference_path=reference_path,
                    identity=identity,
                    model_directory=model_paths.get(name),
                ))
    if workers == 1:
        rows = [_execute_cell_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            max_tasks_per_child=1,
        ) as executor:
            rows = list(executor.map(_execute_cell_job, jobs))
    summary = pd.DataFrame(rows)
    if stage == "ablation":
        summary["horizon"] = summary["method"].map(
            {job.method.name: job.method.horizon for job in jobs}
        )
        summary["beam_width"] = summary["method"].map(
            {job.method.name: job.method.beam_width for job in jobs}
        )
    validate_summary_matrix(config, stage, summary)
    write_csv_atomic(summary, stage_dir / "summary.csv")
    series = []
    for job in jobs:
        path = job.directory / "timeseries_diagnostic.csv"
        if path.is_file():
            frame = pd.read_csv(path)
            frame["condition"] = job.trace.condition
            frame["trace_id"] = job.trace.trace_id
            series.append(frame)
    write_csv_atomic(
        pd.concat(series, ignore_index=True) if series else pd.DataFrame(),
        stage_dir / "timeseries.csv",
    )
    failure_count = int((summary["status"] != RunState.COMPLETED.value).sum())
    write_json_atomic(stage_manifest(
        config,
        stage=stage,
        status="completed" if failure_count == 0 else "completed_with_failures",
        cell_count=len(summary),
        failure_count=failure_count,
        extra={
            "expected_cell_count": config.expected_cells(stage),
            "workers": workers,
            "methods": list(methods),
            "budgets": list(budgets),
            **dict(extra_manifest or {}),
        },
    ), stage_dir / "manifest.json")
    return stage_dir


def _execute_cell_job(job: EvaluationCellJob) -> dict[str, object]:
    manifest_path = job.directory / "manifest.json"
    summary_path = job.directory / "summary.csv"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        validate_cell_manifest(manifest, job.identity)
        if not summary_path.is_file():
            raise ValueError(f"cell summary is missing: {job.directory}")
        frame = pd.read_csv(summary_path)
        if len(frame) != 1:
            raise ValueError(f"cell summary has {len(frame)} rows: {job.directory}")
        return frame.iloc[0].to_dict()
    job.directory.mkdir(parents=True, exist_ok=True)
    try:
        row, diagnostic = _run_cell(job)
    except Exception as exc:
        row = _failed_row(job, RunState.INFRASTRUCTURE_FAILED, type(exc).__name__, str(exc))
        diagnostic = {
            "failure_type": type(exc).__name__, "message": str(exc),
        }
    write_csv_atomic(pd.DataFrame([row]), summary_path)
    write_json_atomic({
        "schema": PAPER_CELL_SCHEMA,
        "identity": job.identity,
        "status": row["status"],
        "diagnostic": diagnostic,
    }, manifest_path)
    return row


def _run_cell(job: EvaluationCellJob) -> tuple[dict[str, object], dict[str, object]]:
    policy = None
    if job.model_directory is not None:
        policy = RtlolaReducerPolicy(
            ReducerPolicy.load(job.model_directory),
            default_action_catalog(job.method.candidate_names),
        )
    benchmark = RtlolaBenchmarkConfig(
        scenario="robot_arm",
        trace_kind=job.trace.condition,
        length=len(job.trace.events),
        budget=job.budget,
        horizon=job.method.horizon,
        beam_width=max(1, job.method.beam_width),
        prediction_step_seconds=0.1,
        seeds=1,
        methods=[job.runtime_method],
        reference_mode="exact",
        mpc_reference="rollout",
        reference_cache=str(job.reference_path),
        mpc_candidate_names=list(job.method.candidate_names),
    )
    result = run_event_trace_benchmark(
        benchmark,
        job.trace.events,
        trace_kind=job.trace.trace_id,
        seed=job.trace.seed,
        method=job.runtime_method,
        policy=policy,
    )
    if result.failures:
        failure = result.failures[0]
        partial = result.failed_timeseries.copy()
        if not partial.empty:
            partial["method"] = job.method.name
            partial["condition"] = job.trace.condition
            partial["trace_id"] = job.trace.trace_id
            write_csv_atomic(partial, job.directory / "timeseries_diagnostic.csv")
        elapsed_ms = (
            float(partial["decision_time_ms"].sum()) if not partial.empty else 0.0
        )
        diagnostic = {
            "first_failure_event": failure.step,
            "completed_fraction": len(partial) / len(job.trace.events),
            "pre_failure_mean_loss": (
                float(partial["approx_loss"].mean()) if not partial.empty else None
            ),
            "pre_failure_throughput_events_per_second": (
                len(partial) * 1000.0 / elapsed_ms if elapsed_ms > 0 else None
            ),
        }
        return _failed_row(
            job, RunState.NATIVE_FAILED, failure.failure_type, failure.message,
            first_event=failure.step,
            diagnostic={
                "completed_fraction": diagnostic["completed_fraction"],
                "pre_fallback_mean_loss": diagnostic["pre_failure_mean_loss"],
                "pre_fallback_throughput_events_per_second": diagnostic[
                    "pre_failure_throughput_events_per_second"
                ],
            },
        ), {**asdict(failure), **diagnostic}
    if len(result.summary) != 1:
        raise RuntimeError("paper evaluation cell did not produce exactly one summary")
    timeseries = result.timeseries.copy()
    timeseries["method"] = job.method.name
    timeseries["condition"] = job.trace.condition
    timeseries["trace_id"] = job.trace.trace_id
    write_csv_atomic(timeseries, job.directory / "timeseries_diagnostic.csv")
    fallback_rows = np.flatnonzero(timeseries["fallback_used"].astype(bool).to_numpy())
    if len(fallback_rows):
        first = int(fallback_rows[0])
        prefix = timeseries.iloc[:first]
        elapsed_ms = float(prefix["decision_time_ms"].sum()) if len(prefix) else 0.0
        diagnostic = {
            "first_fallback_event": int(timeseries.iloc[first]["step"]),
            "completed_fraction": first / len(timeseries),
            "pre_fallback_mean_loss": (
                float(prefix["approx_loss"].mean()) if len(prefix) else None
            ),
            "pre_fallback_throughput_events_per_second": (
                len(prefix) * 1000.0 / elapsed_ms if elapsed_ms > 0 else None
            ),
        }
        return _failed_row(
            job,
            RunState.FALLBACK_FAILED,
            "IntervalFallback",
            "ordinary run used interval fallback",
            first_event=int(timeseries.iloc[first]["step"]),
            diagnostic=diagnostic,
        ), diagnostic
    row = result.summary.iloc[0].to_dict()
    row.update(_row_identity(job))
    row.update({
        "status": RunState.COMPLETED.value,
        "event_count": len(job.trace.events),
        "first_fallback_event": np.nan,
        "completed_fraction": 1.0,
        "pre_fallback_mean_loss": np.nan,
        "pre_fallback_throughput_events_per_second": np.nan,
        "failure_type": "",
        "failure_message": "",
    })
    return row, {}


def _failed_row(
    job: EvaluationCellJob,
    state: RunState,
    failure_type: str,
    message: str,
    *,
    first_event: int | None = None,
    diagnostic: Mapping[str, object] | None = None,
) -> dict[str, object]:
    details = dict(diagnostic or {})
    return {
        **_row_identity(job),
        "status": state.value,
        "event_count": len(job.trace.events),
        "false_positive_count": 0,
        "false_negative_count": 0,
        "reference_negative_count": 0,
        "reference_positive_count": 0,
        "fpr": np.nan,
        "fnr": np.nan,
        "mean_approx_loss": np.nan,
        "final_approx_loss": np.nan,
        "max_approx_loss": np.nan,
        "sum_approx_loss": np.nan,
        "mean_state_width": np.nan,
        "max_state_width": np.nan,
        "total_time_ms": np.nan,
        "fallback_count": 1 if state is RunState.FALLBACK_FAILED else 0,
        "infeasible_candidate_count": 0,
        "first_fallback_event": details.get("first_fallback_event", first_event),
        "completed_fraction": details.get("completed_fraction", 0.0),
        "pre_fallback_mean_loss": details.get("pre_fallback_mean_loss", np.nan),
        "pre_fallback_throughput_events_per_second": details.get(
            "pre_fallback_throughput_events_per_second", np.nan,
        ),
        "failure_type": failure_type,
        "failure_message": message,
    }


def _row_identity(job: EvaluationCellJob) -> dict[str, object]:
    return {
        "trace_id": job.trace.trace_id,
        "trace_sha256": job.trace.trace_sha256,
        "trace_kind": job.trace.condition,
        "condition": job.trace.condition,
        "seed": job.trace.seed,
        "budget": job.budget,
        "method": job.method.name,
        "horizon": job.method.horizon,
        "beam_width": job.method.beam_width,
        "cell_fingerprint": job.identity["fingerprint"],
    }


def _prepare_references(
    config: PaperExperimentConfig,
    stage_dir: Path,
    traces: Sequence[EvaluationTrace],
) -> dict[str, Path]:
    scenario = scenario_by_name("robot_arm")
    paths = {}
    for trace in traces:
        path = stage_dir / "references" / f"{_safe(trace.trace_id)}.json"
        load_or_compute_reference(
            trace.events,
            scenario=scenario,
            trace_kind=trace.trace_id,
            seed=trace.seed,
            cache_path=path,
            include_approximation=True,
        )
        paths[trace.trace_id] = path
    return paths


def _model_paths(
    config: PaperExperimentConfig,
    methods: Sequence[str],
) -> dict[str, Path]:
    paths = {
        method: config.output_root / "train" / relative
        for method, relative in LEARNED_METHODS.items()
        if method in methods
    }
    missing = [path for path in paths.values() if not (path / "training.json").is_file()]
    if missing:
        raise ValueError(f"trained paper models are missing: {missing}")
    return paths


def _stored_traces(path: Path) -> tuple[EvaluationTrace, ...]:
    store = load_random_waypoint_trace_store(path)
    return tuple(EvaluationTrace(
        trace_id=item.trace_id,
        condition=item.condition,
        seed=item.seed,
        events=item.trace.events,
        trace_sha256=item.trace.metadata.trace_sha256,
    ) for item in store.traces)


def _fixed_figure8_traces(config: PaperExperimentConfig) -> tuple[EvaluationTrace, ...]:
    scenario = scenario_by_name("robot_arm")
    traces = []
    for condition in config.figure8_conditions:
        generated = scenario.generate_trace(0, 0, trace_kind=condition)
        events = generated.events
        if not config.enforce_canonical_scope:
            events = events[:config.event_count]
        traces.append(EvaluationTrace(
            trace_id=condition,
            condition=condition,
            seed=0,
            events=events,
            trace_sha256=ROBOT_ARM_TRACE_SHA256[condition],
        ))
    return tuple(traces)


def _run_timing(config: PaperExperimentConfig) -> Path:
    """Run contention-free warm-ups and rotated measured repetitions."""
    stage_dir = config.output_root / "timing"
    traces = _fixed_figure8_traces(config)
    methods = HEADLINE_METHODS
    references = _prepare_references(config, stage_dir, traces)
    model_paths = _model_paths(config, methods)
    policies = {
        name: RtlolaReducerPolicy(
            ReducerPolicy.load(path), default_action_catalog(config.candidate_names),
        )
        for name, path in model_paths.items()
    }
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    rows = []
    warm_trace = traces[0]
    for budget in config.budgets:
        for name in methods:
            method = config.method_by_name[name]
            _timed_call(
                warm_trace,
                budget,
                method,
                references[warm_trace.trace_id],
                policies.get(name),
                event_limit=config.timing_warmup_events,
            )
        for condition_index, trace in enumerate(traces):
            for repetition in range(config.timing_repetitions):
                offset = (condition_index + repetition) % len(methods)
                order = (*methods[offset:], *methods[:offset])
                for order_index, name in enumerate(order):
                    method = config.method_by_name[name]
                    elapsed, event_count, status = _timed_call(
                        trace,
                        budget,
                        method,
                        references[trace.trace_id],
                        policies.get(name),
                    )
                    rows.append({
                        "condition": trace.condition,
                        "budget": budget,
                        "method": name,
                        "repetition": repetition,
                        "order_index": order_index,
                        "event_count": event_count,
                        "elapsed_seconds": elapsed,
                        "status": status.value,
                        "throughput_events_per_second": (
                            event_count / elapsed
                            if status is RunState.COMPLETED else np.nan
                        ),
                    })
    raw = pd.DataFrame(rows)
    summary_rows = []
    for keys, frame in raw.groupby(["condition", "budget", "method"], sort=True):
        condition, budget, method = keys
        completed = frame[frame["status"] == RunState.COMPLETED.value]
        available = len(completed) == len(frame)
        values = completed["throughput_events_per_second"]
        summary_rows.append({
            "condition": condition, "budget": budget, "method": method,
            "available": available,
            "valid_count": len(completed), "failed_count": len(frame) - len(completed),
            "median_throughput_events_per_second": (
                float(values.median()) if available else np.nan
            ),
            "valid_only_median_throughput_events_per_second": (
                float(values.median()) if len(values) else np.nan
            ),
            "min_throughput_events_per_second": (
                float(values.min()) if available else np.nan
            ),
            "max_throughput_events_per_second": (
                float(values.max()) if available else np.nan
            ),
        })
    summary = pd.DataFrame(summary_rows)
    write_csv_atomic(raw, stage_dir / "timing_repetitions.csv")
    write_csv_atomic(summary, stage_dir / "summary.csv")
    write_json_atomic(stage_manifest(
        config,
        stage="timing",
        status=(
            "completed" if bool(summary["available"].all())
            else "completed_with_failures"
        ),
        cell_count=len(raw),
        failure_count=int((raw["status"] != RunState.COMPLETED.value).sum()),
        extra={
            "workers": 1,
            "native_threads": 1,
            "warmup_events_per_method_budget": config.timing_warmup_events,
            "measured_repetitions": config.timing_repetitions,
            "included": "event_loop_and_exact_metric_computation",
            "excluded": ["trace_generation", "reference_preparation", "artifact_io"],
            "method_order": "deterministic_rotation_by_condition_and_repetition",
        },
    ), stage_dir / "manifest.json")
    return stage_dir


def _timed_call(
    trace: EvaluationTrace,
    budget: int,
    method: MethodConfig,
    reference_path: Path,
    policy: RtlolaReducerPolicy | None,
    *,
    event_limit: int | None = None,
) -> tuple[float, int, RunState]:
    events = trace.events[:event_limit] if event_limit is not None else trace.events
    # Warm-up prefixes need their own exact cache to keep cache length semantics explicit.
    cache = reference_path
    if len(events) != len(trace.events):
        cache = reference_path.with_name(f"{reference_path.stem}-warmup-{len(events)}.json")
        scenario = scenario_by_name("robot_arm")
        load_or_compute_reference(
            events, scenario=scenario, trace_kind=trace.trace_id,
            seed=trace.seed, cache_path=cache, include_approximation=True,
        )
    scenario = scenario_by_name("robot_arm")
    reference = load_or_compute_reference(
        events,
        scenario=scenario,
        trace_kind=trace.trace_id,
        seed=trace.seed,
        cache_path=cache,
        include_approximation=True,
    )
    benchmark = RtlolaBenchmarkConfig(
        scenario="robot_arm", trace_kind=trace.condition, length=len(events), budget=budget,
        horizon=method.horizon, beam_width=max(1, method.beam_width), seeds=1,
        methods=[method.name], reference_mode="exact", mpc_reference="rollout",
        reference_cache=str(cache), mpc_candidate_names=list(method.candidate_names),
    )
    result = run_event_trace_benchmark(
        benchmark, events, trace_kind=trace.trace_id, seed=trace.seed,
        method=method.name, policy=policy,
        reference_steps=reference,
    )
    if result.failures or result.summary.empty:
        partial_ms = (
            float(result.failed_timeseries["decision_time_ms"].sum())
            if not result.failed_timeseries.empty else np.nan
        )
        return partial_ms / 1000.0, len(events), RunState.NATIVE_FAILED
    elapsed = float(result.summary.iloc[0]["event_loop_time_ms"]) / 1000.0
    if int(result.summary.iloc[0]["fallback_count"]) > 0:
        return elapsed, len(events), RunState.FALLBACK_FAILED
    return elapsed, len(events), RunState.COMPLETED


def _run_report(config: PaperExperimentConfig) -> Path:
    from pzr.rtlola.paper_artifacts import write_terminal_loss_reports

    inputs = {
        stage: config.output_root / stage
        for stage in ("headline", "generalization", "ablation", "timing")
    }
    for stage, path in inputs.items():
        if not (path / "manifest.json").is_file():
            raise ValueError(f"report input stage is missing: {stage}")
    generalization_timeseries = pd.read_csv(inputs["generalization"] / "timeseries.csv")
    output = write_terminal_loss_reports(
        config,
        headline_summary=pd.read_csv(inputs["headline"] / "summary.csv"),
        generalization_summary=pd.read_csv(inputs["generalization"] / "summary.csv"),
        ablation_summary=pd.read_csv(inputs["ablation"] / "summary.csv"),
        timing_summary=pd.read_csv(inputs["timing"] / "summary.csv"),
        composition_timeseries=generalization_timeseries,
    )
    write_json_atomic(stage_manifest(
        config, stage="report", status="completed",
        extra={"artifact_directory": str(output)},
    ), config.output_root / "report" / "manifest.json")
    return output


def _run_validate(config: PaperExperimentConfig) -> Path:
    validations = {}
    evaluation_stages = ["pilot", "headline", "generalization", "ablation"]
    if config.enforce_canonical_scope:
        evaluation_stages.append("objective-comparison")
    current_source_hash = pzr_source_sha256()
    for stage in evaluation_stages:
        directory = config.output_root / stage
        manifest = load_json(directory / "manifest.json")
        if manifest.get("config_sha256") != config.config_sha256:
            raise ValueError(f"stale {stage} stage manifest")
        if manifest.get("pzr_source_sha256") != current_source_hash:
            raise ValueError(f"stale {stage} source manifest")
        summary = pd.read_csv(directory / "summary.csv")
        validate_summary_matrix(config, stage, summary)
        cell_manifests = tuple((directory / "cells").rglob("manifest.json"))
        if len(cell_manifests) != config.expected_cells(stage):
            raise ValueError(f"{stage} cell manifest count differs")
        manifest_statuses = {}
        for path in cell_manifests:
            cell = load_json(path)
            identity = cell.get("identity")
            if not isinstance(identity, dict) or "fingerprint" not in identity:
                raise ValueError(f"invalid cell identity: {path}")
            manifest_statuses[str(identity["fingerprint"])] = str(cell["status"])
        summary_statuses = dict(zip(
            summary["cell_fingerprint"].astype(str), summary["status"].astype(str),
        ))
        if manifest_statuses != summary_statuses:
            raise ValueError(f"{stage} cell manifests and summary differ")
        validations[stage] = {
            "cell_count": len(summary),
            "failure_count": int((summary["status"] != RunState.COMPLETED.value).sum()),
        }
    artifact_manifest = config.paper_artifact_dir / "artifact_hashes.json"
    if not artifact_manifest.is_file():
        raise ValueError("generated paper artifact hash manifest is missing")
    destination = config.output_root / "validate"
    write_json_atomic(stage_manifest(
        config,
        stage="validate",
        status="completed",
        extra={"validated_stages": validations, "artifact_hash_manifest": str(artifact_manifest)},
    ), destination / "manifest.json")
    return destination


def _require_contiguous_seeds(name: str, seeds: Sequence[int]) -> None:
    if tuple(seeds) != tuple(range(min(seeds), min(seeds) + len(seeds))):
        raise ValueError(f"{name} seeds must be contiguous for the trace-store schema")


def _safe(value: str) -> str:
    return value.replace(":", "_").replace("/", "_")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the versioned terminal-loss paper experiment",
    )
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--paper-artifacts", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--smoke", action="store_true",
        help="run the same stage contract with one short trace per scope",
    )
    parser.add_argument(
        "--approve-long-run", action="store_true",
        help="continue the unchanged held-out scope after a >72-hour pilot projection",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_paper_experiment_config(args.config)
    if args.smoke:
        smoke_root = args.output or Path("/tmp/pzr-terminal-loss-paper-smoke")
        config = replace(
            config,
            output_root=smoke_root,
            paper_artifact_dir=(
                args.paper_artifacts or smoke_root / "generated-paper-artifacts"
            ),
            event_count=20,
            budgets=(40, 80),
            conditions=("random_waypoint",),
            figure8_conditions=("figure8",),
            teacher_workers=1,
            evaluation_workers=1,
            training_epochs=2,
            train_seeds=(0,),
            validation_seeds=(1,),
            reserved_exploration_seeds=(26,),
            pilot_seeds=(90,),
            pilot_budgets=(40, 80),
            generalization_seeds=(100,),
            ablation_seeds=(60,),
            ablation_budget=40,
            ablation_horizons=(1,),
            ablation_widths=(1,),
            timing_warmup_events=2,
            timing_repetitions=1,
            enforce_canonical_scope=False,
        )
    if args.output is not None:
        config = replace(config, output_root=args.output)
    if args.paper_artifacts is not None:
        config = replace(config, paper_artifact_dir=args.paper_artifacts)
    output = run_paper_stage(
        config,
        args.stage,
        workers=args.workers,
        approve_long_run=args.approve_long_run,
    )
    print(f"Terminal-loss paper stage complete: {args.stage} -> {output}")


if __name__ == "__main__":
    main()
