"""Staged execution for the versioned paper evaluation."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from typing import IO, Iterator, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from pzr.artifact_io import write_csv_atomic, write_json_atomic
from pzr.learning.provenance import (
    model_sha256,
    payload_sha256,
    pzr_source_sha256,
    sha256_files,
)
from pzr.learning.ranker import ReducerPolicy
from pzr.learning.training import (
    NamedDataset,
    ReducerTrainingConfig,
    dataset_sha256,
    run_reducer_training,
)
from pzr.rtlola.actions import default_action_catalog
from pzr.rtlola.benchmark import RtlolaBenchmarkConfig, run_event_trace_benchmark
from pzr.rtlola.binding import (
    BINDING_BUILD_PROFILE,
    BINDING_REVISION,
    INTERPRETER_REVISION,
)
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
    PAPER_RUN_SCHEMA,
    PAPER_STAGE_SCHEMA,
    PILOT_METHODS,
    STAGES,
    ExecutionRegime,
    MethodConfig,
    PaperExperimentConfig,
    RunState,
    TraceSource,
    cell_identity,
    load_json,
    load_paper_experiment_config,
    pilot_projection,
    stage_manifest,
    validate_cell_manifest,
    validate_summary_matrix,
)
from pzr.rtlola.reference import REFERENCE_CACHE_SCHEMA, load_or_compute_reference
from pzr.rtlola.robot_arm import (
    RLOLAEVAL_REVISION,
    ROBOT_ARM_SPEC_SHA256,
    ROBOT_ARM_TRACE_ROWS,
    ROBOT_ARM_TRACE_SHA256,
    trace_path,
)
from pzr.rtlola.robot_arm_random import RANDOM_WAYPOINT_SOURCE_REVISION
from pzr.rtlola.scenarios import scenario_by_name


DEFAULT_CONFIG = Path("experiments/paper_evaluation_v2.yaml")
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RUN_EXIT_COMPLETE = 0
RUN_EXIT_FAILED_POINTS = 2
RUN_EXIT_APPROVAL_REQUIRED = 75
RUN_EXIT_PRIMARY_READINESS_FAILED = 76
PAPER_PREFLIGHT_MARKER_EXPRESSION = "not rlola_parity"
SCIENTIFIC_STAGES = (
    "pilot", "objective-comparison", "headline", "generalization", "ablation",
)
PRIMARY_EVALUATION_STAGES = (
    "headline", "objective-comparison", "generalization", "ablation",
)
LEARNED_METHODS = {"pairwise_ranking_policy"}


@dataclass(frozen=True)
class EvaluationTrace:
    trace_id: str
    condition: str
    seed: int
    events: tuple[RtlolaEvent, ...]
    trace_sha256: str
    trace_source: TraceSource
    trace_kind: str
    provenance: Mapping[str, object]


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
    model_training_budget: int | None = None


@dataclass(frozen=True)
class PaperRunResult:
    status: str
    exit_code: int
    failure_count: int
    manifest: Path


class _Tee:
    def __init__(self, *streams: IO[str]) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


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
        "science-report": _run_science_report,
        "science-validate": _run_science_validate,
        "report": _run_report,
        "validate": _run_validate,
    }
    if stage in {"pilot", "objective-comparison", "headline", "generalization", "ablation"}:
        default_workers = (
            config.ablation_workers if stage == "ablation" else config.evaluation_workers
        )
        selected_workers = default_workers if workers is None else workers
        if stage == "ablation" and selected_workers != config.ablation_workers:
            raise ValueError("paper ablation must use its configured single worker")
        return dispatch[stage](config, workers=selected_workers)  # type: ignore[call-arg]
    return dispatch[stage](config)  # type: ignore[call-arg]


def run_complete_paper_evaluation(
    config: PaperExperimentConfig,
    *,
    approve_long_run: bool = False,
    smoke: bool = False,
) -> PaperRunResult:
    """Run or resume every prerequisite and primary paper-evaluation stage."""
    projection_path = config.output_root / "pilot" / "projection.json"
    pilot_existed_at_start = projection_path.is_file()
    if approve_long_run and not pilot_existed_at_start:
        raise ValueError(
            "long-run approval is accepted only after a pilot projection exists"
        )
    config.output_root.mkdir(parents=True, exist_ok=True)
    provenance = _run_provenance(config)
    if provenance["dirty_source_paths"] and not smoke:
        raise ValueError(
            "paper evaluation requires clean scientific sources; dirty paths: "
            f"{provenance['dirty_source_paths']}"
        )
    _write_run_manifest(
        config, status="running", failure_count=0,
        extra={"provenance": provenance, "approval_recorded": False},
    )

    _run_preflight(config, provenance)
    for stage in ("prepare", "train", "pilot"):
        _run_or_skip_stage(config, stage)

    primary_failure_count = _pilot_primary_failure_count(config)
    if primary_failure_count:
        manifest = _write_run_manifest(
            config,
            status="primary_readiness_failed",
            failure_count=primary_failure_count,
            extra={
                "provenance": provenance,
                "approval_recorded": False,
            },
        )
        return PaperRunResult(
            "primary_readiness_failed",
            RUN_EXIT_PRIMARY_READINESS_FAILED,
            primary_failure_count,
            manifest,
        )

    projection = load_json(projection_path)
    _validate_projection(config, projection)
    approval_required = bool(projection.get("approval_required"))
    approval_path = config.output_root / "pilot" / "approval.json"
    if approval_required and not approve_long_run:
        manifest = _write_run_manifest(
            config,
            status="approval_required",
            failure_count=_scientific_failure_count(config),
            extra={
                "provenance": provenance,
                "projection": projection,
                "approval_recorded": False,
            },
        )
        return PaperRunResult(
            "approval_required", RUN_EXIT_APPROVAL_REQUIRED,
            _scientific_failure_count(config), manifest,
        )
    if approval_required:
        write_json_atomic({
            "schema": "pzr.paper-evaluation-approval.v3",
            "approved": True,
            "config_sha256": config.config_sha256,
            "pzr_source_sha256": pzr_source_sha256(),
            "projection_sha256": sha256_files((projection_path,)),
            "recorded_at": _utc_now(),
        }, approval_path)

    for stage in (
        "headline", "objective-comparison", "generalization", "ablation",
        "timing", "report", "validate",
    ):
        _run_or_skip_stage(
            config,
            stage,
            approve_long_run=approval_required,
        )
    failure_count = _scientific_failure_count(config)
    status = "completed" if failure_count == 0 else "completed_with_failures"
    manifest = _write_run_manifest(
        config,
        status=status,
        failure_count=failure_count,
        extra={
            "provenance": provenance,
            "projection": projection,
            "approval_recorded": approval_path.is_file(),
            "artifact_directory": str(config.paper_artifact_dir),
        },
    )
    return PaperRunResult(
        status,
        RUN_EXIT_COMPLETE if failure_count == 0 else RUN_EXIT_FAILED_POINTS,
        failure_count,
        manifest,
    )


def run_exploratory_bundle(
    config: PaperExperimentConfig,
    *,
    smoke: bool = False,
) -> PaperRunResult:
    """Reuse/prepare teacher data, train budget specialists, and run the pilot."""
    config.output_root.mkdir(parents=True, exist_ok=True)
    provenance = _run_provenance(config)
    if provenance["dirty_source_paths"] and not smoke:
        raise ValueError(
            "exploratory bundle requires clean scientific sources; dirty paths: "
            f"{provenance['dirty_source_paths']}"
        )
    _run_preflight(config, provenance)
    for stage in ("prepare", "train", "pilot"):
        _run_or_skip_stage(config, stage)

    projection = load_json(config.output_root / "pilot" / "projection.json")
    _validate_projection(config, projection)
    failure_count = _stage_failure_count(config, "pilot")
    status = (
        "exploration_completed"
        if failure_count == 0 else "exploration_completed_with_failures"
    )
    manifest = config.output_root / "explore" / "manifest.json"
    write_json_atomic({
        "schema": "pzr.paper-evaluation-exploration.v3",
        "experiment_id": config.experiment_id,
        "status": status,
        "updated_at": _utc_now(),
        "config_sha256": config.config_sha256,
        "pzr_source_sha256": pzr_source_sha256(),
        **_runtime_provenance(),
        "provenance": provenance,
        "failure_count": failure_count,
        "projection": projection,
        "included_stages": ["preflight", "prepare", "train", "pilot"],
        "excluded_stages": [
            "objective-comparison", "headline", "generalization", "ablation",
            "timing", "report", "validate",
            "bounded-exploration",
        ],
    }, manifest)
    return PaperRunResult(
        status,
        RUN_EXIT_COMPLETE if failure_count == 0 else RUN_EXIT_FAILED_POINTS,
        failure_count,
        manifest,
    )


def run_scientific_paper_evaluation(
    config: PaperExperimentConfig,
    *,
    approve_long_run: bool = False,
    smoke: bool = False,
) -> PaperRunResult:
    """Run the paper's scientific cells and reports, deferring timing."""
    projection_path = config.output_root / "pilot" / "projection.json"
    if approve_long_run and not projection_path.is_file():
        raise ValueError(
            "long-run approval is accepted only after a pilot projection exists"
        )
    config.output_root.mkdir(parents=True, exist_ok=True)
    provenance = _run_provenance(config)
    if provenance["dirty_source_paths"] and not smoke:
        raise ValueError(
            "paper evaluation requires clean scientific sources; dirty paths: "
            f"{provenance['dirty_source_paths']}"
        )
    _run_preflight(config, provenance)
    for stage in ("prepare", "train", "pilot"):
        _run_or_skip_stage(config, stage)
        _require_no_hard_failures(config, stage)

    projection = load_json(projection_path)
    _validate_projection(config, projection)
    primary_failure_count = _pilot_primary_failure_count(config)
    if primary_failure_count:
        manifest = _write_evaluate_manifest(
            config,
            status="primary_readiness_failed",
            failure_count=primary_failure_count,
            extra={"provenance": provenance, "projection": projection},
        )
        return PaperRunResult(
            "primary_readiness_failed",
            RUN_EXIT_PRIMARY_READINESS_FAILED,
            primary_failure_count,
            manifest,
        )

    # These fixed-scope/ablation stages are independent of the held-out gate.
    for stage in ("headline", "objective-comparison", "ablation"):
        _run_or_skip_stage(config, stage)
        _require_no_hard_failures(config, stage)

    approval_required = bool(projection.get("approval_required"))
    approval_path = config.output_root / "pilot" / "approval.json"
    if approval_required and not approve_long_run:
        manifest = _write_evaluate_manifest(
            config,
            status="approval_required",
            failure_count=_scientific_failure_count(config),
            extra={
                "provenance": provenance,
                "projection": projection,
                "completed_ungated_stages": [
                    "headline", "objective-comparison", "ablation",
                ],
                "gated_pending_stage": "generalization",
            },
        )
        return PaperRunResult(
            "approval_required",
            RUN_EXIT_APPROVAL_REQUIRED,
            _scientific_failure_count(config),
            manifest,
        )
    if approval_required:
        write_json_atomic({
            "schema": "pzr.paper-evaluation-approval.v3",
            "approved": True,
            "config_sha256": config.config_sha256,
            "pzr_source_sha256": pzr_source_sha256(),
            "projection_sha256": sha256_files((projection_path,)),
            "recorded_at": _utc_now(),
        }, approval_path)

    for stage in ("generalization", "science-report", "science-validate"):
        _run_or_skip_stage(
            config,
            stage,
            approve_long_run=approval_required,
        )
        if stage == "generalization":
            _require_no_hard_failures(config, stage)
    failure_count = _scientific_failure_count(config)
    status = "completed" if failure_count == 0 else "completed_with_failures"
    manifest = _write_evaluate_manifest(
        config,
        status=status,
        failure_count=failure_count,
        extra={
            "provenance": provenance,
            "projection": projection,
            "timing_deferred": True,
            "science_artifact_directory": str(
                config.output_root / "science-report" / "artifacts"
            ),
        },
    )
    return PaperRunResult(
        status,
        RUN_EXIT_COMPLETE if failure_count == 0 else RUN_EXIT_FAILED_POINTS,
        failure_count,
        manifest,
    )


def _require_no_hard_failures(config: PaperExperimentConfig, stage: str) -> None:
    if stage not in SCIENTIFIC_STAGES:
        return
    summary = pd.read_csv(config.output_root / stage / "summary.csv")
    hard = summary[summary["status"].isin((
        RunState.NATIVE_FAILED.value,
        RunState.INFRASTRUCTURE_FAILED.value,
    ))]
    if not hard.empty:
        raise RuntimeError(f"{stage} contains {len(hard)} native/infrastructure failures")


def _pilot_primary_failure_count(config: PaperExperimentConfig) -> int:
    summary = pd.read_csv(config.output_root / "pilot" / "summary.csv")
    primary = summary[summary["method"] == "pairwise_ranking_policy"]
    expected = len(config.pilot_seeds) * len(config.pilot_budgets)
    if len(primary) != expected:
        raise ValueError(
            f"pilot has {len(primary)} primary-policy cells, expected {expected}"
        )
    return int((primary["status"] != RunState.COMPLETED.value).sum())


def _write_evaluate_manifest(
    config: PaperExperimentConfig,
    *,
    status: str,
    failure_count: int,
    extra: Mapping[str, object],
) -> Path:
    path = config.output_root / "evaluate" / "manifest.json"
    write_json_atomic({
        "schema": "pzr.paper-evaluation-scientific-run.v3",
        "experiment_id": config.experiment_id,
        "status": status,
        "updated_at": _utc_now(),
        "config_sha256": config.config_sha256,
        "pzr_source_sha256": pzr_source_sha256(),
        **_runtime_provenance(),
        "failure_count": failure_count,
        "expected_prerequisite_pilot_cells": config.expected_cells("pilot"),
        "expected_main_scientific_manifest_cells": sum(
            config.expected_cells(stage) for stage in PRIMARY_EVALUATION_STAGES
        ),
        "expected_new_main_scientific_executions": (
            sum(config.expected_cells(stage) for stage in PRIMARY_EVALUATION_STAGES)
            - len(config.fixed_figure8_trace_kinds) * len(config.budgets)
        ),
        "timing_included": False,
        **dict(extra),
    }, path)
    return path


def paper_evaluation_status(config: PaperExperimentConfig) -> dict[str, object]:
    """Return a non-mutating summary of the current paper-evaluation output."""
    stages: dict[str, object] = {}
    for stage in STAGES:
        manifest_path = config.output_root / stage / "manifest.json"
        if not manifest_path.is_file():
            stages[stage] = {"status": "missing"}
            continue
        try:
            _validate_completed_stage(config, stage)
            manifest = load_json(manifest_path)
            stages[stage] = {
                "status": str(manifest.get("status", "unknown")),
                "cell_count": manifest.get("cell_count"),
                "failure_count": manifest.get("failure_count"),
            }
        except (OSError, ValueError) as exc:
            stages[stage] = {"status": "stale_or_invalid", "message": str(exc)}
    projection_path = config.output_root / "pilot" / "projection.json"
    run_manifest = config.output_root / "run" / "manifest.json"
    exploration_manifest = config.output_root / "explore" / "manifest.json"
    evaluate_manifest = config.output_root / "evaluate" / "manifest.json"
    exploration: dict[str, object] = {"status": "missing"}
    if exploration_manifest.is_file():
        try:
            exploration = load_json(exploration_manifest)
            if exploration.get("schema") != "pzr.paper-evaluation-exploration.v3":
                raise ValueError("unsupported exploration manifest schema")
            if exploration.get("config_sha256") != config.config_sha256:
                raise ValueError("stale exploration config manifest")
            if exploration.get("pzr_source_sha256") != pzr_source_sha256():
                raise ValueError("stale exploration source manifest")
            _validate_runtime_provenance(exploration, "exploration")
        except (OSError, ValueError) as exc:
            exploration = {"status": "stale_or_invalid", "message": str(exc)}
    return {
        "schema": "pzr.paper-evaluation-status.v3",
        "experiment_id": config.experiment_id,
        "output_root": str(config.output_root),
        "paper_artifact_dir": str(config.paper_artifact_dir),
        "run": _validated_auxiliary_manifest(
            run_manifest, PAPER_RUN_SCHEMA, config, "run",
        ),
        "exploration": exploration,
        "scientific_evaluation": _validated_auxiliary_manifest(
            evaluate_manifest,
            "pzr.paper-evaluation-scientific-run.v3",
            config,
            "scientific evaluation",
        ),
        "projection": _validated_auxiliary_manifest(
            projection_path, "pzr.paper-evaluation-pilot-projection.v3", config,
            "pilot projection",
        ),
        "approval_recorded": _approval_is_valid(config),
        "stages": stages,
    }


def _run_or_skip_stage(
    config: PaperExperimentConfig,
    stage: str,
    *,
    approve_long_run: bool = False,
) -> None:
    manifest = config.output_root / stage / "manifest.json"
    if manifest.is_file():
        _validate_completed_stage(config, stage)
        print(f"skip validated stage: {stage}", flush=True)
        return
    with _stage_log(config, stage):
        run_paper_stage(
            config,
            stage,
            approve_long_run=approve_long_run,
        )
    _validate_completed_stage(config, stage)


def _validate_completed_stage(config: PaperExperimentConfig, stage: str) -> None:
    manifest_path = config.output_root / stage / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema") != PAPER_STAGE_SCHEMA:
        raise ValueError(f"unsupported {stage} stage manifest schema")
    if manifest.get("config_sha256") != config.config_sha256:
        raise ValueError(f"stale {stage} config manifest")
    if manifest.get("pzr_source_sha256") != pzr_source_sha256():
        raise ValueError(f"stale {stage} source manifest")
    _validate_runtime_provenance(manifest, stage)
    if stage == "prepare":
        dataset = config.output_root / "prepare" / "teacher" / "dataset"
        if not (dataset / "manifest.json").is_file():
            raise ValueError("prepare teacher dataset is missing")
        if dataset_sha256(dataset) != manifest.get("teacher_dataset_sha256"):
            raise ValueError("prepare teacher dataset hash differs")
    elif stage == "train":
        models = manifest.get("models_by_budget")
        if not isinstance(models, dict):
            raise ValueError("train manifest lacks models")
        if set(models) != {str(budget) for budget in config.budgets}:
            raise ValueError("train manifest specialist budgets differ")
        expected_hyperparameters = {
            "epochs": config.training_epochs,
            "batch_size": config.training_batch_size,
            "learning_rate": config.training_learning_rate,
            "weight_decay": config.training_weight_decay,
            "patience": config.training_patience,
            "seed": config.training_seed,
        }
        if manifest.get("shared_hyperparameters") != expected_hyperparameters:
            raise ValueError("train manifest shared hyperparameters differ")
        for budget_text, payload in models.items():
            if not isinstance(payload, dict):
                raise ValueError("train manifest model entry is invalid")
            budget = int(budget_text)
            if payload.get("budget_filter") != [budget]:
                raise ValueError("train manifest specialist filter differs")
            path = Path(str(payload["path"]))
            if model_sha256(path) != payload.get("sha256"):
                raise ValueError(f"trained model hash differs: {path}")
    elif stage in SCIENTIFIC_STAGES:
        _validate_scientific_stage(config, stage)
    elif stage == "timing":
        _validate_timing_stage(config, manifest)
    elif stage in {"science-report", "report"}:
        artifact_dir = (
            config.output_root / "science-report" / "artifacts"
            if stage == "science-report" else config.paper_artifact_dir
        )
        hashes_path = artifact_dir / "artifact_hashes.json"
        report_path = artifact_dir / "report_manifest.json"
        if not hashes_path.is_file():
            raise ValueError("paper artifact hashes are missing")
        if load_json(hashes_path).get("schema") != "pzr.paper-generated-artifact-hashes.v3":
            raise ValueError("unsupported paper artifact hash schema")
        report = load_json(report_path)
        if report.get("schema") != "pzr.paper-evaluation-report.v3":
            raise ValueError("unsupported paper report schema")
        if report.get("config_sha256") != config.config_sha256:
            raise ValueError("stale paper report config")
    elif stage in {"science-validate", "validate"}:
        if manifest.get("status") not in {"completed", "completed_with_failures"}:
            raise ValueError("validation stage did not complete")


def _validate_scientific_stage(config: PaperExperimentConfig, stage: str) -> None:
    directory = config.output_root / stage
    summary = pd.read_csv(directory / "summary.csv")
    validate_summary_matrix(config, stage, summary)
    cell_manifests = tuple((directory / "cells").rglob("manifest.json"))
    if len(cell_manifests) != config.expected_cells(stage):
        raise ValueError(f"{stage} cell manifest count differs")


def _run_preflight(
    config: PaperExperimentConfig,
    provenance: Mapping[str, object],
) -> None:
    from pzr.rtlola.binding import require_binding

    require_binding()
    __import__("mujoco")
    stage_dir = config.output_root / "preflight"
    manifest_path = stage_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        if manifest.get("schema") != "pzr.paper-evaluation-preflight.v2":
            raise ValueError("unsupported preflight manifest schema")
        if (
            manifest.get("config_sha256") != config.config_sha256
            or manifest.get("pzr_source_sha256") != pzr_source_sha256()
        ):
            raise ValueError("stale preflight manifest")
        _validate_runtime_provenance(manifest, "preflight")
        if (
            manifest.get("status") != "completed"
            or manifest.get("skipped") != 0
            or manifest.get("pytest_marker_expression")
            != PAPER_PREFLIGHT_MARKER_EXPRESSION
        ):
            raise ValueError("preflight manifest is incomplete")
        print("skip validated stage: preflight", flush=True)
        return
    stage_dir.mkdir(parents=True, exist_ok=True)
    junit = stage_dir / "pytest.xml"
    log = stage_dir / "pytest.log"
    command = _paper_preflight_command(junit)
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log.write_text(completed.stdout)
    print(completed.stdout, end="", flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"release validation failed; see {log}")
    counts = _junit_counts(junit)
    if counts["failures"] or counts["errors"] or counts["skipped"]:
        raise RuntimeError(f"release validation was not pass-only: {counts}")
    write_json_atomic({
        "schema": "pzr.paper-evaluation-preflight.v2",
        "status": "completed",
        "config_sha256": config.config_sha256,
        "pzr_source_sha256": pzr_source_sha256(),
        **_runtime_provenance(),
        "tests": counts["tests"],
        "failures": counts["failures"],
        "errors": counts["errors"],
        "skipped": counts["skipped"],
        "pytest_marker_expression": PAPER_PREFLIGHT_MARKER_EXPRESSION,
        "command": command,
        "provenance": dict(provenance),
    }, manifest_path)


def _paper_preflight_command(junit: Path) -> list[str]:
    """Return the paper preflight command, excluding standalone parity checks."""
    return [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-m",
        PAPER_PREFLIGHT_MARKER_EXPRESSION,
        f"--junitxml={junit}",
    ]


def _scientific_failure_count(config: PaperExperimentConfig) -> int:
    count = 0
    for stage in SCIENTIFIC_STAGES:
        summary = config.output_root / stage / "summary.csv"
        if summary.is_file():
            frame = pd.read_csv(summary)
            count += int((frame["status"] != RunState.COMPLETED.value).sum())
    timing_manifest = config.output_root / "timing" / "manifest.json"
    if timing_manifest.is_file():
        count += int(load_json(timing_manifest).get("failure_count", 0))
    return count


def _stage_failure_count(config: PaperExperimentConfig, stage: str) -> int:
    summary = config.output_root / stage / "summary.csv"
    if not summary.is_file():
        raise ValueError(f"{stage} summary is missing")
    frame = pd.read_csv(summary)
    return int((frame["status"] != RunState.COMPLETED.value).sum())


def _run_provenance(config: PaperExperimentConfig) -> dict[str, object]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        [
            "git", "status", "--porcelain", "--untracked-files=all", "--",
            "src/pzr", "experiments", "rlolapythonbinding", "pyproject.toml",
            "tools/run_paper_evaluation.sh",
        ],
        cwd=REPOSITORY_ROOT, text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    return {
        "git_revision": revision,
        "dirty_source_paths": dirty,
        "config_sha256": config.config_sha256,
        "pzr_source_sha256": pzr_source_sha256(),
        **_runtime_provenance(),
    }


def _runtime_provenance() -> dict[str, str]:
    return {
        "spec_sha256": ROBOT_ARM_SPEC_SHA256,
        "rlolaeval_revision": RLOLAEVAL_REVISION,
        "binding_revision": BINDING_REVISION,
        "interpreter_revision": INTERPRETER_REVISION,
        "binding_build_profile": BINDING_BUILD_PROFILE,
        "reference_cache_schema": REFERENCE_CACHE_SCHEMA,
    }


def _validate_runtime_provenance(
    manifest: Mapping[str, object],
    label: str,
) -> None:
    for key, expected in _runtime_provenance().items():
        if manifest.get(key) != expected:
            raise ValueError(f"stale {label} {key}")


def _validate_projection(
    config: PaperExperimentConfig,
    projection: Mapping[str, object],
) -> None:
    if projection.get("schema") != "pzr.paper-evaluation-pilot-projection.v3":
        raise ValueError("unsupported pilot projection schema")
    if projection.get("config_sha256") != config.config_sha256:
        raise ValueError("stale pilot projection config")
    if projection.get("pzr_source_sha256") != pzr_source_sha256():
        raise ValueError("stale pilot projection source")
    if projection.get("gated_stage") != "generalization":
        raise ValueError("pilot projection gate has the wrong stage")
    if projection.get("trace_scope") != TraceSource.GENERATED_NOMINAL.value:
        raise ValueError("pilot projection gate has the wrong trace scope")
    if int(projection.get("target_cell_count", -1)) != config.expected_cells(
        "generalization"
    ):
        raise ValueError("pilot projection target count differs")


def _validated_auxiliary_manifest(
    path: Path,
    schema: str,
    config: PaperExperimentConfig,
    label: str,
) -> dict[str, object] | None:
    if not path.is_file():
        return {"status": "missing"} if label == "run" else None
    try:
        payload = load_json(path)
        if payload.get("schema") != schema:
            raise ValueError(f"unsupported {label} schema")
        if payload.get("config_sha256") != config.config_sha256:
            raise ValueError(f"stale {label} config")
        if payload.get("pzr_source_sha256") != pzr_source_sha256():
            raise ValueError(f"stale {label} source")
        if label == "pilot projection":
            _validate_projection(config, payload)
        return payload
    except (OSError, ValueError) as exc:
        return {"status": "stale_or_invalid", "message": str(exc)}


def _approval_is_valid(config: PaperExperimentConfig) -> bool:
    path = config.output_root / "pilot" / "approval.json"
    if not path.is_file():
        return False
    try:
        payload = load_json(path)
        return (
            payload.get("schema") == "pzr.paper-evaluation-approval.v3"
            and payload.get("config_sha256") == config.config_sha256
            and payload.get("pzr_source_sha256") == pzr_source_sha256()
        )
    except (OSError, ValueError):
        return False


def _validate_timing_stage(
    config: PaperExperimentConfig,
    manifest: Mapping[str, object],
) -> None:
    directory = config.output_root / "timing"
    raw = pd.read_csv(directory / "timing_repetitions.csv")
    summary = pd.read_csv(directory / "summary.csv")
    raw_keys = [
        "trace_source", "trace_kind", "condition", "budget", "method",
        "repetition",
    ]
    summary_keys = [
        "trace_source", "trace_kind", "condition", "budget", "method",
    ]
    if len(raw) != config.expected_cells("timing"):
        raise ValueError("timing measured repetition count differs")
    if len(summary) != config.expected_timing_summary_points:
        raise ValueError("timing summary point count differs")
    if bool(raw.duplicated(raw_keys).any()) or bool(summary.duplicated(summary_keys).any()):
        raise ValueError("timing contains duplicate identities")
    expected_summary = {
        (TraceSource.FIXED_RLOLAEVAL.value, condition, condition, budget, method)
        for condition in config.fixed_figure8_trace_kinds
        for budget in config.budgets
        for method in HEADLINE_METHODS
    }
    actual_summary = set(summary[summary_keys].itertuples(index=False, name=None))
    if actual_summary != expected_summary:
        raise ValueError("timing summary identities differ")
    expected_raw = {
        (*identity, repetition)
        for identity in expected_summary
        for repetition in range(config.timing_repetitions)
    }
    actual_raw = set(raw[raw_keys].itertuples(index=False, name=None))
    if actual_raw != expected_raw:
        raise ValueError("timing repetition identities differ")
    if int(manifest.get("cell_count", -1)) != config.expected_cells("timing"):
        raise ValueError("timing manifest measured count differs")
    if int(manifest.get("warmup_count", -1)) != config.expected_timing_warmups:
        raise ValueError("timing manifest warm-up count differs")


def _write_run_manifest(
    config: PaperExperimentConfig,
    *,
    status: str,
    failure_count: int,
    extra: Mapping[str, object],
) -> Path:
    path = config.output_root / "run" / "manifest.json"
    write_json_atomic({
        "schema": PAPER_RUN_SCHEMA,
        "experiment_id": config.experiment_id,
        "status": status,
        "updated_at": _utc_now(),
        "config_sha256": config.config_sha256,
        "pzr_source_sha256": pzr_source_sha256(),
        "failure_count": failure_count,
        "expected_scientific_cell_count": sum(
            config.expected_cells(stage) for stage in SCIENTIFIC_STAGES
        ),
        "expected_timing_measured_repetition_count": config.expected_cells("timing"),
        "expected_timing_summary_point_count": config.expected_timing_summary_points,
        "expected_timing_warmup_count": config.expected_timing_warmups,
        **dict(extra),
    }, path)
    return path


@contextmanager
def _stage_log(config: PaperExperimentConfig, stage: str) -> Iterator[None]:
    log_path = config.output_root / "logs" / f"{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        tee_out = _Tee(sys.stdout, stream)
        tee_err = _Tee(sys.stderr, stream)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print(f"start stage: {stage}", flush=True)
            yield
            print(f"complete stage: {stage}", flush=True)


def _junit_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        name: sum(int(suite.attrib.get(name, 0)) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_prepare(config: PaperExperimentConfig) -> Path:
    stage_dir = config.output_root / "prepare"
    parent = config.teacher_dataset_parent
    if parent is not None and parent.is_dir():
        _validate_teacher_dataset_parent(config, parent)
        dataset = stage_dir / "teacher" / "dataset"
        dataset.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(parent, dataset, dirs_exist_ok=True)
        if dataset_sha256(dataset) != config.teacher_dataset_parent_sha256:
            raise ValueError("copied teacher dataset hash differs")
        source_manifest = load_json(parent / "manifest.json")
        write_json_atomic(stage_manifest(
            config,
            stage="prepare",
            status="completed",
            extra={
                "teacher_dataset": str(dataset),
                "teacher_dataset_sha256": dataset_sha256(dataset),
                "teacher_dataset_origin": "reused_verified_parent",
                "teacher_dataset_parent": str(parent),
                "teacher_dataset_parent_pzr_source_sha256": source_manifest.get(
                    "pzr_source_sha256"
                ),
                "teacher_budgets": list(config.budgets),
                "teacher_seed_count": len(config.train_seeds)
                + len(config.validation_seeds),
                "trace_scope": TraceSource.GENERATED_NOMINAL.value,
                "trace_kind": config.generated_nominal_trace_kind,
                "training_trace_store": source_manifest.get("trace_store"),
                "training_trace_store_manifest_sha256": source_manifest.get(
                    "trace_store_manifest_sha256"
                ),
            },
        ), stage_dir / "manifest.json")
        return stage_dir

    trace_root = stage_dir / "traces"
    training_store = generate_random_waypoint_trace_store(RandomWaypointTraceStoreConfig(
        output=trace_root / "training",
        event_count=config.event_count,
        conditions=(config.generated_nominal_trace_kind,),
        seed_start=min(config.train_seeds),
        seed_count=len(config.train_seeds) + len(config.validation_seeds),
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
            "teacher_dataset_sha256": dataset_sha256(dataset),
            "teacher_dataset_origin": "collected",
            "teacher_budgets": list(config.budgets),
            "teacher_seed_count": len(config.train_seeds) + len(config.validation_seeds),
            "trace_scope": TraceSource.GENERATED_NOMINAL.value,
            "trace_kind": config.generated_nominal_trace_kind,
            "training_trace_store": str(training_store.root),
            "training_trace_store_manifest_sha256": training_store.manifest_sha256,
        },
    ), stage_dir / "manifest.json")
    return stage_dir


def _validate_teacher_dataset_parent(
    config: PaperExperimentConfig,
    parent: Path,
) -> None:
    """Validate the immutable v1 teacher payload by its scientific contract."""
    actual_hash = dataset_sha256(parent)
    if actual_hash != config.teacher_dataset_parent_sha256:
        raise ValueError("teacher dataset parent hash differs")
    manifest = load_json(parent / "manifest.json")
    expected = {
        "schema": "pzr.reducer-cost-dataset.v5",
        "scenario": "robot_arm",
        "event_count": config.event_count,
        "budgets": list(config.budgets),
        "candidate_names": list(config.candidate_names),
        "conditions": [config.generated_nominal_trace_kind],
        "binding_revision": BINDING_REVISION,
        "interpreter_revision": INTERPRETER_REVISION,
        "binding_build_profile": BINDING_BUILD_PROFILE,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"teacher dataset parent {key} differs")
    cost = manifest.get("cost_contract")
    if not isinstance(cost, dict) or cost.get("teacher_cost") != (
        "binding_native_two_event_full_width_terminal_cost"
    ):
        raise ValueError("teacher dataset parent objective differs")
    feature = manifest.get("feature_schema")
    if not isinstance(feature, dict) or (
        feature.get("name"), feature.get("version")
    ) != ("rtlola.current-zonotope", 2):
        raise ValueError("teacher dataset parent feature schema differs")
    traces = manifest.get("traces")
    if not isinstance(traces, list):
        raise ValueError("teacher dataset parent trace list is missing")
    expected_split = {
        **{seed: "train" for seed in config.train_seeds},
        **{seed: "validation" for seed in config.validation_seeds},
    }
    actual_split = {int(item["seed"]): str(item["split"]) for item in traces}
    if actual_split != expected_split:
        raise ValueError("teacher dataset parent seed splits differ")


def _run_train(config: PaperExperimentConfig) -> Path:
    stage_dir = config.output_root / "train"
    dataset = config.output_root / "prepare" / "teacher" / "dataset"
    if not (dataset / "manifest.json").is_file():
        raise ValueError("prepare stage teacher dataset is missing")
    common = dict(
        datasets=(NamedDataset("terminal_full_width_teacher", dataset),),
        objective="pairwise",
        epochs=config.training_epochs,
        batch_size=config.training_batch_size,
        learning_rate=config.training_learning_rate,
        weight_decay=config.training_weight_decay,
        patience=config.training_patience,
        seed=config.training_seed,
    )
    models = {}
    for budget in config.budgets:
        output = run_reducer_training(ReducerTrainingConfig(
            output=stage_dir / f"model-budget-{budget}",
            budget_filter=(budget,),
            **common,
        ))
        models[str(budget)] = {
            "path": str(output),
            "sha256": model_sha256(output),
            "budget_filter": [budget],
            "training_budget": budget,
        }
    write_json_atomic(stage_manifest(
        config,
        stage="train",
        status="completed",
        extra={
            "models_by_budget": models,
            "shared_hyperparameters": {
                "epochs": config.training_epochs,
                "batch_size": config.training_batch_size,
                "learning_rate": config.training_learning_rate,
                "weight_decay": config.training_weight_decay,
                "patience": config.training_patience,
                "seed": config.training_seed,
            },
        },
    ), stage_dir / "manifest.json")
    return stage_dir


def _run_pilot(config: PaperExperimentConfig, *, workers: int) -> Path:
    traces = _generated_nominal_stage_traces(config, "pilot", config.pilot_seeds)
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
    pilot_manifest = load_json(stage_dir / "manifest.json")
    projection = pilot_projection(
        summary,
        target_cell_count=config.expected_cells("generalization"),
        worker_count=config.evaluation_workers,
        disk_bytes=disk_bytes,
        threshold_hours=config.maximum_projected_wall_hours,
        observed_pilot_wall_seconds=float(pilot_manifest["matrix_wall_seconds"]),
        separate_fixed_workloads={
            "headline_cells": config.expected_cells("headline"),
            "objective_comparison_cells": config.expected_cells("objective-comparison"),
            "objective_comparison_new_executions": (
                len(config.fixed_figure8_trace_kinds) * len(config.budgets)
            ),
            "timing_measured_repetitions": config.expected_cells("timing"),
            "timing_warmups": config.expected_timing_warmups,
        },
    )
    projection["schema"] = "pzr.paper-evaluation-pilot-projection.v3"
    projection["config_sha256"] = config.config_sha256
    projection["pzr_source_sha256"] = pzr_source_sha256()
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
    _validate_projection(config, projection)
    if bool(projection.get("approval_required")) and not approve_long_run:
        raise PermissionError(
            "pilot projects more than 72 four-worker hours; publish the pilot "
            "manifest and rerun with --approve-long-run"
        )
    traces = _generated_nominal_stage_traces(
        config, "generalization", config.generalization_seeds,
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
    headline_dir = config.output_root / "headline"
    _validate_completed_stage(config, "headline")
    traces = _fixed_figure8_traces(config)
    stage_dir = _run_evaluation_matrix(
        config,
        stage="objective-comparison",
        traces=traces,
        budgets=config.budgets,
        methods=("mpc_cumulative_beam",),
        workers=workers,
        finalize=False,
    )
    cumulative = pd.read_csv(stage_dir / "summary.csv")
    cumulative["execution_origin"] = "executed_objective_comparison"
    headline = pd.read_csv(headline_dir / "summary.csv")
    terminal = headline[headline["method"] == "mpc_terminal_beam"].copy()
    references = _prepare_references(config, stage_dir, traces)
    trace_by_condition = {trace.condition: trace for trace in traces}
    reused_rows = []
    for row in terminal.to_dict("records"):
        trace = trace_by_condition[str(row["condition"])]
        budget = int(row["budget"])
        method = config.method_by_name["mpc_terminal_beam"]
        identity = cell_identity(
            config,
            stage="objective-comparison",
            trace_id=trace.trace_id,
            trace_sha256=trace.trace_sha256,
            trace_source=trace.trace_source,
            trace_kind=trace.trace_kind,
            trace_provenance=trace.provenance,
            condition=trace.condition,
            seed=trace.seed,
            event_count=len(trace.events),
            budget=budget,
            method=method,
            reference_path=references[trace.trace_id],
            model_sha256=None,
        )
        source_dir = (
            headline_dir / "cells" / trace.trace_source.value / trace.condition
            / f"seed-{trace.seed}" / f"budget-{budget}" / method.name
        )
        source_manifest_path = source_dir / "manifest.json"
        source_manifest = load_json(source_manifest_path)
        source_identity = source_manifest.get("identity")
        if not isinstance(source_identity, dict):
            raise ValueError(f"invalid headline source cell: {source_dir}")
        for key in (
            "trace_id", "trace_sha256", "trace_source", "trace_kind", "condition",
            "seed", "event_count", "budget", "method", "model_sha256",
            "model_training_budget", "spec_sha256", "binding_revision",
            "interpreter_revision", "binding_build_profile", "reference_semantics",
            "reference_cache_sha256",
        ):
            if source_identity.get(key) != identity.get(key):
                raise ValueError(f"headline terminal reuse differs at {key}: {source_dir}")
        target_dir = (
            stage_dir / "cells" / trace.trace_source.value / trace.condition
            / f"seed-{trace.seed}" / f"budget-{budget}" / method.name
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        source_series = source_dir / "timeseries_diagnostic.csv"
        if source_series.is_file():
            shutil.copy2(source_series, target_dir / source_series.name)
        reused = dict(row)
        reused["cell_fingerprint"] = identity["fingerprint"]
        reused["cell_elapsed_ms"] = 0.0
        reused["execution_origin"] = "reused_headline_terminal"
        reused["source_cell_fingerprint"] = source_identity["fingerprint"]
        reused_rows.append(reused)
        write_csv_atomic(pd.DataFrame([reused]), target_dir / "summary.csv")
        write_json_atomic({
            "schema": PAPER_CELL_SCHEMA,
            "identity": identity,
            "status": reused["status"],
            "cell_elapsed_ms": 0.0,
            "diagnostic": {
                "execution_origin": "reused_headline_terminal",
                "source_stage": "headline",
                "source_cell_manifest": str(source_manifest_path),
                "source_cell_manifest_sha256": sha256_files((source_manifest_path,)),
            },
        }, target_dir / "manifest.json")
    combined = pd.concat((cumulative, pd.DataFrame(reused_rows)), ignore_index=True)
    validate_summary_matrix(config, "objective-comparison", combined)
    write_csv_atomic(combined, stage_dir / "summary.csv")
    series = [pd.read_csv(path) for path in (stage_dir / "cells").rglob(
        "timeseries_diagnostic.csv"
    )]
    write_csv_atomic(pd.concat(series, ignore_index=True), stage_dir / "timeseries.csv")
    failure_count = int((combined["status"] != RunState.COMPLETED.value).sum())
    write_json_atomic(stage_manifest(
        config,
        stage="objective-comparison",
        status="completed" if failure_count == 0 else "completed_with_failures",
        cell_count=len(combined),
        failure_count=failure_count,
        extra={
            "expected_cell_count": config.expected_cells("objective-comparison"),
            "executed_cell_count": len(cumulative),
            "reused_cell_count": len(reused_rows),
            "reuse_source_stage": "headline",
            "reuse_source_manifest_sha256": sha256_files((headline_dir / "manifest.json",)),
            "methods": list(OBJECTIVE_METHODS),
            "budgets": list(config.budgets),
            "workers": workers,
        },
    ), stage_dir / "manifest.json")
    return stage_dir


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
        traces=_generated_nominal_stage_traces(
            config, "ablation", config.ablation_seeds,
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
    finalize: bool = True,
) -> Path:
    if workers < 1:
        raise ValueError("evaluation workers must be positive")
    stage_dir = config.output_root / stage
    references = _prepare_references(config, stage_dir, traces)
    source_hash = pzr_source_sha256()
    overrides = dict(method_overrides or {})
    runtime = dict(runtime_overrides or {})
    jobs = []
    for trace in traces:
        reference_path = references[trace.trace_id]
        for budget in budgets:
            for name in methods:
                model_path = _model_path(config, name, int(budget))
                model_hash = model_sha256(model_path) if model_path is not None else None
                model_training_budget = int(budget) if model_path is not None else None
                method = (
                    overrides[name] if name in overrides else config.method_by_name[name]
                )
                identity = cell_identity(
                    config,
                    stage=stage,
                    trace_id=trace.trace_id,
                    trace_sha256=trace.trace_sha256,
                    trace_source=trace.trace_source,
                    trace_kind=trace.trace_kind,
                    trace_provenance=trace.provenance,
                    condition=trace.condition,
                    seed=trace.seed,
                    event_count=len(trace.events),
                    budget=int(budget),
                    method=method,
                    reference_path=reference_path,
                    model_sha256=model_hash,
                    model_training_budget=model_training_budget,
                    source_sha256=source_hash,
                )
                jobs.append(EvaluationCellJob(
                    stage=stage,
                    directory=(
                        stage_dir / "cells" / trace.trace_source.value
                        / trace.condition / f"seed-{trace.seed}"
                        / f"budget-{budget}" / name
                    ),
                    trace=trace,
                    budget=int(budget),
                    method=method,
                    runtime_method=runtime.get(name, name),
                    reference_path=reference_path,
                    identity=identity,
                    model_directory=model_path,
                    model_training_budget=model_training_budget,
                ))
    matrix_started = perf_counter()
    if workers == 1:
        rows = [_execute_cell_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            max_tasks_per_child=1,
        ) as executor:
            rows = list(executor.map(_execute_cell_job, jobs))
    matrix_wall_seconds = perf_counter() - matrix_started
    summary = pd.DataFrame(rows)
    if stage == "ablation":
        summary["horizon"] = summary["method"].map(
            {job.method.name: job.method.horizon for job in jobs}
        )
        summary["beam_width"] = summary["method"].map(
            {job.method.name: job.method.beam_width for job in jobs}
        )
    if finalize:
        validate_summary_matrix(config, stage, summary)
    write_csv_atomic(summary, stage_dir / "summary.csv")
    series = []
    for job in jobs:
        path = job.directory / "timeseries_diagnostic.csv"
        if path.is_file():
            frame = pd.read_csv(path)
            frame["condition"] = job.trace.condition
            frame["trace_source"] = job.trace.trace_source.value
            frame["trace_kind"] = job.trace.trace_kind
            frame["trace_id"] = job.trace.trace_id
            series.append(frame)
    write_csv_atomic(
        pd.concat(series, ignore_index=True) if series else pd.DataFrame(columns=(
            "trace_source", "trace_kind", "condition", "trace_id", "budget",
            "method", "reducer_used", "fallback_used",
            "infeasible_candidate_count",
        )),
        stage_dir / "timeseries.csv",
    )
    failure_count = int((summary["status"] != RunState.COMPLETED.value).sum())
    if not finalize:
        return stage_dir
    write_json_atomic(stage_manifest(
        config,
        stage=stage,
        status="completed" if failure_count == 0 else "completed_with_failures",
        cell_count=len(summary),
        failure_count=failure_count,
        extra={
            "expected_cell_count": config.expected_cells(stage),
            "workers": workers,
            "matrix_wall_seconds": matrix_wall_seconds,
            "methods": list(methods),
            "budgets": list(budgets),
            "trace_sources": sorted({trace.trace_source.value for trace in traces}),
            "trace_kinds": sorted({trace.trace_kind for trace in traces}),
            "trace_manifest": [
                {
                    "trace_id": trace.trace_id,
                    "trace_source": trace.trace_source.value,
                    "trace_kind": trace.trace_kind,
                    "condition": trace.condition,
                    "seed": trace.seed,
                    "event_count": len(trace.events),
                    "trace_sha256": trace.trace_sha256,
                    "provenance": dict(trace.provenance),
                }
                for trace in traces
            ],
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
    started = perf_counter()
    try:
        row, diagnostic = _run_cell(job)
    except Exception as exc:
        row = _failed_row(job, RunState.INFRASTRUCTURE_FAILED, type(exc).__name__, str(exc))
        diagnostic = {
            "failure_type": type(exc).__name__, "message": str(exc),
        }
    cell_elapsed_ms = (perf_counter() - started) * 1000.0
    row["cell_elapsed_ms"] = cell_elapsed_ms
    write_csv_atomic(pd.DataFrame([row]), summary_path)
    write_json_atomic({
        "schema": PAPER_CELL_SCHEMA,
        "identity": job.identity,
        "status": row["status"],
        "cell_elapsed_ms": cell_elapsed_ms,
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
            partial["trace_source"] = job.trace.trace_source.value
            partial["trace_kind"] = job.trace.trace_kind
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
    timeseries["trace_source"] = job.trace.trace_source.value
    timeseries["trace_kind"] = job.trace.trace_kind
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
        "trace_source": job.trace.trace_source.value,
        "trace_kind": job.trace.trace_kind,
        "condition": job.trace.condition,
        "seed": job.trace.seed,
        "budget": job.budget,
        "method": job.method.name,
        "horizon": job.method.horizon,
        "beam_width": job.method.beam_width,
        "model_training_budget": job.model_training_budget,
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


def _model_path(
    config: PaperExperimentConfig,
    method: str,
    budget: int,
) -> Path | None:
    if method not in LEARNED_METHODS:
        return None
    if budget not in config.budgets:
        raise ValueError(f"learned policy has no configured budget: {budget}")
    path = config.output_root / "train" / f"model-budget-{budget}"
    training_path = path / "training.json"
    if not training_path.is_file():
        raise ValueError(f"trained paper specialist is missing: {path}")
    training = load_json(training_path)
    if training.get("budget_filter") != [budget]:
        raise ValueError(f"trained specialist budget differs: {path}")
    return path


def _generated_nominal_stage_traces(
    config: PaperExperimentConfig,
    stage: str,
    seeds: Sequence[int],
) -> tuple[EvaluationTrace, ...]:
    """Create or validate a stage-owned nominal random-waypoint trace store."""
    _require_contiguous_seeds(stage, seeds)
    path = config.output_root / stage / "traces" / "generated-nominal"
    generate_random_waypoint_trace_store(RandomWaypointTraceStoreConfig(
        output=path,
        event_count=config.event_count,
        conditions=(config.generated_nominal_trace_kind,),
        seed_start=min(seeds),
        seed_count=len(seeds),
    ))
    return _stored_traces(path, expected_seeds=seeds, config=config)


def _stored_traces(
    path: Path,
    *,
    expected_seeds: Sequence[int],
    config: PaperExperimentConfig,
) -> tuple[EvaluationTrace, ...]:
    store = load_random_waypoint_trace_store(path)
    if store.conditions != (config.generated_nominal_trace_kind,):
        raise ValueError("paper generated trace store must be nominal-only")
    if tuple(range(store.seed_start, store.seed_start + store.seed_count)) != tuple(
        expected_seeds
    ):
        raise ValueError("paper generated trace-store seed coverage differs")
    return tuple(EvaluationTrace(
        trace_id=item.trace_id,
        condition=item.condition,
        seed=item.seed,
        events=item.trace.events,
        trace_sha256=item.trace.metadata.trace_sha256,
        trace_source=TraceSource.GENERATED_NOMINAL,
        trace_kind=item.condition,
        provenance={
            "source_revision": RANDOM_WAYPOINT_SOURCE_REVISION,
            "trace_store_manifest_sha256": store.manifest_sha256,
            "generator_config_sha256": payload_sha256(
                item.trace.metadata.generator_config,
            ),
            "generator_config": item.trace.metadata.generator_config,
        },
    ) for item in store.traces)


def _fixed_figure8_traces(config: PaperExperimentConfig) -> tuple[EvaluationTrace, ...]:
    scenario = scenario_by_name("robot_arm")
    traces = []
    for condition in config.fixed_figure8_trace_kinds:
        source_path = trace_path(condition)
        actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        expected_hash = ROBOT_ARM_TRACE_SHA256[condition]
        if actual_hash != expected_hash:
            raise ValueError(f"fixed RLolaEval trace hash differs: {condition}")
        generated = scenario.generate_trace(0, 0, trace_kind=condition)
        full_events = generated.events
        expected_length = ROBOT_ARM_TRACE_ROWS[condition]
        if len(full_events) != expected_length:
            raise ValueError(
                f"fixed RLolaEval trace length differs: {condition} has "
                f"{len(full_events)}, expected {expected_length}"
            )
        events = full_events
        if not config.enforce_canonical_scope:
            events = events[:config.event_count]
        traces.append(EvaluationTrace(
            trace_id=condition,
            condition=condition,
            seed=0,
            events=events,
            trace_sha256=actual_hash,
            trace_source=TraceSource.FIXED_RLOLAEVAL,
            trace_kind=condition,
            provenance={
                "source_revision": RLOLAEVAL_REVISION,
                "source_path": str(source_path.relative_to(REPOSITORY_ROOT)),
                "source_file_sha256": actual_hash,
                "source_event_count": expected_length,
                "evaluation_event_count": len(events),
            },
        ))
    return tuple(traces)


def _run_timing(config: PaperExperimentConfig) -> Path:
    """Run contention-free warm-ups and rotated measured repetitions."""
    stage_dir = config.output_root / "timing"
    traces = _fixed_figure8_traces(config)
    methods = HEADLINE_METHODS
    references = _prepare_references(config, stage_dir, traces)
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    rows = []
    warm_trace = traces[0]
    for budget in config.budgets:
        policies = {}
        for name in methods:
            model_path = _model_path(config, name, budget)
            if model_path is not None:
                policies[name] = RtlolaReducerPolicy(
                    ReducerPolicy.load(model_path),
                    default_action_catalog(config.candidate_names),
                )
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
                        "trace_source": trace.trace_source.value,
                        "trace_kind": trace.trace_kind,
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
            "trace_source": TraceSource.FIXED_RLOLAEVAL.value,
            "trace_kind": condition,
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
            "expected_measured_repetition_count": config.expected_cells("timing"),
            "expected_summary_point_count": config.expected_timing_summary_points,
            "warmup_count": config.expected_timing_warmups,
            "trace_source": TraceSource.FIXED_RLOLAEVAL.value,
            "trace_manifest": [
                {
                    "trace_id": trace.trace_id,
                    "trace_source": trace.trace_source.value,
                    "trace_kind": trace.trace_kind,
                    "condition": trace.condition,
                    "seed": trace.seed,
                    "event_count": len(trace.events),
                    "trace_sha256": trace.trace_sha256,
                    "provenance": dict(trace.provenance),
                }
                for trace in traces
            ],
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


def _run_science_report(config: PaperExperimentConfig) -> Path:
    return _run_report_common(config, include_timing=False)


def _run_report(config: PaperExperimentConfig) -> Path:
    return _run_report_common(config, include_timing=True)


def _run_report_common(
    config: PaperExperimentConfig,
    *,
    include_timing: bool,
) -> Path:
    from pzr.rtlola.paper_artifacts import write_paper_evaluation_reports

    inputs = {
        stage: config.output_root / stage
        for stage in (
            "pilot", "objective-comparison", "headline", "generalization",
            "ablation",
        )
    }
    if include_timing:
        inputs["timing"] = config.output_root / "timing"
    for stage, path in inputs.items():
        if not (path / "manifest.json").is_file():
            raise ValueError(f"report input stage is missing: {stage}")
    generalization_timeseries = pd.read_csv(inputs["generalization"] / "timeseries.csv")
    headline_timeseries = pd.read_csv(inputs["headline"] / "timeseries.csv")
    output = write_paper_evaluation_reports(
        config,
        headline_summary=pd.read_csv(inputs["headline"] / "summary.csv"),
        generalization_summary=pd.read_csv(inputs["generalization"] / "summary.csv"),
        objective_summary=pd.read_csv(inputs["objective-comparison"] / "summary.csv"),
        ablation_summary=pd.read_csv(inputs["ablation"] / "summary.csv"),
        timing_summary=(
            pd.read_csv(inputs["timing"] / "summary.csv")
            if include_timing else None
        ),
        nominal_composition_timeseries=generalization_timeseries,
        fixed_composition_timeseries=headline_timeseries,
        pilot_projection=load_json(inputs["pilot"] / "projection.json"),
        output=(
            None
            if include_timing
            else config.output_root / "science-report" / "artifacts"
        ),
    )
    stage = "report" if include_timing else "science-report"
    write_json_atomic(stage_manifest(
        config, stage=stage, status="completed",
        extra={
            "artifact_directory": str(output),
            "timing_included": include_timing,
        },
    ), config.output_root / stage / "manifest.json")
    return output


def _run_science_validate(config: PaperExperimentConfig) -> Path:
    return _run_validate_common(config, include_timing=False)


def _run_validate(config: PaperExperimentConfig) -> Path:
    return _run_validate_common(config, include_timing=True)


def _run_validate_common(
    config: PaperExperimentConfig,
    *,
    include_timing: bool,
) -> Path:
    validations = {}
    evaluation_stages = ["pilot", "headline", "generalization", "ablation"]
    if (config.output_root / "objective-comparison" / "manifest.json").is_file():
        evaluation_stages.append("objective-comparison")
    current_source_hash = pzr_source_sha256()
    for stage in evaluation_stages:
        directory = config.output_root / stage
        manifest = load_json(directory / "manifest.json")
        if manifest.get("config_sha256") != config.config_sha256:
            raise ValueError(f"stale {stage} stage manifest")
        if manifest.get("pzr_source_sha256") != current_source_hash:
            raise ValueError(f"stale {stage} source manifest")
        _validate_runtime_provenance(manifest, stage)
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
    if include_timing:
        timing_manifest = load_json(config.output_root / "timing" / "manifest.json")
        if timing_manifest.get("schema") != PAPER_STAGE_SCHEMA:
            raise ValueError("unsupported timing stage manifest schema")
        if timing_manifest.get("config_sha256") != config.config_sha256:
            raise ValueError("stale timing config manifest")
        if timing_manifest.get("pzr_source_sha256") != current_source_hash:
            raise ValueError("stale timing source manifest")
        _validate_runtime_provenance(timing_manifest, "timing")
        _validate_timing_stage(config, timing_manifest)
        validations["timing"] = {
            "cell_count": int(timing_manifest.get("cell_count", 0)),
            "failure_count": int(timing_manifest.get("failure_count", 0)),
        }
    artifact_root = (
        config.paper_artifact_dir
        if include_timing else config.output_root / "science-report" / "artifacts"
    )
    artifact_manifest = artifact_root / "artifact_hashes.json"
    if not artifact_manifest.is_file():
        raise ValueError("generated paper artifact hash manifest is missing")
    stage = "validate" if include_timing else "science-validate"
    destination = config.output_root / stage
    failure_count = sum(
        int(stage["failure_count"]) for stage in validations.values()
    )
    write_json_atomic(stage_manifest(
        config, stage=stage,
        status="completed" if failure_count == 0 else "completed_with_failures",
        failure_count=failure_count,
        extra={
            "validated_stages": validations,
            "artifact_hash_manifest": str(artifact_manifest),
            "timing_included": include_timing,
        },
    ), destination / "manifest.json")
    return destination


def _require_contiguous_seeds(name: str, seeds: Sequence[int]) -> None:
    if tuple(seeds) != tuple(range(min(seeds), min(seeds) + len(seeds))):
        raise ValueError(f"{name} seeds must be contiguous for the trace-store schema")


def _safe(value: str) -> str:
    return value.replace(":", "_").replace("/", "_")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the versioned paper evaluation",
    )
    parser.add_argument(
        "stage", choices=(*STAGES, "explore", "evaluate", "run", "status"),
    )
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
        smoke_root = args.output or Path("/tmp/pzr-paper-evaluation-smoke")
        config = replace(
            config,
            output_root=smoke_root,
            paper_artifact_dir=(
                args.paper_artifacts or smoke_root / "generated-paper-artifacts"
            ),
            event_count=20,
            budgets=(40, 80),
            generated_nominal_trace_kind="random_waypoint",
            fixed_figure8_trace_kinds=("figure8",),
            teacher_workers=1,
            evaluation_workers=1,
            training_epochs=2,
            teacher_dataset_parent=None,
            teacher_dataset_parent_sha256=None,
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
    if args.stage == "status":
        print(json.dumps(paper_evaluation_status(config), indent=2, sort_keys=True))
        return
    if args.stage == "explore":
        if args.approve_long_run:
            raise ValueError("explore never starts the approval-gated held-out sweep")
        if args.workers not in {None, config.evaluation_workers}:
            raise ValueError(
                "explore uses the configured pilot worker count so its projection "
                "retains the declared semantics"
            )
        try:
            result = run_exploratory_bundle(config, smoke=args.smoke)
        except Exception as exc:
            if config.output_root.is_dir():
                write_json_atomic({
                    "schema": "pzr.paper-evaluation-exploration.v3",
                    "experiment_id": config.experiment_id,
                    "status": "failed",
                    "updated_at": _utc_now(),
                    "config_sha256": config.config_sha256,
                    "pzr_source_sha256": pzr_source_sha256(),
                    **_runtime_provenance(),
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                }, config.output_root / "explore" / "manifest.json")
            raise
        print(
            f"Exploratory bundle {result.status}: failures={result.failure_count}, "
            f"manifest={result.manifest}",
        )
        raise SystemExit(result.exit_code)
    if args.stage == "evaluate":
        if args.workers not in {None, config.evaluation_workers}:
            raise ValueError(
                "evaluate uses the configured worker counts to retain its declared semantics"
            )
        result = run_scientific_paper_evaluation(
            config,
            approve_long_run=args.approve_long_run,
            smoke=args.smoke,
        )
        print(
            f"Scientific evaluation {result.status}: failures={result.failure_count}, "
            f"manifest={result.manifest}",
        )
        raise SystemExit(result.exit_code)
    if args.stage == "run":
        try:
            result = run_complete_paper_evaluation(
                config,
                approve_long_run=args.approve_long_run,
                smoke=args.smoke,
            )
        except Exception as exc:
            if config.output_root.is_dir():
                _write_run_manifest(
                    config,
                    status="failed",
                    failure_count=_scientific_failure_count(config),
                    extra={
                        "failure_type": type(exc).__name__,
                        "failure_message": str(exc),
                    },
                )
            raise
        print(
            f"Paper evaluation {result.status}: failures={result.failure_count}, "
            f"manifest={result.manifest}",
        )
        raise SystemExit(result.exit_code)
    output = run_paper_stage(
        config,
        args.stage,
        workers=args.workers,
        approve_long_run=args.approve_long_run,
    )
    print(f"Paper-evaluation stage complete: {args.stage} -> {output}")


if __name__ == "__main__":
    main()
