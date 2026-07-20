"""Resumable shard orchestration for RTLola reducer-cost collection."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd

from pzr.artifact_io import write_csv_atomic
from pzr.learning.artifacts import load_reducer_cost_dataset, write_reducer_cost_dataset
from pzr.learning.dart import DartCalibration
from pzr.learning.dataset import ReducerCostDataset
from pzr.learning.provenance import pzr_source_sha256, sha256_files
from pzr.rtlola.actions import MPC_ACTION_NAMES, default_action_catalog
from pzr.rtlola.binding import (
    BINDING_BUILD_PROFILE,
    BINDING_REVISION,
    INTERPRETER_REVISION,
)
from pzr.rtlola.engine import RtlolaEvent
from pzr.rtlola.features import RTL_RANKING_FEATURE_SCHEMA
from pzr.rtlola.learning_data import collect_teacher_episode, write_collected_dataset
from pzr.rtlola.learning_traces import load_random_waypoint_trace_store
from pzr.rtlola.scenarios import scenario_by_name


CollectionMode = Literal["teacher", "dart"]


@dataclass(frozen=True)
class LearningCollectionConfig:
    output: Path
    trace_store: Path
    budgets: tuple[int, ...]
    candidate_names: tuple[str, ...] = MPC_ACTION_NAMES
    train_seeds: int = 4
    validation_seeds: int = 1
    test_seeds: int = 0
    seed_start: int = 0
    workers: int = 1
    collection_mode: CollectionMode = "teacher"
    dart_calibration: Path | None = None
    disturbance_seed: int = 20260717

    def __post_init__(self) -> None:
        if not self.budgets:
            raise ValueError("at least one budget is required")
        if self.train_seeds < 1 or min(self.validation_seeds, self.test_seeds) < 0:
            raise ValueError("training needs a seed and split counts cannot be negative")
        if self.seed_start < 0 or self.disturbance_seed < 0:
            raise ValueError("seed values must be non-negative")
        if self.workers < 1:
            raise ValueError("collection workers must be positive")
        if self.collection_mode == "dart" and self.dart_calibration is None:
            raise ValueError("DART collection requires a calibration")
        if self.collection_mode == "teacher" and self.dart_calibration is not None:
            raise ValueError("teacher collection does not accept a DART calibration")
        default_action_catalog(self.candidate_names)


def run_learning_collection(config: LearningCollectionConfig) -> Path:
    """Run or resume every collection shard and write one validated dataset."""
    calibration = (
        DartCalibration.load(config.dart_calibration)
        if config.dart_calibration is not None else None
    )
    calibration_hash = (
        sha256_files((
            config.dart_calibration / "calibration.json",
            config.dart_calibration / "dart_budget_calibration.csv",
            config.dart_calibration / "dart_direction_kernel.csv",
        ))
        if config.dart_calibration is not None else None
    )
    source_sha256 = pzr_source_sha256()
    _validate_calibration(config, calibration, source_sha256)
    trace_store = load_random_waypoint_trace_store(config.trace_store)
    scenario = scenario_by_name("robot_arm")
    jobs: list[_CollectionShardJob] = []
    trace_records: list[dict[str, object]] = []
    for split, seed in split_seeds(config):
        for stored_trace in trace_store.traces_for_seed(seed):
            trace = stored_trace.trace
            trace_records.append({
                "trace_id": stored_trace.trace_id,
                "split": split,
                "condition": stored_trace.condition,
                "seed": seed,
                "trace_sha256": trace.metadata.trace_sha256,
                "trace_store_relative_path": str(stored_trace.relative_path),
            })
            for budget in config.budgets:
                shard_dir = (
                    config.output / "shards" / split / stored_trace.condition
                    / f"seed-{seed}" / f"budget-{budget}"
                )
                identity = _collection_identity(
                    config=config,
                    event_count=trace_store.event_count,
                    trace_id=stored_trace.trace_id,
                    split=split,
                    condition=stored_trace.condition,
                    seed=seed,
                    budget=budget,
                    trace_sha256=trace.metadata.trace_sha256,
                    trace_store_manifest_sha256=trace_store.manifest_sha256,
                    calibration=calibration,
                    calibration_hash=calibration_hash,
                    source_sha256=source_sha256,
                )
                jobs.append(_CollectionShardJob(
                    directory=shard_dir,
                    identity=identity,
                    events=trace.events,
                    trace_id=stored_trace.trace_id,
                    split=split,
                    condition=stored_trace.condition,
                    seed=seed,
                    budget=budget,
                    candidate_names=config.candidate_names,
                    collection_mode=config.collection_mode,
                    dart_calibration=config.dart_calibration,
                    dart_calibration_sha256=calibration_hash,
                    disturbance_seed=config.disturbance_seed,
                ))
    _run_collection_jobs(jobs, config.workers)
    loaded_shards = [_load_collection_shard(job) for job in jobs]
    dataset = ReducerCostDataset.concatenate([item[0] for item in loaded_shards])
    if dataset.num_samples == 0:
        raise ValueError("teacher collection produced no reduction decisions")
    sample_metadata = pd.concat([item[1] for item in loaded_shards], ignore_index=True)
    metadata: dict[str, object] = {
        "scenario": scenario.name,
        "collection_mode": config.collection_mode,
        "event_count": trace_store.event_count,
        "budgets": list(config.budgets),
        "conditions": list(trace_store.conditions),
        "seed_start": config.seed_start,
        "trace_store": str(trace_store.root),
        "trace_store_manifest_sha256": trace_store.manifest_sha256,
        "binding_revision": BINDING_REVISION,
        "interpreter_revision": INTERPRETER_REVISION,
        "binding_build_profile": BINDING_BUILD_PROFILE,
        "feature_schema": feature_schema_payload(),
        "pzr_source_sha256": source_sha256,
        "shard_count": len(loaded_shards),
        "traces": trace_records,
    }
    if calibration is not None:
        metadata.update({
            "dart_calibration_sha256": calibration_hash,
            "dart_contract": calibration.contract(),
            "disturbance_seed": config.disturbance_seed,
        })
    dataset_dir = config.output / "dataset"
    write_reducer_cost_dataset(dataset, dataset_dir, sample_metadata, metadata)
    write_collection_summaries(sample_metadata, dataset_dir)
    return dataset_dir


def feature_schema_payload() -> dict[str, object]:
    return {
        "name": RTL_RANKING_FEATURE_SCHEMA.name,
        "version": RTL_RANKING_FEATURE_SCHEMA.version,
        "feature_names": list(RTL_RANKING_FEATURE_SCHEMA.feature_names),
        "log1p_features": list(RTL_RANKING_FEATURE_SCHEMA.log1p_features),
    }


def split_seeds(config: LearningCollectionConfig) -> tuple[tuple[str, int], ...]:
    counts = (
        ("train", config.train_seeds),
        ("validation", config.validation_seeds),
        ("test", config.test_seeds),
    )
    offset = config.seed_start
    result: list[tuple[str, int]] = []
    for split, count in counts:
        result.extend((split, offset + index) for index in range(count))
        offset += count
    return tuple(result)


@dataclass(frozen=True)
class _CollectionShardJob:
    directory: Path
    identity: dict[str, object]
    events: tuple[RtlolaEvent, ...]
    trace_id: str
    split: str
    condition: str
    seed: int
    budget: int
    candidate_names: tuple[str, ...]
    collection_mode: CollectionMode
    dart_calibration: Path | None
    dart_calibration_sha256: str | None
    disturbance_seed: int


def _validate_calibration(
    config: LearningCollectionConfig,
    calibration: DartCalibration | None,
    source_sha256: str,
) -> None:
    if calibration is None:
        return
    if calibration.candidate_names != config.candidate_names:
        raise ValueError("DART calibration candidate catalog differs")
    if not set(config.budgets) <= set(calibration.budgets):
        raise ValueError("DART calibration does not cover every collection budget")
    expected_context = {
        "candidate_names": list(config.candidate_names),
        "feature_schema": feature_schema_payload(),
        "binding_revision": BINDING_REVISION,
        "interpreter_revision": INTERPRETER_REVISION,
        "binding_build_profile": BINDING_BUILD_PROFILE,
        "pzr_source_sha256": source_sha256,
    }
    mismatched = [
        name for name, value in expected_context.items()
        if calibration.context.get(name) != value
    ]
    if mismatched:
        raise ValueError(
            "DART calibration context differs for: " + ", ".join(sorted(mismatched))
        )


def _collection_identity(
    *,
    config: LearningCollectionConfig,
    event_count: int,
    trace_id: str,
    split: str,
    condition: str,
    seed: int,
    budget: int,
    trace_sha256: str,
    trace_store_manifest_sha256: str,
    calibration: DartCalibration | None,
    calibration_hash: str | None,
    source_sha256: str,
) -> dict[str, object]:
    identity: dict[str, object] = {
        "collection_shard": True,
        "scenario": "robot_arm",
        "collection_mode": config.collection_mode,
        "event_count": event_count,
        "trace_id": trace_id,
        "split": split,
        "condition": condition,
        "seed": seed,
        "budget": budget,
        "candidate_names": list(config.candidate_names),
        "feature_schema": feature_schema_payload(),
        "trace_sha256": trace_sha256,
        "trace_store_manifest_sha256": trace_store_manifest_sha256,
        "binding_revision": BINDING_REVISION,
        "interpreter_revision": INTERPRETER_REVISION,
        "binding_build_profile": BINDING_BUILD_PROFILE,
        "pzr_source_sha256": source_sha256,
    }
    if calibration is not None:
        identity.update({
            "dart_calibration_sha256": calibration_hash,
            "dart_contract": calibration.contract(),
            "disturbance_seed": config.disturbance_seed,
        })
    return identity


def _run_collection_jobs(jobs: Sequence[_CollectionShardJob], workers: int) -> None:
    missing = []
    for job in jobs:
        if (job.directory / "manifest.json").is_file():
            _load_collection_shard(job)
        else:
            missing.append(job)
    if workers == 1:
        for job in missing:
            _run_collection_shard(job)
    elif missing:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("spawn"),
        ) as executor:
            tuple(executor.map(_run_collection_shard, missing))


def _run_collection_shard(job: _CollectionShardJob) -> None:
    calibration = DartCalibration.load(job.dart_calibration) if job.dart_calibration else None
    samples = collect_teacher_episode(
        scenario=scenario_by_name("robot_arm"),
        events=job.events,
        trace_id=job.trace_id,
        split=job.split,
        condition=job.condition,
        seed=job.seed,
        budget=job.budget,
        candidate_names=job.candidate_names,
        collection_mode=job.collection_mode,
        dart_calibration=calibration,
        dart_calibration_sha256=job.dart_calibration_sha256,
        disturbance_seed=job.disturbance_seed,
    )
    write_collected_dataset(
        samples, job.directory, job.identity, candidate_names=job.candidate_names,
    )


def _load_collection_shard(
    job: _CollectionShardJob,
) -> tuple[ReducerCostDataset, pd.DataFrame]:
    dataset, sample_metadata, manifest = load_reducer_cost_dataset(job.directory)
    mismatched = [
        name for name, value in job.identity.items() if manifest.get(name) != value
    ]
    if mismatched:
        raise ValueError(
            "learning collection shard identity differs for: "
            + ", ".join(sorted(mismatched))
        )
    return dataset, sample_metadata


def write_collection_summaries(metadata: pd.DataFrame, output: Path) -> None:
    """Write core collection accounting and optional DART-only diagnostics."""
    groups = ["split", "condition", "budget"]
    summary = metadata.groupby(groups, dropna=False).agg(
        sample_count=("sample_id", "size"),
        mean_evaluated_leaves=("evaluated_leaves", "mean"),
        teacher_reducer_failure_count=("teacher_reducer_failure_count", "sum"),
        teacher_infeasible_candidate_count=("teacher_infeasible_candidate_count", "sum"),
        execution_fallback_count=("execution_fallback_used", "sum"),
    ).reset_index()
    write_csv_atomic(summary, output / "collection_summary.csv")
    teacher_counts = (
        metadata.groupby([*groups, "teacher_action"], dropna=False)
        .size().rename("count").reset_index()
    )
    write_csv_atomic(teacher_counts, output / "teacher_action_counts.csv")
    if "disturbed" not in metadata:
        return
    working = metadata.copy()
    working["cap_blocked"] = (
        ~working["recovery_forced"].astype(bool)
        & ~working["disturbance_eligible"].astype(bool)
    )
    dart_summary = metadata.groupby(groups, dropna=False).agg(
        sample_count=("sample_id", "size"),
        disturbance_eligible_count=("disturbance_eligible", "sum"),
        disturbance_attempt_count=("disturbance_attempted", "sum"),
        disturbed_count=("disturbed", "sum"),
        recovery_forced_count=("recovery_forced", "sum"),
        mean_target_disturbance_rate=("target_disturbance_rate", "mean"),
        mean_injection_probability=("injection_probability", "mean"),
        mean_disturbance_probability=("disturbance_probability", "mean"),
        mean_regret_cap=("regret_cap", "mean"),
        mean_sampled_normalized_regret=("sampled_normalized_regret", "mean"),
        median_sampled_normalized_regret=("sampled_normalized_regret", "median"),
        max_sampled_normalized_regret=("sampled_normalized_regret", "max"),
    ).reset_index()
    blocked = (
        working.groupby(groups, dropna=False)["cap_blocked"].sum()
        .rename("cap_blocked_state_count").reset_index()
    )
    runs = pd.DataFrame([
        {
            **dict(zip(groups, key if isinstance(key, tuple) else (key,))),
            "maximum_consecutive_disturbances": _maximum_disturbance_run(frame),
        }
        for key, frame in working.groupby(groups, dropna=False)
    ])
    dart_summary = dart_summary.merge(
        blocked, on=groups, validate="one_to_one",
    ).merge(runs, on=groups, validate="one_to_one")
    dart_summary["realized_disturbance_rate"] = (
        dart_summary["disturbed_count"] / dart_summary["sample_count"]
    )
    write_csv_atomic(dart_summary, output / "dart_collection_summary.csv")
    executed_counts = (
        metadata.groupby([*groups, "executed_action"], dropna=False)
        .size().rename("count").reset_index()
    )
    confusion = (
        metadata.groupby(
            [*groups, "teacher_action", "executed_action"], dropna=False,
        ).size().rename("count").reset_index()
    )
    write_csv_atomic(executed_counts, output / "executed_action_counts.csv")
    write_csv_atomic(confusion, output / "teacher_executed_confusion.csv")


def _maximum_disturbance_run(frame: pd.DataFrame) -> int:
    maximum = 0
    for _, trace in frame.groupby("trace_id", sort=False):
        current = 0
        for disturbed in trace.sort_values("step")["disturbed"].astype(bool):
            current = current + 1 if disturbed else 0
            maximum = max(maximum, current)
    return maximum
