#!/usr/bin/env python3
"""Disposable clean-data scaling check on nominal robot-arm traces.

The experiment keeps the seven exact-budget PRP specialists and all learning
hyperparameters fixed.  It compares the existing Clean20/Clean36 models with
new Clean52/Clean68/Clean84 models on the already observed nominal seeds
100--139.  It is retrospective exploration, not a paper evaluation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from pzr.artifact_io import write_csv_atomic, write_json_atomic
from pzr.learning.artifacts import load_reducer_cost_dataset
from pzr.learning.dataset import ReducerCostDataset
from pzr.learning.diagnostics import validation_metrics
from pzr.learning.provenance import (
    model_sha256,
    payload_sha256,
    pzr_source_sha256,
    sha256_files,
)
from pzr.learning.ranker import train_reducer_policy
from pzr.learning.training import dataset_sha256
from pzr.rtlola.binding import (
    BINDING_BUILD_PROFILE,
    BINDING_REVISION,
    INTERPRETER_REVISION,
)
from pzr.rtlola.features import RTL_RANKING_FEATURE_SCHEMA
from pzr.rtlola.learning_collection import (
    LearningCollectionConfig,
    run_learning_collection,
)
from pzr.rtlola.learning_traces import (
    RandomWaypointTraceStore,
    RandomWaypointTraceStoreConfig,
    generate_random_waypoint_trace_store,
    load_random_waypoint_trace_store,
)
from pzr.rtlola.paper_experiment import (
    ExecutionRegime,
    MethodConfig,
    Objective,
    Predictor,
    RunState,
    TraceSource,
    load_json,
    trace_level_metrics,
)
from pzr.rtlola.paper_pipeline import (
    EvaluationCellJob,
    EvaluationTrace,
    _failed_row as paper_failed_row,
    _run_cell as run_paper_cell,
)
from pzr.rtlola.reference import REFERENCE_CACHE_SCHEMA
from pzr.rtlola.robot_arm import RLOLAEVAL_REVISION, ROBOT_ARM_SPEC_SHA256
from pzr.rtlola.robot_arm_random import RANDOM_WAYPOINT_SOURCE_REVISION


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "clean-scaling-exploratory"
PAPER_ROOT = ROOT / "results" / "paper-evaluation-v2"
DART_ROOT = ROOT / "results" / "dart-rescue-v1"
PAPER_CONFIG = ROOT / "experiments" / "paper_evaluation_v2.yaml"
DART_CONFIG = ROOT / "experiments" / "dart_rescue_v1.yaml"
PAPER_CONFIG_SHA256 = "a7a911f641b0227aa3a6657231afef97aedbd8428ab25f2caf9d9d6dd486f074"
DART_CONFIG_SHA256 = "73dfbd34a6cd5aeadfc5a1f341db2590b6015e97a8176b82754df18c3503154d"
DART_EFFECTIVE_CONFIG_SHA256 = (
    "6c930dc948b254c27125463e2147f43c9867befa4afa5513c74dd74d2340538f"
)
PZR_SOURCE_SHA256 = "f230d481022de2c69c610c917deae901e7a87e4322c979c248f2cf8f4fa1e5ca"
PAPER_DATASET = PAPER_ROOT / "prepare" / "teacher" / "dataset"
PAPER_DATASET_SHA256 = (
    "885c3dfbf70ddf614db72f564877e667e056a59966e62094d365606e0b503602"
)
DART_CLEAN_DATASET = DART_ROOT / "prepare" / "extra-clean" / "dataset"
DART_CLEAN_DATASET_SHA256 = (
    "3540f04b561197746862d7b2bbdc3f2880fc2e3613dd975c3fde43ff8b3c08f6"
)

SCHEMA = "pzr.clean-scaling-exploratory.v1"
CELL_SCHEMA = "pzr.clean-scaling-exploratory-cell.v1"
BUDGETS = (40, 80, 120, 150, 200, 250, 500)
BASE_TRAIN_SEEDS = tuple(range(20)) + tuple(range(26, 42))
NEW_TRAIN_SEEDS = tuple(range(200, 248))
VALIDATION_SEEDS = tuple(range(20, 26))
OBSERVED_SEEDS = tuple(range(100, 140))
NEW_TRAINING_SIZES = (52, 68, 84)
ALL_SIZES = (20, 36, *NEW_TRAINING_SIZES)
CANDIDATES = ("girard", "scott", "pca", "combastel")
EVENT_COUNT = 500
WORKERS = 10
TRAINING = {
    "epochs": 100,
    "batch_size": 256,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "patience": 10,
    "seed": 42,
}
STAGES = ("check", "prepare", "train", "evaluate", "report")


@dataclass(frozen=True)
class ExploreConfig:
    output: Path = OUTPUT
    budgets: tuple[int, ...] = BUDGETS
    base_train_seeds: tuple[int, ...] = BASE_TRAIN_SEEDS
    new_train_seeds: tuple[int, ...] = NEW_TRAIN_SEEDS
    validation_seeds: tuple[int, ...] = VALIDATION_SEEDS
    observed_seeds: tuple[int, ...] = OBSERVED_SEEDS
    new_training_sizes: tuple[int, ...] = NEW_TRAINING_SIZES
    event_count: int = EVENT_COUNT
    workers: int = WORKERS
    reuse_existing: bool = True
    smoke: bool = False

    @property
    def master_train_seeds(self) -> tuple[int, ...]:
        return self.base_train_seeds + self.new_train_seeds

    @property
    def all_sizes(self) -> tuple[int, ...]:
        return (
            (20, 36, *self.new_training_sizes)
            if self.reuse_existing
            else self.new_training_sizes
        )

    @property
    def identity(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "output": str(self.output.resolve()),
            "budgets": list(self.budgets),
            "base_train_seeds": list(self.base_train_seeds),
            "new_train_seeds": list(self.new_train_seeds),
            "validation_seeds": list(self.validation_seeds),
            "observed_seeds": list(self.observed_seeds),
            "new_training_sizes": list(self.new_training_sizes),
            "event_count": self.event_count,
            "workers": self.workers,
            "reuse_existing": self.reuse_existing,
            "smoke": self.smoke,
            "training": TRAINING,
            "tool_sha256": tool_sha256(),
            "pzr_source_sha256": pzr_source_sha256(),
            "binding_revision": BINDING_REVISION,
            "interpreter_revision": INTERPRETER_REVISION,
            "binding_build_profile": BINDING_BUILD_PROFILE,
        }

    @property
    def fingerprint(self) -> str:
        return payload_sha256(self.identity)

    @property
    def expected_new_shards(self) -> int:
        return len(self.new_train_seeds) * len(self.budgets)

    @property
    def expected_new_models(self) -> int:
        return len(self.new_training_sizes) * len(self.budgets)

    @property
    def expected_reported_cells(self) -> int:
        return len(self.observed_seeds) * len(self.budgets) * len(self.all_sizes)

    @property
    def expected_new_cells(self) -> int:
        return (
            len(self.observed_seeds)
            * len(self.budgets)
            * len(self.new_training_sizes)
        )


def smoke_config() -> ExploreConfig:
    return ExploreConfig(
        output=Path("/tmp/pzr-clean-scaling-exploratory-smoke"),
        budgets=(40,),
        base_train_seeds=(0, 1),
        new_train_seeds=(200,),
        validation_seeds=VALIDATION_SEEDS,
        observed_seeds=(100,),
        new_training_sizes=(3,),
        event_count=100,
        workers=1,
        reuse_existing=False,
        smoke=True,
    )


def training_seeds(config: ExploreConfig, size: int) -> tuple[int, ...]:
    if size < 1 or size > len(config.master_train_seeds):
        raise ValueError(f"invalid clean training size: {size}")
    return config.master_train_seeds[:size]


def method_name(size: int) -> str:
    return f"clean{size}"


def tool_sha256() -> str:
    return sha256_files(
        (
            Path(__file__).resolve(),
            ROOT / "tools" / "run_clean_scaling.sh",
        ),
        relative_to=ROOT,
    )


def _manifest_path(config: ExploreConfig, stage: str) -> Path:
    return config.output / stage / "manifest.json"


def _write_manifest(
    config: ExploreConfig,
    stage: str,
    extra: Mapping[str, object],
) -> Path:
    path = _manifest_path(config, stage)
    write_json_atomic(
        {
            **config.identity,
            "stage": stage,
            "experiment_fingerprint": config.fingerprint,
            "status": "completed",
            **dict(extra),
        },
        path,
    )
    return path


def _load_manifest(config: ExploreConfig, stage: str) -> dict[str, object]:
    path = _manifest_path(config, stage)
    manifest = load_json(path)
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"unsupported exploratory manifest: {path}")
    if manifest.get("experiment_fingerprint") != config.fingerprint:
        raise ValueError(f"stale exploratory manifest: {path}")
    return manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paper_models(config: ExploreConfig) -> dict[int, dict[str, object]]:
    manifest = load_json(PAPER_ROOT / "train" / "manifest.json")
    records = {}
    for budget in config.budgets:
        record = dict(manifest["models_by_budget"][str(budget)])
        path = Path(str(record["path"]))
        if record.get("budget_filter") != [budget]:
            raise ValueError(f"Clean20 model budget differs: {budget}")
        if model_sha256(path) != record["sha256"]:
            raise ValueError(f"Clean20 model hash differs: {budget}")
        records[budget] = {**record, "path": str(path)}
    return records


def _dart_clean36_models(config: ExploreConfig) -> dict[int, dict[str, object]]:
    manifest = load_json(DART_ROOT / "train" / "manifest.json")
    if manifest.get("effective_config_sha256") != DART_EFFECTIVE_CONFIG_SHA256:
        raise ValueError("DART training manifest differs")
    records = {}
    for budget in config.budgets:
        record = dict(manifest["models"][f"clean36:{budget}"])
        path = Path(str(record["path"]))
        if record.get("budget_filter") != [budget]:
            raise ValueError(f"Clean36 model budget differs: {budget}")
        if model_sha256(path) != record["sha256"]:
            raise ValueError(f"Clean36 model hash differs: {budget}")
        records[budget] = {**record, "path": str(path)}
    return records


def run_check(config: ExploreConfig) -> Path:
    if _sha256(PAPER_CONFIG) != PAPER_CONFIG_SHA256:
        raise ValueError("paper config hash differs")
    if _sha256(DART_CONFIG) != DART_CONFIG_SHA256:
        raise ValueError("DART config hash differs")
    if pzr_source_sha256() != PZR_SOURCE_SHA256:
        raise ValueError("PZR source differs from the completed input artifacts")
    if BINDING_BUILD_PROFILE != "release":
        raise ValueError("clean scaling requires the release binding")
    if dataset_sha256(PAPER_DATASET) != PAPER_DATASET_SHA256:
        raise ValueError("Clean20 teacher dataset hash differs")
    if dataset_sha256(DART_CLEAN_DATASET) != DART_CLEAN_DATASET_SHA256:
        raise ValueError("Clean36 extension dataset hash differs")
    _paper_models(config)
    _dart_clean36_models(config)
    return _write_manifest(config, "check", {
        "scientific_role": "retrospective exploratory diagnostic",
        "paper_dataset_sha256": PAPER_DATASET_SHA256,
        "dart_clean_dataset_sha256": DART_CLEAN_DATASET_SHA256,
    })


def run_prepare(config: ExploreConfig) -> Path:
    _load_manifest(config, "check")
    trace_root = config.output / "prepare" / "traces-new-clean"
    store = generate_random_waypoint_trace_store(
        RandomWaypointTraceStoreConfig(
            output=trace_root,
            event_count=config.event_count,
            conditions=("random_waypoint",),
            seed_start=min(config.new_train_seeds),
            seed_count=len(config.new_train_seeds),
        )
    )
    if tuple(item.seed for item in store.traces) != config.new_train_seeds:
        raise ValueError("new clean trace seeds differ")
    dataset = run_learning_collection(
        LearningCollectionConfig(
            output=config.output / "prepare" / "new-clean",
            trace_store=trace_root,
            budgets=config.budgets,
            candidate_names=CANDIDATES,
            train_seeds=len(config.new_train_seeds),
            validation_seeds=0,
            test_seeds=0,
            seed_start=min(config.new_train_seeds),
            workers=config.workers,
            collection_mode="teacher",
        )
    )
    manifest = load_json(dataset / "manifest.json")
    if int(manifest["shard_count"]) != config.expected_new_shards:
        raise ValueError("new clean teacher shard count differs")
    return _write_manifest(config, "prepare", {
        "trace_store": str(trace_root),
        "trace_store_manifest_sha256": store.manifest_sha256,
        "dataset": str(dataset),
        "dataset_sha256": dataset_sha256(dataset),
        "teacher_shard_count": config.expected_new_shards,
    })


def _master_dataset(
    config: ExploreConfig,
) -> tuple[ReducerCostDataset, pd.DataFrame, dict[str, str]]:
    prepare = _load_manifest(config, "prepare")
    sources = (
        ("paper_clean20", PAPER_DATASET, PAPER_DATASET_SHA256),
        ("dart_extra_clean16", DART_CLEAN_DATASET, DART_CLEAN_DATASET_SHA256),
        (
            "new_clean48",
            Path(str(prepare["dataset"])),
            str(prepare["dataset_sha256"]),
        ),
    )
    datasets = []
    frames = []
    hashes = {}
    for label, path, expected_hash in sources:
        if dataset_sha256(path) != expected_hash:
            raise ValueError(f"training source hash differs: {label}")
        dataset, metadata, _ = load_reducer_cost_dataset(path)
        datasets.append(dataset)
        frames.append(metadata)
        hashes[label] = expected_hash
    dataset = ReducerCostDataset.concatenate(datasets)
    metadata = pd.concat(frames, ignore_index=True)
    if tuple(metadata["sample_id"].astype(str)) != dataset.sample_ids:
        raise ValueError("master clean dataset alignment differs")
    return dataset, metadata, hashes


def select_training_rows(
    dataset: ReducerCostDataset,
    metadata: pd.DataFrame,
    *,
    seeds: Sequence[int],
    validation_seeds: Sequence[int],
    budget: int,
) -> tuple[ReducerCostDataset, pd.DataFrame]:
    split = metadata["split"].astype(str)
    selected = (
        (metadata["budget"].astype(int) == budget)
        & (
            ((split == "train") & metadata["seed"].astype(int).isin(seeds))
            | (
                (split == "validation")
                & metadata["seed"].astype(int).isin(validation_seeds)
            )
        )
    )
    indices = np.flatnonzero(selected.to_numpy())
    subset = dataset.subset(indices)
    frame = metadata.iloc[indices].reset_index(drop=True)
    if set(frame.loc[frame["split"] == "train", "seed"].astype(int)) != set(seeds):
        raise ValueError("clean training seed coverage differs")
    if set(
        frame.loc[frame["split"] == "validation", "seed"].astype(int)
    ) != set(validation_seeds):
        raise ValueError("clean validation seed coverage differs")
    if set(frame["budget"].astype(int)) != {budget}:
        raise ValueError("clean training budget differs")
    frame.insert(0, "dataset_label", "clean")
    return subset, frame


def _subset_hash(
    dataset: ReducerCostDataset,
    *,
    seeds: Sequence[int],
    budget: int,
    source_hashes: Mapping[str, str],
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({
        "seeds": list(seeds),
        "budget": budget,
        "sources": dict(source_hashes),
        "sample_ids": list(dataset.sample_ids),
    }, sort_keys=True).encode())
    for array in (dataset.features, dataset.teacher_costs, dataset.feasible):
        values = np.ascontiguousarray(array)
        digest.update(str(values.dtype).encode())
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _new_model_path(config: ExploreConfig, size: int, budget: int) -> Path:
    return config.output / "train" / f"clean{size}" / f"budget-{budget}"


def run_train(config: ExploreConfig) -> Path:
    dataset, metadata, source_hashes = _master_dataset(config)
    records = {}
    for size in config.new_training_sizes:
        seeds = training_seeds(config, size)
        for budget in config.budgets:
            subset, subset_metadata = select_training_rows(
                dataset,
                metadata,
                seeds=seeds,
                validation_seeds=config.validation_seeds,
                budget=budget,
            )
            subset_hash = _subset_hash(
                subset,
                seeds=seeds,
                budget=budget,
                source_hashes=source_hashes,
            )
            output = _new_model_path(config, size, budget)
            identity = {
                "schema": SCHEMA,
                "experiment_fingerprint": config.fingerprint,
                "training_size": size,
                "training_seeds": list(seeds),
                "budget": budget,
                "subset_sha256": subset_hash,
                "source_hashes": source_hashes,
                "training": TRAINING,
            }
            artifact = output / "exploratory_training.json"
            if artifact.is_file():
                existing = load_json(artifact)
                if existing.get("identity") != identity:
                    raise ValueError(f"stale exploratory model: {output}")
                if model_sha256(output) != existing["model_sha256"]:
                    raise ValueError(f"exploratory model hash differs: {output}")
                records[f"{size}:{budget}"] = existing["record"]
                continue
            policy, result = train_reducer_policy(
                subset,
                RTL_RANKING_FEATURE_SCHEMA,
                objective="pairwise",
                epochs=int(TRAINING["epochs"]),
                batch_size=int(TRAINING["batch_size"]),
                learning_rate=float(TRAINING["learning_rate"]),
                weight_decay=float(TRAINING["weight_decay"]),
                patience=int(TRAINING["patience"]),
                seed=int(TRAINING["seed"]),
            )
            policy.save(output)
            write_csv_atomic(
                validation_metrics(policy, subset, subset_metadata),
                output / "validation_metrics.csv",
            )
            record = {
                "method": method_name(size),
                "training_size": size,
                "training_seeds": list(seeds),
                "budget": budget,
                "path": str(output),
                "sha256": model_sha256(output),
                "subset_sha256": subset_hash,
                "best_epoch": result.best_epoch,
                "epochs_completed": result.epochs,
            }
            write_json_atomic({
                "identity": identity,
                "record": record,
                "model_sha256": record["sha256"],
                "validation_metrics": asdict(result.val_metrics),
            }, artifact)
            records[f"{size}:{budget}"] = record
    if len(records) != config.expected_new_models:
        raise ValueError("new exploratory model count differs")
    return _write_manifest(config, "train", {
        "models": records,
        "new_model_count": len(records),
        "training_sizes": list(config.new_training_sizes),
        "training_seed_lists": {
            str(size): list(training_seeds(config, size))
            for size in config.new_training_sizes
        },
        "shared_hyperparameters": TRAINING,
    })


def _stored_traces(store: RandomWaypointTraceStore) -> tuple[EvaluationTrace, ...]:
    return tuple(
        EvaluationTrace(
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
                "generator_config": item.trace.metadata.generator_config,
                "generator_config_sha256": payload_sha256(
                    item.trace.metadata.generator_config
                ),
            },
        )
        for item in store.traces
    )


def _observed_traces(config: ExploreConfig) -> tuple[EvaluationTrace, ...]:
    if config.smoke:
        root = config.output / "evaluate" / "traces"
        store = generate_random_waypoint_trace_store(
            RandomWaypointTraceStoreConfig(
                output=root,
                event_count=config.event_count,
                conditions=("random_waypoint",),
                seed_start=min(config.observed_seeds),
                seed_count=len(config.observed_seeds),
            )
        )
        return _stored_traces(store)
    paper = load_random_waypoint_trace_store(
        PAPER_ROOT / "generalization" / "traces" / "generated-nominal"
    )
    dart = load_random_waypoint_trace_store(
        DART_ROOT / "evaluate" / "confirmation" / "traces" / "generated-nominal"
    )
    traces = _stored_traces(paper) + _stored_traces(dart)
    if tuple(item.seed for item in traces) != config.observed_seeds:
        raise ValueError("observed nominal trace coverage differs")
    return traces


def _reference_path(config: ExploreConfig, trace: EvaluationTrace) -> Path:
    if config.smoke:
        from pzr.rtlola.reference import load_or_compute_reference
        from pzr.rtlola.scenarios import scenario_by_name

        path = (
            config.output / "evaluate" / "references"
            / f"random_waypoint_seed-{trace.seed}.json"
        )
        load_or_compute_reference(
            trace.events,
            scenario=scenario_by_name("robot_arm"),
            trace_kind=trace.trace_id,
            seed=trace.seed,
            cache_path=path,
            include_approximation=True,
        )
        return path
    if trace.seed < 120:
        path = (
            PAPER_ROOT / "generalization" / "references"
            / f"random_waypoint_seed-{trace.seed}.json"
        )
    else:
        path = (
            DART_ROOT / "evaluate" / "confirmation" / "references"
            / f"random_waypoint_seed-{trace.seed}.json"
        )
    if not path.is_file():
        raise ValueError(f"missing exact reference: {path}")
    return path


def _model_records(
    config: ExploreConfig,
) -> dict[tuple[int, int], dict[str, object]]:
    records = {}
    if config.reuse_existing:
        for budget, record in _paper_models(config).items():
            records[(20, budget)] = {**record, "training_size": 20}
        for budget, record in _dart_clean36_models(config).items():
            records[(36, budget)] = {**record, "training_size": 36}
    trained = _load_manifest(config, "train")
    for record in trained["models"].values():
        records[(int(record["training_size"]), int(record["budget"]))] = dict(record)
    return records


def _method(size: int) -> MethodConfig:
    return MethodConfig(
        name=method_name(size),
        execution_regime=ExecutionRegime.LEARNED_ONLINE,
        predictor=Predictor.NONE,
        horizon=0,
        beam_width=1,
        objective=Objective.LEARNED_TERMINAL_TEACHER,
        candidate_names=CANDIDATES,
    )


def _cell_identity(
    config: ExploreConfig,
    *,
    trace: EvaluationTrace,
    size: int,
    budget: int,
    reference: Path,
    record: Mapping[str, object],
) -> dict[str, object]:
    payload = {
        "schema": CELL_SCHEMA,
        "experiment_fingerprint": config.fingerprint,
        "trace_id": trace.trace_id,
        "trace_sha256": trace.trace_sha256,
        "trace_source": trace.trace_source.value,
        "trace_kind": trace.trace_kind,
        "trace_provenance": dict(trace.provenance),
        "seed": trace.seed,
        "event_count": len(trace.events),
        "training_size": size,
        "training_seeds": list(training_seeds(config, size)),
        "budget": budget,
        "model_sha256": record["sha256"],
        "model_training_budget": budget,
        "method": {
            **asdict(_method(size)),
            "execution_regime": ExecutionRegime.LEARNED_ONLINE.value,
            "predictor": Predictor.NONE.value,
            "objective": Objective.LEARNED_TERMINAL_TEACHER.value,
            "candidate_names": list(CANDIDATES),
        },
        "spec_sha256": ROBOT_ARM_SPEC_SHA256,
        "rlolaeval_revision": RLOLAEVAL_REVISION,
        "binding_revision": BINDING_REVISION,
        "interpreter_revision": INTERPRETER_REVISION,
        "binding_build_profile": BINDING_BUILD_PROFILE,
        "reference_cache_schema": REFERENCE_CACHE_SCHEMA,
        "reference_cache_sha256": sha256_files((reference,)),
    }
    return {**payload, "fingerprint": payload_sha256(payload)}


def _execute_cell(job: EvaluationCellJob) -> dict[str, object]:
    manifest_path = job.directory / "manifest.json"
    summary_path = job.directory / "summary.csv"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        if manifest.get("schema") != CELL_SCHEMA or manifest.get(
            "identity"
        ) != job.identity:
            raise ValueError(f"stale exploratory cell: {job.directory}")
        frame = pd.read_csv(summary_path)
        if len(frame) != 1:
            raise ValueError(f"exploratory cell summary differs: {job.directory}")
        return frame.iloc[0].to_dict()
    job.directory.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    try:
        row, diagnostic = run_paper_cell(job)
    except Exception as exc:
        row = paper_failed_row(
            job,
            RunState.INFRASTRUCTURE_FAILED,
            type(exc).__name__,
            str(exc),
        )
        diagnostic = {"failure_type": type(exc).__name__, "message": str(exc)}
    row["cell_elapsed_ms"] = (perf_counter() - started) * 1000.0
    row["training_size"] = int(job.identity["training_size"])
    write_csv_atomic(pd.DataFrame([row]), summary_path)
    write_json_atomic({
        "schema": CELL_SCHEMA,
        "identity": job.identity,
        "status": row["status"],
        "diagnostic": diagnostic,
    }, manifest_path)
    timeseries = job.directory / "timeseries_diagnostic.csv"
    if row["status"] == RunState.COMPLETED.value and timeseries.is_file():
        timeseries.unlink()
    return row


def _import_baselines(config: ExploreConfig) -> pd.DataFrame:
    if not config.reuse_existing:
        return pd.DataFrame()
    frames = []
    for root in (
        DART_ROOT / "evaluate" / "replay",
        DART_ROOT / "evaluate" / "confirmation",
    ):
        frame = pd.read_csv(root / "summary.csv")
        frame = frame[
            frame["method"].isin(("clean20", "clean36"))
            & frame["seed"].astype(int).isin(config.observed_seeds)
            & frame["budget"].astype(int).isin(config.budgets)
        ].copy()
        frame["training_size"] = frame["method"].str.removeprefix("clean").astype(int)
        frames.append(frame)
    imported = pd.concat(frames, ignore_index=True)
    expected = (
        len(config.observed_seeds) * len(config.budgets) * 2
    )
    if len(imported) != expected:
        raise ValueError("imported Clean20/Clean36 cell count differs")
    return imported


def run_evaluate(config: ExploreConfig) -> Path:
    _load_manifest(config, "train")
    traces = _observed_traces(config)
    references = {trace.trace_id: _reference_path(config, trace) for trace in traces}
    records = _model_records(config)
    jobs = []
    for trace in traces:
        for budget in config.budgets:
            for size in config.new_training_sizes:
                record = records[(size, budget)]
                identity = _cell_identity(
                    config,
                    trace=trace,
                    size=size,
                    budget=budget,
                    reference=references[trace.trace_id],
                    record=record,
                )
                jobs.append(EvaluationCellJob(
                    stage="generalization",
                    directory=(
                        config.output / "evaluate" / "cells"
                        / f"seed-{trace.seed}" / f"budget-{budget}"
                        / method_name(size)
                    ),
                    trace=trace,
                    budget=budget,
                    method=_method(size),
                    runtime_method="pairwise_ranking_policy",
                    reference_path=references[trace.trace_id],
                    identity=identity,
                    model_directory=Path(str(record["path"])),
                    model_training_budget=budget,
                ))
    started = perf_counter()
    if config.workers == 1:
        rows = [_execute_cell(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=config.workers,
            mp_context=get_context("spawn"),
            max_tasks_per_child=1,
        ) as executor:
            rows = list(executor.map(_execute_cell, jobs))
    summary = pd.DataFrame(rows)
    imported = _import_baselines(config)
    if not imported.empty:
        summary = pd.concat([imported, summary], ignore_index=True)
    if len(summary) != config.expected_reported_cells:
        raise ValueError("exploratory reported cell count differs")
    if summary.duplicated(["trace_id", "budget", "method"]).any():
        raise ValueError("exploratory evaluation contains duplicate cells")
    if set(summary["trace_source"]) != {
        TraceSource.GENERATED_NOMINAL.value
    }:
        raise ValueError("exploratory evaluation is not nominal-only")
    if not (
        summary["model_training_budget"].astype(int)
        == summary["budget"].astype(int)
    ).all():
        raise ValueError("exploratory model/evaluation budget differs")
    write_csv_atomic(summary, config.output / "evaluate" / "summary.csv")
    failures = int((summary["status"] != RunState.COMPLETED.value).sum())
    return _write_manifest(config, "evaluate", {
        "reported_cell_count": len(summary),
        "new_cell_count": len(jobs),
        "imported_cell_count": len(imported),
        "failure_count": failures,
        "matrix_wall_seconds": perf_counter() - started,
        "trace_manifest": [
            {
                "trace_id": trace.trace_id,
                "trace_sha256": trace.trace_sha256,
                "seed": trace.seed,
                "trace_source": trace.trace_source.value,
                "trace_kind": trace.trace_kind,
                "event_count": len(trace.events),
                "provenance": dict(trace.provenance),
                "reference_sha256": sha256_files((references[trace.trace_id],)),
            }
            for trace in traces
        ],
        "summary_sha256": sha256_files(
            (config.output / "evaluate" / "summary.csv",)
        ),
    })


def _aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    data = trace_level_metrics(summary)
    rows = []
    for (budget, size, method), frame in data.groupby(
        ["budget", "training_size", "method"],
        sort=True,
    ):
        completed = frame[frame["status"] == RunState.COMPLETED.value]
        failed = len(frame) - len(completed)
        fpr = completed["fpr"].dropna().to_numpy(dtype=np.float64)
        loss = completed["mean_approx_loss"].dropna().to_numpy(dtype=np.float64)
        negative = float(completed["reference_negative_count"].sum())
        rows.append({
            "budget": int(budget),
            "training_size": int(size),
            "method": method,
            "trace_count": len(frame),
            "valid_count": len(completed),
            "failed_count": failed,
            "available": failed == 0,
            "macro_fpr": (
                float(np.mean(fpr)) if failed == 0 and len(fpr) else np.nan
            ),
            "pooled_fpr": (
                float(completed["false_positive_count"].sum()) / negative
                if failed == 0 and negative > 0.0
                else np.nan
            ),
            "mean_loss": (
                float(np.mean(loss)) if failed == 0 and len(loss) else np.nan
            ),
            "median_loss": (
                float(np.median(loss)) if failed == 0 and len(loss) else np.nan
            ),
            "loss_q90": (
                float(np.quantile(loss, 0.9)) if failed == 0 and len(loss) else np.nan
            ),
            "max_loss": (
                float(np.max(loss)) if failed == 0 and len(loss) else np.nan
            ),
            "fallback_rate": float(
                (frame["status"] == RunState.FALLBACK_FAILED.value).mean()
            ),
        })
    return pd.DataFrame(rows)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def _plot_metric(
    aggregate: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    output: Path,
) -> tuple[Path, Path]:
    plt = _pyplot()
    budgets = tuple(sorted(aggregate["budget"].astype(int).unique()))
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.8), squeeze=False)
    for index, budget in enumerate(budgets):
        axis = axes.flat[index]
        frame = aggregate[aggregate["budget"] == budget].sort_values(
            "training_size"
        )
        available = frame[frame["available"].astype(bool)]
        axis.plot(
            available["training_size"],
            available[metric],
            color="#0072B2",
            marker="o",
            linestyle="-",
        )
        failed = frame[~frame["available"].astype(bool)]
        if not failed.empty:
            axis.scatter(
                failed["training_size"],
                np.zeros(len(failed)),
                color="#D55E00",
                marker="x",
                label="Unavailable",
            )
        if metric == "mean_loss" and len(available) and bool(
            (available[metric] > 0.0).all()
        ):
            axis.set_yscale("log")
        axis.set_title(f"B={budget}")
        axis.set_xlabel("Clean training trajectories")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="0.9", linewidth=0.6)
    for index in range(len(budgets), len(axes.flat)):
        axes.flat[index].set_visible(False)
    fig.tight_layout()
    pdf = output.with_suffix(".pdf")
    png = output.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png, dpi=250, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return pdf, png


def run_report(config: ExploreConfig) -> Path:
    evaluation = _load_manifest(config, "evaluate")
    summary_path = config.output / "evaluate" / "summary.csv"
    if sha256_files((summary_path,)) != evaluation["summary_sha256"]:
        raise ValueError("exploratory evaluation summary hash differs")
    summary = pd.read_csv(summary_path)
    aggregate = _aggregate(summary)
    if len(aggregate) != len(config.budgets) * len(config.all_sizes):
        raise ValueError("exploratory aggregate row count differs")
    report = config.output / "report"
    write_csv_atomic(aggregate, report / "training_size_summary.csv")
    figures = (
        *_plot_metric(
            aggregate,
            metric="macro_fpr",
            ylabel="Macro FPR",
            output=report / "nominal_fpr_by_training_size",
        ),
        *_plot_metric(
            aggregate,
            metric="mean_loss",
            ylabel="Mean native loss",
            output=report / "nominal_loss_by_training_size",
        ),
    )
    return _write_manifest(config, "report", {
        "scientific_role": "retrospective exploratory diagnostic",
        "aggregate_row_count": len(aggregate),
        "summary": str(report / "training_size_summary.csv"),
        "summary_sha256": sha256_files(
            (report / "training_size_summary.csv",)
        ),
        "figures": {
            str(path): sha256_files((path,)) for path in figures
        },
        "failure_count": int(evaluation["failure_count"]),
        "claims": {
            "nominal_only": True,
            "fresh_confirmation": False,
            "paper_ready": False,
            "automatic_promotion": False,
        },
    })


def run_stage(config: ExploreConfig, stage: str) -> Path:
    path = _manifest_path(config, stage)
    if path.is_file():
        _load_manifest(config, stage)
        print(f"skip completed stage: {stage}", flush=True)
        return path
    functions = {
        "check": run_check,
        "prepare": run_prepare,
        "train": run_train,
        "evaluate": run_evaluate,
        "report": run_report,
    }
    print(f"start exploratory stage: {stage}", flush=True)
    result = functions[stage](config)
    print(f"complete exploratory stage: {stage}", flush=True)
    return result


def run_all(config: ExploreConfig) -> Path:
    for stage in STAGES:
        run_stage(config, stage)
    return _manifest_path(config, "report")


def status(config: ExploreConfig) -> dict[str, object]:
    stages = {}
    for stage in STAGES:
        path = _manifest_path(config, stage)
        if not path.is_file():
            stages[stage] = "missing"
            continue
        try:
            _load_manifest(config, stage)
            stages[stage] = "completed"
        except ValueError as exc:
            stages[stage] = f"stale: {exc}"
    return {
        "output": str(config.output),
        "smoke": config.smoke,
        "new_teacher_shards": config.expected_new_shards,
        "new_models": config.expected_new_models,
        "reported_cells": config.expected_reported_cells,
        "new_cells": config.expected_new_cells,
        "stages": stages,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(*STAGES, "run", "status"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    config = smoke_config() if args.smoke else ExploreConfig()
    if args.command == "status":
        print(json.dumps(status(config), indent=2))
        return 0
    path = run_all(config) if args.command == "run" else run_stage(
        config, args.command
    )
    print(f"exploratory clean scaling complete: {path}")
    if args.command in {"run", "report"}:
        manifest = load_json(_manifest_path(config, "report"))
        return 2 if int(manifest.get("failure_count", 0)) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
