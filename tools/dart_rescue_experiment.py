#!/usr/bin/env python3
"""Versioned Clean20/Clean36/DART36 rescue experiment.

The mathematical comparison is paired at each exact transform bound:

* Clean36 - Clean20 isolates the effect of sixteen additional clean rollouts.
* DART36 - Clean36 isolates guarded DART on the same sixteen base paths.
* DART36 - Clean20 measures the total retrospective/confirmatory rescue.

Guarded DART fits one categorical disturbance calibration per budget from the
matching frozen Clean20 specialist's meaningful validation errors. A disturbed
action is restricted to the Q90 normalized-regret radius and is followed by one
forced teacher recovery decision.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import sys
from time import perf_counter
from typing import IO, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from pzr.artifact_io import write_csv_atomic, write_json_atomic
from pzr.learning.artifacts import load_reducer_cost_dataset
from pzr.learning.dart import (
    DartCalibration,
    DartCalibrationConfig,
    calibrate_dart,
)
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
    filter_training_budgets,
    run_reducer_training,
)
from pzr.rtlola.binding import (
    BINDING_BUILD_PROFILE,
    BINDING_REVISION,
    INTERPRETER_REVISION,
)
from pzr.rtlola.features import RTL_RANKING_FEATURE_SCHEMA
from pzr.rtlola.learning_collection import (
    LearningCollectionConfig,
    feature_schema_payload,
    run_learning_collection,
)
from pzr.rtlola.learning_traces import (
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
    _fixed_figure8_traces,
    _run_cell as run_paper_cell,
)
from pzr.rtlola.reference import REFERENCE_CACHE_SCHEMA, load_or_compute_reference
from pzr.rtlola.robot_arm import (
    RLOLAEVAL_REVISION,
    ROBOT_ARM_SPEC_SHA256,
)
from pzr.rtlola.robot_arm_random import RANDOM_WAYPOINT_SOURCE_REVISION
from pzr.rtlola.scenarios import scenario_by_name


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "experiments" / "dart_rescue_v1.yaml"
CONFIG_SCHEMA = "pzr.dart-rescue-config.v1"
STAGE_SCHEMA = "pzr.dart-rescue-stage.v1"
CELL_SCHEMA = "pzr.dart-rescue-cell.v1"
VALIDATION_SCHEMA = "pzr.dart-rescue-validation.v1"
METHODS = ("clean20", "clean36", "dart36")
COMPARISONS = (
    ("dart_effect", "dart36", "clean36"),
    ("data_scale", "clean36", "clean20"),
    ("total_rescue", "dart36", "clean20"),
)
STAGES = (
    "preflight",
    "prepare",
    "calibrate",
    "collect",
    "train",
    "evaluate",
    "report",
    "validate",
)
RUN_STATES = {item.value for item in RunState}
SMOKE_EVENT_COUNT = 100


@dataclass(frozen=True)
class DartRescueConfig:
    source: Path
    raw_config_sha256: str
    schema: str
    experiment_id: str
    output_root: Path
    base_paper_config: Path
    base_paper_output_root: Path
    base_paper_config_sha256: str
    expected_pzr_source_sha256: str
    teacher_dataset: Path
    teacher_dataset_sha256: str
    event_count: int
    budgets: tuple[int, ...]
    candidate_names: tuple[str, ...]
    clean_train_seeds: tuple[int, ...]
    clean_validation_seeds: tuple[int, ...]
    extra_training_seeds: tuple[int, ...]
    replay_seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    fixed_figure8_trace_kinds: tuple[str, ...]
    collection_workers: int
    evaluation_workers: int
    training_epochs: int
    training_batch_size: int
    training_learning_rate: float
    training_weight_decay: float
    training_patience: int
    training_seed: int
    dart_config: DartCalibrationConfig
    disturbance_seed: int
    bootstrap_replicates: int
    bootstrap_seed: int
    instability_budgets: tuple[int, ...]
    smoke: bool = False

    def __post_init__(self) -> None:
        if self.schema != CONFIG_SCHEMA:
            raise ValueError(f"unsupported DART rescue config schema: {self.schema}")
        if not self.budgets or len(set(self.budgets)) != len(self.budgets):
            raise ValueError("DART rescue budgets must be non-empty and unique")
        if not self.candidate_names or len(set(self.candidate_names)) != len(
            self.candidate_names
        ):
            raise ValueError("DART rescue candidates must be non-empty and unique")
        seed_groups = {
            "clean_train": set(self.clean_train_seeds),
            "clean_validation": set(self.clean_validation_seeds),
            "extra_training": set(self.extra_training_seeds),
            "replay": set(self.replay_seeds),
            "confirmation": set(self.confirmation_seeds),
        }
        for left, left_values in seed_groups.items():
            for right, right_values in seed_groups.items():
                if left < right and left_values & right_values:
                    raise ValueError(f"DART rescue seed groups overlap: {left} and {right}")
        if min(self.collection_workers, self.evaluation_workers) < 1:
            raise ValueError("DART rescue workers must be positive")
        if self.bootstrap_replicates < 1:
            raise ValueError("DART rescue bootstrap replicate count must be positive")
        if not set(self.instability_budgets) <= set(self.budgets):
            raise ValueError("instability budgets must be evaluated budgets")
        if not self.smoke:
            expected = {
                "budgets": (40, 80, 120, 150, 200, 250, 500),
                "train": tuple(range(20)),
                "validation": tuple(range(20, 26)),
                "extra": tuple(range(26, 42)),
                "replay": tuple(range(100, 120)),
                "confirmation": tuple(range(120, 140)),
                "fixed": (
                    "figure8",
                    "figure8_drift",
                    "figure8_geofence",
                    "figure8_drift_geofence",
                ),
            }
            actual = {
                "budgets": self.budgets,
                "train": self.clean_train_seeds,
                "validation": self.clean_validation_seeds,
                "extra": self.extra_training_seeds,
                "replay": self.replay_seeds,
                "confirmation": self.confirmation_seeds,
                "fixed": self.fixed_figure8_trace_kinds,
            }
            if actual != expected:
                raise ValueError(f"canonical DART rescue scope differs: {actual}")
            if self.event_count != 500:
                raise ValueError("canonical DART rescue traces must have 500 events")
            if (self.collection_workers, self.evaluation_workers) != (10, 10):
                raise ValueError("canonical DART rescue worker contract differs")

    @property
    def effective_config_sha256(self) -> str:
        payload = asdict(self)
        payload["source"] = str(self.source.resolve())
        payload["output_root"] = str(self.output_root.resolve())
        payload["base_paper_config"] = str(self.base_paper_config.resolve())
        payload["base_paper_output_root"] = str(self.base_paper_output_root.resolve())
        payload["teacher_dataset"] = str(self.teacher_dataset.resolve())
        return payload_sha256(payload)

    @property
    def expected_clean_shards(self) -> int:
        return len(self.extra_training_seeds) * len(self.budgets)

    @property
    def expected_dart_shards(self) -> int:
        return len(self.extra_training_seeds) * len(self.budgets)

    @property
    def expected_new_models(self) -> int:
        return 2 * len(self.budgets)

    def reported_cells(self, scope: str) -> int:
        traces = {
            "replay": len(self.replay_seeds),
            "confirmation": len(self.confirmation_seeds),
            "fixed": len(self.fixed_figure8_trace_kinds),
        }
        if scope not in traces:
            raise ValueError(f"unknown DART rescue scope: {scope}")
        return traces[scope] * len(self.budgets) * len(METHODS)

    def new_cells(self, scope: str) -> int:
        if self.smoke:
            return self.reported_cells(scope)
        imported = 1 if scope in {"replay", "fixed"} else 0
        return self.reported_cells(scope) - (
            imported * self.reported_cells(scope) // len(METHODS)
        )


def load_config(path: Path, *, smoke: bool = False) -> DartRescueConfig:
    raw_bytes = path.read_bytes()
    raw = yaml.safe_load(raw_bytes)
    base = raw["base_paper"]
    seeds = raw["seeds"]
    training = raw["training"]
    dart = raw["dart"]
    reporting = raw["reporting"]
    config = DartRescueConfig(
        source=path,
        raw_config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        schema=str(raw["schema"]),
        experiment_id=str(raw["experiment_id"]),
        output_root=Path(raw["output_root"]),
        base_paper_config=Path(base["config"]),
        base_paper_output_root=Path(base["output_root"]),
        base_paper_config_sha256=str(base["config_sha256"]),
        expected_pzr_source_sha256=str(base["pzr_source_sha256"]),
        teacher_dataset=Path(base["teacher_dataset"]),
        teacher_dataset_sha256=str(base["teacher_dataset_sha256"]),
        event_count=int(raw["event_count"]),
        budgets=tuple(int(value) for value in raw["budgets"]),
        candidate_names=tuple(str(value) for value in raw["candidate_names"]),
        clean_train_seeds=tuple(int(value) for value in seeds["clean_train"]),
        clean_validation_seeds=tuple(
            int(value) for value in seeds["clean_validation"]
        ),
        extra_training_seeds=tuple(int(value) for value in seeds["extra_training"]),
        replay_seeds=tuple(int(value) for value in seeds["retrospective_replay"]),
        confirmation_seeds=tuple(
            int(value) for value in seeds["untouched_confirmation"]
        ),
        fixed_figure8_trace_kinds=tuple(str(value) for value in raw["fixed_figure8"]),
        collection_workers=int(raw["workers"]["collection"]),
        evaluation_workers=int(raw["workers"]["evaluation"]),
        training_epochs=int(training["epochs"]),
        training_batch_size=int(training["batch_size"]),
        training_learning_rate=float(training["learning_rate"]),
        training_weight_decay=float(training["weight_decay"]),
        training_patience=int(training["patience"]),
        training_seed=int(training["seed"]),
        dart_config=DartCalibrationConfig(
            regret_cap_quantile=float(dart["regret_cap_quantile"]),
            direction_pseudocount=float(dart["direction_pseudocount"]),
            recovery_decisions=int(dart["recovery_decisions"]),
        ),
        disturbance_seed=int(dart["disturbance_seed"]),
        bootstrap_replicates=int(reporting["bootstrap_replicates"]),
        bootstrap_seed=int(reporting["bootstrap_seed"]),
        instability_budgets=tuple(int(value) for value in reporting["instability_budgets"]),
    )
    if not smoke:
        return config
    return replace(
        config,
        output_root=Path("/tmp/pzr-dart-rescue-v1-smoke"),
        event_count=SMOKE_EVENT_COUNT,
        budgets=(40,),
        extra_training_seeds=(26,),
        replay_seeds=(100,),
        confirmation_seeds=(120,),
        fixed_figure8_trace_kinds=("figure8",),
        collection_workers=1,
        evaluation_workers=1,
        bootstrap_replicates=200,
        instability_budgets=(40,),
        smoke=True,
    )


def tool_sha256() -> str:
    return sha256_files(
        (
            Path(__file__).resolve(),
            REPOSITORY_ROOT / "tools" / "run_dart_rescue.sh",
        ),
        relative_to=REPOSITORY_ROOT,
    )


def _stage_manifest_base(config: DartRescueConfig, stage: str) -> dict[str, object]:
    return {
        "schema": STAGE_SCHEMA,
        "experiment_id": config.experiment_id,
        "stage": stage,
        "status": "completed",
        "raw_config_sha256": config.raw_config_sha256,
        "effective_config_sha256": config.effective_config_sha256,
        "tool_sha256": tool_sha256(),
        "pzr_source_sha256": pzr_source_sha256(),
        "binding_revision": BINDING_REVISION,
        "interpreter_revision": INTERPRETER_REVISION,
        "binding_build_profile": BINDING_BUILD_PROFILE,
        "smoke": config.smoke,
    }


def _write_stage_manifest(
    config: DartRescueConfig,
    stage: str,
    extra: Mapping[str, object],
) -> Path:
    path = config.output_root / stage / "manifest.json"
    write_json_atomic({**_stage_manifest_base(config, stage), **dict(extra)}, path)
    return path


def _validate_stage_manifest(config: DartRescueConfig, stage: str) -> dict[str, object]:
    path = config.output_root / stage / "manifest.json"
    manifest = load_json(path)
    expected = _stage_manifest_base(config, stage)
    mismatched = [
        name for name, value in expected.items() if manifest.get(name) != value
    ]
    if mismatched:
        raise ValueError(
            f"stale {stage} DART rescue manifest: {', '.join(sorted(mismatched))}"
        )
    return manifest


def _paper_model_records(config: DartRescueConfig) -> dict[int, dict[str, object]]:
    manifest = load_json(config.base_paper_output_root / "train" / "manifest.json")
    if manifest.get("config_sha256") != config.base_paper_config_sha256:
        raise ValueError("base paper train config hash differs")
    if manifest.get("pzr_source_sha256") != config.expected_pzr_source_sha256:
        raise ValueError("base paper train source hash differs")
    records: dict[int, dict[str, object]] = {}
    for budget in config.budgets:
        record = dict(manifest["models_by_budget"][str(budget)])
        path = Path(str(record["path"]))
        if record.get("budget_filter") != [budget] or record.get(
            "training_budget"
        ) != budget:
            raise ValueError(f"base Clean20 specialist budget differs: {budget}")
        if model_sha256(path) != record.get("sha256"):
            raise ValueError(f"base Clean20 specialist hash differs: {budget}")
        records[budget] = {**record, "path": str(path)}
    return records


def run_preflight(config: DartRescueConfig) -> Path:
    if hashlib.sha256(config.base_paper_config.read_bytes()).hexdigest() != (
        config.base_paper_config_sha256
    ):
        raise ValueError("base paper config hash differs")
    if pzr_source_sha256() != config.expected_pzr_source_sha256:
        raise ValueError(
            "PZR source differs from the completed paper evaluation; preserve or "
            "explicitly version the new scientific core before running DART"
        )
    if BINDING_BUILD_PROFILE != "release":
        raise ValueError("DART rescue requires the release binding")
    if dataset_sha256(config.teacher_dataset) != config.teacher_dataset_sha256:
        raise ValueError("base teacher dataset hash differs")
    teacher_manifest = load_json(config.teacher_dataset / "manifest.json")
    if teacher_manifest.get("budgets") != list((40, 80, 120, 150, 200, 250, 500)):
        raise ValueError("base teacher dataset budgets differ")
    if teacher_manifest.get("collection_mode") != "teacher":
        raise ValueError("base teacher dataset is not clean teacher data")
    if teacher_manifest.get("event_count") != 500:
        raise ValueError("base teacher dataset event count differs")
    _paper_model_records(config)
    paper_inputs = {}
    for stage in ("generalization", "headline", "science-validate"):
        path = config.base_paper_output_root / stage / "manifest.json"
        manifest = load_json(path)
        if manifest.get("config_sha256") != config.base_paper_config_sha256:
            raise ValueError(f"base paper {stage} config hash differs")
        if manifest.get("pzr_source_sha256") != config.expected_pzr_source_sha256:
            raise ValueError(f"base paper {stage} source hash differs")
        paper_inputs[stage] = sha256_files((path,))
    return _write_stage_manifest(config, "preflight", {
        "teacher_dataset_sha256": config.teacher_dataset_sha256,
        "base_paper_manifest_sha256": paper_inputs,
        "base_model_count": len(config.budgets),
        "release_tests_reused_from": str(
            config.base_paper_output_root / "preflight" / "manifest.json"
        ),
    })


def run_prepare(config: DartRescueConfig) -> Path:
    root = config.output_root / "prepare"
    trace_root = root / "traces-extra-clean16"
    store = generate_random_waypoint_trace_store(RandomWaypointTraceStoreConfig(
        output=trace_root,
        event_count=config.event_count,
        conditions=("random_waypoint",),
        seed_start=min(config.extra_training_seeds),
        seed_count=len(config.extra_training_seeds),
    ))
    if tuple(item.seed for item in store.traces) != config.extra_training_seeds:
        raise ValueError("extra clean trace seed coverage differs")
    clean_dataset = run_learning_collection(LearningCollectionConfig(
        output=root / "extra-clean",
        trace_store=trace_root,
        budgets=config.budgets,
        candidate_names=config.candidate_names,
        train_seeds=len(config.extra_training_seeds),
        validation_seeds=0,
        test_seeds=0,
        seed_start=min(config.extra_training_seeds),
        workers=config.collection_workers,
        collection_mode="teacher",
    ))
    clean_manifest = load_json(clean_dataset / "manifest.json")
    if clean_manifest.get("shard_count") != config.expected_clean_shards:
        raise ValueError("extra clean shard count differs")
    return _write_stage_manifest(config, "prepare", {
        "trace_store": str(store.root),
        "trace_store_manifest_sha256": store.manifest_sha256,
        "extra_clean_dataset": str(clean_dataset),
        "extra_clean_dataset_sha256": dataset_sha256(clean_dataset),
        "clean_shard_count": config.expected_clean_shards,
        "extra_training_seeds": list(config.extra_training_seeds),
        "trace_kind": "random_waypoint",
        "event_count": config.event_count,
    })


def _calibration_hash(path: Path) -> str:
    return sha256_files((
        path / "calibration.json",
        path / "dart_budget_calibration.csv",
        path / "dart_direction_kernel.csv",
    ))


def run_calibrate(config: DartRescueConfig) -> Path:
    root = config.output_root / "calibrate"
    dataset, metadata, manifest = load_reducer_cost_dataset(config.teacher_dataset)
    records = _paper_model_records(config)
    calibrations = {}
    for budget in config.budgets:
        filtered, filtered_metadata = filter_training_budgets(
            dataset, metadata, (budget,),
        )
        policy = ReducerPolicy.load(Path(str(records[budget]["path"])))
        context = {
            "model_sha256": str(records[budget]["sha256"]),
            "dataset_name": "terminal_full_width_teacher",
            "dataset_sha256": config.teacher_dataset_sha256,
            "split": "validation",
            "calibration_budget": budget,
            "candidate_names": list(filtered.candidate_names),
            "feature_schema": feature_schema_payload(),
            "cost_contract": manifest["cost_contract"],
            "binding_revision": BINDING_REVISION,
            "interpreter_revision": INTERPRETER_REVISION,
            "binding_build_profile": BINDING_BUILD_PROFILE,
            "pzr_source_sha256": pzr_source_sha256(),
            "dart_rescue_config_sha256": config.effective_config_sha256,
            "dart_rescue_tool_sha256": tool_sha256(),
        }
        calibration, budget_diagnostics, direction_diagnostics = calibrate_dart(
            policy,
            filtered,
            filtered_metadata,
            split="validation",
            context=context,
            config=config.dart_config,
        )
        if calibration.budgets != (budget,):
            raise ValueError(f"DART calibration is not budget-specialized: {budget}")
        output = root / f"budget-{budget}"
        calibration.save(output, budget_diagnostics, direction_diagnostics)
        from pzr.learning.reporting import write_dart_calibration_plot

        write_dart_calibration_plot(
            budget_diagnostics,
            direction_diagnostics,
            output / "dart_calibration.png",
        )
        calibrations[str(budget)] = {
            "path": str(output),
            "sha256": _calibration_hash(output),
            "model_sha256": records[budget]["sha256"],
            "validation_sample_count": filtered.indices_for_split(
                "validation"
            ).size,
            "budget": budget,
        }
    return _write_stage_manifest(config, "calibrate", {
        "calibration_count": len(calibrations),
        "calibrations_by_budget": calibrations,
        "calibration_contract": asdict(config.dart_config),
        "disturbance_seed": config.disturbance_seed,
    })


def run_collect(config: DartRescueConfig) -> Path:
    prepare = _validate_stage_manifest(config, "prepare")
    calibrate = _validate_stage_manifest(config, "calibrate")
    root = config.output_root / "collect"
    datasets = {}
    total_shards = 0
    for budget in config.budgets:
        calibration_path = Path(
            calibrate["calibrations_by_budget"][str(budget)]["path"]
        )
        dataset_path = run_learning_collection(LearningCollectionConfig(
            output=root / f"budget-{budget}",
            trace_store=Path(str(prepare["trace_store"])),
            budgets=(budget,),
            candidate_names=config.candidate_names,
            train_seeds=len(config.extra_training_seeds),
            validation_seeds=0,
            test_seeds=0,
            seed_start=min(config.extra_training_seeds),
            workers=config.collection_workers,
            collection_mode="dart",
            dart_calibration=calibration_path,
            disturbance_seed=config.disturbance_seed,
        ))
        manifest = load_json(dataset_path / "manifest.json")
        expected = len(config.extra_training_seeds)
        if manifest.get("shard_count") != expected:
            raise ValueError(f"DART shard count differs at budget {budget}")
        total_shards += expected
        datasets[str(budget)] = {
            "path": str(dataset_path),
            "sha256": dataset_sha256(dataset_path),
            "shard_count": expected,
            "sample_count": int(manifest["num_samples"]),
            "calibration_sha256": manifest["dart_calibration_sha256"],
        }
    if total_shards != config.expected_dart_shards:
        raise ValueError("total DART shard count differs")
    return _write_stage_manifest(config, "collect", {
        "datasets_by_budget": datasets,
        "dart_shard_count": total_shards,
        "disturbance_seed": config.disturbance_seed,
    })


def _training_common(config: DartRescueConfig) -> dict[str, object]:
    return {
        "objective": "pairwise",
        "epochs": config.training_epochs,
        "batch_size": config.training_batch_size,
        "learning_rate": config.training_learning_rate,
        "weight_decay": config.training_weight_decay,
        "patience": config.training_patience,
        "seed": config.training_seed,
    }


def run_train(config: DartRescueConfig) -> Path:
    prepare = _validate_stage_manifest(config, "prepare")
    collected = _validate_stage_manifest(config, "collect")
    root = config.output_root / "train"
    models: dict[str, dict[str, object]] = {}
    common = _training_common(config)
    for budget in config.budgets:
        variants = {
            "clean36": NamedDataset(
                "extra_clean16", Path(str(prepare["extra_clean_dataset"]))
            ),
            "dart36": NamedDataset(
                "extra_dart16",
                Path(str(collected["datasets_by_budget"][str(budget)]["path"])),
            ),
        }
        for variant, extra in variants.items():
            output = run_reducer_training(ReducerTrainingConfig(
                datasets=(
                    NamedDataset("primary_clean20", config.teacher_dataset),
                    extra,
                ),
                output=root / variant / f"model-budget-{budget}",
                budget_filter=(budget,),
                **common,
            ))
            training = load_json(output / "training.json")
            if training.get("budget_filter") != [budget] or training.get(
                "training_budgets"
            ) != [budget]:
                raise ValueError(f"{variant} training budget differs: {budget}")
            if training.get("validation_seeds") != list(
                config.clean_validation_seeds
            ):
                raise ValueError(f"{variant} validation seeds differ: {budget}")
            models[f"{variant}:{budget}"] = {
                "variant": variant,
                "budget": budget,
                "path": str(output),
                "sha256": model_sha256(output),
                "budget_filter": [budget],
                "datasets": training["datasets"],
            }
    if len(models) != config.expected_new_models:
        raise ValueError("new DART rescue model count differs")
    return _write_stage_manifest(config, "train", {
        "models": models,
        "new_model_count": len(models),
        "shared_hyperparameters": {
            key: value for key, value in common.items() if key != "objective"
        },
        "base_clean20_models": _paper_model_records(config),
    })


def _generated_traces(
    config: DartRescueConfig,
    *,
    scope: str,
    seeds: Sequence[int],
) -> tuple[EvaluationTrace, ...]:
    if scope == "replay" and not config.smoke:
        path = (
            config.base_paper_output_root
            / "generalization" / "traces" / "generated-nominal"
        )
    else:
        path = config.output_root / "evaluate" / scope / "traces" / "generated-nominal"
        generate_random_waypoint_trace_store(RandomWaypointTraceStoreConfig(
            output=path,
            event_count=config.event_count,
            conditions=("random_waypoint",),
            seed_start=min(seeds),
            seed_count=len(seeds),
        ))
    store = load_random_waypoint_trace_store(path)
    if store.event_count != config.event_count or store.conditions != (
        "random_waypoint",
    ):
        raise ValueError(f"{scope} nominal trace-store contract differs")
    if tuple(item.seed for item in store.traces) != tuple(seeds):
        raise ValueError(f"{scope} nominal trace seeds differ")
    traces = tuple(EvaluationTrace(
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
    ) for item in store.traces)
    if scope == "replay" and not config.smoke:
        paper_manifest = load_json(
            config.base_paper_output_root / "generalization" / "manifest.json"
        )
        expected = {
            str(item["trace_id"]): str(item["trace_sha256"])
            for item in paper_manifest["trace_manifest"]
        }
        actual = {item.trace_id: item.trace_sha256 for item in traces}
        if actual != expected:
            raise ValueError("replay trace hashes differ from the paper evaluation")
    return traces


def _fixed_traces(config: DartRescueConfig) -> tuple[EvaluationTrace, ...]:
    from pzr.rtlola.paper_experiment import load_paper_experiment_config

    paper_config = load_paper_experiment_config(config.base_paper_config)
    traces = tuple(
        item for item in _fixed_figure8_traces(paper_config)
        if item.trace_kind in config.fixed_figure8_trace_kinds
    )
    if not config.smoke:
        return traces
    return tuple(replace(item, events=item.events[:config.event_count]) for item in traces)


def _reference_path(
    config: DartRescueConfig,
    scope: str,
    trace: EvaluationTrace,
) -> Path:
    if scope == "replay" and not config.smoke:
        path = (
            config.base_paper_output_root / "generalization" / "references"
            / f"random_waypoint_seed-{trace.seed}.json"
        )
    elif scope == "fixed" and not config.smoke:
        path = (
            config.base_paper_output_root / "headline" / "references"
            / f"{trace.trace_kind}.json"
        )
    else:
        path = (
            config.output_root / "evaluate" / scope / "references"
            / f"{trace.trace_kind}_seed-{trace.seed}.json"
        )
        load_or_compute_reference(
            trace.events,
            scenario=scenario_by_name("robot_arm"),
            trace_kind=trace.trace_id,
            seed=trace.seed,
            cache_path=path,
            include_approximation=True,
        )
    if not path.is_file():
        raise ValueError(f"missing exact reference cache: {path}")
    return path


def _method_config(config: DartRescueConfig, name: str) -> MethodConfig:
    return MethodConfig(
        name=name,
        execution_regime=ExecutionRegime.LEARNED_ONLINE,
        predictor=Predictor.NONE,
        horizon=0,
        beam_width=1,
        objective=Objective.LEARNED_TERMINAL_TEACHER,
        candidate_names=config.candidate_names,
    )


def _model_records_by_method(
    config: DartRescueConfig,
) -> dict[tuple[str, int], tuple[Path, str]]:
    trained = _validate_stage_manifest(config, "train")
    records: dict[tuple[str, int], tuple[Path, str]] = {}
    for budget, record in _paper_model_records(config).items():
        records[("clean20", budget)] = (
            Path(str(record["path"])), str(record["sha256"])
        )
    for key, record in trained["models"].items():
        variant, raw_budget = key.split(":")
        records[(variant, int(raw_budget))] = (
            Path(str(record["path"])), str(record["sha256"])
        )
    return records


def rescue_cell_identity(
    config: DartRescueConfig,
    *,
    scope: str,
    trace: EvaluationTrace,
    budget: int,
    method: MethodConfig,
    reference_path: Path,
    model_hash: str,
) -> dict[str, object]:
    payload = {
        "schema": CELL_SCHEMA,
        "experiment_id": config.experiment_id,
        "scope": scope,
        "trace_id": trace.trace_id,
        "trace_sha256": trace.trace_sha256,
        "trace_source": trace.trace_source.value,
        "trace_kind": trace.trace_kind,
        "trace_provenance": dict(trace.provenance),
        "condition": trace.condition,
        "seed": trace.seed,
        "event_count": len(trace.events),
        "budget": budget,
        "method": {
            **asdict(method),
            "execution_regime": method.execution_regime.value,
            "predictor": method.predictor.value,
            "objective": method.objective.value,
            "candidate_names": list(method.candidate_names),
        },
        "model_sha256": model_hash,
        "model_training_budget": budget,
        "spec_sha256": ROBOT_ARM_SPEC_SHA256,
        "rlolaeval_revision": RLOLAEVAL_REVISION,
        "binding_revision": BINDING_REVISION,
        "interpreter_revision": INTERPRETER_REVISION,
        "binding_build_profile": BINDING_BUILD_PROFILE,
        "reference_cache_schema": REFERENCE_CACHE_SCHEMA,
        "reference_cache_sha256": sha256_files((reference_path,)),
        "reference_semantics": {
            "selection": "learned_direct_inference",
            "metrics": "exact_cache_dynamic_and_total_radius",
        },
        "seed_lists": {
            "clean_train": list(config.clean_train_seeds),
            "clean_validation": list(config.clean_validation_seeds),
            "extra_training": list(config.extra_training_seeds),
            "replay": list(config.replay_seeds),
            "confirmation": list(config.confirmation_seeds),
        },
        "effective_config_sha256": config.effective_config_sha256,
        "tool_sha256": tool_sha256(),
        "pzr_source_sha256": pzr_source_sha256(),
    }
    return {**payload, "fingerprint": payload_sha256(payload)}


def _execute_rescue_cell(job: EvaluationCellJob) -> dict[str, object]:
    manifest_path = job.directory / "manifest.json"
    summary_path = job.directory / "summary.csv"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        if manifest.get("schema") != CELL_SCHEMA:
            raise ValueError(f"unsupported DART rescue cell schema: {job.directory}")
        if manifest.get("identity") != job.identity:
            raise ValueError(f"stale DART rescue cell: {job.directory}")
        frame = pd.read_csv(summary_path)
        if len(frame) != 1:
            raise ValueError(f"DART rescue cell summary is not unique: {job.directory}")
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
    elapsed_ms = (perf_counter() - started) * 1000.0
    row["cell_elapsed_ms"] = elapsed_ms
    write_csv_atomic(pd.DataFrame([row]), summary_path)
    write_json_atomic({
        "schema": CELL_SCHEMA,
        "identity": job.identity,
        "status": row["status"],
        "cell_elapsed_ms": elapsed_ms,
        "diagnostic": diagnostic,
    }, manifest_path)
    return row


def _import_paper_clean20(
    config: DartRescueConfig,
    scope: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    stage = "generalization" if scope == "replay" else "headline"
    directory = config.base_paper_output_root / stage
    manifest_path = directory / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("config_sha256") != config.base_paper_config_sha256:
        raise ValueError(f"imported {scope} paper config differs")
    if manifest.get("pzr_source_sha256") != config.expected_pzr_source_sha256:
        raise ValueError(f"imported {scope} paper source differs")
    summary = pd.read_csv(directory / "summary.csv")
    summary = summary[summary["method"] == "pairwise_ranking_policy"].copy()
    timeseries = pd.read_csv(directory / "timeseries.csv")
    timeseries = timeseries[
        timeseries["method"] == "pairwise_ranking_policy"
    ].copy()
    summary["method"] = "clean20"
    timeseries["method"] = "clean20"
    expected = config.reported_cells(scope) // len(METHODS)
    if len(summary) != expected:
        raise ValueError(f"imported {scope} Clean20 cell count differs")
    if not (
        summary["model_training_budget"].astype(int)
        == summary["budget"].astype(int)
    ).all():
        raise ValueError(f"imported {scope} Clean20 budgets are not matched")
    return summary, timeseries, {
        "stage": stage,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_files((manifest_path,)),
        "summary_sha256": sha256_files((directory / "summary.csv",)),
        "timeseries_sha256": sha256_files((directory / "timeseries.csv",)),
        "cell_count": len(summary),
    }


def _run_scope(
    config: DartRescueConfig,
    *,
    scope: str,
    traces: Sequence[EvaluationTrace],
    methods_to_run: Sequence[str],
) -> dict[str, object]:
    root = config.output_root / "evaluate" / scope
    models = _model_records_by_method(config)
    references = {
        trace.trace_id: _reference_path(config, scope, trace) for trace in traces
    }
    jobs = []
    for trace in traces:
        for budget in config.budgets:
            for name in methods_to_run:
                path, model_hash = models[(name, budget)]
                method = _method_config(config, name)
                identity = rescue_cell_identity(
                    config,
                    scope=scope,
                    trace=trace,
                    budget=budget,
                    method=method,
                    reference_path=references[trace.trace_id],
                    model_hash=model_hash,
                )
                jobs.append(EvaluationCellJob(
                    stage="generalization" if scope != "fixed" else "headline",
                    directory=(
                        root / "cells" / trace.trace_source.value / trace.condition
                        / f"seed-{trace.seed}" / f"budget-{budget}" / name
                    ),
                    trace=trace,
                    budget=budget,
                    method=method,
                    runtime_method="pairwise_ranking_policy",
                    reference_path=references[trace.trace_id],
                    identity=identity,
                    model_directory=path,
                    model_training_budget=budget,
                ))
    started = perf_counter()
    if config.evaluation_workers == 1:
        rows = [_execute_rescue_cell(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=config.evaluation_workers,
            mp_context=get_context("spawn"),
            max_tasks_per_child=1,
        ) as executor:
            rows = list(executor.map(_execute_rescue_cell, jobs))
    wall_seconds = perf_counter() - started
    summary = pd.DataFrame(rows)
    series = []
    for job in jobs:
        path = job.directory / "timeseries_diagnostic.csv"
        if path.is_file():
            series.append(pd.read_csv(path))
    timeseries = pd.concat(series, ignore_index=True) if series else pd.DataFrame()
    imported = None
    if not config.smoke and scope in {"replay", "fixed"}:
        base_summary, base_timeseries, imported = _import_paper_clean20(config, scope)
        summary = pd.concat([base_summary, summary], ignore_index=True)
        timeseries = pd.concat([base_timeseries, timeseries], ignore_index=True)
    if len(summary) != config.reported_cells(scope):
        raise ValueError(f"{scope} reported DART rescue cell count differs")
    keys = ["trace_id", "budget", "method"]
    if summary.duplicated(keys).any():
        raise ValueError(f"{scope} DART rescue contains duplicate cells")
    if set(summary["method"]) != set(METHODS):
        raise ValueError(f"{scope} DART rescue method set differs")
    if not (
        summary["model_training_budget"].astype(int)
        == summary["budget"].astype(int)
    ).all():
        raise ValueError(f"{scope} DART rescue model budgets are not matched")
    write_csv_atomic(summary, root / "summary.csv")
    write_csv_atomic(timeseries, root / "timeseries.csv")
    write_json_atomic({
        "schema": STAGE_SCHEMA,
        "scope": scope,
        "effective_config_sha256": config.effective_config_sha256,
        "tool_sha256": tool_sha256(),
        "pzr_source_sha256": pzr_source_sha256(),
        "reported_cell_count": len(summary),
        "new_cell_count": len(jobs),
        "failure_count": int((summary["status"] != "completed").sum()),
        "workers": config.evaluation_workers,
        "matrix_wall_seconds": wall_seconds,
        "methods": list(METHODS),
        "trace_manifest": [
            {
                "trace_id": trace.trace_id,
                "trace_sha256": trace.trace_sha256,
                "trace_source": trace.trace_source.value,
                "trace_kind": trace.trace_kind,
                "condition": trace.condition,
                "seed": trace.seed,
                "event_count": len(trace.events),
                "provenance": dict(trace.provenance),
                "reference_sha256": sha256_files((references[trace.trace_id],)),
            }
            for trace in traces
        ],
        "imported_clean20": imported,
    }, root / "manifest.json")
    return {
        "scope": scope,
        "reported_cell_count": len(summary),
        "new_cell_count": len(jobs),
        "failure_count": int((summary["status"] != "completed").sum()),
        "manifest": str(root / "manifest.json"),
        "manifest_sha256": sha256_files((root / "manifest.json",)),
        "matrix_wall_seconds": wall_seconds,
    }


def run_evaluate(config: DartRescueConfig) -> Path:
    replay = _generated_traces(config, scope="replay", seeds=config.replay_seeds)
    confirmation = _generated_traces(
        config, scope="confirmation", seeds=config.confirmation_seeds
    )
    fixed = _fixed_traces(config)
    methods_replay = METHODS if config.smoke else ("clean36", "dart36")
    methods_fixed = METHODS if config.smoke else ("clean36", "dart36")
    scopes = {
        "replay": _run_scope(
            config, scope="replay", traces=replay, methods_to_run=methods_replay
        ),
        "confirmation": _run_scope(
            config,
            scope="confirmation",
            traces=confirmation,
            methods_to_run=METHODS,
        ),
        "fixed": _run_scope(
            config, scope="fixed", traces=fixed, methods_to_run=methods_fixed
        ),
    }
    total_reported = sum(int(item["reported_cell_count"]) for item in scopes.values())
    total_new = sum(int(item["new_cell_count"]) for item in scopes.values())
    expected_reported = sum(config.reported_cells(scope) for scope in scopes)
    expected_new = sum(config.new_cells(scope) for scope in scopes)
    if (total_reported, total_new) != (expected_reported, expected_new):
        raise ValueError("DART rescue evaluation totals differ")
    return _write_stage_manifest(config, "evaluate", {
        "scopes": scopes,
        "reported_cell_count": total_reported,
        "new_cell_count": total_new,
        "failure_count": sum(int(item["failure_count"]) for item in scopes.values()),
    })


def aggregate_scope(
    summary: pd.DataFrame,
    *,
    scope: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    data = trace_level_metrics(summary)
    rng = np.random.default_rng(bootstrap_seed)
    rows = []
    group_keys = ["trace_source", "trace_kind", "condition", "budget", "method"]
    if scope != "fixed":
        group_keys = ["trace_source", "trace_kind", "budget", "method"]
    for keys, frame in data.groupby(group_keys, sort=True):
        identity = dict(zip(group_keys, keys if isinstance(keys, tuple) else (keys,)))
        valid = frame["status"] == "completed"
        failures = int((~valid).sum())
        completed = frame[valid]
        fpr = completed["fpr"].to_numpy(dtype=np.float64)
        loss = completed["mean_approx_loss"].to_numpy(dtype=np.float64)
        main_available = failures == 0 and len(completed) == len(frame)
        if main_available and len(fpr):
            indices = rng.integers(
                0, len(fpr), size=(bootstrap_replicates, len(fpr))
            )
            sampled = np.mean(fpr[indices], axis=1)
            fpr_low, fpr_high = np.quantile(sampled, (0.025, 0.975))
        else:
            fpr_low = fpr_high = np.nan
        rows.append({
            "scope": scope,
            **identity,
            "trace_count": len(frame),
            "valid_count": len(completed),
            "failed_count": failures,
            "main_available": main_available,
            "macro_fpr": float(np.mean(fpr)) if main_available else np.nan,
            "macro_fpr_ci_low": float(fpr_low),
            "macro_fpr_ci_high": float(fpr_high),
            "pooled_fpr": (
                float(completed["false_positive_count"].sum())
                / float(completed["reference_negative_count"].sum())
                if main_available
                and float(completed["reference_negative_count"].sum()) > 0.0
                else np.nan
            ),
            "mean_loss": float(np.mean(loss)) if main_available else np.nan,
            "median_loss": float(np.median(loss)) if main_available else np.nan,
            "loss_q25": float(np.quantile(loss, 0.25)) if main_available else np.nan,
            "loss_q75": float(np.quantile(loss, 0.75)) if main_available else np.nan,
            "loss_q90": float(np.quantile(loss, 0.9)) if main_available else np.nan,
            "max_loss": float(np.max(loss)) if main_available else np.nan,
            "fallback_rate": float(
                (frame["status"] == RunState.FALLBACK_FAILED.value).mean()
            ),
        })
    return pd.DataFrame(rows)


def paired_effects(
    summary: pd.DataFrame,
    *,
    scope: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = trace_level_metrics(summary)
    identity = ["trace_id", "seed", "budget"]
    if scope == "fixed":
        identity = ["trace_id", "condition", "budget"]
    trace_rows = []
    effect_rows = []
    rng = np.random.default_rng(bootstrap_seed)
    for comparison, challenger, reference in COMPARISONS:
        left = data[data["method"] == challenger].set_index(identity)
        right = data[data["method"] == reference].set_index(identity)
        if set(left.index) != set(right.index):
            raise ValueError(f"{scope} paired cells do not align: {comparison}")
        pairs = []
        for key in sorted(left.index):
            lhs = left.loc[key]
            rhs = right.loc[key]
            completed = (
                lhs["status"] == "completed" and rhs["status"] == "completed"
            )
            lhs_loss = float(lhs["mean_approx_loss"])
            rhs_loss = float(rhs["mean_approx_loss"])
            positive_loss = completed and lhs_loss > 0.0 and rhs_loss > 0.0
            row_identity = dict(
                zip(identity, key if isinstance(key, tuple) else (key,))
            )
            row = {
                "scope": scope,
                "comparison": comparison,
                "challenger": challenger,
                "reference": reference,
                **row_identity,
                "completed_pair": completed,
                "fpr_difference": (
                    float(lhs["fpr"] - rhs["fpr"]) if completed else np.nan
                ),
                "loss_ratio": lhs_loss / rhs_loss if positive_loss else np.nan,
                "log10_loss_ratio": (
                    float(np.log10(lhs_loss / rhs_loss)) if positive_loss else np.nan
                ),
                "challenger_loss": lhs_loss if completed else np.nan,
                "reference_loss": rhs_loss if completed else np.nan,
                "challenger_fpr": float(lhs["fpr"]) if completed else np.nan,
                "reference_fpr": float(rhs["fpr"]) if completed else np.nan,
            }
            pairs.append(row)
            trace_rows.append(row)
        pair_frame = pd.DataFrame(pairs)
        for budget, frame in pair_frame.groupby("budget", sort=True):
            all_completed = bool(frame["completed_pair"].all())
            log_ratios = frame["log10_loss_ratio"].to_numpy(dtype=np.float64)
            log_available = all_completed and bool(np.isfinite(log_ratios).all())
            fpr_differences = frame["fpr_difference"].to_numpy(dtype=np.float64)
            if scope != "fixed" and all_completed:
                indices = rng.integers(
                    0,
                    len(frame),
                    size=(bootstrap_replicates, len(frame)),
                )
                fpr_boot = np.mean(fpr_differences[indices], axis=1)
                fpr_ci = np.quantile(fpr_boot, (0.025, 0.975))
                if log_available:
                    log_boot = np.mean(log_ratios[indices], axis=1)
                    log_ci = np.quantile(log_boot, (0.025, 0.975))
                else:
                    log_ci = (np.nan, np.nan)
            else:
                fpr_ci = (np.nan, np.nan)
                log_ci = (np.nan, np.nan)
            effect_rows.append({
                "scope": scope,
                "comparison": comparison,
                "challenger": challenger,
                "reference": reference,
                "budget": int(budget),
                "pair_count": len(frame),
                "valid_pair_count": int(frame["completed_pair"].sum()),
                "main_available": all_completed,
                "mean_fpr_difference": (
                    float(np.mean(fpr_differences)) if all_completed else np.nan
                ),
                "fpr_difference_ci_low": float(fpr_ci[0]),
                "fpr_difference_ci_high": float(fpr_ci[1]),
                "geometric_mean_loss_ratio": (
                    float(10.0 ** np.mean(log_ratios)) if log_available else np.nan
                ),
                "log10_loss_ratio_ci_low": float(log_ci[0]),
                "log10_loss_ratio_ci_high": float(log_ci[1]),
                "loss_win_fraction": (
                    float(np.mean(log_ratios < 0.0)) if log_available else np.nan
                ),
            })
    return pd.DataFrame(effect_rows), pd.DataFrame(trace_rows)


def _write_tex_tables(
    aggregates: pd.DataFrame,
    effects: pd.DataFrame,
    fixed_pairs: pd.DataFrame,
    output: Path,
) -> tuple[Path, ...]:
    paths = []
    for scope in ("replay", "confirmation"):
        table = effects[
            (effects["scope"] == scope)
            & effects["comparison"].isin(("dart_effect", "data_scale", "total_rescue"))
        ][[
            "budget",
            "comparison",
            "valid_pair_count",
            "mean_fpr_difference",
            "fpr_difference_ci_low",
            "fpr_difference_ci_high",
            "geometric_mean_loss_ratio",
            "loss_win_fraction",
        ]].copy()
        path = output / f"{scope}_paired_effects.tex"
        path.write_text(
            table.to_latex(index=False, float_format="%.4g", escape=True)
        )
        paths.append(path)
    fixed = fixed_pairs[[
        "condition",
        "budget",
        "comparison",
        "fpr_difference",
        "loss_ratio",
    ]].copy()
    fixed_path = output / "fixed_figure8_effects.tex"
    fixed_path.write_text(
        fixed.to_latex(index=False, float_format="%.4g", escape=True)
    )
    paths.append(fixed_path)
    aggregate_path = output / "all_scope_summary.tex"
    aggregates[[
        "scope",
        "condition",
        "budget",
        "method",
        "valid_count",
        "failed_count",
        "macro_fpr",
        "mean_loss",
        "median_loss",
        "loss_q90",
    ]].to_latex(
        aggregate_path,
        index=False,
        float_format="%.4g",
        escape=True,
    )
    paths.append(aggregate_path)
    return tuple(paths)


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
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def _save_figure(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(
        path.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )


def _plot_paired_effects(effects: pd.DataFrame, output: Path) -> None:
    plt = _pyplot()
    data = effects[
        effects["scope"].isin(("replay", "confirmation"))
        & effects["comparison"].isin(("dart_effect", "data_scale"))
    ].copy()
    colors = {"dart_effect": "#0072B2", "data_scale": "#D55E00"}
    markers = {"dart_effect": "o", "data_scale": "s"}
    labels = {"dart_effect": "DART36 − Clean36", "data_scale": "Clean36 − Clean20"}
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.4), sharex=True)
    for row, scope in enumerate(("replay", "confirmation")):
        for comparison in ("dart_effect", "data_scale"):
            frame = data[
                (data["scope"] == scope) & (data["comparison"] == comparison)
            ].sort_values("budget")
            x = frame["budget"].to_numpy()
            fpr = 100.0 * frame["mean_fpr_difference"].to_numpy()
            fpr_low = 100.0 * frame["fpr_difference_ci_low"].to_numpy()
            fpr_high = 100.0 * frame["fpr_difference_ci_high"].to_numpy()
            axes[row, 0].errorbar(
                x,
                fpr,
                yerr=np.vstack((fpr - fpr_low, fpr_high - fpr)),
                color=colors[comparison],
                marker=markers[comparison],
                linestyle="-" if comparison == "dart_effect" else "--",
                capsize=2,
                label=labels[comparison],
            )
            ratio = frame["geometric_mean_loss_ratio"].to_numpy()
            low = 10.0 ** frame["log10_loss_ratio_ci_low"].to_numpy()
            high = 10.0 ** frame["log10_loss_ratio_ci_high"].to_numpy()
            axes[row, 1].errorbar(
                x,
                ratio,
                yerr=np.vstack((ratio - low, high - ratio)),
                color=colors[comparison],
                marker=markers[comparison],
                linestyle="-" if comparison == "dart_effect" else "--",
                capsize=2,
            )
        axes[row, 0].axhline(0.0, color="0.45", linewidth=0.8)
        axes[row, 1].axhline(1.0, color="0.45", linewidth=0.8)
        axes[row, 0].set_ylabel(
            ("Replay" if scope == "replay" else "Confirmation")
            + "\nFPR difference (pp)"
        )
        axes[row, 1].set_ylabel("Geometric mean\nloss ratio")
        axes[row, 1].set_yscale("log")
        for axis in axes[row]:
            axis.set_xscale("log")
            axis.grid(axis="y", color="0.9", linewidth=0.6)
    axes[0, 0].legend(frameon=False, loc="best")
    axes[1, 0].set_xlabel("Transform bound")
    axes[1, 1].set_xlabel("Transform bound")
    fig.tight_layout()
    _save_figure(fig, output / "nominal_paired_effects")
    plt.close(fig)


def _plot_nominal_ecdfs(
    summaries: Mapping[str, pd.DataFrame],
    config: DartRescueConfig,
    output: Path,
) -> None:
    plt = _pyplot()
    colors = {"clean20": "#666666", "clean36": "#D55E00", "dart36": "#0072B2"}
    styles = {"clean20": ":", "clean36": "--", "dart36": "-"}
    budgets = config.instability_budgets
    fig, axes = plt.subplots(
        2,
        len(budgets),
        figsize=(max(7.0, 1.75 * len(budgets)), 3.7),
        squeeze=False,
        sharey=True,
    )
    for row, scope in enumerate(("replay", "confirmation")):
        data = trace_level_metrics(summaries[scope])
        for column, budget in enumerate(budgets):
            axis = axes[row, column]
            positive = True
            for method in METHODS:
                values = data[
                    (data["budget"] == budget)
                    & (data["method"] == method)
                    & (data["status"] == "completed")
                ]["mean_approx_loss"].to_numpy(dtype=np.float64)
                positive &= bool(np.all(values > 0.0))
                ordered = np.sort(values)
                y = np.arange(1, len(ordered) + 1) / len(ordered)
                axis.step(
                    ordered,
                    y,
                    where="post",
                    color=colors[method],
                    linestyle=styles[method],
                    label=method,
                )
            if positive:
                axis.set_xscale("log")
            axis.set_title(f"B={budget}")
            axis.grid(axis="y", color="0.9", linewidth=0.6)
            if column == 0:
                axis.set_ylabel(
                    ("Replay" if scope == "replay" else "Confirmation") + "\nECDF"
                )
            if row == 1:
                axis.set_xlabel("Mean loss" + (" (log)" if positive else ""))
    axes[0, 0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save_figure(fig, output / "nominal_loss_ecdfs")
    plt.close(fig)


def _plot_fixed(summary: pd.DataFrame, output: Path) -> None:
    plt = _pyplot()
    data = trace_level_metrics(summary)
    conditions = tuple(dict.fromkeys(data["condition"].astype(str)))
    colors = {"clean20": "#666666", "clean36": "#D55E00", "dart36": "#0072B2"}
    markers = {"clean20": "^", "clean36": "s", "dart36": "o"}
    styles = {"clean20": ":", "clean36": "--", "dart36": "-"}
    fig, axes = plt.subplots(len(conditions), 2, figsize=(7.0, 1.65 * len(conditions)))
    if len(conditions) == 1:
        axes = np.asarray([axes])
    for row, condition in enumerate(conditions):
        for method in METHODS:
            frame = data[
                (data["condition"] == condition) & (data["method"] == method)
            ].sort_values("budget")
            axes[row, 0].plot(
                frame["budget"],
                100.0 * frame["fpr"],
                color=colors[method],
                marker=markers[method],
                linestyle=styles[method],
                label=method,
            )
            axes[row, 1].plot(
                frame["budget"],
                frame["mean_approx_loss"],
                color=colors[method],
                marker=markers[method],
                linestyle=styles[method],
            )
        axes[row, 0].set_ylabel(f"{condition}\nFPR (%)")
        axes[row, 1].set_ylabel("Mean loss")
        axes[row, 0].set_xscale("log")
        axes[row, 1].set_xscale("log")
        completed_loss = data[
            (data["condition"] == condition) & (data["status"] == "completed")
        ]["mean_approx_loss"]
        if bool((completed_loss > 0.0).all()):
            axes[row, 1].set_yscale("log")
        for axis in axes[row]:
            axis.grid(axis="y", color="0.9", linewidth=0.6)
    axes[0, 0].legend(frameon=False, loc="best")
    axes[-1, 0].set_xlabel("Transform bound")
    axes[-1, 1].set_xlabel("Transform bound")
    fig.tight_layout()
    _save_figure(fig, output / "fixed_figure8_rescue")
    plt.close(fig)


def _plot_dart_diagnostics(config: DartRescueConfig, output: Path) -> pd.DataFrame:
    rows = []
    for budget in config.budgets:
        calibration = pd.read_csv(
            config.output_root / "calibrate" / f"budget-{budget}"
            / "dart_budget_calibration.csv"
        ).iloc[0]
        collection = pd.read_csv(
            config.output_root / "collect" / f"budget-{budget}" / "dataset"
            / "dart_collection_summary.csv"
        )
        rows.append({
            "budget": budget,
            "target_disturbance_rate": float(
                calibration["target_disturbance_rate"]
            ),
            "expected_disturbance_rate": float(
                calibration["expected_disturbance_rate"]
            ),
            "realized_disturbance_rate": float(
                np.average(
                    collection["realized_disturbance_rate"],
                    weights=collection["sample_count"],
                )
            ),
            "regret_cap": float(calibration["regret_cap"]),
            "saturated": bool(calibration["saturated"]),
        })
    frame = pd.DataFrame(rows)
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(3.5, 2.4))
    styles = (
        ("target_disturbance_rate", "Target", "#666666", ":"),
        ("expected_disturbance_rate", "Expected", "#D55E00", "--"),
        ("realized_disturbance_rate", "Realized", "#0072B2", "-"),
    )
    for column, label, color, linestyle in styles:
        axis.plot(
            frame["budget"],
            frame[column],
            marker="o",
            color=color,
            linestyle=linestyle,
            label=label,
        )
    axis.set_xscale("log")
    axis.set_xlabel("Transform bound")
    axis.set_ylabel("Disturbed decision fraction")
    axis.grid(axis="y", color="0.9", linewidth=0.6)
    axis.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, output / "dart_disturbance_diagnostics")
    plt.close(fig)
    return frame


def run_report(config: DartRescueConfig) -> Path:
    _validate_stage_manifest(config, "evaluate")
    output = config.output_root / "report"
    output.mkdir(parents=True, exist_ok=True)
    summaries = {
        scope: pd.read_csv(config.output_root / "evaluate" / scope / "summary.csv")
        for scope in ("replay", "confirmation", "fixed")
    }
    aggregate_frames = [
        aggregate_scope(
            summary,
            scope=scope,
            bootstrap_replicates=config.bootstrap_replicates,
            bootstrap_seed=config.bootstrap_seed,
        )
        for scope, summary in summaries.items()
    ]
    aggregates = pd.concat(aggregate_frames, ignore_index=True)
    if "condition" not in aggregates:
        aggregates["condition"] = ""
    aggregates["condition"] = aggregates["condition"].fillna("")
    effects = []
    trace_effects = []
    for scope, summary in summaries.items():
        effect, trace = paired_effects(
            summary,
            scope=scope,
            bootstrap_replicates=config.bootstrap_replicates,
            bootstrap_seed=config.bootstrap_seed,
        )
        effects.append(effect)
        trace_effects.append(trace)
    effect_frame = pd.concat(effects, ignore_index=True)
    trace_frame = pd.concat(trace_effects, ignore_index=True)
    write_csv_atomic(aggregates, output / "aggregate_metrics.csv")
    write_csv_atomic(effect_frame, output / "paired_effects.csv")
    write_csv_atomic(trace_frame, output / "paired_trace_effects.csv")
    fixed_pairs = trace_frame[trace_frame["scope"] == "fixed"].copy()
    write_csv_atomic(fixed_pairs, output / "fixed_figure8_effects.csv")
    composition = []
    for scope in ("replay", "confirmation", "fixed"):
        series = pd.read_csv(
            config.output_root / "evaluate" / scope / "timeseries.csv"
        )
        counts = (
            series.groupby(
                ["condition", "budget", "method", "reducer_used"], dropna=False
            ).size().rename("count").reset_index()
        )
        counts["scope"] = scope
        totals = counts.groupby(
            ["scope", "condition", "budget", "method"], dropna=False
        )["count"].transform("sum")
        counts["fraction"] = counts["count"] / totals
        composition.append(counts)
    write_csv_atomic(
        pd.concat(composition, ignore_index=True),
        output / "reducer_composition.csv",
    )
    diagnostic_frame = _plot_dart_diagnostics(config, output)
    write_csv_atomic(diagnostic_frame, output / "dart_disturbance_diagnostics.csv")
    tex_paths = _write_tex_tables(aggregates, effect_frame, fixed_pairs, output)
    _plot_paired_effects(effect_frame, output)
    _plot_nominal_ecdfs(summaries, config, output)
    _plot_fixed(summaries["fixed"], output)
    source_paths = tuple(
        config.output_root / "evaluate" / scope / name
        for scope in ("replay", "confirmation", "fixed")
        for name in ("summary.csv", "timeseries.csv", "manifest.json")
    )
    output_paths = (
        output / "aggregate_metrics.csv",
        output / "paired_effects.csv",
        output / "paired_trace_effects.csv",
        output / "fixed_figure8_effects.csv",
        output / "reducer_composition.csv",
        output / "dart_disturbance_diagnostics.csv",
        *tex_paths,
        *(output / name for name in (
            "nominal_paired_effects.pdf",
            "nominal_paired_effects.png",
            "nominal_loss_ecdfs.pdf",
            "nominal_loss_ecdfs.png",
            "fixed_figure8_rescue.pdf",
            "fixed_figure8_rescue.png",
            "dart_disturbance_diagnostics.pdf",
            "dart_disturbance_diagnostics.png",
        )),
    )
    write_json_atomic({
        "schema": "pzr.dart-rescue-report.v1",
        "input_hashes": {str(path): sha256_files((path,)) for path in source_paths},
        "output_hashes": {str(path): sha256_files((path,)) for path in output_paths},
        "bootstrap": {
            "unit": "seed-aligned nominal trace",
            "replicates": config.bootstrap_replicates,
            "seed": config.bootstrap_seed,
            "interval": "paired percentile 95%",
        },
        "claims": {
            "replay": "retrospective diagnostic only",
            "confirmation": "untouched nominal random-trajectory confirmation",
            "fixed": "controlled patterned case study; no population interval",
            "randomized_fault_generalization": False,
        },
    }, output / "report_manifest.json")
    return _write_stage_manifest(config, "report", {
        "report_manifest": str(output / "report_manifest.json"),
        "report_manifest_sha256": sha256_files((output / "report_manifest.json",)),
        "aggregate_row_count": len(aggregates),
        "paired_effect_row_count": len(effect_frame),
        "paired_trace_row_count": len(trace_frame),
    })


def _validate_collection_diagnostics(config: DartRescueConfig) -> None:
    for budget in config.budgets:
        root = config.output_root / "collect" / f"budget-{budget}" / "dataset"
        summary = pd.read_csv(root / "dart_collection_summary.csv")
        if bool((summary["maximum_consecutive_disturbances"] > 1).any()):
            raise ValueError(f"DART recovery guard failed at budget {budget}")
        metadata = pd.read_csv(root / "samples.csv")
        disturbed = metadata["disturbed"].astype(bool)
        if bool((
            metadata.loc[disturbed, "sampled_normalized_regret"]
            > metadata.loc[disturbed, "regret_cap"] + 1e-15
        ).any()):
            raise ValueError(f"DART regret cap failed at budget {budget}")


def run_validate(config: DartRescueConfig) -> Path:
    for stage in STAGES[:-1]:
        _validate_stage_manifest(config, stage)
    _validate_collection_diagnostics(config)
    validations = {}
    total_failures = 0
    for scope in ("replay", "confirmation", "fixed"):
        root = config.output_root / "evaluate" / scope
        manifest = load_json(root / "manifest.json")
        summary = pd.read_csv(root / "summary.csv")
        if len(summary) != config.reported_cells(scope):
            raise ValueError(f"{scope} validation cell count differs")
        if set(summary["status"]) - RUN_STATES:
            raise ValueError(f"{scope} contains an invalid run state")
        if summary.duplicated(["trace_id", "budget", "method"]).any():
            raise ValueError(f"{scope} contains duplicate scientific cells")
        trace_level_metrics(summary)
        failures = int((summary["status"] != "completed").sum())
        if failures != int(manifest["failure_count"]):
            raise ValueError(f"{scope} failure count differs")
        total_failures += failures
        validations[scope] = {
            "reported_cell_count": len(summary),
            "new_cell_count": int(manifest["new_cell_count"]),
            "failure_count": failures,
            "manifest_sha256": sha256_files((root / "manifest.json",)),
        }
    report = load_json(config.output_root / "report" / "report_manifest.json")
    for raw_path, expected_hash in report["output_hashes"].items():
        if sha256_files((Path(raw_path),)) != expected_hash:
            raise ValueError(f"stale DART rescue report output: {raw_path}")
    path = config.output_root / "validate" / "manifest.json"
    write_json_atomic({
        **_stage_manifest_base(config, "validate"),
        "schema": VALIDATION_SCHEMA,
        "status": "completed" if total_failures == 0 else "completed_with_failures",
        "validated_scopes": validations,
        "reported_cell_count": sum(
            int(item["reported_cell_count"]) for item in validations.values()
        ),
        "new_cell_count": sum(
            int(item["new_cell_count"]) for item in validations.values()
        ),
        "failure_count": total_failures,
        "paper_evaluation_compatibility": {
            "expected_pzr_source_sha256": config.expected_pzr_source_sha256,
            "actual_pzr_source_sha256": pzr_source_sha256(),
            "unchanged": pzr_source_sha256() == config.expected_pzr_source_sha256,
        },
    }, path)
    return path


@contextmanager
def _stage_log(config: DartRescueConfig, stage: str) -> Iterator[None]:
    path = config.output_root / "logs" / f"{stage}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        tee_out = _Tee(sys.stdout, stream)
        tee_err = _Tee(sys.stderr, stream)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print(f"start stage: {stage}", flush=True)
            yield
            print(f"complete stage: {stage}", flush=True)


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


def _run_or_skip(config: DartRescueConfig, stage: str) -> None:
    manifest = config.output_root / stage / "manifest.json"
    if manifest.is_file():
        if stage == "validate":
            validation = load_json(manifest)
            if validation.get("effective_config_sha256") != (
                config.effective_config_sha256
            ) or validation.get("tool_sha256") != tool_sha256():
                raise ValueError("stale validate DART rescue manifest")
        else:
            _validate_stage_manifest(config, stage)
        print(f"skip validated stage: {stage}", flush=True)
        return
    functions = {
        "preflight": run_preflight,
        "prepare": run_prepare,
        "calibrate": run_calibrate,
        "collect": run_collect,
        "train": run_train,
        "evaluate": run_evaluate,
        "report": run_report,
        "validate": run_validate,
    }
    with _stage_log(config, stage):
        functions[stage](config)


def run_all(config: DartRescueConfig) -> Path:
    for stage in STAGES:
        _run_or_skip(config, stage)
    return config.output_root / "validate" / "manifest.json"


def status(config: DartRescueConfig) -> dict[str, object]:
    stages = {}
    for stage in STAGES:
        path = config.output_root / stage / "manifest.json"
        if not path.is_file():
            stages[stage] = {"status": "missing"}
            continue
        try:
            manifest = load_json(path)
            if manifest.get("effective_config_sha256") != (
                config.effective_config_sha256
            ) or manifest.get("tool_sha256") != tool_sha256():
                raise ValueError("config/tool hash differs")
            stages[stage] = {
                "status": str(manifest.get("status")),
                "failure_count": int(manifest.get("failure_count", 0)),
            }
        except Exception as exc:
            stages[stage] = {"status": "stale_or_invalid", "message": str(exc)}
    return {
        "experiment_id": config.experiment_id,
        "output_root": str(config.output_root),
        "smoke": config.smoke,
        "stages": stages,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(*STAGES, "run", "status"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one budget/trace per scope in /tmp without touching canonical outputs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, smoke=args.smoke)
    if args.command == "status":
        print(json.dumps(status(config), indent=2))
        return 0
    if args.command == "run":
        path = run_all(config)
    else:
        _run_or_skip(config, args.command)
        path = config.output_root / args.command / "manifest.json"
    print(f"DART rescue {args.command} complete: {path}")
    if args.command in {"run", "validate"}:
        manifest = load_json(config.output_root / "validate" / "manifest.json")
        return 2 if int(manifest.get("failure_count", 0)) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
