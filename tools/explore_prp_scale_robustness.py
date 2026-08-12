#!/usr/bin/env python3
"""Disposable PRP clean-data scaling and optimizer-robustness experiment.

The seed-42 size curve extends the existing nested clean datasets from 84 to
148 nominal trajectories.  Clean84 and Clean148 are additionally trained with
four more optimizer seeds.  Every frozen model is evaluated at its matching
transform budget on fresh nominal seeds 140--159 and on the four fixed
figure-eight case studies.

This is exploratory model selection, not a canonical paper evaluation.  Trace
seeds are the independent units for nominal bootstrap intervals; optimizer
seeds are reported as five training replicates and are never pooled with trace
seeds as if they were independent observations.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
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
)
from pzr.rtlola.paper_experiment import (
    ExecutionRegime,
    MethodConfig,
    Objective,
    Predictor,
    RunState,
    TraceSource,
    load_json,
    load_paper_experiment_config,
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
    ROBOT_ARM_TRACE_ROWS,
    ROBOT_ARM_TRACE_SHA256,
)
from pzr.rtlola.robot_arm_random import RANDOM_WAYPOINT_SOURCE_REVISION
from pzr.rtlola.scenarios import scenario_by_name


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "prp-scale-robustness-exploratory"
PAPER_ROOT = ROOT / "results" / "paper-evaluation-v2"
DART_ROOT = ROOT / "results" / "dart-rescue-v1"
OLD_SCALING_ROOT = ROOT / "results" / "clean-scaling-exploratory"
PAPER_CONFIG = ROOT / "experiments" / "paper_evaluation_v2.yaml"

SCHEMA = "pzr.prp-scale-robustness-exploratory.v1"
CELL_SCHEMA = "pzr.prp-scale-robustness-exploratory-cell.v1"
FREEZE_SCHEMA = "pzr.prp-scale-robustness-freeze.v1"
EXPECTED_PZR_SOURCE_SHA256 = (
    "f230d481022de2c69c610c917deae901e7a87e4322c979c248f2cf8f4fa1e5ca"
)
PAPER_CONFIG_SHA256 = (
    "a7a911f641b0227aa3a6657231afef97aedbd8428ab25f2caf9d9d6dd486f074"
)
PAPER_DATASET = PAPER_ROOT / "prepare" / "teacher" / "dataset"
PAPER_DATASET_SHA256 = (
    "885c3dfbf70ddf614db72f564877e667e056a59966e62094d365606e0b503602"
)
DART_CLEAN_DATASET = DART_ROOT / "prepare" / "extra-clean" / "dataset"
DART_CLEAN_DATASET_SHA256 = (
    "3540f04b561197746862d7b2bbdc3f2880fc2e3613dd975c3fde43ff8b3c08f6"
)
OLD_SCALING_DATASET = OLD_SCALING_ROOT / "prepare" / "new-clean" / "dataset"
OLD_SCALING_DATASET_SHA256 = (
    "1b44e03f0125943e1b62a1f184264416e1bad84e4a31a024646b0443962ebbe7"
)
DART_EFFECTIVE_CONFIG_SHA256 = (
    "6c930dc948b254c27125463e2147f43c9867befa4afa5513c74dd74d2340538f"
)

BUDGETS = (40, 80, 120, 150, 200, 250, 500)
BASE_TRAIN_SEEDS = tuple(range(20)) + tuple(range(26, 42)) + tuple(range(200, 248))
NEW_TRAIN_SEEDS = tuple(range(248, 312))
VALIDATION_SEEDS = tuple(range(20, 26))
EVALUATION_SEEDS = tuple(range(140, 160))
SIZE_CURVE = (20, 36, 52, 68, 84, 100, 116, 132, 148)
NEW_SIZES = (100, 116, 132, 148)
OPTIMIZER_SEEDS = (42, 1042, 2042, 3042, 4042)
STABILITY_SIZES = (84, 148)
FIXED_TRACE_KINDS = (
    "figure8",
    "figure8_drift",
    "figure8_geofence",
    "figure8_drift_geofence",
)
CANDIDATES = ("girard", "scott", "pca", "combastel")
EVENT_COUNT = 500
WORKERS = 10
HIGH_LOSS_THRESHOLD = 1e-3
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260726
TRAINING = {
    "epochs": 100,
    "batch_size": 256,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "patience": 10,
}
STAGES = ("check", "prepare", "train", "freeze", "evaluate", "report")


@dataclass(frozen=True, order=True)
class ModelVariant:
    training_size: int
    optimizer_seed: int

    @property
    def name(self) -> str:
        if self.optimizer_seed == 42:
            return f"clean{self.training_size}"
        return f"clean{self.training_size}_opt{self.optimizer_seed}"


@dataclass(frozen=True)
class ExploreConfig:
    output: Path = OUTPUT
    budgets: tuple[int, ...] = BUDGETS
    base_train_seeds: tuple[int, ...] = BASE_TRAIN_SEEDS
    new_train_seeds: tuple[int, ...] = NEW_TRAIN_SEEDS
    validation_seeds: tuple[int, ...] = VALIDATION_SEEDS
    evaluation_seeds: tuple[int, ...] = EVALUATION_SEEDS
    size_curve: tuple[int, ...] = SIZE_CURVE
    new_sizes: tuple[int, ...] = NEW_SIZES
    optimizer_seeds: tuple[int, ...] = OPTIMIZER_SEEDS
    stability_sizes: tuple[int, ...] = STABILITY_SIZES
    fixed_trace_kinds: tuple[str, ...] = FIXED_TRACE_KINDS
    event_count: int = EVENT_COUNT
    workers: int = WORKERS
    reuse_existing: bool = True
    smoke: bool = False

    def __post_init__(self) -> None:
        if len(set(self.master_train_seeds)) != len(self.master_train_seeds):
            raise ValueError("training seeds must be unique")
        seed_groups = (
            set(self.master_train_seeds),
            set(self.validation_seeds),
            set(self.evaluation_seeds),
        )
        if any(left & right for index, left in enumerate(seed_groups) for right in seed_groups[index + 1:]):
            raise ValueError("training, validation, and evaluation seeds must be disjoint")
        if tuple(sorted(self.budgets)) != self.budgets:
            raise ValueError("budgets must be sorted")
        if 42 not in self.optimizer_seeds:
            raise ValueError("optimizer seed 42 must define the size curve")
        if max(self.size_curve) > len(self.master_train_seeds):
            raise ValueError("size curve exceeds the available training trajectories")

    @property
    def master_train_seeds(self) -> tuple[int, ...]:
        return self.base_train_seeds + self.new_train_seeds

    @property
    def variants(self) -> tuple[ModelVariant, ...]:
        variants = {ModelVariant(size, 42) for size in self.size_curve}
        variants.update(
            ModelVariant(size, seed)
            for size in self.stability_sizes
            for seed in self.optimizer_seeds
        )
        return tuple(sorted(variants))

    @property
    def reused_variants(self) -> tuple[ModelVariant, ...]:
        if not self.reuse_existing:
            return ()
        return tuple(ModelVariant(size, 42) for size in (20, 36, 52, 68, 84))

    @property
    def new_variants(self) -> tuple[ModelVariant, ...]:
        reused = set(self.reused_variants)
        return tuple(variant for variant in self.variants if variant not in reused)

    @property
    def identity(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "output": str(self.output.resolve()),
            "budgets": list(self.budgets),
            "base_train_seeds": list(self.base_train_seeds),
            "new_train_seeds": list(self.new_train_seeds),
            "validation_seeds": list(self.validation_seeds),
            "evaluation_seeds": list(self.evaluation_seeds),
            "size_curve": list(self.size_curve),
            "new_sizes": list(self.new_sizes),
            "optimizer_seeds": list(self.optimizer_seeds),
            "stability_sizes": list(self.stability_sizes),
            "fixed_trace_kinds": list(self.fixed_trace_kinds),
            "event_count": self.event_count,
            "workers": self.workers,
            "reuse_existing": self.reuse_existing,
            "smoke": self.smoke,
            "training": TRAINING,
            "high_loss_threshold": HIGH_LOSS_THRESHOLD,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
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
        trace_count = len(self.new_train_seeds)
        if not self.reuse_existing:
            trace_count += len(self.validation_seeds)
        return trace_count * len(self.budgets)

    @property
    def expected_new_models(self) -> int:
        return len(self.new_variants) * len(self.budgets)

    @property
    def expected_nominal_cells(self) -> int:
        return len(self.evaluation_seeds) * len(self.budgets) * len(self.variants)

    @property
    def expected_fixed_cells(self) -> int:
        method_count = len(self.variants) + (1 if self.reuse_existing else 0)
        return len(self.fixed_trace_kinds) * len(self.budgets) * method_count

    @property
    def expected_imported_fixed_cells(self) -> int:
        if not self.reuse_existing:
            return 0
        return len(self.fixed_trace_kinds) * len(self.budgets) * 3

    @property
    def expected_new_fixed_cells(self) -> int:
        return self.expected_fixed_cells - self.expected_imported_fixed_cells

    @property
    def expected_reported_cells(self) -> int:
        return self.expected_nominal_cells + self.expected_fixed_cells

    @property
    def expected_new_cells(self) -> int:
        return self.expected_nominal_cells + self.expected_new_fixed_cells


def smoke_config() -> ExploreConfig:
    return ExploreConfig(
        output=Path("/tmp/pzr-prp-scale-robustness-smoke"),
        budgets=(40,),
        base_train_seeds=(),
        new_train_seeds=(248, 249),
        validation_seeds=(250,),
        evaluation_seeds=(140,),
        size_curve=(1, 2),
        new_sizes=(1, 2),
        optimizer_seeds=(42, 1042),
        stability_sizes=(1, 2),
        fixed_trace_kinds=("figure8",),
        event_count=100,
        workers=1,
        reuse_existing=False,
        smoke=True,
    )


def training_seeds(config: ExploreConfig, size: int) -> tuple[int, ...]:
    if size < 1 or size > len(config.master_train_seeds):
        raise ValueError(f"invalid clean training size: {size}")
    return config.master_train_seeds[:size]


def tool_sha256() -> str:
    return sha256_files(
        (
            Path(__file__).resolve(),
            ROOT / "tools" / "run_prp_scale_robustness.sh",
        ),
        relative_to=ROOT,
    )


def _raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_artifact_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


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
        raise ValueError(f"unsupported PRP robustness manifest: {path}")
    if manifest.get("experiment_fingerprint") != config.fingerprint:
        raise ValueError(f"stale PRP robustness manifest: {path}")
    return manifest


def _validate_dataset(path: Path, expected_hash: str, label: str) -> None:
    actual = dataset_sha256(path)
    if actual != expected_hash:
        raise ValueError(f"{label} dataset hash differs: {actual}")


def _paper_model_records(config: ExploreConfig) -> dict[tuple[str, int], dict[str, object]]:
    manifest = load_json(PAPER_ROOT / "train" / "manifest.json")
    records: dict[tuple[str, int], dict[str, object]] = {}
    for budget in config.budgets:
        record = dict(manifest["models_by_budget"][str(budget)])
        path = _resolve_artifact_path(record["path"])
        if record.get("budget_filter") != [budget]:
            raise ValueError(f"Clean20 training budget differs: {budget}")
        if model_sha256(path) != record["sha256"]:
            raise ValueError(f"Clean20 model hash differs: {budget}")
        records[("clean20", budget)] = {
            **record,
            "path": str(path),
            "training_size": 20,
            "optimizer_seed": 42,
            "training_seeds": list(training_seeds(config, 20)),
        }
    return records


def _dart_model_records(config: ExploreConfig) -> dict[tuple[str, int], dict[str, object]]:
    manifest = load_json(DART_ROOT / "train" / "manifest.json")
    if manifest.get("effective_config_sha256") != DART_EFFECTIVE_CONFIG_SHA256:
        raise ValueError("DART Clean36 manifest differs")
    records: dict[tuple[str, int], dict[str, object]] = {}
    for budget in config.budgets:
        record = dict(manifest["models"][f"clean36:{budget}"])
        path = _resolve_artifact_path(record["path"])
        if record.get("budget_filter") != [budget]:
            raise ValueError(f"Clean36 training budget differs: {budget}")
        if model_sha256(path) != record["sha256"]:
            raise ValueError(f"Clean36 model hash differs: {budget}")
        records[("clean36", budget)] = {
            **record,
            "path": str(path),
            "training_size": 36,
            "optimizer_seed": 42,
            "training_seeds": list(training_seeds(config, 36)),
        }
    return records


def _old_scaling_model_records(
    config: ExploreConfig,
) -> dict[tuple[str, int], dict[str, object]]:
    manifest = load_json(OLD_SCALING_ROOT / "train" / "manifest.json")
    if manifest.get("schema") != "pzr.clean-scaling-exploratory.v1":
        raise ValueError("old clean-scaling model schema differs")
    if manifest.get("pzr_source_sha256") != EXPECTED_PZR_SOURCE_SHA256:
        raise ValueError("old clean-scaling source differs")
    records: dict[tuple[str, int], dict[str, object]] = {}
    for size in (52, 68, 84):
        for budget in config.budgets:
            record = dict(manifest["models"][f"{size}:{budget}"])
            path = _resolve_artifact_path(record["path"])
            if int(record["training_size"]) != size:
                raise ValueError(f"old Clean{size} size differs")
            if tuple(record["training_seeds"]) != training_seeds(config, size):
                raise ValueError(f"old Clean{size} training seeds differ")
            if model_sha256(path) != record["sha256"]:
                raise ValueError(f"old Clean{size} model hash differs: {budget}")
            records[(f"clean{size}", budget)] = {
                **record,
                "path": str(path),
                "optimizer_seed": 42,
            }
    return records


def _reused_model_records(
    config: ExploreConfig,
) -> dict[tuple[str, int], dict[str, object]]:
    if not config.reuse_existing:
        return {}
    records = {
        **_paper_model_records(config),
        **_dart_model_records(config),
        **_old_scaling_model_records(config),
    }
    expected = {
        (variant.name, budget)
        for variant in config.reused_variants
        for budget in config.budgets
    }
    if set(records) != expected:
        raise ValueError("reused PRP model matrix differs")
    return records


def run_check(config: ExploreConfig) -> Path:
    if BINDING_BUILD_PROFILE != "release":
        raise ValueError("PRP robustness exploration requires the release binding")
    if pzr_source_sha256() != EXPECTED_PZR_SOURCE_SHA256:
        raise ValueError("PZR source differs from the frozen upstream artifacts")
    upstream: dict[str, object] = {}
    if config.reuse_existing:
        if _raw_sha256(PAPER_CONFIG) != PAPER_CONFIG_SHA256:
            raise ValueError("paper configuration hash differs")
        _validate_dataset(PAPER_DATASET, PAPER_DATASET_SHA256, "Clean20")
        _validate_dataset(DART_CLEAN_DATASET, DART_CLEAN_DATASET_SHA256, "Clean36")
        _validate_dataset(
            OLD_SCALING_DATASET,
            OLD_SCALING_DATASET_SHA256,
            "Clean84 extension",
        )
        records = _reused_model_records(config)
        serializable_records = {
            f"{name}:{budget}": record
            for (name, budget), record in sorted(records.items())
        }
        upstream = {
            "paper_config_sha256": PAPER_CONFIG_SHA256,
            "dataset_hashes": {
                "paper_clean20": PAPER_DATASET_SHA256,
                "dart_extra_clean16": DART_CLEAN_DATASET_SHA256,
                "old_scaling_extra_clean48": OLD_SCALING_DATASET_SHA256,
            },
            "reused_model_count": len(records),
            "reused_model_matrix_sha256": payload_sha256(serializable_records),
        }
    return _write_manifest(
        config,
        "check",
        {
            "scientific_role": "disposable exploratory model selection",
            "upstream": upstream,
        },
    )


def run_prepare(config: ExploreConfig) -> Path:
    _load_manifest(config, "check")
    trace_root = config.output / "prepare" / "traces-new-clean"
    seed_count = len(config.new_train_seeds)
    validation_count = 0
    if not config.reuse_existing:
        seed_count += len(config.validation_seeds)
        validation_count = len(config.validation_seeds)
    store = generate_random_waypoint_trace_store(
        RandomWaypointTraceStoreConfig(
            output=trace_root,
            event_count=config.event_count,
            conditions=("random_waypoint",),
            seed_start=min(config.new_train_seeds),
            seed_count=seed_count,
        )
    )
    expected_seeds = config.new_train_seeds + (
        config.validation_seeds if not config.reuse_existing else ()
    )
    if tuple(item.seed for item in store.traces) != expected_seeds:
        raise ValueError("new clean trace seeds differ")
    dataset = run_learning_collection(
        LearningCollectionConfig(
            output=config.output / "prepare" / "new-clean",
            trace_store=trace_root,
            budgets=config.budgets,
            candidate_names=CANDIDATES,
            train_seeds=len(config.new_train_seeds),
            validation_seeds=validation_count,
            test_seeds=0,
            seed_start=min(config.new_train_seeds),
            workers=config.workers,
            collection_mode="teacher",
        )
    )
    manifest = load_json(dataset / "manifest.json")
    if int(manifest["shard_count"]) != config.expected_new_shards:
        raise ValueError("new clean teacher shard count differs")
    return _write_manifest(
        config,
        "prepare",
        {
            "trace_store": str(trace_root),
            "trace_store_manifest_sha256": store.manifest_sha256,
            "dataset": str(dataset),
            "dataset_sha256": dataset_sha256(dataset),
            "teacher_shard_count": config.expected_new_shards,
        },
    )


def _master_dataset(
    config: ExploreConfig,
) -> tuple[ReducerCostDataset, pd.DataFrame, dict[str, str]]:
    prepared = _load_manifest(config, "prepare")
    new_dataset = Path(str(prepared["dataset"]))
    sources: tuple[tuple[str, Path, str], ...]
    if config.reuse_existing:
        sources = (
            ("paper_clean20", PAPER_DATASET, PAPER_DATASET_SHA256),
            ("dart_extra_clean16", DART_CLEAN_DATASET, DART_CLEAN_DATASET_SHA256),
            (
                "old_scaling_extra_clean48",
                OLD_SCALING_DATASET,
                OLD_SCALING_DATASET_SHA256,
            ),
            ("new_clean64", new_dataset, str(prepared["dataset_sha256"])),
        )
    else:
        sources = (("smoke_clean", new_dataset, str(prepared["dataset_sha256"])),)
    datasets = []
    frames = []
    hashes = {}
    for label, path, expected_hash in sources:
        _validate_dataset(path, expected_hash, label)
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
    digest.update(
        json.dumps(
            {
                "seeds": list(seeds),
                "budget": budget,
                "sources": dict(source_hashes),
                "sample_ids": list(dataset.sample_ids),
            },
            sort_keys=True,
        ).encode()
    )
    for array in (dataset.features, dataset.teacher_costs, dataset.feasible):
        values = np.ascontiguousarray(array)
        digest.update(str(values.dtype).encode())
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _new_model_path(
    config: ExploreConfig,
    variant: ModelVariant,
    budget: int,
) -> Path:
    return config.output / "train" / variant.name / f"budget-{budget}"


def run_train(config: ExploreConfig) -> Path:
    dataset, metadata, source_hashes = _master_dataset(config)
    records: dict[str, object] = {}
    for variant in config.new_variants:
        seeds = training_seeds(config, variant.training_size)
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
            output = _new_model_path(config, variant, budget)
            identity = {
                "schema": SCHEMA,
                "experiment_fingerprint": config.fingerprint,
                "variant": variant.name,
                "training_size": variant.training_size,
                "training_seeds": list(seeds),
                "optimizer_seed": variant.optimizer_seed,
                "budget": budget,
                "subset_sha256": subset_hash,
                "source_hashes": source_hashes,
                "training": TRAINING,
            }
            artifact = output / "exploratory_training.json"
            if artifact.is_file():
                existing = load_json(artifact)
                if existing.get("identity") != identity:
                    raise ValueError(f"stale PRP robustness model: {output}")
                if model_sha256(output) != existing["model_sha256"]:
                    raise ValueError(f"PRP robustness model hash differs: {output}")
                records[f"{variant.name}:{budget}"] = existing["record"]
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
                seed=variant.optimizer_seed,
            )
            policy.save(output)
            write_csv_atomic(
                validation_metrics(policy, subset, subset_metadata),
                output / "validation_metrics.csv",
            )
            record = {
                "variant": variant.name,
                "training_size": variant.training_size,
                "training_seeds": list(seeds),
                "optimizer_seed": variant.optimizer_seed,
                "budget": budget,
                "path": str(output),
                "sha256": model_sha256(output),
                "subset_sha256": subset_hash,
                "best_epoch": result.best_epoch,
                "epochs_completed": result.epochs,
            }
            write_json_atomic(
                {
                    "identity": identity,
                    "record": record,
                    "model_sha256": record["sha256"],
                    "validation_metrics": asdict(result.val_metrics),
                },
                artifact,
            )
            records[f"{variant.name}:{budget}"] = record
    if len(records) != config.expected_new_models:
        raise ValueError("new PRP robustness model count differs")
    return _write_manifest(
        config,
        "train",
        {
            "models": records,
            "new_model_count": len(records),
            "new_variants": [variant.name for variant in config.new_variants],
            "shared_hyperparameters": TRAINING,
            "training_seed_lists": {
                str(size): list(training_seeds(config, size))
                for size in sorted({item.training_size for item in config.new_variants})
            },
        },
    )


def _all_model_records(
    config: ExploreConfig,
) -> dict[tuple[str, int], dict[str, object]]:
    records = _reused_model_records(config)
    trained = _load_manifest(config, "train")
    for raw in trained["models"].values():
        record = dict(raw)
        records[(str(record["variant"]), int(record["budget"]))] = record
    return records


def run_freeze(config: ExploreConfig) -> Path:
    records = _all_model_records(config)
    expected = {
        (variant.name, budget)
        for variant in config.variants
        for budget in config.budgets
    }
    if set(records) != expected:
        raise ValueError("frozen PRP model matrix differs")
    frozen: dict[str, object] = {}
    for (name, budget), record in sorted(records.items()):
        path = _resolve_artifact_path(record["path"])
        actual_hash = model_sha256(path)
        if actual_hash != record["sha256"]:
            raise ValueError(f"model changed before freeze: {name}, B={budget}")
        frozen[f"{name}:{budget}"] = {
            **record,
            "path": str(path),
            "sha256": actual_hash,
        }
    matrix_hash = payload_sha256(frozen)
    return _write_manifest(
        config,
        "freeze",
        {
            "freeze_schema": FREEZE_SCHEMA,
            "models": frozen,
            "model_count": len(frozen),
            "model_matrix_sha256": matrix_hash,
            "planned_nominal_seeds": list(config.evaluation_seeds),
            "planned_fixed_trace_kinds": list(config.fixed_trace_kinds),
            "planned_variants": [variant.name for variant in config.variants],
            "planned_budgets": list(config.budgets),
        },
    )


def _frozen_model_records(
    config: ExploreConfig,
) -> dict[tuple[str, int], dict[str, object]]:
    manifest = _load_manifest(config, "freeze")
    if manifest.get("freeze_schema") != FREEZE_SCHEMA:
        raise ValueError("unsupported PRP model freeze")
    raw = dict(manifest["models"])
    if payload_sha256(raw) != manifest["model_matrix_sha256"]:
        raise ValueError("frozen PRP model matrix hash differs")
    records = {}
    for key, value in raw.items():
        name, raw_budget = key.rsplit(":", 1)
        record = dict(value)
        path = Path(str(record["path"]))
        if model_sha256(path) != record["sha256"]:
            raise ValueError(f"frozen model changed: {name}, B={raw_budget}")
        records[(name, int(raw_budget))] = record
    return records


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


def _nominal_traces(config: ExploreConfig) -> tuple[EvaluationTrace, ...]:
    store = generate_random_waypoint_trace_store(
        RandomWaypointTraceStoreConfig(
            output=config.output / "evaluate" / "nominal" / "traces",
            event_count=config.event_count,
            conditions=("random_waypoint",),
            seed_start=min(config.evaluation_seeds),
            seed_count=len(config.evaluation_seeds),
        )
    )
    traces = _stored_traces(store)
    if tuple(item.seed for item in traces) != config.evaluation_seeds:
        raise ValueError("fresh nominal evaluation seeds differ")
    return traces


def _fixed_traces(config: ExploreConfig) -> tuple[EvaluationTrace, ...]:
    paper = load_paper_experiment_config(PAPER_CONFIG)
    traces = tuple(
        trace
        for trace in _fixed_figure8_traces(paper)
        if trace.trace_kind in config.fixed_trace_kinds
    )
    if tuple(trace.trace_kind for trace in traces) != config.fixed_trace_kinds:
        raise ValueError("fixed figure-eight trace order differs")
    for trace in traces:
        expected_rows = ROBOT_ARM_TRACE_ROWS[trace.trace_kind]
        expected_hash = ROBOT_ARM_TRACE_SHA256[trace.trace_kind]
        if len(trace.events) != expected_rows or trace.trace_sha256 != expected_hash:
            raise ValueError(f"fixed trace provenance differs: {trace.trace_kind}")
    if config.smoke:
        return tuple(replace(trace, events=trace.events[: config.event_count]) for trace in traces)
    return traces


def _reference_path(
    config: ExploreConfig,
    *,
    scope: str,
    trace: EvaluationTrace,
) -> Path:
    if scope == "fixed" and not config.smoke:
        path = PAPER_ROOT / "headline" / "references" / f"{trace.trace_kind}.json"
    else:
        path = (
            config.output
            / "evaluate"
            / scope
            / "references"
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


def _method(name: str) -> MethodConfig:
    return MethodConfig(
        name=name,
        execution_regime=ExecutionRegime.LEARNED_ONLINE,
        predictor=Predictor.NONE,
        horizon=0,
        beam_width=1,
        objective=Objective.LEARNED_TERMINAL_TEACHER,
        candidate_names=CANDIDATES,
    )


def cell_identity(
    config: ExploreConfig,
    *,
    scope: str,
    trace: EvaluationTrace,
    budget: int,
    variant: ModelVariant,
    reference: Path,
    record: Mapping[str, object],
    freeze_hash: str,
) -> dict[str, object]:
    method = _method(variant.name)
    payload = {
        "schema": CELL_SCHEMA,
        "experiment_fingerprint": config.fingerprint,
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
        "variant": variant.name,
        "training_size": variant.training_size,
        "training_seeds": list(training_seeds(config, variant.training_size)),
        "optimizer_seed": variant.optimizer_seed,
        "method": {
            **asdict(method),
            "execution_regime": method.execution_regime.value,
            "predictor": method.predictor.value,
            "objective": method.objective.value,
            "candidate_names": list(method.candidate_names),
        },
        "model_sha256": record["sha256"],
        "model_training_budget": budget,
        "model_freeze_sha256": freeze_hash,
        "spec_sha256": ROBOT_ARM_SPEC_SHA256,
        "rlolaeval_revision": RLOLAEVAL_REVISION,
        "binding_revision": BINDING_REVISION,
        "interpreter_revision": INTERPRETER_REVISION,
        "binding_build_profile": BINDING_BUILD_PROFILE,
        "reference_cache_schema": REFERENCE_CACHE_SCHEMA,
        "reference_cache_sha256": sha256_files((reference,)),
        "reference_semantics": {
            "selection": "learned_direct_inference",
            "metrics": "exact_cache_dynamic_and_total_radius",
        },
        "tool_sha256": tool_sha256(),
        "pzr_source_sha256": pzr_source_sha256(),
    }
    return {**payload, "fingerprint": payload_sha256(payload)}


def _execute_cell(job: EvaluationCellJob) -> dict[str, object]:
    manifest_path = job.directory / "manifest.json"
    summary_path = job.directory / "summary.csv"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        if manifest.get("schema") != CELL_SCHEMA:
            raise ValueError(f"unsupported PRP robustness cell: {job.directory}")
        if manifest.get("identity") != job.identity:
            raise ValueError(f"stale PRP robustness cell: {job.directory}")
        frame = pd.read_csv(summary_path)
        if len(frame) != 1:
            raise ValueError(f"PRP robustness cell summary differs: {job.directory}")
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
    row["optimizer_seed"] = int(job.identity["optimizer_seed"])
    write_csv_atomic(pd.DataFrame([row]), summary_path)
    write_json_atomic(
        {
            "schema": CELL_SCHEMA,
            "identity": job.identity,
            "status": row["status"],
            "diagnostic": diagnostic,
        },
        manifest_path,
    )
    timeseries = job.directory / "timeseries_diagnostic.csv"
    if row["status"] == RunState.COMPLETED.value and timeseries.is_file():
        timeseries.unlink()
    return row


def _import_fixed_baselines(config: ExploreConfig) -> tuple[pd.DataFrame, dict[str, object]]:
    if not config.reuse_existing:
        return pd.DataFrame(), {}
    paper_manifest_path = PAPER_ROOT / "headline" / "manifest.json"
    paper_manifest = load_json(paper_manifest_path)
    if paper_manifest.get("config_sha256") != PAPER_CONFIG_SHA256:
        raise ValueError("fixed paper manifest config differs")
    if paper_manifest.get("pzr_source_sha256") != EXPECTED_PZR_SOURCE_SHA256:
        raise ValueError("fixed paper manifest source differs")
    paper_summary_path = PAPER_ROOT / "headline" / "summary.csv"
    paper = pd.read_csv(paper_summary_path)
    clean20 = paper[paper["method"] == "pairwise_ranking_policy"].copy()
    clean20["method"] = "clean20"
    clean20["training_size"] = 20
    clean20["optimizer_seed"] = 42
    terminal = paper[paper["method"] == "mpc_terminal_beam"].copy()
    terminal["training_size"] = np.nan
    terminal["optimizer_seed"] = np.nan

    dart_manifest_path = DART_ROOT / "evaluate" / "fixed" / "manifest.json"
    dart_manifest = load_json(dart_manifest_path)
    if dart_manifest.get("effective_config_sha256") != DART_EFFECTIVE_CONFIG_SHA256:
        raise ValueError("fixed DART manifest differs")
    dart_summary_path = DART_ROOT / "evaluate" / "fixed" / "summary.csv"
    dart = pd.read_csv(dart_summary_path)
    clean36 = dart[dart["method"] == "clean36"].copy()
    clean36["training_size"] = 36
    clean36["optimizer_seed"] = 42

    imported = pd.concat([clean20, clean36, terminal], ignore_index=True)
    if len(imported) != config.expected_imported_fixed_cells:
        raise ValueError("imported fixed baseline count differs")
    return imported, {
        "paper_manifest": str(paper_manifest_path),
        "paper_manifest_sha256": sha256_files((paper_manifest_path,)),
        "paper_summary_sha256": sha256_files((paper_summary_path,)),
        "dart_manifest": str(dart_manifest_path),
        "dart_manifest_sha256": sha256_files((dart_manifest_path,)),
        "dart_summary_sha256": sha256_files((dart_summary_path,)),
        "cell_count": len(imported),
    }


def _run_scope(
    config: ExploreConfig,
    *,
    scope: str,
    traces: Sequence[EvaluationTrace],
) -> dict[str, object]:
    root = config.output / "evaluate" / scope
    freeze = _load_manifest(config, "freeze")
    freeze_hash = str(freeze["model_matrix_sha256"])
    models = _frozen_model_records(config)
    references = {
        trace.trace_id: _reference_path(config, scope=scope, trace=trace)
        for trace in traces
    }
    imported = pd.DataFrame()
    import_record: dict[str, object] = {}
    variants = config.variants
    if scope == "fixed" and config.reuse_existing:
        imported, import_record = _import_fixed_baselines(config)
        imported_names = {"clean20", "clean36"}
        variants = tuple(item for item in variants if item.name not in imported_names)

    jobs = []
    variant_by_name = {variant.name: variant for variant in config.variants}
    for trace in traces:
        for budget in config.budgets:
            for variant in variants:
                record = models[(variant.name, budget)]
                identity = cell_identity(
                    config,
                    scope=scope,
                    trace=trace,
                    budget=budget,
                    variant=variant,
                    reference=references[trace.trace_id],
                    record=record,
                    freeze_hash=freeze_hash,
                )
                jobs.append(
                    EvaluationCellJob(
                        stage="generalization" if scope == "nominal" else "headline",
                        directory=(
                            root
                            / "cells"
                            / trace.trace_source.value
                            / trace.condition
                            / f"seed-{trace.seed}"
                            / f"budget-{budget}"
                            / variant.name
                        ),
                        trace=trace,
                        budget=budget,
                        method=_method(variant.name),
                        runtime_method="pairwise_ranking_policy",
                        reference_path=references[trace.trace_id],
                        identity=identity,
                        model_directory=Path(str(record["path"])),
                        model_training_budget=budget,
                    )
                )
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
    if not imported.empty:
        summary = pd.concat([imported, summary], ignore_index=True)
    expected = (
        config.expected_nominal_cells
        if scope == "nominal"
        else config.expected_fixed_cells
    )
    if len(summary) != expected:
        raise ValueError(f"{scope} PRP robustness cell count differs")
    if summary.duplicated(["trace_id", "budget", "method"]).any():
        raise ValueError(f"{scope} PRP robustness cells are not unique")
    expected_methods = {item.name for item in config.variants}
    if scope == "fixed" and config.reuse_existing:
        expected_methods.add("mpc_terminal_beam")
    if set(summary["method"].astype(str)) != expected_methods:
        raise ValueError(f"{scope} PRP robustness method set differs")
    learned = summary[summary["method"].isin(variant_by_name)]
    if not (
        learned["model_training_budget"].astype(int)
        == learned["budget"].astype(int)
    ).all():
        raise ValueError(f"{scope} PRP robustness model budgets differ")
    expected_source = (
        TraceSource.GENERATED_NOMINAL.value
        if scope == "nominal"
        else TraceSource.FIXED_RLOLAEVAL.value
    )
    if set(summary["trace_source"]) != {expected_source}:
        raise ValueError(f"{scope} PRP robustness trace source differs")
    summary_path = root / "summary.csv"
    write_csv_atomic(summary, summary_path)
    manifest_path = root / "manifest.json"
    write_json_atomic(
        {
            "schema": SCHEMA,
            "experiment_fingerprint": config.fingerprint,
            "scope": scope,
            "status": "completed",
            "reported_cell_count": len(summary),
            "new_cell_count": len(jobs),
            "failure_count": int(
                (summary["status"] != RunState.COMPLETED.value).sum()
            ),
            "workers": config.workers,
            "matrix_wall_seconds": perf_counter() - started,
            "methods": sorted(expected_methods),
            "model_freeze_sha256": freeze_hash,
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
            "imported": import_record,
            "summary": str(summary_path),
            "summary_sha256": sha256_files((summary_path,)),
        },
        manifest_path,
    )
    return {
        "scope": scope,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_files((manifest_path,)),
        "reported_cell_count": len(summary),
        "new_cell_count": len(jobs),
        "failure_count": int(
            (summary["status"] != RunState.COMPLETED.value).sum()
        ),
    }


def run_evaluate(config: ExploreConfig) -> Path:
    _load_manifest(config, "freeze")
    nominal = _run_scope(config, scope="nominal", traces=_nominal_traces(config))
    fixed = _run_scope(config, scope="fixed", traces=_fixed_traces(config))
    if int(nominal["reported_cell_count"]) + int(fixed["reported_cell_count"]) != config.expected_reported_cells:
        raise ValueError("total PRP robustness reported cell count differs")
    if int(nominal["new_cell_count"]) + int(fixed["new_cell_count"]) != config.expected_new_cells:
        raise ValueError("total PRP robustness new cell count differs")
    return _write_manifest(
        config,
        "evaluate",
        {
            "scopes": {"nominal": nominal, "fixed": fixed},
            "reported_cell_count": config.expected_reported_cells,
            "new_cell_count": config.expected_new_cells,
            "failure_count": int(nominal["failure_count"]) + int(fixed["failure_count"]),
        },
    )


def aggregate_nominal(
    summary: pd.DataFrame,
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    high_loss_threshold: float = HIGH_LOSS_THRESHOLD,
) -> pd.DataFrame:
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap replicate count must be positive")
    data = trace_level_metrics(summary)
    rng = np.random.default_rng(bootstrap_seed)
    rows = []
    for (budget, method, size, optimizer_seed), frame in data.groupby(
        ["budget", "method", "training_size", "optimizer_seed"],
        sort=True,
    ):
        frame = frame.sort_values("seed")
        completed = frame[frame["status"] == RunState.COMPLETED.value]
        failures = len(frame) - len(completed)
        available = failures == 0
        fpr = completed["fpr"].to_numpy(dtype=np.float64)
        loss = completed["mean_approx_loss"].to_numpy(dtype=np.float64)
        if available and len(fpr):
            indices = rng.integers(
                0,
                len(fpr),
                size=(bootstrap_replicates, len(fpr)),
            )
            draws = np.mean(fpr[indices], axis=1)
            ci_low, ci_high = np.quantile(draws, (0.025, 0.975))
        else:
            ci_low = ci_high = np.nan
        negative = float(completed["reference_negative_count"].sum())
        rows.append(
            {
                "budget": int(budget),
                "method": str(method),
                "training_size": int(size),
                "optimizer_seed": int(optimizer_seed),
                "trace_count": len(frame),
                "valid_count": len(completed),
                "failed_count": failures,
                "available": available,
                "macro_fpr": float(np.mean(fpr)) if available and len(fpr) else np.nan,
                "macro_fpr_ci_low": float(ci_low),
                "macro_fpr_ci_high": float(ci_high),
                "pooled_fpr": (
                    float(completed["false_positive_count"].sum()) / negative
                    if available and negative > 0.0
                    else np.nan
                ),
                "mean_loss": float(np.mean(loss)) if available and len(loss) else np.nan,
                "median_loss": float(np.median(loss)) if available and len(loss) else np.nan,
                "loss_q90": (
                    float(np.quantile(loss, 0.9))
                    if available and len(loss)
                    else np.nan
                ),
                "max_loss": float(np.max(loss)) if available and len(loss) else np.nan,
                "high_loss_count": (
                    int(np.count_nonzero(loss > high_loss_threshold))
                    if available
                    else np.nan
                ),
                "high_loss_rate": (
                    float(np.mean(loss > high_loss_threshold))
                    if available and len(loss)
                    else np.nan
                ),
                "fallback_rate": float(
                    (frame["status"] == RunState.FALLBACK_FAILED.value).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_size_effects(
    summary: pd.DataFrame,
    *,
    reference_size: int = 84,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    high_loss_threshold: float = HIGH_LOSS_THRESHOLD,
) -> pd.DataFrame:
    data = trace_level_metrics(summary)
    data = data[data["optimizer_seed"].astype(int) == 42]
    rng = np.random.default_rng(bootstrap_seed)
    rows = []
    for budget in sorted(data["budget"].astype(int).unique()):
        point = data[data["budget"].astype(int) == budget]
        reference = point[
            point["training_size"].astype(int) == reference_size
        ].set_index("trace_id")
        for size in sorted(point["training_size"].astype(int).unique()):
            challenger = point[
                point["training_size"].astype(int) == size
            ].set_index("trace_id")
            if set(challenger.index) != set(reference.index):
                raise ValueError("size-curve traces are not paired")
            challenger = challenger.loc[sorted(challenger.index)]
            reference_aligned = reference.loc[sorted(reference.index)]
            completed = (
                (challenger["status"] == RunState.COMPLETED.value)
                & (reference_aligned["status"] == RunState.COMPLETED.value)
            )
            available = bool(completed.all())
            fpr_difference = (
                challenger["fpr"].to_numpy(dtype=np.float64)
                - reference_aligned["fpr"].to_numpy(dtype=np.float64)
            )
            challenger_loss = challenger["mean_approx_loss"].to_numpy(dtype=np.float64)
            reference_loss = reference_aligned["mean_approx_loss"].to_numpy(dtype=np.float64)
            positive = bool(
                np.all(challenger_loss > 0.0) and np.all(reference_loss > 0.0)
            )
            if available:
                indices = rng.integers(
                    0,
                    len(fpr_difference),
                    size=(bootstrap_replicates, len(fpr_difference)),
                )
                draws = np.mean(fpr_difference[indices], axis=1)
                ci_low, ci_high = np.quantile(draws, (0.025, 0.975))
            else:
                ci_low = ci_high = np.nan
            rows.append(
                {
                    "budget": budget,
                    "training_size": size,
                    "reference_size": reference_size,
                    "pair_count": len(challenger),
                    "valid_pair_count": int(completed.sum()),
                    "available": available,
                    "mean_fpr_difference": (
                        float(np.mean(fpr_difference)) if available else np.nan
                    ),
                    "fpr_difference_ci_low": float(ci_low),
                    "fpr_difference_ci_high": float(ci_high),
                    "geometric_mean_loss_ratio": (
                        float(np.exp(np.mean(np.log(challenger_loss / reference_loss))))
                        if available and positive
                        else np.nan
                    ),
                    "high_loss_rate_difference": (
                        float(
                            np.mean(challenger_loss > high_loss_threshold)
                            - np.mean(reference_loss > high_loss_threshold)
                        )
                        if available
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def optimizer_stability(
    aggregate: pd.DataFrame,
    *,
    stability_sizes: Sequence[int] = STABILITY_SIZES,
    optimizer_seeds: Sequence[int] = OPTIMIZER_SEEDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stability_sizes = tuple(stability_sizes)
    if len(stability_sizes) != 2:
        raise ValueError("optimizer comparison requires exactly two training sizes")
    selected = aggregate[
        aggregate["training_size"].astype(int).isin(stability_sizes)
    ].copy()
    expected = set(optimizer_seeds)
    rows = []
    differences = []
    for budget in sorted(selected["budget"].astype(int).unique()):
        point = selected[selected["budget"].astype(int) == budget]
        for size in stability_sizes:
            frame = point[point["training_size"].astype(int) == size].sort_values(
                "optimizer_seed"
            )
            if set(frame["optimizer_seed"].astype(int)) != expected:
                raise ValueError("optimizer replicate coverage differs")
            available = bool(frame["available"].astype(bool).all())
            for metric in ("macro_fpr", "mean_loss", "high_loss_rate"):
                values = frame[metric].to_numpy(dtype=np.float64)
                rows.append(
                    {
                        "budget": budget,
                        "training_size": size,
                        "metric": metric,
                        "optimizer_count": len(frame),
                        "available": available,
                        "median": float(np.median(values)) if available else np.nan,
                        "q25": float(np.quantile(values, 0.25)) if available else np.nan,
                        "q75": float(np.quantile(values, 0.75)) if available else np.nan,
                        "minimum": float(np.min(values)) if available else np.nan,
                        "maximum": float(np.max(values)) if available else np.nan,
                    }
                )
        left = point[
            point["training_size"].astype(int) == stability_sizes[0]
        ].set_index("optimizer_seed")
        right = point[
            point["training_size"].astype(int) == stability_sizes[1]
        ].set_index("optimizer_seed")
        if set(left.index.astype(int)) != expected or set(right.index.astype(int)) != expected:
            raise ValueError("paired optimizer seeds differ")
        for optimizer_seed in sorted(expected):
            lhs = left.loc[optimizer_seed]
            rhs = right.loc[optimizer_seed]
            available = bool(lhs["available"]) and bool(rhs["available"])
            differences.append(
                {
                    "budget": budget,
                    "optimizer_seed": optimizer_seed,
                    "reference_size": stability_sizes[0],
                    "challenger_size": stability_sizes[1],
                    "available": available,
                    "macro_fpr_difference": (
                        float(rhs["macro_fpr"] - lhs["macro_fpr"])
                        if available
                        else np.nan
                    ),
                    "mean_loss_difference": (
                        float(rhs["mean_loss"] - lhs["mean_loss"])
                        if available
                        else np.nan
                    ),
                    "high_loss_rate_difference": (
                        float(rhs["high_loss_rate"] - lhs["high_loss_rate"])
                        if available
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(differences)


def fixed_case_table(summary: pd.DataFrame) -> pd.DataFrame:
    data = trace_level_metrics(summary)
    data["available"] = data["status"] == RunState.COMPLETED.value
    data["high_loss"] = np.where(
        data["available"],
        data["mean_approx_loss"].astype(float) > HIGH_LOSS_THRESHOLD,
        np.nan,
    )
    columns = [
        "trace_source",
        "trace_kind",
        "condition",
        "budget",
        "method",
        "training_size",
        "optimizer_seed",
        "status",
        "available",
        "fpr",
        "mean_approx_loss",
        "high_loss",
    ]
    return data[columns].sort_values(
        ["condition", "budget", "training_size", "optimizer_seed", "method"],
        na_position="first",
    )


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _save_figure(fig, output: Path) -> tuple[Path, Path]:
    pdf = output.with_suffix(".pdf")
    png = output.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png, dpi=250, bbox_inches="tight", pad_inches=0.03)
    return pdf, png


def _plot_size_metric(
    aggregate: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    output: Path,
) -> tuple[Path, Path]:
    plt = _pyplot()
    frame = aggregate[aggregate["optimizer_seed"].astype(int) == 42]
    budgets = tuple(sorted(frame["budget"].astype(int).unique()))
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.8), squeeze=False)
    for index, budget in enumerate(budgets):
        axis = axes.flat[index]
        point = frame[frame["budget"].astype(int) == budget].sort_values(
            "training_size"
        )
        valid = point[point["available"].astype(bool)]
        axis.plot(
            valid["training_size"],
            valid[metric],
            color="#0072B2",
            marker="o",
            linestyle="-",
            linewidth=1.2,
            markersize=3.5,
        )
        if metric == "macro_fpr":
            axis.fill_between(
                valid["training_size"].to_numpy(dtype=float),
                valid["macro_fpr_ci_low"].to_numpy(dtype=float),
                valid["macro_fpr_ci_high"].to_numpy(dtype=float),
                color="#56B4E9",
                alpha=0.25,
                linewidth=0,
            )
        invalid = point[~point["available"].astype(bool)]
        if not invalid.empty:
            axis.scatter(
                invalid["training_size"],
                np.zeros(len(invalid)),
                marker="x",
                color="#D55E00",
            )
        values = valid[metric].to_numpy(dtype=float)
        if metric == "mean_loss" and len(values) and bool(np.all(values > 0.0)):
            axis.set_yscale("log")
        axis.set_title(f"B={budget}")
        axis.set_xlabel("Clean training trajectories")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="0.9", linewidth=0.6)
    for index in range(len(budgets), len(axes.flat)):
        axes.flat[index].set_visible(False)
    fig.tight_layout()
    paths = _save_figure(fig, output)
    plt.close(fig)
    return paths


def _plot_optimizer_metric(
    aggregate: pd.DataFrame,
    *,
    stability_sizes: Sequence[int],
    optimizer_seeds: Sequence[int],
    metric: str,
    ylabel: str,
    output: Path,
) -> tuple[Path, Path]:
    plt = _pyplot()
    stability_sizes = tuple(stability_sizes)
    frame = aggregate[
        aggregate["training_size"].astype(int).isin(stability_sizes)
    ]
    budgets = tuple(sorted(frame["budget"].astype(int).unique()))
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.8), squeeze=False)
    for index, budget in enumerate(budgets):
        axis = axes.flat[index]
        point = frame[frame["budget"].astype(int) == budget]
        for optimizer_seed in optimizer_seeds:
            run = point[
                point["optimizer_seed"].astype(int) == optimizer_seed
            ].sort_values("training_size")
            axis.plot(
                run["training_size"],
                run[metric],
                color="0.65",
                marker="o",
                linewidth=0.7,
                markersize=3,
            )
        medians = (
            point.groupby("training_size", sort=True)[metric].median().reset_index()
        )
        axis.plot(
            medians["training_size"],
            medians[metric],
            color="#D55E00",
            marker="D",
            linewidth=1.4,
            markersize=4,
            label="Median",
        )
        values = point[metric].dropna().to_numpy(dtype=float)
        if metric == "mean_loss" and len(values) and bool(np.all(values > 0.0)):
            axis.set_yscale("log")
        axis.set_xticks(stability_sizes)
        axis.set_title(f"B={budget}")
        axis.set_xlabel("Clean training trajectories")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="0.9", linewidth=0.6)
    for index in range(len(budgets), len(axes.flat)):
        axes.flat[index].set_visible(False)
    fig.tight_layout()
    paths = _save_figure(fig, output)
    plt.close(fig)
    return paths


def _plot_fixed_heatmap(
    fixed: pd.DataFrame,
    *,
    size_curve: Sequence[int],
    conditions: Sequence[str],
    budgets: Sequence[int],
    metric: str,
    label: str,
    output: Path,
) -> tuple[Path, Path]:
    plt = _pyplot()
    from matplotlib.colors import LogNorm

    methods = ("mpc_terminal_beam",) + tuple(
        f"clean{size}" for size in size_curve
    )
    conditions = tuple(conditions)
    budgets = tuple(budgets)
    if not bool((fixed["method"] == "mpc_terminal_beam").any()):
        methods = tuple(method for method in methods if method != "mpc_terminal_beam")
    selected = fixed[
        fixed["method"].isin(methods)
        & (
            fixed["optimizer_seed"].isna()
            | (fixed["optimizer_seed"].astype("Int64") == 42)
        )
    ]
    finite_values = selected.loc[
        selected["available"].astype(bool), metric
    ].to_numpy(dtype=float)
    norm = None
    if metric == "mean_approx_loss" and len(finite_values) and bool(
        np.all(finite_values > 0.0)
    ):
        norm = LogNorm(vmin=float(np.min(finite_values)), vmax=float(np.max(finite_values)))
    fig, axes = plt.subplots(1, len(conditions), figsize=(7.2, 3.2), squeeze=False)
    image = None
    for index, condition in enumerate(conditions):
        axis = axes.flat[index]
        point = selected[selected["trace_kind"] == condition]
        matrix = np.full((len(methods), len(budgets)), np.nan, dtype=float)
        for row_index, method in enumerate(methods):
            for column_index, budget in enumerate(budgets):
                cell = point[
                    (point["method"] == method)
                    & (point["budget"].astype(int) == budget)
                ]
                if len(cell) == 1 and bool(cell.iloc[0]["available"]):
                    matrix[row_index, column_index] = float(cell.iloc[0][metric])
        masked = np.ma.masked_invalid(matrix)
        image = axis.imshow(
            masked,
            aspect="auto",
            cmap="viridis",
            norm=norm,
            interpolation="nearest",
        )
        axis.set_title(condition.replace("figure8", "figure8").replace("_", "\n"))
        axis.set_xticks(range(len(budgets)), budgets, rotation=45, ha="right")
        axis.set_xlabel("Budget")
        if index == 0:
            axis.set_yticks(range(len(methods)), methods)
        else:
            axis.set_yticks(range(len(methods)), ())
        for row_index, column_index in np.argwhere(np.isnan(matrix)):
            axis.text(column_index, row_index, "×", ha="center", va="center", color="black")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label=label, shrink=0.78)
    fig.subplots_adjust(left=0.17, right=0.93, bottom=0.18, top=0.86, wspace=0.12)
    paths = _save_figure(fig, output)
    plt.close(fig)
    return paths


def run_report(config: ExploreConfig) -> Path:
    evaluation = _load_manifest(config, "evaluate")
    nominal_manifest = load_json(config.output / "evaluate" / "nominal" / "manifest.json")
    fixed_manifest = load_json(config.output / "evaluate" / "fixed" / "manifest.json")
    nominal_path = config.output / "evaluate" / "nominal" / "summary.csv"
    fixed_path = config.output / "evaluate" / "fixed" / "summary.csv"
    if sha256_files((nominal_path,)) != nominal_manifest["summary_sha256"]:
        raise ValueError("nominal summary hash differs")
    if sha256_files((fixed_path,)) != fixed_manifest["summary_sha256"]:
        raise ValueError("fixed summary hash differs")
    nominal = pd.read_csv(nominal_path)
    fixed_summary = pd.read_csv(fixed_path)
    aggregate = aggregate_nominal(nominal)
    effects = paired_size_effects(
        nominal,
        reference_size=config.stability_sizes[0],
    )
    optimizer_summary, optimizer_differences = optimizer_stability(
        aggregate,
        stability_sizes=config.stability_sizes,
        optimizer_seeds=config.optimizer_seeds,
    )
    fixed = fixed_case_table(fixed_summary)

    report = config.output / "report"
    tables = {
        "nominal_variant_summary.csv": aggregate,
        "nominal_size_effects_vs_clean84.csv": effects,
        "optimizer_stability_summary.csv": optimizer_summary,
        "optimizer_paired_clean148_minus_clean84.csv": optimizer_differences,
        "fixed_figure8_cases.csv": fixed,
    }
    for name, frame in tables.items():
        write_csv_atomic(frame, report / name)
    figures = (
        *_plot_size_metric(
            aggregate,
            metric="macro_fpr",
            ylabel="Macro FPR",
            output=report / "nominal_fpr_by_training_size",
        ),
        *_plot_size_metric(
            aggregate,
            metric="mean_loss",
            ylabel="Mean native loss",
            output=report / "nominal_loss_by_training_size",
        ),
        *_plot_size_metric(
            aggregate,
            metric="high_loss_rate",
            ylabel=f"Trace fraction with loss > {HIGH_LOSS_THRESHOLD:g}",
            output=report / "nominal_high_loss_by_training_size",
        ),
        *_plot_optimizer_metric(
            aggregate,
            stability_sizes=config.stability_sizes,
            optimizer_seeds=config.optimizer_seeds,
            metric="macro_fpr",
            ylabel="Macro FPR",
            output=report / "optimizer_seed_fpr",
        ),
        *_plot_optimizer_metric(
            aggregate,
            stability_sizes=config.stability_sizes,
            optimizer_seeds=config.optimizer_seeds,
            metric="high_loss_rate",
            ylabel=f"Trace fraction with loss > {HIGH_LOSS_THRESHOLD:g}",
            output=report / "optimizer_seed_high_loss",
        ),
        *_plot_fixed_heatmap(
            fixed,
            size_curve=config.size_curve,
            conditions=config.fixed_trace_kinds,
            budgets=config.budgets,
            metric="fpr",
            label="FPR",
            output=report / "fixed_figure8_fpr",
        ),
        *_plot_fixed_heatmap(
            fixed,
            size_curve=config.size_curve,
            conditions=config.fixed_trace_kinds,
            budgets=config.budgets,
            metric="mean_approx_loss",
            label="Mean native loss",
            output=report / "fixed_figure8_loss",
        ),
    )
    artifact_paths = tuple(report / name for name in tables) + figures
    return _write_manifest(
        config,
        "report",
        {
            "scientific_role": "prospective held-out exploratory model selection",
            "reported_cell_count": int(evaluation["reported_cell_count"]),
            "failure_count": int(evaluation["failure_count"]),
            "tables": {
                str(path): sha256_files((path,))
                for path in artifact_paths
                if path.suffix == ".csv"
            },
            "figures": {
                str(path): sha256_files((path,))
                for path in artifact_paths
                if path.suffix in {".pdf", ".png"}
            },
            "statistics": {
                "nominal_observational_unit": "generated trace seed",
                "nominal_trace_count": len(config.evaluation_seeds),
                "optimizer_replicates": len(config.optimizer_seeds),
                "optimizer_trace_products_treated_as_independent": False,
                "bootstrap": {
                    "type": "paired percentile bootstrap over trace seeds",
                    "replicates": BOOTSTRAP_REPLICATES,
                    "seed": BOOTSTRAP_SEED,
                },
                "high_loss_threshold": HIGH_LOSS_THRESHOLD,
                "fixed_case_population_interval": False,
            },
            "claims": {
                "paper_ready": False,
                "automatic_promotion": False,
                "fresh_nominal_mpc": False,
                "randomized_fault_generalization": False,
                "fresh_nominal_seeds_now_exploratory": list(config.evaluation_seeds),
                "reserved_untouched_followup_seeds": list(range(160, 180)),
            },
        },
    )


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
        "freeze": run_freeze,
        "evaluate": run_evaluate,
        "report": run_report,
    }
    print(f"start exploratory stage: {stage}", flush=True)
    result = functions[stage](config)
    print(f"complete exploratory stage: {stage}", flush=True)
    return result


def run_all(config: ExploreConfig) -> Path:
    print(json.dumps(config.identity, indent=2), flush=True)
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
    print(f"exploratory PRP robustness stage complete: {path}")
    if args.command in {"run", "report"}:
        manifest = load_json(_manifest_path(config, "report"))
        return 2 if int(manifest.get("failure_count", 0)) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
