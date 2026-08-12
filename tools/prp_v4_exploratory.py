#!/usr/bin/env python3
"""Disposable PRP v4 causal-feature, DAgger, and challenger study.

This tool deliberately lives outside :mod:`pzr`.  It reads the frozen v3
paper artifact, writes only to ``results/prp-v4-exploratory-v1``, and stops at
selection unless confirmation is requested explicitly.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from enum import Enum
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import yaml

from pzr.artifact_io import write_csv_atomic, write_json_atomic
from pzr.learning.artifacts import load_reducer_cost_dataset
from pzr.learning.dataset import ReducerCostDataset
from pzr.learning.diagnostics import validation_metrics
from pzr.learning.objectives import normalized_regrets, tolerant_best_mask
from pzr.learning.provenance import (
    model_sha256,
    payload_sha256,
    pzr_source_sha256,
    sha256_files,
)
from pzr.learning.ranker import FeatureSchema, ReducerPolicy, train_reducer_policy
from pzr.learning.training import dataset_sha256
from pzr.rtlola.actions import RtlolaActionCatalog, default_action_catalog
from pzr.rtlola.benchmark import (
    RtlolaBenchmarkConfig,
    RtlolaRunResult,
    run_event_trace_benchmark,
)
from pzr.rtlola.binding import (
    BINDING_BUILD_PROFILE,
    BINDING_REVISION,
    INTERPRETER_REVISION,
)
from pzr.rtlola.engine import (
    RtlolaBindingError,
    RtlolaEngine,
    RtlolaEvent,
    RtlolaStateRef,
)
from pzr.rtlola.features import (
    RTL_RANKING_FEATURE_NAMES,
    RTL_RANKING_FEATURE_SCHEMA,
    extract_ranking_features,
)
from pzr.rtlola.input_prediction import predict_future_events
from pzr.rtlola.learning_data import _aligned_root_costs
from pzr.rtlola.learning_traces import (
    RandomWaypointTraceStore,
    RandomWaypointTraceStoreConfig,
    generate_random_waypoint_trace_store,
    load_random_waypoint_trace_store,
)
from pzr.rtlola.reference import load_or_compute_reference
from pzr.rtlola.robot_arm import (
    ROBOT_ARM_SPEC_SHA256,
    ROBOT_ARM_TRACE_ROWS,
    ROBOT_ARM_TRACE_SHA256,
)
from pzr.rtlola.scenarios import scenario_by_name
from pzr.rtlola.search import (
    RtlolaNoFeasibleAction,
    RtlolaSearchResult,
    beam_search,
    full_width_terminal_search,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "prp-v4-exploratory-v1"
PAPER_ROOT = ROOT / "results" / "paper-evaluation-v3"
PAPER_CONFIG = ROOT / "experiments" / "paper_evaluation_v3.yaml"

SCHEMA = "pzr.prp-v4-exploratory.v1"
CELL_SCHEMA = "pzr.prp-v4-exploratory-cell.v1"
OVERLAY_SCHEMA = "pzr.prp-v4-clean-feature-overlay.v1"
FREEZE_SCHEMA = "pzr.prp-v4-confirmation-freeze.v1"
REFERENCE_METHOD = "mpc_terminal_beam_predictive_linear"
CHALLENGER_METHOD = "prp_causal_challenger_h2"
V3_METHOD = "g15_clean148"

BUDGETS = (40, 80, 120, 150, 200, 250, 500)
CANDIDATES = ("girard", "scott", "pca", "combastel")
CLEAN_TRAIN_SEEDS = tuple(range(20)) + tuple(range(26, 42)) + tuple(range(200, 312))
VALIDATION_SEEDS = tuple(range(20, 26))
DAGGER_SEEDS = tuple(range(312, 320))
SELECTION_SEEDS = tuple(range(320, 328))
CONFIRMATION_SEEDS = tuple(range(328, 348))
FIXED_TRACE_KINDS = (
    "figure8",
    "figure8_drift",
    "figure8_geofence",
    "figure8_drift_geofence",
)
BAD_DIAGNOSTIC_CELLS = ((100, 80), (111, 80), (114, 80), (107, 120), (109, 120))
MATCHED_DIAGNOSTIC_CELLS = ((112, 80), (115, 80), (117, 80), (102, 120), (110, 120))
PILOT_BUDGETS = (80, 120, 500)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260801
TAIL_MULTIPLIER = 1_000.0
WORKERS = 10
EVENT_COUNT = 500
TRAINING = {
    "epochs": 100,
    "batch_size": 256,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "patience": 10,
    "seed": 42,
}

PARENT_DATASETS = (
    (
        "paper_clean20",
        ROOT / "results/paper-evaluation-v2/prepare/teacher/dataset",
        ROOT / "results/paper-evaluation-v1/prepare/traces/training",
    ),
    (
        "dart_extra_clean16",
        ROOT / "results/dart-rescue-v1/prepare/extra-clean/dataset",
        ROOT / "results/dart-rescue-v1/prepare/traces-extra-clean16",
    ),
    (
        "old_scaling_extra_clean48",
        ROOT / "results/clean-scaling-exploratory/prepare/new-clean/dataset",
        ROOT / "results/clean-scaling-exploratory/prepare/traces-new-clean",
    ),
    (
        "new_clean64",
        ROOT / "results/prp-scale-robustness-exploratory/prepare/new-clean/dataset",
        ROOT / "results/prp-scale-robustness-exploratory/prepare/traces-new-clean",
    ),
)


class FeatureVariant(str, Enum):
    G15 = "g15"
    G20 = "g20"
    G25 = "g25"

    @property
    def clean_method(self) -> str:
        return f"{self.value}_clean148"

    @property
    def dagger_method(self) -> str:
        return f"{self.value}_dagger1"


CURRENT_JOINT_FEATURE_NAMES = tuple(f"current_{name}" for name in ("a1m", "a2m", "a3m", "a4m", "a5m"))
PREDICTED_JOINT_FEATURE_NAMES = tuple(f"predicted_{name}" for name in ("a1m", "a2m", "a3m", "a4m", "a5m"))
GEOMETRY20_SCHEMA = FeatureSchema(
    name="rtlola.current-zonotope-current-joints",
    version=1,
    feature_names=(*RTL_RANKING_FEATURE_NAMES, *CURRENT_JOINT_FEATURE_NAMES),
    log1p_features=RTL_RANKING_FEATURE_SCHEMA.log1p_features,
)
GEOMETRY25_SCHEMA = FeatureSchema(
    name="rtlola.current-zonotope-current-and-predicted-joints",
    version=1,
    feature_names=(
        *RTL_RANKING_FEATURE_NAMES,
        *CURRENT_JOINT_FEATURE_NAMES,
        *PREDICTED_JOINT_FEATURE_NAMES,
    ),
    log1p_features=RTL_RANKING_FEATURE_SCHEMA.log1p_features,
)
FEATURE_SCHEMAS = {
    FeatureVariant.G15: RTL_RANKING_FEATURE_SCHEMA,
    FeatureVariant.G20: GEOMETRY20_SCHEMA,
    FeatureVariant.G25: GEOMETRY25_SCHEMA,
}


@dataclass(frozen=True)
class ExploreConfig:
    output: Path = DEFAULT_OUTPUT
    budgets: tuple[int, ...] = BUDGETS
    clean_train_seeds: tuple[int, ...] = CLEAN_TRAIN_SEEDS
    validation_seeds: tuple[int, ...] = VALIDATION_SEEDS
    dagger_seeds: tuple[int, ...] = DAGGER_SEEDS
    selection_seeds: tuple[int, ...] = SELECTION_SEEDS
    confirmation_seeds: tuple[int, ...] = CONFIRMATION_SEEDS
    event_count: int = EVENT_COUNT
    workers: int = WORKERS
    epochs: int = int(TRAINING["epochs"])
    smoke: bool = False

    def __post_init__(self) -> None:
        groups = (
            set(self.clean_train_seeds),
            set(self.validation_seeds),
            set(self.dagger_seeds),
            set(self.selection_seeds),
            set(self.confirmation_seeds),
        )
        if any(left & right for index, left in enumerate(groups) for right in groups[index + 1:]):
            raise ValueError("training, validation, DAgger, selection, and confirmation seeds must be disjoint")
        if not self.budgets or tuple(sorted(self.budgets)) != self.budgets:
            raise ValueError("budgets must be non-empty and sorted")
        if self.event_count < 3 or self.workers < 1 or self.epochs < 1:
            raise ValueError("event count, workers, and epochs must be positive")

    @property
    def identity(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "output": str(self.output.resolve()),
            "budgets": list(self.budgets),
            "clean_train_seeds": list(self.clean_train_seeds),
            "validation_seeds": list(self.validation_seeds),
            "dagger_seeds": list(self.dagger_seeds),
            "selection_seeds": list(self.selection_seeds),
            "confirmation_seeds": list(self.confirmation_seeds),
            "event_count": self.event_count,
            "workers": self.workers,
            "smoke": self.smoke,
            "training": {**TRAINING, "epochs": self.epochs},
            "diagnostic_bad_cells": [list(item) for item in BAD_DIAGNOSTIC_CELLS],
            "diagnostic_matched_cells": [list(item) for item in MATCHED_DIAGNOSTIC_CELLS],
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "tail_multiplier": TAIL_MULTIPLIER,
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
    def expected_feature_cells(self) -> int:
        return len(self.selection_seeds) * len(self.budgets) * 4

    def expected_selection_cells(self, clean_winner: FeatureVariant) -> int:
        method_count = 6 if clean_winner is FeatureVariant.G15 else 7
        return len(self.selection_seeds) * len(self.budgets) * method_count


def smoke_config(*, workers: int = 1) -> ExploreConfig:
    return ExploreConfig(
        output=Path("/tmp/pzr-prp-v4-exploratory-smoke"),
        budgets=(40,),
        clean_train_seeds=(0,),
        validation_seeds=(20,),
        dagger_seeds=(312,),
        selection_seeds=(320,),
        confirmation_seeds=(328,),
        event_count=30,
        workers=workers,
        epochs=2,
        smoke=True,
    )


@dataclass(frozen=True)
class TraceRecord:
    trace_id: str
    trace_kind: str
    seed: int
    events: tuple[RtlolaEvent, ...]
    trace_sha256: str
    provenance: Mapping[str, object]


@dataclass(frozen=True)
class CleanBundle:
    dataset_g15: ReducerCostDataset
    dataset_g20: ReducerCostDataset
    dataset_g25: ReducerCostDataset
    metadata: pd.DataFrame

    def dataset(self, variant: FeatureVariant) -> ReducerCostDataset:
        return {
            FeatureVariant.G15: self.dataset_g15,
            FeatureVariant.G20: self.dataset_g20,
            FeatureVariant.G25: self.dataset_g25,
        }[variant]


@dataclass(frozen=True)
class EvaluationJob:
    config: ExploreConfig
    scope: str
    trace: TraceRecord
    budget: int
    method: str
    model_path: Path | None
    feature_variant: FeatureVariant | None
    challenger: bool
    reference_path: Path
    directory: Path


@dataclass(frozen=True)
class DiagnosticJob:
    config: ExploreConfig
    label: str
    trace: TraceRecord
    budget: int
    model_path: Path
    reference_path: Path
    directory: Path


def tool_sha256() -> str:
    wrapper = ROOT / "tools/run_prp_v4_exploratory.sh"
    paths = [Path(__file__).resolve()]
    if wrapper.is_file():
        paths.append(wrapper)
    return sha256_files(tuple(paths), relative_to=ROOT)


def _raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage_path(config: ExploreConfig, stage: str) -> Path:
    return config.output / stage / "manifest.json"


def _write_stage(config: ExploreConfig, stage: str, extra: Mapping[str, object]) -> Path:
    path = _stage_path(config, stage)
    write_json_atomic(
        {
            **config.identity,
            "experiment_fingerprint": config.fingerprint,
            "stage": stage,
            "status": "completed",
            **dict(extra),
        },
        path,
    )
    return path


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _load_stage(config: ExploreConfig, stage: str) -> dict[str, object]:
    path = _stage_path(config, stage)
    manifest = _load_json(path)
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"unsupported PRP v4 manifest: {path}")
    if manifest.get("experiment_fingerprint") != config.fingerprint:
        raise ValueError(f"stale PRP v4 manifest: {path}")
    return manifest


def schema_for(variant: FeatureVariant) -> FeatureSchema:
    return FEATURE_SCHEMAS[variant]


def schema_payload(variant: FeatureVariant) -> dict[str, object]:
    schema = schema_for(variant)
    return {
        "name": schema.name,
        "version": schema.version,
        "feature_names": list(schema.feature_names),
        "log1p_features": list(schema.log1p_features),
    }


def causal_joint_features(
    events: Sequence[RtlolaEvent],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return current and causally predicted joint rows with shapes ``(T, 5)``."""
    if not events:
        raise ValueError("joint features require at least one event")
    current_rows: list[tuple[float, ...]] = []
    predicted_rows: list[tuple[float, ...]] = []
    for step, event in enumerate(events):
        current = tuple(float(value) for value in event.values[1:6])
        history = events[max(0, step - 2):step + 1]
        prediction = predict_future_events(
            history,
            predictor="linear",
            horizon=1,
            step_seconds=0.1,
            timestamp_channel_indices=(0,),
        )
        predicted = tuple(float(value) for value in prediction.events[0].values[1:6])
        current_rows.append(current)
        predicted_rows.append(predicted)
    current_array = np.asarray(current_rows, dtype=np.float32)
    predicted_array = np.asarray(predicted_rows, dtype=np.float32)
    if current_array.shape != (len(events), 5) or predicted_array.shape != current_array.shape:
        raise AssertionError("robot-joint feature shape differs")
    if not np.all(np.isfinite(current_array)) or not np.all(np.isfinite(predicted_array)):
        raise ValueError("robot-joint features contain non-finite values")
    return current_array, predicted_array


def augment_features(
    geometry15: NDArray[np.floating],
    current_joints: NDArray[np.floating],
    predicted_joints: NDArray[np.floating],
    variant: FeatureVariant,
) -> NDArray[np.float32]:
    base = np.asarray(geometry15, dtype=np.float32)
    current = np.asarray(current_joints, dtype=np.float32)
    predicted = np.asarray(predicted_joints, dtype=np.float32)
    if base.shape != (15,) or current.shape != (5,) or predicted.shape != (5,):
        raise ValueError("Geometry15/current/predicted feature shapes must be 15/5/5")
    if variant is FeatureVariant.G15:
        result = base
    elif variant is FeatureVariant.G20:
        result = np.concatenate((base, current)).astype(np.float32)
    else:
        result = np.concatenate((base, current, predicted)).astype(np.float32)
    if result.shape != (len(schema_for(variant).feature_names),):
        raise AssertionError("augmented feature schema differs")
    if not np.all(np.isfinite(result)):
        raise ValueError("augmented features contain non-finite values")
    return result


def _paper_contract() -> tuple[dict[str, object], dict[str, object]]:
    config_payload = yaml.safe_load(PAPER_CONFIG.read_text())
    if config_payload.get("experiment_id") != "paper-evaluation-v3":
        raise ValueError("canonical paper configuration is not v3")
    frozen = dict(config_payload["frozen_policy"])
    return config_payload, frozen


def _v3_snapshot() -> dict[str, object]:
    paths = (
        PAPER_CONFIG,
        PAPER_ROOT / "train/manifest.json",
        PAPER_ROOT / "generalization/manifest.json",
        PAPER_ROOT / "generalization/summary.csv",
        PAPER_ROOT / "generalization/timeseries.csv",
        PAPER_ROOT / "science-validate/manifest.json",
    )
    return {str(path.relative_to(ROOT)): _raw_sha256(path) for path in paths}


def _v3_model_records(config: ExploreConfig) -> dict[int, dict[str, object]]:
    manifest = _load_json(PAPER_ROOT / "train/manifest.json")
    records: dict[int, dict[str, object]] = {}
    for budget in config.budgets:
        record = dict(manifest["models_by_budget"][str(budget)])
        path = ROOT / str(record["path"])
        if record.get("training_budget") != budget or record.get("optimizer_seed") != 42:
            raise ValueError(f"v3 specialist identity differs at budget {budget}")
        if tuple(record.get("training_seeds", ())) != CLEAN_TRAIN_SEEDS:
            raise ValueError(f"v3 specialist training seeds differ at budget {budget}")
        if model_sha256(path) != record["sha256"]:
            raise ValueError(f"v3 specialist model hash differs at budget {budget}")
        records[budget] = {**record, "path": str(path)}
    return records


def run_preflight(config: ExploreConfig) -> Path:
    if BINDING_BUILD_PROFILE != "release":
        raise ValueError("PRP v4 exploration requires a release binding")
    paper, frozen = _paper_contract()
    train_manifest = _load_json(PAPER_ROOT / "train/manifest.json")
    validate_manifest = _load_json(PAPER_ROOT / "science-validate/manifest.json")
    config_hash = _raw_sha256(PAPER_CONFIG)
    if train_manifest.get("config_sha256") != config_hash:
        raise ValueError("v3 train manifest configuration hash differs")
    if train_manifest.get("pzr_source_sha256") != pzr_source_sha256():
        raise ValueError("current PZR scientific source differs from v3")
    if validate_manifest.get("status") not in {"completed", "completed_with_failures"}:
        raise ValueError("v3 scientific validation is incomplete")
    expected_hashes = dict(frozen["source_dataset_sha256"])
    parent_records = {}
    for name, dataset_path, trace_store_path in PARENT_DATASETS:
        actual = dataset_sha256(dataset_path)
        if actual != expected_hashes[name]:
            raise ValueError(f"{name} teacher dataset hash differs: {actual}")
        store = load_random_waypoint_trace_store(trace_store_path)
        parent_records[name] = {
            "dataset": str(dataset_path),
            "dataset_sha256": actual,
            "trace_store": str(trace_store_path),
            "trace_store_manifest_sha256": store.manifest_sha256,
            "seed_count": store.seed_count,
        }
    models = _v3_model_records(config)
    return _write_stage(
        config,
        "preflight",
        {
            "scientific_role": "disposable exploratory model selection",
            "paper_experiment_id": paper["experiment_id"],
            "paper_config_sha256": config_hash,
            "v3_snapshot": _v3_snapshot(),
            "parent_datasets": parent_records,
            "v3_models": {str(key): value for key, value in models.items()},
        },
    )


def _selected_parent_rows(
    config: ExploreConfig,
) -> tuple[ReducerCostDataset, pd.DataFrame, dict[int, object]]:
    datasets = []
    frames = []
    trace_by_seed: dict[int, object] = {}
    allowed = set(config.clean_train_seeds) | set(config.validation_seeds)
    for name, dataset_path, trace_store_path in PARENT_DATASETS:
        dataset, metadata, _ = load_reducer_cost_dataset(dataset_path)
        selected = metadata["seed"].astype(int).isin(allowed) & metadata["budget"].astype(int).isin(config.budgets)
        indices = np.flatnonzero(selected.to_numpy())
        if not len(indices):
            continue
        subset = dataset.subset(indices)
        frame = metadata.iloc[indices].reset_index(drop=True).copy()
        frame.insert(0, "source_dataset", name)
        datasets.append(subset)
        frames.append(frame)
        store = load_random_waypoint_trace_store(trace_store_path)
        for item in store.traces:
            if item.seed not in allowed:
                continue
            prior = trace_by_seed.get(item.seed)
            if prior is not None and prior.trace.metadata.trace_sha256 != item.trace.metadata.trace_sha256:
                raise ValueError(f"conflicting clean trace hash for seed {item.seed}")
            trace_by_seed[item.seed] = item
    combined = ReducerCostDataset.concatenate(datasets)
    metadata = pd.concat(frames, ignore_index=True)
    order = np.lexsort((metadata["step"], metadata["budget"], metadata["seed"]))
    combined = combined.subset(order)
    metadata = metadata.iloc[order].reset_index(drop=True)
    expected = set(config.clean_train_seeds) | set(config.validation_seeds)
    if set(metadata["seed"].astype(int)) != expected:
        raise ValueError("Clean148 overlay seed coverage differs")
    if tuple(metadata["sample_id"].astype(str)) != combined.sample_ids:
        raise ValueError("teacher rows and metadata are not aligned")
    return combined, metadata, trace_by_seed


def _write_overlay_arrays(path: Path, **arrays: NDArray[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def run_prepare(config: ExploreConfig) -> Path:
    preflight = _load_stage(config, "preflight")
    dataset, metadata, trace_by_seed = _selected_parent_rows(config)
    joint_cache = {
        seed: causal_joint_features(item.trace.events)
        for seed, item in trace_by_seed.items()
    }
    g20 = np.empty((dataset.num_samples, 20), dtype=np.float32)
    g25 = np.empty((dataset.num_samples, 25), dtype=np.float32)
    trace_hashes = []
    for row_index, row in metadata.iterrows():
        seed = int(row["seed"])
        step = int(row["step"])
        item = trace_by_seed[seed]
        if str(row["trace_id"]) != item.trace_id or not 0 <= step < len(item.trace.events):
            raise ValueError(f"sample-to-trace join differs: {row['sample_id']}")
        current, predicted = joint_cache[seed]
        g20[row_index] = augment_features(
            dataset.features[row_index], current[step], predicted[step], FeatureVariant.G20,
        )
        g25[row_index] = augment_features(
            dataset.features[row_index], current[step], predicted[step], FeatureVariant.G25,
        )
        trace_hashes.append(item.trace.metadata.trace_sha256)
    metadata = metadata.copy()
    metadata["trace_sha256"] = trace_hashes
    overlay_path = config.output / "prepare/clean-feature-overlay.npz"
    metadata_path = config.output / "prepare/clean-feature-overlay.csv"
    _write_overlay_arrays(
        overlay_path,
        sample_ids=np.asarray(dataset.sample_ids),
        features_g20=g20,
        features_g25=g25,
    )
    write_csv_atomic(metadata, metadata_path)

    generated = generate_random_waypoint_trace_store(
        RandomWaypointTraceStoreConfig(
            output=config.output / "prepare/exploration-traces",
            event_count=config.event_count,
            conditions=("random_waypoint",),
            seed_start=min(config.dagger_seeds),
            seed_count=max(config.selection_seeds) - min(config.dagger_seeds) + 1,
        )
    )
    expected_generated = tuple(
        range(min(config.dagger_seeds), max(config.selection_seeds) + 1)
    )
    if tuple(item.seed for item in generated.traces) != expected_generated:
        raise ValueError("exploration trace seed coverage differs")
    return _write_stage(
        config,
        "prepare",
        {
            "preflight_sha256": sha256_files((_stage_path(config, "preflight"),)),
            "overlay_schema": OVERLAY_SCHEMA,
            "sample_count": dataset.num_samples,
            "overlay": str(overlay_path),
            "overlay_sha256": sha256_files((overlay_path,)),
            "metadata": str(metadata_path),
            "metadata_sha256": sha256_files((metadata_path,)),
            "feature_schemas": {
                item.value: schema_payload(item) for item in FeatureVariant
            },
            "exploration_trace_store": str(generated.root),
            "exploration_trace_store_manifest_sha256": generated.manifest_sha256,
            "v3_snapshot": preflight["v3_snapshot"],
        },
    )


def load_clean_bundle(config: ExploreConfig) -> CleanBundle:
    manifest = _load_stage(config, "prepare")
    dataset, metadata, _ = _selected_parent_rows(config)
    overlay_path = Path(str(manifest["overlay"]))
    metadata_path = Path(str(manifest["metadata"]))
    if sha256_files((overlay_path,)) != manifest["overlay_sha256"]:
        raise ValueError("clean feature overlay hash differs")
    if sha256_files((metadata_path,)) != manifest["metadata_sha256"]:
        raise ValueError("clean feature metadata hash differs")
    persisted = pd.read_csv(metadata_path)
    if tuple(persisted["sample_id"].astype(str)) != dataset.sample_ids:
        raise ValueError("clean feature overlay sample alignment differs")
    with np.load(overlay_path, allow_pickle=False) as arrays:
        if tuple(str(value) for value in arrays["sample_ids"]) != dataset.sample_ids:
            raise ValueError("clean feature overlay identifiers differ")
        g20 = arrays["features_g20"]
        g25 = arrays["features_g25"]
    common = {
        "teacher_costs": dataset.teacher_costs,
        "feasible": dataset.feasible,
        "candidate_names": dataset.candidate_names,
        "splits": dataset.splits,
        "sample_ids": dataset.sample_ids,
    }
    return CleanBundle(
        dataset,
        ReducerCostDataset(features=g20, feature_names=GEOMETRY20_SCHEMA.feature_names, **common),
        ReducerCostDataset(features=g25, feature_names=GEOMETRY25_SCHEMA.feature_names, **common),
        persisted,
    )


def _subset_for_budget(
    dataset: ReducerCostDataset,
    metadata: pd.DataFrame,
    budget: int,
) -> tuple[ReducerCostDataset, pd.DataFrame]:
    selected = metadata["budget"].astype(int).to_numpy() == budget
    indices = np.flatnonzero(selected)
    subset = dataset.subset(indices)
    frame = metadata.iloc[indices].reset_index(drop=True)
    if set(frame["split"].astype(str)) != {"train", "validation"}:
        raise ValueError(f"budget {budget} lacks train/validation rows")
    return subset, frame


def _model_record(path: Path, method: str, budget: int, variant: FeatureVariant) -> dict[str, object]:
    return {
        "method": method,
        "budget": budget,
        "feature_variant": variant.value,
        "feature_schema": schema_payload(variant),
        "path": str(path),
        "sha256": model_sha256(path),
    }


def run_train_clean(config: ExploreConfig) -> Path:
    bundle = load_clean_bundle(config)
    records: dict[str, object] = {}
    v3 = _v3_model_records(config)
    for budget, raw in v3.items():
        records[f"{V3_METHOD}:{budget}"] = {
            **raw,
            "method": V3_METHOD,
            "feature_variant": FeatureVariant.G15.value,
        }
    for variant in (FeatureVariant.G20, FeatureVariant.G25):
        for budget in config.budgets:
            output = config.output / "train-clean" / variant.clean_method / f"budget-{budget}"
            subset, frame = _subset_for_budget(bundle.dataset(variant), bundle.metadata, budget)
            frame = frame.copy()
            frame["dataset_label"] = "clean148"
            identity = {
                "experiment_fingerprint": config.fingerprint,
                "method": variant.clean_method,
                "budget": budget,
                "feature_schema": schema_payload(variant),
                "sample_ids_sha256": payload_sha256({"sample_ids": list(subset.sample_ids)}),
                "training": {**TRAINING, "epochs": config.epochs},
            }
            artifact = output / "exploratory_training.json"
            if artifact.is_file():
                loaded = _load_json(artifact)
                if loaded.get("identity") != identity or model_sha256(output) != loaded.get("model_sha256"):
                    raise ValueError(f"stale clean exploratory model: {output}")
                records[f"{variant.clean_method}:{budget}"] = loaded["record"]
                continue
            policy, result = train_reducer_policy(
                subset,
                schema_for(variant),
                objective="pairwise",
                epochs=config.epochs,
                batch_size=int(TRAINING["batch_size"]),
                learning_rate=float(TRAINING["learning_rate"]),
                weight_decay=float(TRAINING["weight_decay"]),
                patience=min(int(TRAINING["patience"]), config.epochs),
                seed=int(TRAINING["seed"]),
            )
            policy.save(output)
            write_csv_atomic(validation_metrics(policy, subset, frame), output / "validation_metrics.csv")
            record = {
                **_model_record(output, variant.clean_method, budget, variant),
                "best_epoch": result.best_epoch,
                "epochs_completed": result.epochs,
                "training_source": "clean148_feature_overlay",
            }
            write_json_atomic(
                {
                    "identity": identity,
                    "model_sha256": record["sha256"],
                    "record": record,
                    "validation_metrics": asdict(result.val_metrics),
                },
                artifact,
            )
            records[f"{variant.clean_method}:{budget}"] = record
    expected = len(config.budgets) * 3
    if len(records) != expected:
        raise ValueError(f"clean model matrix has {len(records)} records, expected {expected}")
    return _write_stage(config, "train-clean", {"models": records, "model_count": len(records)})


def _clean_model_records(config: ExploreConfig) -> dict[tuple[str, int], dict[str, object]]:
    manifest = _load_stage(config, "train-clean")
    records = {}
    for key, raw in dict(manifest["models"]).items():
        method, raw_budget = key.rsplit(":", 1)
        record = dict(raw)
        path = Path(str(record["path"]))
        if model_sha256(path) != record["sha256"]:
            raise ValueError(f"clean model hash differs: {key}")
        records[(method, int(raw_budget))] = record
    return records


def _trace_record(item: object, *, limit: int | None = None) -> TraceRecord:
    events = tuple(item.trace.events if limit is None else item.trace.events[:limit])
    return TraceRecord(
        trace_id=item.trace_id,
        trace_kind=item.condition,
        seed=item.seed,
        events=events,
        trace_sha256=item.trace.metadata.trace_sha256,
        provenance={
            "trace_store_relative_path": str(item.relative_path),
            "generator_config": item.trace.metadata.generator_config,
        },
    )


def _exploration_traces(config: ExploreConfig, seeds: Sequence[int]) -> tuple[TraceRecord, ...]:
    manifest = _load_stage(config, "prepare")
    store = load_random_waypoint_trace_store(Path(str(manifest["exploration_trace_store"])))
    traces = tuple(_trace_record(store.traces_for_seed(seed)[0]) for seed in seeds)
    if tuple(trace.seed for trace in traces) != tuple(seeds):
        raise ValueError("exploration traces differ from requested seeds")
    return traces


def _diagnostic_traces(config: ExploreConfig) -> dict[int, TraceRecord]:
    generalization = _load_json(PAPER_ROOT / "generalization/manifest.json")
    trace_store_path = PAPER_ROOT / "generalization/traces/generated-nominal"
    store = load_random_waypoint_trace_store(trace_store_path)
    records = {}
    by_seed = {item.seed: item for item in store.traces}
    manifest_by_seed = {int(item["seed"]): item for item in generalization["trace_manifest"]}
    for seed, _ in (*BAD_DIAGNOSTIC_CELLS, *MATCHED_DIAGNOSTIC_CELLS):
        item = by_seed[seed]
        expected = manifest_by_seed[seed]
        if item.trace.metadata.trace_sha256 != expected["trace_sha256"]:
            raise ValueError(f"v3 diagnostic trace hash differs for seed {seed}")
        records[seed] = _trace_record(item, limit=config.event_count if config.smoke else None)
    return records


def extract_policy_features(
    engine: RtlolaEngine,
    state: RtlolaStateRef,
    event: RtlolaEvent,
    history: Sequence[RtlolaEvent],
    budget: int,
    variant: FeatureVariant,
) -> NDArray[np.float32]:
    geometry15 = extract_ranking_features(engine, state, budget)
    current = np.asarray(event.values[1:6], dtype=np.float32)
    prediction = predict_future_events(
        history,
        predictor="linear",
        horizon=1,
        step_seconds=0.1,
        timestamp_channel_indices=(0,),
    )
    predicted = np.asarray(prediction.events[0].values[1:6], dtype=np.float32)
    return augment_features(geometry15, current, predicted, variant)


def _direct_decision(
    policy: ReducerPolicy,
    catalog: RtlolaActionCatalog,
    engine: RtlolaEngine,
    state: RtlolaStateRef,
    event: RtlolaEvent,
    budget: int,
    features: NDArray[np.float32],
) -> tuple[RtlolaSearchResult, NDArray[np.float64], tuple[str, ...]]:
    metrics = engine.metrics(state)
    if metrics.dynamic_generator_count <= budget:
        step = engine.branch_step(state, event, catalog.no_op, budget)
        decision = RtlolaSearchResult(
            first_action=catalog.no_op,
            first_action_budget=budget,
            first_step=step,
            predicted_cost=0.0,
            predicted_sequence=(catalog.no_op.name,),
            evaluated_leaves=1,
            pruned_branches=0,
            mpc_variant="direct_policy",
            root_strategy="ranked_direct",
        )
        return decision, np.full(len(CANDIDATES), np.nan), (catalog.no_op.name,)
    scores = np.asarray(policy.predict_scores(features), dtype=np.float64)
    if scores.shape != (len(CANDIDATES),) or not np.all(np.isfinite(scores)):
        raise ValueError("exploratory policy returned invalid scores")
    order = np.argsort(scores, kind="stable")
    failures = 0
    ranking = tuple(policy.candidate_names[int(index)] for index in order)
    for index in order:
        name = policy.candidate_names[int(index)]
        action = catalog.by_name[name]
        if action.explicit_budget and budget < metrics.dimension:
            failures += 1
            continue
        try:
            step = engine.branch_step(state, event, action, budget)
        except RtlolaBindingError:
            failures += 1
            continue
        return RtlolaSearchResult(
            first_action=action,
            first_action_budget=budget,
            first_step=step,
            predicted_cost=float(scores[index]),
            predicted_sequence=(name,),
            evaluated_leaves=1,
            pruned_branches=0,
            reducer_failure_count=failures,
            infeasible_candidate_count=failures,
            mpc_variant="direct_policy",
            root_strategy="ranked_direct",
        ), scores, ranking
    try:
        step = engine.branch_step(state, event, catalog.fallback, budget)
    except RtlolaBindingError as exc:
        raise RtlolaNoFeasibleAction("exploratory policy and fallback were infeasible") from exc
    return RtlolaSearchResult(
        first_action=catalog.fallback,
        first_action_budget=budget,
        first_step=step,
        predicted_cost=float("nan"),
        predicted_sequence=(catalog.fallback.name,),
        evaluated_leaves=1,
        pruned_branches=0,
        fallback_used=True,
        reducer_failure_count=failures,
        infeasible_candidate_count=failures,
        mpc_variant="direct_policy",
        root_strategy="ranked_direct",
    ), scores, ranking


def challenger_root_names(ranking: Sequence[str]) -> tuple[str, str]:
    """Return the deterministic PRP/Scott root shortlist."""
    ordered = tuple(str(name) for name in ranking)
    if set(ordered) != set(CANDIDATES) or len(ordered) != len(CANDIDATES):
        raise ValueError("challenger ranking must contain the candidate catalog exactly once")
    proposal = ordered[0]
    return (proposal, "scott") if proposal != "scott" else ("scott", ordered[1])


def tolerance_aware_policy_error(
    costs: NDArray[np.float64],
    feasible: NDArray[np.bool_],
    selected_index: int | None,
) -> bool:
    """Treat every tolerance-aware best action as correct."""
    if selected_index is None:
        return True
    best = tolerant_best_mask(costs[None, :], feasible[None, :])[0]
    return not bool(best[selected_index])


def causal_challenger_decision(
    policy: ReducerPolicy,
    catalog: RtlolaActionCatalog,
    engine: RtlolaEngine,
    state: RtlolaStateRef,
    event: RtlolaEvent,
    history: Sequence[RtlolaEvent],
    budget: int,
    features: NDArray[np.float32],
) -> tuple[RtlolaSearchResult, NDArray[np.float64], tuple[str, ...]]:
    """Choose between two PRP-shortlisted roots using H=2 causal terminal loss."""
    metrics = engine.metrics(state)
    if metrics.dynamic_generator_count <= budget:
        return _direct_decision(policy, catalog, engine, state, event, budget, features)
    scores = np.asarray(policy.predict_scores(features), dtype=np.float64)
    order = np.argsort(scores, kind="stable")
    ranking = tuple(policy.candidate_names[int(index)] for index in order)
    roots = challenger_root_names(ranking)
    future = predict_future_events(
        history,
        predictor="linear",
        horizon=1,
        step_seconds=0.1,
        timestamp_channel_indices=(0,),
    ).events
    ordinary = []
    fallback = []
    for name in roots:
        decision = beam_search(
            engine,
            state,
            event,
            future,
            catalog.mpc_candidates,
            budget,
            beam_width=4,
            fallback=catalog.fallback,
            none_action=catalog.no_op,
            use_reference_loss=True,
            forced_first_action=catalog.by_name[name],
            configured_horizon=1,
        )
        (fallback if decision.fallback_used else ordinary).append(decision)
    choices = ordinary or fallback
    best = min(choices, key=lambda item: (item.predicted_cost, item.predicted_sequence))
    merged = replace(
        best,
        evaluated_leaves=sum(item.evaluated_leaves for item in choices),
        reducer_failure_count=sum(item.reducer_failure_count for item in choices),
        infeasible_candidate_count=sum(item.infeasible_candidate_count for item in choices),
        mpc_variant=CHALLENGER_METHOD,
        root_strategy="prp_scott_two_root_full_continuation",
        input_predictor="linear",
        prediction_step_seconds=0.1,
        root_evaluations=tuple(row for item in choices for row in item.root_evaluations),
    )
    if merged.evaluated_leaves > 8:
        raise AssertionError("causal challenger evaluated more than eight terminal leaves")
    return merged, scores, ranking


class ExploratoryPolicy:
    """Direct causal policy with optional shadow teachers and challenger search."""

    def __init__(
        self,
        policy: ReducerPolicy,
        variant: FeatureVariant,
        *,
        events: Sequence[RtlolaEvent],
        challenger: bool = False,
        shadow: bool = False,
    ) -> None:
        if policy.feature_schema != schema_for(variant):
            raise ValueError("exploratory policy feature schema differs")
        if policy.candidate_names != CANDIDATES:
            raise ValueError("exploratory policy candidate catalog differs")
        self.policy = policy
        self.variant = variant
        self.events = tuple(events)
        self.challenger = challenger
        self.shadow = shadow
        self.catalog = default_action_catalog(CANDIDATES)
        self.history: list[RtlolaEvent] = []
        self.diagnostics: list[dict[str, object]] = []

    def choose(
        self,
        engine: RtlolaEngine,
        state: RtlolaStateRef,
        event: RtlolaEvent,
        budget: int,
    ) -> RtlolaSearchResult:
        step = len(self.history)
        if step >= len(self.events) or self.events[step] != event:
            raise ValueError("exploratory policy event history is not aligned")
        self.history.append(event)
        features = extract_policy_features(
            engine, state, event, self.history, budget, self.variant,
        )
        chooser = causal_challenger_decision if self.challenger else _direct_decision
        if self.challenger:
            decision, scores, ranking = chooser(
                self.policy, self.catalog, engine, state, event, self.history, budget, features,
            )
        else:
            decision, scores, ranking = chooser(
                self.policy, self.catalog, engine, state, event, budget, features,
            )
        row: dict[str, object] = {
            "step": step,
            "budget": budget,
            "over_bound": engine.metrics(state).dynamic_generator_count > budget,
            "selected_action": decision.first_action.name,
            "ranking": json.dumps(ranking),
            "top_two_margin": (
                float(np.sort(scores, kind="stable")[1] - np.sort(scores, kind="stable")[0])
                if np.all(np.isfinite(scores)) else np.nan
            ),
            "evaluated_leaves": decision.evaluated_leaves,
            "fallback_used": decision.fallback_used,
            **{f"score_{name}": scores[index] for index, name in enumerate(CANDIDATES)},
            **{name: features[index] for index, name in enumerate(schema_for(self.variant).feature_names)},
        }
        if self.shadow and bool(row["over_bound"]) and step + 1 < len(self.events):
            before_dynamic, before_total = engine.matrices(state)
            offline = full_width_terminal_search(
                engine,
                state,
                event,
                (self.events[step + 1],),
                self.catalog.mpc_candidates,
                budget,
                fallback=self.catalog.fallback,
                none_action=self.catalog.no_op,
                configured_horizon=1,
            )
            predicted = predict_future_events(
                self.history,
                predictor="linear",
                horizon=1,
                step_seconds=0.1,
                timestamp_channel_indices=(0,),
            )
            causal = full_width_terminal_search(
                engine,
                state,
                event,
                predicted.events,
                self.catalog.mpc_candidates,
                budget,
                fallback=self.catalog.fallback,
                none_action=self.catalog.no_op,
                configured_horizon=1,
            )
            after_dynamic, after_total = engine.matrices(state)
            if not np.array_equal(before_dynamic, after_dynamic) or not np.array_equal(before_total, after_total):
                raise AssertionError("shadow teacher mutated the learner state")
            offline_costs, offline_feasible = _aligned_root_costs(offline.root_evaluations, CANDIDATES)
            causal_costs, causal_feasible = _aligned_root_costs(causal.root_evaluations, CANDIDATES)
            selected_index = CANDIDATES.index(decision.first_action.name) if decision.first_action.name in CANDIDATES else None
            offline_regrets = normalized_regrets(offline_costs[None, :], offline_feasible[None, :])[0]
            causal_regrets = normalized_regrets(causal_costs[None, :], causal_feasible[None, :])[0]
            offline_best = tolerant_best_mask(offline_costs[None, :], offline_feasible[None, :])[0]
            causal_best = tolerant_best_mask(causal_costs[None, :], causal_feasible[None, :])[0]
            row.update({
                "offline_teacher_action": offline.first_action.name,
                "causal_teacher_action": causal.first_action.name,
                "offline_causal_action_disagree": offline.first_action.name != causal.first_action.name,
                "offline_causal_disagree": not bool(np.any(offline_best & causal_best)),
                "offline_selected_regret": offline_regrets[selected_index] if selected_index is not None else np.nan,
                "causal_selected_regret": causal_regrets[selected_index] if selected_index is not None else np.nan,
                "offline_policy_error": tolerance_aware_policy_error(offline_costs, offline_feasible, selected_index),
                "causal_policy_error": tolerance_aware_policy_error(causal_costs, causal_feasible, selected_index),
                **{f"offline_cost_{name}": offline_costs[index] for index, name in enumerate(CANDIDATES)},
                **{f"causal_cost_{name}": causal_costs[index] for index, name in enumerate(CANDIDATES)},
                **{f"offline_feasible_{name}": offline_feasible[index] for index, name in enumerate(CANDIDATES)},
                **{f"causal_feasible_{name}": causal_feasible[index] for index, name in enumerate(CANDIDATES)},
            })
        self.diagnostics.append(row)
        return decision


def _reference_path(config: ExploreConfig, scope: str, trace: TraceRecord) -> Path:
    if scope == "diagnose" and not config.smoke:
        path = PAPER_ROOT / "generalization/references" / f"random_waypoint_seed-{trace.seed}.json"
    else:
        path = config.output / scope / "references" / f"{trace.trace_kind}_seed-{trace.seed}.json"
    load_or_compute_reference(
        trace.events,
        scenario=scenario_by_name("robot_arm"),
        trace_kind=trace.trace_id,
        seed=trace.seed,
        cache_path=path,
        include_approximation=True,
    )
    return path


def _benchmark_row(
    result: object,
    job: EvaluationJob,
    elapsed: float,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    if len(result.summary) == 1:
        row = result.summary.iloc[0].to_dict()
        status = "completed"
    elif result.failures:
        failure = result.failures[0]
        row = {
            "method": job.method,
            "seed": job.trace.seed,
            "budget": job.budget,
            "trace_kind": job.trace.trace_kind,
            "event_count": len(job.trace.events),
            "mean_approx_loss": np.nan,
            "fpr": np.nan,
            "fnr": np.nan,
            "failure_type": failure.failure_type,
            "failure_message": failure.message,
        }
        status = "fallback_failed" if failure.failure_type == "RtlolaNoFeasibleAction" else "native_failed"
    else:
        raise ValueError("benchmark produced neither one result nor a failure")
    row.update({
        "status": status,
        "scope": job.scope,
        "trace_id": job.trace.trace_id,
        "trace_sha256": job.trace.trace_sha256,
        "condition": job.trace.trace_kind,
        "seed": job.trace.seed,
        "budget": job.budget,
        "event_count": len(job.trace.events),
        "method": job.method,
        "feature_variant": job.feature_variant.value if job.feature_variant else "none",
        "challenger": job.challenger,
        "cell_elapsed_seconds": elapsed,
        "model_sha256": model_sha256(job.model_path) if job.model_path else None,
    })
    timeseries = result.timeseries.copy()
    failed = result.failed_timeseries.copy()
    return row, timeseries, failed


def _execute_evaluation(job: EvaluationJob) -> dict[str, object]:
    manifest_path = job.directory / "manifest.json"
    summary_path = job.directory / "summary.csv"
    identity = {
        "schema": CELL_SCHEMA,
        "experiment_fingerprint": job.config.fingerprint,
        "scope": job.scope,
        "trace_id": job.trace.trace_id,
        "trace_sha256": job.trace.trace_sha256,
        "event_count": len(job.trace.events),
        "budget": job.budget,
        "method": job.method,
        "feature_variant": job.feature_variant.value if job.feature_variant else None,
        "challenger": job.challenger,
        "model_sha256": model_sha256(job.model_path) if job.model_path else None,
        "reference_sha256": sha256_files((job.reference_path,)),
    }
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        if manifest.get("identity") != identity:
            raise ValueError(f"stale exploratory cell: {job.directory}")
        return pd.read_csv(summary_path).iloc[0].to_dict()
    job.directory.mkdir(parents=True, exist_ok=True)
    policy = None
    direct = None
    runtime_method = job.method
    if job.model_path is not None:
        policy = ReducerPolicy.load(job.model_path)
        assert job.feature_variant is not None
        direct = ExploratoryPolicy(
            policy,
            job.feature_variant,
            events=job.trace.events,
            challenger=job.challenger,
        )
        runtime_method = job.method
    reference = load_or_compute_reference(
        job.trace.events,
        scenario=scenario_by_name("robot_arm"),
        trace_kind=job.trace.trace_id,
        seed=job.trace.seed,
        cache_path=job.reference_path,
        include_approximation=True,
    )
    benchmark_config = RtlolaBenchmarkConfig(
        scenario="robot_arm",
        trace_kind=job.trace.trace_kind,
        length=len(job.trace.events),
        budget=job.budget,
        horizon=4 if job.method == REFERENCE_METHOD else 0,
        beam_width=4,
        prediction_step_seconds=0.1,
        seeds=1,
        methods=[runtime_method],
        reference_mode="exact",
        mpc_reference="rollout",
        output_dir=str(job.directory),
        mpc_candidate_names=list(CANDIDATES),
    )
    started = perf_counter()
    result = run_event_trace_benchmark(
        benchmark_config,
        job.trace.events,
        trace_kind=job.trace.trace_kind,
        seed=job.trace.seed,
        method=runtime_method,
        policy=direct,
        reference_steps=reference,
    )
    row, timeseries, failed = _benchmark_row(result, job, perf_counter() - started)
    write_csv_atomic(pd.DataFrame([row]), summary_path)
    if not timeseries.empty:
        write_csv_atomic(timeseries, job.directory / "timeseries.csv")
    if not failed.empty:
        write_csv_atomic(failed, job.directory / "failed_timeseries.csv")
    if direct is not None:
        write_csv_atomic(pd.DataFrame(direct.diagnostics), job.directory / "decisions.csv")
    write_json_atomic({"schema": CELL_SCHEMA, "identity": identity, "status": row["status"]}, manifest_path)
    return row


def _jobs_for_methods(
    config: ExploreConfig,
    *,
    scope: str,
    traces: Sequence[TraceRecord],
    methods: Sequence[tuple[str, Path | None, FeatureVariant | None, bool]],
) -> list[EvaluationJob]:
    jobs = []
    for trace in traces:
        reference = _reference_path(config, scope, trace)
        for budget in config.budgets:
            for method, model_root, variant, challenger in methods:
                model_path = model_root / f"budget-{budget}" if model_root is not None else None
                if method == V3_METHOD:
                    model_path = Path(str(_v3_model_records(config)[budget]["path"]))
                jobs.append(EvaluationJob(
                    config=config,
                    scope=scope,
                    trace=trace,
                    budget=budget,
                    method=method,
                    model_path=model_path,
                    feature_variant=variant,
                    challenger=challenger,
                    reference_path=reference,
                    directory=(
                        config.output / scope / "cells" / trace.trace_kind
                        / f"seed-{trace.seed}" / f"budget-{budget}" / method
                    ),
                ))
    return jobs


def _run_jobs(config: ExploreConfig, jobs: Sequence[EvaluationJob]) -> pd.DataFrame:
    if config.workers == 1:
        rows = [_execute_evaluation(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=config.workers,
            mp_context=get_context("spawn"),
            max_tasks_per_child=1,
        ) as executor:
            rows = list(executor.map(_execute_evaluation, jobs))
    return pd.DataFrame(rows)


def _execute_diagnostic_job(job: DiagnosticJob) -> dict[str, object]:
    summary_path = job.directory / "summary.csv"
    decisions_path = job.directory / "shadow_decisions.csv"
    timeseries_path = job.directory / "timeseries.csv"
    manifest_path = job.directory / "manifest.json"
    identity = {
        "schema": CELL_SCHEMA,
        "experiment_fingerprint": job.config.fingerprint,
        "label": job.label,
        "trace_id": job.trace.trace_id,
        "trace_sha256": job.trace.trace_sha256,
        "event_count": len(job.trace.events),
        "budget": job.budget,
        "model_sha256": model_sha256(job.model_path),
        "reference_sha256": sha256_files((job.reference_path,)),
    }
    if manifest_path.is_file():
        if _load_json(manifest_path).get("identity") != identity:
            raise ValueError(f"stale diagnostic cell: {job.directory}")
        return {
            "summary": str(summary_path),
            "decisions": str(decisions_path),
            "timeseries": str(timeseries_path),
        }
    job.directory.mkdir(parents=True, exist_ok=True)
    policy = ExploratoryPolicy(
        ReducerPolicy.load(job.model_path),
        FeatureVariant.G15,
        events=job.trace.events,
        shadow=True,
    )
    reference = load_or_compute_reference(
        job.trace.events,
        scenario=scenario_by_name("robot_arm"),
        trace_kind=job.trace.trace_id,
        seed=job.trace.seed,
        cache_path=job.reference_path,
        include_approximation=True,
    )
    benchmark_config = RtlolaBenchmarkConfig(
        scenario="robot_arm",
        trace_kind=job.trace.trace_kind,
        length=len(job.trace.events),
        budget=job.budget,
        horizon=0,
        beam_width=1,
        seeds=1,
        methods=[V3_METHOD],
        reference_mode="exact",
        output_dir=str(job.directory),
        mpc_candidate_names=list(CANDIDATES),
    )
    result = run_event_trace_benchmark(
        benchmark_config,
        job.trace.events,
        trace_kind=job.trace.trace_kind,
        seed=job.trace.seed,
        method=V3_METHOD,
        policy=policy,
        reference_steps=reference,
    )
    evaluation_job = EvaluationJob(
        job.config, "diagnose", job.trace, job.budget, V3_METHOD,
        job.model_path, FeatureVariant.G15, False, job.reference_path, job.directory,
    )
    row, trace_frame, _ = _benchmark_row(result, evaluation_job, float("nan"))
    row["diagnostic_group"] = job.label
    decision_frame = pd.DataFrame(policy.diagnostics)
    decision_frame.insert(0, "diagnostic_group", job.label)
    decision_frame.insert(1, "seed", job.trace.seed)
    trace_frame.insert(0, "diagnostic_group", job.label)
    write_csv_atomic(pd.DataFrame([row]), summary_path)
    write_csv_atomic(decision_frame, decisions_path)
    write_csv_atomic(trace_frame, timeseries_path)
    write_json_atomic({"schema": CELL_SCHEMA, "identity": identity, "status": row["status"]}, manifest_path)
    return {
        "summary": str(summary_path),
        "decisions": str(decisions_path),
        "timeseries": str(timeseries_path),
    }


def run_diagnose(config: ExploreConfig) -> Path:
    _load_stage(config, "train-clean")
    traces = _diagnostic_traces(config)
    records = _v3_model_records(config)
    diagnostic_groups = (
        (("bad", ((100, config.budgets[0]),)), ("matched_success", ((102, config.budgets[0]),)))
        if config.smoke
        else (("bad", BAD_DIAGNOSTIC_CELLS), ("matched_success", MATCHED_DIAGNOSTIC_CELLS))
    )
    jobs = []
    for label, cells in diagnostic_groups:
        for seed, budget in cells:
            trace = traces[seed]
            jobs.append(DiagnosticJob(
                config,
                label,
                trace,
                budget,
                Path(str(records[budget]["path"])),
                _reference_path(config, "diagnose", trace),
                config.output / "diagnose/cells" / label / f"seed-{seed}" / f"budget-{budget}",
            ))
    if config.workers == 1:
        artifacts = [_execute_diagnostic_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=config.workers,
            mp_context=get_context("spawn"),
            max_tasks_per_child=1,
        ) as executor:
            artifacts = list(executor.map(_execute_diagnostic_job, jobs))
    rows = [pd.read_csv(item["summary"]) for item in artifacts]
    decisions = [pd.read_csv(item["decisions"]) for item in artifacts]
    timeseries = [pd.read_csv(item["timeseries"]) for item in artifacts]
    summary = pd.concat(rows, ignore_index=True)
    decision_table = pd.concat(decisions, ignore_index=True)
    time_table = pd.concat(timeseries, ignore_index=True)
    summary_path = config.output / "diagnose/summary.csv"
    decisions_path = config.output / "diagnose/shadow_decisions.csv"
    timeseries_path = config.output / "diagnose/timeseries.csv"
    write_csv_atomic(summary, summary_path)
    write_csv_atomic(decision_table, decisions_path)
    write_csv_atomic(time_table, timeseries_path)
    return _write_stage(config, "diagnose", {
        "cell_count": len(summary),
        "summary": str(summary_path),
        "decisions": str(decisions_path),
        "timeseries": str(timeseries_path),
        "offline_causal_disagreement_count": int(decision_table.get("offline_causal_disagree", pd.Series(dtype=bool)).fillna(False).sum()),
    })


def _clean_methods(config: ExploreConfig) -> list[tuple[str, Path | None, FeatureVariant | None, bool]]:
    return [
        (V3_METHOD, None, FeatureVariant.G15, False),
        (FeatureVariant.G20.clean_method, config.output / "train-clean" / FeatureVariant.G20.clean_method, FeatureVariant.G20, False),
        (FeatureVariant.G25.clean_method, config.output / "train-clean" / FeatureVariant.G25.clean_method, FeatureVariant.G25, False),
        (REFERENCE_METHOD, None, None, False),
    ]


def _tail_selection_table(summary: pd.DataFrame, methods: Sequence[str]) -> pd.DataFrame:
    reference = summary[summary["method"] == REFERENCE_METHOD].set_index(["seed", "budget"])
    rows = []
    for method in methods:
        frame = summary[summary["method"] == method].set_index(["seed", "budget"])
        if set(frame.index) != set(reference.index):
            raise ValueError(f"selection cells do not align for {method}")
        frame = frame.loc[reference.index]
        completed = (frame["status"] == "completed") & (reference["status"] == "completed")
        ratios = frame.loc[completed, "mean_approx_loss"].to_numpy(float) / reference.loc[completed, "mean_approx_loss"].to_numpy(float)
        fpr_differences = frame.loc[completed, "fpr"].to_numpy(float) - reference.loc[completed, "fpr"].to_numpy(float)
        catastrophic = ratios > TAIL_MULTIPLIER
        rows.append({
            "method": method,
            "cell_count": len(frame),
            "failure_count": int((~completed).sum()),
            "catastrophic_count": int(catastrophic.sum()),
            "worst_loss_ratio": float(np.max(ratios)) if len(ratios) else np.inf,
            "p95_loss_ratio": float(np.quantile(ratios, 0.95)) if len(ratios) else np.inf,
            "median_loss_ratio": float(np.median(ratios)) if len(ratios) else np.inf,
            "mean_fpr_difference": float(np.mean(fpr_differences)) if len(fpr_differences) else np.inf,
        })
    table = pd.DataFrame(rows)
    dimensions = {
        V3_METHOD: 15,
        FeatureVariant.G20.clean_method: 20,
        FeatureVariant.G25.clean_method: 25,
        FeatureVariant.G15.dagger_method: 15,
        FeatureVariant.G20.dagger_method: 20,
        FeatureVariant.G25.dagger_method: 25,
    }
    table["feature_dimension"] = table["method"].map(dimensions).fillna(100).astype(int)
    return table.sort_values([
        "failure_count", "catastrophic_count", "worst_loss_ratio",
        "p95_loss_ratio", "mean_fpr_difference", "feature_dimension", "method",
    ]).reset_index(drop=True)


def run_feature_screen(config: ExploreConfig) -> Path:
    _load_stage(config, "train-clean")
    traces = _exploration_traces(config, config.selection_seeds)
    jobs = _jobs_for_methods(config, scope="feature-screen", traces=traces, methods=_clean_methods(config))
    started = perf_counter()
    summary = _run_jobs(config, jobs)
    if len(summary) != config.expected_feature_cells:
        raise ValueError(f"feature screen has {len(summary)} cells, expected {config.expected_feature_cells}")
    table = _tail_selection_table(summary, [item.clean_method for item in FeatureVariant])
    winner_method = str(table.iloc[0]["method"])
    winner = next(item for item in FeatureVariant if item.clean_method == winner_method)
    summary_path = config.output / "feature-screen/summary.csv"
    ranking_path = config.output / "feature-screen/feature_ranking.csv"
    write_csv_atomic(summary, summary_path)
    write_csv_atomic(table, ranking_path)
    return _write_stage(config, "feature-screen", {
        "cell_count": len(summary),
        "failure_count": int((summary["status"] != "completed").sum()),
        "matrix_wall_seconds": perf_counter() - started,
        "clean_winner": winner.value,
        "selection_rule": "failures_catastrophic_worst_p95_mean_fpr_dimension_name",
        "summary": str(summary_path),
        "ranking": str(ranking_path),
    })


def run_pilot(config: ExploreConfig) -> Path:
    _load_stage(config, "train-clean")
    pilot_budgets = tuple(value for value in PILOT_BUDGETS if value in config.budgets) or config.budgets[:1]
    pilot_config = replace(config, budgets=pilot_budgets, selection_seeds=config.selection_seeds[:1], workers=1)
    traces = _exploration_traces(config, pilot_config.selection_seeds)
    jobs = _jobs_for_methods(pilot_config, scope="pilot", traces=traces, methods=_clean_methods(config))
    g25_root = config.output / "train-clean" / FeatureVariant.G25.clean_method
    jobs.extend(_jobs_for_methods(
        pilot_config,
        scope="pilot",
        traces=traces,
        methods=[(f"{CHALLENGER_METHOD}__g25_clean148", g25_root, FeatureVariant.G25, True)],
    ))
    started = perf_counter()
    summary = _run_jobs(pilot_config, jobs)
    wall = perf_counter() - started
    collection_rows = []
    dagger_trace = _exploration_traces(config, config.dagger_seeds[:1])[0]
    clean_records = _clean_model_records(config)
    for variant in (FeatureVariant.G15, FeatureVariant.G25):
        for budget in pilot_budgets:
            method = variant.clean_method
            model_path = Path(str(clean_records[(method, budget)]["path"]))
            collection_started = perf_counter()
            dataset, _ = _collect_dagger_shard(
                config,
                dagger_trace,
                budget,
                variant,
                model_path,
                config.output / "pilot/collection" / variant.value / f"budget-{budget}",
            )
            collection_rows.append({
                "feature_variant": variant.value,
                "budget": budget,
                "sample_count": dataset.num_samples,
                "elapsed_seconds": perf_counter() - collection_started,
            })
    collection_summary = pd.DataFrame(collection_rows)
    by_method = summary.groupby("method", sort=True)["cell_elapsed_seconds"].mean().to_dict()
    projected_cells = config.expected_feature_cells + len(config.selection_seeds) * len(config.budgets) * 3
    projected_evaluation_cpu = sum(float(value) for value in by_method.values()) * projected_cells / max(len(by_method), 1)
    projected_collection_shards = 2 * len(config.dagger_seeds) * len(config.budgets)
    projected_collection_cpu = float(collection_summary["elapsed_seconds"].mean()) * projected_collection_shards
    projected_cpu = projected_evaluation_cpu + projected_collection_cpu
    projection = {
        "pilot_cell_count": len(summary),
        "pilot_wall_seconds": wall,
        "projected_selection_cells_upper_bound": projected_cells,
        "projected_cpu_seconds": projected_cpu,
        "projected_ten_worker_wall_seconds": projected_cpu / config.workers,
        "projected_evaluation_cpu_seconds": projected_evaluation_cpu,
        "projected_collection_shards_upper_bound": projected_collection_shards,
        "projected_collection_cpu_seconds": projected_collection_cpu,
        "projection_scope": "selection evaluation plus both autonomous DAgger collectors",
    }
    summary_path = config.output / "pilot/summary.csv"
    collection_path = config.output / "pilot/collection_summary.csv"
    write_csv_atomic(summary, summary_path)
    write_csv_atomic(collection_summary, collection_path)
    return _write_stage(config, "pilot", {
        **projection,
        "summary": str(summary_path),
        "collection_summary": str(collection_path),
    })


def _dagger_variants(config: ExploreConfig) -> tuple[FeatureVariant, ...]:
    winner = FeatureVariant(str(_load_stage(config, "feature-screen")["clean_winner"]))
    return (FeatureVariant.G15,) if winner is FeatureVariant.G15 else (FeatureVariant.G15, winner)


def _collect_dagger_shard(
    config: ExploreConfig,
    trace: TraceRecord,
    budget: int,
    variant: FeatureVariant,
    model_path: Path,
    directory: Path,
) -> tuple[ReducerCostDataset, pd.DataFrame]:
    manifest_path = directory / "manifest.json"
    identity = {
        "schema": SCHEMA,
        "experiment_fingerprint": config.fingerprint,
        "collection": "autonomous_dagger_round1",
        "trace_id": trace.trace_id,
        "trace_sha256": trace.trace_sha256,
        "seed": trace.seed,
        "budget": budget,
        "feature_variant": variant.value,
        "model_sha256": model_sha256(model_path),
    }
    if manifest_path.is_file():
        dataset, metadata, manifest = load_reducer_cost_dataset(directory)
        if manifest.get("identity") != identity:
            raise ValueError(f"stale DAgger shard: {directory}")
        return dataset, metadata
    policy = ReducerPolicy.load(model_path)
    catalog = default_action_catalog(CANDIDATES)
    engine = RtlolaEngine(
        scenario_by_name("robot_arm").spec,
        event_arity=scenario_by_name("robot_arm").event_arity,
        expected_verdict_keys=scenario_by_name("robot_arm").expected_verdict_keys,
    )
    history: list[RtlolaEvent] = []
    features_rows = []
    costs_rows = []
    feasible_rows = []
    sample_ids = []
    metadata_rows = []
    final_unlabelled = False
    for step, event in enumerate(trace.events):
        state = engine.snapshot(step=step, time=event.time)
        history.append(event)
        features = extract_policy_features(engine, state, event, history, budget, variant)
        decision, scores, ranking = _direct_decision(
            policy, catalog, engine, state, event, budget, features,
        )
        over_bound = engine.metrics(state).dynamic_generator_count > budget
        if over_bound and step + 1 < len(trace.events):
            before_dynamic, before_total = engine.matrices(state)
            teacher = full_width_terminal_search(
                engine,
                state,
                event,
                (trace.events[step + 1],),
                catalog.mpc_candidates,
                budget,
                fallback=catalog.fallback,
                none_action=catalog.no_op,
                configured_horizon=1,
            )
            after_dynamic, after_total = engine.matrices(state)
            if not np.array_equal(before_dynamic, after_dynamic) or not np.array_equal(before_total, after_total):
                raise AssertionError("DAgger shadow teacher mutated learner state")
            costs, feasible = _aligned_root_costs(teacher.root_evaluations, CANDIDATES)
            sample_id = f"{trace.trace_id}:{variant.value}:dagger1:budget-{budget}:step-{step}"
            features_rows.append(features)
            costs_rows.append(costs)
            feasible_rows.append(feasible)
            sample_ids.append(sample_id)
            selected = CANDIDATES.index(decision.first_action.name) if decision.first_action.name in CANDIDATES else None
            regrets = normalized_regrets(costs[None, :], feasible[None, :])[0]
            metadata_rows.append({
                "sample_id": sample_id,
                "trace_id": trace.trace_id,
                "trace_sha256": trace.trace_sha256,
                "split": "train",
                "condition": trace.trace_kind,
                "seed": trace.seed,
                "budget": budget,
                "step": step,
                "executed_action": decision.first_action.name,
                "policy_ranking": json.dumps(ranking),
                "teacher_action": teacher.first_action.name,
                "teacher_action_disagreement": decision.first_action.name != teacher.first_action.name,
                "teacher_error": tolerance_aware_policy_error(costs, feasible, selected),
                "selected_normalized_regret": regrets[selected] if selected is not None else np.nan,
                "teacher_evaluated_leaves": teacher.evaluated_leaves,
                "teacher_reducer_failure_count": teacher.reducer_failure_count,
                "teacher_infeasible_candidate_count": teacher.infeasible_candidate_count,
                "execution_fallback_used": decision.fallback_used,
                **{f"score_{name}": scores[index] for index, name in enumerate(CANDIDATES)},
            })
        elif over_bound:
            final_unlabelled = True
        engine.live_step(event, decision.first_action, budget, step=step + 1)
    if not features_rows:
        raise ValueError("DAgger shard contains no labelable over-bound states")
    dataset = ReducerCostDataset(
        features=np.asarray(features_rows, dtype=np.float32),
        teacher_costs=np.asarray(costs_rows, dtype=np.float64),
        feasible=np.asarray(feasible_rows, dtype=np.bool_),
        candidate_names=CANDIDATES,
        feature_names=schema_for(variant).feature_names,
        splits=tuple("train" for _ in sample_ids),
        sample_ids=tuple(sample_ids),
    )
    metadata = pd.DataFrame(metadata_rows)
    from pzr.learning.artifacts import write_reducer_cost_dataset

    write_reducer_cost_dataset(dataset, directory, metadata, {
        "identity": identity,
        "autonomous_execution": True,
        "teacher_changes_executed_action": False,
        "forced_recovery_count": 0,
        "final_over_bound_unlabelled": final_unlabelled,
    })
    return dataset, metadata


@dataclass(frozen=True)
class DaggerCollectionJob:
    config: ExploreConfig
    trace: TraceRecord
    budget: int
    variant: FeatureVariant
    model_path: Path
    directory: Path


def _execute_dagger_job(job: DaggerCollectionJob) -> dict[str, object]:
    started = perf_counter()
    dataset, metadata = _collect_dagger_shard(
        job.config, job.trace, job.budget, job.variant, job.model_path, job.directory,
    )
    return {
        "feature_variant": job.variant.value,
        "seed": job.trace.seed,
        "budget": job.budget,
        "sample_count": dataset.num_samples,
        "teacher_error_count": int(metadata["teacher_error"].sum()),
        "action_disagreement_count": int(metadata["teacher_action_disagreement"].sum()),
        "fallback_count": int(metadata["execution_fallback_used"].sum()),
        "elapsed_seconds": perf_counter() - started,
        "path": str(job.directory),
    }


def run_collect_dagger(config: ExploreConfig) -> Path:
    variants = _dagger_variants(config)
    clean = _clean_model_records(config)
    traces = _exploration_traces(config, config.dagger_seeds)
    jobs = []
    for variant in variants:
        for trace in traces:
            for budget in config.budgets:
                method = variant.clean_method
                model_path = Path(str(clean[(method, budget)]["path"]))
                jobs.append(DaggerCollectionJob(
                    config,
                    trace,
                    budget,
                    variant,
                    model_path,
                    config.output / "collect-dagger/shards" / variant.value / f"seed-{trace.seed}" / f"budget-{budget}",
                ))
    started = perf_counter()
    if config.workers == 1:
        rows = [_execute_dagger_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=config.workers,
            mp_context=get_context("spawn"),
            max_tasks_per_child=1,
        ) as executor:
            rows = list(executor.map(_execute_dagger_job, jobs))
    summary = pd.DataFrame(rows)
    expected = len(variants) * len(config.dagger_seeds) * len(config.budgets)
    if len(summary) != expected:
        raise ValueError(f"DAgger collection has {len(summary)} shards, expected {expected}")
    path = config.output / "collect-dagger/summary.csv"
    write_csv_atomic(summary, path)
    return _write_stage(config, "collect-dagger", {
        "variants": [item.value for item in variants],
        "shard_count": len(summary),
        "sample_count": int(summary["sample_count"].sum()),
        "matrix_wall_seconds": perf_counter() - started,
        "summary": str(path),
    })


def balanced_dagger_dataset(
    clean: ReducerCostDataset,
    clean_metadata: pd.DataFrame,
    dagger: ReducerCostDataset,
    dagger_metadata: pd.DataFrame,
    *,
    seed: int = 42,
) -> tuple[ReducerCostDataset, pd.DataFrame]:
    """Combine all clean rows with deterministic 50:50 learner-row oversampling."""
    train_indices = clean.indices_for_split("train")
    validation_indices = clean.indices_for_split("validation")
    if dagger.num_samples == 0 or len(train_indices) == 0 or len(validation_indices) == 0:
        raise ValueError("balanced DAgger training needs clean train/validation and learner rows")
    rng = np.random.default_rng(seed)
    learner_indices = rng.choice(dagger.num_samples, size=len(train_indices), replace=True)
    learner_sample_ids = tuple(
        f"{dagger.sample_ids[index]}:resample-{offset}"
        for offset, index in enumerate(learner_indices)
    )
    learner_dataset = ReducerCostDataset(
        features=dagger.features[learner_indices],
        teacher_costs=dagger.teacher_costs[learner_indices],
        feasible=dagger.feasible[learner_indices],
        candidate_names=dagger.candidate_names,
        feature_names=dagger.feature_names,
        splits=tuple("train" for _ in learner_indices),
        sample_ids=learner_sample_ids,
    )
    datasets = (
        clean.subset(train_indices),
        learner_dataset,
        clean.subset(validation_indices),
    )
    result = ReducerCostDataset.concatenate(datasets)
    clean_train = clean_metadata.iloc[train_indices].copy()
    clean_train["training_source"] = "clean148"
    learner = dagger_metadata.iloc[learner_indices].copy().reset_index(drop=True)
    learner["sample_id"] = learner_sample_ids
    learner["split"] = "train"
    learner["training_source"] = "learner_visited_dagger1"
    clean_validation = clean_metadata.iloc[validation_indices].copy()
    clean_validation["training_source"] = "clean_validation"
    metadata = pd.concat((clean_train, learner, clean_validation), ignore_index=True)
    if tuple(metadata["sample_id"].astype(str)) != result.sample_ids:
        raise AssertionError("balanced DAgger dataset and metadata are not aligned")
    counts = metadata.loc[metadata["split"] == "train", "training_source"].value_counts()
    if counts.get("clean148", 0) != counts.get("learner_visited_dagger1", -1):
        raise AssertionError("DAgger training sources are not balanced 50:50")
    return result, metadata


def _load_dagger_variant(config: ExploreConfig, variant: FeatureVariant) -> tuple[ReducerCostDataset, pd.DataFrame]:
    datasets = []
    frames = []
    for seed in config.dagger_seeds:
        for budget in config.budgets:
            path = config.output / "collect-dagger/shards" / variant.value / f"seed-{seed}" / f"budget-{budget}"
            dataset, metadata, _ = load_reducer_cost_dataset(path)
            datasets.append(dataset)
            frames.append(metadata)
    return ReducerCostDataset.concatenate(datasets), pd.concat(frames, ignore_index=True)


def run_train_dagger(config: ExploreConfig) -> Path:
    _load_stage(config, "collect-dagger")
    bundle = load_clean_bundle(config)
    records = {}
    for variant in _dagger_variants(config):
        dagger, dagger_metadata = _load_dagger_variant(config, variant)
        for budget in config.budgets:
            clean_subset, clean_frame = _subset_for_budget(bundle.dataset(variant), bundle.metadata, budget)
            selected = dagger_metadata["budget"].astype(int).to_numpy() == budget
            dagger_subset = dagger.subset(np.flatnonzero(selected))
            dagger_frame = dagger_metadata.loc[selected].reset_index(drop=True)
            aggregate, aggregate_frame = balanced_dagger_dataset(
                clean_subset, clean_frame, dagger_subset, dagger_frame, seed=42,
            )
            aggregate_frame = aggregate_frame.copy()
            aggregate_frame["dataset_label"] = aggregate_frame["training_source"]
            output = config.output / "train-dagger" / variant.dagger_method / f"budget-{budget}"
            identity = {
                "experiment_fingerprint": config.fingerprint,
                "method": variant.dagger_method,
                "budget": budget,
                "feature_schema": schema_payload(variant),
                "balanced_train_rows_per_source": int(len(clean_subset.indices_for_split("train"))),
                "training": {**TRAINING, "epochs": config.epochs},
            }
            artifact = output / "exploratory_training.json"
            if artifact.is_file():
                existing = _load_json(artifact)
                if existing.get("identity") != identity or model_sha256(output) != existing.get("model_sha256"):
                    raise ValueError(f"stale DAgger model: {output}")
                records[f"{variant.dagger_method}:{budget}"] = existing["record"]
                continue
            policy, result = train_reducer_policy(
                aggregate,
                schema_for(variant),
                objective="pairwise",
                epochs=config.epochs,
                batch_size=int(TRAINING["batch_size"]),
                learning_rate=float(TRAINING["learning_rate"]),
                weight_decay=float(TRAINING["weight_decay"]),
                patience=min(int(TRAINING["patience"]), config.epochs),
                seed=42,
            )
            policy.save(output)
            write_csv_atomic(validation_metrics(policy, aggregate, aggregate_frame), output / "validation_metrics.csv")
            record = {
                **_model_record(output, variant.dagger_method, budget, variant),
                "best_epoch": result.best_epoch,
                "epochs_completed": result.epochs,
                "training_source": "clean148_plus_autonomous_dagger1_balanced_50_50",
            }
            write_json_atomic({
                "identity": identity,
                "model_sha256": record["sha256"],
                "record": record,
                "validation_metrics": asdict(result.val_metrics),
            }, artifact)
            records[f"{variant.dagger_method}:{budget}"] = record
    return _write_stage(config, "train-dagger", {"models": records, "model_count": len(records)})


def _dagger_model_records(config: ExploreConfig) -> dict[tuple[str, int], dict[str, object]]:
    records = {}
    for key, raw in dict(_load_stage(config, "train-dagger")["models"]).items():
        method, raw_budget = key.rsplit(":", 1)
        record = dict(raw)
        if model_sha256(Path(str(record["path"]))) != record["sha256"]:
            raise ValueError(f"DAgger model hash differs: {key}")
        records[(method, int(raw_budget))] = record
    return records


def _method_root(config: ExploreConfig, method: str) -> tuple[Path, FeatureVariant]:
    for variant in FeatureVariant:
        if method == variant.clean_method:
            if variant is FeatureVariant.G15:
                raise ValueError("v3 model root is budget-specific")
            return config.output / "train-clean" / method, variant
        if method == variant.dagger_method:
            return config.output / "train-dagger" / method, variant
    raise ValueError(f"unknown exploratory method: {method}")


def run_select(config: ExploreConfig) -> Path:
    feature = _load_stage(config, "feature-screen")
    _load_stage(config, "train-dagger")
    feature_summary = pd.read_csv(str(feature["summary"]))
    traces = _exploration_traces(config, config.selection_seeds)
    dagger_methods = [variant.dagger_method for variant in _dagger_variants(config)]
    dagger_specs = []
    for method in dagger_methods:
        root, variant = _method_root(config, method)
        dagger_specs.append((method, root, variant, False))
    dagger_jobs = _jobs_for_methods(config, scope="select", traces=traces, methods=dagger_specs)
    dagger_summary = _run_jobs(config, dagger_jobs)
    unguarded_summary = pd.concat((feature_summary, dagger_summary), ignore_index=True)
    unguarded_methods = [item.clean_method for item in FeatureVariant] + dagger_methods
    ranking = _tail_selection_table(unguarded_summary, unguarded_methods)
    winner = str(ranking.iloc[0]["method"])
    if winner == V3_METHOD:
        winner_variant = FeatureVariant.G15
        winner_root = None
    else:
        winner_root, winner_variant = _method_root(config, winner)
    challenger_name = f"{CHALLENGER_METHOD}__{winner}"
    challenger_jobs = _jobs_for_methods(
        config,
        scope="select",
        traces=traces,
        methods=[(challenger_name, winner_root, winner_variant, True)],
    )
    if winner == V3_METHOD:
        v3_records = _v3_model_records(config)
        challenger_jobs = [replace(job, model_path=Path(str(v3_records[job.budget]["path"]))) for job in challenger_jobs]
    challenger_summary = _run_jobs(config, challenger_jobs)
    combined = pd.concat((unguarded_summary, challenger_summary), ignore_index=True)
    expected = config.expected_selection_cells(FeatureVariant(str(feature["clean_winner"])))
    if len(combined) != expected:
        raise ValueError(f"selection has {len(combined)} cells, expected {expected}")
    summary_path = config.output / "select/summary.csv"
    ranking_path = config.output / "select/unguarded_ranking.csv"
    write_csv_atomic(combined, summary_path)
    write_csv_atomic(ranking, ranking_path)
    return _write_stage(config, "select", {
        "cell_count": len(combined),
        "failure_count": int((combined["status"] != "completed").sum()),
        "best_unguarded_method": winner,
        "best_unguarded_feature_variant": winner_variant.value,
        "challenger_method": challenger_name,
        "summary": str(summary_path),
        "unguarded_ranking": str(ranking_path),
    })


def _paired_bootstrap(values: NDArray[np.float64]) -> tuple[float, float, float]:
    if not len(values):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    estimates = np.mean(values[draws], axis=1)
    return float(np.mean(values)), float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def selection_metrics(
    summary: pd.DataFrame,
    *,
    bootstrap: bool = True,
) -> pd.DataFrame:
    """Return seed-paired diagnostics against both scientific references."""
    index_columns = ["seed", "budget"]
    if "trace_id" in summary.columns:
        index_columns.append("trace_id")
    rows = []
    for reference_method in (V3_METHOD, REFERENCE_METHOD):
        reference = summary[summary["method"] == reference_method].set_index(index_columns)
        if reference.empty:
            raise ValueError(f"missing paired reference method: {reference_method}")
        candidates = summary[summary["method"] != reference_method]
        for (budget, method), frame in candidates.groupby(["budget", "method"], sort=True):
            frame = frame.set_index(index_columns)
            if not set(frame.index).issubset(set(reference.index)):
                raise ValueError(f"cells do not align with {reference_method}: {method}")
            ref = reference.loc[frame.index]
            valid = (frame["status"] == "completed") & (ref["status"] == "completed")
            loss_ratio = (
                frame.loc[valid, "mean_approx_loss"].to_numpy(float)
                / ref.loc[valid, "mean_approx_loss"].to_numpy(float)
            )
            valid_seeds = frame.loc[valid].index.get_level_values("seed").to_numpy(int)
            worst_index = int(np.argmax(loss_ratio)) if len(loss_ratio) else None
            worst_seed = int(valid_seeds[worst_index]) if worst_index is not None else None
            if worst_index is not None and "trace_id" in index_columns:
                worst_trace_id = str(
                    frame.loc[valid].index.get_level_values("trace_id")[worst_index]
                )
            else:
                worst_trace_id = None
            fpr_difference = (
                frame.loc[valid, "fpr"].to_numpy(float)
                - ref.loc[valid, "fpr"].to_numpy(float)
            )
            if bootstrap:
                fpr_mean, fpr_low, fpr_high = _paired_bootstrap(fpr_difference)
            else:
                fpr_mean = float(np.mean(fpr_difference)) if len(fpr_difference) else np.nan
                fpr_low = np.nan
                fpr_high = np.nan
            rows.append({
                "budget": budget,
                "method": method,
                "reference_method": reference_method,
                "trace_count": len(frame),
                "valid_count": int(valid.sum()),
                "failure_count": int((~valid).sum()),
                "median_loss_ratio": float(np.median(loss_ratio)) if len(loss_ratio) else np.nan,
                "q25_loss_ratio": float(np.quantile(loss_ratio, 0.25)) if len(loss_ratio) else np.nan,
                "q75_loss_ratio": float(np.quantile(loss_ratio, 0.75)) if len(loss_ratio) else np.nan,
                "p95_loss_ratio": float(np.quantile(loss_ratio, 0.95)) if len(loss_ratio) else np.nan,
                "worst_loss_ratio": float(np.max(loss_ratio)) if len(loss_ratio) else np.nan,
                "worst_seed": worst_seed,
                "worst_trace_id": worst_trace_id,
                "catastrophic_count": int((loss_ratio > TAIL_MULTIPLIER).sum()),
                "mean_fpr_difference": fpr_mean,
                "mean_fpr_difference_ci_low": fpr_low,
                "mean_fpr_difference_ci_high": fpr_high,
                "median_events_per_second": float(np.median(
                    frame.loc[valid, "event_count"].to_numpy(float)
                    / (frame.loc[valid, "event_loop_time_ms"].to_numpy(float) / 1000.0)
                )) if bool(valid.any()) else np.nan,
                "mean_evaluated_leaves": float(frame.loc[valid, "mean_evaluated_leaves"].mean()) if bool(valid.any()) else np.nan,
                "max_evaluated_leaves": float(frame.loc[valid, "max_evaluated_leaves"].max()) if bool(valid.any()) else np.nan,
            })
    return pd.DataFrame(rows)


def _display_method(method: str) -> str:
    labels = {
        V3_METHOD: "G15 clean",
        FeatureVariant.G20.clean_method: "G20 clean",
        FeatureVariant.G25.clean_method: "G25 clean",
        FeatureVariant.G15.dagger_method: "G15 DAgger",
        FeatureVariant.G20.dagger_method: "G20 DAgger",
        FeatureVariant.G25.dagger_method: "G25 DAgger",
        REFERENCE_METHOD: "Predictive MPC",
    }
    if method.startswith(f"{CHALLENGER_METHOD}__"):
        return "H=2 causal challenger"
    return labels.get(method, method)


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


def _plot_selection_ecdf(summary: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    """Show the complete trace-level paired loss-ratio distribution."""
    plt = _pyplot()
    reference = summary[summary["method"] == REFERENCE_METHOD].set_index(["seed", "budget"])
    methods = tuple(method for method in summary["method"].unique() if method != REFERENCE_METHOD)
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 3.8), squeeze=False, sharey=True)
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00")
    for axis, budget in zip(axes.flat, sorted(summary["budget"].astype(int).unique())):
        for index, method in enumerate(methods):
            frame = summary[(summary["budget"] == budget) & (summary["method"] == method)].set_index(["seed", "budget"])
            ref = reference.loc[frame.index]
            valid = (frame["status"] == "completed") & (ref["status"] == "completed")
            ratios = np.sort(frame.loc[valid, "mean_approx_loss"].to_numpy(float) / ref.loc[valid, "mean_approx_loss"].to_numpy(float))
            if len(ratios) and np.all(ratios > 0.0):
                axis.step(
                    ratios,
                    np.arange(1, len(ratios) + 1) / len(ratios),
                    where="post",
                    color=colors[index % len(colors)],
                    linestyle=("-", "--", "-.", ":")[index % 4],
                    marker=("o", "s", "^", "D")[index % 4],
                    markersize=2.3,
                    label=_display_method(method),
                )
        axis.axvline(1.0, color="0.35", linewidth=0.7)
        axis.axvline(TAIL_MULTIPLIER, color="#D55E00", linewidth=0.7, linestyle=":")
        axis.set_xscale("log")
        axis.set_title(f"B={budget}")
        axis.set_xlabel("Mean-loss ratio vs. predictive MPC")
        axis.set_ylim(0.0, 1.02)
        axis.grid(axis="both", color="0.9", linewidth=0.5)
    for axis in axes[:, 0]:
        axis.set_ylabel("Trace ECDF")
    for axis in axes.flat[len(summary["budget"].unique()):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    pdf = output.with_suffix(".pdf")
    png = output.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png, dpi=250, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return pdf, png


def _plot_fpr_differences(metrics: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    """Show seed-paired macro-FPR differences and their bootstrap intervals."""
    plt = _pyplot()
    metrics = metrics[metrics["reference_method"] == REFERENCE_METHOD].copy()
    budgets = tuple(sorted(metrics["budget"].astype(int).unique()))
    methods = tuple(metrics["method"].drop_duplicates())
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 4.2), squeeze=False, sharex=True)
    for panel_index, (axis, budget) in enumerate(zip(axes.flat, budgets)):
        frame = metrics[metrics["budget"].astype(int) == budget].set_index("method").reindex(methods)
        positions = np.arange(len(methods))
        values = frame["mean_fpr_difference"].to_numpy(float) * 100.0
        low = frame["mean_fpr_difference_ci_low"].to_numpy(float) * 100.0
        high = frame["mean_fpr_difference_ci_high"].to_numpy(float) * 100.0
        axis.errorbar(
            values,
            positions,
            xerr=np.vstack((values - low, high - values)),
            fmt="o",
            color="#0072B2",
            ecolor="#56B4E9",
            markersize=3.5,
            linewidth=0.9,
            capsize=2,
        )
        axis.axvline(0.0, color="0.35", linewidth=0.7)
        axis.set_title(f"B={budget}")
        axis.set_xlabel("\u0394FPR (pp)")
        axis.set_yticks(
            positions,
            tuple(_display_method(method) for method in methods) if panel_index % 4 == 0 else (),
        )
        axis.grid(axis="x", color="0.9", linewidth=0.5)
    for axis in axes.flat[len(budgets):]:
        axis.set_visible(False)
    fig.text(0.5, 0.01, "Paired difference from predictive MPC", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    pdf = output.with_suffix(".pdf")
    png = output.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png, dpi=250, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return pdf, png


def _decision_tables(config: ExploreConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    for scope in ("feature-screen", "select"):
        for path in sorted((config.output / scope / "cells").rglob("decisions.csv")):
            frame = pd.read_csv(path)
            parts = path.parts
            budget = int(next(part.split("-", 1)[1] for part in parts if part.startswith("budget-")))
            seed = int(next(part.split("-", 1)[1] for part in parts if part.startswith("seed-")))
            frame.insert(0, "method", path.parent.name)
            frame.insert(1, "seed", seed)
            frame["budget"] = budget
            frames.append(frame)
    if not frames:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    decisions = pd.concat(frames, ignore_index=True)
    over_bound = decisions[decisions["over_bound"].astype(bool)].copy()
    ordinary = over_bound[over_bound["selected_action"].isin(CANDIDATES)]
    composition = ordinary.groupby(
        ["method", "budget", "selected_action"], sort=True,
    ).size().rename("count").reset_index()
    totals = composition.groupby(["method", "budget"])["count"].transform("sum")
    composition["percentage"] = 100.0 * composition["count"] / totals
    diagnostic = over_bound[~over_bound["selected_action"].isin(CANDIDATES)].groupby(
        ["method", "budget", "selected_action"], sort=True,
    ).size().rename("count").reset_index()
    transitions = []
    for (method, seed, budget), frame in over_bound.groupby(["method", "seed", "budget"], sort=True):
        actions = frame.sort_values("step")["selected_action"].astype(str).tolist()
        for prior, current in zip(actions[:-1], actions[1:]):
            transitions.append({
                "method": method,
                "seed": seed,
                "budget": budget,
                "from_action": prior,
                "to_action": current,
            })
    transition_table = (
        pd.DataFrame(transitions).groupby(
            ["method", "budget", "from_action", "to_action"], sort=True,
        ).size().rename("count").reset_index()
        if transitions else pd.DataFrame(columns=("method", "budget", "from_action", "to_action", "count"))
    )
    return composition, diagnostic, transition_table


def margin_guard_diagnostics(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    data = decisions[
        decisions["offline_selected_regret"].notna()
        & decisions["top_two_margin"].notna()
    ].copy()
    for (group, budget), frame in data.groupby(["diagnostic_group", "budget"], sort=True):
        margins = frame["top_two_margin"].to_numpy(float)
        regrets = frame["offline_selected_regret"].to_numpy(float)
        errors = frame["offline_policy_error"].astype(bool).to_numpy()
        spearman = float(pd.Series(margins).rank().corr(pd.Series(regrets).rank()))
        for fraction in (0.10, 0.15):
            deferred_count = max(1, int(np.ceil(fraction * len(frame))))
            deferred = np.argsort(margins, kind="stable")[:deferred_count]
            retained = np.ones(len(frame), dtype=bool)
            retained[deferred] = False
            total_error = int(errors.sum())
            total_regret = float(regrets.sum())
            rows.append({
                "diagnostic_group": group,
                "budget": budget,
                "decision_count": len(frame),
                "deferral_fraction": fraction,
                "deferred_count": deferred_count,
                "initial_error_rate": float(errors.mean()),
                "retained_error_rate": float(errors[retained].mean()) if retained.any() else np.nan,
                "error_recall": float(errors[deferred].sum() / total_error) if total_error else np.nan,
                "regret_caught_fraction": float(regrets[deferred].sum() / total_regret) if total_regret > 0.0 else np.nan,
                "margin_regret_spearman": spearman,
            })
    return pd.DataFrame(rows)


def run_report(config: ExploreConfig) -> Path:
    selection = _load_stage(config, "select")
    summary = pd.read_csv(str(selection["summary"]))
    metrics = selection_metrics(summary)
    artifact_dir = config.output / "report/artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifact_dir / "selection_tail_metrics.csv"
    trace_path = artifact_dir / "selection_trace_results.csv"
    write_csv_atomic(metrics, metrics_path)
    write_csv_atomic(summary, trace_path)
    ecdf = _plot_selection_ecdf(summary, artifact_dir / "selection_loss_ratio_ecdf")
    fpr_figure = _plot_fpr_differences(metrics, artifact_dir / "selection_fpr_differences")
    composition, action_diagnostics, transitions = _decision_tables(config)
    composition_path = artifact_dir / "selection_reducer_composition.csv"
    action_diagnostics_path = artifact_dir / "selection_nonordinary_actions.csv"
    transitions_path = artifact_dir / "selection_action_transitions.csv"
    write_csv_atomic(composition, composition_path)
    write_csv_atomic(action_diagnostics, action_diagnostics_path)
    write_csv_atomic(transitions, transitions_path)
    artifacts = [
        metrics_path,
        trace_path,
        composition_path,
        action_diagnostics_path,
        transitions_path,
        *ecdf,
        *fpr_figure,
    ]
    if _stage_path(config, "diagnose").is_file():
        diagnosis = _load_stage(config, "diagnose")
        decisions = pd.read_csv(str(diagnosis["decisions"]))
        guard = decisions[decisions["offline_selected_regret"].notna()].copy()
        if not guard.empty:
            guard["margin_rank"] = guard["top_two_margin"].rank(method="average", pct=True)
            guard_path = artifact_dir / "diagnostic_margin_regret.csv"
            guard_summary_path = artifact_dir / "diagnostic_margin_deferral.csv"
            write_csv_atomic(guard, guard_path)
            write_csv_atomic(margin_guard_diagnostics(guard), guard_summary_path)
            artifacts.extend((guard_path, guard_summary_path))
    hash_manifest = {
        str(path.relative_to(artifact_dir)): _raw_sha256(path) for path in artifacts
    }
    hash_path = artifact_dir / "artifact_hashes.json"
    write_json_atomic(hash_manifest, hash_path)
    if _v3_snapshot() != _load_stage(config, "preflight")["v3_snapshot"]:
        raise ValueError("canonical v3 artifacts changed during PRP v4 exploration")
    return _write_stage(config, "report", {
        "selection_method": selection["best_unguarded_method"],
        "challenger_method": selection["challenger_method"],
        "artifact_directory": str(artifact_dir),
        "artifact_count": len(artifacts) + 1,
        "artifact_hashes": str(hash_path),
        "v3_unchanged": True,
    })


def _confirmation_methods(
    config: ExploreConfig,
    variant_name: str,
) -> tuple[list[tuple[str, Path | None, FeatureVariant | None, bool]], FeatureVariant]:
    selection = _load_stage(config, "select")
    allowed = set(pd.read_csv(str(selection["unguarded_ranking"]))["method"].astype(str))
    if variant_name not in allowed:
        raise ValueError(f"confirmation variant was not an unguarded selection candidate: {variant_name}")
    if variant_name == V3_METHOD:
        selected_spec = (V3_METHOD, None, FeatureVariant.G15, False)
        selected_variant = FeatureVariant.G15
    else:
        root, selected_variant = _method_root(config, variant_name)
        selected_spec = (variant_name, root, selected_variant, False)
    specs = [(V3_METHOD, None, FeatureVariant.G15, False)]
    if variant_name != V3_METHOD:
        specs.append(selected_spec)
    root = selected_spec[1]
    specs.extend([
        (f"{CHALLENGER_METHOD}__{variant_name}", root, selected_variant, True),
        (REFERENCE_METHOD, None, None, False),
    ])
    return specs, selected_variant


def _freeze_confirmation(config: ExploreConfig, variant_name: str) -> dict[str, object]:
    path = config.output / "confirm/freeze.json"
    selection = _load_stage(config, "select")
    payload = {
        "schema": FREEZE_SCHEMA,
        "experiment_fingerprint": config.fingerprint,
        "variant": variant_name,
        "selection_manifest_sha256": sha256_files((_stage_path(config, "select"),)),
        "selection_summary_sha256": sha256_files((Path(str(selection["summary"])),)),
    }
    if path.is_file():
        if _load_json(path) != payload:
            raise ValueError("confirmation variant is already frozen differently")
    else:
        write_json_atomic(payload, path)
    return payload


def _fixed_traces(config: ExploreConfig) -> tuple[TraceRecord, ...]:
    from pzr.rtlola.robot_arm import generate_robot_arm_events, load_robot_arm_trace

    traces = []
    for name in FIXED_TRACE_KINDS:
        rows = load_robot_arm_trace(name)
        if len(rows) != ROBOT_ARM_TRACE_ROWS[name]:
            raise ValueError(f"fixed trace length differs: {name}")
        events = generate_robot_arm_events(0, trace_kind=name)
        if config.smoke:
            events = events[: config.event_count]
        traces.append(TraceRecord(
            trace_id=name,
            trace_kind=name,
            seed=0,
            events=tuple(events),
            trace_sha256=ROBOT_ARM_TRACE_SHA256[name],
            provenance={"spec_sha256": ROBOT_ARM_SPEC_SHA256, "fixed": True},
        ))
    return tuple(traces)


def run_confirm(config: ExploreConfig, variant_name: str) -> Path:
    freeze = _freeze_confirmation(config, variant_name)
    specs, selected_variant = _confirmation_methods(config, variant_name)
    generated = generate_random_waypoint_trace_store(
        RandomWaypointTraceStoreConfig(
            output=config.output / "confirm/nominal-traces",
            event_count=config.event_count,
            conditions=("random_waypoint",),
            seed_start=min(config.confirmation_seeds),
            seed_count=len(config.confirmation_seeds),
        )
    )
    nominal = tuple(_trace_record(generated.traces_for_seed(seed)[0]) for seed in config.confirmation_seeds)
    nominal_jobs = _jobs_for_methods(config, scope="confirm/nominal", traces=nominal, methods=specs)
    fixed_jobs = _jobs_for_methods(config, scope="confirm/fixed", traces=_fixed_traces(config), methods=specs)
    if variant_name == V3_METHOD:
        records = _v3_model_records(config)
        nominal_jobs = [replace(job, model_path=Path(str(records[job.budget]["path"]))) if job.challenger else job for job in nominal_jobs]
        fixed_jobs = [replace(job, model_path=Path(str(records[job.budget]["path"]))) if job.challenger else job for job in fixed_jobs]
    nominal_summary = _run_jobs(config, nominal_jobs)
    fixed_summary = _run_jobs(config, fixed_jobs)
    expected_nominal = len(config.confirmation_seeds) * len(config.budgets) * len(specs)
    expected_fixed = len(FIXED_TRACE_KINDS) * len(config.budgets) * len(specs)
    if len(nominal_summary) != expected_nominal or len(fixed_summary) != expected_fixed:
        raise ValueError("confirmation cell completeness differs")
    nominal_path = config.output / "confirm/nominal_summary.csv"
    fixed_path = config.output / "confirm/fixed_summary.csv"
    nominal_metrics_path = config.output / "confirm/nominal_paired_metrics.csv"
    fixed_metrics_path = config.output / "confirm/fixed_paired_metrics.csv"
    write_csv_atomic(nominal_summary, nominal_path)
    write_csv_atomic(fixed_summary, fixed_path)
    write_csv_atomic(selection_metrics(nominal_summary), nominal_metrics_path)
    write_csv_atomic(selection_metrics(fixed_summary, bootstrap=False), fixed_metrics_path)
    return _write_stage(config, "confirm", {
        "freeze": freeze,
        "selected_feature_variant": selected_variant.value,
        "method_count": len(specs),
        "nominal_cell_count": len(nominal_summary),
        "fixed_cell_count": len(fixed_summary),
        "nominal_summary": str(nominal_path),
        "fixed_summary": str(fixed_path),
        "nominal_paired_metrics": str(nominal_metrics_path),
        "fixed_paired_metrics": str(fixed_metrics_path),
        "summary_sha256": sha256_files((nominal_path, fixed_path)),
        "paired_metrics_sha256": sha256_files((nominal_metrics_path, fixed_metrics_path)),
        "fixed_cases_are_descriptive": True,
    })


STAGES = (
    "preflight",
    "prepare",
    "train-clean",
    "diagnose",
    "pilot",
    "feature-screen",
    "collect-dagger",
    "train-dagger",
    "select",
    "report",
)


def run_stage(config: ExploreConfig, stage: str) -> Path:
    path = _stage_path(config, stage)
    if path.is_file():
        _load_stage(config, stage)
        print(f"skip completed exploratory stage: {stage}", flush=True)
        return path
    functions = {
        "preflight": run_preflight,
        "prepare": run_prepare,
        "train-clean": run_train_clean,
        "diagnose": run_diagnose,
        "pilot": run_pilot,
        "feature-screen": run_feature_screen,
        "collect-dagger": run_collect_dagger,
        "train-dagger": run_train_dagger,
        "select": run_select,
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
    return _stage_path(config, "report")


def status(config: ExploreConfig) -> dict[str, object]:
    stages = {}
    for stage in (*STAGES, "confirm"):
        path = _stage_path(config, stage)
        if not path.is_file():
            stages[stage] = "missing"
            continue
        try:
            _load_stage(config, stage)
            stages[stage] = "completed"
        except ValueError as exc:
            stages[stage] = f"stale: {exc}"
    return {
        "output": str(config.output),
        "smoke": config.smoke,
        "feature_screen_cells": config.expected_feature_cells,
        "stages": stages,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(*STAGES, "confirm", "run", "status"))
    parser.add_argument("--variant")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    config = smoke_config(workers=args.workers) if args.smoke else ExploreConfig(workers=args.workers)
    if args.command == "status":
        print(json.dumps(status(config), indent=2))
        return 0
    if args.command == "confirm":
        if not args.variant:
            parser.error("confirm requires --variant NAME")
        path = run_confirm(config, args.variant)
    elif args.command == "run":
        path = run_all(config)
    else:
        path = run_stage(config, args.command)
    print(f"PRP v4 exploratory stage complete: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
