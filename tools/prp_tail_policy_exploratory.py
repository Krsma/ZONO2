#!/usr/bin/env python3
"""Disposable policy-time PRP tail-robustness experiment.

This driver only writes below ``results/prp-tail-policy-exploratory-v1``.
It reuses hash-verified Clean148, autonomous G15 DAgger, selection-trace,
reference-cell, and optimizer-replica artifacts; it never collects new states
or enters the reserved confirmation/fixed-trace evaluation.
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
from numpy.typing import NDArray
import pandas as pd
import yaml

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
from pzr.learning.ranker import ReducerPolicy, train_reducer_policy
from pzr.learning.training import dataset_sha256
from pzr.rtlola.actions import RtlolaActionCatalog, default_action_catalog
from pzr.rtlola.benchmark import RtlolaBenchmarkConfig, run_event_trace_benchmark
from pzr.rtlola.binding import (
    BINDING_BUILD_PROFILE,
    BINDING_REVISION,
    INTERPRETER_REVISION,
)
from pzr.rtlola.engine import RtlolaBindingError, RtlolaEngine, RtlolaEvent, RtlolaStateRef
from pzr.rtlola.features import RTL_RANKING_FEATURE_SCHEMA, extract_ranking_features
from pzr.rtlola.learning_traces import load_random_waypoint_trace_store
from pzr.rtlola.reference import load_or_compute_reference
from pzr.rtlola.scenarios import scenario_by_name
from pzr.rtlola.search import RtlolaNoFeasibleAction, RtlolaSearchResult

from prp_v4_exploratory import (
    CLEAN_TRAIN_SEEDS,
    PARENT_DATASETS,
    VALIDATION_SEEDS,
    TraceRecord,
    _trace_record,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/prp-tail-policy-exploratory-v1"
V3_ROOT = ROOT / "results/paper-evaluation-v3"
V3_CONFIG = ROOT / "experiments/paper_evaluation_v3.yaml"
V4_ROOT = ROOT / "results/prp-v4-exploratory-v1"
SCALE_ROOT = ROOT / "results/prp-scale-robustness-exploratory"

SCHEMA = "pzr.prp-tail-policy-exploratory.v1"
CELL_SCHEMA = "pzr.prp-tail-policy-cell.v1"
CANDIDATES = ("girard", "scott", "pca", "combastel")
BUDGETS = (40, 80, 120, 150, 200, 250, 500)
MIXTURE_SHARES = (0.05, 0.10, 0.20)
OPTIMIZER_SEEDS = (42, 1042, 2042)
DAGGER_TRAIN_SEEDS = tuple(range(312, 318))
DAGGER_VALIDATION_SEEDS = (318, 319)
SELECTION_SEEDS = tuple(range(320, 328))
CONFIRMATION_SEEDS = tuple(range(328, 348))
PILOT_BUDGETS = (40, 80, 120, 500)
EVENT_COUNT = 500
WORKERS = 10
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260802
TAIL_MULTIPLIER = 1_000.0
G15_METHOD = "g15_clean148"
PREDICTIVE_METHOD = "mpc_terminal_beam_predictive_linear"
CLEAN_ENSEMBLE_METHOD = "clean148_ensemble3"
TRAINING = {
    "epochs": 100,
    "batch_size": 256,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "patience": 10,
}
STAGES = ("preflight", "prepare", "train", "pilot", "evaluate", "report", "validate")


def mixture_label(share: float) -> str:
    if share not in MIXTURE_SHARES:
        raise ValueError(f"unsupported DAgger mixture share: {share}")
    return f"dagger{int(round(share * 100)):02d}"


def mixture_single_method(share: float) -> str:
    return f"{mixture_label(share)}_seed42"


def mixture_ensemble_method(share: float) -> str:
    return f"{mixture_label(share)}_ensemble3"


def new_methods() -> tuple[str, ...]:
    return (
        CLEAN_ENSEMBLE_METHOD,
        *(name for share in MIXTURE_SHARES for name in (
            mixture_single_method(share), mixture_ensemble_method(share),
        )),
    )


@dataclass(frozen=True)
class ExploreConfig:
    output: Path = DEFAULT_OUTPUT
    budgets: tuple[int, ...] = BUDGETS
    clean_train_seeds: tuple[int, ...] = CLEAN_TRAIN_SEEDS
    clean_validation_seeds: tuple[int, ...] = VALIDATION_SEEDS
    dagger_train_seeds: tuple[int, ...] = DAGGER_TRAIN_SEEDS
    dagger_validation_seeds: tuple[int, ...] = DAGGER_VALIDATION_SEEDS
    selection_seeds: tuple[int, ...] = SELECTION_SEEDS
    confirmation_seeds: tuple[int, ...] = CONFIRMATION_SEEDS
    event_count: int = EVENT_COUNT
    workers: int = WORKERS
    epochs: int = int(TRAINING["epochs"])
    smoke: bool = False

    def __post_init__(self) -> None:
        groups = tuple(map(set, (
            self.clean_train_seeds,
            self.clean_validation_seeds,
            self.dagger_train_seeds,
            self.dagger_validation_seeds,
            self.selection_seeds,
            self.confirmation_seeds,
        )))
        if any(left & right for index, left in enumerate(groups) for right in groups[index + 1:]):
            raise ValueError("all clean, DAgger, selection, and confirmation seed roles must be disjoint")
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
            "clean_validation_seeds": list(self.clean_validation_seeds),
            "dagger_train_seeds": list(self.dagger_train_seeds),
            "dagger_validation_seeds": list(self.dagger_validation_seeds),
            "selection_seeds": list(self.selection_seeds),
            "confirmation_seeds": list(self.confirmation_seeds),
            "event_count": self.event_count,
            "workers": self.workers,
            "smoke": self.smoke,
            "mixture_shares": list(MIXTURE_SHARES),
            "optimizer_seeds": list(OPTIMIZER_SEEDS),
            "training": {**TRAINING, "epochs": self.epochs},
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
    def expected_new_cells(self) -> int:
        return len(new_methods()) * len(self.selection_seeds) * len(self.budgets)

    @property
    def expected_reference_cells(self) -> int:
        return 2 * len(self.selection_seeds) * len(self.budgets)

    @property
    def expected_report_cells(self) -> int:
        return self.expected_new_cells + self.expected_reference_cells


def smoke_config(output: Path, *, workers: int = 1) -> ExploreConfig:
    return ExploreConfig(
        output=output,
        budgets=(40,),
        clean_train_seeds=(0,),
        clean_validation_seeds=(20,),
        dagger_train_seeds=(312,),
        dagger_validation_seeds=(318,),
        selection_seeds=(320,),
        confirmation_seeds=(328,),
        event_count=30,
        workers=workers,
        epochs=2,
        smoke=True,
    )


@dataclass(frozen=True)
class MethodSpec:
    name: str
    member_paths: tuple[Path, ...]
    ensemble: bool


@dataclass(frozen=True)
class EvaluationJob:
    config: ExploreConfig
    trace: TraceRecord
    budget: int
    spec: MethodSpec
    reference_path: Path
    directory: Path


def _raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tool_sha256() -> str:
    paths = [Path(__file__).resolve(), ROOT / "tools/run_prp_tail_policy_exploratory.sh"]
    return sha256_files(tuple(path for path in paths if path.is_file()), relative_to=ROOT)


def _stage_path(config: ExploreConfig, stage: str) -> Path:
    return config.output / stage / "manifest.json"


def _write_stage(config: ExploreConfig, stage: str, extra: Mapping[str, object]) -> Path:
    path = _stage_path(config, stage)
    write_json_atomic({
        **config.identity,
        "experiment_fingerprint": config.fingerprint,
        "stage": stage,
        "status": "completed",
        **dict(extra),
    }, path)
    return path


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _load_stage(config: ExploreConfig, stage: str) -> dict[str, object]:
    path = _stage_path(config, stage)
    manifest = _load_json(path)
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"unsupported tail-policy manifest: {path}")
    if manifest.get("experiment_fingerprint") != config.fingerprint:
        raise ValueError(f"stale tail-policy manifest: {path}")
    return manifest


def _v3_model_records(config: ExploreConfig) -> dict[int, dict[str, object]]:
    manifest = _load_json(V3_ROOT / "train/manifest.json")
    records: dict[int, dict[str, object]] = {}
    for budget in config.budgets:
        record = dict(manifest["models_by_budget"][str(budget)])
        path = ROOT / str(record["path"])
        if int(record["optimizer_seed"]) != 42 or int(record["training_size"]) != 148:
            raise ValueError(f"v3 Clean148 model identity differs at budget {budget}")
        if model_sha256(path) != record["sha256"]:
            raise ValueError(f"v3 Clean148 model hash differs at budget {budget}")
        records[budget] = {**record, "path": str(path)}
    return records


def _replica_model_records(config: ExploreConfig) -> dict[tuple[int, int], dict[str, object]]:
    manifest = _load_json(SCALE_ROOT / "train/manifest.json")
    if manifest.get("schema") != "pzr.prp-scale-robustness-exploratory.v1":
        raise ValueError("unsupported clean-scaling parent manifest")
    records: dict[tuple[int, int], dict[str, object]] = {}
    for optimizer_seed in OPTIMIZER_SEEDS[1:]:
        for budget in config.budgets:
            key = f"clean148_opt{optimizer_seed}:{budget}"
            record = dict(manifest["models"][key])
            path = Path(str(record["path"]))
            if int(record["optimizer_seed"]) != optimizer_seed or int(record["training_size"]) != 148:
                raise ValueError(f"Clean148 replica identity differs: {key}")
            if model_sha256(path) != record["sha256"]:
                raise ValueError(f"Clean148 replica model hash differs: {key}")
            records[(optimizer_seed, budget)] = {**record, "path": str(path)}
    return records


def _dagger_shard_path(seed: int, budget: int) -> Path:
    return V4_ROOT / f"collect-dagger/shards/g15/seed-{seed}/budget-{budget}"


def _source_snapshot(config: ExploreConfig) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    paper = yaml.safe_load(V3_CONFIG.read_text())
    expected = dict(paper["frozen_policy"]["source_dataset_sha256"])
    for name, dataset_path, _ in PARENT_DATASETS:
        actual = dataset_sha256(dataset_path)
        if actual != expected[name]:
            raise ValueError(f"Clean148 source hash differs for {name}: {actual}")
        snapshot[f"clean_dataset:{name}"] = actual
    for budget, record in _v3_model_records(config).items():
        snapshot[f"clean_model:42:{budget}"] = str(record["sha256"])
    for (seed, budget), record in _replica_model_records(config).items():
        snapshot[f"clean_model:{seed}:{budget}"] = str(record["sha256"])
    for seed in (*config.dagger_train_seeds, *config.dagger_validation_seeds):
        for budget in config.budgets:
            path = _dagger_shard_path(seed, budget)
            dataset, metadata, manifest = load_reducer_cost_dataset(path)
            identity = dict(manifest.get("identity", {}))
            if (
                identity.get("collection") != "autonomous_dagger_round1"
                or identity.get("feature_variant") != "g15"
                or int(identity.get("seed", -1)) != seed
                or int(identity.get("budget", -1)) != budget
                or tuple(dataset.candidate_names) != CANDIDATES
                or tuple(dataset.feature_names) != RTL_RANKING_FEATURE_SCHEMA.feature_names
                or tuple(metadata["sample_id"].astype(str)) != dataset.sample_ids
            ):
                raise ValueError(f"v4 autonomous DAgger shard identity differs: {path}")
            snapshot[f"dagger:{seed}:{budget}"] = dataset_sha256(path)
    v4_prepare = _load_json(V4_ROOT / "prepare/manifest.json")
    trace_manifest = V4_ROOT / "prepare/exploration-traces/manifest.json"
    if sha256_files((trace_manifest,)) != v4_prepare["exploration_trace_store_manifest_sha256"]:
        raise ValueError("v4 selection trace-store hash differs")
    snapshot["v4_trace_store"] = sha256_files((trace_manifest,))
    feature_manifest = V4_ROOT / "feature-screen/manifest.json"
    feature_summary = V4_ROOT / "feature-screen/summary.csv"
    snapshot["v4_feature_manifest"] = _raw_sha256(feature_manifest)
    snapshot["v4_feature_summary"] = _raw_sha256(feature_summary)
    for seed in config.selection_seeds:
        reference = V4_ROOT / f"feature-screen/references/random_waypoint_seed-{seed}.json"
        snapshot[f"v4_reference:{seed}"] = sha256_files((reference,))
        for budget in config.budgets:
            for method in (G15_METHOD, PREDICTIVE_METHOD):
                cell = V4_ROOT / (
                    f"feature-screen/cells/random_waypoint/seed-{seed}/budget-{budget}/{method}/manifest.json"
                )
                identity = dict(_load_json(cell)["identity"])
                if identity.get("reference_sha256") != snapshot[f"v4_reference:{seed}"]:
                    raise ValueError(f"v4 reference/cell hash differs: {cell}")
    for path in (
        V3_CONFIG,
        V3_ROOT / "train/manifest.json",
        V3_ROOT / "science-validate/manifest.json",
        V4_ROOT / "preflight/manifest.json",
        V4_ROOT / "prepare/manifest.json",
        V4_ROOT / "collect-dagger/manifest.json",
        SCALE_ROOT / "train/manifest.json",
    ):
        snapshot[str(path.relative_to(ROOT))] = _raw_sha256(path)
    return snapshot


def run_preflight(config: ExploreConfig) -> Path:
    if BINDING_BUILD_PROFILE != "release":
        raise ValueError("tail-policy exploration requires a release binding")
    if set(config.selection_seeds) & set(config.confirmation_seeds):
        raise ValueError("selection and confirmation seeds overlap")
    snapshot = _source_snapshot(config)
    return _write_stage(config, "preflight", {
        "scientific_role": "disposable policy-time tail-robustness selection",
        "source_snapshot": snapshot,
        "source_snapshot_sha256": payload_sha256(snapshot),
        "confirmation_automatic": False,
        "fixed_traces_automatic": False,
    })


def _load_clean(config: ExploreConfig) -> tuple[ReducerCostDataset, pd.DataFrame]:
    allowed = set(config.clean_train_seeds) | set(config.clean_validation_seeds)
    datasets: list[ReducerCostDataset] = []
    frames: list[pd.DataFrame] = []
    for name, dataset_path, _ in PARENT_DATASETS:
        dataset, metadata, _ = load_reducer_cost_dataset(dataset_path)
        selected = metadata["seed"].astype(int).isin(allowed) & metadata["budget"].astype(int).isin(config.budgets)
        indices = np.flatnonzero(selected.to_numpy())
        if not len(indices):
            continue
        datasets.append(dataset.subset(indices))
        frame = metadata.iloc[indices].reset_index(drop=True).copy()
        frame["source_dataset"] = name
        frames.append(frame)
    combined = ReducerCostDataset.concatenate(datasets)
    metadata = pd.concat(frames, ignore_index=True)
    order = np.lexsort((metadata["step"], metadata["budget"], metadata["seed"]))
    combined = combined.subset(order)
    metadata = metadata.iloc[order].reset_index(drop=True)
    expected = allowed
    if set(metadata["seed"].astype(int)) != expected:
        raise ValueError("Clean148 seed coverage differs")
    if tuple(metadata["sample_id"].astype(str)) != combined.sample_ids:
        raise ValueError("Clean148 rows and metadata are not aligned")
    if combined.candidate_names != CANDIDATES or combined.feature_names != RTL_RANKING_FEATURE_SCHEMA.feature_names:
        raise ValueError("Clean148 Geometry15/candidate schema differs")
    return combined, metadata


def _load_dagger_budget(
    config: ExploreConfig,
    budget: int,
) -> tuple[ReducerCostDataset, pd.DataFrame]:
    datasets = []
    frames = []
    train = set(config.dagger_train_seeds)
    validation = set(config.dagger_validation_seeds)
    for seed in (*config.dagger_train_seeds, *config.dagger_validation_seeds):
        dataset, metadata, _ = load_reducer_cost_dataset(_dagger_shard_path(seed, budget))
        frame = metadata.copy()
        frame["split"] = "train" if seed in train else "validation"
        if seed not in train | validation:
            raise AssertionError("unassigned DAgger seed")
        dataset = ReducerCostDataset(
            features=dataset.features,
            teacher_costs=dataset.teacher_costs,
            feasible=dataset.feasible,
            candidate_names=dataset.candidate_names,
            feature_names=dataset.feature_names,
            splits=tuple(frame["split"].astype(str)),
            sample_ids=dataset.sample_ids,
        )
        datasets.append(dataset)
        frames.append(frame)
    combined = ReducerCostDataset.concatenate(datasets)
    metadata = pd.concat(frames, ignore_index=True)
    if tuple(metadata["sample_id"].astype(str)) != combined.sample_ids:
        raise ValueError("DAgger rows and metadata are not aligned")
    return combined, metadata


def learner_count_for_share(clean_count: int, target_share: float) -> int:
    if clean_count < 1 or not 0.0 < target_share < 1.0:
        raise ValueError("mixture counts require positive clean rows and a proper share")
    return int(round(clean_count * target_share / (1.0 - target_share)))


def _sampling_seed(optimizer_seed: int, budget: int, share: float, split: str) -> int:
    payload = f"{SCHEMA}:{optimizer_seed}:{budget}:{share:.8f}:{split}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def regret_resample_plan(
    clean_count: int,
    learner_metadata: pd.DataFrame,
    *,
    target_share: float,
    optimizer_seed: int,
    budget: int,
    split: str,
) -> pd.DataFrame:
    """Return trace-balanced regret-weighted learner draws for one split."""
    required = {"sample_id", "seed", "trace_id", "selected_normalized_regret", "split"}
    if not required <= set(learner_metadata):
        raise ValueError(f"learner metadata lacks columns: {sorted(required - set(learner_metadata))}")
    frame = learner_metadata[learner_metadata["split"].astype(str) == split].reset_index(drop=True)
    if frame.empty or clean_count < 1:
        raise ValueError(f"mixture split {split!r} lacks clean or learner rows")
    regrets = frame["selected_normalized_regret"].to_numpy(float)
    if not np.all(np.isfinite(regrets)) or np.any(regrets < 0.0):
        raise ValueError("selected normalized regrets must be finite and non-negative")
    target = learner_count_for_share(clean_count, target_share)
    traces = tuple(sorted(frame["trace_id"].astype(str).unique()))
    if not traces:
        raise ValueError("learner mixture has no trace identities")
    rng = np.random.default_rng(_sampling_seed(optimizer_seed, budget, target_share, split))
    base, remainder = divmod(target, len(traces))
    extra = set(rng.permutation(len(traces))[:remainder].tolist())
    rows: list[dict[str, object]] = []
    draw_offset = 0
    for trace_index, trace_id in enumerate(traces):
        trace_rows = np.flatnonzero(frame["trace_id"].astype(str).to_numpy() == trace_id)
        quota = base + int(trace_index in extra)
        weights = 0.01 + regrets[trace_rows]
        probabilities = weights / weights.sum()
        selected = rng.choice(trace_rows, size=quota, replace=True, p=probabilities)
        for local_draw, source_index in enumerate(selected):
            source = frame.iloc[int(source_index)]
            rows.append({
                "sample_id": (
                    f"{source['sample_id']}:tailmix-{int(round(target_share * 100)):02d}"
                    f":opt-{optimizer_seed}:{split}:draw-{draw_offset + local_draw}"
                ),
                "source_sample_id": str(source["sample_id"]),
                "source_seed": int(source["seed"]),
                "trace_id": trace_id,
                "budget": budget,
                "split": split,
                "target_share": target_share,
                "optimizer_seed": optimizer_seed,
                "selected_normalized_regret": float(source["selected_normalized_regret"]),
                "sampling_weight": float(0.01 + source["selected_normalized_regret"]),
                "trace_quota": quota,
            })
        draw_offset += quota
    result = pd.DataFrame(rows)
    if len(result) != target or result["sample_id"].duplicated().any():
        raise AssertionError("learner resampling count or identifiers differ")
    quotas = result.groupby("trace_id").size().to_numpy(int)
    if int(quotas.max() - quotas.min()) > 1:
        raise AssertionError("learner trace quotas are not balanced")
    realized = len(result) / (clean_count + len(result))
    nearest_error = abs(realized - target_share)
    if nearest_error > 1.0 / (clean_count + len(result)):
        raise AssertionError("realized learner share is not within one-row rounding")
    return result


def _clean_budget(
    clean: ReducerCostDataset,
    metadata: pd.DataFrame,
    budget: int,
) -> tuple[ReducerCostDataset, pd.DataFrame]:
    indices = np.flatnonzero(metadata["budget"].astype(int).to_numpy() == budget)
    subset = clean.subset(indices)
    frame = metadata.iloc[indices].reset_index(drop=True)
    if set(subset.splits) != {"train", "validation"}:
        raise ValueError(f"Clean148 budget {budget} lacks both splits")
    return subset, frame


def run_prepare(config: ExploreConfig) -> Path:
    _load_stage(config, "preflight")
    clean, clean_metadata = _load_clean(config)
    plan_records: dict[str, dict[str, object]] = {}
    for budget in config.budgets:
        clean_subset, _ = _clean_budget(clean, clean_metadata, budget)
        _, dagger_metadata = _load_dagger_budget(config, budget)
        for share in MIXTURE_SHARES:
            for optimizer_seed in OPTIMIZER_SEEDS:
                split_plans = []
                split_counts: dict[str, object] = {}
                for split in ("train", "validation"):
                    clean_count = len(clean_subset.indices_for_split(split))
                    plan = regret_resample_plan(
                        clean_count,
                        dagger_metadata,
                        target_share=share,
                        optimizer_seed=optimizer_seed,
                        budget=budget,
                        split=split,
                    )
                    split_plans.append(plan)
                    split_counts[split] = {
                        "clean_count": clean_count,
                        "learner_count": len(plan),
                        "realized_learner_share": len(plan) / (clean_count + len(plan)),
                        "learner_seeds": sorted(plan["source_seed"].astype(int).unique().tolist()),
                    }
                combined = pd.concat(split_plans, ignore_index=True)
                path = config.output / (
                    f"prepare/resampling/{mixture_label(share)}/opt-{optimizer_seed}/budget-{budget}.csv"
                )
                write_csv_atomic(combined, path)
                key = f"{mixture_label(share)}:{optimizer_seed}:{budget}"
                plan_records[key] = {
                    "path": str(path),
                    "sha256": sha256_files((path,)),
                    "rows": len(combined),
                    "splits": split_counts,
                }
    expected = len(MIXTURE_SHARES) * len(OPTIMIZER_SEEDS) * len(config.budgets)
    if len(plan_records) != expected:
        raise AssertionError("resampling plan matrix is incomplete")
    return _write_stage(config, "prepare", {
        "clean_sample_count": clean.num_samples,
        "plan_count": len(plan_records),
        "plans": plan_records,
        "sampling_rule": "trace_balanced_with_replacement_probability_0.01_plus_selected_normalized_regret",
        "complete_candidate_blocks_retained": True,
        "candidate_symmetry": list(CANDIDATES),
    })


def build_mixture_dataset(
    clean: ReducerCostDataset,
    clean_metadata: pd.DataFrame,
    dagger: ReducerCostDataset,
    dagger_metadata: pd.DataFrame,
    plan: pd.DataFrame,
) -> tuple[ReducerCostDataset, pd.DataFrame]:
    """Combine clean rows once with the aligned full candidate blocks in ``plan``."""
    if tuple(clean_metadata["sample_id"].astype(str)) != clean.sample_ids:
        raise ValueError("clean dataset and metadata differ")
    if tuple(dagger_metadata["sample_id"].astype(str)) != dagger.sample_ids:
        raise ValueError("DAgger dataset and metadata differ")
    if clean.candidate_names != dagger.candidate_names or clean.feature_names != dagger.feature_names:
        raise ValueError("clean and DAgger schemas differ")
    source_index = {sample_id: index for index, sample_id in enumerate(dagger.sample_ids)}
    try:
        selected = np.asarray([source_index[value] for value in plan["source_sample_id"].astype(str)], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"resampling plan references an unknown learner row: {exc}") from exc
    learner = ReducerCostDataset(
        features=dagger.features[selected],
        teacher_costs=dagger.teacher_costs[selected],
        feasible=dagger.feasible[selected],
        candidate_names=dagger.candidate_names,
        feature_names=dagger.feature_names,
        splits=tuple(plan["split"].astype(str)),
        sample_ids=tuple(plan["sample_id"].astype(str)),
    )
    clean_frame = clean_metadata.copy()
    clean_frame["training_source"] = np.where(
        clean_frame["split"].astype(str) == "train", "clean148", "clean_validation",
    )
    learner_frame = dagger_metadata.iloc[selected].reset_index(drop=True).copy()
    learner_frame["source_sample_id"] = learner_frame["sample_id"].astype(str)
    learner_frame["sample_id"] = plan["sample_id"].astype(str).to_numpy()
    learner_frame["split"] = plan["split"].astype(str).to_numpy()
    learner_frame["training_source"] = np.where(
        learner_frame["split"].astype(str) == "train",
        "learner_visited_dagger_train",
        "learner_visited_dagger_validation",
    )
    # Keep each split contiguous so train/validation accounting is transparent.
    parts = []
    frames = []
    for split in ("train", "validation"):
        clean_indices = clean.indices_for_split(split)
        learner_indices = learner.indices_for_split(split)
        parts.extend((clean.subset(clean_indices), learner.subset(learner_indices)))
        frames.extend((
            clean_frame.iloc[clean_indices].copy(),
            learner_frame.iloc[learner_indices].copy(),
        ))
    aggregate = ReducerCostDataset.concatenate(parts)
    metadata = pd.concat(frames, ignore_index=True)
    if tuple(metadata["sample_id"].astype(str)) != aggregate.sample_ids:
        raise AssertionError("mixture sample/cost alignment differs")
    if aggregate.candidate_names != CANDIDATES or aggregate.teacher_costs.shape[1] != 4:
        raise AssertionError("mixture did not retain the complete symmetric candidate block")
    train_learner_seeds = set(metadata.loc[
        metadata["training_source"] == "learner_visited_dagger_train", "seed"
    ].astype(int))
    validation_learner_seeds = set(metadata.loc[
        metadata["training_source"] == "learner_visited_dagger_validation", "seed"
    ].astype(int))
    if train_learner_seeds & validation_learner_seeds:
        raise AssertionError("DAgger train/validation seed leakage")
    return aggregate, metadata


def _plan_record(config: ExploreConfig, share: float, optimizer_seed: int, budget: int) -> dict[str, object]:
    prepare = _load_stage(config, "prepare")
    key = f"{mixture_label(share)}:{optimizer_seed}:{budget}"
    record = dict(prepare["plans"][key])
    path = Path(str(record["path"]))
    if sha256_files((path,)) != record["sha256"]:
        raise ValueError(f"resampling plan hash differs: {key}")
    return record


def _train_model_record(
    path: Path,
    *,
    share: float,
    optimizer_seed: int,
    budget: int,
) -> dict[str, object]:
    return {
        "mixture_share": share,
        "optimizer_seed": optimizer_seed,
        "budget": budget,
        "path": str(path),
        "sha256": model_sha256(path),
        "feature_schema": asdict(RTL_RANKING_FEATURE_SCHEMA),
        "candidate_names": list(CANDIDATES),
    }


def run_train(config: ExploreConfig) -> Path:
    _load_stage(config, "prepare")
    clean, clean_metadata = _load_clean(config)
    records: dict[str, dict[str, object]] = {}
    for budget in config.budgets:
        clean_subset, clean_frame = _clean_budget(clean, clean_metadata, budget)
        dagger, dagger_metadata = _load_dagger_budget(config, budget)
        for share in MIXTURE_SHARES:
            for optimizer_seed in OPTIMIZER_SEEDS:
                plan_record = _plan_record(config, share, optimizer_seed, budget)
                plan = pd.read_csv(str(plan_record["path"]))
                aggregate, aggregate_frame = build_mixture_dataset(
                    clean_subset, clean_frame, dagger, dagger_metadata, plan,
                )
                aggregate_frame = aggregate_frame.copy()
                aggregate_frame["dataset_label"] = aggregate_frame["training_source"]
                output = config.output / (
                    f"train/{mixture_label(share)}/opt-{optimizer_seed}/budget-{budget}"
                )
                identity = {
                    "experiment_fingerprint": config.fingerprint,
                    "mixture_share": share,
                    "optimizer_seed": optimizer_seed,
                    "budget": budget,
                    "plan_sha256": plan_record["sha256"],
                    "sample_ids_sha256": payload_sha256({"sample_ids": list(aggregate.sample_ids)}),
                    "training": {**TRAINING, "epochs": config.epochs},
                }
                artifact = output / "exploratory_training.json"
                key = f"{mixture_label(share)}:{optimizer_seed}:{budget}"
                if artifact.is_file():
                    existing = _load_json(artifact)
                    if existing.get("identity") != identity or model_sha256(output) != existing.get("model_sha256"):
                        raise ValueError(f"stale tail-policy model: {output}")
                    records[key] = dict(existing["record"])
                    continue
                policy, result = train_reducer_policy(
                    aggregate,
                    RTL_RANKING_FEATURE_SCHEMA,
                    objective="pairwise",
                    epochs=config.epochs,
                    batch_size=int(TRAINING["batch_size"]),
                    learning_rate=float(TRAINING["learning_rate"]),
                    weight_decay=float(TRAINING["weight_decay"]),
                    patience=min(int(TRAINING["patience"]), config.epochs),
                    seed=optimizer_seed,
                )
                policy.save(output)
                write_csv_atomic(
                    validation_metrics(policy, aggregate, aggregate_frame),
                    output / "validation_metrics.csv",
                )
                record = {
                    **_train_model_record(
                        output,
                        share=share,
                        optimizer_seed=optimizer_seed,
                        budget=budget,
                    ),
                    "best_epoch": result.best_epoch,
                    "epochs_completed": result.epochs,
                    "train_rows": len(aggregate.indices_for_split("train")),
                    "validation_rows": len(aggregate.indices_for_split("validation")),
                }
                write_json_atomic({
                    "identity": identity,
                    "model_sha256": record["sha256"],
                    "record": record,
                    "validation_metrics": asdict(result.val_metrics),
                }, artifact)
                records[key] = record
    expected = len(MIXTURE_SHARES) * len(OPTIMIZER_SEEDS) * len(config.budgets)
    if len(records) != expected:
        raise AssertionError(f"trained {len(records)} models, expected {expected}")
    return _write_stage(config, "train", {"model_count": len(records), "models": records})


def stable_rank_ensemble(
    member_scores: NDArray[np.floating],
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Order candidates by mean ordinal rank, standardized score, then catalog."""
    scores = np.asarray(member_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(CANDIDATES) or scores.shape[0] < 1:
        raise ValueError("ensemble scores must have shape (members, four candidates)")
    if not np.all(np.isfinite(scores)):
        raise ValueError("ensemble scores contain non-finite values")
    ranks = np.empty_like(scores, dtype=np.float64)
    for member, row in enumerate(scores):
        order = np.argsort(row, kind="stable")
        ranks[member, order] = np.arange(len(CANDIDATES), dtype=np.float64)
    mean_ranks = ranks.mean(axis=0)
    centered = scores - scores.mean(axis=1, keepdims=True)
    scale = scores.std(axis=1, keepdims=True)
    standardized = np.divide(
        centered,
        scale,
        out=np.zeros_like(centered),
        where=scale > 1e-12,
    )
    mean_standardized = standardized.mean(axis=0)
    order = np.lexsort((np.arange(len(CANDIDATES)), mean_standardized, mean_ranks)).astype(np.int64)
    return order, mean_ranks, mean_standardized, ranks


class RankEnsemblePolicy:
    """Inference-only rank ensemble with one direct native branch per event."""

    def __init__(self, policies: Sequence[ReducerPolicy], events: Sequence[RtlolaEvent]) -> None:
        self.policies = tuple(policies)
        if not self.policies:
            raise ValueError("rank ensemble requires at least one policy")
        if any(policy.candidate_names != CANDIDATES for policy in self.policies):
            raise ValueError("rank ensemble candidate catalogs differ")
        if any(policy.feature_schema != RTL_RANKING_FEATURE_SCHEMA for policy in self.policies):
            raise ValueError("rank ensemble requires Geometry15 models")
        self.events = tuple(events)
        self.catalog: RtlolaActionCatalog = default_action_catalog(CANDIDATES)
        self.step = 0
        self.diagnostics: list[dict[str, object]] = []

    def choose(
        self,
        engine: RtlolaEngine,
        state: RtlolaStateRef,
        event: RtlolaEvent,
        budget: int,
    ) -> RtlolaSearchResult:
        if self.step >= len(self.events) or self.events[self.step] != event:
            raise ValueError("rank-ensemble event history is not aligned")
        step_index = self.step
        self.step += 1
        metrics = engine.metrics(state)
        if metrics.dynamic_generator_count <= budget:
            branch = engine.branch_step(state, event, self.catalog.no_op, budget)
            decision = RtlolaSearchResult(
                first_action=self.catalog.no_op,
                first_action_budget=budget,
                first_step=branch,
                predicted_cost=0.0,
                predicted_sequence=(self.catalog.no_op.name,),
                evaluated_leaves=1,
                pruned_branches=0,
                mpc_variant="direct_rank_ensemble",
                root_strategy="rank_average_direct",
            )
            self.diagnostics.append({
                "step": step_index,
                "budget": budget,
                "over_bound": False,
                "selected_action": self.catalog.no_op.name,
                "evaluated_leaves": 1,
                "fallback_used": False,
                "member_count": len(self.policies),
                "member_top1_disagreement": False,
                "mean_rank_variance": 0.0,
            })
            return decision
        features = extract_ranking_features(engine, state, budget)
        scores = np.asarray([
            np.asarray(policy.predict_scores(features), dtype=np.float64)
            for policy in self.policies
        ])
        order, mean_ranks, standardized, ranks = stable_rank_ensemble(scores)
        failures = 0
        selected_index: int | None = None
        branch = None
        for raw_index in order:
            candidate = int(raw_index)
            action = self.catalog.by_name[CANDIDATES[candidate]]
            if action.explicit_budget and budget < metrics.dimension:
                failures += 1
                continue
            try:
                branch = engine.branch_step(state, event, action, budget)
            except RtlolaBindingError:
                failures += 1
                continue
            selected_index = candidate
            break
        fallback_used = selected_index is None
        if fallback_used:
            try:
                branch = engine.branch_step(state, event, self.catalog.fallback, budget)
            except RtlolaBindingError as exc:
                raise RtlolaNoFeasibleAction("rank ensemble and interval fallback were infeasible") from exc
            action = self.catalog.fallback
            predicted_cost = float("nan")
        else:
            action = self.catalog.by_name[CANDIDATES[selected_index]]
            predicted_cost = float(mean_ranks[selected_index])
        assert branch is not None
        member_top1 = np.argmin(ranks, axis=1)
        self.diagnostics.append({
            "step": step_index,
            "budget": budget,
            "over_bound": True,
            "selected_action": action.name,
            "evaluated_leaves": 1,
            "fallback_used": fallback_used,
            "member_count": len(self.policies),
            "member_top1_disagreement": len(set(member_top1.tolist())) > 1,
            "distinct_member_top1": len(set(member_top1.tolist())),
            "mean_rank_variance": float(np.mean(np.var(ranks, axis=0))),
            "ranking": json.dumps([CANDIDATES[int(index)] for index in order]),
            **{f"mean_rank_{name}": mean_ranks[index] for index, name in enumerate(CANDIDATES)},
            **{f"mean_standardized_score_{name}": standardized[index] for index, name in enumerate(CANDIDATES)},
        })
        return RtlolaSearchResult(
            first_action=action,
            first_action_budget=budget,
            first_step=branch,
            predicted_cost=predicted_cost,
            predicted_sequence=(action.name,),
            evaluated_leaves=1,
            pruned_branches=0,
            fallback_used=fallback_used,
            reducer_failure_count=failures,
            infeasible_candidate_count=failures,
            mpc_variant="direct_rank_ensemble",
            root_strategy="rank_average_direct",
        )


def _trained_record(config: ExploreConfig, share: float, optimizer_seed: int, budget: int) -> dict[str, object]:
    manifest = _load_stage(config, "train")
    key = f"{mixture_label(share)}:{optimizer_seed}:{budget}"
    record = dict(manifest["models"][key])
    path = Path(str(record["path"]))
    if model_sha256(path) != record["sha256"]:
        raise ValueError(f"tail-policy model hash differs: {key}")
    return record


def _method_specs(config: ExploreConfig, budget: int) -> tuple[MethodSpec, ...]:
    v3 = _v3_model_records(config)
    replicas = _replica_model_records(config)
    clean_members = (
        Path(str(v3[budget]["path"])),
        *(Path(str(replicas[(seed, budget)]["path"])) for seed in OPTIMIZER_SEEDS[1:]),
    )
    specs = [MethodSpec(CLEAN_ENSEMBLE_METHOD, tuple(clean_members), True)]
    for share in MIXTURE_SHARES:
        members = tuple(
            Path(str(_trained_record(config, share, seed, budget)["path"]))
            for seed in OPTIMIZER_SEEDS
        )
        specs.extend((
            MethodSpec(mixture_single_method(share), (members[0],), False),
            MethodSpec(mixture_ensemble_method(share), members, True),
        ))
    return tuple(specs)


def _selection_traces(config: ExploreConfig) -> tuple[TraceRecord, ...]:
    store = load_random_waypoint_trace_store(V4_ROOT / "prepare/exploration-traces")
    records = tuple(
        _trace_record(
            store.traces_for_seed(seed)[0],
            limit=config.event_count if config.smoke else None,
        )
        for seed in config.selection_seeds
    )
    if tuple(item.seed for item in records) != config.selection_seeds:
        raise ValueError("selection trace seed coverage differs")
    return records


def _reference_path(config: ExploreConfig, trace: TraceRecord) -> Path:
    if not config.smoke:
        return V4_ROOT / f"feature-screen/references/random_waypoint_seed-{trace.seed}.json"
    path = config.output / f"evaluate/references/random_waypoint_seed-{trace.seed}.json"
    load_or_compute_reference(
        trace.events,
        scenario=scenario_by_name("robot_arm"),
        trace_kind=trace.trace_id,
        seed=trace.seed,
        cache_path=path,
        include_approximation=True,
    )
    return path


def _execute_evaluation(job: EvaluationJob) -> dict[str, object]:
    manifest_path = job.directory / "manifest.json"
    summary_path = job.directory / "summary.csv"
    member_hashes = [model_sha256(path) for path in job.spec.member_paths]
    identity = {
        "schema": CELL_SCHEMA,
        "experiment_fingerprint": job.config.fingerprint,
        "trace_id": job.trace.trace_id,
        "trace_sha256": job.trace.trace_sha256,
        "event_count": len(job.trace.events),
        "budget": job.budget,
        "method": job.spec.name,
        "member_model_sha256": member_hashes,
        "ensemble": job.spec.ensemble,
        "reference_sha256": sha256_files((job.reference_path,)),
    }
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        if manifest.get("identity") != identity:
            raise ValueError(f"stale tail-policy cell: {job.directory}")
        return pd.read_csv(summary_path).iloc[0].to_dict()
    job.directory.mkdir(parents=True, exist_ok=True)
    policies = tuple(ReducerPolicy.load(path) for path in job.spec.member_paths)
    policy = RankEnsemblePolicy(policies, job.trace.events)
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
        beam_width=4,
        prediction_step_seconds=0.1,
        seeds=1,
        methods=[job.spec.name],
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
        method=job.spec.name,
        policy=policy,
        reference_steps=reference,
    )
    elapsed = perf_counter() - started
    if len(result.summary) == 1:
        row = result.summary.iloc[0].to_dict()
        status = "completed"
    elif result.failures:
        failure = result.failures[0]
        row = {
            "mean_approx_loss": np.nan,
            "fpr": np.nan,
            "fnr": np.nan,
            "event_count": len(job.trace.events),
            "failure_type": failure.failure_type,
            "failure_message": failure.message,
        }
        status = "fallback_failed" if failure.failure_type == "RtlolaNoFeasibleAction" else "native_failed"
    else:
        raise ValueError("benchmark produced neither a result nor a failure")
    row.update({
        "status": status,
        "scope": "evaluate",
        "trace_id": job.trace.trace_id,
        "trace_sha256": job.trace.trace_sha256,
        "condition": job.trace.trace_kind,
        "seed": job.trace.seed,
        "budget": job.budget,
        "event_count": len(job.trace.events),
        "method": job.spec.name,
        "ensemble": job.spec.ensemble,
        "member_count": len(job.spec.member_paths),
        "cell_elapsed_seconds": elapsed,
        "member_model_sha256": json.dumps(member_hashes),
    })
    write_csv_atomic(pd.DataFrame([row]), summary_path)
    if not result.timeseries.empty:
        write_csv_atomic(result.timeseries, job.directory / "timeseries.csv")
    if not result.failed_timeseries.empty:
        write_csv_atomic(result.failed_timeseries, job.directory / "failed_timeseries.csv")
    write_csv_atomic(pd.DataFrame(policy.diagnostics), job.directory / "decisions.csv")
    write_json_atomic({"schema": CELL_SCHEMA, "identity": identity, "status": status}, manifest_path)
    return row


def _jobs(
    config: ExploreConfig,
    traces: Sequence[TraceRecord],
    budgets: Sequence[int],
) -> list[EvaluationJob]:
    jobs = []
    for trace in traces:
        reference = _reference_path(config, trace)
        for budget in budgets:
            for spec in _method_specs(config, budget):
                jobs.append(EvaluationJob(
                    config=config,
                    trace=trace,
                    budget=budget,
                    spec=spec,
                    reference_path=reference,
                    directory=config.output / (
                        f"evaluate/cells/random_waypoint/seed-{trace.seed}/budget-{budget}/{spec.name}"
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


def run_pilot(config: ExploreConfig) -> Path:
    _load_stage(config, "train")
    budgets = tuple(value for value in PILOT_BUDGETS if value in config.budgets) or config.budgets[:1]
    trace = _selection_traces(config)[:1]
    started = perf_counter()
    summary = _run_jobs(config, _jobs(config, trace, budgets))
    wall = perf_counter() - started
    expected = len(new_methods()) * len(budgets)
    if len(summary) != expected:
        raise ValueError(f"pilot has {len(summary)} cells, expected {expected}")
    path = config.output / "pilot/summary.csv"
    write_csv_atomic(summary, path)
    cpu = float(summary["cell_elapsed_seconds"].sum())
    projected_cpu = cpu * config.expected_new_cells / len(summary)
    return _write_stage(config, "pilot", {
        "pilot_cell_count": len(summary),
        "pilot_seed": int(trace[0].seed),
        "pilot_budgets": list(budgets),
        "reused_by_evaluate": True,
        "pilot_wall_seconds": wall,
        "pilot_cpu_seconds": cpu,
        "projected_new_cell_count": config.expected_new_cells,
        "projected_cpu_seconds": projected_cpu,
        "projected_wall_seconds_at_workers": projected_cpu / config.workers,
        "summary": str(path),
    })


def _execute_smoke_reference(
    config: ExploreConfig,
    trace: TraceRecord,
    budget: int,
    method: str,
) -> dict[str, object]:
    reference_path = _reference_path(config, trace)
    directory = config.output / f"evaluate/reference-cells/seed-{trace.seed}/budget-{budget}/{method}"
    manifest_path = directory / "manifest.json"
    summary_path = directory / "summary.csv"
    identity = {
        "schema": CELL_SCHEMA,
        "experiment_fingerprint": config.fingerprint,
        "trace_sha256": trace.trace_sha256,
        "event_count": len(trace.events),
        "budget": budget,
        "method": method,
        "reference_sha256": sha256_files((reference_path,)),
    }
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        if manifest.get("identity") != identity:
            raise ValueError(f"stale smoke reference cell: {directory}")
        return pd.read_csv(summary_path).iloc[0].to_dict()
    directory.mkdir(parents=True, exist_ok=True)
    direct = None
    horizon = 4 if method == PREDICTIVE_METHOD else 0
    if method == G15_METHOD:
        path = Path(str(_v3_model_records(config)[budget]["path"]))
        direct = RankEnsemblePolicy((ReducerPolicy.load(path),), trace.events)
    reference = load_or_compute_reference(
        trace.events,
        scenario=scenario_by_name("robot_arm"),
        trace_kind=trace.trace_id,
        seed=trace.seed,
        cache_path=reference_path,
        include_approximation=True,
    )
    benchmark_config = RtlolaBenchmarkConfig(
        scenario="robot_arm",
        trace_kind=trace.trace_kind,
        length=len(trace.events),
        budget=budget,
        horizon=horizon,
        beam_width=4,
        prediction_step_seconds=0.1,
        seeds=1,
        methods=[method],
        reference_mode="exact",
        mpc_reference="rollout",
        output_dir=str(directory),
        mpc_candidate_names=list(CANDIDATES),
    )
    result = run_event_trace_benchmark(
        benchmark_config,
        trace.events,
        trace_kind=trace.trace_kind,
        seed=trace.seed,
        method=method,
        policy=direct,
        reference_steps=reference,
    )
    if len(result.summary) != 1:
        raise ValueError(f"smoke reference failed: {method}")
    row = result.summary.iloc[0].to_dict()
    row.update({
        "status": "completed",
        "scope": "reused_reference",
        "trace_id": trace.trace_id,
        "trace_sha256": trace.trace_sha256,
        "condition": trace.trace_kind,
        "seed": trace.seed,
        "budget": budget,
        "event_count": len(trace.events),
        "method": method,
        "ensemble": False,
        "member_count": 1 if method == G15_METHOD else 0,
    })
    write_csv_atomic(pd.DataFrame([row]), summary_path)
    write_json_atomic({"schema": CELL_SCHEMA, "identity": identity, "status": "completed"}, manifest_path)
    return row


def _reference_rows(config: ExploreConfig) -> pd.DataFrame:
    if config.smoke:
        rows = [
            _execute_smoke_reference(config, trace, budget, method)
            for trace in _selection_traces(config)
            for budget in config.budgets
            for method in (G15_METHOD, PREDICTIVE_METHOD)
        ]
        return pd.DataFrame(rows)
    summary = pd.read_csv(V4_ROOT / "feature-screen/summary.csv")
    selected = summary[
        summary["method"].isin((G15_METHOD, PREDICTIVE_METHOD))
        & summary["seed"].astype(int).isin(config.selection_seeds)
        & summary["budget"].astype(int).isin(config.budgets)
    ].copy()
    expected = config.expected_reference_cells
    keys = selected[["seed", "budget", "method"]]
    if len(selected) != expected or keys.duplicated().any():
        raise ValueError(f"v4 reference rows have {len(selected)} cells, expected {expected}")
    if set(selected["status"].astype(str)) != {"completed"}:
        raise ValueError("v4 reference rows include failures")
    selected["scope"] = "reused_reference"
    selected["source_summary_sha256"] = _raw_sha256(V4_ROOT / "feature-screen/summary.csv")
    return selected


def run_evaluate(config: ExploreConfig) -> Path:
    _load_stage(config, "pilot")
    traces = _selection_traces(config)
    started = perf_counter()
    fresh = _run_jobs(config, _jobs(config, traces, config.budgets))
    expected = config.expected_new_cells
    if len(fresh) != expected:
        raise ValueError(f"new evaluation has {len(fresh)} cells, expected {expected}")
    references = _reference_rows(config)
    combined = pd.concat((fresh, references), ignore_index=True, sort=False)
    if len(combined) != config.expected_report_cells:
        raise ValueError(
            f"joined evaluation has {len(combined)} cells, expected {config.expected_report_cells}"
        )
    fresh_path = config.output / "evaluate/new_summary.csv"
    reference_path = config.output / "evaluate/reused_reference_summary.csv"
    joined_path = config.output / "evaluate/summary.csv"
    write_csv_atomic(fresh, fresh_path)
    write_csv_atomic(references, reference_path)
    write_csv_atomic(combined, joined_path)
    return _write_stage(config, "evaluate", {
        "new_cell_count": len(fresh),
        "reference_cell_count": len(references),
        "joined_cell_count": len(combined),
        "failure_count": int((fresh["status"].astype(str) != "completed").sum()),
        "matrix_wall_seconds": perf_counter() - started,
        "pilot_cells_reused": len(new_methods()) * len(
            tuple(value for value in PILOT_BUDGETS if value in config.budgets) or config.budgets[:1]
        ),
        "new_summary": str(fresh_path),
        "reference_summary": str(reference_path),
        "summary": str(joined_path),
    })


def _paired_values(
    summary: pd.DataFrame,
    method: str,
    reference_method: str,
    budget: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["seed", "budget"]
    candidate = summary[summary["method"] == method]
    reference = summary[summary["method"] == reference_method]
    if budget is not None:
        candidate = candidate[candidate["budget"].astype(int) == budget]
        reference = reference[reference["budget"].astype(int) == budget]
    candidate = candidate.set_index(keys).sort_index()
    reference = reference.set_index(keys).sort_index()
    if not candidate.index.equals(reference.index):
        raise ValueError(f"paired cells differ: {method} vs {reference_method}")
    return candidate, reference


def _bootstrap_mean(values: NDArray[np.float64]) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    means = values[draws].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def policy_metrics(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-budget and overall paired tail/accuracy/throughput metrics."""
    per_budget = []
    overall = []
    for method in new_methods():
        all_candidate, all_g15 = _paired_values(summary, method, G15_METHOD)
        _, all_predictive = _paired_values(summary, method, PREDICTIVE_METHOD)
        valid = (
            (all_candidate["status"].astype(str) == "completed")
            & (all_g15["status"].astype(str) == "completed")
            & (all_predictive["status"].astype(str) == "completed")
        )
        candidate_loss = all_candidate.loc[valid, "mean_approx_loss"].to_numpy(float)
        g15_loss = all_g15.loc[valid, "mean_approx_loss"].to_numpy(float)
        predictive_loss = all_predictive.loc[valid, "mean_approx_loss"].to_numpy(float)
        ratio_g15 = candidate_loss / g15_loss
        ratio_predictive = candidate_loss / predictive_loss
        fpr_difference = (
            all_candidate.loc[valid, "fpr"].to_numpy(float)
            - all_g15.loc[valid, "fpr"].to_numpy(float)
        )
        throughput_candidate = (
            all_candidate.loc[valid, "event_count"].to_numpy(float)
            / (all_candidate.loc[valid, "event_loop_time_ms"].to_numpy(float) / 1000.0)
        )
        throughput_g15 = (
            all_g15.loc[valid, "event_count"].to_numpy(float)
            / (all_g15.loc[valid, "event_loop_time_ms"].to_numpy(float) / 1000.0)
        )
        fpr_mean, fpr_low, fpr_high = _bootstrap_mean(fpr_difference)
        fallback_events = int(pd.to_numeric(all_candidate["fallback_count"], errors="coerce").fillna(0).sum())
        overall.append({
            "method": method,
            "cell_count": len(all_candidate),
            "valid_count": int(valid.sum()),
            "failure_count": int((all_candidate["status"].astype(str) != "completed").sum()),
            "fallback_count": fallback_events,
            "severe_tail_count": int((ratio_predictive > TAIL_MULTIPLIER).sum()),
            "worst_loss_ratio_vs_predictive": float(np.max(ratio_predictive)),
            "p95_loss_ratio_vs_predictive": float(np.quantile(ratio_predictive, 0.95)),
            "median_loss_ratio_vs_g15": float(np.median(ratio_g15)),
            "mean_fpr_difference_vs_g15": fpr_mean,
            "mean_fpr_difference_ci_low": fpr_low,
            "mean_fpr_difference_ci_high": fpr_high,
            "max_fpr_difference_vs_g15": float(np.max(fpr_difference)),
            "median_paired_throughput_retention": float(np.median(throughput_candidate / throughput_g15)),
            "ensemble": method.endswith("ensemble3"),
        })
        for budget in sorted(summary["budget"].astype(int).unique()):
            candidate, g15 = _paired_values(summary, method, G15_METHOD, budget)
            _, predictive = _paired_values(summary, method, PREDICTIVE_METHOD, budget)
            budget_valid = (
                (candidate["status"].astype(str) == "completed")
                & (g15["status"].astype(str) == "completed")
                & (predictive["status"].astype(str) == "completed")
            )
            loss = candidate.loc[budget_valid, "mean_approx_loss"].to_numpy(float)
            loss_g15 = g15.loc[budget_valid, "mean_approx_loss"].to_numpy(float)
            loss_predictive = predictive.loc[budget_valid, "mean_approx_loss"].to_numpy(float)
            fpr = (
                candidate.loc[budget_valid, "fpr"].to_numpy(float)
                - g15.loc[budget_valid, "fpr"].to_numpy(float)
            )
            fpr_mean_b, fpr_low_b, fpr_high_b = _bootstrap_mean(fpr)
            per_budget.append({
                "method": method,
                "budget": budget,
                "cell_count": len(candidate),
                "valid_count": int(budget_valid.sum()),
                "failure_count": int((~budget_valid).sum()),
                "median_loss": float(np.median(loss)),
                "q25_loss": float(np.quantile(loss, 0.25)),
                "q75_loss": float(np.quantile(loss, 0.75)),
                "p95_loss": float(np.quantile(loss, 0.95)),
                "worst_loss": float(np.max(loss)),
                "median_loss_ratio_vs_g15": float(np.median(loss / loss_g15)),
                "p95_loss_ratio_vs_predictive": float(np.quantile(loss / loss_predictive, 0.95)),
                "worst_loss_ratio_vs_predictive": float(np.max(loss / loss_predictive)),
                "severe_tail_count": int(((loss / loss_predictive) > TAIL_MULTIPLIER).sum()),
                "mean_fpr_difference_vs_g15": fpr_mean_b,
                "mean_fpr_difference_ci_low": fpr_low_b,
                "mean_fpr_difference_ci_high": fpr_high_b,
                "max_fpr_difference_vs_g15": float(np.max(fpr)),
            })
    per_budget_frame = pd.DataFrame(per_budget)
    overall_frame = pd.DataFrame(overall)
    max_regression = per_budget_frame.groupby("method")["median_loss_ratio_vs_g15"].max()
    overall_frame["max_budget_median_loss_ratio_vs_g15"] = overall_frame["method"].map(max_regression)
    return per_budget_frame, overall_frame


def eligibility_table(per_budget: pd.DataFrame, overall: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in overall.iterrows():
        budget_rows = per_budget[per_budget["method"] == row["method"]]
        checks = {
            "zero_severe_tail": int(row["severe_tail_count"]) == 0,
            "zero_failures": int(row["failure_count"]) == 0,
            "zero_fallbacks": int(row["fallback_count"]) == 0,
            "all_budget_medians_within_1_25_g15": bool(
                (budget_rows["median_loss_ratio_vs_g15"] <= 1.25).all()
            ),
            "throughput_at_least_half_g15": float(row["median_paired_throughput_retention"]) >= 0.50,
            "mean_fpr_regression_at_most_0_005": float(row["mean_fpr_difference_vs_g15"]) <= 0.005,
            "individual_fpr_regression_at_most_0_05": float(row["max_fpr_difference_vs_g15"]) <= 0.05,
        }
        rows.append({**row.to_dict(), **checks, "eligible": all(checks.values())})
    table = pd.DataFrame(rows)
    table = table.sort_values(
        [
            "eligible",
            "worst_loss_ratio_vs_predictive",
            "p95_loss_ratio_vs_predictive",
            "max_budget_median_loss_ratio_vs_g15",
            "median_paired_throughput_retention",
            "ensemble",
            "method",
        ],
        ascending=[False, True, True, True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    table["eligible_rank"] = np.where(
        table["eligible"], np.arange(1, len(table) + 1), np.nan,
    )
    return table


def _decision_tables(config: ExploreConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for path in sorted((config.output / "evaluate/cells").rglob("decisions.csv")):
        frame = pd.read_csv(path)
        method = path.parent.name
        budget = int(next(part.split("-", 1)[1] for part in path.parts if part.startswith("budget-")))
        seed = int(next(part.split("-", 1)[1] for part in path.parts if part.startswith("seed-")))
        frame.insert(0, "method", method)
        frame.insert(1, "seed", seed)
        frame["budget"] = budget
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    decisions = pd.concat(frames, ignore_index=True)
    ordinary = decisions[
        decisions["over_bound"].astype(bool)
        & decisions["selected_action"].isin(CANDIDATES)
    ]
    composition = ordinary.groupby(
        ["method", "budget", "selected_action"], sort=True,
    ).size().rename("count").reset_index()
    totals = composition.groupby(["method", "budget"])["count"].transform("sum")
    composition["percentage"] = 100.0 * composition["count"] / totals
    disagreement = decisions[decisions["over_bound"].astype(bool)].groupby(
        ["method", "budget"], sort=True,
    ).agg(
        decision_count=("step", "size"),
        member_top1_disagreement_count=("member_top1_disagreement", "sum"),
        member_top1_disagreement_rate=("member_top1_disagreement", "mean"),
        mean_rank_variance=("mean_rank_variance", "mean"),
        max_rank_variance=("mean_rank_variance", "max"),
    ).reset_index()
    return composition, disagreement


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    return plt


def _paired_long(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in new_methods():
        candidate, g15 = _paired_values(summary, method, G15_METHOD)
        _, predictive = _paired_values(summary, method, PREDICTIVE_METHOD)
        for key in candidate.index:
            cand = candidate.loc[key]
            base = g15.loc[key]
            ref = predictive.loc[key]
            rows.append({
                "method": method,
                "seed": int(key[0]),
                "budget": int(key[1]),
                "loss_ratio_vs_predictive": float(cand["mean_approx_loss"] / ref["mean_approx_loss"]),
                "loss_ratio_vs_g15": float(cand["mean_approx_loss"] / base["mean_approx_loss"]),
                "fpr_difference_vs_g15": float(cand["fpr"] - base["fpr"]),
                "throughput_retention_vs_g15": float(
                    (cand["event_count"] / cand["event_loop_time_ms"])
                    / (base["event_count"] / base["event_loop_time_ms"])
                ),
            })
    return pd.DataFrame(rows)


def _save_figure(fig: object, output: Path) -> tuple[Path, Path]:
    pdf = output.with_suffix(".pdf")
    png = output.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png, dpi=250, bbox_inches="tight", pad_inches=0.03)
    return pdf, png


def _plot_loss_ecdf(paired: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(7.1, 3.4))
    for index, method in enumerate(new_methods()):
        values = np.sort(paired.loc[paired["method"] == method, "loss_ratio_vs_predictive"].to_numpy(float))
        axis.step(
            values,
            np.arange(1, len(values) + 1) / len(values),
            where="post",
            label=method,
            linewidth=1.0,
            linestyle=("-", "--", "-.", ":")[index % 4],
        )
    axis.axvline(1.0, color="0.4", linewidth=0.7)
    axis.axvline(TAIL_MULTIPLIER, color="#D55E00", linewidth=0.7, linestyle=":")
    axis.set_xscale("log")
    axis.set_xlabel("Paired mean-loss ratio vs predictive MPC")
    axis.set_ylabel("Cell ECDF")
    axis.grid(color="0.9", linewidth=0.5)
    axis.legend(ncol=2, fontsize=6, frameon=False)
    fig.tight_layout()
    paths = _save_figure(fig, output)
    plt.close(fig)
    return paths


def _plot_paired_metric(
    paired: pd.DataFrame,
    column: str,
    ylabel: str,
    output: Path,
    *,
    reference: float,
) -> tuple[Path, Path]:
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(7.1, 3.4))
    positions = np.arange(len(new_methods()))
    data = [paired.loc[paired["method"] == method, column].to_numpy(float) for method in new_methods()]
    axis.boxplot(data, positions=positions, widths=0.62, showfliers=True)
    axis.axhline(reference, color="0.4", linewidth=0.7)
    axis.set_xticks(positions, new_methods(), rotation=30, ha="right", fontsize=6)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="0.9", linewidth=0.5)
    fig.tight_layout()
    paths = _save_figure(fig, output)
    plt.close(fig)
    return paths


def run_report(config: ExploreConfig) -> Path:
    evaluation = _load_stage(config, "evaluate")
    summary = pd.read_csv(str(evaluation["summary"]))
    per_budget, overall = policy_metrics(summary)
    eligibility = eligibility_table(per_budget, overall)
    paired = _paired_long(summary)
    composition, disagreement = _decision_tables(config)
    artifact_dir = config.output / "report/artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "trace_cells.csv": summary,
        "paired_cells.csv": paired,
        "per_budget_metrics.csv": per_budget,
        "overall_metrics.csv": overall,
        "eligibility.csv": eligibility,
        "reducer_composition.csv": composition,
        "ensemble_disagreement.csv": disagreement,
    }
    artifacts: list[Path] = []
    for name, frame in tables.items():
        path = artifact_dir / name
        write_csv_atomic(frame, path)
        artifacts.append(path)
    artifacts.extend(_plot_loss_ecdf(paired, artifact_dir / "loss_ratio_ecdf"))
    artifacts.extend(_plot_paired_metric(
        paired,
        "fpr_difference_vs_g15",
        "Paired FPR difference vs G15",
        artifact_dir / "paired_fpr",
        reference=0.0,
    ))
    artifacts.extend(_plot_paired_metric(
        paired,
        "throughput_retention_vs_g15",
        "Paired throughput retention vs G15",
        artifact_dir / "throughput_retention",
        reference=0.5,
    ))
    eligible = eligibility[eligibility["eligible"].astype(bool)]
    selected = G15_METHOD if eligible.empty else str(eligible.iloc[0]["method"])
    decision = {
        "selected_method": selected,
        "eligible_challenger_count": len(eligible),
        "fallback_if_none": G15_METHOD,
        "confirmation_launched": False,
        "fixed_traces_launched": False,
        "ranking_rule": (
            "worst predictive loss ratio, P95 predictive ratio, maximum budget median "
            "regression vs G15, descending throughput, single model on exact ties"
        ),
    }
    decision_path = artifact_dir / "decision.json"
    write_json_atomic(decision, decision_path)
    artifacts.append(decision_path)
    hashes = {str(path.relative_to(artifact_dir)): _raw_sha256(path) for path in artifacts}
    hash_path = artifact_dir / "artifact_hashes.json"
    write_json_atomic(hashes, hash_path)
    return _write_stage(config, "report", {
        **decision,
        "artifact_directory": str(artifact_dir),
        "artifact_count": len(artifacts) + 1,
        "artifact_hashes": str(hash_path),
    })


def run_validate(config: ExploreConfig) -> Path:
    report = _load_stage(config, "report")
    for stage in STAGES[:-1]:
        _load_stage(config, stage)
    preflight = _load_stage(config, "preflight")
    current_snapshot = _source_snapshot(config)
    if current_snapshot != preflight["source_snapshot"]:
        raise ValueError("v3/v4 parent artifacts changed during tail-policy exploration")
    prepare = _load_stage(config, "prepare")
    train = _load_stage(config, "train")
    evaluate = _load_stage(config, "evaluate")
    expected_plans = len(MIXTURE_SHARES) * len(OPTIMIZER_SEEDS) * len(config.budgets)
    if int(prepare["plan_count"]) != expected_plans or int(train["model_count"]) != expected_plans:
        raise ValueError("resampling/model matrix count differs")
    if int(evaluate["new_cell_count"]) != config.expected_new_cells:
        raise ValueError("new evaluation cell count differs")
    if int(evaluate["reference_cell_count"]) != config.expected_reference_cells:
        raise ValueError("reference cell count differs")
    if int(evaluate["joined_cell_count"]) != config.expected_report_cells:
        raise ValueError("report cell count differs")
    hash_path = Path(str(report["artifact_hashes"]))
    hashes = _load_json(hash_path)
    artifact_root = hash_path.parent
    for relative, expected in hashes.items():
        path = artifact_root / relative
        if _raw_sha256(path) != expected:
            raise ValueError(f"report artifact hash differs: {path}")
    reserved = set(config.confirmation_seeds)
    for path in config.output.rglob("seed-*"):
        if path.is_dir():
            try:
                seed = int(path.name.split("-", 1)[1])
            except ValueError:
                continue
            if seed in reserved:
                raise ValueError(f"confirmation seed was touched: {path}")
    return _write_stage(config, "validate", {
        "source_snapshot_unchanged": True,
        "new_cell_count": config.expected_new_cells,
        "reference_cell_count": config.expected_reference_cells,
        "report_cell_count": config.expected_report_cells,
        "confirmation_seeds_untouched": True,
        "fixed_traces_untouched": True,
        "selected_method": report["selected_method"],
        "artifact_hash_count": len(hashes),
    })


def run_stage(config: ExploreConfig, stage: str) -> Path:
    path = _stage_path(config, stage)
    if path.is_file():
        _load_stage(config, stage)
        print(f"skip completed tail-policy stage: {stage}", flush=True)
        return path
    functions = {
        "preflight": run_preflight,
        "prepare": run_prepare,
        "train": run_train,
        "pilot": run_pilot,
        "evaluate": run_evaluate,
        "report": run_report,
        "validate": run_validate,
    }
    print(f"start tail-policy stage: {stage}", flush=True)
    result = functions[stage](config)
    print(f"complete tail-policy stage: {stage}", flush=True)
    return result


def run_all(config: ExploreConfig) -> Path:
    print(json.dumps(config.identity, indent=2), flush=True)
    for stage in STAGES:
        run_stage(config, stage)
    return _stage_path(config, "validate")


def status(config: ExploreConfig) -> dict[str, object]:
    stages: dict[str, str] = {}
    for stage in STAGES:
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
        "expected_new_cells": config.expected_new_cells,
        "expected_reference_cells": config.expected_reference_cells,
        "expected_report_cells": config.expected_report_cells,
        "stages": stages,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(*STAGES, "run", "status"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--epochs", type=int, default=int(TRAINING["epochs"]))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        config = smoke_config(args.output, workers=args.workers)
        if args.epochs != int(TRAINING["epochs"]):
            config = replace(config, epochs=args.epochs)
    else:
        config = ExploreConfig(output=args.output, workers=args.workers, epochs=args.epochs)
    if args.command == "status":
        print(json.dumps(status(config), indent=2))
        return 0
    path = run_all(config) if args.command == "run" else run_stage(config, args.command)
    print(f"PRP tail-policy exploratory stage complete: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
