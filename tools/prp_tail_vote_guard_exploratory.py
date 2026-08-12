#!/usr/bin/env python3
"""Disposable plurality-vote and native-loss-guard PRP tail experiment.

All writes are confined to ``results/prp-tail-vote-guard-exploratory-v1``.
The completed 5%-DAgger rank-3 experiment is an immutable, hash-verified
parent.  This driver trains only optimizer seeds 3042 and 4042 and never
evaluates confirmation seeds or fixed robot-arm traces.
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

from pzr.artifact_io import write_csv_atomic, write_json_atomic
from pzr.learning.diagnostics import validation_metrics
from pzr.learning.provenance import model_sha256, payload_sha256, pzr_source_sha256, sha256_files
from pzr.learning.ranker import ReducerPolicy, train_reducer_policy
from pzr.rtlola.actions import RtlolaActionCatalog, default_action_catalog
from pzr.rtlola.benchmark import RtlolaBenchmarkConfig, run_event_trace_benchmark
from pzr.rtlola.binding import BINDING_BUILD_PROFILE, BINDING_REVISION, INTERPRETER_REVISION
from pzr.rtlola.engine import RtlolaBindingError, RtlolaEngine, RtlolaEvent, RtlolaStateRef
from pzr.rtlola.features import RTL_RANKING_FEATURE_SCHEMA, extract_ranking_features
from pzr.rtlola.reference import load_or_compute_reference
from pzr.rtlola.scenarios import scenario_by_name
from pzr.rtlola.search import RtlolaNoFeasibleAction, RtlolaSearchResult

import prp_tail_policy_exploratory as parent
from prp_v4_exploratory import TraceRecord


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/prp-tail-vote-guard-exploratory-v1"
PARENT_ROOT = ROOT / "results/prp-tail-policy-exploratory-v1-rerun"
SCHEMA = "pzr.prp-tail-vote-guard-exploratory.v1"
CELL_SCHEMA = "pzr.prp-tail-vote-guard-cell.v1"
CANDIDATES = parent.CANDIDATES
BUDGETS = parent.BUDGETS
REUSED_OPTIMIZER_SEEDS = parent.OPTIMIZER_SEEDS
NEW_OPTIMIZER_SEEDS = (3042, 4042)
ALL_OPTIMIZER_SEEDS = (*REUSED_OPTIMIZER_SEEDS, *NEW_OPTIMIZER_SEEDS)
SELECTION_SEEDS = parent.SELECTION_SEEDS
CONFIRMATION_SEEDS = parent.CONFIRMATION_SEEDS
EVENT_COUNT = parent.EVENT_COUNT
WORKERS = 10
MIXTURE_SHARE = 0.05
G15_METHOD = parent.G15_METHOD
PREDICTIVE_METHOD = parent.PREDICTIVE_METHOD
RANK3_METHOD = "dagger05_ensemble3"
METHODS = (
    "dagger05_vote3",
    "dagger05_vote5",
    "dagger05_vote3_guarded",
    "dagger05_vote5_guarded",
)
PILOT_CELLS = ((321, 40), (326, 80), (321, 80), (326, 120))
TRAINING = dict(parent.TRAINING)
STAGES = ("preflight", "prepare", "train", "pilot", "evaluate", "report", "validate")
LOSS_TOLERANCE_ABS = 1e-15
LOSS_TOLERANCE_REL = 1e-9


@dataclass(frozen=True)
class ExploreConfig:
    output: Path = DEFAULT_OUTPUT
    budgets: tuple[int, ...] = BUDGETS
    clean_train_seeds: tuple[int, ...] = parent.CLEAN_TRAIN_SEEDS
    clean_validation_seeds: tuple[int, ...] = parent.VALIDATION_SEEDS
    dagger_train_seeds: tuple[int, ...] = parent.DAGGER_TRAIN_SEEDS
    dagger_validation_seeds: tuple[int, ...] = parent.DAGGER_VALIDATION_SEEDS
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
            "epochs": self.epochs,
            "smoke": self.smoke,
            "mixture_share": MIXTURE_SHARE,
            "reused_optimizer_seeds": list(REUSED_OPTIMIZER_SEEDS),
            "new_optimizer_seeds": list(NEW_OPTIMIZER_SEEDS),
            "methods": list(METHODS),
            "training": {**TRAINING, "epochs": self.epochs},
            "loss_tolerance_absolute": LOSS_TOLERANCE_ABS,
            "loss_tolerance_relative": LOSS_TOLERANCE_REL,
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
        return len(METHODS) * len(self.selection_seeds) * len(self.budgets)

    @property
    def expected_rank3_cells(self) -> int:
        return len(self.selection_seeds) * len(self.budgets)

    @property
    def expected_reference_cells(self) -> int:
        return 2 * len(self.selection_seeds) * len(self.budgets)

    @property
    def expected_report_cells(self) -> int:
        return self.expected_new_cells + self.expected_rank3_cells + self.expected_reference_cells


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
class VoteOrder:
    order: NDArray[np.int64]
    votes: NDArray[np.int64]
    mean_ranks: NDArray[np.float64]
    mean_standardized_scores: NDArray[np.float64]
    member_top1: NDArray[np.int64]
    winner_margin: int
    tie_resolution: str


@dataclass(frozen=True)
class MethodSpec:
    name: str
    member_paths: tuple[Path, ...]
    guarded: bool


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


def _resolve(path: str | Path) -> Path:
    result = Path(path)
    return result if result.is_absolute() else ROOT / result


def tool_sha256() -> str:
    paths = (
        Path(__file__).resolve(),
        ROOT / "tools/run_prp_tail_vote_guard_exploratory.sh",
        ROOT / "tools/prp_tail_policy_exploratory.py",
    )
    return sha256_files(tuple(path for path in paths if path.is_file()), relative_to=ROOT)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


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


def _load_stage(config: ExploreConfig, stage: str) -> dict[str, object]:
    path = _stage_path(config, stage)
    manifest = _load_json(path)
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"unsupported vote-guard manifest: {path}")
    if manifest.get("experiment_fingerprint") != config.fingerprint:
        raise ValueError(f"stale vote-guard manifest: {path}")
    return manifest


def plurality_order(member_scores: NDArray[np.floating]) -> VoteOrder:
    """Order four reducers by votes, mean rank, standardized score, and catalog."""
    scores = np.asarray(member_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(CANDIDATES) or scores.shape[0] not in (3, 5):
        raise ValueError("plurality scores must have shape (3 or 5, four candidates)")
    if not np.all(np.isfinite(scores)):
        raise ValueError("plurality scores contain non-finite values")
    ranks = np.empty_like(scores)
    for member, row in enumerate(scores):
        member_order = np.argsort(row, kind="stable")
        ranks[member, member_order] = np.arange(len(CANDIDATES), dtype=np.float64)
    member_top1 = np.argmin(scores, axis=1).astype(np.int64)
    votes = np.bincount(member_top1, minlength=len(CANDIDATES)).astype(np.int64)
    mean_ranks = ranks.mean(axis=0)
    centered = scores - scores.mean(axis=1, keepdims=True)
    scale = scores.std(axis=1, keepdims=True)
    standardized = np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 1e-12)
    mean_standardized = standardized.mean(axis=0)
    catalog = np.arange(len(CANDIDATES))
    order = np.lexsort((catalog, mean_standardized, mean_ranks, -votes)).astype(np.int64)
    sorted_votes = np.sort(votes)[::-1]
    margin = int(sorted_votes[0] - sorted_votes[1])
    top = np.flatnonzero(votes == votes[order[0]])
    if len(top) == 1:
        tie_resolution = "vote_count"
    else:
        best_rank = np.min(mean_ranks[top])
        rank_tied = top[mean_ranks[top] == best_rank]
        if len(rank_tied) == 1:
            tie_resolution = "mean_ordinal_rank"
        else:
            best_score = np.min(mean_standardized[rank_tied])
            score_tied = rank_tied[mean_standardized[rank_tied] == best_score]
            tie_resolution = "mean_standardized_score" if len(score_tied) == 1 else "catalog_order"
    return VoteOrder(
        order=order,
        votes=votes,
        mean_ranks=mean_ranks,
        mean_standardized_scores=mean_standardized,
        member_top1=member_top1,
        winner_margin=margin,
        tie_resolution=tie_resolution,
    )


def strong_disagreement(votes: NDArray[np.integer]) -> bool:
    values = np.asarray(votes, dtype=np.int64)
    if values.shape != (len(CANDIDATES),) or int(values.sum()) not in (3, 5) or np.any(values < 0):
        raise ValueError("vote counts must describe a three- or five-member ensemble")
    ordered = np.sort(values)[::-1]
    return int(ordered[0] - ordered[1]) <= 1


def tolerance_aware_minimum(
    costs: Mapping[int, float],
    plurality: Sequence[int],
) -> int:
    """Select the native-loss minimum; tolerance ties follow plurality order."""
    if not costs:
        raise ValueError("native-loss selection requires a feasible contender")
    if any(index not in plurality for index in costs):
        raise ValueError("native-loss contender is absent from plurality order")
    values = np.asarray(list(costs.values()), dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("native losses must be finite and non-negative")
    minimum = float(values.min())
    tied = {
        index
        for index, cost in costs.items()
        if abs(float(cost) - minimum)
        <= max(LOSS_TOLERANCE_ABS, LOSS_TOLERANCE_REL * max(abs(float(cost)), abs(minimum)))
    }
    return next(int(index) for index in plurality if int(index) in tied)


def _deterministic_checks(config: ExploreConfig) -> dict[str, object]:
    scores = np.asarray([
        (0.0, 1.0, 2.0, 3.0),
        (1.0, 0.0, 2.0, 3.0),
        (0.0, 2.0, 1.0, 3.0),
    ])
    vote = plurality_order(scores)
    if tuple(vote.order[:3]) != (0, 1, 2) or tuple(vote.votes) != (2, 1, 0, 0):
        raise AssertionError("deterministic plurality ordering check failed")
    patterns = {
        (2, 1, 0, 0): True,
        (3, 0, 0, 0): False,
        (3, 2, 0, 0): True,
        (3, 1, 1, 0): False,
        (4, 1, 0, 0): False,
    }
    if any(strong_disagreement(np.asarray(pattern)) != expected for pattern, expected in patterns.items()):
        raise AssertionError("guard activation check failed")
    selected = tolerance_aware_minimum({0: 1.0 + 5e-10, 1: 1.0}, (0, 1, 2, 3))
    if selected != 0:
        raise AssertionError("tolerance-aware plurality tie check failed")

    scenario = scenario_by_name("robot_arm")
    events = parent._selection_traces(config)[0].events
    engine = RtlolaEngine(
        scenario.spec,
        event_arity=scenario.event_arity,
        expected_verdict_keys=scenario.expected_verdict_keys,
    )
    catalog = default_action_catalog(CANDIDATES)
    prefix = min(20, len(events) - 1)
    for step, event in enumerate(events[:prefix], start=1):
        engine.live_step(event, catalog.no_op, budget=500, step=step)
    state = engine.snapshot(step=prefix, time=events[prefix - 1].time)
    exact = engine.branch_step(state, events[prefix], catalog.no_op, budget=40)
    candidate = engine.branch_step(state, events[prefix], catalog.by_name["girard"], budget=40)
    before = engine.matrices(candidate.state)
    loss = engine.approx_loss(exact.state, candidate.state)
    after = engine.matrices(candidate.state)
    if not np.isfinite(loss):
        raise AssertionError("native-loss check returned a non-finite value")
    for left, right in zip(before, after, strict=True):
        np.testing.assert_array_equal(left, right)
    return {
        "plurality_ordering": True,
        "vote_margin_guard_activation": True,
        "native_loss_tolerance_tie": True,
        "planner_state_restoration": True,
        "native_probe_loss": loss,
    }


def _parent_manifests() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    prepare = _load_json(PARENT_ROOT / "prepare/manifest.json")
    train = _load_json(PARENT_ROOT / "train/manifest.json")
    evaluate = _load_json(PARENT_ROOT / "evaluate/manifest.json")
    for stage, manifest in (("prepare", prepare), ("train", train), ("evaluate", evaluate)):
        if manifest.get("schema") != parent.SCHEMA or manifest.get("status") != "completed":
            raise ValueError(f"completed parent {stage} manifest has an unexpected identity")
    return prepare, train, evaluate


def _parent_plan(seed: int, budget: int) -> dict[str, object]:
    prepare, _, _ = _parent_manifests()
    record = dict(prepare["plans"][f"dagger05:{seed}:{budget}"])
    path = _resolve(str(record["path"]))
    if sha256_files((path,)) != record["sha256"]:
        raise ValueError(f"parent 5% resampling plan hash differs: seed={seed}, budget={budget}")
    return {**record, "path": str(path)}


def _parent_model(seed: int, budget: int) -> dict[str, object]:
    _, train, _ = _parent_manifests()
    record = dict(train["models"][f"dagger05:{seed}:{budget}"])
    path = _resolve(str(record["path"]))
    if model_sha256(path) != record["sha256"]:
        raise ValueError(f"parent 5% model hash differs: seed={seed}, budget={budget}")
    if int(record["optimizer_seed"]) != seed or float(record["mixture_share"]) != MIXTURE_SHARE:
        raise ValueError("parent 5% model identity differs")
    return {**record, "path": str(path)}


def _parent_rank3_snapshot(config: ExploreConfig) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for seed in config.selection_seeds:
        for budget in config.budgets:
            directory = PARENT_ROOT / (
                f"evaluate/cells/random_waypoint/seed-{seed}/budget-{budget}/{RANK3_METHOD}"
            )
            manifest_path = directory / "manifest.json"
            summary_path = directory / "summary.csv"
            decisions_path = directory / "decisions.csv"
            manifest = _load_json(manifest_path)
            identity = dict(manifest["identity"])
            expected_members = [
                str(_parent_model(member_seed, budget)["sha256"])
                for member_seed in REUSED_OPTIMIZER_SEEDS
            ]
            if (
                identity.get("method") != RANK3_METHOD
                or identity.get("member_model_sha256") != expected_members
                or manifest.get("status") != "completed"
            ):
                raise ValueError(f"parent rank-3 cell identity differs: {directory}")
            for name, path in (("manifest", manifest_path), ("summary", summary_path), ("decisions", decisions_path)):
                snapshot[f"rank3:{seed}:{budget}:{name}"] = _raw_sha256(path)
    return snapshot


def _source_snapshot(config: ExploreConfig) -> dict[str, str]:
    snapshot = dict(parent._source_snapshot(config))
    for stage in ("prepare", "train", "evaluate", "validate"):
        path = PARENT_ROOT / stage / "manifest.json"
        snapshot[f"parent:{stage}:manifest"] = _raw_sha256(path)
    for seed in REUSED_OPTIMIZER_SEEDS:
        for budget in config.budgets:
            plan = _parent_plan(seed, budget)
            model = _parent_model(seed, budget)
            snapshot[f"parent:plan:{seed}:{budget}"] = str(plan["sha256"])
            snapshot[f"parent:model:{seed}:{budget}"] = str(model["sha256"])
            training_manifest = Path(str(model["path"])) / "exploratory_training.json"
            snapshot[f"parent:training_manifest:{seed}:{budget}"] = _raw_sha256(training_manifest)
    for seed in config.selection_seeds:
        for budget in config.budgets:
            for method in (G15_METHOD, PREDICTIVE_METHOD):
                directory = parent.V4_ROOT / (
                    f"feature-screen/cells/random_waypoint/seed-{seed}/budget-{budget}/{method}"
                )
                for path in sorted(directory.iterdir()):
                    if path.is_file():
                        snapshot[f"reference_cell:{seed}:{budget}:{method}:{path.name}"] = _raw_sha256(path)
    snapshot.update(_parent_rank3_snapshot(config))
    return snapshot


def run_preflight(config: ExploreConfig) -> Path:
    if BINDING_BUILD_PROFILE != "release":
        raise ValueError("vote-guard exploration requires a release binding")
    snapshot = _source_snapshot(config)
    checks = _deterministic_checks(config)
    return _write_stage(config, "preflight", {
        "scientific_role": "final disposable PRP vote-and-guard tail-remedy selection",
        "source_snapshot": snapshot,
        "source_snapshot_sha256": payload_sha256(snapshot),
        "deterministic_checks": checks,
        "confirmation_automatic": False,
        "fixed_traces_automatic": False,
    })


def _new_plan_path(config: ExploreConfig, seed: int, budget: int) -> Path:
    return config.output / f"prepare/resampling/dagger05/opt-{seed}/budget-{budget}.csv"


def run_prepare(config: ExploreConfig) -> Path:
    _load_stage(config, "preflight")
    clean, clean_metadata = parent._load_clean(config)
    records: dict[str, dict[str, object]] = {}
    reused: dict[str, dict[str, object]] = {}
    for budget in config.budgets:
        clean_subset, _ = parent._clean_budget(clean, clean_metadata, budget)
        _, dagger_metadata = parent._load_dagger_budget(config, budget)
        for seed in REUSED_OPTIMIZER_SEEDS:
            reused[f"dagger05:{seed}:{budget}"] = _parent_plan(seed, budget)
        for seed in NEW_OPTIMIZER_SEEDS:
            plans = []
            splits: dict[str, object] = {}
            for split in ("train", "validation"):
                clean_count = len(clean_subset.indices_for_split(split))
                plan = parent.regret_resample_plan(
                    clean_count,
                    dagger_metadata,
                    target_share=MIXTURE_SHARE,
                    optimizer_seed=seed,
                    budget=budget,
                    split=split,
                )
                plans.append(plan)
                splits[split] = {
                    "clean_count": clean_count,
                    "learner_count": len(plan),
                    "realized_learner_share": len(plan) / (clean_count + len(plan)),
                    "learner_seeds": sorted(plan["source_seed"].astype(int).unique().tolist()),
                }
            combined = pd.concat(plans, ignore_index=True)
            path = _new_plan_path(config, seed, budget)
            write_csv_atomic(combined, path)
            records[f"dagger05:{seed}:{budget}"] = {
                "path": str(path),
                "sha256": sha256_files((path,)),
                "rows": len(combined),
                "splits": splits,
            }
    expected_new = len(NEW_OPTIMIZER_SEEDS) * len(config.budgets)
    expected_reused = len(REUSED_OPTIMIZER_SEEDS) * len(config.budgets)
    if len(records) != expected_new or len(reused) != expected_reused:
        raise AssertionError("5% resampling-plan matrix is incomplete")
    return _write_stage(config, "prepare", {
        "new_plan_count": len(records),
        "reused_plan_count": len(reused),
        "new_plans": records,
        "reused_plans": reused,
        "sampling_rule": "unchanged_parent_trace_balanced_regret_weighted_resampling",
        "complete_candidate_blocks_retained": True,
    })


def _new_plan(config: ExploreConfig, seed: int, budget: int) -> dict[str, object]:
    record = dict(_load_stage(config, "prepare")["new_plans"][f"dagger05:{seed}:{budget}"])
    path = _resolve(str(record["path"]))
    if sha256_files((path,)) != record["sha256"]:
        raise ValueError(f"new resampling plan hash differs: seed={seed}, budget={budget}")
    return {**record, "path": str(path)}


def run_train(config: ExploreConfig) -> Path:
    _load_stage(config, "prepare")
    clean, clean_metadata = parent._load_clean(config)
    new_records: dict[str, dict[str, object]] = {}
    reused_records = {
        f"dagger05:{seed}:{budget}": _parent_model(seed, budget)
        for seed in REUSED_OPTIMIZER_SEEDS
        for budget in config.budgets
    }
    for budget in config.budgets:
        clean_subset, clean_frame = parent._clean_budget(clean, clean_metadata, budget)
        dagger, dagger_metadata = parent._load_dagger_budget(config, budget)
        for seed in NEW_OPTIMIZER_SEEDS:
            plan_record = _new_plan(config, seed, budget)
            plan = pd.read_csv(str(plan_record["path"]))
            aggregate, metadata = parent.build_mixture_dataset(
                clean_subset, clean_frame, dagger, dagger_metadata, plan,
            )
            metadata = metadata.copy()
            metadata["dataset_label"] = metadata["training_source"]
            output = config.output / f"train/dagger05/opt-{seed}/budget-{budget}"
            identity = {
                "experiment_fingerprint": config.fingerprint,
                "mixture_share": MIXTURE_SHARE,
                "optimizer_seed": seed,
                "budget": budget,
                "plan_sha256": plan_record["sha256"],
                "sample_ids_sha256": payload_sha256({"sample_ids": list(aggregate.sample_ids)}),
                "training": {**TRAINING, "epochs": config.epochs},
            }
            artifact = output / "exploratory_training.json"
            key = f"dagger05:{seed}:{budget}"
            if artifact.is_file():
                existing = _load_json(artifact)
                if existing.get("identity") != identity or model_sha256(output) != existing.get("model_sha256"):
                    raise ValueError(f"stale vote-guard model: {output}")
                new_records[key] = dict(existing["record"])
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
                seed=seed,
            )
            policy.save(output)
            write_csv_atomic(validation_metrics(policy, aggregate, metadata), output / "validation_metrics.csv")
            record = {
                "mixture_share": MIXTURE_SHARE,
                "optimizer_seed": seed,
                "budget": budget,
                "path": str(output),
                "sha256": model_sha256(output),
                "feature_schema": asdict(RTL_RANKING_FEATURE_SCHEMA),
                "candidate_names": list(CANDIDATES),
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
            new_records[key] = record
    if len(new_records) != len(NEW_OPTIMIZER_SEEDS) * len(config.budgets):
        raise AssertionError("new model matrix is incomplete")
    return _write_stage(config, "train", {
        "new_model_count": len(new_records),
        "reused_model_count": len(reused_records),
        "new_models": new_records,
        "reused_models": reused_records,
    })


def _new_model(config: ExploreConfig, seed: int, budget: int) -> dict[str, object]:
    record = dict(_load_stage(config, "train")["new_models"][f"dagger05:{seed}:{budget}"])
    path = _resolve(str(record["path"]))
    if model_sha256(path) != record["sha256"]:
        raise ValueError(f"new model hash differs: seed={seed}, budget={budget}")
    return {**record, "path": str(path)}


def _member_paths(config: ExploreConfig, budget: int) -> tuple[Path, ...]:
    reused = tuple(Path(str(_parent_model(seed, budget)["path"])) for seed in REUSED_OPTIMIZER_SEEDS)
    new = tuple(Path(str(_new_model(config, seed, budget)["path"])) for seed in NEW_OPTIMIZER_SEEDS)
    return (*reused, *new)


def _method_specs(config: ExploreConfig, budget: int) -> tuple[MethodSpec, ...]:
    members = _member_paths(config, budget)
    return (
        MethodSpec(METHODS[0], members[:3], False),
        MethodSpec(METHODS[1], members, False),
        MethodSpec(METHODS[2], members[:3], True),
        MethodSpec(METHODS[3], members, True),
    )


class VoteGuardPolicy:
    """Top-1 plurality inference with an optional current-event native-loss guard."""

    def __init__(
        self,
        policies: Sequence[ReducerPolicy],
        events: Sequence[RtlolaEvent],
        *,
        guarded: bool,
    ) -> None:
        self.policies = tuple(policies)
        if len(self.policies) not in (3, 5):
            raise ValueError("vote policy requires exactly three or five members")
        if any(policy.candidate_names != CANDIDATES for policy in self.policies):
            raise ValueError("vote-policy candidate catalogs differ")
        if any(policy.feature_schema != RTL_RANKING_FEATURE_SCHEMA for policy in self.policies):
            raise ValueError("vote policy requires Geometry15 models")
        self.events = tuple(events)
        self.guarded = bool(guarded)
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
            raise ValueError("vote-policy event history is not aligned")
        step_index = self.step
        self.step += 1
        metrics = engine.metrics(state)
        if metrics.dynamic_generator_count <= budget:
            branch = engine.branch_step(state, event, self.catalog.no_op, budget)
            self.diagnostics.append(self._diagnostic(
                step_index, budget, over_bound=False, action=self.catalog.no_op.name,
                fallback=False, branch_count=1,
            ))
            return RtlolaSearchResult(
                first_action=self.catalog.no_op,
                first_action_budget=budget,
                first_step=branch,
                predicted_cost=0.0,
                predicted_sequence=(self.catalog.no_op.name,),
                evaluated_leaves=1,
                pruned_branches=0,
                mpc_variant="direct_plurality_vote",
                root_strategy="current_event_vote_guard" if self.guarded else "top1_plurality_direct",
            )

        features = extract_ranking_features(engine, state, budget)
        scores = np.asarray([
            np.asarray(policy.predict_scores(features), dtype=np.float64)
            for policy in self.policies
        ])
        vote = plurality_order(scores)
        winner = int(vote.order[0])
        invoke_guard = self.guarded and strong_disagreement(vote.votes)
        guard_started = perf_counter()
        branches: dict[int, object] = {}
        native_costs: dict[int, float] = {}
        failures = 0
        branch_count = 0
        contenders = [winner]
        if invoke_guard:
            reference = engine.branch_step(state, event, self.catalog.no_op, budget)
            branch_count += 1
            contenders = [int(index) for index in vote.order if vote.votes[int(index)] > 0]
            for candidate in contenders:
                action = self.catalog.by_name[CANDIDATES[candidate]]
                if action.explicit_budget and budget < metrics.dimension:
                    failures += 1
                    continue
                branch_count += 1
                try:
                    candidate_branch = engine.branch_step(state, event, action, budget)
                except RtlolaBindingError:
                    failures += 1
                    continue
                branches[candidate] = candidate_branch
                native_costs[candidate] = engine.approx_loss(reference.state, candidate_branch.state)
            selected = tolerance_aware_minimum(native_costs, vote.order) if native_costs else None
        else:
            action = self.catalog.by_name[CANDIDATES[winner]]
            selected = winner
            if action.explicit_budget and budget < metrics.dimension:
                failures += 1
                selected = None
            else:
                branch_count += 1
                try:
                    branches[winner] = engine.branch_step(state, event, action, budget)
                except RtlolaBindingError:
                    failures += 1
                    selected = None

        fallback_used = selected is None
        if fallback_used:
            branch_count += 1
            try:
                branch = engine.branch_step(state, event, self.catalog.fallback, budget)
            except RtlolaBindingError as exc:
                raise RtlolaNoFeasibleAction("all voted reducers and interval fallback were infeasible") from exc
            action = self.catalog.fallback
            predicted_cost = float("nan")
        else:
            action = self.catalog.by_name[CANDIDATES[selected]]
            branch = branches[selected]
            predicted_cost = native_costs.get(selected, float(vote.mean_ranks[selected]))
        guard_ms = (perf_counter() - guard_started) * 1000.0 if invoke_guard else 0.0
        self.diagnostics.append(self._diagnostic(
            step_index,
            budget,
            over_bound=True,
            action=action.name,
            fallback=fallback_used,
            branch_count=branch_count,
            vote=vote,
            guard_invoked=invoke_guard,
            contender_count=len(contenders) if invoke_guard else 1,
            selected=selected,
            unguarded_winner=winner,
            native_costs=native_costs,
            infeasible_count=failures,
            guard_ms=guard_ms,
        ))
        return RtlolaSearchResult(
            first_action=action,
            first_action_budget=budget,
            first_step=branch,
            predicted_cost=float(predicted_cost),
            predicted_sequence=(action.name,),
            evaluated_leaves=branch_count,
            pruned_branches=0,
            fallback_used=fallback_used,
            reducer_failure_count=failures,
            infeasible_candidate_count=failures,
            mpc_variant="guarded_plurality_vote" if self.guarded else "direct_plurality_vote",
            root_strategy="current_event_vote_guard" if self.guarded else "top1_plurality_direct",
        )

    def _diagnostic(
        self,
        step: int,
        budget: int,
        *,
        over_bound: bool,
        action: str,
        fallback: bool,
        branch_count: int,
        vote: VoteOrder | None = None,
        guard_invoked: bool = False,
        contender_count: int = 0,
        selected: int | None = None,
        unguarded_winner: int | None = None,
        native_costs: Mapping[int, float] | None = None,
        infeasible_count: int = 0,
        guard_ms: float = 0.0,
    ) -> dict[str, object]:
        native_costs = {} if native_costs is None else native_costs
        record: dict[str, object] = {
            "step": step,
            "budget": budget,
            "over_bound": over_bound,
            "selected_action": action,
            "evaluated_leaves": branch_count,
            "branch_count": branch_count,
            "fallback_used": fallback,
            "member_count": len(self.policies),
            "guard_enabled": self.guarded,
            "guard_invoked": guard_invoked,
            "contender_count": contender_count,
            "guard_override": bool(guard_invoked and selected is not None and selected != unguarded_winner),
            "infeasible_candidate_count": infeasible_count,
            "guard_added_decision_time_ms": guard_ms,
            "member_top1_disagreement": False,
            "distinct_member_top1": 0,
            "winner_vote_count": 0,
            "runner_up_vote_count": 0,
            "winner_margin": 0,
            "vote_pattern": "under_bound",
            "tie_resolution": "under_bound",
            "plurality_order": "[]",
            "member_top1": "[]",
            "native_costs": json.dumps({CANDIDATES[index]: cost for index, cost in native_costs.items()}),
        }
        for name in CANDIDATES:
            record[f"votes_{name}"] = 0
            record[f"mean_rank_{name}"] = np.nan
            record[f"mean_standardized_score_{name}"] = np.nan
            record[f"native_cost_{name}"] = native_costs.get(CANDIDATES.index(name), np.nan)
        if vote is None:
            return record
        sorted_votes = np.sort(vote.votes)[::-1]
        nonzero = sorted((int(value) for value in vote.votes if value > 0), reverse=True)
        record.update({
            "member_top1_disagreement": len(set(vote.member_top1.tolist())) > 1,
            "distinct_member_top1": len(set(vote.member_top1.tolist())),
            "winner_vote_count": int(sorted_votes[0]),
            "runner_up_vote_count": int(sorted_votes[1]),
            "winner_margin": vote.winner_margin,
            "vote_pattern": "-".join(map(str, nonzero)),
            "tie_resolution": vote.tie_resolution,
            "plurality_order": json.dumps([CANDIDATES[int(index)] for index in vote.order]),
            "member_top1": json.dumps([CANDIDATES[int(index)] for index in vote.member_top1]),
        })
        for index, name in enumerate(CANDIDATES):
            record[f"votes_{name}"] = int(vote.votes[index])
            record[f"mean_rank_{name}"] = float(vote.mean_ranks[index])
            record[f"mean_standardized_score_{name}"] = float(vote.mean_standardized_scores[index])
        return record


def _selection_traces(config: ExploreConfig) -> tuple[TraceRecord, ...]:
    return parent._selection_traces(config)


def _reference_path(config: ExploreConfig, trace: TraceRecord) -> Path:
    return parent._reference_path(config, trace)


def _jobs(
    config: ExploreConfig,
    traces: Sequence[TraceRecord],
    budgets: Sequence[int] | None = None,
    *,
    pairs: Sequence[tuple[int, int]] | None = None,
) -> list[EvaluationJob]:
    jobs = []
    allowed_pairs = None if pairs is None else set(pairs)
    selected_budgets = config.budgets if budgets is None else tuple(budgets)
    for trace in traces:
        reference = _reference_path(config, trace)
        for budget in selected_budgets:
            if allowed_pairs is not None and (trace.seed, budget) not in allowed_pairs:
                continue
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
        "guarded": job.spec.guarded,
        "reference_sha256": sha256_files((job.reference_path,)),
    }
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        if manifest.get("identity") != identity:
            raise ValueError(f"stale vote-guard cell: {job.directory}")
        return pd.read_csv(summary_path).iloc[0].to_dict()
    job.directory.mkdir(parents=True, exist_ok=True)
    policies = tuple(ReducerPolicy.load(path) for path in job.spec.member_paths)
    policy = VoteGuardPolicy(policies, job.trace.events, guarded=job.spec.guarded)
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
        "ensemble": True,
        "guarded": job.spec.guarded,
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


def _pilot_pairs(config: ExploreConfig) -> tuple[tuple[int, int], ...]:
    pairs = tuple(
        pair for pair in PILOT_CELLS
        if pair[0] in config.selection_seeds and pair[1] in config.budgets
    )
    return pairs or ((config.selection_seeds[0], config.budgets[0]),)


def run_pilot(config: ExploreConfig) -> Path:
    _load_stage(config, "train")
    pairs = _pilot_pairs(config)
    traces = tuple(trace for trace in _selection_traces(config) if trace.seed in {seed for seed, _ in pairs})
    started = perf_counter()
    summary = _run_jobs(config, _jobs(config, traces, pairs=pairs))
    expected = len(METHODS) * len(pairs)
    if len(summary) != expected:
        raise ValueError(f"pilot has {len(summary)} cells, expected {expected}")
    path = config.output / "pilot/summary.csv"
    write_csv_atomic(summary, path)
    return _write_stage(config, "pilot", {
        "pilot_cell_count": len(summary),
        "pilot_cells": [list(pair) for pair in pairs],
        "reused_by_evaluate": True,
        "pilot_wall_seconds": perf_counter() - started,
        "pilot_cpu_seconds": float(summary["cell_elapsed_seconds"].sum()),
        "summary": str(path),
    })


def _reference_rows(config: ExploreConfig) -> pd.DataFrame:
    if config.smoke:
        rows = [
            parent._execute_smoke_reference(config, trace, budget, method)
            for trace in _selection_traces(config)
            for budget in config.budgets
            for method in (G15_METHOD, PREDICTIVE_METHOD)
        ]
        return pd.DataFrame(rows)
    summary = pd.read_csv(parent.V4_ROOT / "feature-screen/summary.csv")
    selected = summary[
        summary["method"].isin((G15_METHOD, PREDICTIVE_METHOD))
        & summary["seed"].astype(int).isin(config.selection_seeds)
        & summary["budget"].astype(int).isin(config.budgets)
    ].copy()
    keys = selected[["seed", "budget", "method"]]
    if len(selected) != config.expected_reference_cells or keys.duplicated().any():
        raise ValueError("G15/MPC reference-cell matrix differs")
    if set(selected["status"].astype(str)) != {"completed"}:
        raise ValueError("G15/MPC reference cells include failures")
    selected["scope"] = "reused_reference"
    return selected


def _rank3_rows(config: ExploreConfig) -> pd.DataFrame:
    summary = pd.read_csv(PARENT_ROOT / "evaluate/new_summary.csv")
    selected = summary[
        (summary["method"] == RANK3_METHOD)
        & summary["seed"].astype(int).isin(config.selection_seeds)
        & summary["budget"].astype(int).isin(config.budgets)
    ].copy()
    keys = selected[["seed", "budget", "method"]]
    if len(selected) != config.expected_rank3_cells or keys.duplicated().any():
        raise ValueError("parent rank-3 cell matrix differs")
    if set(selected["status"].astype(str)) != {"completed"}:
        raise ValueError("parent rank-3 cells include failures")
    selected["scope"] = "reused_rank3"
    selected["source_summary_sha256"] = _raw_sha256(PARENT_ROOT / "evaluate/new_summary.csv")
    return selected


def run_evaluate(config: ExploreConfig) -> Path:
    _load_stage(config, "pilot")
    started = perf_counter()
    fresh = _run_jobs(config, _jobs(config, _selection_traces(config)))
    if len(fresh) != config.expected_new_cells:
        raise ValueError(f"new evaluation has {len(fresh)} cells, expected {config.expected_new_cells}")
    rank3 = _rank3_rows(config)
    references = _reference_rows(config)
    combined = pd.concat((fresh, rank3, references), ignore_index=True, sort=False)
    keys = combined[["seed", "budget", "method"]]
    if len(combined) != config.expected_report_cells or keys.duplicated().any():
        raise ValueError("joined report matrix is incomplete or duplicated")
    fresh_path = config.output / "evaluate/new_summary.csv"
    rank3_path = config.output / "evaluate/reused_rank3_summary.csv"
    reference_path = config.output / "evaluate/reused_reference_summary.csv"
    joined_path = config.output / "evaluate/summary.csv"
    write_csv_atomic(fresh, fresh_path)
    write_csv_atomic(rank3, rank3_path)
    write_csv_atomic(references, reference_path)
    write_csv_atomic(combined, joined_path)
    return _write_stage(config, "evaluate", {
        "new_cell_count": len(fresh),
        "rank3_cell_count": len(rank3),
        "reference_cell_count": len(references),
        "joined_cell_count": len(combined),
        "failure_count": int((fresh["status"].astype(str) != "completed").sum()),
        "matrix_wall_seconds": perf_counter() - started,
        "pilot_cells_reused": len(METHODS) * len(_pilot_pairs(config)),
        "new_summary": str(fresh_path),
        "rank3_summary": str(rank3_path),
        "reference_summary": str(reference_path),
        "summary": str(joined_path),
    })


def _paired(summary: pd.DataFrame, method: str, reference: str, budget: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = summary[summary["method"] == method]
    right = summary[summary["method"] == reference]
    if budget is not None:
        left = left[left["budget"].astype(int) == budget]
        right = right[right["budget"].astype(int) == budget]
    left = left.set_index(["seed", "budget"]).sort_index()
    right = right.set_index(["seed", "budget"]).sort_index()
    if not left.index.equals(right.index):
        raise ValueError(f"paired cells differ: {method} vs {reference}")
    return left, right


def _bootstrap_mean(values: NDArray[np.float64]) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(parent.BOOTSTRAP_SEED)
    draws = rng.integers(0, len(values), size=(parent.BOOTSTRAP_REPLICATES, len(values)))
    means = values[draws].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def policy_metrics(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_budget = []
    overall = []
    severe = []
    for method in METHODS:
        candidate, g15 = _paired(summary, method, G15_METHOD)
        _, predictive = _paired(summary, method, PREDICTIVE_METHOD)
        valid = (
            (candidate["status"].astype(str) == "completed")
            & (g15["status"].astype(str) == "completed")
            & (predictive["status"].astype(str) == "completed")
        )
        loss = candidate.loc[valid, "mean_approx_loss"].to_numpy(float)
        g15_loss = g15.loc[valid, "mean_approx_loss"].to_numpy(float)
        predictive_loss = predictive.loc[valid, "mean_approx_loss"].to_numpy(float)
        ratio_g15 = loss / g15_loss
        ratio_predictive = loss / predictive_loss
        fpr = candidate.loc[valid, "fpr"].to_numpy(float) - g15.loc[valid, "fpr"].to_numpy(float)
        throughput = candidate.loc[valid, "event_count"].to_numpy(float) / (
            candidate.loc[valid, "event_loop_time_ms"].to_numpy(float) / 1000.0
        )
        g15_throughput = g15.loc[valid, "event_count"].to_numpy(float) / (
            g15.loc[valid, "event_loop_time_ms"].to_numpy(float) / 1000.0
        )
        fpr_mean, fpr_low, fpr_high = _bootstrap_mean(fpr)
        fallback_count = int(pd.to_numeric(candidate["fallback_count"], errors="coerce").fillna(0).sum())
        severe_mask = ratio_predictive > parent.TAIL_MULTIPLIER
        valid_keys = candidate.index[valid]
        for key, ratio in zip(valid_keys[severe_mask], ratio_predictive[severe_mask], strict=True):
            severe.append({"method": method, "seed": int(key[0]), "budget": int(key[1]), "loss_ratio_vs_predictive": ratio})
        overall.append({
            "method": method,
            "member_count": int(candidate["member_count"].iloc[0]),
            "guarded": method.endswith("_guarded"),
            "cell_count": len(candidate),
            "valid_count": int(valid.sum()),
            "failure_count": int((candidate["status"].astype(str) != "completed").sum()),
            "fallback_count": fallback_count,
            "severe_tail_count": int(severe_mask.sum()),
            "worst_loss_ratio_vs_predictive": float(np.max(ratio_predictive)),
            "p95_loss_ratio_vs_predictive": float(np.quantile(ratio_predictive, 0.95)),
            "median_loss_ratio_vs_g15": float(np.median(ratio_g15)),
            "mean_fpr_difference_vs_g15": fpr_mean,
            "mean_fpr_difference_ci_low": fpr_low,
            "mean_fpr_difference_ci_high": fpr_high,
            "max_fpr_difference_vs_g15": float(np.max(fpr)),
            "median_paired_throughput_retention": float(np.median(throughput / g15_throughput)),
        })
        for budget in config_budgets(summary):
            budget_candidate, budget_g15 = _paired(summary, method, G15_METHOD, budget)
            _, budget_predictive = _paired(summary, method, PREDICTIVE_METHOD, budget)
            budget_valid = (
                (budget_candidate["status"].astype(str) == "completed")
                & (budget_g15["status"].astype(str) == "completed")
                & (budget_predictive["status"].astype(str) == "completed")
            )
            budget_loss = budget_candidate.loc[budget_valid, "mean_approx_loss"].to_numpy(float)
            budget_g15_loss = budget_g15.loc[budget_valid, "mean_approx_loss"].to_numpy(float)
            budget_predictive_loss = budget_predictive.loc[budget_valid, "mean_approx_loss"].to_numpy(float)
            budget_fpr = budget_candidate.loc[budget_valid, "fpr"].to_numpy(float) - budget_g15.loc[budget_valid, "fpr"].to_numpy(float)
            fpr_mean_b, fpr_low_b, fpr_high_b = _bootstrap_mean(budget_fpr)
            per_budget.append({
                "method": method,
                "budget": budget,
                "cell_count": len(budget_candidate),
                "valid_count": int(budget_valid.sum()),
                "failure_count": int((~budget_valid).sum()),
                "median_loss": float(np.median(budget_loss)),
                "q25_loss": float(np.quantile(budget_loss, 0.25)),
                "q75_loss": float(np.quantile(budget_loss, 0.75)),
                "p95_loss": float(np.quantile(budget_loss, 0.95)),
                "worst_loss": float(np.max(budget_loss)),
                "median_loss_ratio_vs_g15": float(np.median(budget_loss / budget_g15_loss)),
                "p95_loss_ratio_vs_predictive": float(np.quantile(budget_loss / budget_predictive_loss, 0.95)),
                "worst_loss_ratio_vs_predictive": float(np.max(budget_loss / budget_predictive_loss)),
                "severe_tail_count": int(((budget_loss / budget_predictive_loss) > parent.TAIL_MULTIPLIER).sum()),
                "mean_fpr_difference_vs_g15": fpr_mean_b,
                "mean_fpr_difference_ci_low": fpr_low_b,
                "mean_fpr_difference_ci_high": fpr_high_b,
                "max_fpr_difference_vs_g15": float(np.max(budget_fpr)),
            })
    per_budget_frame = pd.DataFrame(per_budget)
    overall_frame = pd.DataFrame(overall)
    maxima = per_budget_frame.groupby("method")["median_loss_ratio_vs_g15"].max()
    overall_frame["max_budget_median_loss_ratio_vs_g15"] = overall_frame["method"].map(maxima)
    return per_budget_frame, overall_frame, pd.DataFrame(severe)


def config_budgets(summary: pd.DataFrame) -> tuple[int, ...]:
    return tuple(sorted(summary["budget"].astype(int).unique()))


def eligibility_table(per_budget: pd.DataFrame, overall: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in overall.iterrows():
        budget_rows = per_budget[per_budget["method"] == row["method"]]
        checks = {
            "zero_severe_tail": int(row["severe_tail_count"]) == 0,
            "zero_failures": int(row["failure_count"]) == 0,
            "zero_fallbacks": int(row["fallback_count"]) == 0,
            "all_budget_medians_within_1_25_g15": bool((budget_rows["median_loss_ratio_vs_g15"] <= 1.25).all()),
            "throughput_at_least_half_g15": float(row["median_paired_throughput_retention"]) >= 0.50,
            "mean_fpr_regression_at_most_0_005": float(row["mean_fpr_difference_vs_g15"]) <= 0.005,
            "individual_fpr_regression_at_most_0_05": float(row["max_fpr_difference_vs_g15"]) <= 0.05,
        }
        rows.append({**row.to_dict(), **checks, "eligible": all(checks.values())})
    table = pd.DataFrame(rows).sort_values(
        [
            "eligible",
            "worst_loss_ratio_vs_predictive",
            "p95_loss_ratio_vs_predictive",
            "max_budget_median_loss_ratio_vs_g15",
            "median_paired_throughput_retention",
            "member_count",
            "guarded",
            "method",
        ],
        ascending=[False, True, True, True, False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    table["eligible_rank"] = np.where(table["eligible"], np.arange(1, len(table) + 1), np.nan)
    return table


def _paired_long(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        candidate, g15 = _paired(summary, method, G15_METHOD)
        _, predictive = _paired(summary, method, PREDICTIVE_METHOD)
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


def _decision_tables(config: ExploreConfig) -> dict[str, pd.DataFrame]:
    frames = []
    for path in sorted((config.output / "evaluate/cells").rglob("decisions.csv")):
        frame = pd.read_csv(path)
        frame.insert(0, "method", path.parent.name)
        frame.insert(1, "seed", int(next(part.split("-", 1)[1] for part in path.parts if part.startswith("seed-"))))
        frame["budget"] = int(next(part.split("-", 1)[1] for part in path.parts if part.startswith("budget-")))
        frames.append(frame)
    decisions = pd.concat(frames, ignore_index=True)
    over = decisions[decisions["over_bound"].astype(bool)].copy()
    ordinary = over[over["selected_action"].isin(CANDIDATES)]
    composition = ordinary.groupby(["method", "budget", "selected_action"], sort=True).size().rename("count").reset_index()
    composition["percentage"] = 100.0 * composition["count"] / composition.groupby(["method", "budget"])["count"].transform("sum")
    disagreement = over.groupby(["method", "budget"], sort=True).agg(
        decision_count=("step", "size"),
        member_top1_disagreement_count=("member_top1_disagreement", "sum"),
        member_top1_disagreement_rate=("member_top1_disagreement", "mean"),
        mean_winner_margin=("winner_margin", "mean"),
    ).reset_index()
    vote_patterns = over.groupby(["method", "budget", "vote_pattern"], sort=True).size().rename("count").reset_index()
    vote_patterns["rate"] = vote_patterns["count"] / vote_patterns.groupby(["method", "budget"])["count"].transform("sum")
    guard_rates = over.groupby(["method", "budget"], sort=True).agg(
        decision_count=("step", "size"),
        guard_count=("guard_invoked", "sum"),
        guard_rate=("guard_invoked", "mean"),
        conditional_override_rate=("guard_override", lambda values: float(values.sum()) / max(1, int(over.loc[values.index, "guard_invoked"].sum()))),
        mean_contenders_when_guarded=("contender_count", lambda values: float(values[over.loc[values.index, "guard_invoked"].astype(bool)].mean())),
        mean_guard_added_decision_time_ms=("guard_added_decision_time_ms", "mean"),
        max_guard_added_decision_time_ms=("guard_added_decision_time_ms", "max"),
    ).reset_index()
    return {
        "reducer_composition.csv": composition,
        "ensemble_disagreement.csv": disagreement,
        "vote_patterns.csv": vote_patterns,
        "guard_rates.csv": guard_rates,
    }


def _throughput_pairs(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pure, guarded in ((METHODS[0], METHODS[2]), (METHODS[1], METHODS[3])):
        guard_rows, pure_rows = _paired(summary, guarded, pure)
        for key in guard_rows.index:
            guarded_throughput = guard_rows.loc[key, "event_count"] / (guard_rows.loc[key, "event_loop_time_ms"] / 1000.0)
            pure_throughput = pure_rows.loc[key, "event_count"] / (pure_rows.loc[key, "event_loop_time_ms"] / 1000.0)
            rows.append({
                "guarded_method": guarded,
                "unguarded_method": pure,
                "seed": int(key[0]),
                "budget": int(key[1]),
                "guarded_throughput": guarded_throughput,
                "unguarded_throughput": pure_throughput,
                "guarded_vs_unguarded_retention": guarded_throughput / pure_throughput,
            })
    return pd.DataFrame(rows)


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    return plt


def _save_figure(fig: object, output: Path) -> tuple[Path, Path]:
    pdf = output.with_suffix(".pdf")
    png = output.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png, dpi=250, bbox_inches="tight", pad_inches=0.03)
    return pdf, png


def _plot_metric(paired: pd.DataFrame, column: str, ylabel: str, output: Path, *, reference: float, log_x: bool = False) -> tuple[Path, Path]:
    plt = _pyplot()
    if log_x:
        fig, axis = plt.subplots(figsize=(7.1, 3.4))
        for method in METHODS:
            values = np.sort(paired.loc[paired["method"] == method, column].to_numpy(float))
            axis.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=method, linewidth=1.0)
        axis.axvline(reference, color="0.4", linewidth=0.7)
        axis.set_xscale("log")
        axis.set_xlabel(ylabel)
        axis.set_ylabel("Cell ECDF")
        axis.legend(ncol=2, fontsize=6, frameon=False)
    else:
        fig, axis = plt.subplots(figsize=(7.1, 3.4))
        values = [paired.loc[paired["method"] == method, column].to_numpy(float) for method in METHODS]
        axis.boxplot(values, positions=np.arange(len(METHODS)), widths=0.62, showfliers=True)
        axis.axhline(reference, color="0.4", linewidth=0.7)
        axis.set_xticks(np.arange(len(METHODS)), METHODS, rotation=25, ha="right", fontsize=6)
        axis.set_ylabel(ylabel)
    axis.grid(color="0.9", linewidth=0.5)
    fig.tight_layout()
    paths = _save_figure(fig, output)
    plt.close(fig)
    return paths


def run_report(config: ExploreConfig) -> Path:
    evaluation = _load_stage(config, "evaluate")
    summary = pd.read_csv(str(evaluation["summary"]))
    per_budget, overall, severe = policy_metrics(summary)
    eligibility = eligibility_table(per_budget, overall)
    paired = _paired_long(summary)
    throughput_pairs = _throughput_pairs(summary)
    failure_fallback = overall[["method", "failure_count", "fallback_count", "severe_tail_count"]].copy()
    artifact_dir = config.output / "report/artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "trace_cells.csv": summary,
        "paired_cells.csv": paired,
        "per_budget_metrics.csv": per_budget,
        "overall_metrics.csv": overall,
        "eligibility.csv": eligibility,
        "severe_tail_cells.csv": severe,
        "failure_fallback_summary.csv": failure_fallback,
        "guarded_vs_unguarded_throughput.csv": throughput_pairs,
        **_decision_tables(config),
    }
    artifacts: list[Path] = []
    for name, frame in tables.items():
        path = artifact_dir / name
        write_csv_atomic(frame, path)
        artifacts.append(path)
    artifacts.extend(_plot_metric(paired, "loss_ratio_vs_predictive", "Paired mean-loss ratio vs predictive MPC", artifact_dir / "loss_ratio_ecdf", reference=1.0, log_x=True))
    artifacts.extend(_plot_metric(paired, "fpr_difference_vs_g15", "Paired FPR difference vs G15", artifact_dir / "paired_fpr", reference=0.0))
    artifacts.extend(_plot_metric(paired, "throughput_retention_vs_g15", "Paired throughput retention vs G15", artifact_dir / "throughput_retention", reference=0.5))
    eligible = eligibility[eligibility["eligible"].astype(bool)]
    selected = G15_METHOD if eligible.empty else str(eligible.iloc[0]["method"])
    decision = {
        "selected_method": selected,
        "eligible_challenger_count": len(eligible),
        "fallback_if_none": G15_METHOD,
        "confirmation_launched": False,
        "fixed_traces_launched": False,
        "ranking_rule": "worst predictive-loss ratio, P95 ratio, maximum budget median regression, descending throughput; exact ties prefer fewer members then unguarded",
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
    if _source_snapshot(config) != preflight["source_snapshot"]:
        raise ValueError("parent exploratory artifacts changed during vote-guard exploration")
    prepare = _load_stage(config, "prepare")
    train = _load_stage(config, "train")
    evaluate = _load_stage(config, "evaluate")
    if int(prepare["new_plan_count"]) != len(NEW_OPTIMIZER_SEEDS) * len(config.budgets):
        raise ValueError("new resampling-plan count differs")
    if int(train["new_model_count"]) != len(NEW_OPTIMIZER_SEEDS) * len(config.budgets):
        raise ValueError("new model count differs")
    expected = (
        ("new_cell_count", config.expected_new_cells),
        ("rank3_cell_count", config.expected_rank3_cells),
        ("reference_cell_count", config.expected_reference_cells),
        ("joined_cell_count", config.expected_report_cells),
    )
    for field, value in expected:
        if int(evaluate[field]) != value:
            raise ValueError(f"{field} differs")
    hashes = _load_json(Path(str(report["artifact_hashes"])))
    artifact_root = Path(str(report["artifact_hashes"])).parent
    for relative, expected_hash in hashes.items():
        if _raw_sha256(artifact_root / relative) != expected_hash:
            raise ValueError(f"report artifact hash differs: {relative}")
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
        "new_model_count": len(NEW_OPTIMIZER_SEEDS) * len(config.budgets),
        "new_cell_count": config.expected_new_cells,
        "rank3_cell_count": config.expected_rank3_cells,
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
        print(f"skip completed vote-guard stage: {stage}", flush=True)
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
    print(f"start vote-guard stage: {stage}", flush=True)
    result = functions[stage](config)
    print(f"complete vote-guard stage: {stage}", flush=True)
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
        "expected_new_models": len(NEW_OPTIMIZER_SEEDS) * len(config.budgets),
        "expected_new_cells": config.expected_new_cells,
        "expected_rank3_cells": config.expected_rank3_cells,
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
    print(f"PRP vote-guard exploratory stage complete: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
