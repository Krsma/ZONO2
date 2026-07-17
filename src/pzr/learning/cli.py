"""Staged commands for RTLola reducer-ranking experiments."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import json
from multiprocessing import get_context
from pathlib import Path
import re
from typing import Sequence

import pandas as pd

from pzr.learning.artifacts import load_ranking_dataset, write_ranking_dataset
from pzr.learning.dataset import RankingDataset
from pzr.learning.diagnostics import (
    candidate_diagnostics,
    dataset_diagnostics,
    validation_metrics,
)
from pzr.learning.provenance import model_sha256, pzr_source_sha256, sha256_files
from pzr.learning.ranker import RankingPolicy, train_ranking_policy
from pzr.learning.targets import TARGET_CONTRACT
from pzr.rtlola.actions import MPC_ACTION_NAMES, default_action_catalog
from pzr.rtlola.binding import (
    BINDING_BUILD_PROFILE,
    BINDING_REVISION,
    INTERPRETER_REVISION,
)
from pzr.rtlola.engine import RtlolaEvent
from pzr.rtlola.features import RTL_RANKING_FEATURE_SCHEMA
from pzr.rtlola.learned_policy import RtlolaRankingPolicy
from pzr.rtlola.learning_data import collect_teacher_episode, write_collected_dataset
from pzr.rtlola.learning_evaluation import (
    FixedLearningEvaluationConfig,
    run_fixed_learning_evaluation,
)
from pzr.rtlola.learning_traces import (
    RandomWaypointTraceStoreConfig,
    generate_random_waypoint_trace_store,
    load_random_waypoint_trace_store,
)
from pzr.rtlola.robot_arm import TRACE_KINDS
from pzr.rtlola.scenarios import scenario_by_name


def _csv_strings(value: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("comma-separated value must not be empty")
    return values


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part) for part in _csv_strings(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("integer values must be non-negative")
    return values


@dataclass(frozen=True)
class NamedPath:
    name: str
    path: Path


def _named_path(value: str) -> NamedPath:
    name, separator, raw_path = value.partition("=")
    if not separator or not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME=/path with a filesystem-safe name")
    return NamedPath(name=name, path=Path(raw_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build RTLola learning artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser(
        "generate", help="Build a validated random-waypoint trace store",
    )
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--event-count", type=int, required=True)
    generate.add_argument(
        "--conditions", type=_csv_strings, default=("random_waypoint",),
    )
    generate.add_argument("--seed-start", type=int, default=0)
    generate.add_argument("--seed-count", type=int, required=True)
    collect = subparsers.add_parser("collect", help="Label a validated trace store")
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--trace-store", type=Path, required=True)
    collect.add_argument("--budgets", type=_csv_ints, required=True)
    collect.add_argument("--candidates", type=_csv_strings, default=MPC_ACTION_NAMES)
    collect.add_argument("--train-seeds", type=int, default=4)
    collect.add_argument("--validation-seeds", type=int, default=1)
    collect.add_argument("--test-seeds", type=int, default=1)
    collect.add_argument("--seed-start", type=int, default=0)
    collect.add_argument("--workers", type=int, default=1)
    collect.add_argument(
        "--behavior-model",
        type=Path,
        default=None,
        help="Optional ranker used to visit states for one aggregation round",
    )
    train = subparsers.add_parser("train", help="Train a fixed-catalog PyTorch ranker")
    train.add_argument("--dataset", type=_named_path, action="append", required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--seed", type=int, default=42)
    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate generalization on fixed robot-arm traces",
    )
    evaluate.add_argument("--model", type=_named_path, action="append", required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--budgets", type=_csv_ints, required=True)
    evaluate.add_argument("--candidates", type=_csv_strings, default=MPC_ACTION_NAMES)
    evaluate.add_argument("--trace-kinds", type=_csv_strings, default=TRACE_KINDS)
    evaluate.add_argument(
        "--baselines",
        type=_csv_strings,
        default=("girard", "scott", "pca", "combastel", "mpc_terminal_full_width"),
    )
    evaluate.add_argument(
        "--length", type=int, default=None,
        help="Optional common fixed-trace prefix length for timing smokes",
    )
    evaluate.add_argument("--horizon", type=int, default=1)
    evaluate.add_argument("--beam-width", type=int, default=4)
    evaluate.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        run_generate(args)
    elif args.command == "collect":
        run_collect(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError(args.command)


def run_generate(args: argparse.Namespace) -> None:
    store = generate_random_waypoint_trace_store(
        RandomWaypointTraceStoreConfig(
            output=args.output,
            event_count=args.event_count,
            conditions=tuple(args.conditions),
            seed_start=args.seed_start,
            seed_count=args.seed_count,
        ),
    )
    print(
        f"Random-waypoint trace store complete: {store.root} "
        f"({len(store.traces)} traces)",
    )


def run_collect(args: argparse.Namespace) -> None:
    if not args.budgets:
        raise ValueError("at least one budget is required")
    if args.train_seeds < 1 or min(args.validation_seeds, args.test_seeds) < 0:
        raise ValueError("training needs a seed and split seed counts cannot be negative")
    if args.seed_start < 0:
        raise ValueError("seed start must be non-negative")
    if args.workers < 1:
        raise ValueError("collection workers must be positive")
    trace_store = load_random_waypoint_trace_store(args.trace_store)
    candidate_names = tuple(args.candidates)
    catalog = default_action_catalog(candidate_names)
    behavior = None
    behavior_model_sha256 = None
    if args.behavior_model is not None:
        if args.test_seeds:
            raise ValueError(
                "learned-behavior aggregation supports held-out validation, not test shards"
            )
        behavior = RtlolaRankingPolicy(
            policy=_load_policy(args.behavior_model),
            catalog=catalog,
        )
        behavior_model_sha256 = model_sha256(args.behavior_model)
    scenario = scenario_by_name("robot_arm")
    collection = "dagger" if behavior is not None else "teacher"
    source_sha256 = pzr_source_sha256()
    jobs = []
    trace_records = []
    for split, seed in _split_seeds(args):
        for stored_trace in trace_store.traces_for_seed(seed):
            trace_id = stored_trace.trace_id
            condition = stored_trace.condition
            trace = stored_trace.trace
            trace_records.append({
                "trace_id": trace_id,
                "split": split,
                "condition": condition,
                "seed": seed,
                "trace_sha256": trace.metadata.trace_sha256,
                "trace_store_relative_path": str(stored_trace.relative_path),
            })
            for budget in args.budgets:
                shard_dir = (
                    args.output / "shards" / split / condition
                    / f"seed-{seed}" / f"budget-{budget}"
                )
                shard_identity = {
                    "collection_shard": True,
                    "scenario": scenario.name,
                    "collection": collection,
                    "event_count": trace_store.event_count,
                    "trace_id": trace_id,
                    "split": split,
                    "condition": condition,
                    "seed": seed,
                    "budget": budget,
                    "candidate_names": list(candidate_names),
                    "feature_schema": _feature_schema_payload(),
                    "trace_sha256": trace.metadata.trace_sha256,
                    "trace_store_manifest_sha256": trace_store.manifest_sha256,
                    "behavior_model_sha256": behavior_model_sha256,
                    "binding_revision": BINDING_REVISION,
                    "interpreter_revision": INTERPRETER_REVISION,
                    "binding_build_profile": BINDING_BUILD_PROFILE,
                    "pzr_source_sha256": source_sha256,
                }
                jobs.append(_CollectionShardJob(
                    directory=shard_dir,
                    identity=shard_identity,
                    scenario_name=scenario.name,
                    events=trace.events,
                    trace_id=trace_id,
                    split=split,
                    condition=condition,
                    seed=seed,
                    budget=budget,
                    candidate_names=candidate_names,
                    behavior_model=args.behavior_model,
                ))
    _run_collection_jobs(jobs, args.workers)
    loaded_shards = [_load_collection_shard(job) for job in jobs]
    shard_datasets = [item[0] for item in loaded_shards]
    shard_metadata_frames = [item[1] for item in loaded_shards]
    dataset = RankingDataset.concatenate(shard_datasets)
    if dataset.num_samples == 0:
        raise ValueError("teacher collection produced no reduction decisions")
    sample_metadata = pd.concat(shard_metadata_frames, ignore_index=True)
    metadata = {
        "scenario": scenario.name,
        "collection": collection,
        "event_count": trace_store.event_count,
        "budgets": list(args.budgets),
        "conditions": list(trace_store.conditions),
        "seed_start": args.seed_start,
        "trace_store": str(trace_store.root),
        "trace_store_manifest_sha256": trace_store.manifest_sha256,
        "binding_revision": BINDING_REVISION,
        "interpreter_revision": INTERPRETER_REVISION,
        "binding_build_profile": BINDING_BUILD_PROFILE,
        "behavior_model_sha256": behavior_model_sha256,
        "feature_schema": _feature_schema_payload(),
        "pzr_source_sha256": source_sha256,
        "shard_count": len(shard_datasets),
        "traces": trace_records,
    }
    write_ranking_dataset(
        dataset,
        args.output / "dataset",
        sample_metadata,
        metadata,
    )
    _write_collection_summaries(sample_metadata, args.output / "dataset")
    print(f"Learning dataset complete: {args.output / 'dataset'}")


@dataclass(frozen=True)
class _CollectionShardJob:
    directory: Path
    identity: dict[str, object]
    scenario_name: str
    events: tuple[RtlolaEvent, ...]
    trace_id: str
    split: str
    condition: str
    seed: int
    budget: int
    candidate_names: tuple[str, ...]
    behavior_model: Path | None


def _run_collection_jobs(
    jobs: Sequence[_CollectionShardJob],
    workers: int,
) -> None:
    missing = []
    for job in jobs:
        if (job.directory / "manifest.json").is_file():
            _load_collection_shard(job)
        else:
            missing.append(job)
    if not missing:
        return
    if workers == 1:
        for job in missing:
            _run_collection_shard(job)
        return
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
    ) as executor:
        tuple(executor.map(_run_collection_shard, missing))


def _run_collection_shard(job: _CollectionShardJob) -> None:
    behavior = None
    if job.behavior_model is not None:
        behavior = RtlolaRankingPolicy(
            policy=_load_policy(job.behavior_model),
            catalog=default_action_catalog(job.candidate_names),
        )
    samples = collect_teacher_episode(
        scenario=scenario_by_name(job.scenario_name),
        events=job.events,
        trace_id=job.trace_id,
        split=job.split,
        condition=job.condition,
        seed=job.seed,
        budget=job.budget,
        candidate_names=job.candidate_names,
        behavior_policy=behavior,
    )
    write_collected_dataset(
        samples,
        job.directory,
        job.identity,
        candidate_names=job.candidate_names,
    )


def _load_collection_shard(
    job: _CollectionShardJob,
) -> tuple[RankingDataset, pd.DataFrame]:
    dataset, sample_metadata, manifest = load_ranking_dataset(job.directory)
    _validate_shard_manifest(manifest, job.identity)
    return dataset, sample_metadata


def _validate_shard_manifest(
    manifest: dict[str, object],
    expected: dict[str, object],
) -> None:
    mismatched = [
        name for name, value in expected.items()
        if manifest.get(name) != value
    ]
    if mismatched:
        raise ValueError(
            "learning collection shard identity differs for: "
            + ", ".join(sorted(mismatched))
        )


def _feature_schema_payload() -> dict[str, object]:
    return {
        "name": RTL_RANKING_FEATURE_SCHEMA.name,
        "version": RTL_RANKING_FEATURE_SCHEMA.version,
        "feature_names": list(RTL_RANKING_FEATURE_SCHEMA.feature_names),
        "log1p_features": list(RTL_RANKING_FEATURE_SCHEMA.log1p_features),
    }


def run_train(args: argparse.Namespace) -> None:
    inputs = tuple(args.dataset)
    names = tuple(item.name for item in inputs)
    if len(set(names)) != len(names):
        raise ValueError("named training datasets must have unique names")
    loaded = [load_ranking_dataset(item.path) for item in inputs]
    _validate_named_datasets(inputs, loaded)
    dataset = RankingDataset.concatenate([item[0] for item in loaded])
    metadata_frames = []
    for named_input, (_, metadata, _) in zip(inputs, loaded):
        frame = metadata.copy()
        frame.insert(0, "dataset_label", named_input.name)
        metadata_frames.append(frame)
    sample_metadata = pd.concat(metadata_frames, ignore_index=True)
    policy, result = train_ranking_policy(
        dataset,
        RTL_RANKING_FEATURE_SCHEMA,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    policy.save(args.output)
    payload = {
        "schema": "pzr.ranking-training.v2",
        "datasets": [
            {
                "name": item.name,
                "path": str(item.path),
                "sha256": sha256_files((
                    item.path / "manifest.json",
                    item.path / "samples.npz",
                    item.path / "samples.csv",
                    item.path / "candidate_costs.csv",
                )),
            }
            for item in inputs
        ],
        "candidate_names": list(policy.candidate_names),
        "feature_schema": asdict(policy.feature_schema),
        "target_contract": TARGET_CONTRACT,
        "training": asdict(result),
        "seed": args.seed,
        "binding_revision": BINDING_REVISION,
        "interpreter_revision": INTERPRETER_REVISION,
        "binding_build_profile": BINDING_BUILD_PROFILE,
        "pzr_source_sha256": pzr_source_sha256(),
    }
    (args.output / "training.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
    )
    validation_metrics(policy, dataset, sample_metadata).to_csv(
        args.output / "validation_metrics.csv", index=False,
    )
    dataset_diagnostics(dataset, sample_metadata).to_csv(
        args.output / "dataset_diagnostics.csv", index=False,
    )
    candidate_diagnostics(policy, dataset, sample_metadata).to_csv(
        args.output / "candidate_diagnostics.csv", index=False,
    )
    print(f"Reducer ranker complete: {args.output}")


def _validate_named_datasets(
    inputs: Sequence[NamedPath],
    loaded: Sequence[tuple[RankingDataset, pd.DataFrame, dict[str, object]]],
) -> None:
    first_dataset, _, first_manifest = loaded[0]
    for named_input, (dataset, _, manifest) in zip(inputs, loaded):
        if dataset.candidate_names != first_dataset.candidate_names:
            raise ValueError(f"dataset {named_input.name!r} candidate catalog differs")
        if dataset.feature_names != first_dataset.feature_names:
            raise ValueError(f"dataset {named_input.name!r} feature schema differs")
        if manifest.get("target_contract") != first_manifest.get("target_contract"):
            raise ValueError(f"dataset {named_input.name!r} target schema differs")
        if dataset.indices_for_split("train").size == 0:
            raise ValueError(f"dataset {named_input.name!r} has no training samples")
        if dataset.indices_for_split("validation").size == 0:
            raise ValueError(f"dataset {named_input.name!r} has no validation samples")


def _write_collection_summaries(metadata: pd.DataFrame, output: Path) -> None:
    groups = ["split", "condition", "budget"]
    summary = metadata.groupby(groups, dropna=False).agg(
        sample_count=("sample_id", "size"),
        mean_evaluated_leaves=("evaluated_leaves", "mean"),
        teacher_reducer_failure_count=("teacher_reducer_failure_count", "sum"),
        teacher_infeasible_candidate_count=("teacher_infeasible_candidate_count", "sum"),
        behavior_reducer_failure_count=("behavior_reducer_failure_count", "sum"),
        behavior_infeasible_candidate_count=("behavior_infeasible_candidate_count", "sum"),
        behavior_fallback_count=("behavior_fallback_used", "sum"),
    ).reset_index()
    summary["behavior_fallback_rate"] = (
        summary["behavior_fallback_count"] / summary["sample_count"]
    )
    summary.to_csv(output / "collection_summary.csv", index=False)
    for column, filename in (
        ("teacher_action", "teacher_action_counts.csv"),
        ("behavior_action", "behavior_action_counts.csv"),
    ):
        counts = (
            metadata.groupby([*groups, column], dropna=False)
            .size()
            .rename("count")
            .reset_index()
        )
        counts.to_csv(output / filename, index=False)


def run_evaluate(args: argparse.Namespace) -> None:
    unknown_traces = set(args.trace_kinds) - set(TRACE_KINDS)
    if unknown_traces:
        raise ValueError(f"unknown fixed robot-arm traces: {sorted(unknown_traces)}")
    candidate_names = tuple(args.candidates)
    model_inputs = tuple(args.model)
    model_names = tuple(item.name for item in model_inputs)
    if len(set(model_names)) != len(model_names):
        raise ValueError("named evaluation models must have unique names")
    policies = {
        item.name: RtlolaRankingPolicy(
            _load_policy(item.path), default_action_catalog(candidate_names),
        )
        for item in model_inputs
    }
    model_hashes = {
        item.name: model_sha256(item.path) for item in model_inputs
    }
    model_directories = {item.name: item.path for item in model_inputs}
    run_fixed_learning_evaluation(
        FixedLearningEvaluationConfig(
            output=args.output,
            model_names=model_names,
            trace_kinds=tuple(args.trace_kinds),
            budgets=tuple(args.budgets),
            baselines=tuple(args.baselines),
            candidate_names=candidate_names,
            length=args.length,
            horizon=args.horizon,
            beam_width=args.beam_width,
        ),
        policies,
        model_sha256=model_hashes,
        source_sha256=pzr_source_sha256(),
        model_directories=model_directories,
        workers=args.workers,
    )
    print(f"Learning evaluation complete: {args.output}")


def _load_policy(path: Path):
    from pzr.learning.ranker import RankingPolicy

    return RankingPolicy.load(path)


def _split_seeds(args: argparse.Namespace) -> Sequence[tuple[str, int]]:
    counts = (
        ("train", args.train_seeds),
        ("validation", args.validation_seeds),
        ("test", args.test_seeds),
    )
    offset = args.seed_start
    result = []
    for split, count in counts:
        result.extend((split, offset + index) for index in range(count))
        offset += count
    return tuple(result)


if __name__ == "__main__":
    main()
