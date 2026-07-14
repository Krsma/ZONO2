"""Staged commands for RTLola reducer-ranking experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from pzr.learning.artifacts import load_ranking_dataset, write_ranking_dataset
from pzr.learning.dataset import RankingDataset
from pzr.learning.provenance import model_sha256, pzr_source_sha256
from pzr.learning.ranker import RankingPolicy, evaluate_ranking, train_ranking_policy
from pzr.rtlola.actions import MPC_ACTION_NAMES, default_action_catalog
from pzr.rtlola.binding import (
    BINDING_BUILD_PROFILE,
    BINDING_REVISION,
    INTERPRETER_REVISION,
)
from pzr.rtlola.features import RTL_RANKING_FEATURE_SCHEMA
from pzr.rtlola.learned_policy import RtlolaRankingPolicy
from pzr.rtlola.learning_data import collect_teacher_episode, write_collected_dataset
from pzr.rtlola.learning_evaluation import (
    FixedLearningEvaluationConfig,
    run_fixed_learning_evaluation,
)
from pzr.rtlola.robot_arm import TRACE_KINDS
from pzr.rtlola.robot_arm_random import (
    RANDOM_WAYPOINT_CONDITIONS,
    RandomWaypointConfig,
    RandomWaypointTrace,
    generate_random_waypoint_trace,
    load_random_waypoint_trace,
    write_random_waypoint_trace,
)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build RTLola learning artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect", help="Generate and label waypoint traces")
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--event-count", type=int, required=True)
    collect.add_argument("--budgets", type=_csv_ints, required=True)
    collect.add_argument("--candidates", type=_csv_strings, default=MPC_ACTION_NAMES)
    collect.add_argument("--conditions", type=_csv_strings, default=RANDOM_WAYPOINT_CONDITIONS)
    collect.add_argument("--train-seeds", type=int, default=4)
    collect.add_argument("--validation-seeds", type=int, default=1)
    collect.add_argument("--test-seeds", type=int, default=1)
    collect.add_argument("--seed-start", type=int, default=0)
    collect.add_argument(
        "--behavior-model",
        type=Path,
        default=None,
        help="Optional ranker used to visit states for one aggregation round",
    )
    train = subparsers.add_parser("train", help="Train a fixed-catalog PyTorch ranker")
    train.add_argument("--dataset", type=Path, action="append", required=True)
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
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--model-name", default="learned_geometry15")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--budgets", type=_csv_ints, required=True)
    evaluate.add_argument("--candidates", type=_csv_strings, default=MPC_ACTION_NAMES)
    evaluate.add_argument("--trace-kinds", type=_csv_strings, default=TRACE_KINDS)
    evaluate.add_argument(
        "--baselines",
        type=_csv_strings,
        default=("girard", "mpc_terminal_full_width"),
    )
    evaluate.add_argument(
        "--length", type=int, default=None,
        help="Optional common fixed-trace prefix length for timing smokes",
    )
    evaluate.add_argument("--horizon", type=int, default=1)
    evaluate.add_argument("--beam-width", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "collect":
        run_collect(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError(args.command)


def run_collect(args: argparse.Namespace) -> None:
    if args.event_count < 2:
        raise ValueError("event count must be at least two")
    if not args.budgets:
        raise ValueError("at least one budget is required")
    if args.train_seeds < 1 or min(args.validation_seeds, args.test_seeds) < 0:
        raise ValueError("training needs a seed and split seed counts cannot be negative")
    if args.seed_start < 0:
        raise ValueError("seed start must be non-negative")
    unknown_conditions = set(args.conditions) - set(RANDOM_WAYPOINT_CONDITIONS)
    if unknown_conditions:
        raise ValueError(f"unknown random-waypoint conditions: {sorted(unknown_conditions)}")
    candidate_names = tuple(args.candidates)
    catalog = default_action_catalog(candidate_names)
    behavior = None
    behavior_model_sha256 = None
    if args.behavior_model is not None:
        if args.validation_seeds or args.test_seeds:
            raise ValueError(
                "learned-behavior aggregation must contain training trajectories only"
            )
        behavior = RtlolaRankingPolicy(
            policy=_load_policy(args.behavior_model),
            catalog=catalog,
        )
        behavior_model_sha256 = model_sha256(args.behavior_model)
    scenario = scenario_by_name("robot_arm")
    collection = "dagger" if behavior is not None else "teacher"
    source_sha256 = pzr_source_sha256()
    shard_datasets = []
    shard_metadata_frames = []
    trace_records = []
    for split, seed in _split_seeds(args):
        for condition in args.conditions:
            trace_id = f"{condition}:seed-{seed}"
            trace_config = RandomWaypointConfig(
                seed=seed,
                condition=condition,
                event_count=args.event_count,
            )
            trace_dir = args.output / "traces" / split / trace_id
            trace = _load_or_generate_trace(trace_config, trace_dir)
            trace_records.append({
                "trace_id": trace_id,
                "split": split,
                "condition": condition,
                "seed": seed,
                "trace_sha256": trace.metadata.trace_sha256,
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
                    "event_count": args.event_count,
                    "trace_id": trace_id,
                    "split": split,
                    "condition": condition,
                    "seed": seed,
                    "budget": budget,
                    "candidate_names": list(candidate_names),
                    "feature_schema": _feature_schema_payload(),
                    "trace_sha256": trace.metadata.trace_sha256,
                    "behavior_model_sha256": behavior_model_sha256,
                    "binding_revision": BINDING_REVISION,
                    "interpreter_revision": INTERPRETER_REVISION,
                    "binding_build_profile": BINDING_BUILD_PROFILE,
                    "pzr_source_sha256": source_sha256,
                }
                if shard_dir.exists():
                    dataset, sample_metadata, manifest = load_ranking_dataset(
                        shard_dir,
                    )
                    _validate_shard_manifest(manifest, shard_identity)
                else:
                    samples = collect_teacher_episode(
                        scenario=scenario,
                        events=trace.events,
                        trace_id=trace_id,
                        split=split,
                        condition=condition,
                        seed=seed,
                        budget=budget,
                        candidate_names=candidate_names,
                        behavior_policy=behavior,
                    )
                    write_collected_dataset(
                        samples,
                        shard_dir,
                        shard_identity,
                        candidate_names=candidate_names,
                    )
                    dataset, sample_metadata, manifest = load_ranking_dataset(
                        shard_dir,
                    )
                    _validate_shard_manifest(manifest, shard_identity)
                shard_datasets.append(dataset)
                shard_metadata_frames.append(sample_metadata)
    dataset = RankingDataset.concatenate(shard_datasets)
    if dataset.num_samples == 0:
        raise ValueError("teacher collection produced no reduction decisions")
    sample_metadata = pd.concat(shard_metadata_frames, ignore_index=True)
    metadata = {
        "scenario": scenario.name,
        "collection": collection,
        "event_count": args.event_count,
        "budgets": list(args.budgets),
        "conditions": list(args.conditions),
        "seed_start": args.seed_start,
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


def _load_or_generate_trace(
    config: RandomWaypointConfig,
    directory: Path,
) -> RandomWaypointTrace:
    trace_path = directory / "trace.csv"
    metadata_path = directory / "metadata.json"
    if trace_path.exists() != metadata_path.exists():
        raise ValueError(f"incomplete random-waypoint trace artifact: {directory}")
    if trace_path.exists():
        trace = load_random_waypoint_trace(directory)
        if trace.metadata.generator_config != asdict(config):
            raise ValueError(f"random-waypoint trace configuration differs: {directory}")
        return trace
    trace = generate_random_waypoint_trace(config)
    write_random_waypoint_trace(trace, directory)
    return trace


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
    loaded = [load_ranking_dataset(path) for path in args.dataset]
    dataset = RankingDataset.concatenate([item[0] for item in loaded])
    sample_metadata = pd.concat([item[1] for item in loaded], ignore_index=True)
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
        "schema": "pzr.ranking-training.v1",
        "datasets": [str(path) for path in args.dataset],
        "candidate_names": list(policy.candidate_names),
        "feature_schema": asdict(policy.feature_schema),
        "training": asdict(result),
        "seed": args.seed,
    }
    (args.output / "training.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
    )
    _validation_metric_frame(policy, dataset, sample_metadata).to_csv(
        args.output / "validation_metrics.csv", index=False,
    )
    print(f"Reducer ranker complete: {args.output}")


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


def _validation_metric_frame(
    policy: RankingPolicy,
    dataset: RankingDataset,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    validation = np.asarray(dataset.splits) == "validation"
    if not np.any(validation):
        raise ValueError("training diagnostics require a validation split")
    grouped_metadata = metadata.copy()
    if "condition" not in grouped_metadata:
        grouped_metadata["condition"] = "__unspecified__"
    rows = []
    groups: list[tuple[str, int | None, np.ndarray]] = [
        ("__all__", None, validation),
    ]
    for (condition, budget), indices in grouped_metadata[validation].groupby(
        ["condition", "budget"], dropna=False,
    ).groups.items():
        mask = np.zeros(dataset.num_samples, dtype=np.bool_)
        mask[np.asarray(list(indices), dtype=np.int64)] = True
        groups.append((str(condition), int(budget), mask))
    for condition, budget, mask in groups:
        metrics = evaluate_ranking(policy, dataset.subset(np.flatnonzero(mask)))
        rows.append({
            "condition": condition,
            "budget": budget,
            "sample_count": int(np.count_nonzero(mask)),
            **asdict(metrics),
        })
    return pd.DataFrame(rows)


def run_evaluate(args: argparse.Namespace) -> None:
    unknown_traces = set(args.trace_kinds) - set(TRACE_KINDS)
    if unknown_traces:
        raise ValueError(f"unknown fixed robot-arm traces: {sorted(unknown_traces)}")
    candidate_names = tuple(args.candidates)
    policy = RtlolaRankingPolicy(
        _load_policy(args.model), default_action_catalog(candidate_names),
    )
    run_fixed_learning_evaluation(
        FixedLearningEvaluationConfig(
            output=args.output,
            model_name=args.model_name,
            trace_kinds=tuple(args.trace_kinds),
            budgets=tuple(args.budgets),
            baselines=tuple(args.baselines),
            candidate_names=candidate_names,
            length=args.length,
            horizon=args.horizon,
            beam_width=args.beam_width,
        ),
        policy,
        model_sha256=model_sha256(args.model),
        source_sha256=pzr_source_sha256(),
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
