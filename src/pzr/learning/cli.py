"""Staged commands for RTLola reducer-ranking experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

from pzr.learning.artifacts import load_ranking_dataset
from pzr.learning.dataset import RankingDataset
from pzr.learning.ranker import train_ranking_policy
from pzr.rtlola.actions import MPC_ACTION_NAMES, default_action_catalog
from pzr.rtlola.binding import (
    BINDING_BUILD_PROFILE,
    BINDING_REVISION,
    INTERPRETER_REVISION,
)
from pzr.rtlola.features import RTL_RANKING_FEATURE_SCHEMA
from pzr.rtlola.learned_policy import RtlolaRankingPolicy
from pzr.rtlola.learning_data import collect_teacher_episode, write_collected_dataset
from pzr.rtlola.robot_arm_random import (
    RANDOM_WAYPOINT_CONDITIONS,
    RandomWaypointConfig,
    generate_random_waypoint_trace,
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
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "collect":
        run_collect(args)
    elif args.command == "train":
        run_train(args)
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
    if args.behavior_model is not None:
        if args.validation_seeds or args.test_seeds:
            raise ValueError(
                "learned-behavior aggregation must contain training trajectories only"
            )
        behavior = RtlolaRankingPolicy(
            policy=_load_policy(args.behavior_model),
            catalog=catalog,
        )
    scenario = scenario_by_name("robot_arm")
    samples = []
    trace_records = []
    for split, seed in _split_seeds(args):
        for condition in args.conditions:
            trace_id = f"{condition}:seed-{seed}"
            trace = generate_random_waypoint_trace(RandomWaypointConfig(
                seed=seed,
                condition=condition,
                event_count=args.event_count,
            ))
            trace_dir = args.output / "traces" / split / trace_id
            write_random_waypoint_trace(trace, trace_dir)
            trace_records.append({
                "trace_id": trace_id,
                "split": split,
                "condition": condition,
                "seed": seed,
                "trace_sha256": trace.metadata.trace_sha256,
            })
            for budget in args.budgets:
                samples.extend(collect_teacher_episode(
                    scenario=scenario,
                    events=trace.events,
                    trace_id=trace_id,
                    split=split,
                    condition=condition,
                    seed=seed,
                    budget=budget,
                    candidate_names=candidate_names,
                    behavior_policy=behavior,
                ))
    metadata = {
        "scenario": scenario.name,
        "collection": "dagger" if behavior is not None else "teacher",
        "event_count": args.event_count,
        "budgets": list(args.budgets),
        "conditions": list(args.conditions),
        "seed_start": args.seed_start,
        "binding_revision": BINDING_REVISION,
        "interpreter_revision": INTERPRETER_REVISION,
        "binding_build_profile": BINDING_BUILD_PROFILE,
        "traces": trace_records,
    }
    write_collected_dataset(samples, args.output / "dataset", metadata)
    print(f"Learning dataset complete: {args.output / 'dataset'}")


def run_train(args: argparse.Namespace) -> None:
    loaded = [load_ranking_dataset(path)[0] for path in args.dataset]
    dataset = RankingDataset.concatenate(loaded)
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
    print(f"Reducer ranker complete: {args.output}")


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
