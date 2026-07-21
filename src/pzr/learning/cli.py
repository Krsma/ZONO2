"""Argument parsing and dispatch for RTLola reducer-policy experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

from pzr.learning.dart import (
    DartCalibrationArtifactConfig,
    DartCalibrationConfig,
    run_dart_calibration_artifact,
)
from pzr.learning.exploration import (
    ChallengerSelectionConfig,
    run_challenger_selection,
)
from pzr.learning.provenance import pzr_source_sha256
from pzr.learning.training import (
    NamedDataset,
    ReducerTrainingConfig,
    run_reducer_training,
)
from pzr.rtlola.actions import MPC_ACTION_NAMES
from pzr.rtlola.learning_collection import (
    LearningCollectionConfig,
    run_learning_collection,
)
from pzr.rtlola.learning_traces import (
    RandomWaypointTraceStoreConfig,
    generate_random_waypoint_trace_store,
)
from pzr.rtlola.policy_evaluation import (
    FixedPolicyEvaluationConfig,
    PolicyComparison,
    run_policy_evaluation_from_models,
)
from pzr.rtlola.paper_reporting import PaperReportConfig, write_paper_reports
from pzr.rtlola.robot_arm import TRACE_KINDS


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


def _csv_floats(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part) for part in _csv_strings(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if any(item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError("temperatures must be positive")
    return values


def _named_dataset(value: str) -> NamedDataset:
    name, separator, raw_path = value.partition("=")
    if not separator or not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME=/path with a filesystem-safe name")
    return NamedDataset(name=name, path=Path(raw_path))


def _policy_comparison(value: str) -> PolicyComparison:
    name, separator, methods = value.partition("=")
    challenger, method_separator, reference = methods.partition(":")
    if not separator or not method_separator:
        raise argparse.ArgumentTypeError(
            "expected NAME=CHALLENGER:REFERENCE for a policy comparison"
        )
    try:
        return PolicyComparison(name, challenger, reference)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Pairwise Ranking Policy artifacts and explicit secondary studies",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate", help="build a validated random-waypoint trace store",
    )
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--event-count", type=int, required=True)
    generate.add_argument("--conditions", type=_csv_strings, default=("random_waypoint",))
    generate.add_argument("--seed-start", type=int, default=0)
    generate.add_argument("--seed-count", type=int, required=True)

    collect = subparsers.add_parser(
        "collect", help="collect clean teacher labels or an explicit secondary DART dataset",
    )
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--trace-store", type=Path, required=True)
    collect.add_argument("--budgets", type=_csv_ints, required=True)
    collect.add_argument("--candidates", type=_csv_strings, default=MPC_ACTION_NAMES)
    collect.add_argument("--train-seeds", type=int, default=4)
    collect.add_argument("--validation-seeds", type=int, default=1)
    collect.add_argument("--test-seeds", type=int, default=0)
    collect.add_argument("--seed-start", type=int, default=0)
    collect.add_argument("--workers", type=int, default=1)
    collect.add_argument("--collection-mode", choices=("teacher", "dart"), default="teacher")
    collect.add_argument("--dart-calibration", type=Path)
    collect.add_argument("--disturbance-seed", type=int, default=20260717)

    train = subparsers.add_parser(
        "train", help="train Pairwise Ranking Policy (Soft-KL/expected-regret are secondary)",
    )
    train.add_argument("--dataset", type=_named_dataset, action="append", required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument(
        "--objective",
        choices=("pairwise", "soft-kl", "expected-regret"),
        default="pairwise",
        help="pairwise is primary; soft-kl and expected-regret are secondary objectives",
    )
    train.add_argument("--temperature-grid", type=_csv_floats)
    train.add_argument("--temperature-from", type=Path)
    train.add_argument("--feasibility-penalty", type=float, default=1.0)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--budget-filter",
        type=_csv_ints,
        help="Train only on samples at these recorded transform bounds",
    )

    calibrate = subparsers.add_parser(
        "calibrate-dart", help="secondary: fit guarded-DART disturbance calibration",
    )
    calibrate.add_argument("--model", type=Path, required=True)
    calibrate.add_argument("--dataset", type=_named_dataset, required=True)
    calibrate.add_argument("--split", default="validation")
    calibrate.add_argument("--regret-cap-quantile", type=float, default=0.9)
    calibrate.add_argument("--direction-pseudocount", type=float, default=1.0)
    calibrate.add_argument("--recovery-decisions", type=int, default=1)
    calibrate.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate policies with static and MPC benchmark methods",
    )
    evaluate.add_argument("--model", type=_named_dataset, action="append", required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--budgets", type=_csv_ints, required=True)
    evaluate.add_argument("--candidates", type=_csv_strings, default=MPC_ACTION_NAMES)
    evaluate.add_argument("--trace-kinds", type=_csv_strings, default=TRACE_KINDS)
    evaluate.add_argument(
        "--benchmark-methods",
        type=_csv_strings,
        default=("girard", "scott", "pca", "combastel", "mpc_terminal_full_width"),
    )
    evaluate.add_argument("--length", type=int, default=None)
    evaluate.add_argument("--horizon", type=int, default=1)
    evaluate.add_argument("--beam-width", type=int, default=4)
    evaluate.add_argument("--prediction-step-seconds", type=float, default=0.1)
    evaluate.add_argument("--workers", type=int, default=1)
    evaluate.add_argument("--expected-cell-count", type=int)
    evaluate.add_argument(
        "--comparison",
        type=_policy_comparison,
        action="append",
        default=[],
        help="exploratory: repeat NAME=CHALLENGER:REFERENCE",
    )

    select = subparsers.add_parser(
        "select-challenger", help="exploratory: apply the bounded promotion gate",
    )
    select.add_argument("--evaluation", type=Path, required=True)
    select.add_argument("--model", type=_named_dataset, action="append", required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--expected-cell-count", type=int, default=60)

    report = subparsers.add_parser(
        "report-paper", help="validate and join primary and MPC add-on evaluations",
    )
    report.add_argument("--primary", type=Path, required=True)
    report.add_argument("--mpc-addon", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    dispatch = {
        "generate": _dispatch_generate,
        "collect": _dispatch_collect,
        "train": _dispatch_train,
        "calibrate-dart": _dispatch_calibrate_dart,
        "evaluate": _dispatch_evaluate,
        "select-challenger": _dispatch_select_challenger,
        "report-paper": _dispatch_report_paper,
    }
    dispatch[args.command](args)


def _dispatch_generate(args: argparse.Namespace) -> None:
    store = generate_random_waypoint_trace_store(RandomWaypointTraceStoreConfig(
        output=args.output,
        event_count=args.event_count,
        conditions=tuple(args.conditions),
        seed_start=args.seed_start,
        seed_count=args.seed_count,
    ))
    print(f"Random-waypoint trace store complete: {store.root} ({len(store.traces)} traces)")


def _dispatch_collect(args: argparse.Namespace) -> None:
    output = run_learning_collection(LearningCollectionConfig(
        output=args.output,
        trace_store=args.trace_store,
        budgets=tuple(args.budgets),
        candidate_names=tuple(args.candidates),
        train_seeds=args.train_seeds,
        validation_seeds=args.validation_seeds,
        test_seeds=args.test_seeds,
        seed_start=args.seed_start,
        workers=args.workers,
        collection_mode=args.collection_mode,
        dart_calibration=args.dart_calibration,
        disturbance_seed=args.disturbance_seed,
    ))
    print(f"Learning dataset complete: {output}")


def _dispatch_train(args: argparse.Namespace) -> None:
    output = run_reducer_training(ReducerTrainingConfig(
        datasets=tuple(args.dataset),
        output=args.output,
        objective=args.objective,
        temperature_grid=args.temperature_grid,
        temperature_from=args.temperature_from,
        feasibility_penalty=args.feasibility_penalty,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        budget_filter=(
            tuple(args.budget_filter) if args.budget_filter is not None else None
        ),
    ))
    print(f"Reducer scorer complete: {output}")


def _dispatch_calibrate_dart(args: argparse.Namespace) -> None:
    output = run_dart_calibration_artifact(DartCalibrationArtifactConfig(
        model=args.model,
        dataset=args.dataset,
        output=args.output,
        split=args.split,
        calibration=DartCalibrationConfig(
            regret_cap_quantile=args.regret_cap_quantile,
            direction_pseudocount=args.direction_pseudocount,
            recovery_decisions=args.recovery_decisions,
        ),
    ))
    print(f"DART calibration complete: {output}")


def _dispatch_evaluate(args: argparse.Namespace) -> None:
    unknown_traces = set(args.trace_kinds) - set(TRACE_KINDS)
    if unknown_traces:
        raise ValueError(f"unknown fixed robot-arm traces: {sorted(unknown_traces)}")
    models = tuple(args.model)
    config = FixedPolicyEvaluationConfig(
        output=args.output,
        model_names=tuple(model.name for model in models),
        trace_kinds=tuple(args.trace_kinds),
        budgets=tuple(args.budgets),
        benchmark_methods=tuple(args.benchmark_methods),
        candidate_names=tuple(args.candidates),
        length=args.length,
        horizon=args.horizon,
        beam_width=args.beam_width,
        prediction_step_seconds=args.prediction_step_seconds,
        comparisons=tuple(args.comparison),
        expected_cell_count=args.expected_cell_count,
    )
    run_policy_evaluation_from_models(
        config, models, source_sha256=pzr_source_sha256(), workers=args.workers,
    )
    print(f"Policy evaluation complete: {args.output}")


def _dispatch_select_challenger(args: argparse.Namespace) -> None:
    selection = run_challenger_selection(ChallengerSelectionConfig(
        evaluation=args.evaluation,
        models=tuple(args.model),
        output=args.output,
        expected_cell_count=args.expected_cell_count,
    ))
    winner = selection["winner"]
    print(
        "No challenger passed; stop method expansion"
        if winner is None else f"Selected challenger: {winner['challenger']}"
    )


def _dispatch_report_paper(args: argparse.Namespace) -> None:
    joined = write_paper_reports(PaperReportConfig(
        primary=args.primary,
        mpc_addon=args.mpc_addon,
        output=args.output,
    ))
    print(f"Joined paper dataset complete: {args.output} ({len(joined)} cells)")


if __name__ == "__main__":
    main()
