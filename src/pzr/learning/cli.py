"""Staged commands for RTLola reducer-cost learning experiments."""

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

from pzr.learning.artifacts import load_reducer_cost_dataset, write_reducer_cost_dataset
from pzr.learning.dart import DartCalibration, calibrate_dart
from pzr.learning.dataset import ReducerCostDataset
from pzr.learning.diagnostics import candidate_diagnostics, dataset_diagnostics, validation_metrics
from pzr.learning.provenance import model_sha256, pzr_source_sha256, sha256_files
from pzr.learning.ranker import ReducerPolicy, ReducerTrainingResult, train_reducer_policy
from pzr.learning.reporting import write_dart_calibration_plot, write_training_plots
from pzr.rtlola.actions import MPC_ACTION_NAMES, default_action_catalog
from pzr.rtlola.binding import BINDING_BUILD_PROFILE, BINDING_REVISION, INTERPRETER_REVISION
from pzr.rtlola.engine import RtlolaEvent
from pzr.rtlola.features import RTL_RANKING_FEATURE_SCHEMA
from pzr.rtlola.learned_policy import RtlolaReducerPolicy
from pzr.rtlola.learning_data import collect_teacher_episode, write_collected_dataset
from pzr.rtlola.learning_evaluation import FixedLearningEvaluationConfig, run_fixed_learning_evaluation
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


def _csv_floats(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part) for part in _csv_strings(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if any(item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError("temperatures must be positive")
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
    generate = subparsers.add_parser("generate", help="Build a validated random-waypoint trace store")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--event-count", type=int, required=True)
    generate.add_argument("--conditions", type=_csv_strings, default=("random_waypoint",))
    generate.add_argument("--seed-start", type=int, default=0)
    generate.add_argument("--seed-count", type=int, required=True)

    collect = subparsers.add_parser("collect", help="Label a validated trace store")
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

    train = subparsers.add_parser("train", help="Train a fixed-catalog reducer scorer")
    train.add_argument("--dataset", type=_named_path, action="append", required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--objective", choices=("pairwise", "soft-kl"), required=True)
    train.add_argument("--temperature-grid", type=_csv_floats)
    train.add_argument("--temperature-from", type=Path)
    train.add_argument("--feasibility-penalty", type=float, default=1.0)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--seed", type=int, default=42)

    calibrate = subparsers.add_parser("calibrate-dart", help="Fit held-out novice confusion")
    calibrate.add_argument("--model", type=Path, required=True)
    calibrate.add_argument("--dataset", type=_named_path, required=True)
    calibrate.add_argument("--split", default="validation")
    calibrate.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate fixed robot-arm generalization")
    evaluate.add_argument("--model", type=_named_path, action="append", required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--budgets", type=_csv_ints, required=True)
    evaluate.add_argument("--candidates", type=_csv_strings, default=MPC_ACTION_NAMES)
    evaluate.add_argument("--trace-kinds", type=_csv_strings, default=TRACE_KINDS)
    evaluate.add_argument(
        "--baselines", type=_csv_strings,
        default=("girard", "scott", "pca", "combastel", "mpc_terminal_full_width"),
    )
    evaluate.add_argument("--length", type=int, default=None)
    evaluate.add_argument("--horizon", type=int, default=1)
    evaluate.add_argument("--beam-width", type=int, default=4)
    evaluate.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    commands = {
        "generate": run_generate,
        "collect": run_collect,
        "train": run_train,
        "calibrate-dart": run_calibrate_dart,
        "evaluate": run_evaluate,
    }
    commands[args.command](args)


def run_generate(args: argparse.Namespace) -> None:
    store = generate_random_waypoint_trace_store(RandomWaypointTraceStoreConfig(
        output=args.output,
        event_count=args.event_count,
        conditions=tuple(args.conditions),
        seed_start=args.seed_start,
        seed_count=args.seed_count,
    ))
    print(f"Random-waypoint trace store complete: {store.root} ({len(store.traces)} traces)")


def run_collect(args: argparse.Namespace) -> None:
    if not args.budgets:
        raise ValueError("at least one budget is required")
    if args.train_seeds < 1 or min(args.validation_seeds, args.test_seeds) < 0:
        raise ValueError("training needs a seed and split counts cannot be negative")
    if args.seed_start < 0 or args.disturbance_seed < 0:
        raise ValueError("seed values must be non-negative")
    if args.workers < 1:
        raise ValueError("collection workers must be positive")
    if args.collection_mode == "dart" and args.dart_calibration is None:
        raise ValueError("DART collection requires --dart-calibration")
    if args.collection_mode == "teacher" and args.dart_calibration is not None:
        raise ValueError("teacher collection does not accept --dart-calibration")
    calibration = DartCalibration.load(args.dart_calibration) if args.dart_calibration else None
    calibration_hash = (
        sha256_files((args.dart_calibration / "calibration.json", args.dart_calibration / "dart_calibration.csv"))
        if args.dart_calibration else None
    )
    candidate_names = tuple(args.candidates)
    default_action_catalog(candidate_names)
    source_sha256 = pzr_source_sha256()
    if calibration is not None:
        if calibration.candidate_names != candidate_names:
            raise ValueError("DART calibration candidate catalog differs")
        if not set(args.budgets) <= set(calibration.budgets):
            raise ValueError("DART calibration does not cover every collection budget")
        expected_context = {
            "candidate_names": list(candidate_names),
            "feature_schema": _feature_schema_payload(),
            "binding_revision": BINDING_REVISION,
            "interpreter_revision": INTERPRETER_REVISION,
            "binding_build_profile": BINDING_BUILD_PROFILE,
            "pzr_source_sha256": source_sha256,
        }
        mismatched = [
            name for name, value in expected_context.items()
            if calibration.context.get(name) != value
        ]
        if mismatched:
            raise ValueError("DART calibration context differs for: " + ", ".join(sorted(mismatched)))
    trace_store = load_random_waypoint_trace_store(args.trace_store)
    scenario = scenario_by_name("robot_arm")
    jobs: list[_CollectionShardJob] = []
    trace_records = []
    for split, seed in _split_seeds(args):
        for stored_trace in trace_store.traces_for_seed(seed):
            trace = stored_trace.trace
            trace_records.append({
                "trace_id": stored_trace.trace_id,
                "split": split,
                "condition": stored_trace.condition,
                "seed": seed,
                "trace_sha256": trace.metadata.trace_sha256,
                "trace_store_relative_path": str(stored_trace.relative_path),
            })
            for budget in args.budgets:
                shard_dir = args.output / "shards" / split / stored_trace.condition / f"seed-{seed}" / f"budget-{budget}"
                identity = {
                    "collection_shard": True,
                    "scenario": scenario.name,
                    "collection_mode": args.collection_mode,
                    "event_count": trace_store.event_count,
                    "trace_id": stored_trace.trace_id,
                    "split": split,
                    "condition": stored_trace.condition,
                    "seed": seed,
                    "budget": budget,
                    "candidate_names": list(candidate_names),
                    "feature_schema": _feature_schema_payload(),
                    "trace_sha256": trace.metadata.trace_sha256,
                    "trace_store_manifest_sha256": trace_store.manifest_sha256,
                    "dart_calibration_sha256": calibration_hash,
                    "disturbance_seed": args.disturbance_seed if calibration else None,
                    "binding_revision": BINDING_REVISION,
                    "interpreter_revision": INTERPRETER_REVISION,
                    "binding_build_profile": BINDING_BUILD_PROFILE,
                    "pzr_source_sha256": source_sha256,
                }
                jobs.append(_CollectionShardJob(
                    directory=shard_dir,
                    identity=identity,
                    scenario_name=scenario.name,
                    events=trace.events,
                    trace_id=stored_trace.trace_id,
                    split=split,
                    condition=stored_trace.condition,
                    seed=seed,
                    budget=budget,
                    candidate_names=candidate_names,
                    collection_mode=args.collection_mode,
                    dart_calibration=args.dart_calibration,
                    dart_calibration_sha256=calibration_hash,
                    disturbance_seed=args.disturbance_seed,
                ))
    _run_collection_jobs(jobs, args.workers)
    loaded_shards = [_load_collection_shard(job) for job in jobs]
    dataset = ReducerCostDataset.concatenate([item[0] for item in loaded_shards])
    if dataset.num_samples == 0:
        raise ValueError("teacher collection produced no reduction decisions")
    sample_metadata = pd.concat([item[1] for item in loaded_shards], ignore_index=True)
    metadata = {
        "scenario": scenario.name,
        "collection_mode": args.collection_mode,
        "event_count": trace_store.event_count,
        "budgets": list(args.budgets),
        "conditions": list(trace_store.conditions),
        "seed_start": args.seed_start,
        "trace_store": str(trace_store.root),
        "trace_store_manifest_sha256": trace_store.manifest_sha256,
        "binding_revision": BINDING_REVISION,
        "interpreter_revision": INTERPRETER_REVISION,
        "binding_build_profile": BINDING_BUILD_PROFILE,
        "dart_calibration_sha256": calibration_hash,
        "disturbance_seed": args.disturbance_seed if calibration else None,
        "feature_schema": _feature_schema_payload(),
        "pzr_source_sha256": source_sha256,
        "shard_count": len(loaded_shards),
        "traces": trace_records,
    }
    write_reducer_cost_dataset(dataset, args.output / "dataset", sample_metadata, metadata)
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
    collection_mode: str
    dart_calibration: Path | None
    dart_calibration_sha256: str | None
    disturbance_seed: int


def _run_collection_jobs(jobs: Sequence[_CollectionShardJob], workers: int) -> None:
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
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
        tuple(executor.map(_run_collection_shard, missing))


def _run_collection_shard(job: _CollectionShardJob) -> None:
    calibration = DartCalibration.load(job.dart_calibration) if job.dart_calibration else None
    samples = collect_teacher_episode(
        scenario=scenario_by_name(job.scenario_name),
        events=job.events,
        trace_id=job.trace_id,
        split=job.split,
        condition=job.condition,
        seed=job.seed,
        budget=job.budget,
        candidate_names=job.candidate_names,
        collection_mode=job.collection_mode,
        dart_calibration=calibration,
        dart_calibration_sha256=job.dart_calibration_sha256,
        disturbance_seed=job.disturbance_seed,
    )
    write_collected_dataset(samples, job.directory, job.identity, candidate_names=job.candidate_names)


def _load_collection_shard(job: _CollectionShardJob) -> tuple[ReducerCostDataset, pd.DataFrame]:
    dataset, sample_metadata, manifest = load_reducer_cost_dataset(job.directory)
    _validate_shard_manifest(manifest, job.identity)
    return dataset, sample_metadata


def _validate_shard_manifest(manifest: dict[str, object], expected: dict[str, object]) -> None:
    mismatched = [name for name, value in expected.items() if manifest.get(name) != value]
    if mismatched:
        raise ValueError("learning collection shard identity differs for: " + ", ".join(sorted(mismatched)))


def _feature_schema_payload() -> dict[str, object]:
    return {
        "name": RTL_RANKING_FEATURE_SCHEMA.name,
        "version": RTL_RANKING_FEATURE_SCHEMA.version,
        "feature_names": list(RTL_RANKING_FEATURE_SCHEMA.feature_names),
        "log1p_features": list(RTL_RANKING_FEATURE_SCHEMA.log1p_features),
    }


def run_train(args: argparse.Namespace) -> None:
    inputs = tuple(args.dataset)
    if len({item.name for item in inputs}) != len(inputs):
        raise ValueError("named training datasets must have unique names")
    loaded = [load_reducer_cost_dataset(item.path) for item in inputs]
    _validate_named_datasets(inputs, loaded)
    dataset = ReducerCostDataset.concatenate([item[0] for item in loaded])
    metadata_frames = []
    for named_input, (_, metadata, _) in zip(inputs, loaded):
        frame = metadata.copy()
        frame.insert(0, "dataset_label", named_input.name)
        metadata_frames.append(frame)
    sample_metadata = pd.concat(metadata_frames, ignore_index=True)
    temperatures, temperature_source_hash = _training_temperatures(args, dataset)
    candidates: list[tuple[float | None, ReducerPolicy, ReducerTrainingResult]] = []
    for temperature in temperatures:
        policy, result = train_reducer_policy(
            dataset,
            RTL_RANKING_FEATURE_SCHEMA,
            objective=args.objective,
            temperature=temperature,
            feasibility_penalty=args.feasibility_penalty,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
            seed=args.seed,
        )
        candidates.append((temperature, policy, result))
    selected_index = min(range(len(candidates)), key=lambda index: _temperature_selection_key(candidates[index]))
    selected_temperature, policy, result = candidates[selected_index]
    args.output.mkdir(parents=True, exist_ok=True)
    policy.save(args.output)
    temperature_frame = pd.DataFrame([
        {
            "temperature": temperature,
            "selected": index == selected_index,
            "infeasible_selection_count": candidate_result.val_metrics.infeasible_selection_count,
            "mean_selected_normalized_regret": candidate_result.val_metrics.mean_chosen_normalized_regret,
            "max_selected_normalized_regret": candidate_result.val_metrics.max_chosen_normalized_regret,
            "validation_kl": candidate_result.val_metrics.kl_divergence,
            "validation_loss": candidate_result.val_loss_history[candidate_result.best_epoch],
        }
        for index, (temperature, _, candidate_result) in enumerate(candidates)
    ])
    temperature_frame.to_csv(args.output / "temperature_selection.csv", index=False)
    write_training_plots(
        temperature_frame, result, args.output,
    )
    payload = {
        "schema": "pzr.reducer-training.v3",
        "datasets": [
            {
                "name": item.name,
                "path": str(item.path),
                "sha256": _dataset_sha256(item.path),
            }
            for item in inputs
        ],
        "candidate_names": list(policy.candidate_names),
        "feature_schema": asdict(policy.feature_schema),
        "objective_contract": policy.objective_contract,
        "selected_temperature": selected_temperature,
        "temperature_candidates": [value for value, _, _ in candidates],
        "temperature_source_model_sha256": temperature_source_hash,
        "temperature_selection": "infeasible_count_mean_regret_max_regret_kl_lower_temperature",
        "training": asdict(result),
        "seed": args.seed,
        "binding_revision": BINDING_REVISION,
        "interpreter_revision": INTERPRETER_REVISION,
        "binding_build_profile": BINDING_BUILD_PROFILE,
        "pzr_source_sha256": pzr_source_sha256(),
    }
    (args.output / "training.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    validation_metrics(policy, dataset, sample_metadata).to_csv(args.output / "validation_metrics.csv", index=False)
    dataset_diagnostics(dataset, sample_metadata).to_csv(args.output / "dataset_diagnostics.csv", index=False)
    candidate_diagnostics(policy, dataset, sample_metadata).to_csv(args.output / "candidate_diagnostics.csv", index=False)
    print(f"Reducer scorer complete: {args.output}")


def _training_temperatures(
    args: argparse.Namespace,
    dataset: ReducerCostDataset,
) -> tuple[tuple[float | None, ...], str | None]:
    if args.objective == "pairwise":
        if args.temperature_grid is not None or args.temperature_from is not None:
            raise ValueError("pairwise training does not accept temperature options")
        return (None,), None
    if (args.temperature_grid is None) == (args.temperature_from is None):
        raise ValueError("soft-KL training needs exactly one of --temperature-grid or --temperature-from")
    if args.temperature_grid is not None:
        if len(set(args.temperature_grid)) != len(args.temperature_grid):
            raise ValueError("temperature grid values must be unique")
        return tuple(args.temperature_grid), None
    source = ReducerPolicy.load(args.temperature_from)
    if source.objective_contract.get("schema") != "pzr.reducer-objective.soft-kl-v1":
        raise ValueError("--temperature-from must reference a soft-KL model")
    if source.candidate_names != dataset.candidate_names:
        raise ValueError("temperature source candidate catalog differs")
    if source.feature_schema.feature_names != dataset.feature_names:
        raise ValueError("temperature source feature schema differs")
    return (float(source.objective_contract["temperature"]),), model_sha256(args.temperature_from)


def _temperature_selection_key(
    candidate: tuple[float | None, ReducerPolicy, ReducerTrainingResult],
) -> tuple[float, ...]:
    temperature, _, result = candidate
    metrics = result.val_metrics
    return (
        float(metrics.infeasible_selection_count),
        metrics.mean_chosen_normalized_regret,
        metrics.max_chosen_normalized_regret,
        metrics.kl_divergence if pd.notna(metrics.kl_divergence) else 0.0,
        float(temperature or 0.0),
    )


def _validate_named_datasets(
    inputs: Sequence[NamedPath],
    loaded: Sequence[tuple[ReducerCostDataset, pd.DataFrame, dict[str, object]]],
) -> None:
    first_dataset, _, first_manifest = loaded[0]
    for named_input, (dataset, _, manifest) in zip(inputs, loaded):
        if dataset.candidate_names != first_dataset.candidate_names:
            raise ValueError(f"dataset {named_input.name!r} candidate catalog differs")
        if dataset.feature_names != first_dataset.feature_names:
            raise ValueError(f"dataset {named_input.name!r} feature schema differs")
        if manifest.get("cost_contract") != first_manifest.get("cost_contract"):
            raise ValueError(f"dataset {named_input.name!r} cost schema differs")
        if dataset.indices_for_split("train").size == 0:
            raise ValueError(f"dataset {named_input.name!r} has no training samples")
        if dataset.indices_for_split("validation").size == 0:
            raise ValueError(f"dataset {named_input.name!r} has no validation samples")


def run_calibrate_dart(args: argparse.Namespace) -> None:
    policy = ReducerPolicy.load(args.model)
    if policy.objective_contract.get("schema") != "pzr.reducer-objective.soft-kl-v1":
        raise ValueError("DART calibration requires a soft-KL novice model")
    dataset, metadata, manifest = load_reducer_cost_dataset(args.dataset.path)
    dataset_hash = _dataset_sha256(args.dataset.path)
    context = {
        "model_sha256": model_sha256(args.model),
        "dataset_name": args.dataset.name,
        "dataset_sha256": dataset_hash,
        "split": args.split,
        "candidate_names": list(dataset.candidate_names),
        "feature_schema": _feature_schema_payload(),
        "cost_contract": manifest["cost_contract"],
        "binding_revision": BINDING_REVISION,
        "interpreter_revision": INTERPRETER_REVISION,
        "binding_build_profile": BINDING_BUILD_PROFILE,
        "pzr_source_sha256": pzr_source_sha256(),
    }
    calibration, diagnostics = calibrate_dart(
        policy, dataset, metadata, split=args.split, context=context,
    )
    calibration.save(args.output, diagnostics)
    write_dart_calibration_plot(diagnostics, args.output / "dart_calibration.png")
    print(f"DART calibration complete: {args.output}")


def _dataset_sha256(path: Path) -> str:
    return sha256_files((
        path / "manifest.json", path / "samples.npz", path / "samples.csv", path / "candidate_costs.csv",
    ))


def _write_collection_summaries(metadata: pd.DataFrame, output: Path) -> None:
    groups = ["split", "condition", "budget"]
    summary = metadata.groupby(groups, dropna=False).agg(
        sample_count=("sample_id", "size"),
        mean_evaluated_leaves=("evaluated_leaves", "mean"),
        teacher_reducer_failure_count=("teacher_reducer_failure_count", "sum"),
        teacher_infeasible_candidate_count=("teacher_infeasible_candidate_count", "sum"),
        execution_fallback_count=("execution_fallback_used", "sum"),
        disturbed_count=("disturbed", "sum"),
        mean_disturbance_probability=("disturbance_probability", "mean"),
        mean_infeasible_probability_redirected=("infeasible_probability_redirected", "mean"),
        mean_sampled_normalized_regret=("sampled_normalized_regret", "mean"),
        median_sampled_normalized_regret=("sampled_normalized_regret", "median"),
        max_sampled_normalized_regret=("sampled_normalized_regret", "max"),
    ).reset_index()
    summary["realized_disturbance_rate"] = summary["disturbed_count"] / summary["sample_count"]
    summary.to_csv(output / "collection_summary.csv", index=False)
    for column, filename in (
        ("teacher_action", "teacher_action_counts.csv"),
        ("executed_action", "executed_action_counts.csv"),
    ):
        metadata.groupby([*groups, column], dropna=False).size().rename("count").reset_index().to_csv(output / filename, index=False)
    metadata.groupby(
        [*groups, "teacher_action", "executed_action"], dropna=False,
    ).size().rename("count").reset_index().to_csv(
        output / "teacher_executed_confusion.csv", index=False,
    )


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
        item.name: RtlolaReducerPolicy(ReducerPolicy.load(item.path), default_action_catalog(candidate_names))
        for item in model_inputs
    }
    model_hashes = {item.name: model_sha256(item.path) for item in model_inputs}
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
        model_directories={item.name: item.path for item in model_inputs},
        workers=args.workers,
    )
    print(f"Learning evaluation complete: {args.output}")


def _split_seeds(args: argparse.Namespace) -> Sequence[tuple[str, int]]:
    counts = (("train", args.train_seeds), ("validation", args.validation_seeds), ("test", args.test_seeds))
    offset = args.seed_start
    result = []
    for split, count in counts:
        result.extend((split, offset + index) for index in range(count))
        offset += count
    return tuple(result)


if __name__ == "__main__":
    main()
