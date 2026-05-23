"""CoRL-style closed-loop robotics intervention experiment suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import tarfile
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from pzr.control.policies import ReductionDecision
from pzr.experiments.benchmark import (
    BenchmarkConfig,
    MethodSpec,
    focused_geometry_reducer_factories,
    wide_rollout_reducer_factories,
    _make_policy,
    _method_candidate_names,
)
from pzr.learning.distill_cli import train_policy
from pzr.learning.features import DECISION_FEATURE_NAMES, DECISION_FEATURE_SCHEMA_VERSION, decision_feature_values
from pzr.monitoring.base import TriggerSpec, Verdict, evaluate_triggers
from pzr.reduction.paper_reducers import CombastelReducer, GirardReducer, PcaReducer
from pzr.reduction.reducers import BoxReducer, ProtectedReducer, ScoredKeepReducer
from pzr.robotics import (
    InterventionManager,
    IrosGateMonitor,
    IrosObservation,
    NoisySensorModel,
    make_env_client,
    preflight_safe_control_gym,
)
from pzr.robotics.iros import IROS_STREAM_NAMES, iros_stream_values
from pzr.robotics.safe_control_gym import IrosEnvClient, IrosEnvSnapshot


@dataclass(frozen=True)
class CorlProfile:
    """Configuration knobs for a CoRL robotics run."""

    budget: int
    horizon: int
    max_steps: int
    train_seeds: int
    eval_seeds: int
    dagger_iterations: int
    bootstrap_samples: int
    sensor_bias_bound: float
    sensor_noise_bound: float
    stream_memory_decay: float
    fallback_hold_steps: int
    distill_epochs: int
    distill_batch_size: int


PROFILES = {
    "smoke": CorlProfile(
        budget=8,
        horizon=2,
        max_steps=30,
        train_seeds=1,
        eval_seeds=1,
        dagger_iterations=1,
        bootstrap_samples=50,
        sensor_bias_bound=0.01,
        sensor_noise_bound=0.02,
        stream_memory_decay=0.65,
        fallback_hold_steps=2,
        distill_epochs=3,
        distill_batch_size=4,
    ),
    "overnight": CorlProfile(
        budget=8,
        horizon=6,
        max_steps=1000,
        train_seeds=20,
        eval_seeds=50,
        dagger_iterations=3,
        bootstrap_samples=5000,
        sensor_bias_bound=0.015,
        sensor_noise_bound=0.03,
        stream_memory_decay=0.85,
        fallback_hold_steps=2,
        distill_epochs=150,
        distill_batch_size=64,
    ),
    "paper": CorlProfile(
        budget=8,
        horizon=6,
        max_steps=1000,
        train_seeds=40,
        eval_seeds=100,
        dagger_iterations=4,
        bootstrap_samples=10000,
        sensor_bias_bound=0.015,
        sensor_noise_bound=0.03,
        stream_memory_decay=0.85,
        fallback_hold_steps=2,
        distill_epochs=300,
        distill_batch_size=64,
    ),
}

HEADLINE_METRICS = (
    "task_completed",
    "gates_passed",
    "collision_episode",
    "constraint_violation_episode",
    "fallback_activation_count",
    "fallback_duration_fraction",
    "spurious_intervention_rate",
    "justified_intervention_rate",
    "missed_violation_rate",
    "time_to_target",
    "mean_reducer_latency_ms",
)

DAGGER_EXPERTS = ("mpc_wide_fixed_girard", "mpc_focused_sequence")
CORL_METHOD_SETS = ("core", "extended")
LEARNED_MODES = ("none", "dagger", "checkpoint")
LABEL_DIVERSITY_MIN_CLASSES = 3
LABEL_DIVERSITY_MAX_TOP_FRACTION = 0.9
FAILURE_EVENT_COLUMNS = (
    "phase",
    "method",
    "method_kind",
    "seed",
    "step",
    "event_type",
    "exception_type",
    "message",
    "traceback",
    "generator_count",
    "candidate_reducer_names",
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)
    if args.preflight:
        result = preflight_safe_control_gym(
            profile=args.profile,
            safe_control_gym_root=args.safe_control_gym_root,
            safe_control_python=args.safe_control_python,
            safe_control_config=args.safe_control_config,
            safe_control_controller_mode=args.safe_control_controller_mode,
            allow_debug_pid=args.allow_debug_pid,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0 if result.ok else 2
    if args.calibration:
        run_calibration_suite(args)
    elif args.controller_validation:
        run_controller_validation(args)
    else:
        run_corl_suite(args)
    return 0


def run_controller_validation(args: argparse.Namespace) -> Path:
    """Run a nominal-only controller validation suite."""

    profile = _profile_from_args(args, controller_validation=True)
    _validate_control_mode(args, profile)
    out_dir = _resolve_out_dir(args.out)
    if out_dir.exists():
        if not args.force:
            raise FileExistsError(f"output directory already exists: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preflight = preflight_safe_control_gym(
        profile=args.profile,
        safe_control_gym_root=args.safe_control_gym_root,
        safe_control_python=args.safe_control_python,
        safe_control_config=args.safe_control_config,
        safe_control_controller_mode=args.safe_control_controller_mode,
        allow_debug_pid=args.allow_debug_pid,
    )
    if not preflight.ok:
        raise RuntimeError(
            "CoRL controller-validation preflight failed:\n"
            + "\n".join(f"- {message}" for message in preflight.messages)
        )

    manifest: dict[str, Any] = {
        "suite": "pzr_corl_controller_validation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "profile_config": asdict(profile),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "preflight": preflight.to_dict(),
        "steps": [
            {
                "kind": "nominal_controller_validation",
                "episodes": profile.eval_seeds,
                "eval_seed_start": 0,
            }
        ],
    }

    episode_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    monitor_rows: list[dict[str, Any]] = []
    for seed in range(profile.eval_seeds):
        client = make_env_client(
            profile=args.profile,
            safe_control_gym_root=args.safe_control_gym_root,
            safe_control_python=args.safe_control_python,
            safe_control_config=args.safe_control_config,
            safe_control_controller_mode=args.safe_control_controller_mode,
            allow_debug_pid=args.allow_debug_pid,
        )
        try:
            episode, interventions, monitor, _, _ = _run_episode(
                client,
                profile,
                "nominal_no_monitor",
                seed,
                phase="controller_validation",
            )
        finally:
            client.close()
        episode_rows.append(episode)
        intervention_rows.extend(interventions)
        monitor_rows.extend(monitor)

    raw = pd.DataFrame(episode_rows)
    interventions = pd.DataFrame(intervention_rows)
    monitor = pd.DataFrame(monitor_rows)
    failures = pd.DataFrame(columns=FAILURE_EVENT_COLUMNS)
    summary = _controller_validation_summary(raw)
    notes = _controller_validation_notes(raw, summary)

    raw.to_csv(out_dir / "raw_episodes.csv", index=False)
    interventions.to_csv(out_dir / "intervention_timeseries.csv", index=False)
    monitor.to_csv(out_dir / "monitor_timeseries.csv", index=False)
    failures.to_csv(out_dir / "failure_events.csv", index=False)
    summary.to_csv(out_dir / "controller_validation_summary.csv", index=False)
    (out_dir / "analysis_notes.json").write_text(
        json.dumps(_json_safe(notes), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "config.json").write_text(
        json.dumps(
            _json_safe({"profile": asdict(profile), "args": vars(args), "mode": "controller_validation"}),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_artifact_index(out_dir, out_dir / "artifact_index.csv")
    if not args.no_archive:
        _write_archive(out_dir, out_dir.with_suffix(".tar.gz"))
    return out_dir


def run_calibration_suite(args: argparse.Namespace) -> Path:
    """Run a compact monitor/task calibration sweep."""

    profile = _profile_from_args(args)
    profile = replace(profile, train_seeds=0, eval_seeds=args.calibration_seeds, max_steps=args.calibration_max_steps)
    _validate_control_mode(args, profile)
    out_dir = _resolve_out_dir(args.out)
    if out_dir.exists():
        if not args.force:
            raise FileExistsError(f"output directory already exists: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = _calibration_configs(args, profile)
    preflight_records = _preflight_calibration_configs(args, configs)
    episode_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    methods: tuple[MethodSpec | str, ...] = ("nominal_no_monitor", "reference_unbounded", *_headline_methods("core"))
    _log_progress(out_dir, "calibration_start", config_count=len(configs), seeds=profile.eval_seeds)
    for config in configs:
        run_profile = config["profile"]
        safe_control_config = config["safe_control_config"]
        for method in methods:
            method_name = method if isinstance(method, str) else method.name
            _log_progress(
                out_dir,
                "calibration_method_start",
                config_id=config["config_id"],
                method=method_name,
                safe_control_config=safe_control_config or "",
            )
            for seed in range(profile.eval_seeds):
                client = make_env_client(
                    profile=args.profile,
                    safe_control_gym_root=args.safe_control_gym_root,
                    safe_control_python=args.safe_control_python,
                    safe_control_config=safe_control_config,
                    safe_control_controller_mode=args.safe_control_controller_mode,
                    allow_debug_pid=args.allow_debug_pid,
                )
                try:
                    episode, _, _, _, failures = _run_episode(
                        client,
                        run_profile,
                        method,
                        seed,
                        phase="calibration",
                    )
                finally:
                    client.close()
                episode["config_id"] = config["config_id"]
                episode["safe_control_config"] = safe_control_config or ""
                episode_rows.append(episode)
                for failure in failures:
                    failure["config_id"] = config["config_id"]
                    failure_rows.append(failure)
    raw = pd.DataFrame(episode_rows)
    failures = pd.DataFrame(failure_rows, columns=("config_id", *FAILURE_EVENT_COLUMNS))
    summary = _calibration_summary(raw)
    recommendations = _calibration_recommendations(summary, failures)

    raw.to_csv(out_dir / "calibration_runs.csv", index=False)
    summary.to_csv(out_dir / "calibration_summary.csv", index=False)
    failures.to_csv(out_dir / "failure_events.csv", index=False)
    (out_dir / "calibration_recommendations.json").write_text(
        json.dumps(_json_safe(recommendations), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "analysis_notes.json").write_text(
        json.dumps(_json_safe(recommendations), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "config.json").write_text(
        json.dumps(
            _json_safe(
                {
                    "profile": asdict(profile),
                    "args": vars(args),
                    "mode": "calibration",
                    "preflight": preflight_records,
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_artifact_index(out_dir, out_dir / "artifact_index.csv")
    if not args.no_archive:
        _write_archive(out_dir, out_dir.with_suffix(".tar.gz"))
    _log_progress(out_dir, "calibration_complete", out=str(out_dir))
    return out_dir


def run_corl_suite(args: argparse.Namespace) -> Path:
    """Run the CoRL robotics suite and return the output directory."""

    if args.learned_mode in {"dagger", "checkpoint"}:
        _require_torch()
    profile = _profile_from_args(args)
    _validate_control_mode(args, profile)
    out_dir = _resolve_out_dir(args.out)
    if out_dir.exists():
        if not args.force:
            raise FileExistsError(f"output directory already exists: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "learning").mkdir(parents=True, exist_ok=True)

    preflight = preflight_safe_control_gym(
        profile=args.profile,
        safe_control_gym_root=args.safe_control_gym_root,
        safe_control_python=args.safe_control_python,
        safe_control_config=args.safe_control_config,
        safe_control_controller_mode=args.safe_control_controller_mode,
        allow_debug_pid=args.allow_debug_pid,
    )
    if not preflight.ok:
        raise RuntimeError(
            "CoRL preflight failed:\n"
            + "\n".join(f"- {message}" for message in preflight.messages)
        )

    manifest: dict[str, Any] = {
        "suite": "pzr_corl_suite",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "profile_config": asdict(profile),
        "dagger_expert": args.dagger_expert,
        "method_set": args.method_set,
        "learned_mode": args.learned_mode,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "preflight": preflight.to_dict(),
        "steps": [],
    }

    _log_progress(
        out_dir,
        "suite_start",
        profile=args.profile,
        dagger_expert=args.dagger_expert,
        method_set=args.method_set,
        learned_mode=args.learned_mode,
    )
    checkpoint: Path | None = None
    label_quality = _dagger_label_quality(pd.DataFrame())
    failure_rows: list[dict[str, Any]] = []
    if args.learned_mode == "dagger":
        training_rows, training_failures = _run_dagger_training(args, profile, out_dir)
        failure_rows.extend(training_failures)
        label_summary = _dagger_label_summary(training_rows)
        label_summary.to_csv(out_dir / "learning" / "dagger_label_summary.csv", index=False)
        label_quality = _dagger_label_quality(label_summary)
        checkpoint = out_dir / "learning" / "dagger_final.pt"
        manifest["steps"].append(
            {
                "kind": "dagger_training",
                "rows": int(training_rows.shape[0]),
                "checkpoint": "learning/dagger_final.pt",
                "expert": args.dagger_expert,
                "label_quality": label_quality,
            }
        )
    elif args.learned_mode == "checkpoint":
        if args.learned_checkpoint is None:
            raise ValueError("--learned-mode checkpoint requires --learned-checkpoint")
        checkpoint = Path(args.learned_checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"learned checkpoint does not exist: {checkpoint}")
        manifest["steps"].append(
            {
                "kind": "learned_checkpoint",
                "checkpoint": str(checkpoint),
            }
        )
    else:
        _empty_dagger_label_summary().to_csv(out_dir / "learning" / "dagger_label_summary.csv", index=False)

    _log_progress(out_dir, "evaluation_start", checkpoint="" if checkpoint is None else str(checkpoint))
    episode_rows, intervention_rows, monitor_rows, decision_rows, evaluation_failures = _run_evaluation(
        args,
        profile,
        checkpoint,
        out_dir,
        label_quality=label_quality,
    )
    failure_rows.extend(evaluation_failures)
    manifest["steps"].append(
        {
            "kind": "heldout_evaluation",
            "episodes": len(episode_rows),
            "eval_seeds": profile.eval_seeds,
            "eval_seed_start": profile.train_seeds,
        }
    )

    raw = pd.DataFrame(episode_rows)
    interventions = pd.DataFrame(intervention_rows)
    monitor = pd.DataFrame(monitor_rows)
    decisions = pd.DataFrame(decision_rows)
    if decisions.empty:
        decisions = _empty_decision_features()
    failures = pd.DataFrame(failure_rows, columns=FAILURE_EVENT_COLUMNS)
    selection = _selection_summary(decisions)
    sequence_summary = _sequence_summary(decisions)
    headline = _headline_table(raw, profile.bootstrap_samples, seed=args.bootstrap_seed)
    notes = _analysis_notes(
        raw,
        headline,
        label_quality=label_quality,
        failures=failures,
        interventions=interventions,
        monitor=monitor,
        decisions=decisions,
    )

    raw.to_csv(out_dir / "raw_episodes.csv", index=False)
    interventions.to_csv(out_dir / "intervention_timeseries.csv", index=False)
    monitor.to_csv(out_dir / "monitor_timeseries.csv", index=False)
    decisions.to_csv(out_dir / "decision_features.csv", index=False)
    failures.to_csv(out_dir / "failure_events.csv", index=False)
    selection.to_csv(out_dir / "selection_summary.csv", index=False)
    sequence_summary.to_csv(out_dir / "predicted_sequence_summary.csv", index=False)
    headline.to_csv(out_dir / "headline_table.csv", index=False)
    (out_dir / "headline_table.md").write_text(_markdown_table(headline), encoding="utf-8")
    (out_dir / "headline_quality.md").write_text(_headline_quality_markdown(notes), encoding="utf-8")
    (out_dir / "analysis_notes.json").write_text(
        json.dumps(_json_safe(notes), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "config.json").write_text(
        json.dumps(_json_safe({"profile": asdict(profile), "args": vars(args)}), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _log_progress(out_dir, "suite_complete", out=str(out_dir))
    _write_artifact_index(out_dir, out_dir / "artifact_index.csv")
    if not args.no_archive:
        _write_archive(out_dir, out_dir.with_suffix(".tar.gz"))
    if args.fail_on_unusable and not bool(notes.get("paper_usable", False)):
        reasons = notes.get("paper_usable_reasons", [])
        reason_text = "\n".join(f"- {reason}" for reason in reasons)
        raise RuntimeError(f"CoRL run is not usable as headline evidence:\n{reason_text}")
    return out_dir


def _run_dagger_training(
    args: argparse.Namespace,
    profile: CorlProfile,
    out_dir: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    aggregate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    expert = _dagger_expert_method(args.dagger_expert)
    learned_checkpoint: Path | None = None
    for iteration in range(profile.dagger_iterations + 1):
        failure_count_before_iteration = len(failure_rows)
        method = (
            expert
            if iteration == 0 or learned_checkpoint is None
            else _learned_method(learned_checkpoint, args.dagger_expert)
        )
        rows: list[dict[str, Any]] = []
        iteration_start = perf_counter()
        _log_progress(
            out_dir,
            "dagger_iteration_start",
            iteration=iteration,
            learner=method.name,
            expert=expert.name,
            train_seeds=profile.train_seeds,
        )
        for seed in range(profile.train_seeds):
            seed_start = perf_counter()
            decisions: list[dict[str, Any]] = []
            _log_progress(
                out_dir,
                "dagger_seed_start",
                iteration=iteration,
                seed=seed,
                learner=method.name,
                expert=expert.name,
            )
            client = make_env_client(
                profile=args.profile,
                safe_control_gym_root=args.safe_control_gym_root,
                safe_control_python=args.safe_control_python,
                safe_control_config=args.safe_control_config,
                safe_control_controller_mode=args.safe_control_controller_mode,
                allow_debug_pid=args.allow_debug_pid,
            )
            try:
                _, _, _, decisions, failures = _run_episode(
                    client,
                    profile,
                    method,
                    seed,
                    phase="train",
                    expert_for_labels=expert if method.kind == "learned" else None,
                )
                rows.extend(decisions)
                failure_rows.extend(failures)
            finally:
                client.close()
            _log_progress(
                out_dir,
                "dagger_seed_complete",
                iteration=iteration,
                seed=seed,
                decision_rows=len(decisions),
                elapsed_seconds=perf_counter() - seed_start,
            )
        for row in rows:
            row["dagger_iteration"] = iteration
        aggregate_rows.extend(rows)
        if len(failure_rows) > failure_count_before_iteration:
            raise RuntimeError(
                "DAgger reducer labeling failed; inspect progress output and failure context before training a checkpoint"
            )
        if not aggregate_rows:
            raise RuntimeError("DAgger produced no reducer-decision rows; lower budget or increase monitor memory")
        dataset = pd.DataFrame(aggregate_rows)
        dataset.to_csv(out_dir / "dagger_dataset.csv", index=False)
        learned_checkpoint = out_dir / "learning" / f"dagger_iter{iteration}.pt"
        _train_checkpoint(dataset, learned_checkpoint, profile, iteration, expert.name, args.dagger_expert)
        _log_progress(
            out_dir,
            "dagger_iteration_complete",
            iteration=iteration,
            decision_rows=len(rows),
            aggregate_rows=len(aggregate_rows),
            checkpoint=str(learned_checkpoint),
            label_counts=_label_counts(pd.DataFrame(rows)),
            elapsed_seconds=perf_counter() - iteration_start,
        )
    final = out_dir / "learning" / "dagger_final.pt"
    if learned_checkpoint is None:
        raise RuntimeError("DAgger did not produce a checkpoint")
    shutil.copy2(learned_checkpoint, final)
    label_summary = _dagger_label_summary(pd.DataFrame(aggregate_rows))
    label_quality = _dagger_label_quality(label_summary)
    metrics = {
        "iterations": profile.dagger_iterations,
        "row_count": len(aggregate_rows),
        "final_checkpoint": "learning/dagger_final.pt",
        "expert_method": expert.name,
        "label_quality": label_quality,
    }
    (out_dir / "learning" / "dagger_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return pd.DataFrame(aggregate_rows), failure_rows


def _run_evaluation(
    args: argparse.Namespace,
    profile: CorlProfile,
    checkpoint: Path | None,
    out_dir: Path,
    *,
    label_quality: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    methods = (
        "nominal_no_monitor",
        "reference_unbounded",
        *_headline_methods(args.method_set),
    )
    if _include_learned_method(args, checkpoint, label_quality):
        methods = (*methods, _learned_method(checkpoint, args.dagger_expert))
    raw_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    monitor_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for method in methods:
        method_name = method if isinstance(method, str) else method.name
        _log_progress(out_dir, "evaluation_method_start", method=method_name)
        for seed in range(profile.train_seeds, profile.train_seeds + profile.eval_seeds):
            seed_start = perf_counter()
            _log_progress(
                out_dir,
                "evaluation_seed_start",
                method=method_name,
                seed=seed,
            )
            client = make_env_client(
                profile=args.profile,
                safe_control_gym_root=args.safe_control_gym_root,
                safe_control_python=args.safe_control_python,
                safe_control_config=args.safe_control_config,
                safe_control_controller_mode=args.safe_control_controller_mode,
                allow_debug_pid=args.allow_debug_pid,
            )
            try:
                episode, interventions, monitor, decisions, failures = _run_episode(
                    client,
                    profile,
                    method,
                    seed,
                    phase="eval",
                )
            finally:
                client.close()
            raw_rows.append(episode)
            intervention_rows.extend(interventions)
            monitor_rows.extend(monitor)
            decision_rows.extend(decisions)
            failure_rows.extend(failures)
            _log_progress(
                out_dir,
                "evaluation_seed_complete",
                method=method_name,
                seed=seed,
                steps=episode["steps"],
                decision_rows=len(decisions),
                elapsed_seconds=perf_counter() - seed_start,
            )
    return raw_rows, intervention_rows, monitor_rows, decision_rows, failure_rows


def _violated_trigger_names(verdicts: Sequence[Any]) -> list[str]:
    return [verdict.trigger.name for verdict in verdicts if verdict.status == "violation"]


def _oracle_verdicts(monitor: IrosGateMonitor, snapshot: IrosEnvSnapshot) -> tuple[Verdict, ...]:
    verdicts = list(monitor.oracle_verdicts(snapshot.pose, snapshot.velocity, snapshot.target_gate_index))
    if snapshot.collision:
        verdicts.append(_simulator_violation("simulator_collision"))
    if snapshot.constraint_violation:
        verdicts.append(_simulator_violation("simulator_constraint_violation"))
    return tuple(verdicts)


def _simulator_violation(name: str) -> Verdict:
    return Verdict(TriggerSpec(name, 0, 0.0, direction="below"), "violation", -1.0, -1.0)


def _failure_event(
    *,
    phase: str,
    method: str,
    method_kind: str,
    seed: int,
    step: int,
    event_type: str,
    exc: BaseException,
    generator_count: int | None = None,
    candidate_reducer_names: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "phase": phase,
        "method": method,
        "method_kind": method_kind,
        "seed": int(seed),
        "step": int(step),
        "event_type": event_type,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=8)),
        "generator_count": "" if generator_count is None else int(generator_count),
        "candidate_reducer_names": json.dumps(list(candidate_reducer_names)),
    }


def _snapshot_metadata(snapshot: IrosEnvSnapshot) -> dict[str, Any]:
    info = snapshot.info
    return {
        "controller_mode": str(info.get("controller_mode", "unknown")),
        "pycffirmware_available": bool(info.get("pycffirmware_available", False)),
        "ctrl_freq": float(info.get("ctrl_freq", np.nan)),
        "firmware_freq": float(info.get("firmware_freq", np.nan)),
        "episode_len_sec": float(info.get("episode_len_sec", np.nan)),
        "simulator_time": float(snapshot.time),
    }


def _run_episode(
    client: IrosEnvClient,
    profile: CorlProfile,
    method: MethodSpec | str,
    seed: int,
    *,
    phase: str,
    expert_for_labels: MethodSpec | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    snapshot = client.reset(seed)
    scenario = client.scenario
    monitor = IrosGateMonitor(
        scenario,
        stream_memory_decay=profile.stream_memory_decay,
    )
    state = monitor.initial_state()
    sensor = NoisySensorModel(
        bias_bound=profile.sensor_bias_bound,
        noise_bound=profile.sensor_noise_bound,
        seed=10_000 + seed,
    )
    manager = InterventionManager(
        client.fallback_command(snapshot),
        fallback_hold_steps=profile.fallback_hold_steps,
        expected_gate_count=len(scenario.gates),
    )
    policy = None if isinstance(method, str) else _make_policy(method, monitor, _benchmark_config(profile))
    candidate_names = () if isinstance(method, str) else _method_candidate_names(method)
    expert_policy = (
        None
        if expert_for_labels is None
        else _make_policy(expert_for_labels, monitor, _benchmark_config(profile))
    )
    expert_candidate_names = (
        ()
        if expert_for_labels is None
        else _method_candidate_names(expert_for_labels)
    )
    method_name = method if isinstance(method, str) else method.name
    method_kind = method if isinstance(method, str) else method.kind
    interventions: list[dict[str, Any]] = []
    monitor_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    collision_count = 0
    constraint_count = 0
    reduction_failure_count = 0
    start = perf_counter()

    for step in range(profile.max_steps):
        nominal = client.nominal_command(snapshot)
        reduction_applied = False
        reducer_name = ""
        reduction_seconds = 0.0
        unsound = False
        budget_violation = False
        predicted_sequence: tuple[str, ...] = ()
        evaluated_sequences = 0
        pruned_sequences = 0
        predicted_cost = 0.0
        true_streams = iros_stream_values(
            scenario,
            IrosObservation(snapshot.pose, snapshot.velocity, target_gate_index=snapshot.target_gate_index),
        )

        if method_name == "nominal_no_monitor":
            command = nominal
            oracle = _oracle_verdicts(monitor, snapshot)
            verdicts = ()
            fallback_active = False
        else:
            observation = sensor.observe(
                snapshot.pose,
                snapshot.velocity,
                target_gate_index=snapshot.target_gate_index,
                command=nominal,
                time=snapshot.time,
            )
            result = monitor.step(state, observation)
            state = result.state
            if state.zonotope.generator_count > profile.budget and method_name != "reference_unbounded":
                features_before = decision_feature_values(
                    monitor,
                    state,
                    budget=profile.budget,
                    horizon=profile.horizon,
                )
                if expert_policy is not None and expert_for_labels is not None:
                    try:
                        expert_decision = _reduce_with_policy(
                            expert_policy,
                            expert_for_labels,
                            monitor,
                            state,
                            profile,
                            snapshot,
                        )
                    except Exception as exc:
                        reduction_failure_count += 1
                        failure_rows.append(
                            _failure_event(
                                phase=phase,
                                method=expert_for_labels.name,
                                method_kind=expert_for_labels.kind,
                                seed=seed,
                                step=step + 1,
                                event_type="expert_label_selection",
                                exc=exc,
                                generator_count=state.zonotope.generator_count,
                                candidate_reducer_names=expert_candidate_names,
                            )
                        )
                    else:
                        decisions.append(
                            _decision_row(
                                scenario="iros_gate",
                                method=expert_for_labels.name,
                                method_kind=expert_for_labels.kind,
                                seed=seed,
                                phase=phase,
                                profile=profile,
                                step=step + 1,
                                decision=expert_decision,
                                features=features_before,
                                candidate_reducer_names=expert_candidate_names,
                            )
                        )
                reduction_start = perf_counter()
                try:
                    decision = _reduce_with_policy(policy, method, monitor, state, profile, snapshot)
                except Exception as exc:
                    reduction_failure_count += 1
                    failure_rows.append(
                        _failure_event(
                            phase=phase,
                            method=method_name,
                            method_kind=str(method_kind),
                            seed=seed,
                            step=step + 1,
                            event_type="reducer_selection",
                            exc=exc,
                            generator_count=state.zonotope.generator_count,
                            candidate_reducer_names=candidate_names,
                        )
                    )
                else:
                    reduction_seconds = perf_counter() - reduction_start
                    state = decision.state
                    reducer_name = decision.reducer_name
                    reduction_applied = not decision.is_no_op
                    unsound = not decision.result.certificate.is_sound
                    predicted_sequence = decision.predicted_sequence
                    evaluated_sequences = decision.evaluated_sequences
                    pruned_sequences = decision.pruned_sequences
                    predicted_cost = decision.predicted_cost
                    if expert_policy is None:
                        decisions.append(
                            _decision_row(
                                scenario="iros_gate",
                                method=method_name,
                                method_kind=str(method_kind),
                                seed=seed,
                                phase=phase,
                                profile=profile,
                                step=step + 1,
                                decision=decision,
                                features=features_before,
                                candidate_reducer_names=candidate_names,
                            )
                        )
            verdicts = evaluate_triggers(state.zonotope, monitor.triggers)
            oracle = _oracle_verdicts(monitor, snapshot)
            command = manager.choose_command(
                nominal,
                verdicts,
                oracle,
                fallback_command=client.fallback_command(snapshot),
                gates_passed=snapshot.gates_passed,
                time=snapshot.time,
                reducer_name=reducer_name,
                reducer_latency_seconds=reduction_seconds,
                budget_violation=state.zonotope.generator_count > profile.budget and method_name != "reference_unbounded",
                unsound_certificate=unsound,
            )
            budget_violation = state.zonotope.generator_count > profile.budget and method_name != "reference_unbounded"
            fallback_active = not np.allclose(command, nominal)

        next_snapshot = client.step(command)
        collision_count += int(next_snapshot.collision)
        constraint_count += int(next_snapshot.constraint_violation)
        interventions.append(
            {
                "phase": phase,
                "method": method_name,
                "method_kind": str(method_kind),
                "seed": seed,
                "step": step + 1,
                "sensor_bias_bound": profile.sensor_bias_bound,
                "sensor_noise_bound": profile.sensor_noise_bound,
                "stream_memory_decay": profile.stream_memory_decay,
                "fallback_hold_steps": profile.fallback_hold_steps,
                "time": snapshot.time,
                **_snapshot_metadata(snapshot),
                "monitor_triggered": any(v.status == "violation" for v in verdicts),
                "oracle_violated": any(v.status == "violation" for v in oracle),
                "monitor_trigger_names": json.dumps(_violated_trigger_names(verdicts)),
                "oracle_trigger_names": json.dumps(_violated_trigger_names(oracle)),
                "fallback_active": fallback_active,
                "collision": next_snapshot.collision,
                "constraint_violation": next_snapshot.constraint_violation,
                "gates_passed": next_snapshot.gates_passed,
                "task_completed": next_snapshot.task_completed,
                "pose_x": float(snapshot.pose[0]),
                "pose_y": float(snapshot.pose[1]),
                "pose_z": float(snapshot.pose[2]),
                "velocity_x": float(snapshot.velocity[0]),
                "velocity_y": float(snapshot.velocity[1]),
                "velocity_z": float(snapshot.velocity[2]),
                **{f"stream_{name}": float(value) for name, value in zip(IROS_STREAM_NAMES, true_streams)},
            }
        )
        monitor_rows.append(
            {
                "phase": phase,
                "method": method_name,
                "method_kind": str(method_kind),
                "seed": seed,
                "step": step + 1,
                "sensor_bias_bound": profile.sensor_bias_bound,
                "sensor_noise_bound": profile.sensor_noise_bound,
                "stream_memory_decay": profile.stream_memory_decay,
                "fallback_hold_steps": profile.fallback_hold_steps,
                "generator_count": 0 if method_name == "nominal_no_monitor" else state.zonotope.generator_count,
                "budget": profile.budget,
                "budget_violation": budget_violation,
                "reduction_applied": reduction_applied,
                "reducer_name": reducer_name,
                "reduction_seconds": reduction_seconds,
                "unsound_certificate": unsound,
                "predicted_cost": predicted_cost,
                "predicted_sequence": json.dumps(list(predicted_sequence)),
                "evaluated_sequence_count": evaluated_sequences,
                "pruned_sequence_count": pruned_sequences,
            }
        )
        snapshot = next_snapshot
        if snapshot.done:
            break

    metrics = manager.metrics
    duration = max(1, len(interventions))
    completed = snapshot.task_completed or metrics.task_completed
    episode = {
        "phase": phase,
        "scenario": "iros_gate",
        "method": method_name,
        "method_kind": str(method_kind),
        "seed": seed,
        "budget": profile.budget,
        "horizon": profile.horizon,
        "sensor_bias_bound": profile.sensor_bias_bound,
        "sensor_noise_bound": profile.sensor_noise_bound,
        "stream_memory_decay": profile.stream_memory_decay,
        "fallback_hold_steps": profile.fallback_hold_steps,
        **_snapshot_metadata(snapshot),
        "steps": duration,
        "total_seconds": perf_counter() - start,
        "task_completed": completed,
        "gates_passed": max(snapshot.gates_passed, metrics.gates_passed),
        "collision_count": collision_count,
        "collision_episode": collision_count > 0,
        "constraint_violation_count": constraint_count,
        "constraint_violation_episode": constraint_count > 0,
        "fallback_activation_count": metrics.fallback_activation_count,
        "fallback_duration": metrics.fallback_duration,
        "fallback_duration_fraction": metrics.fallback_duration / duration,
        "spurious_intervention_count": metrics.spurious_intervention_count,
        "spurious_intervention_rate": metrics.spurious_intervention_rate,
        "justified_intervention_count": metrics.justified_intervention_count,
        "justified_intervention_rate": metrics.justified_intervention_count / duration,
        "missed_violation_count": metrics.missed_violation_count,
        "missed_violation_rate": metrics.missed_violation_rate,
        "time_to_target": (
            metrics.time_to_target
            if metrics.time_to_target is not None
            else (snapshot.time if completed else np.nan)
        ),
        "mean_reducer_latency_ms": 1000.0 * metrics.reducer_latency_seconds / max(1, sum(metrics.reducer_choices.values())),
        "budget_violation_count": int(sum(row["budget_violation"] for row in monitor_rows)),
        "unsound_certificate_count": int(sum(row["unsound_certificate"] for row in monitor_rows)),
        "reduction_failure_count": reduction_failure_count,
        "reducer_choices": json.dumps(metrics.reducer_choices, sort_keys=True),
    }
    return episode, interventions, monitor_rows, decisions, failure_rows


def _reduce_with_policy(
    policy: Any,
    method: MethodSpec | str,
    monitor: IrosGateMonitor,
    state: Any,
    profile: CorlProfile,
    snapshot: IrosEnvSnapshot,
) -> ReductionDecision:
    if policy is None or isinstance(method, str):
        raise ValueError("method has no reducer policy")
    if method.kind in {"mpc", "mpc_sequence", "mpc_rollout"}:
        predicted = _predicted_observations(monitor, snapshot, profile)
        return policy.reduce_state(monitor, state, predicted)
    return policy.reduce_state(monitor, state)


def _predicted_observations(
    monitor: IrosGateMonitor,
    snapshot: IrosEnvSnapshot,
    profile: CorlProfile,
) -> tuple[IrosObservation, ...]:
    observations: list[IrosObservation] = []
    dt = _snapshot_dt(snapshot)
    pose = snapshot.pose.copy()
    velocity = snapshot.velocity.copy()
    target_gate_index = snapshot.target_gate_index
    for index in range(profile.horizon):
        gate = monitor.scenario.gate(target_gate_index)
        velocity = 0.85 * velocity + 0.15 * (gate.center - pose)
        pose = pose + dt * velocity
        if (
            np.linalg.norm(pose - gate.center) <= monitor.scenario.gate_pass_radius
            and target_gate_index < len(monitor.scenario.gates) - 1
        ):
            target_gate_index += 1
        observations.append(
            IrosObservation(
                pose,
                velocity,
                target_gate_index=target_gate_index,
                bias_radius=np.full(6, profile.sensor_bias_bound),
                noise_radius=np.full(6, profile.sensor_noise_bound),
                time=snapshot.time + (index + 1) * dt,
            )
        )
    return tuple(observations)


def _snapshot_dt(snapshot: IrosEnvSnapshot) -> float:
    info = snapshot.info
    if "ctrl_timestep" in info:
        try:
            value = float(info["ctrl_timestep"])
            if np.isfinite(value) and value > 0.0:
                return value
        except (TypeError, ValueError):
            pass
    if "ctrl_freq" in info:
        try:
            freq = float(info["ctrl_freq"])
            if np.isfinite(freq) and freq > 0.0:
                return 1.0 / freq
        except (TypeError, ValueError):
            pass
    return 0.05


def _decision_row(
    *,
    scenario: str,
    method: str,
    method_kind: str,
    seed: int,
    phase: str,
    profile: CorlProfile,
    step: int,
    decision: ReductionDecision,
    features: dict[str, float],
    candidate_reducer_names: Sequence[str],
) -> dict[str, Any]:
    return {
        "feature_schema_version": DECISION_FEATURE_SCHEMA_VERSION,
        "scenario": scenario,
        "method": method,
        "method_kind": method_kind,
        "seed": seed,
        "phase": phase,
        "length": profile.max_steps,
        "budget": profile.budget,
        "horizon": profile.horizon,
        "predictor_mode": "online",
        "step": step,
        "chosen_reducer_label": decision.reducer_name,
        "predicted_cost": decision.predicted_cost,
        "predicted_sequence": json.dumps(list(decision.predicted_sequence)),
        "evaluated_sequence_count": decision.evaluated_sequences,
        "pruned_sequence_count": decision.pruned_sequences,
        "candidate_reducer_names": json.dumps(list(candidate_reducer_names)),
        "no_op_selected": decision.is_no_op,
        **features,
    }


def _empty_decision_features() -> pd.DataFrame:
    return pd.DataFrame(
        columns=(
            "feature_schema_version",
            "scenario",
            "method",
            "method_kind",
            "seed",
            "phase",
            "length",
            "budget",
            "horizon",
            "predictor_mode",
            "step",
            "chosen_reducer_label",
            "predicted_cost",
            "predicted_sequence",
            "evaluated_sequence_count",
            "pruned_sequence_count",
            "candidate_reducer_names",
            "no_op_selected",
            *DECISION_FEATURE_NAMES,
        )
    )


def _train_checkpoint(
    dataset: pd.DataFrame,
    checkpoint: Path,
    profile: CorlProfile,
    iteration: int,
    expert_method: str,
    dagger_expert: str,
) -> None:
    train_policy(
        argparse.Namespace(
            data=_write_training_dataset(dataset, checkpoint),
            expert_method=expert_method,
            predictor_mode="online",
            out=checkpoint,
            seed=iteration,
            epochs=profile.distill_epochs,
            batch_size=profile.distill_batch_size,
            lr=1e-3,
            weight_decay=0.0,
            validation_fraction=0.2,
            hidden_sizes=(64, 64),
            class_balanced=True,
            training_mode="dagger",
            candidate_reducer_names=_learned_candidate_names(dagger_expert),
            dagger_metadata={
                "iteration": iteration,
                "row_count": int(dataset.shape[0]),
                "expert_method": expert_method,
                "dagger_expert": dagger_expert,
            },
        )
    )


def _write_training_dataset(dataset: pd.DataFrame, checkpoint: Path) -> Path:
    path = checkpoint.with_suffix(".dataset.csv")
    dataset.to_csv(path, index=False)
    return path


def _headline_methods(method_set: str = "extended") -> tuple[MethodSpec, ...]:
    core = (
        MethodSpec.static("box", _protected(BoxReducer)),
        MethodSpec.static("girard", _protected(GirardReducer)),
        MethodSpec.static("keep_calibration_aware", _protected(ScoredKeepReducer.calibration_aware)),
        MethodSpec.rollout_mpc(
            "mpc_focused_fixed_girard",
            focused_geometry_reducer_factories(),
            _protected(GirardReducer),
            _protected(BoxReducer),
        ),
        MethodSpec.rollout_mpc(
            "mpc_wide_fixed_girard",
            wide_rollout_reducer_factories(),
            _protected(GirardReducer),
            _protected(BoxReducer),
        ),
    )
    if method_set == "core":
        return core
    if method_set != "extended":
        raise ValueError(f"unsupported CoRL method set: {method_set}")
    return (
        MethodSpec.static("box", _protected(BoxReducer)),
        MethodSpec.static("girard", _protected(GirardReducer)),
        MethodSpec.static("combastel", _protected(CombastelReducer)),
        MethodSpec.static("pca", _protected(PcaReducer)),
        MethodSpec.static("keep_norm", _protected(ScoredKeepReducer.by_norm)),
        MethodSpec.static("keep_calibration_aware", _protected(ScoredKeepReducer.calibration_aware)),
        _expert_method(),
        *core[3:],
    )


def _include_learned_method(
    args: argparse.Namespace,
    checkpoint: Path | None,
    label_quality: dict[str, Any],
) -> bool:
    if args.learned_mode == "none" or checkpoint is None:
        return False
    if args.learned_mode == "checkpoint":
        return True
    if args.include_failed_learned:
        return True
    return bool(label_quality.get("passes_gate", False))


def _expert_method() -> MethodSpec:
    return MethodSpec.sequence_mpc(
        "mpc_focused_sequence",
        focused_geometry_reducer_factories(),
        _protected(BoxReducer),
    )


def _dagger_expert_method(name: str) -> MethodSpec:
    if name == "mpc_focused_sequence":
        return _expert_method()
    if name == "mpc_wide_fixed_girard":
        return MethodSpec.rollout_mpc(
            "mpc_wide_fixed_girard",
            wide_rollout_reducer_factories(),
            _protected(GirardReducer),
            _protected(BoxReducer),
        )
    raise ValueError(f"unsupported DAgger expert: {name}")


def _learned_method(checkpoint: Path, dagger_expert: str = "mpc_wide_fixed_girard") -> MethodSpec:
    return MethodSpec(
        "learned_dagger",
        "learned",
        mpc_reducer_factories=_learned_reducer_factories(dagger_expert),
        mpc_fallback_reducer_factory=_protected(BoxReducer),
        learned_policy_path=str(checkpoint),
    )


def _learned_reducer_factories(dagger_expert: str) -> tuple[Callable[[], Any], ...]:
    if dagger_expert == "mpc_focused_sequence":
        return focused_geometry_reducer_factories()
    if dagger_expert == "mpc_wide_fixed_girard":
        return wide_rollout_reducer_factories()
    raise ValueError(f"unsupported DAgger expert: {dagger_expert}")


def _learned_candidate_names(dagger_expert: str) -> tuple[str, ...]:
    names: list[str] = []
    for factory in _learned_reducer_factories(dagger_expert):
        names.append(factory().name)
    names.append(_protected(BoxReducer)().name)
    return tuple(dict.fromkeys(names))


def _protected(factory: Callable[[], Any]) -> Callable[[], Any]:
    def make() -> Any:
        return ProtectedReducer(factory())

    return make


def _benchmark_config(profile: CorlProfile) -> BenchmarkConfig:
    return BenchmarkConfig(
        length=profile.max_steps,
        budget=profile.budget,
        horizon=profile.horizon,
        seeds=(),
        predictor_mode="online",
        include_reference=False,
        bootstrap_samples=profile.bootstrap_samples,
    )


def _log_progress(out_dir: Path, event: str, **fields: Any) -> None:
    record = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **_json_safe(fields),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"[corl] {event} {json.dumps(_json_safe(fields), sort_keys=True)}", flush=True)


def _label_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty or "chosen_reducer_label" not in frame:
        return {}
    return {
        str(label): int(count)
        for label, count in frame["chosen_reducer_label"].astype(str).value_counts().sort_index().items()
    }


def _dagger_label_summary(rows: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "dagger_iteration",
        "selected_reducer",
        "selection_count",
        "selection_fraction",
        "decision_count",
    )
    if rows.empty or "chosen_reducer_label" not in rows:
        return pd.DataFrame(columns=columns)
    summaries: list[dict[str, Any]] = []
    for iteration, group in rows.groupby("dagger_iteration", sort=True):
        total = int(group.shape[0])
        counts = group["chosen_reducer_label"].astype(str).value_counts().sort_index()
        for reducer, count in counts.items():
            summaries.append(
                {
                    "dagger_iteration": int(iteration),
                    "selected_reducer": str(reducer),
                    "selection_count": int(count),
                    "selection_fraction": float(count / total) if total else 0.0,
                    "decision_count": total,
                }
            )
    return pd.DataFrame(summaries, columns=columns)


def _empty_dagger_label_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=(
            "dagger_iteration",
            "selected_reducer",
            "selection_count",
            "selection_fraction",
            "decision_count",
        )
    )


def _dagger_label_quality(summary: pd.DataFrame) -> dict[str, Any]:
    if summary.empty:
        return {
            "decision_count": 0,
            "unique_reducer_count": 0,
            "top_reducer": "",
            "top_reducer_fraction": 0.0,
            "passes_gate": False,
            "gate": (
                f"at least {LABEL_DIVERSITY_MIN_CLASSES} reducers and top reducer at most "
                f"{LABEL_DIVERSITY_MAX_TOP_FRACTION:.0%}"
            ),
        }
    totals = (
        summary.groupby("selected_reducer", sort=True)["selection_count"]
        .sum()
        .sort_values(ascending=False)
    )
    decision_count = int(totals.sum())
    top_reducer = str(totals.index[0])
    top_fraction = float(totals.iloc[0] / decision_count) if decision_count else 0.0
    unique_count = int((totals > 0).sum())
    passes = unique_count >= LABEL_DIVERSITY_MIN_CLASSES and top_fraction <= LABEL_DIVERSITY_MAX_TOP_FRACTION
    return {
        "decision_count": decision_count,
        "unique_reducer_count": unique_count,
        "top_reducer": top_reducer,
        "top_reducer_fraction": top_fraction,
        "passes_gate": bool(passes),
        "gate": (
            f"at least {LABEL_DIVERSITY_MIN_CLASSES} reducers and top reducer at most "
            f"{LABEL_DIVERSITY_MAX_TOP_FRACTION:.0%}"
        ),
    }


def _selection_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame(columns=("phase", "method", "selected_reducer", "selection_count"))
    return (
        decisions.groupby(["phase", "method", "chosen_reducer_label"], dropna=False)
        .size()
        .reset_index(name="selection_count")
        .rename(columns={"chosen_reducer_label": "selected_reducer"})
    )


def _sequence_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame(columns=("phase", "method", "decision_count", "mean_sequence_length"))
    rows: list[dict[str, Any]] = []
    for (phase, method), group in decisions.groupby(["phase", "method"], sort=True):
        lengths = [len(json.loads(value)) for value in group["predicted_sequence"].fillna("[]")]
        rows.append(
            {
                "phase": phase,
                "method": method,
                "decision_count": int(group.shape[0]),
                "mean_sequence_length": float(np.mean(lengths)) if lengths else 0.0,
                "evaluated_sequence_count": int(group["evaluated_sequence_count"].fillna(0).sum()),
                "pruned_sequence_count": int(group["pruned_sequence_count"].fillna(0).sum()),
            }
        )
    return pd.DataFrame(rows)


def _preflight_calibration_configs(args: argparse.Namespace, configs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for config in configs:
        safe_control_config = config["safe_control_config"]
        key = "" if safe_control_config is None else str(safe_control_config)
        if key in seen:
            continue
        seen.add(key)
        result = preflight_safe_control_gym(
            profile=args.profile,
            safe_control_gym_root=args.safe_control_gym_root,
            safe_control_python=args.safe_control_python,
            safe_control_config=safe_control_config,
            safe_control_controller_mode=args.safe_control_controller_mode,
            allow_debug_pid=args.allow_debug_pid,
        )
        record = {
            "safe_control_config": key,
            "result": result.to_dict(),
        }
        records.append(record)
        if not result.ok:
            raise RuntimeError(
                "CoRL calibration preflight failed"
                + (f" for {key}" if key else "")
                + ":\n"
                + "\n".join(f"- {message}" for message in result.messages)
            )
    return records


def _calibration_configs(args: argparse.Namespace, profile: CorlProfile) -> list[dict[str, Any]]:
    if args.profile == "smoke" and args.safe_control_config is None:
        config_paths = [None]
    elif args.safe_control_config is not None:
        config_paths = [args.safe_control_config]
    else:
        config_paths = ["competition/level0.yaml", "competition/level1.yaml"]
    variants: list[tuple[str, CorlProfile]] = [("base", profile)]
    if args.profile == "smoke":
        variants.append(("low_budget", replace(profile, budget=max(1, profile.budget - 2))))
    else:
        for value in (0.0, 0.005, 0.015):
            variants.append((f"bias_{value:g}", replace(profile, sensor_bias_bound=value)))
        for value in (0.005, 0.015, 0.03):
            variants.append((f"noise_{value:g}", replace(profile, sensor_noise_bound=value)))
        for value in (0.0, 0.5, 0.85):
            variants.append((f"memory_{value:g}", replace(profile, stream_memory_decay=value)))
        for value in (6, 8, 10):
            variants.append((f"budget_{value}", replace(profile, budget=value)))
        for value in (1, 2):
            variants.append((f"fallback_hold_{value}", replace(profile, fallback_hold_steps=value)))
    configs: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for path in config_paths:
        for variant_name, variant in variants:
            key = (
                path,
                variant.budget,
                variant.sensor_bias_bound,
                variant.sensor_noise_bound,
                variant.stream_memory_decay,
                variant.fallback_hold_steps,
            )
            if key in seen:
                continue
            seen.add(key)
            configs.append(
                {
                    "config_id": f"{Path(path).stem if path else 'fake'}:{variant_name}",
                    "safe_control_config": path,
                    "profile": variant,
                }
            )
    return configs


def _calibration_summary(raw: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "config_id",
        "safe_control_config",
        "method",
        "episode_count",
        "task_completion_rate",
        "mean_gates_passed",
        "collision_rate",
        "constraint_violation_rate",
        "fallback_duration_fraction",
        "spurious_intervention_rate",
        "missed_violation_rate",
        "reduction_failure_count",
        "budget_violation_count",
        "unsound_certificate_count",
        "paper_candidate",
        "rejection_reasons",
    )
    if raw.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (config_id, method), group in raw.groupby(["config_id", "method"], sort=True):
        row = {
            "config_id": config_id,
            "safe_control_config": str(group["safe_control_config"].iloc[0]),
            "method": method,
            "episode_count": int(group.shape[0]),
            "task_completion_rate": float(group["task_completed"].astype(float).mean()),
            "mean_gates_passed": float(group["gates_passed"].astype(float).mean()),
            "collision_rate": float(group["collision_episode"].astype(float).mean()),
            "constraint_violation_rate": float(group["constraint_violation_episode"].astype(float).mean()),
            "fallback_duration_fraction": float(group["fallback_duration_fraction"].astype(float).mean()),
            "spurious_intervention_rate": float(group["spurious_intervention_rate"].astype(float).mean()),
            "missed_violation_rate": float(group["missed_violation_rate"].astype(float).mean()),
            "reduction_failure_count": int(group["reduction_failure_count"].sum()),
            "budget_violation_count": int(group["budget_violation_count"].sum()),
            "unsound_certificate_count": int(group["unsound_certificate_count"].sum()),
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    candidates: list[dict[str, Any]] = []
    for config_id, group in summary.groupby("config_id", sort=True):
        reasons: list[str] = []
        nominal = group[group["method"] == "nominal_no_monitor"]
        bounded = group[~group["method"].isin({"nominal_no_monitor", "reference_unbounded"})]
        if nominal.empty or float(nominal["task_completion_rate"].iloc[0]) < 0.8:
            reasons.append("nominal controller completion rate below 0.8")
        if bounded.empty:
            reasons.append("no bounded methods")
        else:
            fallback = bounded["fallback_duration_fraction"].astype(float)
            if (fallback <= 0.02).all() or (fallback >= 0.98).all():
                reasons.append("bounded-method fallback duration saturated")
            if int(bounded["reduction_failure_count"].sum()):
                reasons.append("reduction failures")
            if int(bounded["budget_violation_count"].sum()):
                reasons.append("budget violations")
            if int(bounded["unsound_certificate_count"].sum()):
                reasons.append("unsound certificates")
            headline_method = bounded[bounded["method"] == "mpc_focused_fixed_girard"]
            if headline_method.empty:
                reasons.append("headline MPC method missing")
            elif (headline_method["missed_violation_rate"].astype(float) > 0.0).any():
                reasons.append("headline MPC missed violations")
        candidates.append(
            {
                "config_id": config_id,
                "paper_candidate": not reasons,
                "rejection_reasons": "; ".join(reasons),
            }
        )
    return summary.merge(pd.DataFrame(candidates), on="config_id", how="left")[list(columns)]


def _calibration_recommendations(summary: pd.DataFrame, failures: pd.DataFrame) -> dict[str, Any]:
    candidates = summary[summary["paper_candidate"].astype(bool)] if not summary.empty else pd.DataFrame()
    ranked = (
        candidates[candidates["method"] == "mpc_focused_fixed_girard"]
        .sort_values(["task_completion_rate", "fallback_duration_fraction", "spurious_intervention_rate"], ascending=[False, True, True])
        if not candidates.empty
        else pd.DataFrame()
    )
    recommended = None if ranked.empty else str(ranked.iloc[0]["config_id"])
    return {
        "recommended_config_id": recommended,
        "paper_candidate_config_ids": sorted(candidates["config_id"].astype(str).unique().tolist()) if not candidates.empty else [],
        "failure_event_count": int(failures.shape[0]) if failures is not None else 0,
        "summary": {
            "config_count": int(summary["config_id"].nunique()) if not summary.empty else 0,
            "paper_candidate_count": int(candidates["config_id"].nunique()) if not candidates.empty else 0,
        },
    }


def _headline_table(raw: pd.DataFrame, bootstrap_samples: int, *, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, group in raw.groupby("method", sort=True):
        row: dict[str, Any] = {"method": method, "episode_count": int(group.shape[0])}
        for metric in HEADLINE_METRICS:
            values = group[metric].astype(float).dropna().to_numpy()
            if metric == "task_completed":
                name = "task_completion_rate"
            elif metric == "collision_episode":
                name = "collision_rate"
            elif metric == "constraint_violation_episode":
                name = "constraint_violation_rate"
            elif metric == "gates_passed":
                name = "mean_gates_passed"
            else:
                name = metric
            mean, lo, hi = _bootstrap_mean_ci(values, bootstrap_samples, seed=seed)
            row[name] = mean
            row[f"{name}_ci_low"] = lo
            row[f"{name}_ci_high"] = hi
        row["budget_violation_count"] = int(group["budget_violation_count"].sum())
        row["unsound_certificate_count"] = int(group["unsound_certificate_count"].sum())
        row["reduction_failure_count"] = int(group["reduction_failure_count"].sum())
        rows.append(row)
    result = pd.DataFrame(rows)
    return _add_deltas(result, "reference_unbounded", ("box", "girard"))


def _add_deltas(
    table: pd.DataFrame,
    reference: str,
    extra_refs: tuple[str, ...],
) -> pd.DataFrame:
    if table.empty or reference not in set(table["method"]):
        return table
    refs = (reference, *extra_refs)
    metric_names = [column for column in table.columns if column not in {"method", "episode_count"} and not column.endswith(("_ci_low", "_ci_high"))]
    output = table.copy()
    for ref in refs:
        if ref not in set(table["method"]):
            continue
        ref_row = table[table["method"] == ref].iloc[0]
        for metric in metric_names:
            output[f"{metric}_delta_vs_{ref}"] = output[metric] - float(ref_row[metric])
    return output


def _bootstrap_mean_ci(values: np.ndarray, samples: int, *, seed: int) -> tuple[float, float, float]:
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if values.size == 1 or samples <= 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, values.size), replace=True)
    means = np.mean(draws, axis=1)
    return mean, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _analysis_notes(
    raw: pd.DataFrame,
    headline: pd.DataFrame,
    *,
    label_quality: dict[str, Any] | None = None,
    failures: pd.DataFrame | None = None,
    interventions: pd.DataFrame | None = None,
    monitor: pd.DataFrame | None = None,
    decisions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    budgeted = raw[~raw["method"].isin({"reference_unbounded", "nominal_no_monitor"})]
    flags: list[str] = []
    for column in ("budget_violation_count", "unsound_certificate_count", "reduction_failure_count"):
        count = int(budgeted[column].sum()) if column in budgeted else 0
        if count:
            flags.append(f"{column}={count}")
    failure_event_count = 0 if failures is None else int(failures.shape[0])
    if failure_event_count:
        flags.append(f"failure_event_count={failure_event_count}")
    if not headline.empty and "mpc_focused_sequence" in set(headline["method"]):
        mpc = headline[headline["method"] == "mpc_focused_sequence"].iloc[0]
        ref = headline[headline["method"] == "reference_unbounded"].iloc[0]
        if float(mpc["missed_violation_rate"]) > float(ref["missed_violation_rate"]):
            flags.append("mpc_focused_sequence_increases_missed_violation_rate")
    quality = _paper_usable_notes(raw, headline, failures, interventions, monitor, decisions, label_quality)
    if not quality["paper_usable"]:
        flags.append("paper_usable=false")
    if decisions is None or decisions.empty:
        flags.append("decision_features_empty")
    elif "candidate_reducer_names" in decisions:
        empty_candidates = decisions["candidate_reducer_names"].fillna("[]").astype(str).map(_json_list_is_empty)
        if bool(empty_candidates.all()):
            flags.append("candidate_reducer_names_empty")
    if interventions is not None and not interventions.empty and "monitor_triggered" in interventions:
        monitored_interventions = interventions[
            ~interventions["method"].isin({"nominal_no_monitor"})
        ]
        if not monitored_interventions.empty and not monitored_interventions["monitor_triggered"].astype(bool).any():
            flags.append("monitor_never_triggered")
    if label_quality and label_quality.get("decision_count", 0) and not label_quality.get("passes_gate", False):
        flags.append("learned_label_quality_failed")
    return {
        "soundness_checks": {
            "budget_violation_count": int(budgeted["budget_violation_count"].sum()) if not budgeted.empty else 0,
            "unsound_certificate_count": int(budgeted["unsound_certificate_count"].sum()) if not budgeted.empty else 0,
            "reduction_failure_count": int(budgeted["reduction_failure_count"].sum()) if not budgeted.empty else 0,
            "failure_event_count": failure_event_count,
        },
        "learning_label_quality": {} if label_quality is None else label_quality,
        "paper_usable": quality["paper_usable"],
        "paper_usable_reasons": quality["paper_usable_reasons"],
        "warning_flags": flags,
        "best_methods": _best_methods(headline),
    }


def _paper_usable_notes(
    raw: pd.DataFrame,
    headline: pd.DataFrame,
    failures: pd.DataFrame | None,
    interventions: pd.DataFrame | None,
    monitor: pd.DataFrame | None,
    decisions: pd.DataFrame | None,
    label_quality: dict[str, Any] | None,
) -> dict[str, Any]:
    reasons: list[str] = []
    if raw.empty:
        reasons.append("no episode rows were produced")
    nominal = raw[raw["method"] == "nominal_no_monitor"] if "method" in raw else pd.DataFrame()
    if nominal.empty:
        reasons.append("nominal controller baseline is missing")
    elif float(nominal["task_completed"].astype(float).mean()) < 0.8:
        reasons.append("nominal controller completion rate is below 0.8")
    budgeted = raw[~raw["method"].isin({"reference_unbounded", "nominal_no_monitor"})] if "method" in raw else pd.DataFrame()
    for column in ("budget_violation_count", "unsound_certificate_count", "reduction_failure_count"):
        if column in budgeted and int(budgeted[column].sum()):
            reasons.append(f"{column} is nonzero")
    if failures is not None and not failures.empty:
        reasons.append("failure_events.csv is nonempty")
    if interventions is None or interventions.empty:
        reasons.append("intervention_timeseries.csv is empty")
    if monitor is None or monitor.empty:
        reasons.append("monitor_timeseries.csv is empty")
    if decisions is None or decisions.empty:
        reasons.append("decision_features.csv has no reducer-decision rows")
    monitored = (
        headline[~headline["method"].isin({"nominal_no_monitor", "reference_unbounded"})]
        if "method" in headline
        else pd.DataFrame()
    )
    if monitored.empty:
        reasons.append("no bounded monitored methods were evaluated")
    elif "fallback_duration_fraction" in monitored:
        fallback = monitored["fallback_duration_fraction"].astype(float).dropna()
        if not fallback.empty and ((fallback <= 0.02).all() or (fallback >= 0.98).all()):
            reasons.append("fallback duration is saturated for all bounded methods")
    if "girard" in set(monitored.get("method", pd.Series(dtype=str))) and monitored.shape[0] > 1:
        girard = monitored[monitored["method"] == "girard"].iloc[0]
        comparable = monitored[monitored["method"] != "girard"]
        metric_names = [
            column
            for column in ("fallback_duration_fraction", "spurious_intervention_rate", "missed_violation_rate")
            if column in comparable and column in girard
        ]
        if metric_names:
            differs = any(
                np.any(np.abs(comparable[metric].astype(float).to_numpy() - float(girard[metric])) > 1e-9)
                for metric in metric_names
            )
            if not differs:
                reasons.append("bounded methods are indistinguishable from Girard on headline metrics")
    if label_quality and label_quality.get("decision_count", 0) and not label_quality.get("passes_gate", False):
        reasons.append("learned label diversity gate failed")
    return {
        "paper_usable": not reasons,
        "paper_usable_reasons": ["all headline quality gates passed"] if not reasons else reasons,
    }


def _json_list_is_empty(value: str) -> bool:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return False
    return isinstance(decoded, list) and not decoded


def _controller_validation_summary(raw: pd.DataFrame) -> pd.DataFrame:
    episode_count = int(raw.shape[0])
    if raw.empty:
        return pd.DataFrame(
            [
                {
                    "episode_count": 0,
                    "task_completion_rate": float("nan"),
                    "mean_gates_passed": float("nan"),
                    "collision_rate": float("nan"),
                    "constraint_violation_rate": float("nan"),
                    "mean_steps": float("nan"),
                    "mean_simulator_time": float("nan"),
                    "pass_gate": False,
                }
            ]
        )
    completed = raw["task_completed"].astype(bool)
    collision = raw["collision_episode"].astype(bool)
    constraint = raw["constraint_violation_episode"].astype(bool)
    pass_gate = (
        episode_count >= 10
        and int(completed.sum()) >= 8
        and float(collision.mean()) <= 0.2
        and float(constraint.mean()) <= 0.2
    )
    return pd.DataFrame(
        [
            {
                "episode_count": episode_count,
                "task_completion_rate": float(completed.mean()),
                "completed_episodes": int(completed.sum()),
                "mean_gates_passed": float(raw["gates_passed"].astype(float).mean()),
                "collision_rate": float(collision.mean()),
                "collision_episodes": int(collision.sum()),
                "constraint_violation_rate": float(constraint.mean()),
                "constraint_violation_episodes": int(constraint.sum()),
                "mean_steps": float(raw["steps"].astype(float).mean()),
                "mean_simulator_time": float(raw["simulator_time"].astype(float).mean()),
                "pass_gate": bool(pass_gate),
            }
        ]
    )


def _controller_validation_notes(raw: pd.DataFrame, summary: pd.DataFrame) -> dict[str, Any]:
    failed = raw[~raw["task_completed"].astype(bool)] if "task_completed" in raw else pd.DataFrame()
    failure_rows = []
    for _, row in failed.iterrows():
        failure_rows.append(
            {
                "seed": int(row["seed"]),
                "gates_passed": int(row["gates_passed"]),
                "collision_episode": bool(row["collision_episode"]),
                "constraint_violation_episode": bool(row["constraint_violation_episode"]),
                "steps": int(row["steps"]),
                "simulator_time": float(row["simulator_time"]),
            }
        )
    summary_row = summary.iloc[0].to_dict() if not summary.empty else {}
    return {
        "decision": "advance_to_level0_monitored" if bool(summary_row.get("pass_gate", False)) else "fix_controller_before_pzr",
        "success_gate": "at least 8/10 completed Level0 firmware nominal episodes without systematic collisions or constraints",
        "summary": summary_row,
        "failed_episodes": failure_rows,
    }


def _best_methods(headline: pd.DataFrame) -> dict[str, str]:
    best: dict[str, str] = {}
    if headline.empty:
        return best
    for metric, ascending in (
        ("spurious_intervention_rate", True),
        ("missed_violation_rate", True),
        ("task_completion_rate", False),
        ("collision_rate", True),
    ):
        if metric in headline:
            ranked = headline.sort_values([metric, "method"], ascending=[ascending, True])
            best[metric] = str(ranked.iloc[0]["method"])
    return best


def _profile_from_args(args: argparse.Namespace, *, controller_validation: bool = False) -> CorlProfile:
    profile = PROFILES[args.profile]
    updates = {
        "budget": args.budget,
        "horizon": args.horizon,
        "max_steps": args.max_steps,
        "train_seeds": args.train_seeds,
        "eval_seeds": args.eval_seeds,
        "dagger_iterations": args.dagger_iterations,
        "bootstrap_samples": args.bootstrap_samples,
        "distill_epochs": args.distill_epochs,
        "sensor_bias_bound": args.sensor_bias_bound,
        "sensor_noise_bound": args.sensor_noise_bound,
        "stream_memory_decay": args.stream_memory_decay,
        "fallback_hold_steps": args.fallback_hold_steps,
    }
    if controller_validation and args.eval_seeds is None:
        updates["eval_seeds"] = 10
    if controller_validation and args.train_seeds is None:
        updates["train_seeds"] = 0
    return replace(profile, **{key: value for key, value in updates.items() if value is not None})


def _validate_control_mode(args: argparse.Namespace, profile: CorlProfile) -> None:
    if args.profile == "smoke" and args.safe_control_python is None:
        return
    if args.safe_control_controller_mode == "debug_pid":
        if not args.allow_debug_pid:
            raise RuntimeError("debug_pid sidecar mode requires --allow-debug-pid and is diagnostic only")
        if profile.max_steps < 2200:
            raise RuntimeError("debug_pid diagnostics at 60 Hz require --max-steps >= 2200 for a 33 s episode")
    elif args.safe_control_controller_mode != "firmware":
        raise RuntimeError("--safe-control-controller-mode must be 'firmware' or 'debug_pid'")


def _resolve_out_dir(value: str | None) -> Path:
    if value:
        return Path(value)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("results") / f"corl-main-{timestamp}"


def _write_artifact_index(out_dir: Path, index_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in out_dir.rglob("*") if item.is_file()):
        if path == index_path:
            continue
        rows.append(
            {
                "path": str(path.relative_to(out_dir)),
                "kind": _artifact_kind(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    pd.DataFrame(rows, columns=("path", "kind", "bytes", "sha256")).to_csv(index_path, index=False)


def _write_archive(out_dir: Path, archive_path: Path) -> None:
    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(out_dir, arcname=out_dir.name)


def _artifact_kind(path: Path) -> str:
    if path.suffix == ".csv":
        return "csv"
    if path.suffix == ".json":
        return "json"
    if path.suffix == ".md":
        return "markdown"
    if path.suffix == ".pt":
        return "checkpoint"
    return path.suffix.lstrip(".") or "file"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "\n"
    columns = [str(column) for column in frame.columns]
    rows = [[_format_markdown_cell(value) for value in row] for row in frame.to_numpy()]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def _headline_quality_markdown(notes: dict[str, Any]) -> str:
    status = "PASS" if notes.get("paper_usable", False) else "FAIL"
    reasons = notes.get("paper_usable_reasons", [])
    lines = [f"# Headline Quality: {status}", ""]
    for reason in reasons:
        lines.append(f"- {reason}")
    lines.append("")
    return "\n".join(lines)


def _format_markdown_cell(value: Any) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for CoRL learned-policy training or checkpoint evaluation. "
            "Install the learning extra with `python -m pip install -e .[learning]`."
        ) from exc


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pzr-run-corl",
        description="Run CoRL-style safe-control-gym intervention experiments.",
    )
    parser.add_argument("--profile", choices=tuple(PROFILES), default="overnight")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--controller-validation", action="store_true")
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--safe-control-gym-root", type=str, default=None)
    parser.add_argument("--safe-control-python", type=str, default=None)
    parser.add_argument("--safe-control-config", type=str, default=None)
    parser.add_argument("--safe-control-controller-mode", choices=("firmware", "debug_pid"), default="firmware")
    parser.add_argument("--allow-debug-pid", action="store_true")
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--sensor-bias-bound", type=float, default=None)
    parser.add_argument("--sensor-noise-bound", type=float, default=None)
    parser.add_argument("--stream-memory-decay", type=float, default=None)
    parser.add_argument("--fallback-hold-steps", type=int, default=None)
    parser.add_argument("--train-seeds", type=int, default=None)
    parser.add_argument("--eval-seeds", type=int, default=None)
    parser.add_argument("--dagger-iterations", type=int, default=None)
    parser.add_argument("--dagger-expert", choices=DAGGER_EXPERTS, default="mpc_wide_fixed_girard")
    parser.add_argument("--method-set", choices=CORL_METHOD_SETS, default="core")
    parser.add_argument("--learned-mode", choices=LEARNED_MODES, default="none")
    parser.add_argument("--learned-checkpoint", type=str, default=None)
    parser.add_argument("--include-failed-learned", action="store_true")
    parser.add_argument("--fail-on-unusable", action="store_true")
    parser.add_argument("--calibration-seeds", type=int, default=5)
    parser.add_argument("--calibration-max-steps", type=int, default=1000)
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument("--distill-epochs", type=int, default=None)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
