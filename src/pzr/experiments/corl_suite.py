"""CoRL-style closed-loop robotics intervention experiment suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import tarfile
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
)
from pzr.learning.distill_cli import train_policy
from pzr.learning.features import DECISION_FEATURE_NAMES, DECISION_FEATURE_SCHEMA_VERSION, decision_feature_values
from pzr.monitoring.base import evaluate_triggers
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
    generator_memory_decay: float
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
        generator_memory_decay=0.65,
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
        generator_memory_decay=0.85,
        distill_epochs=150,
        distill_batch_size=64,
    ),
    "paper": CorlProfile(
        budget=8,
        horizon=6,
        max_steps=1500,
        train_seeds=40,
        eval_seeds=100,
        dagger_iterations=4,
        bootstrap_samples=10000,
        sensor_bias_bound=0.015,
        sensor_noise_bound=0.03,
        generator_memory_decay=0.85,
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)
    if args.preflight:
        result = preflight_safe_control_gym(
            profile=args.profile,
            safe_control_gym_root=args.safe_control_gym_root,
            safe_control_python=args.safe_control_python,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0 if result.ok else 2
    run_corl_suite(args)
    return 0


def run_corl_suite(args: argparse.Namespace) -> Path:
    """Run the CoRL robotics suite and return the output directory."""

    _require_torch()
    profile = _profile_from_args(args)
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
        "python": platform.python_version(),
        "platform": platform.platform(),
        "preflight": preflight.to_dict(),
        "steps": [],
    }

    training_rows = _run_dagger_training(args, profile, out_dir)
    manifest["steps"].append(
        {
            "kind": "dagger_training",
            "rows": int(training_rows.shape[0]),
            "checkpoint": "learning/dagger_final.pt",
        }
    )

    checkpoint = out_dir / "learning" / "dagger_final.pt"
    episode_rows, intervention_rows, monitor_rows, decision_rows = _run_evaluation(
        args,
        profile,
        checkpoint,
    )
    manifest["steps"].append(
        {
            "kind": "heldout_evaluation",
            "episodes": len(episode_rows),
            "eval_seeds": profile.eval_seeds,
        }
    )

    raw = pd.DataFrame(episode_rows)
    interventions = pd.DataFrame(intervention_rows)
    monitor = pd.DataFrame(monitor_rows)
    decisions = pd.DataFrame(decision_rows)
    selection = _selection_summary(decisions)
    sequence_summary = _sequence_summary(decisions)
    headline = _headline_table(raw, profile.bootstrap_samples, seed=args.bootstrap_seed)
    notes = _analysis_notes(raw, headline)

    raw.to_csv(out_dir / "raw_episodes.csv", index=False)
    interventions.to_csv(out_dir / "intervention_timeseries.csv", index=False)
    monitor.to_csv(out_dir / "monitor_timeseries.csv", index=False)
    decisions.to_csv(out_dir / "decision_features.csv", index=False)
    selection.to_csv(out_dir / "selection_summary.csv", index=False)
    sequence_summary.to_csv(out_dir / "predicted_sequence_summary.csv", index=False)
    headline.to_csv(out_dir / "headline_table.csv", index=False)
    (out_dir / "headline_table.md").write_text(_markdown_table(headline), encoding="utf-8")
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
    _write_artifact_index(out_dir, out_dir / "artifact_index.csv")
    if not args.no_archive:
        _write_archive(out_dir, out_dir.with_suffix(".tar.gz"))
    return out_dir


def _run_dagger_training(
    args: argparse.Namespace,
    profile: CorlProfile,
    out_dir: Path,
) -> pd.DataFrame:
    aggregate_rows: list[dict[str, Any]] = []
    expert = _expert_method()
    learned_checkpoint: Path | None = None
    for iteration in range(profile.dagger_iterations + 1):
        method = expert if iteration == 0 or learned_checkpoint is None else _learned_method(learned_checkpoint)
        rows: list[dict[str, Any]] = []
        for seed in range(profile.train_seeds):
            client = make_env_client(
                profile=args.profile,
                safe_control_gym_root=args.safe_control_gym_root,
                safe_control_python=args.safe_control_python,
            )
            try:
                _, _, _, decisions = _run_episode(
                    client,
                    profile,
                    method,
                    seed,
                    phase="train",
                    expert_for_labels=expert if method.kind == "learned" else None,
                )
                rows.extend(decisions)
            finally:
                client.close()
        for row in rows:
            row["dagger_iteration"] = iteration
        aggregate_rows.extend(rows)
        if not aggregate_rows:
            raise RuntimeError("DAgger produced no reducer-decision rows; lower budget or increase monitor memory")
        dataset = pd.DataFrame(aggregate_rows)
        dataset.to_csv(out_dir / "dagger_dataset.csv", index=False)
        learned_checkpoint = out_dir / "learning" / f"dagger_iter{iteration}.pt"
        _train_checkpoint(dataset, learned_checkpoint, profile, iteration)
    final = out_dir / "learning" / "dagger_final.pt"
    if learned_checkpoint is None:
        raise RuntimeError("DAgger did not produce a checkpoint")
    shutil.copy2(learned_checkpoint, final)
    metrics = {
        "iterations": profile.dagger_iterations,
        "row_count": len(aggregate_rows),
        "final_checkpoint": "learning/dagger_final.pt",
    }
    (out_dir / "learning" / "dagger_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return pd.DataFrame(aggregate_rows)


def _run_evaluation(
    args: argparse.Namespace,
    profile: CorlProfile,
    checkpoint: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    methods = (
        "nominal_no_monitor",
        "reference_unbounded",
        *_headline_methods(),
        _learned_method(checkpoint),
    )
    raw_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    monitor_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for method in methods:
        for seed in range(profile.eval_seeds):
            client = make_env_client(
                profile=args.profile,
                safe_control_gym_root=args.safe_control_gym_root,
                safe_control_python=args.safe_control_python,
            )
            try:
                episode, interventions, monitor, decisions = _run_episode(
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
    return raw_rows, intervention_rows, monitor_rows, decision_rows


def _run_episode(
    client: IrosEnvClient,
    profile: CorlProfile,
    method: MethodSpec | str,
    seed: int,
    *,
    phase: str,
    expert_for_labels: MethodSpec | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    snapshot = client.reset(seed)
    scenario = client.scenario
    monitor = IrosGateMonitor(
        scenario,
        generator_memory_decay=profile.generator_memory_decay,
    )
    state = monitor.initial_state()
    sensor = NoisySensorModel(
        bias_bound=profile.sensor_bias_bound,
        noise_bound=profile.sensor_noise_bound,
        seed=10_000 + seed,
    )
    manager = InterventionManager(
        client.fallback_command(snapshot),
        fallback_hold_steps=2,
        expected_gate_count=len(scenario.gates),
    )
    policy = None if isinstance(method, str) else _make_policy(method, monitor, _benchmark_config(profile))
    expert_policy = (
        None
        if expert_for_labels is None
        else _make_policy(expert_for_labels, monitor, _benchmark_config(profile))
    )
    method_name = method if isinstance(method, str) else method.name
    method_kind = method if isinstance(method, str) else method.kind
    interventions: list[dict[str, Any]] = []
    monitor_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
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

        if method_name == "nominal_no_monitor":
            command = nominal
            oracle = monitor.oracle_verdicts(snapshot.pose, snapshot.velocity, snapshot.target_gate_index)
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
                    expert_decision = _reduce_with_policy(
                        expert_policy,
                        expert_for_labels,
                        monitor,
                        state,
                        profile,
                        snapshot,
                    )
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
                        )
                    )
                reduction_start = perf_counter()
                try:
                    decision = _reduce_with_policy(policy, method, monitor, state, profile, snapshot)
                except Exception:
                    reduction_failure_count += 1
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
                            )
                        )
            verdicts = evaluate_triggers(state.zonotope, monitor.triggers)
            oracle = monitor.oracle_verdicts(snapshot.pose, snapshot.velocity, snapshot.target_gate_index)
            command = manager.choose_command(
                nominal,
                verdicts,
                oracle,
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
                "time": snapshot.time,
                "monitor_triggered": any(v.status == "violation" for v in verdicts),
                "oracle_violated": any(v.status == "violation" for v in oracle),
                "fallback_active": fallback_active,
                "collision": next_snapshot.collision,
                "constraint_violation": next_snapshot.constraint_violation,
                "gates_passed": next_snapshot.gates_passed,
                "task_completed": next_snapshot.task_completed,
            }
        )
        monitor_rows.append(
            {
                "phase": phase,
                "method": method_name,
                "method_kind": str(method_kind),
                "seed": seed,
                "step": step + 1,
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
        "time_to_target": metrics.time_to_target if metrics.time_to_target is not None else np.nan,
        "mean_reducer_latency_ms": 1000.0 * metrics.reducer_latency_seconds / max(1, sum(metrics.reducer_choices.values())),
        "budget_violation_count": int(sum(row["budget_violation"] for row in monitor_rows)),
        "unsound_certificate_count": int(sum(row["unsound_certificate"] for row in monitor_rows)),
        "reduction_failure_count": reduction_failure_count,
        "reducer_choices": json.dumps(metrics.reducer_choices, sort_keys=True),
    }
    return episode, interventions, monitor_rows, decisions


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
    dt = 0.05
    pose = snapshot.pose.copy()
    velocity = snapshot.velocity.copy()
    for index in range(profile.horizon):
        gate = monitor.scenario.gate(snapshot.target_gate_index)
        velocity = 0.85 * velocity + 0.15 * (gate.center - pose)
        pose = pose + dt * velocity
        observations.append(
            IrosObservation(
                pose,
                velocity,
                target_gate_index=snapshot.target_gate_index,
                noise_radius=np.full(6, profile.sensor_bias_bound + profile.sensor_noise_bound),
                time=snapshot.time + (index + 1) * dt,
            )
        )
    return tuple(observations)


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
        "candidate_reducer_names": json.dumps([]),
        "no_op_selected": decision.is_no_op,
        **features,
    }


def _train_checkpoint(
    dataset: pd.DataFrame,
    checkpoint: Path,
    profile: CorlProfile,
    iteration: int,
) -> None:
    train_policy(
        argparse.Namespace(
            data=_write_training_dataset(dataset, checkpoint),
            expert_method="mpc_focused_sequence",
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
            dagger_metadata={"iteration": iteration, "row_count": int(dataset.shape[0])},
        )
    )


def _write_training_dataset(dataset: pd.DataFrame, checkpoint: Path) -> Path:
    path = checkpoint.with_suffix(".dataset.csv")
    dataset.to_csv(path, index=False)
    return path


def _headline_methods() -> tuple[MethodSpec, ...]:
    return (
        MethodSpec.static("box", _protected(BoxReducer)),
        MethodSpec.static("girard", _protected(GirardReducer)),
        MethodSpec.static("combastel", _protected(CombastelReducer)),
        MethodSpec.static("pca", _protected(PcaReducer)),
        MethodSpec.static("keep_norm", _protected(ScoredKeepReducer.by_norm)),
        MethodSpec.static("keep_calibration_aware", _protected(ScoredKeepReducer.calibration_aware)),
        _expert_method(),
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


def _expert_method() -> MethodSpec:
    return MethodSpec.sequence_mpc(
        "mpc_focused_sequence",
        focused_geometry_reducer_factories(),
        _protected(BoxReducer),
    )


def _learned_method(checkpoint: Path) -> MethodSpec:
    return MethodSpec(
        "learned_dagger",
        "learned",
        mpc_reducer_factories=focused_geometry_reducer_factories(),
        mpc_fallback_reducer_factory=_protected(BoxReducer),
        learned_policy_path=str(checkpoint),
    )


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


def _analysis_notes(raw: pd.DataFrame, headline: pd.DataFrame) -> dict[str, Any]:
    budgeted = raw[~raw["method"].isin({"reference_unbounded", "nominal_no_monitor"})]
    flags: list[str] = []
    for column in ("budget_violation_count", "unsound_certificate_count", "reduction_failure_count"):
        count = int(budgeted[column].sum()) if column in budgeted else 0
        if count:
            flags.append(f"{column}={count}")
    if not headline.empty and "mpc_focused_sequence" in set(headline["method"]):
        mpc = headline[headline["method"] == "mpc_focused_sequence"].iloc[0]
        ref = headline[headline["method"] == "reference_unbounded"].iloc[0]
        if float(mpc["missed_violation_rate"]) > float(ref["missed_violation_rate"]):
            flags.append("mpc_focused_sequence_increases_missed_violation_rate")
    return {
        "soundness_checks": {
            "budget_violation_count": int(budgeted["budget_violation_count"].sum()) if not budgeted.empty else 0,
            "unsound_certificate_count": int(budgeted["unsound_certificate_count"].sum()) if not budgeted.empty else 0,
            "reduction_failure_count": int(budgeted["reduction_failure_count"].sum()) if not budgeted.empty else 0,
        },
        "warning_flags": flags,
        "best_methods": _best_methods(headline),
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


def _profile_from_args(args: argparse.Namespace) -> CorlProfile:
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
    }
    return replace(profile, **{key: value for key, value in updates.items() if value is not None})


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
            "PyTorch is required for the CoRL suite because DAgger is always included. "
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
    parser.add_argument("--safe-control-gym-root", type=str, default=None)
    parser.add_argument("--safe-control-python", type=str, default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--train-seeds", type=int, default=None)
    parser.add_argument("--eval-seeds", type=int, default=None)
    parser.add_argument("--dagger-iterations", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument("--distill-epochs", type=int, default=None)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
