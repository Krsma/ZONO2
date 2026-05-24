"""PPO residual-controller training with monitor action shielding."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
import warnings
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - exercised when optional extra is absent.
    gym = None  # type: ignore[assignment]

from pzr.control.policies import ReductionDecision, StaticReductionPolicy
from pzr.experiments.benchmark import BenchmarkConfig, MethodSpec, _make_policy, wide_rollout_reducer_factories
from pzr.experiments.corl_suite import (
    FAILURE_EVENT_COLUMNS,
    _json_safe,
    _oracle_verdicts,
    _predicted_observations,
    _write_artifact_index,
)
from pzr.learning.ppo import PPOConfig, PPOTrainer, RolloutBuffer
from pzr.monitoring.base import Verdict, evaluate_triggers
from pzr.reduction.paper_reducers import GirardReducer
from pzr.reduction.reducers import BoxReducer, ProtectedReducer
from pzr.robotics import IrosGateMonitor, IrosObservation, NoisySensorModel, make_env_client, preflight_safe_control_gym
from pzr.robotics.iros import IROS_STREAM_NAMES, iros_stream_values
from pzr.robotics.safe_control_gym import IrosEnvClient, IrosEnvSnapshot


SHIELD_METHODS = {
    "unshielded": "ppo_unshielded",
    "shield_box": "ppo_shield_box",
    "shield_girard": "ppo_shield_girard",
    "shield_pzr": "ppo_shield_pzr",
    "ppo_unshielded": "ppo_unshielded",
    "ppo_shield_box": "ppo_shield_box",
    "ppo_shield_girard": "ppo_shield_girard",
    "ppo_shield_pzr": "ppo_shield_pzr",
}

PPO_BACKENDS = ("sb3", "custom")

TRAINING_CURVE_COLUMNS = (
    "method",
    "update",
    "environment_steps",
    "episodes",
    "mean_episode_reward",
    "completion_rate",
    "mean_gates_passed",
    "collision_rate",
    "constraint_violation_rate",
    "shield_rate",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
)

EPISODE_COLUMNS = (
    "phase",
    "method",
    "seed",
    "episode_index",
    "environment_steps",
    "episode_steps",
    "reward",
    "task_completed",
    "gates_passed",
    "collision",
    "constraint_violation",
    "shield_interventions",
    "shield_rate",
    "partial",
)

SHIELD_TIMESERIES_COLUMNS = (
    "phase",
    "method",
    "seed",
    "episode_index",
    "step",
    "environment_steps",
    "time",
    "monitor_triggered",
    "monitor_trigger_names",
    "oracle_violated",
    "oracle_trigger_names",
    "shield_active",
    "reducer_name",
    "reduction_applied",
    "generator_count",
    "budget_violation",
    "unsound_certificate",
    "predicted_cost",
    "predicted_sequence",
    "evaluated_sequence_count",
    "pruned_sequence_count",
    "command_source",
    "candidate_ax",
    "candidate_ay",
    "candidate_az",
    "applied_ax",
    "applied_ay",
    "applied_az",
    "pose_x",
    "pose_y",
    "pose_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "gates_passed",
    "task_completed",
    "collision",
    "constraint_violation",
    *(f"stream_{name}" for name in IROS_STREAM_NAMES),
)


@dataclass(frozen=True)
class CorlLearningProfile:
    """Default runtime settings for PPO controller training."""

    level: str
    budget: int
    horizon: int
    max_episode_steps: int
    sensor_bias_bound: float
    sensor_noise_bound: float
    stream_memory_decay: float


PROFILES = {
    "smoke": CorlLearningProfile(
        level="smoke",
        budget=8,
        horizon=2,
        max_episode_steps=40,
        sensor_bias_bound=0.01,
        sensor_noise_bound=0.02,
        stream_memory_decay=0.65,
    ),
    "level0": CorlLearningProfile(
        level="level0",
        budget=8,
        horizon=4,
        max_episode_steps=1000,
        sensor_bias_bound=0.015,
        sensor_noise_bound=0.03,
        stream_memory_decay=0.85,
    ),
    "level1": CorlLearningProfile(
        level="level1",
        budget=8,
        horizon=6,
        max_episode_steps=1500,
        sensor_bias_bound=0.015,
        sensor_noise_bound=0.03,
        stream_memory_decay=0.85,
    ),
}


@dataclass
class EpisodeAccumulator:
    """Mutable per-episode training or evaluation totals."""

    episode_index: int = 0
    step: int = 0
    reward: float = 0.0
    task_completed: bool = False
    gates_passed: int = 0
    collision: bool = False
    constraint_violation: bool = False
    shield_interventions: int = 0


@dataclass
class ControllerRuntime:
    """One shielded residual-control episode state."""

    client: IrosEnvClient
    profile: CorlLearningProfile
    method: str
    seed: int
    phase: str
    residual_scale: float
    accel_clip: float
    global_step_start: int = 0
    snapshot: IrosEnvSnapshot | None = None
    monitor: IrosGateMonitor | None = None
    state: Any = None
    sensor: NoisySensorModel | None = None
    policy: Any = None
    previous_command: np.ndarray | None = None
    previous_shield_active: bool = False
    episode: EpisodeAccumulator | None = None
    episode_seed_base: int | None = None

    def reset(self, seed: int | None = None, *, preserve_seed_base: bool = False) -> np.ndarray:
        episode_index = 0 if self.episode is None else int(self.episode.episode_index)
        if seed is not None:
            self.seed = int(seed)
            if not preserve_seed_base:
                self.episode_seed_base = int(seed)
        self.snapshot = self.client.reset(self.seed)
        self.monitor = IrosGateMonitor(
            self.client.scenario,
            stream_memory_decay=self.profile.stream_memory_decay,
        )
        self.state = self.monitor.initial_state()
        self.sensor = NoisySensorModel(
            bias_bound=self.profile.sensor_bias_bound,
            noise_bound=self.profile.sensor_noise_bound,
            seed=20_000 + self.seed,
        )
        self.policy = _make_shield_policy(self.method, self.monitor, self.profile)
        self.previous_command = np.zeros(3, dtype=float)
        self.previous_shield_active = False
        self.episode = EpisodeAccumulator(episode_index=episode_index)
        return encode_observation(
            self.client.scenario,
            self.snapshot,
            self.previous_command,
            self.previous_shield_active,
        )

    def step(
        self,
        policy_action: np.ndarray,
        *,
        environment_steps: int,
        auto_reset: bool = True,
    ) -> tuple[np.ndarray, float, bool, dict[str, Any], dict[str, Any] | None]:
        if self.snapshot is None or self.monitor is None or self.sensor is None or self.episode is None:
            raise RuntimeError("controller runtime must be reset before stepping")
        snapshot = self.snapshot
        monitor = self.monitor
        episode = self.episode
        planner_hint = self.client.nominal_command(snapshot)
        candidate = residual_acceleration_command(
            np.asarray(policy_action, dtype=float),
            planner_hint,
            snapshot,
            self.client.scenario,
            residual_scale=self.residual_scale,
            accel_clip=self.accel_clip,
        )
        true_streams = iros_stream_values(
            self.client.scenario,
            IrosObservation(snapshot.pose, snapshot.velocity, target_gate_index=snapshot.target_gate_index),
        )
        verdicts: tuple[Verdict, ...] = ()
        oracle = _oracle_verdicts(monitor, snapshot)
        reducer_name = ""
        reduction_applied = False
        reduction_seconds = 0.0
        unsound = False
        budget_violation = False
        predicted_cost = 0.0
        predicted_sequence: tuple[str, ...] = ()
        evaluated_sequences = 0
        pruned_sequences = 0

        observation = self.sensor.observe(
            snapshot.pose,
            snapshot.velocity,
            target_gate_index=snapshot.target_gate_index,
            command=candidate,
            time=snapshot.time,
        )
        result = monitor.step(self.state, observation)
        self.state = result.state
        if self.state.zonotope.generator_count > self.profile.budget and self.method != "ppo_unshielded":
            reduction_start = perf_counter()
            decision = _reduce_with_shield_policy(
                self.policy,
                self.method,
                monitor,
                self.state,
                self.profile,
                snapshot,
            )
            reduction_seconds = perf_counter() - reduction_start
            self.state = decision.state
            reducer_name = decision.reducer_name
            reduction_applied = not decision.is_no_op
            unsound = not decision.result.certificate.is_sound
            predicted_cost = decision.predicted_cost
            predicted_sequence = decision.predicted_sequence
            evaluated_sequences = decision.evaluated_sequences
            pruned_sequences = decision.pruned_sequences
        verdicts = evaluate_triggers(self.state.zonotope, monitor.triggers)
        monitor_triggered = any(verdict.status == "violation" for verdict in verdicts)
        shield_enabled = self.method != "ppo_unshielded"
        shield_active = bool(shield_enabled and monitor_triggered)
        applied = shield_acceleration(snapshot, accel_clip=self.accel_clip) if shield_active else candidate
        next_snapshot = self.client.step(applied)
        new_gates = max(0, int(next_snapshot.gates_passed) - int(snapshot.gates_passed))
        reward = controller_reward(next_snapshot, applied, shield_active, new_gates)
        budget_violation = self.state.zonotope.generator_count > self.profile.budget and self.method != "ppo_unshielded"

        episode.step += 1
        episode.reward += reward
        episode.task_completed = episode.task_completed or bool(next_snapshot.task_completed)
        episode.gates_passed = max(episode.gates_passed, int(next_snapshot.gates_passed))
        episode.collision = episode.collision or bool(next_snapshot.collision)
        episode.constraint_violation = episode.constraint_violation or bool(next_snapshot.constraint_violation)
        episode.shield_interventions += int(shield_active)

        row = {
            "phase": self.phase,
            "method": self.method,
            "seed": self.seed,
            "episode_index": episode.episode_index,
            "step": episode.step,
            "environment_steps": environment_steps,
            "time": snapshot.time,
            "monitor_triggered": monitor_triggered,
            "monitor_trigger_names": json.dumps(_violated_trigger_names(verdicts)),
            "oracle_violated": any(verdict.status == "violation" for verdict in oracle),
            "oracle_trigger_names": json.dumps(_violated_trigger_names(oracle)),
            "shield_active": shield_active,
            "reducer_name": reducer_name,
            "reduction_applied": reduction_applied,
            "generator_count": self.state.zonotope.generator_count,
            "budget_violation": budget_violation,
            "unsound_certificate": unsound,
            "predicted_cost": predicted_cost,
            "predicted_sequence": json.dumps(list(predicted_sequence)),
            "evaluated_sequence_count": evaluated_sequences,
            "pruned_sequence_count": pruned_sequences,
            "command_source": "shield" if shield_active else "policy",
            "candidate_ax": float(candidate[0]),
            "candidate_ay": float(candidate[1]),
            "candidate_az": float(candidate[2]),
            "applied_ax": float(applied[0]),
            "applied_ay": float(applied[1]),
            "applied_az": float(applied[2]),
            "pose_x": float(snapshot.pose[0]),
            "pose_y": float(snapshot.pose[1]),
            "pose_z": float(snapshot.pose[2]),
            "velocity_x": float(snapshot.velocity[0]),
            "velocity_y": float(snapshot.velocity[1]),
            "velocity_z": float(snapshot.velocity[2]),
            "gates_passed": int(next_snapshot.gates_passed),
            "task_completed": bool(next_snapshot.task_completed),
            "collision": bool(next_snapshot.collision),
            "constraint_violation": bool(next_snapshot.constraint_violation),
            **{f"stream_{name}": float(value) for name, value in zip(IROS_STREAM_NAMES, true_streams)},
        }
        done = bool(next_snapshot.done or episode.step >= self.profile.max_episode_steps)
        self.snapshot = next_snapshot
        self.previous_command = applied.copy()
        self.previous_shield_active = shield_active
        next_observation = encode_observation(
            self.client.scenario,
            next_snapshot,
            self.previous_command,
            self.previous_shield_active,
        )
        episode_row = self.finish_episode(environment_steps, partial=False) if done else None
        if done:
            episode.episode_index += 1
            if auto_reset:
                seed_base = self.seed if self.episode_seed_base is None else self.episode_seed_base
                next_observation = self.reset(seed_base + episode.episode_index, preserve_seed_base=True)
                self.episode.episode_index = episode.episode_index
        _ = reduction_seconds
        return next_observation, reward, done, row, episode_row

    def finish_episode(self, environment_steps: int, *, partial: bool) -> dict[str, Any]:
        if self.episode is None:
            raise RuntimeError("controller runtime must be reset before finishing an episode")
        episode = self.episode
        steps = max(1, episode.step)
        return {
            "phase": self.phase,
            "method": self.method,
            "seed": self.seed,
            "episode_index": episode.episode_index,
            "environment_steps": environment_steps,
            "episode_steps": episode.step,
            "reward": episode.reward,
            "task_completed": episode.task_completed,
            "gates_passed": episode.gates_passed,
            "collision": episode.collision,
            "constraint_violation": episode.constraint_violation,
            "shield_interventions": episode.shield_interventions,
            "shield_rate": episode.shield_interventions / steps,
            "partial": partial,
        }


@dataclass(frozen=True)
class WorkerSpec:
    """One subprocess worker assigned to one canonical PPO method."""

    method: str
    method_index: int
    worker_dir: Path
    command: tuple[str, ...]


@dataclass(frozen=True)
class WorkerResult:
    """Completed subprocess status and log locations."""

    method: str
    returncode: int
    worker_dir: Path
    stdout_log: Path
    stderr_log: Path


class Sb3ControllerEnv(gym.Env if gym is not None else object):  # type: ignore[misc]
    """Gymnasium adapter that records the existing CoRL controller artifacts."""

    metadata = {"render_modes": ()}

    def __init__(
        self,
        args: argparse.Namespace,
        profile: CorlLearningProfile,
        method: str,
        seed: int,
        phase: str,
        artifacts: dict[str, list[dict[str, Any]]],
        episode_sink: list[dict[str, Any]],
        *,
        environment_steps: int = 0,
        training: bool = True,
    ) -> None:
        gym, spaces = _require_gymnasium()
        self._gym = gym
        self.args = args
        self.profile = profile
        self.method = method
        self.seed = int(seed)
        self.phase = phase
        self.artifacts = artifacts
        self.episode_sink = episode_sink
        self.training = bool(training)
        self.environment_steps = int(environment_steps)
        self.client = make_env_client(
            profile="smoke" if args.profile == "smoke" else "overnight",
            safe_control_gym_root=args.safe_control_gym_root,
            safe_control_python=args.safe_control_python,
            safe_control_config=args.safe_control_config,
            safe_control_controller_mode=args.safe_control_controller_mode,
            allow_debug_pid=args.allow_debug_pid,
            fake_max_steps=profile.max_episode_steps,
        )
        self.runtime = ControllerRuntime(
            self.client,
            profile,
            method,
            self.seed,
            phase,
            residual_scale=args.residual_scale,
            accel_clip=args.accel_clip,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)
        self.step_rewards: list[float] = []
        self._closed = False

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        _ = options
        if seed is not None:
            self.seed = int(seed)
        reset_seed = self.seed
        if seed is None and self.runtime.episode is not None and self.runtime.episode.step:
            seed_base = self.runtime.seed if self.runtime.episode_seed_base is None else self.runtime.episode_seed_base
            reset_seed = int(seed_base + self.runtime.episode.episode_index)
        observation = self.runtime.reset(reset_seed, preserve_seed_base=(seed is None))
        self.seed = int(self.runtime.seed)
        return observation, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        step = self.environment_steps + 1 if self.training else self.environment_steps
        observation, reward, done, row, episode_row = self.runtime.step(
            action,
            environment_steps=step,
            auto_reset=False,
        )
        self.artifacts["shield_timeseries"].append(row)
        self.step_rewards.append(float(reward))
        if self.training:
            self.environment_steps += 1
            if self.args.debug_raise_after_steps is not None and self.environment_steps >= self.args.debug_raise_after_steps:
                raise RuntimeError("debug controller-training failure")
        if episode_row is not None:
            self.episode_sink.append(episode_row)
        return observation, float(reward), bool(done), False, {}

    def close(self) -> None:
        if not self._closed:
            self.client.close()
            self._closed = True


def main(argv: Sequence[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)
    run_corl_controller_training(args)
    return 0


def run_corl_controller_training(args: argparse.Namespace) -> Path:
    """Train CoRL PPO controllers, optionally in method-level worker processes."""

    if int(args.jobs) < 1:
        raise ValueError("--jobs must be at least 1")
    if int(args.worker_threads) < 1:
        raise ValueError("--worker-threads must be at least 1")
    _apply_worker_thread_caps(int(args.worker_threads))
    if int(args.jobs) > 1:
        return _run_parallel_corl_controller_training(args)
    return _run_serial_corl_controller_training(args)


def _run_serial_corl_controller_training(args: argparse.Namespace) -> Path:
    """Train and evaluate PPO residual controllers with optional monitor shielding."""

    profile = _profile_from_args(args)
    methods = _parse_method_set(args.method_set)
    out_dir = Path(args.out) if args.out else Path("results") / f"corl-ppo-{args.profile}"
    if out_dir.exists():
        if not args.force:
            raise FileExistsError(f"output directory already exists: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "policy_checkpoints").mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "suite": "pzr_corl_ppo_controller",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "profile_config": asdict(profile),
        "methods": list(methods),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ppo_backend": args.ppo_backend,
        "ppo": _ppo_manifest_config(args),
        "status": "running",
        "steps": [],
    }
    artifacts: dict[str, list[dict[str, Any]]] = {
        "training_curve": [],
        "raw_train_episodes": [],
        "eval_episodes": [],
        "shield_timeseries": [],
        "failure_events": [],
    }
    try:
        if args.skip_preflight:
            manifest["preflight"] = {
                "ok": True,
                "skipped": True,
                "messages": ["preflight was already run by the parent process"],
            }
        else:
            preflight = preflight_safe_control_gym(
                profile="smoke" if args.profile == "smoke" else "overnight",
                safe_control_gym_root=args.safe_control_gym_root,
                safe_control_python=args.safe_control_python,
                safe_control_config=args.safe_control_config,
                safe_control_controller_mode=args.safe_control_controller_mode,
                allow_debug_pid=args.allow_debug_pid,
            )
            manifest["preflight"] = preflight.to_dict()
            if not preflight.ok:
                raise RuntimeError(
                    "CoRL controller-training preflight failed:\n"
                    + "\n".join(f"- {message}" for message in preflight.messages)
                )
        offset = int(args.method_index_offset)
        for index, method in enumerate(methods):
            _train_one_method(args, profile, method, offset + index, out_dir, artifacts, manifest)
        manifest["status"] = "success"
        _write_learning_artifacts(out_dir, args, profile, manifest, artifacts)
    except Exception as exc:
        artifacts["failure_events"].append(
            _controller_failure_event(
                phase="train",
                method="",
                seed=-1,
                step=-1,
                event_type="controller_training_abort",
                exc=exc,
            )
        )
        manifest["status"] = "failed"
        manifest["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_learning_artifacts(out_dir, args, profile, manifest, artifacts)
        raise
    return out_dir


def _run_parallel_corl_controller_training(args: argparse.Namespace) -> Path:
    """Run one PPO method per subprocess and aggregate worker artifacts."""

    profile = _profile_from_args(args)
    methods = _parse_method_set(args.method_set)
    out_dir = Path(args.out) if args.out else Path("results") / f"corl-ppo-{args.profile}"
    if out_dir.exists():
        if not args.force:
            raise FileExistsError(f"output directory already exists: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "policy_checkpoints").mkdir(parents=True, exist_ok=True)

    preflight = preflight_safe_control_gym(
        profile="smoke" if args.profile == "smoke" else "overnight",
        safe_control_gym_root=args.safe_control_gym_root,
        safe_control_python=args.safe_control_python,
        safe_control_config=args.safe_control_config,
        safe_control_controller_mode=args.safe_control_controller_mode,
        allow_debug_pid=args.allow_debug_pid,
    )
    if not preflight.ok:
        raise RuntimeError(
            "CoRL controller-training preflight failed:\n"
            + "\n".join(f"- {message}" for message in preflight.messages)
        )

    manifest: dict[str, Any] = {
        "suite": "pzr_corl_ppo_controller",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "profile_config": asdict(profile),
        "methods": list(methods),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ppo_backend": args.ppo_backend,
        "ppo": _ppo_manifest_config(args),
        "status": "running",
        "preflight": preflight.to_dict(),
        "steps": [],
    }
    specs = _worker_specs(args, methods, out_dir)
    results = _run_worker_specs(specs, jobs=int(args.jobs), worker_threads=int(args.worker_threads))
    artifacts, worker_statuses, worker_steps = _aggregate_worker_artifacts(out_dir, specs, results)
    manifest["steps"].extend(worker_steps)
    manifest["parallel"] = {
        "jobs": int(args.jobs),
        "worker_threads": int(args.worker_threads),
        "worker_count": len(specs),
        "worker_dirs": [str(spec.worker_dir.relative_to(out_dir)) for spec in specs],
        "worker_statuses": worker_statuses,
    }

    failed = [status for status in worker_statuses if not status["success"]]
    if failed:
        for status in failed:
            artifacts["failure_events"].append(_parallel_worker_failure_event(status))
        manifest["status"] = "failed"
        manifest["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_learning_artifacts(out_dir, args, profile, manifest, artifacts)
        failed_methods = ", ".join(str(status["method"]) for status in failed)
        raise RuntimeError(f"parallel CoRL controller workers failed: {failed_methods}")

    manifest["status"] = "success"
    _write_learning_artifacts(out_dir, args, profile, manifest, artifacts)
    return out_dir


def _worker_specs(args: argparse.Namespace, methods: Sequence[str], out_dir: Path) -> list[WorkerSpec]:
    workers_dir = out_dir / "workers"
    return [
        WorkerSpec(
            method=method,
            method_index=index,
            worker_dir=workers_dir / method,
            command=tuple(_worker_command(args, method, index, workers_dir / method)),
        )
        for index, method in enumerate(methods)
    ]


def _ppo_manifest_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "backend": args.ppo_backend,
        "requested_total_steps": int(args.total_steps),
        "requested_eval_interval": int(args.eval_interval),
        "requested_rollout_steps": int(args.rollout_steps),
        "minibatch_size": int(args.minibatch_size),
        "update_epochs": int(args.update_epochs),
    }


def _worker_command(args: argparse.Namespace, method: str, method_index: int, worker_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pzr.experiments.corl_learning",
        "--profile",
        str(args.profile),
        "--method-set",
        method,
        "--out",
        str(worker_dir),
        "--force",
        "--jobs",
        "1",
        "--worker-threads",
        str(args.worker_threads),
        "--ppo-backend",
        str(args.ppo_backend),
        "--method-index-offset",
        str(method_index),
        "--skip-preflight",
        "--safe-control-controller-mode",
        str(args.safe_control_controller_mode),
        "--total-steps",
        str(args.total_steps),
        "--eval-interval",
        str(args.eval_interval),
        "--eval-seeds",
        str(args.eval_seeds),
        "--seed",
        str(args.seed),
        "--residual-scale",
        str(args.residual_scale),
        "--accel-clip",
        str(args.accel_clip),
        "--rollout-steps",
        str(args.rollout_steps),
        "--minibatch-size",
        str(args.minibatch_size),
        "--update-epochs",
        str(args.update_epochs),
        "--gamma",
        str(args.gamma),
        "--gae-lambda",
        str(args.gae_lambda),
        "--clip-ratio",
        str(args.clip_ratio),
        "--learning-rate",
        str(args.learning_rate),
        "--entropy-coefficient",
        str(args.entropy_coefficient),
        "--value-coefficient",
        str(args.value_coefficient),
        "--max-grad-norm",
        str(args.max_grad_norm),
    ]
    for name, option in (
        ("safe_control_gym_root", "--safe-control-gym-root"),
        ("safe_control_python", "--safe-control-python"),
        ("safe_control_config", "--safe-control-config"),
        ("budget", "--budget"),
        ("horizon", "--horizon"),
        ("max_episode_steps", "--max-episode-steps"),
        ("sensor_bias_bound", "--sensor-bias-bound"),
        ("sensor_noise_bound", "--sensor-noise-bound"),
        ("stream_memory_decay", "--stream-memory-decay"),
        ("debug_raise_after_steps", "--debug-raise-after-steps"),
    ):
        value = getattr(args, name)
        if value is not None:
            command.extend([option, str(value)])
    if args.allow_debug_pid:
        command.append("--allow-debug-pid")
    return command


def _run_worker_specs(specs: Sequence[WorkerSpec], *, jobs: int, worker_threads: int) -> list[WorkerResult]:
    workers_root = specs[0].worker_dir.parent if specs else Path()
    if specs:
        workers_root.mkdir(parents=True, exist_ok=True)
    results: list[WorkerResult] = []
    env = _worker_env(worker_threads)
    for start in range(0, len(specs), jobs):
        launched: list[tuple[WorkerSpec, subprocess.Popen[bytes], Any, Any, Path, Path]] = []
        for spec in specs[start : start + jobs]:
            stdout_tmp = workers_root / f"{spec.method}.stdout.log.tmp"
            stderr_tmp = workers_root / f"{spec.method}.stderr.log.tmp"
            stdout_handle = stdout_tmp.open("wb")
            stderr_handle = stderr_tmp.open("wb")
            process = subprocess.Popen(spec.command, stdout=stdout_handle, stderr=stderr_handle, env=env)
            launched.append((spec, process, stdout_handle, stderr_handle, stdout_tmp, stderr_tmp))
        while launched:
            completed_index = next(
                (index for index, (_, process, *_rest) in enumerate(launched) if process.poll() is not None),
                None,
            )
            if completed_index is None:
                sleep(0.1)
                continue
            spec, process, stdout_handle, stderr_handle, stdout_tmp, stderr_tmp = launched.pop(completed_index)
            returncode = int(process.wait())
            stdout_handle.close()
            stderr_handle.close()
            spec.worker_dir.mkdir(parents=True, exist_ok=True)
            stdout_log = spec.worker_dir / "stdout.log"
            stderr_log = spec.worker_dir / "stderr.log"
            if stdout_tmp.exists():
                stdout_tmp.replace(stdout_log)
            if stderr_tmp.exists():
                stderr_tmp.replace(stderr_log)
            results.append(
                WorkerResult(
                    method=spec.method,
                    returncode=returncode,
                    worker_dir=spec.worker_dir,
                    stdout_log=stdout_log,
                    stderr_log=stderr_log,
                )
            )
    return results


def _worker_env(worker_threads: int) -> dict[str, str]:
    env = os.environ.copy()
    thread_value = str(int(worker_threads))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = thread_value
    env["TORCH_NUM_THREADS"] = thread_value
    env.setdefault("MPLCONFIGDIR", "/tmp")
    src_root = Path(__file__).resolve().parents[2]
    pythonpath = env.get("PYTHONPATH")
    if pythonpath:
        paths = pythonpath.split(os.pathsep)
        if str(src_root) not in paths:
            env["PYTHONPATH"] = os.pathsep.join([str(src_root), *paths])
    else:
        env["PYTHONPATH"] = str(src_root)
    return env


def _apply_worker_thread_caps(worker_threads: int) -> None:
    thread_value = str(int(worker_threads))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, thread_value)
    try:
        import torch

        torch.set_num_threads(int(worker_threads))
    except Exception:
        pass


def _require_gymnasium() -> tuple[Any, Any]:
    if gym is None:
        raise ImportError(
            "Gymnasium is required for the SB3 PPO backend. "
            "Install the learning extra with `python -m pip install -e .[learning]`."
        )
    return gym, gym.spaces


def _require_sb3_ppo() -> Any:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            from stable_baselines3 import PPO as Sb3PPO
        except ImportError as exc:
            raise ImportError(
                "Stable-Baselines3 is required for the default PPO backend. "
                "Install the learning extra with `python -m pip install -e .[learning]` "
                "or pass `--ppo-backend custom`."
            ) from exc
    return Sb3PPO


def _sb3_loss_stats(model: Any) -> dict[str, float]:
    values = getattr(model.logger, "name_to_value", {})
    entropy_loss = values.get("train/entropy_loss", np.nan)
    return {
        "policy_loss": float(values.get("train/policy_gradient_loss", np.nan)),
        "value_loss": float(values.get("train/value_loss", np.nan)),
        "entropy": float(-entropy_loss) if np.isfinite(entropy_loss) else float(np.nan),
        "approx_kl": float(values.get("train/approx_kl", np.nan)),
    }


def _aggregate_worker_artifacts(
    out_dir: Path,
    specs: Sequence[WorkerSpec],
    results: Sequence[WorkerResult],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    result_by_method = {result.method: result for result in results}
    artifacts: dict[str, list[dict[str, Any]]] = {
        "training_curve": [],
        "raw_train_episodes": [],
        "eval_episodes": [],
        "shield_timeseries": [],
        "failure_events": [],
    }
    csv_specs = (
        ("training_curve", "training_curve.csv"),
        ("raw_train_episodes", "raw_train_episodes.csv"),
        ("eval_episodes", "eval_episodes.csv"),
        ("shield_timeseries", "shield_timeseries.csv"),
        ("failure_events", "failure_events.csv"),
    )
    worker_statuses: list[dict[str, Any]] = []
    worker_steps: list[dict[str, Any]] = []
    for spec in specs:
        result = result_by_method.get(spec.method)
        manifest_path = spec.worker_dir / "manifest.json"
        worker_manifest: dict[str, Any] = {}
        manifest_status = "missing"
        if manifest_path.exists():
            try:
                worker_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest_status = str(worker_manifest.get("status", "missing"))
            except json.JSONDecodeError:
                manifest_status = "invalid"
        for key, filename in csv_specs:
            path = spec.worker_dir / filename
            if path.exists():
                artifacts[key].extend(pd.read_csv(path).to_dict("records"))
        checkpoint_dir = spec.worker_dir / "policy_checkpoints"
        if checkpoint_dir.exists():
            for checkpoint in (*checkpoint_dir.glob("*.pt"), *checkpoint_dir.glob("*.zip")):
                shutil.copy2(checkpoint, out_dir / "policy_checkpoints" / checkpoint.name)
        for step in worker_manifest.get("steps", []):
            step = dict(step)
            if "worker_dir" not in step:
                step["worker_dir"] = str(spec.worker_dir.relative_to(out_dir))
            worker_steps.append(step)
        returncode = None if result is None else result.returncode
        success = returncode == 0 and manifest_status == "success"
        worker_statuses.append(
            {
                "method": spec.method,
                "method_index": spec.method_index,
                "worker_dir": str(spec.worker_dir.relative_to(out_dir)),
                "returncode": returncode,
                "manifest_status": manifest_status,
                "success": success,
                "stdout_log": str((spec.worker_dir / "stdout.log").relative_to(out_dir)),
                "stderr_log": str((spec.worker_dir / "stderr.log").relative_to(out_dir)),
            }
        )
    return artifacts, worker_statuses, worker_steps


def _parallel_worker_failure_event(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": "train",
        "method": status["method"],
        "method_kind": "ppo_controller",
        "seed": -1,
        "step": -1,
        "elapsed_seconds": "",
        "event_type": "parallel_worker_failed",
        "exception_type": "RuntimeError",
        "message": (
            f"worker exited with returncode={status['returncode']} "
            f"and manifest_status={status['manifest_status']}"
        ),
        "traceback": "",
        "generator_count": "",
        "candidate_reducer_names": json.dumps([]),
    }


def encode_observation(
    scenario: Any,
    snapshot: IrosEnvSnapshot,
    previous_command: np.ndarray,
    previous_shield_active: bool,
) -> np.ndarray:
    """Encode a deterministic fixed-scale controller observation."""

    pose = np.asarray(snapshot.pose, dtype=float).reshape(3)
    velocity = np.asarray(snapshot.velocity, dtype=float).reshape(3)
    gate = scenario.gate(snapshot.target_gate_index)
    rel_gate = gate.center - pose
    gate_distance = float(np.linalg.norm(rel_gate))
    gate_fraction = float(snapshot.gates_passed / max(1, len(scenario.gates)))
    if scenario.obstacles:
        clearances = [
            (
                float(np.linalg.norm(pose - obstacle.center) - obstacle.radius - scenario.collision_radius),
                obstacle.center - pose,
            )
            for obstacle in scenario.obstacles
        ]
        clearance, rel_obstacle = min(clearances, key=lambda item: item[0])
    else:
        clearance = 5.0
        rel_obstacle = np.zeros(3, dtype=float)
    streams = iros_stream_values(
        scenario,
        IrosObservation(pose, velocity, target_gate_index=snapshot.target_gate_index),
    )
    vector = np.concatenate(
        [
            pose / 5.0,
            velocity / 4.0,
            rel_gate / 5.0,
            np.asarray([gate_distance / 5.0, gate_fraction], dtype=float),
            np.asarray(rel_obstacle, dtype=float).reshape(3) / 5.0,
            np.asarray([np.clip(clearance, -5.0, 5.0) / 5.0], dtype=float),
            streams / np.asarray([5.0, 5.0, 5.0, 3.0, 3.0, 4.0, 5.0], dtype=float),
            np.asarray(previous_command, dtype=float).reshape(3) / 4.0,
            np.asarray([1.0 if previous_shield_active else 0.0], dtype=float),
        ]
    )
    return np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)


def residual_acceleration_command(
    policy_action: np.ndarray,
    planner_hint: np.ndarray,
    snapshot: IrosEnvSnapshot,
    scenario: Any,
    *,
    residual_scale: float = 2.0,
    accel_clip: float = 4.0,
) -> np.ndarray:
    """Convert a bounded residual policy action into a clipped acceleration command."""

    hint = np.asarray(planner_hint, dtype=float).reshape(-1)
    if hint.size == 3:
        planner_acc = np.clip(hint, -accel_clip, accel_clip)
    elif hint.size >= 6:
        target_position = hint[:3]
        target_velocity = hint[3:6]
        planner_acc = np.clip(
            2.0 * (target_position - snapshot.pose) + 1.2 * (target_velocity - snapshot.velocity),
            -accel_clip,
            accel_clip,
        )
    else:
        gate = scenario.gate(snapshot.target_gate_index)
        planner_acc = np.clip(2.0 * (gate.center - snapshot.pose) - 1.2 * snapshot.velocity, -accel_clip, accel_clip)
    residual = float(residual_scale) * np.clip(np.asarray(policy_action, dtype=float).reshape(3), -1.0, 1.0)
    return np.clip(planner_acc + residual, -accel_clip, accel_clip).astype(float)


def shield_acceleration(snapshot: IrosEnvSnapshot, *, accel_clip: float = 4.0) -> np.ndarray:
    """Brake horizontal velocity and climb back toward one meter altitude."""

    climb = np.asarray([0.0, 0.0, 1.5 * (1.0 - float(snapshot.pose[2]))], dtype=float)
    return np.clip(-1.8 * snapshot.velocity + climb, -accel_clip, accel_clip).astype(float)


def controller_reward(
    snapshot: IrosEnvSnapshot,
    command: np.ndarray,
    shield_active: bool,
    newly_passed_gates: int,
) -> float:
    reward = -0.02
    reward += 100.0 * float(snapshot.task_completed)
    reward += 20.0 * float(newly_passed_gates)
    reward -= 150.0 * float(snapshot.collision)
    reward -= 100.0 * float(snapshot.constraint_violation)
    reward -= 0.01 * float(np.dot(command, command))
    reward -= 0.5 * float(shield_active)
    return float(reward)


def _train_one_method(
    args: argparse.Namespace,
    profile: CorlLearningProfile,
    method: str,
    method_index: int,
    out_dir: Path,
    artifacts: dict[str, list[dict[str, Any]]],
    manifest: dict[str, Any],
) -> None:
    if args.ppo_backend == "sb3":
        _train_one_method_sb3(args, profile, method, method_index, out_dir, artifacts, manifest)
        return
    if args.ppo_backend != "custom":
        raise ValueError(f"unsupported PPO backend: {args.ppo_backend}")
    seed = int(args.seed + 1000 * method_index)
    client = make_env_client(
        profile="smoke" if args.profile == "smoke" else "overnight",
        safe_control_gym_root=args.safe_control_gym_root,
        safe_control_python=args.safe_control_python,
        safe_control_config=args.safe_control_config,
        safe_control_controller_mode=args.safe_control_controller_mode,
        allow_debug_pid=args.allow_debug_pid,
        fake_max_steps=profile.max_episode_steps,
    )
    try:
        runtime = ControllerRuntime(
            client,
            profile,
            method,
            seed,
            "train",
            residual_scale=args.residual_scale,
            accel_clip=args.accel_clip,
        )
        observation = runtime.reset(seed)
        config = PPOConfig(
            rollout_steps=args.rollout_steps,
            minibatch_size=args.minibatch_size,
            update_epochs=args.update_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_ratio=args.clip_ratio,
            learning_rate=args.learning_rate,
            entropy_coefficient=args.entropy_coefficient,
            value_coefficient=args.value_coefficient,
            max_grad_norm=args.max_grad_norm,
        )
        trainer = PPOTrainer(observation.size, 3, config, seed=seed)
        total_steps = 0
        next_eval = int(args.eval_interval)
        update = 0
        completed_since_update = 0
        episode_rows_since_update: list[dict[str, Any]] = []
        best_eval_score: tuple[float, ...] | None = None
        best_eval_checkpoint: Path | None = None
        best_eval_steps: int | None = None
        while total_steps < int(args.total_steps):
            rollout = RolloutBuffer.empty()
            rewards: list[float] = []
            shield_flags: list[bool] = []
            while len(rollout) < config.rollout_steps and total_steps < int(args.total_steps):
                action, log_prob, value = trainer.act(observation, deterministic=False)
                next_observation, reward, done, row, episode_row = runtime.step(
                    action,
                    environment_steps=total_steps + 1,
                )
                rollout.add(observation, action, log_prob, value, reward, done)
                artifacts["shield_timeseries"].append(row)
                rewards.append(reward)
                shield_flags.append(bool(row["shield_active"]))
                total_steps += 1
                observation = next_observation
                if episode_row is not None:
                    artifacts["raw_train_episodes"].append(episode_row)
                    episode_rows_since_update.append(episode_row)
                    completed_since_update += 1
                if args.debug_raise_after_steps is not None and total_steps >= args.debug_raise_after_steps:
                    raise RuntimeError("debug controller-training failure")
            last_value = 0.0 if rollout.dones and rollout.dones[-1] else trainer.value(observation)
            losses = trainer.update(rollout, last_value=last_value)
            update += 1
            curve = _training_curve_row(
                method,
                update,
                total_steps,
                completed_since_update,
                episode_rows_since_update,
                rewards,
                shield_flags,
                losses,
            )
            artifacts["training_curve"].append(curve)
            completed_since_update = 0
            episode_rows_since_update = []
            if total_steps >= next_eval or total_steps >= int(args.total_steps):
                checkpoint = out_dir / "policy_checkpoints" / f"{method}_step{total_steps}.pt"
                trainer.save(checkpoint, metadata={"method": method, "environment_steps": total_steps})
                eval_rows, eval_timeseries = _evaluate_method(args, profile, method, trainer, seed + 50_000, total_steps)
                artifacts["eval_episodes"].extend(eval_rows)
                artifacts["shield_timeseries"].extend(eval_timeseries)
                best_eval_score, best_eval_checkpoint, best_eval_steps = _maybe_update_best_eval_checkpoint(
                    out_dir,
                    method,
                    checkpoint,
                    total_steps,
                    eval_rows,
                    best_eval_score,
                    best_eval_checkpoint,
                    best_eval_steps,
                    ".pt",
                )
                next_eval += int(args.eval_interval)
        final = out_dir / "policy_checkpoints" / f"{method}_final.pt"
        trainer.save(final, metadata={"method": method, "environment_steps": total_steps, "final": True})
        if runtime.episode is not None and runtime.episode.step:
            artifacts["raw_train_episodes"].append(runtime.finish_episode(total_steps, partial=True))
        manifest["steps"].append(
            {
                "kind": "ppo_training",
                "ppo_backend": "custom",
                "method": method,
                "environment_steps": total_steps,
                "final_checkpoint": str(final.relative_to(out_dir)),
                "best_eval_checkpoint": "" if best_eval_checkpoint is None else str(best_eval_checkpoint.relative_to(out_dir)),
                "best_eval_environment_steps": best_eval_steps,
            }
        )
    finally:
        client.close()


def _train_one_method_sb3(
    args: argparse.Namespace,
    profile: CorlLearningProfile,
    method: str,
    method_index: int,
    out_dir: Path,
    artifacts: dict[str, list[dict[str, Any]]],
    manifest: dict[str, Any],
) -> None:
    sb3_ppo = _require_sb3_ppo()
    seed = int(args.seed + 1000 * method_index)
    episode_rows_since_update: list[dict[str, Any]] = []
    env = Sb3ControllerEnv(
        args,
        profile,
        method,
        seed,
        "train",
        artifacts,
        artifacts["raw_train_episodes"],
        training=True,
    )
    try:
        n_steps = max(2, min(int(args.rollout_steps), int(args.eval_interval), int(args.total_steps)))
        batch_size = min(max(2, int(args.minibatch_size)), n_steps)
        model = sb3_ppo(
            "MlpPolicy",
            env,
            learning_rate=float(args.learning_rate),
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=int(args.update_epochs),
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            clip_range=float(args.clip_ratio),
            ent_coef=float(args.entropy_coefficient),
            vf_coef=float(args.value_coefficient),
            max_grad_norm=float(args.max_grad_norm),
            seed=seed,
            device="cpu",
            verbose=0,
            policy_kwargs={"net_arch": list(PPOConfig().hidden_sizes)},
        )
        total_steps = 0
        next_eval = int(args.eval_interval)
        update = 0
        eval_checkpoints: list[int] = []
        best_eval_score: tuple[float, ...] | None = None
        best_eval_checkpoint: Path | None = None
        best_eval_steps: int | None = None
        while total_steps < int(args.total_steps):
            target = min(next_eval, int(args.total_steps))
            chunk_steps = max(1, target - total_steps)
            row_start = len(artifacts["shield_timeseries"])
            reward_start = len(env.step_rewards)
            episode_start = len(artifacts["raw_train_episodes"])
            model.learn(
                total_timesteps=chunk_steps,
                reset_num_timesteps=(total_steps == 0),
                progress_bar=False,
            )
            total_steps = int(model.num_timesteps)
            update += 1
            episode_rows_since_update = artifacts["raw_train_episodes"][episode_start:]
            rows = artifacts["shield_timeseries"][row_start:]
            rewards = env.step_rewards[reward_start:]
            shield_flags = [bool(row["shield_active"]) for row in rows]
            artifacts["training_curve"].append(
                _training_curve_row(
                    method,
                    update,
                    total_steps,
                    len(episode_rows_since_update),
                    episode_rows_since_update,
                    rewards,
                    shield_flags,
                    _sb3_loss_stats(model),
                )
            )
            if total_steps >= next_eval or total_steps >= int(args.total_steps):
                checkpoint = out_dir / "policy_checkpoints" / f"{method}_step{total_steps}.zip"
                model.save(checkpoint)
                eval_rows, eval_timeseries = _evaluate_method_sb3(
                    args,
                    profile,
                    method,
                    model,
                    seed + 50_000,
                    total_steps,
                )
                artifacts["eval_episodes"].extend(eval_rows)
                artifacts["shield_timeseries"].extend(eval_timeseries)
                eval_checkpoints.append(total_steps)
                best_eval_score, best_eval_checkpoint, best_eval_steps = _maybe_update_best_eval_checkpoint(
                    out_dir,
                    method,
                    checkpoint,
                    total_steps,
                    eval_rows,
                    best_eval_score,
                    best_eval_checkpoint,
                    best_eval_steps,
                    ".zip",
                )
                while next_eval <= total_steps:
                    next_eval += int(args.eval_interval)
        final = out_dir / "policy_checkpoints" / f"{method}_final.zip"
        model.save(final)
        if env.runtime.episode is not None and env.runtime.episode.step:
            artifacts["raw_train_episodes"].append(env.runtime.finish_episode(total_steps, partial=True))
        manifest["steps"].append(
            {
                "kind": "ppo_training",
                "ppo_backend": "sb3",
                "method": method,
                "environment_steps": total_steps,
                "requested_total_steps": int(args.total_steps),
                "actual_total_steps": total_steps,
                "rollout_steps": n_steps,
                "requested_eval_interval": int(args.eval_interval),
                "actual_eval_checkpoints": eval_checkpoints,
                "final_checkpoint": str(final.relative_to(out_dir)),
                "best_eval_checkpoint": "" if best_eval_checkpoint is None else str(best_eval_checkpoint.relative_to(out_dir)),
                "best_eval_environment_steps": best_eval_steps,
            }
        )
    finally:
        env.close()


def _evaluate_method(
    args: argparse.Namespace,
    profile: CorlLearningProfile,
    method: str,
    trainer: PPOTrainer,
    seed_start: int,
    environment_steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episode_rows: list[dict[str, Any]] = []
    timeseries_rows: list[dict[str, Any]] = []
    for offset in range(int(args.eval_seeds)):
        client = make_env_client(
            profile="smoke" if args.profile == "smoke" else "overnight",
            safe_control_gym_root=args.safe_control_gym_root,
            safe_control_python=args.safe_control_python,
            safe_control_config=args.safe_control_config,
            safe_control_controller_mode=args.safe_control_controller_mode,
            allow_debug_pid=args.allow_debug_pid,
            fake_max_steps=profile.max_episode_steps,
        )
        try:
            seed = seed_start + offset
            runtime = ControllerRuntime(
                client,
                profile,
                method,
                seed,
                "eval",
                residual_scale=args.residual_scale,
                accel_clip=args.accel_clip,
            )
            observation = runtime.reset(seed)
            done = False
            local_steps = 0
            while not done and local_steps < profile.max_episode_steps:
                action, _, _ = trainer.act(observation, deterministic=True)
                observation, _, done, row, episode_row = runtime.step(
                    action,
                    environment_steps=environment_steps,
                )
                timeseries_rows.append(row)
                local_steps += 1
                if episode_row is not None:
                    episode_row["environment_steps"] = environment_steps
                    episode_rows.append(episode_row)
            if runtime.episode is not None and runtime.episode.step:
                episode_rows.append(runtime.finish_episode(environment_steps, partial=not done))
        finally:
            client.close()
    return episode_rows, timeseries_rows


def _evaluate_method_sb3(
    args: argparse.Namespace,
    profile: CorlLearningProfile,
    method: str,
    model: Any,
    seed_start: int,
    environment_steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episode_rows: list[dict[str, Any]] = []
    timeseries_rows: list[dict[str, Any]] = []
    for offset in range(int(args.eval_seeds)):
        client = make_env_client(
            profile="smoke" if args.profile == "smoke" else "overnight",
            safe_control_gym_root=args.safe_control_gym_root,
            safe_control_python=args.safe_control_python,
            safe_control_config=args.safe_control_config,
            safe_control_controller_mode=args.safe_control_controller_mode,
            allow_debug_pid=args.allow_debug_pid,
            fake_max_steps=profile.max_episode_steps,
        )
        try:
            seed = seed_start + offset
            runtime = ControllerRuntime(
                client,
                profile,
                method,
                seed,
                "eval",
                residual_scale=args.residual_scale,
                accel_clip=args.accel_clip,
            )
            observation = runtime.reset(seed)
            done = False
            local_steps = 0
            while not done and local_steps < profile.max_episode_steps:
                action, _ = model.predict(observation, deterministic=True)
                observation, _, done, row, episode_row = runtime.step(
                    action,
                    environment_steps=environment_steps,
                )
                timeseries_rows.append(row)
                local_steps += 1
                if episode_row is not None:
                    episode_row["environment_steps"] = environment_steps
                    episode_rows.append(episode_row)
            if runtime.episode is not None and runtime.episode.step:
                episode_rows.append(runtime.finish_episode(environment_steps, partial=not done))
        finally:
            client.close()
    return episode_rows, timeseries_rows


def _maybe_update_best_eval_checkpoint(
    out_dir: Path,
    method: str,
    checkpoint: Path,
    environment_steps: int,
    eval_rows: Sequence[dict[str, Any]],
    best_score: tuple[float, ...] | None,
    best_checkpoint: Path | None,
    best_steps: int | None,
    suffix: str,
) -> tuple[tuple[float, ...] | None, Path | None, int | None]:
    score = _eval_score(eval_rows)
    if best_score is not None and score <= best_score:
        return best_score, best_checkpoint, best_steps
    best_path = out_dir / "policy_checkpoints" / f"{method}_best_eval{suffix}"
    shutil.copy2(checkpoint, best_path)
    return score, best_path, int(environment_steps)


def _eval_score(rows: Sequence[dict[str, Any]]) -> tuple[float, ...]:
    if not rows:
        return (float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    frame = pd.DataFrame(rows)
    return (
        float(frame["task_completed"].astype(float).mean()),
        float(frame["gates_passed"].astype(float).mean()),
        -float(frame["collision"].astype(float).mean()),
        -float(frame["constraint_violation"].astype(float).mean()),
        float(frame["reward"].astype(float).mean()),
    )


def _training_curve_row(
    method: str,
    update: int,
    environment_steps: int,
    completed_episodes: int,
    episode_rows: Sequence[dict[str, Any]],
    rewards: Sequence[float],
    shield_flags: Sequence[bool],
    losses: dict[str, float],
) -> dict[str, Any]:
    completed = [bool(row["task_completed"]) for row in episode_rows]
    gates = [float(row["gates_passed"]) for row in episode_rows]
    collisions = [bool(row["collision"]) for row in episode_rows]
    constraints = [bool(row["constraint_violation"]) for row in episode_rows]
    return {
        "method": method,
        "update": update,
        "environment_steps": environment_steps,
        "episodes": completed_episodes,
        "mean_episode_reward": float(np.mean([row["reward"] for row in episode_rows])) if episode_rows else float(np.sum(rewards)),
        "completion_rate": float(np.mean(completed)) if completed else 0.0,
        "mean_gates_passed": float(np.mean(gates)) if gates else 0.0,
        "collision_rate": float(np.mean(collisions)) if collisions else 0.0,
        "constraint_violation_rate": float(np.mean(constraints)) if constraints else 0.0,
        "shield_rate": float(np.mean(shield_flags)) if shield_flags else 0.0,
        "policy_loss": float(losses.get("policy_loss", np.nan)),
        "value_loss": float(losses.get("value_loss", np.nan)),
        "entropy": float(losses.get("entropy", np.nan)),
        "approx_kl": float(losses.get("approx_kl", np.nan)),
    }


def _make_shield_policy(method: str, monitor: IrosGateMonitor, profile: CorlLearningProfile) -> Any:
    if method == "ppo_unshielded":
        return None
    if method == "ppo_shield_box":
        return StaticReductionPolicy(ProtectedReducer(BoxReducer()), profile.budget)
    if method == "ppo_shield_girard":
        return StaticReductionPolicy(ProtectedReducer(GirardReducer()), profile.budget)
    if method == "ppo_shield_pzr":
        spec = MethodSpec.rollout_mpc(
            "mpc_wide_fixed_girard",
            wide_rollout_reducer_factories(),
            lambda: ProtectedReducer(GirardReducer()),
            lambda: ProtectedReducer(BoxReducer()),
        )
        return _make_policy(spec, monitor, _benchmark_config(profile))
    raise ValueError(f"unsupported shield method: {method}")


def _reduce_with_shield_policy(
    policy: Any,
    method: str,
    monitor: IrosGateMonitor,
    state: Any,
    profile: CorlLearningProfile,
    snapshot: IrosEnvSnapshot,
) -> ReductionDecision:
    if policy is None:
        raise ValueError("unshielded method has no reduction policy")
    if method == "ppo_shield_pzr":
        return policy.reduce_state(monitor, state, _predicted_observations(monitor, snapshot, _corl_profile_adapter(profile)))
    return policy.reduce_state(monitor, state)


def _benchmark_config(profile: CorlLearningProfile) -> BenchmarkConfig:
    return BenchmarkConfig(
        length=profile.max_episode_steps,
        budget=profile.budget,
        horizon=profile.horizon,
        seeds=(),
        predictor_mode="online",
        include_reference=False,
        bootstrap_samples=20,
    )


def _corl_profile_adapter(profile: CorlLearningProfile) -> Any:
    return argparse.Namespace(
        budget=profile.budget,
        horizon=profile.horizon,
        max_steps=profile.max_episode_steps,
        sensor_bias_bound=profile.sensor_bias_bound,
        sensor_noise_bound=profile.sensor_noise_bound,
        stream_memory_decay=profile.stream_memory_decay,
    )


def _write_learning_artifacts(
    out_dir: Path,
    args: argparse.Namespace,
    profile: CorlLearningProfile,
    manifest: dict[str, Any],
    artifacts: dict[str, list[dict[str, Any]]],
) -> None:
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["failure_event_count"] = len(artifacts["failure_events"])
    training = _rows_frame(artifacts["training_curve"], TRAINING_CURVE_COLUMNS)
    train_episodes = _rows_frame(artifacts["raw_train_episodes"], EPISODE_COLUMNS)
    eval_episodes = _rows_frame(artifacts["eval_episodes"], EPISODE_COLUMNS)
    shield = _rows_frame(artifacts["shield_timeseries"], SHIELD_TIMESERIES_COLUMNS)
    failures = _rows_frame(artifacts["failure_events"], FAILURE_EVENT_COLUMNS)
    training.to_csv(out_dir / "training_curve.csv", index=False)
    train_episodes.to_csv(out_dir / "raw_train_episodes.csv", index=False)
    eval_episodes.to_csv(out_dir / "eval_episodes.csv", index=False)
    shield.to_csv(out_dir / "shield_timeseries.csv", index=False)
    failures.to_csv(out_dir / "failure_events.csv", index=False)
    notes = _analysis_notes(training, train_episodes, eval_episodes, shield, failures)
    _add_execution_warnings(notes, args, manifest)
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


def _rows_frame(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
    extras = [column for column in frame.columns if column not in columns]
    return frame[list(columns) + extras]


def _analysis_notes(
    training: pd.DataFrame,
    train_episodes: pd.DataFrame,
    eval_episodes: pd.DataFrame,
    shield: pd.DataFrame,
    failures: pd.DataFrame,
) -> dict[str, Any]:
    notes: dict[str, Any] = {
        "primary_metrics": {},
        "final_metrics": {},
        "best_checkpoint_metrics": {},
        "level0_success_gate": {
            "unshielded_nontrivial_gate_progress": False,
            "shielded_variant_reduces_unsafe_training": False,
            "no_runtime_failures": failures.empty,
        },
        "warning_flags": [],
    }
    if not eval_episodes.empty:
        for method, group in eval_episodes.groupby("method", sort=True):
            notes["primary_metrics"][method] = {
                "heldout_completion_rate": float(group["task_completed"].astype(float).mean()),
                "heldout_collision_rate": float(group["collision"].astype(float).mean()),
                "mean_gates_passed": float(group["gates_passed"].astype(float).mean()),
                "shield_intervention_rate": float(group["shield_rate"].astype(float).mean()),
            }
        if float(eval_episodes["gates_passed"].astype(float).max()) <= 0.0:
            notes["warning_flags"].append("heldout_eval_zero_gate_progress")
        final_metrics, best_metrics = _checkpoint_eval_metrics(eval_episodes)
        notes["final_metrics"] = final_metrics
        notes["best_checkpoint_metrics"] = best_metrics
    if not train_episodes.empty and "ppo_unshielded" in set(train_episodes["method"]):
        unshielded = train_episodes[train_episodes["method"] == "ppo_unshielded"]
        notes["level0_success_gate"]["unshielded_nontrivial_gate_progress"] = bool(
            unshielded["gates_passed"].astype(float).max() > 0
        )
        unsafe = train_episodes.assign(
            unsafe=lambda frame: frame["collision"].astype(bool) | frame["constraint_violation"].astype(bool)
        )
        unshielded_unsafe = float(unsafe[unsafe["method"] == "ppo_unshielded"]["unsafe"].astype(float).mean())
        shielded = unsafe[unsafe["method"] != "ppo_unshielded"]
        if not shielded.empty:
            shielded_rate = shielded.groupby("method")["unsafe"].mean().min()
            notes["level0_success_gate"]["shielded_variant_reduces_unsafe_training"] = bool(
                shielded_rate <= unshielded_unsafe
            )
    if training.empty:
        notes["warning_flags"].append("training_curve_empty")
    if eval_episodes.empty:
        notes["warning_flags"].append("eval_episodes_empty")
    if shield.empty:
        notes["warning_flags"].append("shield_timeseries_empty")
    if not failures.empty:
        notes["warning_flags"].append(f"failure_event_count={failures.shape[0]}")
    return notes


def _checkpoint_eval_metrics(eval_episodes: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    final_metrics: dict[str, Any] = {}
    best_metrics: dict[str, Any] = {}
    required = {
        "method",
        "environment_steps",
        "task_completed",
        "collision",
        "constraint_violation",
        "gates_passed",
        "reward",
        "shield_rate",
    }
    if not required <= set(eval_episodes.columns):
        return final_metrics, best_metrics
    for method, method_group in eval_episodes.groupby("method", sort=True):
        checkpoint_rows: list[dict[str, Any]] = []
        for environment_steps, checkpoint_group in method_group.groupby("environment_steps", sort=True):
            checkpoint_rows.append(_eval_checkpoint_row(method, int(environment_steps), checkpoint_group))
        if not checkpoint_rows:
            continue
        final_metrics[str(method)] = max(checkpoint_rows, key=lambda row: int(row["environment_steps"]))
        best_metrics[str(method)] = max(
            checkpoint_rows,
            key=lambda row: (
                float(row["heldout_completion_rate"]),
                float(row["mean_gates_passed"]),
                -float(row["heldout_collision_rate"]),
                -float(row["heldout_constraint_violation_rate"]),
                float(row["mean_reward"]),
            ),
        )
    return final_metrics, best_metrics


def _eval_checkpoint_row(method: str, environment_steps: int, group: pd.DataFrame) -> dict[str, Any]:
    return {
        "method": method,
        "environment_steps": int(environment_steps),
        "episode_count": int(group.shape[0]),
        "heldout_completion_rate": float(group["task_completed"].astype(float).mean()),
        "heldout_collision_rate": float(group["collision"].astype(float).mean()),
        "heldout_constraint_violation_rate": float(group["constraint_violation"].astype(float).mean()),
        "mean_gates_passed": float(group["gates_passed"].astype(float).mean()),
        "max_gates_passed": int(group["gates_passed"].astype(int).max()),
        "mean_reward": float(group["reward"].astype(float).mean()),
        "shield_intervention_rate": float(group["shield_rate"].astype(float).mean()),
    }


def _add_execution_warnings(notes: dict[str, Any], args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    warnings_list = notes.setdefault("warning_flags", [])
    if args.ppo_backend != "sb3":
        return
    rounded_steps = [
        step
        for step in manifest.get("steps", [])
        if step.get("ppo_backend") == "sb3"
        and int(step.get("environment_steps", 0)) > int(step.get("requested_total_steps", args.total_steps))
    ]
    if rounded_steps and "sb3_actual_steps_exceed_requested_total_steps" not in warnings_list:
        warnings_list.append("sb3_actual_steps_exceed_requested_total_steps")


def _controller_failure_event(
    *,
    phase: str,
    method: str,
    seed: int,
    step: int,
    event_type: str,
    exc: BaseException,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "method": method,
        "method_kind": "ppo_controller",
        "seed": int(seed),
        "step": int(step),
        "elapsed_seconds": "",
        "event_type": event_type,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=8)),
        "generator_count": "",
        "candidate_reducer_names": json.dumps([]),
    }


def _violated_trigger_names(verdicts: Sequence[Verdict]) -> list[str]:
    return [verdict.trigger.name for verdict in verdicts if verdict.status == "violation"]


def _parse_method_set(value: str) -> tuple[str, ...]:
    methods: list[str] = []
    for item in value.split(","):
        key = item.strip()
        if not key:
            continue
        if key not in SHIELD_METHODS:
            raise ValueError(f"unsupported PPO shield method: {key}")
        methods.append(SHIELD_METHODS[key])
    if not methods:
        raise ValueError("--method-set must include at least one method")
    return tuple(dict.fromkeys(methods))


def _profile_from_args(args: argparse.Namespace) -> CorlLearningProfile:
    profile = PROFILES[args.profile]
    updates = {
        "budget": args.budget,
        "horizon": args.horizon,
        "max_episode_steps": args.max_episode_steps,
        "sensor_bias_bound": args.sensor_bias_bound,
        "sensor_noise_bound": args.sensor_noise_bound,
        "stream_memory_decay": args.stream_memory_decay,
    }
    return replace(profile, **{key: value for key, value in updates.items() if value is not None})


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pzr-train-corl-controller",
        description="Train PPO residual controllers with monitor action shielding.",
    )
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument("--safe-control-gym-root", type=str, default=None)
    parser.add_argument("--safe-control-python", type=str, default=None)
    parser.add_argument("--safe-control-config", type=str, default=None)
    parser.add_argument("--safe-control-controller-mode", choices=("firmware", "debug_pid"), default="firmware")
    parser.add_argument("--allow-debug-pid", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--method-set", type=str, default="unshielded,shield_box,shield_girard,shield_pzr")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument("--ppo-backend", choices=PPO_BACKENDS, default="sb3")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--eval-seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--sensor-bias-bound", type=float, default=None)
    parser.add_argument("--sensor-noise-bound", type=float, default=None)
    parser.add_argument("--stream-memory-decay", type=float, default=None)
    parser.add_argument("--residual-scale", type=float, default=2.0)
    parser.add_argument("--accel-clip", type=float, default=4.0)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--method-index-offset", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--skip-preflight", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--debug-raise-after-steps", type=int, default=None, help=argparse.SUPPRESS)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
