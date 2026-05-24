import json

import numpy as np
import pandas as pd
import pytest

from pzr.experiments.corl_learning import (
    ControllerRuntime,
    CorlLearningProfile,
    _analysis_notes,
    _make_parser,
    _parse_method_set,
    _worker_env,
    _worker_specs,
    encode_observation,
    main as corl_learning_main,
    residual_acceleration_command,
)
from pzr.learning.ppo import PPOConfig, PPOTrainer, RolloutBuffer
from pzr.robotics.safe_control_gym import FakeIrosEnvClient, IrosEnvSnapshot, make_env_client


def _profile() -> CorlLearningProfile:
    return CorlLearningProfile(
        level="smoke",
        budget=4,
        horizon=2,
        max_episode_steps=8,
        sensor_bias_bound=0.0,
        sensor_noise_bound=0.0,
        stream_memory_decay=0.0,
    )


def test_corl_controller_observation_is_stable_and_finite() -> None:
    client = FakeIrosEnvClient(max_steps=5)
    snapshot = client.reset(3)

    first = encode_observation(client.scenario, snapshot, np.zeros(3), False)
    second = encode_observation(client.scenario, snapshot, np.zeros(3), False)

    assert first.shape == second.shape
    assert first.shape == (26,)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first, second)


def test_residual_acceleration_command_clips_3d_and_6d_hints() -> None:
    client = FakeIrosEnvClient(max_steps=5)
    snapshot = client.reset(0)

    command_3d = residual_acceleration_command(
        np.asarray([1.0, -1.0, 0.5]),
        np.asarray([10.0, -10.0, 0.0]),
        snapshot,
        client.scenario,
        residual_scale=2.0,
        accel_clip=4.0,
    )
    command_6d = residual_acceleration_command(
        np.asarray([0.0, 0.0, 0.0]),
        np.concatenate([snapshot.pose + np.asarray([10.0, 0.0, 0.0]), snapshot.velocity]),
        snapshot,
        client.scenario,
        residual_scale=2.0,
        accel_clip=4.0,
    )

    np.testing.assert_allclose(command_3d, [4.0, -4.0, 1.0])
    np.testing.assert_allclose(command_6d, [4.0, 0.0, 0.0])


def test_fake_smoke_client_honors_configured_max_steps() -> None:
    client = make_env_client(
        profile="smoke",
        safe_control_gym_root=None,
        safe_control_python=None,
        fake_max_steps=60,
    )
    try:
        snapshot = client.reset(0)
        for _ in range(60):
            snapshot = client.step(client.nominal_command(snapshot))
            if snapshot.done:
                break
    finally:
        client.close()

    assert snapshot.gates_passed == 3
    assert snapshot.task_completed


def test_ppo_update_runs_and_writes_checkpoint(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    trainer = PPOTrainer(
        observation_dim=4,
        action_dim=3,
        config=PPOConfig(rollout_steps=4, minibatch_size=2, update_epochs=1),
        seed=1,
    )
    rollout = RolloutBuffer.empty()
    for index in range(4):
        obs = np.full(4, index / 10.0, dtype=np.float32)
        action, log_prob, value = trainer.act(obs)
        rollout.add(obs, action, log_prob, value, reward=1.0, done=index == 3)

    losses = trainer.update(rollout, last_value=0.0)
    checkpoint = tmp_path / "policy.pt"
    trainer.save(checkpoint)

    assert checkpoint.stat().st_size > 0
    assert np.isfinite(losses["policy_loss"])
    assert np.isfinite(losses["value_loss"])
    assert np.isfinite(losses["approx_kl"])
    assert losses["approx_kl"] >= 0.0


def test_custom_ppo_stored_actions_match_stored_log_probabilities() -> None:
    torch = pytest.importorskip("torch")
    trainer = PPOTrainer(
        observation_dim=4,
        action_dim=3,
        config=PPOConfig(),
        seed=7,
    )
    obs = np.asarray([2.0, -1.0, 0.5, -0.25], dtype=np.float32)
    actions = []
    old_log_probs = []
    for _ in range(256):
        action, log_prob, _ = trainer.act(obs)
        actions.append(action)
        old_log_probs.append(log_prob)

    observations = torch.as_tensor(np.repeat(obs[None, :], len(actions), axis=0), dtype=torch.float32)
    stored_actions = torch.as_tensor(np.asarray(actions), dtype=torch.float32)
    with torch.no_grad():
        new_log_probs = trainer.model.distribution(observations).log_prob(stored_actions).sum(dim=-1).numpy()
    ratios = np.exp(new_log_probs - np.asarray(old_log_probs))

    assert np.max(np.abs(ratios - 1.0)) < 1e-5
    assert np.max(np.abs(actions)) > 1.0


def test_controller_runtime_auto_reset_uses_linear_episode_seeds() -> None:
    pytest.importorskip("torch")
    client = FakeIrosEnvClient(max_steps=1)
    runtime = ControllerRuntime(
        client,
        _profile(),
        "ppo_unshielded",
        seed=10,
        phase="train",
        residual_scale=2.0,
        accel_clip=4.0,
    )
    try:
        observation = runtime.reset(10)
        seeds = []
        for index in range(4):
            seeds.append(runtime.seed)
            observation, _, done, _, _ = runtime.step(np.zeros(3), environment_steps=index + 1)
            assert done
            assert observation.shape == (26,)
    finally:
        client.close()

    assert seeds == [10, 11, 12, 13]


def test_shield_overrides_unsafe_candidate_action() -> None:
    pytest.importorskip("torch")
    client = FakeIrosEnvClient(max_steps=5)
    runtime = ControllerRuntime(
        client,
        _profile(),
        "ppo_shield_box",
        seed=0,
        phase="train",
        residual_scale=2.0,
        accel_clip=4.0,
    )
    runtime.reset(0)
    client._velocity = np.asarray([5.0, 0.0, 0.0])
    runtime.snapshot = IrosEnvSnapshot(
        client._pose,
        client._velocity,
        target_gate_index=client._target_gate,
        gates_passed=client._gates_passed,
        time=client._step * client.dt,
    )

    _, _, _, row, _ = runtime.step(np.ones(3), environment_steps=1)

    assert row["shield_active"]
    assert row["command_source"] == "shield"
    assert row["applied_ax"] != row["candidate_ax"]


def test_corl_learning_smoke_writes_artifacts(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("gymnasium")
    out = tmp_path / "corl-ppo"

    exit_code = corl_learning_main(
        [
            "--profile",
            "smoke",
            "--method-set",
            "unshielded,shield_box",
            "--total-steps",
            "16",
            "--eval-interval",
            "8",
            "--eval-seeds",
            "1",
            "--rollout-steps",
            "8",
            "--minibatch-size",
            "4",
            "--update-epochs",
            "1",
            "--max-episode-steps",
            "8",
            "--out",
            str(out),
            "--force",
        ]
    )

    assert exit_code == 0
    for name in (
        "training_curve.csv",
        "raw_train_episodes.csv",
        "eval_episodes.csv",
        "shield_timeseries.csv",
        "failure_events.csv",
        "config.json",
        "manifest.json",
        "analysis_notes.json",
        "artifact_index.csv",
    ):
        assert (out / name).stat().st_size > 0
    training = pd.read_csv(out / "training_curve.csv")
    eval_episodes = pd.read_csv(out / "eval_episodes.csv")
    assert {"ppo_unshielded", "ppo_shield_box"} <= set(training["method"])
    assert {"ppo_unshielded", "ppo_shield_box"} <= set(eval_episodes["method"])
    assert list((out / "policy_checkpoints").glob("*_final.zip"))
    assert list((out / "policy_checkpoints").glob("*_best_eval.zip"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert manifest["ppo_backend"] == "sb3"
    assert manifest["steps"][0]["best_eval_checkpoint"]
    notes = json.loads((out / "analysis_notes.json").read_text(encoding="utf-8"))
    assert "final_metrics" in notes
    assert "best_checkpoint_metrics" in notes


def test_corl_learning_custom_backend_smoke_writes_pt_checkpoint(tmp_path) -> None:
    pytest.importorskip("torch")
    out = tmp_path / "corl-ppo-custom"

    exit_code = corl_learning_main(
        [
            "--profile",
            "smoke",
            "--method-set",
            "unshielded",
            "--ppo-backend",
            "custom",
            "--total-steps",
            "16",
            "--eval-interval",
            "8",
            "--eval-seeds",
            "1",
            "--rollout-steps",
            "8",
            "--minibatch-size",
            "4",
            "--update-epochs",
            "1",
            "--max-episode-steps",
            "8",
            "--out",
            str(out),
            "--force",
        ]
    )

    assert exit_code == 0
    assert list((out / "policy_checkpoints").glob("ppo_unshielded_final.pt"))
    assert list((out / "policy_checkpoints").glob("ppo_unshielded_best_eval.pt"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["ppo_backend"] == "custom"
    assert manifest["steps"][0]["best_eval_checkpoint"].endswith("_best_eval.pt")


def test_corl_learning_max_episode_steps_reaches_fake_smoke_client(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("gymnasium")
    out = tmp_path / "corl-ppo-smoke60"

    corl_learning_main(
        [
            "--profile",
            "smoke",
            "--method-set",
            "unshielded",
            "--total-steps",
            "64",
            "--eval-interval",
            "64",
            "--eval-seeds",
            "1",
            "--rollout-steps",
            "16",
            "--minibatch-size",
            "8",
            "--update-epochs",
            "1",
            "--max-episode-steps",
            "60",
            "--out",
            str(out),
            "--force",
        ]
    )

    raw = pd.read_csv(out / "raw_train_episodes.csv")
    assert int(raw["episode_steps"].max()) > 40


def test_corl_learning_sb3_records_step_rounding_warning(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("gymnasium")
    out = tmp_path / "corl-ppo-rounded"

    corl_learning_main(
        [
            "--profile",
            "smoke",
            "--method-set",
            "unshielded",
            "--total-steps",
            "20",
            "--eval-interval",
            "20",
            "--eval-seeds",
            "1",
            "--rollout-steps",
            "8",
            "--minibatch-size",
            "4",
            "--update-epochs",
            "1",
            "--max-episode-steps",
            "8",
            "--out",
            str(out),
            "--force",
        ]
    )

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    notes = json.loads((out / "analysis_notes.json").read_text(encoding="utf-8"))
    step = manifest["steps"][0]
    assert manifest["ppo"]["requested_total_steps"] == 20
    assert step["requested_total_steps"] == 20
    assert step["rollout_steps"] == 8
    assert step["environment_steps"] > 20
    assert step["actual_eval_checkpoints"] == [step["environment_steps"]]
    assert "sb3_actual_steps_exceed_requested_total_steps" in notes["warning_flags"]


def test_corl_learning_parallel_smoke_writes_aggregate_artifacts(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("gymnasium")
    out = tmp_path / "corl-ppo-parallel"

    exit_code = corl_learning_main(
        [
            "--profile",
            "smoke",
            "--method-set",
            "unshielded,shield_box",
            "--jobs",
            "2",
            "--worker-threads",
            "1",
            "--total-steps",
            "16",
            "--eval-interval",
            "8",
            "--eval-seeds",
            "1",
            "--rollout-steps",
            "8",
            "--minibatch-size",
            "4",
            "--update-epochs",
            "1",
            "--max-episode-steps",
            "8",
            "--out",
            str(out),
            "--force",
        ]
    )

    assert exit_code == 0
    training = pd.read_csv(out / "training_curve.csv")
    eval_episodes = pd.read_csv(out / "eval_episodes.csv")
    assert {"ppo_unshielded", "ppo_shield_box"} <= set(training["method"])
    assert {"ppo_unshielded", "ppo_shield_box"} <= set(eval_episodes["method"])
    assert list((out / "policy_checkpoints").glob("ppo_unshielded_final.zip"))
    assert list((out / "policy_checkpoints").glob("ppo_shield_box_final.zip"))
    assert list((out / "policy_checkpoints").glob("ppo_unshielded_best_eval.zip"))
    assert list((out / "policy_checkpoints").glob("ppo_shield_box_best_eval.zip"))
    for method in ("ppo_unshielded", "ppo_shield_box"):
        worker_dir = out / "workers" / method
        assert worker_dir.is_dir()
        assert (worker_dir / "stdout.log").exists()
        assert (worker_dir / "stderr.log").exists()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert manifest["parallel"]["jobs"] == 2
    assert manifest["parallel"]["worker_threads"] == 1
    assert manifest["parallel"]["worker_count"] == 2
    assert all(status["success"] for status in manifest["parallel"]["worker_statuses"])
    assert manifest["ppo_backend"] == "sb3"


def test_corl_learning_failure_writes_partial_artifacts(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("gymnasium")
    out = tmp_path / "corl-ppo-fail"

    with pytest.raises(RuntimeError, match="debug controller-training failure"):
        corl_learning_main(
            [
                "--profile",
                "smoke",
                "--method-set",
                "shield_box",
                "--total-steps",
                "16",
                "--eval-interval",
                "8",
                "--eval-seeds",
                "1",
                "--rollout-steps",
                "8",
                "--minibatch-size",
                "4",
                "--update-epochs",
                "1",
                "--max-episode-steps",
                "8",
                "--debug-raise-after-steps",
                "3",
                "--out",
                str(out),
                "--force",
            ]
        )

    failures = pd.read_csv(out / "failure_events.csv")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    shield = pd.read_csv(out / "shield_timeseries.csv")
    assert not failures.empty
    assert manifest["status"] == "failed"
    assert not shield.empty


def test_corl_learning_parallel_failure_writes_partial_artifacts(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("gymnasium")
    out = tmp_path / "corl-ppo-parallel-fail"

    with pytest.raises(RuntimeError, match="parallel CoRL controller workers failed"):
        corl_learning_main(
            [
                "--profile",
                "smoke",
                "--method-set",
                "unshielded,shield_box",
                "--jobs",
                "2",
                "--worker-threads",
                "1",
                "--total-steps",
                "16",
                "--eval-interval",
                "8",
                "--eval-seeds",
                "1",
                "--rollout-steps",
                "8",
                "--minibatch-size",
                "4",
                "--update-epochs",
                "1",
                "--max-episode-steps",
                "8",
                "--debug-raise-after-steps",
                "3",
                "--out",
                str(out),
                "--force",
            ]
        )

    failures = pd.read_csv(out / "failure_events.csv")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    shield = pd.read_csv(out / "shield_timeseries.csv")
    assert not failures.empty
    assert "parallel_worker_failed" in set(failures["event_type"])
    assert manifest["status"] == "failed"
    assert all(not status["success"] for status in manifest["parallel"]["worker_statuses"])
    assert not shield.empty
    assert (out / "artifact_index.csv").stat().st_size > 0
    for method in ("ppo_unshielded", "ppo_shield_box"):
        assert (out / "workers" / method / "stdout.log").exists()
        assert (out / "workers" / method / "stderr.log").exists()


def test_parallel_worker_specs_preserve_canonical_methods_and_seed_offsets(tmp_path) -> None:
    args = _make_parser().parse_args(
        [
            "--method-set",
            "unshielded,shield_box",
            "--jobs",
            "2",
            "--worker-threads",
            "3",
            "--total-steps",
            "32",
            "--ppo-backend",
            "custom",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    methods = _parse_method_set(args.method_set)
    specs = _worker_specs(args, methods, tmp_path / "out")

    assert [spec.method for spec in specs] == ["ppo_unshielded", "ppo_shield_box"]
    assert specs[0].worker_dir.name == "ppo_unshielded"
    assert specs[1].worker_dir.name == "ppo_shield_box"
    first = list(specs[0].command)
    second = list(specs[1].command)
    assert first[first.index("--method-set") + 1] == "ppo_unshielded"
    assert second[second.index("--method-set") + 1] == "ppo_shield_box"
    assert first[first.index("--method-index-offset") + 1] == "0"
    assert second[second.index("--method-index-offset") + 1] == "1"
    assert first[first.index("--jobs") + 1] == "1"
    assert first[first.index("--worker-threads") + 1] == "3"
    assert first[first.index("--ppo-backend") + 1] == "custom"


def test_parallel_worker_env_caps_cpu_threads() -> None:
    env = _worker_env(2)

    assert env["OMP_NUM_THREADS"] == "2"
    assert env["MKL_NUM_THREADS"] == "2"
    assert env["OPENBLAS_NUM_THREADS"] == "2"
    assert env["NUMEXPR_NUM_THREADS"] == "2"


def test_analysis_notes_warn_when_heldout_eval_has_zero_gate_progress() -> None:
    eval_episodes = pd.DataFrame(
        [
            {
                "method": "ppo_unshielded",
                "task_completed": False,
                "collision": False,
                "gates_passed": 0,
                "shield_rate": 0.0,
            }
        ]
    )

    notes = _analysis_notes(
        pd.DataFrame([{"method": "ppo_unshielded"}]),
        pd.DataFrame(),
        eval_episodes,
        pd.DataFrame([{"method": "ppo_unshielded"}]),
        pd.DataFrame(),
    )

    assert "heldout_eval_zero_gate_progress" in notes["warning_flags"]


def test_analysis_notes_report_final_and_best_checkpoint_metrics() -> None:
    eval_episodes = pd.DataFrame(
        [
            {
                "method": "ppo_unshielded",
                "environment_steps": 10,
                "task_completed": False,
                "collision": False,
                "constraint_violation": False,
                "gates_passed": 2,
                "reward": 20.0,
                "shield_rate": 0.0,
            },
            {
                "method": "ppo_unshielded",
                "environment_steps": 20,
                "task_completed": False,
                "collision": False,
                "constraint_violation": False,
                "gates_passed": 1,
                "reward": 10.0,
                "shield_rate": 0.0,
            },
        ]
    )

    notes = _analysis_notes(
        pd.DataFrame([{"method": "ppo_unshielded"}]),
        pd.DataFrame(),
        eval_episodes,
        pd.DataFrame([{"method": "ppo_unshielded"}]),
        pd.DataFrame(),
    )

    assert notes["final_metrics"]["ppo_unshielded"]["environment_steps"] == 20
    assert notes["final_metrics"]["ppo_unshielded"]["mean_gates_passed"] == 1.0
    assert notes["best_checkpoint_metrics"]["ppo_unshielded"]["environment_steps"] == 10
    assert notes["best_checkpoint_metrics"]["ppo_unshielded"]["mean_gates_passed"] == 2.0
