#!/usr/bin/env python3
"""Collect a safe-control-gym quadrotor trace as JSONL.

This script is meant to run inside the safe-control-gym sidecar environment.
It keeps simulator-specific imports outside the main PZR package.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _load_level_config(path: Path) -> dict[str, Any]:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def _apply_seeded_stress_config(config: dict[str, Any], seed: int) -> dict[str, Any]:
    """Create a seed-varying live scenario without editing sidecar YAML files."""
    rng = np.random.default_rng(seed)
    config = json.loads(json.dumps(config))
    quad_config = dict(config["quadrotor_config"])

    init = dict(quad_config.get("init_state", {}))
    init["init_x"] = float(init.get("init_x", -0.9) + rng.uniform(-0.12, 0.12))
    init["init_y"] = float(init.get("init_y", -2.9) + rng.uniform(-0.12, 0.12))
    init["init_z"] = float(max(0.025, init.get("init_z", 0.03) + rng.uniform(0.0, 0.04)))
    init["init_psi"] = float(init.get("init_psi", 0.0) + rng.uniform(-0.10, 0.10))
    quad_config["init_state"] = init

    gates = np.asarray(quad_config.get("gates", []), dtype=float)
    if gates.size:
        gates = gates.copy()
        gates[:, 0] += rng.normal(0.0, 0.10, size=gates.shape[0])
        gates[:, 1] += rng.normal(0.0, 0.10, size=gates.shape[0])
        if gates.shape[1] > 5:
            gates[:, 5] += rng.normal(0.0, 0.08, size=gates.shape[0])
        quad_config["gates"] = gates.tolist()

    obstacles = np.asarray(quad_config.get("obstacles", []), dtype=float)
    if obstacles.size:
        obstacles = obstacles.copy()
        obstacles[:, 0] += rng.normal(0.0, 0.12, size=obstacles.shape[0])
        obstacles[:, 1] += rng.normal(0.0, 0.12, size=obstacles.shape[0])
        quad_config["obstacles"] = obstacles.tolist()

    goal = np.asarray(quad_config.get("task_info", {}).get("stabilization_goal", []), dtype=float)
    if goal.size >= 3:
        goal = goal.copy()
        goal[:2] += rng.normal(0.0, 0.08, size=2)
        goal[2] = float(np.clip(goal[2] + rng.normal(0.0, 0.04), 0.45, 1.35))
        task_info = dict(quad_config.get("task_info", {}))
        task_info["stabilization_goal"] = goal[:3].tolist()
        quad_config["task_info"] = task_info

    quad_config["seed"] = int(seed)
    quad_config["reseed_on_reset"] = True
    quad_config["pzr_seeded_stress"] = True
    config["quadrotor_config"] = quad_config
    return config


def _make_controller_obs(obs: np.ndarray) -> list[float]:
    return [
        float(obs[0]), 0.0,
        float(obs[2]), 0.0,
        float(obs[4]), 0.0,
        float(obs[6]), float(obs[7]), float(obs[8]),
        0.0, 0.0, 0.0,
    ]


def collect_trace(
    *,
    safe_control_root: Path,
    config_path: Path,
    output: Path,
    length: int,
    seed: int,
    controller_mode: str,
    stress_randomize: bool,
) -> None:
    sys.path.insert(0, str(safe_control_root))
    sys.path.insert(0, str(safe_control_root / "competition"))

    from safe_control_gym.utils.registration import make
    from competition_utils import Command, thrusts
    from edit_this import Controller

    config = _load_level_config(config_path)
    if stress_randomize:
        config = _apply_seeded_stress_config(config, seed)
    config["use_firmware"] = controller_mode == "firmware"
    config["verbose"] = False
    quad_config = dict(config["quadrotor_config"])
    quad_config["gui"] = False
    quad_config["seed"] = int(seed)
    quad_config["reseed_on_reset"] = True
    if not config["use_firmware"]:
        quad_config["ctrl_freq"] = 60
        quad_config["pyb_freq"] = 240

    ctrl_freq = int(quad_config["ctrl_freq"])
    ctrl_dt = 1.0 / ctrl_freq

    firmware_wrapper = None
    if config["use_firmware"]:
        from functools import partial

        firmware_freq = 500
        quad_config["ctrl_freq"] = firmware_freq
        env_func = partial(make, "quadrotor", **quad_config)
        firmware_wrapper = make("firmware", env_func, firmware_freq, ctrl_freq)
        obs, info = firmware_wrapper.reset()
        info["ctrl_timestep"] = ctrl_dt
        info["ctrl_freq"] = ctrl_freq
        env = firmware_wrapper.env
    else:
        env = make("quadrotor", **quad_config)
        obs, info = env.reset()

    info["ctrl_timestep"] = ctrl_dt
    info["ctrl_freq"] = ctrl_freq
    ctrl_obs = _make_controller_obs(np.asarray(obs, dtype=float)) if config["use_firmware"] else obs
    controller = Controller(ctrl_obs, info, config["use_firmware"], verbose=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    reward = 0.0
    done = False
    action = np.zeros(4, dtype=float)
    records = []

    try:
        for i in range(length):
            curr_time = i * ctrl_dt
            if config["use_firmware"]:
                vicon_obs = _make_controller_obs(np.asarray(obs, dtype=float))
                command_type, args = controller.cmdFirmware(
                    curr_time, vicon_obs, reward, done, info,
                )
                if command_type == Command.FULLSTATE:
                    firmware_wrapper.sendFullStateCmd(*args, curr_time)
                elif command_type == Command.TAKEOFF:
                    firmware_wrapper.sendTakeoffCmd(*args)
                elif command_type == Command.LAND:
                    firmware_wrapper.sendLandCmd(*args)
                elif command_type == Command.STOP:
                    firmware_wrapper.sendStopCmd()
                elif command_type == Command.GOTO:
                    firmware_wrapper.sendGotoCmd(*args)
                elif command_type == Command.NOTIFYSETPOINTSTOP:
                    firmware_wrapper.notifySetpointStop()
                elif command_type == Command.FINISHED:
                    done = True
                obs, reward, done, info, action = firmware_wrapper.step(curr_time, action)
            else:
                target_pos, target_vel = controller.cmdSimOnly(
                    curr_time, obs, reward, done, info,
                )
                action = thrusts(
                    controller.ctrl,
                    controller.CTRL_TIMESTEP,
                    controller.KF,
                    obs,
                    target_pos,
                    target_vel,
                )
                obs, reward, done, info = env.step(action)

            controller.interStepLearn(action, obs, reward, done, info)
            records.append({
                "step": i,
                "time": curr_time,
                "obs": _json_safe(obs),
                "action": _json_safe(action),
                "reward": _json_safe(reward),
                "done": bool(done),
                "info": _json_safe(info),
                "controller_mode": controller_mode,
                "gates": _json_safe(quad_config.get("gates", [])),
                "obstacles": _json_safe(quad_config.get("obstacles", [])),
                "constraints": _json_safe(quad_config.get("constraints", [])),
                "pzr_seeded_stress": bool(stress_randomize),
            })
            if done:
                break
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    with open(output, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-control-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--controller-mode", choices=("sim", "firmware"), default="sim")
    parser.add_argument("--stress-randomize", action="store_true")
    args = parser.parse_args()
    collect_trace(
        safe_control_root=args.safe_control_root,
        config_path=args.config,
        output=args.output,
        length=args.length,
        seed=args.seed,
        controller_mode=args.controller_mode,
        stress_randomize=args.stress_randomize,
    )


if __name__ == "__main__":
    main()
