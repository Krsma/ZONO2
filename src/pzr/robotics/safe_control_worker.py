"""JSON-lines sidecar worker for the safe-control-gym beta IROS task."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
import types
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p
import yaml


class IrosSidecarSession:
    """Owns one headless safe-control-gym quadrotor environment."""

    def __init__(self, root: Path, scenario_config: Path | None) -> None:
        self.root = root
        self.scenario_config = scenario_config or root / "competition" / "level3.yaml"
        if not self.scenario_config.is_absolute():
            self.scenario_config = root / self.scenario_config
        self.env: Any | None = None
        self.obs: np.ndarray | None = None
        self.info: dict[str, Any] = {}
        self.reward = 0.0
        self.done = False
        self.time = 0.0
        self.ctrl_dt = 1.0 / 60.0
        self._pid: Any | None = None
        self._thrusts: Any | None = None

    def reset(self, seed: int) -> dict[str, Any]:
        self.close()
        _prepare_imports(self.root)
        from competition_utils import PIDController, thrusts
        from safe_control_gym.utils.registration import make

        config = _load_yaml(self.scenario_config)
        quad_config = dict(config.get("quadrotor_config", {}))
        quad_config.update(
            {
                "seed": int(seed),
                "gui": False,
                "ctrl_freq": 60,
                "pyb_freq": 240,
                "info_in_reset": True,
            }
        )
        self.env = make("quadrotor", **quad_config)
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, info = reset_result
        else:
            obs, info = reset_result, self.env._get_reset_info()
        self.obs = np.asarray(obs, dtype=float)
        self.info = dict(info)
        self.reward = 0.0
        self.done = False
        self.time = 0.0
        self.ctrl_dt = float(self.info.get("ctrl_timestep", 1.0 / quad_config["ctrl_freq"]))
        self._pid = PIDController(kf=float(getattr(self.env, "KF", self.info.get("quadrotor_kf", 3.16e-10))))
        self._thrusts = thrusts
        return self._payload()

    def step(self, action: list[float] | tuple[float, ...]) -> dict[str, Any]:
        if self.env is None or self.obs is None or self._pid is None or self._thrusts is None:
            raise RuntimeError("environment has not been reset")
        command = np.asarray(action, dtype=float).reshape(3)
        command = np.clip(command, -4.0, 4.0)
        pose = _pose_from_obs(self.obs)
        velocity = _velocity_from_obs(self.obs)
        target_velocity = np.clip(velocity + command * self.ctrl_dt, -2.5, 2.5)
        target_position = pose + target_velocity * self.ctrl_dt
        motor_action = self._thrusts(
            self._pid,
            self.ctrl_dt,
            float(getattr(self.env, "KF")),
            self.obs,
            target_position,
            target_velocity,
        )
        obs, reward, done, info = self.env.step(motor_action)
        self.obs = np.asarray(obs, dtype=float)
        self.reward = float(reward)
        self.done = bool(done)
        self.info = dict(info)
        self.time += self.ctrl_dt
        return self._payload()

    def close(self) -> None:
        if self.env is not None:
            close = getattr(self.env, "close", None)
            if callable(close):
                close()
        self.env = None
        self.obs = None
        self.info = {}

    def _payload(self) -> dict[str, Any]:
        if self.env is None or self.obs is None:
            raise RuntimeError("environment has not been reset")
        return {
            "scenario": _scenario_payload(self.env, self.info),
            "snapshot": _snapshot_payload(self.env, self.obs, self.info, self.done, self.time),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-control-gym-root", required=True)
    parser.add_argument("--scenario-config", default=None)
    args = parser.parse_args()
    session = IrosSidecarSession(
        Path(args.safe_control_gym_root).expanduser().resolve(),
        None if args.scenario_config is None else Path(args.scenario_config).expanduser(),
    )
    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request.get("command")
            if command == "reset":
                _write({"ok": True, **session.reset(int(request.get("seed", 0)))})
            elif command == "step":
                _write({"ok": True, **session.step(request.get("action", [0.0, 0.0, 0.0]))})
            elif command == "close":
                session.close()
                _write({"ok": True})
                return 0
            else:
                _write({"ok": False, "error": f"unknown sidecar command: {command}"})
        except Exception as exc:
            _write({"ok": False, "error": f"{exc}\n{traceback.format_exc()}"})
    session.close()
    return 0


def _prepare_imports(root: Path) -> None:
    script_dir = str(Path(__file__).resolve().parent)
    sys.path = [path for path in sys.path if path != script_dir]
    for path in (root, root / "competition"):
        text = str(path)
        sys.path = [entry for entry in sys.path if entry != text]
        sys.path.insert(0, text)
    try:
        import torch  # noqa: F401
    except ImportError:
        torch_stub = types.ModuleType("torch")

        class Tensor:
            pass

        def manual_seed(seed: int) -> None:
            _ = seed

        torch_stub.Tensor = Tensor
        torch_stub.manual_seed = manual_seed
        sys.modules["torch"] = torch_stub


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"scenario config must be a YAML mapping: {path}")
    return data


def _pose_from_obs(obs: np.ndarray) -> np.ndarray:
    return np.asarray([obs[0], obs[2], obs[4]], dtype=float)


def _velocity_from_obs(obs: np.ndarray) -> np.ndarray:
    return np.asarray([obs[1], obs[3], obs[5]], dtype=float)


def _snapshot_payload(
    env: Any,
    obs: np.ndarray,
    info: dict[str, Any],
    done: bool,
    time: float,
) -> dict[str, Any]:
    target_gate = int(info.get("current_target_gate_id", getattr(env, "current_gate", 0)))
    num_gates = int(getattr(env, "NUM_GATES", max(1, target_gate + 1)))
    gates_passed = num_gates if target_gate < 0 else max(0, min(target_gate, num_gates))
    collision = info.get("collision", (None, False))
    if isinstance(collision, (list, tuple)) and len(collision) >= 2:
        collision_flag = bool(collision[1])
    else:
        collision_flag = bool(collision)
    return {
        "pose": _pose_from_obs(obs).tolist(),
        "velocity": _velocity_from_obs(obs).tolist(),
        "target_gate_index": 0 if target_gate < 0 else min(target_gate, num_gates - 1),
        "gates_passed": gates_passed,
        "collision": collision_flag,
        "constraint_violation": bool(info.get("constraint_violation", False)),
        "task_completed": bool(info.get("task_completed", False)),
        "done": bool(done),
        "time": float(time),
        "info": _json_safe(info),
    }


def _scenario_payload(env: Any, info: dict[str, Any]) -> dict[str, Any]:
    gates = []
    gate_dims = info.get("gate_dimensions", {})
    effective_gates = list(getattr(env, "EFFECTIVE_GATES_POSITIONS", []))
    nominal_gates = list(getattr(env, "GATES", []))
    for index, gate in enumerate(effective_gates):
        gate_type = int(nominal_gates[index][6]) if index < len(nominal_gates) and len(nominal_gates[index]) > 6 else 0
        dim_key = "low" if gate_type == 1 else "tall"
        dims = gate_dims.get(dim_key, {})
        gates.append(
            {
                "center": [float(gate[0]), float(gate[1]), float(gate[2])],
                "width": float(dims.get("edge", 0.45)),
                "height": float(dims.get("height", 1.0)),
                "type": gate_type,
            }
        )
    obstacles = []
    obstacle_radius = float(info.get("obstacle_dimensions", {}).get("radius", 0.05))
    obstacle_ids = list(getattr(env, "OBSTACLES_IDS", []))
    for obs_id in obstacle_ids:
        position, _ = p.getBasePositionAndOrientation(obs_id, physicsClientId=getattr(env, "PYB_CLIENT"))
        obstacles.append({"center": [float(position[0]), float(position[1]), float(position[2])], "radius": obstacle_radius})
    if not obstacles:
        for obstacle in getattr(env, "OBSTACLES", []):
            obstacles.append(
                {
                    "center": [float(obstacle[0]), float(obstacle[1]), float(obstacle[2])],
                    "radius": obstacle_radius,
                }
            )
    return {
        "gates": gates,
        "obstacles": obstacles,
        "corridor_radius": 1.0,
        "min_obstacle_clearance": obstacle_radius,
        "collision_radius": float(getattr(env, "COLLISION_R", 0.0)),
        "altitude_min": -0.1,
        "altitude_max": 2.0,
        "speed_max": 3.0,
        "gate_pass_radius": 0.25,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
