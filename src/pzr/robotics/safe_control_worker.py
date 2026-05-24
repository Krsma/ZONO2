"""JSON-lines sidecar worker for the safe-control-gym beta IROS task."""

from __future__ import annotations

import argparse
from functools import partial
import importlib
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

    def __init__(self, root: Path, scenario_config: Path | None, *, controller_mode: str = "firmware") -> None:
        self.root = root
        self.scenario_config = scenario_config or root / "competition" / "level3.yaml"
        if not self.scenario_config.is_absolute():
            self.scenario_config = root / self.scenario_config
        if controller_mode not in {"firmware", "debug_pid"}:
            raise ValueError("controller_mode must be 'firmware' or 'debug_pid'")
        self.controller_mode = controller_mode
        self.env: Any | None = None
        self.obs: np.ndarray | None = None
        self.info: dict[str, Any] = {}
        self.config: dict[str, Any] = {}
        self.reward = 0.0
        self.done = False
        self.time = 0.0
        self.ctrl_dt = 1.0 / 60.0
        self._logical_ctrl_freq = 60
        self._episode_len_sec = float("nan")
        self._firmware_wrapper: Any | None = None
        self._last_motor_action = np.zeros(4, dtype=float)
        self._pid: Any | None = None
        self._thrusts: Any | None = None
        self._planner: LegalGatePlanner | None = None
        self._firmware_status_cache: dict[str, bool] = {}
        self._takeoff_seconds = 3.0
        self._safe_hover_height = 1.0
        self._arena_margin = 0.05
        self._takeoff_arena_margin = 0.05

    def reset(self, seed: int) -> dict[str, Any]:
        self.close()
        _prepare_imports(self.root)
        from safe_control_gym.utils.registration import make

        config = _load_yaml(self.scenario_config)
        self.config = config
        quad_config = dict(config.get("quadrotor_config", {}))
        firmware_status = _firmware_status(self.root)
        self._firmware_status_cache = firmware_status
        quad_config.update({"seed": int(seed), "gui": False, "info_in_reset": True})
        if self.controller_mode == "firmware":
            if not firmware_status["pycffirmware_available"]:
                raise RuntimeError("pycffirmware is not available; use firmware sidecar env for headline runs")
            ctrl_freq = int(quad_config.get("ctrl_freq", 30))
            firmware_freq = 500
            pyb_freq = int(quad_config.get("pyb_freq", firmware_freq))
            if pyb_freq % firmware_freq != 0:
                raise ValueError("firmware mode requires pyb_freq to be a multiple of 500")
            quad_config["ctrl_freq"] = firmware_freq
            quad_config["pyb_freq"] = pyb_freq
            env_func = partial(make, "quadrotor", **quad_config)
            self._firmware_wrapper = make("firmware", env_func, firmware_freq, ctrl_freq)
            obs, info = self._firmware_wrapper.reset()
            self.env = self._firmware_wrapper.env
            info = dict(info)
            info["ctrl_timestep"] = 1.0 / ctrl_freq
            info["ctrl_freq"] = ctrl_freq
            info["firmware_freq"] = firmware_freq
            self._last_motor_action = np.zeros(4, dtype=float)
        else:
            from competition_utils import PIDController, thrusts

            quad_config.update({"ctrl_freq": 60, "pyb_freq": 240})
            self.env = make("quadrotor", **quad_config)
            reset_result = self.env.reset()
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                obs, info = reset_result
            else:
                obs, info = reset_result, self.env._get_reset_info()
            self._pid = PIDController()
            self._thrusts = thrusts
        self.obs = np.asarray(obs, dtype=float)
        self.info = dict(info)
        self.info["controller_mode"] = self.controller_mode
        self.info["pycffirmware_available"] = firmware_status["pycffirmware_available"]
        self._logical_ctrl_freq = int(self.info.get("ctrl_freq", quad_config.get("ctrl_freq", 60)))
        self._episode_len_sec = float(self.info.get("episode_len_sec", quad_config.get("episode_len_sec", float("nan"))))
        self.reward = 0.0
        self.done = False
        self.time = 0.0
        self.ctrl_dt = float(self.info.get("ctrl_timestep", 1.0 / int(self.info.get("ctrl_freq", 60))))
        self._planner = LegalGatePlanner(self.info)
        return self._payload()

    def step(self, action: list[float] | tuple[float, ...]) -> dict[str, Any]:
        if self.env is None or self.obs is None:
            raise RuntimeError("environment has not been reset")
        target_position, target_velocity = self._target_from_action(action)
        if self.controller_mode == "firmware":
            if self._firmware_wrapper is None:
                raise RuntimeError("firmware wrapper has not been reset")
            self._firmware_wrapper.sendFullStateCmd(
                target_position,
                target_velocity,
                np.zeros(3, dtype=float),
                0.0,
                np.zeros(3, dtype=float),
                self.time,
            )
            obs, reward, done, info, motor_action = self._firmware_wrapper.step(self.time, self._last_motor_action)
            self._last_motor_action = np.asarray(motor_action, dtype=float)
        else:
            if self._pid is None or self._thrusts is None:
                raise RuntimeError("debug PID controller has not been reset")
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
        self.info["controller_mode"] = self.controller_mode
        self.info["pycffirmware_available"] = self._firmware_status_cache.get("pycffirmware_available", False)
        self.info["ctrl_freq"] = self._logical_ctrl_freq
        self.info["episode_len_sec"] = self._episode_len_sec
        if self.controller_mode == "firmware":
            self.info.setdefault("firmware_freq", 500)
        self.time += self.ctrl_dt
        return self._payload()

    def nominal_command(self) -> dict[str, Any]:
        return {"command": self._nominal_reference().tolist()}

    def fallback_command(self) -> dict[str, Any]:
        return {"command": self._fallback_reference().tolist()}

    def close(self) -> None:
        if self.env is not None:
            close = getattr(self.env, "close", None)
            if callable(close):
                close()
        self.env = None
        self.obs = None
        self.info = {}
        self.config = {}
        self._firmware_wrapper = None
        self._planner = None

    def _target_from_action(
        self,
        action: list[float] | tuple[float, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.obs is None:
            raise RuntimeError("environment has not been reset")
        command = np.asarray(action, dtype=float).reshape(-1)
        if command.size == 6:
            return command[:3], command[3:]
        if command.size != 3:
            raise ValueError("sidecar action must be a 3D acceleration command or 6D target reference")
        command = np.clip(command, -4.0, 4.0)
        pose = _pose_from_obs(self.obs)
        velocity = _velocity_from_obs(self.obs)
        target_velocity = np.clip(velocity + command * self.ctrl_dt, -2.5, 2.5)
        target_position = pose + target_velocity * self.ctrl_dt
        return target_position, target_velocity

    def _nominal_reference(self) -> np.ndarray:
        if self.obs is None or self._planner is None:
            raise RuntimeError("environment has not been reset")
        pose = _pose_from_obs(self.obs)
        if self.time < self._takeoff_seconds:
            safe_xy = self._safe_xy_target(pose[:2], margin=self._takeoff_arena_margin)
            target_position = np.asarray([safe_xy[0], safe_xy[1], self._safe_hover_height], dtype=float)
            target_velocity = np.zeros(3, dtype=float)
        else:
            target_position, target_velocity = self._planner.reference(self.obs, self.info, self.time)
            target_position[:2] = self._safe_xy_target(target_position[:2])
        return np.concatenate([target_position, target_velocity])

    def _fallback_reference(self) -> np.ndarray:
        if self.obs is None:
            raise RuntimeError("environment has not been reset")
        pose = _pose_from_obs(self.obs)
        safe_xy = self._safe_xy_target(pose[:2], margin=self._takeoff_arena_margin)
        target_position = np.asarray([safe_xy[0], safe_xy[1], max(pose[2], self._safe_hover_height)], dtype=float)
        return np.concatenate([target_position, np.zeros(3, dtype=float)])

    def _safe_xy_target(self, xy: np.ndarray, *, margin: float | None = None) -> np.ndarray:
        bound = 3.0 - (self._arena_margin if margin is None else margin)
        return np.clip(np.asarray(xy, dtype=float).reshape(2), -bound, bound)

    def _payload(self) -> dict[str, Any]:
        if self.env is None or self.obs is None:
            raise RuntimeError("environment has not been reset")
        return {
            "scenario": _scenario_payload(self.env, self.info, self.config),
            "snapshot": _snapshot_payload(self.env, self.obs, self.info, self.done, self.time, self.config),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-control-gym-root", required=True)
    parser.add_argument("--scenario-config", default=None)
    parser.add_argument("--controller-mode", choices=("firmware", "debug_pid"), default="firmware")
    args = parser.parse_args()
    session = IrosSidecarSession(
        Path(args.safe_control_gym_root).expanduser().resolve(),
        None if args.scenario_config is None else Path(args.scenario_config).expanduser(),
        controller_mode=args.controller_mode,
    )
    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request.get("command")
            if command == "reset":
                _write({"ok": True, **session.reset(int(request.get("seed", 0)))})
            elif command == "step":
                _write({"ok": True, **session.step(request.get("action", [0.0, 0.0, 0.0]))})
            elif command == "nominal":
                _write({"ok": True, **session.nominal_command()})
            elif command == "fallback":
                _write({"ok": True, **session.fallback_command()})
            elif command == "status":
                status = session._firmware_status_cache or _firmware_status(session.root)
                _write({"ok": True, **status, "controller_mode": session.controller_mode})
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


class LegalGatePlanner:
    """Current-gate waypoint planner using only IROS-visible online fields."""

    def __init__(self, initial_info: dict[str, Any]) -> None:
        self._gate_dimensions = dict(initial_info.get("gate_dimensions", {}))
        self._x_reference = np.asarray(initial_info.get("x_reference", [-0.5, 0.0, 2.9, 0.0, 0.75, 0.0]), dtype=float)
        self._last_gate_id: int | None = None
        self._phase = "pre"

    def reference(self, obs: np.ndarray, info: dict[str, Any], time: float) -> tuple[np.ndarray, np.ndarray]:
        pose = _pose_from_obs(obs)
        velocity = _velocity_from_obs(obs)
        gate_id = int(info.get("current_target_gate_id", -1))
        if gate_id < 0:
            target = self._final_goal()
        else:
            if gate_id != self._last_gate_id:
                self._last_gate_id = gate_id
                self._phase = "pre"
            center, yaw = self._current_gate_center_and_yaw(info)
            normal = np.asarray([-np.sin(yaw), np.cos(yaw)], dtype=float)
            if np.linalg.norm(normal) < 1e-9:
                normal = np.asarray([1.0, 0.0], dtype=float)
            direction = 1.0 if np.dot(pose[:2] - center[:2], normal) >= 0.0 else -1.0
            pre = center.copy()
            post = center.copy()
            pre[:2] = center[:2] + direction * 0.42 * normal
            post[:2] = center[:2] - direction * 0.48 * normal
            speed = float(np.linalg.norm(velocity))
            if self._phase == "pre" and np.linalg.norm(pose - pre) < 0.06 and speed < 0.25:
                self._phase = "cross"
            if self._phase == "cross" and np.linalg.norm(pose - center) < 0.12:
                self._phase = "post"
            target = {"pre": pre, "cross": center, "post": post}[self._phase]
        error = target - pose
        desired_speed = np.clip(0.6 * error - 0.8 * velocity, -0.45, 0.45)
        lookahead = target + 0.02 * desired_speed
        return lookahead.astype(float), desired_speed.astype(float)

    def _current_gate_center_and_yaw(self, info: dict[str, Any]) -> tuple[np.ndarray, float]:
        raw = np.asarray(info.get("current_target_gate_pos", []), dtype=float).reshape(-1)
        gate_type = int(info.get("current_target_gate_type", 0))
        if raw.size >= 2:
            center = np.asarray([raw[0], raw[1], self._gate_height(gate_type)], dtype=float)
            if raw.size >= 3 and abs(raw[2]) > 0.2:
                center[2] = raw[2]
            yaw = float(raw[5]) if raw.size >= 6 else 0.0
        else:
            center = self._final_goal()
            yaw = 0.0
        return center, yaw

    def _gate_height(self, gate_type: int) -> float:
        dim_key = "low" if gate_type == 1 else "tall"
        dims = self._gate_dimensions.get(dim_key, {})
        if isinstance(dims, dict) and "height" in dims:
            return float(dims["height"])
        return 0.75 if gate_type == 1 else 1.0

    def _final_goal(self) -> np.ndarray:
        if self._x_reference.size >= 5:
            return np.asarray([self._x_reference[0], self._x_reference[2], self._x_reference[4]], dtype=float)
        return np.asarray(self._x_reference[:3], dtype=float)


def _firmware_status(root: Path) -> dict[str, bool]:
    _prepare_imports(root)
    return {
        "pycffirmware_available": _can_import("pycffirmware"),
        "firmware_wrapper_available": _can_import("safe_control_gym.controllers.firmware.firmware_wrapper"),
    }


def _can_import(module: str) -> bool:
    try:
        importlib.import_module(module)
    except Exception:
        return False
    return True


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
    config: dict[str, Any],
) -> dict[str, Any]:
    target_gate = int(info.get("current_target_gate_id", getattr(env, "current_gate", 0)))
    num_gates = int(getattr(env, "NUM_GATES", max(1, target_gate + 1)))
    gates_passed = num_gates if target_gate < 0 else max(0, min(target_gate, num_gates))
    collision = info.get("collision", (None, False))
    if isinstance(collision, (list, tuple)) and len(collision) >= 2:
        collision_flag = bool(collision[1])
    else:
        collision_flag = bool(collision)
    bounds_violation = _state_bounds_violated(obs, config)
    task_completed = bool(info.get("task_completed", False))
    return {
        "pose": _pose_from_obs(obs).tolist(),
        "velocity": _velocity_from_obs(obs).tolist(),
        "target_gate_index": 0 if target_gate < 0 else min(target_gate, num_gates - 1),
        "gates_passed": gates_passed,
        "collision": collision_flag,
        "constraint_violation": bool(info.get("constraint_violation", False)) or bounds_violation or collision_flag,
        "task_completed": task_completed,
        "done": bool(done),
        "time": float(time),
        "info": _json_safe(info),
    }


def _scenario_payload(env: Any, info: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
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
    state_bounds = _state_bounds_from_config(config)
    x_bounds = state_bounds.get(0, (-3.0, 3.0))
    y_bounds = state_bounds.get(2, (-3.0, 3.0))
    z_bounds = state_bounds.get(4, (-0.1, 2.0))
    arena_radius = float(max(abs(x_bounds[0]), abs(x_bounds[1]), abs(y_bounds[0]), abs(y_bounds[1])))
    return {
        "gates": gates,
        "obstacles": obstacles,
        "corridor_radius": arena_radius,
        "min_obstacle_clearance": obstacle_radius,
        "collision_radius": float(getattr(env, "COLLISION_R", 0.0)),
        "altitude_min": float(z_bounds[0]),
        "altitude_max": float(z_bounds[1]),
        "speed_max": 3.0,
        "gate_pass_radius": 0.25,
    }


def _state_bounds_from_config(config: dict[str, Any]) -> dict[int, tuple[float, float]]:
    quad_config = config.get("quadrotor_config", {}) if isinstance(config, dict) else {}
    constraints = quad_config.get("constraints", []) if isinstance(quad_config, dict) else []
    if not isinstance(constraints, list):
        return {}
    for constraint in constraints:
        if not isinstance(constraint, dict):
            continue
        if constraint.get("constraint_form") != "bounded_constraint":
            continue
        if constraint.get("constrained_variable") != "state":
            continue
        active_dims = list(constraint.get("active_dims", []))
        lower_bounds = list(constraint.get("lower_bounds", []))
        upper_bounds = list(constraint.get("upper_bounds", []))
        bounds: dict[int, tuple[float, float]] = {}
        for dim, lower, upper in zip(active_dims, lower_bounds, upper_bounds):
            bounds[int(dim)] = (float(lower), float(upper))
        return bounds
    return {}


def _state_bounds_violated(obs: np.ndarray, config: dict[str, Any]) -> bool:
    bounds = _state_bounds_from_config(config)
    if not bounds:
        return False
    for dim, (lower, upper) in bounds.items():
        if dim >= obs.size:
            continue
        value = float(obs[dim])
        if value < lower or value > upper:
            return True
    return False


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
