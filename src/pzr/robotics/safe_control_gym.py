"""safe-control-gym adapter boundary for CoRL intervention experiments."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pzr.robotics.iros import Gate, IrosScenario, Obstacle, load_safe_control_gym_iros


@dataclass(frozen=True)
class IrosEnvSnapshot:
    """Normalized simulator state consumed by the CoRL suite."""

    pose: NDArray[np.float64]
    velocity: NDArray[np.float64]
    target_gate_index: int
    gates_passed: int
    collision: bool
    constraint_violation: bool
    task_completed: bool
    done: bool
    time: float
    info: dict[str, Any]

    def __init__(
        self,
        pose: ArrayLike,
        velocity: ArrayLike,
        *,
        target_gate_index: int = 0,
        gates_passed: int = 0,
        collision: bool = False,
        constraint_violation: bool = False,
        task_completed: bool = False,
        done: bool = False,
        time: float = 0.0,
        info: dict[str, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "pose", np.asarray(pose, dtype=float).reshape(3))
        object.__setattr__(self, "velocity", np.asarray(velocity, dtype=float).reshape(3))
        object.__setattr__(self, "target_gate_index", int(target_gate_index))
        object.__setattr__(self, "gates_passed", int(gates_passed))
        object.__setattr__(self, "collision", bool(collision))
        object.__setattr__(self, "constraint_violation", bool(constraint_violation))
        object.__setattr__(self, "task_completed", bool(task_completed))
        object.__setattr__(self, "done", bool(done))
        object.__setattr__(self, "time", float(time))
        object.__setattr__(self, "info", {} if info is None else dict(info))


class IrosEnvClient(Protocol):
    """Minimal environment client used by the CoRL suite."""

    @property
    def scenario(self) -> IrosScenario:
        ...

    @property
    def action_dimension(self) -> int:
        ...

    def reset(self, seed: int) -> IrosEnvSnapshot:
        ...

    def step(self, command: ArrayLike) -> IrosEnvSnapshot:
        ...

    def nominal_command(self, snapshot: IrosEnvSnapshot) -> NDArray[np.float64]:
        ...

    def fallback_command(self, snapshot: IrosEnvSnapshot) -> NDArray[np.float64]:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class PreflightResult:
    """Preflight result for safe-control-gym setup diagnostics."""

    ok: bool
    checks: dict[str, bool]
    messages: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": self.checks,
            "messages": list(self.messages),
        }


class FakeIrosEnvClient:
    """Deterministic safe-control-gym-compatible environment for smoke tests."""

    def __init__(self, max_steps: int = 80, dt: float = 0.05) -> None:
        self._scenario = IrosScenario(
            gates=(
                Gate([1.0, 0.0, 1.0], 0.8, 0.8),
                Gate([2.2, 0.5, 1.1], 0.8, 0.8),
                Gate([3.4, 0.0, 1.0], 0.8, 0.8),
            ),
            obstacles=(Obstacle([1.7, 0.25, 1.0], 0.18),),
            corridor_radius=1.0,
            min_obstacle_clearance=0.08,
            altitude_min=0.25,
            altitude_max=2.4,
            speed_max=2.4,
            gate_pass_radius=0.25,
        )
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self._rng = np.random.default_rng(0)
        self._pose = np.zeros(3)
        self._velocity = np.zeros(3)
        self._target_gate = 0
        self._gates_passed = 0
        self._step = 0
        self._done = False

    @property
    def scenario(self) -> IrosScenario:
        return self._scenario

    @property
    def action_dimension(self) -> int:
        return 3

    def reset(self, seed: int) -> IrosEnvSnapshot:
        self._rng = np.random.default_rng(seed)
        self._pose = np.asarray([0.0, -0.35, 1.0]) + self._rng.normal(0.0, 0.03, 3)
        self._velocity = np.zeros(3)
        self._target_gate = 0
        self._gates_passed = 0
        self._step = 0
        self._done = False
        return self._snapshot()

    def step(self, command: ArrayLike) -> IrosEnvSnapshot:
        if self._done:
            return self._snapshot()
        cmd = np.asarray(command, dtype=float).reshape(3)
        cmd = np.clip(cmd, -3.0, 3.0)
        self._velocity = 0.82 * self._velocity + 0.18 * cmd
        self._pose = self._pose + self.dt * self._velocity
        gate = self._scenario.gate(self._target_gate)
        if np.linalg.norm(self._pose - gate.center) <= self._scenario.gate_pass_radius:
            self._gates_passed = max(self._gates_passed, self._target_gate + 1)
            self._target_gate = min(self._target_gate + 1, len(self._scenario.gates) - 1)
        self._step += 1
        self._done = self._step >= self.max_steps or self._gates_passed == len(self._scenario.gates)
        return self._snapshot()

    def nominal_command(self, snapshot: IrosEnvSnapshot) -> NDArray[np.float64]:
        gate = self._scenario.gate(snapshot.target_gate_index)
        error = gate.center - snapshot.pose
        return 2.8 * error - 0.8 * snapshot.velocity

    def fallback_command(self, snapshot: IrosEnvSnapshot) -> NDArray[np.float64]:
        return -1.2 * snapshot.velocity + np.asarray([0.0, 0.0, 0.2 * (1.0 - snapshot.pose[2])])

    def close(self) -> None:
        self._done = True

    def _snapshot(self) -> IrosEnvSnapshot:
        clearance = min(
            np.linalg.norm(self._pose - obstacle.center) - obstacle.radius
            for obstacle in self._scenario.obstacles
        )
        collision = clearance <= 0.0
        constraint = (
            collision
            or self._pose[2] < self._scenario.altitude_min
            or self._pose[2] > self._scenario.altitude_max
            or np.linalg.norm(self._velocity) > self._scenario.speed_max
        )
        completed = self._gates_passed == len(self._scenario.gates)
        return IrosEnvSnapshot(
            self._pose,
            self._velocity,
            target_gate_index=self._target_gate,
            gates_passed=self._gates_passed,
            collision=collision,
            constraint_violation=constraint,
            task_completed=completed,
            done=self._done,
            time=self._step * self.dt,
            info={"step": self._step},
        )


class DirectSafeControlGymClient:
    """Fail-fast direct safe-control-gym adapter.

    The public safe-control-gym APIs vary across competition branches, so the
    first production implementation intentionally validates imports and then
    reports that no concrete IROS task factory has been registered instead of
    silently running the wrong environment.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        configured = root or os.environ.get("PZR_SAFE_CONTROL_GYM_ROOT")
        self.root = None if configured is None else Path(configured).expanduser()
        self._package: Any | None = None

    @property
    def scenario(self) -> IrosScenario:
        raise RuntimeError("safe-control-gym IROS scenario extraction is unavailable before reset")

    @property
    def action_dimension(self) -> int:
        return 0

    def preflight(self) -> PreflightResult:
        checks: dict[str, bool] = {
            "root_exists": self.root is not None and self.root.exists(),
            "package_import": False,
            "iros_task_detected": False,
            "headless_reset": False,
        }
        messages: list[str] = []
        if not checks["root_exists"]:
            messages.append(
                "safe-control-gym root is missing; set PZR_SAFE_CONTROL_GYM_ROOT or pass --safe-control-gym-root"
            )
            return PreflightResult(False, checks, tuple(messages))
        try:
            self._package = load_safe_control_gym_iros(self.root)
        except Exception as exc:
            messages.append(f"safe_control_gym import failed: {exc}")
            return PreflightResult(False, checks, tuple(messages))
        checks["package_import"] = True
        task_markers = (
            self.root / "safe_control_gym",
            self.root / "examples",
            self.root / "competition",
        )
        checks["iros_task_detected"] = any(path.exists() for path in task_markers)
        messages.append(
            "safe-control-gym imports, but the concrete beta IROS task factory is not wired in this adapter yet"
        )
        return PreflightResult(False, checks, tuple(messages))

    def reset(self, seed: int) -> IrosEnvSnapshot:
        _ = seed
        raise RuntimeError("safe-control-gym IROS reset is not available; run --preflight for setup details")

    def step(self, command: ArrayLike) -> IrosEnvSnapshot:
        _ = command
        raise RuntimeError("safe-control-gym IROS step is not available")

    def nominal_command(self, snapshot: IrosEnvSnapshot) -> NDArray[np.float64]:
        _ = snapshot
        raise RuntimeError("safe-control-gym nominal controller is not available")

    def fallback_command(self, snapshot: IrosEnvSnapshot) -> NDArray[np.float64]:
        _ = snapshot
        return np.zeros(3)

    def close(self) -> None:
        return None


class SidecarSafeControlGymClient:
    """JSON-lines sidecar client for running safe-control-gym in another Python."""

    def __init__(
        self,
        python: str | Path,
        root: str | Path,
        scenario_config: str | Path | None = None,
        *,
        controller_mode: str = "firmware",
    ) -> None:
        self.python = str(python)
        self.root = str(root)
        if controller_mode not in {"firmware", "debug_pid"}:
            raise ValueError("controller_mode must be 'firmware' or 'debug_pid'")
        self.controller_mode = controller_mode
        worker = Path(__file__).with_name("safe_control_worker.py")
        command = [
            self.python,
            str(worker),
            "--safe-control-gym-root",
            self.root,
            "--controller-mode",
            self.controller_mode,
        ]
        if scenario_config is not None:
            command.extend(["--scenario-config", str(scenario_config)])
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
        )
        self._scenario: IrosScenario | None = None

    @property
    def scenario(self) -> IrosScenario:
        if self._scenario is None:
            raise RuntimeError("sidecar scenario is unavailable before reset")
        return self._scenario

    @property
    def action_dimension(self) -> int:
        return 6

    def reset(self, seed: int) -> IrosEnvSnapshot:
        response = self._request({"command": "reset", "seed": seed})
        self._scenario = _scenario_from_payload(response["scenario"])
        return _snapshot_from_payload(response["snapshot"])

    def status(self) -> dict[str, Any]:
        return self._request({"command": "status"})

    def step(self, command: ArrayLike) -> IrosEnvSnapshot:
        response = self._request({"command": "step", "action": np.asarray(command).tolist()})
        return _snapshot_from_payload(response["snapshot"])

    def nominal_command(self, snapshot: IrosEnvSnapshot) -> NDArray[np.float64]:
        _ = snapshot
        response = self._request({"command": "nominal"})
        return np.asarray(response["command"], dtype=float)

    def fallback_command(self, snapshot: IrosEnvSnapshot) -> NDArray[np.float64]:
        _ = snapshot
        response = self._request({"command": "fallback"})
        return np.asarray(response["command"], dtype=float)

    def close(self) -> None:
        if self._process.poll() is None:
            try:
                self._request({"command": "close"})
            except Exception:
                pass
            self._process.terminate()

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("sidecar worker pipes are closed")
        self._process.stdin.write(json.dumps(payload) + "\n")
        self._process.stdin.flush()
        response: dict[str, Any] | None = None
        while response is None:
            line = self._process.stdout.readline()
            if not line:
                stderr = self._process.stderr.read() if self._process.stderr is not None else ""
                raise RuntimeError(f"safe-control-gym sidecar stopped: {stderr}")
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                continue
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "safe-control-gym sidecar request failed"))
        return response


def make_env_client(
    *,
    profile: str,
    safe_control_gym_root: str | Path | None,
    safe_control_python: str | Path | None,
    safe_control_config: str | Path | None = None,
    safe_control_controller_mode: str = "firmware",
    allow_debug_pid: bool = False,
    fake_max_steps: int | None = None,
) -> IrosEnvClient:
    """Create the requested environment client."""

    if profile == "smoke" and safe_control_gym_root is None and safe_control_python is None:
        return FakeIrosEnvClient(max_steps=40 if fake_max_steps is None else int(fake_max_steps))
    if safe_control_python is not None:
        if safe_control_controller_mode != "firmware" and not allow_debug_pid:
            raise RuntimeError("debug_pid sidecar mode requires --allow-debug-pid and is diagnostic only")
        root = safe_control_gym_root or os.environ.get("PZR_SAFE_CONTROL_GYM_ROOT")
        if root is None:
            raise RuntimeError("--safe-control-python requires --safe-control-gym-root or PZR_SAFE_CONTROL_GYM_ROOT")
        return SidecarSafeControlGymClient(
            safe_control_python,
            root,
            safe_control_config,
            controller_mode=safe_control_controller_mode,
        )
    return DirectSafeControlGymClient(safe_control_gym_root)


def preflight_safe_control_gym(
    *,
    profile: str,
    safe_control_gym_root: str | Path | None,
    safe_control_python: str | Path | None,
    safe_control_config: str | Path | None = None,
    safe_control_controller_mode: str = "firmware",
    allow_debug_pid: bool = False,
) -> PreflightResult:
    """Run setup checks for the requested CoRL environment path."""

    try:
        import torch  # noqa: F401
        torch_ok = True
    except ImportError:
        torch_ok = False
    if profile == "smoke" and safe_control_gym_root is None and safe_control_python is None:
        client = FakeIrosEnvClient(max_steps=5)
        snapshot = client.reset(0)
        checks = {
            "fake_env_reset": True,
            "geometry": bool(client.scenario.gates),
            "snapshot_pose_velocity": snapshot.pose.shape == (3,) and snapshot.velocity.shape == (3,),
            "torch": torch_ok,
        }
        required = checks["fake_env_reset"] and checks["geometry"] and checks["snapshot_pose_velocity"]
        return PreflightResult(required, checks, ("fake CoRL smoke environment is available",))
    if safe_control_python is not None:
        root = safe_control_gym_root or os.environ.get("PZR_SAFE_CONTROL_GYM_ROOT")
        checks = {
            "safe_control_python_exists": Path(safe_control_python).exists(),
            "safe_control_gym_root_exists": bool(root) and Path(root).exists(),
            "sidecar_controller_firmware": safe_control_controller_mode == "firmware",
            "debug_pid_explicitly_allowed": allow_debug_pid or safe_control_controller_mode == "firmware",
            "pycffirmware_available": False,
            "firmware_wrapper_available": False,
            "firmware_reset": False,
            "firmware_step": False,
            "sidecar_reset": False,
            "sidecar_step": False,
            "torch": torch_ok,
        }
        messages: list[str] = []
        if safe_control_controller_mode not in {"firmware", "debug_pid"}:
            messages.append("safe-control controller mode must be 'firmware' or 'debug_pid'")
        if safe_control_controller_mode != "firmware" and not allow_debug_pid:
            messages.append("debug_pid sidecar mode is diagnostic only; pass --allow-debug-pid to use it")
        if not checks["safe_control_python_exists"]:
            messages.append(f"safe-control sidecar Python is missing: {safe_control_python}")
        if not checks["safe_control_gym_root_exists"]:
            messages.append(
                "safe-control-gym root is missing; pass --safe-control-gym-root or set PZR_SAFE_CONTROL_GYM_ROOT"
            )
        if messages:
            return PreflightResult(False, checks, tuple(messages))
        client = SidecarSafeControlGymClient(
            safe_control_python,
            root,
            safe_control_config,
            controller_mode=safe_control_controller_mode,
        )
        try:
            status = client.status()
            checks["pycffirmware_available"] = bool(status.get("pycffirmware_available", False))
            checks["firmware_wrapper_available"] = bool(status.get("firmware_wrapper_available", False))
            snapshot = client.reset(0)
            checks["sidecar_reset"] = snapshot.pose.shape == (3,) and snapshot.velocity.shape == (3,)
            command = client.nominal_command(snapshot)
            next_snapshot = client.step(command)
            checks["sidecar_step"] = next_snapshot.pose.shape == (3,) and next_snapshot.velocity.shape == (3,)
            checks["firmware_reset"] = checks["sidecar_reset"] and safe_control_controller_mode == "firmware"
            checks["firmware_step"] = checks["sidecar_step"] and safe_control_controller_mode == "firmware"
        except Exception as exc:
            messages.append(f"safe-control-gym sidecar reset/step failed: {exc}")
        finally:
            client.close()
        if not messages:
            if safe_control_controller_mode == "firmware":
                messages.append("safe-control-gym firmware sidecar reset and step succeeded")
            else:
                messages.append("safe-control-gym debug PID sidecar reset and step succeeded (diagnostic only)")
        headline_required = (
            checks["safe_control_python_exists"]
            and checks["safe_control_gym_root_exists"]
            and checks["sidecar_reset"]
            and checks["sidecar_step"]
            and checks["debug_pid_explicitly_allowed"]
        )
        if safe_control_controller_mode == "firmware":
            headline_required = (
                headline_required
                and checks["sidecar_controller_firmware"]
                and checks["pycffirmware_available"]
                and checks["firmware_wrapper_available"]
                and checks["firmware_reset"]
                and checks["firmware_step"]
            )
        return PreflightResult(bool(headline_required), checks, tuple(messages))
    direct = DirectSafeControlGymClient(safe_control_gym_root)
    result = direct.preflight()
    checks = dict(result.checks)
    checks["torch"] = torch_ok
    return PreflightResult(result.ok, checks, result.messages)


def _scenario_from_payload(payload: dict[str, Any]) -> IrosScenario:
    return IrosScenario(
        gates=tuple(Gate(gate["center"], gate.get("width", 0.8), gate.get("height", 0.8)) for gate in payload["gates"]),
        obstacles=tuple(Obstacle(obstacle["center"], obstacle["radius"]) for obstacle in payload.get("obstacles", [])),
        corridor_radius=float(payload.get("corridor_radius", 1.5)),
        min_obstacle_clearance=float(payload.get("min_obstacle_clearance", 0.1)),
        collision_radius=float(payload.get("collision_radius", 0.0)),
        altitude_min=float(payload.get("altitude_min", 0.2)),
        altitude_max=float(payload.get("altitude_max", 3.0)),
        speed_max=float(payload.get("speed_max", 4.0)),
        gate_pass_radius=float(payload.get("gate_pass_radius", 0.35)),
    )


def _snapshot_from_payload(payload: dict[str, Any]) -> IrosEnvSnapshot:
    return IrosEnvSnapshot(
        payload["pose"],
        payload["velocity"],
        target_gate_index=payload.get("target_gate_index", 0),
        gates_passed=payload.get("gates_passed", 0),
        collision=payload.get("collision", False),
        constraint_violation=payload.get("constraint_violation", False),
        task_completed=payload.get("task_completed", False),
        done=payload.get("done", False),
        time=payload.get("time", 0.0),
        info=payload.get("info", {}),
    )
