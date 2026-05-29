"""MuJoCo 3-joint planar robot arm environment.

A planar arm with 3 revolute joints reaches toward targets near a forbidden
zone. Forward kinematics and Jacobian are computed analytically for zonotope
propagation through the monitor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

MODEL_PATH = Path(__file__).parent / "mujoco_models" / "robot_arm.xml"

LINK_LENGTHS = (0.3, 0.25, 0.2)
TOTAL_REACH = sum(LINK_LENGTHS)
NUM_JOINTS = 3
STATE_DIM = 2 * NUM_JOINTS  # angles + velocities

HOME_ANGLES = np.array([0.0, np.pi / 4, -np.pi / 4], dtype=np.float64)

FORBIDDEN_ZONE_CENTER = np.array([0.55, -0.15], dtype=np.float64)
FORBIDDEN_ZONE_HALF = np.array([0.15, 0.15], dtype=np.float64)


def forward_kinematics(
    angles: NDArray[np.float64],
    link_lengths: tuple[float, ...] = LINK_LENGTHS,
) -> NDArray[np.float64]:
    """Planar FK: joint angles -> (x, y) end-effector position."""
    cumulative = np.cumsum(angles)
    x = sum(L * np.cos(cumulative[i]) for i, L in enumerate(link_lengths))
    y = sum(L * np.sin(cumulative[i]) for i, L in enumerate(link_lengths))
    return np.array([float(x), float(y)], dtype=np.float64)


def fk_jacobian(
    angles: NDArray[np.float64],
    link_lengths: tuple[float, ...] = LINK_LENGTHS,
) -> NDArray[np.float64]:
    """FK Jacobian: 2 x n_joints matrix, d(x,y)/d(theta)."""
    n = len(angles)
    cumulative = np.cumsum(angles)
    J = np.zeros((2, n), dtype=np.float64)
    for j in range(n):
        for k in range(j, n):
            J[0, j] += -link_lengths[k] * np.sin(cumulative[k])
            J[1, j] += link_lengths[k] * np.cos(cumulative[k])
    return J


@dataclass
class RobotArmConfig:
    link_lengths: tuple[float, ...] = LINK_LENGTHS
    max_steps: int = 200


@dataclass
class RobotArmEnv:
    """MuJoCo 3-joint planar arm."""

    config: RobotArmConfig = field(default_factory=RobotArmConfig)
    _model: Any = field(default=None, repr=False)
    _data: Any = field(default=None, repr=False)
    _step_count: int = 0

    def reset(self, seed: int = 0) -> NDArray[np.float64]:
        self._model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self._data = mujoco.MjData(self._model)
        mujoco.mj_resetData(self._model, self._data)

        rng = np.random.default_rng(seed)
        perturbation = rng.uniform(-0.1, 0.1, NUM_JOINTS)
        self._data.qpos[:NUM_JOINTS] = HOME_ANGLES + perturbation
        mujoco.mj_forward(self._model, self._data)
        self._step_count = 0
        return self.true_state()

    def step(
        self, action: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float, bool, dict[str, Any]]:
        act = np.clip(action, -1.0, 1.0)
        self._data.ctrl[:NUM_JOINTS] = act
        mujoco.mj_step(self._model, self._data)
        self._step_count += 1

        state = self.true_state()
        ee = forward_kinematics(state[:NUM_JOINTS], self.config.link_lengths)

        dist_to_zone = np.maximum(
            np.abs(ee - FORBIDDEN_ZONE_CENTER) - FORBIDDEN_ZONE_HALF, 0.0,
        )
        in_zone = np.all(dist_to_zone == 0.0)

        done = in_zone or self._step_count >= self.config.max_steps
        reward = -0.01 if not in_zone else -5.0

        info = {
            "ee_pos": ee,
            "in_forbidden_zone": in_zone,
            "step": self._step_count,
        }
        return state, reward, done, info

    def true_state(self) -> NDArray[np.float64]:
        angles = self._data.qpos[:NUM_JOINTS].copy()
        velocities = self._data.qvel[:NUM_JOINTS].copy()
        return np.concatenate([angles, velocities]).astype(np.float64)

    def close(self) -> None:
        self._model = None
        self._data = None


def joint_pd_controller(
    state: NDArray[np.float64],
    target_angles: NDArray[np.float64],
    kp: float = 5.0,
    kd: float = 1.0,
) -> NDArray[np.float64]:
    """PD control in joint space."""
    angles = state[:NUM_JOINTS]
    velocities = state[NUM_JOINTS:]
    action = kp * (target_angles - angles) - kd * velocities
    return np.clip(action, -1.0, 1.0)
