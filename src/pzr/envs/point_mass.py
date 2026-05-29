"""MuJoCo point-mass environment: 2D navigation with obstacles.

A point-mass agent navigates through an arena with obstacles toward a goal.
Provides true state for oracle comparison and noisy observations for the monitor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

MODEL_PATH = Path(__file__).parent / "mujoco_models" / "point_mass.xml"

OBSTACLE_POSITIONS = np.array([
    [0.5, 0.5],
    [-0.5, 1.5],
    [1.5, -0.5],
], dtype=np.float64)

OBSTACLE_RADII = np.array([0.4, 0.3, 0.35], dtype=np.float64)

GOAL_POSITION = np.array([2.0, 2.0], dtype=np.float64)
ARENA_HALF_SIZE = 3.0
AGENT_RADIUS = 0.1


@dataclass
class PointMassConfig:
    max_steps: int = 200
    goal_radius: float = 0.3
    goal_reward: float = 10.0
    step_penalty: float = -0.01
    collision_penalty: float = -5.0
    min_obstacle_clearance: float = 0.15
    max_speed: float = 3.0
    boundary_margin: float = 0.3


@dataclass
class PointMassEnv:
    """MuJoCo point-mass with obstacles."""

    config: PointMassConfig = field(default_factory=PointMassConfig)
    _model: Any = field(default=None, repr=False)
    _data: Any = field(default=None, repr=False)
    _step_count: int = 0

    def reset(self, seed: int = 0) -> NDArray[np.float64]:
        self._model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self._data = mujoco.MjData(self._model)
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
        self._step_count = 0
        return self.true_state()

    def step(self, action: NDArray[np.float64]) -> tuple[NDArray[np.float64], float, bool, dict[str, Any]]:
        act = np.clip(action, -1.0, 1.0)
        self._data.ctrl[:] = act
        mujoco.mj_step(self._model, self._data)
        self._step_count += 1

        state = self.true_state()
        pos = state[:2]
        vel = state[2:]

        dist_to_goal = float(np.linalg.norm(pos - GOAL_POSITION))
        reached_goal = dist_to_goal < self.config.goal_radius
        collision = self._check_collision(pos)
        speed = float(np.linalg.norm(vel))

        reward = self.config.step_penalty - 0.1 * dist_to_goal
        if reached_goal:
            reward += self.config.goal_reward
        if collision:
            reward += self.config.collision_penalty

        done = reached_goal or collision or self._step_count >= self.config.max_steps

        info = {
            "dist_to_goal": dist_to_goal,
            "reached_goal": reached_goal,
            "collision": collision,
            "speed": speed,
            "min_obstacle_clearance": self._min_obstacle_clearance(pos),
            "step": self._step_count,
        }
        return state, reward, done, info

    def true_state(self) -> NDArray[np.float64]:
        pos_x = self._data.qpos[0]
        pos_y = self._data.qpos[1]
        vel_x = self._data.qvel[0]
        vel_y = self._data.qvel[1]
        return np.array([pos_x, pos_y, vel_x, vel_y], dtype=np.float64)

    def close(self) -> None:
        self._model = None
        self._data = None

    def _check_collision(self, pos: NDArray[np.float64]) -> bool:
        for obs_pos, obs_r in zip(OBSTACLE_POSITIONS, OBSTACLE_RADII):
            dist = float(np.linalg.norm(pos - obs_pos))
            if dist < obs_r + AGENT_RADIUS:
                return True
        return False

    def _min_obstacle_clearance(self, pos: NDArray[np.float64]) -> float:
        clearances = []
        for obs_pos, obs_r in zip(OBSTACLE_POSITIONS, OBSTACLE_RADII):
            dist = float(np.linalg.norm(pos - obs_pos))
            clearances.append(dist - obs_r - AGENT_RADIUS)
        return min(clearances) if clearances else float("inf")


def simple_pd_controller(
    state: NDArray[np.float64],
    goal: NDArray[np.float64] = GOAL_POSITION,
    kp: float = 1.0,
    kd: float = 0.5,
) -> NDArray[np.float64]:
    """Simple PD controller for reaching a goal position."""
    pos = state[:2]
    vel = state[2:]
    error = goal - pos
    action = kp * error - kd * vel
    return np.clip(action, -1.0, 1.0)
