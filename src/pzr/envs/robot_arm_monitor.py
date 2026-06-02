"""Monitor adapter for the 3-joint planar robot arm.

State: [theta_1, theta_2, theta_3, omega_1, omega_2, omega_3] (6D joint space)
Triggers: evaluated in Cartesian end-effector space via FK Jacobian

The zonotope lives in joint space (where generators grow from encoder noise).
Safety is checked in Cartesian space: the FK Jacobian maps the joint-space
zonotope to a 2D end-effector zonotope, and triggers evaluate on that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from pzr.envs.base import NoisySensorModel
from pzr.envs.robot_arm import (
    FORBIDDEN_ZONE_CENTER,
    FORBIDDEN_ZONE_HALF,
    LINK_LENGTHS,
    NUM_JOINTS,
    RobotArmConfig,
    STATE_DIM,
    fk_jacobian,
    forward_kinematics,
)
from pzr.monitoring.base import MonitorResult, MonitorState, TriggerSpec
from pzr.monitoring.triggers import evaluate_triggers
from pzr.zonotope.core import Zonotope

ANGLE_0 = 0
ANGLE_1 = 1
ANGLE_2 = 2
VEL_0 = 3
VEL_1 = 4
VEL_2 = 5

EE_X = 0
EE_Y = 1

JOINT_COUPLING_UNSCALED = np.array([
    [1.0, 0.2, -0.2],
    [-0.2, 1.0, 0.2],
    [0.2, -0.2, 1.0],
], dtype=np.float64)
JOINT_COUPLING_SCALE = float(
    np.max(np.sum(np.abs(np.linalg.inv(JOINT_COUPLING_UNSCALED)), axis=1))
)
JOINT_COUPLING_BASIS = JOINT_COUPLING_SCALE * JOINT_COUPLING_UNSCALED


@dataclass(frozen=True)
class RobotArmMeasurement:
    """Noisy observation from the robot arm."""

    time: float
    joint_angles: tuple[float, float, float]
    joint_velocities: tuple[float, float, float]


@dataclass(frozen=True)
class RobotArmTraceRecord:
    """One MuJoCo trace step with true state and noisy monitor input."""

    time: float
    true_state: NDArray[np.float64]
    measurement: RobotArmMeasurement
    target_angles: NDArray[np.float64]
    action: NDArray[np.float64]
    ee_pos: NDArray[np.float64]
    in_forbidden_zone: bool
    done: bool
    episode_id: int


@dataclass(frozen=True)
class RobotArmMonitor:
    """Zonotope monitor for the 3-joint planar arm.

    The zonotope state is in joint space (6D). Triggers are evaluated on a
    derived 2D Cartesian zonotope obtained via the FK Jacobian.
    """

    noise_model: NoisySensorModel
    config: RobotArmConfig = field(default_factory=RobotArmConfig)
    forbidden_zone_x_lo: float = FORBIDDEN_ZONE_CENTER[0] - FORBIDDEN_ZONE_HALF[0]
    forbidden_zone_x_hi: float = FORBIDDEN_ZONE_CENTER[0] + FORBIDDEN_ZONE_HALF[0]
    forbidden_zone_y_lo: float = FORBIDDEN_ZONE_CENTER[1] - FORBIDDEN_ZONE_HALF[1]
    forbidden_zone_y_hi: float = FORBIDDEN_ZONE_CENTER[1] + FORBIDDEN_ZONE_HALF[1]
    wall_x: float = 0.7
    floor_y: float = -0.1

    @property
    def triggers(self) -> tuple[TriggerSpec, ...]:
        return (
            TriggerSpec("ee_floor", EE_Y, self.floor_y, "below", overlap=0.05),
            TriggerSpec("ee_wall", EE_X, self.wall_x, "above", overlap=0.05),
            TriggerSpec("ee_zone_x_lo", EE_X, self.forbidden_zone_x_lo, "above", overlap=0.05),
            TriggerSpec("ee_zone_y_lo", EE_Y, self.forbidden_zone_y_lo, "below", overlap=0.05),
        )

    @property
    def num_calibration_generators(self) -> int:
        return NUM_JOINTS

    def initial_state(self) -> MonitorState:
        center = np.zeros(STATE_DIM, dtype=np.float64)
        generators = _coupled_joint_generators(self.noise_model.bias_bound)
        return MonitorState(
            zonotope=Zonotope(center, generators),
            step=0,
            calibration_indices=tuple(range(NUM_JOINTS)),
        )

    def clone_state(self, state: MonitorState) -> MonitorState:
        z = state.zonotope
        return MonitorState(
            zonotope=Zonotope(z.center.copy(), z.generators.copy()),
            step=state.step,
            calibration_indices=state.calibration_indices,
            payload=state.payload,
        )

    def replace_zonotope(self, state: MonitorState, zonotope: Zonotope) -> MonitorState:
        return state.with_zonotope(zonotope)

    def trigger_zonotope(self, state: MonitorState) -> Zonotope:
        return self._cartesian_zonotope(state.zonotope)

    def step(self, state: MonitorState, measurement: RobotArmMeasurement) -> MonitorResult:
        old_z = state.zonotope
        old_g = old_z.generators
        n_existing = old_z.generator_count
        cal = state.calibration_indices

        new_center = np.array([
            measurement.joint_angles[0],
            measurement.joint_angles[1],
            measurement.joint_angles[2],
            measurement.joint_velocities[0],
            measurement.joint_velocities[1],
            measurement.joint_velocities[2],
        ], dtype=np.float64)

        bias_bound = self.noise_model.bias_bound
        noise_bound = self.noise_model.noise_bound

        n_new = n_existing + NUM_JOINTS
        new_g = np.zeros((STATE_DIM, n_new), dtype=np.float64)

        if n_existing > 0:
            new_g[:, :n_existing] = old_g

        calibration_g = _coupled_joint_generators(bias_bound)
        for j in range(NUM_JOINTS):
            if j < len(cal) and cal[j] < n_existing:
                new_g[:, cal[j]] = calibration_g[:, j]

        new_g[:, n_existing:n_new] = _coupled_joint_generators(noise_bound)

        new_z = Zonotope(new_center, new_g)
        new_state = MonitorState(
            zonotope=new_z,
            step=state.step + 1,
            calibration_indices=cal,
            payload=measurement.time,
        )

        cart_z = self._cartesian_zonotope(new_z)
        verdicts = evaluate_triggers(cart_z, self.triggers)
        return MonitorResult(new_state, verdicts)

    def _cartesian_zonotope(self, joint_z: Zonotope) -> Zonotope:
        """Map joint-space zonotope to 2D Cartesian end-effector zonotope."""
        angles = joint_z.center[:NUM_JOINTS]
        ee_pos = forward_kinematics(angles, self.config.link_lengths)
        J = fk_jacobian(angles, self.config.link_lengths)

        # Build 2x6 matrix: [J | 0_{2x3}] — Jacobian on angles, zero on velocities
        M = np.zeros((2, STATE_DIM), dtype=np.float64)
        M[:, :NUM_JOINTS] = J

        bias = ee_pos - M @ joint_z.center
        return joint_z.affine_map(M, bias)


def _coupled_joint_generators(bounds: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return 3 coupled generator columns for angle and velocity bounds."""
    b = np.asarray(bounds, dtype=np.float64).ravel()
    angle_bounds = np.zeros(NUM_JOINTS, dtype=np.float64)
    velocity_bounds = np.zeros(NUM_JOINTS, dtype=np.float64)
    for j in range(NUM_JOINTS):
        angle_bounds[j] = float(b[j]) if j < b.size else 0.0
        v_idx = NUM_JOINTS + j
        velocity_bounds[j] = float(b[v_idx]) if v_idx < b.size else 0.5 * angle_bounds[j]

    generators = np.zeros((STATE_DIM, NUM_JOINTS), dtype=np.float64)
    generators[:NUM_JOINTS, :] = np.diag(angle_bounds) @ JOINT_COUPLING_BASIS
    generators[NUM_JOINTS:, :] = np.diag(velocity_bounds) @ JOINT_COUPLING_BASIS
    return generators


def generate_robot_arm_trace(
    length: int,
    *,
    seed: int = 0,
    bias_bound: NDArray[np.float64] | None = None,
    noise_bound: NDArray[np.float64] | None = None,
) -> tuple[RobotArmMeasurement, ...]:
    """Generate a trace of noisy measurements from a robot arm episode.

    Drives the arm through waypoints near the forbidden zone boundary,
    creating trajectories that stress the safety certificate.
    """
    records = generate_robot_arm_trace_records(
        length,
        seed=seed,
        bias_bound=bias_bound,
        noise_bound=noise_bound,
    )
    return tuple(record.measurement for record in records)


def generate_robot_arm_trace_records(
    length: int,
    *,
    seed: int = 0,
    bias_bound: NDArray[np.float64] | None = None,
    noise_bound: NDArray[np.float64] | None = None,
) -> tuple[RobotArmTraceRecord, ...]:
    """Generate a MuJoCo trace with true state, control, and noisy measurement."""
    from pzr.envs.robot_arm import RobotArmEnv, joint_pd_controller

    if bias_bound is None:
        bias_bound = np.array([0.02, 0.02, 0.02, 0.01, 0.01, 0.01])
    if noise_bound is None:
        noise_bound = np.array([0.01, 0.01, 0.01, 0.005, 0.005, 0.005])

    noise_model = NoisySensorModel(bias_bound=bias_bound, noise_bound=noise_bound)
    rng = np.random.default_rng(seed)
    noise_model.reset(rng)

    num_waypoints = max(3, length // 30)
    waypoint_angles = _generate_waypoint_angles(num_waypoints, rng)
    waypoint_idx = 0

    env = RobotArmEnv(config=RobotArmConfig(max_steps=length + 10))
    true_state = env.reset(seed=seed)
    episode_id = 0
    records: list[RobotArmTraceRecord] = []

    try:
        for t in range(length):
            target = waypoint_angles[waypoint_idx]
            action = joint_pd_controller(true_state, target)
            true_state, _, done, info = env.step(action)
            measurement = _make_measurement(true_state, noise_model, rng, float(t))
            ee_pos = np.asarray(
                info.get("ee_pos", forward_kinematics(true_state[:NUM_JOINTS])),
                dtype=np.float64,
            )
            records.append(RobotArmTraceRecord(
                time=float(t),
                true_state=true_state.copy(),
                measurement=measurement,
                target_angles=target.copy(),
                action=action.copy(),
                ee_pos=ee_pos.copy(),
                in_forbidden_zone=bool(info.get("in_forbidden_zone", False)),
                done=bool(done),
                episode_id=episode_id,
            ))

            angle_error = float(np.linalg.norm(true_state[:NUM_JOINTS] - target))
            if angle_error < 0.15 and waypoint_idx < num_waypoints - 1:
                waypoint_idx += 1

            if done and t < length - 1:
                env.close()
                env = RobotArmEnv(config=RobotArmConfig(max_steps=length + 10))
                episode_id += 1
                true_state = env.reset(seed=seed + t + 1)
    finally:
        env.close()

    return tuple(records)


def _generate_waypoint_angles(
    num_waypoints: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Generate joint-angle waypoints that put the end-effector near the forbidden zone."""
    waypoints = []
    for _ in range(num_waypoints):
        # Mix of configurations: some near the zone, some away
        if rng.random() < 0.6:
            # Target near forbidden zone: ee around (0.4-0.7, -0.3 to 0.0)
            target_x = rng.uniform(0.35, 0.75)
            target_y = rng.uniform(-0.35, 0.05)
            angles = _ik_numerical(np.array([target_x, target_y]), rng)
        else:
            # Random reachable configuration
            angles = rng.uniform(
                [-1.5, -1.0, -1.5],
                [1.5, 1.5, 1.0],
            )
        waypoints.append(angles)
    return np.array(waypoints, dtype=np.float64)


def _ik_numerical(
    target: NDArray[np.float64],
    rng: np.random.Generator,
    max_iter: int = 50,
) -> NDArray[np.float64]:
    """Simple iterative IK using the Jacobian pseudoinverse."""
    angles = rng.uniform(-0.5, 0.5, NUM_JOINTS)
    for _ in range(max_iter):
        ee = forward_kinematics(angles)
        error = target - ee
        if np.linalg.norm(error) < 0.01:
            break
        J = fk_jacobian(angles)
        dtheta = np.linalg.lstsq(J, error, rcond=None)[0]
        angles = angles + 0.5 * dtheta
        angles = np.clip(angles, -3.0, 3.0)
    return angles


def _make_measurement(
    true_state: NDArray[np.float64],
    noise_model: NoisySensorModel,
    rng: np.random.Generator,
    time: float,
) -> RobotArmMeasurement:
    """Create a noisy measurement from the true state."""
    noisy = noise_model.observe(true_state, rng)
    return RobotArmMeasurement(
        time=time,
        joint_angles=(float(noisy[0]), float(noisy[1]), float(noisy[2])),
        joint_velocities=(float(noisy[3]), float(noisy[4]), float(noisy[5])),
    )
