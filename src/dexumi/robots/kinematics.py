"""Shared pyroki-based bimanual end-effector IK.

This module is the single implementation of the IK solve loop. Robot-specific
modules (``axol/solver.py``, ``piper/solver.py``) only need to supply a
:class:`RobotKinematicsSpec` describing their URDF link/joint names and an
optional collision-pair builder, then call :func:`make_kinematics_solver` to
get a ready-to-use ``KinematicsSolver`` class bound to that spec.
"""

from __future__ import annotations

import functools
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import yourdfpy

_logger = logging.getLogger(__name__)


@dataclass
class KinematicsConfig:
    """IK solver cost weights and runtime limits (shared by all embodiments)."""

    pos_weight: float = 50.0
    ori_weight: float = 10.0
    rest_weight: float = 7.5
    posture_weight: float = 5.0
    manipulability_weight: float = 0.05
    limit_weight: float = 75.0
    self_collision_margin: float = 0.1
    self_collision_weight: float = 75.0
    max_iterations: int = 8
    cost_tolerance: float = 1e-2
    max_joint_delta: float = 0.0055 * 2 * math.pi
    max_reach: float = 0.8


CollisionBuilder = Callable[[yourdfpy.URDF, pk.Robot, object], pk.collision.RobotCollision]


@dataclass(frozen=True)
class RobotKinematicsSpec:
    """Robot-specific naming/configuration needed by the shared IK solver.

    All link/joint names must match the robot's URDF exactly.

    ``left_arm_joint_names`` / ``right_arm_joint_names`` are the joints solved
    by IK (revolute only for robots like Piper that also have prismatic fingers).
    ``left_control_joint_names`` / ``right_control_joint_names`` are the joints
    used to build the output command; they default to the IK joint names when
    not provided.

    ``left_elbow_link`` / ``right_elbow_link`` are optional link names used to
    expose elbow indices on the solver (e.g. for visualization helpers).
    """

    name: str
    urdf_path: Path
    left_ee_link: str
    right_ee_link: str
    left_shoulder_link: str
    right_shoulder_link: str
    left_arm_joint_names: tuple[str, ...]
    right_arm_joint_names: tuple[str, ...]
    left_control_joint_names: tuple[str, ...] | None = None
    right_control_joint_names: tuple[str, ...] | None = None
    left_elbow_link: str | None = None
    right_elbow_link: str | None = None
    collision_builder: CollisionBuilder | None = None


@functools.partial(jax.jit, static_argnames=("max_iterations",))
def _solve_ee_ik(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    target_L: jaxlie.SE3 | None,
    target_R: jaxlie.SE3 | None,
    L_ee_idx: jax.Array,
    R_ee_idx: jax.Array,
    q_current: jax.Array,
    posture_pose: jax.Array,
    pos_weight: float,
    ori_weight: float,
    rest_weight: float,
    posture_weight: float,
    manipulability_weight: float,
    limit_weight: float,
    self_collision_margin: float,
    self_collision_weight: float,
    max_iterations: int,
    cost_tolerance: float,
) -> jax.Array:
    JointVar = robot.joint_var_cls
    ee_indices = jnp.array([L_ee_idx, R_ee_idx], dtype=jnp.int32)

    costs = [
        pk.costs.rest_cost(JointVar(0), rest_pose=q_current, weight=rest_weight),
        pk.costs.rest_cost(JointVar(0), rest_pose=posture_pose, weight=posture_weight),
        pk.costs.manipulability_cost(
            robot,
            JointVar(0),
            ee_indices,
            weight=manipulability_weight,
        ),
    ]

    if target_L is not None:
        costs.append(
            pk.costs.pose_cost_analytic_jac(
                robot,
                JointVar(0),
                target_L,
                jnp.array(L_ee_idx, dtype=jnp.int32),
                pos_weight=pos_weight,
                ori_weight=ori_weight,
            )
        )
    if target_R is not None:
        costs.append(
            pk.costs.pose_cost_analytic_jac(
                robot,
                JointVar(0),
                target_R,
                jnp.array(R_ee_idx, dtype=jnp.int32),
                pos_weight=pos_weight,
                ori_weight=ori_weight,
            )
        )

    costs.append(pk.costs.limit_cost(robot, JointVar(0), weight=limit_weight))
    costs.append(
        pk.costs.self_collision_cost(
            robot,
            robot_coll,
            JointVar(0),
            margin=self_collision_margin,
            weight=self_collision_weight,
        )
    )

    var_joints = JointVar(jnp.array([0]))
    initial_vals = jaxls.VarValues.make(
        [var_joints.with_value(q_current[jnp.newaxis, :])]
    )
    problem = jaxls.LeastSquaresProblem(costs, [var_joints])
    solution_vals = problem.analyze().solve(
        initial_vals=initial_vals,
        verbose=False,
        linear_solver="dense_cholesky",
        trust_region=jaxls.TrustRegionConfig(),
        termination=jaxls.TerminationConfig(
            max_iterations=max_iterations,
            cost_tolerance=cost_tolerance,
        ),
    )
    return solution_vals[var_joints][0]


def _default_collision_builder(
    urdf: yourdfpy.URDF,
    robot: pk.Robot,
    config: object,
) -> pk.collision.RobotCollision:
    del robot, config
    return pk.collision.RobotCollision.from_urdf(urdf)


def _clamp_reach(pos: np.ndarray, center: np.ndarray, max_reach: float) -> np.ndarray:
    d = pos - center
    dist = np.linalg.norm(d)
    if dist > max_reach:
        return (center + d * (max_reach / dist)).astype(np.float32)
    return pos.astype(np.float32)


def _rot_3x3_to_wxyz(R: np.ndarray) -> np.ndarray:
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        r = np.sqrt(t + 1.0)
        s = 0.5 / r
        return np.array(
            [
                0.5 * r,
                (R[2, 1] - R[1, 2]) * s,
                (R[0, 2] - R[2, 0]) * s,
                (R[1, 0] - R[0, 1]) * s,
            ],
            np.float32,
        )
    if R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        r = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        s = 0.5 / r
        return np.array(
            [
                (R[2, 1] - R[1, 2]) * s,
                0.5 * r,
                (R[0, 1] + R[1, 0]) * s,
                (R[0, 2] + R[2, 0]) * s,
            ],
            np.float32,
        )
    if R[1, 1] >= R[2, 2]:
        r = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        s = 0.5 / r
        return np.array(
            [
                (R[0, 2] - R[2, 0]) * s,
                (R[0, 1] + R[1, 0]) * s,
                0.5 * r,
                (R[1, 2] + R[2, 1]) * s,
            ],
            np.float32,
        )
    r = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    s = 0.5 / r
    return np.array(
        [
            (R[1, 0] - R[0, 1]) * s,
            (R[0, 2] + R[2, 0]) * s,
            (R[1, 2] + R[2, 1]) * s,
            0.5 * r,
        ],
        np.float32,
    )


def _np_to_se3(pos: np.ndarray, rot_3x3: np.ndarray) -> jaxlie.SE3:
    return jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(wxyz=jnp.asarray(_rot_3x3_to_wxyz(rot_3x3))),
        jnp.asarray(pos, dtype=jnp.float32),
    )


class BimanualPyrokiSolver:
    """Shared bimanual IK solver that tracks end-effectors only."""

    def __init__(self, *, spec: RobotKinematicsSpec, config: object) -> None:
        self.spec = spec
        self.config = config

        _logger.info("Loading %s URDF...", spec.name)
        urdf = yourdfpy.URDF.load(str(spec.urdf_path), mesh_dir=str(spec.urdf_path.parent))
        self.urdf = urdf
        self.robot = pk.Robot.from_urdf(urdf)
        collision_builder = spec.collision_builder or _default_collision_builder
        self.robot_coll = collision_builder(urdf, self.robot, config)

        names = self.robot.links.names
        self.l_ee_idx = names.index(spec.left_ee_link)
        self.r_ee_idx = names.index(spec.right_ee_link)
        self._l_ee_idx_jax = jnp.asarray(self.l_ee_idx, dtype=jnp.int32)
        self._r_ee_idx_jax = jnp.asarray(self.r_ee_idx, dtype=jnp.int32)

        fk0 = self.robot.forward_kinematics(
            jnp.zeros(self.robot.joints.num_actuated_joints)
        )
        self._left_shoulder_pos = np.asarray(
            jaxlie.SE3(fk0[names.index(spec.left_shoulder_link)]).translation(),
            dtype=np.float32,
        )
        self._right_shoulder_pos = np.asarray(
            jaxlie.SE3(fk0[names.index(spec.right_shoulder_link)]).translation(),
            dtype=np.float32,
        )

        actuated = list(self.robot.joints.actuated_names)
        name_to_idx = {name: i for i, name in enumerate(actuated)}
        self.left_indices = [name_to_idx[name] for name in spec.left_arm_joint_names]
        self.right_indices = [name_to_idx[name] for name in spec.right_arm_joint_names]
        self.left_joint_indices = [
            name_to_idx[name] for name in (spec.left_control_joint_names or spec.left_arm_joint_names)
        ]
        self.right_joint_indices = [
            name_to_idx[name]
            for name in (spec.right_control_joint_names or spec.right_arm_joint_names)
        ]

        self._posture_pose = jnp.zeros(self.num_joints, dtype=jnp.float32)
        self._warmup()

    @property
    def joint_names(self) -> list[str]:
        return list(self.robot.joints.actuated_names)

    @property
    def num_joints(self) -> int:
        return self.robot.joints.num_actuated_joints

    def set_posture_pose(self, q: np.ndarray) -> None:
        self._posture_pose = jnp.asarray(q, dtype=jnp.float32)

    def fk(self, q: np.ndarray) -> tuple[jaxlie.SE3, jaxlie.SE3]:
        fk = self.robot.forward_kinematics(jnp.asarray(q, dtype=jnp.float32))
        return jaxlie.SE3(fk[self.l_ee_idx]), jaxlie.SE3(fk[self.r_ee_idx])

    def link_positions(self, q: np.ndarray, link_indices: list[int]) -> np.ndarray:
        fk = self.robot.forward_kinematics(jnp.asarray(q, dtype=jnp.float32))
        return np.asarray(
            [jaxlie.SE3(fk[index]).translation() for index in link_indices],
            dtype=np.float32,
        )

    def link_pose(self, q: np.ndarray, link_index: int) -> tuple[np.ndarray, np.ndarray]:
        fk = self.robot.forward_kinematics(jnp.asarray(q, dtype=jnp.float32))
        pose = jaxlie.SE3(fk[link_index])
        return (
            np.asarray(pose.translation(), dtype=np.float32),
            np.asarray(pose.rotation().as_matrix(), dtype=np.float32),
        )

    def ik(
        self,
        q_current: np.ndarray,
        left_pose: tuple[np.ndarray, np.ndarray] | None = None,
        right_pose: tuple[np.ndarray, np.ndarray] | None = None,
        left_elbow_pos: np.ndarray | None = None,
        right_elbow_pos: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve one frame using end-effector pose targets only.

        ``left_elbow_pos`` and ``right_elbow_pos`` are accepted for compatibility
        with older callers and intentionally ignored.
        """
        del left_elbow_pos, right_elbow_pos
        if left_pose is None and right_pose is None:
            return np.asarray(q_current, dtype=np.float32)

        cfg = self.config
        q_current = np.asarray(q_current, dtype=np.float32)

        target_L: jaxlie.SE3 | None = None
        if left_pose is not None:
            pos, rot = left_pose
            target_L = _np_to_se3(
                _clamp_reach(
                    np.asarray(pos, dtype=np.float32),
                    self._left_shoulder_pos,
                    cfg.max_reach,
                ),
                np.asarray(rot, dtype=np.float32),
            )

        target_R: jaxlie.SE3 | None = None
        if right_pose is not None:
            pos, rot = right_pose
            target_R = _np_to_se3(
                _clamp_reach(
                    np.asarray(pos, dtype=np.float32),
                    self._right_shoulder_pos,
                    cfg.max_reach,
                ),
                np.asarray(rot, dtype=np.float32),
            )

        q_result = _solve_ee_ik(
            self.robot,
            self.robot_coll,
            target_L,
            target_R,
            self._l_ee_idx_jax,
            self._r_ee_idx_jax,
            jnp.asarray(q_current, dtype=jnp.float32),
            self._posture_pose,
            cfg.pos_weight,
            cfg.ori_weight,
            cfg.rest_weight,
            cfg.posture_weight,
            cfg.manipulability_weight,
            cfg.limit_weight,
            cfg.self_collision_margin,
            cfg.self_collision_weight,
            cfg.max_iterations,
            cfg.cost_tolerance,
        )
        q_result_np = np.asarray(q_result, dtype=np.float32)
        delta = np.clip(
            q_result_np - q_current,
            -cfg.max_joint_delta,
            cfg.max_joint_delta,
        )
        return (q_current + delta).astype(np.float32)

    def _warmup(self) -> None:
        _logger.info("Warming up %s EE IK solver (JIT compile)...", self.spec.name)
        dummy_q = np.zeros(self.num_joints, dtype=np.float32)
        left_ee, right_ee = self.fk(dummy_q)
        try:
            self.ik(
                q_current=dummy_q,
                left_pose=(
                    np.asarray(left_ee.translation(), dtype=np.float32),
                    np.asarray(left_ee.rotation().as_matrix(), dtype=np.float32),
                ),
                right_pose=(
                    np.asarray(right_ee.translation(), dtype=np.float32),
                    np.asarray(right_ee.rotation().as_matrix(), dtype=np.float32),
                ),
            )
        except Exception:
            _logger.exception("%s EE IK warmup failed.", self.spec.name)
        _logger.info("%s EE IK solver ready.", self.spec.name)


def make_kinematics_solver(spec: RobotKinematicsSpec) -> type[BimanualPyrokiSolver]:
    """Return a ``KinematicsSolver`` class pre-bound to *spec*.

    The returned class accepts only ``config`` at construction time, keeping
    call sites robot-agnostic::

        solver = KinematicsSolver(config=config)

    Elbow link indices (``l_elbow_idx``, ``r_elbow_idx``) are resolved from
    ``spec.left_elbow_link`` / ``spec.right_elbow_link`` when present, and set
    to ``-1`` otherwise.  These attributes are used by visualization helpers
    that need to draw elbow positions.
    """

    class _KinematicsSolver(BimanualPyrokiSolver):
        def __init__(self, config: KinematicsConfig = KinematicsConfig()) -> None:
            super().__init__(spec=spec, config=config)
            names = self.robot.links.names
            self.l_elbow_idx: int = (
                names.index(spec.left_elbow_link) if spec.left_elbow_link else -1
            )
            self.r_elbow_idx: int = (
                names.index(spec.right_elbow_link) if spec.right_elbow_link else -1
            )

    _KinematicsSolver.__name__ = f"{spec.name}KinematicsSolver"
    _KinematicsSolver.__qualname__ = f"{spec.name}KinematicsSolver"
    return _KinematicsSolver
