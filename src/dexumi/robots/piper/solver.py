"""Standalone bimanual IK solver for the dual Piper URDF.

Uses pyroki + jaxls to solve for joint positions given absolute Cartesian
end-effector poses in the robot's world frame.
"""

from __future__ import annotations

import functools
import logging

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import yourdfpy

from .config import KinematicsConfig
from .shared import (
    URDF_PATH,
    Joint,
    urdf_arm_joint_names,
    urdf_body_name,
    urdf_revolute_joint_names,
)

_logger = logging.getLogger(__name__)


def _build_robot_collision(
    urdf: yourdfpy.URDF,
    robot: pk.Robot,
    *,
    home_margin: float,
) -> pk.collision.RobotCollision:
    """Build a collision model restricted to inter-arm and base-arm pairs."""

    link_names = [link.name for link in urdf.robot.links]

    def is_left_arm(name: str) -> bool:
        return name.startswith("izq_link")

    def is_right_arm(name: str) -> bool:
        return name.startswith("der_link")

    def is_left_base(name: str) -> bool:
        return name == "base_izquierdo"

    def is_right_base(name: str) -> bool:
        return name == "base_derecho"

    def keep_pair(a: str, b: str) -> bool:
        cross_arm = (is_left_arm(a) and is_right_arm(b)) or (
            is_right_arm(a) and is_left_arm(b)
        )
        opposite_base_arm = (
            (is_left_base(a) and is_right_arm(b))
            or (is_left_base(b) and is_right_arm(a))
            or (is_right_base(a) and is_left_arm(b))
            or (is_right_base(b) and is_left_arm(a))
        )
        return cross_arm or opposite_base_arm

    ignore: set[tuple[str, str]] = set()
    for i, a in enumerate(link_names):
        for b in link_names[i + 1 :]:
            if not keep_pair(a, b):
                ignore.add((a, b))

    rc = pk.collision.RobotCollision.from_urdf(urdf, user_ignore_pairs=tuple(ignore))

    q0 = jnp.zeros(robot.joints.num_actuated_joints)
    distances = np.asarray(rc.compute_self_collision_distance(robot, q0))
    active_i = np.asarray(rc.active_idx_i)
    active_j = np.asarray(rc.active_idx_j)
    for index in np.where(distances < home_margin)[0]:
        ignore.add((rc.link_names[active_i[index]], rc.link_names[active_j[index]]))

    rc = pk.collision.RobotCollision.from_urdf(urdf, user_ignore_pairs=tuple(ignore))
    _logger.info(
        "RobotCollision: restricted to %d Piper safety pairs.",
        len(rc.active_idx_i),
    )
    return rc


_LEFT_EE = urdf_body_name(Joint.JOINT_6, is_left=True)
_RIGHT_EE = urdf_body_name(Joint.JOINT_6, is_left=False)
_LEFT_ELBOW = urdf_body_name(Joint.JOINT_3, is_left=True)
_RIGHT_ELBOW = urdf_body_name(Joint.JOINT_3, is_left=False)
_LEFT_SHOULDER = urdf_body_name(Joint.JOINT_1, is_left=True)
_RIGHT_SHOULDER = urdf_body_name(Joint.JOINT_1, is_left=False)

_LEFT_REVOLUTE_JOINT_NAMES = urdf_revolute_joint_names(is_left=True)
_RIGHT_REVOLUTE_JOINT_NAMES = urdf_revolute_joint_names(is_left=False)
_LEFT_JOINT_NAMES = urdf_arm_joint_names(is_left=True)
_RIGHT_JOINT_NAMES = urdf_arm_joint_names(is_left=False)


@functools.partial(jax.jit, static_argnames=("max_iterations",))
def _solve_ik(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    target_L: jaxlie.SE3 | None,
    target_R: jaxlie.SE3 | None,
    L_ee_idx: jax.Array,
    R_ee_idx: jax.Array,
    elbow_L: jaxlie.SE3 | None,
    elbow_R: jaxlie.SE3 | None,
    L_elbow_idx: jax.Array,
    R_elbow_idx: jax.Array,
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
    elbow_weight: float,
    max_iterations: int,
    cost_tolerance: float,
) -> jax.Array:
    JointVar = robot.joint_var_cls

    costs = [
        pk.costs.rest_cost(JointVar(0), rest_pose=q_current, weight=rest_weight),
        pk.costs.rest_cost(JointVar(0), rest_pose=posture_pose, weight=posture_weight),
        pk.costs.manipulability_cost(
            robot,
            JointVar(0),
            jnp.array([L_ee_idx, R_ee_idx], dtype=jnp.int32),
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

    if elbow_L is not None:
        costs.append(
            pk.costs.pose_cost_analytic_jac(
                robot,
                JointVar(0),
                elbow_L,
                jnp.array(L_elbow_idx, dtype=jnp.int32),
                pos_weight=elbow_weight,
                ori_weight=0.0,
            )
        )

    if elbow_R is not None:
        costs.append(
            pk.costs.pose_cost_analytic_jac(
                robot,
                JointVar(0),
                elbow_R,
                jnp.array(R_elbow_idx, dtype=jnp.int32),
                pos_weight=elbow_weight,
                ori_weight=0.0,
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
    analyzed = problem.analyze()
    solution_vals = analyzed.solve(
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


def _clamp_reach(pos: np.ndarray, center: np.ndarray, max_reach: float) -> np.ndarray:
    """Clamp EE target position to within max_reach of center (shoulder position)."""
    d = pos - center
    dist = np.linalg.norm(d)
    if dist > max_reach:
        return (center + d * (max_reach / dist)).astype(np.float32)
    return pos


def _rot_3x3_to_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix → unit quaternion (w, x, y, z), float32.

    Pure NumPy (Shepperd method) — avoids JAX dispatch overhead outside JIT.
    """
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
    """Construct SE3 from numpy pos + rot_3x3 at the JAX boundary."""
    return jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(wxyz=jnp.asarray(_rot_3x3_to_wxyz(rot_3x3))),
        jnp.asarray(pos, dtype=jnp.float32),
    )


def _pos3_to_se3(pos: np.ndarray) -> jaxlie.SE3:
    """Convert a (3,) position array to an identity-rotation SE3."""
    return jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(wxyz=jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)),
        jnp.asarray(pos, dtype=jnp.float32),
    )


class KinematicsSolver:
    """Bimanual IK solver for the dual Piper URDF.

    Loads the bundled URDF, builds a pyroki + jaxls solver, and resolves
    absolute Cartesian end-effector poses (world frame) to joint angles.
    JIT compilation is triggered during ``__init__`` so the first call to
    :meth:`ik` is fast.

    Args:
        config: Solver cost weights and parameters.

    Example::

        solver = KinematicsSolver()
        q = np.zeros(solver.num_joints, dtype=np.float32)
        pos = np.array([0.3, 0.2, 0.4], dtype=np.float32)
        rot = np.eye(3, dtype=np.float32)
        q = solver.ik(q, left_pose=(pos, rot))
    """

    def __init__(self, config: KinematicsConfig = KinematicsConfig()) -> None:
        """Load the bundled Piper URDF, build the pyroki robot and collision model, and warm up JIT.

        Args:
            config: Cost weights and solver parameters.
        """
        self.config = config

        _logger.info("Loading Piper URDF...")
        urdf = yourdfpy.URDF.load(str(URDF_PATH), mesh_dir=str(URDF_PATH.parent))
        self.robot = pk.Robot.from_urdf(urdf)
        self.robot_coll = _build_robot_collision(
            urdf,
            self.robot,
            home_margin=config.self_collision_margin,
        )

        names = self.robot.links.names
        self.l_ee_idx = names.index(_LEFT_EE)
        self.r_ee_idx = names.index(_RIGHT_EE)
        self.l_elbow_idx = names.index(_LEFT_ELBOW)
        self.r_elbow_idx = names.index(_RIGHT_ELBOW)

        self._l_ee_idx_jax = jnp.asarray(self.l_ee_idx, dtype=jnp.int32)
        self._r_ee_idx_jax = jnp.asarray(self.r_ee_idx, dtype=jnp.int32)
        self._l_elbow_idx_jax = jnp.asarray(self.l_elbow_idx, dtype=jnp.int32)
        self._r_elbow_idx_jax = jnp.asarray(self.r_elbow_idx, dtype=jnp.int32)

        L_sh_idx = names.index(_LEFT_SHOULDER)
        R_sh_idx = names.index(_RIGHT_SHOULDER)
        fk0 = self.robot.forward_kinematics(
            jnp.zeros(self.robot.joints.num_actuated_joints)
        )
        self._left_shoulder_pos = np.asarray(
            jaxlie.SE3(fk0[L_sh_idx]).translation(), dtype=np.float32
        )
        self._right_shoulder_pos = np.asarray(
            jaxlie.SE3(fk0[R_sh_idx]).translation(), dtype=np.float32
        )

        actuated = list(self.robot.joints.actuated_names)
        name_to_idx = {name: i for i, name in enumerate(actuated)}
        self.left_indices = [name_to_idx[name] for name in _LEFT_REVOLUTE_JOINT_NAMES]
        self.right_indices = [name_to_idx[name] for name in _RIGHT_REVOLUTE_JOINT_NAMES]
        self.left_joint_indices = [name_to_idx[name] for name in _LEFT_JOINT_NAMES]
        self.right_joint_indices = [name_to_idx[name] for name in _RIGHT_JOINT_NAMES]

        self._posture_pose = jnp.zeros(
            self.robot.joints.num_actuated_joints, dtype=jnp.float32
        )

        self._warmup()

    def set_posture_pose(self, q: np.ndarray) -> None:
        """Set the global preferred posture used as a persistent attractor.

        Args:
            q: Full ``(N,)`` joint array in radians (same ordering as :meth:`ik`).
        """
        self._posture_pose = jnp.asarray(q, dtype=jnp.float32)

    @property
    def joint_names(self) -> list[str]:
        """Ordered list of all actuated joint names (left arm then right arm)."""
        return list(self.robot.joints.actuated_names)

    @property
    def num_joints(self) -> int:
        """Total number of actuated joints across both arms."""
        return self.robot.joints.num_actuated_joints

    def fk(self, q: np.ndarray) -> tuple[jaxlie.SE3, jaxlie.SE3]:
        """Compute end-effector poses from joint positions.

        Args:
            q: Full ``(N,)`` joint array in radians.

        Returns:
            Tuple ``(left_pose, right_pose)`` as :class:`jaxlie.SE3` transforms.
        """
        fk = self.robot.forward_kinematics(jnp.asarray(q, dtype=jnp.float32))
        return jaxlie.SE3(fk[self.l_ee_idx]), jaxlie.SE3(fk[self.r_ee_idx])

    def ik(
        self,
        q_current: np.ndarray,
        left_pose: tuple[np.ndarray, np.ndarray] | None = None,
        right_pose: tuple[np.ndarray, np.ndarray] | None = None,
        left_elbow_pos: np.ndarray | None = None,
        right_elbow_pos: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute joint positions for absolute Cartesian end-effector targets.

        All positions and orientations must be expressed in the robot's world frame.
        End-effector targets are clamped to ``config.max_reach`` from each shoulder
        before solving, and joint changes are clamped to ``config.max_joint_delta``
        per call.

        Args:
            q_current: Full ``(N,)`` joint array in radians used as the solver
                seed and rest-cost target.
            left_pose: ``(pos, rot_3x3)`` numpy tuple for the left end-effector,
                or ``None`` to skip the left arm.
            right_pose: Same as ``left_pose`` for the right end-effector.
            left_elbow_pos: ``(3,)`` optional left elbow position hint in world frame.
            right_elbow_pos: ``(3,)`` optional right elbow position hint in world frame.

        Returns:
            Updated full ``(N,)`` joint array in radians.
        """
        if left_pose is None and right_pose is None:
            return q_current

        cfg = self.config
        q_current = np.asarray(q_current, dtype=np.float32)

        target_L: jaxlie.SE3 | None = None
        if left_pose is not None:
            lp, lr = left_pose
            lp = _clamp_reach(
                np.asarray(lp, dtype=np.float32), self._left_shoulder_pos, cfg.max_reach
            )
            target_L = _np_to_se3(lp, np.asarray(lr, dtype=np.float32))

        target_R: jaxlie.SE3 | None = None
        if right_pose is not None:
            rp, rr = right_pose
            rp = _clamp_reach(
                np.asarray(rp, dtype=np.float32),
                self._right_shoulder_pos,
                cfg.max_reach,
            )
            target_R = _np_to_se3(rp, np.asarray(rr, dtype=np.float32))

        elbow_L = (
            _pos3_to_se3(np.asarray(left_elbow_pos, dtype=np.float32))
            if left_elbow_pos is not None
            else None
        )
        elbow_R = (
            _pos3_to_se3(np.asarray(right_elbow_pos, dtype=np.float32))
            if right_elbow_pos is not None
            else None
        )

        q_result = _solve_ik(
            self.robot,
            self.robot_coll,
            target_L,
            target_R,
            self._l_ee_idx_jax,
            self._r_ee_idx_jax,
            elbow_L,
            elbow_R,
            self._l_elbow_idx_jax,
            self._r_elbow_idx_jax,
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
            cfg.elbow_weight,
            cfg.max_iterations,
            cfg.cost_tolerance,
        )
        q_result_np = np.asarray(q_result, dtype=np.float32)
        delta = np.clip(
            q_result_np - q_current, -cfg.max_joint_delta, cfg.max_joint_delta
        )
        return (q_current + delta).astype(np.float32)

    def _warmup(self) -> None:
        """Trigger JIT compilation with a dummy solve."""
        _logger.info("Warming up Piper IK solver (JIT compile)...")
        dummy_q = np.zeros(self.num_joints, dtype=np.float32)
        fk = self.robot.forward_kinematics(jnp.asarray(dummy_q, dtype=jnp.float32))
        left_ee = jaxlie.SE3(fk[self.l_ee_idx])
        right_ee = jaxlie.SE3(fk[self.r_ee_idx])
        kwargs: dict = dict(
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
        if self.config.elbow_weight > 0:
            kwargs["left_elbow_pos"] = np.asarray(
                jaxlie.SE3(fk[self.l_elbow_idx]).translation(), dtype=np.float32
            )
            kwargs["right_elbow_pos"] = np.asarray(
                jaxlie.SE3(fk[self.r_elbow_idx]).translation(), dtype=np.float32
            )
        try:
            self.ik(**kwargs)
        except Exception:
            _logger.exception("Piper IK warmup failed.")
        _logger.info("Piper IK solver ready.")
