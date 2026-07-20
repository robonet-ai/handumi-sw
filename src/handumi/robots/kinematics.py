"""Small pyroki bimanual IK wrapper driven by robot YAML configs."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk


@dataclass(frozen=True)
class KinematicsConfig:
    """Position-dominant IK weights."""

    pos_weight: float = 100.0
    ori_weight: float = 15.0
    rest_weight: float = 2.0
    posture_weight: float = 0.0
    manipulability_weight: float = 0.0
    max_joint_delta: float | None = None
    max_reach: float | None = None


def limit_joint_delta(
    q_current: np.ndarray,
    q_target: np.ndarray,
    max_delta: float | None,
) -> np.ndarray:
    """Limit each joint's change while preserving the solver's direction."""
    current = np.asarray(q_current, dtype=np.float32)
    target = np.asarray(q_target, dtype=np.float32)
    if max_delta is None:
        return target
    if max_delta <= 0.0:
        raise ValueError("max_joint_delta must be > 0")
    delta = np.clip(target - current, -float(max_delta), float(max_delta))
    return (current + delta).astype(np.float32)


@jdc.jit
def _solve(
    robot,
    ee_indices,
    tgt_pos,
    tgt_wxyz,
    q_prev,
    pos_weight,
    ori_weight,
    rest_weight,
):
    JointVar = robot.joint_var_cls
    target_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(tgt_wxyz), tgt_pos
    )
    batch = target_pose.get_batch_axes()
    costs = [
        pk.costs.pose_cost_analytic_jac(
            jax.tree.map(lambda x: x[None], robot),
            JointVar(jnp.full(batch, 0)),
            target_pose,
            ee_indices,
            pos_weight=pos_weight,
            ori_weight=ori_weight,
        ),
        pk.costs.rest_cost(
            JointVar(0),
            rest_pose=q_prev,
            weight=rest_weight,
        ),
        pk.costs.limit_constraint(robot, JointVar(0)),
    ]
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[JointVar(0)])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=10.0),
        )
    )
    return sol[JointVar(0)]


def solve_bimanual(
    robot: pk.Robot,
    ee_indices,
    tgt_pos,
    tgt_wxyz,
    q_prev=None,
    pos_weight=100.0,
    ori_weight=15.0,
    rest_weight=2.0,
) -> np.ndarray:
    """Solve two end-effector targets and return the full actuated config."""
    nq = robot.joints.num_actuated_joints
    if q_prev is None:
        q_prev = np.zeros(nq, dtype=np.float32)
    cfg = _solve(
        robot,
        jnp.array(ee_indices),
        jnp.array(tgt_pos),
        jnp.array(tgt_wxyz),
        jnp.array(q_prev),
        pos_weight,
        ori_weight,
        rest_weight,
    )
    return np.array(cfg, dtype=np.float32)


class BimanualKinematicsSolver:
    """Compatibility wrapper around :func:`solve_bimanual`."""

    def __init__(
        self,
        *,
        robot: pk.Robot,
        ee_indices: tuple[int, int],
        arm_joint_indices: dict[str, list[int]] | None = None,
        home_q: np.ndarray,
        config: KinematicsConfig,
        locked_joint_indices: tuple[int, ...] = (),
    ) -> None:
        self.robot = robot
        self.ee_indices = ee_indices
        self.home_q = np.asarray(home_q, dtype=np.float32)
        self.config = config
        self.locked_joint_indices = tuple(locked_joint_indices)
        self.l_ee_idx, self.r_ee_idx = ee_indices
        arm_joint_indices = arm_joint_indices or {}
        self.left_indices = list(
            arm_joint_indices.get("left") or _side_indices(robot, "left")
        )
        self.right_indices = list(
            arm_joint_indices.get("right") or _side_indices(robot, "right")
        )
        self.left_joint_indices = self.left_indices
        self.right_joint_indices = self.right_indices
        self.l_elbow_idx = -1
        self.r_elbow_idx = -1

    @property
    def num_joints(self) -> int:
        return self.robot.joints.num_actuated_joints

    @property
    def joint_names(self) -> list[str]:
        return list(self.robot.joints.actuated_names)

    def set_posture_pose(self, q: np.ndarray) -> None:
        self.home_q = np.asarray(q, dtype=np.float32)

    def _with_locked_joints(self, q: np.ndarray) -> np.ndarray:
        if not self.locked_joint_indices:
            return np.asarray(q, dtype=np.float32)
        out = np.asarray(q, dtype=np.float32).copy()
        for index in self.locked_joint_indices:
            out[index] = self.home_q[index]
        return out

    def fk(self, q: np.ndarray) -> tuple[jaxlie.SE3, jaxlie.SE3]:
        fk = self.robot.forward_kinematics(jnp.asarray(q, dtype=jnp.float32))
        return jaxlie.SE3(fk[self.l_ee_idx]), jaxlie.SE3(fk[self.r_ee_idx])

    def fk_pose7(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return left/right FK as ``[x, y, z, qx, qy, qz, qw]`` poses."""
        left, right = self.fk(q)
        return se3_to_pose7(left), se3_to_pose7(right)

    def link_positions(self, q: np.ndarray, link_indices: list[int]) -> np.ndarray:
        fk = self.robot.forward_kinematics(jnp.asarray(q, dtype=jnp.float32))
        return np.asarray(
            [jaxlie.SE3(fk[index]).translation() for index in link_indices],
            dtype=np.float32,
        )

    def ik(
        self,
        q_current: np.ndarray,
        left_pose: tuple[np.ndarray, np.ndarray] | None = None,
        right_pose: tuple[np.ndarray, np.ndarray] | None = None,
        left_elbow_pos: np.ndarray | None = None,
        right_elbow_pos: np.ndarray | None = None,
    ) -> np.ndarray:
        del left_elbow_pos, right_elbow_pos
        if left_pose is None and right_pose is None:
            return self._with_locked_joints(q_current)

        q_prev = self._with_locked_joints(q_current)
        left_fk, right_fk = self.fk(q_prev)
        tgt_pos = []
        tgt_wxyz = []
        for pose, fallback in ((left_pose, left_fk), (right_pose, right_fk)):
            if pose is None:
                tgt_pos.append(np.asarray(fallback.translation(), dtype=np.float32))
                tgt_wxyz.append(np.asarray(fallback.rotation().wxyz, dtype=np.float32))
            else:
                pos, rot = pose
                tgt_pos.append(np.asarray(pos, dtype=np.float32))
                tgt_wxyz.append(
                    _pose_rotation_to_wxyz(np.asarray(rot, dtype=np.float32))
                )

        q_target = solve_bimanual(
            self.robot,
            self.ee_indices,
            np.asarray(tgt_pos, dtype=np.float32),
            np.asarray(tgt_wxyz, dtype=np.float32),
            q_prev=q_prev,
            pos_weight=self.config.pos_weight,
            ori_weight=self.config.ori_weight,
            rest_weight=self.config.rest_weight,
        )
        q_limited = limit_joint_delta(q_prev, q_target, self.config.max_joint_delta)
        return self._with_locked_joints(q_limited)


def _side_indices(robot: pk.Robot, side: str) -> list[int]:
    return [
        i
        for i, name in enumerate(robot.joints.actuated_names)
        if name.startswith(f"{side}_")
    ]


def se3_to_pose7(transform: jaxlie.SE3) -> np.ndarray:
    """Convert a JAXLie SE3 to ``[x, y, z, qx, qy, qz, qw]``."""
    translation = np.asarray(transform.translation(), dtype=np.float32)
    wxyz = np.asarray(transform.rotation().wxyz, dtype=np.float32)
    quat = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    else:
        quat = (quat / norm).astype(np.float32)
    return np.concatenate([translation, quat]).astype(np.float32)


def rotation_error_deg(target_pose7: np.ndarray, achieved_pose7: np.ndarray) -> np.ndarray:
    """Shortest quaternion angular distance between pose7 arrays, in degrees."""
    target_quat = np.asarray(target_pose7, dtype=np.float32)[..., 3:7]
    achieved_quat = np.asarray(achieved_pose7, dtype=np.float32)[..., 3:7]
    target_quat = target_quat / np.maximum(
        np.linalg.norm(target_quat, axis=-1, keepdims=True),
        1e-8,
    )
    achieved_quat = achieved_quat / np.maximum(
        np.linalg.norm(achieved_quat, axis=-1, keepdims=True),
        1e-8,
    )
    dot = np.abs(np.sum(target_quat * achieved_quat, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0))).astype(np.float32)


def pose_error_arrays(
    target_left_pose7: np.ndarray,
    target_right_pose7: np.ndarray,
    achieved_left_pose7: np.ndarray,
    achieved_right_pose7: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return per-frame EE position and orientation errors for both arms."""
    target_left_pose7 = np.asarray(target_left_pose7, dtype=np.float32)
    target_right_pose7 = np.asarray(target_right_pose7, dtype=np.float32)
    achieved_left_pose7 = np.asarray(achieved_left_pose7, dtype=np.float32)
    achieved_right_pose7 = np.asarray(achieved_right_pose7, dtype=np.float32)
    return {
        "left_pos_error_m": np.linalg.norm(
            target_left_pose7[:, :3] - achieved_left_pose7[:, :3], axis=1
        ).astype(np.float32),
        "right_pos_error_m": np.linalg.norm(
            target_right_pose7[:, :3] - achieved_right_pose7[:, :3], axis=1
        ).astype(np.float32),
        "left_rot_error_deg": rotation_error_deg(target_left_pose7, achieved_left_pose7),
        "right_rot_error_deg": rotation_error_deg(target_right_pose7, achieved_right_pose7),
    }


def optimization_score_from_errors(
    pos_mean_cm: float,
    pos_max_cm: float,
    rot_mean_deg: float,
    rot_max_deg: float,
) -> float:
    """Single scalar useful for comparing IK weight sweeps."""
    return float(
        pos_mean_cm
        + 0.35 * pos_max_cm
        + 0.25 * rot_mean_deg
        + 0.08 * rot_max_deg
    )


def _pose_rotation_to_wxyz(rot: np.ndarray) -> np.ndarray:
    if rot.shape == (3, 3):
        return _rot_3x3_to_wxyz(rot)
    if rot.shape == (4,):
        norm = float(np.linalg.norm(rot))
        if norm < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        xyzw = (rot / norm).astype(np.float32)
        return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)
    raise ValueError(
        f"Expected rotation as 3x3 matrix or xyzw quaternion, got {rot.shape}."
    )


def _rot_3x3_to_wxyz(rot: np.ndarray) -> np.ndarray:
    t = float(rot[0, 0] + rot[1, 1] + rot[2, 2])
    if t > 0.0:
        r = np.sqrt(t + 1.0)
        s = 0.5 / r
        return np.array(
            [
                0.5 * r,
                (rot[2, 1] - rot[1, 2]) * s,
                (rot[0, 2] - rot[2, 0]) * s,
                (rot[1, 0] - rot[0, 1]) * s,
            ],
            dtype=np.float32,
        )
    if rot[0, 0] >= rot[1, 1] and rot[0, 0] >= rot[2, 2]:
        r = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        s = 0.5 / r
        return np.array(
            [
                (rot[2, 1] - rot[1, 2]) * s,
                0.5 * r,
                (rot[0, 1] + rot[1, 0]) * s,
                (rot[0, 2] + rot[2, 0]) * s,
            ],
            dtype=np.float32,
        )
    if rot[1, 1] >= rot[2, 2]:
        r = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        s = 0.5 / r
        return np.array(
            [
                (rot[0, 2] - rot[2, 0]) * s,
                (rot[0, 1] + rot[1, 0]) * s,
                0.5 * r,
                (rot[1, 2] + rot[2, 1]) * s,
            ],
            dtype=np.float32,
        )
    r = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
    s = 0.5 / r
    return np.array(
        [
            (rot[1, 0] - rot[0, 1]) * s,
            (rot[0, 2] + rot[2, 0]) * s,
            (rot[1, 2] + rot[2, 1]) * s,
            0.5 * r,
        ],
        dtype=np.float32,
    )
