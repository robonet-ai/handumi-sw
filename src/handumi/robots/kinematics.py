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
        home_q: np.ndarray,
        config: KinematicsConfig,
    ) -> None:
        self.robot = robot
        self.ee_indices = ee_indices
        self.home_q = np.asarray(home_q, dtype=np.float32)
        self.config = config
        self.l_ee_idx, self.r_ee_idx = ee_indices
        self.left_indices = _side_indices(robot, "left")
        self.right_indices = _side_indices(robot, "right")
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

    def fk(self, q: np.ndarray) -> tuple[jaxlie.SE3, jaxlie.SE3]:
        fk = self.robot.forward_kinematics(jnp.asarray(q, dtype=jnp.float32))
        return jaxlie.SE3(fk[self.l_ee_idx]), jaxlie.SE3(fk[self.r_ee_idx])

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
            return np.asarray(q_current, dtype=np.float32)

        q_prev = np.asarray(q_current, dtype=np.float32)
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
                tgt_wxyz.append(_rot_3x3_to_wxyz(np.asarray(rot, dtype=np.float32)))

        return solve_bimanual(
            self.robot,
            self.ee_indices,
            np.asarray(tgt_pos, dtype=np.float32),
            np.asarray(tgt_wxyz, dtype=np.float32),
            q_prev=q_prev,
            pos_weight=self.config.pos_weight,
            ori_weight=self.config.ori_weight,
            rest_weight=self.config.rest_weight,
        )


def _side_indices(robot: pk.Robot, side: str) -> list[int]:
    return [
        i
        for i, name in enumerate(robot.joints.actuated_names)
        if name.startswith(f"{side}_")
    ]


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
