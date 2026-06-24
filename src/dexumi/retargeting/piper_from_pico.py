"""Retarget PICO upper-body poses to the Piper IK solver."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from dexumi.retargeting.pico_upper_body import (
    LEFT_ELBOW,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ELBOW,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    parse_axis_map,
)
from dexumi.robots.piper.shared import (
    COMMAND_SIZE,
    GRIPPER_INDEX,
    gripper_to_finger_positions,
)
from dexumi.robots.piper.solver import KinematicsSolver


REST_LEFT_ARM = np.array([0.0, 1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)
REST_RIGHT_ARM = np.array([0.0, 1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)


@dataclass(frozen=True)
class ArmReference:
    """First-frame calibration data for one arm."""

    human_wrist_rel: np.ndarray
    human_elbow_rel: np.ndarray
    robot_wrist_pos: np.ndarray
    robot_elbow_pos: np.ndarray
    robot_wrist_rot: np.ndarray


@dataclass(frozen=True)
class RetargetReferences:
    """Left/right first-frame calibration data."""

    left: ArmReference
    right: ArmReference


def _set_gripper_q(
    solver: KinematicsSolver,
    q: np.ndarray,
    *,
    left: float,
    right: float,
) -> None:
    left_a, left_b = gripper_to_finger_positions(left)
    right_a, right_b = gripper_to_finger_positions(right)
    q[solver.left_joint_indices[6:8]] = np.array([left_a, left_b], dtype=np.float32)
    q[solver.right_joint_indices[6:8]] = np.array([right_a, right_b], dtype=np.float32)


def make_rest_q(
    solver: KinematicsSolver,
    *,
    gripper: float = 1.0,
) -> np.ndarray:
    """Create the full Piper joint vector for the visual rest pose."""

    q = np.zeros(solver.num_joints, dtype=np.float32)
    q[solver.left_indices] = REST_LEFT_ARM
    q[solver.right_indices] = REST_RIGHT_ARM
    _set_gripper_q(solver, q, left=gripper, right=gripper)
    return q


def _body_position(body_pose: np.ndarray, joint_index: int) -> np.ndarray:
    return np.asarray(body_pose[joint_index, :3], dtype=np.float32)


def _human_arm_reference(
    body_pose: np.ndarray,
    *,
    shoulder_index: int,
    elbow_index: int,
    wrist_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    shoulder = _body_position(body_pose, shoulder_index)
    elbow = _body_position(body_pose, elbow_index)
    wrist = _body_position(body_pose, wrist_index)
    return wrist - shoulder, elbow - shoulder


def robot_link_pose(
    solver: KinematicsSolver,
    q: np.ndarray,
    link_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(position, rotation_matrix)`` for one Piper link."""

    import jax.numpy as jnp
    import jaxlie

    fk = solver.robot.forward_kinematics(jnp.asarray(q, dtype=jnp.float32))
    pose = jaxlie.SE3(fk[link_index])
    return (
        np.asarray(pose.translation(), dtype=np.float32),
        np.asarray(pose.rotation().as_matrix(), dtype=np.float32),
    )


def piper_link_positions(
    solver: KinematicsSolver,
    q: np.ndarray,
    link_indices: list[int],
) -> np.ndarray:
    """Return link positions for drawing the Piper skeleton."""

    import jax.numpy as jnp
    import jaxlie

    fk = solver.robot.forward_kinematics(jnp.asarray(q, dtype=jnp.float32))
    return np.asarray(
        [jaxlie.SE3(fk[index]).translation() for index in link_indices],
        dtype=np.float32,
    )


def calibrate_from_first_frame(
    solver: KinematicsSolver,
    first_body_pose: np.ndarray,
    q_rest: np.ndarray,
) -> RetargetReferences:
    """Use the first human frame and Piper rest pose as the shared zero motion."""

    left_wrist_rel, left_elbow_rel = _human_arm_reference(
        first_body_pose,
        shoulder_index=LEFT_SHOULDER,
        elbow_index=LEFT_ELBOW,
        wrist_index=LEFT_WRIST,
    )
    right_wrist_rel, right_elbow_rel = _human_arm_reference(
        first_body_pose,
        shoulder_index=RIGHT_SHOULDER,
        elbow_index=RIGHT_ELBOW,
        wrist_index=RIGHT_WRIST,
    )

    left_wrist_pos, left_wrist_rot = robot_link_pose(solver, q_rest, solver.l_ee_idx)
    right_wrist_pos, right_wrist_rot = robot_link_pose(solver, q_rest, solver.r_ee_idx)
    left_elbow_pos, _ = robot_link_pose(solver, q_rest, solver.l_elbow_idx)
    right_elbow_pos, _ = robot_link_pose(solver, q_rest, solver.r_elbow_idx)

    return RetargetReferences(
        left=ArmReference(
            human_wrist_rel=left_wrist_rel,
            human_elbow_rel=left_elbow_rel,
            robot_wrist_pos=left_wrist_pos,
            robot_elbow_pos=left_elbow_pos,
            robot_wrist_rot=left_wrist_rot,
        ),
        right=ArmReference(
            human_wrist_rel=right_wrist_rel,
            human_elbow_rel=right_elbow_rel,
            robot_wrist_pos=right_wrist_pos,
            robot_elbow_pos=right_elbow_pos,
            robot_wrist_rot=right_wrist_rot,
        ),
    )


class PicoToPiperArmRetargeter:
    """Relative-position retargeter from PICO upper body to Piper arms."""

    def __init__(
        self,
        *,
        solver: KinematicsSolver,
        first_body_pose: np.ndarray,
        scale: float,
        axis_map: str,
        enable_left: bool = True,
        enable_right: bool = True,
        gripper: float = 1.0,
    ) -> None:
        self.solver = solver
        self.scale = float(scale)
        self.transform = parse_axis_map(axis_map)
        self.enable_left = enable_left
        self.enable_right = enable_right
        self.gripper = float(gripper)

        self.q_rest = make_rest_q(solver, gripper=gripper)
        self.solver.set_posture_pose(self.q_rest)
        self.refs = calibrate_from_first_frame(solver, first_body_pose, self.q_rest)

    def _arm_targets(
        self,
        body_pose: np.ndarray,
        *,
        ref: ArmReference,
        shoulder_index: int,
        elbow_index: int,
        wrist_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        wrist_rel, elbow_rel = _human_arm_reference(
            body_pose,
            shoulder_index=shoulder_index,
            elbow_index=elbow_index,
            wrist_index=wrist_index,
        )
        wrist_delta = self.transform(wrist_rel - ref.human_wrist_rel) * self.scale
        elbow_delta = self.transform(elbow_rel - ref.human_elbow_rel) * self.scale
        return ref.robot_wrist_pos + wrist_delta, ref.robot_elbow_pos + elbow_delta

    def retarget_frame(
        self,
        body_pose: np.ndarray,
        q_current: np.ndarray,
    ) -> np.ndarray:
        """Solve one frame of PICO upper-body data into a Piper joint vector."""

        left_pose = None
        left_elbow_pos = None
        if self.enable_left:
            left_wrist_pos, left_elbow_pos = self._arm_targets(
                body_pose,
                ref=self.refs.left,
                shoulder_index=LEFT_SHOULDER,
                elbow_index=LEFT_ELBOW,
                wrist_index=LEFT_WRIST,
            )
            left_pose = (left_wrist_pos, self.refs.left.robot_wrist_rot)

        right_pose = None
        right_elbow_pos = None
        if self.enable_right:
            right_wrist_pos, right_elbow_pos = self._arm_targets(
                body_pose,
                ref=self.refs.right,
                shoulder_index=RIGHT_SHOULDER,
                elbow_index=RIGHT_ELBOW,
                wrist_index=RIGHT_WRIST,
            )
            right_pose = (right_wrist_pos, self.refs.right.robot_wrist_rot)

        q_out = self.solver.ik(
            q_current=q_current,
            left_pose=left_pose,
            right_pose=right_pose,
            left_elbow_pos=left_elbow_pos,
            right_elbow_pos=right_elbow_pos,
        )
        _set_gripper_q(self.solver, q_out, left=self.gripper, right=self.gripper)
        return q_out

    def split_for_sim(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert the full IK vector into left/right Piper command arrays."""

        left = np.zeros(COMMAND_SIZE, dtype=np.float32)
        right = np.zeros(COMMAND_SIZE, dtype=np.float32)
        left[:6] = q[self.solver.left_indices]
        right[:6] = q[self.solver.right_indices]
        left[GRIPPER_INDEX] = self.gripper
        right[GRIPPER_INDEX] = self.gripper
        return left, right


def move_retargeter_to_front_workspace(
    retargeter: PicoToPiperArmRetargeter,
    *,
    wrist_forward: float,
    wrist_height: float,
    wrist_lateral: float,
    elbow_forward: float,
    elbow_height: float,
    elbow_lateral: float,
) -> None:
    """Use a chest/front Piper workspace instead of the raw URDF rest pose."""

    left_wrist = np.array(
        [wrist_forward, wrist_lateral, wrist_height],
        dtype=np.float32,
    )
    right_wrist = np.array(
        [wrist_forward, -wrist_lateral, wrist_height], dtype=np.float32
    )
    left_elbow = np.array(
        [elbow_forward, elbow_lateral, elbow_height],
        dtype=np.float32,
    )
    right_elbow = np.array(
        [elbow_forward, -elbow_lateral, elbow_height], dtype=np.float32
    )

    retargeter.refs = replace(
        retargeter.refs,
        left=replace(
            retargeter.refs.left,
            robot_wrist_pos=left_wrist,
            robot_elbow_pos=left_elbow,
        ),
        right=replace(
            retargeter.refs.right,
            robot_wrist_pos=right_wrist,
            robot_elbow_pos=right_elbow,
        ),
    )


def settle_first_frame(
    retargeter: PicoToPiperArmRetargeter,
    first_body_pose: np.ndarray,
    iterations: int,
) -> np.ndarray:
    """Run IK on the first frame so playback starts from the chosen workspace."""

    q = retargeter.q_rest.copy()
    for _ in range(max(0, iterations)):
        q = retargeter.retarget_frame(first_body_pose, q)
    return q
