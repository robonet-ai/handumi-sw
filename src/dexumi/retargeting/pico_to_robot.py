"""Generic PICO upper-body to bimanual robot end-effector retargeting."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from dexumi.retargeting.pico_upper_body import (
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    parse_axis_map,
)


@dataclass(frozen=True)
class ArmReference:
    """First-frame calibration for one arm, using only the wrist/EE target."""

    human_wrist_rel: np.ndarray
    robot_wrist_pos: np.ndarray
    robot_wrist_rot: np.ndarray


@dataclass(frozen=True)
class RetargetReferences:
    left: ArmReference
    right: ArmReference


@dataclass(frozen=True)
class RetargetingSpec:
    """Embodiment-specific command layout and rest pose details."""

    name: str
    rest_left_arm: np.ndarray
    rest_right_arm: np.ndarray
    command_size: int
    gripper_index: int
    left_front_wrist: Callable[[float, float, float], np.ndarray]
    right_front_wrist: Callable[[float, float, float], np.ndarray]

    @property
    def arm_joint_count(self) -> int:
        return int(len(self.rest_left_arm))


def make_rest_q(solver, spec: RetargetingSpec) -> np.ndarray:
    q = np.zeros(solver.num_joints, dtype=np.float32)
    q[solver.left_indices] = spec.rest_left_arm
    q[solver.right_indices] = spec.rest_right_arm
    return q


def robot_link_pose(
    solver,
    q: np.ndarray,
    link_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    return solver.link_pose(q, link_index)


def robot_link_positions(
    solver,
    q: np.ndarray,
    link_indices: list[int],
) -> np.ndarray:
    return solver.link_positions(q, link_indices)


def _body_position(body_pose: np.ndarray, joint_index: int) -> np.ndarray:
    return np.asarray(body_pose[joint_index, :3], dtype=np.float32)


def _human_wrist_reference(
    body_pose: np.ndarray,
    *,
    shoulder_index: int,
    wrist_index: int,
) -> np.ndarray:
    shoulder = _body_position(body_pose, shoulder_index)
    wrist = _body_position(body_pose, wrist_index)
    return wrist - shoulder


def calibrate_from_first_frame(
    solver,
    first_body_pose: np.ndarray,
    q_rest: np.ndarray,
) -> RetargetReferences:
    left_wrist_rel = _human_wrist_reference(
        first_body_pose,
        shoulder_index=LEFT_SHOULDER,
        wrist_index=LEFT_WRIST,
    )
    right_wrist_rel = _human_wrist_reference(
        first_body_pose,
        shoulder_index=RIGHT_SHOULDER,
        wrist_index=RIGHT_WRIST,
    )

    left_wrist_pos, left_wrist_rot = robot_link_pose(solver, q_rest, solver.l_ee_idx)
    right_wrist_pos, right_wrist_rot = robot_link_pose(solver, q_rest, solver.r_ee_idx)

    return RetargetReferences(
        left=ArmReference(
            human_wrist_rel=left_wrist_rel,
            robot_wrist_pos=left_wrist_pos,
            robot_wrist_rot=left_wrist_rot,
        ),
        right=ArmReference(
            human_wrist_rel=right_wrist_rel,
            robot_wrist_pos=right_wrist_pos,
            robot_wrist_rot=right_wrist_rot,
        ),
    )


class PicoToRobotArmRetargeter:
    """Relative-position retargeter from PICO wrists to robot end-effectors."""

    def __init__(
        self,
        *,
        solver,
        spec: RetargetingSpec,
        first_body_pose: np.ndarray,
        scale: float,
        axis_map: str,
        enable_left: bool = True,
        enable_right: bool = True,
        gripper: float = 1.0,
    ) -> None:
        self.solver = solver
        self.spec = spec
        self.scale = float(scale)
        self.transform = parse_axis_map(axis_map)
        self.enable_left = enable_left
        self.enable_right = enable_right
        self.gripper = float(gripper)

        self.q_rest = make_rest_q(solver, spec)
        self.solver.set_posture_pose(self.q_rest)
        self.refs = calibrate_from_first_frame(solver, first_body_pose, self.q_rest)

    def wrist_target(
        self,
        body_pose: np.ndarray,
        *,
        ref: ArmReference,
        shoulder_index: int,
        wrist_index: int,
    ) -> np.ndarray:
        wrist_rel = _human_wrist_reference(
            body_pose,
            shoulder_index=shoulder_index,
            wrist_index=wrist_index,
        )
        wrist_delta = self.transform(wrist_rel - ref.human_wrist_rel) * self.scale
        return ref.robot_wrist_pos + wrist_delta

    def _arm_targets(
        self,
        body_pose: np.ndarray,
        *,
        ref: ArmReference,
        shoulder_index: int,
        elbow_index: int | None = None,
        wrist_index: int,
    ) -> tuple[np.ndarray, None]:
        del elbow_index
        wrist = self.wrist_target(
            body_pose,
            ref=ref,
            shoulder_index=shoulder_index,
            wrist_index=wrist_index,
        )
        shoulder = (
            self.solver._left_shoulder_pos
            if ref is self.refs.left
            else self.solver._right_shoulder_pos
        )
        pseudo_elbow = shoulder + 0.55 * (wrist - shoulder)
        return wrist, pseudo_elbow.astype(np.float32)

    def target_poses(
        self,
        body_pose: np.ndarray,
    ) -> tuple[tuple[np.ndarray, np.ndarray] | None, tuple[np.ndarray, np.ndarray] | None]:
        left_pose = None
        if self.enable_left:
            left_wrist = self.wrist_target(
                body_pose,
                ref=self.refs.left,
                shoulder_index=LEFT_SHOULDER,
                wrist_index=LEFT_WRIST,
            )
            left_pose = (left_wrist, self.refs.left.robot_wrist_rot)

        right_pose = None
        if self.enable_right:
            right_wrist = self.wrist_target(
                body_pose,
                ref=self.refs.right,
                shoulder_index=RIGHT_SHOULDER,
                wrist_index=RIGHT_WRIST,
            )
            right_pose = (right_wrist, self.refs.right.robot_wrist_rot)

        return left_pose, right_pose

    def target_points(self, body_pose: np.ndarray) -> np.ndarray:
        left_pose, right_pose = self.target_poses(body_pose)
        left = (
            left_pose[0]
            if left_pose is not None
            else np.asarray(self.refs.left.robot_wrist_pos, dtype=np.float32)
        )
        right = (
            right_pose[0]
            if right_pose is not None
            else np.asarray(self.refs.right.robot_wrist_pos, dtype=np.float32)
        )
        return np.asarray(
            [
                self.solver._left_shoulder_pos,
                left,
                self.solver._right_shoulder_pos,
                right,
            ],
            dtype=np.float32,
        )

    def retarget_frame(self, body_pose: np.ndarray, q_current: np.ndarray) -> np.ndarray:
        left_pose, right_pose = self.target_poses(body_pose)
        return self.solver.ik(
            q_current=q_current,
            left_pose=left_pose,
            right_pose=right_pose,
        )

    def split_for_sim(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left = np.zeros(self.spec.command_size, dtype=np.float32)
        right = np.zeros(self.spec.command_size, dtype=np.float32)
        left[: self.spec.arm_joint_count] = q[self.solver.left_indices]
        right[: self.spec.arm_joint_count] = q[self.solver.right_indices]
        left[self.spec.gripper_index] = self.gripper
        right[self.spec.gripper_index] = self.gripper
        return left, right


def move_retargeter_to_front_workspace(
    retargeter: PicoToRobotArmRetargeter,
    *,
    wrist_forward: float,
    wrist_height: float,
    wrist_lateral: float,
    elbow_forward: float | None = None,
    elbow_height: float | None = None,
    elbow_lateral: float | None = None,
) -> None:
    """Set the calibrated wrist workspace; elbow arguments are ignored."""

    del elbow_forward, elbow_height, elbow_lateral
    left_wrist = retargeter.spec.left_front_wrist(
        wrist_forward,
        wrist_lateral,
        wrist_height,
    )
    right_wrist = retargeter.spec.right_front_wrist(
        wrist_forward,
        wrist_lateral,
        wrist_height,
    )
    retargeter.refs = replace(
        retargeter.refs,
        left=replace(retargeter.refs.left, robot_wrist_pos=left_wrist),
        right=replace(retargeter.refs.right, robot_wrist_pos=right_wrist),
    )


def settle_first_frame(
    retargeter: PicoToRobotArmRetargeter,
    first_body_pose: np.ndarray,
    iterations: int,
) -> np.ndarray:
    q = retargeter.q_rest.copy()
    for _ in range(max(0, iterations)):
        q = retargeter.retarget_frame(first_body_pose, q)
    return q
