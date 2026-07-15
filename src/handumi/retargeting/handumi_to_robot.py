"""Helpers for converting HandUMI raw state vectors into pose targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from handumi.dataset.raw import (
    HANDUMI_RAW_STATE_SIZE,
    LEFT_GRIPPER_INDEX,
    LEFT_POSE_SLICE,
    RIGHT_GRIPPER_INDEX,
    RIGHT_POSE_SLICE,
)
from handumi.robots.utils import mat_to_pose7, pose7_to_mat, pose_between, pose_mul

HANDUMI_POSE_ONLY_STATE_SIZE = 14

# PICO/Quest controller world -> robot world.
#   robot X (forward) <- -VR Z
#   robot Y (left +)  <- -VR X
#   robot Z (up)      <- +VR Y
VR_TO_ROBOT = np.array(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)


@dataclass(frozen=True)
class HandumiRawState:
    left_position: np.ndarray
    left_rotation: np.ndarray
    right_position: np.ndarray
    right_rotation: np.ndarray
    left_gripper_width: float
    right_gripper_width: float


@dataclass(frozen=True)
class RetargetAnchors:
    """Reference frames used to map raw HandUMI poses into robot world poses."""

    left_raw_position: np.ndarray
    left_raw_rotation: np.ndarray
    right_raw_position: np.ndarray
    right_raw_rotation: np.ndarray
    left_robot_pose7: np.ndarray
    right_robot_pose7: np.ndarray
    max_reach: float | None = None


def quaternion_xyzw_normalize(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
    norm = float(np.linalg.norm(quat_xyzw))
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (quat_xyzw / norm).astype(np.float32)


def quaternion_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert an ``[qx, qy, qz, qw]`` quaternion to a 3x3 rotation matrix."""
    qx, qy, qz, qw = quaternion_xyzw_normalize(quat_xyzw)

    return np.asarray(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float32,
    )


def split_raw_state(state: np.ndarray) -> HandumiRawState:
    """Split a 16D raw state or 14D pose-only state into left/right values."""
    arr = _as_supported_state(state)

    left_pose = arr[LEFT_POSE_SLICE]
    right_pose = arr[RIGHT_POSE_SLICE]
    has_grippers = len(arr) == HANDUMI_RAW_STATE_SIZE
    return HandumiRawState(
        left_position=left_pose[:3].copy(),
        left_rotation=quaternion_xyzw_to_matrix(left_pose[3:7]),
        right_position=right_pose[:3].copy(),
        right_rotation=quaternion_xyzw_to_matrix(right_pose[3:7]),
        left_gripper_width=float(arr[LEFT_GRIPPER_INDEX]) if has_grippers else np.nan,
        right_gripper_width=float(arr[RIGHT_GRIPPER_INDEX]) if has_grippers else np.nan,
    )


def raw_state_pose7_pair(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return left/right raw poses as ``[x, y, z, qx, qy, qz, qw]`` arrays."""
    arr = _as_supported_state(state)
    left = arr[LEFT_POSE_SLICE].copy()
    right = arr[RIGHT_POSE_SLICE].copy()
    left[3:7] = quaternion_xyzw_normalize(left[3:7])
    right[3:7] = quaternion_xyzw_normalize(right[3:7])
    return left, right


def absolute_table_robot_target_pose7(
    state: np.ndarray,
    robot_from_table_pose7: np.ndarray,
    *,
    left_tool_adapter_pose7: np.ndarray | None = None,
    right_tool_adapter_pose7: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map both table-frame TCPs through one shared deployment transform.

    Unlike per-arm anchoring, this rigid transform preserves distances and
    intersections between the left and right trajectories.
    """
    left, right = raw_state_pose7_pair(state)
    robot_from_table = np.asarray(robot_from_table_pose7, dtype=np.float32)
    left = pose_mul(robot_from_table, left)
    right = pose_mul(robot_from_table, right)
    if left_tool_adapter_pose7 is not None:
        left = pose_mul(left, left_tool_adapter_pose7)
    if right_tool_adapter_pose7 is not None:
        right = pose_mul(right, right_tool_adapter_pose7)
    return left, right


def orientation_only_pose_adapter(
    source_pose7: np.ndarray,
    target_pose7: np.ndarray,
) -> np.ndarray:
    """Return a right-side rotation adapter without changing TCP position."""
    adapter = pose_between(source_pose7, target_pose7)
    adapter[:3] = 0.0
    return adapter.astype(np.float32)


def pose7_to_wxyz(pose7: np.ndarray) -> np.ndarray:
    """Return a pose quaternion in Pyroki/JAXLie ``[qw, qx, qy, qz]`` order."""
    quat = quaternion_xyzw_normalize(np.asarray(pose7, dtype=np.float32)[3:7])
    return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float32)


def _as_supported_state(state: np.ndarray) -> np.ndarray:
    arr = np.asarray(state, dtype=np.float32)
    if len(arr) in (HANDUMI_POSE_ONLY_STATE_SIZE, HANDUMI_RAW_STATE_SIZE):
        return arr
    raise ValueError(
        "Expected HandUMI state length "
        f"{HANDUMI_RAW_STATE_SIZE} (poses + grippers) or "
        f"{HANDUMI_POSE_ONLY_STATE_SIZE} (poses only), got {len(arr)}."
    )


def matrix_to_quaternion_xyzw(rot: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to ``[qx, qy, qz, qw]``."""
    rot = np.asarray(rot, dtype=np.float32)
    t = float(rot[0, 0] + rot[1, 1] + rot[2, 2])
    if t > 0.0:
        r = np.sqrt(t + 1.0)
        s = 0.5 / r
        quat = np.array(
            [
                (rot[2, 1] - rot[1, 2]) * s,
                (rot[0, 2] - rot[2, 0]) * s,
                (rot[1, 0] - rot[0, 1]) * s,
                0.5 * r,
            ],
            dtype=np.float32,
        )
    elif rot[0, 0] >= rot[1, 1] and rot[0, 0] >= rot[2, 2]:
        r = np.sqrt(max(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2], 1e-8))
        s = 0.5 / r
        quat = np.array(
            [
                0.5 * r,
                (rot[0, 1] + rot[1, 0]) * s,
                (rot[0, 2] + rot[2, 0]) * s,
                (rot[2, 1] - rot[1, 2]) * s,
            ],
            dtype=np.float32,
        )
    elif rot[1, 1] >= rot[2, 2]:
        r = np.sqrt(max(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2], 1e-8))
        s = 0.5 / r
        quat = np.array(
            [
                (rot[0, 1] + rot[1, 0]) * s,
                0.5 * r,
                (rot[1, 2] + rot[2, 1]) * s,
                (rot[0, 2] - rot[2, 0]) * s,
            ],
            dtype=np.float32,
        )
    else:
        r = np.sqrt(max(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1], 1e-8))
        s = 0.5 / r
        quat = np.array(
            [
                (rot[0, 2] + rot[2, 0]) * s,
                (rot[1, 2] + rot[2, 1]) * s,
                0.5 * r,
                (rot[1, 0] - rot[0, 1]) * s,
            ],
            dtype=np.float32,
        )
    return quaternion_xyzw_normalize(quat)


def retarget_anchors_from_raw_state(
    state: np.ndarray,
    *,
    left_robot_pose7: np.ndarray,
    right_robot_pose7: np.ndarray,
    max_reach: float | None = None,
) -> RetargetAnchors:
    """Build anchors from the first raw frame and the robot FK home poses."""
    raw = split_raw_state(state)
    return RetargetAnchors(
        left_raw_position=raw.left_position,
        left_raw_rotation=raw.left_rotation,
        right_raw_position=raw.right_position,
        right_raw_rotation=raw.right_rotation,
        left_robot_pose7=np.asarray(left_robot_pose7, dtype=np.float32).copy(),
        right_robot_pose7=np.asarray(right_robot_pose7, dtype=np.float32).copy(),
        max_reach=max_reach,
    )


def raw_state_robot_target_pose7(
    state: np.ndarray,
    anchors: RetargetAnchors,
) -> tuple[np.ndarray, np.ndarray]:
    """Map one raw HandUMI frame to left/right robot-world pose targets.

    Positions are relative to the first raw frame and added to the robot home
    TCP. Orientations preserve the raw relative wrist rotation, expressed from
    the robot home TCP orientation.
    """
    raw = split_raw_state(state)
    left = _retarget_side_pose7(
        raw.left_position,
        raw.left_rotation,
        anchors.left_raw_position,
        anchors.left_raw_rotation,
        anchors.left_robot_pose7,
        anchors.max_reach,
    )
    right = _retarget_side_pose7(
        raw.right_position,
        raw.right_rotation,
        anchors.right_raw_position,
        anchors.right_raw_rotation,
        anchors.right_robot_pose7,
        anchors.max_reach,
    )
    return left, right


def _retarget_side_pose7(
    raw_position: np.ndarray,
    raw_rotation: np.ndarray,
    raw_home_position: np.ndarray,
    raw_home_rotation: np.ndarray,
    robot_home_pose7: np.ndarray,
    max_reach: float | None,
) -> np.ndarray:
    robot_home_pose7 = np.asarray(robot_home_pose7, dtype=np.float32)
    delta = np.asarray(raw_position, dtype=np.float32) - np.asarray(
        raw_home_position, dtype=np.float32
    )
    if max_reach is not None:
        norm = float(np.linalg.norm(delta))
        if norm > max_reach:
            delta = delta * (max_reach / max(norm, 1e-8))

    robot_home_rot = quaternion_xyzw_to_matrix(robot_home_pose7[3:7])
    rel_rot = np.asarray(raw_rotation, dtype=np.float32) @ np.asarray(
        raw_home_rotation, dtype=np.float32
    ).T
    target_rot = robot_home_rot @ rel_rot

    out = np.zeros(7, dtype=np.float32)
    out[:3] = robot_home_pose7[:3] + delta
    out[3:7] = matrix_to_quaternion_xyzw(target_rot)
    return out


def local_frame_adapter(
    source_current_pose7: np.ndarray,
    robot_current_pose7: np.ndarray,
    source_world_to_robot_world: np.ndarray = VR_TO_ROBOT,
) -> np.ndarray:
    """Return the local-frame rotation adapter for relative TCP motions.

    The source pose is still in the tracker/PICO world. The adapter aligns the
    source TCP local frame at frame 0 to the robot end-effector local frame at
    home, while applying the source-world -> robot-world axis convention.
    """
    source_rot = quaternion_xyzw_to_matrix(np.asarray(source_current_pose7)[3:7])
    robot_rot = quaternion_xyzw_to_matrix(np.asarray(robot_current_pose7)[3:7])
    world_map = np.asarray(source_world_to_robot_world, dtype=np.float32)
    return (source_rot.T @ world_map.T @ robot_rot).astype(np.float32)


def adapt_relative_pose(
    relative_pose7: np.ndarray,
    adapter_rot: np.ndarray,
    *,
    translation_scale: float = 1.0,
) -> np.ndarray:
    """Convert a source-local relative pose into a robot-EE-local relative pose."""
    relative_mat = pose7_to_mat(np.asarray(relative_pose7, dtype=np.float32))
    adapter = np.eye(4, dtype=np.float32)
    adapter[:3, :3] = np.asarray(adapter_rot, dtype=np.float32)
    adapted = np.linalg.inv(adapter) @ relative_mat @ adapter
    out = mat_to_pose7(adapted)
    out[:3] *= float(translation_scale)
    out[3:7] = quaternion_xyzw_normalize(out[3:7])
    return out.astype(np.float32)


def one_step_local_relative(poses7: np.ndarray) -> np.ndarray:
    """Return frame-to-frame local SE(3) deltas for a pose7 trajectory."""
    poses = np.asarray(poses7, dtype=np.float32)
    if len(poses) < 2:
        return np.zeros((0, 7), dtype=np.float32)
    return np.stack(
        [pose_between(poses[i], poses[i + 1]) for i in range(len(poses) - 1)],
        axis=0,
    ).astype(np.float32)


def local_relative_robot_target_pose7(
    *,
    previous_source_pose7: np.ndarray,
    current_source_pose7: np.ndarray,
    base_robot_pose7: np.ndarray,
    adapter_rot: np.ndarray,
    home_robot_pose7: np.ndarray,
    translation_scale: float = 1.0,
    max_reach: float | None = None,
) -> np.ndarray:
    """Compose one source local TCP delta onto a robot-world EE target."""
    source_step = pose_between(previous_source_pose7, current_source_pose7)
    robot_step = adapt_relative_pose(
        source_step,
        adapter_rot,
        translation_scale=translation_scale,
    )
    target = pose_mul(base_robot_pose7, robot_step).astype(np.float32)
    target[3:7] = quaternion_xyzw_normalize(target[3:7])
    return _clamp_pose7_to_reach(target, home_robot_pose7, max_reach)


def _clamp_pose7_to_reach(
    pose7: np.ndarray,
    home_pose7: np.ndarray,
    max_reach: float | None,
) -> np.ndarray:
    out = np.asarray(pose7, dtype=np.float32).copy()
    if max_reach is None:
        return out
    home = np.asarray(home_pose7, dtype=np.float32)
    delta = out[:3] - home[:3]
    norm = float(np.linalg.norm(delta))
    if norm > max_reach:
        out[:3] = home[:3] + delta * (float(max_reach) / max(norm, 1e-8))
    return out


def raw_state_target_poses(
    state: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return ``(left_pose, right_pose)`` tuples compatible with IK solvers."""
    raw = split_raw_state(state)
    return (
        (raw.left_position, raw.left_rotation),
        (raw.right_position, raw.right_rotation),
    )
