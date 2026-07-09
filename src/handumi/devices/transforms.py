"""Coordinate / calibration transforms for Meta Quest tracking (Phase 2A, Step 2).

Everything here is explicit, side-effect-free, and unit-tested. The Quest app
streams raw **Unity** poses (left-handed, X right / Y up / Z forward); this
module converts them at the boundary and applies HandUMI calibration:

    raw Unity controller pose
      -> unity_pose_to_handumi   (Unity left-handed -> right-handed)
      -> apply mounting offset    (controller anchor -> gripper TCP)
      -> to_workspace             (re-center on the reset reference / HMD)
      => gripper TCP pose in handumi_workspace

The Unity->right-handed mapping is ported from yubi-sw
(`airoa_quest/.../quest_bridge_node.py::_unity_pose_to_ros`):

    position:    Unity(x, y, z)     -> (z, -x, y)
    quaternion:  Unity(x, y, z, w)  -> (z, -x, y, -w)

Poses are stored as position (3,) + quaternion `[x, y, z, w]`, the same layout
the rest of HandUMI uses (`retargeting.handumi_to_robot`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# Quaternion helpers ([x, y, z, w], right-handed).
# ---------------------------------------------------------------------------


def quat_normalize(q: npt.ArrayLike) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        return _IDENTITY_QUAT.copy()
    return q / n


def quat_conjugate(q: npt.ArrayLike) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([-x, -y, -z, w], dtype=np.float64)


def quat_multiply(a: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray:
    """Hamilton product of two `[x, y, z, w]` quaternions (a then b applied)."""
    ax, ay, az, aw = np.asarray(a, dtype=np.float64).reshape(4)
    bx, by, bz, bw = np.asarray(b, dtype=np.float64).reshape(4)
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def quat_rotate(q: npt.ArrayLike, v: npt.ArrayLike) -> np.ndarray:
    """Rotate a 3-vector by a quaternion."""
    qx, qy, qz, qw = quat_normalize(q)
    u = np.array([qx, qy, qz], dtype=np.float64)
    v = np.asarray(v, dtype=np.float64).reshape(3)
    uv = np.cross(u, v)
    uuv = np.cross(u, uv)
    return v + 2.0 * (qw * uv + uuv)


def quat_to_matrix(q: npt.ArrayLike) -> np.ndarray:
    """Convert `[x, y, z, w]` to a 3x3 rotation matrix."""
    x, y, z, w = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat(m: npt.ArrayLike) -> np.ndarray:
    """Convert a 3x3 rotation matrix to `[x, y, z, w]` (w >= 0)."""
    m = np.asarray(m, dtype=np.float64).reshape(3, 3)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    if q[3] < 0:
        q = -q
    return quat_normalize(q)


# ---------------------------------------------------------------------------
# Rigid pose (position + quaternion).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, init=False)
class Pose:
    """A rigid transform: translation (3,) + rotation quaternion `[x, y, z, w]`."""

    position: np.ndarray
    quaternion: np.ndarray

    def __init__(self, position: npt.ArrayLike, quaternion: npt.ArrayLike) -> None:
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat = quat_normalize(quaternion)
        object.__setattr__(self, "position", pos)
        object.__setattr__(self, "quaternion", quat)

    @classmethod
    def identity(cls) -> "Pose":
        return cls(np.zeros(3), _IDENTITY_QUAT.copy())

    @classmethod
    def from_matrix(cls, m: npt.ArrayLike) -> "Pose":
        m = np.asarray(m, dtype=np.float64).reshape(4, 4)
        return cls(m[:3, 3], matrix_to_quat(m[:3, :3]))

    def as_matrix(self) -> np.ndarray:
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] = quat_to_matrix(self.quaternion)
        m[:3, 3] = self.position
        return m

    def compose(self, other: "Pose") -> "Pose":
        """``self @ other`` — apply ``other`` in ``self``'s local frame."""
        position = self.position + quat_rotate(self.quaternion, other.position)
        quaternion = quat_multiply(self.quaternion, other.quaternion)
        return Pose(position, quaternion)

    def inverse(self) -> "Pose":
        inv_q = quat_conjugate(self.quaternion)
        inv_pos = -quat_rotate(inv_q, self.position)
        return Pose(inv_pos, inv_q)

    def __matmul__(self, other: "Pose") -> "Pose":
        return self.compose(other)


# ---------------------------------------------------------------------------
# Unity -> HandUMI (right-handed) conversion.
# ---------------------------------------------------------------------------


def unity_position_to_handumi(position: npt.ArrayLike) -> np.ndarray:
    """Unity (x, y, z) -> right-handed (z, -x, y)."""
    x, y, z = np.asarray(position, dtype=np.float64).reshape(3)
    return np.array([z, -x, y], dtype=np.float64)


def unity_quaternion_to_handumi(quaternion: npt.ArrayLike) -> np.ndarray:
    """Unity `[x, y, z, w]` -> right-handed `[z, -x, y, -w]`."""
    x, y, z, w = np.asarray(quaternion, dtype=np.float64).reshape(4)
    return quat_normalize(np.array([z, -x, y, -w], dtype=np.float64))


def unity_pose_to_handumi(position: npt.ArrayLike, quaternion: npt.ArrayLike) -> Pose:
    """Convert a raw Unity pose to a right-handed HandUMI :class:`Pose`."""
    return Pose(
        unity_position_to_handumi(position),
        unity_quaternion_to_handumi(quaternion),
    )


# ---------------------------------------------------------------------------
# Calibration: mounting offset + workspace reset.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MountingOffsets:
    """Fixed controller-anchor -> gripper-TCP offsets, in HandUMI convention.

    Applied on the right of the (already converted) controller pose:
    ``gripper_tcp = controller_pose @ offset``.
    """

    left: Pose
    right: Pose

    @classmethod
    def identity(cls) -> "MountingOffsets":
        return cls(Pose.identity(), Pose.identity())

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MountingOffsets":
        data = data or {}
        return cls(left=_pose_from_dict(data.get("left")),
                   right=_pose_from_dict(data.get("right")))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MountingOffsets":
        with Path(path).open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        calib = (data.get("calibration") or {}).get("controller_to_gripper_tcp")
        return cls.from_dict(calib)


@dataclass(frozen=True)
class WorkspaceCalibration:
    """Maps poses from the Quest tracking frame into ``handumi_workspace``.

    ``pose_ws = workspace_from_quest @ pose_quest``. A reset captures a reference
    pose (typically the converted HMD pose) and re-centers the workspace on it,
    so the reference becomes the workspace origin.
    """

    workspace_from_quest: Pose

    @classmethod
    def identity(cls) -> "WorkspaceCalibration":
        return cls(Pose.identity())

    @classmethod
    def from_reference(cls, reference: Pose) -> "WorkspaceCalibration":
        """Re-center the workspace so ``reference`` maps to the origin."""
        return cls(reference.inverse())

    def apply(self, pose_quest: Pose) -> Pose:
        return self.workspace_from_quest.compose(pose_quest)


def apply_mounting_offset(controller_pose: Pose, offset: Pose) -> Pose:
    """Controller anchor pose -> gripper TCP pose (offset in controller frame)."""
    return controller_pose.compose(offset)


def gripper_pose_in_workspace(
    unity_position: npt.ArrayLike,
    unity_quaternion: npt.ArrayLike,
    *,
    mounting_offset: Pose,
    workspace: WorkspaceCalibration,
) -> Pose:
    """Full per-controller pipeline: raw Unity pose -> gripper TCP in workspace.

    ``workspace_from_quest @ unity_to_handumi(pose) @ mounting_offset``.
    """
    controller = unity_pose_to_handumi(unity_position, unity_quaternion)
    gripper_tcp = apply_mounting_offset(controller, mounting_offset)
    return workspace.apply(gripper_tcp)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _pose_from_dict(data: dict[str, Any] | None) -> Pose:
    if not data:
        return Pose.identity()
    position = data.get("position", [0.0, 0.0, 0.0])
    quaternion = data.get("quaternion", [0.0, 0.0, 0.0, 1.0])
    return Pose(np.asarray(position, dtype=np.float64),
                np.asarray(quaternion, dtype=np.float64))


__all__ = [
    "MountingOffsets",
    "Pose",
    "WorkspaceCalibration",
    "apply_mounting_offset",
    "gripper_pose_in_workspace",
    "matrix_to_quat",
    "quat_conjugate",
    "quat_multiply",
    "quat_normalize",
    "quat_rotate",
    "quat_to_matrix",
    "unity_pose_to_handumi",
    "unity_position_to_handumi",
    "unity_quaternion_to_handumi",
]
