"""Piper URDF adapter — the single source of truth for Piper link/joint names.

Everything that needs to map between Piper's logical joints (the ``Joint`` enum)
and the names that appear verbatim in ``piper.urdf`` lives here. No other module
should hard-code Piper URDF strings; they should call the helpers below instead.

Downstream consumers:
- ``piper/solver.py``    — builds ``PIPER_KINEMATICS_SPEC`` from these names.
- ``robots/registry.py`` — wires ``command_to_arm_q`` into :class:`~dexumi.robots.sim.ViserSim`.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np


class Joint(Enum):
    """Joints on one Piper arm, in URDF/control order.

    ``JOINT_1`` through ``JOINT_6`` are the revolute arm joints solved by IK.
    ``FINGER_1`` and ``FINGER_2`` are the two prismatic gripper joints and are
    commanded separately from arm IK.
    """

    JOINT_1 = "joint1"
    JOINT_2 = "joint2"
    JOINT_3 = "joint3"
    JOINT_4 = "joint4"
    JOINT_5 = "joint5"
    JOINT_6 = "joint6"
    FINGER_1 = "joint7"
    FINGER_2 = "joint8"


ARM_JOINTS: list[Joint] = [
    Joint.JOINT_1,
    Joint.JOINT_2,
    Joint.JOINT_3,
    Joint.JOINT_4,
    Joint.JOINT_5,
    Joint.JOINT_6,
]
FINGER_JOINTS: list[Joint] = [Joint.FINGER_1, Joint.FINGER_2]

ARM_JOINT_COUNT = 6
GRIPPER_OPEN_WIDTH_M = 0.035

# Shape (8,): 6 arm joints in radians, one unused slot, then gripper in [0, 1].
COMMAND_SIZE = 8
GRIPPER_INDEX = 7


def _resolve_urdf_path() -> Path:
    here = Path(__file__).resolve()
    candidates = (
        here.parents[2] / "assets" / "piper" / "piper.urdf",
        here.parents[4] / "assets" / "piper" / "piper.urdf",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Could not find piper.urdf; expected it under dexumi/assets/piper "
        "or repo assets/piper"
    )


URDF_PATH: Path = _resolve_urdf_path()


def _side_prefix(*, is_left: bool) -> str:
    return "left" if is_left else "right"


def _link_prefix(*, is_left: bool) -> str:
    return "left" if is_left else "right"


def urdf_joint_name(joint: Joint, *, is_left: bool) -> str:
    """URDF joint name for ``joint`` on the requested Piper side."""

    return f"{_side_prefix(is_left=is_left)}_{joint.value}"


def urdf_body_name(joint: Joint, *, is_left: bool) -> str:
    """URDF link driven by ``joint`` on the requested Piper side."""

    number = int(joint.value.replace("joint", ""))
    return f"{_link_prefix(is_left=is_left)}_link{number}"


def urdf_revolute_joint_names(*, is_left: bool) -> list[str]:
    """URDF revolute joint names for one arm, in IK/control order."""

    return [urdf_joint_name(joint, is_left=is_left) for joint in ARM_JOINTS]


def urdf_finger_joint_names(*, is_left: bool) -> list[str]:
    """URDF prismatic gripper joint names for one arm."""

    return [urdf_joint_name(joint, is_left=is_left) for joint in FINGER_JOINTS]


def urdf_arm_joint_names(*, is_left: bool) -> list[str]:
    """All URDF actuated joint names for one arm, in URDF order (joint1..joint8)."""

    prefix = "left" if is_left else "right"
    return [f"{prefix}_joint{i}" for i in range(1, 9)]


def urdf_revolute_body_names(*, is_left: bool) -> list[str]:
    """URDF bodies driven by the six revolute arm joints."""

    return [urdf_body_name(joint, is_left=is_left) for joint in ARM_JOINTS]


def gripper_to_finger_positions(gripper: float) -> tuple[float, float]:
    """Map a normalized gripper command to the two prismatic finger joints."""
    width = float(max(0.0, min(1.0, gripper))) * GRIPPER_OPEN_WIDTH_M
    return width, -width


def command_to_arm_q(command: np.ndarray) -> np.ndarray:
    """Map one Piper arm command to the URDF actuated-joint sub-vector.

    Indices ``0..5`` are the six revolute arm joints. Index ``6`` is unused.
    Index ``7`` is the normalized gripper opening in ``[0, 1]``, converted to
    the two prismatic finger joint positions.
    """
    finger_a, finger_b = gripper_to_finger_positions(command[GRIPPER_INDEX])
    return np.concatenate(
        [
            command[:ARM_JOINT_COUNT].astype(float),
            np.array([finger_a, finger_b], dtype=float),
        ]
    )
