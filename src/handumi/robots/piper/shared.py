"""Piper URDF adapter — the single source of truth for Piper link/joint names.

Everything that needs to map between Piper's logical joints (the ``Joint`` enum)
and the names that appear verbatim in ``piper.urdf`` lives here. No other module
should hard-code Piper URDF strings; they should call the helpers below instead.

Downstream consumers:
- ``piper/solver.py``    — builds ``PIPER_KINEMATICS_SPEC`` from these names.
- ``robots/registry.py`` — wires ``command_to_arm_q`` into :class:`~handumi.robots.sim.ViserSim`.
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
GRIPPER_STROKE_M = 2.0 * GRIPPER_OPEN_WIDTH_M

JOINT_LIMITS_RAD = np.array(
    [
        [-2.6179, 2.6179],
        [0.0, 3.14],
        [-2.967, 0.0],
        [-1.745, 1.745],
        [-1.22, 1.22],
        [-2.09439, 2.09439],
    ],
    dtype=np.float32,
)

LEROBOT_JOINT_NAMES = [
    "left_shoulder_pan.pos",
    "left_shoulder_lift.pos",
    "left_elbow_flex.pos",
    "left_forearm_roll.pos",
    "left_wrist_flex.pos",
    "left_wrist_roll.pos",
    "left_gripper.pos",
    "right_shoulder_pan.pos",
    "right_shoulder_lift.pos",
    "right_elbow_flex.pos",
    "right_forearm_roll.pos",
    "right_wrist_flex.pos",
    "right_wrist_roll.pos",
    "right_gripper.pos",
]

# The right-arm IK state is expressed in the mirrored dual-arm URDF frame. These
# signs map it back to the single-arm Piper SDK joint convention.
_RIGHT_SOLVER_TO_ROBOT_SIGNS = np.array(
    [1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
    dtype=np.float32,
)

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
        "Could not find piper.urdf; expected it under handumi/assets/piper "
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


def solver_arm_to_robot_arm(arm_q: np.ndarray, *, is_left: bool) -> np.ndarray:
    """Convert one arm from solver/URDF radians to Piper SDK joint radians."""
    q = np.asarray(arm_q[:ARM_JOINT_COUNT], dtype=np.float32).copy()
    if not is_left:
        q *= _RIGHT_SOLVER_TO_ROBOT_SIGNS
    return np.clip(q, JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1]).astype(
        np.float32
    )


def normalized_gripper_to_meters(gripper: float) -> float:
    """Convert the internal normalized gripper command to Piper stroke meters."""
    return float(np.clip(gripper, 0.0, 1.0) * GRIPPER_STROKE_M)


def solver_q_to_robot_state(
    q: np.ndarray,
    *,
    left_indices: list[int],
    right_indices: list[int],
    gripper: float,
) -> np.ndarray:
    """Return LeRobot state/action values in Piper physical robot units.

    Output layout is ``[left j1..j6 rad, left gripper m, right j1..j6 rad,
    right gripper m]``. This is the continuous equivalent of Piper SDK feedback:
    joint integers are divided by ``1000 deg/rad`` and gripper integers by
    ``1_000_000 m``.
    """
    grip_m = normalized_gripper_to_meters(gripper)
    return np.concatenate(
        [
            solver_arm_to_robot_arm(q[left_indices], is_left=True),
            np.array([grip_m], dtype=np.float32),
            solver_arm_to_robot_arm(q[right_indices], is_left=False),
            np.array([grip_m], dtype=np.float32),
        ]
    ).astype(np.float32)


def robot_arm_to_sdk_joint_ctrl(
    arm_state: np.ndarray,
) -> tuple[int, int, int, int, int, int]:
    """Convert six Piper joint radians to ``JointCtrl`` integer arguments."""
    q = np.clip(
        np.asarray(arm_state[:ARM_JOINT_COUNT], dtype=np.float32),
        JOINT_LIMITS_RAD[:, 0],
        JOINT_LIMITS_RAD[:, 1],
    )
    values = np.rint(np.degrees(q) * 1000.0).astype(np.int32)
    return tuple(int(v) for v in values)


def robot_gripper_to_sdk_ctrl(gripper_m: float) -> int:
    """Convert Piper gripper stroke meters to ``GripperCtrl`` integer units."""
    return int(
        round(float(np.clip(gripper_m, 0.0, GRIPPER_STROKE_M)) * 1_000_000.0)
    )
