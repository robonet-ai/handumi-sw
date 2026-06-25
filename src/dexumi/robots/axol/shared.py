"""Axol URDF adapter — the single source of truth for Axol link/joint names.

Everything that needs to map between Axol's logical joints (the ``Joint`` enum)
and the names that appear verbatim in ``axol.urdf`` lives here. No other module
should hard-code Axol URDF strings; they should call the helpers below instead.

Downstream consumers:
- ``axol/solver.py``   — builds ``AXOL_KINEMATICS_SPEC`` from these names.
- ``robots/registry.py`` — wires ``command_to_arm_q`` into :class:`~dexumi.robots.sim.ViserSim`.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np


class Joint(Enum):
    """All motor joints on one arm, in control order.

    The seven arm joints (``SHOULDER_1`` through ``WRIST_3``) are collected in
    ``ARM_JOINTS``. ``GRIPPER`` is the eighth entry and is handled separately
    from the arm joints throughout the control stack.
    """

    SHOULDER_1 = "shoulder_1"
    SHOULDER_2 = "shoulder_2"
    SHOULDER_3 = "shoulder_3"
    ELBOW = "elbow"
    WRIST_1 = "wrist_1"
    WRIST_2 = "wrist_2"
    WRIST_3 = "wrist_3"
    GRIPPER = "gripper"


CAN_LEFT = "can_alm_axol_l"
CAN_RIGHT = "can_alm_axol_r"

ARM_JOINTS: list[Joint] = [j for j in Joint if j != Joint.GRIPPER]

ARM_JOINT_COUNT = len(ARM_JOINTS)
# Shape (8,): 7 arm joints in radians, then gripper in [0, 1] (not actuated in URDF).
COMMAND_SIZE = 8
GRIPPER_INDEX = 7


def _resolve_urdf_path() -> Path:
    here = Path(__file__).resolve()
    candidates = (
        here.parents[2] / "assets" / "axol" / "urdf" / "axol.urdf",
        here.parents[4] / "assets" / "axol" / "urdf" / "axol.urdf",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Could not find axol.urdf; expected it under dexumi/assets/axol or repo assets/axol"
    )


URDF_PATH: Path = _resolve_urdf_path()


# Single source of truth for URDF joint and body names. All helpers
# (gravity comp, IK solver, simulation) compose ``f"{side}_{suffix}"`` from
# these tables via the ``urdf_*_name`` helpers below.

# ``Joint.GRIPPER`` is intentionally absent: the gripper is a fixed URDF
# joint with no actuator counterpart.
_ARM_JOINT_URDF_SUFFIX: dict[Joint, str] = {
    Joint.SHOULDER_1: "s1_0",
    Joint.SHOULDER_2: "s2_0",
    Joint.SHOULDER_3: "s3_0",
    Joint.ELBOW: "e1_0",
    Joint.WRIST_1: "e2_0",
    Joint.WRIST_2: "w1_0",
    Joint.WRIST_3: "w2_0",
}

# Body driven by each joint. ``Joint.GRIPPER`` maps to the (fixed-jointed)
# gripper link itself; MuJoCo merges this body into ``*_w2`` at load time.
_BODY_URDF_SUFFIX: dict[Joint, str] = {
    Joint.SHOULDER_1: "s2",
    Joint.SHOULDER_2: "s3",
    Joint.SHOULDER_3: "e1",
    Joint.ELBOW: "e2",
    Joint.WRIST_1: "w0",
    Joint.WRIST_2: "w1",
    Joint.WRIST_3: "w2",
    Joint.GRIPPER: "gripper",
}


def urdf_joint_name(joint: Joint, *, is_left: bool) -> str:
    """URDF revolute-joint name driving ``joint`` on the given arm.

    Example::

        urdf_joint_name(Joint.SHOULDER_1, is_left=True) == "left_s1_0"

    Raises ``KeyError`` for ``Joint.GRIPPER`` (no actuator joint in the URDF).
    """
    side = "left" if is_left else "right"
    return f"{side}_{_ARM_JOINT_URDF_SUFFIX[joint]}"


def urdf_body_name(joint: Joint, *, is_left: bool) -> str:
    """URDF body driven by ``joint`` on the given arm.

    Example::

        urdf_body_name(Joint.SHOULDER_1, is_left=True) == "left_s2"
        urdf_body_name(Joint.GRIPPER,    is_left=True) == "left_gripper"
    """
    side = "left" if is_left else "right"
    return f"{side}_{_BODY_URDF_SUFFIX[joint]}"


def urdf_arm_joint_names(*, is_left: bool) -> list[str]:
    """URDF revolute-joint names for the 7 arm joints, in :data:`ARM_JOINTS` order."""
    return [urdf_joint_name(j, is_left=is_left) for j in ARM_JOINTS]


def urdf_arm_body_names(*, is_left: bool) -> list[str]:
    """URDF bodies driven by the 7 arm joints, in :data:`ARM_JOINTS` order."""
    return [urdf_body_name(j, is_left=is_left) for j in ARM_JOINTS]


def command_to_arm_q(command: np.ndarray) -> np.ndarray:
    """Map one Axol arm command to the URDF actuated-joint sub-vector.

    Indices ``0..6`` are the seven revolute arm joints. Index ``7`` is the
    normalized gripper value and is not forwarded (Axol has no prismatic
    gripper joint in the URDF).
    """
    return command[:ARM_JOINT_COUNT].astype(float)
