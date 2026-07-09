"""HandUMI raw robot-agnostic dataset schema.

The raw state is the source-of-truth representation recorded from the wearable
HandUMI hardware before any robot-specific IK or retargeting.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from handumi.devices.transforms import Pose

HANDUMI_RAW_STATE_NAMES: tuple[str, ...] = (
    "left_x",
    "left_y",
    "left_z",
    "left_qx",
    "left_qy",
    "left_qz",
    "left_qw",
    "right_x",
    "right_y",
    "right_z",
    "right_qx",
    "right_qy",
    "right_qz",
    "right_qw",
    "left_gripper_width",
    "right_gripper_width",
)

HANDUMI_RAW_STATE_SIZE = len(HANDUMI_RAW_STATE_NAMES)

LEFT_POSE_SLICE = slice(0, 7)
RIGHT_POSE_SLICE = slice(7, 14)
LEFT_GRIPPER_INDEX = 14
RIGHT_GRIPPER_INDEX = 15

HANDUMI_RAW_IMAGE_KEYS: tuple[str, ...] = (
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)


def raw_state_feature() -> dict[str, Any]:
    """Return the LeRobot feature metadata for raw state/action vectors."""
    return {
        "dtype": "float32",
        "shape": [HANDUMI_RAW_STATE_SIZE],
        "names": list(HANDUMI_RAW_STATE_NAMES),
    }


def validate_raw_state_shape(value: Sequence[object], *, name: str = "raw state") -> None:
    """Raise ``ValueError`` if ``value`` is not a single 16D raw state vector."""
    if len(value) != HANDUMI_RAW_STATE_SIZE:
        raise ValueError(
            f"Expected {name} length {HANDUMI_RAW_STATE_SIZE}, got {len(value)}."
        )


def pose_to_state_vector(
    left: "Pose",
    right: "Pose",
    left_width_m: float,
    right_width_m: float,
) -> np.ndarray:
    """Assemble the 16D HandUMI raw state from calibrated left/right poses + widths.

    Backend-neutral: any tracking source (Quest, PICO) that produces workspace
    ``Pose`` values plus gripper widths feeds the same raw-state layout.
    """
    state = np.zeros(HANDUMI_RAW_STATE_SIZE, dtype=np.float32)
    state[LEFT_POSE_SLICE] = np.concatenate([left.position, left.quaternion])
    state[RIGHT_POSE_SLICE] = np.concatenate([right.position, right.quaternion])
    state[LEFT_GRIPPER_INDEX] = float(left_width_m)
    state[RIGHT_GRIPPER_INDEX] = float(right_width_m)
    return state


__all__ = [
    "HANDUMI_RAW_IMAGE_KEYS",
    "HANDUMI_RAW_STATE_NAMES",
    "HANDUMI_RAW_STATE_SIZE",
    "LEFT_GRIPPER_INDEX",
    "LEFT_POSE_SLICE",
    "RIGHT_GRIPPER_INDEX",
    "RIGHT_POSE_SLICE",
    "pose_to_state_vector",
    "raw_state_feature",
    "validate_raw_state_shape",
]

