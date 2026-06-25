"""Axol retargeting spec — PICO wrist-to-EE mapping for the Axol arm.

Defines the robot-specific constants that parameterize the generic
:class:`~dexumi.retargeting.pico_to_robot.PicoToRobotArmRetargeter`:

- ``REST_LEFT_ARM`` / ``REST_RIGHT_ARM`` — joint angles that put Axol in a
  comfortable resting pose (slight elbow bend, neutral wrist).
- ``AXOL_RETARGETING_SPEC`` — wires the above into the shared retargeter.
- ``PicoToAxolArmRetargeter`` — convenience subclass that binds the spec so
  callers don't need to pass it explicitly.

Re-exports ``move_retargeter_to_front_workspace`` and ``settle_first_frame``
from the generic module so callers have a single import point.
"""

from __future__ import annotations

import numpy as np

from dexumi.retargeting.pico_to_robot import (
    PicoToRobotArmRetargeter,
    RetargetingSpec,
    move_retargeter_to_front_workspace,
    robot_link_positions,
    settle_first_frame,
)

REST_LEFT_ARM = np.array(
    [-0.025 * 2 * np.pi, 0.0, 0.0, 0.05 * 2 * np.pi, 0.0, 0.0, -0.025 * 2 * np.pi],
    dtype=np.float32,
)
REST_RIGHT_ARM = np.array(
    [0.025 * 2 * np.pi, 0.0, 0.0, -0.05 * 2 * np.pi, 0.0, 0.0, 0.025 * 2 * np.pi],
    dtype=np.float32,
)


def _left_front_wrist(forward: float, lateral: float, height: float) -> np.ndarray:
    return np.array([lateral, forward, height], dtype=np.float32)


def _right_front_wrist(forward: float, lateral: float, height: float) -> np.ndarray:
    return np.array([-lateral, forward, height], dtype=np.float32)


AXOL_RETARGETING_SPEC = RetargetingSpec(
    name="axol",
    rest_left_arm=REST_LEFT_ARM,
    rest_right_arm=REST_RIGHT_ARM,
    command_size=8,
    gripper_index=7,
    left_front_wrist=_left_front_wrist,
    right_front_wrist=_right_front_wrist,
)


class PicoToAxolArmRetargeter(PicoToRobotArmRetargeter):
    """PICO wrist retargeter for Axol end-effectors."""

    def __init__(
        self,
        *,
        solver,
        first_body_pose: np.ndarray,
        scale: float,
        axis_map: str,
        enable_left: bool = True,
        enable_right: bool = True,
        gripper: float = 1.0,
    ) -> None:
        super().__init__(
            solver=solver,
            spec=AXOL_RETARGETING_SPEC,
            first_body_pose=first_body_pose,
            scale=scale,
            axis_map=axis_map,
            enable_left=enable_left,
            enable_right=enable_right,
            gripper=gripper,
        )


def axol_link_positions(solver, q: np.ndarray, link_indices: list[int]) -> np.ndarray:
    return robot_link_positions(solver, q, link_indices)


__all__ = [
    "AXOL_RETARGETING_SPEC",
    "PicoToAxolArmRetargeter",
    "REST_LEFT_ARM",
    "REST_RIGHT_ARM",
    "axol_link_positions",
    "move_retargeter_to_front_workspace",
    "settle_first_frame",
]
