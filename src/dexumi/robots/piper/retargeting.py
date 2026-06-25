"""Piper retargeting spec — PICO wrist-to-EE mapping for the Piper arm.

Defines the robot-specific constants that parameterize the generic
:class:`~dexumi.retargeting.pico_to_robot.PicoToRobotArmRetargeter`:

- ``REST_LEFT_ARM`` / ``REST_RIGHT_ARM`` — joint angles used as the posture
  prior when no motion is commanded (zeros for Piper = natural hanging pose).
- ``PIPER_RETARGETING_SPEC`` — wires the above into the shared retargeter.
- ``PicoToPiperArmRetargeter`` — convenience subclass that binds the spec so
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

REST_LEFT_ARM = np.zeros(6, dtype=np.float32)
REST_RIGHT_ARM = np.zeros(6, dtype=np.float32)


def _left_front_wrist(forward: float, lateral: float, height: float) -> np.ndarray:
    return np.array([lateral, forward, height], dtype=np.float32)


def _right_front_wrist(forward: float, lateral: float, height: float) -> np.ndarray:
    return np.array([-lateral, forward, height], dtype=np.float32)


PIPER_RETARGETING_SPEC = RetargetingSpec(
    name="piper",
    rest_left_arm=REST_LEFT_ARM,
    rest_right_arm=REST_RIGHT_ARM,
    command_size=8,
    gripper_index=7,
    left_front_wrist=_left_front_wrist,
    right_front_wrist=_right_front_wrist,
)


class PicoToPiperArmRetargeter(PicoToRobotArmRetargeter):
    """PICO wrist retargeter for Piper end-effectors."""

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
            spec=PIPER_RETARGETING_SPEC,
            first_body_pose=first_body_pose,
            scale=scale,
            axis_map=axis_map,
            enable_left=enable_left,
            enable_right=enable_right,
            gripper=gripper,
        )


def piper_link_positions(solver, q: np.ndarray, link_indices: list[int]) -> np.ndarray:
    return robot_link_positions(solver, q, link_indices)


__all__ = [
    "PIPER_RETARGETING_SPEC",
    "PicoToPiperArmRetargeter",
    "REST_LEFT_ARM",
    "REST_RIGHT_ARM",
    "move_retargeter_to_front_workspace",
    "piper_link_positions",
    "settle_first_frame",
]
