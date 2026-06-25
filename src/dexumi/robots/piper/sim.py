"""Piper-specific Viser simulation.

Thin subclass of :class:`~dexumi.robots.sim.ViserSim` that wires Piper's URDF
path and command layout. The Piper URDF has two prismatic finger joints, so the
normalized gripper value (index 7 of the command) is converted to physical finger
positions before being sent to viser.
"""

from __future__ import annotations

import numpy as np

from dexumi.robots.sim import ViserSim

from .shared import (
    ARM_JOINT_COUNT,
    COMMAND_SIZE,
    GRIPPER_INDEX,
    URDF_PATH,
    gripper_to_finger_positions,
    urdf_arm_joint_names,
)


class Sim(ViserSim):
    """Viser-based dual Piper simulation.

    Each arm command is shape ``(8,)``: indices ``0..5`` are the six revolute
    arm joints in radians, index ``6`` is unused, and index ``7`` is the gripper
    opening normalized to ``[0, 1]`` (0 = closed, 1 = fully open).
    """

    def __init__(
        self,
        *,
        joint_names: list[str] | None = None,
        default_q: np.ndarray | None = None,
        port: int = 8003,
    ) -> None:
        super().__init__(
            urdf_path=URDF_PATH,
            left_joint_names=urdf_arm_joint_names(is_left=True),
            right_joint_names=urdf_arm_joint_names(is_left=False),
            command_size=COMMAND_SIZE,
            joint_names=joint_names,
            default_q=default_q,
            port=port,
        )

    def _arm_q(self, command: np.ndarray) -> np.ndarray:
        finger_a, finger_b = gripper_to_finger_positions(command[GRIPPER_INDEX])
        return np.concatenate(
            [
                command[:ARM_JOINT_COUNT].astype(float),
                np.array([finger_a, finger_b], dtype=float),
            ]
        )
