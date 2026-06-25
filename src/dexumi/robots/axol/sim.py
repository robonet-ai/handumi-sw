"""Axol-specific Viser simulation.

Thin subclass of :class:`~dexumi.robots.sim.ViserSim` that wires Axol's URDF
path and command layout. The Axol URDF has **no** prismatic gripper joint, so
the gripper value (index 7 of the command) is not forwarded to the renderer.
"""

from __future__ import annotations

import numpy as np

from dexumi.robots.sim import ViserSim

from .shared import ARM_JOINTS, URDF_PATH, urdf_arm_joint_names

_ARM_JOINT_COUNT = len(ARM_JOINTS)
_COMMAND_SIZE = 8


class Sim(ViserSim):
    """Viser-based dual Axol simulation.

    Each arm command is shape ``(8,)``: indices ``0..6`` are the seven revolute
    arm joints in radians, index ``7`` is the gripper opening in ``[0, 1]``
    (not rendered — Axol has no prismatic gripper joint in the URDF).
    """

    def __init__(
        self,
        *,
        joint_names: list[str] | None = None,
        default_q: np.ndarray | None = None,
        port: int = 8002,
    ) -> None:
        super().__init__(
            urdf_path=URDF_PATH,
            left_joint_names=urdf_arm_joint_names(is_left=True),
            right_joint_names=urdf_arm_joint_names(is_left=False),
            command_size=_COMMAND_SIZE,
            joint_names=joint_names,
            default_q=default_q,
            port=port,
        )

    def _arm_q(self, command: np.ndarray) -> np.ndarray:
        return command[:_ARM_JOINT_COUNT].astype(float)
