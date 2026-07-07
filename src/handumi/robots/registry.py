"""Embodiment registry — load robot specs and Viser simulators by name.

Typical usage::

    from handumi.robots.registry import load_embodiment

    runtime = load_embodiment("axol")
    solver = runtime.solver_cls()
    sim = runtime.make_sim()
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from handumi.robots.kinematics import KinematicsConfig
from handumi.sim.mujoco_sim import MujocoSim, SceneConfig
from handumi.sim.viser_sim import ViserSim

DEFAULT_COMPARE_AXIS_MAPS: dict[str, tuple[str, ...]] = {
    "piper": (
        "x,z,y",
        "x,z,-y",
        "x,-z,y",
        "x,-z,-y",
        "-x,z,y",
        "-x,z,-y",
        "-x,-z,y",
        "-x,-z,-y",
    ),
    "axol": (
        "z,x,y",
        "z,x,-y",
        "z,-x,y",
        "z,-x,-y",
        "-z,x,y",
        "-z,x,-y",
        "-z,-x,y",
        "-z,-x,-y",
    ),
}

EMBODIMENT_NAMES: tuple[str, ...] = ("piper", "axol")


@dataclass(frozen=True)
class EmbodimentRuntime:
    """Resolved embodiment configuration for IK, retargeting, and visualization."""

    name: str
    config_cls: type
    solver_cls: type
    retargeter_cls: type
    move_to_front_workspace: Any
    settle_first_frame: Any
    urdf_path: Path
    urdf_arm_joint_names: Callable[..., list[str]]
    command_size: int
    command_to_arm_q: Callable[[np.ndarray], np.ndarray]
    default_port: int
    default_axis_map: str
    default_compare_axis_maps: tuple[str, ...]
    default_workspace: str
    wrist_forward: float
    wrist_height: float
    wrist_lateral: float
    # MuJoCo physics is optional per embodiment: an embodiment with no MJCF
    # (e.g. axol today, which only has a URDF) leaves these None, and
    # make_physics() returns None — Viser then stays kinematics-only for it,
    # same as before physics existed at all.
    mjcf_path: Path | None = None
    mujoco_arm_joint_names: Callable[..., list[str]] | None = None

    def make_sim(
        self,
        *,
        port: int | None = None,
        joint_names: list[str] | None = None,
        default_q: np.ndarray | None = None,
        scene_bodies: list | None = None,
    ) -> ViserSim:
        """Build a :class:`~handumi.sim.viser_sim.ViserSim` for this embodiment."""
        return ViserSim(
            urdf_path=self.urdf_path,
            left_joint_names=self.urdf_arm_joint_names(is_left=True),
            right_joint_names=self.urdf_arm_joint_names(is_left=False),
            command_size=self.command_size,
            arm_q_fn=self.command_to_arm_q,
            joint_names=joint_names,
            default_q=default_q,
            scene_bodies=scene_bodies,
            port=self.default_port if port is None else port,
        )

    def make_physics(
        self,
        *,
        scene_config: SceneConfig | None = None,
    ) -> MujocoSim | None:
        """Build a :class:`~handumi.sim.mujoco_sim.MujocoSim` for this
        embodiment, or ``None`` if it has no MJCF (no real physics available)."""
        if self.mjcf_path is None or self.mujoco_arm_joint_names is None:
            return None
        return MujocoSim(
            mjcf_path=self.mjcf_path,
            left_joint_names=self.mujoco_arm_joint_names(is_left=True),
            right_joint_names=self.mujoco_arm_joint_names(is_left=False),
            command_to_arm_q_fn=self.command_to_arm_q,
            scene_config=scene_config,
        )


def load_embodiment(name: str) -> EmbodimentRuntime:
    """Return the runtime bundle for ``name`` (``"piper"`` or ``"axol"``)."""
    if name == "piper":
        from handumi.robots.piper.retargeting import (
            PicoToPiperArmRetargeter,
            move_retargeter_to_front_workspace,
            settle_first_frame,
        )
        from handumi.robots.piper.shared import (
            COMMAND_SIZE,
            MJCF_PATH,
            URDF_PATH,
            command_to_arm_q,
            mujoco_arm_joint_names,
            urdf_arm_joint_names,
        )
        from handumi.robots.piper.solver import KinematicsSolver

        return EmbodimentRuntime(
            name="piper",
            config_cls=KinematicsConfig,
            solver_cls=KinematicsSolver,
            retargeter_cls=PicoToPiperArmRetargeter,
            move_to_front_workspace=move_retargeter_to_front_workspace,
            settle_first_frame=settle_first_frame,
            urdf_path=URDF_PATH,
            urdf_arm_joint_names=urdf_arm_joint_names,
            command_size=COMMAND_SIZE,
            command_to_arm_q=command_to_arm_q,
            default_port=8003,
            default_axis_map="x,z,y",
            default_compare_axis_maps=DEFAULT_COMPARE_AXIS_MAPS["piper"],
            default_workspace="rest",
            wrist_forward=0.34,
            wrist_height=0.24,
            wrist_lateral=0.23,
            mjcf_path=MJCF_PATH,
            mujoco_arm_joint_names=mujoco_arm_joint_names,
        )

    if name == "axol":
        from handumi.robots.axol.retargeting import (
            PicoToAxolArmRetargeter,
            move_retargeter_to_front_workspace,
            settle_first_frame,
        )
        from handumi.robots.axol.shared import (
            COMMAND_SIZE,
            URDF_PATH,
            command_to_arm_q,
            urdf_arm_joint_names,
        )
        from handumi.robots.axol.solver import KinematicsSolver

        return EmbodimentRuntime(
            name="axol",
            config_cls=KinematicsConfig,
            solver_cls=KinematicsSolver,
            retargeter_cls=PicoToAxolArmRetargeter,
            move_to_front_workspace=move_retargeter_to_front_workspace,
            settle_first_frame=settle_first_frame,
            urdf_path=URDF_PATH,
            urdf_arm_joint_names=urdf_arm_joint_names,
            command_size=COMMAND_SIZE,
            command_to_arm_q=command_to_arm_q,
            default_port=8002,
            default_axis_map="x,z,y",
            default_compare_axis_maps=DEFAULT_COMPARE_AXIS_MAPS["axol"],
            default_workspace="rest",
            wrist_forward=0.28,
            wrist_height=0.28,
            wrist_lateral=0.20,
        )

    raise ValueError(
        f"Unsupported embodiment: {name!r}. Choose from {', '.join(EMBODIMENT_NAMES)}."
    )


__all__ = [
    "DEFAULT_COMPARE_AXIS_MAPS",
    "EMBODIMENT_NAMES",
    "EmbodimentRuntime",
    "load_embodiment",
]
