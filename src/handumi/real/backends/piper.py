"""Generic backend adapter around the existing Piper implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from handumi.real.piper_can import (
    PiperCanEnvironment,
    load_piper_can_settings,
    piper_mdeg_to_q,
    q_to_piper_mdeg,
)
from handumi.real.can_setup import ensure_can_interfaces_ready
from handumi.robots.registry import RobotRuntime


class PiperBackend:
    def __init__(
        self, environment: PiperCanEnvironment, *, max_width_mm: float
    ) -> None:
        self.environment = environment
        self.max_width_mm = float(max_width_mm)

    @classmethod
    def from_config(cls, *, runtime: RobotRuntime, rig_config: Path) -> "PiperBackend":
        return cls(
            PiperCanEnvironment(
                load_piper_can_settings(rig_config, runtime.config.real)
            ),
            max_width_mm=runtime.config.gripper_max_width_m * 1000.0,
        )

    def connect(self) -> None:
        self.environment.connect()

    def prepare(self, *, repair: bool = True) -> None:
        settings = self.environment.settings
        ensure_can_interfaces_ready(
            [settings.left_port, settings.right_port],
            bitrate=settings.bitrate,
            restart_ms=settings.restart_ms,
            repair=repair,
        )

    def home(self, q: np.ndarray, joint_names: list[str]) -> None:
        self.environment.home(q_to_piper_mdeg(q, joint_names))

    def move_home(self, q: np.ndarray, joint_names: list[str]) -> None:
        self.environment.move_home(q_to_piper_mdeg(q, joint_names))

    def command(
        self,
        q: np.ndarray,
        joint_names: list[str],
        gripper_openings: dict[str, float],
    ) -> None:
        self.environment.set_q(q, joint_names)
        self.environment.set_gripper_widths_mm(
            {
                side: float(np.clip(value, 0.0, 1.0)) * self.max_width_mm
                for side, value in gripper_openings.items()
            }
        )

    def hold(self, base_q: np.ndarray, joint_names: list[str]) -> np.ndarray:
        held = self.environment.hold_current_commands_mdeg()
        return piper_mdeg_to_q(
            left_mdeg=held["left"],
            right_mdeg=held["right"],
            actuated_names=joint_names,
            base_q=base_q,
        )

    def check_health(self) -> None:
        self.environment.raise_if_failed()

    def close(self) -> None:
        self.environment.close()
