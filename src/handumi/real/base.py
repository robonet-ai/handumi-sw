"""Common real-robot backend API used by HandUMI teleoperation."""

from __future__ import annotations

from typing import Protocol

import numpy as np

SIDE_NAMES: tuple[str, str] = ("left", "right")


class TeleopRobotBackend(Protocol):
    """Manufacturer-neutral contract for real HandUMI teleoperation.

    Implementations own all robot-specific details: CAN setup, SDK connection,
    unit conversion, joint-name mapping, gripper mapping, health checks, and
    slow homing. The teleop loop only deals in full robot ``q`` vectors and
    normalized gripper openings.
    """

    name: str
    active_sides: tuple[str, ...]

    def setup(self, *, repair: bool = True) -> None:
        """Prepare host transports before opening robot SDK connections."""

    def connect(self) -> None:
        """Open physical robot connections."""

    def disconnect(self) -> None:
        """Close physical robot connections and leave actuators disabled."""

    def read(self, base_q: np.ndarray | None = None) -> np.ndarray:
        """Return the latest measured or scheduled full robot configuration."""

    def home(self, q: np.ndarray) -> None:
        """Start streaming and move slowly to a safe starting pose."""

    def move_home(self, q: np.ndarray) -> None:
        """Return slowly to a safe pose while already connected and streaming."""

    def write(
        self,
        q: np.ndarray,
        gripper_openings: dict[str, float],
    ) -> None:
        """Publish the newest full target without waiting for motion to finish.

        Backends must treat this as latest-target delivery, not a FIFO of
        trajectories: their own fixed-rate command streamer interpolates and
        retransmits the newest target.  This keeps tracking/IK cadence
        independent from each robot SDK/CAN cadence and prevents stale hand
        poses from accumulating as lag.
        """

    def hold(self, base_q: np.ndarray) -> np.ndarray:
        """Cancel pending motion and hold the current robot-side command."""

    def check_health(self) -> None:
        """Raise if the robot command stream or SDK reports a failure."""


__all__ = ["SIDE_NAMES", "TeleopRobotBackend"]
