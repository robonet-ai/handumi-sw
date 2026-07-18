"""Lazy registry for optional real-robot backends."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from handumi.robots.registry import RobotRuntime

REGISTERED_BACKENDS: tuple[str, ...] = ("openarm_can", "piper_can")


class RobotBackend(Protocol):
    """Manufacturer-neutral contract consumed by real teleoperation."""

    def prepare(self, *, repair: bool = True) -> None: ...

    def connect(self) -> None: ...

    def home(self, q: np.ndarray, joint_names: list[str]) -> None: ...

    def move_home(self, q: np.ndarray, joint_names: list[str]) -> None: ...

    def command(
        self,
        q: np.ndarray,
        joint_names: list[str],
        gripper_openings: dict[str, float],
    ) -> None: ...

    def hold(self, base_q: np.ndarray, joint_names: list[str]) -> np.ndarray: ...

    def check_health(self) -> None: ...

    def close(self) -> None: ...


def make_real_backend(
    robot: str,
    *,
    runtime: RobotRuntime,
    rig_config: Path,
    active_sides: tuple[str, ...] = ("left", "right"),
) -> RobotBackend:
    """Create a backend without importing SDKs for unused robots."""
    backend = runtime.config.real.backend
    if backend is None:
        raise ValueError(f"Robot {robot!r} does not declare real.backend.")
    if backend == "piper_can":
        from handumi.real.backends.piper import PiperBackend

        return PiperBackend.from_config(runtime=runtime, rig_config=rig_config)
    if backend == "openarm_can":
        from handumi.real.openarm_gripper_calibration import (
            user_openarm_gripper_calibration_path,
        )
        from handumi.real.openarm_can import (
            OpenArmCanEnvironment,
            load_openarm_settings,
        )

        return OpenArmCanEnvironment(
            load_openarm_settings(
                rig_config,
                runtime.config.real_options,
                user_openarm_gripper_calibration_path(),
            ),
            active_sides=active_sides,
            joint_limits={
                name: (float(lower), float(upper))
                for name, lower, upper in zip(
                    runtime.joint_names,
                    runtime.robot.joints.lower_limits,
                    runtime.robot.joints.upper_limits,
                    strict=True,
                )
            },
        )
    raise ValueError(
        f"No real hardware backend {backend!r} registered for robot {robot!r}."
    )


REAL_BACKEND_NAMES: tuple[str, ...] = ("openarmv1", "piper")

__all__ = [
    "REAL_BACKEND_NAMES",
    "REGISTERED_BACKENDS",
    "RobotBackend",
    "make_real_backend",
]
