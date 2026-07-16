"""Per-arm physical J8 endpoint calibration for OpenArm v1."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class OpenArmGripperLimits:
    closed_position_rad: float
    open_position_rad: float

    @property
    def travel_rad(self) -> float:
        return abs(self.open_position_rad - self.closed_position_rad)

    def validate(self) -> None:
        # OpenArm v1 closes in the positive J8 direction and opens in the
        # negative direction. Its nominal physical travel is 60 degrees.
        if self.open_position_rad >= self.closed_position_rad:
            raise ValueError(
                "OpenArm v1 J8 open position must be below its closed position."
            )
        if not 0.5 <= self.travel_rad <= 1.6:
            raise ValueError(
                "OpenArm v1 J8 travel must be between 0.5 and 1.6 rad; "
                f"measured {self.travel_rad:.4f} rad."
            )


def user_openarm_gripper_calibration_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".cache"
    return root / "handumi" / "openarmv1_grippers.yaml"


def load_openarm_gripper_limits(
    path: Path | None = None,
) -> dict[str, OpenArmGripperLimits]:
    path = path or user_openarm_gripper_calibration_path()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}
    result: dict[str, OpenArmGripperLimits] = {}
    for side in ("left", "right"):
        values = data.get(side) or {}
        if "closed_position_rad" not in values or "open_position_rad" not in values:
            continue
        limits = OpenArmGripperLimits(
            closed_position_rad=float(values["closed_position_rad"]),
            open_position_rad=float(values["open_position_rad"]),
        )
        limits.validate()
        result[side] = limits
    return result


def save_openarm_gripper_limits(
    side: str,
    limits: OpenArmGripperLimits,
    path: Path | None = None,
) -> Path:
    if side not in ("left", "right"):
        raise ValueError(f"Invalid OpenArm side: {side!r}.")
    limits.validate()
    path = path or user_openarm_gripper_calibration_path()
    existing = load_openarm_gripper_limits(path)
    existing[side] = limits
    data = {
        name: {
            "closed_position_rad": value.closed_position_rad,
            "open_position_rad": value.open_position_rad,
        }
        for name, value in existing.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    temporary.replace(path)
    return path


__all__ = [
    "OpenArmGripperLimits",
    "load_openarm_gripper_limits",
    "save_openarm_gripper_limits",
    "user_openarm_gripper_calibration_path",
]
