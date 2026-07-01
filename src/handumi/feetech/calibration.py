"""Calibration helpers for HandUMI Feetech gripper encoders."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# The repo ships a checked-in template; the real, machine-specific calibration
# (ports + tick ranges) lives in a per-user cache so it is never committed and
# each laptop calibrates once. See resolve_config_path().
REPO_TEMPLATE_PATH = Path("configs/feetech.yaml")


def user_config_path() -> Path:
    """Per-user Feetech config path: ``$XDG_CACHE_HOME/handumi/feetech.yaml``
    (falling back to ``~/.cache/handumi/feetech.yaml``)."""
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".cache"
    return root / "handumi" / "feetech.yaml"


def resolve_config_path(explicit: Path | None = None, *, seed: bool = False) -> Path:
    """Resolve which Feetech config file to use.

    Precedence: explicit ``--config`` override > per-user cache > repo template.
    With ``seed=True`` (used by the setup/calibration tools that write back), the
    user cache is created from the repo template on first use so there is always
    a writable, machine-local file to calibrate into.
    """
    if explicit is not None:
        return explicit
    cache = user_config_path()
    if cache.exists():
        return cache
    if seed:
        cache.parent.mkdir(parents=True, exist_ok=True)
        if REPO_TEMPLATE_PATH.exists():
            shutil.copy(REPO_TEMPLATE_PATH, cache)
        return cache
    return REPO_TEMPLATE_PATH


@dataclass(frozen=True)
class GripperCalibration:
    servo_id: int
    closed_ticks: int | None = None
    open_ticks: int | None = None
    max_width_mm: float | None = None
    port: str | None = None

    @property
    def is_complete(self) -> bool:
        return (
            self.closed_ticks is not None
            and self.open_ticks is not None
            and self.max_width_mm is not None
        )

    def normalized_width(self, ticks: int) -> float:
        if self.closed_ticks is None or self.open_ticks is None:
            raise ValueError(f"Servo {self.servo_id} is not calibrated.")
        span = self.open_ticks - self.closed_ticks
        if span == 0:
            raise ValueError(f"Servo {self.servo_id} has identical open/closed ticks.")
        return float(min(1.0, max(0.0, (ticks - self.closed_ticks) / span)))

    def width_mm(self, ticks: int) -> float:
        if self.max_width_mm is None:
            raise ValueError(f"Servo {self.servo_id} is missing max_width_mm.")
        return self.normalized_width(ticks) * self.max_width_mm

    def width_m(self, ticks: int) -> float:
        return self.width_mm(ticks) / 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "servo_id": self.servo_id,
            "port": self.port,
            "closed_ticks": self.closed_ticks,
            "open_ticks": self.open_ticks,
            "max_width_mm": self.max_width_mm,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GripperCalibration":
        return cls(
            servo_id=int(data["servo_id"]),
            port=_optional_str(data.get("port")),
            closed_ticks=_optional_int(data.get("closed_ticks")),
            open_ticks=_optional_int(data.get("open_ticks")),
            max_width_mm=_read_max_width_mm(data),
        )


@dataclass(frozen=True)
class FeetechConfig:
    port: str | None
    baudrate: int
    protocol_version: int
    left: GripperCalibration
    right: GripperCalibration

    def to_dict(self) -> dict[str, Any]:
        return {
            "port": self.port,
            "baudrate": self.baudrate,
            "protocol_version": self.protocol_version,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }


def default_config() -> FeetechConfig:
    return FeetechConfig(
        port=None,
        baudrate=1_000_000,
        protocol_version=0,
        left=GripperCalibration(servo_id=0),
        right=GripperCalibration(servo_id=1),
    )


def load_config(path: Path) -> FeetechConfig:
    if not path.exists():
        return default_config()
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return FeetechConfig(
        port=_optional_str(data.get("port")),
        baudrate=int(data.get("baudrate", 1_000_000)),
        protocol_version=int(data.get("protocol_version", 0)),
        left=GripperCalibration.from_dict(data.get("left", {"servo_id": 0})),
        right=GripperCalibration.from_dict(data.get("right", {"servo_id": 1})),
    )


def save_config(config: FeetechConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config.to_dict(), fh, sort_keys=False)


def update_side(
    config: FeetechConfig,
    *,
    side: str,
    calibration: GripperCalibration,
) -> FeetechConfig:
    if side == "left":
        return FeetechConfig(
            config.port,
            config.baudrate,
            config.protocol_version,
            calibration,
            config.right,
        )
    if side == "right":
        return FeetechConfig(
            config.port,
            config.baudrate,
            config.protocol_version,
            config.left,
            calibration,
        )
    raise ValueError(f"Unknown side {side!r}; expected 'left' or 'right'.")


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _read_max_width_mm(data: dict[str, Any]) -> float | None:
    if "max_width_mm" in data:
        return _optional_float(data.get("max_width_mm"))
    max_width_m = _optional_float(data.get("max_width_m"))
    return None if max_width_m is None else max_width_m * 1000.0
