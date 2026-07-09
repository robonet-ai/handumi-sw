"""Calibration helpers for HandUMI Feetech gripper encoders.

Two concerns, two homes:

- **Ports** (``servo_id``, ``port``) are per-machine wiring that can change on
  every replug/reboot. They live in the tracked ``configs/feetech.yaml``,
  edited directly — the same pattern as ``configs/cameras.yaml``.
- **Calibration** (``closed_ticks``, ``open_ticks``, ``max_width_mm``) is a
  measured property of the physical gripper that rarely changes once set. It
  lives in a per-user cache so it is never committed and each laptop
  calibrates once. See :func:`user_calibration_path`.

:func:`load_config` merges both into the single :class:`FeetechConfig` the
rest of the codebase (bus/gripper/recorders) works with.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PORTS_PATH = Path("configs/feetech.yaml")


def user_calibration_path() -> Path:
    """Per-user calibration cache: ``$XDG_CACHE_HOME/handumi/calibration.yaml``
    (falling back to ``~/.cache/handumi/calibration.yaml``)."""
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".cache"
    return root / "handumi" / "calibration.yaml"


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


@dataclass(frozen=True)
class FeetechConfig:
    port: str | None
    baudrate: int
    protocol_version: int
    left: GripperCalibration
    right: GripperCalibration


def default_config() -> FeetechConfig:
    return FeetechConfig(
        port=None,
        baudrate=1_000_000,
        protocol_version=0,
        left=GripperCalibration(servo_id=0),
        right=GripperCalibration(servo_id=1),
    )


def assert_calibrated(config: FeetechConfig, *, source: Path | None = None) -> None:
    """Fail fast with an actionable message if either gripper is uncalibrated.

    Call this at startup (before the record/monitor loop) so an uncalibrated rig
    is reported clearly instead of crashing mid-run inside width computation.
    """
    missing = [side for side in ("left", "right") if not getattr(config, side).is_complete]
    if not missing:
        return
    where = f" in {source}" if source else ""
    raise SystemExit(
        f"Feetech calibration is incomplete for {', '.join(missing)}{where}.\n"
        "Calibrate this laptop first (see README_gripper.md):\n"
        "  python scripts/setup/setup_ports.py           # set servo_id/port\n"
        "  python scripts/setup/home_servos.py           # centre the encoder range\n"
        "  python scripts/setup/calibrate_grippers.py calibrate\n"
        "Or pass --skip-feetech to run without gripper widths."
    )


def load_ports(path: Path = PORTS_PATH) -> FeetechConfig:
    """Load servo_id/port/baudrate/protocol_version from the tracked ports file.

    Calibration fields (closed/open ticks, max_width_mm) are always ``None``
    here; use :func:`load_config` to get the full, merged runtime config.
    """
    if not path.exists():
        return default_config()
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return FeetechConfig(
        port=_optional_str(data.get("port")),
        baudrate=int(data.get("baudrate", 1_000_000)),
        protocol_version=int(data.get("protocol_version", 0)),
        left=_port_only_calibration(data.get("left", {}), default_servo_id=0),
        right=_port_only_calibration(data.get("right", {}), default_servo_id=1),
    )


def load_calibration_values(path: Path) -> dict[str, dict[str, Any]]:
    """Load ``{side: {closed_ticks, open_ticks, max_width_mm}}`` from the
    per-user calibration cache. Missing file/side -> empty dict (uncalibrated)."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return {side: data.get(side) or {} for side in ("left", "right")}


def load_config(
    ports_path: Path = PORTS_PATH, calibration_path: Path | None = None
) -> FeetechConfig:
    """Merge the tracked ports file with the per-user calibration cache into
    the full runtime :class:`FeetechConfig`."""
    ports = load_ports(ports_path)
    values = load_calibration_values(calibration_path or user_calibration_path())

    def merged(side_ports: GripperCalibration, side: str) -> GripperCalibration:
        v = values.get(side, {})
        return GripperCalibration(
            servo_id=side_ports.servo_id,
            port=side_ports.port,
            closed_ticks=_optional_int(v.get("closed_ticks")),
            open_ticks=_optional_int(v.get("open_ticks")),
            max_width_mm=_read_max_width_mm(v),
        )

    return FeetechConfig(
        port=ports.port,
        baudrate=ports.baudrate,
        protocol_version=ports.protocol_version,
        left=merged(ports.left, "left"),
        right=merged(ports.right, "right"),
    )


def save_calibration(config: FeetechConfig, path: Path | None = None) -> Path:
    """Persist only the measured calibration (closed/open ticks, max_width_mm)
    to the per-user cache. Ports are untouched — they live in ``configs/feetech.yaml``."""
    path = path or user_calibration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        side: {
            "closed_ticks": getattr(config, side).closed_ticks,
            "open_ticks": getattr(config, side).open_ticks,
            "max_width_mm": getattr(config, side).max_width_mm,
        }
        for side in ("left", "right")
    }
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    return path


def _port_only_calibration(data: dict[str, Any], *, default_servo_id: int) -> GripperCalibration:
    return GripperCalibration(
        servo_id=int(data.get("servo_id", default_servo_id)),
        port=_optional_str(data.get("port")),
    )


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
