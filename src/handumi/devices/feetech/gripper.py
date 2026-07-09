"""HandUMI gripper aperture sensing backed by Feetech servo encoders."""

from __future__ import annotations

from dataclasses import dataclass

from handumi.devices.feetech.bus import FeetechBus
from handumi.devices.feetech.calibration import FeetechConfig, GripperCalibration


_ENCODER_RESOLUTION = 4096
_HALF_TURN = _ENCODER_RESOLUTION // 2


@dataclass(frozen=True)
class GripperWidths:
    left: float
    right: float
    left_mm: float
    right_mm: float
    left_normalized: float
    right_normalized: float
    left_ticks: int
    right_ticks: int

    @classmethod
    def zero(cls) -> "GripperWidths":
        """All-zero widths, used when Feetech is skipped or unavailable."""
        return cls(
            left=0.0,
            right=0.0,
            left_mm=0.0,
            right_mm=0.0,
            left_normalized=0.0,
            right_normalized=0.0,
            left_ticks=0,
            right_ticks=0,
        )


def zero_gripper_widths() -> GripperWidths:
    """Backend-neutral zero widths (thin wrapper over :meth:`GripperWidths.zero`)."""
    return GripperWidths.zero()


class _EncoderUnwrapper:
    """Turn raw 0-4095 Feetech readings into a continuous tick stream.

    The servo reports ``Present_Position`` modulo 4096, so a gripper whose range
    crosses the 0/4095 seam (like the right HandUMI gripper) makes the raw value
    jump a full revolution between consecutive frames. We sample fast enough that
    real motion never exceeds half a turn per frame, so any jump larger than that
    is a wraparound we cancel by accumulating turns.

    The first frame is trusted as-is (``turns == 0``) rather than guessed from the
    calibration: any guess is ambiguous when the range hugs the seam, and a wrong
    guess latches the whole stream onto the wrong revolution. Start a recording
    with the grippers roughly closed (away from the seam) and tracking is exact.
    """

    def __init__(self) -> None:
        self._prev_raw: int | None = None
        self._turns = 0

    def __call__(self, raw: int) -> int:
        if self._prev_raw is not None:
            delta = raw - self._prev_raw
            if delta > _HALF_TURN:
                self._turns -= 1
            elif delta < -_HALF_TURN:
                self._turns += 1
        self._prev_raw = raw
        return raw + self._turns * _ENCODER_RESOLUTION


class FeetechGripperPair:
    def __init__(self, config: FeetechConfig) -> None:
        self.config = config
        left_port = _side_port(config, config.left)
        right_port = _side_port(config, config.right)
        self._buses: dict[str, FeetechBus] = {}
        for port in {left_port, right_port}:
            self._buses[port] = FeetechBus(
                port=port,
                baudrate=config.baudrate,
                protocol_version=config.protocol_version,
            )
        self._left_port = left_port
        self._right_port = right_port
        self._left_unwrap = _EncoderUnwrapper()
        self._right_unwrap = _EncoderUnwrapper()

    def open(self) -> None:
        for bus in self._buses.values():
            bus.open()

    def close(self) -> None:
        for bus in self._buses.values():
            bus.close()

    def __enter__(self) -> "FeetechGripperPair":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def read_normalized_widths(self) -> GripperWidths:
        left = _read_width(self._buses[self._left_port], self.config.left, self._left_unwrap)
        right = _read_width(self._buses[self._right_port], self.config.right, self._right_unwrap)
        return GripperWidths(
            left=left["width_m"],
            right=right["width_m"],
            left_mm=left["width_mm"],
            right_mm=right["width_mm"],
            left_normalized=left["normalized"],
            right_normalized=right["normalized"],
            left_ticks=left["ticks"],
            right_ticks=right["ticks"],
        )


def _read_width(
    bus: FeetechBus,
    calibration: GripperCalibration,
    unwrap: _EncoderUnwrapper,
) -> dict[str, float | int]:
    ticks = unwrap(bus.read_position(calibration.servo_id))
    normalized = calibration.normalized_width(ticks)
    width_mm = calibration.width_mm(ticks)
    return {
        "ticks": ticks,
        "normalized": normalized,
        "width_mm": width_mm,
        "width_m": width_mm / 1000.0,
    }


def _side_port(config: FeetechConfig, calibration: GripperCalibration) -> str:
    port = calibration.port or config.port
    if not port:
        raise ValueError(
            "Feetech port is not configured. Set a shared `port` or per-side `left.port` / `right.port`."
        )
    return port
