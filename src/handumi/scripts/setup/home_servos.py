"""Re-home HandUMI Feetech servos so each gripper range avoids the 0/4095 seam.

The Feetech encoder reports ``Present_Position`` modulo 4096. If a gripper's
travel sits on the 0/4095 wrap (as the right HandUMI gripper did), the readout
jumps a full revolution mid-stroke and the normalized/mm widths flip or
saturate. The robust fix is to re-home the servo: hold the shaft at the
gripper's MID-travel position and write the "middle position calibration"
(value 128 to ``Torque_Enable``), which stores a correction in EEPROM so the
mid-travel position reads 2048. The whole stroke then sits centred at 2048,
comfortably away from the seam.

Workflow (gripper disassembled is easiest):
  1. Move the servo shaft to roughly half of the gripper's travel.
  2. Run this script and press ENTER to capture/centre.
  3. Reassemble and recalibrate with ``handumi-calibrate-grippers calibrate``.

Usage
-----
::

    handumi-home-servos              # both sides
    handumi-home-servos --side right
"""

from __future__ import annotations

import argparse
import select
import sys
from pathlib import Path

from handumi.config import DEFAULT_RIG_CONFIG
from handumi.feetech.bus import FeetechBus
from handumi.feetech.calibration import (
    FeetechConfig,
    GripperCalibration,
    load_ports,
)

_ENCODER_CENTER = 2048
_OK_TOLERANCE = 150


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Centre HandUMI Feetech servos at 2048 (homing) to avoid the encoder seam."
    )
    parser.add_argument(
        "--rig-config",
        "--config",
        dest="rig_config",
        type=Path,
        default=DEFAULT_RIG_CONFIG,
        help="Rig file containing Feetech servo_id/port values.",
    )
    parser.add_argument(
        "--side",
        choices=["left", "right", "both"],
        default="both",
        help="Which gripper servo(s) to re-home.",
    )
    parser.add_argument("--interval-s", type=float, default=0.1)
    args = parser.parse_args()

    print(f"Using rig: {args.rig_config}")
    config = load_ports(args.rig_config)
    sides = ["left", "right"] if args.side == "both" else [args.side]
    for side in sides:
        calibration = getattr(config, side)
        port = _side_port(config, calibration, side)
        _home_side(
            side=side,
            port=port,
            calibration=calibration,
            baudrate=config.baudrate,
            protocol_version=config.protocol_version,
            interval_s=args.interval_s,
        )

    print(
        "\nDone. Reassemble the gripper(s), then recalibrate so closed/open match "
        "the new centred range:\n"
        "  handumi-calibrate-grippers calibrate"
    )


def _home_side(
    *,
    side: str,
    port: str,
    calibration: GripperCalibration,
    baudrate: int,
    protocol_version: int,
    interval_s: float,
) -> None:
    print(f"\n=== Homing {side} servo: port={port}, servo_id={calibration.servo_id} ===")
    with FeetechBus(port=port, baudrate=baudrate, protocol_version=protocol_version) as bus:
        try:
            bus.disable_torque(calibration.servo_id)
        except RuntimeError as exc:
            print(f"Warning: could not disable torque on {side} servo: {exc}")

        before = _watch_until_enter(
            bus,
            calibration.servo_id,
            f"Move the {side} servo to the gripper's MID-travel position",
            interval_s,
        )
        bus.set_middle_position(calibration.servo_id)
        after = bus.read_position(calibration.servo_id)

        delta = after - _ENCODER_CENTER
        status = "OK" if abs(delta) <= _OK_TOLERANCE else "CHECK"
        print(
            f"{side}: {before} -> {after} ticks "
            f"(target {_ENCODER_CENTER}, off by {delta:+d}) [{status}]"
        )
        if status == "CHECK":
            print(
                f"  WARNING: {side} did not land near {_ENCODER_CENTER}. The 128 middle "
                "calibration may not be supported on this servo/firmware, or the shaft "
                "moved during the write. Re-run, or set the offset manually."
            )


def _watch_until_enter(bus: FeetechBus, servo_id: int, prompt: str, interval_s: float) -> int:
    print(f"{prompt}. Press ENTER to capture and centre.")
    latest = bus.read_position(servo_id)
    while True:
        latest = bus.read_position(servo_id)
        sys.stdout.write(f"\r  ticks={latest:5d}")
        sys.stdout.flush()
        ready, _, _ = select.select([sys.stdin], [], [], interval_s)
        if ready:
            sys.stdin.readline()
            sys.stdout.write("\n")
            return latest


def _side_port(config: FeetechConfig, calibration: GripperCalibration, side: str) -> str:
    port = calibration.port or config.port
    if not port:
        raise SystemExit(f"{side} Feetech port is not configured.")
    return port


if __name__ == "__main__":
    main()
