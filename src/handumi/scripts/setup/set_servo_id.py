"""Assign a new internal ID to one Feetech servo.

Use this only during rig setup, with exactly one servo connected to the target
bus unless you are certain there is no ID conflict.
"""

from __future__ import annotations

import argparse

from handumi.feetech.bus import FeetechBus
from handumi.feetech.calibration import default_config

MIN_SERVO_ID = 0
MAX_SERVO_ID = 253


def build_parser() -> argparse.ArgumentParser:
    defaults = default_config()
    parser = argparse.ArgumentParser(
        description=(
            "Change the internal Feetech servo ID stored in EEPROM. "
            "Connect only the servo you intend to modify."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/ttyUSB0.")
    parser.add_argument(
        "--old-id",
        type=int,
        default=None,
        help="Current servo ID. Omit to auto-detect when exactly one servo is connected.",
    )
    parser.add_argument("--new-id", type=int, required=True, help="New servo ID to write.")
    parser.add_argument("--baudrate", type=int, default=defaults.baudrate)
    parser.add_argument("--protocol-version", type=int, default=defaults.protocol_version)
    parser.add_argument("--scan-start-id", type=int, default=MIN_SERVO_ID)
    parser.add_argument("--scan-end-id", type=int, default=MAX_SERVO_ID)
    parser.add_argument("--yes", action="store_true", help="Do not prompt for confirmation.")
    return parser


def _validate_id(value: int, *, name: str) -> None:
    if not MIN_SERVO_ID <= value <= MAX_SERVO_ID:
        raise SystemExit(f"{name} must be in [{MIN_SERVO_ID}, {MAX_SERVO_ID}], got {value}.")


def _confirm(args: argparse.Namespace, *, old_id: int) -> None:
    if args.yes:
        return
    print("About to change one Feetech servo ID.")
    print(f"  port  : {args.port}")
    print(f"  old ID: {old_id}")
    print(f"  new ID: {args.new_id}")
    print()
    print("Connect only the servo you want to modify, or make sure no other servo")
    print("on this bus already uses either ID.")
    answer = input("Write the new ID? [y/N]: ").strip().lower()
    if answer != "y":
        raise SystemExit("Aborted; servo ID was not changed.")


def _resolve_old_id(args: argparse.Namespace, bus) -> int:
    if args.old_id is not None:
        _validate_id(args.old_id, name="--old-id")
        return int(args.old_id)

    _validate_id(args.scan_start_id, name="--scan-start-id")
    _validate_id(args.scan_end_id, name="--scan-end-id")
    if args.scan_start_id > args.scan_end_id:
        raise SystemExit("--scan-start-id must be <= --scan-end-id.")

    ids = bus.scan(range(args.scan_start_id, args.scan_end_id + 1))
    if len(ids) == 1:
        print(f"Detected servo ID {ids[0]} on {args.port}.")
        return int(ids[0])
    if not ids:
        raise SystemExit(
            f"No servo replied on {args.port}. "
            "Check power, wiring, baudrate, and the scan range."
        )
    raise SystemExit(
        f"Multiple servo IDs replied on {args.port}: {ids}. "
        "Connect only the target servo or pass --old-id explicitly."
    )


def set_servo_id(args: argparse.Namespace, *, bus_cls=FeetechBus) -> None:
    _validate_id(args.new_id, name="--new-id")

    with bus_cls(
        port=args.port,
        baudrate=args.baudrate,
        protocol_version=args.protocol_version,
    ) as bus:
        old_id = _resolve_old_id(args, bus)
        if old_id == args.new_id:
            print(f"Servo on {args.port} is already ID {args.new_id}; nothing to change.")
            return

        if not bus.ping(old_id):
            raise SystemExit(
                f"No servo replied at ID {old_id} on {args.port}. "
                "Check --port, --old-id, power, wiring, and baudrate."
            )
        if bus.ping(args.new_id):
            raise SystemExit(
                f"A servo already replies at new ID {args.new_id} on {args.port}. "
                "Choose a free ID or disconnect the other servo first."
            )

        _confirm(args, old_id=old_id)

        print(f"Writing servo ID {old_id} -> {args.new_id} on {args.port} ...")
        bus.write_servo_id(old_id, args.new_id)

        old_still_present = bus.ping(old_id)
        new_present = bus.ping(args.new_id)

    if not new_present:
        raise SystemExit(
            f"Write finished, but ID {args.new_id} did not reply. "
            "Power-cycle the servo and re-run handumi-setup-ports."
        )
    if old_still_present:
        print(
            f"Warning: ID {old_id} still replies. If multiple servos are connected, "
            "disconnect all but the target servo and verify again."
        )
    print("Servo ID updated.")
    print("Next: run `handumi-setup-ports` and update `configs/rig.yaml`.")


def main() -> None:
    set_servo_id(build_parser().parse_args())


if __name__ == "__main__":
    main()
