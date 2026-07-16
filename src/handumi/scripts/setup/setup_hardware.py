#!/usr/bin/env python3
"""Interactive hardware setup for novice HandUMI + real robot users."""

from __future__ import annotations

import argparse
from pathlib import Path

from handumi.config import DEFAULT_RIG_CONFIG
from handumi.feetech.calibration import (
    FeetechConfig,
    GripperCalibration,
    load_config,
    save_calibration,
    user_calibration_path,
)
from handumi.feetech.setup import ensure_feetech_serial_permissions, run_feetech_wizard
from handumi.calibration.control_tcp import calibration_path_for_robot_device
from handumi.real.backends import REAL_BACKEND_NAMES
from handumi.real.backends.setup import RobotSetupOptions, run_robot_setup
from handumi.real.can_setup import ensure_rig_config
from handumi.scripts.setup import calibrate_grippers, home_servos
from handumi.tracking.pico import prepare_pico_adb_session


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=REAL_BACKEND_NAMES, default="piper")
    parser.add_argument("--device", choices=("pico", "meta"), default="pico")
    parser.add_argument("--rig-config", type=Path, default=DEFAULT_RIG_CONFIG)
    parser.add_argument("--bitrate", type=int, default=1_000_000)
    parser.add_argument("--restart-ms", type=int, default=100)
    parser.add_argument("--dbitrate", type=int, default=5_000_000)
    parser.add_argument(
        "--skip-can-map",
        action="store_true",
        help="Use the existing rig.yaml CAN mapping instead of the mapping wizard.",
    )
    parser.add_argument(
        "--skip-can-repair",
        action="store_true",
        help="Do not run sudo/ip-link repair after mapping CAN.",
    )
    parser.add_argument(
        "--skip-feetech-map",
        action="store_true",
        help="Use the existing rig.yaml Feetech mapping instead of the replug wizard.",
    )
    parser.add_argument("--feetech-start-id", type=int, default=0)
    parser.add_argument("--feetech-end-id", type=int, default=20)
    parser.add_argument(
        "--skip-feetech-calibration",
        action="store_true",
        help="Do not guide servo homing / width calibration after Feetech mapping.",
    )
    parser.add_argument(
        "--force-feetech-calibration",
        action="store_true",
        help="Run Feetech homing / width calibration even if a cache already exists.",
    )
    parser.add_argument(
        "--skip-feetech-home",
        action="store_true",
        help="Skip the servo middle-position homing step during guided calibration.",
    )
    parser.add_argument("--feetech-calibration-config", type=Path, default=None)
    parser.add_argument("--feetech-max-width-mm", type=float, default=None)
    parser.add_argument("--left-feetech-max-width-mm", type=float, default=None)
    parser.add_argument("--right-feetech-max-width-mm", type=float, default=None)
    parser.add_argument("--feetech-calibration-interval-s", type=float, default=0.1)
    parser.add_argument(
        "--skip-pico",
        action="store_true",
        help="Skip ADB reverse and PICO keep-awake setup.",
    )
    parser.add_argument("--skip-adb-check", action="store_true")
    parser.add_argument(
        "--skip-openarm-motor-check",
        action="store_true",
        help="Skip the read-only J1-J8 OpenArm diagnostic.",
    )
    parser.add_argument(
        "--calibrate-openarm-zero",
        action="store_true",
        help="Explicitly run the vendor mechanical-zero procedure; this moves motors.",
    )
    parser.add_argument(
        "--openarm-zero-side",
        choices=("right", "left", "both"),
        default="both",
        help="Arm(s) calibrated by --calibrate-openarm-zero (default: both).",
    )
    parser.add_argument(
        "--controller-tcp-calibration",
        type=Path,
        default=None,
        help="Use an explicit Controller-to-TCP calibration file.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    ensure_rig_config(args.rig_config)

    if not args.skip_feetech_map:
        ensure_feetech_serial_permissions()

    run_robot_setup(
        RobotSetupOptions(
            robot=args.robot,
            rig_config=args.rig_config,
            bitrate=args.bitrate,
            dbitrate=args.dbitrate,
            restart_ms=args.restart_ms,
            skip_can_map=args.skip_can_map,
            skip_can_repair=args.skip_can_repair,
            skip_motor_check=args.skip_openarm_motor_check,
            calibrate_openarm_zero=args.calibrate_openarm_zero,
            openarm_zero_side=args.openarm_zero_side,
        )
    )
    print("Robot transport listo.")

    if not args.skip_feetech_map:
        run_feetech_wizard(
            rig_config=args.rig_config,
            start_id=args.feetech_start_id,
            end_id=args.feetech_end_id,
        )
        print("Feetech listo.")

    if not args.skip_feetech_calibration:
        ensure_feetech_calibration(args)

    if args.device == "pico" and not args.skip_pico:
        prepare_pico_adb_session(skip_adb_check=args.skip_adb_check)
        print("PICO listo por USB/ADB.")

    default_calibration_path, _ = calibration_path_for_robot_device(
        args.robot, args.device
    )
    calibration_path = args.controller_tcp_calibration or default_calibration_path
    if not calibration_path.exists():
        raise SystemExit(
            f"Missing {args.robot}/{args.device} Controller->TCP calibration: "
            f"{calibration_path}\n"
            "Capture the physical tool pivot calibration before real teleop."
        )

    print("\nSetup listo. Prueba:")
    command = f"  uv run handumi-teleop-real --device {args.device} --robot {args.robot}"
    if args.controller_tcp_calibration is not None:
        command += f" --controller-tcp-calibration {calibration_path}"
    print(command)


def ensure_feetech_calibration(args: argparse.Namespace) -> None:
    calibration_path = args.feetech_calibration_config or user_calibration_path()
    current = load_config(args.rig_config, calibration_path)
    sides = _calibration_sides(current, force=args.force_feetech_calibration)
    if not sides:
        print(f"Feetech calibration listo: {calibration_path}")
        return

    print("\nFeetech calibration falta o fue forzada.")
    print(f"Cache: {calibration_path}")
    print("Este paso guiado hara homing de servos y calibrara ancho open/closed.")
    answer = input("Continuar ahora? [Y/n]: ").strip().lower()
    if answer not in ("", "y", "yes", "s", "si"):
        raise SystemExit(
            "Feetech calibration pendiente. Puedes correr despues:\n"
            "  uv run handumi-setup-hardware --robot piper --device pico --skip-can-map"
        )

    results = {"left": current.left, "right": current.right}
    side_width = {
        "left": args.left_feetech_max_width_mm,
        "right": args.right_feetech_max_width_mm,
    }
    for side in sides:
        calibration = getattr(current, side)
        port = _side_port(current, calibration, side)
        if not args.skip_feetech_home:
            print(
                f"\nHoming {side}: coloca el gripper a mitad de recorrido. "
                "El script capturara ENTER y centrara el encoder."
            )
            home_servos._home_side(
                side=side,
                port=port,
                calibration=calibration,
                baudrate=current.baudrate,
                protocol_version=current.protocol_version,
                interval_s=args.feetech_calibration_interval_s,
            )
        closed, open_, width_mm = calibrate_grippers._calibrate_side(
            side=side,
            port=port,
            calibration=calibration,
            baudrate=current.baudrate,
            protocol_version=current.protocol_version,
            max_width_mm=side_width[side] or args.feetech_max_width_mm,
            interval_s=args.feetech_calibration_interval_s,
        )
        results[side] = GripperCalibration(
            calibration.servo_id, closed, open_, width_mm, calibration.port
        )

    saved = save_calibration(
        FeetechConfig(
            port=current.port,
            baudrate=current.baudrate,
            protocol_version=current.protocol_version,
            left=results["left"],
            right=results["right"],
        ),
        calibration_path,
    )
    print(f"Feetech calibration guardada: {saved}")


def _calibration_sides(config: FeetechConfig, *, force: bool) -> list[str]:
    if force:
        return ["left", "right"]
    return [side for side in ("left", "right") if not getattr(config, side).is_complete]


def _side_port(
    config: FeetechConfig, calibration: GripperCalibration, side: str
) -> str:
    port = calibration.port or config.port
    if not port:
        raise SystemExit(f"{side} Feetech port is not configured.")
    return port


if __name__ == "__main__":
    main()
