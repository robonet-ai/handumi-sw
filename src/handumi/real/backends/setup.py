"""Robot-specific setup steps behind one wizard-facing registry."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from handumi.real.can_setup import (
    ensure_can_fd_interfaces_ready,
    ensure_can_interfaces_ready,
    run_openarm_can_wizard,
    run_piper_can_wizard,
)
from handumi.real.openarm_can import load_openarm_settings, require_openarm_can
from handumi.real.piper_can import load_piper_can_settings
from handumi.robots.registry import load_robot_config


@dataclass(frozen=True)
class RobotSetupOptions:
    robot: str
    rig_config: Path
    bitrate: int
    dbitrate: int
    restart_ms: int
    skip_can_map: bool
    skip_can_repair: bool
    skip_motor_check: bool
    calibrate_openarm_zero: bool
    openarm_zero_side: str = "both"


def run_robot_setup(options: RobotSetupOptions) -> None:
    handlers = {
        "piper": _setup_piper,
        "openarmv1": _setup_openarm,
    }
    try:
        handler = handlers[options.robot]
    except KeyError as exc:
        raise SystemExit(f"No setup provider for {options.robot!r}.") from exc
    handler(options)


def _setup_piper(options: RobotSetupOptions) -> None:
    if not options.skip_can_map:
        run_piper_can_wizard(
            rig_config=options.rig_config,
            bitrate=options.bitrate,
            restart_ms=options.restart_ms,
        )
    settings = load_piper_can_settings(options.rig_config)
    ensure_can_interfaces_ready(
        [settings.left_port, settings.right_port],
        bitrate=settings.bitrate,
        restart_ms=settings.restart_ms,
        repair=not options.skip_can_repair,
    )


def _setup_openarm(options: RobotSetupOptions) -> None:
    if shutil.which("openarm-can-cli") is None:
        raise SystemExit(
            "Missing openarm-can-cli. Install libopenarm-can-dev and "
            "openarm-can-utils from the official OpenArm repository."
        )
    try:
        require_openarm_can()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if not options.skip_can_map:
        run_openarm_can_wizard(
            rig_config=options.rig_config,
            bitrate=options.bitrate,
            dbitrate=options.dbitrate,
        )
    settings = load_openarm_settings(
        options.rig_config, load_robot_config("openarmv1").real_options
    )
    ensure_can_fd_interfaces_ready(
        [settings.left_port, settings.right_port],
        bitrate=settings.bitrate,
        dbitrate=settings.dbitrate,
        repair=not options.skip_can_repair,
    )
    if not options.skip_motor_check:
        for side, port in (
            ("right", settings.right_port),
            ("left", settings.left_port),
        ):
            _check_openarm_motors(
                side,
                port,
                bitrate=settings.bitrate,
                dbitrate=settings.dbitrate,
            )
    if options.calibrate_openarm_zero:
        calibration_executable = shutil.which(
            "openarm-can-zero-position-calibration"
        )
        if calibration_executable is None:
            raise SystemExit(
                "Missing openarm-can-zero-position-calibration from OpenArm v1."
            )
        selected = {
            "right": ("right_arm", settings.right_port),
            "left": ("left_arm", settings.left_port),
        }
        if options.openarm_zero_side not in (*selected, "both"):
            raise SystemExit(
                f"Invalid OpenArm zero side: {options.openarm_zero_side}."
            )
        side_names = (
            ("right", "left")
            if options.openarm_zero_side == "both"
            else (options.openarm_zero_side,)
        )
        for side_name in side_names:
            side, port = selected[side_name]
            answer = input(
                f"OpenArm {side_name} zero calibration on {port} moves joints "
                "to mechanical stops. Put this arm near the official zero pose, "
                "close its gripper, clear the workspace, and hold the emergency "
                f"stop ready. Type CALIBRATE {side_name.upper()}: "
            ).strip()
            if answer != f"CALIBRATE {side_name.upper()}":
                raise SystemExit(
                    f"OpenArm {side_name} zero calibration cancelled; "
                    "no calibration command was sent for this arm."
                )
            _run_openarm_zero_calibration(
                side,
                port,
                executable=calibration_executable,
            )


def _run_openarm_zero_calibration(side: str, port: str, *, executable: str) -> None:
    """Run the OpenArm v1 calibrator inside the active HandUMI Python env."""
    subprocess.run(
        [
            sys.executable,
            executable,
            "--canport",
            port,
            "--arm-side",
            side,
        ],
        check=True,
    )


def _check_openarm_motors(
    side: str,
    port: str,
    *,
    bitrate: int = 1_000_000,
    dbitrate: int = 5_000_000,
) -> None:
    """Read motor parameters without enabling motor output."""
    result = subprocess.run(
        [
            "openarm-can-cli",
            "-i",
            port,
            "show_param",
            "--id",
            "1,2,3,4,5,6,7,8",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15.0,
    )
    output = f"{result.stdout or ''}\n{result.stderr or ''}"
    responded = output.count("MOTOR ID:")
    if result.returncode != 0 or responded != 8 or "NO RESPONSE FROM MOTOR" in output:
        raise SystemExit(
            f"OpenArm {side} motor diagnostic failed on {port}. Expected J1-J8 "
            f"at CAN-FD {bitrate}/{dbitrate} bps. Restore the interface with "
            f"'openarm-can-cli -i {port} can_configure' and verify motor IDs and "
            f"internal 5 Mbps baudrate before retrying:\n{output.strip()}"
        )
    print(
        f"OpenArm {side}: J1-J8 responded on {port} at configured "
        f"CAN-FD {bitrate}/{dbitrate} bps."
    )


__all__ = ["RobotSetupOptions", "run_robot_setup"]
