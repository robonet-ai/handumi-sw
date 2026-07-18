"""Calibrate the physical closed/open J8 endpoints of one OpenArm v1 gripper."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from handumi.config import DEFAULT_RIG_CONFIG
from handumi.real.can_setup import ensure_can_fd_interfaces_ready
from handumi.real.openarm_can import (
    GRIPPER_RECV_CAN_ID,
    GRIPPER_SEND_CAN_ID,
    load_openarm_settings,
    require_openarm_can,
)
from handumi.real.openarm_gripper_calibration import (
    OpenArmGripperLimits,
    save_openarm_gripper_limits,
    user_openarm_gripper_calibration_path,
)
from handumi.robots.registry import load_robot_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", required=True, choices=("left", "right"))
    parser.add_argument("--rig-config", type=Path, default=DEFAULT_RIG_CONFIG)
    parser.add_argument("--calibration-config", type=Path, default=None)
    parser.add_argument("--skip-can-repair", action="store_true")
    parser.add_argument(
        "--automatic",
        action="store_true",
        help=(
            "Experimentally seek endpoints using motor torque. The default "
            "manual capture keeps J8 disabled and is safer and more reliable."
        ),
    )
    parser.add_argument("--step-deg", type=float, default=0.2)
    parser.add_argument("--max-travel-deg", type=float, default=90.0)
    parser.add_argument("--torque-threshold", type=float, default=0.3)
    parser.add_argument("--velocity-threshold", type=float, default=0.3)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    settings = load_openarm_settings(
        args.rig_config, load_robot_config("openarmv1").real_options
    )
    port = settings.left_port if args.side == "left" else settings.right_port
    output = args.calibration_config or user_openarm_gripper_calibration_path()

    print(f"OpenArm v1 {args.side} J8 calibration on {port}.")
    if args.automatic:
        print("Only J8 will be enabled; J1-J7 are not registered or commanded.")
    else:
        print("J8 will remain disabled while its position is read; move the jaws by hand.")
    print("Remove objects from the jaws and keep the emergency stop reachable.")
    expected = f"CALIBRATE {args.side.upper()} J8"
    answer = input(f"Type {expected}: ").strip()
    if answer != expected:
        raise SystemExit("Calibration cancelled; no motor command was sent.")

    ensure_can_fd_interfaces_ready(
        [port],
        bitrate=settings.bitrate,
        dbitrate=settings.dbitrate,
        repair=not args.skip_can_repair,
    )
    sdk = require_openarm_can()
    if args.automatic:
        limits = calibrate_side(
            port=port,
            sdk=sdk,
            step_rad=math.radians(args.step_deg),
            max_travel_rad=math.radians(args.max_travel_deg),
            torque_threshold=args.torque_threshold,
            velocity_threshold=args.velocity_threshold,
        )
    else:
        limits = calibrate_side_manually(port=port, sdk=sdk, side=args.side)
    saved = save_openarm_gripper_limits(args.side, limits, output)
    print(
        f"Measured {args.side}: closed={limits.closed_position_rad:+.5f} rad, "
        f"open={limits.open_position_rad:+.5f} rad, "
        f"travel={math.degrees(limits.travel_rad):.2f} deg."
    )
    print(f"Saved: {saved}")
    print("The next handumi-teleop-real run will load these endpoints automatically.")


def calibrate_side_manually(
    *,
    port: str,
    sdk: ModuleType,
    side: str,
    input_fn: Any = input,
) -> OpenArmGripperLimits:
    """Capture useful physical endpoints while J8 remains torque-disabled."""
    arm = sdk.OpenArm(port, True)
    arm.init_gripper_motor(
        sdk.MotorType.DM4310,
        GRIPPER_SEND_CAN_ID,
        GRIPPER_RECV_CAN_ID,
        sdk.ControlMode.MIT,
    )
    arm.set_callback_mode_all(sdk.CallbackMode.STATE)
    arm.disable_all()
    arm.recv_all(1_000)
    try:
        input_fn(
            f"Manually close the {side} jaws until they just touch, without "
            "forcing the linkage. Press ENTER to capture: "
        )
        closed = _read_stable_position(arm)
        print(f"  captured closed={closed:+.5f} rad")
        input_fn(
            f"Manually open the {side} jaws to the maximum useful opening. "
            "Press ENTER to capture: "
        )
        open_ = _read_stable_position(arm)
        print(f"  captured open={open_:+.5f} rad")
        limits = OpenArmGripperLimits(closed, open_)
        limits.validate()
        return limits
    finally:
        arm.disable_all()
        arm.recv_all(1_000)


def _read_stable_position(arm: Any) -> float:
    # The user may move J8 a full stroke while this process is blocked in
    # input(). The first responses can therefore still represent the previous
    # endpoint. Refresh and discard a short window before measuring stability.
    for _ in range(5):
        arm.refresh_all()
        time.sleep(0.01)
        arm.recv_all(2_000)
        arm.get_gripper().get_motors()[0].get_position()

    readings: list[float] = []
    for _ in range(7):
        arm.refresh_all()
        time.sleep(0.01)
        arm.recv_all(2_000)
        readings.append(float(arm.get_gripper().get_motors()[0].get_position()))
        time.sleep(0.02)
    spread = max(readings) - min(readings)
    if spread > math.radians(0.5):
        raise RuntimeError(
            f"J8 moved {math.degrees(spread):.2f} deg while capturing. "
            "Hold the jaws still and retry."
        )
    return float(np.median(readings))


def calibrate_side(
    *,
    port: str,
    sdk: ModuleType,
    step_rad: float = math.radians(0.2),
    max_travel_rad: float = math.radians(90.0),
    torque_threshold: float = 0.3,
    velocity_threshold: float = 0.3,
) -> OpenArmGripperLimits:
    if step_rad <= 0.0 or max_travel_rad <= 0.0:
        raise ValueError("Calibration step and maximum travel must be positive.")

    arm = sdk.OpenArm(port, True)
    arm.init_gripper_motor(
        sdk.MotorType.DM4310,
        GRIPPER_SEND_CAN_ID,
        GRIPPER_RECV_CAN_ID,
        sdk.ControlMode.MIT,
    )
    arm.set_callback_mode_all(sdk.CallbackMode.STATE)
    arm.enable_all()
    time.sleep(0.1)
    arm.recv_all(2_000)
    gripper = arm.get_gripper()
    try:
        print("Seeking fully closed stop (+J8) ...")
        closed = _seek_stop(
            arm,
            gripper,
            sdk,
            direction=1.0,
            step_rad=step_rad,
            max_travel_rad=max_travel_rad,
            torque_threshold=torque_threshold,
            velocity_threshold=velocity_threshold,
        )
        time.sleep(0.3)
        print("Seeking fully open stop (-J8) ...")
        open_ = _seek_stop(
            arm,
            gripper,
            sdk,
            direction=-1.0,
            step_rad=step_rad,
            max_travel_rad=max_travel_rad,
            torque_threshold=torque_threshold,
            velocity_threshold=velocity_threshold,
        )
        limits = OpenArmGripperLimits(
            closed_position_rad=closed,
            open_position_rad=open_,
        )
        limits.validate()
        print("Returning J8 slowly to the measured closed position ...")
        _move_to(arm, gripper, sdk, closed, speed_rad_s=0.35)
        return limits
    finally:
        arm.disable_all()
        arm.recv_all(1_000)


def _seek_stop(
    arm: Any,
    gripper: Any,
    sdk: ModuleType,
    *,
    direction: float,
    step_rad: float,
    max_travel_rad: float,
    torque_threshold: float,
    velocity_threshold: float,
) -> float:
    arm.refresh_all()
    arm.recv_all(2_000)
    # openarm_can returns Motor values by copy. Fetch a fresh snapshot after
    # every response; retaining one object would freeze all feedback fields.
    start = float(gripper.get_motors()[0].get_position())
    target = start
    consecutive_hits = 0
    max_steps = int(math.ceil(max_travel_rad / step_rad))
    latest_position = start
    maximum_torque = 0.0
    maximum_excursion = 0.0
    last_reported_bucket = -1
    # Once a possible stop is seen, hold the same target while confirming it;
    # do not keep winding the requested position farther into the mechanism.
    advances = 0
    while advances < max_steps or consecutive_hits:
        if consecutive_hits == 0:
            target += math.copysign(step_rad, direction)
            advances += 1
        # Match the vendor v1 bump calibration gains and thresholds for J8.
        gripper.mit_control_one(0, sdk.MITParam(45.0, 1.2, target, 0.0, 0.0))
        arm.recv_all(2_000)
        motor = gripper.get_motors()[0]
        latest_position = float(motor.get_position())
        velocity = abs(float(motor.get_velocity()))
        torque = abs(float(motor.get_torque()))
        maximum_torque = max(maximum_torque, torque)
        maximum_excursion = max(maximum_excursion, abs(latest_position - start))
        bucket = int(math.degrees(abs(target - start)) // 5.0)
        if bucket > last_reported_bucket:
            last_reported_bucket = bucket
            print(
                f"  command={target:+.4f} measured={latest_position:+.4f} rad "
                f"velocity={velocity:.3f} torque={torque:.3f}"
            )
        if velocity < velocity_threshold and torque > torque_threshold:
            consecutive_hits += 1
            if consecutive_hits >= 5:
                return latest_position
        else:
            consecutive_hits = 0
        time.sleep(0.005)
    raise RuntimeError(
        f"J8 mechanical stop was not detected within "
        f"{math.degrees(max_travel_rad):.1f} commanded degrees "
        f"(measured excursion={math.degrees(maximum_excursion):.1f} deg, "
        f"max torque={maximum_torque:.3f}, final={latest_position:+.4f} rad). "
        "Motor was disabled; check assembly, control mode, feedback, and zero calibration."
    )


def _move_to(
    arm: Any,
    gripper: Any,
    sdk: ModuleType,
    target: float,
    *,
    speed_rad_s: float,
) -> None:
    motor = gripper.get_motors()[0]
    start = float(motor.get_position())
    duration = max(0.25, abs(target - start) / speed_rad_s)
    steps = max(2, int(math.ceil(duration / 0.01)))
    for value in np.linspace(start, target, steps):
        gripper.mit_control_one(
            0, sdk.MITParam(10.0, 0.9, float(value), 0.0, 0.0)
        )
        arm.recv_all(2_000)
        time.sleep(duration / steps)


if __name__ == "__main__":
    main()
