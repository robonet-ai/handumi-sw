#!/usr/bin/env python3
"""Run live HandUMI teleop on a registered real-robot backend.

The controller TCP pose is anchored at the robot home TCP, then relative
controller motion drives the IK target. The IK solution ``q`` is the source of
truth; the selected backend converts it to hardware commands and streams them
over CAN.

Safety behavior:

* controller->TCP calibration is required;
* the robot homes before teleop starts;
* arms stay idle until double-clap or explicit ``--space-start``;
* double-clap during teleop clears anchors and returns home.

Examples:

    handumi teleop-real --device pico --robot piper
    handumi teleop-real --device pico --robot piper --space-start
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from handumi.config import DEFAULT_RIG_CONFIG
from handumi.feetech.setup import list_feetech_serial_ports
from handumi.real.registry import REAL_BACKEND_NAMES, make_real_backend
from handumi.robots.registry import load_embodiment, resolve_home_q
from handumi.teleop.common import (
    DEFAULT_TELEOP_FPS,
    SIDE_CHOICES,
    KeyboardSpaceListener,
    TeleopLoopTimer,
    TeleopMotionSmoother,
    enabled_sides as _enabled_sides,
    enabled_tracking_ok as _enabled_tracking_ok,
    latest_widths as _latest_widths,
    tracking_world_map as _tracking_world_map,
)
from handumi.teleop.core import TeleopController
from handumi.teleop.session import TeleopSession
from handumi.teleop.trajectory import DelayedJointCommandPlayer
from handumi.teleop.hardware import (
    load_required_controller_tcp_calibration as _load_required_calibration,
    validate_feetech_ports_exist,
    validate_feetech_ready as _validate_feetech_ready,
)
from handumi.teleop.tracking import TrackingRecoveryPolicy
from handumi.tracking.gestures import DoubleClapDetector
from handumi.utils.speech import log_say

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
real_log = logging.getLogger("handumi.teleop_real")

DEFAULT_REAL_SMOOTHING_TIME_CONSTANT_S = 0.05
DEFAULT_REAL_POSITION_DEADBAND_MM = 0.5
DEFAULT_REAL_ORIENTATION_DEADBAND_DEG = 0.25
DEFAULT_REAL_COMMAND_RATE_HZ = 100.0
DEFAULT_REAL_TRAJECTORY_DELAY_MS = 80.0


def _parse_real_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    show_advanced = "--help-advanced" in raw_argv
    raw_argv = [value for value in raw_argv if value != "--help-advanced"]
    parser = argparse.ArgumentParser(
        description="Teleoperate a supported physical robot with HandUMI."
    )
    parser.add_argument(
        "--help-advanced", action="store_true", help="Show expert hardware options."
    )
    parser.add_argument("--device", choices=("pico", "meta"), required=True)
    parser.add_argument("--robot", choices=REAL_BACKEND_NAMES, default="piper")
    parser.add_argument(
        "--home-pose",
        default=None,
        help="Override a legacy named home pose. Omit to use the robot home_q.",
    )
    parser.add_argument("--side", choices=SIDE_CHOICES, default="both")
    parser.add_argument("--fps", type=int, default=DEFAULT_TELEOP_FPS)
    parser.add_argument(
        "--command-rate-hz",
        type=float,
        default=DEFAULT_REAL_COMMAND_RATE_HZ,
        help="Fixed-rate playback frequency for interpolated joint commands.",
    )
    parser.add_argument(
        "--trajectory-delay-ms",
        type=float,
        default=DEFAULT_REAL_TRAJECTORY_DELAY_MS,
        help="Playback delay used to bracket and interpolate IK results.",
    )
    parser.add_argument(
        "--motion-smoothing-time-constant-s",
        type=float,
        default=DEFAULT_REAL_SMOOTHING_TIME_CONSTANT_S,
        help="TCP and post-IK low-pass time constant; 0 disables it.",
    )
    parser.add_argument(
        "--motion-position-deadband-mm",
        type=float,
        default=DEFAULT_REAL_POSITION_DEADBAND_MM,
        help="Ignore controller translation jitter below this distance.",
    )
    parser.add_argument(
        "--motion-orientation-deadband-deg",
        type=float,
        default=DEFAULT_REAL_ORIENTATION_DEADBAND_DEG,
        help="Ignore controller rotation jitter below this angle.",
    )
    parser.add_argument(
        "--duration-s", type=float, default=0.0, help="0 means run until Ctrl+C."
    )
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=1.0,
        help="Scale HandUMI translation deltas before applying them to the robot TCP.",
    )
    parser.add_argument(
        "--space-start",
        action="store_true",
        help="Allow keyboard Space to start any unanchored enabled arms.",
    )
    parser.add_argument(
        "--no-sounds", action="store_true", help="Disable spoken feedback."
    )
    parser.add_argument(
        "--controller-tcp-calibration",
        type=Path,
        default=None,
        help="Override the robot/device Controller->TCP setup calibration.",
    )
    parser.add_argument(
        "--rig-config",
        type=Path,
        default=DEFAULT_RIG_CONFIG,
        help="Machine-local Feetech, tracking, and robot CAN configuration.",
    )

    # Shared Feetech options.
    parser.add_argument("--feetech-port", type=str, default=None)
    parser.add_argument("--skip-feetech", action="store_true")

    # Shared tracking options consumed by build_tracker.
    parser.add_argument("--quest-ip", type=str, default=None)
    parser.add_argument("--tcp-port", type=int, default=None)
    parser.add_argument("--sync-port", type=int, default=None)
    parser.add_argument(
        "--pico-mode", choices=("mandos", "object", "whole-body"), default="mandos"
    )
    pico_transport = parser.add_mutually_exclusive_group()
    pico_transport.add_argument("--pico-adb", action="store_true")
    pico_transport.add_argument("--pico-wifi", action="store_true")
    parser.add_argument("--skip-adb-check", action="store_true")
    parser.add_argument(
        "--skip-can-repair",
        action="store_true",
        help="Validate but do not auto-repair CAN with sudo before connecting.",
    )
    if not show_advanced:
        normal = {
            "help",
            "help_advanced",
            "device",
            "robot",
            "side",
            "space_start",
            "no_sounds",
        }
        for action in parser._actions:
            if action.dest not in normal:
                action.help = argparse.SUPPRESS
    else:
        parser.print_help()
        raise SystemExit(0)
    return parser.parse_args(raw_argv)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _parse_real_args(argv)


def _validate_real_args(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise SystemExit("--fps must be > 0.")
    if args.duration_s < 0.0:
        raise SystemExit("--duration-s must be >= 0.")
    if args.command_rate_hz <= 0.0:
        raise SystemExit("--command-rate-hz must be > 0.")
    if args.trajectory_delay_ms < 0.0:
        raise SystemExit("--trajectory-delay-ms must be >= 0.")
    if args.motion_smoothing_time_constant_s < 0.0:
        raise SystemExit("--motion-smoothing-time-constant-s must be >= 0.")
    if args.motion_position_deadband_mm < 0.0:
        raise SystemExit("--motion-position-deadband-mm must be >= 0.")
    if args.motion_orientation_deadband_deg < 0.0:
        raise SystemExit("--motion-orientation-deadband-deg must be >= 0.")
    if args.skip_feetech and not args.space_start:
        raise SystemExit(
            "--skip-feetech disables double-clap; add --space-start so teleop can begin."
        )


def _validate_feetech_ports_exist(feetech_config, *, robot: str = "piper") -> None:
    return validate_feetech_ports_exist(
        feetech_config,
        robot=robot,
        list_ports=list_feetech_serial_ports,
    )


def _run_real() -> None:
    args = _parse_real_args()
    _validate_real_args(args)
    from handumi.scripts.record import build_tracker, connect_feetech

    real_log.info("Loading %s IK solver.", args.robot)
    runtime = load_embodiment(args.robot)
    try:
        home_pose_name, home_q = resolve_home_q(
            runtime, rig_config=args.rig_config, explicit_name=args.home_pose
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    controller = TeleopController(
        runtime,
        home_q=home_q,
        enabled_sides=_enabled_sides(args.side),
        source_world_to_robot_world=_tracking_world_map(args.device),
        translation_scale=args.translation_scale,
    )
    q = home_q.copy()
    real_log.info("Selected home pose: %s", home_pose_name)
    real_log.info("Warming IK solver before touching hardware.")
    controller.warmup()
    _validate_feetech_ready(args)

    calibration = _load_required_calibration(args)
    tracker = build_tracker(args, calibration, reset_workspace_on_x=False)
    grippers = None
    enabled_sides = _enabled_sides(args.side)
    real_env = make_real_backend(
        args.robot,
        runtime=runtime,
        rig_config=args.rig_config,
        active_sides=enabled_sides,
    )
    space_listener = KeyboardSpaceListener(enabled=args.space_start)
    tracker_started = False

    clap = DoubleClapDetector()
    play_sounds = not args.no_sounds
    loop_timer = TeleopLoopTimer(args.fps)
    motion_smoother = TeleopMotionSmoother(
        args.motion_smoothing_time_constant_s,
        position_deadband_m=args.motion_position_deadband_mm / 1000.0,
        orientation_deadband_rad=np.deg2rad(args.motion_orientation_deadband_deg),
    )
    teleop_session = TeleopSession(controller, motion_smoother)
    command_player = DelayedJointCommandPlayer(
        real_env.write,
        command_rate_hz=args.command_rate_hz,
        delay_s=args.trajectory_delay_ms / 1000.0,
    )
    episode_start: float | None = None
    frame = 0
    tracking_recovery = TrackingRecoveryPolicy()

    try:
        real_log.info("Starting tracking before moving real arms.")
        tracker.start()
        tracker_started = True
        grippers = connect_feetech(args)

        real_env.setup(repair=not args.skip_can_repair)
        real_env.connect()
        real_env.home(home_q)
        motion_smoother.reset(home_q)
        real_log.info(
            "Joint trajectory playback: %.1f Hz with %.0f ms delay.",
            args.command_rate_hz,
            args.trajectory_delay_ms,
        )

        space_listener.start()
        if args.space_start:
            real_log.info(
                "Real %s is at home. Start idle arms with Space, or double clap "
                "to start enabled arms.",
                args.robot,
            )
        else:
            real_log.info(
                "Real %s is at home. Double clap a gripper to start enabled arms.",
                args.robot,
            )

        while True:
            loop_start, _ = loop_timer.tick()
            if episode_start is not None and args.duration_s > 0.0:
                if loop_start - episode_start >= args.duration_s:
                    break

            sample = tracker.latest()
            side_tracked = {"left": sample.left_tracked, "right": sample.right_tracked}
            tracking_ok = _enabled_tracking_ok(side_tracked, enabled_sides)

            # Tracker startup and a fresh SDK reconnect can briefly expose an
            # empty sample.  Before an operator anchors an arm, there is no
            # robot motion to cancel, so do not hold the robot or restart the
            # PICO service for that transient state.
            if not controller.active and not tracking_ok:
                tracking_recovery.reset()
                widths = _latest_widths(grippers)
                if args.space_start:
                    space_listener.consume_space()
                clap.update(widths.left_mm, widths.right_mm, loop_start)
                loop_timer.sleep(loop_start)
                continue

            if not tracking_ok:
                if tracking_recovery.note_missing(loop_start):
                    command_player.stop()
                    held = real_env.hold(q)
                    controller.tracking_lost(held)
                    motion_smoother.reset(held)
                    q = held
                    real_log.warning(
                        "Tracking lost; pending motion cancelled at the current robot "
                        "command. Re-anchor after recovery."
                    )
                    log_say("tracking lost", play_sounds=play_sounds)
                recover = getattr(tracker, "recover", None)
                if callable(recover) and tracking_recovery.should_recover(loop_start):
                    if recover():
                        real_log.info(
                            "Tracking recovered; double clap or Space to re-anchor."
                        )
                        log_say("tracking recovered", play_sounds=play_sounds)
                loop_timer.sleep(loop_start)
                continue
            if tracking_recovery.lost:
                real_log.info("Tracking stream is valid again; waiting for a fresh anchor.")
            tracking_recovery.reset()

            widths = _latest_widths(grippers)
            inputs = teleop_session.inputs(sample, widths)
            start_sides: tuple[str, ...] = ()
            if args.space_start and space_listener.consume_space():
                start_sides = controller.idle_sides()
                if start_sides:
                    real_log.info("Space pressed; starting %s.", "/".join(start_sides))
            if clap.update(widths.left_mm, widths.right_mm, loop_start):
                if controller.active:
                    command_player.stop()
                    q = controller.reset()
                    motion_smoother.reset(home_q)
                    episode_start = None
                    frame = 0
                    real_log.info(
                        "Double clap detected; teleop reset, robot returning home slowly."
                    )
                    log_say("returning home", play_sounds=play_sounds)
                    real_env.move_home(home_q)
                    log_say("teleop reset", play_sounds=play_sounds)
                    continue
                start_sides = enabled_sides
                real_log.info("Double clap detected; starting %s.", "/".join(start_sides))

            teleop_frame = teleop_session.advance(
                inputs, now_s=loop_start, start_sides=start_sides
            )
            anchored_sides = teleop_frame.anchored_sides
            anchored_this_frame = bool(anchored_sides)
            for side in anchored_sides:
                real_log.info("%s arm anchored; real robot follows from home.", side)
                log_say(f"{side} anchored", play_sounds=play_sounds)

            if episode_start is None and anchored_this_frame:
                episode_start = loop_start
                frame = 0
                real_log.info("Teleop timer started.")

            q = teleop_frame.q
            if anchored_this_frame:
                # A fresh anchor defines a new trajectory epoch. Do not
                # interpolate from commands left over from an earlier epoch.
                command_player.stop()
                command_player.start(
                    q,
                    inputs.openings,
                    time_s=loop_start,
                )
            elif controller.active:
                if not command_player.running:
                    command_player.start(
                        q,
                        inputs.openings,
                        time_s=loop_start,
                    )
                else:
                    command_player.push(
                        q,
                        inputs.openings,
                        time_s=loop_start,
                    )
            real_env.check_health()

            loop_timer.sleep(loop_start)
            if episode_start is not None:
                frame += 1
    except KeyboardInterrupt:
        real_log.info("Stopping.")
    finally:
        space_listener.close()
        try:
            command_player.stop()
        finally:
            try:
                real_env.disconnect()
            finally:
                if grippers is not None:
                    grippers.close()
                if tracker_started:
                    tracker.stop()


def main() -> None:
    _run_real()


if __name__ == "__main__":
    main()
