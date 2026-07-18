#!/usr/bin/env python3
"""Live HandUMI teleop for registered real-robot backends.

This is the real-hardware sibling of ``handumi-teleop-sim``. The tracking and
retargeting semantics are intentionally the same: the current HandUMI TCP pose
is anchored to the robot home TCP, then relative controller motion drives the
IK target. The one IK solution ``q`` is the source of truth; the selected lazy
backend converts it to vendor commands and streams them over CAN.

Safety defaults:

* controller->TCP calibration is required;
* both arms home slowly to the selected named pose before teleop;
* arms stay at home until double-clap or explicit ``--space-start``;
* a double-clap while teleop is active clears anchors and returns home.

Usage
-----
::

    handumi-teleop-real --device pico --robot piper
    handumi-teleop-real --device pico --robot piper --space-start
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np

from handumi.calibration.control_tcp import (
    calibration_path_for_robot_device,
    load_controller_tcp_calibration,
)
from handumi.config import DEFAULT_RIG_CONFIG
from handumi.feetech import zero_gripper_widths
from handumi.feetech.calibration import (
    assert_calibrated,
    load_config,
    user_calibration_path,
)
from handumi.feetech.setup import list_feetech_serial_ports
from handumi.real.backends import REAL_BACKEND_NAMES, make_real_backend
from handumi.retargeting.handumi_to_robot import raw_state_pose7_pair
from handumi.robots.registry import load_embodiment, resolve_home_q
from handumi.scripts.record import build_tracker, connect_feetech
from handumi.scripts.teleop_sim import (
    KeyboardSpaceListener,
    _enabled_sides,
    _sample_state,
    _tracking_world_map,
)
from handumi.tracking.gestures import DoubleClapDetector
from handumi.teleop.core import TeleopController
from handumi.utils.speech import log_say

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("handumi.teleop_real")

SIDE_CHOICES = ("left", "right", "both")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", choices=("pico", "meta"), required=True)
    parser.add_argument("--robot", choices=REAL_BACKEND_NAMES, default="piper")
    parser.add_argument(
        "--home-pose",
        default=None,
        help="Named safe pose from the robot YAML (OpenArm: forward_open, arms_90, down).",
    )
    parser.add_argument("--side", choices=SIDE_CHOICES, default="both")
    parser.add_argument("--fps", type=int, default=30)
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

    # Feetech flags, same names as handumi-record and handumi-teleop-sim.
    parser.add_argument("--feetech-port", type=str, default=None)
    parser.add_argument("--skip-feetech", action="store_true")

    # Tracking flags, same names as handumi-record (shared build_tracker).
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
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise SystemExit("--fps must be > 0.")
    if args.duration_s < 0.0:
        raise SystemExit("--duration-s must be >= 0.")
    if args.skip_feetech and not args.space_start:
        raise SystemExit(
            "--skip-feetech disables double-clap; add --space-start so teleop can begin."
        )


def _validate_feetech_ready(args: argparse.Namespace) -> None:
    if args.skip_feetech:
        return
    feetech_config = load_config(args.rig_config)
    if args.feetech_port is not None:
        feetech_config = type(feetech_config)(
            port=args.feetech_port,
            baudrate=feetech_config.baudrate,
            protocol_version=feetech_config.protocol_version,
            left=feetech_config.left,
            right=feetech_config.right,
        )
    assert_calibrated(feetech_config, source=user_calibration_path())
    _validate_feetech_ports_exist(feetech_config, robot=args.robot)


def _validate_feetech_ports_exist(feetech_config, *, robot: str = "piper") -> None:
    ports = {
        side: getattr(feetech_config, side).port or feetech_config.port
        for side in ("left", "right")
    }
    missing = {
        side: port
        for side, port in ports.items()
        if not port or not Path(port).exists()
    }
    if missing:
        current = sorted(list_feetech_serial_ports())
        missing_text = ", ".join(
            f"{side}={port or '<unset>'}" for side, port in missing.items()
        )
        current_text = ", ".join(current) if current else "ninguno"
        raise SystemExit(
            "Feetech port configured in rig.yaml is missing: "
            f"{missing_text}.\n"
            f"Puertos Feetech actuales: {current_text}\n"
            "Remapea Feetech sin tocar CAN/PICO:\n"
            f"  uv run handumi-setup-hardware --robot {robot} --device pico "
            "--skip-can-map --skip-can-repair --skip-pico "
            "--force-feetech-calibration"
        )

    denied = {
        side: port
        for side, port in ports.items()
        if port and not os.access(port, os.R_OK | os.W_OK)
    }
    if denied:
        denied_text = ", ".join(f"{side}={port}" for side, port in denied.items())
        raise SystemExit(
            f"No tengo permisos para abrir Feetech: {denied_text}.\n"
            "Corre primero:\n"
            f"  uv run handumi-setup-hardware --robot {robot} --device pico "
            "--skip-can-map --skip-can-repair --skip-feetech-map --skip-pico"
        )


def _load_required_calibration(args: argparse.Namespace):
    path, source = calibration_path_for_robot_device(
        args.robot,
        args.device,
        explicit_path=args.controller_tcp_calibration,
    )
    if not path.exists():
        raise SystemExit(
            f"Missing controller->TCP calibration: {path}\n"
            "Run the TCP calibration before real teleop, or pass "
            "--controller-tcp-calibration <path>."
        )
    calibration = load_controller_tcp_calibration(path)
    log.info("controller->TCP calibration: %s", source)
    return calibration


def _latest_widths(grippers):
    return (
        zero_gripper_widths() if grippers is None else grippers.read_normalized_widths()
    )


def _ik_home_target(pose7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return pose7[:3], pose7[3:7]


def _enabled_tracking_ok(
    side_tracked: dict[str, bool],
    enabled_sides: tuple[str, ...],
) -> bool:
    return all(side_tracked[side] for side in enabled_sides)


def _clear_enabled_anchors(
    anchors: dict[str, dict[str, np.ndarray] | None],
    enabled_sides: tuple[str, ...],
) -> None:
    for side in enabled_sides:
        anchors[side] = None


def _has_enabled_anchors(
    anchors: dict[str, dict[str, np.ndarray] | None],
    enabled_sides: tuple[str, ...],
) -> bool:
    return any(anchors[side] is not None for side in enabled_sides)


def _apply_inactive_side_policy(
    q: np.ndarray,
    previous_q: np.ndarray,
    home_q: np.ndarray,
    anchors: dict[str, dict[str, np.ndarray] | None],
    side_indices: dict[str, list[int]],
    tracking_hold_sides: set[str],
) -> None:
    """Keep recovery-held arms still; park other inactive arms at home."""
    for side in ("left", "right"):
        if anchors[side] is not None:
            continue
        source = previous_q if side in tracking_hold_sides else home_q
        q[side_indices[side]] = source[side_indices[side]]


def main() -> None:
    args = parse_args()
    _validate_args(args)

    log.info("Loading %s IK solver.", args.robot)
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
    log.info("Selected home pose: %s", home_pose_name)
    actuated_names = list(runtime.robot.joints.actuated_names)

    log.info("Warming IK solver before touching hardware.")
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
    interval = 1.0 / args.fps
    episode_start: float | None = None
    frame = 0
    tracking_lost_since: float | None = None
    last_recovery_attempt = 0.0

    try:
        log.info("Starting tracking before moving real arms.")
        tracker.start()
        tracker_started = True
        grippers = connect_feetech(args)

        real_env.prepare(repair=not args.skip_can_repair)
        real_env.connect()
        real_env.home(home_q, actuated_names)

        space_listener.start()
        if args.space_start:
            log.info(
                "Real %s is at home. Start idle arms with Space, or double clap "
                "to start enabled arms.",
                args.robot,
            )
        else:
            log.info(
                "Real %s is at home. Double clap a gripper to start enabled arms.",
                args.robot,
            )

        while True:
            loop_start = time.perf_counter()
            if episode_start is not None and args.duration_s > 0.0:
                if loop_start - episode_start >= args.duration_s:
                    break

            sample = tracker.latest()
            side_tracked = {"left": sample.left_tracked, "right": sample.right_tracked}
            tracking_ok = _enabled_tracking_ok(side_tracked, enabled_sides)
            if not tracking_ok:
                if tracking_lost_since is None:
                    tracking_lost_since = loop_start
                    held = real_env.hold(q, actuated_names)
                    controller.tracking_lost(held)
                    q = held
                    log.warning(
                        "Tracking lost; pending motion cancelled at the current robot "
                        "command. Re-anchor after recovery."
                    )
                    log_say("tracking lost", play_sounds=play_sounds)
                recover = getattr(tracker, "recover", None)
                if callable(recover) and loop_start - last_recovery_attempt >= 3.0:
                    last_recovery_attempt = loop_start
                    if recover():
                        log.info(
                            "Tracking recovered; double clap or Space to re-anchor."
                        )
                        log_say("tracking recovered", play_sounds=play_sounds)
                dt = time.perf_counter() - loop_start
                if (sleep := interval - dt) > 0:
                    time.sleep(sleep)
                continue
            if tracking_lost_since is not None:
                tracking_lost_since = None
                log.info("Tracking stream is valid again; waiting for a fresh anchor.")

            widths = _latest_widths(grippers)
            state = _sample_state(sample, widths)
            source_poses: dict[str, np.ndarray] = dict(
                zip(("left", "right"), raw_state_pose7_pair(state), strict=True)
            )

            start_sides: tuple[str, ...] = ()
            if args.space_start and space_listener.consume_space():
                start_sides = controller.idle_sides()
                if start_sides:
                    log.info("Space pressed; starting %s.", "/".join(start_sides))
            if clap.update(widths.left_mm, widths.right_mm, loop_start):
                if controller.active:
                    q = controller.reset()
                    episode_start = None
                    frame = 0
                    log.info(
                        "Double clap detected; teleop reset, robot returning home slowly."
                    )
                    log_say("returning home", play_sounds=play_sounds)
                    real_env.move_home(home_q, actuated_names)
                    log_say("teleop reset", play_sounds=play_sounds)
                    continue
                start_sides = enabled_sides
                log.info("Double clap detected; starting %s.", "/".join(start_sides))

            anchored_sides = controller.anchor(source_poses, side_tracked, start_sides)
            anchored_this_frame = bool(anchored_sides)
            for side in anchored_sides:
                log.info("%s arm anchored; real robot follows from home.", side)
                log_say(f"{side} anchored", play_sounds=play_sounds)

            if episode_start is None and anchored_this_frame:
                episode_start = loop_start
                frame = 0
                log.info("Teleop timer started.")

            openings = {
                "left": widths.left_normalized,
                "right": widths.right_normalized,
            }
            q = controller.step(source_poses, side_tracked, openings).q
            real_env.command(q, actuated_names, openings)
            real_env.check_health()

            dt = time.perf_counter() - loop_start
            if (sleep := interval - dt) > 0:
                time.sleep(sleep)
            if episode_start is not None:
                frame += 1
    except KeyboardInterrupt:
        log.info("Stopping.")
    finally:
        space_listener.close()
        try:
            real_env.close()
        finally:
            if grippers is not None:
                grippers.close()
            if tracker_started:
                tracker.stop()


if __name__ == "__main__":
    main()
