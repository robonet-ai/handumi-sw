#!/usr/bin/env python3
"""Live HandUMI teleop for real Piper arms.

This is the real-hardware sibling of ``handumi-teleop-sim``. The tracking and
retargeting semantics are intentionally the same: the current HandUMI TCP pose
is anchored to the robot home TCP, then relative controller motion drives the
IK target. The one IK solution ``q`` is the source of truth; for real Piper it
is converted to SDK milli-degrees and streamed over CAN.

Safety defaults:

* only ``--robot piper`` is accepted in this first real backend;
* controller->TCP calibration is required;
* both arms home slowly to the XHUMAN Piper start pose before teleop;
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
from handumi.feetech.calibration import assert_calibrated, load_config, user_calibration_path
from handumi.feetech.setup import list_feetech_serial_ports
from handumi.real.can_setup import ensure_can_interfaces_ready
from handumi.real.piper_can import (
    PiperCanEnvironment,
    format_mdeg,
    load_piper_can_settings,
    piper_mdeg_to_q,
    q_to_piper_mdeg,
)
from handumi.retargeting.handumi_to_robot import (
    local_frame_adapter,
    local_relative_robot_target_pose7,
    raw_state_pose7_pair,
)
from handumi.robots.registry import EMBODIMENT_NAMES, load_embodiment
from handumi.scripts.record import build_tracker, connect_feetech
from handumi.scripts.teleop_sim import (
    KeyboardSpaceListener,
    _enabled_sides,
    _sample_state,
    _side_joint_indices,
    _start_sides,
    _tracking_world_map,
)
from handumi.tracking.gestures import DoubleClapDetector
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
    parser.add_argument("--robot", choices=EMBODIMENT_NAMES, default="piper")
    parser.add_argument("--side", choices=SIDE_CHOICES, default="both")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration-s", type=float, default=0.0, help="0 means run until Ctrl+C.")
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
    parser.add_argument("--no-sounds", action="store_true", help="Disable spoken feedback.")
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
        help="Machine-local Feetech, tracking, and Piper CAN configuration.",
    )

    # Feetech flags, same names as handumi-record and handumi-teleop-sim.
    parser.add_argument("--feetech-port", type=str, default=None)
    parser.add_argument("--skip-feetech", action="store_true")

    # Tracking flags, same names as handumi-record (shared build_tracker).
    parser.add_argument("--quest-ip", type=str, default=None)
    parser.add_argument("--tcp-port", type=int, default=None)
    parser.add_argument("--sync-port", type=int, default=None)
    parser.add_argument("--pico-mode", choices=("mandos", "object", "whole-body"), default="mandos")
    pico_transport = parser.add_mutually_exclusive_group()
    pico_transport.add_argument("--pico-adb", action="store_true")
    pico_transport.add_argument("--pico-wifi", action="store_true")
    parser.add_argument("--skip-adb-check", action="store_true")
    parser.add_argument(
        "--skip-can-repair",
        action="store_true",
        help="Do not auto-repair CAN with sudo before connecting Piper.",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.robot != "piper":
        raise SystemExit("Real teleop currently supports only --robot piper.")
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
    _validate_feetech_ports_exist(feetech_config)


def _validate_feetech_ports_exist(feetech_config) -> None:
    ports = {
        side: getattr(feetech_config, side).port or feetech_config.port
        for side in ("left", "right")
    }
    missing = {side: port for side, port in ports.items() if not port or not Path(port).exists()}
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
            "  uv run handumi-setup-hardware --robot piper --device pico "
            "--skip-can-map --skip-can-repair --skip-pico "
            "--force-feetech-calibration"
        )

    denied = {side: port for side, port in ports.items() if port and not os.access(port, os.R_OK | os.W_OK)}
    if denied:
        denied_text = ", ".join(f"{side}={port}" for side, port in denied.items())
        raise SystemExit(
            f"No tengo permisos para abrir Feetech: {denied_text}.\n"
            "Corre primero:\n"
            "  uv run handumi-setup-hardware --robot piper --device pico "
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
    return zero_gripper_widths() if grippers is None else grippers.read_normalized_widths()


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
    solver = runtime.solver_cls()
    q = runtime.config.home_q.astype(np.float32).copy()
    home_q = q.copy()
    home_left_pose7, home_right_pose7 = solver.fk_pose7(q)
    max_reach = runtime.config.ik_weights.max_reach
    side_indices = _side_joint_indices(runtime)
    actuated_names = list(runtime.robot.joints.actuated_names)
    home_targets_mdeg = q_to_piper_mdeg(home_q, actuated_names)

    log.info("Warming IK solver before touching hardware.")
    solver.ik(
        q,
        left_pose=_ik_home_target(home_left_pose7),
        right_pose=_ik_home_target(home_right_pose7),
    )

    settings = load_piper_can_settings(args.rig_config, runtime.config.real)
    log.info(
        "Piper CAN rig: left=%s right=%s bitrate=%d command_rate=%.1fHz",
        settings.left_port,
        settings.right_port,
        settings.bitrate,
        settings.command_rate_hz,
    )
    log.info(
        "Piper home mdeg: left=%s right=%s",
        format_mdeg(home_targets_mdeg["left"]),
        format_mdeg(home_targets_mdeg["right"]),
    )
    if not args.skip_can_repair:
        ensure_can_interfaces_ready(
            [settings.left_port, settings.right_port],
            bitrate=settings.bitrate,
            restart_ms=settings.restart_ms,
        )
    _validate_feetech_ready(args)

    calibration = _load_required_calibration(args)
    tracker = build_tracker(args, calibration, reset_workspace_on_x=False)
    grippers = None
    real_env = PiperCanEnvironment(settings)
    space_listener = KeyboardSpaceListener(enabled=args.space_start)
    tracker_started = False

    anchor_ref = {"left": home_left_pose7.copy(), "right": home_right_pose7.copy()}
    anchors: dict[str, dict[str, np.ndarray] | None] = {"left": None, "right": None}
    enabled_sides = _enabled_sides(args.side)
    clap = DoubleClapDetector()
    play_sounds = not args.no_sounds
    interval = 1.0 / args.fps
    episode_start: float | None = None
    frame = 0
    tracking_lost_since: float | None = None
    tracking_hold_sides: set[str] = set()
    last_recovery_attempt = 0.0

    try:
        log.info("Starting tracking before moving real arms.")
        tracker.start()
        tracker_started = True
        grippers = connect_feetech(args)

        real_env.connect()
        real_env.home(home_targets_mdeg)
        real_env.set_q(home_q, actuated_names)

        space_listener.start()
        if args.space_start:
            log.info(
                "Real Piper is at home. Start idle arms with Space, or double clap "
                "to start enabled arms."
            )
        else:
            log.info(
                "Real Piper is at home. Double clap a gripper to start enabled arms."
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
                    _clear_enabled_anchors(anchors, enabled_sides)
                    held = real_env.hold_current_commands_mdeg()
                    q = piper_mdeg_to_q(
                        left_mdeg=held["left"],
                        right_mdeg=held["right"],
                        actuated_names=actuated_names,
                        base_q=q,
                    )
                    tracking_hold_sides.update(enabled_sides)
                    log.warning(
                        "Tracking lost; pending motion cancelled at the current Piper "
                        "command. Re-anchor after recovery."
                    )
                    log_say("tracking lost", play_sounds=play_sounds)
                recover = getattr(tracker, "recover", None)
                if callable(recover) and loop_start - last_recovery_attempt >= 3.0:
                    last_recovery_attempt = loop_start
                    if recover():
                        log.info("Tracking recovered; double clap or Space to re-anchor.")
                        log_say("tracking recovered", play_sounds=play_sounds)
                dt = time.perf_counter() - loop_start
                if (sleep := interval - dt) > 0:
                    time.sleep(sleep)
                continue
            if tracking_lost_since is not None:
                tracking_lost_since = None
                log.info("Tracking stream is valid again; waiting for a fresh anchor.")

            widths = _latest_widths(grippers)
            if grippers is not None:
                real_env.set_gripper_widths_mm(
                    {"left": widths.left_mm, "right": widths.right_mm}
                )
            state = _sample_state(sample, widths)
            source_poses = dict(zip(("left", "right"), raw_state_pose7_pair(state), strict=True))

            start_sides: tuple[str, ...] = ()
            if args.space_start and space_listener.consume_space():
                start_sides = _start_sides(anchors, enabled_sides)
                if start_sides:
                    log.info("Space pressed; starting %s.", "/".join(start_sides))
            if clap.update(widths.left_mm, widths.right_mm, loop_start):
                if _has_enabled_anchors(anchors, enabled_sides):
                    _clear_enabled_anchors(anchors, enabled_sides)
                    tracking_hold_sides.difference_update(enabled_sides)
                    q = home_q.copy()
                    episode_start = None
                    frame = 0
                    log.info(
                        "Double clap detected; teleop reset, Piper returning home slowly."
                    )
                    log_say("returning home", play_sounds=play_sounds)
                    real_env.move_home(home_targets_mdeg)
                    log_say("teleop reset", play_sounds=play_sounds)
                    continue
                start_sides = enabled_sides
                log.info("Double clap detected; starting %s.", "/".join(start_sides))

            anchored_this_frame = False
            for side in ("left", "right"):
                if side not in enabled_sides or side not in start_sides:
                    continue
                if not side_tracked[side]:
                    log.warning("%s anchor ignored; that controller is not tracked.", side)
                    continue
                source_pose = source_poses[side]
                anchors[side] = {
                    "source": source_pose.copy(),
                    "adapter": local_frame_adapter(
                        source_pose,
                        anchor_ref[side],
                        source_world_to_robot_world=_tracking_world_map(args.device),
                    ),
                }
                tracking_hold_sides.discard(side)
                anchored_this_frame = True
                log.info("%s arm anchored; real Piper follows from home.", side)
                log_say(f"{side} anchored", play_sounds=play_sounds)

            if episode_start is None and anchored_this_frame:
                episode_start = loop_start
                frame = 0
                log.info("Teleop timer started.")

            ik_targets: dict[str, tuple[np.ndarray, np.ndarray] | None] = {
                "left": None,
                "right": None,
            }
            for side in ("left", "right"):
                anchor = anchors[side]
                if anchor is None or not side_tracked[side]:
                    continue
                pose7 = local_relative_robot_target_pose7(
                    previous_source_pose7=anchor["source"],
                    current_source_pose7=source_poses[side],
                    base_robot_pose7=anchor_ref[side],
                    adapter_rot=anchor["adapter"],
                    home_robot_pose7=anchor_ref[side],
                    translation_scale=args.translation_scale,
                    max_reach=max_reach,
                )
                ik_targets[side] = (pose7[:3], pose7[3:7])

            previous_q = q.copy()
            q = solver.ik(q, left_pose=ik_targets["left"], right_pose=ik_targets["right"])
            _apply_inactive_side_policy(
                q,
                previous_q,
                home_q,
                anchors,
                side_indices,
                tracking_hold_sides,
            )
            runtime.set_finger_positions(
                q, {"left": widths.left_normalized, "right": widths.right_normalized}
            )
            real_env.set_q(q, actuated_names)
            real_env.raise_if_failed()

            dt = time.perf_counter() - loop_start
            if (sleep := interval - dt) > 0:
                time.sleep(sleep)
            if episode_start is not None:
                frame += 1
    except KeyboardInterrupt:
        log.info("Stopping.")
    finally:
        space_listener.close()
        real_env.close()
        if grippers is not None:
            grippers.close()
        if tracker_started:
            tracker.stop()


if __name__ == "__main__":
    main()
