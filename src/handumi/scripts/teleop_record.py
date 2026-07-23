
"""Record joint-level real-robot teleoperation demonstrations.

This is the recording sibling of ``handumi teleop-real``. The operator drives
the real robot with HandUMI tracking and Feetech gripper widths, while each
LeRobot row stores canonical robot joints directly:

* ``observation.state`` is the robot feedback read from the real backend.
* ``action`` is the next joint command produced by the teleop controller.

Before recording, controller->TCP calibration and Feetech calibration must be
available. Episode control matches ``handumi record --clap-control``:

* double-squeeze right: start or stop/save the current episode
* double-squeeze left while recording: discard and restart the same episode
* ``Esc`` / ``Ctrl+C``: discard the active episode and stop

PICO tracking uses ADB.

Examples
--------
::

    handumi teleop-record --device pico --robot piper
    handumi teleop-record --device pico --robot openarmv1
    handumi teleop-record --device pico --robot piper --side right
    handumi teleop-record --device pico --robot piper \
        --output-dir outputs/my_dataset --resume

Common options:

* ``--device``        pico|meta tracking device.
* ``--robot``         Registered real backend, for example piper or openarmv1.
* ``--side``          left|right|both enabled arms.
* ``--fps``           Recording/control frequency in Hz.
* ``--episode-time-s`` Maximum episode duration in seconds.
* ``--num-episodes``  Number of episodes to record; 0 means until stopped.
* ``--task``          Task description stored in the dataset.
* ``--output-dir``    Destination directory; defaults to outputs/teleop_<date>.
* ``--resume``        Append episodes to an existing dataset.
"""

import argparse
import logging
import signal
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from handumi.config import DEFAULT_RIG_CONFIG
from handumi.dataset.canonical import canonical_joint_layout, canonicalize_command
from handumi.dataset.capture import (
    GRIPPER_STALE_TIMEOUT_S,
    MAX_SYNC_SKEW_S,
    SENSOR_LOSS_TIMEOUT_S,
    SYNC_LAG_S,
    TRACKING_LOSS_TIMEOUT_S,
)
from handumi.dataset.raw import (
    HANDUMI_CAPTURE_SCHEMA,
    camera_health_features,
    capture_timing_features,
    feetech_features,
)
from handumi.feetech import FeetechGripperPair, FeetechGripperSampler, GripperWidths
from handumi.real.registry import REAL_BACKEND_NAMES, make_real_backend
from handumi.robots.registry import load_embodiment, resolve_home_q
from handumi.scripts.record import (
    _EscapeStopListener,
    _robot_metadata,
    _wait_for_clap,
    _wait_for_tracking,
    build_tracker,
    connect_feetech,
)
from handumi.synchronization import (
    SustainedHealthGate,
    synchronized_gripper_frame,
)
from handumi.teleop.common import (
    DEFAULT_GRIPPER_SAMPLE_HZ,
    DEFAULT_MOTION_SMOOTHING_TIME_CONSTANT_S,
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
    validate_feetech_ready as _validate_feetech_ready,
)
from handumi.teleop.tracking import TrackingRecoveryPolicy
from handumi.tracking.base import TrackingProvider
from handumi.tracking.gestures import DoubleClapDetector
from handumi.utils.speech import log_say

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
record_log = logging.getLogger("handumi.record_teleop")

DEFAULT_TRANSLATION_SCALE = 1.0
SPACE_START_ENABLED = False
PLAY_SOUNDS = True
MOTION_SMOOTHING_TIME_CONSTANT_S = DEFAULT_MOTION_SMOOTHING_TIME_CONSTANT_S
REPAIR_CAN_ON_SETUP = True
RIG_CONFIG_PATH = DEFAULT_RIG_CONFIG
CONTROLLER_TCP_CALIBRATION_PATH = None
FEETECH_PORT_OVERRIDE = None
REQUIRE_FEETECH_GRIPPERS = True
META_QUEST_IP_OVERRIDE = None
META_TCP_PORT_OVERRIDE = None
META_SYNC_PORT_OVERRIDE = None
PICO_TRACKING_MODE = "mandos"
PICO_USE_WIFI = False
SKIP_ADB_CHECK = False
DEFAULT_RECORD_COMMAND_RATE_HZ = 100.0
DEFAULT_RECORD_TRAJECTORY_DELAY_MS = 80.0


def build_features(
    cam_names: list[str],
    cam_width: int,
    cam_height: int,
    use_videos: bool,
    joint_names: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    img_dtype = "video" if use_videos else "image"
    features: dict[str, Any] = {}
    for cam in cam_names:
        features[f"observation.images.{cam}"] = {
            "dtype": img_dtype,
            "shape": (cam_height, cam_width, 3),
            "names": ["height", "width", "channel"],
        }
    state_action = joint_state_feature(joint_names)
    features["observation.state"] = state_action
    features["action"] = dict(state_action)
    features.update(feetech_features())
    features.update(capture_timing_features())
    features.update(camera_health_features(cam_names))
    return features


def joint_state_feature(joint_names: list[str] | tuple[str, ...]) -> dict[str, Any]:
    names = list(joint_names)
    return {
        "dtype": "float32",
        "shape": (len(names),),
        "names": names,
    }


def build_joint_frame(
    *,
    observation_q: np.ndarray,
    action_q: np.ndarray,
    widths: GripperWidths,
) -> dict[str, np.ndarray]:
    return {
        "observation.state": np.asarray(observation_q, dtype=np.float32).copy(),
        "action": np.asarray(action_q, dtype=np.float32).copy(),
        "observation.feetech.left_ticks": np.array([widths.left_ticks], dtype=np.int64),
        "observation.feetech.right_ticks": np.array(
            [widths.right_ticks], dtype=np.int64
        ),
        "observation.feetech.left_width_mm": np.array(
            [widths.left_mm], dtype=np.float32
        ),
        "observation.feetech.right_width_mm": np.array(
            [widths.right_mm], dtype=np.float32
        ),
        "observation.feetech.left_normalized": np.array(
            [widths.left_normalized], dtype=np.float32
        ),
        "observation.feetech.right_normalized": np.array(
            [widths.right_normalized], dtype=np.float32
        ),
    }


def _parse_record_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record real-robot HandUMI teleoperation demonstrations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--device", choices=("pico", "meta"), required=True)
    p.add_argument("--robot", choices=REAL_BACKEND_NAMES, default="piper")
    p.add_argument("--home-pose", default=None)
    p.add_argument("--side", choices=SIDE_CHOICES, default="both")
    p.add_argument("--fps", type=int, default=DEFAULT_TELEOP_FPS)
    p.add_argument(
        "--command-rate-hz",
        type=float,
        default=DEFAULT_RECORD_COMMAND_RATE_HZ,
        help="Fixed-rate playback frequency for interpolated joint commands.",
    )
    p.add_argument(
        "--trajectory-delay-ms",
        type=float,
        default=DEFAULT_RECORD_TRAJECTORY_DELAY_MS,
        help="Playback delay used to bracket and interpolate IK results.",
    )
    p.add_argument(
        "--motion-smoothing-time-constant-s",
        type=float,
        default=MOTION_SMOOTHING_TIME_CONSTANT_S,
        help=(
            "Shared TCP-pose and joint-command low-pass time constant; "
            "0 disables smoothing."
        ),
    )
    p.add_argument("--episode-time-s", type=float, default=60.0)
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--task", type=str, default="HandUMI real teleop recording")
    p.add_argument("--repo-id", type=str, default="local/handumi_teleop_dataset")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    _apply_recording_defaults(args)
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _parse_record_args(argv)


def _apply_recording_defaults(args: argparse.Namespace) -> None:
    args.translation_scale = DEFAULT_TRANSLATION_SCALE
    args.space_start = SPACE_START_ENABLED
    args.no_sounds = not PLAY_SOUNDS
    args.sync_lag_s = SYNC_LAG_S
    args.max_sync_skew_s = MAX_SYNC_SKEW_S
    args.gripper_stale_timeout_s = GRIPPER_STALE_TIMEOUT_S
    args.sensor_loss_timeout_s = SENSOR_LOSS_TIMEOUT_S
    args.feetech_sample_hz = DEFAULT_GRIPPER_SAMPLE_HZ
    args.tracking_loss_timeout_s = TRACKING_LOSS_TIMEOUT_S
    args.skip_can_repair = not REPAIR_CAN_ON_SETUP
    args.rig_config = RIG_CONFIG_PATH
    args.controller_tcp_calibration = CONTROLLER_TCP_CALIBRATION_PATH
    args.feetech_port = FEETECH_PORT_OVERRIDE
    args.skip_feetech = not REQUIRE_FEETECH_GRIPPERS
    args.quest_ip = META_QUEST_IP_OVERRIDE
    args.tcp_port = META_TCP_PORT_OVERRIDE
    args.sync_port = META_SYNC_PORT_OVERRIDE
    args.pico_mode = PICO_TRACKING_MODE
    args.pico_adb = not PICO_USE_WIFI
    args.pico_wifi = PICO_USE_WIFI
    args.skip_adb_check = SKIP_ADB_CHECK


def _validate_record_args(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise SystemExit("--fps must be > 0.")
    if args.episode_time_s <= 0:
        raise SystemExit("--episode-time-s must be > 0.")
    if args.num_episodes < 0:
        raise SystemExit("--num-episodes must be >= 0.")
    if args.command_rate_hz <= 0.0:
        raise SystemExit("--command-rate-hz must be > 0.")
    if args.trajectory_delay_ms < 0.0:
        raise SystemExit("--trajectory-delay-ms must be >= 0.")
    if args.motion_smoothing_time_constant_s < 0.0:
        raise SystemExit("--motion-smoothing-time-constant-s must be >= 0.")
    for name in (
        "sync_lag_s",
        "max_sync_skew_s",
        "gripper_stale_timeout_s",
        "sensor_loss_timeout_s",
        "feetech_sample_hz",
        "tracking_loss_timeout_s",
    ):
        value = getattr(args, name)
        if value <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be greater than zero.")


def record_episode(
    *,
    tracker: TrackingProvider,
    grippers: FeetechGripperSampler | FeetechGripperPair | None,
    real_env,
    controller: TeleopController,
    runtime,
    home_q: np.ndarray,
    enabled_sides: tuple[str, ...],
    space_listener: KeyboardSpaceListener,
    clap_detector: DoubleClapDetector,
    episode_time_s: float,
    fps: int,
    task: str,
    stop_event: threading.Event,
    play_sounds: bool,
    initial_start_sides: tuple[str, ...],
    sync_lag_s: float,
    max_sync_skew_s: float,
    gripper_stale_timeout_s: float,
    sensor_loss_timeout_s: float,
    tracking_loss_timeout_s: float,
    command_player: DelayedJointCommandPlayer,
    motion_smoother: TeleopMotionSmoother | None = None,
) -> tuple[np.ndarray, np.ndarray, int, str, np.ndarray]:
    loop_timer = TeleopLoopTimer(fps)
    n_frames = 0
    start_t: float | None = None
    episode_start_ns: int | None = None
    status = "recorded"
    pending_start_sides = initial_start_sides
    tracking_recovery = TrackingRecoveryPolicy()
    health_gate = SustainedHealthGate(sensor_loss_timeout_s)
    max_sync_skew_ns = int(max_sync_skew_s * 1e9)
    sync_lag_ns = int(sync_lag_s * 1e9)
    q = controller.q.copy()
    if motion_smoother is None:
        motion_smoother = TeleopMotionSmoother()
    motion_smoother.reset(q)
    teleop_session = TeleopSession(controller, motion_smoother)
    observations: list[np.ndarray] = []
    commands: list[np.ndarray] = []
    command_player.stop()

    while True:
        loop_start, _ = loop_timer.tick()
        record_time_ns = time.monotonic_ns()
        if episode_start_ns is None:
            episode_start_ns = record_time_ns

        if stop_event.is_set():
            status = "interrupted"
            observations.clear()
            commands.clear()
            break
        if start_t is not None and loop_start - start_t >= episode_time_s:
            break

        sample = tracker.latest()
        side_tracked = {"left": sample.left_tracked, "right": sample.right_tracked}
        tracking_ok = _enabled_tracking_ok(side_tracked, enabled_sides)

        if not controller.active and not tracking_ok:
            tracking_recovery.reset()
            immediate_widths = _latest_widths(grippers)
            space_listener.consume_space()
            clap_detector.update_side(
                immediate_widths.left_mm,
                immediate_widths.right_mm,
                loop_start,
            )
            loop_timer.sleep(loop_start)
            continue

        if not tracking_ok:
            if tracking_recovery.note_missing(loop_start):
                command_player.stop()
                held = real_env.hold(q)
                controller.tracking_lost(held)
                motion_smoother.reset(held)
                q = held
                record_log.warning("Tracking lost; robot command held and episode discarded.")
                log_say("tracking lost", play_sounds=play_sounds)
            if (
                tracking_recovery.lost
                and tracking_recovery.lost_for(loop_start) >= tracking_loss_timeout_s
            ):
                status = "tracking_lost"
                observations.clear()
                commands.clear()
                break
            recover = getattr(tracker, "recover", None)
            if callable(recover) and tracking_recovery.should_recover(loop_start):
                if recover():
                    record_log.info("Tracking recovered; double clap or Space to re-anchor.")
                    log_say("tracking recovered", play_sounds=play_sounds)
            loop_timer.sleep(loop_start)
            continue
        if tracking_recovery.lost:
            record_log.info("Tracking stream recovered; waiting for a fresh anchor.")
        tracking_recovery.reset()

        observation_q = real_env.read(base_q=q)
        immediate_widths = _latest_widths(grippers)
        start_sides = pending_start_sides
        pending_start_sides = ()
        if space_listener.consume_space():
            start_sides = controller.idle_sides()
        clap_side = clap_detector.update_side(
            immediate_widths.left_mm, immediate_widths.right_mm, loop_start
        )
        if clap_side == "right":
            if start_t is not None:
                status = "recorded"
                break
            start_sides = enabled_sides
        elif clap_side == "left" and start_t is not None:
            status = "repeat"
            observations.clear()
            commands.clear()
            break

        teleop_frame = teleop_session.advance(
            teleop_session.inputs(sample, immediate_widths),
            now_s=loop_start,
            start_sides=start_sides,
        )
        anchored = teleop_frame.anchored_sides
        if anchored and start_t is None:
            start_t = loop_start
            record_log.info("Teleop episode started after anchoring %s.", "/".join(anchored))
            log_say("recording episode", play_sounds=play_sounds)

        action_q = teleop_frame.q
        if anchored:
            command_player.stop()
            command_player.start(
                action_q,
                teleop_frame.inputs.openings,
                time_s=loop_start,
            )
        elif controller.active:
            if not command_player.running:
                command_player.start(
                    action_q,
                    teleop_frame.inputs.openings,
                    time_s=loop_start,
                )
            else:
                command_player.push(
                    action_q,
                    teleop_frame.inputs.openings,
                    time_s=loop_start,
                )
        real_env.check_health()
        q = action_q

        played_command = command_player.latest()
        if played_command is None:
            played_action_q = action_q
            played_openings = teleop_frame.inputs.openings
        else:
            played_action_q, played_openings = played_command

        if start_t is None:
            loop_timer.sleep(loop_start)
            continue

        target_time_ns = max(episode_start_ns, record_time_ns - sync_lag_ns)
        gripper_frame = synchronized_gripper_frame(
            grippers,
            target_time_ns=target_time_ns,
            record_time_ns=record_time_ns,
            stale_timeout_s=gripper_stale_timeout_s,
            max_sync_skew_s=max_sync_skew_s,
        )
        tracking_time_ns = int(sample.aligned_time_ns or sample.pc_monotonic_ns)
        tracking_sync_ok = bool(
            tracking_time_ns > 0
            and abs(tracking_time_ns - target_time_ns) <= max_sync_skew_ns
        )
        sensor_health = {
            "feetech": gripper_frame.healthy_for_gate,
            "tracking": tracking_sync_ok,
        }
        _, timed_out_sensors = health_gate.update(sensor_health, record_time_ns)
        if timed_out_sensors:
            status = "sensor_unhealthy"
            record_log.error(
                "Sensor health timed out: %s.",
                ", ".join(sorted(timed_out_sensors)),
            )
            observations.clear()
            commands.clear()
            break

        observations.append(
            canonicalize_command(
                observation_q,
                runtime=runtime,
                openings={
                    "left": gripper_frame.widths.left_normalized,
                    "right": gripper_frame.widths.right_normalized,
                },
            )
        )
        commands.append(
            canonicalize_command(
                played_action_q,
                runtime=runtime,
                openings=played_openings,
            )
        )
        n_frames += 1
        loop_timer.sleep(loop_start)

    command_player.stop()
    if len(observations) < 2:
        return (
            np.empty((0, canonical_joint_layout(runtime).size), dtype=np.float32),
            np.empty((0, canonical_joint_layout(runtime).size), dtype=np.float32),
            n_frames,
            status,
            q,
        )
    states = np.asarray(observations[:-1], dtype=np.float32)
    actions = np.asarray(commands[1:], dtype=np.float32)
    return states, actions, len(states), status, q


def _run_record() -> None:
    args = _parse_record_args()
    _validate_record_args(args)
    if args.output_dir is None:
        args.output_dir = _default_output_dir()
    play_sounds = not args.no_sounds
    stop_event = threading.Event()

    record_log.info("Loading %s IK solver.", args.robot)
    runtime = load_embodiment(args.robot)
    try:
        home_pose_name, home_q = resolve_home_q(
            runtime, rig_config=args.rig_config, explicit_name=args.home_pose
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    enabled_sides = _enabled_sides(args.side)
    controller = TeleopController(
        runtime,
        home_q=home_q,
        enabled_sides=enabled_sides,
        source_world_to_robot_world=_tracking_world_map(args.device),
        translation_scale=args.translation_scale,
    )
    controller.warmup()
    _validate_feetech_ready(args)

    calibration = _load_required_calibration(args)
    tracker = build_tracker(args, calibration, reset_workspace_on_x=False)
    real_env = make_real_backend(
        args.robot,
        runtime=runtime,
        rig_config=args.rig_config,
        active_sides=enabled_sides,
    )
    gripper_pair = None
    grippers = None
    tracker_started = False
    space_listener = KeyboardSpaceListener(enabled=args.space_start)
    motion_smoother = TeleopMotionSmoother(args.motion_smoothing_time_constant_s)
    command_player = DelayedJointCommandPlayer(
        real_env.write,
        command_rate_hz=args.command_rate_hz,
        delay_s=args.trajectory_delay_ms / 1000.0,
    )

    def _on_signal(signum, frame):
        del signum, frame
        record_log.info("Signal received - discarding active episode and stopping ...")
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    escape_listener = _EscapeStopListener(stop_event)
    escape_listener.start()

    try:
        record_log.info("Starting tracking before moving real arms.")
        tracker.start()
        tracker_started = True
        gripper_pair = connect_feetech(args)
        if gripper_pair is not None:
            grippers = FeetechGripperSampler(
                gripper_pair, sample_hz=args.feetech_sample_hz
            )
            grippers.start()

        real_env.setup(repair=not args.skip_can_repair)
        real_env.connect()
        record_log.info("Selected home pose: %s", home_pose_name)
        real_env.home(home_q)
        record_log.info(
            "Joint trajectory playback: %.1f Hz with %.0f ms delay.",
            args.command_rate_hz,
            args.trajectory_delay_ms,
        )
        space_listener.start()

        from handumi.dataset import EpisodeResult, write_dataset

        layout = canonical_joint_layout(runtime)
        existing_episodes = _existing_episode_count(args.output_dir) if args.resume else 0
        results: list[EpisodeResult] = []
        record_log.info("Recording vector dataset at: %s", args.output_dir)

        clap_detector = DoubleClapDetector()
        recorded = 0
        restart_active = False
        while (
            args.num_episodes <= 0 or recorded < args.num_episodes
        ) and not stop_event.is_set():
            ep_num = existing_episodes + recorded + 1
            ep_total = "inf" if args.num_episodes <= 0 else str(args.num_episodes)
            if not _wait_for_tracking(tracker, stop_event):
                break
            record_log.info("--- Episode %d/%s ---", ep_num, ep_total)
            if restart_active:
                restart_active = False
                record_log.info("  Restarting episode %d immediately ...", ep_num)
            elif not args.space_start:
                record_log.info(
                    "  Double-squeeze right gripper to start episode %d ...",
                    ep_num,
                )
                if not _wait_for_clap(
                    grippers,
                    clap_detector,
                    stop_event,
                    side="right",
                ):
                    break
            else:
                record_log.info(
                    "  Press Space%s to start episode %d ...",
                    " or double-squeeze right gripper" if not args.skip_feetech else "",
                    ep_num,
                )
            controller.reset()
            states, actions, n_frames, status, _ = record_episode(
                tracker=tracker,
                grippers=grippers,
                real_env=real_env,
                controller=controller,
                runtime=runtime,
                home_q=home_q,
                enabled_sides=enabled_sides,
                space_listener=space_listener,
                clap_detector=clap_detector,
                episode_time_s=args.episode_time_s,
                fps=args.fps,
                task=args.task,
                stop_event=stop_event,
                play_sounds=play_sounds,
                initial_start_sides=enabled_sides if not args.space_start else (),
                sync_lag_s=args.sync_lag_s,
                max_sync_skew_s=args.max_sync_skew_s,
                gripper_stale_timeout_s=args.gripper_stale_timeout_s,
                sensor_loss_timeout_s=args.sensor_loss_timeout_s,
                tracking_loss_timeout_s=args.tracking_loss_timeout_s,
                command_player=command_player,
                motion_smoother=motion_smoother,
            )
            if status == "repeat":
                record_log.warning("Episode restart requested (%d frames discarded).", n_frames)
                log_say("Restart recording", play_sounds=play_sounds)
                real_env.move_home(home_q)
                restart_active = True
                continue
            if n_frames == 0 or status in {
                "tracking_lost",
                "sensor_unhealthy",
                "interrupted",
            }:
                record_log.warning("Episode discarded (%s, %d frames).", status, n_frames)
                log_say("Episode discarded", play_sounds=play_sounds)
                if status == "interrupted":
                    break
                real_env.move_home(home_q)
                continue
            results.append(
                EpisodeResult(
                    episode_index=recorded,
                    states=states,
                    actions=actions,
                    task=args.task,
                    calibration_id=-1,
                    source_kind=1,
                )
            )
            recorded += 1
            record_log.info("Episode %d saved (%d frames).", ep_num, n_frames)
            log_say(
                f"Episode {ep_num} saved, {n_frames} frames",
                play_sounds=play_sounds,
            )
            real_env.move_home(home_q)

        if results:
            write_dataset(
                output_root=args.output_dir,
                source_root=args.output_dir,
                source_info={"features": {}, "handumi": {}},
                episodes=results,
                robot_type=runtime.config.kind,
                joint_names=layout.names,
                fps=args.fps,
                resume=args.resume,
                handumi_metadata={
                    "recording_device": args.device,
                    "capture_schema": HANDUMI_CAPTURE_SCHEMA,
                    "state_layout": "yaml_arm_joints_plus_logical_gripper_width_m",
                    "state_semantics": "real_robot_joint_feedback",
                    "action_semantics": "next_step_teleop_joint_command",
                    "trajectory_command_rate_hz": args.command_rate_hz,
                    "trajectory_delay_ms": args.trajectory_delay_ms,
                    "observation_action_alignment": (
                        "observation.state[t] is canonical backend feedback; "
                        "action[t] is the next recorded teleop command."
                    ),
                    "source_kind_ids": {"converted": 0, "teleop": 1, "unknown": -1},
                    "calibration_id_semantics": (
                        "-1 means no per-episode calibration artifact is referenced"
                    ),
                    "sync_lag_s": args.sync_lag_s,
                    "max_sync_skew_s": args.max_sync_skew_s,
                    "joint_names": layout.names,
                    "target_robot": _robot_metadata(args.robot),
                    "repo_id": args.repo_id,
                },
            )
        record_log.info("Done. Recorded %d episode(s). Dataset at: %s", recorded, args.output_dir)
    finally:
        escape_listener.stop()
        space_listener.close()
        try:
            command_player.stop()
        finally:
            try:
                real_env.disconnect()
            finally:
                if grippers is not None:
                    grippers.stop()
                if gripper_pair is not None:
                    gripper_pair.close()
                if tracker_started:
                    tracker.stop()
                log_say("Exiting", play_sounds=play_sounds, blocking=True)


def _existing_episode_count(root: Path) -> int:
    info_path = Path(root) / "meta" / "info.json"
    if not info_path.exists():
        return 0
    import json

    return int(json.loads(info_path.read_text()).get("total_episodes", 0))


def _default_output_dir() -> Path:
    return Path("outputs") / f"teleop_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def main() -> None:
    _run_record()


if __name__ == "__main__":
    main()
