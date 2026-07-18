#!/usr/bin/env python3
"""Record joint-level real-robot teleoperation demonstrations.

This recorder is the data-capture sibling of ``handumi-teleop-real``.  The
operator still drives the robot with HandUMI tracking and real gripper widths,
but every LeRobot row stores robot joints directly:

* ``observation.state`` is the robot feedback/configuration read from the real
  backend for this control tick.
* ``action`` is the joint command produced by the HandUMI teleop controller and
  sent to the robot on that same tick.

Those columns intentionally differ.  The action is the policy target; the
observation is what the robot reports or schedules after transport/backend
latency and vendor-side smoothing.
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from handumi.calibration.control_tcp import (
    calibration_path_for_robot_device,
    load_controller_tcp_calibration,
)
from handumi.cameras import (
    build_camera_specs,
    connect_cameras,
    disconnect_cameras,
    read_camera_samples,
    resolve_camera_ids,
)
from handumi.config import DEFAULT_RIG_CONFIG
from handumi.dataset.raw import (
    HANDUMI_CAPTURE_SCHEMA,
    camera_health_features,
    capture_timing_features,
    feetech_features,
)
from handumi.feetech import FeetechGripperPair, FeetechGripperSampler, GripperWidths
from handumi.real.registry import REAL_BACKEND_NAMES, make_real_backend
from handumi.retargeting.handumi_to_robot import raw_state_pose7_pair
from handumi.robots.registry import load_embodiment, resolve_home_q
from handumi.scripts.record import (
    _EscapeStopListener,
    _camera_arg,
    _capture_sources_metadata,
    _recording_tcp_calibration_metadata,
    _robot_metadata,
    _selected_camera_names,
    _update_info_json,
    _validate_finalized_lerobot_dataset,
    _validate_unique_camera_ids,
    _wait_for_clap,
    _wait_for_tracking,
    _write_dataset_readme,
    build_tracker,
    connect_feetech,
)
from handumi.scripts.teleop_real import (
    _enabled_tracking_ok,
    _enabled_sides,
    _latest_widths,
    _tracking_world_map,
    _validate_feetech_ready,
)
from handumi.scripts.teleop_sim import KeyboardSpaceListener, _sample_state
from handumi.synchronization import (
    SustainedHealthGate,
    capture_timing_frame,
    synchronized_gripper_frame,
)
from handumi.teleop.core import TeleopController
from handumi.tracking.base import TrackingProvider
from handumi.tracking.gestures import DoubleClapDetector
from handumi.utils.speech import log_say

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("handumi.record_teleop")

SIDE_CHOICES = ("left", "right", "both")


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--device", choices=("pico", "meta"), required=True)
    p.add_argument("--robot", choices=REAL_BACKEND_NAMES, default="piper")
    p.add_argument("--home-pose", default=None)
    p.add_argument("--side", choices=SIDE_CHOICES, default="both")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--episode-time-s", type=float, default=60.0)
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--task", type=str, default="HandUMI real teleop recording")
    p.add_argument("--repo-id", type=str, default="local/handumi_teleop_dataset")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--translation-scale", type=float, default=1.0)
    p.add_argument("--space-start", action="store_true")
    p.add_argument("--no-sounds", action="store_true")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--dataset-license", default="other")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--vcodec", type=str, default="h264")
    p.add_argument("--sync-lag-s", type=float, default=0.04)
    p.add_argument("--max-sync-skew-s", type=float, default=0.06)
    p.add_argument("--camera-stale-timeout-s", type=float, default=0.25)
    p.add_argument("--gripper-stale-timeout-s", type=float, default=0.10)
    p.add_argument("--sensor-loss-timeout-s", type=float, default=1.0)
    p.add_argument("--feetech-sample-hz", type=float, default=100.0)
    p.add_argument("--tracking-loss-timeout-s", type=float, default=1.0)
    p.add_argument("--skip-can-repair", action="store_true")
    p.add_argument("--rig-config", type=Path, default=DEFAULT_RIG_CONFIG)
    p.add_argument("--controller-tcp-calibration", type=Path, default=None)

    p.add_argument("--cam-ids", nargs="+", type=_camera_arg, default=None)
    p.add_argument("--wrist-cameras", action="store_true")
    p.add_argument("--workspace-camera", action="store_true")
    only_camera = p.add_mutually_exclusive_group()
    only_camera.add_argument(
        "--only-left-camera",
        "--only-left-cameras",
        dest="only_left_camera",
        action="store_true",
    )
    only_camera.add_argument(
        "--only-right-camera",
        "--only-right-cameras",
        dest="only_right_camera",
        action="store_true",
    )
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--cam-fps", type=int, default=30)

    p.add_argument("--feetech-port", type=str, default=None)
    p.add_argument("--skip-feetech", action="store_true")
    p.add_argument("--quest-ip", type=str, default=None)
    p.add_argument("--tcp-port", type=int, default=None)
    p.add_argument("--sync-port", type=int, default=None)
    p.add_argument(
        "--pico-mode", choices=("mandos", "object", "whole-body"), default="mandos"
    )
    pico_transport = p.add_mutually_exclusive_group()
    pico_transport.add_argument("--pico-adb", action="store_true")
    pico_transport.add_argument("--pico-wifi", action="store_true")
    p.add_argument("--skip-adb-check", action="store_true")
    return p.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise SystemExit("--fps must be > 0.")
    if args.episode_time_s <= 0:
        raise SystemExit("--episode-time-s must be > 0.")
    if args.num_episodes < 0:
        raise SystemExit("--num-episodes must be >= 0.")
    if args.skip_feetech and not args.space_start:
        raise SystemExit(
            "--skip-feetech disables double-clap; add --space-start so teleop "
            "can begin."
        )
    for name in (
        "sync_lag_s",
        "max_sync_skew_s",
        "camera_stale_timeout_s",
        "gripper_stale_timeout_s",
        "sensor_loss_timeout_s",
        "feetech_sample_hz",
        "tracking_loss_timeout_s",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be greater than zero.")


def record_episode(
    *,
    dataset,
    cameras: list,
    cam_names: list[str],
    tracker: TrackingProvider,
    grippers: FeetechGripperSampler | FeetechGripperPair | None,
    real_env,
    controller: TeleopController,
    home_q: np.ndarray,
    enabled_sides: tuple[str, ...],
    space_listener: KeyboardSpaceListener,
    clap_detector: DoubleClapDetector,
    episode_time_s: float,
    fps: int,
    task: str,
    cam_width: int,
    cam_height: int,
    stop_event: threading.Event,
    play_sounds: bool,
    initial_start_sides: tuple[str, ...],
    sync_lag_s: float,
    max_sync_skew_s: float,
    camera_stale_timeout_s: float,
    gripper_stale_timeout_s: float,
    sensor_loss_timeout_s: float,
    tracking_loss_timeout_s: float,
) -> tuple[int, str, np.ndarray]:
    interval = 1.0 / fps
    n_frames = 0
    start_t: float | None = None
    episode_start_ns: int | None = None
    status = "recorded"
    pending_start_sides = initial_start_sides
    tracking_lost_since: float | None = None
    last_recovery_attempt = 0.0
    health_gate = SustainedHealthGate(sensor_loss_timeout_s)
    max_sync_skew_ns = int(max_sync_skew_s * 1e9)
    sync_lag_ns = int(sync_lag_s * 1e9)
    q = controller.q.copy()

    while True:
        loop_start = time.perf_counter()
        record_time_ns = time.monotonic_ns()
        if episode_start_ns is None:
            episode_start_ns = record_time_ns

        if stop_event.is_set():
            status = "interrupted"
            dataset.clear_episode_buffer()
            break
        if start_t is not None and loop_start - start_t >= episode_time_s:
            break

        sample = tracker.latest()
        side_tracked = {"left": sample.left_tracked, "right": sample.right_tracked}
        if not _enabled_tracking_ok(side_tracked, enabled_sides):
            if tracking_lost_since is None:
                tracking_lost_since = loop_start
                held = real_env.hold(q)
                controller.tracking_lost(held)
                q = held
                log.warning("Tracking lost; robot command held and episode discarded.")
                log_say("tracking lost", play_sounds=play_sounds)
            if loop_start - tracking_lost_since >= tracking_loss_timeout_s:
                status = "tracking_lost"
                dataset.clear_episode_buffer()
                break
            recover = getattr(tracker, "recover", None)
            if callable(recover) and loop_start - last_recovery_attempt >= 3.0:
                last_recovery_attempt = loop_start
                recover()
            _sleep_until_next_tick(interval, loop_start)
            continue
        if tracking_lost_since is not None:
            tracking_lost_since = None
            log.info("Tracking stream recovered; waiting for a fresh anchor.")

        observation_q = real_env.read(base_q=q)
        immediate_widths = _latest_widths(grippers)
        state = _sample_state(sample, immediate_widths)
        source_poses: dict[str, np.ndarray] = dict(
            zip(("left", "right"), raw_state_pose7_pair(state), strict=True)
        )

        start_sides = pending_start_sides
        pending_start_sides = ()
        if space_listener.consume_space():
            start_sides = controller.idle_sides()
        if clap_detector.update(
            immediate_widths.left_mm, immediate_widths.right_mm, loop_start
        ):
            if controller.active:
                q = controller.reset()
                start_t = None
                n_frames = 0
                dataset.clear_episode_buffer()
                log.info("Double clap detected; restarting episode from home.")
                log_say("restart recording", play_sounds=play_sounds)
                real_env.move_home(home_q)
                continue
            start_sides = enabled_sides

        anchored = controller.anchor(source_poses, side_tracked, start_sides)
        if anchored and start_t is None:
            start_t = loop_start
            log.info("Teleop episode started after anchoring %s.", "/".join(anchored))
            log_say("recording episode", play_sounds=play_sounds)

        openings = {
            "left": immediate_widths.left_normalized,
            "right": immediate_widths.right_normalized,
        }
        teleop_step = controller.step(source_poses, side_tracked, openings)
        action_q = teleop_step.q
        real_env.write(action_q, openings)
        real_env.check_health()
        q = action_q

        if start_t is None:
            _sleep_until_next_tick(interval, loop_start)
            continue

        target_time_ns = max(episode_start_ns, record_time_ns - sync_lag_ns)
        cam_frames, camera_health = read_camera_samples(
            cameras,
            cam_names,
            target_time_ns=target_time_ns,
            record_time_ns=record_time_ns,
            width=cam_width,
            height=cam_height,
            stale_timeout_s=camera_stale_timeout_s,
            max_sync_skew_s=max_sync_skew_s,
        )
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
            **camera_health,
            "feetech": gripper_frame.healthy_for_gate,
            "tracking": tracking_sync_ok,
        }
        _, timed_out_sensors = health_gate.update(sensor_health, record_time_ns)
        if timed_out_sensors:
            status = "sensor_unhealthy"
            log.error(
                "Sensor health timed out: %s.",
                ", ".join(sorted(timed_out_sensors)),
            )
            dataset.clear_episode_buffer()
            break

        dataset.add_frame(
            {
                **cam_frames,
                **build_joint_frame(
                    observation_q=observation_q,
                    action_q=action_q,
                    widths=gripper_frame.widths,
                ),
                **gripper_frame.frame,
                **capture_timing_frame(target_time_ns, record_time_ns),
                "task": task,
            }
        )
        n_frames += 1
        _sleep_until_next_tick(interval, loop_start)

    return n_frames, status, q


def main() -> None:
    args = parse_args()
    _validate_args(args)
    if args.output_dir is None:
        args.output_dir = _default_output_dir()
    play_sounds = not args.no_sounds
    stop_event = threading.Event()

    log.info("Loading %s IK solver.", args.robot)
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
    cameras: list[Any] = []
    tracker_started = False
    space_listener = KeyboardSpaceListener(enabled=args.space_start)

    def _on_signal(signum, frame):
        del signum, frame
        log.info("Signal received - discarding active episode and stopping ...")
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    escape_listener = _EscapeStopListener(stop_event)
    escape_listener.start()

    try:
        log.info("Starting tracking before moving real arms.")
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
        log.info("Selected home pose: %s", home_pose_name)
        real_env.home(home_q)
        space_listener.start()

        camera_names = _selected_camera_names(args)
        cam_ids = resolve_camera_ids(
            args.cam_ids,
            args.rig_config,
            camera_names=camera_names,
        )
        _validate_unique_camera_ids(camera_names, cam_ids)
        camera_specs, _ = build_camera_specs(
            cam_ids,
            camera_names=camera_names,
            laptop_camera=False,
            laptop_cam_id=0,
            laptop_cam_name="laptop",
        )
        cam_names = [spec["name"] for spec in camera_specs]
        cameras = connect_cameras(
            camera_specs,
            fps=args.cam_fps,
            width=args.cam_width,
            height=args.cam_height,
            zero_non_laptop=False,
        )

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        use_videos = not args.no_video
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            root=args.output_dir,
            robot_type=f"{args.robot}_teleop_real",
            features=build_features(
                cam_names,
                args.cam_width,
                args.cam_height,
                use_videos,
                runtime.joint_names,
            ),
            use_videos=use_videos,
            image_writer_processes=0,
            image_writer_threads=max(1, 4 * len(cam_names)),
            vcodec=args.vcodec,
        )
        log.info("Dataset created at: %s", dataset.root)

        clap_detector = DoubleClapDetector()
        recorded = 0
        while (
            args.num_episodes <= 0 or recorded < args.num_episodes
        ) and not stop_event.is_set():
            ep_num = dataset.num_episodes + 1
            ep_total = "inf" if args.num_episodes <= 0 else str(args.num_episodes)
            if not _wait_for_tracking(tracker, stop_event):
                break
            log.info(
                "--- Episode %d/%s: double clap%s to start ---",
                ep_num,
                ep_total,
                " or Space" if args.space_start else "",
            )
            if not args.space_start:
                if not _wait_for_clap(
                    grippers,
                    clap_detector,
                    stop_event,
                    side="right",
                ):
                    break
                clap_detector = DoubleClapDetector()
            controller.reset()
            n_frames, status, _ = record_episode(
                dataset=dataset,
                cameras=cameras,
                cam_names=cam_names,
                tracker=tracker,
                grippers=grippers,
                real_env=real_env,
                controller=controller,
                home_q=home_q,
                enabled_sides=enabled_sides,
                space_listener=space_listener,
                clap_detector=clap_detector,
                episode_time_s=args.episode_time_s,
                fps=args.fps,
                task=args.task,
                cam_width=args.cam_width,
                cam_height=args.cam_height,
                stop_event=stop_event,
                play_sounds=play_sounds,
                initial_start_sides=enabled_sides if not args.space_start else (),
                sync_lag_s=args.sync_lag_s,
                max_sync_skew_s=args.max_sync_skew_s,
                camera_stale_timeout_s=args.camera_stale_timeout_s,
                gripper_stale_timeout_s=args.gripper_stale_timeout_s,
                sensor_loss_timeout_s=args.sensor_loss_timeout_s,
                tracking_loss_timeout_s=args.tracking_loss_timeout_s,
            )
            if n_frames == 0 or status in {
                "tracking_lost",
                "sensor_unhealthy",
                "interrupted",
            }:
                log.warning("Episode discarded (%s, %d frames).", status, n_frames)
                log_say("Episode discarded", play_sounds=play_sounds)
                dataset.clear_episode_buffer()
                if status == "interrupted":
                    break
                real_env.move_home(home_q)
                continue
            dataset.save_episode()
            recorded += 1
            log.info("Episode %d saved (%d frames).", ep_num, n_frames)
            log_say(
                f"Episode {ep_num} saved, {n_frames} frames",
                play_sounds=play_sounds,
            )
            real_env.move_home(home_q)

        _finalize_dataset(
            dataset,
            args=args,
            camera_specs=camera_specs,
            cameras=cameras,
            grippers=grippers,
            runtime=runtime,
        )
        log.info("Done. Recorded %d episode(s). Dataset at: %s", recorded, dataset.root)
    finally:
        escape_listener.stop()
        space_listener.close()
        try:
            real_env.disconnect()
        finally:
            if grippers is not None:
                grippers.stop()
            if gripper_pair is not None:
                gripper_pair.close()
            disconnect_cameras(cameras)
            if tracker_started:
                tracker.stop()
            log_say("Exiting", play_sounds=play_sounds, blocking=True)


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
    log.info("controller->TCP calibration: %s", source)
    return load_controller_tcp_calibration(path)


def _finalize_dataset(
    dataset,
    *,
    args: argparse.Namespace,
    camera_specs: list[dict[str, object]],
    cameras: list[object | None],
    grippers: object | None,
    runtime,
) -> None:
    dataset.finalize()
    root = Path(dataset.root)
    robot_metadata = _robot_metadata(args.robot)
    calibration_metadata, _ = _recording_tcp_calibration_metadata(
        robot_metadata=robot_metadata,
        device=args.device,
        explicit_path=args.controller_tcp_calibration,
    )
    updated_info = _update_info_json(
        root,
        {
            "recording_device": args.device,
            "capture_schema": HANDUMI_CAPTURE_SCHEMA,
            "state_semantics": "real_robot_joint_feedback",
            "action_semantics": "handumi_teleop_joint_command",
            "observation_action_alignment": (
                "observation.state is backend.read(...) before write; action is "
                "the TeleopController command sent on the same row"
            ),
            "sync_lag_s": args.sync_lag_s,
            "max_sync_skew_s": args.max_sync_skew_s,
            "joint_names": list(runtime.joint_names),
            "target_robot": robot_metadata,
            "controller_tcp_calibration": calibration_metadata,
            "cameras": [
                {"name": spec["name"], "index_or_path": spec["id"]}
                for spec in camera_specs
            ],
            "sources": _capture_sources_metadata(camera_specs, cameras, grippers),
        },
    )
    if updated_info is not None:
        dataset.meta.info = updated_info
    card_kwargs = _write_dataset_readme(
        root,
        repo_id=args.repo_id,
        task=args.task,
        license_id=args.dataset_license,
    )
    _validate_finalized_lerobot_dataset(root)
    if args.push_to_hub:
        dataset.push_to_hub(
            license=args.dataset_license,
            tags=["HandUMI", args.robot, "real-teleop"],
            **card_kwargs,
        )


def _default_output_dir() -> Path:
    return Path("outputs") / f"teleop_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _sleep_until_next_tick(interval: float, loop_start: float) -> None:
    dt = time.perf_counter() - loop_start
    if (sleep := interval - dt) > 0:
        time.sleep(sleep)
    else:
        log.warning("Loop slower than target (%.1f Hz actual).", 1.0 / max(dt, 1e-6))


if __name__ == "__main__":
    main()
