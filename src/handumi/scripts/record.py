#!/usr/bin/env python3
"""Unified HandUMI recorder for PICO and Meta Quest tracking backends.

Episode control: timed by default (--episode-time-s), PICO buttons with
--manual-control, or hands-free with --clap-control (a double clap — close
both grippers twice within ~1.2s — starts and stops each episode; works
with either device since it reads the Feetech widths).

Spoken status announcements ("Recording episode 3", "Episode 3 saved, 812
frames", ...) are on by default — pass --no-sounds to disable them. Without
--output-dir each run writes a fresh outputs/<YYYYMMDD_HHMMSS>/ folder
(outputs/ is gitignored).
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from handumi.calibration.control_tcp import ControllerTcpCalibration
from handumi.cameras import (
    build_camera_specs,
    connect_cameras,
    disconnect_cameras,
    read_camera_frames,
    resolve_camera_ids,
)
from handumi.dataset.raw import (
    feetech_features,
    pose_to_state_vector,
    raw_state_feature,
    raw_tracking_features,
)
from handumi.feetech import (
    PORTS_PATH,
    FeetechGripperPair,
    GripperWidths,
    assert_calibrated,
    load_config,
    user_calibration_path,
    zero_gripper_widths,
)
from handumi.feetech.bus import FeetechUnavailableError
from handumi.robots.utils import IDENTITY_POSE7
from handumi.tracking.base import ControllerPairSample, TrackingProvider
from handumi.tracking.gestures import DoubleClapDetector
from handumi.tracking.meta_quest import MetaQuestConfig, MetaQuestTrackingProvider
from handumi.tracking.pico import (
    START_BUTTON_CHOICES,
    PicoTrackingProvider,
    read_start_button_value,
    wait_for_manual_start,
    wait_for_start_button,
)
from handumi.tracking.transforms import Pose
from handumi.utils.speech import log_say

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("handumi.record")


def build_features(
    cam_names: list[str],
    cam_width: int,
    cam_height: int,
    use_videos: bool,
) -> dict:
    img_dtype = "video" if use_videos else "image"
    features: dict = {}
    for cam in cam_names:
        features[f"observation.images.{cam}"] = {
            "dtype": img_dtype,
            "shape": (cam_height, cam_width, 3),
            "names": ["height", "width", "channel"],
        }
    features["observation.state"] = _tuple_shape(raw_state_feature())
    features["action"] = _tuple_shape(raw_state_feature())
    features.update(feetech_features())
    features.update(raw_tracking_features())
    return features


def _tuple_shape(feature: dict) -> dict:
    feature = dict(feature)
    feature["shape"] = tuple(feature["shape"])
    return feature


def build_observation(sample: ControllerPairSample, widths: GripperWidths) -> dict:
    left_controller = _pose_from_pose7(sample.left_controller_pose)
    right_controller = _pose_from_pose7(sample.right_controller_pose)
    state = pose_to_state_vector(
        left_controller,
        right_controller,
        widths.left,
        widths.right,
    )
    tracking_frame = {
        key: value
        for key, value in sample.tracking_frame().items()
        if "_tcp_pose" not in key
    }
    return {
        "observation.state": state,
        "action": state.copy(),
        "observation.feetech.left_ticks": np.array([widths.left_ticks], dtype=np.int64),
        "observation.feetech.right_ticks": np.array([widths.right_ticks], dtype=np.int64),
        "observation.feetech.left_width_mm": np.array([widths.left_mm], dtype=np.float32),
        "observation.feetech.right_width_mm": np.array([widths.right_mm], dtype=np.float32),
        "observation.feetech.left_normalized": np.array([widths.left_normalized], dtype=np.float32),
        "observation.feetech.right_normalized": np.array([widths.right_normalized], dtype=np.float32),
        **tracking_frame,
    }


def _pose_from_pose7(pose7: np.ndarray) -> Pose:
    pose = np.asarray(pose7, dtype=np.float32).reshape(7)
    return Pose(pose[:3], pose[3:7])


def record_episode(
    *,
    dataset,
    cameras: list,
    cam_names: list[str],
    tracker: TrackingProvider,
    grippers: FeetechGripperPair | None,
    episode_time_s: float,
    fps: int,
    task: str,
    cam_width: int,
    cam_height: int,
    stop_event: threading.Event,
    manual_control: bool,
    start_button: str,
    repeat_button: str,
    finish_button: str,
    start_threshold: float,
    clap_detector: DoubleClapDetector | None = None,
) -> tuple[int, str]:
    control_interval = 1.0 / fps
    n_frames = 0
    start_t = time.perf_counter()
    status = "recorded"
    clap_control = clap_detector is not None
    xrt = getattr(tracker, "xrt", None)
    prev_start = (
        read_start_button_value(xrt, start_button) >= start_threshold
        if manual_control and xrt is not None
        else False
    )
    prev_repeat = (
        read_start_button_value(xrt, repeat_button) >= start_threshold
        if manual_control and xrt is not None
        else False
    )
    prev_finish = (
        read_start_button_value(xrt, finish_button) >= start_threshold
        if manual_control and xrt is not None
        else False
    )

    # Clap-controlled episodes end on the next double clap, not on a timer.
    timed = not manual_control and not clap_control

    while True:
        loop_start = time.perf_counter()
        elapsed = loop_start - start_t
        if (timed and elapsed >= episode_time_s) or stop_event.is_set():
            break

        if manual_control and xrt is not None:
            start_pressed = read_start_button_value(xrt, start_button) >= start_threshold
            repeat_pressed = read_start_button_value(xrt, repeat_button) >= start_threshold
            finish_pressed = read_start_button_value(xrt, finish_button) >= start_threshold
            start_rise = start_pressed and not prev_start
            repeat_rise = repeat_pressed and not prev_repeat
            finish_rise = finish_pressed and not prev_finish
            prev_start, prev_repeat, prev_finish = start_pressed, repeat_pressed, finish_pressed
            if repeat_rise:
                status = "repeat"
                dataset.clear_episode_buffer()
                break
            if finish_rise:
                status = "finish"
                break
            if start_rise:
                status = "recorded"
                break

        cam_frames = read_camera_frames(cameras, cam_names, width=cam_width, height=cam_height)
        widths = zero_gripper_widths() if grippers is None else grippers.read_normalized_widths()
        if clap_control and clap_detector.update(widths.left_mm, widths.right_mm, loop_start):
            break
        sample = tracker.latest()
        dataset.add_frame({**cam_frames, **build_observation(sample, widths), "task": task})
        n_frames += 1

        dt = time.perf_counter() - loop_start
        sleep = control_interval - dt
        if sleep > 0:
            time.sleep(sleep)
        else:
            log.warning("Loop slower than %d Hz (%.1f Hz actual).", fps, 1.0 / max(dt, 1e-6))

    return n_frames, status


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record HandUMI data with PICO or Meta Quest.")
    p.add_argument("--device", choices=("pico", "meta"), required=True)
    p.add_argument("--cam-ids", nargs="+", type=_camera_arg, default=None)
    p.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--cam-fps", type=int, default=30)
    p.add_argument("--feetech-config", type=Path, default=PORTS_PATH)
    p.add_argument("--feetech-port", type=str, default=None)
    p.add_argument("--skip-feetech", action="store_true")
    p.add_argument("--repo-id", type=str, default="local/handumi_dataset")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Dataset folder. Defaults to a fresh outputs/<YYYYMMDD_HHMMSS>/ "
        "named after when recording started (outputs/ is gitignored).",
    )
    p.add_argument("--task", type=str, default="HandUMI recording")
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--episode-time-s", type=float, default=60.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--vcodec", type=str, default="h264")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument(
        "--controller-tcp-calibration",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )

    p.add_argument("--tracking-config", type=Path, default=Path("configs/tracking_meta_quest.yaml"))
    p.add_argument("--quest-ip", type=str, default=None)
    p.add_argument("--tcp-port", type=int, default=None)
    p.add_argument("--sync-port", type=int, default=None)

    p.add_argument("--pico-mode", choices=("mandos", "object", "whole-body"), default="mandos")
    pico_transport = p.add_mutually_exclusive_group()
    pico_transport.add_argument("--pico-adb", action="store_true")
    pico_transport.add_argument("--pico-wifi", action="store_true")
    p.add_argument("--skip-adb-check", action="store_true")
    p.add_argument("--start-button", choices=START_BUTTON_CHOICES, default="enter")
    p.add_argument("--start-threshold", type=float, default=0.75)
    p.add_argument("--manual-control", action="store_true")
    p.add_argument("--repeat-button", choices=START_BUTTON_CHOICES, default="B")
    p.add_argument("--finish-button", choices=START_BUTTON_CHOICES, default="Y")
    p.add_argument(
        "--clap-control",
        action="store_true",
        help="Hands-free: a double clap (close both grippers twice within "
        "~1.2s) starts and stops each episode. Needs real Feetech widths.",
    )
    p.add_argument(
        "--no-sounds",
        action="store_true",
        help="Disable spoken episode-status announcements (start/save/discard/stop).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.manual_control and args.device != "pico":
        raise SystemExit("--manual-control currently requires --device pico.")
    if args.manual_control and args.start_button == "enter":
        args.start_button = "A"
        log.info("--manual-control set: using PICO A as start/stop button.")
    if args.clap_control and args.skip_feetech:
        raise SystemExit("--clap-control needs real Feetech widths; drop --skip-feetech.")
    if args.clap_control and args.manual_control:
        raise SystemExit("--clap-control and --manual-control are mutually exclusive.")
    if args.output_dir is None:
        args.output_dir = _default_output_dir()
    play_sounds = not args.no_sounds

    log.info("--- Tracking setup ---")
    calibration = ControllerTcpCalibration(
        left=IDENTITY_POSE7.astype(np.float32).copy(),
        right=IDENTITY_POSE7.astype(np.float32).copy(),
        source=None,
    )
    tracker = build_tracker(args, calibration)
    tracker.start()

    log.info("--- Camera setup ---")
    cam_ids = resolve_camera_ids(args.cam_ids, args.camera_config)
    camera_specs, _ = build_camera_specs(
        cam_ids, laptop_camera=False, laptop_cam_id=0, laptop_cam_name="laptop"
    )
    cam_names = [spec["name"] for spec in camera_specs]
    cameras = connect_cameras(
        camera_specs,
        fps=args.cam_fps,
        width=args.cam_width,
        height=args.cam_height,
        zero_non_laptop=False,
    )

    log.info("--- Feetech setup ---")
    grippers = connect_feetech(args)

    log.info("--- Dataset setup ---")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    use_videos = not args.no_video
    features = build_features(cam_names, args.cam_width, args.cam_height, use_videos)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=args.output_dir,
        robot_type="handumi_raw",
        features=features,
        use_videos=use_videos,
        image_writer_processes=0,
        image_writer_threads=max(1, 4 * len(cam_names)),
        vcodec=args.vcodec,
    )
    log.info("Dataset created at: %s", dataset.root)

    stop_event = threading.Event()

    def _on_signal(signum, frame):
        log.info("Signal received - stopping after current episode ...")
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    recorded = 0
    clap_detector = DoubleClapDetector() if args.clap_control else None
    try:
        while (args.num_episodes <= 0 or recorded < args.num_episodes) and not stop_event.is_set():
            ep_num = dataset.num_episodes + 1
            ep_total = "inf" if args.num_episodes <= 0 else str(args.num_episodes)
            log.info("--- Episode %d/%s ---", ep_num, ep_total)
            if args.clap_control:
                log.info("  Double-clap both grippers to start episode %d ...", ep_num)
                if not _wait_for_clap(grippers, clap_detector, stop_event):
                    break
            elif args.manual_control:
                action = wait_for_manual_start(
                    getattr(tracker, "xrt"),
                    start_button=args.start_button,
                    finish_button=args.finish_button,
                    threshold=args.start_threshold,
                    stop_event=stop_event,
                )
                if action == "finish":
                    break
            elif args.start_button == "enter":
                input(f"  Press ENTER to start recording episode {ep_num} ...")
            elif args.device == "pico":
                if not wait_for_start_button(
                    getattr(tracker, "xrt"),
                    button=args.start_button,
                    threshold=args.start_threshold,
                    stop_event=stop_event,
                ):
                    break
            else:
                raise SystemExit("--start-button other than enter currently requires --device pico.")

            stop_event.clear()
            log_say(f"Recording episode {ep_num}", play_sounds=play_sounds)
            n_frames, status = record_episode(
                dataset=dataset,
                cameras=cameras,
                cam_names=cam_names,
                tracker=tracker,
                grippers=grippers,
                episode_time_s=args.episode_time_s,
                fps=args.fps,
                task=args.task,
                cam_width=args.cam_width,
                cam_height=args.cam_height,
                stop_event=stop_event,
                manual_control=args.manual_control,
                start_button=args.start_button,
                repeat_button=args.repeat_button,
                finish_button=args.finish_button,
                start_threshold=args.start_threshold,
                clap_detector=clap_detector,
            )
            if n_frames == 0 or status == "repeat":
                log.warning("Episode discarded (%s, %d frames).", status, n_frames)
                log_say("Episode discarded", play_sounds=play_sounds)
                dataset.clear_episode_buffer()
                if status == "finish":
                    break
                continue
            dataset.save_episode()
            recorded += 1
            log.info("Episode %d saved (%d frames).", ep_num, n_frames)
            log_say(f"Episode {ep_num} saved, {n_frames} frames", play_sounds=play_sounds)
            if status == "finish":
                break
    finally:
        log_say("Stop recording", play_sounds=play_sounds, blocking=True)
        log.info("--- Finalising ---")
        dataset.finalize()
        _update_info_json(
            Path(dataset.root),
            {
                "recording_device": args.device,
                "tracking_schema": "controller_raw_v1",
                "state_semantics": "raw_controller_pose7_plus_gripper_widths",
            },
        )
        if args.push_to_hub:
            dataset.push_to_hub()
        disconnect_cameras(cameras)
        if grippers is not None:
            grippers.close()
        tracker.stop()
        log.info("Done. Recorded %d episode(s). Dataset at: %s", recorded, dataset.root)
        log_say("Exiting", play_sounds=play_sounds)


def build_tracker(args: argparse.Namespace, calibration) -> TrackingProvider:
    if args.device == "pico":
        transport = "wifi" if args.pico_wifi else "adb"
        return PicoTrackingProvider(
            calibration=calibration,
            mode=args.pico_mode,
            transport=transport,
            skip_adb_check=args.skip_adb_check,
        )

    base = (
        MetaQuestConfig.from_yaml(args.tracking_config)
        if args.tracking_config.exists()
        else MetaQuestConfig(quest_ip="")
    )
    config = MetaQuestConfig(
        quest_ip=args.quest_ip if args.quest_ip is not None else base.quest_ip,
        tcp_port=args.tcp_port if args.tcp_port is not None else base.tcp_port,
        sync_port=args.sync_port if args.sync_port is not None else base.sync_port,
        connect_retry_s=base.connect_retry_s,
    )
    return MetaQuestTrackingProvider(config=config, calibration=calibration)


def connect_feetech(args: argparse.Namespace) -> FeetechGripperPair | None:
    if args.skip_feetech:
        log.info("Feetech disabled: gripper widths will be zero-filled.")
        return None
    feetech_config = load_config(args.feetech_config)
    if args.feetech_port is not None:
        feetech_config = type(feetech_config)(
            port=args.feetech_port,
            baudrate=feetech_config.baudrate,
            protocol_version=feetech_config.protocol_version,
            left=feetech_config.left,
            right=feetech_config.right,
        )
    assert_calibrated(feetech_config, source=user_calibration_path())
    grippers = FeetechGripperPair(feetech_config)
    try:
        grippers.open()
    except FeetechUnavailableError as exc:
        raise SystemExit(str(exc)) from exc
    return grippers


def _wait_for_clap(
    grippers: FeetechGripperPair | None,
    clap_detector: DoubleClapDetector,
    stop_event: threading.Event,
) -> bool:
    """Poll Feetech widths until a double-clap fires (or ``stop_event`` sets)."""
    while not stop_event.is_set():
        widths = zero_gripper_widths() if grippers is None else grippers.read_normalized_widths()
        if clap_detector.update(widths.left_mm, widths.right_mm, time.perf_counter()):
            return True
        time.sleep(0.02)
    return False


def _default_output_dir() -> Path:
    """``outputs/<YYYYMMDD_HHMMSS>/`` named after the moment recording starts.

    ``outputs/`` is gitignored — datasets never get committed by accident.
    """
    return Path("outputs") / datetime.now().strftime("%Y%m%d_%H%M%S")


def _update_info_json(root: Path, handumi: dict[str, str]) -> None:
    path = root / "meta" / "info.json"
    if not path.exists():
        log.warning("Cannot write HandUMI metadata; missing %s", path)
        return
    info = json.loads(path.read_text())
    info["handumi"] = {**info.get("handumi", {}), **handumi}
    path.write_text(json.dumps(info, indent=4) + "\n")


def _camera_arg(value: str) -> int | str:
    return int(value) if value.isdigit() else value


if __name__ == "__main__":
    main()
