#!/usr/bin/env python3
"""
Record HandUMI raw data from wrist cameras, optional tracking, and Feetech encoders.

What gets recorded per frame
─────────────────────────────
  observation.images.left_wrist     RGB frames from left wrist camera  (video)
  observation.images.right_wrist    RGB frames from right wrist camera (video)
  observation.state                 HandUMI raw state                  (float32[16])
  action                            same as observation.state          (float32[16])
  observation.feetech.left_ticks          raw encoder ticks                      (int64[1])
  observation.feetech.right_ticks         raw encoder ticks                      (int64[1])
  observation.feetech.left_width_mm       calibrated aperture                    (float32[1])
  observation.feetech.right_width_mm      calibrated aperture                    (float32[1])
  observation.pico.*                      optional tracking features

No follower robot is required. Action currently mirrors the raw HandUMI state.

Usage
─────
  handumi-record-pico \
      --repo-id local/my_dataset \
      --output-dir datasets/my_dataset \
      --task "Pick and place cube" \
      --num-episodes 10 \
      --episode-time-s 60 \
      --fps 30

  (See bin/record_pico.sh for a ready-made launcher.)
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

import numpy as np

from handumi.cameras.preview import LaptopPreview, draw_laptop_overlay
from handumi.cameras.usb import (
    build_camera_specs,
    connect_cameras,
    disconnect_cameras,
    read_camera_frames,
    resolve_camera_ids,
)
from handumi.capture.reach import (
    REACH_BUDGETS_M,
    compute_reach_features,
    empty_reach_features,
    update_episode_reach_flags,
)
from handumi.dataset.raw import raw_state_feature
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
from handumi.tracking.pico import (
    MAX_MOTION_TRACKERS,
    START_BUTTON_CHOICES,
    empty_pico_frame,
    guess_lan_ip,
    init_xrt,
    launch_xrt_service,
    read_pico_frame,
    read_start_button_value,
    setup_adb_reverse,
    verify_adb_connection,
    wait_for_manual_start,
    wait_for_pico_data,
    wait_for_start_button,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s – %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("handumi.record")

# ── Dataset feature schema ─────────────────────────────────────────────────────


def build_features(
    cam_names: list[str],
    cam_width: int,
    cam_height: int,
    use_videos: bool,
    include_pico: bool,
) -> dict:
    """
    Build the feature schema dict expected by LeRobotDataset.create().
    """
    img_dtype = "video" if use_videos else "image"
    features: dict = {}

    for cam in cam_names:
        features[f"observation.images.{cam}"] = {
            "dtype": img_dtype,
            "shape": (cam_height, cam_width, 3),
            "names": ["height", "width", "channel"],
        }

    features["observation.state"] = raw_state_feature()
    features["action"] = raw_state_feature()

    features["observation.feetech.left_ticks"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    }
    features["observation.feetech.right_ticks"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    }
    features["observation.feetech.left_width_mm"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    }
    features["observation.feetech.right_width_mm"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    }
    features["observation.feetech.left_normalized"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    }
    features["observation.feetech.right_normalized"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    }

    if not include_pico:
        return features

    # PICO body tracking
    features["observation.pico.body_joints_pose"] = {
        "dtype": "float32",
        "shape": (24, 7),
        "names": None,
    }
    features["observation.pico.body_joints_velocity"] = {
        "dtype": "float32",
        "shape": (24, 6),
        "names": None,
    }
    features["observation.pico.body_joints_accel"] = {
        "dtype": "float32",
        "shape": (24, 6),
        "names": None,
    }

    # PICO controller / hand poses
    features["observation.pico.left_controller_pose"] = {
        "dtype": "float32",
        "shape": (7,),
        "names": ["x", "y", "z", "qx", "qy", "qz", "qw"],
    }
    features["observation.pico.headset_pose"] = {
        "dtype": "float32",
        "shape": (7,),
        "names": ["x", "y", "z", "qx", "qy", "qz", "qw"],
    }
    features["observation.pico.right_controller_pose"] = {
        "dtype": "float32",
        "shape": (7,),
        "names": ["x", "y", "z", "qx", "qy", "qz", "qw"],
    }
    features["observation.pico.left_hand_pose"] = {
        "dtype": "float32",
        "shape": (27, 7),
        "names": None,
    }
    features["observation.pico.right_hand_pose"] = {
        "dtype": "float32",
        "shape": (27, 7),
        "names": None,
    }
    features["observation.pico.timestamp_ns"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    }
    features["observation.pico.motion_tracker_pose"] = {
        "dtype": "float32",
        "shape": (MAX_MOTION_TRACKERS, 7),
        "names": None,
    }
    features["observation.pico.motion_tracker_velocity"] = {
        "dtype": "float32",
        "shape": (MAX_MOTION_TRACKERS, 6),
        "names": None,
    }
    features["observation.pico.motion_tracker_accel"] = {
        "dtype": "float32",
        "shape": (MAX_MOTION_TRACKERS, 6),
        "names": None,
    }
    features["observation.pico.motion_tracker_count"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    }
    features["observation.pico.motion_tracker_serial_hash"] = {
        "dtype": "int64",
        "shape": (MAX_MOTION_TRACKERS,),
        "names": None,
    }

    for robot in REACH_BUDGETS_M:
        for name in ("left_ratio", "right_ratio", "max_ratio"):
            features[f"observation.reach.{robot}_{name}"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": None,
            }
        for name in ("frame_feasible", "episode_feasible"):
            features[f"observation.reach.{robot}_{name}"] = {
                "dtype": "int64",
                "shape": (1,),
                "names": None,
            }
    features["observation.reach.any_episode_feasible"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    }

    return features


def build_raw_state(pico_frame: dict, widths: GripperWidths) -> np.ndarray:
    """Build the 16D HandUMI raw state from tracking poses and Feetech widths."""

    left_pose = np.asarray(
        pico_frame.get("observation.pico.left_controller_pose", _identity_pose()),
        dtype=np.float32,
    ).reshape(-1)[:7]
    right_pose = np.asarray(
        pico_frame.get("observation.pico.right_controller_pose", _identity_pose()),
        dtype=np.float32,
    ).reshape(-1)[:7]
    return np.concatenate(
        [
            left_pose,
            right_pose,
            np.array([widths.left, widths.right], dtype=np.float32),
        ]
    ).astype(np.float32)


def _identity_pose() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


# ── Recording loop ─────────────────────────────────────────────────────────────


def record_episode(
    *,
    dataset,
    cameras: list,
    cam_names: list[str],
    grippers: FeetechGripperPair | None,
    xrt,
    episode_time_s: float,
    fps: int,
    task: str,
    stop_event,
    pico_mode: str,
    include_pico: bool,
    cam_width: int,
    cam_height: int,
    manual_control: bool,
    start_button: str,
    start_threshold: float,
    repeat_button: str,
    finish_button: str,
    laptop_cam_name: str | None,
    laptop_overlay: bool,
    laptop_preview: LaptopPreview | None,
    save_unreachable: bool,
) -> tuple[int, str, dict[str, bool]]:
    """
    Record one episode.  Returns the number of frames saved.
    stop_event is a threading.Event (or any object with .is_set()) that
    signals early termination.
    """
    control_interval = 1.0 / fps
    n_frames = 0
    start_t = time.perf_counter()
    anchor_l = None
    anchor_r = None
    status = "recorded"
    episode_feasible = {robot: False for robot in REACH_BUDGETS_M} if include_pico else {}
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

    while True:
        loop_start = time.perf_counter()

        elapsed = loop_start - start_t
        if (not manual_control and elapsed >= episode_time_s) or stop_event.is_set():
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

        # ── Camera frames ──────────────────────────────────────────────────
        cam_frames = read_camera_frames(
            cameras,
            cam_names,
            width=cam_width,
            height=cam_height,
        )

        # ── PICO data ──────────────────────────────────────────────────────
        if not include_pico:
            pico_frame = {}
            reach_frame = {}
            reach_metrics = {}
        else:
            pico_frame = (
                read_pico_frame(xrt, mode=pico_mode)
                if xrt is not None
                else empty_pico_frame()
            )
            if anchor_l is None or anchor_r is None:
                anchor_l = np.asarray(
                    pico_frame["observation.pico.left_controller_pose"], dtype=np.float32
                )[:3].copy()
                anchor_r = np.asarray(
                    pico_frame["observation.pico.right_controller_pose"], dtype=np.float32
                )[:3].copy()
            reach_frame, reach_metrics = compute_reach_features(pico_frame, anchor_l, anchor_r)

        # ── Feetech gripper widths ─────────────────────────────────────────
        widths = zero_gripper_widths() if grippers is None else grippers.read_normalized_widths()
        state_vec = build_raw_state(pico_frame, widths)
        feetech_frame = {
            "observation.feetech.left_ticks": np.array([widths.left_ticks], dtype=np.int64),
            "observation.feetech.right_ticks": np.array([widths.right_ticks], dtype=np.int64),
            "observation.feetech.left_width_mm": np.array([widths.left_mm], dtype=np.float32),
            "observation.feetech.right_width_mm": np.array([widths.right_mm], dtype=np.float32),
            "observation.feetech.left_normalized": np.array(
                [widths.left_normalized], dtype=np.float32
            ),
            "observation.feetech.right_normalized": np.array(
                [widths.right_normalized], dtype=np.float32
            ),
        }

        if laptop_cam_name:
            laptop_key = f"observation.images.{laptop_cam_name}"
            if laptop_key in cam_frames:
                if laptop_overlay:
                    tracker_count = int(
                        np.asarray(
                            pico_frame.get(
                                "observation.pico.motion_tracker_count",
                                np.zeros((1,), dtype=np.int64),
                            )
                        ).reshape(-1)[0]
                    )
                    cam_frames[laptop_key] = draw_laptop_overlay(
                        cam_frames[laptop_key],
                        elapsed_s=elapsed,
                        n_frames=n_frames + 1,
                        tracker_count=tracker_count,
                        reach_metrics=reach_metrics,
                        manual_control=manual_control,
                    )
                if laptop_preview is not None:
                    laptop_preview.show(cam_frames[laptop_key])

        # ── Assemble and save frame ────────────────────────────────────────
        data_frame = {
            **cam_frames,
            "observation.state": state_vec,
            "action": state_vec.copy(),
            **feetech_frame,
            **pico_frame,
            **reach_frame,
            "task": task,
        }
        dataset.add_frame(data_frame)
        n_frames += 1

        # ── Timing ────────────────────────────────────────────────────────
        dt = time.perf_counter() - loop_start
        sleep = control_interval - dt
        if sleep < 0:
            log.warning(
                f"Loop running slower than {fps} Hz "
                f"({1/dt:.1f} Hz actual). Consider reducing fps or camera resolution."
            )
        else:
            time.sleep(sleep)

    if include_pico and n_frames > 0 and status != "repeat":
        should_save, episode_feasible = update_episode_reach_flags(
            dataset,
            save_unreachable=save_unreachable,
        )
        if not should_save:
            status = "unreachable"
            dataset.clear_episode_buffer()

    return n_frames, status, episode_feasible


# ── Main ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record HandUMI raw data from cameras, optional tracking, and Feetech encoders."
    )

    # Cameras
    p.add_argument(
        "--cam-ids",
        nargs="+",
        type=_camera_arg,
        default=None,
        metavar="ID",
        help="OpenCV camera indices or paths for left_wrist right_wrist.",
    )
    p.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--cam-fps", type=int, default=30, help="Camera capture FPS.")

    # Feetech encoder source
    p.add_argument(
        "--feetech-config",
        type=Path,
        default=PORTS_PATH,
        help="Feetech ports file (servo_id/port); calibration is per-user cache.",
    )
    p.add_argument(
        "--feetech-port",
        type=str,
        default=None,
        help="Override shared Feetech serial port from config.",
    )
    p.add_argument(
        "--skip-feetech",
        action="store_true",
        help="Disable Feetech encoder reads and fill gripper widths with zeros.",
    )

    # Dataset
    p.add_argument(
        "--repo-id",
        type=str,
        default="local/handumi_dataset",
        help="Dataset repo-id in '{namespace}/{name}' format.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Local root directory for the dataset (default: $HF_LEROBOT_HOME/<repo-id>).",
    )
    p.add_argument(
        "--task",
        type=str,
        default="Teleoperation recording",
        help="Short task description stored in every frame.",
    )
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument(
        "--episode-time-s",
        type=float,
        default=60.0,
        help="Duration of each recording episode in seconds.",
    )
    p.add_argument("--fps", type=int, default=30, help="Dataset recording FPS.")
    p.add_argument(
        "--start-button",
        choices=START_BUTTON_CHOICES,
        default="enter",
        help=(
            "How to start each episode. Use 'enter' for keyboard, or a PICO "
            "controller input such as A, B, X, Y, left_trigger, or right_trigger."
        ),
    )
    p.add_argument(
        "--start-threshold",
        type=float,
        default=0.75,
        help="Analog threshold for trigger/grip start buttons.",
    )
    p.add_argument(
        "--manual-control",
        action="store_true",
        help=(
            "Use PICO buttons for open-ended episodes: start button stops/saves, "
            "repeat button discards, finish button stops all recording."
        ),
    )
    p.add_argument(
        "--repeat-button",
        choices=START_BUTTON_CHOICES,
        default="B",
        help="Manual-control button used to discard the current episode.",
    )
    p.add_argument(
        "--finish-button",
        choices=START_BUTTON_CHOICES,
        default="Y",
        help="Manual-control button used to finish all recording.",
    )
    p.add_argument(
        "--no-video",
        action="store_true",
        help="Store camera frames as PNG images instead of video.",
    )
    p.add_argument(
        "--vcodec",
        type=str,
        default="h264",
        help="Video codec for encoding (h264, hevc, libsvtav1, auto).",
    )
    p.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the completed dataset to the Hugging Face Hub.",
    )
    p.add_argument(
        "--skip-pico",
        action="store_true",
        help="Disable PICO / XRoboToolkit. This is the default unless --use-pico is set.",
    )
    p.add_argument(
        "--use-pico",
        action="store_true",
        help="Enable PICO / XRoboToolkit tracking streams.",
    )
    p.add_argument(
        "--only-pico",
        action="store_true",
        help=(
            "Record only PICO/XR data. Camera images stay in the schema but are "
            "zero-filled, except laptop camera when --laptop-camera is enabled. "
            "Feetech widths are zero-filled."
        ),
    )
    p.add_argument(
        "--laptop-camera",
        action="store_true",
        help="Append/record a laptop camera stream with stopwatch + reach overlay.",
    )
    p.add_argument(
        "--laptop-cam-id",
        type=int,
        default=0,
        help="OpenCV index for the laptop camera when --laptop-camera is enabled.",
    )
    p.add_argument(
        "--laptop-cam-name",
        type=str,
        default="laptop",
        help=(
            "Feature suffix for the laptop stream. If it matches an existing cam "
            "name, that camera receives the overlay instead of appending a new one."
        ),
    )
    p.add_argument(
        "--no-laptop-overlay",
        action="store_true",
        help="Record the laptop camera without drawing stopwatch/reach overlay.",
    )
    p.add_argument(
        "--no-laptop-preview",
        action="store_true",
        help="Do not open the live preview window for the saved laptop video stream.",
    )
    p.add_argument(
        "--save-unreachable",
        action="store_true",
        help=(
            "Save episodes even when both Piper and OpenArm exceed the reach budget. "
            "By default those episodes are discarded."
        ),
    )
    pico_mode = p.add_mutually_exclusive_group()
    pico_mode.add_argument(
        "--pico-mandos",
        action="store_true",
        help=(
            "Record PICO headset/controllers/hands only; body joints and motion "
            "trackers are zero-filled. Use this for controller-based arm inference."
        ),
    )
    pico_mode.add_argument(
        "--pico-object",
        action="store_true",
        help=(
            "Record PICO motion tracker/object-tracking poses. Body joints are "
            "zero-filled; headset/controller/hand poses are still recorded."
        ),
    )
    pico_mode.add_argument(
        "--pico-whole-body",
        action="store_true",
        help="Record the original PICO 24-joint body-tracking stream.",
    )
    pico_transport = p.add_mutually_exclusive_group()
    pico_transport.add_argument(
        "--pico-adb",
        action="store_true",
        help=(
            "Connect PICO through USB/ADB reverse tunnel. This is the default; "
            "set PC-service IP to 127.0.0.1 in the PICO app."
        ),
    )
    pico_transport.add_argument(
        "--pico-wifi",
        action="store_true",
        help=(
            "Connect PICO over WiFi/LAN. Skips ADB checks and adb reverse; set "
            "the PC-service IP in the PICO app to this computer's LAN IP."
        ),
    )
    p.add_argument(
        "--skip-adb-check",
        action="store_true",
        help="ADB mode only: skip the ADB device presence check.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.skip_pico = not args.use_pico or args.skip_pico
    if args.skip_pico and args.only_pico:
        raise SystemExit("--only-pico requires PICO; do not combine it with --skip-pico.")
    if args.manual_control and args.skip_pico:
        raise SystemExit("--manual-control requires PICO buttons; do not combine it with --skip-pico.")
    if args.manual_control and args.start_button == "enter":
        args.start_button = "A"
        log.info("--manual-control set: using PICO A as start/stop button.")
    if args.only_pico:
        args.skip_feetech = True

    if args.pico_object:
        pico_mode = "object"
    elif args.pico_mandos:
        pico_mode = "mandos"
    else:
        pico_mode = "whole-body"
    pico_transport = "wifi" if args.pico_wifi else "adb"

    if args.only_pico:
        log.info(
            f"--only-pico set: non-laptop cameras and Feetech widths will be zero-filled; "
            f"PICO mode={pico_mode!r}, transport={pico_transport!r}."
        )

    # ── 1. PICO / XRoboToolkit initialisation ─────────────────────────────────
    xrt = None
    if not args.skip_pico:
        log.info("─── PICO / XRoboToolkit setup ─────────────────────────")
        if pico_transport == "adb" and not args.skip_adb_check:
            log.info("Checking ADB connection …")
            if not verify_adb_connection(timeout_s=15.0):
                log.error(
                    "No ADB device found after 15 s.\n"
                    "  • Make sure the PICO headset is connected via USB cable.\n"
                    "  • Enable USB debugging on the headset.\n"
                    "  • Or pass --pico-wifi to connect through LAN/WiFi.\n"
                    "  • Or pass --skip-pico to record without the headset."
                )
                sys.exit(1)
            setup_adb_reverse()
        elif pico_transport == "wifi":
            lan_ip = guess_lan_ip()
            if lan_ip:
                log.info(
                    "PICO WiFi mode: skipping ADB. In the PICO XRoboToolkit app, "
                    f"set PC-service IP to {lan_ip} and port {PICO_SERVICE_PORT}."
                )
            else:
                log.info(
                    "PICO WiFi mode: skipping ADB. Set the PICO XRoboToolkit "
                    f"PC-service IP to this computer's LAN IP and port {PICO_SERVICE_PORT}."
                )
        else:
            log.info("ADB check skipped. Assuming XRoboToolkit can reach the PC service.")

        launch_xrt_service()
        xrt = init_xrt()

        if not wait_for_pico_data(xrt, mode=pico_mode, timeout_s=15.0):
            log.warning(
                f"PICO {pico_mode!r} data not available after 15 s. "
                "Missing streams will be filled with zeros."
            )
    else:
        log.info("--skip-pico set: PICO / XRoboToolkit disabled.")

    # ── 2. Camera initialisation ───────────────────────────────────────────────
    log.info("─── Camera setup ───────────────────────────────────────")
    cam_ids = resolve_camera_ids(args.cam_ids, args.camera_config)
    camera_specs, laptop_cam_name = build_camera_specs(
        cam_ids,
        laptop_camera=args.laptop_camera,
        laptop_cam_id=args.laptop_cam_id,
        laptop_cam_name=args.laptop_cam_name,
    )
    cam_names = [spec["name"] for spec in camera_specs]
    cameras = connect_cameras(
        camera_specs,
        fps=args.cam_fps,
        width=args.cam_width,
        height=args.cam_height,
        zero_non_laptop=args.only_pico,
    )

    # ── 3. Feetech encoder initialisation ─────────────────────────────────────
    log.info("─── Feetech encoder setup ──────────────────────────────")
    grippers = None
    if args.skip_feetech:
        log.info("Feetech disabled: left/right gripper widths will be zero-filled.")
    else:
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

    # ── 4. Dataset creation ────────────────────────────────────────────────────
    log.info("─── Dataset setup ──────────────────────────────────────")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    use_videos = not args.no_video
    features = build_features(
        cam_names=cam_names,
        cam_width=args.cam_width,
        cam_height=args.cam_height,
        use_videos=use_videos,
        include_pico=not args.skip_pico,
    )

    n_cams = len(cam_names)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=args.output_dir,
        robot_type="handumi_raw",
        features=features,
        use_videos=use_videos,
        image_writer_processes=0,
        image_writer_threads=max(1, 4 * n_cams),
        vcodec=args.vcodec,
    )
    log.info(f"Dataset created at: {dataset.root}")
    log.info(f"Features: {list(features.keys())}")

    laptop_preview = None
    if args.laptop_camera and not args.no_laptop_preview:
        laptop_preview = LaptopPreview(
            width=args.cam_width,
            height=args.cam_height,
            fps=args.fps,
            title="handumi saved laptop video",
        )

    # ── 5. Keyboard / signal stop event ───────────────────────────────────────
    import threading

    stop_event = threading.Event()

    def _on_signal(signum, frame):
        log.info("Signal received – stopping after current episode …")
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # ── 6. Recording loop ──────────────────────────────────────────────────────
    log.info("─── Recording ──────────────────────────────────────────")
    if args.manual_control:
        limit = "unlimited" if args.num_episodes <= 0 else str(args.num_episodes)
        log.info(
            f"Manual recording: {limit} episode(s) @ {args.fps} Hz. "
            f"{args.start_button}=stop/save, {args.repeat_button}=repeat, "
            f"{args.finish_button}=finish."
        )
    else:
        log.info(
            f"Will record {args.num_episodes} episode(s) × {args.episode_time_s}s @ {args.fps} Hz."
        )
        log.info("Press Ctrl+C to stop early (current episode will still be saved).")

    recorded = 0
    try:
        while (args.num_episodes <= 0 or recorded < args.num_episodes) and not stop_event.is_set():
            ep_num = dataset.num_episodes + 1
            ep_total = "∞" if args.num_episodes <= 0 else str(args.num_episodes)
            log.info(f"\n── Episode {ep_num}/{ep_total} ─────────────────────────────────")
            if args.manual_control:
                if xrt is None:
                    raise RuntimeError("--manual-control requires PICO/XRoboToolkit.")
                action = wait_for_manual_start(
                    xrt,
                    start_button=args.start_button,
                    finish_button=args.finish_button,
                    threshold=args.start_threshold,
                    stop_event=stop_event,
                )
                if action == "finish":
                    break
            elif args.start_button == "enter":
                input(f"  Press ENTER to start recording episode {ep_num} …")
            else:
                if xrt is None:
                    raise RuntimeError(
                        "--start-button requires PICO/XRoboToolkit; use 'enter' with --skip-pico."
                    )
                if not wait_for_start_button(
                    xrt,
                    button=args.start_button,
                    threshold=args.start_threshold,
                    stop_event=stop_event,
                ):
                    break

            if args.manual_control:
                log.info("  Recording …")
            else:
                log.info(f"  Recording for {args.episode_time_s}s …  (Ctrl+C to end early)")
            stop_event.clear()  # allow Ctrl+C to end only this episode
            n_frames, status, episode_feasible = record_episode(
                dataset=dataset,
                cameras=cameras,
                cam_names=cam_names,
                grippers=grippers,
                xrt=xrt,
                episode_time_s=args.episode_time_s,
                fps=args.fps,
                task=args.task,
                stop_event=stop_event,
                pico_mode=pico_mode,
                include_pico=not args.skip_pico,
                cam_width=args.cam_width,
                cam_height=args.cam_height,
                manual_control=args.manual_control,
                start_button=args.start_button,
                start_threshold=args.start_threshold,
                repeat_button=args.repeat_button,
                finish_button=args.finish_button,
                laptop_cam_name=laptop_cam_name,
                laptop_overlay=args.laptop_camera and not args.no_laptop_overlay,
                laptop_preview=laptop_preview,
                save_unreachable=args.save_unreachable,
            )

            if n_frames == 0 or status in {"repeat", "unreachable"}:
                if status == "repeat":
                    log.warning("  Episode discarded by repeat button.")
                elif status == "unreachable":
                    log.warning("  Episode discarded: out of reach for both Piper and OpenArm.")
                else:
                    log.warning("  No frames recorded – discarding episode.")
                dataset.clear_episode_buffer()
                if status == "finish":
                    break
                continue

            log.info(f"  Saving {n_frames} frames …")
            dataset.save_episode()
            recorded += 1
            if episode_feasible:
                log.info(
                    f"  Episode {ep_num} saved. ({recorded}/{ep_total} done) "
                    f"Piper={int(episode_feasible['piper'])} "
                    f"OpenArm={int(episode_feasible['openarm'])}"
                )
            else:
                log.info(f"  Episode {ep_num} saved. ({recorded}/{ep_total} done)")

            if status == "finish":
                break

            # After Ctrl+C ends the episode, ask whether to continue
            if stop_event.is_set() and (args.num_episodes <= 0 or recorded < args.num_episodes):
                ans = input("\nContinue recording more episodes? [y/N] ").strip().lower()
                if ans == "y":
                    stop_event.clear()
                else:
                    break

    finally:
        log.info("─── Finalising ─────────────────────────────────────────")
        if laptop_preview is not None:
            laptop_preview.close()

        dataset.finalize()

        disconnect_cameras(cameras)

        if grippers is not None:
            grippers.close()

        if xrt is not None:
            try:
                xrt.close()
            except Exception:
                pass

        log.info(f"Done. Recorded {recorded} episode(s).")
        log.info(f"Dataset stored at: {dataset.root}")

        if args.push_to_hub:
            log.info("Pushing dataset to Hugging Face Hub …")
            dataset.push_to_hub()


def _camera_arg(value: str) -> int | str:
    return int(value) if value.isdigit() else value


if __name__ == "__main__":
    main()
