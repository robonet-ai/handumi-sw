#!/usr/bin/env python3
"""Record HandUMI raw data with Meta Quest tracking (Phase 2A recording path).

Companion to the PICO recorder (`record_handumi_pico.py`). This one sources left/right
gripper poses from the native Quest app (TCP/JSON, see
`handumi.tracking.meta_quest`), calibrates them into `handumi_workspace`
(`handumi.tracking.transforms`), merges Feetech gripper width, and writes the
**same 16D HandUMI raw state** the rest of the pipeline expects.

Per-frame schema
────────────────
  observation.images.left_wrist / right_wrist    wrist cameras
  observation.state / action                      HandUMI raw state  float32[16]
  observation.feetech.*                           ticks / mm / normalized
  observation.quest.left_controller_pose          calibrated TCP pose [x,y,z,qx,qy,qz,qw]
  observation.quest.right_controller_pose
  observation.quest.headset_pose                  HMD pose in workspace
  observation.quest.left_tracked / right_tracked  OVR tracking flags  int64[1]
  observation.quest.device_time_ns                Quest clock         int64[1]
  observation.quest.pc_monotonic_ns               PC receive clock    int64[1]
  observation.quest.seq                           packet sequence     int64[1]

The device + PC clocks are recorded per frame so Quest poses can be aligned with
camera/Feetech frames in post-processing (see docs/phase-2-motion-tracking.md).

Controls (workstation has the only UI; the headset has none):
  left X   reset workspace on the current HMD pose (auto-inits on first frame)
  right A  start / stop an episode   (with --button-control)
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

from handumi.dataset.raw import pose_to_state_vector, raw_state_feature
from handumi.feetech import (
    FeetechGripperPair,
    GripperWidths,
    assert_calibrated,
    load_config,
    resolve_config_path,
    zero_gripper_widths,
)
from handumi.feetech.bus import FeetechUnavailableError
from handumi.tracking.meta_quest import (
    MetaQuestConfig,
    MetaQuestReceiver,
    QuestFrame,
    controller_pose_in_workspace,
    workspace_from_hmd,
)
from handumi.tracking.transforms import (
    MountingOffsets,
    Pose,
    WorkspaceCalibration,
    unity_pose_to_handumi,
)

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("handumi.record_quest")

_POSE_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw"]


# ── Schema ──────────────────────────────────────────────────────────────────


def build_features(
    cam_names: list[str], cam_width: int, cam_height: int, use_videos: bool
) -> dict:
    """Feature schema for the Quest recorder (cameras + 16D state + Feetech + Quest)."""
    img_dtype = "video" if use_videos else "image"
    features: dict = {}
    for cam in cam_names:
        features[f"observation.images.{cam}"] = {
            "dtype": img_dtype,
            "shape": (cam_height, cam_width, 3),
            "names": ["height", "width", "channel"],
        }

    # LeRobot's frame validation compares value.shape (a tuple) against
    # feature["shape"]; the shared schema uses a list, so coerce to a tuple.
    features["observation.state"] = _tuple_shape(raw_state_feature())
    features["action"] = _tuple_shape(raw_state_feature())

    for side in ("left", "right"):
        features[f"observation.feetech.{side}_ticks"] = _f("int64")
        features[f"observation.feetech.{side}_width_mm"] = _f("float32")
        features[f"observation.feetech.{side}_normalized"] = _f("float32")

    for key in ("left_controller_pose", "right_controller_pose", "headset_pose"):
        features[f"observation.quest.{key}"] = {
            "dtype": "float32",
            "shape": (7,),
            "names": list(_POSE_NAMES),
        }
    for key in ("left_tracked", "right_tracked", "device_time_ns", "pc_monotonic_ns", "seq"):
        features[f"observation.quest.{key}"] = _f("int64")

    return features


def _f(dtype: str) -> dict:
    return {"dtype": dtype, "shape": (1,), "names": None}


def _tuple_shape(feature: dict) -> dict:
    feature["shape"] = tuple(feature["shape"])
    return feature


# ── Per-frame assembly (pure, testable) ─────────────────────────────────────


def _pose7(pose: Pose) -> np.ndarray:
    return np.concatenate([pose.position, pose.quaternion]).astype(np.float32)


def build_observation(
    frame: QuestFrame | None,
    *,
    mounts: MountingOffsets,
    workspace: WorkspaceCalibration,
    widths: GripperWidths,
) -> dict:
    """Build the non-image part of one recorded frame from a Quest sample.

    A missing/untracked frame records identity poses with the tracking flags set
    to 0, so the fixed schema is always satisfied.
    """
    if frame is not None:
        left_pose = controller_pose_in_workspace(
            frame.left, mounting_offset=mounts.left, workspace=workspace
        )
        right_pose = controller_pose_in_workspace(
            frame.right, mounting_offset=mounts.right, workspace=workspace
        )
        hmd_pose = workspace.apply(
            unity_pose_to_handumi(frame.hmd.position, frame.hmd.quaternion)
        )
        left_tracked = int(frame.left.tracked and frame.left.valid)
        right_tracked = int(frame.right.tracked and frame.right.valid)
        device_time_ns = int(frame.device_time_ns)
        pc_monotonic_ns = int(frame.pc_monotonic_ns)
        seq = int(frame.seq)
    else:
        left_pose = right_pose = hmd_pose = Pose.identity()
        left_tracked = right_tracked = 0
        device_time_ns = pc_monotonic_ns = seq = 0

    state = pose_to_state_vector(left_pose, right_pose, widths.left, widths.right)
    return {
        "observation.state": state,
        "action": state.copy(),
        "observation.feetech.left_ticks": np.array([widths.left_ticks], dtype=np.int64),
        "observation.feetech.right_ticks": np.array([widths.right_ticks], dtype=np.int64),
        "observation.feetech.left_width_mm": np.array([widths.left_mm], dtype=np.float32),
        "observation.feetech.right_width_mm": np.array([widths.right_mm], dtype=np.float32),
        "observation.feetech.left_normalized": np.array([widths.left_normalized], dtype=np.float32),
        "observation.feetech.right_normalized": np.array([widths.right_normalized], dtype=np.float32),
        "observation.quest.left_controller_pose": _pose7(left_pose),
        "observation.quest.right_controller_pose": _pose7(right_pose),
        "observation.quest.headset_pose": _pose7(hmd_pose),
        "observation.quest.left_tracked": np.array([left_tracked], dtype=np.int64),
        "observation.quest.right_tracked": np.array([right_tracked], dtype=np.int64),
        "observation.quest.device_time_ns": np.array([device_time_ns], dtype=np.int64),
        "observation.quest.pc_monotonic_ns": np.array([pc_monotonic_ns], dtype=np.int64),
        "observation.quest.seq": np.array([seq], dtype=np.int64),
    }


# ── Workspace state machine ─────────────────────────────────────────────────


class WorkspaceState:
    """Owns the workspace calibration + left-X reset edge detection."""

    def __init__(self) -> None:
        self.calibration = WorkspaceCalibration.identity()
        self._set = False
        self._prev_reset = False

    @property
    def is_set(self) -> bool:
        return self._set

    def update(self, frame: QuestFrame | None) -> None:
        if frame is None or not frame.hmd.tracked:
            return
        reset_pressed = frame.left.buttons.primary
        reset_edge = reset_pressed and not self._prev_reset
        self._prev_reset = reset_pressed
        if reset_edge or not self._set:
            self.calibration = workspace_from_hmd(frame.hmd)
            self._set = True
            log.info("Workspace %s on HMD pose.", "reset" if reset_edge else "initialized")


# ── Recording loop ──────────────────────────────────────────────────────────


def record_episode(
    *,
    dataset,
    cameras: list,
    cam_names: list[str],
    receiver: MetaQuestReceiver,
    mounts: MountingOffsets,
    workspace: WorkspaceState,
    grippers: FeetechGripperPair | None,
    episode_time_s: float,
    fps: int,
    task: str,
    cam_width: int,
    cam_height: int,
    button_control: bool,
    stop_event: threading.Event,
) -> tuple[int, str]:
    """Record one episode. Returns (n_frames, status)."""
    from handumi.cameras.usb import read_camera_frames

    control_interval = 1.0 / fps
    n_frames = 0
    start_t = time.perf_counter()
    prev_stop_button = _right_a(receiver)

    while True:
        loop_start = time.perf_counter()
        elapsed = loop_start - start_t
        if stop_event.is_set():
            return n_frames, "finish"
        if not button_control and elapsed >= episode_time_s:
            return n_frames, "recorded"

        frame = receiver.latest()
        workspace.update(frame)

        if button_control:
            pressed = _right_a(receiver)
            if pressed and not prev_stop_button:
                return n_frames, "recorded"
            prev_stop_button = pressed

        cam_frames = read_camera_frames(cameras, cam_names, width=cam_width, height=cam_height)
        widths = zero_gripper_widths() if grippers is None else grippers.read_normalized_widths()
        observation = build_observation(
            frame, mounts=mounts, workspace=workspace.calibration, widths=widths
        )
        dataset.add_frame({**cam_frames, **observation, "task": task})
        n_frames += 1

        _print_status(frame, widths, n_frames, workspace.is_set)
        dt = time.perf_counter() - loop_start
        time.sleep(max(control_interval - dt, 0.0))


def _right_a(receiver: MetaQuestReceiver) -> bool:
    frame = receiver.latest()
    return bool(frame.right.buttons.primary) if frame is not None else False


def _print_status(frame, widths, n_frames, ws_set) -> None:
    trk = (
        f"L{int(frame.left.tracked)} R{int(frame.right.tracked)}" if frame is not None else "L0 R0"
    )
    sys.stdout.write(
        f"\r  rec frame={n_frames:06d} ws={'set' if ws_set else 'unset'} trk={trk} "
        f"L={widths.left_mm:6.1f}mm R={widths.right_mm:6.1f}mm   "
    )
    sys.stdout.flush()


# ── Main ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record HandUMI data with Meta Quest tracking.")
    p.add_argument("--tracking-config", type=Path, default=Path("configs/tracking_meta_quest.yaml"))
    p.add_argument("--quest-ip", type=str, default=None)
    p.add_argument("--tcp-port", type=int, default=None)
    p.add_argument("--sync-port", type=int, default=None)
    p.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
    p.add_argument("--cam-ids", nargs="+", type=_camera_arg, default=None)
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--cam-fps", type=int, default=30)
    p.add_argument("--feetech-config", type=Path, default=None)
    p.add_argument("--feetech-port", type=str, default=None)
    p.add_argument("--skip-feetech", action="store_true")
    p.add_argument("--repo-id", type=str, default="local/handumi_quest")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--task", type=str, default="Quest teleoperation recording")
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--episode-time-s", type=float, default=60.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--vcodec", type=str, default="h264")
    p.add_argument(
        "--button-control",
        action="store_true",
        help="Use right A to start/stop each episode (otherwise ENTER + timer).",
    )
    p.add_argument("--push-to-hub", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = _resolve_config(args)
    mounts = (
        MountingOffsets.from_yaml(args.tracking_config)
        if args.tracking_config.exists()
        else MountingOffsets.identity()
    )

    log.info("─── Camera setup ───")
    cameras, cam_names = _connect_cameras(args)

    log.info("─── Feetech setup ───")
    grippers = _connect_feetech(args)

    log.info("─── Quest receiver ───")
    receiver = MetaQuestReceiver(config)
    receiver.start()
    log.info("Connecting to Quest at %s:%d ...", config.quest_ip, config.tcp_port)

    log.info("─── Dataset setup ───")
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

    workspace = WorkspaceState()
    stop_event = threading.Event()

    def _on_signal(signum, frame):
        log.info("Signal received - stopping after current episode ...")
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    recorded = 0
    try:
        while (args.num_episodes <= 0 or recorded < args.num_episodes) and not stop_event.is_set():
            ep_num = dataset.num_episodes + 1
            if args.button_control:
                log.info("Episode %d: press right A to start ...", ep_num)
                if not _wait_for_right_a(receiver, stop_event):
                    break
            else:
                input(f"  Press ENTER to start episode {ep_num} (Ctrl+C to stop) ...")
            stop_event.clear()

            log.info("  Recording ...")
            n_frames, status = record_episode(
                dataset=dataset,
                cameras=cameras,
                cam_names=cam_names,
                receiver=receiver,
                mounts=mounts,
                workspace=workspace,
                grippers=grippers,
                episode_time_s=args.episode_time_s,
                fps=args.fps,
                task=args.task,
                cam_width=args.cam_width,
                cam_height=args.cam_height,
                button_control=args.button_control,
                stop_event=stop_event,
            )
            print()
            if n_frames == 0:
                log.warning("  No frames recorded - discarding episode.")
                dataset.clear_episode_buffer()
            else:
                log.info("  Saving %d frames ...", n_frames)
                dataset.save_episode()
                recorded += 1
            if status == "finish":
                break
    finally:
        log.info("─── Finalising ───")
        dataset.finalize()
        receiver.stop()
        if cameras:
            from handumi.cameras.usb import disconnect_cameras

            disconnect_cameras(cameras)
        if grippers is not None:
            grippers.close()
        log.info("Done. Recorded %d episode(s). Dataset at: %s", recorded, dataset.root)
        if args.push_to_hub:
            dataset.push_to_hub()


def _wait_for_right_a(receiver: MetaQuestReceiver, stop_event: threading.Event) -> bool:
    prev = _right_a(receiver)
    while not stop_event.is_set():
        pressed = _right_a(receiver)
        if pressed and not prev:
            return True
        prev = pressed
        time.sleep(0.02)
    return False


def _resolve_config(args) -> MetaQuestConfig:
    base = (
        MetaQuestConfig.from_yaml(args.tracking_config)
        if args.tracking_config.exists()
        else MetaQuestConfig(quest_ip="")
    )
    return MetaQuestConfig(
        quest_ip=args.quest_ip if args.quest_ip is not None else base.quest_ip,
        tcp_port=args.tcp_port if args.tcp_port is not None else base.tcp_port,
        sync_port=args.sync_port if args.sync_port is not None else base.sync_port,
        connect_retry_s=base.connect_retry_s,
    )


def _connect_cameras(args):
    from handumi.cameras.usb import build_camera_specs, connect_cameras, resolve_camera_ids

    cam_ids = resolve_camera_ids(args.cam_ids, args.camera_config)
    camera_specs, _ = build_camera_specs(
        cam_ids, laptop_camera=False, laptop_cam_id=0, laptop_cam_name="laptop"
    )
    cam_names = [spec["name"] for spec in camera_specs]
    cameras = connect_cameras(
        camera_specs, fps=args.cam_fps, width=args.cam_width, height=args.cam_height,
        zero_non_laptop=False,
    )
    return cameras, cam_names


def _connect_feetech(args) -> FeetechGripperPair | None:
    if args.skip_feetech:
        log.info("Feetech disabled: gripper widths will be zero-filled.")
        return None
    feetech_path = resolve_config_path(args.feetech_config)
    feetech_config = load_config(feetech_path)
    if args.feetech_port is not None:
        feetech_config = type(feetech_config)(
            port=args.feetech_port,
            baudrate=feetech_config.baudrate,
            protocol_version=feetech_config.protocol_version,
            left=feetech_config.left,
            right=feetech_config.right,
        )
    assert_calibrated(feetech_config, source=feetech_path)
    grippers = FeetechGripperPair(feetech_config)
    try:
        grippers.open()
    except FeetechUnavailableError as exc:
        raise SystemExit(str(exc)) from exc
    return grippers


def _camera_arg(value: str) -> int | str:
    return int(value) if value.isdigit() else value


if __name__ == "__main__":
    main()
