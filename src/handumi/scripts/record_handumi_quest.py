#!/usr/bin/env python3
"""Record HandUMI raw data with Meta Quest tracking (Phase 2A recording path).

Companion to the PICO recorder (`handumi-record-pico`). This one sources left/right
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
  right A  start / stop an episode              (with --button-control)
  clap x2  start / stop an episode, hands-free   (with --clap-control)

By default (neither flag), episodes start on ENTER and stop after
--episode-time-s. Without --output-dir, each run creates a fresh
outputs/<YYYYMMDD_HHMMSS>/ folder named after when recording started
(outputs/ is gitignored — never committed).

Pass --robot piper to mirror each episode live in Viser (real MuJoCo physics,
same as live_tracking_quest.py --robot) while it's being recorded, so the
collector can watch the robot follow their hands during the real HandUMI
task. Spoken status announcements ("Recording episode 3", "Episode 3 saved,
812 frames", ...) are on by default — pass --no-sounds to disable them.

Usage
-----
::

    handumi-record-quest \
        --repo-id local/handumi_quest_test \
        --output-dir outputs/datasets/handumi_quest_test \
        --task "quest tracking test" \
        --num-episodes 1 \
        --episode-time-s 20
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from handumi.dataset.raw import pose_to_state_vector, raw_state_feature
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
from handumi.tracking.gestures import DoubleClapDetector
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
from handumi.utils.speech import log_say

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

    def update(self, frame: QuestFrame | None) -> bool:
        """Update the calibration; returns True if it was (re)initialized this call."""
        if frame is None or not frame.hmd.tracked:
            return False
        reset_pressed = frame.left.buttons.primary
        reset_edge = reset_pressed and not self._prev_reset
        self._prev_reset = reset_pressed
        if reset_edge or not self._set:
            self.calibration = workspace_from_hmd(frame.hmd)
            self._set = True
            log.info("Workspace %s on HMD pose.", "reset" if reset_edge else "initialized")
            return True
        return False


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
    clap_control: bool = False,
    clap_detector: DoubleClapDetector | None = None,
    robot_follower=None,
) -> tuple[int, str]:
    """Record one episode. Returns (n_frames, status).

    ``robot_follower`` (see ``handumi.capture.robot_follow.RobotFollower``) is
    optional and mirrors this episode's tracked poses into the Viser/MuJoCo
    sim live, exactly like ``live_tracking_quest.py --robot`` does — so the
    collector can watch the robot (and, if configured, the task scene) follow
    their hands while HandUMI data is being recorded.
    """
    from handumi.cameras.usb import read_camera_frames

    control_interval = 1.0 / fps
    n_frames = 0
    start_t = time.perf_counter()
    prev_stop_button = _right_a(receiver)
    timed = not button_control and not clap_control

    while True:
        loop_start = time.perf_counter()
        elapsed = loop_start - start_t
        if stop_event.is_set():
            return n_frames, "finish"
        if timed and elapsed >= episode_time_s:
            return n_frames, "recorded"

        frame = receiver.latest()
        if workspace.update(frame) and robot_follower is not None:
            robot_follower.reset()

        if button_control:
            pressed = _right_a(receiver)
            if pressed and not prev_stop_button:
                return n_frames, "recorded"
            prev_stop_button = pressed

        cam_frames = read_camera_frames(cameras, cam_names, width=cam_width, height=cam_height)
        widths = zero_gripper_widths() if grippers is None else grippers.read_normalized_widths()

        if clap_control and clap_detector is not None:
            if clap_detector.update(widths.left_mm, widths.right_mm, loop_start):
                return n_frames, "recorded"

        observation = build_observation(
            frame, mounts=mounts, workspace=workspace.calibration, widths=widths
        )

        if robot_follower is not None and workspace.is_set:
            left_tracked = bool(observation["observation.quest.left_tracked"][0])
            right_tracked = bool(observation["observation.quest.right_tracked"][0])
            robot_follower.step(
                observation["observation.state"],
                left_tracked=left_tracked,
                right_tracked=right_tracked,
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
    p.add_argument("--feetech-config", type=Path, default=PORTS_PATH)
    p.add_argument("--feetech-port", type=str, default=None)
    p.add_argument("--skip-feetech", action="store_true")
    p.add_argument("--repo-id", type=str, default="local/handumi_quest")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Dataset root. Defaults to outputs/<YYYYMMDD_HHMMSS>/ named after "
        "when recording started (outputs/ is gitignored).",
    )
    p.add_argument("--task", type=str, default="Quest teleoperation recording")
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--episode-time-s", type=float, default=60.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--vcodec", type=str, default="h264")
    control_group = p.add_mutually_exclusive_group()
    control_group.add_argument(
        "--button-control",
        action="store_true",
        help="Use right A to start/stop each episode (otherwise ENTER + timer).",
    )
    control_group.add_argument(
        "--clap-control",
        action="store_true",
        help="Double-clap either gripper shut to start/stop each episode, "
        "hands-free (otherwise ENTER + timer). Requires Feetech (not --skip-feetech).",
    )
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument(
        "--robot",
        choices=["piper"],
        default=None,
        help="Mirror this episode live in Viser, IK-following the tracked poses "
        "while recording (same as live_tracking_quest.py --robot).",
    )
    p.add_argument("--robot-port", type=int, default=None, help="Viser port (default 8003).")
    p.add_argument(
        "--robot-z-lift",
        type=float,
        default=0.55,
        help="Meters added to workspace Z: HMD-origin poses sit below the head; "
        "the robot base sits on the floor plate.",
    )
    p.add_argument("--robot-x-shift", type=float, default=0.0, help="Meters added to workspace X.")
    p.add_argument(
        "--no-robot-browser",
        action="store_true",
        help="Don't auto-open the Viser robot view in a browser tab (e.g. headless/SSH).",
    )
    p.add_argument(
        "--no-sounds",
        action="store_true",
        help="Disable spoken episode-status announcements (start/save/discard/stop).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.clap_control and args.skip_feetech:
        raise SystemExit(
            "--clap-control needs real Feetech gripper widths to detect claps; "
            "cannot combine with --skip-feetech."
        )
    output_dir = args.output_dir or _default_output_dir()
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

    robot_follower = None
    if args.robot:
        from handumi.capture.robot_follow import RobotFollower

        log.info("─── Robot sim setup (%s) ───", args.robot)
        robot_follower = RobotFollower(
            embodiment=args.robot,
            port=args.robot_port,
            z_lift=args.robot_z_lift,
            x_shift=args.robot_x_shift,
            open_browser=not args.no_robot_browser,
        )

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
        root=output_dir,
        robot_type="handumi_raw",
        features=features,
        use_videos=use_videos,
        image_writer_processes=0,
        image_writer_threads=max(1, 4 * len(cam_names)),
        vcodec=args.vcodec,
    )
    log.info("Dataset created at: %s", dataset.root)

    workspace = WorkspaceState()
    clap_detector = DoubleClapDetector()
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
            elif args.clap_control:
                log.info("Episode %d: double-clap either gripper to start ...", ep_num)
                if not _wait_for_clap(grippers, clap_detector, stop_event):
                    break
            else:
                input(f"  Press ENTER to start episode {ep_num} (Ctrl+C to stop) ...")
            stop_event.clear()

            log_say(f"Recording episode {ep_num}", play_sounds=not args.no_sounds)
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
                clap_control=args.clap_control,
                clap_detector=clap_detector,
                robot_follower=robot_follower,
            )
            print()
            if n_frames == 0:
                log_say("No frames recorded, discarding episode", play_sounds=not args.no_sounds)
                dataset.clear_episode_buffer()
            else:
                log_say(f"Episode {ep_num} saved, {n_frames} frames", play_sounds=not args.no_sounds)
                dataset.save_episode()
                recorded += 1
            if status == "finish":
                break
    finally:
        log_say("Stop recording", play_sounds=not args.no_sounds, blocking=True)
        log.info("─── Finalising ───")
        dataset.finalize()
        receiver.stop()
        if cameras:
            from handumi.cameras.usb import disconnect_cameras

            disconnect_cameras(cameras)
        if grippers is not None:
            grippers.close()
        if robot_follower is not None:
            robot_follower.close()
        log.info("Done. Recorded %d episode(s). Dataset at: %s", recorded, dataset.root)
        if args.push_to_hub:
            dataset.push_to_hub()
        log_say("Exiting", play_sounds=not args.no_sounds)


def _wait_for_right_a(receiver: MetaQuestReceiver, stop_event: threading.Event) -> bool:
    prev = _right_a(receiver)
    while not stop_event.is_set():
        pressed = _right_a(receiver)
        if pressed and not prev:
            return True
        prev = pressed
        time.sleep(0.02)
    return False


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


def _camera_arg(value: str) -> int | str:
    return int(value) if value.isdigit() else value


if __name__ == "__main__":
    main()
