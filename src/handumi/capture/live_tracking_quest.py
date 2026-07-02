"""Live HandUMI motion tracking (Phase 2A live view + optional 2B robot).

Ties the Phase 2A pieces together:

  1. receive Quest controller frames           (handumi.tracking.meta_quest)
  2. calibrate poses into handumi_workspace     (handumi.tracking.transforms)
  3. read Feetech gripper width                 (handumi.feetech)
  -> build the 16D HandUMI raw state
  -> log to Rerun: wrist cameras + Feetech width series + a live 3D trajectory
     of each controller (rolling trails), the UMI-style view from yubi-sw.

Pass ``--robot piper`` to also solve bimanual IK each frame and render the
robot following your hands in Viser (Phase 2B, handumi.capture.robot_follow).

The left X button captures a workspace reset (re-centers on the current HMD
pose); the workspace also auto-initializes on the first tracked frame. Python
owns the (minimal) state machine — the device only forwards button states.

Run with the mock Quest for a dry run:

    python -m handumi.tracking.mock_quest_sender
    python -m handumi.capture.live_tracking_quest --skip-cameras --skip-feetech
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import numpy.typing as npt

from handumi.dataset.raw import HANDUMI_RAW_STATE_SIZE, pose_to_state_vector
from handumi.feetech import PORTS_PATH
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
)

log = logging.getLogger("handumi.live_tracking_quest")

# Pastel palette (warm cream/mustard/taupe), kept saturated enough on the
# cream 3D background to stay readable at a glance.
BACKGROUND_COLOR = (238, 233, 220)  # warm cream — the 3D view background
LEFT_COLOR = (240, 189, 63)  # mustard gold — matches Feetech left series
RIGHT_COLOR = (150, 130, 110)  # warm taupe — matches Feetech right series
ORIGIN_COLOR = (70, 58, 46)  # dark warm brown — the workspace origin (HMD pose at last reset)


# ---------------------------------------------------------------------------
# Pure helpers (no I/O — unit-tested).
# Quest pose->workspace math lives in handumi.tracking.meta_quest; the 16D raw
# state assembly lives in handumi.dataset.raw. Only the Rerun trail is local.
# ---------------------------------------------------------------------------


class TrajectoryTrail:
    """Rolling buffer of recent 3D positions for one controller."""

    def __init__(self, max_points: int) -> None:
        self._points: deque[np.ndarray] = deque(maxlen=max(1, max_points))

    def append(self, position: npt.ArrayLike) -> None:
        self._points.append(np.asarray(position, dtype=np.float32).reshape(3))

    def clear(self) -> None:
        self._points.clear()

    def points(self) -> np.ndarray:
        if not self._points:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(self._points, dtype=np.float32)


# ---------------------------------------------------------------------------
# Rerun setup + logging.
# ---------------------------------------------------------------------------


def _init_rerun(*, spawn: bool, ip: str | None, port: int | None) -> bool:
    try:
        import rerun as rr
    except ImportError:
        log.warning("rerun is not installed; running without visualization.")
        return False
    rr.init("handumi_live_tracking_quest")
    if ip and port:
        rr.connect_grpc(url=f"rerun+http://{ip}:{port}/proxy")
    elif spawn:
        rr.spawn()
    _send_blueprint()
    _send_styles()
    return True


# Diagnostic-signal colors, kept distinct from the LEFT/RIGHT tracking palette
# above so a glance tells "which hand" (mustard/taupe) apart from "is this
# healthy" (terracotta=fps, dark brown=clock skew) — same pastel family.
FPS_COLOR = (196, 108, 78)  # muted terracotta
OFFSET_COLOR = (90, 80, 70)  # dark warm brown

# How much recent history the 2D charts show by default. Rerun's default is
# "the whole recording", which keeps compressing as a session runs longer —
# a rolling window keeps oscillations (gripper opening/closing, fps) readable
# no matter how long you've been recording.
_CHART_WINDOW_S = 20.0


def _send_styles() -> None:
    import rerun as rr

    styles = {
        "observation.feetech.left_width_mm": ("left_width_mm", [*LEFT_COLOR, 255]),
        "observation.feetech.right_width_mm": ("right_width_mm", [*RIGHT_COLOR, 255]),
        "tracking.fps": ("fps", [*FPS_COLOR, 255]),
        "tracking.offset_s": ("clock_offset_s", [*OFFSET_COLOR, 255]),
        "tracking.left_tracked": ("left_tracked", [*LEFT_COLOR, 255]),
        "tracking.right_tracked": ("right_tracked", [*RIGHT_COLOR, 255]),
    }
    for path, (name, color) in styles.items():
        rr.log(path, rr.SeriesLines(colors=[color], widths=[2.5], names=[name]), static=True)
    # handumi_workspace is right-handed, X forward / Y left / Z up.
    rr.log("tracking", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)


def _recent_window():
    import rerun.blueprint as rrb
    import rerun.datatypes as rdt

    return rrb.VisibleTimeRanges(
        rrb.VisibleTimeRange(
            timeline="log_time",
            range=rdt.TimeRange(
                start=rdt.TimeRangeBoundary.cursor_relative(seconds=-_CHART_WINDOW_S),
                end=rdt.TimeRangeBoundary.cursor_relative(seconds=0.0),
            ),
        )
    )


def _send_blueprint() -> None:
    import rerun.blueprint as rrb

    recent = _recent_window()

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/tracking",
                name="controller_trajectory",
                background=rrb.Background(color=[*BACKGROUND_COLOR, 255]),
            ),
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="/observation.images.left_wrist", name="left_wrist"),
                    rrb.Spatial2DView(origin="/observation.images.right_wrist", name="right_wrist"),
                ),
                rrb.TimeSeriesView(
                    origin="/",
                    contents=[
                        "/observation.feetech.left_width_mm",
                        "/observation.feetech.right_width_mm",
                    ],
                    name="gripper_width_mm",
                    axis_y=rrb.ScalarAxis(range=(0.0, 90.0)),
                    time_ranges=recent,
                    plot_legend=rrb.Corner2D.LeftTop,
                ),
                row_shares=[3, 2],
            ),
            column_shares=[2, 3],
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed"),
    )
    import rerun as rr

    rr.send_blueprint(blueprint, make_active=True, make_default=True)


def _log_workspace_origin() -> None:
    """Mark the workspace origin (the HMD pose captured at the last reset).

    Both controllers are logged relative to this same point, but it is never
    itself drawn — only each controller's *current* pose gets an axis gizmo, so
    the shared reference is invisible otherwise. By definition the origin is
    always at [0,0,0] in the workspace frame, so this needs no pose input.
    """
    import rerun as rr

    rr.log(
        "tracking/origin",
        rr.Transform3D(translation=[0.0, 0.0, 0.0], axis_length=0.15),
    )
    rr.log(
        "tracking/origin/marker",
        rr.Points3D([[0.0, 0.0, 0.0]], colors=[ORIGIN_COLOR], radii=0.018),
    )


def _log_pose(path: str, pose: Pose, color: tuple[int, int, int], trail: TrajectoryTrail) -> None:
    import rerun as rr

    rr.log(
        path,
        rr.Transform3D(
            translation=pose.position,
            quaternion=rr.Quaternion(xyzw=pose.quaternion),
            axis_length=0.1,
        ),
    )
    rr.log(f"{path}/tip", rr.Points3D([pose.position], colors=[color], radii=0.012))
    points = trail.points()
    if len(points) >= 2:
        rr.log(f"{path}/trail", rr.LineStrips3D([points], colors=[color], radii=0.003))


def _log_cameras(cam_frames: dict, compress: bool) -> None:
    from lerobot.utils.visualization_utils import log_rerun_data

    if cam_frames:
        log_rerun_data(observation=cam_frames, compress_images=compress)


def _log_scalars(observation: dict) -> None:
    import rerun as rr

    for key, value in observation.items():
        rr.log(key, rr.Scalars(float(value)))


# ---------------------------------------------------------------------------
# Live loop.
# ---------------------------------------------------------------------------


def run_live_tracking(
    *,
    receiver: MetaQuestReceiver,
    mounts: MountingOffsets,
    cameras: list | None,
    cam_names: list[str],
    grippers,
    fps: int,
    trail_seconds: float,
    cam_width: int,
    cam_height: int,
    compress_images: bool,
    rerun_enabled: bool,
    duration_s: float | None,
    robot_follower=None,
    stop_check=lambda: False,
) -> None:
    """Run the live tracking loop. Returns when stopped / duration elapsed."""
    max_points = max(2, int(trail_seconds * fps))
    left_trail = TrajectoryTrail(max_points)
    right_trail = TrajectoryTrail(max_points)
    workspace = WorkspaceCalibration.identity()
    workspace_set = False
    prev_reset_pressed = False
    last_state = np.zeros(HANDUMI_RAW_STATE_SIZE, dtype=np.float32)

    control_interval = 1.0 / fps
    start = time.perf_counter()
    frame_index = 0

    while not stop_check():
        loop_start = time.perf_counter()
        elapsed = loop_start - start
        if duration_s is not None and elapsed >= duration_s:
            break

        frame: QuestFrame | None = receiver.latest()
        metrics = receiver.metrics()

        cam_frames = {}
        if cameras:
            from handumi.cameras.usb import read_camera_frames

            cam_frames = read_camera_frames(cameras, cam_names, width=cam_width, height=cam_height)

        widths = _read_widths(grippers)

        left_tracked = right_tracked = False
        if frame is not None:
            # Reset edge: left X re-centers the workspace on the current HMD pose.
            reset_pressed = frame.left.buttons.primary
            reset_edge = reset_pressed and not prev_reset_pressed
            prev_reset_pressed = reset_pressed

            if frame.hmd.tracked and (reset_edge or not workspace_set):
                workspace = workspace_from_hmd(frame.hmd)
                workspace_set = True
                left_trail.clear()
                right_trail.clear()
                if rerun_enabled:
                    _log_workspace_origin()
                log.info("Workspace %s on HMD pose.", "reset" if reset_edge else "initialized")

            left_tracked = frame.left.tracked and frame.left.valid
            right_tracked = frame.right.tracked and frame.right.valid
            left_pose = controller_pose_in_workspace(
                frame.left, mounting_offset=mounts.left, workspace=workspace
            )
            right_pose = controller_pose_in_workspace(
                frame.right, mounting_offset=mounts.right, workspace=workspace
            )
            last_state = pose_to_state_vector(
                left_pose, right_pose, widths["left_m"], widths["right_m"]
            )
            if left_tracked:
                left_trail.append(left_pose.position)
            if right_tracked:
                right_trail.append(right_pose.position)

            if rerun_enabled:
                _log_pose("tracking/left", left_pose, LEFT_COLOR, left_trail)
                _log_pose("tracking/right", right_pose, RIGHT_COLOR, right_trail)

            if robot_follower is not None and workspace_set:
                robot_follower.step(
                    last_state, left_tracked=left_tracked, right_tracked=right_tracked
                )

        if rerun_enabled:
            _log_cameras(cam_frames, compress_images)
            _log_scalars(
                {
                    "observation.feetech.left_width_mm": widths["left_mm"],
                    "observation.feetech.right_width_mm": widths["right_mm"],
                    "observation.feetech.left_normalized": widths["left_norm"],
                    "observation.feetech.right_normalized": widths["right_norm"],
                    "tracking.fps": metrics["fps"],
                    "tracking.offset_s": metrics["offset_s"],
                    "tracking.left_tracked": float(left_tracked),
                    "tracking.right_tracked": float(right_tracked),
                }
            )

        _print_status(frame, metrics, widths, frame_index, workspace_set)
        frame_index += 1
        dt = time.perf_counter() - loop_start
        time.sleep(max(control_interval - dt, 0.0))

    print()


def _read_widths(grippers) -> dict:
    if grippers is None:
        return {"left_m": 0.0, "right_m": 0.0, "left_mm": 0.0, "right_mm": 0.0,
                "left_norm": 0.0, "right_norm": 0.0}
    w = grippers.read_normalized_widths()
    return {
        "left_m": w.left, "right_m": w.right,
        "left_mm": w.left_mm, "right_mm": w.right_mm,
        "left_norm": w.left_normalized, "right_norm": w.right_normalized,
    }


def _print_status(frame, metrics, widths, frame_index, workspace_set) -> None:
    if frame is None:
        sys.stdout.write(
            f"\rframe={frame_index:06d} connected={metrics['connected']} "
            f"streaming={metrics['streaming']} (waiting for Quest frames)      "
        )
    else:
        sys.stdout.write(
            "\r"
            f"frame={frame_index:06d} fps={metrics['fps']:5.1f} "
            f"ws={'set' if workspace_set else 'unset'} "
            f"L trk={int(frame.left.tracked)} R trk={int(frame.right.tracked)} "
            f"L={widths['left_mm']:6.1f}mm R={widths['right_mm']:6.1f}mm   "
        )
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live HandUMI Quest + Feetech tracking to Rerun.")
    p.add_argument("--tracking-config", type=Path, default=Path("configs/tracking_meta_quest.yaml"))
    p.add_argument("--quest-ip", type=str, default=None, help="Override quest_ip from config.")
    p.add_argument("--tcp-port", type=int, default=None)
    p.add_argument("--sync-port", type=int, default=None)
    p.add_argument("--feetech-config", type=Path, default=PORTS_PATH, help="Feetech ports file.")
    p.add_argument("--feetech-port", type=str, default=None)
    p.add_argument("--skip-feetech", action="store_true")
    p.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
    p.add_argument("--cam-ids", nargs="+", type=_camera_arg, default=None)
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--cam-fps", type=int, default=30)
    p.add_argument("--skip-cameras", action="store_true")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--trail-seconds", type=float, default=10.0)
    p.add_argument("--duration-s", type=float, default=None)
    p.add_argument(
        "--robot",
        choices=["piper"],
        default=None,
        help="Render this robot in Viser, IK-following the tracked poses (Phase 2B).",
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
    p.add_argument("--compress-images", action="store_true")
    p.add_argument("--display-ip", type=str, default=None)
    p.add_argument("--display-port", type=int, default=None)
    p.add_argument("--no-rerun-spawn", action="store_true", help="Init Rerun without a viewer.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    args = parse_args()

    config = _load_tracking_config(args)
    mounts = _load_mounts(args.tracking_config)

    rerun_enabled = _init_rerun(
        spawn=not args.no_rerun_spawn, ip=args.display_ip, port=args.display_port
    )

    cameras, cam_names = _connect_cameras(args)
    grippers = _connect_feetech(args)

    robot_follower = None
    if args.robot:
        from handumi.capture.robot_follow import RobotFollower

        robot_follower = RobotFollower(
            embodiment=args.robot,
            port=args.robot_port,
            z_lift=args.robot_z_lift,
            x_shift=args.robot_x_shift,
            open_browser=not args.no_robot_browser,
        )

    receiver = MetaQuestReceiver(config)
    receiver.start()
    log.info("Connecting to Quest at %s:%d. Ctrl+C to stop.", config.quest_ip, config.tcp_port)

    stop = {"flag": False}

    def _on_signal(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        run_live_tracking(
            receiver=receiver,
            mounts=mounts,
            cameras=cameras,
            cam_names=cam_names,
            grippers=grippers,
            fps=args.fps,
            trail_seconds=args.trail_seconds,
            cam_width=args.cam_width,
            cam_height=args.cam_height,
            compress_images=args.compress_images,
            rerun_enabled=rerun_enabled,
            duration_s=args.duration_s,
            robot_follower=robot_follower,
            stop_check=lambda: stop["flag"],
        )
    finally:
        receiver.stop()
        if robot_follower is not None:
            robot_follower.close()
        if grippers is not None:
            grippers.close()
        if cameras:
            from handumi.cameras.usb import disconnect_cameras

            disconnect_cameras(cameras)


def _load_tracking_config(args) -> MetaQuestConfig:
    if args.tracking_config.exists():
        config = MetaQuestConfig.from_yaml(args.tracking_config)
    else:
        config = MetaQuestConfig(quest_ip="")
    return MetaQuestConfig(
        quest_ip=args.quest_ip if args.quest_ip is not None else config.quest_ip,
        tcp_port=args.tcp_port if args.tcp_port is not None else config.tcp_port,
        sync_port=args.sync_port if args.sync_port is not None else config.sync_port,
        connect_retry_s=config.connect_retry_s,
    )


def _load_mounts(path: Path) -> MountingOffsets:
    if path.exists():
        return MountingOffsets.from_yaml(path)
    return MountingOffsets.identity()


def _connect_cameras(args):
    if args.skip_cameras:
        log.info("Cameras disabled.")
        return None, []
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


def _connect_feetech(args):
    if args.skip_feetech:
        log.info("Feetech disabled: gripper widths will be zero-filled.")
        return None
    from handumi.feetech import FeetechGripperPair, assert_calibrated, load_config, user_calibration_path
    from handumi.feetech.bus import FeetechUnavailableError

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
