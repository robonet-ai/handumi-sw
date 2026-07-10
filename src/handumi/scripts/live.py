#!/usr/bin/env python3
"""Live preview: move the HandUMI and watch the robot follow in Viser (+ Rerun).

Nothing is recorded. The same pipeline the post-hoc replay uses runs live:

    TrackingProvider.latest()                (PICO or Meta Quest)
      -> controller->TCP calibration          configs/calibration/<device>_controller_tcp.yaml
      -> anchored retargeting                (same as handumi-replay-in-sim)
      -> bimanual IK                          robots/kinematics.py
      -> Viser                                robot follows your hands

so what you see is what a recording would replay. Use it to sanity-check
tracking health and TCP calibration before a session.

Rerun (on by default, --no-rerun to disable) shows the calibrated TCP
trails in the workspace frame — tracking-side truth, before retargeting/IK.

Per-arm anchoring (each arm fully independent). Two gestures, SAME action —
(re-)anchor that arm: your hand pose at that instant maps to the arm's home
TCP and the arm follows relative motion from there. Only fires while that
side is tracked.

  X (left controller)   anchor the LEFT arm   (hands free, during setup)
  A (right controller)  anchor the RIGHT arm
  double clap LEFT      anchor the LEFT arm   (hands inside the HandUMIs)
  double clap RIGHT     anchor the RIGHT arm

Both arms start parked at home until their first anchor. Spoken feedback
("left anchored", ...) — --no-sounds to mute.

Usage
-----
::

    handumi-live --device meta
    handumi-live --device meta --quest-ip 127.0.0.1 --no-browser   # vs mock
    handumi-live --device pico --pico-mode mandos
"""

from __future__ import annotations

import argparse
import logging
import time
import webbrowser
from pathlib import Path

import numpy as np

from handumi.calibration.control_tcp import (
    calibration_path_for_device,
    load_controller_tcp_calibration,
)
from handumi.cameras import (
    build_camera_specs,
    connect_cameras,
    disconnect_cameras,
    read_camera_frames,
    resolve_camera_ids,
)
from handumi.dataset.raw import pose_to_state_vector
from handumi.feetech import PORTS_PATH, zero_gripper_widths
from handumi.retargeting.handumi_to_robot import (
    raw_state_robot_target_pose7,
    retarget_anchors_from_raw_state,
)
from handumi.robots.registry import EMBODIMENT_NAMES, load_embodiment
from handumi.robots.utils import IDENTITY_POSE7
from handumi.scripts.record import build_tracker, connect_feetech
from handumi.tracking.gestures import DoubleClapDetector
from handumi.tracking.pico import read_start_button_value
from handumi.tracking.transforms import Pose
from handumi.utils.speech import log_say
from handumi.utils.trajectory import TrajectoryTrail

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("handumi.live")

# Same side palette as replay_in_sim's target markers.
LEFT_COLOR = (255, 190, 50)
RIGHT_COLOR = (80, 220, 130)
BACKGROUND_COLOR = (40, 8, 8)  # dark red — the 3D view background
_TRAIL_SECONDS = 10.0
_CHART_WINDOW_S = 20.0  # rolling window for the gripper-width chart


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--device", choices=("pico", "meta"), required=True)
    p.add_argument("--robot", choices=EMBODIMENT_NAMES, default="piper")
    p.add_argument("--port", type=int, default=8003, help="Viser port.")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-browser", action="store_true", help="Don't auto-open Viser.")
    p.add_argument("--no-rerun", action="store_true", help="Disable the Rerun view.")
    p.add_argument("--no-sounds", action="store_true", help="Disable spoken anchor/home feedback.")
    p.add_argument(
        "--controller-tcp-calibration",
        type=Path,
        default=None,
        help="Override configs/calibration/<device>_controller_tcp.yaml.",
    )

    # Camera + Feetech flags, same names as handumi-record.
    p.add_argument("--cam-ids", nargs="+", type=_camera_arg, default=None)
    p.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--cam-fps", type=int, default=30)
    p.add_argument("--skip-cameras", action="store_true")
    p.add_argument("--feetech-config", type=Path, default=PORTS_PATH)
    p.add_argument("--feetech-port", type=str, default=None)
    p.add_argument("--skip-feetech", action="store_true")

    # Tracking flags, same names as handumi-record (shared build_tracker).
    p.add_argument("--tracking-config", type=Path, default=Path("configs/tracking_meta_quest.yaml"))
    p.add_argument("--quest-ip", type=str, default=None)
    p.add_argument("--tcp-port", type=int, default=None)
    p.add_argument("--sync-port", type=int, default=None)
    p.add_argument("--pico-mode", choices=("mandos", "object", "whole-body"), default="mandos")
    pico_transport = p.add_mutually_exclusive_group()
    pico_transport.add_argument("--pico-adb", action="store_true")
    pico_transport.add_argument("--pico-wifi", action="store_true")
    p.add_argument("--skip-adb-check", action="store_true")
    return p.parse_args()


def _camera_arg(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _load_calibration(args: argparse.Namespace):
    from handumi.calibration.control_tcp import ControllerTcpCalibration

    path = args.controller_tcp_calibration or calibration_path_for_device(args.device)
    if path.exists():
        calibration = load_controller_tcp_calibration(path)
        log.info("controller->TCP calibration: %s", path)
        return calibration
    log.warning(
        "No calibration at %s — previewing RAW controller poses. "
        "See docs/README_tcp_offset.md to calibrate.",
        path,
    )
    return ControllerTcpCalibration(
        left=IDENTITY_POSE7.astype(np.float32).copy(),
        right=IDENTITY_POSE7.astype(np.float32).copy(),
        source=None,
    )


def _side_joint_indices(runtime) -> dict[str, list[int]]:
    """Actuated-joint indices per side (``left_*`` / ``right_*`` prefixes)."""
    names = list(runtime.robot.joints.actuated_names)
    return {
        side: [i for i, name in enumerate(names) if name.startswith(f"{side}_")]
        for side in ("left", "right")
    }


def _anchor_buttons_pressed(tracker) -> dict[str, bool]:
    """Raw pressed-state of the per-arm anchor buttons: X (left), A (right).

    Meta: read from the receiver's latest frame. PICO: read via the XRT SDK.
    Missing data reads as not-pressed.
    """
    if tracker.device == "meta":
        frame = tracker.receiver.latest()
        if frame is None:
            return {"left": False, "right": False}
        return {
            "left": bool(frame.left.buttons.primary),
            "right": bool(frame.right.buttons.primary),
        }
    xrt = getattr(tracker, "xrt", None)
    if xrt is None:
        return {"left": False, "right": False}
    return {
        "left": read_start_button_value(xrt, "X") >= 0.75,
        "right": read_start_button_value(xrt, "A") >= 0.75,
    }


def _sample_state(sample, widths=None) -> np.ndarray:
    """16D raw state from a live sample's calibrated TCP poses + gripper widths."""
    left = Pose(sample.left_tcp_pose[:3], sample.left_tcp_pose[3:7])
    right = Pose(sample.right_tcp_pose[:3], sample.right_tcp_pose[3:7])
    left_w = 0.0 if widths is None else widths.left
    right_w = 0.0 if widths is None else widths.right
    return pose_to_state_vector(left, right, left_w, right_w)


def _init_rerun(enabled: bool, cam_names: list[str]):
    """Start Rerun with the classic live layout: 3D tracking on the left,
    wrist cameras top-right, gripper-width chart bottom-right."""
    if not enabled:
        return None
    import rerun as rr
    import rerun.blueprint as rrb
    import rerun.datatypes as rdt

    rr.init("handumi_live", spawn=True)
    rr.log("tracking", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for path, name, color in (
        ("observation.feetech.left_width_mm", "left_width_mm", LEFT_COLOR),
        ("observation.feetech.right_width_mm", "right_width_mm", RIGHT_COLOR),
    ):
        rr.log(path, rr.SeriesLines(colors=[[*color, 255]], widths=[2.5], names=[name]), static=True)
    # Faint corner markers spanning the working volume: the 3D view auto-fits
    # to data bounds, so these fix the initial framing/zoom (wide horizontally,
    # short vertically) instead of hugging the first few points.
    half_xy, half_z = 0.75, 0.4
    corners = [[sx * half_xy, sy * half_xy, sz * half_z]
               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    rr.log(
        "tracking/bounds",
        rr.Points3D(corners, colors=[[128, 100, 100, 90]] * len(corners), radii=0.004),
        static=True,
    )

    recent = rrb.VisibleTimeRanges(
        rrb.VisibleTimeRange(
            timeline="log_time",
            range=rdt.TimeRange(
                start=rdt.TimeRangeBoundary.cursor_relative(seconds=-_CHART_WINDOW_S),
                end=rdt.TimeRangeBoundary.cursor_relative(seconds=0.0),
            ),
        )
    )
    width_chart = rrb.TimeSeriesView(
        origin="/",
        contents=[
            "/observation.feetech.left_width_mm",
            "/observation.feetech.right_width_mm",
        ],
        name="gripper_width_mm",
        axis_y=rrb.ScalarAxis(range=(0.0, 90.0)),
        time_ranges=recent,
        plot_legend=rrb.Corner2D.LeftTop,
    )
    if cam_names:
        right_column = rrb.Vertical(
            rrb.Horizontal(
                *[
                    rrb.Spatial2DView(origin=f"/observation.images.{name}", name=name)
                    for name in cam_names
                ]
            ),
            width_chart,
            row_shares=[3, 2],
        )
    else:
        right_column = width_chart
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/tracking",
                name="controller_trajectory",
                background=rrb.Background(color=[*BACKGROUND_COLOR, 255]),
            ),
            right_column,
            column_shares=[2, 3],
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed"),
    )
    rr.send_blueprint(blueprint, make_active=True, make_default=True)
    return rr


def _log_rerun(
    rr,
    side: str,
    tcp_pose7: np.ndarray,
    raw_pose7: np.ndarray,
    trail: TrajectoryTrail,
    raw_trail: TrajectoryTrail,
    color,
) -> None:
    """Two trails per side: solid = calibrated TCP, faint = raw controller
    anchor (no mount offset) — they must differ only by a rigid offset."""
    trail.append(tcp_pose7[:3])
    rr.log(f"tracking/{side}/tcp", rr.Points3D([tcp_pose7[:3]], colors=[color], radii=0.012))
    points = trail.points()
    if len(points) >= 2:
        rr.log(f"tracking/{side}/trail", rr.LineStrips3D([points], colors=[color], radii=0.003))

    faint = [*color, 90]
    raw_trail.append(raw_pose7[:3])
    rr.log(f"tracking/{side}/raw", rr.Points3D([raw_pose7[:3]], colors=[faint], radii=0.007))
    raw_points = raw_trail.points()
    if len(raw_points) >= 2:
        rr.log(
            f"tracking/{side}/raw_trail",
            rr.LineStrips3D([raw_points], colors=[faint], radii=0.0015),
        )


def main() -> None:
    args = parse_args()

    calibration = _load_calibration(args)
    # X is the left-arm anchor button here, so the provider must not consume
    # it as a workspace reset (per-arm anchors absorb any workspace shift).
    tracker = build_tracker(args, calibration, reset_workspace_on_x=False)
    tracker.start()

    cameras: list = []
    cam_names: list[str] = []
    if not args.skip_cameras:
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

    grippers = connect_feetech(args)  # honors --skip-feetech internally

    log.info("Loading %s IK solver (JAX JIT warmup, ~30s on CPU) ...", args.robot)
    runtime = load_embodiment(args.robot)
    solver = runtime.solver_cls()
    q = runtime.config.home_q.astype(np.float32).copy()
    home_left_pose7, home_right_pose7 = solver.fk_pose7(q)
    max_reach = runtime.config.ik_weights.max_reach

    import viser
    import yourdfpy
    from viser.extras import ViserUrdf

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/grid", width=3.0, height=3.0, cell_size=0.1)
    urdf = yourdfpy.URDF.load(
        str(runtime.urdf_path), mesh_dir=str(runtime.urdf_path.parent), load_meshes=True
    )
    robot_view = ViserUrdf(server, urdf, root_node_name="/robot")
    robot_view.update_cfg(q)
    target_markers = {
        "left": server.scene.add_icosphere("/target/left", radius=0.018, color=LEFT_COLOR),
        "right": server.scene.add_icosphere("/target/right", radius=0.018, color=RIGHT_COLOR),
    }
    @server.on_client_connect
    def _set_initial_camera(client: viser.ClientHandle) -> None:
        # Behind the arms (operator's point of view — you see their backs),
        # slightly elevated, framed so no manual zoom/orbit is needed.
        client.camera.position = (-1.4, 0.0, 0.9)
        client.camera.look_at = (0.2, 0.0, 0.35)

    url = f"http://localhost:{server.get_port()}"
    log.info("Live view ready: %s (Ctrl+C to stop)", url)
    if not args.no_browser:
        webbrowser.open(url)

    rr = _init_rerun(not args.no_rerun, cam_names)
    max_points = max(2, int(_TRAIL_SECONDS * args.fps))
    trails = {"left": TrajectoryTrail(max_points), "right": TrajectoryTrail(max_points)}
    raw_trails = {"left": TrajectoryTrail(max_points), "right": TrajectoryTrail(max_points)}

    play_sounds = not args.no_sounds
    side_indices = _side_joint_indices(runtime)
    home_q = runtime.config.home_q.astype(np.float32)
    # Per-arm anchors: None = disengaged (arm holds home). X/A engage a side
    # by snapshotting the current state; a per-side double clap disengages it.
    anchors: dict[str, object] = {"left": None, "right": None}
    prev_button = {"left": False, "right": False}
    clap = {"left": DoubleClapDetector(), "right": DoubleClapDetector()}
    interval = 1.0 / args.fps
    log.info("Arms idle at home. Anchor with X (left) / A (right), or with a "
             "double clap of that side's gripper — both gestures re-anchor.")
    try:
        while True:
            loop_start = time.perf_counter()
            sample = tracker.latest()
            side_tracked = {"left": sample.left_tracked, "right": sample.right_tracked}

            widths = zero_gripper_widths() if grippers is None else grippers.read_normalized_widths()
            if rr is not None:
                if cameras:
                    cam_frames = read_camera_frames(
                        cameras, cam_names, width=args.cam_width, height=args.cam_height
                    )
                    for key, frame in cam_frames.items():
                        rr.log(key, rr.Image(frame).compress(jpeg_quality=75))
                rr.log("observation.feetech.left_width_mm", rr.Scalars(float(widths.left_mm)))
                rr.log("observation.feetech.right_width_mm", rr.Scalars(float(widths.right_mm)))

            state = _sample_state(sample, widths)

            # (Re-)anchor a side on either gesture — X/A rising edge (hands
            # free, during setup) or that side's double clap (hands inside
            # the HandUMIs, mid-operation). Both do exactly the same thing:
            # snapshot that hand's current pose <-> the arm's home TCP, and
            # the arm follows relative motion from there.
            pressed = _anchor_buttons_pressed(tracker)
            side_width_mm = {"left": widths.left_mm, "right": widths.right_mm}
            for side in ("left", "right"):
                rise = pressed[side] and not prev_button[side]
                prev_button[side] = pressed[side]
                # Feeding the same width to both detector channels makes the
                # clap side-local.
                clapped = clap[side].update(side_width_mm[side], side_width_mm[side], loop_start)
                if not (rise or clapped):
                    continue
                if not side_tracked[side]:
                    log.warning("%s anchor ignored — that controller is not tracked.", side)
                    continue
                anchors[side] = retarget_anchors_from_raw_state(
                    state,
                    left_robot_pose7=home_left_pose7,
                    right_robot_pose7=home_right_pose7,
                    max_reach=max_reach,
                )
                log.info("%s arm anchored — follows from home.", side)
                log_say(f"{side} anchored", play_sounds=play_sounds)

            # Anchored + tracked sides follow their anchor via IK; anchored
            # but momentarily untracked sides hold the current pose (None
            # target). Never-anchored sides are parked kinematically at
            # home_q every tick (no IK target — chasing the home pose through
            # IK left the arm in a jittery tug-of-war of costs).
            ik_targets: dict[str, tuple | None] = {"left": None, "right": None}
            for index, side in enumerate(("left", "right")):
                if anchors[side] is None or not side_tracked[side]:
                    continue
                pose7 = raw_state_robot_target_pose7(state, anchors[side])[index]
                ik_targets[side] = (pose7[:3], pose7[3:7])
                target_markers[side].position = tuple(pose7[:3])
            q = solver.ik(q, left_pose=ik_targets["left"], right_pose=ik_targets["right"])
            for side in ("left", "right"):
                if anchors[side] is None:
                    q[side_indices[side]] = home_q[side_indices[side]]
            # Gripper fingers always mirror the real HandUMI opening
            # (normalized Feetech width scaled to each finger's URDF range).
            runtime.set_finger_positions(
                q, {"left": widths.left_normalized, "right": widths.right_normalized}
            )
            robot_view.update_cfg(q)

            if rr is not None:
                for side, tcp, raw, color in (
                    ("left", sample.left_tcp_pose, sample.left_controller_pose, LEFT_COLOR),
                    ("right", sample.right_tcp_pose, sample.right_controller_pose, RIGHT_COLOR),
                ):
                    if side_tracked[side]:
                        _log_rerun(rr, side, tcp, raw, trails[side], raw_trails[side], color)

            dt = time.perf_counter() - loop_start
            if (sleep := interval - dt) > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        log.info("Stopping.")
    finally:
        disconnect_cameras(cameras)
        if grippers is not None:
            grippers.close()
        tracker.stop()


if __name__ == "__main__":
    main()
