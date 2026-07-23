#!/usr/bin/env python3
"""Live simulation teleop: move the HandUMI and watch the robot follow in Viser (+ Rerun).

Nothing is recorded. The same pipeline the post-hoc replay uses runs live:

    TrackingProvider.latest()                (PICO or Meta Quest)
      -> controller->TCP calibration          configs/calibration/<device>_controller_tcp.yaml
      -> anchored retargeting                (same as ``handumi replay``)
      -> bimanual IK                          robots/kinematics.py
      -> Viser                                robot follows your hands

so what you see is what a recording would replay. Use it to sanity-check
tracking health and TCP calibration before a session.

Rerun (on by default, --no-rerun to disable) shows the calibrated TCP
trails in the workspace frame — tracking-side truth, before retargeting/IK.

Teleop anchoring maps your current HandUMI pose to the robot home TCP and the
robot follows relative motion from there. The double-clap gesture (close/open
one gripper twice) starts idle arms; once teleop is active, another double clap
resets teleop by clearing anchors and parking enabled arms at home. Keyboard
Space can also start idle arms when explicitly enabled with ``--space-start``.
With ``--auto-start``, stable tracking starts a five-second countdown and then
anchors the enabled arms through the same start path.

  Space                 start both arms that are not anchored yet (--space-start)
  double clap           start teleop, or reset/pause active teleop

Both arms start parked at home until their first anchor. Spoken feedback
("left anchored", ...) — --no-sounds to mute.

--anchor-z <m> enables the table-anchor ritual: anchor with the gripper tip
RESTING ON THE TABLE and that pose maps to the given robot-world height
(0.0 = table at the arm-base plane) instead of the home TCP — absolute
heights then match: the real tip touches the table exactly when the sim
tip reaches z = anchor-z.

Usage
-----
::

    handumi teleop --device meta
    handumi teleop --device meta --quest-ip 127.0.0.1 --no-browser
    handumi teleop --device pico --pico-mode mandos
"""

from __future__ import annotations

import argparse
import logging
import sys
import webbrowser
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
    read_camera_frames,
    resolve_camera_ids,
)
from handumi.config import DEFAULT_RIG_CONFIG
from handumi.robots.registry import EMBODIMENT_NAMES, load_embodiment, resolve_home_q
from handumi.robots.utils import IDENTITY_POSE7
from handumi.scripts.record import _camera_list_arg, build_tracker, connect_feetech
from handumi.teleop.common import (
    DEFAULT_MOTION_SMOOTHING_TIME_CONSTANT_S,
    DEFAULT_TELEOP_FPS,
    SIDE_CHOICES,
    KeyboardSpaceListener,
    TeleopLoopTimer,
    TeleopMotionSmoother,
    enabled_sides as _enabled_sides,
    latest_widths,
    sample_state as _sample_state,
    start_sides as _start_sides,
    tracking_ready_for_sides as _tracking_ready_for_sides,
    tracking_world_map as _tracking_world_map,
)
from handumi.teleop.core import TeleopController
from handumi.teleop.session import TeleopSession
from handumi.teleop.trajectory import DelayedJointCommandPlayer
from handumi.tracking.gestures import DoubleClapDetector
from handumi.utils.speech import log_say
from handumi.utils.trajectory import TrajectoryTrail
from handumi.visualization import BACKGROUND_COLOR, LEFT_COLOR, RIGHT_COLOR

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
sim_log = logging.getLogger("handumi.teleop_sim")

_TRAIL_SECONDS = 10.0
_CHART_WINDOW_S = 20.0  # rolling window for the gripper-width chart
DEFAULT_SIM_COMMAND_RATE_HZ = 100.0
DEFAULT_SIM_TRAJECTORY_DELAY_MS = 80.0


def _parse_sim_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    show_advanced = "--help-advanced" in raw_argv
    raw_argv = [value for value in raw_argv if value != "--help-advanced"]

    def advanced(text: str) -> str:
        return text if show_advanced else argparse.SUPPRESS

    p = argparse.ArgumentParser(
        description=(
            "Teleoperate HandUMI through a robot profile in live simulation."
        )
    )
    p.add_argument("--help-advanced", action="store_true", help="Show expert hardware options.")
    p.add_argument("--device", choices=("pico", "meta"), required=True)
    p.add_argument("--robot", choices=EMBODIMENT_NAMES, default="piper")
    p.add_argument(
        "--home-pose",
        default=None,
        help=advanced("Override a named home pose."),
    )
    p.add_argument("--side", choices=SIDE_CHOICES, default="both")
    p.add_argument("--port", type=int, default=8003, help=advanced("Viser port."))
    p.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_TELEOP_FPS,
        help=advanced("Control frequency."),
    )
    p.add_argument(
        "--command-rate-hz",
        type=float,
        default=DEFAULT_SIM_COMMAND_RATE_HZ,
        help=advanced("Fixed-rate playback frequency for interpolated joints."),
    )
    p.add_argument(
        "--trajectory-delay-ms",
        type=float,
        default=DEFAULT_SIM_TRAJECTORY_DELAY_MS,
        help=advanced("Playback delay used to bracket and interpolate IK results."),
    )
    p.add_argument(
        "--duration-s", type=float, default=0.0, help=advanced("0 runs until Ctrl+C.")
    )
    p.add_argument(
        "--translation-scale",
        type=float,
        default=1.0,
        help=advanced("Scale HandUMI translation deltas."),
    )
    p.add_argument(
        "--motion-smoothing-time-constant-s",
        type=float,
        default=DEFAULT_MOTION_SMOOTHING_TIME_CONSTANT_S,
        help=advanced(
            "Shared TCP-pose and joint-command low-pass time constant; 0 disables smoothing."
        ),
    )
    p.add_argument("--no-browser", action="store_true", help="Don't auto-open Viser.")
    p.add_argument(
        "--no-viser",
        action="store_true",
        help="Disable the Viser server and 3D robot view entirely (Rerun remains enabled).",
    )
    p.add_argument("--no-rerun", action="store_true", help="Disable the Rerun view.")
    p.add_argument(
        "--no-sounds", action="store_true", help="Disable spoken anchor/home feedback."
    )
    p.add_argument(
        "--space-start",
        action="store_true",
        help="Allow keyboard Space to start any unanchored enabled arms.",
    )
    p.add_argument(
        "--auto-start",
        action="store_true",
        help=(
            "Start enabled arms automatically after controller tracking remains "
            "valid for --auto-start-delay-s."
        ),
    )
    p.add_argument(
        "--auto-start-delay-s",
        type=float,
        default=5.0,
        help=advanced("Stable-tracking seconds required by --auto-start."),
    )
    p.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Render a task scene (assets/scenes/<name>/scene.xml) in Viser at "
        "DEFAULT_SCENE_POSITION, e.g. cube_in_box. Static props only.",
    )
    p.add_argument(
        "--anchor-z",
        type=float,
        default=None,
        help=advanced("Table-anchor ritual: anchor with the gripper TIP RESTING ON THE "
        "TABLE, and that pose maps to this robot-world height (meters) instead "
        "of the arm's home TCP — absolute heights then match for the whole "
        "session (real tip touches the table exactly when the sim tip does). "
        "0.0 = table at the arm-base plane. Omit for the default relative "
        "mapping (anchor pose -> home TCP)."),
    )
    p.add_argument(
        "--controller-tcp-calibration",
        type=Path,
        default=None,
        help=advanced("Override the robot/device Controller->TCP calibration."),
    )

    p.add_argument(
        "--rig-config",
        type=Path,
        default=DEFAULT_RIG_CONFIG,
        help=advanced("Machine-local hardware configuration."),
    )

    p.add_argument(
        "--cameras",
        type=_camera_list_arg,
        default=None,
        help="Cameras shown in Rerun; defaults to both wrist cameras.",
    )
    p.add_argument("--cam-width", type=int, default=640, help=advanced("Camera width."))
    p.add_argument("--cam-height", type=int, default=480, help=advanced("Camera height."))
    p.add_argument("--cam-fps", type=int, default=30, help=advanced("Camera FPS."))
    p.add_argument("--skip-cameras", action="store_true", help=advanced("Disable camera views."))
    p.add_argument("--feetech-port", type=str, default=None, help=advanced("Feetech port override."))
    p.add_argument("--skip-feetech", action="store_true", help=advanced("Disable Feetech."))

    # Tracking flags, same names as ``handumi record`` (shared build_tracker).
    p.add_argument("--quest-ip", type=str, default=None, help=advanced("Quest IP override."))
    p.add_argument("--tcp-port", type=int, default=None, help=advanced("Quest TCP port."))
    p.add_argument("--sync-port", type=int, default=None, help=advanced("Quest sync port."))
    p.add_argument(
        "--pico-mode", choices=("mandos", "object", "whole-body"), default="mandos", help=advanced("PICO tracking mode.")
    )
    pico_transport = p.add_mutually_exclusive_group()
    pico_transport.add_argument("--pico-adb", action="store_true", help=advanced("Use PICO ADB."))
    pico_transport.add_argument("--pico-wifi", action="store_true", help=advanced("Use PICO Wi-Fi."))
    p.add_argument("--skip-adb-check", action="store_true", help=advanced("Skip ADB checks."))
    if show_advanced:
        p.print_help()
        raise SystemExit(0)
    return p.parse_args(raw_argv)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _parse_sim_args(argv)


class AutoStartCountdown:
    """One-shot start after continuous valid tracking for a safety delay."""

    def __init__(self, *, enabled: bool, delay_s: float) -> None:
        self.enabled = enabled
        self.delay_s = delay_s
        self.started_at: float | None = None
        self.announced_seconds: int | None = None
        self.completed = False

    def update(
        self,
        *,
        now: float,
        tracking_ready: bool,
        already_active: bool,
        idle_sides: tuple[str, ...],
    ) -> tuple[str, ...]:
        if not self.enabled or self.completed:
            return ()
        if already_active:
            self.completed = True
            return ()
        if not tracking_ready:
            if self.started_at is not None:
                sim_log.info("Auto-start countdown cancelled: controller tracking lost.")
            self.started_at = None
            self.announced_seconds = None
            return ()
        if self.started_at is None:
            self.started_at = now
            self.announced_seconds = int(np.ceil(self.delay_s))
            sim_log.info(
                "Controllers detected. Auto-start in %d s; hold them steady.",
                self.announced_seconds,
            )

        remaining = self.delay_s - (now - self.started_at)
        if remaining <= 0.0:
            self.completed = True
            if idle_sides:
                sim_log.info(
                    "Auto-start countdown complete; starting %s.",
                    "/".join(idle_sides),
                )
            return idle_sides

        seconds = int(np.ceil(remaining))
        if seconds < (self.announced_seconds or seconds):
            self.announced_seconds = seconds
            sim_log.info("Auto-start in %d s ...", seconds)
        return ()


def _resolve_camera_usage(args: argparse.Namespace) -> None:
    """Cameras only ever appear in Rerun, so tie their lifecycle to it.

    Without Rerun, connecting cameras just occupies devices for nothing:
    disable them automatically. Camera-selection flags with --no-rerun are
    almost certainly a mistake, so fail loudly instead of silently ignoring
    them.
    """
    if not args.no_rerun:
        return
    if args.cameras is not None:
        raise SystemExit(
            "Cameras are only shown in Rerun. Remove --no-rerun, or drop "
            "--cameras."
        )
    if not args.skip_cameras:
        sim_log.info("Rerun disabled; skipping cameras (they are only shown in Rerun).")
        args.skip_cameras = True


def _load_calibration(args: argparse.Namespace):
    from handumi.calibration.control_tcp import ControllerTcpCalibration

    path, source = calibration_path_for_robot_device(
        args.robot,
        args.device,
        explicit_path=args.controller_tcp_calibration,
    )
    if path.exists():
        calibration = load_controller_tcp_calibration(path)
        sim_log.info("controller->TCP calibration: %s", source)
        return calibration
    sim_log.warning(
        "No calibration at %s — previewing RAW controller poses. "
        "See docs/README_tcp_offset.md to calibrate.",
        path,
    )
    return ControllerTcpCalibration(
        left=IDENTITY_POSE7.astype(np.float32).copy(),
        right=IDENTITY_POSE7.astype(np.float32).copy(),
        source=None,
    )


def _validate_unique_camera_ids(
    camera_names: list[str], camera_ids: list[int | str]
) -> None:
    """Reject camera mappings that would show one device in multiple grid cells."""
    duplicates = {
        camera_id for camera_id in camera_ids if camera_ids.count(camera_id) > 1
    }
    if not duplicates:
        return
    mappings = ", ".join(
        f"{name}={camera_id}" for name, camera_id in zip(camera_names, camera_ids)
    )
    raise SystemExit(
        f"Selected cameras must use distinct devices ({mappings}). "
        "Fix the cameras section in configs/rig.yaml."
    )


def _init_rerun(enabled: bool, cam_names: list[str]):
    """Start Rerun with the classic live layout: 3D tracking on the left,
    cameras top-right, gripper-width chart bottom-right."""
    if not enabled:
        return None
    import rerun as rr
    import rerun.blueprint as rrb
    import rerun.datatypes as rdt

    rr.init("handumi_teleop_sim", spawn=True)
    rr.log("tracking", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for path, name, color in (
        ("observation.feetech.left_width_mm", "left_width_mm", LEFT_COLOR),
        ("observation.feetech.right_width_mm", "right_width_mm", RIGHT_COLOR),
    ):
        rr.log(
            path,
            rr.SeriesLines(colors=[[*color, 255]], widths=[2.5], names=[name]),
            static=True,
        )
    # Faint corner markers spanning the working volume: the 3D view auto-fits
    # to data bounds, so these fix the initial framing/zoom (wide horizontally,
    # short vertically) instead of hugging the first few points.
    half_xy, half_z = 0.75, 0.4
    corners = [
        [sx * half_xy, sy * half_xy, sz * half_z]
        for sx in (-1, 1)
        for sy in (-1, 1)
        for sz in (-1, 1)
    ]
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
    rr.log(
        f"tracking/{side}/tcp",
        rr.Points3D([tcp_pose7[:3]], colors=[color], radii=0.012),
    )
    points = trail.points()
    if len(points) >= 2:
        rr.log(
            f"tracking/{side}/trail",
            rr.LineStrips3D([points], colors=[color], radii=0.003),
        )

    faint = [*color, 90]
    raw_trail.append(raw_pose7[:3])
    rr.log(
        f"tracking/{side}/raw",
        rr.Points3D([raw_pose7[:3]], colors=[faint], radii=0.007),
    )
    raw_points = raw_trail.points()
    if len(raw_points) >= 2:
        rr.log(
            f"tracking/{side}/raw_trail",
            rr.LineStrips3D([raw_points], colors=[faint], radii=0.0015),
        )


def _run_sim() -> None:
    args = _parse_sim_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be > 0.")
    if args.auto_start_delay_s <= 0.0:
        raise SystemExit("--auto-start-delay-s must be greater than zero.")
    if args.command_rate_hz <= 0.0:
        raise SystemExit("--command-rate-hz must be > 0.")
    if args.trajectory_delay_ms < 0.0:
        raise SystemExit("--trajectory-delay-ms must be >= 0.")
    if args.motion_smoothing_time_constant_s < 0.0:
        raise SystemExit("--motion-smoothing-time-constant-s must be >= 0.")

    _resolve_camera_usage(args)
    calibration = _load_calibration(args)
    # Keep controller buttons from changing the tracking workspace during sim
    # teleop. Gripper double-clap and optional Space are the only start inputs.
    tracker = build_tracker(args, calibration, reset_workspace_on_x=False)
    tracker.start()

    cameras: list = []
    cam_names: list[str] = []
    if not args.skip_cameras:
        camera_names = args.cameras or ["left_wrist", "right_wrist"]
        cam_ids = resolve_camera_ids(None, args.rig_config, camera_names=camera_names)
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

    grippers = connect_feetech(args)  # honors --skip-feetech internally

    sim_log.info("Loading %s IK solver (JAX JIT warmup, ~30s on CPU) ...", args.robot)
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
        anchor_z=args.anchor_z,
    )
    q = home_q.copy()
    sim_log.info("Selected home pose: %s", home_pose_name)
    controller.warmup()

    server = None
    robot_view = None
    if not args.no_viser:
        import viser
        from viser.extras import ViserUrdf

        server = viser.ViserServer(port=args.port)
        server.scene.add_grid("/grid", width=3.0, height=3.0, cell_size=0.1)
        urdf = runtime.load_urdf(load_meshes=True)
        robot_view = ViserUrdf(server, urdf, root_node_name="/robot")
        robot_view.update_cfg(q)

    physics = None
    scene_frames: dict[str, Any] = {}
    if args.scene is not None:
        from handumi.sim.scene import DEFAULT_SCENE_POSITION, load_scene

        scene_position = DEFAULT_SCENE_POSITION
        # Props render under per-body frames so physics can move them.
        if server is not None:
            for body in load_scene(args.scene, position=scene_position):
                scene_frame = server.scene.add_frame(
                    f"/scene/{body.name}",
                    position=tuple(body.rest_position),
                    show_axes=False,
                )
                scene_frames[body.name] = scene_frame
                for i, geom in enumerate(body.geoms):
                    sx, sy, sz = (2.0 * s for s in geom.size)
                    cr, cg, cb = (int(round(c * 255)) for c in geom.rgba[:3])
                    server.scene.add_box(
                        f"/scene/{body.name}/g{i}",
                        dimensions=(sx, sy, sz),
                        color=(cr, cg, cb),
                        position=tuple(geom.local_position),
                    )
        if runtime.config.mjcf is not None:
            from handumi.sim.mujoco_sim import MujocoPhysics

            physics = MujocoPhysics(
                mjcf_path=runtime.config.mjcf,
                actuator_names=[
                    runtime.mjcf_actuator_name(n)
                    for n in runtime.robot.joints.actuated_names
                ],
                scene_name=args.scene,
                scene_position=scene_position,
            )
            physics.start()
            sim_log.info(
                "Scene %r with MuJoCo contact physics at %s.",
                args.scene,
                scene_position,
            )
        else:
            sim_log.info(
                "Scene %r rendered statically (no MJCF for %s).", args.scene, args.robot
            )
    target_markers = {}
    if server is not None:
        target_markers = {
            "left": server.scene.add_icosphere(
                "/target/left", radius=0.018, color=LEFT_COLOR
            ),
            "right": server.scene.add_icosphere(
                "/target/right", radius=0.018, color=RIGHT_COLOR
            ),
        }

        @server.on_client_connect
        def _set_initial_camera(client: viser.ClientHandle) -> None:
            # Behind the arms (operator's point of view — you see their backs),
            # slightly elevated, framed so no manual zoom/orbit is needed.
            client.camera.position = (-1.4, 0.0, 0.9)
            client.camera.look_at = (0.2, 0.0, 0.35)

        url = f"http://localhost:{server.get_port()}"
        sim_log.info("Live view ready: %s (Ctrl+C to stop)", url)
        if not args.no_browser:
            webbrowser.open(url)
    else:
        sim_log.info("Viser disabled; streaming live cameras and tracking only to Rerun.")

    rr = _init_rerun(not args.no_rerun, cam_names)
    max_points = max(2, int(_TRAIL_SECONDS * args.fps))
    trails = {"left": TrajectoryTrail(max_points), "right": TrajectoryTrail(max_points)}
    raw_trails = {
        "left": TrajectoryTrail(max_points),
        "right": TrajectoryTrail(max_points),
    }

    play_sounds = not args.no_sounds
    # Robot pose each side's anchor maps to. With --anchor-z the ritual is
    # "anchor with the tip resting on the table": same home x/y, but the
    # tip's height at anchor time corresponds to anchor-z in robot world,
    # pinning absolute heights (real table touch == sim table touch).
    if args.anchor_z is not None:
        sim_log.info(
            "Table-anchor mode: anchor with the tip ON the table "
            "(maps to z=%.3f in robot world).",
            args.anchor_z,
        )
    clap = DoubleClapDetector()
    space_listener = KeyboardSpaceListener(enabled=args.space_start)
    space_listener.start()
    auto_start = AutoStartCountdown(
        enabled=args.auto_start,
        delay_s=args.auto_start_delay_s,
    )
    episode_start: float | None = None
    frame = 0
    loop_timer = TeleopLoopTimer(args.fps)
    motion_smoother = TeleopMotionSmoother(args.motion_smoothing_time_constant_s)
    teleop_session = TeleopSession(controller, motion_smoother)
    joint_names = list(runtime.robot.joints.actuated_names)

    def write_sim_command(
        command_q: np.ndarray,
        openings: dict[str, float],
    ) -> None:
        del openings
        if physics is not None:
            physics.set_ctrl(
                {
                    runtime.mjcf_actuator_name(name): float(command_q[i])
                    for i, name in enumerate(joint_names)
                }
            )
        elif robot_view is not None:
            robot_view.update_cfg(command_q)

    command_player = DelayedJointCommandPlayer(
        write_sim_command,
        command_rate_hz=args.command_rate_hz,
        delay_s=args.trajectory_delay_ms / 1000.0,
    )
    sim_log.info(
        "Joint trajectory playback: %.1f Hz with %.0f ms delay.",
        args.command_rate_hz,
        args.trajectory_delay_ms,
    )
    if args.auto_start:
        manual_hint = " Space remains available." if args.space_start else ""
        sim_log.info(
            "Arms idle at home. Waiting for controller tracking; auto-start "
            "after %.1f s.%s",
            args.auto_start_delay_s,
            manual_hint,
        )
    elif args.space_start:
        sim_log.info(
            "Arms idle at home. Start with Space, or double clap a gripper "
            "to start enabled arms."
        )
    else:
        sim_log.info("Arms idle at home. Double clap a gripper to start enabled arms.")
    try:
        while True:
            loop_start, _ = loop_timer.tick()
            if episode_start is not None:
                if (
                    args.duration_s > 0.0
                    and loop_start - episode_start >= args.duration_s
                ):
                    break
            sample = tracker.latest()
            widths = latest_widths(grippers)
            inputs = teleop_session.inputs(sample, widths)
            side_tracked = inputs.side_tracked
            if rr is not None:
                if cameras:
                    cam_frames = read_camera_frames(
                        cameras, cam_names, width=args.cam_width, height=args.cam_height
                    )
                    for key, frame in cam_frames.items():
                        rr.log(key, rr.Image(frame).compress(jpeg_quality=75))
                rr.log(
                    "observation.feetech.left_width_mm",
                    rr.Scalars(float(widths.left_mm)),
                )
                rr.log(
                    "observation.feetech.right_width_mm",
                    rr.Scalars(float(widths.right_mm)),
                )

            # Double clap toggles teleop: first clap starts idle arms, next clap
            # clears anchors so the robot parks at home and waits for a fresh
            # start. Space remains an optional start shortcut for idle arms.
            start_sides: tuple[str, ...] = ()
            reset_this_frame = False
            if args.space_start and space_listener.consume_space():
                start_sides = controller.idle_sides()
                if start_sides:
                    sim_log.info("Space pressed; starting %s.", "/".join(start_sides))
            auto_start_sides = auto_start.update(
                now=loop_start,
                tracking_ready=_tracking_ready_for_sides(
                    inputs.raw_source_poses, side_tracked, enabled_sides
                ),
                already_active=controller.active,
                idle_sides=controller.idle_sides(),
            )
            if auto_start_sides:
                start_sides = auto_start_sides
            if clap.update(widths.left_mm, widths.right_mm, loop_start):
                if controller.active:
                    command_player.stop()
                    q = controller.reset()
                    motion_smoother.reset(home_q)
                    episode_start = None
                    frame = 0
                    reset_this_frame = True
                    sim_log.info(
                        "Double clap detected; teleop reset, arms parking at home."
                    )
                    log_say("teleop reset", play_sounds=play_sounds)
                else:
                    start_sides = enabled_sides
                    sim_log.info(
                        "Double clap detected; starting %s.", "/".join(start_sides)
                    )

            teleop_frame = teleop_session.advance(
                inputs, now_s=loop_start, start_sides=start_sides
            )
            anchored_sides = teleop_frame.anchored_sides
            anchored_this_frame = bool(anchored_sides)
            for side in anchored_sides:
                sim_log.info("%s arm anchored — follows from home.", side)
                log_say(f"{side} anchored", play_sounds=play_sounds)

            if (anchored_this_frame or reset_this_frame) and physics is not None:
                # Starting or resetting teleop also puts every scene prop
                # (cube, box, ...) back at its initial pose.
                physics.reset()
                sim_log.info("Scene reset to its initial state.")
            if reset_this_frame:
                write_sim_command(home_q, inputs.openings)
            if episode_start is None and anchored_this_frame:
                episode_start = loop_start
                frame = 0
                sim_log.info("Teleop timer started.")

            # Anchored + tracked sides follow their anchor via IK; anchored
            # but momentarily untracked sides hold the current pose (None
            # target). Never-anchored sides are parked kinematically at
            # home_q every tick (no IK target — chasing the home pose through
            # IK left the arm in a jittery tug-of-war of costs).
            q = teleop_frame.q
            if anchored_this_frame:
                command_player.stop()
                command_player.start(
                    q,
                    inputs.openings,
                    time_s=loop_start,
                )
            elif controller.active:
                if not command_player.running:
                    command_player.start(
                        q,
                        inputs.openings,
                        time_s=loop_start,
                    )
                else:
                    command_player.push(
                        q,
                        inputs.openings,
                        time_s=loop_start,
                    )
            for side, pose7 in teleop_frame.step.target_pose7.items():
                if target_markers:
                    target_markers[side].position = tuple(pose7[:3])

            if physics is not None:
                # The 100 Hz delayed trajectory feeds MuJoCo on its own
                # callback thread. Viser renders what physics actually
                # settled on (grasps included), not the raw 30 Hz IK result.
                settled = physics.joint_positions()
                q_render = q.copy()
                for i, name in enumerate(joint_names):
                    q_render[i] = settled.get(runtime.mjcf_actuator_name(name), q[i])
                if robot_view is not None:
                    robot_view.update_cfg(q_render)
                for body_name, scene_frame in scene_frames.items():
                    pose = physics.body_pose(body_name)
                    if pose is not None:
                        position, quat_wxyz = pose
                        scene_frame.position = tuple(position.tolist())
                        scene_frame.wxyz = tuple(quat_wxyz.tolist())
            if rr is not None:
                for side, tcp, raw, color in (
                    (
                        "left",
                        sample.left_tcp_pose,
                        sample.left_controller_pose,
                        LEFT_COLOR,
                    ),
                    (
                        "right",
                        sample.right_tcp_pose,
                        sample.right_controller_pose,
                        RIGHT_COLOR,
                    ),
                ):
                    if side_tracked[side]:
                        _log_rerun(
                            rr, side, tcp, raw, trails[side], raw_trails[side], color
                        )

            loop_timer.sleep(loop_start)
            if episode_start is not None:
                frame += 1
    except KeyboardInterrupt:
        sim_log.info("Stopping.")
    finally:
        space_listener.close()
        command_player.stop()
        if physics is not None:
            physics.close()
        disconnect_cameras(cameras)
        if grippers is not None:
            grippers.close()
        tracker.stop()


def main() -> None:
    _run_sim()


if __name__ == "__main__":
    main()
