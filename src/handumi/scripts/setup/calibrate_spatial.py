#!/usr/bin/env python3
"""Interactive ChArUco calibration for HandUMI cameras and Quest tracking."""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from handumi.calibration.spatial import (
    CameraIntrinsics,
    CharucoBoardSpec,
    CharucoDetection,
    board_from_table_pose,
    calibrate_fisheye,
    calibration_hash,
    detect_charuco,
    draw_detection,
    estimate_board_pose,
    load_yaml,
    new_spatial_calibration,
    pose7_from_dict,
    pose7_to_dict,
    solve_controller_camera,
    solve_table_camera,
    solve_table_quest,
    write_yaml,
)
from handumi.cameras.opencv import OpenCVCameraDevice
from handumi.config import DEFAULT_RIG_CONFIG, load_rig_section
from handumi.tracking.meta_quest import MetaQuestConfig, MetaQuestReceiver
from handumi.tracking.transforms import unity_pose_to_handumi
from handumi.robots.utils import pose7_to_mat
from handumi.utils.trajectory import TrajectoryTrail
from handumi.visualization import BACKGROUND_COLOR, LEFT_COLOR, RIGHT_COLOR


log = logging.getLogger("handumi.calibrate_spatial")
DEFAULT_SPATIAL = Path("outputs/calibration/spatial.yaml")
DEFAULT_SESSION = Path("outputs/calibration/session.yaml")
MIN_CORNERS = 12
MAX_SYNC_ERROR_MS = 20.0
WORKSPACE_COLOR = (110, 180, 255)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _board_from_rig(path: Path) -> CharucoBoardSpec:
    section = load_rig_section(path, "spatial_calibration")
    return CharucoBoardSpec.from_dict(section.get("board"))


def _camera(path: Path, name: str, fps: int, width: int, height: int) -> OpenCVCameraDevice:
    cameras = load_rig_section(path, "cameras")
    entry = cameras.get(name)
    if not isinstance(entry, dict) or "index_or_path" not in entry:
        raise SystemExit(f"Missing cameras.{name}.index_or_path in {path}.")
    return OpenCVCameraDevice(entry["index_or_path"], fps, width, height)


def _camera_source(path: Path, name: str) -> int | str:
    cameras = load_rig_section(path, "cameras")
    entry = cameras.get(name)
    if not isinstance(entry, dict) or "index_or_path" not in entry:
        raise SystemExit(f"Missing cameras.{name}.index_or_path in {path}.")
    return entry["index_or_path"]


def _load_spatial(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing spatial calibration: {path}")
    data = load_yaml(path)
    if data.get("kind") != "handumi_spatial_calibration":
        raise SystemExit(f"Not a HandUMI spatial calibration: {path}")
    return data


def _intrinsics(spatial: dict, camera: str) -> CameraIntrinsics:
    entry = (spatial.get("cameras") or {}).get(camera)
    if not isinstance(entry, dict):
        raise SystemExit(f"Camera {camera!r} has no intrinsics in spatial calibration.")
    return CameraIntrinsics.from_dict(entry)


def _open_preview(title: str) -> None:
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 960, 720)


def _overlay(preview: np.ndarray, lines: list[str], ok: bool) -> np.ndarray:
    color = (20, 200, 20) if ok else (20, 20, 230)
    for index, line in enumerate(lines):
        cv2.putText(
            preview,
            line,
            (14, 28 + index * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    return preview


def _capture(
    *,
    camera: OpenCVCameraDevice,
    board: CharucoBoardSpec,
    title: str,
    requested_views: int,
    pair_at: Callable[[int], tuple[np.ndarray, float] | None] | None = None,
) -> tuple[list[CharucoDetection], list[np.ndarray]]:
    detections: list[CharucoDetection] = []
    poses: list[np.ndarray] = []
    last_sequence = -1
    _open_preview(title)
    try:
        while len(detections) < requested_views:
            sample = camera.sample_at()
            detection = detect_charuco(sample.image, board, min_corners=MIN_CORNERS)
            pair = None if pair_at is None else pair_at(sample.capture_time_ns)
            sync_ms = None if pair is None else pair[1]
            valid = detection is not None and (pair_at is None or pair is not None)
            preview = draw_detection(sample.image, detection)
            corners = 0 if detection is None else detection.count
            lines = [
                f"views {len(detections)}/{requested_views}  corners {corners}",
                "SPACE capture   Q finish/cancel",
            ]
            if pair_at is not None:
                sync_text = "n/a" if sync_ms is None else f"{sync_ms:.1f} ms"
                lines.insert(1, f"Quest-camera sync {sync_text} (max {MAX_SYNC_ERROR_MS:.0f} ms)")
            cv2.imshow(title, _overlay(preview, lines, valid))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" ") and sample.sequence != last_sequence:
                if not valid or detection is None:
                    log.warning("View rejected: board/tracking/synchronization gate failed.")
                    continue
                last_sequence = sample.sequence
                detections.append(detection)
                if pair is not None:
                    poses.append(pair[0])
                log.info("Accepted view %d/%d (%d corners).", len(detections), requested_views, corners)
    finally:
        cv2.destroyWindow(title)
    return detections, poses


def _board_poses_with_gate(
    detections: list[CharucoDetection],
    controller_poses: list[np.ndarray],
    intrinsics: CameraIntrinsics,
    *,
    max_error_px: float = 1.5,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    accepted_controllers: list[np.ndarray] = []
    accepted_boards: list[np.ndarray] = []
    for controller, detection in zip(controller_poses, detections, strict=True):
        board_pose, error_px = estimate_board_pose(detection, intrinsics)
        if error_px > max_error_px:
            log.warning("Rejected captured view with %.2f px pose error.", error_px)
            continue
        accepted_controllers.append(controller)
        accepted_boards.append(board_pose)
    return accepted_controllers, accepted_boards


def _quest_pairer(
    receiver: MetaQuestReceiver,
    side: str,
) -> Callable[[int], tuple[np.ndarray, float] | None]:
    def pair_at(capture_time_ns: int) -> tuple[np.ndarray, float] | None:
        aligned = receiver.aligned_at(capture_time_ns)
        if aligned is None:
            return None
        controller = getattr(aligned.frame, side)
        if not controller.tracked or not controller.valid:
            return None
        sync_ms = abs(aligned.aligned_time_ns - capture_time_ns) / 1e6
        if sync_ms > MAX_SYNC_ERROR_MS:
            return None
        pose = unity_pose_to_handumi(controller.position, controller.quaternion)
        return np.concatenate([pose.position, pose.quaternion]).astype(np.float32), sync_ms

    return pair_at


def _connect_quest(args: argparse.Namespace) -> MetaQuestReceiver:
    base = MetaQuestConfig.from_yaml(args.rig_config)
    config = MetaQuestConfig(
        quest_ip=args.quest_ip or base.quest_ip,
        tcp_port=args.tcp_port or base.tcp_port,
        sync_port=args.sync_port or base.sync_port,
        connect_retry_s=base.connect_retry_s,
        frame_stale_timeout_s=base.frame_stale_timeout_s,
    )
    receiver = MetaQuestReceiver(config)
    receiver.start()
    log.info("Connecting to Quest at %s:%d ...", config.quest_ip, config.tcp_port)
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if receiver.metrics()["streaming"]:
            return receiver
        time.sleep(0.1)
    receiver.stop()
    raise SystemExit("Quest did not start streaming within 15 seconds.")


def cmd_inspect(args: argparse.Namespace) -> None:
    board = _board_from_rig(args.rig_config)
    camera = _camera(args.rig_config, args.camera, args.fps, args.width, args.height)
    camera.connect()
    try:
        _capture(
            camera=camera,
            board=board,
            title=f"HandUMI board inspection: {args.camera}",
            requested_views=1,
        )
    finally:
        camera.disconnect()


def cmd_intrinsics(args: argparse.Namespace) -> None:
    board = _board_from_rig(args.rig_config)
    camera = _camera(args.rig_config, args.camera, args.fps, args.width, args.height)
    camera.connect()
    try:
        detections, _ = _capture(
            camera=camera,
            board=board,
            title=f"HandUMI intrinsics: {args.camera}",
            requested_views=args.views,
        )
    finally:
        camera.disconnect()
    if len(detections) < args.views:
        raise SystemExit(f"Only {len(detections)} views captured; requested {args.views}.")
    intrinsics = calibrate_fisheye(args.camera, detections, (args.width, args.height))
    if intrinsics.mean_error_px > args.max_mean_error_px:
        raise SystemExit(
            f"Mean reprojection error {intrinsics.mean_error_px:.3f}px exceeds "
            f"{args.max_mean_error_px:.3f}px; calibration not saved."
        )
    spatial = load_yaml(args.output) if args.output.exists() else new_spatial_calibration(board)
    if CharucoBoardSpec.from_dict(spatial.get("board")) != board:
        raise SystemExit(
            f"Board configuration differs from existing {args.output}; use a new output file."
        )
    spatial.setdefault("cameras", {})[args.camera] = {
        **intrinsics.to_dict(),
        "index_or_path": _camera_source(args.rig_config, args.camera),
        "captured_at": _now_iso(),
    }
    write_yaml(args.output, spatial)
    log.info("Saved %s intrinsics to %s (mean %.3f px).", args.camera, args.output, intrinsics.mean_error_px)


def cmd_mount(args: argparse.Namespace) -> None:
    spatial = _load_spatial(args.spatial)
    board = CharucoBoardSpec.from_dict(spatial.get("board"))
    camera_name = f"{args.side}_wrist"
    intrinsics = _intrinsics(spatial, camera_name)
    camera = _camera(args.rig_config, camera_name, args.fps, intrinsics.width, intrinsics.height)
    receiver = _connect_quest(args)
    camera.connect()
    try:
        detections, controller_poses = _capture(
            camera=camera,
            board=board,
            title=f"HandUMI {args.side} controller-camera mount",
            requested_views=args.views,
            pair_at=_quest_pairer(receiver, args.side),
        )
    finally:
        camera.disconnect()
        receiver.stop()
    if len(detections) < args.views:
        raise SystemExit(f"Only {len(detections)} views captured; requested {args.views}.")
    controller_poses, board_poses = _board_poses_with_gate(
        detections, controller_poses, intrinsics
    )
    if len(board_poses) < max(8, args.views - 2):
        raise SystemExit("Too many views failed the reprojection gate.")
    controller_camera, metrics = solve_controller_camera(controller_poses, board_poses)
    if metrics["translation_rms_mm"] > args.max_rms_mm:
        raise SystemExit(
            f"Mount residual {metrics['translation_rms_mm']:.2f} mm exceeds "
            f"{args.max_rms_mm:.2f} mm; calibration not saved."
        )
    spatial.setdefault("controller_camera", {})[args.side] = {
        "camera": camera_name,
        "controller_from_camera": pose7_to_dict(controller_camera),
        "metrics": metrics,
        "captured_at": _now_iso(),
    }
    write_yaml(args.spatial, spatial)
    log.info("Saved %s mount to %s (RMS %.2f mm).", args.side, args.spatial, metrics["translation_rms_mm"])


def cmd_session(args: argparse.Namespace) -> None:
    spatial = _load_spatial(args.spatial)
    board = CharucoBoardSpec.from_dict(spatial.get("board"))
    camera_name = f"{args.side}_wrist"
    intrinsics = _intrinsics(spatial, camera_name)
    mount = (spatial.get("controller_camera") or {}).get(args.side)
    if not isinstance(mount, dict):
        raise SystemExit(f"Missing {args.side} controller-camera mount in {args.spatial}.")
    controller_camera = pose7_from_dict(mount["controller_from_camera"])
    camera = _camera(args.rig_config, camera_name, args.fps, intrinsics.width, intrinsics.height)
    receiver = _connect_quest(args)
    camera.connect()
    try:
        detections, controller_poses = _capture(
            camera=camera,
            board=board,
            title=f"HandUMI table session: {args.side}",
            requested_views=args.views,
            pair_at=_quest_pairer(receiver, args.side),
        )
    finally:
        camera.disconnect()
        receiver.stop()
    if len(detections) < args.views:
        raise SystemExit(f"Only {len(detections)} views captured; requested {args.views}.")
    controller_poses, board_poses = _board_poses_with_gate(
        detections, controller_poses, intrinsics
    )
    if len(board_poses) < max(4, args.views - 1):
        raise SystemExit("Too many views failed the reprojection gate.")
    table_from_quest, metrics = solve_table_quest(
        controller_poses, controller_camera, board_poses, board
    )
    if metrics["translation_rms_mm"] > args.max_rms_mm:
        raise SystemExit(
            f"Session residual {metrics['translation_rms_mm']:.2f} mm exceeds "
            f"{args.max_rms_mm:.2f} mm; calibration not saved."
        )
    table_from_camera: dict[str, dict] = {}
    if not args.skip_workspace:
        workspace_intrinsics = _intrinsics(spatial, "workspace")
        workspace = _camera(
            args.rig_config,
            "workspace",
            args.fps,
            workspace_intrinsics.width,
            workspace_intrinsics.height,
        )
        workspace.connect()
        try:
            workspace_detections, _ = _capture(
                camera=workspace,
                board=board,
                title="HandUMI fixed workspace camera",
                requested_views=args.workspace_views,
            )
        finally:
            workspace.disconnect()
        if len(workspace_detections) < args.workspace_views:
            raise SystemExit(
                f"Only {len(workspace_detections)} workspace views captured; "
                f"requested {args.workspace_views}."
            )
        workspace_board_poses: list[np.ndarray] = []
        for detection in workspace_detections:
            board_pose, error_px = estimate_board_pose(detection, workspace_intrinsics)
            if error_px <= 1.5:
                workspace_board_poses.append(board_pose)
        if len(workspace_board_poses) < max(3, args.workspace_views - 1):
            raise SystemExit("Too many workspace views failed the reprojection gate.")
        workspace_pose, workspace_metrics = solve_table_camera(
            workspace_board_poses, board
        )
        if workspace_metrics["translation_rms_mm"] > args.max_rms_mm:
            raise SystemExit(
                "Workspace-camera residual exceeds the session threshold; "
                "calibration not saved."
            )
        table_from_camera["workspace"] = {
            "pose": pose7_to_dict(workspace_pose),
            "metrics": workspace_metrics,
        }
    session = {
        "schema_version": 1,
        "kind": "handumi_session_calibration",
        "created_at": _now_iso(),
        "spatial_calibration_path": str(args.spatial),
        "spatial_calibration_sha256": calibration_hash(spatial),
        "board": board.to_dict(),
        "source_side": args.side,
        "table_from_quest": pose7_to_dict(table_from_quest),
        "table_from_camera": table_from_camera,
        "metrics": metrics,
    }
    write_yaml(args.output, session)
    log.info("Saved table session to %s (RMS %.2f mm).", args.output, metrics["translation_rms_mm"])


def cmd_verify(args: argparse.Namespace) -> None:
    session = load_yaml(args.session)
    spatial = _load_spatial(args.spatial)
    expected = session.get("spatial_calibration_sha256")
    actual = calibration_hash(spatial)
    if expected != actual:
        raise SystemExit("Session calibration references a different spatial calibration hash.")
    metrics = session.get("metrics") or {}
    print(f"session: {args.session}")
    print(f"spatial_sha256: {actual}")
    print(f"translation_rms_mm: {float(metrics.get('translation_rms_mm', float('nan'))):.3f}")
    print(f"rotation_rms_deg: {float(metrics.get('rotation_rms_deg', float('nan'))):.3f}")
    print("hash and schema: OK")
    if args.metadata_only:
        return

    board = CharucoBoardSpec.from_dict(spatial.get("board"))
    camera_name = f"{args.side}_wrist"
    intrinsics = _intrinsics(spatial, camera_name)
    mount = (spatial.get("controller_camera") or {}).get(args.side)
    if not isinstance(mount, dict):
        raise SystemExit(f"Missing {args.side} controller-camera mount in {args.spatial}.")
    controller_camera = pose7_to_mat(
        pose7_from_dict(mount["controller_from_camera"])
    ).astype(np.float64)
    table_quest = pose7_to_mat(pose7_from_dict(session["table_from_quest"])).astype(np.float64)
    board_table = pose7_to_mat(board_from_table_pose(board)).astype(np.float64)
    camera = _camera(args.rig_config, camera_name, args.fps, intrinsics.width, intrinsics.height)
    receiver = _connect_quest(args)
    camera.connect()
    try:
        detections, controller_poses = _capture(
            camera=camera,
            board=board,
            title=f"HandUMI session verification: {args.side}",
            requested_views=args.views,
            pair_at=_quest_pairer(receiver, args.side),
        )
    finally:
        camera.disconnect()
        receiver.stop()
    if len(detections) < 3:
        raise SystemExit("Need at least 3 verification views.")
    errors_mm: list[float] = []
    errors_deg: list[float] = []
    for controller_pose, detection in zip(controller_poses, detections, strict=True):
        camera_board = pose7_to_mat(estimate_board_pose(detection, intrinsics)[0])
        table_table = (
            table_quest
            @ pose7_to_mat(controller_pose)
            @ controller_camera
            @ camera_board
            @ board_table
        )
        errors_mm.append(float(np.linalg.norm(table_table[:3, 3])) * 1000.0)
        rvec, _ = cv2.Rodrigues(table_table[:3, :3])
        errors_deg.append(float(np.rad2deg(np.linalg.norm(rvec))))
    mean_mm = float(np.mean(errors_mm))
    mean_deg = float(np.mean(errors_deg))
    print(f"live_translation_mean_mm: {mean_mm:.3f}")
    print(f"live_rotation_mean_deg: {mean_deg:.3f}")
    if mean_mm > args.max_error_mm or mean_deg > args.max_error_deg:
        raise SystemExit("Live table verification failed; do not record this session.")
    print("live table verification: OK")


def _init_rerun_view(
    camera_names: list[str], board: CharucoBoardSpec, *, spawn: bool = True
):
    import rerun as rr
    import rerun.blueprint as rrb

    rr.init("handumi_spatial_calibration", spawn=spawn)
    rr.log("table", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    width, height = board.size_m
    outline = np.array(
        [
            [-width / 2, -height / 2, 0],
            [width / 2, -height / 2, 0],
            [width / 2, height / 2, 0],
            [-width / 2, height / 2, 0],
            [-width / 2, -height / 2, 0],
        ],
        dtype=np.float32,
    )
    rr.log(
        "table/charuco",
        rr.LineStrips3D([outline], colors=[[230, 230, 230]], radii=0.002),
        static=True,
    )
    rr.log(
        "table/axes",
        rr.Arrows3D(
            origins=[[0, 0, 0]] * 3,
            vectors=[[0.12, 0, 0], [0, 0.12, 0], [0, 0, 0.12]],
            colors=[[230, 70, 70], [70, 220, 100], [80, 140, 255]],
            radii=0.004,
            labels=["+X", "+Y", "+Z"],
        ),
        static=True,
    )

    camera_views = [
        rrb.Spatial2DView(
            origin=f"/table/cameras/{camera_name}", name=camera_name
        )
        for camera_name in camera_names
    ]
    right_column = (
        rrb.Vertical(*camera_views) if len(camera_views) > 1 else camera_views[0]
    )
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/table",
                name="calibrated_table",
                background=rrb.Background(color=[*BACKGROUND_COLOR, 255]),
            ),
            right_column,
            column_shares=[3, 2],
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed"),
    )
    rr.send_blueprint(blueprint, make_active=True, make_default=True)
    return rr


def _rectification_maps(
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = (intrinsics.width, intrinsics.height)
    rectified_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        intrinsics.matrix,
        intrinsics.distortion,
        size,
        np.eye(3),
        balance=0.7,
        new_size=size,
    )
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
        intrinsics.matrix,
        intrinsics.distortion,
        np.eye(3),
        rectified_matrix,
        size,
        cv2.CV_32FC1,
    )
    return rectified_matrix, map_x, map_y


def _log_camera_model(
    rr,
    name: str,
    intrinsics: CameraIntrinsics,
    color,
    *,
    image_matrix: np.ndarray | None = None,
) -> None:
    rr.log(
        f"table/cameras/{name}",
        rr.Pinhole(
            image_from_camera=(
                intrinsics.matrix if image_matrix is None else image_matrix
            ),
            resolution=[intrinsics.width, intrinsics.height],
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=0.12,
            color=color,
            line_width=0.002,
        ),
        static=True,
    )


def _log_camera_pose(rr, name: str, table_camera: np.ndarray, *, static: bool = False) -> None:
    rr.log(
        f"table/cameras/{name}",
        rr.Transform3D(
            translation=table_camera[:3, 3],
            mat3x3=table_camera[:3, :3],
            relation=rr.TransformRelation.ParentFromChild,
            axis_length=0.08,
        ),
        static=static,
    )


def cmd_visualize(args: argparse.Namespace) -> None:
    """Show calibrated cameras and controller motion in the table frame."""
    spatial = _load_spatial(args.spatial)
    session = load_yaml(args.session)
    if session.get("spatial_calibration_sha256") != calibration_hash(spatial):
        raise SystemExit("Session calibration references a different spatial calibration hash.")

    board = CharucoBoardSpec.from_dict(spatial.get("board"))
    table_quest = pose7_to_mat(pose7_from_dict(session["table_from_quest"])).astype(
        np.float64
    )
    camera_names = [
        name
        for name in ("left_wrist", "right_wrist", "workspace")
        if name in (spatial.get("cameras") or {})
    ]
    workspace_entry = (session.get("table_from_camera") or {}).get("workspace")
    if "workspace" in camera_names and not isinstance(workspace_entry, dict):
        log.warning(
            "Session has no fixed workspace-camera pose; showing wrist cameras only."
        )
        camera_names.remove("workspace")
    if not camera_names:
        raise SystemExit("Spatial calibration contains no calibrated cameras.")

    intrinsics = {name: _intrinsics(spatial, name) for name in camera_names}
    cameras: dict[str, OpenCVCameraDevice] = {}
    receiver = _connect_quest(args)
    try:
        for name in camera_names:
            camera_intrinsics = intrinsics[name]
            camera = _camera(
                args.rig_config,
                name,
                args.fps,
                camera_intrinsics.width,
                camera_intrinsics.height,
            )
            camera.connect()
            cameras[name] = camera

        rr = _init_rerun_view(camera_names, board)
        colors = {
            "left_wrist": LEFT_COLOR,
            "right_wrist": RIGHT_COLOR,
            "workspace": WORKSPACE_COLOR,
        }
        rectification = {
            name: _rectification_maps(intrinsics[name]) for name in camera_names
        }
        for name in camera_names:
            rectified_matrix, _, _ = rectification[name]
            _log_camera_model(
                rr,
                name,
                intrinsics[name],
                colors[name],
                image_matrix=rectified_matrix,
            )

        if "workspace" in cameras:
            assert isinstance(workspace_entry, dict)
            workspace_pose = pose7_to_mat(pose7_from_dict(workspace_entry["pose"]))
            _log_camera_pose(rr, "workspace", workspace_pose, static=True)

        controller_camera: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            name = f"{side}_wrist"
            if name not in cameras:
                continue
            mount = (spatial.get("controller_camera") or {}).get(side)
            if not isinstance(mount, dict):
                raise SystemExit(f"Missing {side} controller-camera mount.")
            controller_camera[side] = pose7_to_mat(
                pose7_from_dict(mount["controller_from_camera"])
            )

        trails = {
            "left": TrajectoryTrail(max(2, args.fps * 10)),
            "right": TrajectoryTrail(max(2, args.fps * 10)),
        }
        interval = 1.0 / args.fps
        log.info("Rerun calibration view started. Ctrl+C to stop.")
        while True:
            loop_start = time.perf_counter()
            for name, camera in cameras.items():
                sample = camera.sample_at()
                _, map_x, map_y = rectification[name]
                image = cv2.remap(
                    sample.image,
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                rr.log(
                    f"table/cameras/{name}",
                    rr.Image(image).compress(jpeg_quality=80),
                )

            aligned = receiver.aligned_at()
            if aligned is not None:
                for side, color in (("left", LEFT_COLOR), ("right", RIGHT_COLOR)):
                    if side not in controller_camera:
                        continue
                    state = getattr(aligned.frame, side)
                    if not state.tracked or not state.valid:
                        continue
                    controller = unity_pose_to_handumi(state.position, state.quaternion)
                    quest_controller = np.eye(4, dtype=np.float64)
                    quest_controller[:3, :3] = controller.as_matrix()[:3, :3]
                    quest_controller[:3, 3] = controller.position
                    table_controller = table_quest @ quest_controller
                    table_camera = table_controller @ controller_camera[side]
                    _log_camera_pose(rr, f"{side}_wrist", table_camera)
                    trails[side].append(table_controller[:3, 3])
                    rr.log(
                        f"table/tracking/{side}/controller",
                        rr.Points3D(
                            [table_controller[:3, 3]], colors=[color], radii=0.012
                        ),
                    )
                    points = trails[side].points()
                    if len(points) >= 2:
                        rr.log(
                            f"table/tracking/{side}/trail",
                            rr.LineStrips3D(
                                [points], colors=[color], radii=0.003
                            ),
                        )

            elapsed = time.perf_counter() - loop_start
            if (sleep := interval - elapsed) > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        log.info("Stopping Rerun calibration view.")
    finally:
        for camera in cameras.values():
            camera.disconnect()
        receiver.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig-config", type=Path, default=DEFAULT_RIG_CONFIG)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--quest-ip")
    parser.add_argument("--tcp-port", type=int)
    parser.add_argument("--sync-port", type=int)
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect-board", help="Check board detection and orientation.")
    inspect.add_argument("--camera", choices=("left_wrist", "right_wrist", "workspace"), default="left_wrist")
    inspect.add_argument("--width", type=int, default=640)
    inspect.add_argument("--height", type=int, default=480)
    inspect.set_defaults(func=cmd_inspect)

    intrinsics = sub.add_parser("intrinsics", help="Calibrate one fisheye camera.")
    intrinsics.add_argument("--camera", choices=("left_wrist", "right_wrist", "workspace"), required=True)
    intrinsics.add_argument("--views", type=int, default=30)
    intrinsics.add_argument("--width", type=int, default=640)
    intrinsics.add_argument("--height", type=int, default=480)
    intrinsics.add_argument("--max-mean-error-px", type=float, default=0.8)
    intrinsics.add_argument("--output", type=Path, default=DEFAULT_SPATIAL)
    intrinsics.set_defaults(func=cmd_intrinsics)

    mount = sub.add_parser("mount", help="Calibrate controller-to-wrist-camera extrinsics.")
    mount.add_argument("--side", choices=("left", "right"), required=True)
    mount.add_argument("--views", type=int, default=24)
    mount.add_argument("--max-rms-mm", type=float, default=2.0)
    mount.add_argument("--spatial", type=Path, default=DEFAULT_SPATIAL)
    mount.set_defaults(func=cmd_mount)

    session = sub.add_parser("session", help="Set the table frame for this Quest session.")
    session.add_argument("--side", choices=("left", "right"), required=True)
    session.add_argument("--views", type=int, default=10)
    session.add_argument("--workspace-views", type=int, default=5)
    session.add_argument(
        "--skip-workspace",
        action="store_true",
        help="Do not calibrate the fixed workspace camera for this session.",
    )
    session.add_argument("--max-rms-mm", type=float, default=3.0)
    session.add_argument("--spatial", type=Path, default=DEFAULT_SPATIAL)
    session.add_argument("--output", type=Path, default=DEFAULT_SESSION)
    session.set_defaults(func=cmd_session)

    verify = sub.add_parser("verify", help="Validate session/spatial identity and metrics.")
    verify.add_argument("--spatial", type=Path, default=DEFAULT_SPATIAL)
    verify.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    verify.add_argument("--side", choices=("left", "right"), default="left")
    verify.add_argument("--views", type=int, default=5)
    verify.add_argument("--max-error-mm", type=float, default=3.0)
    verify.add_argument("--max-error-deg", type=float, default=1.0)
    verify.add_argument("--metadata-only", action="store_true")
    verify.set_defaults(func=cmd_verify)

    visualize = sub.add_parser(
        "visualize", help="Open the calibrated three-camera Rerun view."
    )
    visualize.add_argument("--spatial", type=Path, default=DEFAULT_SPATIAL)
    visualize.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    visualize.set_defaults(func=cmd_visualize)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
