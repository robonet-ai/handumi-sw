#!/usr/bin/env python3
"""Interactive ChArUco calibration for HandUMI cameras and VR tracking."""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from handumi.calibration.control_tcp import ControllerTcpCalibration
from handumi.calibration.spatial import (
    CameraIntrinsics,
    CharucoBoardSpec,
    CharucoDetection,
    board_from_table_pose,
    calibrate_fisheye,
    calibrate_pinhole,
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
    solve_table_device,
    write_yaml,
)
from handumi.cameras.opencv import OpenCVCameraDevice
from handumi.config import DEFAULT_RIG_CONFIG, load_rig_section
from handumi.robots.utils import IDENTITY_POSE7, pose7_to_mat
from handumi.tracking.base import TrackingProvider
from handumi.tracking.meta_quest import MetaQuestConfig, MetaQuestTrackingProvider
from handumi.tracking.pico import PicoTrackingProvider
from handumi.utils.trajectory import TrajectoryTrail
from handumi.visualization import BACKGROUND_COLOR, LEFT_COLOR, RIGHT_COLOR


log = logging.getLogger("handumi.calibrate_spatial")
DEFAULT_SPATIAL = Path("outputs/calibration/spatial.yaml")
DEFAULT_SESSION = Path("outputs/calibration/session.yaml")
MIN_CORNERS = 12
MAX_SYNC_ERROR_MS = 20.0
PICO_MAX_SYNC_ERROR_MS = 80.0
AUTO_CAPTURE_INTERVAL_S = 2.0
MIN_POSE_ROTATION_DEG = 8.0
STABLE_HOLD_S = 0.25
STABLE_TRANSLATION_M = 0.010
STABLE_ROTATION_DEG = 5.0
WORKSPACE_COLOR = (110, 180, 255)
INTRINSIC_VIEW_PROMPTS = (
    "centro frontal",
    "esquina superior izquierda",
    "esquina superior derecha",
    "esquina inferior izquierda",
    "esquina inferior derecha",
    "cerca e inclinado",
    "lejos e inclinado",
    "roll izquierda",
    "roll derecha",
    "inclinacion opuesta",
)
MOUNT_VIEW_PROMPTS = (
    "frontal",
    "roll izquierda",
    "roll derecha",
    "pitch adelante",
    "pitch atras",
    "yaw izquierda",
    "yaw derecha",
    "diagonal combinada",
)


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


def _load_session(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"Missing session calibration: {path}. Run "
            "'handumi calibrate spatial --device <meta|pico> session "
            "--side <calibrated-side>' first."
        )
    data = load_yaml(path)
    if data.get("kind") != "handumi_session_calibration":
        raise SystemExit(f"Not a HandUMI session calibration: {path}")
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


def _pose_is_distinct(pose: np.ndarray, accepted: list[np.ndarray]) -> bool:
    if not accepted:
        return True
    rotation = pose7_to_mat(pose)[:3, :3]
    for previous in accepted:
        previous_rotation = pose7_to_mat(previous)[:3, :3]
        angle = Rotation.from_matrix(previous_rotation.T @ rotation).magnitude()
        if np.rad2deg(angle) < MIN_POSE_ROTATION_DEG:
            return False
    return True


def _image_view_signature(detection: CharucoDetection, shape: tuple[int, ...]) -> np.ndarray:
    points = detection.image_points.reshape(-1, 2)
    height, width = shape[:2]
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = points.mean(axis=0)
    return np.array(
        [
            center[0] / width,
            center[1] / height,
            (high[0] - low[0]) / width,
            (high[1] - low[1]) / height,
        ]
    )


def _image_view_is_distinct(signature: np.ndarray, accepted: list[np.ndarray]) -> bool:
    return not accepted or all(np.linalg.norm(signature - previous) >= 0.05 for previous in accepted)


def _controller_is_stable(pose: np.ndarray, reference: np.ndarray) -> bool:
    current = pose7_to_mat(pose)
    anchor = pose7_to_mat(reference)
    translation_m = np.linalg.norm(current[:3, 3] - anchor[:3, 3])
    rotation_deg = np.rad2deg(
        Rotation.from_matrix(anchor[:3, :3].T @ current[:3, :3]).magnitude()
    )
    return bool(
        translation_m <= STABLE_TRANSLATION_M
        and rotation_deg <= STABLE_ROTATION_DEG
    )


def _capture(
    *,
    camera: OpenCVCameraDevice,
    board: CharucoBoardSpec,
    title: str,
    requested_views: int,
    pair_at: Callable[[int], tuple[np.ndarray, float] | None] | None = None,
    require_distinct: bool = True,
    detection_gate: Callable[[CharucoDetection], bool] | None = None,
    instructions: tuple[str, ...] = (),
    view_prompts: tuple[str, ...] = (),
    sync_label: str = "tracking",
    max_sync_ms: float = MAX_SYNC_ERROR_MS,
) -> tuple[list[CharucoDetection], list[np.ndarray]]:
    detections: list[CharucoDetection] = []
    poses: list[np.ndarray] = []
    image_signatures: list[np.ndarray] = []
    next_capture_at = time.monotonic() + AUTO_CAPTURE_INTERVAL_S
    stable_pose: np.ndarray | None = None
    stable_since = time.monotonic()
    _open_preview(title)
    try:
        while len(detections) < requested_views:
            sample = camera.sample_at()
            detection = detect_charuco(sample.image, board, min_corners=MIN_CORNERS)
            pair = None if pair_at is None else pair_at(sample.capture_time_ns)
            sync_ms = None if pair is None else pair[1]
            valid = detection is not None and (pair_at is None or pair is not None)
            signature = (
                None if detection is None else _image_view_signature(detection, sample.image.shape)
            )
            now = time.monotonic()
            stable = valid
            if pair_at is not None:
                current_pose = None if pair is None else pair[0]
                if current_pose is None:
                    stable_pose = None
                    stable_since = now
                    stable = False
                elif stable_pose is None or not _controller_is_stable(current_pose, stable_pose):
                    stable_pose = current_pose.copy()
                    stable_since = now
                    stable = False
                else:
                    stable = now - stable_since >= STABLE_HOLD_S
            distinct = True
            if require_distinct and valid:
                distinct = (
                    _pose_is_distinct(pair[0], poses)
                    if pair is not None
                    else signature is not None and _image_view_is_distinct(signature, image_signatures)
                )
            preview = draw_detection(sample.image, detection)
            corners = 0 if detection is None else detection.count
            remaining_s = max(0.0, next_capture_at - now)
            lines = [
                *instructions,
                *(
                    (f"Siguiente vista: {view_prompts[len(detections) % len(view_prompts)]}",)
                    if view_prompts
                    else ()
                ),
                f"Vistas {len(detections)}/{requested_views}  esquinas {corners}",
                f"Captura automatica en {remaining_s:.1f}s   Q salir",
            ]
            if valid and not distinct:
                lines.append("Cambia la posicion o rotacion")
            elif valid and not stable:
                lines.append("Sosten el HandUMI quieto")
            if pair_at is not None:
                sync_text = "n/a" if sync_ms is None else f"{sync_ms:.1f} ms"
                lines.insert(
                    1,
                    f"{sync_label}-camera sync {sync_text} "
                    f"(max {max_sync_ms:.0f} ms)",
                )
            cv2.imshow(title, _overlay(preview, lines, valid and stable and distinct))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if now >= next_capture_at:
                next_capture_at = now + AUTO_CAPTURE_INTERVAL_S
                if not valid or detection is None:
                    reason = (
                        "ChArUco detection failed"
                        if pair_at is None
                        else "board/tracking/synchronization gate failed"
                    )
                    log.warning("Auto capture skipped: %s.", reason)
                    continue
                if not stable:
                    log.warning("Auto capture skipped: hold the HandUMI still.")
                    continue
                if not distinct:
                    log.warning("Auto capture skipped: move to a more distinct view.")
                    continue
                if detection_gate is not None and not detection_gate(detection):
                    log.warning("Auto capture skipped: ChArUco pose reprojection failed.")
                    continue
                detections.append(detection)
                if pair is not None:
                    poses.append(pair[0])
                elif signature is not None:
                    image_signatures.append(signature)
                log.info("Accepted view %d/%d (%d corners).", len(detections), requested_views, corners)
    finally:
        cv2.destroyWindow(title)
    return detections, poses


def _reprojection_gate(intrinsics: CameraIntrinsics) -> Callable[[CharucoDetection], bool]:
    def valid(detection: CharucoDetection) -> bool:
        try:
            _, error_px = estimate_board_pose(detection, intrinsics)
        except (ValueError, cv2.error):
            return False
        return bool(np.isfinite(error_px) and error_px <= 1.5)

    return valid


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


def _identity_tcp_calibration() -> ControllerTcpCalibration:
    pose = IDENTITY_POSE7.astype(np.float32)
    return ControllerTcpCalibration(left=pose.copy(), right=pose.copy(), source=None)


def _sync_limit_ms(args: argparse.Namespace) -> float:
    if args.max_sync_ms is not None:
        return float(args.max_sync_ms)
    return PICO_MAX_SYNC_ERROR_MS if args.device == "pico" else MAX_SYNC_ERROR_MS


def _tracking_pairer(
    tracker: TrackingProvider,
    side: str,
    *,
    max_sync_ms: float,
) -> Callable[[int], tuple[np.ndarray, float] | None]:
    def pair_at(capture_time_ns: int) -> tuple[np.ndarray, float] | None:
        sample_at = getattr(tracker, "sample_at", None)
        sample = (
            sample_at(capture_time_ns)
            if sample_at is not None
            else tracker.latest()
        )
        tracked = bool(
            getattr(sample, f"{side}_device_tracked")
            and getattr(sample, f"{side}_pose_valid")
            and sample.streaming
        )
        if not tracked:
            return None
        aligned_time_ns = int(sample.aligned_time_ns or sample.pc_monotonic_ns)
        if aligned_time_ns <= 0:
            return None
        sync_ms = abs(aligned_time_ns - capture_time_ns) / 1e6
        if sync_ms > max_sync_ms:
            return None
        pose = getattr(sample, f"{side}_device_controller_pose")
        return np.asarray(pose, dtype=np.float32).copy(), sync_ms

    return pair_at


def _connect_tracker(args: argparse.Namespace) -> TrackingProvider:
    calibration = _identity_tcp_calibration()
    if args.device == "pico":
        transport = "wifi" if args.pico_wifi else "adb"
        tracker = PicoTrackingProvider(
            calibration=calibration,
            mode=args.pico_mode,
            transport=transport,
            skip_adb_check=args.skip_adb_check,
        )
        log.info("Connecting to PICO through XRoboToolkit (%s) ...", transport)
    else:
        base = MetaQuestConfig.from_yaml(args.rig_config)
        config = MetaQuestConfig(
            quest_ip=args.quest_ip or base.quest_ip,
            tcp_port=args.tcp_port or base.tcp_port,
            sync_port=args.sync_port or base.sync_port,
            connect_retry_s=base.connect_retry_s,
            frame_stale_timeout_s=base.frame_stale_timeout_s,
        )
        tracker = MetaQuestTrackingProvider(
            config=config,
            calibration=calibration,
            reset_workspace_on_x=False,
        )
        log.info("Connecting to Quest at %s:%d ...", config.quest_ip, config.tcp_port)
    tracker.start()
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if tracker.latest().streaming:
            return tracker
        time.sleep(0.1)
    tracker.stop()
    raise SystemExit(f"{args.device} tracking did not start streaming within 15 seconds.")


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
            require_distinct=False,
            instructions=(
                "Ponte frente al borde inferior del ChArUco",
                "IDs 15 y 16 deben quedar hacia ti",
            ),
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
            instructions=(
                "Ponte frente al borde inferior del ChArUco",
                "Mueve la camara: centro, bordes, distancia e inclinacion",
            ),
            view_prompts=INTRINSIC_VIEW_PROMPTS,
        )
    finally:
        camera.disconnect()
    if len(detections) < args.views:
        raise SystemExit(f"Only {len(detections)} views captured; requested {args.views}.")
    calibrate = calibrate_pinhole if args.camera == "workspace" else calibrate_fisheye
    intrinsics = calibrate(args.camera, detections, (args.width, args.height))
    if intrinsics.mean_error_px > args.max_mean_error_px:
        raise SystemExit(
            f"Mean reprojection error {intrinsics.mean_error_px:.3f}px exceeds "
            f"{args.max_mean_error_px:.3f}px; calibration not saved."
        )
    _, map_x, map_y = _rectification_maps(intrinsics)
    valid_coverage = np.mean(
        (map_x >= 0)
        & (map_x < intrinsics.width)
        & (map_y >= 0)
        & (map_y < intrinsics.height)
    )
    if valid_coverage < 0.80:
        raise SystemExit(
            f"Rectification coverage {valid_coverage:.1%} is below 80%; "
            "capture more varied views. Calibration not saved."
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
    tracker = _connect_tracker(args)
    max_sync_ms = _sync_limit_ms(args)
    camera.connect()
    try:
        detections, controller_poses = _capture(
            camera=camera,
            board=board,
            title=f"HandUMI {args.side} controller-camera mount",
            requested_views=args.views,
            pair_at=_tracking_pairer(
                tracker,
                args.side,
                max_sync_ms=max_sync_ms,
            ),
            detection_gate=_reprojection_gate(intrinsics),
            instructions=(
                f"Tablero fijo: toma el HandUMI {args.side}",
                "Muevelo en roll, pitch y yaw entre capturas",
            ),
            view_prompts=MOUNT_VIEW_PROMPTS,
            sync_label=args.device,
            max_sync_ms=max_sync_ms,
        )
    finally:
        camera.disconnect()
        tracker.stop()
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


def _calibrate_workspace(
    args: argparse.Namespace,
    spatial: dict,
    board: CharucoBoardSpec,
    *,
    views: int,
    max_rms_mm: float,
) -> dict:
    intrinsics = _intrinsics(spatial, "workspace")
    camera = _camera(
        args.rig_config,
        "workspace",
        args.fps,
        intrinsics.width,
        intrinsics.height,
    )
    camera.connect()
    try:
        detections, _ = _capture(
            camera=camera,
            board=board,
            title="HandUMI fixed workspace camera",
            requested_views=views,
            require_distinct=False,
            detection_gate=_reprojection_gate(intrinsics),
            instructions=(
                "No muevas el tablero ni la camara workspace",
                "Espera a que termine el contador",
            ),
        )
    finally:
        camera.disconnect()
    if len(detections) < views:
        raise SystemExit(f"Only {len(detections)} workspace views captured; requested {views}.")
    board_poses: list[np.ndarray] = []
    for detection in detections:
        board_pose, error_px = estimate_board_pose(detection, intrinsics)
        if error_px <= 1.5:
            board_poses.append(board_pose)
    if len(board_poses) < max(3, views - 1):
        raise SystemExit("Too many workspace views failed the reprojection gate.")
    pose, metrics = solve_table_camera(board_poses, board)
    if metrics["translation_rms_mm"] > max_rms_mm:
        raise SystemExit(
            f"Workspace-camera residual {metrics['translation_rms_mm']:.2f} mm "
            f"exceeds {max_rms_mm:.2f} mm; tracking session remains saved."
        )
    return {"pose": pose7_to_dict(pose), "metrics": metrics}


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
    tracker = _connect_tracker(args)
    max_sync_ms = _sync_limit_ms(args)
    camera.connect()
    try:
        detections, controller_poses = _capture(
            camera=camera,
            board=board,
            title=f"HandUMI table session: {args.side}",
            requested_views=args.views,
            pair_at=_tracking_pairer(
                tracker,
                args.side,
                max_sync_ms=max_sync_ms,
            ),
            detection_gate=_reprojection_gate(intrinsics),
            instructions=(
                f"{args.device} fijo; ponte frente al borde inferior",
                f"Toma el HandUMI {args.side} y varia su orientacion",
            ),
            view_prompts=MOUNT_VIEW_PROMPTS,
            sync_label=args.device,
            max_sync_ms=max_sync_ms,
        )
    finally:
        camera.disconnect()
        tracker.stop()
    if len(detections) < args.views:
        raise SystemExit(f"Only {len(detections)} views captured; requested {args.views}.")
    controller_poses, board_poses = _board_poses_with_gate(
        detections, controller_poses, intrinsics
    )
    if len(board_poses) < max(4, args.views - 1):
        raise SystemExit("Too many views failed the reprojection gate.")
    table_from_device, metrics = solve_table_device(
        controller_poses, controller_camera, board_poses, board
    )
    if metrics["translation_rms_mm"] > args.max_rms_mm:
        raise SystemExit(
            f"Session residual {metrics['translation_rms_mm']:.2f} mm exceeds "
            f"{args.max_rms_mm:.2f} mm; calibration not saved."
        )
    session = {
        "schema_version": 2,
        "kind": "handumi_session_calibration",
        "created_at": _now_iso(),
        "spatial_calibration_path": str(args.spatial),
        "spatial_calibration_sha256": calibration_hash(spatial),
        "board": board.to_dict(),
        "tracking_device": args.device,
        "source_side": args.side,
        "table_from_device": pose7_to_dict(table_from_device),
        "table_from_camera": {},
        "metrics": metrics,
    }
    if args.device == "meta":
        session["table_from_quest"] = pose7_to_dict(table_from_device)
    write_yaml(args.output, session)
    log.info(
        "Saved %s table session to %s (RMS %.2f mm).",
        args.device,
        args.output,
        metrics["translation_rms_mm"],
    )
    if not args.skip_workspace:
        session["table_from_camera"]["workspace"] = _calibrate_workspace(
            args,
            spatial,
            board,
            views=args.workspace_views,
            max_rms_mm=args.max_rms_mm,
        )
        write_yaml(args.output, session)
        log.info("Saved workspace-camera calibration to %s.", args.output)


def cmd_workspace(args: argparse.Namespace) -> None:
    spatial = _load_spatial(args.spatial)
    session = _load_session(args.session)
    if session.get("spatial_calibration_sha256") != calibration_hash(spatial):
        raise SystemExit("Session calibration references a different spatial calibration hash.")
    board = CharucoBoardSpec.from_dict(spatial.get("board"))
    session.setdefault("table_from_camera", {})["workspace"] = _calibrate_workspace(
        args,
        spatial,
        board,
        views=args.views,
        max_rms_mm=args.max_rms_mm,
    )
    write_yaml(args.session, session)
    log.info("Saved workspace-camera calibration to %s.", args.session)


def cmd_verify(args: argparse.Namespace) -> None:
    session = _load_session(args.session)
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
    table_from_device = session.get("table_from_device") or session.get("table_from_quest")
    if not isinstance(table_from_device, dict):
        raise SystemExit("Session calibration is missing table_from_device.")
    if session.get("tracking_device") not in (None, args.device):
        raise SystemExit(
            f"Session calibration is for {session.get('tracking_device')}, "
            f"but --device {args.device} was selected."
        )
    table_device = pose7_to_mat(pose7_from_dict(table_from_device)).astype(np.float64)
    board_table = pose7_to_mat(board_from_table_pose(board)).astype(np.float64)
    camera = _camera(args.rig_config, camera_name, args.fps, intrinsics.width, intrinsics.height)
    tracker = _connect_tracker(args)
    max_sync_ms = _sync_limit_ms(args)
    camera.connect()
    try:
        detections, controller_poses = _capture(
            camera=camera,
            board=board,
            title=f"HandUMI session verification: {args.side}",
            requested_views=args.views,
            pair_at=_tracking_pairer(
                tracker,
                args.side,
                max_sync_ms=max_sync_ms,
            ),
            detection_gate=_reprojection_gate(intrinsics),
            instructions=(
                f"Verificacion: toma el HandUMI {args.side}",
                "Tablero fijo; varia la orientacion entre capturas",
            ),
            sync_label=args.device,
            max_sync_ms=max_sync_ms,
        )
    finally:
        camera.disconnect()
        tracker.stop()
    if len(detections) < 3:
        raise SystemExit("Need at least 3 verification views.")
    errors_mm: list[float] = []
    errors_deg: list[float] = []
    for controller_pose, detection in zip(controller_poses, detections, strict=True):
        camera_board = pose7_to_mat(estimate_board_pose(detection, intrinsics)[0])
        table_table = (
            table_device
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
    if intrinsics.model == "fisheye":
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
    elif intrinsics.model == "pinhole":
        rectified_matrix, _ = cv2.getOptimalNewCameraMatrix(
            intrinsics.matrix,
            intrinsics.distortion,
            size,
            0.7,
            size,
        )
        map_x, map_y = cv2.initUndistortRectifyMap(
            intrinsics.matrix,
            intrinsics.distortion,
            None,
            rectified_matrix,
            size,
            cv2.CV_32FC1,
        )
    else:
        raise ValueError(f"Unsupported camera model: {intrinsics.model}")
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
    session = _load_session(args.session)
    if session.get("spatial_calibration_sha256") != calibration_hash(spatial):
        raise SystemExit("Session calibration references a different spatial calibration hash.")

    board = CharucoBoardSpec.from_dict(spatial.get("board"))
    table_from_device = session.get("table_from_device") or session.get("table_from_quest")
    if not isinstance(table_from_device, dict):
        raise SystemExit("Session calibration is missing table_from_device.")
    if session.get("tracking_device") not in (None, args.device):
        raise SystemExit(
            f"Session calibration is for {session.get('tracking_device')}, "
            f"but --device {args.device} was selected."
        )
    table_device = pose7_to_mat(pose7_from_dict(table_from_device)).astype(np.float64)
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
    tracker = _connect_tracker(args)
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

            sample = tracker.latest()
            if sample.streaming:
                for side, color in (("left", LEFT_COLOR), ("right", RIGHT_COLOR)):
                    if side not in controller_camera:
                        continue
                    if not (
                        getattr(sample, f"{side}_device_tracked")
                        and getattr(sample, f"{side}_pose_valid")
                    ):
                        continue
                    device_controller = pose7_to_mat(
                        getattr(sample, f"{side}_device_controller_pose")
                    )
                    table_controller = table_device @ device_controller
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
        tracker.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig-config", type=Path, default=DEFAULT_RIG_CONFIG)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", choices=("meta", "pico"), default="meta")
    parser.add_argument("--quest-ip")
    parser.add_argument("--tcp-port", type=int)
    parser.add_argument("--sync-port", type=int)
    parser.add_argument(
        "--pico-mode",
        choices=("mandos", "object", "whole-body"),
        default="mandos",
    )
    pico_transport = parser.add_mutually_exclusive_group()
    pico_transport.add_argument("--pico-adb", action="store_true")
    pico_transport.add_argument("--pico-wifi", action="store_true")
    parser.add_argument("--skip-adb-check", action="store_true")
    parser.add_argument(
        "--max-sync-ms",
        type=float,
        default=None,
        help=(
            "Override camera/tracking timestamp tolerance. Defaults to "
            f"{MAX_SYNC_ERROR_MS:.0f} ms for Meta and "
            f"{PICO_MAX_SYNC_ERROR_MS:.0f} ms for PICO."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect-board", help="Check board detection and orientation.")
    inspect.add_argument("--camera", choices=("left_wrist", "right_wrist", "workspace"), default="left_wrist")
    inspect.add_argument("--width", type=int, default=640)
    inspect.add_argument("--height", type=int, default=480)
    inspect.set_defaults(func=cmd_inspect)

    intrinsics = sub.add_parser("intrinsics", help="Calibrate one camera.")
    intrinsics.add_argument("--camera", choices=("left_wrist", "right_wrist", "workspace"), required=True)
    intrinsics.add_argument("--views", type=int, default=10)
    intrinsics.add_argument("--width", type=int, default=640)
    intrinsics.add_argument("--height", type=int, default=480)
    intrinsics.add_argument("--max-mean-error-px", type=float, default=0.8)
    intrinsics.add_argument("--output", type=Path, default=DEFAULT_SPATIAL)
    intrinsics.set_defaults(func=cmd_intrinsics)

    mount = sub.add_parser("mount", help="Calibrate controller-to-wrist-camera extrinsics.")
    mount.add_argument("--side", choices=("left", "right"), required=True)
    mount.add_argument("--views", type=int, default=8)
    mount.add_argument("--max-rms-mm", type=float, default=8.0)
    mount.add_argument("--spatial", type=Path, default=DEFAULT_SPATIAL)
    mount.set_defaults(func=cmd_mount)

    session = sub.add_parser("session", help="Set the table frame for this tracking session.")
    session.add_argument("--side", choices=("left", "right"), required=True)
    session.add_argument("--views", type=int, default=5)
    session.add_argument("--workspace-views", type=int, default=3)
    session.add_argument(
        "--skip-workspace",
        action="store_true",
        help="Do not calibrate the fixed workspace camera for this session.",
    )
    session.add_argument("--max-rms-mm", type=float, default=8.0)
    session.add_argument("--spatial", type=Path, default=DEFAULT_SPATIAL)
    session.add_argument("--output", type=Path, default=DEFAULT_SESSION)
    session.set_defaults(func=cmd_session)

    workspace = sub.add_parser(
        "workspace", help="Calibrate only the fixed workspace camera for a saved session."
    )
    workspace.add_argument("--views", type=int, default=3)
    workspace.add_argument("--max-rms-mm", type=float, default=8.0)
    workspace.add_argument("--spatial", type=Path, default=DEFAULT_SPATIAL)
    workspace.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    workspace.set_defaults(func=cmd_workspace)

    verify = sub.add_parser("verify", help="Validate session/spatial identity and metrics.")
    verify.add_argument("--spatial", type=Path, default=DEFAULT_SPATIAL)
    verify.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    verify.add_argument("--side", choices=("left", "right"), default="left")
    verify.add_argument("--views", type=int, default=5)
    verify.add_argument("--max-error-mm", type=float, default=8.0)
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
