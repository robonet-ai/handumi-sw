"""ChArUco-based camera, controller-mount, and table-frame calibration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from handumi.robots.utils import mat_to_pose7, pose7_to_mat


SCHEMA_VERSION = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _array(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(shape)


def pose7_to_dict(pose: np.ndarray) -> dict[str, list[float]]:
    value = np.asarray(pose, dtype=np.float64).reshape(7)
    return {
        "translation_m": value[:3].tolist(),
        "quaternion_xyzw": value[3:].tolist(),
    }


def pose7_from_dict(data: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            _array(data["translation_m"], (3,)),
            _array(data["quaternion_xyzw"], (4,)),
        ]
    ).astype(np.float32)


def calibration_hash(data: dict[str, Any] | Path) -> str:
    """Return a stable SHA-256 over calibration contents."""
    if isinstance(data, Path):
        with data.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CharucoBoardSpec:
    board_id: str = "opencv_charuco_5x7_30mm_v1"
    squares_x: int = 5
    squares_y: int = 7
    dictionary: str = "DICT_5X5_100"
    square_length_m: float = 0.030
    marker_length_m: float = 0.015
    legacy_pattern: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CharucoBoardSpec":
        data = data or {}
        return cls(
            board_id=str(data.get("id", cls.board_id)),
            squares_x=int(data.get("squares_x", cls.squares_x)),
            squares_y=int(data.get("squares_y", cls.squares_y)),
            dictionary=str(data.get("dictionary", cls.dictionary)),
            square_length_m=float(data.get("square_length_m", cls.square_length_m)),
            marker_length_m=float(data.get("marker_length_m", cls.marker_length_m)),
            legacy_pattern=bool(data.get("legacy_pattern", cls.legacy_pattern)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.board_id,
            "squares_x": self.squares_x,
            "squares_y": self.squares_y,
            "dictionary": self.dictionary,
            "square_length_m": self.square_length_m,
            "marker_length_m": self.marker_length_m,
            "legacy_pattern": self.legacy_pattern,
        }

    def create(self) -> cv2.aruco.CharucoBoard:
        dictionary_id = getattr(cv2.aruco, self.dictionary, None)
        if dictionary_id is None:
            raise ValueError(f"Unknown ArUco dictionary: {self.dictionary}")
        board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length_m,
            self.marker_length_m,
            cv2.aruco.getPredefinedDictionary(dictionary_id),
        )
        board.setLegacyPattern(self.legacy_pattern)
        return board

    @property
    def size_m(self) -> tuple[float, float]:
        return (
            self.squares_x * self.square_length_m,
            self.squares_y * self.square_length_m,
        )


@dataclass(frozen=True)
class CameraIntrinsics:
    camera: str
    width: int
    height: int
    matrix: np.ndarray
    distortion: np.ndarray
    rms_px: float
    mean_error_px: float
    views: int
    model: str = "fisheye"

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera": self.camera,
            "resolution": [self.width, self.height],
            "model": self.model,
            "matrix": np.asarray(self.matrix).tolist(),
            "distortion": np.asarray(self.distortion).reshape(-1).tolist(),
            "rms_px": float(self.rms_px),
            "mean_error_px": float(self.mean_error_px),
            "views": int(self.views),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraIntrinsics":
        width, height = data["resolution"]
        return cls(
            camera=str(data["camera"]),
            width=int(width),
            height=int(height),
            model=str(data.get("model", "fisheye")),
            matrix=_array(data["matrix"], (3, 3)),
            distortion=np.asarray(data["distortion"], dtype=np.float64).reshape(-1, 1),
            rms_px=float(data["rms_px"]),
            mean_error_px=float(data.get("mean_error_px", data["rms_px"])),
            views=int(data["views"]),
        )


@dataclass(frozen=True)
class CharucoDetection:
    object_points: np.ndarray
    image_points: np.ndarray
    ids: np.ndarray
    marker_corners: tuple[np.ndarray, ...]
    marker_ids: np.ndarray | None

    @property
    def count(self) -> int:
        return int(len(self.ids))


def detect_charuco(
    image: np.ndarray,
    board_spec: CharucoBoardSpec,
    *,
    min_corners: int = 4,
) -> CharucoDetection | None:
    board = board_spec.create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    if charuco_ids is None or charuco_corners is None or len(charuco_ids) < min_corners:
        return None
    object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
    return CharucoDetection(
        object_points=np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3),
        image_points=np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2),
        ids=np.asarray(charuco_ids, dtype=np.int32).reshape(-1),
        marker_corners=tuple(marker_corners or ()),
        marker_ids=None if marker_ids is None else np.asarray(marker_ids, dtype=np.int32),
    )


def draw_detection(image: np.ndarray, detection: CharucoDetection | None) -> np.ndarray:
    preview = image.copy()
    if detection is None:
        return preview
    if detection.marker_corners:
        cv2.aruco.drawDetectedMarkers(preview, list(detection.marker_corners), detection.marker_ids)
    corners = detection.image_points.astype(np.float32)
    cv2.aruco.drawDetectedCornersCharuco(preview, corners, detection.ids.reshape(-1, 1))
    return preview


def calibrate_fisheye(
    camera: str,
    detections: Sequence[CharucoDetection],
    image_size: tuple[int, int],
) -> CameraIntrinsics:
    if len(detections) < 10:
        raise ValueError("At least 10 accepted ChArUco views are required.")
    object_points = [d.object_points for d in detections]
    image_points = [d.image_points for d in detections]
    matrix = np.eye(3, dtype=np.float64)
    matrix[0, 0] = matrix[1, 1] = max(image_size)
    matrix[0, 2] = image_size[0] / 2.0
    matrix[1, 2] = image_size[1] / 2.0
    distortion = np.zeros((4, 1), dtype=np.float64)
    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_CHECK_COND
        | cv2.fisheye.CALIB_FIX_SKEW
        | cv2.fisheye.CALIB_FIX_K3
        | cv2.fisheye.CALIB_FIX_K4
    )
    rms, matrix, distortion, rvecs, tvecs = cv2.fisheye.calibrate(
        object_points,
        image_points,
        image_size,
        matrix,
        distortion,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7),
    )
    errors: list[float] = []
    for detection, rvec, tvec in zip(detections, rvecs, tvecs, strict=True):
        projected, _ = cv2.fisheye.projectPoints(
            detection.object_points, rvec, tvec, matrix, distortion
        )
        errors.extend(
            np.linalg.norm(projected.reshape(-1, 2) - detection.image_points.reshape(-1, 2), axis=1)
        )
    return CameraIntrinsics(
        camera=camera,
        width=image_size[0],
        height=image_size[1],
        matrix=matrix,
        distortion=distortion,
        rms_px=float(rms),
        mean_error_px=float(np.mean(errors)),
        views=len(detections),
    )


def calibrate_pinhole(
    camera: str,
    detections: Sequence[CharucoDetection],
    image_size: tuple[int, int],
) -> CameraIntrinsics:
    if len(detections) < 10:
        raise ValueError("At least 10 accepted ChArUco views are required.")
    object_points = [d.object_points.reshape(-1, 3).astype(np.float32) for d in detections]
    image_points = [d.image_points.reshape(-1, 2).astype(np.float32) for d in detections]
    rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    errors: list[float] = []
    for detection, rvec, tvec in zip(detections, rvecs, tvecs, strict=True):
        projected, _ = cv2.projectPoints(
            detection.object_points, rvec, tvec, matrix, distortion
        )
        errors.extend(
            np.linalg.norm(
                projected.reshape(-1, 2) - detection.image_points.reshape(-1, 2),
                axis=1,
            )
        )
    return CameraIntrinsics(
        camera=camera,
        width=image_size[0],
        height=image_size[1],
        matrix=matrix,
        distortion=distortion,
        rms_px=float(rms),
        mean_error_px=float(np.mean(errors)),
        views=len(detections),
        model="pinhole",
    )


def estimate_board_pose(
    detection: CharucoDetection,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, float]:
    """Return ``T_camera_board`` and mean fisheye reprojection error."""
    if intrinsics.model == "fisheye":
        image_points = cv2.fisheye.undistortPoints(
            detection.image_points,
            intrinsics.matrix,
            intrinsics.distortion,
        )
        solve_matrix = np.eye(3)
        solve_distortion = np.zeros(4)
    elif intrinsics.model == "pinhole":
        image_points = detection.image_points
        solve_matrix = intrinsics.matrix
        solve_distortion = intrinsics.distortion
    else:
        raise ValueError(f"Unsupported camera model: {intrinsics.model}")
    ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        detection.object_points,
        image_points,
        solve_matrix,
        solve_distortion,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not ok or not rvecs:
        raise ValueError("Could not estimate ChArUco board pose.")
    candidates: list[tuple[float, np.ndarray]] = []
    for rvec, tvec in zip(rvecs, tvecs, strict=True):
        if float(np.asarray(tvec).reshape(3)[2]) <= 0.0:
            continue
        if intrinsics.model == "fisheye":
            projected, _ = cv2.fisheye.projectPoints(
                detection.object_points,
                rvec,
                tvec,
                intrinsics.matrix,
                intrinsics.distortion,
            )
        else:
            projected, _ = cv2.projectPoints(
                detection.object_points,
                rvec,
                tvec,
                intrinsics.matrix,
                intrinsics.distortion,
            )
        error = np.linalg.norm(
            projected.reshape(-1, 2) - detection.image_points.reshape(-1, 2),
            axis=1,
        )
        rotation, _ = cv2.Rodrigues(rvec)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = np.asarray(tvec).reshape(3)
        candidates.append((float(np.mean(error)), transform))
    if not candidates:
        raise ValueError("ChArUco pose is behind the camera.")
    error_px, transform = min(candidates, key=lambda candidate: candidate[0])
    return mat_to_pose7(transform), error_px


def _parameters_to_matrix(parameters: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
    matrix[:3, 3] = parameters[3:6]
    return matrix


def _matrix_to_parameters(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [Rotation.from_matrix(matrix[:3, :3]).as_rotvec(), matrix[:3, 3]]
    )


def _transform_residual(error: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [error[:3, 3], Rotation.from_matrix(error[:3, :3]).as_rotvec() * 0.1]
    )


def transform_errors(transforms: Sequence[np.ndarray]) -> tuple[float, float, float, float]:
    """Translation/rotation RMS and max about a robust mean transform."""
    mean = mean_transform(transforms)
    translation: list[float] = []
    rotation: list[float] = []
    for transform in transforms:
        delta = np.linalg.inv(mean) @ transform
        translation.append(float(np.linalg.norm(delta[:3, 3])))
        rotation.append(float(np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec())))
    return (
        float(np.sqrt(np.mean(np.square(translation)))) * 1000.0,
        float(np.max(translation)) * 1000.0,
        float(np.rad2deg(np.sqrt(np.mean(np.square(rotation))))),
        float(np.rad2deg(np.max(rotation))),
    )


def mean_transform(transforms: Sequence[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("At least one transform is required.")
    translations = np.stack([value[:3, 3] for value in transforms])
    rotations = Rotation.from_matrix(np.stack([value[:3, :3] for value in transforms]))
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rotations.mean().as_matrix()
    out[:3, 3] = np.median(translations, axis=0)
    return out


def solve_controller_camera(
    device_controller_poses: Sequence[np.ndarray],
    camera_board_poses: Sequence[np.ndarray],
) -> tuple[np.ndarray, dict[str, float]]:
    """Solve ``T_controller_camera`` from synchronized fixed-board views."""
    if (
        len(device_controller_poses) != len(camera_board_poses)
        or len(device_controller_poses) < 8
    ):
        raise ValueError("At least 8 paired controller/board views are required.")
    gripper = [pose7_to_mat(p).astype(np.float64) for p in device_controller_poses]
    target = [pose7_to_mat(p).astype(np.float64) for p in camera_board_poses]
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    methods = (
        cv2.CALIB_HAND_EYE_TSAI,
        cv2.CALIB_HAND_EYE_PARK,
        cv2.CALIB_HAND_EYE_HORAUD,
        cv2.CALIB_HAND_EYE_ANDREFF,
        cv2.CALIB_HAND_EYE_DANIILIDIS,
    )
    for method in methods:
        try:
            rotation, translation = cv2.calibrateHandEye(
                [g[:3, :3] for g in gripper],
                [g[:3, 3] for g in gripper],
                [t[:3, :3] for t in target],
                [t[:3, 3] for t in target],
                method=method,
            )
        except cv2.error:
            continue
        rotation = np.asarray(rotation, dtype=np.float64)
        translation = np.asarray(translation, dtype=np.float64).reshape(3)
        if (
            not np.all(np.isfinite(rotation))
            or not np.all(np.isfinite(translation))
            or np.linalg.det(rotation) <= 0.0
        ):
            continue
        x = np.eye(4, dtype=np.float64)
        x[:3, :3] = Rotation.from_matrix(rotation).as_matrix()
        x[:3, 3] = translation
        fixed_board = [g @ x @ c for g, c in zip(gripper, target, strict=True)]
        y = mean_transform(fixed_board)
        residual = np.concatenate(
            [_transform_residual(np.linalg.inv(y) @ value) for value in fixed_board]
        )
        candidates.append((float(np.mean(np.square(residual))), x, y))
    if not candidates:
        raise ValueError(
            "Hand-eye calibration produced no valid right-handed solution; "
            "capture more varied controller rotations."
        )
    _, x0, y0 = min(candidates, key=lambda candidate: candidate[0])

    initial = np.concatenate([_matrix_to_parameters(x0), _matrix_to_parameters(y0)])

    def residual(parameters: np.ndarray) -> np.ndarray:
        x = _parameters_to_matrix(parameters[:6])
        y = _parameters_to_matrix(parameters[6:])
        return np.concatenate(
            [_transform_residual(np.linalg.inv(y) @ g @ x @ c) for g, c in zip(gripper, target, strict=True)]
        )

    result = least_squares(residual, initial, loss="huber", f_scale=0.002, max_nfev=1000)
    x = _parameters_to_matrix(result.x[:6])
    fixed_board = [g @ x @ c for g, c in zip(gripper, target, strict=True)]
    rms_mm, max_mm, rms_deg, max_deg = transform_errors(fixed_board)
    return mat_to_pose7(x), {
        "views": float(len(gripper)),
        "translation_rms_mm": rms_mm,
        "translation_max_mm": max_mm,
        "rotation_rms_deg": rms_deg,
        "rotation_max_deg": max_deg,
    }


def board_from_table_pose(board_spec: CharucoBoardSpec) -> np.ndarray:
    """Return ``T_board_table`` for bottom edge (IDs 15/16) facing operator."""
    width, height = board_spec.size_m
    transform = np.eye(4, dtype=np.float64)
    # Printed board coordinates are +X right, +Y toward its bottom edge and
    # +Z into the paper. Table +Y is away and +Z is upward.
    transform[:3, :3] = np.diag([1.0, -1.0, -1.0])
    transform[:3, 3] = [width / 2.0, height / 2.0, 0.0]
    return mat_to_pose7(transform)


def solve_table_device(
    device_controller_poses: Sequence[np.ndarray],
    controller_camera_pose: np.ndarray,
    camera_board_poses: Sequence[np.ndarray],
    board_spec: CharucoBoardSpec,
) -> tuple[np.ndarray, dict[str, float]]:
    if (
        len(device_controller_poses) != len(camera_board_poses)
        or len(device_controller_poses) < 4
    ):
        raise ValueError("At least 4 paired session views are required.")
    controller_camera = pose7_to_mat(controller_camera_pose).astype(np.float64)
    device_board = [
        pose7_to_mat(g).astype(np.float64)
        @ controller_camera
        @ pose7_to_mat(c).astype(np.float64)
        for g, c in zip(device_controller_poses, camera_board_poses, strict=True)
    ]
    rms_mm, max_mm, rms_deg, max_deg = transform_errors(device_board)
    device_table = mean_transform(device_board) @ pose7_to_mat(
        board_from_table_pose(board_spec)
    )
    return mat_to_pose7(np.linalg.inv(device_table)), {
        "views": float(len(device_board)),
        "translation_rms_mm": rms_mm,
        "translation_max_mm": max_mm,
        "rotation_rms_deg": rms_deg,
        "rotation_max_deg": max_deg,
    }


def solve_table_quest(
    quest_controller_poses: Sequence[np.ndarray],
    controller_camera_pose: np.ndarray,
    camera_board_poses: Sequence[np.ndarray],
    board_spec: CharucoBoardSpec,
) -> tuple[np.ndarray, dict[str, float]]:
    """Backward-compatible alias for the device-agnostic table solver."""
    return solve_table_device(
        quest_controller_poses,
        controller_camera_pose,
        camera_board_poses,
        board_spec,
    )


def solve_table_camera(
    camera_board_poses: Sequence[np.ndarray],
    board_spec: CharucoBoardSpec,
) -> tuple[np.ndarray, dict[str, float]]:
    """Solve a fixed ``T_table_camera`` from views of the table board."""
    if len(camera_board_poses) < 3:
        raise ValueError("At least 3 workspace-camera board views are required.")
    table_board = np.linalg.inv(pose7_to_mat(board_from_table_pose(board_spec)))
    table_camera = [
        table_board @ np.linalg.inv(pose7_to_mat(camera_board_pose))
        for camera_board_pose in camera_board_poses
    ]
    rms_mm, max_mm, rms_deg, max_deg = transform_errors(table_camera)
    return mat_to_pose7(mean_transform(table_camera)), {
        "views": float(len(table_camera)),
        "translation_rms_mm": rms_mm,
        "translation_max_mm": max_mm,
        "rotation_rms_deg": rms_deg,
        "rotation_max_deg": max_deg,
    }


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return value


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def new_spatial_calibration(board: CharucoBoardSpec) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "handumi_spatial_calibration",
        "created_at": _now_iso(),
        "board": board.to_dict(),
        "cameras": {},
        "controller_camera": {},
    }


def session_calibration_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    data = load_yaml(path)
    if data.get("kind") != "handumi_session_calibration":
        raise ValueError(f"Not a HandUMI session calibration: {path}")
    spatial_path = Path(str(data.get("spatial_calibration_path", "")))
    if not spatial_path.exists():
        raise ValueError(f"Missing spatial calibration referenced by session: {spatial_path}")
    spatial = load_yaml(spatial_path)
    spatial_sha256 = calibration_hash(spatial)
    if spatial_sha256 != data.get("spatial_calibration_sha256"):
        raise ValueError("Session calibration does not match its spatial calibration file.")
    table_from_device = data.get("table_from_device") or data.get("table_from_quest")
    if not isinstance(table_from_device, dict):
        raise ValueError("Session calibration is missing table_from_device.")
    tracking_device = str(data.get("tracking_device") or "meta")
    return {
        "path": str(path),
        "sha256": calibration_hash(data),
        "configuration": data,
        "spatial_calibration_path": str(spatial_path),
        "spatial_calibration_sha256": spatial_sha256,
        "spatial_calibration": spatial,
        "created_at": data.get("created_at"),
        "board_id": (data.get("board") or {}).get("id"),
        "workspace_frame": "table",
        "tracking_device": tracking_device,
        "table_from_device": table_from_device,
        "table_from_quest": data.get("table_from_quest"),
        "metrics": data.get("metrics", {}),
    }


def session_table_from_device(path: Path) -> np.ndarray:
    data = load_yaml(path)
    if data.get("kind") != "handumi_session_calibration":
        raise ValueError(f"Not a HandUMI session calibration: {path}")
    table_from_device = data.get("table_from_device") or data.get("table_from_quest")
    if not isinstance(table_from_device, dict):
        raise ValueError("Session calibration is missing table_from_device.")
    return pose7_from_dict(table_from_device)


def session_table_from_quest(path: Path) -> np.ndarray:
    """Backward-compatible alias for legacy Quest session calibrations."""
    return session_table_from_device(path)


__all__ = [
    "CameraIntrinsics",
    "CharucoBoardSpec",
    "CharucoDetection",
    "board_from_table_pose",
    "calibrate_fisheye",
    "calibrate_pinhole",
    "calibration_hash",
    "detect_charuco",
    "draw_detection",
    "estimate_board_pose",
    "load_yaml",
    "new_spatial_calibration",
    "pose7_from_dict",
    "pose7_to_dict",
    "session_calibration_metadata",
    "session_table_from_device",
    "session_table_from_quest",
    "solve_controller_camera",
    "solve_table_camera",
    "solve_table_device",
    "solve_table_quest",
    "transform_errors",
    "write_yaml",
]
