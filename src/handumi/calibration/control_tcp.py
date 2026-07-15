"""Controller-frame to physical HandUMI gripper TCP calibration helpers.

The important transform is:

    T_world_tcp = T_world_controller @ T_controller_tcp

`T_controller_tcp` is fixed by the 3D-printed mount. It is not the same as the
robot TCP frame in the URDF; this one corrects recorded controller poses so
trajectories represent the useful gripper point.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation

from handumi.dataset.raw import LEFT_POSE_SLICE, RIGHT_POSE_SLICE
from handumi.robots.utils import IDENTITY_POSE7, pose_mul, quat_normalize

DEFAULT_PARQUET = Path("pico_recording/data/chunk-000/file-000.parquet")
DEFAULT_DEVICE = "pico"
SUPPORTED_DEVICES = ("pico", "meta")
DEFAULT_CALIBRATION_DIR = Path("configs/calibration")
DEFAULT_CALIBRATION = DEFAULT_CALIBRATION_DIR / f"{DEFAULT_DEVICE}_controller_tcp.yaml"
STATE_COLUMN = "observation.state"
SIDES = ("left", "right")
CONTROLLER_TCP_METADATA_SCHEMA_VERSION = 2


def missing_dataset_message(path: Path = DEFAULT_PARQUET) -> str:
    return (
        f"Dataset not found: {path}. "
        "Download NONHUMAN-RESEARCH/pico_laptop_reach to pico_recording, "
        "or pass --parquet/--csv explicitly. See docs/datasets.md."
    )


@dataclass(frozen=True)
class PivotSolve:
    position: np.ndarray
    pivot_world: np.ndarray
    residuals: np.ndarray
    rms_error: float
    max_error: float
    condition: float
    num_samples: int


@dataclass(frozen=True)
class ControllerTcpCalibration:
    left: np.ndarray
    right: np.ndarray
    source: Path | None = None


def controller_tcp_calibration_from_metadata(
    metadata: dict[str, Any],
) -> ControllerTcpCalibration:
    """Load a self-contained controller->TCP snapshot from dataset metadata."""
    root = metadata.get("controller_to_gripper_tcp", metadata)
    if not isinstance(root, dict):
        raise ValueError("controller TCP metadata must be a mapping")
    try:
        left = _side_pose_from_mapping(root, "left")
        right = _side_pose_from_mapping(root, "right")
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError("invalid controller TCP calibration metadata") from exc
    return ControllerTcpCalibration(left=left, right=right, source=None)


def missing_calibration_message(path: Path = DEFAULT_CALIBRATION) -> str:
    return (
        f"Missing controller->UMI TCP calibration: {path}\n"
        "Run pivot calibration once for each side, for example:\n"
        "  uv run handumi-calibrate-tcp-offset --device pico pivot --side left -e EP_CAL_LEFT\n"
        "  uv run handumi-calibrate-tcp-offset --device pico pivot --side right -e EP_CAL_RIGHT\n"
        "For debug only, pass the explicit raw-controller flag in the caller."
    )


def calibration_path_for_device(device: str, root: Path = DEFAULT_CALIBRATION_DIR) -> Path:
    if device not in SUPPORTED_DEVICES:
        raise SystemExit(f"Invalid device {device!r}; use one of {SUPPORTED_DEVICES}")
    return root / f"{device}_controller_tcp.yaml"


def calibration_path_for_robot_device(
    robot: str,
    device: str,
    *,
    explicit_path: Path | None = None,
) -> tuple[Path, str]:
    """Resolve an explicit, robot-tool configured, then legacy device path."""
    if explicit_path is not None:
        return explicit_path, f"explicit {explicit_path}"

    from handumi.robots.registry import load_robot_config

    config = load_robot_config(robot)
    configured = config.controller_tcp_calibrations.get(device)
    if configured is not None:
        return configured, f"configured {robot}/{device}: {configured}"
    fallback = calibration_path_for_device(device)
    return fallback, f"legacy device fallback {device}: {fallback}"


def controller_tcp_calibration_sha256(path: Path) -> str:
    """Return the exact calibration-file fingerprint used for provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_identity_bound_controller_tcp_metadata(metadata: dict[str, Any]) -> bool:
    """Whether a dataset snapshot names the complete source tool assembly."""
    try:
        schema_version = int(metadata.get("schema_version", 0))
    except (TypeError, ValueError):
        return False
    required = (
        "source_robot",
        "source_gripper",
        "tracking_device",
        "controller_mount",
    )
    return (
        schema_version >= CONTROLLER_TCP_METADATA_SCHEMA_VERSION
        and metadata.get("applied_to_state") is not True
        and all(str(metadata.get(key, "")).strip() for key in required)
    )


def _as_pose7(value: Any) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float32).reshape(-1)
    if pose.shape[0] < 7:
        raise SystemExit("Expected pose7 value [x,y,z,qx,qy,qz,qw]")
    pose = pose[:7].copy()
    pose[3:] = quat_normalize(pose[3:])
    return pose


def _continuous_pose7(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float32).copy()
    for i in range(len(poses)):
        poses[i, 3:] = quat_normalize(poses[i, 3:])
        if i > 0 and float(np.dot(poses[i - 1, 3:], poses[i, 3:])) < 0:
            poses[i, 3:] *= -1
    return poses


def load_episode_poses(
    parquet: Path,
    episode: int,
    side: str,
    *,
    column: str | None = None,
) -> np.ndarray:
    if not parquet.exists():
        raise SystemExit(missing_dataset_message(parquet))
    df = pd.read_parquet(parquet)
    if side not in SIDES:
        raise SystemExit(f"Invalid side {side!r}; use left or right")
    column = column or STATE_COLUMN
    if "episode_index" not in df.columns:
        raise SystemExit(f"{parquet} has no episode_index column")
    if column not in df.columns:
        raise SystemExit(f"{parquet} has no column {column!r}")

    ep = df[df["episode_index"] == episode].copy()
    if ep.empty:
        available = sorted(int(x) for x in df["episode_index"].dropna().unique())
        raise SystemExit(f"Episode {episode} not found. Available episodes: {available}")
    sort_cols = [col for col in ("frame_index", "index") if col in ep.columns]
    if sort_cols:
        ep = ep.sort_values(sort_cols)
    pose_slice = LEFT_POSE_SLICE if side == "left" else RIGHT_POSE_SLICE
    poses = np.stack(
        [
            _as_pose7(np.asarray(value).reshape(-1)[pose_slice])
            if column == STATE_COLUMN
            else _as_pose7(value)
            for value in ep[column]
        ],
        axis=0,
    )
    return _continuous_pose7(poses)


def load_csv_poses(csv_path: Path, side: str | None = None) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if side is not None and "side" in df.columns:
        df = df[df["side"].astype(str).str.lower() == side].copy()
    required = ["x", "y", "z", "qx", "qy", "qz", "qw"]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise SystemExit(f"{csv_path} missing columns: {missing}")
    if df.empty:
        raise SystemExit(f"{csv_path} has no rows for side={side!r}")
    poses = df[required].to_numpy(np.float32)
    return _continuous_pose7(poses)


def solve_pivot_offset(controller_poses: np.ndarray) -> PivotSolve:
    """Solve `p_controller + R_controller @ t_controller_tcp = p_fixed_tcp`."""
    poses = _continuous_pose7(controller_poses)
    if len(poses) < 8:
        raise SystemExit(
            "Pivot calibration needs at least 8 frames; record 5-10 seconds if possible"
        )

    rotations = Rotation.from_quat(poses[:, 3:]).as_matrix().astype(np.float64)
    positions = poses[:, :3].astype(np.float64)
    rows = []
    rhs = []
    eye = np.eye(3)
    for rot, pos in zip(rotations, positions, strict=True):
        rows.append(np.concatenate([rot, -eye], axis=1))
        rhs.append(-pos)
    a = np.concatenate(rows, axis=0)
    b = np.concatenate(rhs, axis=0)
    solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    singular = np.linalg.svd(a, compute_uv=False)
    condition = float(singular[0] / max(singular[-1], 1e-12))

    offset = solution[:3].astype(np.float32)
    pivot = solution[3:].astype(np.float32)
    reconstructed = positions + np.einsum("nij,j->ni", rotations, offset.astype(np.float64))
    residuals = (reconstructed - pivot.astype(np.float64)).astype(np.float32)
    errors = np.linalg.norm(residuals, axis=1)
    return PivotSolve(
        position=offset,
        pivot_world=pivot,
        residuals=residuals,
        rms_error=float(np.sqrt(np.mean(errors * errors))),
        max_error=float(np.max(errors)),
        condition=condition,
        num_samples=len(poses),
    )


def solve_orientation_offset(
    controller_poses: np.ndarray,
    tcp_quat_world_xyzw: np.ndarray,
) -> np.ndarray:
    """Estimate `R_controller_tcp = inv(R_world_controller) @ R_world_tcp`."""
    poses = _continuous_pose7(controller_poses)
    desired = Rotation.from_quat(quat_normalize(tcp_quat_world_xyzw))
    controller = Rotation.from_quat(poses[:, 3:])
    offsets = controller.inv() * desired
    return quat_normalize(offsets.mean().as_quat().astype(np.float32))


def calibration_to_dict(
    *,
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, Any]:
    return {
        "calibration": {
            "frame_convention": "pose7=[x,y,z,qx,qy,qz,qw], meters, xyzw quaternion",
            "controller_to_gripper_tcp": {
                "left": {
                    "position": [float(x) for x in left[:3]],
                    "quaternion": [float(x) for x in quat_normalize(left[3:])],
                },
                "right": {
                    "position": [float(x) for x in right[:3]],
                    "quaternion": [float(x) for x in quat_normalize(right[3:])],
                },
            },
        }
    }


def _side_pose_from_mapping(mapping: dict[str, Any], side: str) -> np.ndarray:
    raw = mapping.get(side, {})
    pose = IDENTITY_POSE7.copy()
    if "position" in raw:
        pose[:3] = np.asarray(raw["position"], dtype=np.float32)
    if "quaternion" in raw:
        pose[3:] = quat_normalize(np.asarray(raw["quaternion"], dtype=np.float32))
    return pose


def load_controller_tcp_calibration(path: Path) -> ControllerTcpCalibration:
    if not path.exists():
        raise SystemExit(missing_calibration_message(path))
    data = yaml.safe_load(path.read_text()) or {}
    root = data.get("calibration", data)
    mapping = root.get("controller_to_gripper_tcp", root)
    left = _side_pose_from_mapping(mapping, "left")
    right = _side_pose_from_mapping(mapping, "right")
    return ControllerTcpCalibration(left=left, right=right, source=path)


def controller_tcp_calibration_metadata(
    path: Path,
    *,
    applied_to_state: bool,
    source_robot: str | None = None,
    source_gripper: str | None = None,
    tracking_device: str | None = None,
    controller_mount: str | None = None,
) -> dict[str, Any]:
    """Return a self-contained, fingerprinted robot-tool calibration record.

    Identity fields bind the transform to the physical assembly that produced
    the demonstration. They are optional only for legacy conversion paths.
    """
    calibration = load_controller_tcp_calibration(path)
    payload = calibration_to_dict(
        left=calibration.left,
        right=calibration.right,
    )["calibration"]
    metadata: dict[str, Any] = {
        "schema_version": CONTROLLER_TCP_METADATA_SCHEMA_VERSION,
        "source_path": str(path),
        "sha256": controller_tcp_calibration_sha256(path),
        "applied_to_state": applied_to_state,
        **payload,
    }
    identity = {
        "source_robot": source_robot,
        "source_gripper": source_gripper,
        "tracking_device": tracking_device,
        "controller_mount": controller_mount,
    }
    metadata.update({key: value for key, value in identity.items() if value})
    return metadata


def write_controller_tcp_calibration(
    path: Path,
    *,
    left: np.ndarray,
    right: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = calibration_to_dict(left=left, right=right)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def apply_controller_tcp_calibration(
    left_controller_pose: np.ndarray,
    right_controller_pose: np.ndarray,
    calibration: ControllerTcpCalibration,
) -> tuple[np.ndarray, np.ndarray]:
    left = np.stack([pose_mul(pose, calibration.left) for pose in left_controller_pose], axis=0)
    right = np.stack([pose_mul(pose, calibration.right) for pose in right_controller_pose], axis=0)
    return _continuous_pose7(left), _continuous_pose7(right)


def existing_or_identity(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.exists():
        current = load_controller_tcp_calibration(path)
        return current.left.copy(), current.right.copy()
    return IDENTITY_POSE7.copy(), IDENTITY_POSE7.copy()
