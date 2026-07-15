"""USB camera setup helpers and frame collection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from handumi.cameras.base import CameraDevice, CameraSample
from handumi.cameras.opencv import OpenCVCameraDevice
from handumi.config import load_rig_section

log = logging.getLogger("handumi.record")

CameraSpec = dict[str, Any]


def build_camera_specs(
    cam_ids: list[int | str],
    *,
    camera_names: Sequence[str] | None = None,
    laptop_camera: bool,
    laptop_cam_id: int,
    laptop_cam_name: str,
) -> tuple[list[CameraSpec], str | None]:
    if camera_names is None:
        names = ["left_wrist", "right_wrist"]
        names.extend(f"cam_{i}" for i in range(2, len(cam_ids)))
    else:
        names = list(camera_names)
    if len(names) != len(cam_ids):
        raise ValueError(
            f"Expected {len(names)} camera IDs for {names}, got {len(cam_ids)}."
        )
    specs = []
    for i, cam_id in enumerate(cam_ids):
        name = names[i] if i < len(names) else f"cam_{i}"
        specs.append({"id": cam_id, "name": name, "is_laptop": False})
    resolved_laptop_name = laptop_cam_name if laptop_camera else None
    if laptop_camera:
        for spec in specs:
            if spec["name"] == laptop_cam_name:
                spec["is_laptop"] = True
                spec["id"] = laptop_cam_id
                break
        else:
            specs.append(
                {"id": laptop_cam_id, "name": laptop_cam_name, "is_laptop": True}
            )
    return specs, resolved_laptop_name


def resolve_camera_ids(
    cam_ids: list[int | str] | None,
    rig_config: Path,
    *,
    camera_names: Sequence[str] | None = None,
) -> list[int | str]:
    names = list(camera_names or ("left_wrist", "right_wrist"))
    if cam_ids is not None:
        if camera_names is not None and len(cam_ids) != len(names):
            raise ValueError(
                f"Expected {len(names)} --cam-ids values for {names}, got {len(cam_ids)}."
            )
        return cam_ids
    defaults = {"left_wrist": 0, "right_wrist": 2, "workspace": 4}
    data = load_rig_section(rig_config, "cameras")
    return [
        _read_camera_value(data, name, defaults.get(name, 0))
        for name in names
    ]


def connect_cameras(
    camera_specs: list[CameraSpec],
    *,
    fps: int,
    width: int,
    height: int,
    zero_non_laptop: bool,
    backend: str = "opencv",
) -> list[CameraDevice | None]:
    cameras: list[CameraDevice | None] = []
    for spec in camera_specs:
        cam_id = spec["id"]
        name = spec["name"]
        should_zero = zero_non_laptop and not spec["is_laptop"]
        if should_zero:
            cameras.append(None)
            log.info("Camera '%s' will be zero-filled.", name)
            continue

        cam = _make_camera(
            backend,
            index_or_path=cam_id,
            fps=fps,
            width=width,
            height=height,
        )
        cam.connect()
        cameras.append(cam)
        label = " laptop overlay" if spec["is_laptop"] else ""
        log.info("Camera '%s' (index %s) connected.%s", name, cam_id, label)
    return cameras


def read_camera_frames(
    cameras: list[CameraDevice | None],
    cam_names: list[str],
    *,
    width: int,
    height: int,
) -> dict:
    frames: dict = {}
    for cam, name in zip(cameras, cam_names):
        frame = (
            np.zeros((height, width, 3), dtype=np.uint8)
            if cam is None
            else cam.async_read()
        )
        frames[f"observation.images.{name}"] = frame
    return frames


def read_camera_samples(
    cameras: list[CameraDevice | None],
    cam_names: list[str],
    *,
    target_time_ns: int,
    record_time_ns: int,
    width: int,
    height: int,
    stale_timeout_s: float,
    max_sync_skew_s: float,
) -> tuple[dict, dict[str, bool]]:
    """Read camera frames nearest one clock target plus per-camera diagnostics."""
    frame: dict = {}
    health: dict[str, bool] = {}
    stale_timeout_ns = int(stale_timeout_s * 1e9)
    max_sync_skew_ns = int(max_sync_skew_s * 1e9)

    for camera, name in zip(cameras, cam_names):
        prefix = f"observation.camera.{name}"
        enabled = camera is not None
        try:
            sample = (
                CameraSample(
                    image=np.zeros((height, width, 3), dtype=np.uint8),
                    capture_time_ns=target_time_ns,
                    sequence=0,
                )
                if camera is None
                else camera.sample_at(target_time_ns)
            )
        except Exception as exc:
            log.debug("Camera '%s' read failed: %s", name, exc)
            sample = CameraSample(
                image=np.zeros((height, width, 3), dtype=np.uint8),
                capture_time_ns=0,
                sequence=0,
            )

        age_ns = (
            max(0, record_time_ns - sample.capture_time_ns)
            if sample.capture_time_ns
            else 2**63 - 1
        )
        sync_error_ns = (
            abs(sample.capture_time_ns - target_time_ns)
            if sample.capture_time_ns
            else 2**63 - 1
        )
        healthy = bool(
            not enabled
            or (
                sample.capture_time_ns > 0
                and age_ns <= stale_timeout_ns
                and sync_error_ns <= max_sync_skew_ns
            )
        )
        health[f"camera.{name}"] = healthy
        frame[f"observation.images.{name}"] = sample.image
        frame[f"{prefix}.healthy"] = _scalar_int(healthy if enabled else False)
        frame[f"{prefix}.sample_time_ns"] = _scalar_int(sample.capture_time_ns)
        frame[f"{prefix}.sequence"] = _scalar_int(sample.sequence)
    return frame, health


def disconnect_cameras(cameras: list[CameraDevice | None]) -> None:
    for cam in cameras:
        if cam is None:
            continue
        try:
            cam.disconnect()
        except Exception:
            pass


def _make_camera(
    backend: str,
    *,
    index_or_path: int | str,
    fps: int,
    width: int,
    height: int,
) -> CameraDevice:
    normalized = backend.lower().replace("_", "-")
    if normalized in {"opencv", "cv2"}:
        return OpenCVCameraDevice(
            index_or_path=index_or_path,
            fps=fps,
            width=width,
            height=height,
        )
    raise ValueError(f"Unsupported camera backend {backend!r}.")


def _read_camera_value(data: dict[str, Any], key: str, default: int) -> int | str:
    section = data.get(key) or {}
    value = section.get("index_or_path", default)
    if isinstance(value, int):
        return value
    text = str(value)
    return int(text) if text.isdigit() else text


def _scalar_int(value: int | bool) -> np.ndarray:
    return np.array([int(value)], dtype=np.int64)
