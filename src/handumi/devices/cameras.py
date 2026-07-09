"""Camera device contracts, USB setup, and preview helpers for HandUMI recording."""

from __future__ import annotations

import logging
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

log = logging.getLogger("handumi.record")
FONT = cv2.FONT_HERSHEY_SIMPLEX


class CameraDevice(ABC):
    """Minimal camera interface used by HandUMI recorders."""

    @abstractmethod
    def connect(self) -> None:
        """Open the camera stream."""

    @abstractmethod
    def async_read(self) -> np.ndarray:
        """Return the latest RGB frame."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the camera stream."""


def build_camera_specs(
    cam_ids: list[int | str],
    *,
    laptop_camera: bool,
    laptop_cam_id: int,
    laptop_cam_name: str,
) -> tuple[list[dict[str, Any]], str | None]:
    names = ["left_wrist", "right_wrist"]
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
    camera_config: Path,
) -> list[int | str]:
    if cam_ids is not None:
        return cam_ids
    if not camera_config.exists():
        return [0, 2]
    with camera_config.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return [
        _read_camera_value(data, "left_wrist", 0),
        _read_camera_value(data, "right_wrist", 2),
    ]


def connect_cameras(
    camera_specs: list[dict[str, Any]],
    *,
    fps: int,
    width: int,
    height: int,
    zero_non_laptop: bool,
) -> list:
    from lerobot.cameras.opencv import OpenCVCamera
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    cameras: list = []
    for spec in camera_specs:
        cam_id = spec["id"]
        name = spec["name"]
        should_zero = zero_non_laptop and not spec["is_laptop"]
        if should_zero:
            cameras.append(None)
            log.info(f"Camera '{name}' will be zero-filled.")
        else:
            cfg = OpenCVCameraConfig(
                index_or_path=cam_id,
                fps=fps,
                width=width,
                height=height,
            )
            cam = OpenCVCamera(cfg)
            cam.connect()
            cameras.append(cam)
            label = " laptop overlay" if spec["is_laptop"] else ""
            log.info(f"Camera '{name}' (index {cam_id}) connected.{label}")
    return cameras


def read_camera_frames(
    cameras: list,
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


def disconnect_cameras(cameras: list) -> None:
    for cam in cameras:
        try:
            cam.disconnect()
        except Exception:
            pass


def _read_camera_value(data: dict[str, Any], key: str, default: int) -> int | str:
    section = data.get(key) or {}
    value = section.get("index_or_path", default)
    if isinstance(value, int):
        return value
    text = str(value)
    return int(text) if text.isdigit() else text


def _text(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    scale: float = 0.5,
    color: tuple[int, int, int] = (235, 235, 235),
    thick: int = 1,
) -> None:
    x, y = pos
    cv2.putText(img, text, (x + 1, y + 1), FONT, scale, (0, 0, 0), thick + 2)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick)


def _panel(img: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], alpha: float = 0.55) -> None:
    overlay = img.copy()
    cv2.rectangle(overlay, p1, p2, (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


def _clock(seconds: float) -> str:
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes:02d}:{rem:04.1f}"


def _reach_color(ratio: float) -> tuple[tuple[int, int, int], str]:
    if ratio < 0.85:
        return (80, 220, 120), "OK"
    if ratio <= 1.0:
        return (240, 210, 60), "NEAR"
    return (235, 80, 80), "OUT"


def draw_laptop_overlay(
    frame: np.ndarray,
    *,
    elapsed_s: float,
    n_frames: int,
    tracker_count: int,
    reach_metrics: dict,
    manual_control: bool,
) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()
    _panel(out, (0, 0), (w, 48), 0.52)
    _text(out, "REC", (12, 30), 0.62, (235, 80, 80), 2)
    _text(out, _clock(elapsed_s), (w // 2 - 42, 31), 0.72, (70, 220, 245), 2)
    _text(out, f"{n_frames} fr", (w - 150, 20), 0.42, (220, 220, 220), 1)
    tr_color = (80, 220, 120) if tracker_count > 0 else (235, 130, 80)
    _text(out, f"TR:{tracker_count}", (w - 150, 41), 0.38, tr_color, 1)

    x0 = max(8, w - 260)
    y0 = 58
    _panel(out, (x0 - 8, y0 - 18), (w - 8, y0 + 82), 0.44)
    _text(out, "REACH  Piper + OpenArm", (x0, y0 - 3), 0.40, (220, 220, 220), 1)

    for row, robot in enumerate(("piper", "openarm")):
        vals = reach_metrics.get(robot, {})
        y = y0 + 18 + row * 38
        label = "PIPER" if robot == "piper" else "OPEN"
        _text(out, label, (x0, y + 11), 0.36, (230, 230, 230), 1)
        for side, ratio, xoff in (
            ("L", float(vals.get("left_ratio", 0.0)), 48),
            ("R", float(vals.get("right_ratio", 0.0)), 146),
        ):
            color, tag = _reach_color(ratio)
            bx = x0 + xoff
            bw = 62
            cv2.rectangle(out, (bx, y), (bx + bw, y + 10), (48, 48, 48), -1)
            fill = int(bw * min(ratio, 1.25) / 1.25)
            cv2.rectangle(out, (bx, y), (bx + fill, y + 10), color, -1)
            mark = bx + int(bw / 1.25)
            cv2.line(out, (mark, y - 2), (mark, y + 12), (235, 235, 235), 1)
            _text(out, f"{side} {ratio * 100:3.0f}% {tag}", (bx, y + 25), 0.30, color, 1)

    if manual_control:
        _panel(out, (0, h - 28), (w, h), 0.42)
        _text(out, "A stop/save   B repeat   Y finish", (12, h - 9), 0.42, (230, 230, 230), 1)
    return out


class LaptopPreview:
    """Low-latency ffplay window fed with the exact laptop frame saved to the dataset."""

    def __init__(self, *, width: int, height: int, fps: int, title: str) -> None:
        self.width = width
        self.height = height
        self.proc: subprocess.Popen | None = None
        ffplay = shutil.which("ffplay")
        if ffplay is None:
            log.warning("ffplay not found; laptop preview window disabled.")
            return

        cmd = [
            ffplay,
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-framedrop",
            "-sync",
            "ext",
            "-window_title",
            title,
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-",
        ]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            log.info("Laptop preview window opened.")
        except OSError as exc:
            log.warning(f"Could not open laptop preview window: {exc}")

    def show(self, frame: np.ndarray) -> None:
        if self.proc is None or self.proc.poll() is not None or self.proc.stdin is None:
            return
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError):
            log.warning("Laptop preview window closed.")
            self.close()

    def close(self) -> None:
        if self.proc is None:
            return
        proc = self.proc
        self.proc = None
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
