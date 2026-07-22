import unittest
import tempfile
import time
from pathlib import Path
from unittest import mock

import yaml
import numpy as np

from handumi.cameras.base import CameraSample
from handumi.cameras.opencv import OpenCVCameraDevice
from handumi.cameras.usb import (
    CameraStartupError,
    build_camera_specs,
    connect_cameras,
    read_camera_samples,
    resolve_camera_ids,
    validate_camera_streams,
)


class _SampledCamera:
    def __init__(self, sample_time_ns: int):
        self.sample_time_ns = sample_time_ns

    def sample_at(self, target_time_ns: int):
        return CameraSample(np.ones((2, 3, 3), dtype=np.uint8), self.sample_time_ns, 4)


class _ConnectCamera:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.disconnect_calls = 0

    def connect(self):
        if self.fail:
            raise ConnectionError("injected camera failure")

    def disconnect(self):
        self.disconnect_calls += 1


class _ProgressingCamera:
    def __init__(self, *, frozen: bool = False):
        self.sequence = 0
        self.frozen = frozen
        self.capture_time_ns = time.monotonic_ns()

    def sample_at(self, target_time_ns: int):
        if not self.frozen:
            self.sequence += 1
            self.capture_time_ns = time.monotonic_ns()
        return CameraSample(
            np.ones((2, 3, 3), dtype=np.uint8),
            self.capture_time_ns,
            self.sequence,
        )


class UsbCameraConfigTest(unittest.TestCase):
    def test_partial_connect_failure_disconnects_every_created_camera(self):
        first = _ConnectCamera()
        failing = _ConnectCamera(fail=True)
        specs = [
            {"id": 1, "name": "left_wrist", "is_laptop": False},
            {"id": 2, "name": "right_wrist", "is_laptop": False},
        ]
        with mock.patch(
            "handumi.cameras.usb._make_camera",
            side_effect=[first, failing],
        ):
            with self.assertRaisesRegex(ConnectionError, "injected"):
                connect_cameras(
                    specs,
                    fps=30,
                    width=640,
                    height=480,
                    zero_non_laptop=False,
                )
        self.assertEqual(first.disconnect_calls, 1)
        self.assertEqual(failing.disconnect_calls, 1)

    def test_camera_backend_defaults_to_mjpeg(self):
        camera = OpenCVCameraDevice(0, fps=30, width=640, height=480)
        self.assertEqual(camera.fourcc, "MJPG")

    def test_simultaneous_startup_gate_accepts_progressing_streams(self):
        validate_camera_streams(
            [_ProgressingCamera(), _ProgressingCamera()],
            ["left_wrist", "right_wrist"],
            duration_s=0.03,
            stale_timeout_s=0.02,
            minimum_distinct_frames=2,
            poll_s=0.002,
        )

    def test_simultaneous_startup_gate_rejects_frozen_stream(self):
        with self.assertRaisesRegex(CameraStartupError, "right_wrist"):
            validate_camera_streams(
                [_ProgressingCamera(), _ProgressingCamera(frozen=True)],
                ["left_wrist", "right_wrist"],
                duration_s=0.03,
                stale_timeout_s=0.005,
                minimum_distinct_frames=2,
                poll_s=0.002,
            )

    def test_build_camera_specs_without_laptop_camera(self):
        specs, laptop_name = build_camera_specs(
            [0, 2],
            laptop_camera=False,
            laptop_cam_id=4,
            laptop_cam_name="laptop",
        )

        self.assertIsNone(laptop_name)
        self.assertEqual(
            specs,
            [
                {"id": 0, "name": "left_wrist", "is_laptop": False},
                {"id": 2, "name": "right_wrist", "is_laptop": False},
            ],
        )

    def test_build_camera_specs_reuses_named_camera_for_laptop(self):
        specs, laptop_name = build_camera_specs(
            [0, 2],
            laptop_camera=True,
            laptop_cam_id=9,
            laptop_cam_name="right_wrist",
        )

        self.assertEqual(laptop_name, "right_wrist")
        self.assertEqual(
            specs,
            [
                {"id": 0, "name": "left_wrist", "is_laptop": False},
                {"id": 9, "name": "right_wrist", "is_laptop": True},
            ],
        )

    def test_build_named_workspace_camera_spec(self):
        specs, _ = build_camera_specs(
            [3, 5, 7],
            camera_names=["left_wrist", "right_wrist", "workspace"],
            laptop_camera=False,
            laptop_cam_id=9,
            laptop_cam_name="laptop",
        )

        self.assertEqual(
            [spec["name"] for spec in specs],
            ["left_wrist", "right_wrist", "workspace"],
        )

    def test_resolve_camera_ids_from_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cameras.yaml"
            with path.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(
                    {
                        "cameras": {
                            "left_wrist": {"index_or_path": 3},
                            "right_wrist": {"index_or_path": 5},
                            "workspace": {"index_or_path": 7},
                        }
                    },
                    fh,
                )

            self.assertEqual(resolve_camera_ids(None, path), [3, 5])
            self.assertEqual(
                resolve_camera_ids(
                    None,
                    path,
                    camera_names=["left_wrist", "right_wrist", "workspace"],
                ),
                [3, 5, 7],
            )

    def test_explicit_camera_ids_override_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cameras.yaml"
            self.assertEqual(resolve_camera_ids([7, 8], path), [7, 8])

    def test_timestamped_camera_health_is_recorded(self):
        frame, health = read_camera_samples(
            [_SampledCamera(1_002_000_000)],
            ["left_wrist"],
            target_time_ns=1_000_000_000,
            record_time_ns=1_010_000_000,
            width=3,
            height=2,
            stale_timeout_s=0.1,
            max_sync_skew_s=0.01,
        )

        self.assertTrue(health["camera.left_wrist"])
        self.assertEqual(
            frame["observation.camera.left_wrist.sample_time_ns"].item(),
            1_002_000_000,
        )
        self.assertEqual(frame["observation.camera.left_wrist.healthy"].item(), 1)
        self.assertNotIn("observation.camera.left_wrist.enabled", frame)
        self.assertNotIn("observation.camera.left_wrist.sync_error_ms", frame)

    def test_stale_camera_is_unhealthy(self):
        _, health = read_camera_samples(
            [_SampledCamera(1_000_000_000)],
            ["left_wrist"],
            target_time_ns=2_000_000_000,
            record_time_ns=2_000_000_000,
            width=3,
            height=2,
            stale_timeout_s=0.1,
            max_sync_skew_s=0.01,
        )

        self.assertFalse(health["camera.left_wrist"])


if __name__ == "__main__":
    unittest.main()
