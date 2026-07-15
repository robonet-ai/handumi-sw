import unittest
import tempfile
from pathlib import Path

import yaml
import numpy as np

from handumi.cameras.base import CameraSample
from handumi.cameras.usb import (
    build_camera_specs,
    read_camera_samples,
    resolve_camera_ids,
)


class _SampledCamera:
    def __init__(self, sample_time_ns: int):
        self.sample_time_ns = sample_time_ns

    def sample_at(self, target_time_ns: int):
        return CameraSample(np.ones((2, 3, 3), dtype=np.uint8), self.sample_time_ns, 4)


class UsbCameraConfigTest(unittest.TestCase):
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
