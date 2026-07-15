import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from handumi.feetech.calibration import (
    FeetechConfig,
    GripperCalibration,
    assert_calibrated,
    load_config,
    load_ports,
    save_calibration,
    user_calibration_path,
)


class FeetechCalibrationTest(unittest.TestCase):
    def test_normalized_width(self):
        calibration = GripperCalibration(
            servo_id=1,
            closed_ticks=1000,
            open_ticks=2000,
        )
        self.assertEqual(calibration.normalized_width(900), 0.0)
        self.assertEqual(calibration.normalized_width(1000), 0.0)
        self.assertEqual(calibration.normalized_width(1500), 0.5)
        self.assertEqual(calibration.normalized_width(2000), 1.0)
        self.assertEqual(calibration.normalized_width(2100), 1.0)

    def test_inverted_ticks_are_supported(self):
        calibration = GripperCalibration(
            servo_id=2,
            closed_ticks=2000,
            open_ticks=1000,
        )
        self.assertEqual(calibration.normalized_width(2000), 0.0)
        self.assertEqual(calibration.normalized_width(1500), 0.5)
        self.assertEqual(calibration.normalized_width(1000), 1.0)

    def test_width_units(self):
        calibration = GripperCalibration(
            servo_id=0,
            closed_ticks=1000,
            open_ticks=2000,
            max_width_mm=80.0,
        )
        self.assertEqual(calibration.width_mm(1500), 40.0)
        self.assertEqual(calibration.width_m(1500), 0.04)


class PortsCalibrationSplitTest(unittest.TestCase):
    """Ports (servo_id/port) and calibration (ticks/mm) persist to separate
    files and merge into one runtime FeetechConfig — see calibration.py."""

    def test_load_ports_ignores_calibration_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            ports_path = Path(tmp) / "feetech.yaml"
            ports_path.write_text(
                "feetech:\n"
                "  baudrate: 1000000\n"
                "  protocol_version: 0\n"
                "  left:\n    servo_id: 0\n    port: /dev/ttyUSB0\n"
                "  right:\n    servo_id: 1\n    port: /dev/ttyUSB1\n",
                encoding="utf-8",
            )
            ports = load_ports(ports_path)
        self.assertEqual(ports.left.port, "/dev/ttyUSB0")
        self.assertEqual(ports.right.servo_id, 1)
        self.assertIsNone(ports.left.closed_ticks)
        self.assertFalse(ports.left.is_complete)

    def test_save_calibration_round_trips_and_merges_with_ports(self):
        config = FeetechConfig(
            port=None,
            baudrate=1_000_000,
            protocol_version=0,
            left=GripperCalibration(0, 1000, 2000, 80.0, "/dev/ttyUSB0"),
            right=GripperCalibration(1, 900, 1900, 75.0, "/dev/ttyUSB1"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            ports_path = Path(tmp) / "feetech.yaml"
            ports_path.write_text(
                "feetech:\n"
                "  baudrate: 1000000\n"
                "  protocol_version: 0\n"
                "  left:\n    servo_id: 0\n    port: /dev/ttyUSB0\n"
                "  right:\n    servo_id: 1\n    port: /dev/ttyUSB1\n",
                encoding="utf-8",
            )
            calibration_path = Path(tmp) / "calibration.yaml"
            save_calibration(config, calibration_path)
            merged = load_config(ports_path, calibration_path)
        self.assertEqual(merged, config)

    def test_missing_calibration_file_yields_uncalibrated_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            ports_path = Path(tmp) / "feetech.yaml"
            ports_path.write_text(
                "feetech:\n"
                "  left:\n    servo_id: 0\n    port: /dev/ttyUSB0\n"
                "  right:\n    servo_id: 1\n    port: /dev/ttyUSB1\n",
                encoding="utf-8",
            )
            merged = load_config(ports_path, Path(tmp) / "does-not-exist.yaml")
        self.assertFalse(merged.left.is_complete)
        self.assertEqual(merged.left.port, "/dev/ttyUSB0")

    def test_user_calibration_path_follows_xdg(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"XDG_CACHE_HOME": tmp}):
                self.assertEqual(
                    user_calibration_path(), Path(tmp) / "handumi" / "calibration.yaml"
                )


class AssertCalibratedTest(unittest.TestCase):
    def _config(self, *, right_complete: bool) -> FeetechConfig:
        right = (
            GripperCalibration(1, 900, 1900, 75.0) if right_complete else GripperCalibration(1)
        )
        return FeetechConfig(
            port=None,
            baudrate=1_000_000,
            protocol_version=0,
            left=GripperCalibration(0, 1000, 2000, 80.0),
            right=right,
        )

    def test_passes_when_complete(self):
        assert_calibrated(self._config(right_complete=True))  # no raise

    def test_raises_and_names_missing_side(self):
        with self.assertRaises(SystemExit) as ctx:
            assert_calibrated(self._config(right_complete=False), source=Path("/x/calibration.yaml"))
        msg = str(ctx.exception)
        self.assertIn("right", msg)
        self.assertIn("/x/calibration.yaml", msg)
        self.assertIn("--skip-feetech", msg)


if __name__ == "__main__":
    unittest.main()
