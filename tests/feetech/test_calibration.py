import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from handumi.feetech.calibration import (
    REPO_TEMPLATE_PATH,
    FeetechConfig,
    GripperCalibration,
    assert_calibrated,
    load_config,
    resolve_config_path,
    save_config,
    user_config_path,
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

    def test_round_trip_config(self):
        config = FeetechConfig(
            port=None,
            baudrate=1_000_000,
            protocol_version=0,
            left=GripperCalibration(0, 1000, 2000, 80.0, "/dev/ttyUSB0"),
            right=GripperCalibration(1, 900, 1900, 75.0, "/dev/ttyUSB1"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feetech.yaml"
            save_config(config, path)
            loaded = load_config(path)
        self.assertEqual(loaded, config)

    def test_width_units(self):
        calibration = GripperCalibration(
            servo_id=0,
            closed_ticks=1000,
            open_ticks=2000,
            max_width_mm=80.0,
        )
        self.assertEqual(calibration.width_mm(1500), 40.0)
        self.assertEqual(calibration.width_m(1500), 0.04)


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
            assert_calibrated(self._config(right_complete=False), source=Path("/x/feetech.yaml"))
        msg = str(ctx.exception)
        self.assertIn("right", msg)
        self.assertIn("/x/feetech.yaml", msg)
        self.assertIn("--skip-feetech", msg)


class ResolveConfigPathTest(unittest.TestCase):
    def test_explicit_override_wins(self):
        explicit = Path("/tmp/whatever.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"XDG_CACHE_HOME": tmp}):
                self.assertEqual(resolve_config_path(explicit), explicit)

    def test_user_cache_path_follows_xdg(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"XDG_CACHE_HOME": tmp}):
                self.assertEqual(
                    user_config_path(), Path(tmp) / "handumi" / "feetech.yaml"
                )

    def test_falls_back_to_template_without_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"XDG_CACHE_HOME": tmp}):
                self.assertEqual(resolve_config_path(), REPO_TEMPLATE_PATH)

    def test_prefers_cache_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"XDG_CACHE_HOME": tmp}):
                cache = user_config_path()
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text("port: null\n", encoding="utf-8")
                self.assertEqual(resolve_config_path(), cache)

    def test_seed_creates_cache_from_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"XDG_CACHE_HOME": tmp}):
                path = resolve_config_path(seed=True)
                self.assertEqual(path, user_config_path())
                self.assertTrue(path.exists())
                # Seeded from the template, so it loads as a valid (uncalibrated) config.
                self.assertFalse(load_config(path).left.is_complete)


if __name__ == "__main__":
    unittest.main()
