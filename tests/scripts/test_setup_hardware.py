import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from handumi.feetech.calibration import FeetechConfig, GripperCalibration
from handumi.scripts.setup import setup_hardware
from handumi.scripts.setup.setup_hardware import ensure_feetech_calibration, parse_args


class SetupHardwareArgsTest(unittest.TestCase):
    def test_defaults_to_piper_pico(self):
        args = parse_args([])

        self.assertEqual(args.robot, "piper")
        self.assertEqual(args.device, "pico")
        self.assertEqual(args.bitrate, 1_000_000)
        self.assertEqual(args.restart_ms, 100)
        self.assertEqual(args.dbitrate, 5_000_000)
        self.assertEqual(args.openarm_zero_side, "both")
        self.assertIsNone(args.controller_tcp_calibration)
        self.assertFalse(args.skip_can_map)
        self.assertFalse(args.skip_feetech_map)
        self.assertFalse(args.skip_feetech_calibration)
        self.assertFalse(args.force_feetech_calibration)
        self.assertFalse(args.skip_feetech_home)
        self.assertEqual(args.feetech_start_id, 0)
        self.assertEqual(args.feetech_end_id, 20)

    def test_can_skip_flags_are_available_for_repair_only_runs(self):
        args = parse_args(["--skip-can-map", "--skip-feetech-map", "--skip-pico"])

        self.assertTrue(args.skip_can_map)
        self.assertTrue(args.skip_feetech_map)
        self.assertTrue(args.skip_pico)

    def test_feetech_calibration_flags_are_available(self):
        args = parse_args(
            [
                "--skip-feetech-calibration",
                "--force-feetech-calibration",
                "--skip-feetech-home",
                "--feetech-max-width-mm",
                "82",
            ]
        )

        self.assertTrue(args.skip_feetech_calibration)
        self.assertTrue(args.force_feetech_calibration)
        self.assertTrue(args.skip_feetech_home)
        self.assertEqual(args.feetech_max_width_mm, 82)


class SetupHardwareFeetechCalibrationTest(unittest.TestCase):
    def _args(self, **overrides):
        values = {
            "rig_config": Path("configs/rig.yaml"),
            "feetech_calibration_config": None,
            "force_feetech_calibration": False,
            "skip_feetech_home": False,
            "feetech_max_width_mm": None,
            "left_feetech_max_width_mm": None,
            "right_feetech_max_width_mm": None,
            "feetech_calibration_interval_s": 0.001,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_skips_when_calibration_is_complete(self):
        config = FeetechConfig(
            port=None,
            baudrate=1_000_000,
            protocol_version=0,
            left=GripperCalibration(0, 1000, 2000, 80.0, "/dev/ttyACM0"),
            right=GripperCalibration(1, 900, 1900, 75.0, "/dev/ttyACM1"),
        )

        with (
            mock.patch.object(setup_hardware, "load_config", return_value=config),
            mock.patch.object(setup_hardware.home_servos, "_home_side") as home,
            mock.patch.object(
                setup_hardware.calibrate_grippers, "_calibrate_side"
            ) as calibrate,
        ):
            ensure_feetech_calibration(self._args())

        home.assert_not_called()
        calibrate.assert_not_called()

    def test_guides_missing_side_and_saves_calibration(self):
        config = FeetechConfig(
            port=None,
            baudrate=1_000_000,
            protocol_version=0,
            left=GripperCalibration(0, 1000, 2000, 80.0, "/dev/ttyACM0"),
            right=GripperCalibration(1, port="/dev/ttyACM1"),
        )

        with tempfile.TemporaryDirectory() as tmp:
            calibration_path = Path(tmp) / "calibration.yaml"
            args = self._args(
                feetech_calibration_config=calibration_path,
                right_feetech_max_width_mm=76.0,
            )
            with (
                mock.patch.object(setup_hardware, "load_config", return_value=config),
                mock.patch("builtins.input", return_value=""),
                mock.patch.object(setup_hardware.home_servos, "_home_side") as home,
                mock.patch.object(
                    setup_hardware.calibrate_grippers,
                    "_calibrate_side",
                    return_value=(950, 1950, 76.0),
                ) as calibrate,
            ):
                ensure_feetech_calibration(args)

            home.assert_called_once()
            calibrate.assert_called_once()
            saved = calibration_path.read_text(encoding="utf-8")

        self.assertIn("right:", saved)
        self.assertIn("closed_ticks: 950", saved)
        self.assertIn("open_ticks: 1950", saved)
        self.assertIn("max_width_mm: 76.0", saved)

    def test_declining_guided_calibration_aborts_with_next_command(self):
        config = FeetechConfig(
            port=None,
            baudrate=1_000_000,
            protocol_version=0,
            left=GripperCalibration(0, port="/dev/ttyACM0"),
            right=GripperCalibration(1, port="/dev/ttyACM1"),
        )
        with (
            mock.patch.object(setup_hardware, "load_config", return_value=config),
            mock.patch("builtins.input", return_value="n"),
            self.assertRaisesRegex(SystemExit, "--skip-can-map"),
        ):
            ensure_feetech_calibration(self._args())


if __name__ == "__main__":
    unittest.main()
