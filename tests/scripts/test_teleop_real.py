import unittest
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np

from handumi.feetech.calibration import FeetechConfig, GripperCalibration
from handumi.scripts.teleop_real import (
    DEFAULT_REAL_COMMAND_RATE_HZ,
    DEFAULT_REAL_ORIENTATION_DEADBAND_DEG,
    DEFAULT_REAL_POSITION_DEADBAND_MM,
    DEFAULT_REAL_SMOOTHING_TIME_CONSTANT_S,
    DEFAULT_REAL_TRAJECTORY_DELAY_MS,
    _enabled_tracking_ok,
    _load_required_calibration,
    _validate_feetech_ports_exist,
    _validate_real_args as _validate_args,
    parse_args,
)
from handumi.teleop.common import start_sides as _start_sides


class TeleopRealArgsTest(unittest.TestCase):
    def test_defaults_target_piper_without_space_start(self):
        args = parse_args(["--device", "pico"])

        self.assertEqual(args.robot, "piper")
        self.assertEqual(args.fps, 30)
        self.assertEqual(args.command_rate_hz, DEFAULT_REAL_COMMAND_RATE_HZ)
        self.assertEqual(
            args.trajectory_delay_ms, DEFAULT_REAL_TRAJECTORY_DELAY_MS
        )
        self.assertEqual(
            args.motion_smoothing_time_constant_s,
            DEFAULT_REAL_SMOOTHING_TIME_CONSTANT_S,
        )
        self.assertEqual(
            args.motion_position_deadband_mm, DEFAULT_REAL_POSITION_DEADBAND_MM
        )
        self.assertEqual(
            args.motion_orientation_deadband_deg,
            DEFAULT_REAL_ORIENTATION_DEADBAND_DEG,
        )
        self.assertFalse(args.space_start)
        _validate_args(args)

    def test_space_start_is_opt_in(self):
        args = parse_args(["--device", "pico", "--space-start"])

        self.assertTrue(args.space_start)
        _validate_args(args)

    def test_smoothing_configuration_cannot_be_negative(self):
        for option in (
            "--motion-smoothing-time-constant-s",
            "--motion-position-deadband-mm",
            "--motion-orientation-deadband-deg",
        ):
            args = parse_args(["--device", "pico", option, "-0.01"])
            with self.assertRaises(SystemExit):
                _validate_args(args)

    def test_trajectory_configuration_is_validated(self):
        args = parse_args(["--device", "pico", "--command-rate-hz", "0"])
        with self.assertRaises(SystemExit):
            _validate_args(args)

        args = parse_args(["--device", "pico", "--trajectory-delay-ms", "-1"])
        with self.assertRaises(SystemExit):
            _validate_args(args)

    def test_default_calibration_comes_from_piper_robot_tool_setup(self):
        args = parse_args(["--device", "meta"])

        calibration = _load_required_calibration(args)

        np.testing.assert_allclose(
            calibration.left[:3],
            [0.12068467, 0.02142489, -0.21669616],
        )

    def test_accepts_registered_openarm_backend(self):
        args = parse_args(["--device", "pico", "--robot", "openarmv1", "--space-start"])

        _validate_args(args)
        self.assertEqual(args.robot, "openarmv1")

    def test_skip_feetech_requires_space_start(self):
        args = parse_args(["--device", "pico", "--skip-feetech"])

        with self.assertRaises(SystemExit):
            _validate_args(args)

        args = parse_args(["--device", "pico", "--skip-feetech", "--space-start"])
        _validate_args(args)

    def test_space_starts_only_idle_arms(self):
        anchors = {"left": {"source": object()}, "right": None}

        self.assertEqual(_start_sides(anchors, ("left", "right")), ("right",))

    def test_tracking_loss_policy_requires_all_enabled_sides(self):
        self.assertFalse(
            _enabled_tracking_ok({"left": True, "right": False}, ("left", "right"))
        )

    def test_single_side_mode_only_requires_that_side_tracked(self):
        self.assertTrue(_enabled_tracking_ok({"left": True, "right": False}, ("left",)))
        self.assertFalse(
            _enabled_tracking_ok({"left": True, "right": False}, ("right",))
        )

    def test_feetech_port_validation_reports_missing_rig_ports(self):
        config = FeetechConfig(
            port=None,
            baudrate=1_000_000,
            protocol_version=0,
            left=GripperCalibration(0, 1000, 2000, 80.0, "/dev/ttyACM9"),
            right=GripperCalibration(1, 900, 1900, 75.0, "/dev/ttyACM8"),
        )

        with (
            mock.patch(
                "handumi.scripts.teleop_real.list_feetech_serial_ports",
                return_value={"/dev/ttyACM0"},
            ),
            self.assertRaisesRegex(SystemExit, "Remap Feetech"),
        ):
            _validate_feetech_ports_exist(config)

    def test_feetech_port_validation_accepts_existing_ports(self):
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "ttyACM0"
            right = Path(tmp) / "ttyACM1"
            left.touch()
            right.touch()
            config = FeetechConfig(
                port=None,
                baudrate=1_000_000,
                protocol_version=0,
                left=GripperCalibration(0, 1000, 2000, 80.0, str(left)),
                right=GripperCalibration(1, 900, 1900, 75.0, str(right)),
            )

            _validate_feetech_ports_exist(config)


if __name__ == "__main__":
    unittest.main()
