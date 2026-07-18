import unittest
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np

from handumi.feetech.calibration import FeetechConfig, GripperCalibration
from handumi.scripts.teleop_real import (
    _apply_inactive_side_policy,
    _clear_enabled_anchors,
    _enabled_tracking_ok,
    _has_enabled_anchors,
    _load_required_calibration,
    _validate_feetech_ports_exist,
    _validate_args,
    parse_args,
)
from handumi.scripts.teleop_sim import _start_sides
from handumi.tracking.gestures import DoubleClapDetector


class TeleopRealArgsTest(unittest.TestCase):
    def test_defaults_target_piper_without_space_start(self):
        args = parse_args(["--device", "pico"])

        self.assertEqual(args.robot, "piper")
        self.assertFalse(args.space_start)
        _validate_args(args)

    def test_space_start_is_opt_in(self):
        args = parse_args(["--device", "pico", "--space-start"])

        self.assertTrue(args.space_start)
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

    def test_double_clap_starts_idle_arms_or_resets_active_arms(self):
        detector = DoubleClapDetector(window_s=1.2)
        detector.update(50.0, 50.0, 0.0)
        self.assertFalse(detector.update(2.0, 50.0, 0.05))
        detector.update(50.0, 50.0, 0.2)

        triggered = detector.update(2.0, 50.0, 0.5)
        enabled_sides = ("left", "right")
        idle_anchors = {"left": None, "right": None}
        active_anchors = {"left": {"source": object()}, "right": None}
        start_sides = (
            enabled_sides
            if triggered and not _has_enabled_anchors(idle_anchors, enabled_sides)
            else ()
        )

        self.assertEqual(start_sides, enabled_sides)
        self.assertTrue(_has_enabled_anchors(active_anchors, enabled_sides))
        _clear_enabled_anchors(active_anchors, enabled_sides)
        self.assertFalse(_has_enabled_anchors(active_anchors, enabled_sides))

    def test_tracking_loss_policy_clears_enabled_anchors(self):
        anchors = {"left": {"source": object()}, "right": {"source": object()}}

        self.assertFalse(
            _enabled_tracking_ok({"left": True, "right": False}, ("left", "right"))
        )
        _clear_enabled_anchors(anchors, ("left", "right"))

        self.assertIsNone(anchors["left"])
        self.assertIsNone(anchors["right"])

    def test_tracking_recovery_holds_command_until_reanchored(self):
        previous_q = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        solved_q = np.zeros(4, dtype=np.float32)
        home_q = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        anchors = {"left": None, "right": None}
        side_indices = {"left": [0, 1], "right": [2, 3]}

        _apply_inactive_side_policy(
            solved_q,
            previous_q,
            home_q,
            anchors,
            side_indices,
            {"left"},
        )

        np.testing.assert_array_equal(solved_q[:2], previous_q[:2])
        np.testing.assert_array_equal(solved_q[2:], home_q[2:])

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
            self.assertRaisesRegex(SystemExit, "Remapea Feetech"),
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
