import unittest

from handumi.scripts.teleop_real import _validate_args, parse_args
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

    def test_rejects_non_piper_robot(self):
        args = parse_args(["--device", "pico", "--robot", "axol", "--space-start"])

        with self.assertRaises(SystemExit):
            _validate_args(args)

    def test_skip_feetech_requires_space_start(self):
        args = parse_args(["--device", "pico", "--skip-feetech"])

        with self.assertRaises(SystemExit):
            _validate_args(args)

        args = parse_args(["--device", "pico", "--skip-feetech", "--space-start"])
        _validate_args(args)

    def test_space_starts_only_idle_arms(self):
        anchors = {"left": {"source": object()}, "right": None}

        self.assertEqual(_start_sides(anchors, ("left", "right")), ("right",))

    def test_double_clap_reanchors_all_enabled_arms(self):
        detector = DoubleClapDetector(window_s=1.2)
        detector.update(50.0, 50.0, 0.0)
        self.assertFalse(detector.update(2.0, 50.0, 0.05))
        detector.update(50.0, 50.0, 0.2)

        triggered = detector.update(2.0, 50.0, 0.5)
        enabled_sides = ("left", "right")
        start_sides = enabled_sides if triggered else ()

        self.assertEqual(start_sides, enabled_sides)


if __name__ == "__main__":
    unittest.main()
