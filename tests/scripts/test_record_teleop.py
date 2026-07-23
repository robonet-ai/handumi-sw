import unittest

import numpy as np

from handumi.dataset.capture import SYNC_LAG_S
from handumi.feetech import GripperWidths
from handumi.scripts.teleop_record import (
    DEFAULT_RECORD_COMMAND_RATE_HZ,
    DEFAULT_RECORD_TRAJECTORY_DELAY_MS,
    PICO_TRACKING_MODE,
    _validate_record_args,
    build_features,
    build_joint_frame,
    joint_state_feature,
    parse_args,
)
from handumi.teleop import (
    DEFAULT_GRIPPER_SAMPLE_HZ,
    DEFAULT_MOTION_SMOOTHING_TIME_CONSTANT_S,
    DEFAULT_TELEOP_FPS,
)


def _widths() -> GripperWidths:
    return GripperWidths(
        left=0.01,
        right=0.02,
        left_mm=10.0,
        right_mm=20.0,
        left_normalized=0.25,
        right_normalized=0.5,
        left_ticks=11,
        right_ticks=22,
    )


class TeleopRecordSchemaTest(unittest.TestCase):
    def test_operational_defaults_are_constants_not_cli_flags(self):
        args = parse_args(["--device", "pico"])

        self.assertEqual(args.pico_mode, PICO_TRACKING_MODE)
        self.assertTrue(args.pico_adb)
        self.assertFalse(args.pico_wifi)
        self.assertFalse(args.skip_feetech)
        self.assertFalse(args.space_start)
        self.assertEqual(args.fps, DEFAULT_TELEOP_FPS)
        self.assertEqual(args.command_rate_hz, DEFAULT_RECORD_COMMAND_RATE_HZ)
        self.assertEqual(
            args.trajectory_delay_ms, DEFAULT_RECORD_TRAJECTORY_DELAY_MS
        )
        self.assertEqual(
            args.motion_smoothing_time_constant_s,
            DEFAULT_MOTION_SMOOTHING_TIME_CONSTANT_S,
        )
        self.assertEqual(args.fps, 30)
        self.assertEqual(args.motion_smoothing_time_constant_s, 0.0)
        self.assertEqual(args.sync_lag_s, SYNC_LAG_S)
        self.assertEqual(args.feetech_sample_hz, DEFAULT_GRIPPER_SAMPLE_HZ)

        with self.assertRaises(SystemExit):
            parse_args(["--device", "pico", "--skip-feetech"])
        with self.assertRaises(SystemExit):
            parse_args(["--device", "pico", "--pico-wifi"])

    def test_trajectory_configuration_is_validated(self):
        args = parse_args(["--device", "pico", "--command-rate-hz", "0"])
        with self.assertRaises(SystemExit):
            _validate_record_args(args)

        args = parse_args(["--device", "pico", "--trajectory-delay-ms", "-1"])
        with self.assertRaises(SystemExit):
            _validate_record_args(args)

    def test_joint_state_feature_uses_robot_joint_names(self):
        self.assertEqual(
            joint_state_feature(["left_joint1", "left_joint2"]),
            {
                "dtype": "float32",
                "shape": (2,),
                "names": ["left_joint1", "left_joint2"],
            },
        )

    def test_features_store_joint_feedback_and_joint_action(self):
        features = build_features(
            ["left_wrist"],
            cam_width=320,
            cam_height=240,
            use_videos=True,
            joint_names=["left_joint1", "right_joint1"],
        )

        self.assertEqual(features["observation.state"]["shape"], (2,))
        self.assertEqual(features["action"]["shape"], (2,))
        self.assertEqual(
            features["observation.state"]["names"],
            ["left_joint1", "right_joint1"],
        )
        self.assertEqual(features["action"]["names"], ["left_joint1", "right_joint1"])
        self.assertEqual(features["observation.images.left_wrist"]["dtype"], "video")
        self.assertFalse(
            any(
                key.startswith("observation.tracking") or key == "observation.valid"
                for key in features
            )
        )

    def test_joint_frame_keeps_observation_and_action_separate(self):
        frame = build_joint_frame(
            observation_q=np.array([1.0, 2.0], dtype=np.float64),
            action_q=np.array([3.0, 4.0], dtype=np.float64),
            widths=_widths(),
        )

        np.testing.assert_array_equal(
            frame["observation.state"], np.array([1.0, 2.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            frame["action"], np.array([3.0, 4.0], dtype=np.float32)
        )
        self.assertEqual(frame["observation.feetech.left_ticks"].item(), 11)
        self.assertEqual(frame["observation.feetech.right_normalized"].item(), 0.5)


if __name__ == "__main__":
    unittest.main()
