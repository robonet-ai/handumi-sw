import unittest

import numpy as np

from handumi.feetech import GripperWidths
from handumi.scripts.record_teleop import (
    build_features,
    build_joint_frame,
    joint_state_feature,
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
