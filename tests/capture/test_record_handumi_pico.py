import unittest

from handumi.scripts.record_handumi_pico import build_features


class RecordHandUMIFeaturesTest(unittest.TestCase):
    def test_checkpoint_features_do_not_require_tracking(self):
        features = build_features(
            cam_names=["left_wrist", "right_wrist"],
            cam_width=640,
            cam_height=480,
            use_videos=True,
            include_pico=False,
        )

        self.assertIn("observation.images.left_wrist", features)
        self.assertIn("observation.images.right_wrist", features)
        self.assertIn("observation.feetech.left_ticks", features)
        self.assertIn("observation.feetech.right_width_mm", features)
        self.assertNotIn("observation.pico.left_controller_pose", features)
        self.assertNotIn("observation.reach.piper_left_ratio", features)


if __name__ == "__main__":
    unittest.main()
