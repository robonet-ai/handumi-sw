import unittest

from handumi.capture.teleoperate_handumi import zero_gripper_widths
from handumi.feetech.calibration import FeetechConfig, GripperCalibration, assert_calibrated


class TeleoperateHandUMITest(unittest.TestCase):
    def test_zero_gripper_widths_match_stream_schema(self):
        widths = zero_gripper_widths()
        self.assertEqual(widths.left_mm, 0.0)
        self.assertEqual(widths.right_mm, 0.0)
        self.assertEqual(widths.left_ticks, 0)
        self.assertEqual(widths.right_ticks, 0)

    def test_assert_calibrated_rejects_incomplete_config(self):
        config = FeetechConfig(
            port="/dev/ttyUSB0",
            baudrate=1_000_000,
            protocol_version=0,
            left=GripperCalibration(0, 1000, 2000, 80.0),
            right=GripperCalibration(1),
        )

        with self.assertRaises(SystemExit):
            assert_calibrated(config)


if __name__ == "__main__":
    unittest.main()
