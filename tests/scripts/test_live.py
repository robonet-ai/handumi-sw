import argparse
import unittest
from pathlib import Path

import numpy as np

from handumi.calibration.control_tcp import ControllerTcpCalibration
from handumi.scripts.live import _load_calibration, _sample_state
from handumi.tracking.base import ControllerPairSample


def _sample(left_pos, right_pos) -> ControllerPairSample:
    pose = lambda p: np.array([*p, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # noqa: E731
    return ControllerPairSample(
        device="meta",
        left_controller_pose=pose(left_pos),
        right_controller_pose=pose(right_pos),
        left_tcp_pose=pose(left_pos),
        right_tcp_pose=pose(right_pos),
        left_tracked=True,
        right_tracked=True,
    )


class SampleStateTest(unittest.TestCase):
    def test_layout_uses_tcp_poses_and_zero_widths(self):
        state = _sample_state(_sample([0.1, 0.2, 0.3], [0.4, 0.5, 0.6]))
        self.assertEqual(state.shape, (16,))
        self.assertTrue(np.allclose(state[0:3], [0.1, 0.2, 0.3]))
        self.assertTrue(np.allclose(state[7:10], [0.4, 0.5, 0.6]))
        self.assertEqual(state[14], 0.0)
        self.assertEqual(state[15], 0.0)


class LoadCalibrationTest(unittest.TestCase):
    def _args(self, path) -> argparse.Namespace:
        return argparse.Namespace(controller_tcp_calibration=path, device="meta")

    def test_loads_repo_calibration(self):
        calibration = _load_calibration(
            self._args(Path("configs/calibration/meta_controller_tcp.yaml"))
        )
        self.assertIsInstance(calibration, ControllerTcpCalibration)
        # Repo file carries a non-identity mount offset.
        self.assertFalse(np.allclose(calibration.left[:3], 0.0))

    def test_missing_file_falls_back_to_identity(self):
        calibration = _load_calibration(self._args(Path("/nonexistent/calib.yaml")))
        self.assertTrue(np.allclose(calibration.left[:3], 0.0))
        self.assertTrue(np.allclose(calibration.left[3:7], [0, 0, 0, 1]))


if __name__ == "__main__":
    unittest.main()
