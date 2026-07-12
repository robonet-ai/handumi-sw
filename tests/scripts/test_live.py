import argparse
import unittest
from pathlib import Path

import numpy as np

from handumi.calibration.control_tcp import ControllerTcpCalibration
from handumi.retargeting.handumi_to_robot import VR_TO_ROBOT
from handumi.robots.kinematics import limit_joint_delta
from handumi.robots.registry import load_robot_config
from handumi.scripts.teleop_sim import (
    _load_calibration,
    _sample_state,
    _start_sides,
    _tracking_world_map,
)
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


class PiperLiveConfigTest(unittest.TestCase):
    def test_home_matches_physical_piper_start(self):
        config = load_robot_config("piper")

        np.testing.assert_allclose(
            config.home_q[[0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13]],
            [0, 0, 0, 0, np.deg2rad(25), 0] * 2,
            atol=1e-7,
        )

    def test_joint_delta_is_limited_per_joint(self):
        current = np.array([0.0, 0.5, -0.5], dtype=np.float32)
        target = np.array([0.2, 0.48, -0.8], dtype=np.float32)

        limited = limit_joint_delta(current, target, np.deg2rad(4))

        np.testing.assert_allclose(
            limited,
            [np.deg2rad(4), 0.48, -0.5 - np.deg2rad(4)],
            atol=1e-7,
        )

    def test_live_world_map_matches_tracking_provider_axes(self):
        np.testing.assert_allclose(_tracking_world_map("pico"), VR_TO_ROBOT)
        np.testing.assert_allclose(_tracking_world_map("meta"), np.eye(3))

    def test_piper_uses_validated_ik_weights(self):
        config = load_robot_config("piper")

        self.assertEqual(config.ik_weights.pos_weight, 100.0)
        self.assertEqual(config.ik_weights.ori_weight, 4.5)
        self.assertEqual(config.ik_weights.rest_weight, 12.0)


class TeleopSimStartTest(unittest.TestCase):
    def test_space_start_only_returns_unanchored_enabled_sides(self):
        anchors = {"left": {"source": np.zeros(7)}, "right": None}

        self.assertEqual(_start_sides(anchors, ("left", "right")), ("right",))
        self.assertEqual(_start_sides(anchors, ("left",)), ())


if __name__ == "__main__":
    unittest.main()
