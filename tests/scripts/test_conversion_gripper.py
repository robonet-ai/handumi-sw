import argparse
import unittest

import numpy as np

from handumi.dataset.raw import LEFT_GRIPPER_INDEX, RIGHT_GRIPPER_INDEX
from handumi.scripts.conversion import _write_gripper_joints


class _FakeRuntime:
    # Piper-like: two prismatic fingers per side, mirrored open values.
    finger_joints = {
        "left": ((6, 0.035), (7, -0.035)),
        "right": ((14, 0.035), (15, -0.035)),
    }


def _args(gripper=1.0, max_w=0.08) -> argparse.Namespace:
    return argparse.Namespace(gripper=gripper, gripper_max_width_m=max_w)


def _states(left_m, right_m, n=3) -> np.ndarray:
    states = np.zeros((n, 16), dtype=np.float32)
    states[:, LEFT_GRIPPER_INDEX] = left_m
    states[:, RIGHT_GRIPPER_INDEX] = right_m
    return states


class WriteGripperJointsTest(unittest.TestCase):
    def test_recorded_widths_scale_to_finger_range(self):
        joints = np.zeros((3, 16), dtype=np.float32)
        # 40mm of an 80mm max opening = half open.
        _write_gripper_joints(
            joints, states=_states(0.04, 0.08), runtime=_FakeRuntime(), args=_args()
        )
        self.assertTrue(np.allclose(joints[:, 6], 0.5 * 0.035))
        self.assertTrue(np.allclose(joints[:, 7], 0.5 * -0.035))
        self.assertTrue(np.allclose(joints[:, 14], 0.035))  # fully open, clipped
        self.assertTrue(np.allclose(joints[:, 15], -0.035))

    def test_zero_widths_fall_back_to_constant(self):
        joints = np.zeros((3, 16), dtype=np.float32)
        _write_gripper_joints(
            joints, states=_states(0.0, 0.0), runtime=_FakeRuntime(),
            args=_args(gripper=0.25),
        )
        self.assertTrue(np.allclose(joints[:, 6], 0.25 * 0.035))
        self.assertTrue(np.allclose(joints[:, 14], 0.25 * 0.035))

    def test_overwidth_clips_to_fully_open(self):
        joints = np.zeros((3, 16), dtype=np.float32)
        _write_gripper_joints(
            joints, states=_states(0.2, 0.2), runtime=_FakeRuntime(), args=_args()
        )
        self.assertTrue(np.allclose(joints[:, 6], 0.035))


if __name__ == "__main__":
    unittest.main()
