import socket
import threading
import time
import unittest

import numpy as np

from handumi.tracking.gestures import DoubleClapDetector
from handumi.utils.trajectory import TrajectoryTrail
from handumi.dataset.raw import (
    HANDUMI_RAW_STATE_SIZE,
    LEFT_GRIPPER_INDEX,
    RIGHT_GRIPPER_INDEX,
    pose_to_state_vector,
)
from handumi.tracking import mock_quest_sender as mock
from handumi.tracking.meta_quest import (
    ControllerButtons,
    ControllerState,
    HmdState,
    MetaQuestConfig,
    MetaQuestReceiver,
    controller_pose_in_workspace,
    workspace_from_hmd,
)
from handumi.tracking.transforms import (
    MountingOffsets,
    Pose,
    WorkspaceCalibration,
    unity_pose_to_handumi,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class StateVectorTest(unittest.TestCase):
    def test_layout_and_dtype(self):
        left = Pose([0.1, 0.2, 0.3], [0, 0, 0, 1])
        right = Pose([0.4, 0.5, 0.6], [0, 1, 0, 0])
        state = pose_to_state_vector(left, right, 0.011, 0.022)
        self.assertEqual(state.shape, (HANDUMI_RAW_STATE_SIZE,))
        self.assertEqual(state.dtype, np.float32)
        self.assertTrue(np.allclose(state[0:3], [0.1, 0.2, 0.3]))
        self.assertTrue(np.allclose(state[3:7], [0, 0, 0, 1]))
        self.assertTrue(np.allclose(state[7:10], [0.4, 0.5, 0.6]))
        self.assertTrue(np.allclose(state[10:14], [0, 1, 0, 0]))
        self.assertAlmostEqual(state[LEFT_GRIPPER_INDEX], 0.011, places=5)
        self.assertAlmostEqual(state[RIGHT_GRIPPER_INDEX], 0.022, places=5)


class TrajectoryTrailTest(unittest.TestCase):
    def test_rolling_cap(self):
        trail = TrajectoryTrail(max_points=3)
        for i in range(5):
            trail.append([i, 0, 0])
        pts = trail.points()
        self.assertEqual(pts.shape, (3, 3))
        self.assertTrue(np.allclose(pts[:, 0], [2, 3, 4]))

    def test_empty_and_clear(self):
        trail = TrajectoryTrail(max_points=3)
        self.assertEqual(trail.points().shape, (0, 3))
        trail.append([1, 2, 3])
        trail.clear()
        self.assertEqual(trail.points().shape, (0, 3))


class DoubleClapDetectorTest(unittest.TestCase):
    def test_reports_triggering_side(self):
        det = DoubleClapDetector(window_s=1.2)
        self.assertIsNone(det.update_side(50.0, 50.0, 0.0))
        self.assertIsNone(det.update_side(2.0, 50.0, 0.1))
        self.assertIsNone(det.update_side(50.0, 50.0, 0.2))
        self.assertEqual(det.update_side(2.0, 50.0, 0.3), "left")

    def _clap(self, det, t, left=True, right=True):
        """One clap: open -> closed -> (returns result of the closed sample)."""
        det.update(50.0, 50.0, t)
        return det.update(2.0 if left else 50.0, 3.0 if right else 50.0, t + 0.05)

    def test_double_clap_left_only_triggers(self):
        det = DoubleClapDetector(window_s=1.2)
        self.assertFalse(self._clap(det, 0.0, right=False))
        self.assertTrue(self._clap(det, 0.5, right=False))

    def test_double_clap_right_only_triggers(self):
        det = DoubleClapDetector(window_s=1.2)
        self.assertFalse(self._clap(det, 0.0, left=False))
        self.assertTrue(self._clap(det, 0.5, left=False))

    def test_double_clap_both_triggers(self):
        det = DoubleClapDetector(window_s=1.2)
        self.assertFalse(self._clap(det, 0.0))
        self.assertTrue(self._clap(det, 0.5))

    def test_single_clap_does_not_trigger(self):
        det = DoubleClapDetector(window_s=1.2)
        self.assertFalse(self._clap(det, 0.0))
        # stays closed — no re-arm, no second clap
        self.assertFalse(det.update(2.0, 2.0, 0.5))
        self.assertFalse(det.update(2.0, 2.0, 1.0))

    def test_slow_claps_do_not_trigger(self):
        det = DoubleClapDetector(window_s=1.2)
        self.assertFalse(self._clap(det, 0.0))
        self.assertFalse(self._clap(det, 3.0))  # too late — counts as a new first clap
        self.assertTrue(self._clap(det, 3.5))  # ...which a quick follow-up completes

    def test_alternating_sides_do_not_trigger(self):
        det = DoubleClapDetector(window_s=1.2)
        # one clap left, then one clap right — neither side double-clapped
        self.assertFalse(self._clap(det, 0.0, right=False))
        self.assertFalse(self._clap(det, 0.5, left=False))


class CalibrationHelpersTest(unittest.TestCase):
    def test_identity_calibration_equals_unity_conversion(self):
        ctrl = ControllerState(
            tracked=True, valid=True,
            position=np.array([0.2, 0.9, 0.3]),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            buttons=ControllerButtons(),
        )
        out = controller_pose_in_workspace(
            ctrl, mounting_offset=Pose.identity(), workspace=WorkspaceCalibration.identity()
        )
        expected = unity_pose_to_handumi(ctrl.position, ctrl.quaternion)
        self.assertTrue(np.allclose(out.as_matrix(), expected.as_matrix()))

    def test_workspace_from_hmd_recenters(self):
        hmd = HmdState(tracked=True, position=np.array([0.0, 1.1, 0.2]),
                       quaternion=np.array([0.0, 0.0, 0.0, 1.0]))
        ws = workspace_from_hmd(hmd)
        ref = unity_pose_to_handumi(hmd.position, hmd.quaternion)
        self.assertTrue(np.allclose(ws.apply(ref).as_matrix(), np.eye(4), atol=1e-9))


if __name__ == "__main__":
    unittest.main()
