import unittest

import numpy as np

from handumi.retargeting.handumi_to_robot import (
    local_frame_adapter,
    local_relative_robot_target_pose7,
    quaternion_xyzw_to_matrix,
    raw_state_robot_target_pose7,
    raw_state_target_poses,
    retarget_anchors_from_raw_state,
    split_raw_state,
)


class HandumiToRobotTest(unittest.TestCase):
    def test_quaternion_identity(self):
        rot = quaternion_xyzw_to_matrix(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(rot, np.eye(3), atol=1e-6)

    def test_split_raw_state(self):
        state = np.zeros(16, dtype=np.float32)
        state[0:7] = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
        state[7:14] = [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0]
        state[14] = 0.25
        state[15] = 0.75

        raw = split_raw_state(state)

        np.testing.assert_allclose(raw.left_position, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(raw.right_position, [4.0, 5.0, 6.0])
        self.assertAlmostEqual(raw.left_gripper_width, 0.25)
        self.assertAlmostEqual(raw.right_gripper_width, 0.75)

    def test_raw_state_target_poses(self):
        state = np.zeros(16, dtype=np.float32)
        state[3:7] = [0.0, 0.0, 0.0, 1.0]
        state[10:14] = [0.0, 0.0, 0.0, 1.0]

        left_pose, right_pose = raw_state_target_poses(state)

        np.testing.assert_allclose(left_pose[1], np.eye(3), atol=1e-6)
        np.testing.assert_allclose(right_pose[1], np.eye(3), atol=1e-6)

    def test_pose_only_state_without_grippers(self):
        state = np.zeros(14, dtype=np.float32)
        state[0:7] = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
        state[7:14] = [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0]

        raw = split_raw_state(state)
        left_pose, right_pose = raw_state_target_poses(state)

        np.testing.assert_allclose(raw.left_position, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(raw.right_position, [4.0, 5.0, 6.0])
        np.testing.assert_allclose(left_pose[1], np.eye(3), atol=1e-6)
        np.testing.assert_allclose(right_pose[1], np.eye(3), atol=1e-6)
        self.assertTrue(np.isnan(raw.left_gripper_width))
        self.assertTrue(np.isnan(raw.right_gripper_width))

    def test_robot_target_pose7_is_anchored_at_robot_home(self):
        state = np.zeros(16, dtype=np.float32)
        state[0:7] = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
        state[7:14] = [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0]
        left_home = np.array([0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        right_home = np.array([0.3, -0.2, 0.1, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        anchors = retarget_anchors_from_raw_state(
            state,
            left_robot_pose7=left_home,
            right_robot_pose7=right_home,
        )

        left_target, right_target = raw_state_robot_target_pose7(state, anchors)

        np.testing.assert_allclose(left_target, left_home, atol=1e-6)
        np.testing.assert_allclose(right_target, right_home, atol=1e-6)

    def test_robot_target_pose7_accepts_pose_only_state(self):
        state = np.zeros(14, dtype=np.float32)
        state[0:7] = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]
        state[7:14] = [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0]
        left_home = np.array([0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        right_home = np.array([0.3, -0.2, 0.1, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        anchors = retarget_anchors_from_raw_state(
            state,
            left_robot_pose7=left_home,
            right_robot_pose7=right_home,
        )

        left_target, right_target = raw_state_robot_target_pose7(state, anchors)

        np.testing.assert_allclose(left_target, left_home, atol=1e-6)
        np.testing.assert_allclose(right_target, right_home, atol=1e-6)

    def test_robot_target_pose7_clamps_position_delta(self):
        state0 = np.zeros(16, dtype=np.float32)
        state0[3:7] = [0.0, 0.0, 0.0, 1.0]
        state0[10:14] = [0.0, 0.0, 0.0, 1.0]
        state1 = state0.copy()
        state1[:3] = [2.0, 0.0, 0.0]
        state1[7:10] = [0.0, 3.0, 0.0]
        home = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        anchors = retarget_anchors_from_raw_state(
            state0,
            left_robot_pose7=home,
            right_robot_pose7=home,
            max_reach=0.5,
        )

        left_target, right_target = raw_state_robot_target_pose7(state1, anchors)

        self.assertAlmostEqual(float(np.linalg.norm(left_target[:3])), 0.5)
        self.assertAlmostEqual(float(np.linalg.norm(right_target[:3])), 0.5)

    def test_local_relative_retarget_maps_vr_y_to_robot_z(self):
        previous_source = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )
        current_source = np.array(
            [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )
        home = np.array([0.3, 0.2, 0.4, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        adapter = local_frame_adapter(previous_source, home)

        target = local_relative_robot_target_pose7(
            previous_source_pose7=previous_source,
            current_source_pose7=current_source,
            base_robot_pose7=home,
            adapter_rot=adapter,
            home_robot_pose7=home,
            translation_scale=1.0,
        )

        np.testing.assert_allclose(target[:3], [0.3, 0.2, 0.5], atol=1e-6)
        np.testing.assert_allclose(target[3:], home[3:], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
