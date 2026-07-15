import unittest

import numpy as np

from handumi.calibration.control_tcp import (
    ControllerTcpCalibration,
    apply_controller_tcp_calibration,
)
from handumi.retargeting.handumi_to_robot import (
    absolute_table_robot_target_pose7,
    local_frame_adapter,
    local_relative_robot_target_pose7,
    orientation_only_pose_adapter,
    quaternion_xyzw_to_matrix,
    raw_state_robot_target_pose7,
    raw_state_target_poses,
    retarget_anchors_from_raw_state,
    split_raw_state,
)


class HandumiToRobotTest(unittest.TestCase):
    def test_controller_tcp_then_table_robot_transform_composes_in_order(self):
        half_turn = np.sqrt(0.5)
        left_controller = np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, half_turn, half_turn]],
            dtype=np.float32,
        )
        right_controller = np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        calibration = ControllerTcpCalibration(
            left=np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            right=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        )
        left_tcp, right_tcp = apply_controller_tcp_calibration(
            left_controller,
            right_controller,
            calibration,
        )
        state = np.concatenate(
            [left_tcp[0], right_tcp[0], np.zeros(2, dtype=np.float32)]
        )
        robot_from_table = np.array(
            [0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )

        left_robot, _ = absolute_table_robot_target_pose7(state, robot_from_table)

        np.testing.assert_allclose(left_robot[:3], [0.3, 0.1, 0.2], atol=1e-6)

    def test_orientation_adapter_preserves_position_and_aligns_rotation(self):
        source = np.array(
            [0.2, -0.1, 0.3, 0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)],
            dtype=np.float32,
        )
        target = np.array(
            [-0.4, 0.8, 0.1, np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
            dtype=np.float32,
        )

        adapter = orientation_only_pose_adapter(source, target)
        state = np.concatenate([source, source, np.zeros(2, dtype=np.float32)])
        mapped, _ = absolute_table_robot_target_pose7(
            state,
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            left_tool_adapter_pose7=adapter,
        )

        np.testing.assert_allclose(mapped[:3], source[:3], atol=1e-6)
        self.assertAlmostEqual(abs(float(np.dot(mapped[3:], target[3:]))), 1.0, places=6)

    def test_absolute_table_retarget_preserves_bimanual_geometry(self):
        state = np.zeros(16, dtype=np.float32)
        state[0:7] = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
        state[7:14] = [-0.2, 0.4, 0.3, 0.0, 0.0, 0.0, 1.0]
        robot_from_table = np.array(
            [0.3, 0.0, 0.1, 0.0, 0.0, -np.sqrt(0.5), np.sqrt(0.5)],
            dtype=np.float32,
        )

        left, right = absolute_table_robot_target_pose7(state, robot_from_table)

        source_distance = np.linalg.norm(state[:3] - state[7:10])
        target_distance = np.linalg.norm(left[:3] - right[:3])
        self.assertAlmostEqual(float(target_distance), float(source_distance), places=6)

    def test_absolute_table_retarget_keeps_common_point_common(self):
        state = np.zeros(16, dtype=np.float32)
        common = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
        state[0:7] = common
        state[7:14] = common
        robot_from_table = np.array(
            [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
        )

        left, right = absolute_table_robot_target_pose7(state, robot_from_table)

        np.testing.assert_allclose(left, right, atol=1e-6)

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
