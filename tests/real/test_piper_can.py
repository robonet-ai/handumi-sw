import time
import unittest
from unittest import mock

import numpy as np

from handumi.real.piper_can import (
    PiperCanEnvironment,
    PiperCanSettings,
    PiperJointStreamer,
    load_piper_can_settings,
    piper_mdeg_to_q,
    q_to_piper_mdeg,
    step_mdeg_toward,
)
from handumi.robots.registry import RobotRealConfig, load_robot_config


JOINT_NAMES = [
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "left_gripper",
    "left_gripper_mirror",
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
    "right_gripper",
    "right_gripper_mirror",
]


class FakeArm:
    def __init__(self, start=None):
        self.port = "fake"
        self.current = np.zeros(6, dtype=np.int64) if start is None else np.asarray(start)
        self.sent: list[np.ndarray] = []
        self.grippers: list[tuple[int, int]] = []
        self.closed = False

    def read_mdeg(self):
        return self.current.copy()

    def send_mdeg(self, cmd):
        self.current = np.asarray(cmd, dtype=np.int64).copy()
        self.sent.append(self.current.copy())

    def send_gripper_microm(self, opening_microm, effort):
        self.grippers.append((int(opening_microm), int(effort)))

    def disconnect(self):
        self.closed = True


class PiperCanConfigTest(unittest.TestCase):
    def test_loads_can_from_rig_and_real_defaults_from_robot_config(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as tmp:
            rig = Path(tmp) / "rig.yaml"
            rig.write_text(
                "robots:\n"
                "  piper:\n"
                "    can:\n"
                "      bitrate: 1000000\n"
                "      restart_ms: 100\n"
                "      left_port: can0\n"
                "      right_port: can1\n",
                encoding="utf-8",
            )
            settings = load_piper_can_settings(
                rig,
                RobotRealConfig(command_rate_hz=123, max_joint_speed_deg_s=45),
            )

        self.assertEqual(settings.left_port, "can0")
        self.assertEqual(settings.right_port, "can1")
        self.assertEqual(settings.bitrate, 1_000_000)
        self.assertEqual(settings.restart_ms, 100)
        self.assertEqual(settings.command_rate_hz, 123)
        self.assertEqual(settings.max_joint_speed_deg_s, 45)
        self.assertEqual(settings.gripper_effort, 1000)

    def test_piper_yaml_real_defaults(self):
        config = load_robot_config("piper")

        self.assertEqual(config.real.command_rate_hz, 100.0)
        self.assertEqual(config.real.max_joint_speed_deg_s, 180.0)
        self.assertEqual(config.real.home_max_joint_speed_deg_s, 20.0)
        self.assertEqual(config.real.home_timeout_s, 30.0)
        self.assertEqual(config.real.home_tolerance_deg, 3.0)
        self.assertEqual(config.real.speed_percent, 80)
        self.assertEqual(config.real.gripper_effort, 1000)


class PiperUnitsTest(unittest.TestCase):
    def test_q_to_piper_mdeg_matches_xhuman_home_pose(self):
        q = np.zeros(len(JOINT_NAMES), dtype=np.float32)
        q[JOINT_NAMES.index("left_joint5")] = np.deg2rad(25.0)
        q[JOINT_NAMES.index("right_joint5")] = np.deg2rad(25.0)

        targets = q_to_piper_mdeg(q, JOINT_NAMES)

        np.testing.assert_array_equal(targets["left"], [0, 0, 0, 0, 25000, 0])
        np.testing.assert_array_equal(targets["right"], [0, 0, 0, 0, 25000, 0])

    def test_piper_mdeg_feedback_round_trips_into_full_q(self):
        base_q = np.ones(len(JOINT_NAMES), dtype=np.float32)

        q = piper_mdeg_to_q(
            left_mdeg=np.array([1000, 0, 0, 0, 25000, 0]),
            right_mdeg=np.array([0, 0, -2000, 0, 25000, 0]),
            actuated_names=JOINT_NAMES,
            base_q=base_q,
        )

        self.assertAlmostEqual(q[JOINT_NAMES.index("left_joint1")], np.deg2rad(1.0))
        self.assertAlmostEqual(q[JOINT_NAMES.index("left_joint5")], np.deg2rad(25.0))
        self.assertAlmostEqual(q[JOINT_NAMES.index("right_joint3")], np.deg2rad(-2.0))
        self.assertEqual(q[JOINT_NAMES.index("left_gripper")], 1.0)

    def test_step_mdeg_toward_limits_per_joint(self):
        current = np.array([0, 5000, -5000], dtype=np.int64)
        target = np.array([2000, 4500, -9000], dtype=np.int64)

        limited = step_mdeg_toward(current, target, max_step_mdeg=1000)

        np.testing.assert_array_equal(limited, [1000, 4500, -6000])


class PiperJointStreamerTest(unittest.TestCase):
    def test_streamer_sends_latest_target_to_fake_arms(self):
        left = FakeArm()
        right = FakeArm()
        streamer = PiperJointStreamer(
            {"left": left, "right": right},
            command_rate_hz=500.0,
            max_joint_speed_deg_s=500.0,
            gripper_effort=1000,
        )

        streamer.start()
        try:
            target = {
                "left": np.array([1000, 0, 0, 0, 0, 0], dtype=np.int64),
                "right": np.array([0, -1000, 0, 0, 0, 0], dtype=np.int64),
            }
            streamer.set_targets(target)
            streamer.wait_until_targets(timeout_s=1.0, tolerance_mdeg=0.0)
        finally:
            streamer.stop()

        np.testing.assert_array_equal(left.read_mdeg(), target["left"])
        np.testing.assert_array_equal(right.read_mdeg(), target["right"])
        self.assertGreaterEqual(len(left.sent), 1)
        self.assertGreaterEqual(len(right.sent), 1)

    def test_streamer_can_stop_without_targets(self):
        streamer = PiperJointStreamer(
            {"left": FakeArm(), "right": FakeArm()},
            command_rate_hz=100.0,
            max_joint_speed_deg_s=100.0,
            gripper_effort=1000,
        )
        streamer.start()
        time.sleep(0.02)
        streamer.stop()

    def test_hold_current_commands_cancels_pending_motion(self):
        left = FakeArm()
        right = FakeArm()
        streamer = PiperJointStreamer(
            {"left": left, "right": right},
            command_rate_hz=500.0,
            max_joint_speed_deg_s=500.0,
            gripper_effort=1000,
        )
        streamer.set_targets(
            {
                "left": np.full(6, 10_000, dtype=np.int64),
                "right": np.full(6, -10_000, dtype=np.int64),
            }
        )

        held = streamer.hold_current_commands()
        streamer.start()
        try:
            time.sleep(0.02)
        finally:
            streamer.stop()

        np.testing.assert_array_equal(held["left"], np.zeros(6, dtype=np.int64))
        np.testing.assert_array_equal(held["right"], np.zeros(6, dtype=np.int64))
        np.testing.assert_array_equal(left.read_mdeg(), held["left"])
        np.testing.assert_array_equal(right.read_mdeg(), held["right"])

    def test_streamer_sends_gripper_targets_to_fake_arms(self):
        left = FakeArm()
        right = FakeArm()
        streamer = PiperJointStreamer(
            {"left": left, "right": right},
            command_rate_hz=500.0,
            max_joint_speed_deg_s=500.0,
            gripper_effort=1234,
        )

        streamer.start()
        try:
            streamer.set_gripper_targets_microm({"left": 42000, "right": 17000})
            time.sleep(0.02)
        finally:
            streamer.stop()

        self.assertIn((42000, 1234), left.grippers)
        self.assertIn((17000, 1234), right.grippers)


class PiperCanEnvironmentTest(unittest.TestCase):
    def test_move_home_temporarily_uses_slow_joint_limit(self):
        settings = PiperCanSettings(
            left_port="can0",
            right_port="can1",
            max_joint_speed_deg_s=180.0,
            home_max_joint_speed_deg_s=20.0,
            home_timeout_s=12.0,
            home_tolerance_deg=2.0,
        )
        environment = PiperCanEnvironment(settings)
        environment.streamer = mock.Mock()
        targets = {
            "left": np.zeros(6, dtype=np.int64),
            "right": np.zeros(6, dtype=np.int64),
        }

        environment.move_home(targets)

        self.assertEqual(
            environment.streamer.set_max_joint_speed_deg_s.call_args_list,
            [mock.call(20.0), mock.call(180.0)],
        )
        environment.streamer.set_targets.assert_called_once_with(targets)
        environment.streamer.wait_until_targets.assert_called_once_with(
            timeout_s=12.0,
            tolerance_mdeg=2000.0,
        )


if __name__ == "__main__":
    unittest.main()
