import argparse
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from handumi.calibration.control_tcp import ControllerTcpCalibration
from handumi.retargeting.handumi_to_robot import VR_TO_ROBOT
from handumi.robots.kinematics import limit_joint_delta
from handumi.robots.registry import load_robot_config
from handumi.scripts.teleop_sim import (
    AutoStartCountdown,
    _load_calibration,
    parse_args,
    _resolve_camera_usage,
    _sample_state,
    _selected_camera_names,
    _start_sides,
    _tracking_ready_for_sides,
    _tracking_world_map,
    _validate_unique_camera_ids,
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


class TeleopSimCameraSelectionTest(unittest.TestCase):
    def test_no_viser_flag_is_parsed(self):
        with mock.patch(
            "sys.argv", ["handumi-teleop-sim", "--device", "meta", "--no-viser"]
        ):
            self.assertTrue(parse_args().no_viser)

    def test_context_camera_is_between_wrist_views(self):
        self.assertEqual(
            _selected_camera_names(context_camera=True),
            ["left_wrist", "workspace", "right_wrist"],
        )

    def test_context_camera_flag_is_parsed(self):
        with mock.patch(
            "sys.argv", ["handumi-teleop-sim", "--device", "meta", "--context-camera"]
        ):
            self.assertTrue(parse_args().context_camera)

    def test_duplicate_camera_devices_are_rejected(self):
        with self.assertRaisesRegex(SystemExit, "distinct devices"):
            _validate_unique_camera_ids(
                ["left_wrist", "workspace", "right_wrist"], [2, 2, 4]
            )

    def test_default_uses_both_wrist_cameras(self):
        self.assertEqual(
            _selected_camera_names(context_camera=False),
            ["left_wrist", "right_wrist"],
        )


class ResolveCameraUsageTest(unittest.TestCase):
    def _args(self, **overrides) -> argparse.Namespace:
        base = dict(
            no_rerun=False,
            context_camera=False,
            cam_ids=None,
            skip_cameras=False,
        )
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_rerun_enabled_leaves_cameras_untouched(self):
        args = self._args(no_rerun=False, skip_cameras=False)
        _resolve_camera_usage(args)
        self.assertFalse(args.skip_cameras)

    def test_no_rerun_auto_skips_cameras(self):
        args = self._args(no_rerun=True, skip_cameras=False)
        _resolve_camera_usage(args)
        self.assertTrue(args.skip_cameras)

    def test_no_rerun_with_skip_cameras_is_a_noop(self):
        args = self._args(no_rerun=True, skip_cameras=True)
        _resolve_camera_usage(args)
        self.assertTrue(args.skip_cameras)

    def test_no_rerun_with_context_camera_is_rejected(self):
        args = self._args(no_rerun=True, context_camera=True)
        with self.assertRaisesRegex(SystemExit, "only shown in Rerun"):
            _resolve_camera_usage(args)

    def test_no_rerun_with_explicit_cam_ids_is_rejected(self):
        args = self._args(no_rerun=True, cam_ids=[0, 2])
        with self.assertRaisesRegex(SystemExit, "only shown in Rerun"):
            _resolve_camera_usage(args)


class LoadCalibrationTest(unittest.TestCase):
    def _args(self, path) -> argparse.Namespace:
        return argparse.Namespace(
            controller_tcp_calibration=path,
            device="meta",
            robot="piper",
        )

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

    def test_default_comes_from_piper_robot_tool_setup(self):
        calibration = _load_calibration(self._args(None))

        np.testing.assert_allclose(
            calibration.left[:3],
            [0.12068467, 0.02142489, -0.21669616],
        )


class PiperTeleopSimConfigTest(unittest.TestCase):
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

    def test_world_map_matches_tracking_provider_axes(self):
        np.testing.assert_allclose(_tracking_world_map("pico"), VR_TO_ROBOT)
        np.testing.assert_allclose(_tracking_world_map("meta"), np.eye(3))

    def test_piper_uses_validated_ik_weights(self):
        config = load_robot_config("piper")

        self.assertEqual(config.ik_weights.pos_weight, 100.0)
        self.assertEqual(config.ik_weights.ori_weight, 4.5)
        self.assertEqual(config.ik_weights.rest_weight, 12.0)

    def test_piper_selects_validated_meta_tcp_calibration(self):
        config = load_robot_config("piper")

        path = config.controller_tcp_calibrations["meta"]
        self.assertEqual(path.name, "meta_controller_tcp.yaml")
        self.assertTrue(path.is_file())
        self.assertEqual(config.handumi_gripper, "piper_parallel_v1")
        self.assertEqual(config.handumi_controller_mount, "handumi_v1")


class TeleopSimStartTest(unittest.TestCase):
    def test_auto_start_tracking_requires_nonzero_pose_for_every_enabled_side(self):
        poses = {
            "left": np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]),
            "right": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        }
        tracked = {"left": True, "right": True}

        self.assertFalse(_tracking_ready_for_sides(poses, tracked, ("left", "right")))
        self.assertTrue(_tracking_ready_for_sides(poses, tracked, ("left",)))

    def test_auto_start_flag_defaults_to_five_seconds(self):
        with mock.patch(
            "sys.argv", ["handumi-teleop-sim", "--device", "pico", "--auto-start"]
        ):
            args = parse_args()

        self.assertTrue(args.auto_start)
        self.assertEqual(args.auto_start_delay_s, 5.0)

    def test_auto_start_requires_continuous_tracking_and_fires_once(self):
        countdown = AutoStartCountdown(enabled=True, delay_s=5.0)
        idle = ("left", "right")

        self.assertEqual(
            countdown.update(
                now=10.0,
                tracking_ready=True,
                already_active=False,
                idle_sides=idle,
            ),
            (),
        )
        self.assertEqual(
            countdown.update(
                now=13.0,
                tracking_ready=False,
                already_active=False,
                idle_sides=idle,
            ),
            (),
        )
        self.assertEqual(
            countdown.update(
                now=14.0,
                tracking_ready=True,
                already_active=False,
                idle_sides=idle,
            ),
            (),
        )
        self.assertEqual(
            countdown.update(
                now=18.9,
                tracking_ready=True,
                already_active=False,
                idle_sides=idle,
            ),
            (),
        )
        self.assertEqual(
            countdown.update(
                now=19.0,
                tracking_ready=True,
                already_active=False,
                idle_sides=idle,
            ),
            idle,
        )
        self.assertEqual(
            countdown.update(
                now=25.0,
                tracking_ready=True,
                already_active=False,
                idle_sides=idle,
            ),
            (),
        )

    def test_space_start_only_returns_unanchored_enabled_sides(self):
        anchors = {"left": {"source": np.zeros(7)}, "right": None}

        self.assertEqual(_start_sides(anchors, ("left", "right")), ("right",))
        self.assertEqual(_start_sides(anchors, ("left",)), ())


if __name__ == "__main__":
    unittest.main()
