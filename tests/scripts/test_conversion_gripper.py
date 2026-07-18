import argparse
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from handumi.dataset.raw import LEFT_GRIPPER_INDEX, RIGHT_GRIPPER_INDEX
from handumi.scripts.conversion import (
    _resolve_cli_profile,
    _resolve_conversion_tcp_calibration,
    _piper_command_states_from_rollout,
    _write_gripper_joints,
    build_parser,
    process_episode,
)


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


def _tcp_snapshot(*, identity_bound: bool) -> dict[str, object]:
    snapshot: dict[str, object] = {
        "sha256": "dataset-sha",
        "applied_to_state": False,
        "controller_to_gripper_tcp": {
            "left": {
                "position": [0.01, 0.02, 0.03],
                "quaternion": [0.0, 0.0, 0.0, 1.0],
            },
            "right": {
                "position": [-0.01, -0.02, -0.03],
                "quaternion": [0.0, 0.0, 0.0, 1.0],
            },
        },
    }
    if identity_bound:
        snapshot.update(
            {
                "schema_version": 2,
                "source_robot": "piper",
                "source_gripper": "piper_parallel_v1",
                "tracking_device": "meta",
                "controller_mount": "handumi_v1",
            }
        )
    return snapshot


class ConversionTcpCalibrationTest(unittest.TestCase):
    @staticmethod
    def _args() -> argparse.Namespace:
        return argparse.Namespace(
            controller_device=None,
            controller_tcp_calibration=None,
            embodiment="axol",
        )

    def test_identity_bound_source_snapshot_precedes_target_embodiment(self):
        args = self._args()
        info = {
            "handumi": {
                "recording_device": "meta",
                "target_robot": {"name": "piper"},
                "controller_tcp_calibration": _tcp_snapshot(identity_bound=True),
            }
        }

        selection = _resolve_conversion_tcp_calibration(args, info)

        np.testing.assert_allclose(selection.calibration.left[:3], [0.01, 0.02, 0.03])
        self.assertTrue(selection.metadata["applied_to_state"])
        self.assertEqual(selection.metadata["source_robot"], "piper")
        self.assertEqual(args.controller_device, "meta")
        self.assertTrue(selection.source.startswith("dataset robot-tool snapshot"))

    def test_legacy_snapshot_uses_source_piper_setup_before_snapshot(self):
        args = self._args()
        info = {
            "handumi": {
                "recording_device": "meta",
                "target_robot": {"name": "piper"},
                "controller_tcp_calibration": _tcp_snapshot(identity_bound=False),
            }
        }

        selection = _resolve_conversion_tcp_calibration(args, info)

        np.testing.assert_allclose(
            selection.calibration.left[:3],
            [0.12068467, 0.02142489, -0.21669616],
        )
        self.assertEqual(selection.metadata["source_gripper"], "piper_parallel_v1")
        self.assertTrue(selection.source.startswith("configured piper/meta:"))

    def test_device_override_cannot_contradict_identity_bound_snapshot(self):
        args = self._args()
        args.controller_device = "pico"
        info = {
            "handumi": {
                "controller_tcp_calibration": _tcp_snapshot(identity_bound=True),
            }
        }

        with self.assertRaisesRegex(ValueError, "conflicts"):
            _resolve_conversion_tcp_calibration(args, info)


class ConversionProfileTest(unittest.TestCase):
    def test_piper_profile_selects_replay_parity_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["--piper"])

        _resolve_cli_profile(parser, args)

        self.assertEqual(args.embodiment, "piper")
        self.assertEqual(args.retarget_mode, "absolute-table")
        self.assertEqual(
            args.deployment_calibration,
            Path("configs/calibration/piper_table.yaml"),
        )
        self.assertIsNone(args.gripper_max_width_m)

    def test_piper_profile_rejects_non_parity_retarget_mode(self):
        parser = build_parser()
        args = parser.parse_args(["--piper", "--retarget-mode", "local-relative"])

        with self.assertRaises(SystemExit):
            _resolve_cli_profile(parser, args)

    def test_default_profile_remains_axol_local_relative(self):
        parser = build_parser()
        args = parser.parse_args([])

        _resolve_cli_profile(parser, args)

        self.assertEqual(args.embodiment, "axol")
        self.assertEqual(args.retarget_mode, "local-relative")

    def test_absolute_table_dataset_pairs_are_exact_replay_qpos(self):
        qpos = np.arange(5 * 16, dtype=np.float32).reshape(5, 16)
        rollout = {
            "qpos": qpos,
            "retarget_mode": np.asarray(["absolute-table"]),
            "left_pos_error_m": np.zeros(5, dtype=np.float32),
            "right_pos_error_m": np.zeros(5, dtype=np.float32),
            "left_rot_error_deg": np.zeros(5, dtype=np.float32),
            "right_rot_error_deg": np.zeros(5, dtype=np.float32),
            "initial_solve_iterations": np.asarray([3]),
            "gripper_source": np.asarray(["recorded Feetech normalized"]),
        }
        args = argparse.Namespace(retarget_mode="absolute-table", ik_reports=[])

        with patch(
            "handumi.scripts.conversion._solve_with_replay_pipeline",
            return_value=rollout,
        ):
            result = process_episode(
                args=args,
                states=np.zeros((5, 16), dtype=np.float32),
                episode_index=0,
                source_episode_index=2,
                task="test",
            )

        np.testing.assert_array_equal(result.states, qpos[:-1])
        np.testing.assert_array_equal(result.actions, qpos[1:])

    def test_piper_layout_has_six_arm_joints_and_one_width_per_side(self):
        qpos = np.arange(3 * 16, dtype=np.float32).reshape(3, 16)
        normalized = np.asarray([[0.0, 1.0], [0.25, 0.5], [1.0, 0.0]])
        names = [
            *(f"left_joint{i}" for i in range(1, 9)),
            *(f"right_joint{i}" for i in range(1, 9)),
        ]

        commands = _piper_command_states_from_rollout(
            {"qpos": qpos, "gripper_normalized": normalized},
            actuated_names=names,
            gripper_max_width_m=0.066,
        )

        self.assertEqual(commands.shape, (3, 14))
        np.testing.assert_array_equal(commands[:, :6], qpos[:, :6])
        np.testing.assert_array_equal(commands[:, 7:13], qpos[:, 8:14])
        np.testing.assert_allclose(commands[:, 6], [0.0, 0.0165, 0.066])
        np.testing.assert_allclose(commands[:, 13], [0.066, 0.033, 0.0])


if __name__ == "__main__":
    unittest.main()
