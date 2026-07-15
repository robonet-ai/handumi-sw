import argparse
import unittest

import numpy as np

from handumi.dataset.raw import LEFT_GRIPPER_INDEX, RIGHT_GRIPPER_INDEX
from handumi.scripts.conversion import (
    _resolve_conversion_tcp_calibration,
    _write_gripper_joints,
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


if __name__ == "__main__":
    unittest.main()
