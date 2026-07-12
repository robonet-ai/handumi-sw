import threading
import argparse
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from handumi.cameras.base import CameraSample
from handumi.feetech import GripperWidths
from handumi.scripts.record import (
    _default_output_dir,
    _robot_metadata,
    _selected_camera_names,
    _validate_unique_camera_ids,
    _wait_for_clap,
    _wait_for_tracking,
    build_observation,
    record_episode,
)
from handumi.tracking.base import ControllerPairSample
from handumi.tracking.gestures import DoubleClapDetector


def _widths(left_mm: float, right_mm: float) -> GripperWidths:
    return GripperWidths(
        left=left_mm / 1000.0,
        right=right_mm / 1000.0,
        left_mm=left_mm,
        right_mm=right_mm,
        left_normalized=left_mm / 80.0,
        right_normalized=right_mm / 80.0,
        left_ticks=0,
        right_ticks=0,
    )


class _FakeGrippers:
    """Feeds a scripted sequence of widths; repeats the last one forever."""

    def __init__(self, sequence: list[GripperWidths]):
        self._sequence = list(sequence)

    def read_normalized_widths(self) -> GripperWidths:
        if len(self._sequence) > 1:
            return self._sequence.pop(0)
        return self._sequence[0]


class _FakeTracker:
    device = "meta"

    def latest(self) -> ControllerPairSample:
        return ControllerPairSample.empty("meta")


class _ScriptedTracker:
    device = "meta"

    def __init__(self, tracked: list[bool]):
        self._tracked = list(tracked)

    def latest(self) -> ControllerPairSample:
        value = self._tracked.pop(0) if len(self._tracked) > 1 else self._tracked[0]
        return replace(
            ControllerPairSample.empty("meta"),
            left_tracked=value,
            right_tracked=value,
        )


class _FakeDataset:
    def __init__(self):
        self.frames: list[dict] = []

    def add_frame(self, frame: dict) -> None:
        self.frames.append(frame)

    def clear_episode_buffer(self) -> None:
        self.frames.clear()


class _HealthyTracker:
    device = "meta"

    def sample_at(self, target_time_ns: int) -> ControllerPairSample:
        return replace(
            ControllerPairSample.empty("meta"),
            left_tracked=True,
            right_tracked=True,
            left_device_tracked=True,
            right_device_tracked=True,
            left_pose_valid=True,
            right_pose_valid=True,
            aligned_time_ns=target_time_ns,
            pc_monotonic_ns=target_time_ns,
            connected=True,
            streaming=True,
        )

    def latest(self) -> ControllerPairSample:
        return self.sample_at(1)


class _StaleCamera:
    def sample_at(self, target_time_ns: int) -> CameraSample:
        return CameraSample(
            image=np.zeros((48, 64, 3), dtype=np.uint8),
            capture_time_ns=target_time_ns - 1_000_000_000,
            sequence=1,
        )


def _clap_sequence() -> list[GripperWidths]:
    """open, clap, open, clap -> triggers the double-clap detector."""
    return [
        _widths(50.0, 50.0),
        _widths(2.0, 2.0),
        _widths(50.0, 50.0),
        _widths(2.0, 2.0),
        _widths(50.0, 50.0),
    ]


class DefaultOutputDirTest(unittest.TestCase):
    def test_is_timestamped_under_outputs(self):
        out = _default_output_dir()
        self.assertEqual(out.parent, Path("outputs"))
        self.assertRegex(out.name, r"^\d{8}_\d{6}$")


class CameraSelectionTest(unittest.TestCase):
    @staticmethod
    def _args(**overrides):
        values = {
            "wrist_cameras": False,
            "workspace_camera": False,
            "only_left_camera": False,
            "only_right_camera": False,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_defaults_to_both_wrist_cameras(self):
        self.assertEqual(
            _selected_camera_names(self._args()),
            ["left_wrist", "right_wrist"],
        )

    def test_all_three_cameras_can_be_selected(self):
        self.assertEqual(
            _selected_camera_names(
                self._args(wrist_cameras=True, workspace_camera=True)
            ),
            ["left_wrist", "right_wrist", "workspace"],
        )

    def test_only_right_camera(self):
        self.assertEqual(
            _selected_camera_names(self._args(only_right_camera=True)),
            ["right_wrist"],
        )

    def test_duplicate_camera_devices_are_rejected(self):
        with self.assertRaises(SystemExit):
            _validate_unique_camera_ids(
                ["right_wrist", "workspace"],
                [4, 4],
            )


class RobotMetadataTest(unittest.TestCase):
    def test_snapshots_robot_config_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            path = config_dir / "piper.yaml"
            path.write_text("kind: piper\nhome_q: [0.0]\n")

            metadata = _robot_metadata("piper", config_dir)

        self.assertEqual(metadata["name"], "piper")
        self.assertEqual(metadata["configuration"]["kind"], "piper")
        self.assertEqual(len(metadata["sha256"]), 64)


class WaitForClapTest(unittest.TestCase):
    def test_returns_true_on_double_clap(self):
        grippers = _FakeGrippers(_clap_sequence())
        self.assertTrue(
            _wait_for_clap(grippers, DoubleClapDetector(), threading.Event())
        )

    def test_returns_false_when_stopped(self):
        stop = threading.Event()
        stop.set()
        grippers = _FakeGrippers([_widths(50.0, 50.0)])
        self.assertFalse(_wait_for_clap(grippers, DoubleClapDetector(), stop))


class WaitForTrackingTest(unittest.TestCase):
    def test_waits_until_both_controllers_are_tracked(self):
        tracker = _ScriptedTracker([False, True])
        with mock.patch("handumi.scripts.record.time.sleep"):
            ready = _wait_for_tracking(tracker, threading.Event(), poll_s=0.0)
        self.assertTrue(ready)

    def test_returns_false_when_stopped(self):
        stop = threading.Event()
        stop.set()
        self.assertFalse(_wait_for_tracking(_ScriptedTracker([True]), stop))


class RecordEpisodeClapControlTest(unittest.TestCase):
    def test_double_clap_restarts_the_episode(self):
        dataset = _FakeDataset()
        n_frames, status = record_episode(
            dataset=dataset,
            cameras=[],
            cam_names=[],
            tracker=_FakeTracker(),
            grippers=_FakeGrippers(_clap_sequence()),
            episode_time_s=9999.0,  # would never end on the timer
            fps=1000,
            task="test",
            cam_width=64,
            cam_height=48,
            stop_event=threading.Event(),
            manual_control=False,
            start_button="enter",
            repeat_button="B",
            finish_button="Y",
            start_threshold=0.75,
            clap_detector=DoubleClapDetector(),
        )
        self.assertEqual(status, "repeat")
        self.assertGreaterEqual(n_frames, 0)
        self.assertEqual(dataset.frames, [])


class RecordEpisodeTrackingGateTest(unittest.TestCase):
    def test_sustained_tracking_loss_discards_episode(self):
        dataset = _FakeDataset()
        with (
            mock.patch(
                "handumi.scripts.record.time.perf_counter",
                side_effect=[0.0, 0.0, 0.0, 2.0],
            ),
            mock.patch(
                "handumi.scripts.record.time.monotonic_ns",
                side_effect=[0, 2_000_000_000],
            ),
            mock.patch("handumi.scripts.record.time.sleep"),
        ):
            n_frames, status = record_episode(
                dataset=dataset,
                cameras=[],
                cam_names=[],
                tracker=_FakeTracker(),
                grippers=None,
                episode_time_s=1.0,
                fps=1000,
                task="test",
                cam_width=64,
                cam_height=48,
                stop_event=threading.Event(),
                manual_control=False,
                start_button="enter",
                repeat_button="B",
                finish_button="Y",
                start_threshold=0.75,
                tracking_loss_timeout_s=1.0,
            )

        self.assertEqual(status, "tracking_lost")
        self.assertEqual(n_frames, 1)
        self.assertEqual(len(dataset.frames), n_frames)

    def test_sustained_camera_failure_discards_episode(self):
        dataset = _FakeDataset()
        n_frames, status = record_episode(
            dataset=dataset,
            cameras=[_StaleCamera()],
            cam_names=["left_wrist"],
            tracker=_HealthyTracker(),
            grippers=None,
            episode_time_s=1.0,
            fps=1000,
            task="test",
            cam_width=64,
            cam_height=48,
            stop_event=threading.Event(),
            manual_control=False,
            start_button="enter",
            repeat_button="B",
            finish_button="Y",
            start_threshold=0.75,
            sync_lag_s=0.001,
            max_sync_skew_s=0.01,
            camera_stale_timeout_s=0.1,
            sensor_loss_timeout_s=0.005,
        )

        self.assertEqual(status, "sensor_unhealthy")
        self.assertGreater(n_frames, 0)


class BuildObservationTest(unittest.TestCase):
    def test_state_carries_widths_and_tracking_frame(self):
        obs = build_observation(ControllerPairSample.empty("meta"), _widths(11.0, 22.0))
        state = obs["observation.state"]
        self.assertEqual(state.shape, (16,))
        self.assertAlmostEqual(float(state[14]), 0.011, places=5)
        self.assertAlmostEqual(float(state[15]), 0.022, places=5)
        self.assertIn("observation.tracking.left_controller_pose", obs)
        self.assertNotIn("observation.tracking.left_tcp_pose", obs)


if __name__ == "__main__":
    unittest.main()
