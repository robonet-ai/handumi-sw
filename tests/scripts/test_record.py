import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from handumi.feetech import GripperWidths
from handumi.scripts.record import (
    _default_output_dir,
    _wait_for_clap,
    _wait_for_tracking,
    build_observation,
    record_episode,
)
from handumi.tracking.base import ControllerPairSample
from handumi.tracking.gestures import DoubleClapDetector


def _widths(left_mm: float, right_mm: float) -> GripperWidths:
    return GripperWidths(
        left=left_mm / 1000.0, right=right_mm / 1000.0,
        left_mm=left_mm, right_mm=right_mm,
        left_normalized=left_mm / 80.0, right_normalized=right_mm / 80.0,
        left_ticks=0, right_ticks=0,
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


def _clap_sequence() -> list[GripperWidths]:
    """open, clap, open, clap -> triggers the double-clap detector."""
    return [
        _widths(50.0, 50.0), _widths(2.0, 2.0),
        _widths(50.0, 50.0), _widths(2.0, 2.0),
        _widths(50.0, 50.0),
    ]


class DefaultOutputDirTest(unittest.TestCase):
    def test_is_timestamped_under_outputs(self):
        out = _default_output_dir()
        self.assertEqual(out.parent, Path("outputs"))
        self.assertRegex(out.name, r"^\d{8}_\d{6}$")


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
        self.assertFalse(
            _wait_for_clap(grippers, DoubleClapDetector(), stop)
        )


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
    def test_double_clap_stops_the_episode(self):
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
        self.assertEqual(status, "recorded")
        # The clap frames themselves are not recorded; only pre-clap ones.
        self.assertGreaterEqual(n_frames, 1)
        self.assertLess(n_frames, 10)
        for frame in dataset.frames:
            self.assertIn("observation.state", frame)
            self.assertEqual(frame["observation.state"].dtype, np.float32)


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
