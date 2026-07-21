import threading
import time

import numpy as np

from handumi.teleop.recording_viewer import (
    QueuedRecorderRobotViewer,
    RecorderRobotFrame,
    RecorderRobotViewerConfig,
)


def _frame(sequence: int = 1) -> RecorderRobotFrame:
    left = np.array([sequence, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    right = np.array([0, sequence, 0, 0, 0, 0, 1], dtype=np.float32)
    return RecorderRobotFrame(
        sample_time_ns=sequence,
        left_tcp_pose=left,
        right_tcp_pose=right,
        left_tracked=True,
        right_tracked=True,
        left_gripper_opening=0.25,
        right_gripper_opening=0.75,
    )


class _FakePipeline:
    def __init__(self, *, gate: threading.Event | None = None) -> None:
        self.gate = gate
        self.started = threading.Event()
        self.frames: list[RecorderRobotFrame] = []
        self.states: list[tuple[str, str]] = []
        self.failures: list[str] = []
        self.closed = False

    def set_recording_state(self, state: str, detail: str) -> None:
        self.states.append((state, detail))

    def process(self, frame: RecorderRobotFrame) -> tuple[str, str, str]:
        self.started.set()
        if self.gate is not None:
            self.gate.wait(timeout=2.0)
        self.frames.append(frame)
        return "anchored", "both-tracked", "solved"

    def mark_failure(self, message: str) -> None:
        self.failures.append(message)

    def close(self) -> None:
        self.closed = True


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


def test_worker_copies_aligned_input_and_reports_status():
    pipeline = _FakePipeline()
    viewer = QueuedRecorderRobotViewer(
        RecorderRobotViewerConfig(queue_size=2),
        pipeline_factory=lambda _config: pipeline,
    )
    frame = _frame()

    viewer.set_recording_state("waiting", "episode 1")
    viewer.set_recording_state("recording", "episode 1")
    assert viewer.submit(frame)
    frame.left_tcp_pose[0] = 99.0
    assert _wait_until(lambda: len(pipeline.frames) == 1)

    status = viewer.status()
    assert status.lifecycle == "ready"
    assert status.submitted_frames == 1
    assert status.processed_frames == 1
    assert status.anchor_state == "anchored"
    assert status.recording_state == "RECORDING"
    assert pipeline.states == [
        ("WAITING", "episode 1"),
        ("RECORDING", "episode 1"),
    ]
    np.testing.assert_allclose(pipeline.frames[0].left_tcp_pose[:3], [1, 0, 0])
    viewer.close()
    assert pipeline.closed
    assert viewer.status().lifecycle == "stopped"


def test_slow_pipeline_drops_stale_frames_without_blocking_submitter():
    release = threading.Event()
    pipeline = _FakePipeline(gate=release)
    viewer = QueuedRecorderRobotViewer(
        RecorderRobotViewerConfig(queue_size=2),
        pipeline_factory=lambda _config: pipeline,
    )

    assert viewer.submit(_frame(1))
    assert pipeline.started.wait(timeout=1.0)
    started = time.perf_counter()
    for sequence in range(2, 20):
        viewer.submit(_frame(sequence))
    elapsed = time.perf_counter() - started

    assert elapsed < 0.25
    assert viewer.status().dropped_frames > 0
    release.set()
    assert _wait_until(lambda: viewer.status().processed_frames >= 2)
    viewer.close()


def test_initialization_failure_is_nonfatal_and_observable():
    def fail(_config):
        raise RuntimeError("port occupied")

    viewer = QueuedRecorderRobotViewer(
        RecorderRobotViewerConfig(), pipeline_factory=fail
    )
    assert _wait_until(lambda: viewer.status().lifecycle == "failed")

    assert viewer.submit(_frame())
    status = viewer.status()
    assert status.failures == 1
    assert status.last_error == "RuntimeError: port occupied"
    viewer.close()


def test_per_frame_failure_does_not_kill_worker():
    class _FailOncePipeline(_FakePipeline):
        def process(self, frame):
            if not self.failures:
                raise RuntimeError("IK failed")
            return super().process(frame)

    pipeline = _FailOncePipeline()
    viewer = QueuedRecorderRobotViewer(
        RecorderRobotViewerConfig(queue_size=2),
        pipeline_factory=lambda _config: pipeline,
    )
    viewer.submit(_frame(1))
    assert _wait_until(lambda: viewer.status().failures == 1)
    viewer.submit(_frame(2))
    assert _wait_until(lambda: viewer.status().processed_frames == 1)

    assert pipeline.failures == ["IK failed"]
    assert viewer.status().lifecycle == "degraded"
    viewer.close()
