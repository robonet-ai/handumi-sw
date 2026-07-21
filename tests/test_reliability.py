import errno
import json
import os
import threading
import time
from pathlib import Path

import pytest

from handumi.reliability import (
    BoundedLatestWorker,
    CaptureSession,
    CaptureStorageError,
    ShortWriteError,
    StageProfiler,
    _write_all,
    atomic_write_json,
    check_disk_space,
    recover_interrupted_sessions,
    resolve_capture_profile,
)
from handumi.scripts.soak import run_soak


def test_stage_profiler_tracks_latency_failure_timeout_drop_and_queue_depth():
    profiler = StageProfiler()
    with profiler.measure("camera", items=3):
        pass
    profiler.failure("camera")
    profiler.timeout("camera")
    profiler.drop("camera", 2)
    profiler.queue_depth("camera", 4)
    stages = profiler.snapshot()["stages"]
    assert isinstance(stages, dict)
    stage = stages["camera"]
    assert isinstance(stage, dict)
    assert stage["calls"] == 1
    assert stage["items"] == 3
    assert stage["failures"] == 1
    assert stage["timeouts"] == 1
    assert stage["drops"] == 2
    assert stage["queue_depth_max"] == 4


def test_bounded_worker_drops_without_blocking_and_closes_thread():
    profiler = StageProfiler()
    release = threading.Event()
    started = threading.Event()

    def process(_item: object) -> None:
        started.set()
        release.wait(1)

    worker = BoundedLatestWorker("viewer", process, maxsize=1, profiler=profiler)
    assert worker.submit(1)
    assert started.wait(1)
    assert worker.submit(2)
    assert worker.submit(3)
    release.set()
    assert worker.close()
    assert not any(thread.name == "viewer" for thread in threading.enumerate())
    stages = profiler.snapshot()["stages"]
    assert isinstance(stages, dict)
    viewer = stages["viewer"]
    assert isinstance(viewer, dict)
    drops = viewer["drops"]
    assert isinstance(drops, int)
    assert drops >= 1


def test_disk_check_and_enospc_short_write_are_predictable(tmp_path: Path, monkeypatch):
    assert check_disk_space(tmp_path, minimum_free_bytes=0) >= 0
    with pytest.raises(CaptureStorageError, match="insufficient"):
        check_disk_space(tmp_path, minimum_free_bytes=2**63)
    with pytest.raises(ShortWriteError):
        _write_all(1, b"payload", write=lambda _fd, _payload: 0)

    def no_space(_fd: int, _payload: bytes) -> int:
        raise OSError(errno.ENOSPC, "injected disk full")

    with pytest.raises(CaptureStorageError, match="exhausted"):
        _write_all(1, b"payload", write=no_space)


def test_atomic_manifest_and_complete_promotion_are_recoverable(tmp_path: Path):
    destination = tmp_path / "episode"
    session = CaptureSession(destination, resolve_capture_profile(10, 30, 3))
    (session.staging_root / "data.bin").write_bytes(b"synthetic")
    profiler = StageProfiler()
    promoted = session.complete(profiler)
    assert promoted == destination
    manifest = json.loads((destination / "session-manifest.json").read_text())
    assert manifest["completion_status"] == "complete"
    assert manifest["privacy"]["contains_absolute_paths"] is False
    assert manifest["produced_files"][0]["path"] == "data.bin"


def test_reject_and_crash_recovery_never_present_partial_as_complete(tmp_path: Path):
    rejected = CaptureSession(tmp_path / "bad", resolve_capture_profile(30, 30, 3))
    rejected_path = rejected.reject(StageProfiler(), reason="encoder_failure")
    rejected_manifest = json.loads(
        (rejected_path / "session-manifest.json").read_text()
    )
    assert rejected_manifest["completion_status"] == "rejected"

    interrupted = CaptureSession(
        tmp_path / "interrupted", resolve_capture_profile(10, 30, 3)
    )
    staging = interrupted.staging_root
    recovered = recover_interrupted_sessions(tmp_path)
    assert staging not in recovered
    assert len(recovered) == 1
    manifest = json.loads((recovered[0] / "session-manifest.json").read_text())
    assert manifest["completion_status"] == "incomplete"


def test_atomic_json_failure_does_not_publish_destination(tmp_path: Path, monkeypatch):
    destination = tmp_path / "manifest.json"

    def fail_replace(_source: os.PathLike[str], _target: os.PathLike[str]) -> None:
        raise OSError(errno.EIO, "injected partial filesystem failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        atomic_write_json(destination, {"safe": True})
    assert not destination.exists()


def test_short_software_soak_is_machine_readable_and_leak_bounded(tmp_path: Path):
    output = tmp_path / "soak.json"
    before_threads = {thread.ident for thread in threading.enumerate()}
    result = run_soak(
        duration_s=0.15,
        dataset_hz=30,
        camera_fps=30,
        camera_count=3,
        output=output,
    )
    rows = result["rows"]
    assert isinstance(rows, int)
    assert rows >= 3
    assert result["maintained_requested_profile"] is True
    assert json.loads(output.read_text())["schema"] == "handumi_software_soak_v1"
    time.sleep(0.02)
    assert {thread.ident for thread in threading.enumerate()} == before_threads
