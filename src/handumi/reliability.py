"""Capture reliability primitives for the research-preview recorder.

The manifest intentionally contains only logical labels, relative output paths,
hashes, versions, and aggregate counters.  It must never serialize rig values,
network addresses, device serials, participant identifiers, or absolute paths.
"""

from __future__ import annotations

import errno
import hashlib
import importlib.metadata
import json
import os
import platform
import queue
import shutil
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterator, Mapping

SESSION_SCHEMA = "handumi_capture_session_v1"
PROFILING_SCHEMA = "handumi_capture_profiling_v1"


class CaptureReliabilityError(RuntimeError):
    """Base class for predictable capture failures."""


class CaptureStorageError(CaptureReliabilityError):
    """Storage cannot safely accept more capture data."""


class ShortWriteError(CaptureStorageError):
    """A filesystem write returned zero bytes before completion."""


@dataclass(frozen=True)
class CaptureProfile:
    name: str
    dataset_hz: float
    camera_fps: float
    camera_count: int
    status: str
    evidence: str

    def validate(self) -> None:
        if self.dataset_hz <= 0 or self.camera_fps <= 0:
            raise ValueError("capture rates must be positive")
        if self.camera_count < 0:
            raise ValueError("camera_count cannot be negative")
        if self.status not in {"supported", "experimental", "unsupported"}:
            raise ValueError(f"invalid capture profile status: {self.status}")


SUPPORTED_CAPTURE_PROFILES: tuple[CaptureProfile, ...] = (
    CaptureProfile(
        name="rows10-cameras30x3",
        dataset_hz=10.0,
        camera_fps=30.0,
        camera_count=3,
        status="supported",
        evidence="software-only synthetic/headless soak; hardware remains unverified",
    ),
    CaptureProfile(
        name="rows30-cameras30x3",
        dataset_hz=30.0,
        camera_fps=30.0,
        camera_count=3,
        status="experimental",
        evidence="software-only synthetic/headless soak; fail if sustained row rate is below target",
    ),
)


def resolve_capture_profile(
    dataset_hz: float, camera_fps: float, camera_count: int
) -> CaptureProfile:
    for profile in SUPPORTED_CAPTURE_PROFILES:
        if (
            abs(profile.dataset_hz - dataset_hz) < 1e-6
            and abs(profile.camera_fps - camera_fps) < 1e-6
            and profile.camera_count == camera_count
        ):
            return profile
    return CaptureProfile(
        name=f"custom-{dataset_hz:g}hz-{camera_fps:g}fps-{camera_count}cam",
        dataset_hz=dataset_hz,
        camera_fps=camera_fps,
        camera_count=camera_count,
        status="experimental",
        evidence="no retained soak evidence for this exact profile",
    )


@dataclass
class StageCounter:
    calls: int = 0
    failures: int = 0
    timeouts: int = 0
    drops: int = 0
    items: int = 0
    latency_total_ns: int = 0
    latency_max_ns: int = 0
    queue_depth_max: int = 0

    def snapshot(self, elapsed_s: float) -> dict[str, int | float]:
        average_ms = self.latency_total_ns / self.calls / 1e6 if self.calls else 0.0
        return {
            **asdict(self),
            "latency_average_ms": average_ms,
            "latency_max_ms": self.latency_max_ns / 1e6,
            "throughput_items_s": self.items / elapsed_s if elapsed_s > 0 else 0.0,
        }


class StageProfiler:
    """Thread-safe monotonic stage metrics with explicit failure counters."""

    def __init__(self) -> None:
        self.started_monotonic_ns = time.monotonic_ns()
        self._stages: dict[str, StageCounter] = {}
        self._lock = threading.Lock()

    @contextmanager
    def measure(self, stage: str, *, items: int = 1) -> Iterator[None]:
        started = time.monotonic_ns()
        try:
            yield
        except BaseException:
            self.failure(stage)
            raise
        finally:
            elapsed = max(0, time.monotonic_ns() - started)
            with self._lock:
                counter = self._stages.setdefault(stage, StageCounter())
                counter.calls += 1
                counter.items += max(0, items)
                counter.latency_total_ns += elapsed
                counter.latency_max_ns = max(counter.latency_max_ns, elapsed)

    def failure(self, stage: str, count: int = 1) -> None:
        with self._lock:
            self._stages.setdefault(stage, StageCounter()).failures += count

    def timeout(self, stage: str, count: int = 1) -> None:
        with self._lock:
            self._stages.setdefault(stage, StageCounter()).timeouts += count

    def drop(self, stage: str, count: int = 1) -> None:
        with self._lock:
            self._stages.setdefault(stage, StageCounter()).drops += count

    def queue_depth(self, stage: str, depth: int) -> None:
        with self._lock:
            counter = self._stages.setdefault(stage, StageCounter())
            counter.queue_depth_max = max(counter.queue_depth_max, max(0, depth))

    def snapshot(self) -> dict[str, object]:
        elapsed_s = max(0.0, (time.monotonic_ns() - self.started_monotonic_ns) / 1e9)
        with self._lock:
            stages = {
                name: counter.snapshot(elapsed_s)
                for name, counter in sorted(self._stages.items())
            }
        return {
            "schema": PROFILING_SCHEMA,
            "elapsed_s": elapsed_s,
            "stages": stages,
        }


class BoundedLatestWorker:
    """A nonessential worker that drops stale work and never blocks capture."""

    def __init__(
        self,
        name: str,
        process: Callable[[object], None],
        *,
        maxsize: int,
        profiler: StageProfiler,
    ) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self.name = name
        self._process = process
        self._profiler = profiler
        self._queue: queue.Queue[object] = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, item: object) -> bool:
        if self._stop.is_set():
            return False
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                self._profiler.drop(self.name)
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                self._profiler.drop(self.name)
                return False
        self._profiler.queue_depth(self.name, self._queue.qsize())
        return True

    def close(self, timeout_s: float = 5.0) -> bool:
        self._stop.set()
        self._thread.join(timeout=max(0.0, timeout_s))
        if self._thread.is_alive():
            self._profiler.timeout(self.name)
            return False
        return True

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                with self._profiler.measure(self.name):
                    self._process(item)
            except Exception:
                # The profiler records the failure. Nonessential workers are isolated.
                pass
            finally:
                self._queue.task_done()


def check_disk_space(path: Path, *, minimum_free_bytes: int) -> int:
    """Return available bytes or raise before a capture can exhaust storage."""
    if minimum_free_bytes < 0:
        raise ValueError("minimum_free_bytes cannot be negative")
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError as exc:
        raise CaptureStorageError(
            f"cannot inspect storage for capture root: {exc}"
        ) from exc
    if free < minimum_free_bytes:
        raise CaptureStorageError(
            f"insufficient free space: {free} bytes available, {minimum_free_bytes} required"
        )
    return free


def _write_all(
    fd: int, payload: bytes, write: Callable[[int, bytes], int] = os.write
) -> None:
    view = memoryview(payload)
    while view:
        try:
            written = write(fd, view.tobytes())
        except OSError as exc:
            if exc.errno == errno.ENOSPC:
                raise CaptureStorageError(
                    "storage exhausted while writing capture metadata"
                ) from exc
            raise CaptureStorageError(f"capture metadata write failed: {exc}") from exc
        if written <= 0:
            raise ShortWriteError("filesystem returned a zero-length short write")
        view = view[written:]


def atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        _write_all(fd, payload)
        os.fsync(fd)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    finally:
        os.close(fd)
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_configuration(label: str, path: Path | None) -> dict[str, object]:
    """Hash a configuration without publishing its machine-local path or values."""
    if path is None:
        return {"label": label, "present": False}
    try:
        digest = sha256_file(path)
    except OSError:
        return {"label": label, "present": False}
    return {"label": label, "present": True, "sha256": digest}


def source_commit() -> str:
    """Return the checked-out source revision without exposing repository paths."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for distribution in ("handumi", "numpy", "opencv-python", "rerun-sdk", "lerobot"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


@dataclass
class CaptureSession:
    """Crash-visible session state and atomic dataset-directory promotion."""

    requested_root: Path
    profile: CaptureProfile
    configuration_hashes: list[dict[str, object]] = field(default_factory=list)
    calibration_hashes: list[dict[str, object]] = field(default_factory=list)
    source_commit: str = field(default_factory=source_commit)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    viewer_failures: list[str] = field(default_factory=list)
    defer_initialization: bool = False
    _initialized: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.profile.validate()
        self.requested_root = Path(self.requested_root)
        self.staging_root = self.requested_root.with_name(
            f".{self.requested_root.name}.handumi-inprogress-{self.session_id}"
        )
        if not self.defer_initialization:
            self.initialize()

    def initialize(self) -> None:
        """Make the reserved staging directory crash-visible.

        Dataset implementations such as LeRobot require their root not to exist
        when ``create`` is called. A recorder can therefore reserve the unique
        name with ``defer_initialization=True``, let the dataset create it, and
        call this method immediately afterwards. The default remains eager for
        callers that write files directly.
        """
        if self._initialized:
            return
        self.staging_root.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            raise CaptureStorageError(
                "reserved capture staging directory already has a session manifest"
            )
        self._write_state("incomplete", profiler=None, reason="capture_started")
        self._initialized = True

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise CaptureStorageError("capture session has not been initialized")

    @property
    def manifest_path(self) -> Path:
        return self.staging_root / "session-manifest.json"

    def checkpoint(self, profiler: StageProfiler, *, reason: str = "periodic") -> None:
        self._require_initialized()
        self._write_state("incomplete", profiler=profiler, reason=reason)

    def reject(self, profiler: StageProfiler, *, reason: str) -> Path:
        self._require_initialized()
        self._write_state("rejected", profiler=profiler, reason=reason)
        rejected = self.staging_root.with_name(
            self.staging_root.name.replace("handumi-inprogress", "handumi-rejected")
        )
        os.replace(self.staging_root, rejected)
        self.staging_root = rejected
        return rejected

    def complete(self, profiler: StageProfiler) -> Path:
        self._require_initialized()
        if self.requested_root.exists():
            raise CaptureStorageError(
                f"capture destination already exists: {self.requested_root.name}"
            )
        self._write_state(
            "complete", profiler=profiler, reason="finalization_succeeded"
        )
        os.replace(self.staging_root, self.requested_root)
        self.staging_root = self.requested_root
        return self.requested_root

    def _write_state(
        self,
        status: str,
        *,
        profiler: StageProfiler | None,
        reason: str,
    ) -> None:
        produced: list[dict[str, object]] = []
        if status == "complete":
            for path in sorted(self.staging_root.rglob("*")):
                if path.is_file() and path != self.manifest_path:
                    produced.append(
                        {
                            "path": path.relative_to(self.staging_root).as_posix(),
                            "size_bytes": path.stat().st_size,
                            "sha256": sha256_file(path),
                        }
                    )
        manifest: dict[str, object] = {
            "schema": SESSION_SCHEMA,
            "session_id": self.session_id,
            "source_commit": self.source_commit,
            "runtime_versions": runtime_versions(),
            "schema_versions": {
                "session": SESSION_SCHEMA,
                "profiling": PROFILING_SCHEMA,
            },
            "capture_profile": asdict(self.profile),
            "configuration_hashes": self.configuration_hashes,
            "calibration_hashes": self.calibration_hashes,
            "started_at": self.started_at,
            "ended_at": datetime.now(UTC).isoformat()
            if status != "incomplete"
            else None,
            "completion_status": status,
            "reason": reason,
            "viewer_failures": list(self.viewer_failures),
            "profiling": profiler.snapshot() if profiler is not None else None,
            "produced_files": produced,
            "privacy": {
                "contains_network_addresses": False,
                "contains_device_serials": False,
                "contains_participant_identifiers": False,
                "contains_absolute_paths": False,
            },
        }
        atomic_write_json(self.manifest_path, manifest)


def recover_interrupted_sessions(parent: Path) -> tuple[Path, ...]:
    """Preserve stale staging directories as incomplete forensic evidence."""
    recovered: list[Path] = []
    for path in sorted(parent.glob(".*.handumi-inprogress-*")):
        if not path.is_dir():
            continue
        target = path.with_name(
            path.name.replace("handumi-inprogress", "handumi-recovered")
        )
        suffix = 1
        while target.exists():
            target = target.with_name(f"{target.name}-{suffix}")
            suffix += 1
        os.replace(path, target)
        recovered.append(target)
    return tuple(recovered)


def process_resource_snapshot() -> dict[str, int]:
    rss_bytes = 0
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        rss_bytes = pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        pass
    try:
        fd_count = len(tuple(Path("/proc/self/fd").iterdir()))
    except OSError:
        fd_count = -1
    return {"rss_bytes": rss_bytes, "file_descriptors": fd_count}


def write_publishable_session_manifest(
    root: Path,
    *,
    profiler: StageProfiler,
    profile: CaptureProfile,
    status: str,
    configuration_hashes: list[dict[str, object]],
    calibration_hashes: list[dict[str, object]],
    viewer_failures: list[str],
    started_at: str,
    reason: str,
) -> Path:
    """Write the recorder manifest without leaking machine or participant data."""
    if status not in {"complete", "incomplete", "rejected"}:
        raise ValueError("invalid session completion status")
    manifest_path = root / "session-manifest.json"
    produced: list[dict[str, object]] = []
    if status == "complete":
        for path in sorted(root.rglob("*")):
            if path.is_file() and path != manifest_path:
                produced.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    value: dict[str, object] = {
        "schema": SESSION_SCHEMA,
        "source_commit": source_commit(),
        "runtime_versions": runtime_versions(),
        "configuration_hashes": configuration_hashes,
        "calibration_hashes": calibration_hashes,
        "schema_versions": {
            "session": SESSION_SCHEMA,
            "profiling": PROFILING_SCHEMA,
        },
        "capture_profile": asdict(profile),
        "started_at": started_at,
        "ended_at": datetime.now(UTC).isoformat(),
        "completion_status": status,
        "reason": reason,
        "viewer_failures": viewer_failures,
        "profiling": profiler.snapshot(),
        "produced_files": produced,
        "privacy": {
            "contains_network_addresses": False,
            "contains_device_serials": False,
            "contains_participant_identifiers": False,
            "contains_absolute_paths": False,
        },
    }
    atomic_write_json(manifest_path, value)
    return manifest_path
