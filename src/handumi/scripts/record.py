#!/usr/bin/env python3
"""Unified HandUMI recorder for PICO and Meta Quest tracking backends.

Episode control: timed by default (--episode-time-s), PICO buttons with
--manual-control, or hands-free with --clap-control (double-clap right to
start or stop/save; double-clap left while recording to restart the attempt).

Spoken status announcements ("Recording episode 3", "Episode 3 saved, 812
frames", ...) are on by default — pass --no-sounds to disable them. Without
--output-dir each run writes a fresh outputs/<YYYYMMDD_HHMMSS>/ folder
(outputs/ is gitignored).
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import logging
import os
import queue
import select
import signal
import sys
import termios
import threading
import time
import tty
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence, cast

import numpy as np
import yaml

from handumi.body import (
    AnthropometricTable,
    BodyProfile,
    CanonicalBodyFrame,
    KinematicComEstimator,
    NeutralCalibrationCapture,
    ProfileConstrainedSkeleton,
    ProfileNeutralCalibration,
    canonical_body_features,
    canonical_body_from_packet,
    canonical_body_metadata,
    estimate_profile_neutral_calibration,
    persist_neutral_calibration_capture,
    validate_neutral_capture,
)
from handumi.calibration.control_tcp import (
    ControllerTcpCalibration,
    calibration_path_for_device,
    controller_tcp_calibration_metadata,
)
from handumi.calibration.spatial import (
    session_calibration_metadata,
    session_table_from_device,
)
from handumi.cameras import (
    build_camera_specs,
    connect_cameras,
    disconnect_cameras,
    read_camera_samples,
    resolve_camera_ids,
)
from handumi.config import DEFAULT_RIG_CONFIG
from handumi.dataset.raw import (
    HANDUMI_CAPTURE_SCHEMA,
    HANDUMI_STATE_SEMANTICS,
    HANDUMI_TRACKING_SCHEMA,
    camera_health_features,
    capture_timing_features,
    feetech_features,
    pose_to_state_vector,
    raw_state_feature,
    raw_tracking_features,
)
from handumi.dataset.tracking_sidecar import TrackingSidecarWriter
from handumi.feetech import (
    FeetechGripperPair,
    FeetechGripperSampler,
    GripperWidths,
    assert_calibrated,
    load_config,
    user_calibration_path,
    zero_gripper_widths,
)
from handumi.feetech.gripper import GripperWidthSource
from handumi.feetech.bus import FeetechUnavailableError
from handumi.robots.utils import IDENTITY_POSE7
from handumi.reliability import (
    CaptureSession,
    CaptureStorageError,
    StageProfiler,
    check_disk_space,
    hash_configuration,
    recover_interrupted_sessions,
    resolve_capture_profile,
)
from handumi.synchronization import (
    SustainedHealthGate,
    capture_timing_frame,
    synchronized_gripper_frame,
    tracking_sample_at,
)
from handumi.teleop.recording_viewer import (
    QueuedRecorderRobotViewer,
    RecorderRobotFrame,
    RecorderRobotSink,
    RecorderRobotViewerConfig,
)
from handumi.tracking.base import (
    ControllerPairSample,
    TrackingProvider,
    TrackingSampleSource,
)
from handumi.tracking.gestures import DoubleClapDetector
from handumi.tracking.meta_quest import MetaQuestConfig, MetaQuestTrackingProvider
from handumi.tracking.packet import TrackingPacket
from handumi.tracking.pico import (
    START_BUTTON_CHOICES,
    PicoTrackingProvider,
    read_start_button_value,
    wait_for_manual_start,
    wait_for_start_button,
)
from handumi.tracking.transforms import HandumiWorldCalibration, Pose
from handumi.utils.speech import log_say
from handumi.visualization.controller_trajectory import initialize_rerun

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("handumi.record")

ROBOT_CONFIG_DIR = Path("configs/robots")


class _TrackingSidecarSink(Protocol):
    def drain_provider(self, provider: object) -> int: ...

    def consume_frame_epoch_change(self) -> object | None: ...

    def nearest_packet(self, target_time_ns: int) -> TrackingPacket | None: ...


class _BodyEstimator(Protocol):
    def estimate(self, frame: CanonicalBodyFrame) -> CanonicalBodyFrame: ...


class _RerunSink(Protocol):
    def log(
        self,
        cam_frames: dict,
        sample: ControllerPairSample,
        widths: GripperWidths,
        *,
        body_frame: CanonicalBodyFrame | None = None,
    ) -> None: ...


class _RobotFrameSink(Protocol):
    def submit(self, frame: RecorderRobotFrame) -> bool: ...


class _EscapeStopListener:
    """Turn terminal Escape into the recorder's graceful stop event."""

    def __init__(self, stop_event: threading.Event, fd: int | None = None) -> None:
        self.stop_event = stop_event
        self.fd = fd
        self._original: list | None = None
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        fd = self.fd
        if fd is None:
            if not sys.stdin.isatty():
                log.warning("Escape stop disabled: stdin is not a terminal.")
                return False
            fd = sys.stdin.fileno()
            self.fd = fd
        if not os.isatty(fd):
            return False
        self._original = termios.tcgetattr(fd)
        tty.setcbreak(fd, termios.TCSANOW)
        self._thread = threading.Thread(
            target=self._run,
            name="handumi_escape_stop",
            daemon=True,
        )
        self._thread.start()
        return True

    def _run(self) -> None:
        assert self.fd is not None
        while not self._closed.is_set() and not self.stop_event.is_set():
            readable, _, _ = select.select([self.fd], [], [], 0.1)
            if readable and os.read(self.fd, 1) == b"\x1b":
                log.info("Escape pressed - discarding active episode and stopping ...")
                self.stop_event.set()
                return

    def stop(self) -> None:
        self._closed.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.fd is not None and self._original is not None:
            termios.tcsetattr(self.fd, termios.TCSANOW, self._original)
            self._original = None


def _wait_for_enter(
    stop_event: threading.Event,
    prompt: str,
    *,
    fd: int | None = None,
) -> bool:
    """Wait for a newline while remaining responsive to graceful-stop signals."""
    input_fd = sys.stdin.fileno() if fd is None else fd
    print(prompt, end="", flush=True)
    while not stop_event.is_set():
        readable, _, _ = select.select([input_fd], [], [], 0.1)
        if not readable:
            continue
        value = os.read(input_fd, 1)
        if value == b"":
            print()
            return False
        if value in (b"\n", b"\r"):
            return True
    print()
    return False


def build_features(
    cam_names: list[str],
    cam_width: int,
    cam_height: int,
    use_videos: bool,
    *,
    include_body: bool = True,
) -> dict:
    img_dtype = "video" if use_videos else "image"
    features: dict = {}
    for cam in cam_names:
        features[f"observation.images.{cam}"] = {
            "dtype": img_dtype,
            "shape": (cam_height, cam_width, 3),
            "names": ["height", "width", "channel"],
        }
    features["observation.state"] = _tuple_shape(raw_state_feature())
    features["action"] = _tuple_shape(raw_state_feature())
    features.update(feetech_features())
    features.update(raw_tracking_features())
    features.update(capture_timing_features())
    features.update(camera_health_features(cam_names))
    if include_body:
        features.update(canonical_body_features())
    return features


def _discard_tracking_backlog(provider: object) -> int:
    """Drop native-rate packets received before the recording gate opened."""
    drain = getattr(provider, "drain_packets", None)
    if not callable(drain):
        return 0
    packets = cast(Iterable[object], drain())
    try:
        discarded = len(cast(Sequence[object], packets))
    except TypeError:
        discarded = sum(1 for _ in packets)
    if discarded:
        log.info("Discarded %d pre-episode tracking packets.", discarded)
    return discarded


def _body_calibration_from_workspace(
    workspace_from_device_pose: np.ndarray,
    *,
    device: str,
    qualified: bool,
) -> HandumiWorldCalibration:
    """Use the controller workspace for body geometry and its source ground."""
    value = np.asarray(workspace_from_device_pose, dtype=np.float64).reshape(7)
    world_from_source = Pose(value[:3], value[3:])
    rotation = world_from_source.as_matrix()[:3, :3]
    normal = rotation @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    offset = -float(np.dot(normal, world_from_source.position))
    return HandumiWorldCalibration(
        world_from_source,
        ground_plane=np.concatenate([normal, [offset]]),
        source_frame=f"{device}_right_handed_source",
        qualified=qualified,
    )


def build_body_estimator(args: argparse.Namespace) -> KinematicComEstimator | None:
    profile_path = getattr(args, "body_profile", None)
    height_m = getattr(args, "body_height_m", None)
    mass_kg = getattr(args, "body_mass_kg", None)
    foot_length_m = getattr(args, "body_foot_length_m", None)
    foot_width_m = getattr(args, "body_foot_width_m", None)
    table_path = getattr(args, "anthropometric_table", None)

    direct_values = (height_m, mass_kg, foot_length_m, foot_width_m)
    if profile_path is not None and any(value is not None for value in direct_values):
        raise SystemExit(
            "--body-profile cannot be combined with direct --body-*-m values."
        )
    if profile_path is not None:
        try:
            profile = BodyProfile.from_yaml(profile_path)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"Invalid body profile: {exc}") from exc
    elif height_m is None and mass_kg is None:
        if (
            table_path is not None
            or foot_length_m is not None
            or foot_width_m is not None
        ):
            raise SystemExit(
                "Body dimensions and anthropometric tables require both "
                "--body-height-m and --body-mass-kg, or --body-profile."
            )
        return None
    else:
        if height_m is None or mass_kg is None:
            raise SystemExit(
                "--body-height-m and --body-mass-kg must be provided together."
            )
        try:
            profile = BodyProfile(
                height_m=height_m,
                mass_kg=mass_kg,
                foot_length_m=foot_length_m,
                foot_width_m=foot_width_m,
                source="recording_cli",
            )
        except ValueError as exc:
            raise SystemExit(f"Invalid body profile: {exc}") from exc

    try:
        table = (
            AnthropometricTable.from_yaml(table_path)
            if table_path is not None
            else None
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid anthropometric table: {exc}") from exc
    return KinematicComEstimator(profile, table=table)


def _capture_profile_neutral_calibration(
    tracker: TrackingProvider,
    profile: BodyProfile,
    *,
    duration_s: float,
    stop_event: threading.Event,
) -> tuple[ProfileNeutralCalibration, NeutralCalibrationCapture]:
    """Capture a short upright pose without consuming the sidecar queue."""
    latest_packet = getattr(tracker, "latest_packet", None)
    if not callable(latest_packet):
        raise SystemExit(
            "Selected tracking backend cannot provide native body packets for "
            "profile neutral calibration."
        )
    packet_accessor = cast(Callable[[], TrackingPacket | None], latest_packet)
    if duration_s <= 0:
        raise SystemExit("--body-neutral-calibration-s must be greater than zero.")

    log.info(
        "Stand upright in a neutral or T-pose with both feet on the floor for %.1fs.",
        duration_s,
    )
    source_frames: list[CanonicalBodyFrame] = []
    hmd_poses: list[np.ndarray] = []
    packets: list[TrackingPacket] = []
    seen: set[tuple[int | None, int, int]] = set()
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline and not stop_event.is_set():
        packet = packet_accessor()
        sample = tracker.latest()
        if packet is None or packet.body is None or not packet.body.active:
            time.sleep(0.01)
            continue
        if (
            tracker.device == "meta"
            and packet.body.calibration_state.lower() != "valid"
        ):
            raise SystemExit(
                "Meta body calibration state became invalid during neutral capture; "
                "complete headset body setup and retry."
            )
        if not sample.hmd_tracked:
            time.sleep(0.01)
            continue
        key = (
            packet.sequence,
            int(packet.body.source_time_ns),
            int(packet.body.observation_sequence or -1),
        )
        if key in seen:
            time.sleep(0.005)
            continue
        seen.add(key)
        source_frames.append(canonical_body_from_packet(packet))
        hmd_poses.append(np.asarray(sample.device_hmd_pose, dtype=np.float64).copy())
        packets.append(packet)
        time.sleep(0.005)

    if stop_event.is_set():
        raise SystemExit("Body neutral calibration interrupted.")
    min_samples = max(15, min(60, int(round(duration_s * 10.0))))
    capture = NeutralCalibrationCapture(
        packets=tuple(packets),
        device_hmd_poses=tuple(hmd_poses),
        requested_duration_s=duration_s,
    )
    try:
        validate_neutral_capture(capture, min_samples=min_samples)
        neutral = estimate_profile_neutral_calibration(
            source_frames,
            hmd_poses,
            profile,
            source_frame=f"{tracker.device}_right_handed_source",
            min_samples=min_samples,
        )
    except ValueError as exc:
        raise SystemExit(
            f"Body neutral calibration failed: {exc}. Stand upright and retry; "
            "do not record against an unverified floor."
        ) from exc
    return neutral, capture


def _tuple_shape(feature: dict) -> dict:
    feature = dict(feature)
    feature["shape"] = tuple(feature["shape"])
    return feature


def build_observation(
    sample: ControllerPairSample,
    widths: GripperWidths,
    *,
    body_frame: CanonicalBodyFrame | None = None,
) -> dict:
    left_controller = _pose_from_pose7(sample.left_controller_pose)
    right_controller = _pose_from_pose7(sample.right_controller_pose)
    state = pose_to_state_vector(
        left_controller,
        right_controller,
        widths.left,
        widths.right,
    )
    observation = {
        "observation.state": state,
        "action": state.copy(),
        "observation.feetech.left_ticks": np.array([widths.left_ticks], dtype=np.int64),
        "observation.feetech.right_ticks": np.array(
            [widths.right_ticks], dtype=np.int64
        ),
        "observation.feetech.left_width_mm": np.array(
            [widths.left_mm], dtype=np.float32
        ),
        "observation.feetech.right_width_mm": np.array(
            [widths.right_mm], dtype=np.float32
        ),
        "observation.feetech.left_normalized": np.array(
            [widths.left_normalized], dtype=np.float32
        ),
        "observation.feetech.right_normalized": np.array(
            [widths.right_normalized], dtype=np.float32
        ),
        **sample.tracking_frame(),
    }
    if body_frame is not None:
        observation.update(body_frame.observation())
    return observation


def _pose_from_pose7(pose7: np.ndarray) -> Pose:
    pose = np.asarray(pose7, dtype=np.float32).reshape(7)
    return Pose(pose[:3], pose[3:7])


def _tracking_healthy(sample: ControllerPairSample) -> bool:
    return bool(sample.left_tracked and sample.right_tracked)


def _wait_for_tracking(
    tracker: TrackingSampleSource,
    stop_event: threading.Event,
    *,
    poll_s: float = 0.05,
) -> bool:
    """Wait until both controller poses are fresh and valid."""
    last_report = float("-inf")
    while not stop_event.is_set():
        sample = tracker.latest()
        if _tracking_healthy(sample):
            log.info("Both controllers tracked; recording gate open.")
            return True

        now = time.monotonic()
        if now - last_report >= 2.0:
            log.warning(
                "Waiting for controller tracking (left=%d right=%d) ...",
                int(sample.left_tracked),
                int(sample.right_tracked),
            )
            last_report = now
        time.sleep(poll_s)
    return False


class _RecordingRerun:
    """Bounded asynchronous Rerun stream owned by the recorder."""

    def __init__(self, cam_names: list[str], fps: int) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frames: queue.Queue[
            tuple[
                dict,
                ControllerPairSample,
                GripperWidths,
                CanonicalBodyFrame | None,
            ]
        ] = queue.Queue(maxsize=2)
        self._dropped_frames = 0
        self._worker: threading.Thread | None = None
        self.stream = initialize_rerun(
            "handumi_record",
            cam_names,
            fps=fps,
            spawn=True,
            recorder_status=True,
            include_quality=True,
            on_error=self._on_error,
        )
        if self.stream is not None:
            self._worker = threading.Thread(
                target=self._run,
                name="handumi_rerun",
                daemon=True,
            )
            self._worker.start()
        self.set_status("READY", "Waiting to start the first episode")

    @staticmethod
    def _on_error(exc: BaseException) -> None:
        log.error(
            "Rerun failed; disabling live view while recording continues: %s", exc
        )

    def set_status(self, state: str, detail: str) -> None:
        """Show the current recorder state as a persistent operator flag."""
        if self.stream is not None:
            with self._lock:
                self.stream.set_status(state, detail)

    def log(
        self,
        cam_frames: dict,
        sample: ControllerPairSample,
        widths: GripperWidths,
        *,
        body_frame: CanonicalBodyFrame | None = None,
    ) -> None:
        """Queue the newest synchronized frame without blocking capture."""
        if self.stream is None or self._stop.is_set():
            return
        item = (cam_frames, sample, widths, body_frame)
        try:
            self._frames.put_nowait(item)
        except queue.Full:
            try:
                self._frames.get_nowait()
                self._frames.task_done()
            except queue.Empty:
                pass
            self._dropped_frames += 1
            try:
                self._frames.put_nowait(item)
            except queue.Full:
                self._dropped_frames += 1

    @property
    def pending_frames(self) -> int:
        return self._frames.qsize()

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    def close(self, timeout_s: float = 2.0) -> None:
        """Flush the bounded queue best-effort and stop the worker."""
        self._stop.set()
        worker = self._worker
        if worker is not None:
            worker.join(timeout=max(0.0, timeout_s))
            if worker.is_alive():
                log.warning("Rerun worker did not flush before shutdown.")
        self._worker = None
        if self._dropped_frames:
            log.info(
                "Rerun dropped %d stale live frames to keep recording responsive.",
                self._dropped_frames,
            )

    def _run(self) -> None:
        while not self._stop.is_set() or not self._frames.empty():
            try:
                cam_frames, sample, widths, body_frame = self._frames.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                stream = self.stream
                if stream is not None:
                    with self._lock:
                        stream.log_frame(
                            cam_frames,
                            sample,
                            widths,
                            body_frame=body_frame,
                        )
            finally:
                self._frames.task_done()


def record_episode(
    *,
    dataset,
    cameras: list,
    cam_names: list[str],
    tracker: TrackingSampleSource,
    grippers: FeetechGripperSampler | GripperWidthSource | None,
    episode_time_s: float,
    fps: int,
    task: str,
    cam_width: int,
    cam_height: int,
    stop_event: threading.Event,
    manual_control: bool,
    start_button: str,
    repeat_button: str,
    finish_button: str,
    start_threshold: float,
    clap_detector: DoubleClapDetector | None = None,
    tracking_loss_timeout_s: float = 1.0,
    sync_lag_s: float = 0.04,
    max_sync_skew_s: float = 0.06,
    camera_stale_timeout_s: float = 0.25,
    gripper_stale_timeout_s: float = 0.10,
    sensor_loss_timeout_s: float = 1.0,
    tracking_sidecar: _TrackingSidecarSink | None = None,
    world_calibration: HandumiWorldCalibration | None = None,
    profile_skeleton: ProfileConstrainedSkeleton | None = None,
    body_estimator: _BodyEstimator | None = None,
    rerun: _RerunSink | None = None,
    robot_viewer: _RobotFrameSink | None = None,
    profiler: StageProfiler | None = None,
    capture_session: CaptureSession | None = None,
    minimum_free_bytes: int = 0,
    disk_check_interval_s: float = 5.0,
) -> tuple[int, str]:
    control_interval = 1.0 / fps
    n_frames = 0
    start_t = time.perf_counter()
    status = "recorded"
    clap_control = clap_detector is not None
    xrt = getattr(tracker, "xrt", None)
    prev_start = (
        read_start_button_value(xrt, start_button) >= start_threshold
        if manual_control and xrt is not None
        else False
    )
    prev_repeat = (
        read_start_button_value(xrt, repeat_button) >= start_threshold
        if manual_control and xrt is not None
        else False
    )
    prev_finish = (
        read_start_button_value(xrt, finish_button) >= start_threshold
        if manual_control and xrt is not None
        else False
    )

    # Clap starts episodes hands-free. Once recording, another clap saves the
    # episode; the timer remains a maximum-duration safety limit.
    timed = not manual_control
    tracking_loss_timeout_ns = int(tracking_loss_timeout_s * 1e9)
    tracking_lost_since_ns: int | None = None
    episode_start_ns: int | None = None
    sync_lag_ns = int(sync_lag_s * 1e9)
    max_sync_skew_ns = int(max_sync_skew_s * 1e9)
    health_gate = SustainedHealthGate(sensor_loss_timeout_s)
    next_disk_check = time.monotonic()

    elapsed = 0.0
    while True:
        loop_start = time.perf_counter()
        tracking_now_ns = time.monotonic_ns()
        disk_now = time.monotonic()
        if minimum_free_bytes and disk_now >= next_disk_check:
            try:
                with (
                    profiler.measure("disk_space_check")
                    if profiler is not None
                    else nullcontext()
                ):
                    check_disk_space(
                        Path(dataset.root), minimum_free_bytes=minimum_free_bytes
                    )
            except CaptureStorageError as exc:
                status = "storage_failure"
                dataset.clear_episode_buffer()
                log.error("Capture storage check failed; discarding episode: %s", exc)
                break
            if profiler is not None and capture_session is not None:
                capture_session.checkpoint(profiler, reason="periodic_disk_check")
            next_disk_check = disk_now + disk_check_interval_s
        if episode_start_ns is None:
            episode_start_ns = tracking_now_ns
        elapsed = loop_start - start_t
        if stop_event.is_set():
            status = "interrupted"
            dataset.clear_episode_buffer()
            break
        if timed and elapsed >= episode_time_s:
            if (
                tracking_lost_since_ns is not None
                and tracking_now_ns - tracking_lost_since_ns >= tracking_loss_timeout_ns
            ):
                status = "tracking_lost"
                log.error(
                    "Controller tracking unavailable for %.2fs; discarding episode.",
                    (tracking_now_ns - tracking_lost_since_ns) / 1e9,
                )
            break

        if manual_control and xrt is not None:
            start_pressed = (
                read_start_button_value(xrt, start_button) >= start_threshold
            )
            repeat_pressed = (
                read_start_button_value(xrt, repeat_button) >= start_threshold
            )
            finish_pressed = (
                read_start_button_value(xrt, finish_button) >= start_threshold
            )
            start_rise = start_pressed and not prev_start
            repeat_rise = repeat_pressed and not prev_repeat
            finish_rise = finish_pressed and not prev_finish
            prev_start, prev_repeat, prev_finish = (
                start_pressed,
                repeat_pressed,
                finish_pressed,
            )
            if repeat_rise:
                status = "repeat"
                dataset.clear_episode_buffer()
                break
            if finish_rise:
                status = "finish"
                break
            if start_rise:
                status = "recorded"
                break

        target_time_ns = max(episode_start_ns, tracking_now_ns - sync_lag_ns)
        with (
            profiler.measure("camera_synchronization", items=len(cameras))
            if profiler is not None
            else nullcontext()
        ):
            cam_frames, camera_health = read_camera_samples(
                cameras,
                cam_names,
                target_time_ns=target_time_ns,
                record_time_ns=tracking_now_ns,
                width=cam_width,
                height=cam_height,
                stale_timeout_s=camera_stale_timeout_s,
                max_sync_skew_s=max_sync_skew_s,
            )
        if profiler is not None:
            with profiler.measure("camera_acquisition", items=len(cam_frames)):
                pass
        gripper_frame = synchronized_gripper_frame(
            grippers,
            target_time_ns=target_time_ns,
            record_time_ns=tracking_now_ns,
            stale_timeout_s=gripper_stale_timeout_s,
            max_sync_skew_s=max_sync_skew_s,
        )
        widths = gripper_frame.widths
        with (
            profiler.measure("tracking_reception_clock_alignment")
            if profiler is not None
            else nullcontext()
        ):
            sample = tracking_sample_at(tracker, target_time_ns)
        body_frame = None
        if tracking_sidecar is not None:
            with (
                profiler.measure("tracking_sidecar_write")
                if profiler is not None
                else nullcontext()
            ):
                tracking_sidecar.drain_provider(tracker)
            epoch_event = tracking_sidecar.consume_frame_epoch_change()
            if epoch_event is not None:
                epoch_index = getattr(epoch_event, "index", None)
                epoch_reason = getattr(epoch_event, "reason", None)
                if not isinstance(epoch_index, int) or not isinstance(
                    epoch_reason, str
                ):
                    raise TypeError("invalid tracking frame epoch event")
                status = "frame_epoch_changed"
                dataset.clear_episode_buffer()
                if body_estimator is not None:
                    reset_estimator = getattr(body_estimator, "reset", None)
                    if callable(reset_estimator):
                        reset_estimator()
                log.error(
                    "Tracking frame epoch changed to %d (%s); discarding the "
                    "episode and requiring calibration review.",
                    epoch_index,
                    epoch_reason,
                )
                break
            with (
                profiler.measure("body_canonicalization_derived_estimation")
                if profiler is not None
                else nullcontext()
            ):
                body_frame = canonical_body_from_packet(
                    tracking_sidecar.nearest_packet(target_time_ns),
                    calibration=world_calibration,
                )
                if profile_skeleton is not None:
                    body_frame = profile_skeleton.apply(body_frame)
                if body_estimator is not None:
                    body_frame = body_estimator.estimate(body_frame)
        sample_time_ns = int(sample.aligned_time_ns or sample.pc_monotonic_ns)
        tracking_sync_ok = bool(
            sample_time_ns > 0
            and abs(sample_time_ns - target_time_ns) <= max_sync_skew_ns
        )
        if _tracking_healthy(sample) and tracking_sync_ok:
            if tracking_lost_since_ns is not None:
                log.info("Controller tracking recovered before the episode timeout.")
            tracking_lost_since_ns = None
        elif tracking_lost_since_ns is None:
            # For a stale cached frame, count loss from its receive timestamp
            # instead of adding the freshness timeout to the one-second grace.
            sample_time_ns = int(sample.pc_monotonic_ns)
            tracking_lost_since_ns = (
                min(sample_time_ns, tracking_now_ns)
                if sample_time_ns > 0
                else tracking_now_ns
            )
            log.warning(
                "Controller tracking lost (left=%d right=%d); "
                "discarding if it lasts %.2fs.",
                int(sample.left_tracked),
                int(sample.right_tracked),
                tracking_loss_timeout_s,
            )
        elif tracking_now_ns - tracking_lost_since_ns >= tracking_loss_timeout_ns:
            status = "tracking_lost"
            log.error(
                "Controller tracking unavailable for %.2fs; discarding episode.",
                (tracking_now_ns - tracking_lost_since_ns) / 1e9,
            )
            break

        sensor_health = {
            **camera_health,
            "feetech": gripper_frame.healthy_for_gate,
        }
        recovered, timed_out_sensors = health_gate.update(
            sensor_health, tracking_now_ns
        )
        for sensor in recovered:
            log.info("Sensor health recovered before timeout: %s.", sensor)
        if timed_out_sensors:
            status = "sensor_unhealthy"
            log.error(
                "Sensor health unavailable for %.2fs (%s); discarding episode.",
                sensor_loss_timeout_s,
                ", ".join(sorted(timed_out_sensors)),
            )
            break

        if clap_control:
            clap_side = clap_detector.update_side(
                widths.left_mm, widths.right_mm, loop_start
            )
            if clap_side == "right":
                status = "recorded"
                break
            if clap_side == "left":
                status = "repeat"
                dataset.clear_episode_buffer()
                break
        if rerun is not None:
            # This is the same aligned, already-estimated frame written below.
            with (
                profiler.measure("rerun_enqueue")
                if profiler is not None
                else nullcontext()
            ):
                rerun.log(cam_frames, sample, widths, body_frame=body_frame)
            if profiler is not None:
                profiler.queue_depth("rerun", int(getattr(rerun, "pending_frames", 0)))
        if robot_viewer is not None:
            try:
                with (
                    profiler.measure("viser_robot_ik_enqueue")
                    if profiler is not None
                    else nullcontext()
                ):
                    robot_viewer.submit(
                        RecorderRobotFrame(
                            sample_time_ns=sample_time_ns,
                            left_tcp_pose=sample.left_tcp_pose,
                            right_tcp_pose=sample.right_tcp_pose,
                            left_tracked=sample.left_tracked,
                            right_tracked=sample.right_tracked,
                            left_gripper_opening=widths.left_normalized,
                            right_gripper_opening=widths.right_normalized,
                        )
                    )
                status_reader = getattr(robot_viewer, "status", None)
                viewer_status = status_reader() if callable(status_reader) else None
            except Exception as exc:  # noqa: BLE001 - viewer never owns capture.
                log.exception("Recorder Viser sink failed; recording continues")
                if rerun is not None:
                    status_writer = getattr(rerun, "set_status", None)
                    if callable(status_writer):
                        status_writer(
                            "RECORDING",
                            f"Dataset capture continues; Viser sink failed: {exc}",
                        )
            else:
                if rerun is not None and bool(getattr(viewer_status, "failures", 0)):
                    status_writer = getattr(rerun, "set_status", None)
                    if callable(status_writer):
                        status_writer(
                            "RECORDING",
                            "Dataset capture continues; Viser degraded: "
                            f"{getattr(viewer_status, 'last_error', None)}",
                        )
        try:
            with (
                profiler.measure("dataset_serialization_writes")
                if profiler is not None
                else nullcontext()
            ):
                dataset.add_frame(
                    {
                        **cam_frames,
                        **build_observation(sample, widths, body_frame=body_frame),
                        **gripper_frame.frame,
                        **capture_timing_frame(target_time_ns, tracking_now_ns),
                        "task": task,
                    }
                )
        except OSError as exc:
            if exc.errno != errno.ENOSPC:
                raise
            status = "storage_failure"
            dataset.clear_episode_buffer()
            log.error(
                "Storage exhausted while serializing the episode; it was rejected."
            )
            break
        n_frames += 1

        dt = time.perf_counter() - loop_start
        sleep = control_interval - dt
        if sleep > 0:
            time.sleep(sleep)
        else:
            log.warning(
                "Loop slower than %d Hz (%.1f Hz actual).", fps, 1.0 / max(dt, 1e-6)
            )

    measured_duration_s = max(0.0, elapsed)
    if (
        status == "recorded"
        and timed
        and episode_time_s >= 1.0
        and measured_duration_s >= episode_time_s * 0.95
        and n_frames < int(measured_duration_s * fps * 0.98)
    ):
        status = "profile_unmaintained"
        dataset.clear_episode_buffer()
        log.error(
            "Requested %d Hz row profile was not maintained: %d rows in %.3fs; "
            "episode rejected.",
            fps,
            n_frames,
            measured_duration_s,
        )
    return n_frames, status


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record HandUMI data with PICO or Meta Quest."
    )
    p.add_argument("--device", choices=("pico", "meta"), required=True)
    p.add_argument(
        "--rig-config",
        type=Path,
        default=DEFAULT_RIG_CONFIG,
        help="Machine-local cameras, Feetech, and Meta Quest configuration.",
    )
    p.add_argument("--cam-ids", nargs="+", type=_camera_arg, default=None)
    p.add_argument(
        "--wrist-cameras",
        action="store_true",
        help="Record both wrist cameras. This is the default when no camera-selection flag is used.",
    )
    p.add_argument(
        "--workspace-camera",
        action="store_true",
        help="Record the workspace camera; combine with --wrist-cameras for all three.",
    )
    only_camera = p.add_mutually_exclusive_group()
    only_camera.add_argument(
        "--only-left-camera",
        "--only-left-cameras",
        dest="only_left_camera",
        action="store_true",
    )
    only_camera.add_argument(
        "--only-right-camera",
        "--only-right-cameras",
        dest="only_right_camera",
        action="store_true",
    )
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--cam-fps", type=int, default=30)
    p.add_argument(
        "--minimum-free-gb",
        type=float,
        default=2.0,
        help="Fail before and during capture if usable disk space falls below this value.",
    )
    p.add_argument("--disk-check-interval-s", type=float, default=5.0)
    p.add_argument("--feetech-port", type=str, default=None)
    p.add_argument("--skip-feetech", action="store_true")
    p.add_argument("--repo-id", type=str, default="local/handumi_dataset")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Dataset folder. Defaults to a fresh outputs/<YYYYMMDD_HHMMSS>/ "
        "named after when recording started (outputs/ is gitignored).",
    )
    p.add_argument("--task", type=str, default="HandUMI recording")
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--episode-time-s", type=float, default=60.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--tracking-loss-timeout-s",
        type=float,
        default=1.0,
        help="Discard an episode when either controller remains untracked for this long.",
    )
    p.add_argument(
        "--sync-lag-s",
        type=float,
        default=0.04,
        help="Capture rows this far behind real time so native sensor buffers can align.",
    )
    p.add_argument(
        "--max-sync-skew-s",
        type=float,
        default=0.06,
        help="Maximum source-to-row timestamp difference considered healthy.",
    )
    p.add_argument("--camera-stale-timeout-s", type=float, default=0.25)
    p.add_argument("--gripper-stale-timeout-s", type=float, default=0.10)
    p.add_argument(
        "--sensor-loss-timeout-s",
        type=float,
        default=1.0,
        help="Discard after a camera or encoder remains unhealthy for this long.",
    )
    p.add_argument("--feetech-sample-hz", type=float, default=100.0)
    p.add_argument("--no-video", action="store_true")
    p.add_argument(
        "--rerun",
        action="store_true",
        help="Open a live Rerun view with recorded cameras, controller/TCP trails, and gripper widths.",
    )
    p.add_argument(
        "--viser",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Show recorder-owned robot IK in Viser without opening another "
            "tracking, camera, or Feetech provider (default: disabled)."
        ),
    )
    p.add_argument(
        "--viser-host",
        default="127.0.0.1",
        help="Viser bind host (default: localhost; use LAN addresses explicitly).",
    )
    p.add_argument("--viser-port", type=int, default=8003)
    p.add_argument(
        "--viser-anchor",
        choices=("episode-start", "first-tracked", "disabled"),
        default="episode-start",
        help="When the simulated robot establishes its controller/TCP anchor.",
    )
    p.add_argument("--viser-anchor-z", type=float, default=None)
    p.add_argument("--viser-home-pose", default=None)
    p.add_argument("--viser-scene", default=None)
    p.add_argument("--viser-queue-size", type=int, default=2)
    p.add_argument("--vcodec", type=str, default="h264")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument(
        "--dataset-license",
        default="other",
        help="Hugging Face dataset-card license identifier. Defaults to 'other' so data is not relicensed as software.",
    )
    p.add_argument(
        "--controller-tcp-calibration",
        type=Path,
        default=None,
        help=(
            "Explicit Controller-to-HandUMI-TCP override. Defaults to the "
            "robot/device setup and snapshots its tool identity in dataset metadata. "
            "Raw controller poses remain unchanged."
        ),
    )
    p.add_argument(
        "--session-calibration",
        type=Path,
        default=None,
        help=(
            "Tracking-device-to-table session calibration from handumi-calibrate-spatial. "
            "Locks all episodes to the same table frame."
        ),
    )
    p.add_argument(
        "--robot",
        default="piper",
        help=(
            "Intended robot embodiment. Snapshots configs/robots/<robot>.yaml in "
            "metadata; raw recordings remain robot-agnostic."
        ),
    )

    p.add_argument("--quest-ip", type=str, default=None)
    p.add_argument("--tcp-port", type=int, default=None)
    p.add_argument("--sync-port", type=int, default=None)
    p.add_argument(
        "--body-profile",
        type=Path,
        default=None,
        help=(
            "YAML body profile with required height_m and mass_kg. Enables "
            "kinematic CoM/contact estimation."
        ),
    )
    p.add_argument("--body-height-m", type=float, default=None)
    p.add_argument("--body-mass-kg", type=float, default=None)
    p.add_argument("--body-foot-length-m", type=float, default=None)
    p.add_argument("--body-foot-width-m", type=float, default=None)
    p.add_argument(
        "--body-neutral-calibration-s",
        type=float,
        default=3.0,
        help=(
            "Upright neutral/T-pose dwell used to place the experimental body "
            "floor at z=0 and fit supplied body dimensions (default: 3.0s)."
        ),
    )
    p.add_argument(
        "--anthropometric-table",
        type=Path,
        default=None,
        help="Optional custom versioned anthropometric segment table YAML.",
    )

    p.add_argument(
        "--pico-mode", choices=("mandos", "object", "whole-body"), default="mandos"
    )
    pico_transport = p.add_mutually_exclusive_group()
    pico_transport.add_argument("--pico-adb", action="store_true")
    pico_transport.add_argument("--pico-wifi", action="store_true")
    p.add_argument("--skip-adb-check", action="store_true")
    p.add_argument("--start-button", choices=START_BUTTON_CHOICES, default="enter")
    p.add_argument("--start-threshold", type=float, default=0.75)
    p.add_argument("--manual-control", action="store_true")
    p.add_argument("--repeat-button", choices=START_BUTTON_CHOICES, default="B")
    p.add_argument("--finish-button", choices=START_BUTTON_CHOICES, default="Y")
    p.add_argument(
        "--clap-control",
        action="store_true",
        help="Hands-free: double-squeeze right to start or stop/save; "
        "double-squeeze left while recording to restart the same episode. "
        "Needs real Feetech widths.",
    )
    p.add_argument(
        "--no-sounds",
        action="store_true",
        help="Disable spoken episode-status announcements (start/save/discard/stop).",
    )
    return p.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.manual_control and args.device != "pico":
        raise SystemExit("--manual-control currently requires --device pico.")
    if args.manual_control and args.start_button == "enter":
        args.start_button = "A"
        log.info("--manual-control set: using PICO A as start/stop button.")
    if args.clap_control and args.skip_feetech:
        raise SystemExit(
            "--clap-control needs real Feetech widths; drop --skip-feetech."
        )
    if args.clap_control and args.manual_control:
        raise SystemExit("--clap-control and --manual-control are mutually exclusive.")
    if args.tracking_loss_timeout_s <= 0:
        raise SystemExit("--tracking-loss-timeout-s must be greater than zero.")
    for name in (
        "sync_lag_s",
        "max_sync_skew_s",
        "camera_stale_timeout_s",
        "gripper_stale_timeout_s",
        "sensor_loss_timeout_s",
        "feetech_sample_hz",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be greater than zero.")
    if not 0 <= getattr(args, "viser_port", 8003) <= 65535:
        raise SystemExit("--viser-port must be between 0 and 65535.")
    if getattr(args, "viser_queue_size", 2) <= 0:
        raise SystemExit("--viser-queue-size must be greater than zero.")
    if getattr(args, "minimum_free_gb", 2.0) < 0:
        raise SystemExit("--minimum-free-gb cannot be negative.")
    if getattr(args, "disk_check_interval_s", 5.0) <= 0:
        raise SystemExit("--disk-check-interval-s must be greater than zero.")
    if not str(getattr(args, "viser_host", "127.0.0.1")).strip():
        raise SystemExit("--viser-host must not be empty.")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    if args.output_dir is None:
        args.output_dir = _default_output_dir()
    minimum_free_bytes = int(args.minimum_free_gb * 1024**3)
    check_disk_space(args.output_dir, minimum_free_bytes=minimum_free_bytes)
    profiler = StageProfiler()
    play_sounds = not args.no_sounds

    camera_names = _selected_camera_names(args)
    capture_profile = resolve_capture_profile(args.fps, args.cam_fps, len(camera_names))
    if capture_profile.status != "supported":
        log.warning(
            "Capture profile %s is %s: %s",
            capture_profile.name,
            capture_profile.status,
            capture_profile.evidence,
        )
    requested_output_dir = Path(args.output_dir)
    recovered_sessions = recover_interrupted_sessions(requested_output_dir.parent)
    for recovered_path in recovered_sessions:
        log.warning(
            "Preserved interrupted capture as incomplete forensic evidence: %s",
            recovered_path.name,
        )
    capture_session = CaptureSession(
        requested_output_dir,
        capture_profile,
        configuration_hashes=[hash_configuration("rig", args.rig_config)],
        calibration_hashes=[
            hash_configuration("controller_tcp", args.controller_tcp_calibration),
            hash_configuration("session", args.session_calibration),
            hash_configuration("body_profile", args.body_profile),
        ],
    )
    args.output_dir = capture_session.staging_root
    robot_metadata = _robot_metadata(args.robot)
    calibration_metadata, calibration_source = _recording_tcp_calibration_metadata(
        robot_metadata=robot_metadata,
        device=args.device,
        explicit_path=args.controller_tcp_calibration,
    )
    log.info("Controller->TCP setup: %s", calibration_source)
    try:
        spatial_session_metadata = session_calibration_metadata(
            args.session_calibration
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid session calibration: {exc}") from exc
    body_estimator = build_body_estimator(args)
    profile_skeleton = (
        ProfileConstrainedSkeleton(body_estimator.profile)
        if body_estimator is not None
        else None
    )
    log.info("--- Tracking setup ---")
    calibration = ControllerTcpCalibration(
        left=IDENTITY_POSE7.astype(np.float32).copy(),
        right=IDENTITY_POSE7.astype(np.float32).copy(),
        source=None,
    )
    tracker = build_tracker(
        args,
        calibration,
        reset_workspace_on_x=body_estimator is None,
        level_workspace_on_reset=body_estimator is not None,
    )
    if args.session_calibration is not None:
        assert spatial_session_metadata is not None
        session_device = str(spatial_session_metadata.get("tracking_device") or "")
        if session_device and session_device != args.device:
            raise SystemExit(
                f"Session calibration is for {session_device}, "
                f"but --device {args.device} was selected."
            )
        set_workspace = getattr(tracker, "set_workspace_from_device_pose", None)
        if set_workspace is None:
            raise SystemExit(
                "Selected tracking backend cannot apply a table calibration."
            )
        set_workspace(session_table_from_device(args.session_calibration), locked=True)
    tracker.start()

    log.info("--- Camera setup ---")
    cam_ids = resolve_camera_ids(
        args.cam_ids,
        args.rig_config,
        camera_names=camera_names,
    )
    _validate_unique_camera_ids(camera_names, cam_ids)
    camera_specs, _ = build_camera_specs(
        cam_ids,
        camera_names=camera_names,
        laptop_camera=False,
        laptop_cam_id=0,
        laptop_cam_name="laptop",
    )
    cam_names = [spec["name"] for spec in camera_specs]
    # Rerun may launch an external viewer process. Spawn it before OpenCV opens
    # V4L descriptors so that viewer lifetime cannot keep cameras busy after
    # the recorder exits.
    rerun = _RecordingRerun(cam_names, args.fps) if args.rerun else None
    cameras = connect_cameras(
        camera_specs,
        fps=args.cam_fps,
        width=args.cam_width,
        height=args.cam_height,
        zero_non_laptop=False,
    )

    log.info("--- Feetech setup ---")
    gripper_pair = connect_feetech(args)
    grippers = None
    if gripper_pair is not None:
        grippers = FeetechGripperSampler(
            gripper_pair,
            sample_hz=args.feetech_sample_hz,
        )
        grippers.start()

    log.info("--- Dataset setup ---")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    use_videos = not args.no_video
    features = build_features(cam_names, args.cam_width, args.cam_height, use_videos)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=args.output_dir,
        robot_type="handumi_raw",
        features=features,
        use_videos=use_videos,
        image_writer_processes=0,
        image_writer_threads=max(1, 4 * len(cam_names)),
        vcodec=args.vcodec,
    )
    log.info("Dataset created at: %s", dataset.root)
    tracking_sidecar = TrackingSidecarWriter(dataset.root)
    robot_viewer: RecorderRobotSink | None = None
    if args.viser:
        robot_viewer = QueuedRecorderRobotViewer(
            RecorderRobotViewerConfig(
                robot=args.robot,
                device=args.device,
                host=args.viser_host,
                port=args.viser_port,
                rig_config=args.rig_config,
                home_pose=args.viser_home_pose,
                scene=args.viser_scene,
                anchor_mode=args.viser_anchor,
                anchor_z=args.viser_anchor_z,
                queue_size=args.viser_queue_size,
            )
        )
    world_calibration = HandumiWorldCalibration.identity(
        source_frame=f"{args.device}_right_handed_source",
        qualified=False,
    )
    neutral_calibration_metadata: dict | None = None
    neutral_calibration_artifacts: list[dict] = []
    tracking_workspace = (
        "table" if spatial_session_metadata is not None else "hmd_recentered"
    )

    stop_event = threading.Event()

    def _on_signal(signum, frame):
        log.info("Signal received - discarding active episode and stopping ...")
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    escape_listener = _EscapeStopListener(stop_event)
    if args.clap_control:
        escape_listener.start()

    recorded = 0
    clap_detector = DoubleClapDetector() if args.clap_control else None
    restart_active = False
    try:
        while (
            args.num_episodes <= 0 or recorded < args.num_episodes
        ) and not stop_event.is_set():
            ep_num = dataset.num_episodes + 1
            ep_total = "inf" if args.num_episodes <= 0 else str(args.num_episodes)
            log.info("--- Episode %d/%s ---", ep_num, ep_total)
            if rerun is not None:
                rerun.set_status(
                    "WAITING", f"Episode {ep_num}/{ep_total}: waiting to start"
                )
            if robot_viewer is not None:
                robot_viewer.set_recording_state(
                    "WAITING", f"episode {ep_num}/{ep_total}"
                )
            if args.clap_control:
                assert clap_detector is not None
                if restart_active:
                    restart_active = False
                    log.info("  Restarting episode %d immediately ...", ep_num)
                    if rerun is not None:
                        rerun.set_status(
                            "RESTARTED", f"Episode {ep_num}/{ep_total}: restarting now"
                        )
                else:
                    log.info(
                        "  Double-squeeze right gripper to start episode %d ...", ep_num
                    )
                    if not _wait_for_clap(
                        grippers, clap_detector, stop_event, side="right"
                    ):
                        break
                    # A calibrated table workspace is locked and ignores this
                    # legacy HMD recenter; uncalibrated sessions retain it.
                    if body_estimator is None:
                        reset_workspace = getattr(tracker, "reset_workspace", None)
                        if reset_workspace is not None:
                            reset_workspace()
            elif args.manual_control:
                action = wait_for_manual_start(
                    getattr(tracker, "xrt"),
                    start_button=args.start_button,
                    finish_button=args.finish_button,
                    threshold=args.start_threshold,
                    stop_event=stop_event,
                )
                if action == "finish":
                    break
            elif args.start_button == "enter":
                if not _wait_for_enter(
                    stop_event,
                    f"  Press ENTER to start recording episode {ep_num} ...",
                ):
                    break
            elif args.device == "pico":
                if not wait_for_start_button(
                    getattr(tracker, "xrt"),
                    button=args.start_button,
                    threshold=args.start_threshold,
                    stop_event=stop_event,
                ):
                    break
            else:
                raise SystemExit(
                    "--start-button other than enter currently requires --device pico."
                )

            if not _wait_for_tracking(tracker, stop_event):
                break
            if profile_skeleton is not None and not profile_skeleton.calibrated:
                assert body_estimator is not None
                if spatial_session_metadata is None:
                    if rerun is not None:
                        rerun.set_status(
                            "CALIBRATING",
                            "Stand upright: fitting experimental body profile and floor",
                        )
                    if robot_viewer is not None:
                        robot_viewer.set_recording_state(
                            "CALIBRATING", "neutral/profile capture"
                        )
                    neutral, neutral_capture = _capture_profile_neutral_calibration(
                        tracker,
                        body_estimator.profile,
                        duration_s=args.body_neutral_calibration_s,
                        stop_event=stop_event,
                    )
                    world_calibration = neutral.world
                    neutral_frames = [
                        canonical_body_from_packet(
                            packet, calibration=world_calibration
                        )
                        for packet in neutral_capture.packets
                    ]
                    neutral_calibration_metadata = neutral.metadata()
                    set_workspace = getattr(
                        tracker, "set_workspace_from_device_pose", None
                    )
                    if not callable(set_workspace):
                        raise SystemExit(
                            "Selected tracking backend cannot apply the calibrated body world."
                        )
                    world_pose = world_calibration.world_from_source
                    set_workspace(
                        np.concatenate(
                            (world_pose.position, world_pose.quaternion)
                        ).astype(np.float32),
                        locked=True,
                    )
                    tracking_workspace = "profile_neutral_ground"
                    log.info(
                        "Experimental profile-assisted floor locked at source z=%.3f m; "
                        "Rerun ground is z=0.",
                        neutral.source_ground_height_m,
                    )
                else:
                    alignment_sample = tracker.latest()
                    world_calibration = _body_calibration_from_workspace(
                        alignment_sample.workspace_from_device_pose,
                        device=args.device,
                        qualified=False,
                    )
                    # A table calibration owns the rigid frame. Capture only the
                    # neutral geometry used by the profile-constrained skeleton.
                    neutral, neutral_capture = _capture_profile_neutral_calibration(
                        tracker,
                        body_estimator.profile,
                        duration_s=args.body_neutral_calibration_s,
                        stop_event=stop_event,
                    )
                    neutral_frames = [
                        canonical_body_from_packet(
                            packet, calibration=world_calibration
                        )
                        for packet in neutral_capture.packets
                    ]
                    neutral_calibration_metadata = neutral.metadata()
                try:
                    profile_skeleton.calibrate(neutral_frames)
                except ValueError as exc:
                    raise SystemExit(f"Body profile fitting failed: {exc}") from exc
                tracking_sidecar.set_frame_calibration(
                    world_calibration.metadata(),
                    reason="initial_profile_neutral_calibration",
                )
                _, artifact_reference = persist_neutral_calibration_capture(
                    dataset.root,
                    neutral_capture,
                    neutral,
                    body_estimator.profile,
                    applied_world=world_calibration,
                    profile_skeleton=profile_skeleton.metadata(),
                    frame_epoch=tracking_sidecar.frame_epoch,
                    frame_epoch_reason="initial_profile_neutral_calibration",
                    neutral_world_applied=spatial_session_metadata is None,
                )
                neutral_calibration_artifacts.append(artifact_reference)
            _discard_tracking_backlog(tracker)
            if body_estimator is not None:
                if profile_skeleton is None or not profile_skeleton.calibrated:
                    alignment_sample = tracker.latest()
                    world_calibration = _body_calibration_from_workspace(
                        alignment_sample.workspace_from_device_pose,
                        device=args.device,
                        qualified=False,
                    )
                log.info(
                    "Body/controller visualization locked to %s.",
                    tracking_workspace,
                )
                body_estimator.reset()
                tracking_sidecar.set_frame_calibration(
                    world_calibration.metadata(),
                    reason="recording_world_calibration",
                )
            tracking_sidecar.start_episode(dataset.num_episodes)
            log_say(f"Recording episode {ep_num}", play_sounds=play_sounds)
            if rerun is not None:
                rerun.set_status(
                    "RECORDING", f"Episode {ep_num}/{ep_total} is being recorded"
                )
            if robot_viewer is not None:
                robot_viewer.set_recording_state(
                    "RECORDING", f"episode {ep_num}/{ep_total}"
                )
            try:
                n_frames, status = record_episode(
                    dataset=dataset,
                    cameras=cameras,
                    cam_names=cam_names,
                    tracker=tracker,
                    grippers=grippers,
                    episode_time_s=args.episode_time_s,
                    fps=args.fps,
                    task=args.task,
                    cam_width=args.cam_width,
                    cam_height=args.cam_height,
                    stop_event=stop_event,
                    manual_control=args.manual_control,
                    start_button=args.start_button,
                    repeat_button=args.repeat_button,
                    finish_button=args.finish_button,
                    start_threshold=args.start_threshold,
                    clap_detector=clap_detector,
                    tracking_loss_timeout_s=args.tracking_loss_timeout_s,
                    sync_lag_s=args.sync_lag_s,
                    max_sync_skew_s=args.max_sync_skew_s,
                    camera_stale_timeout_s=args.camera_stale_timeout_s,
                    gripper_stale_timeout_s=args.gripper_stale_timeout_s,
                    sensor_loss_timeout_s=args.sensor_loss_timeout_s,
                    tracking_sidecar=tracking_sidecar,
                    world_calibration=world_calibration,
                    profile_skeleton=profile_skeleton,
                    body_estimator=body_estimator,
                    rerun=rerun,
                    robot_viewer=robot_viewer,
                    profiler=profiler,
                    capture_session=capture_session,
                    minimum_free_bytes=minimum_free_bytes,
                    disk_check_interval_s=args.disk_check_interval_s,
                )
            except BaseException:
                tracking_sidecar.finish_episode(status="interrupted", provider=tracker)
                raise
            if status == "repeat":
                log.warning(
                    "Episode restart requested (%d frames discarded).", n_frames
                )
                log_say("Restart recording", play_sounds=play_sounds)
                if rerun is not None:
                    rerun.set_status(
                        "RESTARTED",
                        f"Episode {ep_num}/{ep_total}: {n_frames} frames discarded; restarting",
                    )
                if robot_viewer is not None:
                    robot_viewer.set_recording_state(
                        "RESTARTED", f"{n_frames} frames discarded"
                    )
                dataset.clear_episode_buffer()
                tracking_sidecar.finish_episode(status="discarded", provider=tracker)
                restart_active = True
                continue
            if n_frames == 0 or status in {
                "frame_epoch_changed",
                "tracking_lost",
                "sensor_unhealthy",
                "interrupted",
                "storage_failure",
                "profile_unmaintained",
            }:
                log.warning("Episode discarded (%s, %d frames).", status, n_frames)
                log_say("Episode discarded", play_sounds=play_sounds)
                if rerun is not None:
                    rerun.set_status(
                        "DISCARDED",
                        f"Episode {ep_num}/{ep_total}: {status} after {n_frames} frames",
                    )
                if robot_viewer is not None:
                    robot_viewer.set_recording_state(
                        "DISCARDED", f"{status} after {n_frames} frames"
                    )
                dataset.clear_episode_buffer()
                tracking_sidecar.finish_episode(status="discarded", provider=tracker)
                if status == "frame_epoch_changed":
                    if profile_skeleton is not None:
                        profile_skeleton.invalidate()
                    world_calibration = HandumiWorldCalibration.identity(
                        source_frame=f"{args.device}_right_handed_source",
                        qualified=False,
                    )
                    if spatial_session_metadata is not None:
                        log.error(
                            "The tracking source frame changed after table calibration; "
                            "stop and create a new spatial session calibration."
                        )
                        break
                if status in {"finish", "interrupted"}:
                    break
                continue
            try:
                with profiler.measure("video_encoding_dataset_episode_write"):
                    dataset.save_episode()
            except OSError as exc:
                if exc.errno == errno.ENOSPC:
                    raise CaptureStorageError(
                        "storage exhausted while closing episode; partial data is not complete"
                    ) from exc
                raise
            tracking_sidecar.finish_episode(status="recorded", provider=tracker)
            recorded += 1
            log.info("Episode %d saved (%d frames).", ep_num, n_frames)
            log_say(
                f"Episode {ep_num} saved, {n_frames} frames", play_sounds=play_sounds
            )
            if rerun is not None:
                rerun.set_status(
                    "SAVED", f"Episode {ep_num}/{ep_total}: {n_frames} frames saved"
                )
            if robot_viewer is not None:
                robot_viewer.set_recording_state(
                    "SAVED", f"episode {ep_num}/{ep_total}; {n_frames} frames"
                )
            if status == "finish":
                break
    finally:
        escape_listener.stop()
        if rerun is not None:
            rerun.set_status(
                "STOPPED", f"Recording stopped: {recorded} episode(s) saved"
            )
        if robot_viewer is not None:
            robot_viewer.set_recording_state("STOPPED", f"{recorded} episode(s) saved")
        log_say("Stop recording", play_sounds=play_sounds, blocking=True)
        log.info("--- Finalising ---")
        finalization_error: BaseException | None = None
        try:
            with profiler.measure("dataset_finalization"):
                dataset.finalize()
            root = Path(dataset.root)
            body_metadata = canonical_body_metadata(
                transforms=[world_calibration.metadata()]
            )
            com_estimator_metadata = (
                body_estimator.metadata()
                if body_estimator is not None
                else {
                    "schema": "handumi_kinematic_com_v1",
                    "enabled": False,
                    "reason": "body_profile_not_supplied",
                }
            )
            if profile_skeleton is not None:
                com_estimator_metadata["profile_constrained_skeleton"] = (
                    profile_skeleton.metadata()
                )
            if neutral_calibration_metadata is not None:
                body_metadata["neutral_calibration"] = neutral_calibration_metadata
            if neutral_calibration_artifacts:
                body_metadata["neutral_calibration_artifacts"] = (
                    neutral_calibration_artifacts
                )
            body_metadata["estimator_version"] = (
                com_estimator_metadata["schema"]
                if body_estimator is not None
                else "not_run"
            )
            body_metadata["com_estimator"] = com_estimator_metadata
            body_tracking_schema = body_metadata.pop("tracking_schema")
            updated_info = _update_info_json(
                root,
                {
                    "recording_device": args.device,
                    "tracking_schema": HANDUMI_TRACKING_SCHEMA,
                    "tracking_workspace": tracking_workspace,
                    "state_semantics": HANDUMI_STATE_SEMANTICS,
                    "capture_schema": HANDUMI_CAPTURE_SCHEMA,
                    "sync_lag_s": args.sync_lag_s,
                    "max_sync_skew_s": args.max_sync_skew_s,
                    "camera_stale_timeout_s": args.camera_stale_timeout_s,
                    "gripper_stale_timeout_s": args.gripper_stale_timeout_s,
                    "cameras": [
                        {"name": spec["name"], "index_or_path": spec["id"]}
                        for spec in camera_specs
                    ],
                    "sources": _capture_sources_metadata(
                        camera_specs, cameras, grippers
                    ),
                    "controller_tcp_calibration": calibration_metadata,
                    "spatial_session_calibration": spatial_session_metadata,
                    "target_robot": robot_metadata,
                    "robot_viewer": (
                        {
                            "enabled": True,
                            "robot": args.robot,
                            "host": args.viser_host,
                            "port": args.viser_port,
                            "anchor_mode": args.viser_anchor,
                            "home_pose": args.viser_home_pose,
                            "scene": args.viser_scene,
                            "status": robot_viewer.status().__dict__,
                            "raw_recording_robot_agnostic": True,
                        }
                        if robot_viewer is not None
                        else {"enabled": False}
                    ),
                    "body_tracking_schema": body_tracking_schema,
                    "canonical_body": body_metadata,
                    "tracking_sidecar": {
                        "schema": "handumi_tracking_sidecar_v1",
                        "manifest": "raw/tracking/manifest.json",
                        "frame_epochs": tracking_sidecar.frame_epoch_metadata(),
                    },
                },
            )
            if updated_info is not None:
                dataset.meta.info = updated_info
            card_kwargs = _write_dataset_readme(
                root,
                repo_id=args.repo_id,
                task=args.task,
                license_id=args.dataset_license,
            )
            _validate_finalized_lerobot_dataset(root)
            log.info("LeRobot v3 integrity validation passed.")
            viewer_failures: list[str] = []
            if robot_viewer is not None:
                viewer_status = robot_viewer.status()
                if viewer_status.last_error:
                    viewer_failures.append(viewer_status.last_error)
            capture_session.viewer_failures.extend(viewer_failures)
            if args.push_to_hub:
                dataset.push_to_hub(
                    license=args.dataset_license,
                    tags=["HandUMI"],
                    **card_kwargs,  # pyright: ignore[reportArgumentType]
                )
            with profiler.measure("checksum_generation"):
                capture_session.complete(profiler)
        except BaseException as exc:
            finalization_error = exc
            log.exception("Dataset finalization failed; do not upload this dataset.")
            try:
                capture_session.reject(
                    profiler,
                    reason=f"finalization_failed:{type(exc).__name__}",
                )
            except BaseException:
                log.exception(
                    "Could not promote the failed staging directory to rejected; "
                    "its incomplete state remains preserved."
                )
        finally:
            if robot_viewer is not None:
                robot_viewer.close()
            if rerun is not None:
                rerun.close()
            disconnect_cameras(cameras)
            if grippers is not None:
                grippers.stop()
            if gripper_pair is not None:
                gripper_pair.close()
            tracking_sidecar.close()
            tracker.stop()
        if finalization_error is not None:
            raise finalization_error
        log.info(
            "Done. Recorded %d episode(s). Dataset at: %s",
            recorded,
            requested_output_dir,
        )
        log_say("Exiting", play_sounds=play_sounds)


def build_tracker(
    args: argparse.Namespace,
    calibration,
    *,
    reset_workspace_on_x: bool = True,
    level_workspace_on_reset: bool = False,
) -> TrackingProvider:
    if args.device == "pico":
        transport = "wifi" if args.pico_wifi else "adb"
        return PicoTrackingProvider(
            calibration=calibration,
            mode=args.pico_mode,
            transport=transport,
            skip_adb_check=args.skip_adb_check,
        )

    base = MetaQuestConfig.from_yaml(args.rig_config)
    config = MetaQuestConfig(
        quest_ip=args.quest_ip if args.quest_ip is not None else base.quest_ip,
        tcp_port=args.tcp_port if args.tcp_port is not None else base.tcp_port,
        sync_port=args.sync_port if args.sync_port is not None else base.sync_port,
        connect_retry_s=base.connect_retry_s,
        frame_stale_timeout_s=base.frame_stale_timeout_s,
        packet_queue_size=base.packet_queue_size,
    )
    return MetaQuestTrackingProvider(
        config=config,
        calibration=calibration,
        reset_workspace_on_x=reset_workspace_on_x,
        level_workspace_on_reset=level_workspace_on_reset,
    )


def connect_feetech(args: argparse.Namespace) -> FeetechGripperPair | None:
    if args.skip_feetech:
        log.info("Feetech disabled: gripper widths will be zero-filled.")
        return None
    feetech_config = load_config(args.rig_config)
    if args.feetech_port is not None:
        feetech_config = type(feetech_config)(
            port=args.feetech_port,
            baudrate=feetech_config.baudrate,
            protocol_version=feetech_config.protocol_version,
            left=feetech_config.left,
            right=feetech_config.right,
        )
    assert_calibrated(feetech_config, source=user_calibration_path())
    grippers = FeetechGripperPair(feetech_config)
    try:
        grippers.open()
    except FeetechUnavailableError as exc:
        raise SystemExit(str(exc)) from exc
    return grippers


def _wait_for_clap(
    grippers: FeetechGripperSampler | GripperWidthSource | None,
    clap_detector: DoubleClapDetector,
    stop_event: threading.Event,
    *,
    side: str = "right",
) -> bool:
    """Poll widths until ``side`` double-claps (or ``stop_event`` sets)."""
    while not stop_event.is_set():
        widths = _latest_gripper_widths(grippers)
        triggered = clap_detector.update_side(
            widths.left_mm, widths.right_mm, time.perf_counter()
        )
        if triggered == side:
            return True
        time.sleep(0.02)
    return False


def _latest_gripper_widths(
    grippers: FeetechGripperSampler | GripperWidthSource | None,
) -> GripperWidths:
    if grippers is None:
        return zero_gripper_widths()
    if isinstance(grippers, FeetechGripperSampler):
        sample = grippers.latest()
        return zero_gripper_widths() if sample is None else sample.widths
    return grippers.read_normalized_widths()


def _default_output_dir() -> Path:
    """``outputs/<YYYYMMDD_HHMMSS>/`` named after the moment recording starts.

    ``outputs/`` is gitignored — datasets never get committed by accident.
    """
    return Path("outputs") / datetime.now().strftime("%Y%m%d_%H%M%S")


def _selected_camera_names(args: argparse.Namespace) -> list[str]:
    only_left = bool(getattr(args, "only_left_camera", False))
    only_right = bool(getattr(args, "only_right_camera", False))
    wrist = bool(getattr(args, "wrist_cameras", False))
    workspace = bool(getattr(args, "workspace_camera", False))
    if (only_left or only_right) and (wrist or workspace):
        raise SystemExit(
            "--only-left-camera/--only-right-camera cannot be combined with "
            "--wrist-cameras or --workspace-camera."
        )
    if only_left:
        return ["left_wrist"]
    if only_right:
        return ["right_wrist"]
    if not wrist and not workspace:
        return ["left_wrist", "right_wrist"]
    names = []
    if wrist:
        names.extend(("left_wrist", "right_wrist"))
    if workspace:
        names.append("workspace")
    return names


def _capture_sources_metadata(
    camera_specs: list[dict[str, object]],
    cameras: Sequence[object | None],
    grippers: object | None,
) -> dict[str, object]:
    """Store source enablement once instead of repeating it on every row."""
    return {
        "tracking": {"enabled": True},
        "feetech": {"enabled": grippers is not None},
        "cameras": {
            str(spec["name"]): {"enabled": camera is not None}
            for spec, camera in zip(camera_specs, cameras, strict=True)
        },
    }


def _robot_metadata(
    name: str, config_dir: Path = ROBOT_CONFIG_DIR
) -> dict[str, object]:
    path = config_dir / f"{name}.yaml"
    if not path.exists():
        available = ", ".join(sorted(item.stem for item in config_dir.glob("*.yaml")))
        raise SystemExit(
            f"Unknown robot {name!r}; expected {path}. Available: {available or 'none'}."
        )
    raw = path.read_bytes()
    config = yaml.safe_load(raw) or {}
    return {
        "name": name,
        "config_path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "configuration": config,
    }


def _recording_tcp_calibration_metadata(
    *,
    robot_metadata: dict[str, object],
    device: str,
    explicit_path: Path | None,
) -> tuple[dict[str, object], str]:
    """Resolve and snapshot the robot/gripper TCP setup used for recording."""
    robot = str(robot_metadata["name"])
    configuration = robot_metadata.get("configuration")
    if not isinstance(configuration, dict):
        raise SystemExit(f"Robot {robot!r} has invalid configuration metadata.")

    configured = configuration.get("controller_tcp_calibrations") or {}
    configured_path_value = (
        configured.get(device) if isinstance(configured, dict) else None
    )
    associated_with_robot_tool = configured_path_value is not None
    if explicit_path is not None:
        calibration_path = explicit_path
        source = f"explicit {calibration_path} for {robot}/{device}"
    elif configured_path_value is not None:
        calibration_path = Path(str(configured_path_value))
        source = f"configured {robot}/{device}: {calibration_path}"
    else:
        calibration_path = calibration_path_for_device(device)
        source = f"legacy device fallback {device}: {calibration_path}"
        log.warning(
            "Robot %s has no controller_tcp_calibrations.%s entry; using %s "
            "without treating it as a verified robot/gripper pairing.",
            robot,
            device,
            calibration_path,
        )

    tool = configuration.get("handumi_tool") or {}
    if not isinstance(tool, dict):
        raise SystemExit(f"Robot {robot!r} handumi_tool must be a mapping.")
    gripper = str(tool["gripper"]) if tool.get("gripper") else None
    controller_mount = (
        str(tool["controller_mount"]) if tool.get("controller_mount") else None
    )
    if associated_with_robot_tool and (gripper is None or controller_mount is None):
        raise SystemExit(
            f"Robot {robot!r} configures a {device} Controller->TCP calibration "
            "but is missing handumi_tool.gripper or handumi_tool.controller_mount."
        )

    metadata = controller_tcp_calibration_metadata(
        calibration_path,
        applied_to_state=False,
        source_robot=robot,
        source_gripper=gripper if associated_with_robot_tool else None,
        tracking_device=device,
        controller_mount=controller_mount if associated_with_robot_tool else None,
    )
    return metadata, source


def _validate_unique_camera_ids(
    camera_names: list[str],
    camera_ids: list[int | str],
) -> None:
    duplicates = {
        camera_id for camera_id in camera_ids if camera_ids.count(camera_id) > 1
    }
    if duplicates:
        mappings = ", ".join(
            f"{name}={camera_id}" for name, camera_id in zip(camera_names, camera_ids)
        )
        raise SystemExit(
            f"Selected cameras must use distinct devices ({mappings}). "
            "Fix the cameras section in configs/rig.yaml or pass matching --cam-ids."
        )


def _update_info_json(
    root: Path, handumi: dict[str, object]
) -> dict[str, object] | None:
    path = root / "meta" / "info.json"
    if not path.exists():
        log.warning("Cannot write HandUMI metadata; missing %s", path)
        return None
    info = json.loads(path.read_text())
    info["handumi"] = {**info.get("handumi", {}), **handumi}
    path.write_text(json.dumps(info, indent=4) + "\n")
    return info


def _dataset_card_kwargs(task: str) -> dict[str, str]:
    return {
        "dataset_description": (
            "Bimanual HandUMI demonstration data recorded in LeRobot v3 format. "
            f"Task: {task}"
        ),
        "url": "https://github.com/robonet-ai/handumi-sw",
    }


def _write_dataset_readme(
    root: Path,
    *,
    repo_id: str,
    task: str,
    license_id: str,
) -> dict[str, str]:
    """Create the same LeRobot dataset card locally that Hub upload uses."""
    from lerobot.datasets.utils import create_lerobot_dataset_card

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    kwargs = _dataset_card_kwargs(task)
    card = create_lerobot_dataset_card(
        tags=["HandUMI"],
        dataset_info=info,
        license=license_id,
        repo_id=repo_id,
        **kwargs,
    )
    card.save(root / "README.md")
    return kwargs


def _validate_finalized_lerobot_dataset(root: Path) -> None:
    """Reject incomplete LeRobot v3 datasets before upload or reported success."""
    import pyarrow.parquet as pq

    info_path = root / "meta" / "info.json"
    readme_path = root / "README.md"
    if not info_path.is_file() or not readme_path.is_file():
        raise RuntimeError("Dataset is missing meta/info.json or README.md.")
    info = json.loads(info_path.read_text())
    if info.get("codebase_version") != "v3.0":
        raise RuntimeError(
            f"Expected LeRobot v3.0, got {info.get('codebase_version')!r}."
        )

    total_episodes = int(info.get("total_episodes", 0))
    total_frames = int(info.get("total_frames", 0))
    if total_episodes <= 0 or total_frames <= 0:
        raise RuntimeError("Dataset contains no completed episodes.")

    required_meta = (root / "meta" / "stats.json", root / "meta" / "tasks.parquet")
    if not all(path.is_file() for path in required_meta):
        raise RuntimeError("Dataset is missing stats.json or tasks.parquet.")

    episode_files = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    data_files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not episode_files or not data_files:
        raise RuntimeError("Dataset is missing episode metadata or data Parquet files.")

    episode_indices: set[int] = set()
    for path in episode_files:
        try:
            table = pq.read_table(path, columns=["episode_index"])
        except Exception as exc:
            raise RuntimeError(f"Invalid episode metadata Parquet: {path}.") from exc
        episode_indices.update(int(value.as_py()) for value in table["episode_index"])
    expected_indices = set(range(total_episodes))
    if episode_indices != expected_indices:
        raise RuntimeError(
            f"Episode metadata mismatch: expected {sorted(expected_indices)}, "
            f"found {sorted(episode_indices)}."
        )

    try:
        parquet_frames = sum(
            pq.ParquetFile(path).metadata.num_rows for path in data_files
        )
    except Exception as exc:
        raise RuntimeError("One or more data Parquet files are incomplete.") from exc
    if parquet_frames != total_frames:
        raise RuntimeError(
            f"Frame count mismatch: info.json={total_frames}, parquet={parquet_frames}."
        )

    video_keys = [
        key
        for key, feature in (info.get("features") or {}).items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]
    missing_videos = [
        key
        for key in video_keys
        if not list((root / "videos" / key).glob("chunk-*/*.mp4"))
    ]
    if missing_videos:
        raise RuntimeError(
            f"Dataset is missing videos for: {', '.join(missing_videos)}."
        )


def _camera_arg(value: str) -> int | str:
    return int(value) if value.isdigit() else value


if __name__ == "__main__":
    main()
