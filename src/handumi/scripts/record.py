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
import hashlib
import json
import logging
import os
import select
import signal
import sys
import termios
import threading
import time
import tty
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

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
from handumi.feetech import (
    FeetechGripperPair,
    FeetechGripperSampler,
    GripperWidths,
    assert_calibrated,
    load_config,
    user_calibration_path,
    zero_gripper_widths,
)
from handumi.feetech.bus import FeetechUnavailableError
from handumi.robots.utils import IDENTITY_POSE7
from handumi.synchronization import (
    SustainedHealthGate,
    capture_timing_frame,
    synchronized_gripper_frame,
    tracking_sample_at,
)
from handumi.tracking.base import ControllerPairSample, TrackingProvider
from handumi.tracking.gestures import DoubleClapDetector
from handumi.tracking.meta_quest import MetaQuestConfig, MetaQuestTrackingProvider
from handumi.tracking.pico import (
    START_BUTTON_CHOICES,
    PicoTrackingProvider,
    read_start_button_value,
    wait_for_manual_start,
    wait_for_start_button,
)
from handumi.tracking.transforms import Pose
from handumi.utils.speech import log_say
from handumi.utils.trajectory import TrajectoryTrail
from handumi.visualization import BACKGROUND_COLOR, LEFT_COLOR, RIGHT_COLOR

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("handumi.record")

ROBOT_CONFIG_DIR = Path("configs/robots")
_RERUN_TRAIL_SECONDS = 10.0
_RERUN_CHART_WINDOW_S = 20.0


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


def build_features(
    cam_names: list[str],
    cam_width: int,
    cam_height: int,
    use_videos: bool,
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
    return features


def _tuple_shape(feature: dict) -> dict:
    feature = dict(feature)
    feature["shape"] = tuple(feature["shape"])
    return feature


def build_observation(sample: ControllerPairSample, widths: GripperWidths) -> dict:
    left_controller = _pose_from_pose7(sample.left_controller_pose)
    right_controller = _pose_from_pose7(sample.right_controller_pose)
    state = pose_to_state_vector(
        left_controller,
        right_controller,
        widths.left,
        widths.right,
    )
    return {
        "observation.state": state,
        "action": state.copy(),
        "observation.feetech.left_ticks": np.array([widths.left_ticks], dtype=np.int64),
        "observation.feetech.right_ticks": np.array([widths.right_ticks], dtype=np.int64),
        "observation.feetech.left_width_mm": np.array([widths.left_mm], dtype=np.float32),
        "observation.feetech.right_width_mm": np.array([widths.right_mm], dtype=np.float32),
        "observation.feetech.left_normalized": np.array([widths.left_normalized], dtype=np.float32),
        "observation.feetech.right_normalized": np.array([widths.right_normalized], dtype=np.float32),
        **sample.tracking_frame(),
    }


def _pose_from_pose7(pose7: np.ndarray) -> Pose:
    pose = np.asarray(pose7, dtype=np.float32).reshape(7)
    return Pose(pose[:3], pose[3:7])


def _tracking_healthy(sample: ControllerPairSample) -> bool:
    return bool(sample.left_tracked and sample.right_tracked)


def _wait_for_tracking(
    tracker: TrackingProvider,
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
    """Live Rerun stream owned by the recorder (never opens another device)."""

    def __init__(self, cam_names: list[str], fps: int) -> None:
        import rerun as rr
        import rerun.blueprint as rrb
        import rerun.datatypes as rdt

        self.rr = rr
        self.trails = {
            "left": TrajectoryTrail(max(2, int(_RERUN_TRAIL_SECONDS * fps))),
            "right": TrajectoryTrail(max(2, int(_RERUN_TRAIL_SECONDS * fps))),
        }
        self.raw_trails = {
            "left": TrajectoryTrail(max(2, int(_RERUN_TRAIL_SECONDS * fps))),
            "right": TrajectoryTrail(max(2, int(_RERUN_TRAIL_SECONDS * fps))),
        }
        self._healthy = True
        rr.init("handumi_record", spawn=True)
        rr.log("tracking", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        for path, name, color in (
            ("observation.feetech.left_width_mm", "left_width_mm", LEFT_COLOR),
            ("observation.feetech.right_width_mm", "right_width_mm", RIGHT_COLOR),
        ):
            rr.log(path, rr.SeriesLines(colors=[[*color, 255]], widths=[2.5], names=[name]), static=True)

        corners = [[sx * 0.75, sy * 0.75, sz * 0.4]
                   for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        rr.log("tracking/bounds", rr.Points3D(corners, colors=[[128, 100, 100, 90]] * 8, radii=0.004), static=True)
        recent = rrb.VisibleTimeRanges(rrb.VisibleTimeRange(
            timeline="log_time",
            range=rdt.TimeRange(
                start=rdt.TimeRangeBoundary.cursor_relative(seconds=-_RERUN_CHART_WINDOW_S),
                end=rdt.TimeRangeBoundary.cursor_relative(seconds=0.0),
            ),
        ))
        chart = rrb.TimeSeriesView(
            origin="/",
            contents=["/observation.feetech.left_width_mm", "/observation.feetech.right_width_mm"],
            name="gripper_width_mm",
            axis_y=rrb.ScalarAxis(range=(0.0, 90.0)),
            time_ranges=recent,
            plot_legend=rrb.Corner2D.LeftTop,
        )
        right = chart if not cam_names else rrb.Vertical(
            rrb.Horizontal(*[
                rrb.Spatial2DView(origin=f"/observation.images.{name}", name=name)
                for name in cam_names
            ]),
            chart,
            row_shares=[3, 2],
        )
        status = rrb.TextDocumentView(
            origin="/recording",
            contents=["/recording/status"],
            name="recording_status",
        )
        rr.send_blueprint(rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial3DView(origin="/tracking", name="controller_trajectory",
                                       background=rrb.Background(color=[*BACKGROUND_COLOR, 255])),
                    right,
                    column_shares=[2, 3],
                ),
                status,
                row_shares=[10, 1],
            ),
            rrb.BlueprintPanel(state="collapsed"),
            rrb.SelectionPanel(state="collapsed"),
            rrb.TimePanel(state="collapsed"),
        ), make_active=True, make_default=True)
        self.set_status("READY", "Waiting to start the first episode")

    def set_status(self, state: str, detail: str) -> None:
        """Show the current recorder state as a persistent operator flag."""
        self.rr.log(
            "recording/status",
            self.rr.TextDocument(
                f"# {state}\n\n{detail}", media_type="text/markdown"
            ),
        )

    def log(self, cam_frames: dict, sample: ControllerPairSample, widths: GripperWidths) -> None:
        """Log a frame without allowing a viewer problem to stop recording."""
        if not self._healthy:
            return
        rr = self.rr
        try:
            # ``cam_frames`` also contains camera-health scalar fields. Only
            # image entries belong in Rerun's image archetype.
            for key, frame in cam_frames.items():
                if key.startswith("observation.images."):
                    rr.log(key, rr.Image(frame).compress(jpeg_quality=75))
            rr.log("observation.feetech.left_width_mm", rr.Scalars(float(widths.left_mm)))
            rr.log("observation.feetech.right_width_mm", rr.Scalars(float(widths.right_mm)))
            for side, tcp, raw, color, tracked in (
                ("left", sample.left_tcp_pose, sample.left_controller_pose, LEFT_COLOR, sample.left_tracked),
                ("right", sample.right_tcp_pose, sample.right_controller_pose, RIGHT_COLOR, sample.right_tracked),
            ):
                if not tracked:
                    continue
                trail = self.trails[side]
                trail.append(tcp[:3])
                rr.log(f"tracking/{side}/tcp", rr.Points3D([tcp[:3]], colors=[color], radii=0.012))
                if len(points := trail.points()) >= 2:
                    rr.log(f"tracking/{side}/trail", rr.LineStrips3D([points], colors=[color], radii=0.003))
                raw_trail = self.raw_trails[side]
                raw_trail.append(raw[:3])
                rr.log(f"tracking/{side}/raw", rr.Points3D([raw[:3]], colors=[[*color, 90]], radii=0.007))
                if len(raw_points := raw_trail.points()) >= 2:
                    rr.log(f"tracking/{side}/raw_trail", rr.LineStrips3D(
                        [raw_points], colors=[[*color, 90]], radii=0.0015
                    ))
        except Exception:
            self._healthy = False
            log.exception("Rerun failed; disabling live view while recording continues.")


def record_episode(
    *,
    dataset,
    cameras: list,
    cam_names: list[str],
    tracker: TrackingProvider,
    grippers: FeetechGripperSampler | FeetechGripperPair | None,
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
    rerun: _RecordingRerun | None = None,
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

    while True:
        loop_start = time.perf_counter()
        tracking_now_ns = time.monotonic_ns()
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
            start_pressed = read_start_button_value(xrt, start_button) >= start_threshold
            repeat_pressed = read_start_button_value(xrt, repeat_button) >= start_threshold
            finish_pressed = read_start_button_value(xrt, finish_button) >= start_threshold
            start_rise = start_pressed and not prev_start
            repeat_rise = repeat_pressed and not prev_repeat
            finish_rise = finish_pressed and not prev_finish
            prev_start, prev_repeat, prev_finish = start_pressed, repeat_pressed, finish_pressed
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
        gripper_frame = synchronized_gripper_frame(
            grippers,
            target_time_ns=target_time_ns,
            record_time_ns=tracking_now_ns,
            stale_timeout_s=gripper_stale_timeout_s,
            max_sync_skew_s=max_sync_skew_s,
        )
        widths = gripper_frame.widths
        sample = tracking_sample_at(tracker, target_time_ns)
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
            rerun.log(cam_frames, sample, widths)
        dataset.add_frame(
            {
                **cam_frames,
                **build_observation(sample, widths),
                **gripper_frame.frame,
                **capture_timing_frame(target_time_ns, tracking_now_ns),
                "task": task,
            }
        )
        n_frames += 1

        dt = time.perf_counter() - loop_start
        sleep = control_interval - dt
        if sleep > 0:
            time.sleep(sleep)
        else:
            log.warning("Loop slower than %d Hz (%.1f Hz actual).", fps, 1.0 / max(dt, 1e-6))

    return n_frames, status


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record HandUMI data with PICO or Meta Quest.")
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

    p.add_argument("--pico-mode", choices=("mandos", "object", "whole-body"), default="mandos")
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


def main() -> None:
    args = parse_args()
    if args.manual_control and args.device != "pico":
        raise SystemExit("--manual-control currently requires --device pico.")
    if args.manual_control and args.start_button == "enter":
        args.start_button = "A"
        log.info("--manual-control set: using PICO A as start/stop button.")
    if args.clap_control and args.skip_feetech:
        raise SystemExit("--clap-control needs real Feetech widths; drop --skip-feetech.")
    if args.clap_control and args.manual_control:
        raise SystemExit("--clap-control and --manual-control are mutually exclusive.")
    if args.session_calibration is not None and args.device != "meta":
        raise SystemExit("--session-calibration currently requires --device meta.")
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
    if args.output_dir is None:
        args.output_dir = _default_output_dir()
    play_sounds = not args.no_sounds

    camera_names = _selected_camera_names(args)
    robot_metadata = _robot_metadata(args.robot)
    calibration_metadata, calibration_source = _recording_tcp_calibration_metadata(
        robot_metadata=robot_metadata,
        device=args.device,
        explicit_path=args.controller_tcp_calibration,
    )
    log.info("Controller->TCP setup: %s", calibration_source)
    try:
        spatial_session_metadata = session_calibration_metadata(args.session_calibration)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid session calibration: {exc}") from exc
    log.info("--- Tracking setup ---")
    calibration = ControllerTcpCalibration(
        left=IDENTITY_POSE7.astype(np.float32).copy(),
        right=IDENTITY_POSE7.astype(np.float32).copy(),
        source=None,
    )
    tracker = build_tracker(args, calibration)
    if args.session_calibration is not None:
        session_device = str(spatial_session_metadata.get("tracking_device") or "")
        if session_device and session_device != args.device:
            raise SystemExit(
                f"Session calibration is for {session_device}, "
                f"but --device {args.device} was selected."
            )
        set_workspace = getattr(tracker, "set_workspace_from_device_pose", None)
        if set_workspace is None:
            raise SystemExit("Selected tracking backend cannot apply a table calibration.")
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
    cameras = connect_cameras(
        camera_specs,
        fps=args.cam_fps,
        width=args.cam_width,
        height=args.cam_height,
        zero_non_laptop=False,
    )
    rerun = _RecordingRerun(cam_names, args.fps) if args.rerun else None

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
        while (args.num_episodes <= 0 or recorded < args.num_episodes) and not stop_event.is_set():
            ep_num = dataset.num_episodes + 1
            ep_total = "inf" if args.num_episodes <= 0 else str(args.num_episodes)
            log.info("--- Episode %d/%s ---", ep_num, ep_total)
            if rerun is not None:
                rerun.set_status("WAITING", f"Episode {ep_num}/{ep_total}: waiting to start")
            if args.clap_control:
                assert clap_detector is not None
                if restart_active:
                    restart_active = False
                    log.info("  Restarting episode %d immediately ...", ep_num)
                    if rerun is not None:
                        rerun.set_status("RESTARTED", f"Episode {ep_num}/{ep_total}: restarting now")
                else:
                    log.info("  Double-squeeze right gripper to start episode %d ...", ep_num)
                    if not _wait_for_clap(
                        grippers, clap_detector, stop_event, side="right"
                    ):
                        break
                    # A calibrated table workspace is locked and ignores this
                    # legacy HMD recenter; uncalibrated sessions retain it.
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
                input(f"  Press ENTER to start recording episode {ep_num} ...")
            elif args.device == "pico":
                if not wait_for_start_button(
                    getattr(tracker, "xrt"),
                    button=args.start_button,
                    threshold=args.start_threshold,
                    stop_event=stop_event,
                ):
                    break
            else:
                raise SystemExit("--start-button other than enter currently requires --device pico.")

            if not _wait_for_tracking(tracker, stop_event):
                break
            log_say(f"Recording episode {ep_num}", play_sounds=play_sounds)
            if rerun is not None:
                rerun.set_status("RECORDING", f"Episode {ep_num}/{ep_total} is being recorded")
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
                rerun=rerun,
            )
            if status == "repeat":
                log.warning("Episode restart requested (%d frames discarded).", n_frames)
                log_say("Restart recording", play_sounds=play_sounds)
                if rerun is not None:
                    rerun.set_status(
                        "RESTARTED",
                        f"Episode {ep_num}/{ep_total}: {n_frames} frames discarded; restarting",
                    )
                dataset.clear_episode_buffer()
                restart_active = True
                continue
            if n_frames == 0 or status in {
                "tracking_lost",
                "sensor_unhealthy",
                "interrupted",
            }:
                log.warning("Episode discarded (%s, %d frames).", status, n_frames)
                log_say("Episode discarded", play_sounds=play_sounds)
                if rerun is not None:
                    rerun.set_status(
                        "DISCARDED",
                        f"Episode {ep_num}/{ep_total}: {status} after {n_frames} frames",
                    )
                dataset.clear_episode_buffer()
                if status in {"finish", "interrupted"}:
                    break
                continue
            dataset.save_episode()
            recorded += 1
            log.info("Episode %d saved (%d frames).", ep_num, n_frames)
            log_say(f"Episode {ep_num} saved, {n_frames} frames", play_sounds=play_sounds)
            if rerun is not None:
                rerun.set_status(
                    "SAVED", f"Episode {ep_num}/{ep_total}: {n_frames} frames saved"
                )
            if status == "finish":
                break
    finally:
        escape_listener.stop()
        if rerun is not None:
            rerun.set_status("STOPPED", f"Recording stopped: {recorded} episode(s) saved")
        log_say("Stop recording", play_sounds=play_sounds, blocking=True)
        log.info("--- Finalising ---")
        finalization_error: BaseException | None = None
        try:
            dataset.finalize()
            root = Path(dataset.root)
            updated_info = _update_info_json(
                root,
                {
                    "recording_device": args.device,
                    "tracking_schema": HANDUMI_TRACKING_SCHEMA,
                    "tracking_workspace": (
                        "table" if spatial_session_metadata is not None else "hmd_recentered"
                    ),
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
            if args.push_to_hub:
                dataset.push_to_hub(
                    license=args.dataset_license,
                    tags=["HandUMI"],
                    **card_kwargs,
                )
        except BaseException as exc:
            finalization_error = exc
            log.exception("Dataset finalization failed; do not upload this dataset.")
        finally:
            disconnect_cameras(cameras)
            if grippers is not None:
                grippers.stop()
            if gripper_pair is not None:
                gripper_pair.close()
            tracker.stop()
        if finalization_error is not None:
            raise finalization_error
        log.info("Done. Recorded %d episode(s). Dataset at: %s", recorded, dataset.root)
        log_say("Exiting", play_sounds=play_sounds)


def build_tracker(
    args: argparse.Namespace, calibration, *, reset_workspace_on_x: bool = True
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
    )
    return MetaQuestTrackingProvider(
        config=config, calibration=calibration, reset_workspace_on_x=reset_workspace_on_x
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
    grippers: FeetechGripperSampler | FeetechGripperPair | None,
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
    grippers: FeetechGripperSampler | FeetechGripperPair | None,
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
    cameras: list[object | None],
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


def _robot_metadata(name: str, config_dir: Path = ROBOT_CONFIG_DIR) -> dict[str, object]:
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
    configured_path_value = configured.get(device) if isinstance(configured, dict) else None
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
        camera_id
        for camera_id in camera_ids
        if camera_ids.count(camera_id) > 1
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
        key for key in video_keys if not list((root / "videos" / key).glob("chunk-*/*.mp4"))
    ]
    if missing_videos:
        raise RuntimeError(f"Dataset is missing videos for: {', '.join(missing_videos)}.")


def _camera_arg(value: str) -> int | str:
    return int(value) if value.isdigit() else value


if __name__ == "__main__":
    main()
