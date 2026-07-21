"""Non-blocking robot visualization driven by recorder-owned samples.

The recorder owns every tracking, camera, and Feetech connection.  This module
only receives immutable copies of the already-aligned controller/TCP and
gripper sample used for a dataset row, then runs IK and Viser on a bounded
worker queue.  Viewer or IK failures are diagnostics and never stop capture.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from handumi.retargeting import VR_TO_ROBOT
from handumi.robots.registry import load_embodiment, resolve_home_q
from handumi.teleop.core import TeleopController
from handumi.visualization import LEFT_COLOR, RIGHT_COLOR

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecorderRobotFrame:
    """One robot-view input copied from an aligned dataset-row sample."""

    sample_time_ns: int
    left_tcp_pose: np.ndarray
    right_tcp_pose: np.ndarray
    left_tracked: bool
    right_tracked: bool
    left_gripper_opening: float
    right_gripper_opening: float

    def copied(self) -> "RecorderRobotFrame":
        return replace(
            self,
            left_tcp_pose=np.asarray(self.left_tcp_pose, dtype=np.float32)
            .reshape(7)
            .copy(),
            right_tcp_pose=np.asarray(self.right_tcp_pose, dtype=np.float32)
            .reshape(7)
            .copy(),
        )


@dataclass(frozen=True)
class RecorderRobotViewerConfig:
    robot: str = "piper"
    device: str = "meta"
    host: str = "127.0.0.1"
    port: int = 8003
    rig_config: Path = Path("configs/rig.yaml")
    home_pose: str | None = None
    scene: str | None = None
    anchor_mode: str = "episode-start"
    anchor_z: float | None = None
    queue_size: int = 2

    def __post_init__(self) -> None:
        if self.device not in {"meta", "pico"}:
            raise ValueError("device must be 'meta' or 'pico'")
        if self.anchor_mode not in {"episode-start", "first-tracked", "disabled"}:
            raise ValueError(
                "anchor_mode must be episode-start, first-tracked, or disabled"
            )
        if not self.host:
            raise ValueError("host must not be empty")
        if not 0 <= self.port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        if self.queue_size <= 0:
            raise ValueError("queue_size must be positive")


@dataclass(frozen=True)
class RecorderRobotViewerStatus:
    lifecycle: str = "starting"
    recording_state: str = "WAITING"
    anchor_state: str = "idle"
    tracking_state: str = "unknown"
    ik_state: str = "idle"
    submitted_frames: int = 0
    processed_frames: int = 0
    dropped_frames: int = 0
    failures: int = 0
    last_error: str | None = None


class RecorderRobotSink(Protocol):
    def submit(self, frame: RecorderRobotFrame) -> bool: ...

    def set_recording_state(self, state: str, detail: str = "") -> None: ...

    def status(self) -> RecorderRobotViewerStatus: ...

    def close(self) -> None: ...


class RobotViewerPipeline(Protocol):
    def set_recording_state(self, state: str, detail: str) -> None: ...

    def process(self, frame: RecorderRobotFrame) -> tuple[str, str, str]: ...

    def mark_failure(self, message: str) -> None: ...

    def close(self) -> None: ...


class QueuedRecorderRobotViewer:
    """Bounded asynchronous sink whose failure boundary is capture-safe."""

    def __init__(
        self,
        config: RecorderRobotViewerConfig,
        *,
        pipeline_factory: Callable[[RecorderRobotViewerConfig], RobotViewerPipeline]
        | None = None,
    ) -> None:
        self.config = config
        self._factory = pipeline_factory or _ViserRobotPipeline
        self._queue: queue.Queue[RecorderRobotFrame] = queue.Queue(
            maxsize=config.queue_size
        )
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._status = RecorderRobotViewerStatus()
        self._state_queue: queue.SimpleQueue[tuple[str, str]] = queue.SimpleQueue()
        self._worker = threading.Thread(
            target=self._run,
            name="handumi-recorder-viser",
            daemon=True,
        )
        self._worker.start()

    def submit(self, frame: RecorderRobotFrame) -> bool:
        if self._stop.is_set():
            return False
        with self._lock:
            self._status = replace(
                self._status,
                submitted_frames=self._status.submitted_frames + 1,
            )
        return self._put_latest(frame.copied())

    def set_recording_state(self, state: str, detail: str = "") -> None:
        normalized = str(state).strip().upper() or "UNKNOWN"
        with self._lock:
            self._status = replace(self._status, recording_state=normalized)
        self._state_queue.put((normalized, str(detail)))

    def status(self) -> RecorderRobotViewerStatus:
        with self._lock:
            return self._status

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._worker.join(timeout=5.0)
        with self._lock:
            lifecycle = "stopped" if not self._worker.is_alive() else "shutdown-timeout"
            self._status = replace(self._status, lifecycle=lifecycle)

    def _put_latest(self, item: RecorderRobotFrame) -> bool:
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            try:
                dropped = self._queue.get_nowait()
            except queue.Empty:
                dropped = None
            if dropped is not None:
                with self._lock:
                    self._status = replace(
                        self._status,
                        dropped_frames=self._status.dropped_frames + 1,
                    )
            try:
                self._queue.put_nowait(item)
                return True
            except queue.Full:
                return False

    def _run(self) -> None:
        pipeline: RobotViewerPipeline | None = None
        try:
            pipeline = self._factory(self.config)
            with self._lock:
                self._status = replace(self._status, lifecycle="ready")
        except Exception as exc:  # noqa: BLE001 - viewer must not stop capture.
            self._record_failure(exc, lifecycle="failed")
            log.exception("Recorder Viser initialization failed; recording continues")
            return

        try:
            while not self._stop.is_set():
                self._forward_pending_states(pipeline)
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                # A state transition can arrive while the worker is blocked in
                # get(). Apply it before the first frame of the new episode.
                self._forward_pending_states(pipeline)
                try:
                    anchor_state, tracking_state, ik_state = pipeline.process(item)
                    with self._lock:
                        self._status = replace(
                            self._status,
                            anchor_state=anchor_state,
                            tracking_state=tracking_state,
                            ik_state=ik_state,
                            processed_frames=self._status.processed_frames + 1,
                        )
                except Exception as exc:  # noqa: BLE001 - per-frame isolation.
                    self._record_failure(exc, lifecycle="degraded")
                    pipeline.mark_failure(str(exc))
                    log.exception("Recorder Viser frame failed; recording continues")
        finally:
            try:
                pipeline.close()
            except Exception as exc:  # noqa: BLE001 - cleanup remains best effort.
                self._record_failure(exc, lifecycle="shutdown-failed")

    def _forward_pending_states(self, pipeline: RobotViewerPipeline) -> None:
        while True:
            try:
                pending_state = self._state_queue.get_nowait()
            except queue.Empty:
                return
            try:
                pipeline.set_recording_state(*pending_state)
            except Exception as exc:  # noqa: BLE001 - viewer isolation.
                self._record_failure(exc, lifecycle="degraded")
                pipeline.mark_failure(str(exc))
                log.exception("Recorder Viser state update failed; recording continues")

    def _record_failure(self, exc: BaseException, *, lifecycle: str) -> None:
        message = f"{type(exc).__name__}: {exc}"
        with self._lock:
            self._status = replace(
                self._status,
                lifecycle=lifecycle,
                failures=self._status.failures + 1,
                last_error=message,
            )


class _ViserRobotPipeline:
    """Worker-thread-owned Viser server and robot IK state."""

    def __init__(self, config: RecorderRobotViewerConfig) -> None:
        import viser
        from viser.extras import ViserUrdf

        self.config = config
        self.runtime = load_embodiment(config.robot)
        _, home_q = resolve_home_q(
            self.runtime,
            rig_config=config.rig_config,
            explicit_name=config.home_pose,
        )
        source_world = (
            VR_TO_ROBOT if config.device == "pico" else np.eye(3, dtype=np.float32)
        )
        self.controller = TeleopController(
            self.runtime,
            home_q=home_q,
            enabled_sides=("left", "right"),
            source_world_to_robot_world=source_world,
            anchor_z=config.anchor_z,
        )
        self.controller.warmup()
        self.server = viser.ViserServer(
            host=config.host,
            port=config.port,
            label="HandUMI recorder robot view",
        )
        self.server.scene.add_grid("/grid", width=3.0, height=3.0, cell_size=0.1)
        self.robot_view = ViserUrdf(
            self.server,
            self.runtime.load_urdf(load_meshes=True),
            root_node_name="/robot",
        )
        self.robot_view.update_cfg(home_q)
        self.targets = {
            "left": self.server.scene.add_icosphere(
                "/target/left", radius=0.018, color=LEFT_COLOR
            ),
            "right": self.server.scene.add_icosphere(
                "/target/right", radius=0.018, color=RIGHT_COLOR
            ),
        }
        self.status_markdown = self.server.gui.add_markdown(
            "HandUMI recorder viewer starting"
        )
        self.recording_state = "WAITING"
        self.recording_detail = ""
        self.pending_episode_anchor = False
        self.last_error: str | None = None
        self._add_scene(config.scene)
        self._update_status("idle", "unknown", "idle")

    def _add_scene(self, scene: str | None) -> None:
        if scene is None:
            return
        from handumi.sim.scene import DEFAULT_SCENE_POSITION, load_scene

        for body in load_scene(scene, position=DEFAULT_SCENE_POSITION):
            frame = self.server.scene.add_frame(
                f"/scene/{body.name}",
                position=tuple(body.rest_position),
                show_axes=False,
            )
            for index, geom in enumerate(body.geoms):
                sx, sy, sz = (2.0 * float(value) for value in geom.size)
                cr, cg, cb = (int(round(float(value) * 255)) for value in geom.rgba[:3])
                self.server.scene.add_box(
                    f"/scene/{body.name}/g{index}",
                    dimensions=(sx, sy, sz),
                    color=(cr, cg, cb),
                    position=tuple(geom.local_position),
                )
            del frame

    def set_recording_state(self, state: str, detail: str) -> None:
        self.recording_state = state
        self.recording_detail = detail
        if state in {"WAITING", "RESTARTED"}:
            self.controller.reset()
            self.pending_episode_anchor = False
        elif state == "RECORDING" and self.config.anchor_mode == "episode-start":
            self.pending_episode_anchor = True
        self._update_status(
            "anchored" if self.controller.active else "idle",
            "unknown",
            "idle",
        )

    def process(self, frame: RecorderRobotFrame) -> tuple[str, str, str]:
        poses = {
            "left": frame.left_tcp_pose,
            "right": frame.right_tcp_pose,
        }
        tracked = {
            "left": bool(frame.left_tracked),
            "right": bool(frame.right_tracked),
        }
        tracking_state = (
            "both-tracked"
            if all(tracked.values())
            else "tracking-loss:"
            + ",".join(side for side, valid in tracked.items() if not valid)
        )
        should_anchor = (
            self.config.anchor_mode == "first-tracked" and not self.controller.active
        ) or (
            self.config.anchor_mode == "episode-start" and self.pending_episode_anchor
        )
        if should_anchor and all(tracked.values()):
            self.controller.anchor(poses, tracked, ("left", "right"))
            self.pending_episode_anchor = False

        step = self.controller.step(
            poses,
            tracked,
            {
                "left": float(frame.left_gripper_opening),
                "right": float(frame.right_gripper_opening),
            },
        )
        self.robot_view.update_cfg(step.q)
        for side, pose7 in step.target_pose7.items():
            self.targets[side].position = tuple(pose7[:3])
        anchor_state = "anchored" if self.controller.active else "idle"
        if step.reach_limited_sides:
            ik_state = "reach-limited:" + ",".join(step.reach_limited_sides)
        else:
            ik_state = "solved" if self.controller.active else "idle"
        self._update_status(anchor_state, tracking_state, ik_state)
        return anchor_state, tracking_state, ik_state

    def mark_failure(self, message: str) -> None:
        self.last_error = message
        self._update_status(
            "anchored" if self.controller.active else "idle",
            "unknown",
            "failed",
        )

    def _update_status(
        self, anchor_state: str, tracking_state: str, ik_state: str
    ) -> None:
        error = "none" if self.last_error is None else self.last_error
        detail = self.recording_detail or "none"
        self.status_markdown.content = (
            "### HandUMI recorder robot view\n"
            f"- Recording: **{self.recording_state}** ({detail})\n"
            f"- Anchor: **{anchor_state}** ({self.config.anchor_mode})\n"
            f"- Tracking: **{tracking_state}**\n"
            f"- IK/reachability: **{ik_state}**; configured reach limits apply\n"
            f"- Last viewer failure: **{error}**"
        )

    def close(self) -> None:
        self.server.stop()


__all__ = [
    "QueuedRecorderRobotViewer",
    "RecorderRobotFrame",
    "RecorderRobotSink",
    "RecorderRobotViewerConfig",
    "RecorderRobotViewerStatus",
    "RobotViewerPipeline",
]
