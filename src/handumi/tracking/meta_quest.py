"""HandUMI Quest App tracking receiver.

The native Quest app is a TCP server that streams one
newline-delimited JSON pose sample per frame; the workstation (this module)
dials in, parses each sample, stamps a PC receive clock, and keeps the latest
frame in a buffer. A companion UDP NTP-style loop estimates the Quest<->PC clock
offset so poses can be aligned with camera/Feetech frames in post-processing.

This step does NOT transform coordinates — poses are kept as raw Unity values.
``handumi.tracking.transforms`` (Step 2) converts them.

App and protocol reference:
  https://github.com/robonet-ai/handumi-quest-app

Wire protocol (see docs/phase-2-motion-tracking.md → TCP/JSON Payload):
  TCP : newline-delimited JSON pose samples (Quest app is the server).
  UDP : NTP-style time-sync. The PC sends ``<B Q>`` = (1, t1_pc_ns); the Quest
        echoes ``<B Q Q>`` = (2, t1_echo, t2_quest_ns) to the sender address.
"""

from __future__ import annotations

import json
import logging
import socket
import statistics
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from handumi.calibration.control_tcp import ControllerTcpCalibration
from handumi.config import load_rig_section
from handumi.tracking.base import ControllerPairSample, apply_tcp_calibration_pose7
from handumi.tracking.transforms import (
    Pose,
    WorkspaceCalibration,
    apply_mounting_offset,
    unity_pose_to_handumi,
)

log = logging.getLogger("handumi.tracking.meta_quest")

# UDP sync wire formats.
_PING = struct.Struct("<BQ")  # (msg_type=1, t1_pc_ns)
_PONG = struct.Struct("<BQQ")  # (msg_type=2, t1_echo, t2_quest_ns)
_PING_TYPE = 1
_PONG_TYPE = 2

_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Frame model (raw Unity coordinates — no transforms at this step).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ControllerButtons:
    """Per-controller inputs. Analog values are UI/debug only — never width.

    The compatibility TCP/JSON format reports button *presses* only (no analog
    trigger/grip), so ``trigger``/``grip`` here are 0.0/1.0 from the pressed
    flags, not continuous values.
    """

    trigger: float = 0.0
    grip: float = 0.0
    thumbstick: tuple[float, float] = (0.0, 0.0)
    thumbstick_click: bool = False
    primary: bool = False  # X (left) / A (right)
    secondary: bool = False  # Y (left) / B (right)


@dataclass(frozen=True)
class ControllerState:
    """A controller pose in raw Unity coordinates plus OVR tracking flags."""

    tracked: bool
    valid: bool
    position: np.ndarray  # (3,) meters, Unity left-handed
    quaternion: np.ndarray  # (4,) [x, y, z, w], Unity left-handed
    buttons: ControllerButtons


@dataclass(frozen=True)
class HmdState:
    """The headset pose (body/chest reference frame), raw Unity coordinates."""

    tracked: bool
    position: np.ndarray  # (3,)
    quaternion: np.ndarray  # (4,) [x, y, z, w]


@dataclass(frozen=True)
class QuestFrame:
    """One parsed pose sample with both device and PC clocks."""

    seq: int
    device_time_ns: int  # Quest monotonic clock at sample generation
    pc_monotonic_ns: int  # time.monotonic_ns() at receive on the workstation
    delta_time_s: float  # Quest-reported frame delta (NaN if absent)
    space: str
    hmd: HmdState
    left: ControllerState
    right: ControllerState
    battery: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    receive_sequence: int = 0


@dataclass(frozen=True)
class AlignedQuestFrame:
    """Quest frame mapped onto the workstation monotonic clock."""

    frame: QuestFrame
    aligned_time_ns: int
    clock_offset_ns: int
    clock_synced: bool


def controller_pose_in_workspace(
    controller: ControllerState,
    *,
    mounting_offset: Pose,
    workspace: WorkspaceCalibration,
) -> Pose:
    """Calibrated gripper-TCP pose for one Quest controller (raw Unity -> workspace)."""
    converted = unity_pose_to_handumi(controller.position, controller.quaternion)
    gripper_tcp = apply_mounting_offset(converted, mounting_offset)
    return workspace.apply(gripper_tcp)


def workspace_from_hmd(hmd: HmdState) -> WorkspaceCalibration:
    """Build a workspace reset that re-centers on the current Quest HMD pose."""
    reference = unity_pose_to_handumi(hmd.position, hmd.quaternion)
    return WorkspaceCalibration.from_reference(reference)


def pose_to_pose7(pose: Pose) -> np.ndarray:
    """Return ``[x,y,z,qx,qy,qz,qw]`` for a tracking :class:`Pose`."""
    return np.concatenate([pose.position, pose.quaternion]).astype(np.float32)


# Flat HandUMI Quest App wire keys, per side.
_POSE_KEYS = {
    "left": {
        "pos": "leftControllerPosition", "rot": "leftControllerRotation",
        "tracked": "leftTracked", "valid": "leftValid",
    },
    "right": {
        "pos": "rightControllerPosition", "rot": "rightControllerRotation",
        "tracked": "rightTracked", "valid": "rightValid",
    },
}
_BUTTON_KEYS = {
    "left": {
        "primary": "buttonXPressed", "secondary": "buttonYPressed",
        "trigger": "leftTriggerPressed", "grip": "leftGripPressed",
        "stick": "leftJoystick", "stick_click": "leftThumbstickClick",
    },
    "right": {
        "primary": "buttonAPressed", "secondary": "buttonBPressed",
        "trigger": "rightTriggerPressed", "grip": "rightGripPressed",
        "stick": "rightJoystick", "stick_click": "rightThumbstickClick",
    },
}


def _controller_from_msg(msg: dict[str, Any], side: str) -> ControllerState:
    pk, bk = _POSE_KEYS[side], _BUTTON_KEYS[side]
    return ControllerState(
        tracked=bool(msg.get(pk["tracked"], False)),
        valid=bool(msg.get(pk["valid"], False)),
        position=_vec3(msg.get(pk["pos"])),
        quaternion=_quat(msg.get(pk["rot"])),
        buttons=ControllerButtons(
            trigger=1.0 if msg.get(bk["trigger"]) else 0.0,
            grip=1.0 if msg.get(bk["grip"]) else 0.0,
            thumbstick=_stick(msg.get(bk["stick"])),
            thumbstick_click=bool(msg.get(bk["stick_click"], False)),
            primary=bool(msg.get(bk["primary"], False)),
            secondary=bool(msg.get(bk["secondary"], False)),
        ),
    )


def parse_frame(msg: dict[str, Any], *, pc_monotonic_ns: int) -> QuestFrame:
    """Build a :class:`QuestFrame` from one HandUMI Quest App JSON sample.

    Wire format uses a flat layout (Unity coordinates, dict-shaped
    vectors). Pure function so it is trivial to unit-test against the contract.
    """
    hmd_pos = msg.get("hmdPosition")
    hmd_rot = msg.get("hmdRotation")
    return QuestFrame(
        seq=int(msg.get("seq", 0) or 0),
        device_time_ns=int(msg.get("ovrTimeNs", 0) or 0),
        pc_monotonic_ns=int(pc_monotonic_ns),
        delta_time_s=_as_float(msg.get("deltaTime"), default=float("nan")),
        space=str(msg.get("space", "")),
        hmd=HmdState(
            tracked=hmd_pos is not None and hmd_rot is not None,
            position=_vec3(hmd_pos),
            quaternion=_quat(hmd_rot),
        ),
        left=_controller_from_msg(msg, "left"),
        right=_controller_from_msg(msg, "right"),
        battery={
            "hmd_pct": msg.get("hmdBattPct"),
            "left_pct": msg.get("leftBattPct"),
            "right_pct": msg.get("rightBattPct"),
            "hmd_charging": msg.get("hmdCharging"),
        },
        raw=msg,
    )


# ---------------------------------------------------------------------------
# Config.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetaQuestConfig:
    quest_ip: str
    tcp_port: int = 65432
    sync_port: int = 42000
    connect_retry_s: float = 1.0
    frame_stale_timeout_s: float = 0.25
    fps_window: int = 30
    offset_history: int = 15
    rtt_accept_ns: int = 8_000_000

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MetaQuestConfig":
        data = load_rig_section(Path(path), "meta_quest")
        conn = data.get("connection", {}) or {}
        health = data.get("health", {}) or {}
        return cls(
            quest_ip=str(conn.get("quest_ip", "")),
            tcp_port=int(conn.get("tcp_port", 65432)),
            sync_port=int(conn.get("sync_port", 42000)),
            connect_retry_s=float(conn.get("connect_retry_s", 1.0)),
            frame_stale_timeout_s=float(
                health.get("frame_stale_timeout_s", 0.25)
            ),
        )


# ---------------------------------------------------------------------------
# Receiver.
# ---------------------------------------------------------------------------


class MetaQuestReceiver:
    """Connect to the Quest app, parse frames, expose latest frame + metrics.

    Threading model: a supervisor thread (re)connects the TCP socket; while
    connected, a TCP recv loop drains frames and a UDP loop estimates the clock
    offset. ``latest()`` and ``metrics()`` are safe to call from any thread.
    """

    def __init__(
        self,
        config: MetaQuestConfig,
        *,
        on_frame: Callable[[QuestFrame], None] | None = None,
    ) -> None:
        self.config = config
        self._on_frame = on_frame

        self._lock = threading.Lock()
        self._running = False
        self._connected = False
        self._thread: threading.Thread | None = None

        self._latest: QuestFrame | None = None
        self._frames: deque[QuestFrame] = deque(maxlen=512)
        self._frame_times: deque[float] = deque(maxlen=config.fps_window)
        self._last_frame_mono: float | None = None
        self._offset_hist: deque[int] = deque(maxlen=config.offset_history)
        self._offset_ns: int = 0
        self._rtt_ns: int | None = None
        self._receive_sequence = 0

    # ---------- public API ----------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="meta_quest_rx", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._thread = None

    def __enter__(self) -> "MetaQuestReceiver":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def latest(self) -> QuestFrame | None:
        with self._lock:
            return self._latest

    def aligned_at(self, target_time_ns: int | None = None) -> AlignedQuestFrame | None:
        """Return the buffered Quest frame nearest one PC monotonic timestamp."""
        with self._lock:
            frames = tuple(self._frames)
            latest = self._latest
            offset_ns = self._offset_ns
            clock_synced = self._rtt_ns is not None
        if not frames:
            frames = () if latest is None else (latest,)
        if not frames:
            return None

        def aligned_time(frame: QuestFrame) -> int:
            if clock_synced and frame.device_time_ns > 0:
                return int(frame.device_time_ns + offset_ns)
            return int(frame.pc_monotonic_ns)

        if target_time_ns is None:
            frame = frames[-1]
        else:
            frame = min(
                frames,
                key=lambda candidate: abs(aligned_time(candidate) - target_time_ns),
            )
        return AlignedQuestFrame(
            frame=frame,
            aligned_time_ns=aligned_time(frame),
            clock_offset_ns=int(offset_ns),
            clock_synced=bool(clock_synced),
        )

    def metrics(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            connected = self._connected
            frame_times = tuple(self._frame_times)
            last_frame_mono = self._last_frame_mono
            offset_ns = self._offset_ns
            rtt_ns = self._rtt_ns

        if len(frame_times) >= 2 and frame_times[-1] > frame_times[0]:
            fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
        else:
            fps = 0.0
        last_frame_age_s = (
            float("nan") if last_frame_mono is None else now - last_frame_mono
        )
        streaming = (
            connected
            and last_frame_age_s == last_frame_age_s  # not NaN
            and last_frame_age_s <= self.config.frame_stale_timeout_s
        )
        return {
            "connected": connected,
            "streaming": streaming,
            "fps": fps,
            "last_frame_age_s": last_frame_age_s,
            "offset_ns": offset_ns,
            "offset_s": offset_ns / 1e9,
            "rtt_ns": rtt_ns,
            "rtt_ms": float("nan") if rtt_ns is None else rtt_ns / 1e6,
        }

    # ---------- internals ----------

    def _run(self) -> None:
        while self._running:
            sock = self._try_connect()
            if sock is None:
                time.sleep(self.config.connect_retry_s)
                continue

            with self._lock:
                self._connected = True
                self._frame_times.clear()
                self._frames.clear()
                self._last_frame_mono = None
                self._offset_hist.clear()
                self._offset_ns = 0
                self._rtt_ns = None

            udp_thread = threading.Thread(
                target=self._udp_sync_loop, name="meta_quest_sync", daemon=True
            )
            udp_thread.start()
            try:
                self._tcp_recv_loop(sock)
            finally:
                with self._lock:
                    self._connected = False
                try:
                    sock.close()
                except OSError:
                    pass
                udp_thread.join(timeout=1.0)

    def _try_connect(self) -> socket.socket | None:
        if not self.config.quest_ip:
            log.warning("quest_ip is not set; cannot connect.")
            return None
        try:
            log.info("Connecting TCP %s:%d ...", self.config.quest_ip, self.config.tcp_port)
            sock = socket.create_connection(
                (self.config.quest_ip, self.config.tcp_port), timeout=2.0
            )
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(0.5)
            log.info("Quest connected.")
            return sock
        except OSError as exc:
            log.debug("Connect failed: %s", exc)
            return None

    def _tcp_recv_loop(self, sock: socket.socket) -> None:
        buf = ""
        while self._running:
            try:
                raw = sock.recv(1024 * 1024)
            except socket.timeout:
                continue
            except OSError as exc:
                log.warning("TCP error: %s", exc)
                return
            if not raw:
                log.warning("Quest closed the connection.")
                return

            buf += raw.decode("utf-8", errors="ignore")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._handle_message(msg)

    def _handle_message(self, msg: dict[str, Any]) -> None:
        pc_mono_ns = time.monotonic_ns()
        frame = parse_frame(msg, pc_monotonic_ns=pc_mono_ns)
        with self._lock:
            self._receive_sequence += 1
            frame = replace(frame, receive_sequence=self._receive_sequence)
            self._latest = frame
            self._frames.append(frame)
            self._last_frame_mono = pc_mono_ns / 1e9
            self._frame_times.append(self._last_frame_mono)
        if self._on_frame is not None:
            try:
                self._on_frame(frame)
            except Exception:  # noqa: BLE001 - a bad callback must not kill rx.
                log.exception("on_frame callback raised")

    def _udp_sync_loop(self) -> None:
        quest_addr = (self.config.quest_ip, self.config.sync_port)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except OSError:
            return
        sock.settimeout(1.0)
        try:
            while self._running and self._connected:
                try:
                    t1 = time.monotonic_ns()
                    sock.sendto(_PING.pack(_PING_TYPE, t1), quest_addr)
                    data, _ = sock.recvfrom(32)
                    t4 = time.monotonic_ns()
                    if len(data) != _PONG.size:
                        continue
                    msg_type, t1_echo, t2_q = _PONG.unpack(data)
                    if msg_type != _PONG_TYPE:
                        continue
                    rtt = t4 - t1_echo
                    if rtt >= self.config.rtt_accept_ns or rtt < 0:
                        continue
                    # offset = pc - device, so device_time + offset ~= pc clock.
                    mid_pc = (t1_echo + t4) // 2
                    offset = mid_pc - t2_q
                    with self._lock:
                        self._rtt_ns = rtt
                        self._offset_hist.append(int(offset))
                        self._offset_ns = int(statistics.median(self._offset_hist))
                except socket.timeout:
                    pass
                except OSError:
                    time.sleep(0.5)
                time.sleep(1.0)
        finally:
            try:
                sock.close()
            except OSError:
                pass


class MetaQuestTrackingProvider:
    """Meta Quest provider normalized to the common controller schema."""

    device = "meta"

    def __init__(
        self,
        *,
        config: MetaQuestConfig,
        calibration: ControllerTcpCalibration,
        reset_workspace_on_x: bool = True,
    ) -> None:
        self.config = config
        self.calibration = calibration
        self.receiver = MetaQuestReceiver(config)
        self.workspace = WorkspaceCalibration.identity()
        self.workspace_set = False
        self.workspace_locked = False
        # When False, controller buttons cannot reset the workspace (for
        # example, ``handumi teleop sim`` owns its anchoring behavior).
        self.reset_workspace_on_x = reset_workspace_on_x
        self._prev_reset = False

    def start(self) -> None:
        self.receiver.start()
        log.info("Connecting to Quest at %s:%d ...", self.config.quest_ip, self.config.tcp_port)

    def stop(self) -> None:
        self.receiver.stop()

    def reset_workspace(self) -> None:
        """Re-center on the next HMD frame unless a table calibration is locked."""
        if self.workspace_locked:
            log.info("Workspace is locked to the calibrated table frame; reset ignored.")
            return
        self.workspace_set = False

    def set_workspace_from_device_pose(self, pose7: np.ndarray, *, locked: bool = True) -> None:
        """Set ``T_workspace_quest`` explicitly, normally from session calibration."""
        value = np.asarray(pose7, dtype=np.float64).reshape(7)
        self.workspace = WorkspaceCalibration(Pose(value[:3], value[3:]))
        self.workspace_set = True
        self.workspace_locked = bool(locked)
        log.info("Workspace set from calibration (%s).", "locked" if locked else "unlocked")

    def latest(self) -> ControllerPairSample:
        return self._sample(self.receiver.aligned_at())

    def sample_at(self, target_time_ns: int) -> ControllerPairSample:
        """Return the native Quest frame nearest a synchronized row target."""
        return self._sample(self.receiver.aligned_at(target_time_ns))

    def _sample(self, aligned: AlignedQuestFrame | None) -> ControllerPairSample:
        if aligned is None:
            return ControllerPairSample.empty(self.device)
        frame = aligned.frame

        # The receiver intentionally retains its last frame across reconnects.
        # Never advertise that cached pose as tracked once the stream is stale.
        metrics = self.receiver.metrics()
        streaming = bool(metrics["streaming"])
        connected = bool(metrics["connected"])
        reset_pressed = (
            streaming and self.reset_workspace_on_x and frame.left.buttons.primary
        )
        reset_edge = reset_pressed and not self._prev_reset
        self._prev_reset = reset_pressed
        if (
            not self.workspace_locked
            and streaming
            and frame.hmd.tracked
            and (reset_edge or not self.workspace_set)
        ):
            self.workspace = workspace_from_hmd(frame.hmd)
            self.workspace_set = True
            log.info("Workspace %s on HMD pose.", "reset" if reset_edge else "initialized")

        left_device = unity_pose_to_handumi(frame.left.position, frame.left.quaternion)
        right_device = unity_pose_to_handumi(frame.right.position, frame.right.quaternion)
        hmd_device = unity_pose_to_handumi(frame.hmd.position, frame.hmd.quaternion)
        left_controller = self.workspace.apply(left_device)
        right_controller = self.workspace.apply(right_device)
        hmd_pose = self.workspace.apply(hmd_device)
        left = pose_to_pose7(left_controller)
        right = pose_to_pose7(right_controller)
        left_tcp, right_tcp = apply_tcp_calibration_pose7(left, right, self.calibration)
        return ControllerPairSample(
            device=self.device,
            left_controller_pose=left,
            right_controller_pose=right,
            left_tcp_pose=left_tcp,
            right_tcp_pose=right_tcp,
            left_tracked=bool(streaming and frame.left.tracked and frame.left.valid),
            right_tracked=bool(streaming and frame.right.tracked and frame.right.valid),
            left_device_tracked=bool(frame.left.tracked),
            right_device_tracked=bool(frame.right.tracked),
            left_pose_valid=bool(frame.left.valid),
            right_pose_valid=bool(frame.right.valid),
            hmd_pose=pose_to_pose7(hmd_pose),
            left_device_controller_pose=pose_to_pose7(left_device),
            right_device_controller_pose=pose_to_pose7(right_device),
            device_hmd_pose=pose_to_pose7(hmd_device),
            hmd_tracked=bool(streaming and frame.hmd.tracked),
            workspace_from_device_pose=pose_to_pose7(
                self.workspace.workspace_from_quest
            ),
            device_time_ns=int(frame.device_time_ns),
            pc_monotonic_ns=int(frame.pc_monotonic_ns),
            aligned_time_ns=int(aligned.aligned_time_ns),
            clock_offset_ns=int(aligned.clock_offset_ns),
            clock_synced=bool(aligned.clock_synced),
            connected=connected,
            streaming=streaming,
            sequence=int(frame.receive_sequence),
        )


# ---------------------------------------------------------------------------
# Parsing helpers.
# ---------------------------------------------------------------------------


def _as_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _vec3(value: Any) -> np.ndarray:
    """Parse a 3-vector from an `{x,y,z}` dict (or an `[x,y,z]` list)."""
    out = np.zeros(3, dtype=np.float32)
    if isinstance(value, dict):
        out[0] = _as_float(value.get("x"))
        out[1] = _as_float(value.get("y"))
        out[2] = _as_float(value.get("z"))
    elif isinstance(value, (list, tuple)):
        for i in range(min(3, len(value))):
            out[i] = _as_float(value[i])
    return out


def _quat(value: Any) -> np.ndarray:
    """Parse a quaternion from an `{x,y,z,w}` dict (or an `[x,y,z,w]` list)."""
    out = np.array(_IDENTITY_QUAT, dtype=np.float32)
    if isinstance(value, dict):
        out[0] = _as_float(value.get("x"))
        out[1] = _as_float(value.get("y"))
        out[2] = _as_float(value.get("z"))
        out[3] = _as_float(value.get("w"), default=1.0)
    elif isinstance(value, (list, tuple)) and len(value) >= 4:
        for i in range(4):
            out[i] = _as_float(value[i])
    return out


def _stick(value: Any) -> tuple[float, float]:
    if isinstance(value, dict):
        return (_as_float(value.get("x")), _as_float(value.get("y")))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (_as_float(value[0]), _as_float(value[1]))
    return (0.0, 0.0)


# ---------------------------------------------------------------------------
# Demo entry point: connect and print frames + diagnostics.
# ---------------------------------------------------------------------------


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Connect to a Quest app (or mock) and print frames + metrics."
    )
    parser.add_argument("--quest-ip", default="127.0.0.1")
    parser.add_argument("--tcp-port", type=int, default=65432)
    parser.add_argument("--sync-port", type=int, default=42000)
    parser.add_argument("--config", type=Path, default=None,
                        help="Optional configs/rig.yaml override.")
    parser.add_argument("--print-raw", action="store_true",
                        help="Dump the first received raw JSON frame (verify wire format).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.config is not None:
        config = MetaQuestConfig.from_yaml(args.config)
    else:
        config = MetaQuestConfig(
            quest_ip=args.quest_ip, tcp_port=args.tcp_port, sync_port=args.sync_port
        )

    receiver = MetaQuestReceiver(config)
    receiver.start()
    log.info("Receiver started. Ctrl+C to stop.")
    raw_dumped = False
    try:
        while True:
            time.sleep(0.5)
            m = receiver.metrics()
            frame = receiver.latest()
            if frame is None:
                print(
                    f"\rconnected={m['connected']} streaming={m['streaming']} "
                    f"(waiting for frames)        ",
                    end="",
                    flush=True,
                )
                continue
            if args.print_raw and not raw_dumped:
                raw_dumped = True
                print("\n--- first raw frame ---")
                print(json.dumps(frame.raw, indent=2, sort_keys=True))
                print("--- end raw frame ---")
            lp = frame.left.position
            rp = frame.right.position
            print(
                "\r"
                f"seq={frame.seq:06d} fps={m['fps']:5.1f} "
                f"off={m['offset_s']:+.4f}s rtt={m['rtt_ms']:.2f}ms | "
                f"L trk={int(frame.left.tracked)} [{lp[0]:+.2f},{lp[1]:+.2f},{lp[2]:+.2f}] "
                f"R trk={int(frame.right.tracked)} [{rp[0]:+.2f},{rp[1]:+.2f},{rp[2]:+.2f}]   ",
                end="",
                flush=True,
            )
    except KeyboardInterrupt:
        print()
    finally:
        receiver.stop()


if __name__ == "__main__":
    _main()
