"""Meta Quest native-app tracking receiver (Phase 2A, Step 1 — the pose pipe).

yubi-sw model: the native Quest app is a TCP server that streams one
newline-delimited JSON pose sample per frame; the workstation (this module)
dials in, parses each sample, stamps a PC receive clock, and keeps the latest
frame in a buffer. A companion UDP NTP-style loop estimates the Quest<->PC clock
offset so poses can be aligned with camera/Feetech frames in post-processing.

This step does NOT transform coordinates — poses are kept as raw Unity values.
``handumi.tracking.transforms`` (Step 2) converts them.

Reference:
  ../yubi-sw/airoa_quest/airoa_quest_bridge/transport/tcp_json.py
  ../yubi-sw/airoa_quest/airoa_quest_msgs/msg/QuestController.msg

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

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

    The YubiQuestApp legacy TCP/JSON reports button *presses* only (no analog
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


# Flat YubiQuestApp wire keys, per side (from yubi-sw quest_bridge_node.py).
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
    """Build a :class:`QuestFrame` from one decoded YubiQuestApp JSON sample.

    Wire format is the yubi-sw flat layout (Unity coordinates, dict-shaped
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
    fps_window: int = 30
    offset_history: int = 15
    rtt_accept_ns: int = 8_000_000

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MetaQuestConfig":
        with Path(path).open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        conn = data.get("connection", {}) or {}
        return cls(
            quest_ip=str(conn.get("quest_ip", "")),
            tcp_port=int(conn.get("tcp_port", 65432)),
            sync_port=int(conn.get("sync_port", 42000)),
            connect_retry_s=float(conn.get("connect_retry_s", 1.0)),
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
        self._frame_times: deque[float] = deque(maxlen=config.fps_window)
        self._last_frame_mono: float | None = None
        self._offset_hist: deque[int] = deque(maxlen=config.offset_history)
        self._offset_ns: int = 0
        self._rtt_ns: int | None = None

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
            and last_frame_age_s <= 1.0
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
            self._latest = frame
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


# ---------------------------------------------------------------------------
# Parsing helpers.
# ---------------------------------------------------------------------------


def _as_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _vec3(value: Any) -> np.ndarray:
    """Parse a 3-vector from a yubi `{x,y,z}` dict (or an `[x,y,z]` list)."""
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
    """Parse a quaternion from a yubi `{x,y,z,w}` dict (or an `[x,y,z,w]` list)."""
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
                        help="Optional configs/tracking_meta_quest.yaml override.")
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
