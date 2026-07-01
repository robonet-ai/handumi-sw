"""Mock Quest app for developing the Phase 2A receiver without hardware.

Emulates the native Quest app end of the contract so the whole pipe — TCP/JSON
pose stream + UDP NTP-style time-sync — can be exercised on the workstation:

  * TCP server: accepts a connection and streams newline-delimited JSON pose
    samples (raw Unity coordinates) at a fixed rate. Controllers gently
    oscillate so the receiver shows changing numbers.
  * UDP server: echoes time-sync pings with a *device clock* that is offset from
    the PC clock by ``--skew-s`` seconds, so the receiver's offset estimate is
    non-zero and verifiable.

Run this in one terminal, then the receiver in another:

    PYTHONPATH=src python -m handumi.tracking.mock_quest_sender
    PYTHONPATH=src python -m handumi.tracking.meta_quest

See docs/phase-2-motion-tracking.md → TCP/JSON Payload for the field layout.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import socket
import struct
import threading
import time

log = logging.getLogger("handumi.tracking.mock_quest_sender")

_PING = struct.Struct("<BQ")  # (msg_type=1, t1_pc_ns)
_PONG = struct.Struct("<BQQ")  # (msg_type=2, t1_echo, t2_quest_ns)
_PING_TYPE = 1
_PONG_TYPE = 2


def _device_time_ns(skew_ns: int) -> int:
    """Fake Quest monotonic clock = PC monotonic clock + a fixed skew."""
    return time.monotonic_ns() + skew_ns


def _xyz(x: float, y: float, z: float) -> dict:
    return {"x": x, "y": y, "z": z}


def _xyzw(x: float, y: float, z: float, w: float) -> dict:
    return {"x": x, "y": y, "z": z, "w": w}


def _make_frame(seq: int, t0: float, skew_ns: int) -> dict:
    """Build one pose sample in the YubiQuestApp flat wire format (Unity coords)."""
    t = time.monotonic() - t0
    sway = 0.05 * math.sin(t)
    bob = 0.05 * math.sin(2.0 * t)
    reach = 0.05 * math.cos(t)
    return {
        # Top-level timing (yubi legacy TCP/JSON has no seq).
        "ovrTimeNs": _device_time_ns(skew_ns),
        "deltaTime": 1.0 / 72.0,
        # HMD pose.
        "hmdPosition": _xyz(0.0, 1.10, 0.05),
        "hmdRotation": _xyzw(0.0, 0.0, 0.0, 1.0),
        # Left controller.
        "leftControllerPosition": _xyz(-0.20 + sway, 0.95 + bob, 0.30 + reach),
        "leftControllerRotation": _xyzw(0.0, 0.0, 0.0, 1.0),
        "leftTracked": True,
        "leftValid": True,
        "leftJoystick": _xyz(0.0, 0.0, 0.0),
        "leftThumbstickClick": False,
        "leftTriggerPressed": False,
        "leftGripPressed": False,
        "buttonXPressed": False,
        "buttonYPressed": False,
        # Right controller.
        "rightControllerPosition": _xyz(0.20 - sway, 0.95 + bob, 0.30 - reach),
        "rightControllerRotation": _xyzw(0.0, 0.0, 0.0, 1.0),
        "rightTracked": True,
        "rightValid": True,
        "rightJoystick": _xyz(0.0, 0.0, 0.0),
        "rightThumbstickClick": False,
        "rightTriggerPressed": False,
        "rightGripPressed": False,
        "buttonAPressed": False,
        "buttonBPressed": False,
        # Battery.
        "hmdBattPct": 87,
        "leftBattPct": 90,
        "rightBattPct": 92,
        "hmdCharging": False,
    }


def _udp_sync_server(host: str, sync_port: int, skew_ns: int, stop: threading.Event) -> None:
    """Echo every ping with the device clock (the Quest end of time-sync)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, sync_port))
    sock.settimeout(0.5)
    log.info("UDP time-sync server on %s:%d", host, sync_port)
    try:
        while not stop.is_set():
            try:
                data, addr = sock.recvfrom(32)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(data) != _PING.size:
                continue
            msg_type, t1 = _PING.unpack(data)
            if msg_type != _PING_TYPE:
                continue
            sock.sendto(_PONG.pack(_PONG_TYPE, t1, _device_time_ns(skew_ns)), addr)
    finally:
        sock.close()


def _serve_client(conn: socket.socket, addr, fps: float, skew_ns: int,
                  stop: threading.Event) -> None:
    log.info("Client connected: %s", addr)
    seq = 0
    t0 = time.monotonic()
    period = 1.0 / fps if fps > 0 else 0.0
    conn.settimeout(1.0)
    try:
        while not stop.is_set():
            loop_start = time.monotonic()
            frame = _make_frame(seq, t0, skew_ns)
            line = (json.dumps(frame) + "\n").encode("utf-8")
            try:
                conn.sendall(line)
            except OSError:
                break
            seq += 1
            dt = time.monotonic() - loop_start
            time.sleep(max(period - dt, 0.0))
    finally:
        conn.close()
        log.info("Client disconnected: %s", addr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock Quest app (TCP/JSON + UDP sync).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--tcp-port", type=int, default=65432)
    parser.add_argument("--sync-port", type=int, default=42000)
    parser.add_argument("--fps", type=float, default=72.0)
    parser.add_argument("--skew-s", type=float, default=5.0,
                        help="Fake device-clock skew vs PC clock (verifies sync).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    skew_ns = int(args.skew_s * 1e9)
    stop = threading.Event()

    udp_thread = threading.Thread(
        target=_udp_sync_server,
        args=(args.host, args.sync_port, skew_ns, stop),
        daemon=True,
    )
    udp_thread.start()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.tcp_port))
    server.listen(1)
    server.settimeout(0.5)
    log.info("TCP pose server on %s:%d (fps=%.0f, skew=%.1fs). Ctrl+C to stop.",
             args.host, args.tcp_port, args.fps, args.skew_s)

    try:
        while True:
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue
            _serve_client(conn, addr, args.fps, skew_ns, stop)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.close()
        udp_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
