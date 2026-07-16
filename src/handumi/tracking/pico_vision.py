"""Stream a local camera to XRoboToolkit Remote Vision on a PICO headset."""

from __future__ import annotations

import logging
import shutil
import socket
import struct
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

log = logging.getLogger("handumi.pico_vision")

PICO_VISION_COMMAND_PORT = 13579
PICO_VISION_STREAM_PORT = 12345
MAX_COMMAND_PACKET_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class CameraRequest:
    width: int = 2560
    height: int = 720
    fps: int = 30
    bitrate: int = 4_000_000
    enable_mv_hevc: int = 0
    render_mode: int = 2
    port: int = PICO_VISION_STREAM_PORT
    camera: str = "ZED"
    ip: str = "127.0.0.1"


class StreamState:
    """Thread-safe latest Remote Vision request and lifecycle signals."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request: CameraRequest | None = None
        self._generation = 0
        self.enabled = threading.Event()
        self.shutdown = threading.Event()

    def open(self, request: CameraRequest) -> None:
        with self._lock:
            self._request = request
            self._generation += 1
            self.enabled.set()

    def close(self) -> None:
        with self._lock:
            self._generation += 1
            self.enabled.clear()

    def snapshot(self) -> tuple[CameraRequest | None, int]:
        with self._lock:
            return self._request, self._generation

    def is_current(self, generation: int) -> bool:
        with self._lock:
            return self.enabled.is_set() and generation == self._generation


def _compact_string(data: bytes, offset: int) -> tuple[str, int]:
    if offset >= len(data):
        raise ValueError("missing compact string length")
    length = data[offset]
    offset += 1
    end = offset + length
    if end > len(data):
        raise ValueError("compact string exceeds payload")
    return data[offset:end].decode("utf-8", errors="replace"), end


def parse_camera_request(data: bytes) -> CameraRequest:
    """Decode the CameraRequestSerializer payload sent by XRoboToolkit."""
    if len(data) < 31 or data[:2] != b"\xca\xfe" or data[2] != 1:
        raise ValueError("invalid camera request payload")
    values = struct.unpack_from("<7i", data, 3)
    camera, offset = _compact_string(data, 31)
    ip, _ = _compact_string(data, offset)
    return CameraRequest(
        width=values[0],
        height=values[1],
        fps=values[2],
        bitrate=values[3],
        enable_mv_hevc=values[4],
        render_mode=values[5],
        port=values[6],
        camera=camera,
        ip=ip,
    )


def parse_network_protocol(data: bytes) -> tuple[str, bytes]:
    """Decode XRoboToolkit's little-endian command envelope."""
    if len(data) < 8:
        raise ValueError("protocol message too small")
    command_len = struct.unpack_from("<i", data, 0)[0]
    if command_len < 0:
        raise ValueError("negative command length")
    command_end = 4 + command_len
    if command_end + 4 > len(data):
        raise ValueError("incomplete command")
    data_len = struct.unpack_from("<i", data, command_end)[0]
    payload_start = command_end + 4
    payload_end = payload_start + data_len
    if data_len < 0 or payload_end > len(data):
        raise ValueError("invalid protocol payload length")
    command = data[4:command_end].rstrip(b"\0").decode("utf-8", errors="replace")
    return command, data[payload_start:payload_end]


def _unwrap_command_packet(packet: bytes) -> bytes:
    if len(packet) >= 4:
        body_len = struct.unpack(">I", packet[:4])[0]
        if body_len and 4 + body_len <= len(packet):
            return packet[4 : 4 + body_len]
    return packet


def iter_annexb_units(stream: BinaryIO, stop: threading.Event) -> Iterator[bytes]:
    """Yield XRoboToolkit decoder units delimited by 4-byte Annex-B codes.

    libx264 may place 3-byte start codes inside a frame. They must remain in
    the same length-prefixed packet: splitting each NAL makes the PICO Android
    decoder reject the buffers as incomplete. This intentionally matches the
    working deployment bridge.
    """
    start_code = b"\x00\x00\x00\x01"
    buffer = b""
    while not stop.is_set():
        chunk = stream.read(4096)
        if not chunk:
            if buffer:
                yield buffer
            return
        buffer += chunk
        while True:
            first = buffer.find(start_code)
            if first < 0:
                buffer = buffer[-3:]
                break
            second = buffer.find(start_code, first + len(start_code))
            if second < 0:
                if first > 0:
                    buffer = buffer[first:]
                break
            yield buffer[first:second]
            buffer = buffer[second:]


def build_camera_ffmpeg_command(
    *,
    camera: Path,
    left_camera: Path | None = None,
    right_camera: Path | None = None,
    input_format: str,
    input_width: int,
    input_height: int,
    input_fps: int,
    output_width: int,
    output_height: int,
    output_fps: int,
    bitrate: int,
    eye_y_offset: int = 0,
) -> list[str]:
    """Build the direct V4L2-to-stereo-H264 pipeline used by Remote Vision."""
    eye_width = max(2, output_width // 2)
    output_width = eye_width * 2
    eye_y_offset = max(-output_height + 1, min(output_height - 1, eye_y_offset))
    padded_height = output_height + abs(eye_y_offset)
    pad_y = max(eye_y_offset, 0)
    crop_y = max(-eye_y_offset, 0)
    shift_filter = (
        f"pad={eye_width}:{padded_height}:0:{pad_y}:black,"
        f"crop={eye_width}:{output_height}:0:{crop_y}"
        if eye_y_offset
        else "null"
    )
    cameras = [camera]
    if left_camera is not None:
        cameras.append(left_camera)
    if right_camera is not None:
        cameras.append(right_camera)

    inputs: list[str] = []
    for source in cameras:
        inputs.extend(
            [
                "-f",
                "v4l2",
                "-input_format",
                input_format,
                "-video_size",
                f"{input_width}x{input_height}",
                "-framerate",
                str(input_fps),
                "-i",
                str(source),
            ]
        )

    if len(cameras) == 1:
        filter_graph = (
            f"[0:v]scale={eye_width}:{output_height}:"
            "force_original_aspect_ratio=decrease,"
            f"pad={eye_width}:{output_height}:(ow-iw)/2:(oh-ih)/2:black"
            f"[eye_centered];[eye_centered]{shift_filter}[eye];"
            "[eye]split=2[left_eye][right_eye];"
            "[left_eye][right_eye]hstack=inputs=2,"
            f"scale={output_width}:{output_height},fps={output_fps}[stereo]"
        )
    else:
        tile = max(2, eye_width // 3)
        center_x = (eye_width - tile) // 2
        wrist_y = (output_height - tile) // 2
        filters = [
            f"color=c=black:s={eye_width}x{output_height}:r={output_fps}[base]",
            f"[0:v]scale={tile}:{output_height}[context]",
        ]
        overlays = [f"[base][context]overlay={center_x}:0[with_context]"]
        previous = "with_context"
        next_input = 1
        if left_camera is not None:
            filters.append(
                f"[{next_input}:v]scale={tile}:{tile}:"
                "force_original_aspect_ratio=decrease,"
                f"pad={tile}:{tile}:(ow-iw)/2:(oh-ih)/2:black[left_wrist]"
            )
            overlays.append(f"[{previous}][left_wrist]overlay=0:{wrist_y}[with_left]")
            previous = "with_left"
            next_input += 1
        if right_camera is not None:
            filters.append(
                f"[{next_input}:v]scale={tile}:{tile}:"
                "force_original_aspect_ratio=decrease,"
                f"pad={tile}:{tile}:(ow-iw)/2:(oh-ih)/2:black[right_wrist]"
            )
            overlays.append(
                f"[{previous}][right_wrist]overlay={eye_width - tile}:"
                f"{wrist_y}[with_right]"
            )
            previous = "with_right"
        filter_graph = ";".join(
            [
                *filters,
                *overlays,
                f"[{previous}]{shift_filter}[eye]",
                "[eye]split=2[left_eye][right_eye]",
                "[left_eye][right_eye]hstack=inputs=2,"
                f"scale={output_width}:{output_height},fps={output_fps}[stereo]",
            ]
        )

    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        *inputs,
        "-filter_complex",
        filter_graph,
        "-map",
        "[stereo]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-profile:v",
        "baseline",
        "-pix_fmt",
        "yuv420p",
        "-b:v",
        str(bitrate),
        "-x264-params",
        "keyint=15:min-keyint=15:scenecut=0:repeat-headers=1:aud=1",
        "-f",
        "h264",
        "-",
    ]


def setup_pico_vision_adb(*, runner=subprocess.run) -> None:
    """Configure the USB command and video directions used by Remote Vision."""
    commands = (
        ("reverse", PICO_VISION_COMMAND_PORT),
        ("forward", PICO_VISION_STREAM_PORT),
    )
    for direction, port in commands:
        result = runner(
            ["adb", direction, f"tcp:{port}", f"tcp:{port}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            hint = (
                " Close XRoboToolkit on the PICO, rerun this command, and then "
                "open the app again."
                if direction == "reverse" and "Address already in use" in detail
                else ""
            )
            raise RuntimeError(
                f"adb {direction} tcp:{port} failed: {detail or 'unknown error'}.{hint}"
            )
        log.info("ADB %s tcp:%d -> tcp:%d ready.", direction, port, port)


def probe_camera_input(
    camera: Path,
    *,
    input_format: str,
    width: int,
    height: int,
    fps: int,
) -> None:
    """Read one frame so busy or incompatible cameras fail before CAN opens."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "v4l2",
            "-input_format",
            input_format,
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            str(camera),
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            f"Cannot capture {camera} as {input_format} {width}x{height}@{fps}: "
            f"{detail or 'unknown ffmpeg error'}. Close OBS or any process using it."
        )
    log.info("Camera preflight OK: %s", camera)


class PicoRemoteVisionBridge:
    """XRoboToolkit command server plus direct local-camera H.264 sender."""

    def __init__(
        self,
        camera: Path,
        *,
        left_camera: Path | None = None,
        right_camera: Path | None = None,
        input_format: str = "mjpeg",
        input_width: int = 1280,
        input_height: int = 720,
        input_fps: int = 30,
        bitrate: int | None = None,
        eye_y_offset: int = 48,
        command_host: str = "0.0.0.0",
        command_port: int = PICO_VISION_COMMAND_PORT,
        stream_host: str = "127.0.0.1",
        stream_port: int = PICO_VISION_STREAM_PORT,
        setup_adb: bool = True,
    ) -> None:
        self.camera = Path(camera)
        self.left_camera = Path(left_camera) if left_camera is not None else None
        self.right_camera = Path(right_camera) if right_camera is not None else None
        self.input_format = input_format
        self.input_width = input_width
        self.input_height = input_height
        self.input_fps = input_fps
        self.bitrate = bitrate
        self.eye_y_offset = int(eye_y_offset)
        self.command_host = command_host
        self.command_port = command_port
        self.stream_host = stream_host
        self.stream_port = stream_port
        self.setup_adb = setup_adb
        self.state = StreamState()
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._server_thread: threading.Thread | None = None
        self._stream_thread: threading.Thread | None = None

    def start(self) -> None:
        missing = [
            source
            for source in (self.camera, self.left_camera, self.right_camera)
            if source is not None and not source.exists()
        ]
        if missing:
            raise RuntimeError(
                "PICO camera device(s) do not exist: "
                + ", ".join(str(source) for source in missing)
            )
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required for PICO Remote Vision streaming.")
        for source in (self.camera, self.left_camera, self.right_camera):
            if source is not None:
                probe_camera_input(
                    source,
                    input_format=self.input_format,
                    width=self.input_width,
                    height=self.input_height,
                    fps=self.input_fps,
                )
        if self.setup_adb:
            setup_pico_vision_adb()
        self._server_thread = threading.Thread(
            target=self._command_server, name="pico-vision-command", daemon=True
        )
        self._stream_thread = threading.Thread(
            target=self._stream_worker, name="pico-vision-stream", daemon=True
        )
        self._server_thread.start()
        if not self._ready.wait(timeout=3.0):
            raise RuntimeError("PICO Remote Vision command server did not start.")
        if self._startup_error is not None:
            raise RuntimeError(
                "PICO Remote Vision command server failed."
            ) from self._startup_error
        self._stream_thread.start()
        log.info(
            "PICO Remote Vision ready for %s. In XRoboToolkit select ZEDMINI, "
            "source IP 127.0.0.1, then Listen.",
            self.camera,
        )

    def stop(self) -> None:
        self.state.shutdown.set()
        self.state.close()
        for thread in (self._server_thread, self._stream_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=3.0)

    def _command_server(self) -> None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind((self.command_host, self.command_port))
                server.listen(1)
                server.settimeout(0.5)
                log.info(
                    "PICO Remote Vision command server listening on %s:%d.",
                    self.command_host,
                    self.command_port,
                )
                self._ready.set()
                while not self.state.shutdown.is_set():
                    try:
                        connection, address = server.accept()
                    except socket.timeout:
                        continue
                    log.info("PICO Remote Vision connected from %s.", address)
                    self._serve_connection(connection)
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            if not self.state.shutdown.is_set():
                log.error("PICO Remote Vision command server failed: %s", exc)

    def _serve_connection(self, connection: socket.socket) -> None:
        with connection:
            connection.settimeout(0.5)
            buffer = bytearray()
            while not self.state.shutdown.is_set():
                try:
                    chunk = connection.recv(65536)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                buffer.extend(chunk)
                while len(buffer) >= 4:
                    body_len = struct.unpack(">I", buffer[:4])[0]
                    if not 0 < body_len <= MAX_COMMAND_PACKET_BYTES:
                        log.warning("Invalid Remote Vision packet length: %d", body_len)
                        buffer.clear()
                        break
                    if len(buffer) < body_len + 4:
                        break
                    packet = bytes(buffer[: body_len + 4])
                    del buffer[: body_len + 4]
                    self._handle_command(packet)
        self.state.close()
        log.info("PICO Remote Vision command connection closed.")

    def _handle_command(self, packet: bytes) -> None:
        try:
            command, payload = parse_network_protocol(_unwrap_command_packet(packet))
            if command == "OPEN_CAMERA":
                request = parse_camera_request(payload)
                log.info(
                    "Remote Vision OPEN_CAMERA %s %dx%d@%d target %s:%d.",
                    request.camera,
                    request.width,
                    request.height,
                    request.fps,
                    request.ip,
                    request.port,
                )
                self.state.open(request)
            elif command == "CLOSE_CAMERA":
                log.info("Remote Vision CLOSE_CAMERA.")
                self.state.close()
            else:
                log.debug("Ignoring Remote Vision command %r.", command)
        except ValueError as exc:
            log.warning("Invalid Remote Vision command: %s", exc)

    def _stream_worker(self) -> None:
        while not self.state.shutdown.is_set():
            if not self.state.enabled.wait(timeout=0.5):
                continue
            request, generation = self.state.snapshot()
            if request is None:
                continue
            self._stream_request(request, generation)

    def _stream_request(self, request: CameraRequest, generation: int) -> None:
        width = max(4, request.width or 2560)
        height = max(2, request.height or 720)
        fps = max(1, min(request.fps or self.input_fps, self.input_fps))
        bitrate = self.bitrate or request.bitrate or 4_000_000
        try:
            stream_socket = socket.create_connection(
                (self.stream_host, self.stream_port), timeout=10
            )
        except OSError as exc:
            if self.state.is_current(generation):
                log.warning(
                    "PICO video decoder is not ready yet: %s. Retrying in 1 s.",
                    exc,
                )
                self.state.shutdown.wait(1.0)
            return

        command = build_camera_ffmpeg_command(
            camera=self.camera,
            left_camera=self.left_camera,
            right_camera=self.right_camera,
            input_format=self.input_format,
            input_width=self.input_width,
            input_height=self.input_height,
            input_fps=self.input_fps,
            output_width=width,
            output_height=height,
            output_fps=fps,
            bitrate=bitrate,
            eye_y_offset=self.eye_y_offset,
        )
        log.info(
            "Streaming %s to PICO as stereo %dx%d@%d H.264.",
            self.camera,
            width,
            height,
            fps,
        )
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        stopped = threading.Event()
        try:
            assert process.stdout is not None
            for unit in iter_annexb_units(process.stdout, stopped):
                if not self.state.is_current(generation):
                    break
                stream_socket.sendall(struct.pack(">I", len(unit)) + unit)
            return_code = process.poll()
            if return_code not in (None, 0) and self.state.is_current(generation):
                log.error(
                    "Camera encoder exited with code %d. If the device is busy, "
                    "close OBS or any other camera viewer.",
                    return_code,
                )
                self.state.close()
        except (BrokenPipeError, ConnectionError, OSError) as exc:
            if self.state.is_current(generation):
                # XRoboToolkit briefly recreates its MediaCodec and TCP server
                # after OPEN_CAMERA. The first forwarded connection can be
                # closed during that hand-off; retain the request so the worker
                # reconnects without requiring another press of Listen.
                log.warning(
                    "PICO video decoder restarted the stream: %s. "
                    "Retrying in 1 s.",
                    exc,
                )
                self.state.shutdown.wait(1.0)
        finally:
            stopped.set()
            stream_socket.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
            log.info("PICO camera stream stopped.")


__all__ = [
    "CameraRequest",
    "PicoRemoteVisionBridge",
    "build_camera_ffmpeg_command",
    "iter_annexb_units",
    "parse_camera_request",
    "parse_network_protocol",
    "probe_camera_input",
    "setup_pico_vision_adb",
]
