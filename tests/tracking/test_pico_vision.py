import io
import struct
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from handumi.tracking.pico_vision import (
    CameraRequest,
    PicoRemoteVisionBridge,
    build_camera_ffmpeg_command,
    iter_annexb_units,
    parse_camera_request,
    parse_network_protocol,
    probe_camera_input,
    setup_pico_vision_adb,
)


def _compact(value: str) -> bytes:
    encoded = value.encode()
    return bytes([len(encoded)]) + encoded


def test_parses_xrobotoolkit_open_camera_payload():
    payload = (
        b"\xca\xfe\x01"
        + struct.pack("<7i", 2560, 720, 30, 4_000_000, 0, 2, 12345)
        + _compact("ZED")
        + _compact("127.0.0.1")
    )

    request = parse_camera_request(payload)

    assert request.camera == "ZED"
    assert (request.width, request.height, request.fps) == (2560, 720, 30)
    assert request.ip == "127.0.0.1"
    assert request.port == 12345


def test_parses_xrobotoolkit_network_envelope():
    command = b"OPEN_CAMERA"
    payload = b"camera-request"
    message = (
        struct.pack("<i", len(command))
        + command
        + struct.pack("<i", len(payload))
        + payload
    )

    assert parse_network_protocol(message) == ("OPEN_CAMERA", payload)


def test_annexb_reader_keeps_three_byte_nals_inside_decoder_unit():
    stream = io.BytesIO(
        b"\x00\x00\x00\x01\x67abc\x00\x00\x01\x68de\x00\x00\x00\x01\x65frame"
    )

    units = list(iter_annexb_units(stream, threading.Event()))

    assert units == [
        b"\x00\x00\x00\x01\x67abc\x00\x00\x01\x68de",
        b"\x00\x00\x00\x01\x65frame",
    ]


def test_single_camera_command_duplicates_one_view_for_both_eyes():
    command = build_camera_ffmpeg_command(
        camera=Path("/dev/video2"),
        input_format="mjpeg",
        input_width=1280,
        input_height=720,
        input_fps=30,
        output_width=2560,
        output_height=720,
        output_fps=30,
        bitrate=4_000_000,
    )

    graph = command[command.index("-filter_complex") + 1]
    assert command.count("-i") == 1
    assert "split=2[left_eye][right_eye]" in graph
    assert "hstack=inputs=2" in graph


def test_positive_eye_offset_moves_view_down_before_stereo_duplication():
    command = build_camera_ffmpeg_command(
        camera=Path("/dev/video2"),
        input_format="mjpeg",
        input_width=1280,
        input_height=720,
        input_fps=30,
        output_width=2560,
        output_height=720,
        output_fps=30,
        bitrate=4_000_000,
        eye_y_offset=48,
    )

    graph = command[command.index("-filter_complex") + 1]
    assert "pad=1280:768:0:48:black,crop=1280:720:0:0[eye]" in graph
    assert graph.index("[eye]split=2") > graph.index("crop=1280:720")


def test_three_camera_command_builds_context_hands_grid():
    command = build_camera_ffmpeg_command(
        camera=Path("/dev/video2"),
        left_camera=Path("/dev/video4"),
        right_camera=Path("/dev/video6"),
        input_format="mjpeg",
        input_width=1280,
        input_height=720,
        input_fps=30,
        output_width=2560,
        output_height=720,
        output_fps=30,
        bitrate=4_000_000,
    )

    graph = command[command.index("-filter_complex") + 1]
    assert command.count("-i") == 3
    assert "[context]" in graph
    assert "[left_wrist]" in graph
    assert "[right_wrist]" in graph
    assert "overlay=0:" in graph


def test_adb_setup_uses_reverse_for_commands_and_forward_for_video():
    runner = mock.Mock(
        side_effect=[
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
        ]
    )

    setup_pico_vision_adb(runner=runner)

    assert runner.call_args_list[0].args[0] == [
        "adb",
        "reverse",
        "tcp:13579",
        "tcp:13579",
    ]
    assert runner.call_args_list[1].args[0] == [
        "adb",
        "forward",
        "tcp:12345",
        "tcp:12345",
    ]


def test_camera_preflight_reports_busy_device_before_hardware_setup():
    completed = SimpleNamespace(
        returncode=1,
        stdout="",
        stderr="Device or resource busy",
    )
    with (
        mock.patch(
            "handumi.tracking.pico_vision.subprocess.run", return_value=completed
        ),
        mock.patch(
            "handumi.tracking.pico_vision.shutil.which", return_value="/usr/bin/ffmpeg"
        ),
        mock.patch("handumi.tracking.pico_vision.subprocess.Popen") as popen,
    ):
        try:
            probe_camera_input(
                Path("/dev/video2"),
                input_format="mjpeg",
                width=1280,
                height=720,
                fps=30,
            )
        except RuntimeError as exc:
            assert "resource busy" in str(exc)
        else:
            raise AssertionError("busy camera was accepted")

    popen.assert_not_called()


def test_decoder_connection_race_keeps_request_enabled_for_retry():
    bridge = PicoRemoteVisionBridge(Path("/dev/video2"), setup_adb=False)
    request = CameraRequest()
    bridge.state.open(request)
    _, generation = bridge.state.snapshot()

    with (
        mock.patch(
            "handumi.tracking.pico_vision.socket.create_connection",
            side_effect=ConnectionRefusedError("decoder restarting"),
        ),
        mock.patch.object(bridge.state.shutdown, "wait", return_value=False),
    ):
        bridge._stream_request(request, generation)

    assert bridge.state.enabled.is_set()
    assert bridge.state.is_current(generation)
