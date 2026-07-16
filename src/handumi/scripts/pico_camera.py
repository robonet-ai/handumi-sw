"""Stream one local camera to the PICO XRoboToolkit Remote Vision panel."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from handumi.tracking.pico_vision import PicoRemoteVisionBridge

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


def _size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x", maxsplit=1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("size dimensions must be positive")
    return width, height


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera",
        type=Path,
        default=Path("/dev/video2"),
        help="Context camera shown in the center of each eye.",
    )
    parser.add_argument(
        "--left-camera",
        type=Path,
        default=None,
        help="Optional left-wrist camera shown on the left.",
    )
    parser.add_argument(
        "--right-camera",
        type=Path,
        default=None,
        help="Optional right-wrist camera shown on the right.",
    )
    parser.add_argument("--input-format", default="mjpeg")
    parser.add_argument("--input-size", type=_size, default=(1280, 720))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--bitrate", type=int, default=None)
    parser.add_argument(
        "--eye-y-offset",
        type=int,
        default=48,
        help=(
            "Vertical image offset per eye in pixels; positive moves the view "
            "down (default: 48)."
        ),
    )
    parser.add_argument(
        "--skip-adb",
        action="store_true",
        help="Do not create adb reverse 13579 and adb forward 12345.",
    )
    return parser.parse_args(argv)


def bridge_from_args(args: argparse.Namespace) -> PicoRemoteVisionBridge:
    width, height = args.input_size
    return PicoRemoteVisionBridge(
        args.camera,
        left_camera=args.left_camera,
        right_camera=args.right_camera,
        input_format=args.input_format,
        input_width=width,
        input_height=height,
        input_fps=args.fps,
        bitrate=args.bitrate,
        eye_y_offset=args.eye_y_offset,
        setup_adb=not args.skip_adb,
    )


def main() -> None:
    args = parse_args()
    bridge = bridge_from_args(args)
    try:
        bridge.start()
        print(
            "Bridge listo. Abre XRoboToolkit > Remote Vision, selecciona "
            "ZEDMINI, usa IP 127.0.0.1 y pulsa Listen. Ctrl+C para terminar."
        )
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop()


if __name__ == "__main__":
    main()
