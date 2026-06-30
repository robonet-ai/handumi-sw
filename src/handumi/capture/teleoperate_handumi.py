"""Stream HandUMI camera and Feetech observations to Rerun without recording."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

import numpy as np

from handumi.cameras.usb import (
    build_camera_specs,
    connect_cameras,
    disconnect_cameras,
    read_camera_frames,
)
from handumi.feetech import FeetechGripperPair, GripperWidths, load_config
from handumi.feetech.bus import FeetechUnavailableError

log = logging.getLogger("handumi.teleoperate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream HandUMI cameras and Feetech gripper widths to Rerun."
    )
    parser.add_argument("--cam-ids", nargs="+", type=int, default=[0, 2])
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--cam-fps", type=int, default=30)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--feetech-config", type=Path, default=Path("configs/feetech.yaml"))
    parser.add_argument("--feetech-port", type=str, default=None)
    parser.add_argument("--skip-feetech", action="store_true")
    parser.add_argument("--display-ip", type=str, default=None)
    parser.add_argument("--display-port", type=int, default=None)
    parser.add_argument("--compress-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    _init_rerun(ip=args.display_ip, port=args.display_port)

    camera_specs, _ = build_camera_specs(
        args.cam_ids,
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

    grippers = None
    if args.skip_feetech:
        log.info("Feetech disabled: gripper widths will be zero-filled.")
    else:
        feetech_config = load_config(args.feetech_config)
        if args.feetech_port is not None:
            feetech_config = type(feetech_config)(
                port=args.feetech_port,
                baudrate=feetech_config.baudrate,
                protocol_version=feetech_config.protocol_version,
                left=feetech_config.left,
                right=feetech_config.right,
            )
        _assert_calibrated(feetech_config)
        grippers = FeetechGripperPair(feetech_config)
        try:
            grippers.open()
        except FeetechUnavailableError as exc:
            raise SystemExit(str(exc)) from exc

    stop = False

    def _on_signal(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    log.info("Streaming to Rerun. Press Ctrl+C to stop.")
    start = time.perf_counter()
    frame_index = 0
    control_interval = 1.0 / args.fps

    try:
        while not stop:
            loop_start = time.perf_counter()
            elapsed = loop_start - start
            if args.duration_s is not None and elapsed >= args.duration_s:
                break

            cam_frames = read_camera_frames(
                cameras,
                cam_names,
                width=args.cam_width,
                height=args.cam_height,
            )
            widths = zero_gripper_widths() if grippers is None else grippers.read_normalized_widths()
            _log_observation(
                cam_frames=cam_frames,
                widths=widths,
                frame_index=frame_index,
                elapsed_s=elapsed,
                compress_images=args.compress_images,
            )
            _print_status(widths=widths, frame_index=frame_index)

            frame_index += 1
            dt = time.perf_counter() - loop_start
            time.sleep(max(control_interval - dt, 0.0))
    finally:
        print()
        if grippers is not None:
            grippers.close()
        disconnect_cameras(cameras)
        _shutdown_rerun()


def _init_rerun(*, ip: str | None, port: int | None) -> None:
    try:
        from lerobot.utils.visualization_utils import init_rerun
    except ImportError as exc:
        raise SystemExit(
            "LeRobot visualization utilities are required. Run `uv sync` in this repo."
        ) from exc
    init_rerun(session_name="handumi_teleoperate", ip=ip, port=port)


def _shutdown_rerun() -> None:
    try:
        from lerobot.utils.visualization_utils import shutdown_rerun

        shutdown_rerun()
    except Exception:
        pass


def _log_observation(
    *,
    cam_frames: dict,
    widths: GripperWidths,
    frame_index: int,
    elapsed_s: float,
    compress_images: bool,
) -> None:
    from lerobot.utils.visualization_utils import log_rerun_data

    observation = {
        **cam_frames,
        "observation.feetech.left_ticks": float(widths.left_ticks),
        "observation.feetech.right_ticks": float(widths.right_ticks),
        "observation.feetech.left_width_mm": float(widths.left_mm),
        "observation.feetech.right_width_mm": float(widths.right_mm),
        "observation.feetech.left_normalized": float(widths.left_normalized),
        "observation.feetech.right_normalized": float(widths.right_normalized),
        "observation.loop.frame_index": float(frame_index),
        "observation.loop.elapsed_s": float(elapsed_s),
    }
    log_rerun_data(observation=observation, compress_images=compress_images)


def _print_status(*, widths: GripperWidths, frame_index: int) -> None:
    sys.stdout.write(
        "\r"
        f"frame={frame_index:06d} "
        f"left={widths.left_mm:7.2f}mm ({widths.left_normalized:0.3f}) "
        f"right={widths.right_mm:7.2f}mm ({widths.right_normalized:0.3f})"
    )
    sys.stdout.flush()


def _assert_calibrated(config) -> None:
    missing = []
    if not config.left.is_complete:
        missing.append("left")
    if not config.right.is_complete:
        missing.append("right")
    if missing:
        raise SystemExit(
            "Feetech calibration is incomplete for "
            + ", ".join(missing)
            + ". Calibrate configs/feetech.yaml before live monitoring."
        )


def zero_gripper_widths() -> GripperWidths:
    return GripperWidths(
        left=0.0,
        right=0.0,
        left_mm=0.0,
        right_mm=0.0,
        left_normalized=0.0,
        right_normalized=0.0,
        left_ticks=0,
        right_ticks=0,
    )


if __name__ == "__main__":
    main()
