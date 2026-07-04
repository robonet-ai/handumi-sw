from __future__ import annotations

import argparse
import glob
import shutil
import subprocess
import time
from datetime import datetime

from handumi.feetech.bus import FeetechBus
from handumi.feetech.calibration import PORTS_PATH, default_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show Feetech serial ports and USB camera devices while plugging/unplugging hardware."
    )
    parser.add_argument("--interval-s", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=20)
    args = parser.parse_args()

    try:
        while True:
            _clear()
            print(f"=== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
            _print_serial_ports(range(args.start_id, args.end_id + 1))
            _print_camera_ports()
            print(f"\nEdit servo_id/port in: {PORTS_PATH} (Feetech), configs/cameras.yaml (cameras)")
            if args.once:
                break
            time.sleep(args.interval_s)
    except KeyboardInterrupt:
        print("\nStopped.")


def _clear() -> None:
    print("\033c", end="")


def _print_serial_ports(scan_ids: range) -> None:
    print("\nFeetech serial ports")
    ports = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    if not ports:
        print("  none")
        return
    for port in ports:
        print(f"  {port}: ids={_scan_feetech_ids(port, scan_ids)}")


def _scan_feetech_ids(port: str, scan_ids: range) -> list[int] | str:
    config = default_config()
    try:
        with FeetechBus(
            port=port,
            baudrate=config.baudrate,
            protocol_version=config.protocol_version,
        ) as bus:
            return bus.scan(scan_ids)
    except Exception as exc:
        return f"unavailable ({exc})"


def _print_camera_ports() -> None:
    print("\nUSB cameras")
    if shutil.which("v4l2-ctl"):
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output = result.stdout.strip()
        print(_format_v4l2_usb_cameras(output) if output else "  none")
    else:
        print("  v4l2-ctl not found; showing /dev/video* only")
        videos = sorted(glob.glob("/dev/video*"))
        print("\n".join(f"  {video}" for video in videos) if videos else "  none")


def _format_v4l2_usb_cameras(output: str) -> str:
    lines: list[str] = []
    current_name: str | None = None
    current_videos: list[str] = []

    def flush() -> None:
        if current_name and current_videos:
            lines.append(current_name)
            lines.extend(f"  {video}" for video in current_videos)

    for raw_line in output.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if not line.startswith("\t"):
            flush()
            current_name = stripped
            current_videos = []
            continue
        if stripped.startswith("/dev/video"):
            current_videos.append(stripped)

    flush()
    return "\n".join(lines) if lines else "  none"


if __name__ == "__main__":
    main()
