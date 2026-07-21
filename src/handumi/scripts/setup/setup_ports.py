"""Identify Feetech serial ports and USB cameras while plugging hardware.

Polls the system every ``--interval-s`` seconds, scanning
``/dev/ttyACM*`` / ``/dev/ttyUSB*`` for Feetech servo IDs and listing USB
cameras via ``v4l2-ctl``. Read-only: it never touches ``configs/rig.yaml``,
it just prints what it finds so you can fill that file in by hand.

Usage
-----
::

    handumi setup ports
    handumi setup ports --once
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import grp
import io
import os
import sys
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime

from handumi.config import DEFAULT_RIG_CONFIG
from handumi.feetech.bus import FeetechBus
from handumi.feetech.calibration import default_config

_USB_SERIAL_ADAPTERS = {
    ("1a86", "55d3"): ("QinHeng CH34x / USB Single Serial", "ch341"),
    ("1a86", "7523"): ("QinHeng CH340/CH341", "ch341"),
    ("10c4", "ea60"): ("Silicon Labs CP210x", "cp210x"),
    ("0403", "6001"): ("FTDI USB Serial", "ftdi_sio"),
    ("067b", "2303"): ("Prolific PL2303", "pl2303"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show Feetech serial ports and USB camera devices while plugging/unplugging hardware."
    )
    parser.add_argument(
        "--interval-s",
        type=float,
        default=2.0,
        help="Seconds between screen refreshes.",
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=20)
    args = parser.parse_args()

    scan_ids = range(args.start_id, args.end_id + 1)
    previous_lines: list[str] | None = None
    try:
        while True:
            lines = _capture_frame(scan_ids)
            _draw_frame(lines, previous_lines)
            previous_lines = lines
            if args.once:
                break
            time.sleep(args.interval_s)
    except KeyboardInterrupt:
        print("\nStopped.")


def _capture_frame(scan_ids: range) -> list[str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"=== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        _print_serial_ports(scan_ids)
        _print_camera_ports()
        print(f"\nEdit camera and Feetech assignments in: {DEFAULT_RIG_CONFIG}")
    return buf.getvalue().splitlines()


def _draw_frame(lines: list[str], previous: list[str] | None) -> None:
    """Redraw only the lines that changed instead of clearing the screen.

    Keeps the display steady between refreshes (like ``top``): unchanged
    lines are left untouched and only the ones that differ (e.g. the
    timestamp, or a port that appeared/disappeared) get rewritten.
    """
    out = io.StringIO()
    if previous is None:
        out.write("\033[H\033[2J")
        for line in lines:
            out.write(line + "\n")
    else:
        out.write("\033[H")
        for index, line in enumerate(lines):
            old_line = previous[index] if index < len(previous) else None
            if line != old_line:
                out.write("\033[K" + line + "\n")
            else:
                out.write("\n")
        if len(previous) > len(lines):
            out.write("\033[J")
    sys.stdout.write(out.getvalue())
    sys.stdout.flush()


def _print_serial_ports(scan_ids: range) -> None:
    print("\nFeetech serial ports")
    ports = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    if not ports:
        print("  none")
        _print_missing_serial_port_diagnostics()
        return
    for port in ports:
        print(f"  {port}: ids={_scan_feetech_ids(port, scan_ids)}")
        if hint := _serial_port_permission_hint(port):
            for line in hint:
                print(f"    {line}")


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


def _print_missing_serial_port_diagnostics() -> None:
    adapters = _detect_usb_serial_adapters()
    if not adapters:
        print("  No /dev/ttyACM* or /dev/ttyUSB* devices are present.")
        print("  If the Feetech adapter is plugged in, try another cable/port/hub.")
        return

    print(
        "  USB serial adapters are connected, but no /dev/ttyUSB* "
        "or /dev/ttyACM* exists."
    )
    print("  Detected adapters:")
    for adapter in adapters:
        label = f"{adapter['vendor']}:{adapter['product']} {adapter['name']}"
        if usb_product := adapter.get("usb_product"):
            if usb_product not in label:
                label += f" ({usb_product})"
        if serial := adapter.get("serial"):
            label += f" serial={serial}"
        print(f"    {label}")

    driver_hints = sorted(
        {str(adapter["driver"]) for adapter in adapters if adapter.get("driver")}
    )
    if driver_hints:
        print(f"  Driver hint: {', '.join(driver_hints)}")
        missing = [
            driver for driver in driver_hints if not _kernel_module_available(driver)
        ]
        if missing:
            print(f"  Missing module for the running kernel: {', '.join(missing)}")

    if kernel_hint := _kernel_module_tree_hint():
        print(f"  {kernel_hint}")

    print("  Try:")
    print("    uname -r")
    print("    modinfo ch341")
    print("    ls /usr/lib/modules/$(uname -r)")
    print("    sudo reboot")


def _detect_usb_serial_adapters(
    sys_bus_usb: Path = Path("/sys/bus/usb/devices"),
) -> list[dict[str, str]]:
    adapters: list[dict[str, str]] = []
    if not sys_bus_usb.exists():
        return adapters

    for device in sorted(sys_bus_usb.iterdir()):
        vendor = _read_sysfs_value(device / "idVendor")
        product = _read_sysfs_value(device / "idProduct")
        if not vendor or not product:
            continue
        key = (vendor.lower(), product.lower())
        if key not in _USB_SERIAL_ADAPTERS:
            continue
        name, driver = _USB_SERIAL_ADAPTERS[key]
        adapters.append(
            {
                "vendor": key[0],
                "product": key[1],
                "name": name,
                "usb_product": _read_sysfs_value(device / "product") or "",
                "driver": driver,
                "serial": _read_sysfs_value(device / "serial") or "",
            }
        )
    return adapters


def _read_sysfs_value(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _kernel_module_available(module: str) -> bool:
    if shutil.which("modinfo") is None:
        return True
    result = subprocess.run(
        ["modinfo", module],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _kernel_module_tree_hint() -> str:
    try:
        result = subprocess.run(
            ["uname", "-r"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return ""
    kernel = result.stdout.strip()
    if result.returncode != 0 or not kernel:
        return ""
    modules_path = Path("/usr/lib/modules") / kernel
    if modules_path.exists():
        return ""
    return (
        f"Kernel module tree is missing for running kernel {kernel}; "
        "this usually means the system was updated and needs a reboot."
    )


def _serial_port_permission_hint(port: str) -> list[str]:
    if os.access(port, os.R_OK | os.W_OK):
        return []
    try:
        stat_result = os.stat(port)
        group_name = grp.getgrgid(stat_result.st_gid).gr_name
    except OSError:
        return []
    except KeyError:
        group_name = str(stat_result.st_gid)

    if stat_result.st_gid in os.getgroups():
        return [
            "Permission check failed even though your user is in the device group.",
            "Check udev rules or another process holding the port.",
        ]

    return [
        f"Permission hint: add your user to the serial device group `{group_name}`.",
        f"Run: sudo usermod -aG {group_name} $USER",
        "Then log out and back in.",
    ]


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
