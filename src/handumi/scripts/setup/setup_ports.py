"""Identify Feetech serial ports and USB cameras while plugging hardware.

Watches udev for serial/camera changes, scans ``/dev/ttyACM*`` /
``/dev/ttyUSB*`` for Feetech servo IDs, and lists USB cameras via
``v4l2-ctl``. Use this to fill in ``configs/rig.yaml``.

Usage
-----
::

    handumi-setup-ports
    handumi-setup-ports --once
"""

from __future__ import annotations

import argparse
import glob
import grp
import os
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
        help="Polling interval used only when udevadm is unavailable or --poll is set.",
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--poll",
        action="store_true",
        help="Poll at --interval-s instead of waiting for udev change events.",
    )
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=20)
    args = parser.parse_args()

    monitor = None if args.once or args.poll else _start_udev_monitor()
    if not args.once and not args.poll and monitor is None:
        print("udevadm monitor unavailable; falling back to polling.")
        time.sleep(1.0)

    scan_ids = range(args.start_id, args.end_id + 1)
    try:
        while True:
            _render_status(scan_ids)
            if args.once:
                break
            if monitor is None:
                time.sleep(args.interval_s)
                continue
            if not _wait_for_udev_event(monitor):
                monitor = None
                time.sleep(args.interval_s)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        _stop_udev_monitor(monitor)


def _render_status(scan_ids: range) -> None:
    _clear()
    print(f"=== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    _print_serial_ports(scan_ids)
    _print_camera_ports()
    print(f"\nEdit camera and Feetech assignments in: {DEFAULT_RIG_CONFIG}")


def _start_udev_monitor() -> subprocess.Popen[str] | None:
    if shutil.which("udevadm") is None:
        return None
    try:
        return subprocess.Popen(
            [
                "udevadm",
                "monitor",
                "--udev",
                "--subsystem-match=usb",
                "--subsystem-match=tty",
                "--subsystem-match=video4linux",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return None


def _wait_for_udev_event(monitor: subprocess.Popen[str]) -> bool:
    if monitor.stdout is None:
        return False
    while True:
        line = monitor.stdout.readline()
        if line == "":
            return monitor.poll() is None
        if "/usb/" in line or "/tty/" in line or "/video4linux/" in line:
            return True


def _stop_udev_monitor(monitor: subprocess.Popen[str] | None) -> None:
    if monitor is None or monitor.poll() is not None:
        return
    monitor.terminate()
    try:
        monitor.wait(timeout=1)
    except subprocess.TimeoutExpired:
        monitor.kill()
        monitor.wait(timeout=1)


def _clear() -> None:
    print("\033c", end="")


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
