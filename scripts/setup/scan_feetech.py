from __future__ import annotations

import argparse

from handumi.feetech.bus import FeetechBus
from handumi.feetech.calibration import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/feetech.yaml")
    parser.add_argument("--port", default=None)
    parser.add_argument("--all-ports", action="store_true")
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=20)
    args = parser.parse_args()

    config = load_config(args.config)
    ports = _ports(args.port, args.all_ports, config.port)
    ids = range(args.start_id, args.end_id + 1)

    for port in ports:
        try:
            with FeetechBus(port=port, baudrate=config.baudrate, protocol_version=config.protocol_version) as bus:
                found = bus.scan(ids)
        except Exception as exc:
            print(f"{port}: ERROR {exc}")
            continue
        print(f"{port}: {found if found else 'no servos found'}")


def _ports(port: str | None, all_ports: bool, config_port: str | None) -> list[str]:
    if port:
        return [port]
    if config_port and not all_ports:
        return [config_port]
    from serial.tools import list_ports

    return sorted(
        item.device
        for item in list_ports.comports()
        if "ttyUSB" in item.device or "ttyACM" in item.device or "tty.usb" in item.device
    )


if __name__ == "__main__":
    main()
