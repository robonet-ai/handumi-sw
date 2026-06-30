from __future__ import annotations

import argparse
from pathlib import Path

from handumi.feetech.bus import FeetechBus
from handumi.feetech.calibration import FeetechConfig, GripperCalibration, load_config, save_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/feetech.yaml"))
    parser.add_argument("--max-width-mm", type=float, required=True)
    args = parser.parse_args()

    current = load_config(args.config)
    left_port = current.left.port or current.port
    right_port = current.right.port or current.port
    if not left_port or not right_port:
        raise SystemExit("Set left/right Feetech ports before calibration.")

    left_bus = FeetechBus(left_port, current.baudrate, current.protocol_version)
    right_bus = left_bus if right_port == left_port else FeetechBus(right_port, current.baudrate, current.protocol_version)
    left_bus.open()
    if right_bus is not left_bus:
        right_bus.open()
    try:
        input("Close both grippers, then press ENTER...")
        left_closed = left_bus.read_position(current.left.servo_id)
        right_closed = right_bus.read_position(current.right.servo_id)
        input("Open both grippers, then press ENTER...")
        left_open = left_bus.read_position(current.left.servo_id)
        right_open = right_bus.read_position(current.right.servo_id)
    finally:
        if right_bus is not left_bus:
            right_bus.close()
        left_bus.close()

    config = FeetechConfig(
        port=current.port,
        baudrate=current.baudrate,
        protocol_version=current.protocol_version,
        left=GripperCalibration(current.left.servo_id, left_closed, left_open, args.max_width_mm, current.left.port),
        right=GripperCalibration(current.right.servo_id, right_closed, right_open, args.max_width_mm, current.right.port),
    )
    save_config(config, args.config)
    print(f"Saved {args.config}")
    print(f"left : closed={left_closed}, open={left_open}, max_width_mm={args.max_width_mm}")
    print(f"right: closed={right_closed}, open={right_open}, max_width_mm={args.max_width_mm}")


if __name__ == "__main__":
    main()
