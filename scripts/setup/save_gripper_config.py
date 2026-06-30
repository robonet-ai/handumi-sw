from __future__ import annotations

import argparse
from pathlib import Path

from handumi.feetech.calibration import FeetechConfig, GripperCalibration, load_config, save_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/feetech.yaml"))
    parser.add_argument("--left-port", required=True)
    parser.add_argument("--right-port", required=True)
    parser.add_argument("--left-id", type=int, default=0)
    parser.add_argument("--right-id", type=int, default=1)
    args = parser.parse_args()

    current = load_config(args.config)
    config = FeetechConfig(
        port=None,
        baudrate=current.baudrate,
        protocol_version=current.protocol_version,
        left=GripperCalibration(
            servo_id=args.left_id,
            port=args.left_port,
            closed_ticks=current.left.closed_ticks,
            open_ticks=current.left.open_ticks,
            max_width_mm=current.left.max_width_mm,
        ),
        right=GripperCalibration(
            servo_id=args.right_id,
            port=args.right_port,
            closed_ticks=current.right.closed_ticks,
            open_ticks=current.right.open_ticks,
            max_width_mm=current.right.max_width_mm,
        ),
    )
    save_config(config, args.config)
    print(f"Saved {args.config}")
    print(f"left : port={config.left.port}, id={config.left.servo_id}")
    print(f"right: port={config.right.port}, id={config.right.servo_id}")


if __name__ == "__main__":
    main()
