from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from handumi.feetech.bus import FeetechBus


@dataclass
class Monitor:
    port: str
    servo_id: int
    bus: FeetechBus
    initial: int
    last: int
    peak_delta: int = 0

    def update(self) -> None:
        self.last = self.bus.read_position(self.servo_id)
        self.peak_delta = max(self.peak_delta, abs(self.last - self.initial))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port-id", nargs=2, action="append", metavar=("PORT", "ID"), required=True)
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument("--protocol-version", type=int, default=0)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--interval-s", type=float, default=0.2)
    parser.add_argument("--keep-torque", action="store_true")
    args = parser.parse_args()

    monitors: list[Monitor] = []
    buses: list[FeetechBus] = []
    try:
        for port, raw_id in args.port_id:
            servo_id = int(raw_id)
            bus = FeetechBus(port=port, baudrate=args.baudrate, protocol_version=args.protocol_version)
            bus.open()
            buses.append(bus)
            if not args.keep_torque:
                bus.disable_torque(servo_id)
            ticks = bus.read_position(servo_id)
            monitors.append(Monitor(port, servo_id, bus, ticks, ticks))

        print("Move one gripper. If ticks stay fixed, the servo encoder is not moving or this register is not the right signal.")
        deadline = time.monotonic() + args.duration_s
        while time.monotonic() < deadline:
            for monitor in monitors:
                monitor.update()
            _print(monitors)
            time.sleep(args.interval_s)
    finally:
        for bus in buses:
            bus.close()


def _print(monitors: list[Monitor]) -> None:
    print("port          id  ticks  delta  peak_delta")
    for monitor in monitors:
        delta = monitor.last - monitor.initial
        print(f"{monitor.port:<12} {monitor.servo_id:>2}  {monitor.last:>5}  {delta:>5}  {monitor.peak_delta:>10}")
    print()


if __name__ == "__main__":
    main()
