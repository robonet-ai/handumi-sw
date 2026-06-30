from __future__ import annotations

import argparse

from handumi.feetech.bus import FeetechBus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--current-id", type=int, required=True)
    parser.add_argument("--new-id", type=int, required=True)
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument("--protocol-version", type=int, default=0)
    args = parser.parse_args()

    with FeetechBus(port=args.port, baudrate=args.baudrate, protocol_version=args.protocol_version) as bus:
        try:
            bus.write_servo_id(args.current_id, args.new_id)
        except RuntimeError as exc:
            if bus.ping(args.new_id):
                print(f"Warning: write returned an error, but servo responds as ID {args.new_id}.")
            else:
                raise SystemExit(f"Could not write ID {args.current_id} -> {args.new_id}: {exc}") from exc

    print(f"{args.port}: ID {args.current_id} -> {args.new_id}")


if __name__ == "__main__":
    main()
