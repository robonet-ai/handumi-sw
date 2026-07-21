"""Print live left/right controller pose for mounting-offset checks.

Two-stance method for the controller-to-TCP calibration quaternion:

  A) Hold the BARE controller in the natural handheld grip, pointing
     forward. Record the printed quaternion ``q_A``.
  B) Mount the controller in the HandUMI and hold the device pointing the
     SAME forward direction. Record ``q_B``.

The offset is ``conj(q_B) * q_A`` (``quat_multiply`` from
``handumi.tracking.transforms``): the mounted controller then reads as a
naturally-held one. Using the *difference* between the stances cancels the
tracking frame's arbitrary yaw and the OVR frame's built-in tilt, so
neither stance needs to be aligned with the tracking origin.

No mounting offset or workspace reset is applied here. This prints the
device-frame controller pose normalized into HandUMI pose7 convention.

Usage
-----
::

    handumi tracking pose --device meta
    handumi tracking pose --device meta --quest-ip 192.168.1.42
    handumi tracking pose --device pico --pico-mode mandos
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from handumi.calibration.control_tcp import ControllerTcpCalibration
from handumi.config import DEFAULT_RIG_CONFIG
from handumi.robots.utils import IDENTITY_POSE7
from handumi.scripts.record import build_tracker


def _identity_tcp_calibration() -> ControllerTcpCalibration:
    pose = IDENTITY_POSE7.astype(np.float32)
    return ControllerTcpCalibration(left=pose.copy(), right=pose.copy(), source=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rig-config",
        dest="rig_config",
        type=Path,
        default=DEFAULT_RIG_CONFIG,
    )
    parser.add_argument("--device", choices=("meta", "pico"), default="meta")
    parser.add_argument("--quest-ip", type=str, default=None)
    parser.add_argument("--tcp-port", type=int, default=None)
    parser.add_argument("--sync-port", type=int, default=None)
    parser.add_argument(
        "--pico-mode",
        choices=("mandos", "object", "whole-body"),
        default="mandos",
    )
    pico_transport = parser.add_mutually_exclusive_group()
    pico_transport.add_argument("--pico-adb", action="store_true")
    pico_transport.add_argument("--pico-wifi", action="store_true")
    parser.add_argument("--skip-adb-check", action="store_true")
    parser.add_argument("--rate-hz", type=float, default=2.0)
    args = parser.parse_args()

    tracker = build_tracker(
        args,
        _identity_tcp_calibration(),
        reset_workspace_on_x=False,
    )
    tracker.start()
    print(f"Connecting to {args.device} tracking. Ctrl+C to stop.\n")
    print("Hold the stance you want to measure, then read the printed quaternion.\n")

    try:
        while True:
            sample = tracker.latest()
            if not sample.streaming:
                sys.stdout.write(
                    f"\r(waiting for {args.device} frames)                                    "
                )
                sys.stdout.flush()
                time.sleep(1.0 / args.rate_hz)
                continue

            for side in ("left", "right"):
                if not getattr(sample, f"{side}_device_tracked"):
                    continue
                pose = getattr(sample, f"{side}_device_controller_pose")
                qx, qy, qz, qw = pose[3:7]
                sys.stdout.write(
                    f"\r{side:5s} tracked=1 quaternion=[{qx:+.4f}, {qy:+.4f}, {qz:+.4f}, {qw:+.4f}]"
                    f"  position=[{pose[0]:+.3f}, {pose[1]:+.3f}, {pose[2]:+.3f}]"
                    "          \n"
                )
            sys.stdout.flush()
            time.sleep(1.0 / args.rate_hz)
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()


if __name__ == "__main__":
    main()
