"""Print live left/right controller orientation to find the mounting-offset rotation.

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

No mounting offset or workspace reset is applied here — this is the raw
Unity->handumi-converted controller pose, on purpose.

Usage
-----
::

    handumi-print-controller-pose
    handumi-print-controller-pose --quest-ip 192.168.1.42
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from handumi.config import DEFAULT_RIG_CONFIG
from handumi.tracking.meta_quest import MetaQuestConfig, MetaQuestReceiver
from handumi.tracking.transforms import unity_pose_to_handumi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rig-config",
        dest="rig_config",
        type=Path,
        default=DEFAULT_RIG_CONFIG,
    )
    parser.add_argument("--quest-ip", type=str, default=None)
    parser.add_argument("--tcp-port", type=int, default=None)
    parser.add_argument("--sync-port", type=int, default=None)
    parser.add_argument("--rate-hz", type=float, default=2.0)
    args = parser.parse_args()

    config = MetaQuestConfig.from_yaml(args.rig_config)
    config = MetaQuestConfig(
        quest_ip=args.quest_ip if args.quest_ip is not None else config.quest_ip,
        tcp_port=args.tcp_port if args.tcp_port is not None else config.tcp_port,
        sync_port=args.sync_port if args.sync_port is not None else config.sync_port,
        connect_retry_s=config.connect_retry_s,
        frame_stale_timeout_s=config.frame_stale_timeout_s,
    )

    receiver = MetaQuestReceiver(config)
    receiver.start()
    print(f"Connecting to Quest at {config.quest_ip}:{config.tcp_port}. Ctrl+C to stop.\n")
    print("Hold the stance you want to measure, then read the printed quaternion.\n")

    try:
        while True:
            frame = receiver.latest()
            if frame is None:
                sys.stdout.write("\r(waiting for Quest frames)                                    ")
                sys.stdout.flush()
                time.sleep(1.0 / args.rate_hz)
                continue

            for side, ctrl in (("left", frame.left), ("right", frame.right)):
                if not ctrl.tracked:
                    continue
                pose = unity_pose_to_handumi(ctrl.position, ctrl.quaternion)
                qx, qy, qz, qw = pose.quaternion
                sys.stdout.write(
                    f"\r{side:5s} tracked=1 quaternion=[{qx:+.4f}, {qy:+.4f}, {qz:+.4f}, {qw:+.4f}]"
                    f"  position=[{pose.position[0]:+.3f}, {pose.position[1]:+.3f}, {pose.position[2]:+.3f}]"
                    "          \n"
                )
            sys.stdout.flush()
            time.sleep(1.0 / args.rate_hz)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()
