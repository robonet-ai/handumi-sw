"""Pivot calibration: solve the controller -> gripper-TCP position offset.

Rest the HandUMI gripper tip on a fixed point (a pencil mark on the table)
and, WITHOUT letting the tip slip, rotate the whole device through as many
orientations as you can for ~20-30 seconds. Every sampled controller pose
then satisfies::

    p_i + R_i @ t = c        (tip pinned at unknown fixed point c)

with ``t`` the tip position in the controller frame — exactly the
``calibration.controller_to_gripper_tcp.<side>.position`` entry of
``configs/tracking_meta_quest.yaml``. Stacking all samples gives a linear
least-squares problem in (t, c); the RMS residual tells you how still the
tip really was (aim for < 5 mm).

This calibrates POSITION only. The mount ROTATION is calibrated separately
with the two-stance method (handumi-print-controller-pose). Run one side at
a time; the other side's value is the Y-mirror (the two HandUMI devices are
physical mirror twins), which this script prints too.

Usage
-----
::

    handumi-calibrate-tcp-offset --side left
    handumi-calibrate-tcp-offset --side right --duration-s 30
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from handumi.devices.meta_quest import MetaQuestConfig, MetaQuestReceiver
from handumi.devices.transforms import quat_to_matrix, unity_pose_to_handumi


def solve_pivot(positions: np.ndarray, rotations: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares pivot fit.

    positions: (N, 3) controller positions; rotations: (N, 3, 3) controller
    orientations, both in the same (handumi) frame. Returns (t, c, rms_m):
    tip offset in the controller frame, the fixed pivot point, and the RMS
    residual in meters.
    """
    n = len(positions)
    A = np.zeros((3 * n, 6))
    b = np.zeros(3 * n)
    for i in range(n):
        A[3 * i : 3 * i + 3, 0:3] = rotations[i]
        A[3 * i : 3 * i + 3, 3:6] = -np.eye(3)
        b[3 * i : 3 * i + 3] = -positions[i]
    solution, *_ = np.linalg.lstsq(A, b, rcond=None)
    t, c = solution[:3], solution[3:]
    residuals = (A @ solution - b).reshape(n, 3)
    rms = float(np.sqrt((residuals**2).sum(axis=1).mean()))
    return t, c, rms


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--side", choices=["left", "right"], required=True)
    parser.add_argument("--tracking-config", type=Path, default=Path("configs/tracking_meta_quest.yaml"))
    parser.add_argument("--quest-ip", type=str, default=None)
    parser.add_argument("--duration-s", type=float, default=25.0)
    parser.add_argument("--rate-hz", type=float, default=30.0)
    args = parser.parse_args()

    config = MetaQuestConfig.from_yaml(args.tracking_config)
    if args.quest_ip is not None:
        config = MetaQuestConfig(
            quest_ip=args.quest_ip, tcp_port=config.tcp_port,
            sync_port=config.sync_port, connect_retry_s=config.connect_retry_s,
        )

    receiver = MetaQuestReceiver(config)
    receiver.start()
    print(f"Connecting to Quest at {config.quest_ip}:{config.tcp_port} ...")
    print(f"\nPin the {args.side.upper()} gripper TIP on a fixed point, then keep it")
    print("pinned while rotating the device in all directions.")
    input("Press Enter to start sampling... ")

    positions: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    deadline = time.monotonic() + args.duration_s
    period = 1.0 / args.rate_hz
    try:
        while time.monotonic() < deadline:
            frame = receiver.latest()
            if frame is not None:
                controller = getattr(frame, args.side)
                if controller.tracked and controller.valid:
                    pose = unity_pose_to_handumi(controller.position, controller.quaternion)
                    positions.append(np.asarray(pose.position, dtype=np.float64))
                    rotations.append(quat_to_matrix(pose.quaternion))
            remaining = deadline - time.monotonic()
            print(f"\r  sampling... {remaining:5.1f}s left, {len(positions)} samples", end="")
            time.sleep(period)
    finally:
        receiver.stop()
    print()

    if len(positions) < 50:
        raise SystemExit(f"Only {len(positions)} tracked samples — check trk=1 and retry.")

    P, R = np.asarray(positions), np.asarray(rotations)
    # Rotation diversity check: a pivot fit is only well-conditioned if the
    # orientations actually varied (a still device makes A rank-deficient).
    spread = float(np.linalg.norm(R - R.mean(axis=0)))
    if spread < 1.0:
        print("WARNING: little rotation detected — rotate the device more next time.")

    t, c, rms = solve_pivot(P, R)
    mirrored = np.array([t[0], -t[1], t[2]])
    print(f"\nsamples: {len(P)}   RMS residual: {rms * 1000:.1f} mm "
          f"({'OK' if rms < 0.005 else 'HIGH — tip moved, consider re-running'})")
    print(f"pivot point (workspace): {np.round(c, 4)}")
    print(f"\nconfigs/tracking_meta_quest.yaml -> calibration.controller_to_gripper_tcp:")
    print(f"  {args.side}:")
    print(f"    position: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
    other = "right" if args.side == "left" else "left"
    print(f"  {other} (mirrored, verify by calibrating that side too):")
    print(f"    position: [{mirrored[0]:.4f}, {mirrored[1]:.4f}, {mirrored[2]:.4f}]")


if __name__ == "__main__":
    main()
