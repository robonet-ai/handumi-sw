"""Calibrate the fixed workspace -> robot-world transform (configs/teleop.yaml).

Teleop is absolute: real-world positions must map to the same robot-world
positions every session, so a task scene placed identically in sim and in
reality stays aligned. This script solves the translation (+ optional yaw)
of that mapping from reference touches:

  1. Wear the Quest as in a normal session and reset the workspace
     (this script uses the first tracked HMD pose as the origin, same as
     live tracking does).
  2. Touch a point whose ROBOT-WORLD coordinates you know with the gripper
     tip (e.g. the scene origin from configs/scene.yaml, a table corner
     measured from the arm base plate) and hold still.
  3. Enter those coordinates when prompted; the script averages ~1s of
     tracked TCP poses at that instant.

One point solves the translation (yaw stays 0). Two or more points also
solve the yaw about Z (least squares). Requires the controller -> TCP
calibration to be done first (handumi-calibrate-tcp-offset +
handumi-print-controller-pose) — a wrong TCP poisons this step.

Usage
-----
::

    handumi-calibrate-workspace                # interactive
    handumi-calibrate-workspace --quest-ip ...
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def solve_transform(
    workspace_points: np.ndarray, robot_points: np.ndarray, *, solve_yaw: bool
) -> tuple[np.ndarray, float, float]:
    """Fit robot = R_yaw(about Z) @ workspace + t. Returns (t, yaw_deg, rms_m)."""
    W, B = np.asarray(workspace_points), np.asarray(robot_points)
    yaw = 0.0
    if solve_yaw and len(W) >= 2:
        # 2D Procrustes about Z on the centered XY coordinates.
        Wc, Bc = W - W.mean(axis=0), B - B.mean(axis=0)
        num = float((Wc[:, 0] * Bc[:, 1] - Wc[:, 1] * Bc[:, 0]).sum())
        den = float((Wc[:, 0] * Bc[:, 0] + Wc[:, 1] * Bc[:, 1]).sum())
        yaw = np.arctan2(num, den)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    t = (B - W @ R.T).mean(axis=0)
    rms = float(np.sqrt(((W @ R.T + t - B) ** 2).sum(axis=1).mean()))
    return t, float(np.rad2deg(yaw)), rms


def _average_tcp(receiver: MetaQuestReceiver, side: str, mounts: MountingOffsets,
                 workspace, seconds: float = 1.0) -> np.ndarray | None:
    """Average the tracked TCP workspace position over ``seconds``."""
    from handumi.tracking.meta_quest import controller_pose_in_workspace

    samples: list[np.ndarray] = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        frame = receiver.latest()
        if frame is not None:
            controller = getattr(frame, side)
            if controller.tracked and controller.valid:
                offset = mounts.left if side == "left" else mounts.right
                pose = controller_pose_in_workspace(
                    controller, mounting_offset=offset, workspace=workspace
                )
                samples.append(np.asarray(pose.position, dtype=np.float64))
        time.sleep(0.02)
    if len(samples) < 10:
        return None
    return np.asarray(samples).mean(axis=0)


def main() -> None:
    from handumi.tracking.meta_quest import MetaQuestConfig, MetaQuestReceiver, workspace_from_hmd
    from handumi.tracking.transforms import MountingOffsets

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tracking-config", type=Path, default=Path("configs/tracking_meta_quest.yaml"))
    parser.add_argument("--quest-ip", type=str, default=None)
    parser.add_argument("--side", choices=["left", "right"], default="right",
                        help="Which gripper does the touching (default right).")
    args = parser.parse_args()

    config = MetaQuestConfig.from_yaml(args.tracking_config)
    if args.quest_ip is not None:
        config = MetaQuestConfig(
            quest_ip=args.quest_ip, tcp_port=config.tcp_port,
            sync_port=config.sync_port, connect_retry_s=config.connect_retry_s,
        )
    mounts = MountingOffsets.from_yaml(args.tracking_config)

    receiver = MetaQuestReceiver(config)
    receiver.start()
    print(f"Connecting to Quest at {config.quest_ip}:{config.tcp_port} ...")

    # Workspace origin: first tracked HMD pose, same convention as live
    # tracking/recording (wear the Quest exactly as in a real session).
    print("Waiting for a tracked HMD pose to set the workspace origin...")
    workspace = None
    while workspace is None:
        frame = receiver.latest()
        if frame is not None and frame.hmd.tracked:
            workspace = workspace_from_hmd(frame.hmd)
        time.sleep(0.05)
    print("Workspace origin set.\n")

    workspace_points: list[np.ndarray] = []
    robot_points: list[np.ndarray] = []
    try:
        while True:
            prompt = (f"Point {len(robot_points) + 1}: robot-world x y z in meters "
                      "(empty to finish): ")
            line = input(prompt).strip()
            if not line:
                break
            try:
                target = np.array([float(v) for v in line.split()], dtype=np.float64)
                assert target.shape == (3,)
            except (ValueError, AssertionError):
                print("  Expected three numbers, e.g.: 0.35 0.0 0.0")
                continue
            input(f"  Touch it with the {args.side} gripper tip, hold still, press Enter... ")
            measured = _average_tcp(receiver, args.side, mounts, workspace)
            if measured is None:
                print("  Not enough tracked samples — check trk=1 and retry this point.")
                continue
            workspace_points.append(measured)
            robot_points.append(target)
            print(f"  captured workspace TCP: {np.round(measured, 4)}")
    finally:
        receiver.stop()

    if not robot_points:
        raise SystemExit("No points captured.")

    solve_yaw = len(robot_points) >= 2
    t, yaw_deg, rms = solve_transform(
        np.asarray(workspace_points), np.asarray(robot_points), solve_yaw=solve_yaw
    )
    print(f"\npoints: {len(robot_points)}   fit RMS: {rms * 1000:.1f} mm"
          + ("" if solve_yaw else "   (yaw fixed at 0 — add a 2nd point to solve it)"))
    print("\nPaste into configs/teleop.yaml:\n")
    print("workspace_to_robot:")
    print(f"  translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
    print(f"  yaw_deg: {yaw_deg:.2f}")


if __name__ == "__main__":
    main()
