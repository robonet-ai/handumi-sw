#!/usr/bin/env python3
"""Replay PICO body arm motion on the Piper Viser simulation.

PICO's frame is calibrated into Piper's world frame: PICO ``x,z,y`` maps to
Piper ``x,y,z`` (front, lateral, vertical). Playback starts from Piper's
standard all-zero joint pose; a front workspace remains available for tuning.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import numpy as np

from dexumi.retargeting.piper_from_pico import (
    PicoToPiperArmRetargeter,
    move_retargeter_to_front_workspace,
    settle_first_frame,
)
from dexumi.robots.piper.config import KinematicsConfig
from dexumi.robots.piper.solver import KinematicsSolver
from dexumi.robots.piper.sim import Sim
from dexumi.utils.lerobot_io import load_pico_body_poses


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay PICO body arm motion on Piper's Viser simulation."
    )
    parser.add_argument("--repo-id", default="NONHUMAN-RESEARCH/dexumi-dataset-v2")
    parser.add_argument("--dataset-root", default="outputs/datasets/dexumi-dataset-v2")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--column", default="observation.pico.body_joints_pose")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--axis-map",
        default="x,z,y",
        help="PICO delta to Piper target delta. Default: x,z,y.",
    )
    parser.add_argument("--left-only", action="store_true")
    parser.add_argument("--right-only", action="store_true")
    parser.add_argument("--gripper", type=float, default=1.0)
    parser.add_argument(
        "--piper-workspace",
        choices=("front", "rest"),
        default="rest",
        help="Use Piper's standard all-zero pose or an optional front workspace.",
    )
    parser.add_argument("--piper-wrist-forward", type=float, default=0.34)
    parser.add_argument("--piper-wrist-height", type=float, default=0.24)
    parser.add_argument("--piper-wrist-lateral", type=float, default=0.23)
    parser.add_argument("--piper-elbow-forward", type=float, default=0.16)
    parser.add_argument("--piper-elbow-height", type=float, default=0.34)
    parser.add_argument("--piper-elbow-lateral", type=float, default=0.20)
    parser.add_argument(
        "--settle-iterations",
        type=int,
        default=20,
        help="IK iterations on the first frame before playback starts.",
    )
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument(
        "--loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Loop dataset replay indefinitely (default: on).",
    )
    parser.add_argument(
        "--hold-after",
        type=float,
        default=None,
        help="Seconds to keep Viser alive after a non-looping replay. Default: indefinite.",
    )
    parser.add_argument("--save", default=None, help="Optional .npz path for solved joints.")

    parser.add_argument("--pos-weight", type=float, default=50.0)
    parser.add_argument("--ori-weight", type=float, default=0.0)
    parser.add_argument("--elbow-weight", type=float, default=5.0)
    parser.add_argument(
        "--manipulability-weight",
        type=float,
        default=0.0,
        help="Keep Piper's standard zero pose stable during calibration.",
    )
    parser.add_argument(
        "--max-joint-delta",
        type=float,
        default=KinematicsConfig().max_joint_delta,
        help="Maximum Piper joint change per dataset frame in radians.",
    )
    parser.add_argument("--max-reach", type=float, default=0.8)
    return parser


async def replay_once(
    *,
    sim: Sim,
    retargeter: PicoToPiperArmRetargeter,
    poses: np.ndarray,
    frame_indices: list[int],
    playback_fps: float,
    save_records: bool,
    initial_q: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    q = initial_q.copy()
    left, right = retargeter.split_for_sim(q)
    await sim.motion_control(left=left, right=right)

    q_records: list[np.ndarray] = []
    left_records: list[np.ndarray] = []
    right_records: list[np.ndarray] = []

    frame_delay = 0.0 if playback_fps <= 0 else 1.0 / playback_fps
    next_time = time.perf_counter()

    for frame_number, frame_index in enumerate(frame_indices):
        # The first selected pose defines the zero-motion calibration already
        # represented by initial_q; solving it again can leave Piper's zero pose.
        if frame_number > 0:
            q = retargeter.retarget_frame(poses[frame_index], q)
        left, right = retargeter.split_for_sim(q)
        await sim.motion_control(left=left, right=right)

        if save_records:
            q_records.append(q.copy())
            left_records.append(left.copy())
            right_records.append(right.copy())

        if frame_number % 30 == 0:
            print(f"frame {frame_index}/{len(poses) - 1}")

        next_time += frame_delay
        if frame_delay > 0:
            await asyncio.sleep(max(0.0, next_time - time.perf_counter()))

    return q_records, left_records, right_records


async def main_async() -> None:
    args = build_parser().parse_args()

    if args.left_only and args.right_only:
        raise ValueError("Use only one of --left-only or --right-only.")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1.")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be >= 0.")

    poses, dataset_fps = load_pico_body_poses(
        repo_id=args.repo_id,
        root=args.dataset_root,
        episode=args.episode,
        column=args.column,
        revision=args.revision,
    )
    if args.start_frame >= len(poses):
        raise ValueError(
            f"--start-frame {args.start_frame} is outside dataset length {len(poses)}."
        )

    frame_indices = list(range(args.start_frame, len(poses), args.stride))
    if args.max_frames is not None:
        frame_indices = frame_indices[: args.max_frames]
    if not frame_indices:
        raise ValueError("No frames selected for replay.")

    playback_fps = float(args.fps if args.fps is not None else dataset_fps)

    config = KinematicsConfig(
        pos_weight=args.pos_weight,
        ori_weight=args.ori_weight,
        elbow_weight=args.elbow_weight,
        manipulability_weight=args.manipulability_weight,
        max_joint_delta=args.max_joint_delta,
        max_reach=args.max_reach,
    )
    solver = KinematicsSolver(config=config)
    retargeter = PicoToPiperArmRetargeter(
        solver=solver,
        first_body_pose=poses[frame_indices[0]],
        scale=args.scale,
        axis_map=args.axis_map,
        enable_left=not args.right_only,
        enable_right=not args.left_only,
        gripper=args.gripper,
    )
    if args.piper_workspace == "front":
        move_retargeter_to_front_workspace(
            retargeter,
            wrist_forward=args.piper_wrist_forward,
            wrist_height=args.piper_wrist_height,
            wrist_lateral=args.piper_wrist_lateral,
            elbow_forward=args.piper_elbow_forward,
            elbow_height=args.piper_elbow_height,
            elbow_lateral=args.piper_elbow_lateral,
        )

    initial_q = settle_first_frame(
        retargeter,
        poses[frame_indices[0]],
        0 if args.piper_workspace == "rest" else args.settle_iterations,
    )
    if args.piper_workspace == "front":
        solver.set_posture_pose(initial_q)

    sim = Sim(port=args.port)
    await sim.enable()
    await asyncio.sleep(0.5)

    print(f"Viser simulation: http://localhost:{args.port}")
    print(
        "Replay config: "
        f"frames={len(frame_indices)}, fps={playback_fps:g}, "
        f"scale={args.scale:g}, axis_map={args.axis_map!r}, "
        f"workspace={args.piper_workspace!r}, ori_weight={args.ori_weight:g}"
    )

    all_q: list[np.ndarray] = []
    all_left: list[np.ndarray] = []
    all_right: list[np.ndarray] = []
    first_pass = True

    while True:
        q_records, left_records, right_records = await replay_once(
            sim=sim,
            retargeter=retargeter,
            poses=poses,
            frame_indices=frame_indices,
            playback_fps=playback_fps,
            save_records=bool(args.save and first_pass),
            initial_q=initial_q,
        )
        all_q.extend(q_records)
        all_left.extend(left_records)
        all_right.extend(right_records)
        first_pass = False

        if not args.loop:
            break

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path,
            q=np.asarray(all_q, dtype=np.float32),
            left=np.asarray(all_left, dtype=np.float32),
            right=np.asarray(all_right, dtype=np.float32),
            frame_indices=np.asarray(frame_indices, dtype=np.int32),
            fps=np.asarray(playback_fps, dtype=np.float32),
            scale=np.asarray(args.scale, dtype=np.float32),
            axis_map=np.asarray(args.axis_map),
            piper_workspace=np.asarray(args.piper_workspace),
        )
        print(f"Saved solved joints to {save_path}")

    if args.hold_after is not None:
        print(f"Holding simulation for {args.hold_after:g}s...")
        await asyncio.sleep(args.hold_after)
    else:
        print("Simulation running indefinitely. Press Ctrl+C to stop.")
        await asyncio.Event().wait()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
