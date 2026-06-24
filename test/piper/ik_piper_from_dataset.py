#!/usr/bin/env python3
"""Replay PICO body arm motion on the Piper Viser simulation.

This is intentionally kept under tests/manual while the coordinate mapping,
scale, and calibration behavior are still being validated visually.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from dexumi.robots.piper.config import KinematicsConfig
from dexumi.robots.piper.solver import KinematicsSolver
from dexumi.robots.piper.sim import Sim
from dexumi.robots.piper.shared import ARM_JOINT_COUNT, COMMAND_SIZE, GRIPPER_INDEX
from dexumi.utils.lerobot_io import load_pico_body_poses

# SMPL24/PICO joint indices used by SONIC's body skeleton.
LEFT_SHOULDER = 16
RIGHT_SHOULDER = 17
LEFT_ELBOW = 18
RIGHT_ELBOW = 19
LEFT_WRIST = 20
RIGHT_WRIST = 21

# Neutral rest pose for the Piper arm (6 revolute joints, in radians).
REST_LEFT_ARM = np.zeros(ARM_JOINT_COUNT, dtype=np.float32)
REST_RIGHT_ARM = np.zeros(ARM_JOINT_COUNT, dtype=np.float32)


@dataclass(frozen=True)
class ArmReference:
    """Calibration data for one arm at the first replay frame."""

    human_wrist_rel: np.ndarray
    human_elbow_rel: np.ndarray
    robot_wrist_pos: np.ndarray
    robot_elbow_pos: np.ndarray
    robot_wrist_rot: np.ndarray


@dataclass(frozen=True)
class RetargetReferences:
    """Left/right calibration data."""

    left: ArmReference
    right: ArmReference


def parse_axis_map(spec: str) -> Callable[[np.ndarray], np.ndarray]:
    """Build a vector transform from a spec like ``z,x,y`` or ``z,y,-x``."""

    axes = {"x": 0, "y": 1, "z": 2}
    parts = [part.strip().lower() for part in spec.split(",")]
    if len(parts) != 3:
        raise ValueError("--axis-map must contain exactly 3 comma-separated axes.")

    rows: list[tuple[int, int]] = []
    used: set[int] = set()
    for part in parts:
        sign = -1 if part.startswith("-") else 1
        axis = part[1:] if part.startswith("-") else part
        if axis not in axes:
            raise ValueError(f"Invalid axis {part!r}; use x, y, z, or a negated axis.")
        index = axes[axis]
        if index in used:
            raise ValueError(f"Axis {axis!r} is repeated in --axis-map.")
        used.add(index)
        rows.append((sign, index))

    def transform(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        return np.array([sign * vector[index] for sign, index in rows], dtype=np.float32)

    return transform


def make_rest_q(solver: KinematicsSolver) -> np.ndarray:
    """Create the full Piper joint vector for the rest pose."""

    q = np.zeros(solver.num_joints, dtype=np.float32)
    q[solver.left_indices] = REST_LEFT_ARM
    q[solver.right_indices] = REST_RIGHT_ARM
    return q


def _body_position(body_pose: np.ndarray, joint_index: int) -> np.ndarray:
    return np.asarray(body_pose[joint_index, :3], dtype=np.float32)


def _human_arm_reference(
    body_pose: np.ndarray,
    *,
    shoulder_index: int,
    elbow_index: int,
    wrist_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    shoulder = _body_position(body_pose, shoulder_index)
    elbow = _body_position(body_pose, elbow_index)
    wrist = _body_position(body_pose, wrist_index)
    return wrist - shoulder, elbow - shoulder


def _robot_link_pose(
    solver: KinematicsSolver,
    q: np.ndarray,
    link_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    import jax.numpy as jnp
    import jaxlie

    fk = solver.robot.forward_kinematics(jnp.asarray(q, dtype=jnp.float32))
    pose = jaxlie.SE3(fk[link_index])
    return (
        np.asarray(pose.translation(), dtype=np.float32),
        np.asarray(pose.rotation().as_matrix(), dtype=np.float32),
    )


def calibrate_from_first_frame(
    solver: KinematicsSolver,
    first_body_pose: np.ndarray,
    q_rest: np.ndarray,
) -> RetargetReferences:
    """Use the first human frame and Piper rest pose as the shared zero motion."""

    left_wrist_rel, left_elbow_rel = _human_arm_reference(
        first_body_pose,
        shoulder_index=LEFT_SHOULDER,
        elbow_index=LEFT_ELBOW,
        wrist_index=LEFT_WRIST,
    )
    right_wrist_rel, right_elbow_rel = _human_arm_reference(
        first_body_pose,
        shoulder_index=RIGHT_SHOULDER,
        elbow_index=RIGHT_ELBOW,
        wrist_index=RIGHT_WRIST,
    )

    left_wrist_pos, left_wrist_rot = _robot_link_pose(solver, q_rest, solver.l_ee_idx)
    right_wrist_pos, right_wrist_rot = _robot_link_pose(solver, q_rest, solver.r_ee_idx)
    left_elbow_pos, _ = _robot_link_pose(solver, q_rest, solver.l_elbow_idx)
    right_elbow_pos, _ = _robot_link_pose(solver, q_rest, solver.r_elbow_idx)

    return RetargetReferences(
        left=ArmReference(
            human_wrist_rel=left_wrist_rel,
            human_elbow_rel=left_elbow_rel,
            robot_wrist_pos=left_wrist_pos,
            robot_elbow_pos=left_elbow_pos,
            robot_wrist_rot=left_wrist_rot,
        ),
        right=ArmReference(
            human_wrist_rel=right_wrist_rel,
            human_elbow_rel=right_elbow_rel,
            robot_wrist_pos=right_wrist_pos,
            robot_elbow_pos=right_elbow_pos,
            robot_wrist_rot=right_wrist_rot,
        ),
    )


class PicoToPiperArmRetargeter:
    """Minimal relative-position retargeter for visual Piper tests."""

    def __init__(
        self,
        *,
        solver: KinematicsSolver,
        first_body_pose: np.ndarray,
        scale: float,
        axis_map: str,
        enable_left: bool = True,
        enable_right: bool = True,
        gripper: float = 1.0,
    ) -> None:
        self.solver = solver
        self.scale = float(scale)
        self.transform = parse_axis_map(axis_map)
        self.enable_left = enable_left
        self.enable_right = enable_right
        self.gripper = float(gripper)

        self.q_rest = make_rest_q(solver)
        self.solver.set_posture_pose(self.q_rest)
        self.refs = calibrate_from_first_frame(solver, first_body_pose, self.q_rest)

    def _arm_targets(
        self,
        body_pose: np.ndarray,
        *,
        ref: ArmReference,
        shoulder_index: int,
        elbow_index: int,
        wrist_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        wrist_rel, elbow_rel = _human_arm_reference(
            body_pose,
            shoulder_index=shoulder_index,
            elbow_index=elbow_index,
            wrist_index=wrist_index,
        )
        wrist_delta = self.transform(wrist_rel - ref.human_wrist_rel) * self.scale
        elbow_delta = self.transform(elbow_rel - ref.human_elbow_rel) * self.scale
        return ref.robot_wrist_pos + wrist_delta, ref.robot_elbow_pos + elbow_delta

    def retarget_frame(self, body_pose: np.ndarray, q_current: np.ndarray) -> np.ndarray:
        """Solve one frame of PICO body data into a Piper joint vector."""

        left_pose = None
        left_elbow_pos = None
        if self.enable_left:
            left_wrist_pos, left_elbow_pos = self._arm_targets(
                body_pose,
                ref=self.refs.left,
                shoulder_index=LEFT_SHOULDER,
                elbow_index=LEFT_ELBOW,
                wrist_index=LEFT_WRIST,
            )
            left_pose = (left_wrist_pos, self.refs.left.robot_wrist_rot)

        right_pose = None
        right_elbow_pos = None
        if self.enable_right:
            right_wrist_pos, right_elbow_pos = self._arm_targets(
                body_pose,
                ref=self.refs.right,
                shoulder_index=RIGHT_SHOULDER,
                elbow_index=RIGHT_ELBOW,
                wrist_index=RIGHT_WRIST,
            )
            right_pose = (right_wrist_pos, self.refs.right.robot_wrist_rot)

        return self.solver.ik(
            q_current=q_current,
            left_pose=left_pose,
            right_pose=right_pose,
            left_elbow_pos=left_elbow_pos,
            right_elbow_pos=right_elbow_pos,
        )

    def split_for_sim(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert the full IK vector into Piper sim command arrays.

        Each output is shape ``(COMMAND_SIZE,)`` = ``(8,)``: 6 arm joints in
        radians, one unused slot (index 6), then gripper in [0, 1] (index 7).
        """
        left = np.zeros(COMMAND_SIZE, dtype=np.float32)
        right = np.zeros(COMMAND_SIZE, dtype=np.float32)
        left[:ARM_JOINT_COUNT] = q[self.solver.left_indices]
        right[:ARM_JOINT_COUNT] = q[self.solver.right_indices]
        left[GRIPPER_INDEX] = self.gripper
        right[GRIPPER_INDEX] = self.gripper
        return left, right


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
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument(
        "--axis-map",
        default="z,x,y",
        help="PICO delta to Piper delta mapping, e.g. z,x,y or z,y,-x.",
    )
    parser.add_argument("--left-only", action="store_true")
    parser.add_argument("--right-only", action="store_true")
    parser.add_argument("--gripper", type=float, default=1.0)
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
    parser.add_argument("--max-joint-delta", type=float, default=0.35)
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
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    q = retargeter.q_rest.copy()
    left, right = retargeter.split_for_sim(q)
    await sim.motion_control(left=left, right=right)

    q_records: list[np.ndarray] = []
    left_records: list[np.ndarray] = []
    right_records: list[np.ndarray] = []

    frame_delay = 0.0 if playback_fps <= 0 else 1.0 / playback_fps
    next_time = time.perf_counter()

    for frame_number, frame_index in enumerate(frame_indices):
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

    sim = Sim(port=args.port)
    await sim.enable()
    await asyncio.sleep(0.5)

    print(f"Viser simulation: http://localhost:{args.port}")
    print(
        "Replay config: "
        f"frames={len(frame_indices)}, fps={playback_fps:g}, "
        f"scale={args.scale:g}, axis_map={args.axis_map!r}"
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
