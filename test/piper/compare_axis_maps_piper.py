#!/usr/bin/env python3
"""Compare multiple PICO-to-Piper axis mappings in one Viser scene."""

from __future__ import annotations

import argparse
import asyncio
import time

import numpy as np

from dexumi.retargeting.piper_from_pico import (
    PicoToPiperArmRetargeter,
    move_retargeter_to_front_workspace,
    settle_first_frame,
)
from dexumi.robots.piper.config import KinematicsConfig
from dexumi.robots.piper.shared import URDF_PATH, urdf_arm_joint_names
from dexumi.robots.piper.solver import KinematicsSolver
from dexumi.utils.lerobot_io import load_pico_body_poses

DEFAULT_AXIS_MAPS = (
    "x,z,y",
    "x,z,-y",
    "x,-z,y",
    "x,-z,-y",
    "-x,z,y",
    "-x,z,-y",
    "-x,-z,y",
    "-x,-z,-y",
)


def _parse_axis_maps(value: str | None) -> list[str]:
    if value is None or value.strip() == "":
        return list(DEFAULT_AXIS_MAPS)
    axis_maps = [item.strip() for item in value.split(";") if item.strip()]
    if not axis_maps:
        raise ValueError("--axis-maps did not contain any mappings.")
    return axis_maps


def _robot_to_viser_order(viser_order: list[str]) -> list[int]:
    robot_order = urdf_arm_joint_names(is_left=True) + urdf_arm_joint_names(
        is_left=False
    )
    mapping: list[int] = []
    for name in viser_order:
        try:
            mapping.append(robot_order.index(name))
        except ValueError:
            mapping.append(-1)
    return mapping


def _to_viser_q(q_robot: np.ndarray, viser_to_robot: list[int]) -> np.ndarray:
    q_out = np.zeros(len(viser_to_robot), dtype=float)
    for viser_index, robot_index in enumerate(viser_to_robot):
        if robot_index >= 0:
            q_out[viser_index] = q_robot[robot_index]
    return q_out


def _split_q_for_robot_order(
    solver: KinematicsSolver,
    q_solver: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            q_solver[solver.left_joint_indices],
            q_solver[solver.right_joint_indices],
        ]
    ).astype(float)


def _layout_position(index: int, *, columns: int, spacing: float) -> np.ndarray:
    row = index // columns
    col = index % columns
    return np.array(
        [
            (col - (columns - 1) / 2.0) * spacing,
            -row * spacing,
            0.0,
        ],
        dtype=np.float32,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare PICO-to-Piper axis-map candidates in a Viser grid."
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
        "--axis-maps",
        default=None,
        help=(
            "Semicolon-separated mappings. Default compares the selected "
            "finalist mappings."
        ),
    )
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--spacing", type=float, default=1.35)
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
    parser.add_argument("--settle-iterations", type=int, default=20)
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--hold-after", type=float, default=20.0)

    parser.add_argument("--pos-weight", type=float, default=50.0)
    parser.add_argument("--ori-weight", type=float, default=0.0)
    parser.add_argument("--elbow-weight", type=float, default=5.0)
    parser.add_argument("--manipulability-weight", type=float, default=0.0)
    parser.add_argument(
        "--max-joint-delta",
        type=float,
        default=KinematicsConfig().max_joint_delta,
    )
    parser.add_argument("--max-reach", type=float, default=0.8)
    return parser


async def main_async() -> None:
    args = build_parser().parse_args()

    if args.left_only and args.right_only:
        raise ValueError("Use only one of --left-only or --right-only.")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1.")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be >= 0.")

    axis_maps = _parse_axis_maps(args.axis_maps)
    if args.columns < 1:
        raise ValueError("--columns must be >= 1.")
    if args.spacing <= 0:
        raise ValueError("--spacing must be > 0.")

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
    retargeters = [
        PicoToPiperArmRetargeter(
            solver=solver,
            first_body_pose=poses[frame_indices[0]],
            scale=args.scale,
            axis_map=axis_map,
            enable_left=not args.right_only,
            enable_right=not args.left_only,
            gripper=args.gripper,
        )
        for axis_map in axis_maps
    ]
    if args.piper_workspace == "front":
        for retargeter in retargeters:
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
        retargeters[0],
        poses[frame_indices[0]],
        0 if args.piper_workspace == "rest" else args.settle_iterations,
    )
    if args.piper_workspace == "front":
        solver.set_posture_pose(initial_q)
    q_states = [initial_q.copy() for _ in retargeters]

    try:
        import viser
        import yourdfpy
        from viser.extras import ViserUrdf
    except ImportError as exc:
        raise RuntimeError(
            "Viser Piper dependencies are missing. Install with: "
            "GIT_LFS_SKIP_SMUDGE=1 uv sync --extra lerobot --extra axol"
        ) from exc

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid(
        "/grid",
        width=5.0,
        height=3.5,
        position=(0.0, -0.65, 0.0),
    )

    urdf = yourdfpy.URDF.load(str(URDF_PATH), mesh_dir=str(URDF_PATH.parent))
    robot_views = []
    viser_to_robot: list[int] | None = None
    for index, axis_map in enumerate(axis_maps):
        root_name = f"/candidate_{index + 1}"
        root_pos = _layout_position(index, columns=args.columns, spacing=args.spacing)
        server.scene.add_frame(
            root_name,
            show_axes=True,
            axes_length=0.18,
            axes_radius=0.006,
            position=root_pos,
        )
        server.scene.add_label(
            f"{root_name}/label",
            f"[{index + 1}] {axis_map}",
            position=(0.0, 0.0, 1.05),
            font_size_mode="screen",
            font_screen_scale=0.75,
            anchor="center-center",
        )
        robot_view = ViserUrdf(
            server,
            urdf_or_path=urdf,
            root_node_name=root_name,
            load_meshes=True,
            load_collision_meshes=False,
        )
        if viser_to_robot is None:
            viser_to_robot = _robot_to_viser_order(
                robot_view.get_actuated_joint_names()
            )
        robot_views.append(robot_view)

    if viser_to_robot is None:
        raise RuntimeError("Could not build Viser joint mapping.")

    for q_state, robot_view in zip(q_states, robot_views, strict=True):
        q_robot = _split_q_for_robot_order(solver, q_state)
        robot_view.update_cfg(_to_viser_q(q_robot, viser_to_robot))

    print(f"Viser comparison: http://localhost:{server.get_port()}")
    print("Axis-map candidates:")
    for index, axis_map in enumerate(axis_maps, start=1):
        print(f"  [{index}] {axis_map}")
    print(
        "Replay config: "
        f"frames={len(frame_indices)}, fps={playback_fps:g}, scale={args.scale:g}, "
        f"candidates={len(axis_maps)}, workspace={args.piper_workspace!r}"
    )

    frame_delay = 0.0 if playback_fps <= 0 else 1.0 / playback_fps

    try:
        while True:
            next_time = time.perf_counter()
            for frame_number, frame_index in enumerate(frame_indices):
                if frame_number > 0:
                    for i, retargeter in enumerate(retargeters):
                        q_states[i] = retargeter.retarget_frame(
                            poses[frame_index],
                            q_states[i],
                        )

                for i in range(len(retargeters)):
                    q_robot = _split_q_for_robot_order(solver, q_states[i])
                    robot_views[i].update_cfg(_to_viser_q(q_robot, viser_to_robot))

                if frame_number % 30 == 0:
                    print(f"frame {frame_index}/{len(poses) - 1}")

                next_time += frame_delay
                if frame_delay > 0:
                    await asyncio.sleep(max(0.0, next_time - time.perf_counter()))

            if not args.loop:
                break

        if args.hold_after > 0:
            print(f"Holding comparison for {args.hold_after:g}s...")
            await asyncio.sleep(args.hold_after)
    finally:
        server.stop()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
