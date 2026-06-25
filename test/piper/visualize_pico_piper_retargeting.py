#!/usr/bin/env python3
"""Visualize PICO arm targets against Piper IK/FK in one Viser scene."""

from __future__ import annotations

import argparse
import asyncio
import time

import numpy as np

from dexumi.retargeting.pico_upper_body import (
    LEFT_ELBOW,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ELBOW,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    SMPL24_PARENT_INDICES,
    UPPER_BODY_INDEX,
    UPPER_BODY_JOINTS,
    parse_axis_map,
)
from dexumi.retargeting.piper_from_pico import (
    PicoToPiperArmRetargeter,
    move_retargeter_to_front_workspace,
    piper_link_positions,
    settle_first_frame,
)
from dexumi.robots.piper.config import KinematicsConfig
from dexumi.robots.piper.shared import URDF_PATH, urdf_arm_joint_names
from dexumi.robots.piper.solver import KinematicsSolver
from dexumi.utils.lerobot_io import load_pico_body_poses


ROBOT_TARGET_LINES = np.asarray(
    [
        [0, 1],
        [1, 2],
        [3, 4],
        [4, 5],
    ],
    dtype=np.int32,
)
RAW_PICO_LINES = np.asarray(
    [
        [UPPER_BODY_INDEX[parent], UPPER_BODY_INDEX[child]]
        for child in UPPER_BODY_JOINTS
        for parent in [SMPL24_PARENT_INDICES[child]]
        if parent in UPPER_BODY_INDEX
    ],
    dtype=np.int32,
)


def _rgb(color: tuple[int, int, int], count: int) -> np.ndarray:
    return np.repeat(np.asarray([color], dtype=np.uint8), count, axis=0)


def _line_points(points: np.ndarray, lines: np.ndarray) -> np.ndarray:
    return np.asarray(points[lines], dtype=np.float32)


def _line_colors(color: tuple[int, int, int], count: int) -> np.ndarray:
    return np.repeat(np.asarray([[color, color]], dtype=np.uint8), count, axis=0)


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


def _target_points(
    retargeter: PicoToPiperArmRetargeter,
    body_pose: np.ndarray,
) -> np.ndarray:
    left_wrist, left_elbow = retargeter._arm_targets(
        body_pose,
        ref=retargeter.refs.left,
        shoulder_index=LEFT_SHOULDER,
        elbow_index=LEFT_ELBOW,
        wrist_index=LEFT_WRIST,
    )
    right_wrist, right_elbow = retargeter._arm_targets(
        body_pose,
        ref=retargeter.refs.right,
        shoulder_index=RIGHT_SHOULDER,
        elbow_index=RIGHT_ELBOW,
        wrist_index=RIGHT_WRIST,
    )
    return np.asarray(
        [
            retargeter.solver._left_shoulder_pos,
            left_elbow,
            left_wrist,
            retargeter.solver._right_shoulder_pos,
            right_elbow,
            right_wrist,
        ],
        dtype=np.float32,
    )


def _actual_points(
    solver: KinematicsSolver,
    q: np.ndarray,
) -> np.ndarray:
    left_elbow, left_wrist, right_elbow, right_wrist = piper_link_positions(
        solver,
        q,
        [solver.l_elbow_idx, solver.l_ee_idx, solver.r_elbow_idx, solver.r_ee_idx],
    )
    return np.asarray(
        [
            solver._left_shoulder_pos,
            left_elbow,
            left_wrist,
            solver._right_shoulder_pos,
            right_elbow,
            right_wrist,
        ],
        dtype=np.float32,
    )


def _raw_pico_points(
    body_pose: np.ndarray,
    *,
    transform,
    root: np.ndarray,
    scale: float,
    offset: np.ndarray,
) -> np.ndarray:
    points = np.asarray(body_pose[UPPER_BODY_JOINTS, :3], dtype=np.float32)
    transformed = np.asarray([transform(point - root) for point in points])
    return transformed * float(scale) + offset


def _error_segments(targets: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [targets[1], actual[1]],
            [targets[2], actual[2]],
            [targets[4], actual[4]],
            [targets[5], actual[5]],
        ],
        dtype=np.float32,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Show raw PICO upper-body motion, retargeted Piper targets, and "
            "Piper FK in a single Viser diagnostic scene."
        )
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
    parser.add_argument(
        "--pico-offset",
        type=float,
        nargs=3,
        default=(-0.9, -1.0, 0.35),
        metavar=("X", "Y", "Z"),
        help="Scene offset for the side raw-PICO skeleton.",
    )
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
    parser.add_argument("--port", type=int, default=8004)
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
        raise ValueError("No frames selected for visualization.")

    playback_fps = float(args.fps if args.fps is not None else dataset_fps)
    transform = parse_axis_map(args.axis_map)
    raw_root = np.asarray(poses[frame_indices[0], 0, :3], dtype=np.float32)
    pico_offset = np.asarray(args.pico_offset, dtype=np.float32)

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
        width=3.5,
        height=3.5,
        plane="xy",
        cell_size=0.25,
        section_size=0.5,
    )
    server.scene.add_frame(
        "/world_axes",
        show_axes=True,
        axes_length=0.25,
        axes_radius=0.008,
    )

    urdf = yourdfpy.URDF.load(str(URDF_PATH), mesh_dir=str(URDF_PATH.parent))
    robot_view = ViserUrdf(
        server,
        urdf_or_path=urdf,
        root_node_name="/robot",
        load_meshes=True,
        load_collision_meshes=False,
    )
    viser_to_robot = _robot_to_viser_order(robot_view.get_actuated_joint_names())

    body_pose = poses[frame_indices[0]]
    q = initial_q.copy()
    targets = _target_points(retargeter, body_pose)
    actual = _actual_points(solver, q)
    raw_pico = _raw_pico_points(
        body_pose,
        transform=transform,
        root=raw_root,
        scale=args.scale,
        offset=pico_offset,
    )

    q_robot = _split_q_for_robot_order(solver, q)
    robot_view.update_cfg(_to_viser_q(q_robot, viser_to_robot))

    raw_points_handle = server.scene.add_point_cloud(
        "/pico_raw/points",
        points=raw_pico,
        colors=_rgb((220, 40, 80), len(raw_pico)),
        point_size=0.035,
        point_shape="circle",
    )
    raw_lines_handle = server.scene.add_line_segments(
        "/pico_raw/bones",
        points=_line_points(raw_pico, RAW_PICO_LINES),
        colors=_line_colors((245, 245, 245), len(RAW_PICO_LINES)),
        line_width=2.5,
    )
    target_points_handle = server.scene.add_point_cloud(
        "/targets/points",
        points=targets,
        colors=np.asarray(
            [
                [40, 120, 255],
                [255, 175, 30],
                [255, 230, 50],
                [40, 120, 255],
                [255, 175, 30],
                [255, 230, 50],
            ],
            dtype=np.uint8,
        ),
        point_size=0.035,
        point_shape="circle",
    )
    target_lines_handle = server.scene.add_line_segments(
        "/targets/bones",
        points=_line_points(targets, ROBOT_TARGET_LINES),
        colors=_line_colors((255, 210, 40), len(ROBOT_TARGET_LINES)),
        line_width=4.0,
    )
    actual_points_handle = server.scene.add_point_cloud(
        "/piper_fk/points",
        points=actual,
        colors=np.asarray(
            [
                [20, 170, 220],
                [20, 220, 190],
                [20, 255, 120],
                [20, 170, 220],
                [20, 220, 190],
                [20, 255, 120],
            ],
            dtype=np.uint8,
        ),
        point_size=0.025,
        point_shape="circle",
    )
    actual_lines_handle = server.scene.add_line_segments(
        "/piper_fk/links",
        points=_line_points(actual, ROBOT_TARGET_LINES),
        colors=_line_colors((20, 220, 190), len(ROBOT_TARGET_LINES)),
        line_width=3.0,
    )
    error_handle = server.scene.add_line_segments(
        "/errors/target_to_fk",
        points=_error_segments(targets, actual),
        colors=_line_colors((255, 40, 220), 4),
        line_width=2.0,
    )

    status_label = server.scene.add_label(
        "/status",
        "",
        position=(-1.45, -1.45, 1.0),
        font_size_mode="screen",
        font_screen_scale=0.8,
    )
    server.scene.add_label(
        "/legend/pico",
        "PICO raw: red/white",
        position=tuple(pico_offset + np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        font_size_mode="screen",
        font_screen_scale=0.75,
        anchor="center-center",
    )
    server.scene.add_label(
        "/legend/targets",
        "Targets: yellow/orange | FK: cyan/green | Error: magenta",
        position=(-0.35, 0.0, 1.05),
        font_size_mode="screen",
        font_screen_scale=0.75,
        anchor="center-center",
    )

    print(f"Viser diagnostic: http://localhost:{server.get_port()}")
    print(
        "Replay config: "
        f"frames={len(frame_indices)}, fps={playback_fps:g}, scale={args.scale:g}, "
        f"axis_map={args.axis_map!r}, workspace={args.piper_workspace!r}, "
        f"ori_weight={args.ori_weight:g}"
    )
    print("Colors: PICO raw red/white, IK targets yellow/orange, Piper FK cyan/green.")

    frame_delay = 0.0 if playback_fps <= 0 else 1.0 / playback_fps

    try:
        while True:
            next_time = time.perf_counter()
            q = initial_q.copy()
            for frame_number, frame_index in enumerate(frame_indices):
                body_pose = poses[frame_index]
                if frame_number > 0:
                    q = retargeter.retarget_frame(body_pose, q)

                q_robot = _split_q_for_robot_order(solver, q)
                robot_view.update_cfg(_to_viser_q(q_robot, viser_to_robot))

                targets = _target_points(retargeter, body_pose)
                actual = _actual_points(solver, q)
                raw_pico = _raw_pico_points(
                    body_pose,
                    transform=transform,
                    root=raw_root,
                    scale=args.scale,
                    offset=pico_offset,
                )

                raw_points_handle.points = raw_pico
                raw_lines_handle.points = _line_points(raw_pico, RAW_PICO_LINES)
                target_points_handle.points = targets
                target_lines_handle.points = _line_points(targets, ROBOT_TARGET_LINES)
                actual_points_handle.points = actual
                actual_lines_handle.points = _line_points(actual, ROBOT_TARGET_LINES)
                error_handle.points = _error_segments(targets, actual)

                left_err = float(np.linalg.norm(targets[2] - actual[2]))
                right_err = float(np.linalg.norm(targets[5] - actual[5]))
                status_label.text = (
                    f"frame {frame_index}/{len(poses) - 1} | "
                    f"L wrist err {left_err:.3f} m | R wrist err {right_err:.3f} m"
                )

                if frame_number % 30 == 0:
                    print(
                        f"frame {frame_index}/{len(poses) - 1} "
                        f"Lerr={left_err:.3f} Rerr={right_err:.3f}"
                    )

                next_time += frame_delay
                if frame_delay > 0:
                    await asyncio.sleep(max(0.0, next_time - time.perf_counter()))

            if not args.loop:
                break

        if args.hold_after is None:
            print("Holding diagnostic viewer. Press Ctrl+C to stop.")
            while True:
                await asyncio.sleep(3600.0)
        if args.hold_after > 0:
            print(f"Holding diagnostic viewer for {args.hold_after:g}s...")
            await asyncio.sleep(args.hold_after)
    finally:
        server.stop()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
