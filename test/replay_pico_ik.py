#!/usr/bin/env python3
"""Replay one PICO dataset episode on a selected robot embodiment.

Examples:
    python test/replay_pico_ik.py --embodiment piper --episode 0
    python test/replay_pico_ik.py --embodiment axol --episode 0 --workspace front
    python test/replay_pico_ik.py --embodiment piper --episode 0 --visualize
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from dexumi.retargeting.pico_to_robot import robot_link_positions
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
from dexumi.dataset import dataset_root_from_repo_id, load_pico_body_poses
from dexumi.robots.registry import EmbodimentRuntime, load_embodiment

DEFAULT_REPO_ID = "NONHUMAN-RESEARCH/dexumi-dataset-v2"

ROBOT_TARGET_LINES = np.asarray(
    [[0, 1], [1, 2], [3, 4], [4, 5]],
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay PICO arm motion with the shared EE-only pyroki IK solver."
    )
    parser.add_argument("--embodiment", choices=("piper", "axol"), default="piper")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Local dataset root. Defaults to outputs/datasets/<repo-id suffix>.",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--column", default="observation.pico.body_joints_pose")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--axis-map", default=None)
    parser.add_argument("--left-only", action="store_true")
    parser.add_argument("--right-only", action="store_true")
    parser.add_argument("--gripper", type=float, default=1.0)
    parser.add_argument("--workspace", choices=("rest", "front"), default=None)
    parser.add_argument("--wrist-forward", type=float, default=None)
    parser.add_argument("--wrist-height", type=float, default=None)
    parser.add_argument("--wrist-lateral", type=float, default=None)
    parser.add_argument("--settle-iterations", type=int, default=20)
    parser.add_argument("--port", type=int, default=None)
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
    parser.add_argument("--manipulability-weight", type=float, default=0.0)
    parser.add_argument("--max-joint-delta", type=float, default=None)
    parser.add_argument("--max-reach", type=float, default=0.8)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help=(
            "Open a diagnostic Viser scene with raw PICO upper-body motion, "
            "retargeted targets, and robot FK."
        ),
    )
    parser.add_argument(
        "--pico-offset",
        type=float,
        nargs=3,
        default=(-0.9, -1.0, 0.35),
        metavar=("X", "Y", "Z"),
        help="Scene offset for the side raw-PICO skeleton (only with --visualize).",
    )
    return parser


def _rgb(color: tuple[int, int, int], count: int) -> np.ndarray:
    return np.repeat(np.asarray([color], dtype=np.uint8), count, axis=0)


def _line_points(points: np.ndarray, lines: np.ndarray) -> np.ndarray:
    return np.asarray(points[lines], dtype=np.float32)


def _line_colors(color: tuple[int, int, int], count: int) -> np.ndarray:
    return np.repeat(np.asarray([[color, color]], dtype=np.uint8), count, axis=0)


def _robot_to_viser_order(
    viser_order: list[str],
    urdf_arm_joint_names: Callable[..., list[str]],
) -> list[int]:
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


def _split_q_for_robot_order(solver, q_solver: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            q_solver[solver.left_joint_indices],
            q_solver[solver.right_joint_indices],
        ]
    ).astype(float)


def _target_points(retargeter, body_pose: np.ndarray) -> np.ndarray:
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


def _actual_points(solver, q: np.ndarray) -> np.ndarray:
    left_elbow, left_wrist, right_elbow, right_wrist = robot_link_positions(
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


async def run_diagnostic_async(
    *,
    args: argparse.Namespace,
    runtime: EmbodimentRuntime,
    poses: np.ndarray,
    frame_indices: list[int],
    playback_fps: float,
    axis_map: str,
    workspace: str,
    retargeter,
    solver,
    initial_q: np.ndarray,
) -> None:
    try:
        import viser
        import yourdfpy
        from viser.extras import ViserUrdf
    except ImportError as exc:
        raise RuntimeError(
            "Viser dependencies are missing. Install with: "
            "GIT_LFS_SKIP_SMUDGE=1 uv sync --extra lerobot --extra axol"
        ) from exc

    transform = parse_axis_map(axis_map)
    raw_root = np.asarray(poses[frame_indices[0], 0, :3], dtype=np.float32)
    pico_offset = np.asarray(args.pico_offset, dtype=np.float32)
    port = args.port or runtime.default_port

    server = viser.ViserServer(port=port)
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

    urdf = yourdfpy.URDF.load(
        str(runtime.urdf_path), mesh_dir=str(runtime.urdf_path.parent)
    )
    robot_view = ViserUrdf(
        server,
        urdf_or_path=urdf,
        root_node_name="/robot",
        load_meshes=True,
        load_collision_meshes=False,
    )
    viser_to_robot = _robot_to_viser_order(
        robot_view.get_actuated_joint_names(),
        runtime.urdf_arm_joint_names,
    )

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

    robot_view.update_cfg(_to_viser_q(_split_q_for_robot_order(solver, q), viser_to_robot))

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
        "/robot_fk/points",
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
        "/robot_fk/links",
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
        f"embodiment={runtime.name}, frames={len(frame_indices)}, fps={playback_fps:g}, "
        f"scale={args.scale:g}, axis_map={axis_map!r}, workspace={workspace!r}, "
        "solver=pyroki-ee"
    )
    print(
        "Colors: PICO raw red/white, IK targets yellow/orange, "
        f"{runtime.name} FK cyan/green."
    )

    frame_delay = 0.0 if playback_fps <= 0 else 1.0 / playback_fps

    try:
        while True:
            next_time = time.perf_counter()
            q = initial_q.copy()
            for frame_number, frame_index in enumerate(frame_indices):
                body_pose = poses[frame_index]
                if frame_number > 0:
                    q = retargeter.retarget_frame(body_pose, q)

                robot_view.update_cfg(
                    _to_viser_q(_split_q_for_robot_order(solver, q), viser_to_robot)
                )

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
        elif args.hold_after > 0:
            print(f"Holding diagnostic viewer for {args.hold_after:g}s...")
            await asyncio.sleep(args.hold_after)
    finally:
        server.stop()


async def replay_once(
    *,
    sim,
    retargeter,
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
    loop = asyncio.get_running_loop()
    next_time = loop.time()

    for frame_number, frame_index in enumerate(frame_indices):
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
            await asyncio.sleep(max(0.0, next_time - loop.time()))

    return q_records, left_records, right_records


async def main_async() -> None:
    args = build_parser().parse_args()
    if args.left_only and args.right_only:
        raise ValueError("Use only one of --left-only or --right-only.")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1.")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be >= 0.")

    runtime = load_embodiment(args.embodiment)
    axis_map = args.axis_map or runtime.default_axis_map
    workspace = args.workspace or runtime.default_workspace
    dataset_root = (
        Path(args.dataset_root)
        if args.dataset_root is not None
        else dataset_root_from_repo_id(args.repo_id)
    )

    poses, dataset_fps = load_pico_body_poses(
        repo_id=args.repo_id,
        root=dataset_root,
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

    config_kwargs = dict(
        pos_weight=args.pos_weight,
        ori_weight=args.ori_weight,
        manipulability_weight=args.manipulability_weight,
        max_reach=args.max_reach,
    )
    if args.max_joint_delta is not None:
        config_kwargs["max_joint_delta"] = args.max_joint_delta
    config = runtime.config_cls(**config_kwargs)
    solver = runtime.solver_cls(config=config)
    retargeter = runtime.retargeter_cls(
        solver=solver,
        first_body_pose=poses[frame_indices[0]],
        scale=args.scale,
        axis_map=axis_map,
        enable_left=not args.right_only,
        enable_right=not args.left_only,
        gripper=args.gripper,
    )

    if workspace == "front":
        runtime.move_to_front_workspace(
            retargeter,
            wrist_forward=args.wrist_forward or runtime.wrist_forward,
            wrist_height=args.wrist_height or runtime.wrist_height,
            wrist_lateral=args.wrist_lateral or runtime.wrist_lateral,
        )

    initial_q = runtime.settle_first_frame(
        retargeter,
        poses[frame_indices[0]],
        0 if workspace == "rest" else args.settle_iterations,
    )
    if workspace == "front":
        solver.set_posture_pose(initial_q)

    playback_fps = float(args.fps if args.fps is not None else dataset_fps)

    if args.visualize:
        await run_diagnostic_async(
            args=args,
            runtime=runtime,
            poses=poses,
            frame_indices=frame_indices,
            playback_fps=playback_fps,
            axis_map=axis_map,
            workspace=workspace,
            retargeter=retargeter,
            solver=solver,
            initial_q=initial_q,
        )
        return

    sim = runtime.make_sim(port=args.port or runtime.default_port)
    await sim.enable()
    await asyncio.sleep(0.5)

    print(f"Viser simulation: http://localhost:{args.port or runtime.default_port}")
    print(
        "Replay config: "
        f"embodiment={runtime.name}, frames={len(frame_indices)}, fps={playback_fps:g}, "
        f"scale={args.scale:g}, axis_map={axis_map!r}, workspace={workspace!r}, "
        "solver=pyroki-ee"
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
            axis_map=np.asarray(axis_map),
            embodiment=np.asarray(runtime.name),
            workspace=np.asarray(workspace),
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
