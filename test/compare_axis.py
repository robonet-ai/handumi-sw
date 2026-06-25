#!/usr/bin/env python3
"""Compare multiple PICO axis mappings for a selected robot embodiment.

Examples:
    python test/compare_axis.py --embodiment piper --episode 0
    python test/compare_axis.py --embodiment axol --episode 0 --workspace front
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from replay_pico_ik import (  # noqa: E402
    DEFAULT_REPO_ID,
    _robot_to_viser_order,
    _split_q_for_robot_order,
    _to_viser_q,
)
from dexumi.dataset import dataset_root_from_repo_id, load_pico_body_poses
from dexumi.robots.registry import load_embodiment


def _parse_axis_maps(value: str | None, defaults: tuple[str, ...]) -> list[str]:
    if value is None or value.strip() == "":
        return list(defaults)
    axis_maps = [item.strip() for item in value.split(";") if item.strip()]
    if not axis_maps:
        raise ValueError("--axis-maps did not contain any mappings.")
    return axis_maps


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
        description="Compare PICO axis-map candidates for piper or axol in a Viser grid."
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
    parser.add_argument(
        "--axis-maps",
        default=None,
        help="Semicolon-separated mappings. Default uses embodiment-specific finalists.",
    )
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--spacing", type=float, default=1.35)
    parser.add_argument("--left-only", action="store_true")
    parser.add_argument("--right-only", action="store_true")
    parser.add_argument("--gripper", type=float, default=1.0)
    parser.add_argument("--workspace", choices=("rest", "front"), default=None)
    parser.add_argument("--wrist-forward", type=float, default=None)
    parser.add_argument("--wrist-height", type=float, default=None)
    parser.add_argument("--wrist-lateral", type=float, default=None)
    parser.add_argument("--settle-iterations", type=int, default=20)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--hold-after", type=float, default=20.0)
    parser.add_argument("--pos-weight", type=float, default=50.0)
    parser.add_argument("--ori-weight", type=float, default=0.0)
    parser.add_argument("--manipulability-weight", type=float, default=0.0)
    parser.add_argument("--max-joint-delta", type=float, default=None)
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

    runtime = load_embodiment(args.embodiment)
    workspace = args.workspace or runtime.default_workspace
    axis_maps = _parse_axis_maps(args.axis_maps, runtime.default_compare_axis_maps)
    if args.columns < 1:
        raise ValueError("--columns must be >= 1.")
    if args.spacing <= 0:
        raise ValueError("--spacing must be > 0.")

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

    playback_fps = float(args.fps if args.fps is not None else dataset_fps)

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
    retargeters = [
        runtime.retargeter_cls(
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
    if workspace == "front":
        for retargeter in retargeters:
            runtime.move_to_front_workspace(
                retargeter,
                wrist_forward=args.wrist_forward or runtime.wrist_forward,
                wrist_height=args.wrist_height or runtime.wrist_height,
                wrist_lateral=args.wrist_lateral or runtime.wrist_lateral,
            )

    initial_q = runtime.settle_first_frame(
        retargeters[0],
        poses[frame_indices[0]],
        0 if workspace == "rest" else args.settle_iterations,
    )
    if workspace == "front":
        solver.set_posture_pose(initial_q)
    q_states = [initial_q.copy() for _ in retargeters]

    try:
        import viser
        import yourdfpy
        from viser.extras import ViserUrdf
    except ImportError as exc:
        raise RuntimeError(
            "Viser dependencies are missing. Install with: "
            "GIT_LFS_SKIP_SMUDGE=1 uv sync --extra lerobot --extra axol"
        ) from exc

    port = args.port or runtime.default_port
    server = viser.ViserServer(port=port)
    server.scene.add_grid(
        "/grid",
        width=5.0,
        height=3.5,
        position=(0.0, -0.65, 0.0),
    )

    urdf = yourdfpy.URDF.load(
        str(runtime.urdf_path), mesh_dir=str(runtime.urdf_path.parent)
    )
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
                robot_view.get_actuated_joint_names(),
                runtime.urdf_arm_joint_names,
            )
        robot_views.append(robot_view)

    if viser_to_robot is None:
        raise RuntimeError("Could not build Viser joint mapping.")

    for q_state, robot_view in zip(q_states, robot_views, strict=True):
        robot_view.update_cfg(
            _to_viser_q(_split_q_for_robot_order(solver, q_state), viser_to_robot)
        )

    print(f"Viser comparison: http://localhost:{server.get_port()}")
    print("Axis-map candidates:")
    for index, axis_map in enumerate(axis_maps, start=1):
        print(f"  [{index}] {axis_map}")
    print(
        "Replay config: "
        f"embodiment={runtime.name}, frames={len(frame_indices)}, fps={playback_fps:g}, "
        f"scale={args.scale:g}, candidates={len(axis_maps)}, workspace={workspace!r}"
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
                    robot_views[i].update_cfg(
                        _to_viser_q(
                            _split_q_for_robot_order(solver, q_states[i]),
                            viser_to_robot,
                        )
                    )

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
