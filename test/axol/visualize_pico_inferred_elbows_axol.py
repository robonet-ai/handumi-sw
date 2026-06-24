#!/usr/bin/env python3
"""Visualize PICO upper body, inferred elbows, and Axol IK side by side.

This tool is for validating the controller-only direction: PICO elbow joints
may exist in old whole-body recordings, but Axol IK is driven with elbows
inferred from shoulder/wrist geometry.  Recorded elbows are used only for the
offline error number shown in the terminal/viewer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from dexumi.robots.axol.config import KinematicsConfig
from dexumi.robots.axol.solver import KinematicsSolver
from dexumi.robots.utils import Joint, urdf_body_name
from dexumi.retargeting.axol_from_pico import (
    PicoToAxolArmRetargeter,
    axol_link_positions,
    move_retargeter_to_front_workspace,
    settle_first_frame,
)
from dexumi.retargeting.pico_upper_body import (
    LEFT_ELBOW,
    LEFT_HAND,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ELBOW,
    RIGHT_HAND,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    UPPER_BODY_INDEX,
    UPPER_BODY_JOINTS,
    estimate_arm_lengths,
    infer_pose_elbows,
    parse_axis_map,
    upper_body_lines,
)


AXOL_LINK_ORDER = [
    "base",
    "s1",
    urdf_body_name(Joint.SHOULDER_1, is_left=True),
    urdf_body_name(Joint.SHOULDER_2, is_left=True),
    urdf_body_name(Joint.SHOULDER_3, is_left=True),
    urdf_body_name(Joint.ELBOW, is_left=True),
    urdf_body_name(Joint.WRIST_1, is_left=True),
    urdf_body_name(Joint.WRIST_2, is_left=True),
    urdf_body_name(Joint.WRIST_3, is_left=True),
    urdf_body_name(Joint.GRIPPER, is_left=True),
    urdf_body_name(Joint.SHOULDER_1, is_left=False),
    urdf_body_name(Joint.SHOULDER_2, is_left=False),
    urdf_body_name(Joint.SHOULDER_3, is_left=False),
    urdf_body_name(Joint.ELBOW, is_left=False),
    urdf_body_name(Joint.WRIST_1, is_left=False),
    urdf_body_name(Joint.WRIST_2, is_left=False),
    urdf_body_name(Joint.WRIST_3, is_left=False),
    urdf_body_name(Joint.GRIPPER, is_left=False),
]
AXOL_LEFT_GRIPPER_INDEX = AXOL_LINK_ORDER.index(urdf_body_name(Joint.GRIPPER, is_left=True))
AXOL_RIGHT_GRIPPER_INDEX = AXOL_LINK_ORDER.index(
    urdf_body_name(Joint.GRIPPER, is_left=False)
)
AXOL_LINES = np.asarray(
    [
        2,
        0,
        1,
        2,
        1,
        2,
        *sum(([2, i, i + 1] for i in range(2, 9)), []),
        2,
        1,
        10,
        *sum(([2, i, i + 1] for i in range(10, 17)), []),
    ],
    dtype=np.int_,
)


def load_pico_body_poses(
    root: Path,
    *,
    episode: int,
    column: str,
) -> tuple[np.ndarray, float]:
    """Load PICO body poses from a local LeRobot dataset root."""

    parquet_files = sorted((root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")

    df = pd.concat([pd.read_parquet(path) for path in parquet_files], ignore_index=True)
    df.sort_values("index", inplace=True)
    df = df[df["episode_index"] == episode]
    if df.empty:
        raise ValueError(f"Episode {episode} not found in {root}")
    if column not in df.columns:
        raise KeyError(f"Column {column!r} not found. Available: {list(df.columns)}")

    poses = np.stack(
        [np.stack(frame).astype(np.float32) for frame in df[column].values]
    )
    if poses.ndim != 3 or poses.shape[1] < 24 or poses.shape[2] < 3:
        raise ValueError(f"{column!r} must have shape (N, >=24, >=3); got {poses.shape}")

    fps = 30.0
    info_path = root / "meta" / "info.json"
    if info_path.is_file():
        with info_path.open("r", encoding="utf-8") as f:
            fps = float(json.load(f).get("fps", fps))
    return poses, fps


def display_positions(
    positions: np.ndarray,
    *,
    first_root: np.ndarray,
    axis_map: str,
    scale: float,
) -> np.ndarray:
    """Center and orient PICO coordinates for side-by-side viewing."""

    transform = parse_axis_map(axis_map)
    centered = np.asarray(positions, dtype=np.float32) - first_root
    return np.asarray([transform(point) for point in centered], dtype=np.float32) * scale


def polyline(points: np.ndarray) -> "pv.PolyData":
    import pyvista as pv

    if len(points) < 2:
        return pv.PolyData(points)
    poly = pv.PolyData(points)
    poly.lines = np.concatenate([[len(points)], np.arange(len(points))]).astype(np.int_)
    return poly


def update_polyline(poly: "pv.PolyData", points: np.ndarray) -> None:
    if len(points) < 2:
        poly.points = points if len(points) else np.zeros((1, 3), dtype=np.float32)
        poly.lines = np.empty(0, dtype=np.int_)
        return
    poly.points = points
    poly.lines = np.concatenate([[len(points)], np.arange(len(points))]).astype(np.int_)


def trail_window(values: np.ndarray, end_index: int, trail_length: int) -> np.ndarray:
    if trail_length <= 0:
        return values[: end_index + 1]
    start = max(0, end_index + 1 - trail_length)
    return values[start : end_index + 1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("outputs/datasets/dexumi-dataset-v2"),
        help="Local LeRobot dataset root.",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--column", default="observation.pico.body_joints_pose")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--axis-map", default="z,x,y")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--spacing", type=float, default=1.25)
    parser.add_argument("--trail-length", type=int, default=180)
    parser.add_argument("--upper-ratio", type=float, default=0.44)
    parser.add_argument("--extension-ratio", type=float, default=0.92)
    parser.add_argument("--length-percentile", type=float, default=95.0)
    parser.add_argument("--bend-forward", type=float, default=0.65)
    parser.add_argument("--bend-down", type=float, default=-1.0)
    parser.add_argument("--bend-side", type=float, default=0.25)
    parser.add_argument("--axol-scale", type=float, default=1.0)
    parser.add_argument("--axol-wrist-forward", type=float, default=-0.34)
    parser.add_argument("--axol-wrist-height", type=float, default=0.58)
    parser.add_argument("--axol-wrist-lateral", type=float, default=0.23)
    parser.add_argument("--axol-elbow-forward", type=float, default=-0.16)
    parser.add_argument("--axol-elbow-height", type=float, default=0.68)
    parser.add_argument("--axol-elbow-lateral", type=float, default=0.20)
    parser.add_argument("--settle-iterations", type=int, default=20)
    parser.add_argument("--pos-weight", type=float, default=50.0)
    parser.add_argument("--ori-weight", type=float, default=0.0)
    parser.add_argument("--elbow-weight", type=float, default=5.0)
    parser.add_argument("--max-joint-delta", type=float, default=0.25)
    parser.add_argument("--max-reach", type=float, default=0.8)
    parser.add_argument("--background", default="black")
    parser.add_argument("--point-size", type=float, default=15.0)
    parser.add_argument("--line-width", type=float, default=4.0)
    parser.add_argument("--save", type=Path, default=None, help="Optional .npz output.")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    poses, dataset_fps = load_pico_body_poses(
        args.dataset_root,
        episode=args.episode,
        column=args.column,
    )
    if args.start_frame < 0 or args.start_frame >= len(poses):
        raise ValueError(f"--start-frame must be inside [0, {len(poses)})")

    frame_indices = list(range(args.start_frame, len(poses), args.stride))
    if args.max_frames is not None:
        frame_indices = frame_indices[: args.max_frames]
    if not frame_indices:
        raise ValueError("No frames selected.")

    left_lengths = estimate_arm_lengths(
        poses,
        shoulder_index=LEFT_SHOULDER,
        wrist_index=LEFT_WRIST,
        upper_ratio=args.upper_ratio,
        extension_ratio=args.extension_ratio,
        percentile=args.length_percentile,
    )
    right_lengths = estimate_arm_lengths(
        poses,
        shoulder_index=RIGHT_SHOULDER,
        wrist_index=RIGHT_WRIST,
        upper_ratio=args.upper_ratio,
        extension_ratio=args.extension_ratio,
        percentile=args.length_percentile,
    )

    config = KinematicsConfig(
        pos_weight=args.pos_weight,
        ori_weight=args.ori_weight,
        elbow_weight=args.elbow_weight,
        max_joint_delta=args.max_joint_delta,
        max_reach=args.max_reach,
    )
    solver = KinematicsSolver(config=config)

    first_inferred = infer_pose_elbows(
        poses[frame_indices[0]],
        left_lengths=left_lengths,
        right_lengths=right_lengths,
        bend_forward=args.bend_forward,
        bend_down=args.bend_down,
        bend_side=args.bend_side,
    )
    retargeter = PicoToAxolArmRetargeter(
        solver=solver,
        first_body_pose=first_inferred,
        scale=args.axol_scale,
        axis_map=args.axis_map,
    )
    move_retargeter_to_front_workspace(
        retargeter,
        wrist_forward=args.axol_wrist_forward,
        wrist_height=args.axol_wrist_height,
        wrist_lateral=args.axol_wrist_lateral,
        elbow_forward=args.axol_elbow_forward,
        elbow_height=args.axol_elbow_height,
        elbow_lateral=args.axol_elbow_lateral,
    )
    q = settle_first_frame(retargeter, first_inferred, args.settle_iterations)
    solver.set_posture_pose(q)

    robot_link_names = list(solver.robot.links.names)
    axol_link_indices = [robot_link_names.index(name) for name in AXOL_LINK_ORDER]

    first_root = np.asarray(poses[frame_indices[0], 0, :3], dtype=np.float32)
    actual_offset = np.array([-args.spacing, 0.0, 0.0], dtype=np.float32)
    inferred_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    axol_offset = np.array([args.spacing, 0.0, 0.0], dtype=np.float32)

    actual_frames: list[np.ndarray] = []
    inferred_frames: list[np.ndarray] = []
    axol_frames: list[np.ndarray] = []
    q_frames: list[np.ndarray] = []
    elbow_errors: list[float] = []

    for frame_index in frame_indices:
        raw_actual = np.asarray(poses[frame_index, :, :3], dtype=np.float32)
        raw_inferred = infer_pose_elbows(
            poses[frame_index],
            left_lengths=left_lengths,
            right_lengths=right_lengths,
            bend_forward=args.bend_forward,
            bend_down=args.bend_down,
            bend_side=args.bend_side,
        )
        elbow_errors.append(
            float(
                np.mean(
                    [
                        np.linalg.norm(raw_actual[LEFT_ELBOW] - raw_inferred[LEFT_ELBOW]),
                        np.linalg.norm(raw_actual[RIGHT_ELBOW] - raw_inferred[RIGHT_ELBOW]),
                    ]
                )
            )
        )
        actual_frames.append(
            display_positions(
                raw_actual[UPPER_BODY_JOINTS],
                first_root=first_root,
                axis_map=args.axis_map,
                scale=args.scale,
            )
            + actual_offset
        )
        inferred_frames.append(
            display_positions(
                raw_inferred[UPPER_BODY_JOINTS],
                first_root=first_root,
                axis_map=args.axis_map,
                scale=args.scale,
            )
            + inferred_offset
        )
        q = retargeter.retarget_frame(raw_inferred, q)
        q_frames.append(q.copy())
        axol_frames.append(axol_link_positions(solver, q, axol_link_indices) + axol_offset)

    actual = np.asarray(actual_frames, dtype=np.float32)
    inferred = np.asarray(inferred_frames, dtype=np.float32)
    axol = np.asarray(axol_frames, dtype=np.float32)
    q_traj = np.asarray(q_frames, dtype=np.float32)
    errors = np.asarray(elbow_errors, dtype=np.float32)

    # Put all three skeletons on the same floor plane for easier visual comparison.
    actual[:, :, 2] -= float(actual[0, :, 2].min())
    inferred[:, :, 2] -= float(inferred[0, :, 2].min())
    axol[:, :, 2] -= float(axol[0, :, 2].min())

    print(
        "PICO inferred-elbow retargeting: "
        f"frames={len(frame_indices)}, axis_map={args.axis_map!r}, "
        f"Axol joints={solver.num_joints} ({solver.num_joints // 2} per arm)"
    )
    print(
        "Estimated arm lengths from shoulder/wrist only: "
        f"L upper={left_lengths[0]:.3f}m fore={left_lengths[1]:.3f}m, "
        f"R upper={right_lengths[0]:.3f}m fore={right_lengths[1]:.3f}m"
    )
    print(
        "Recorded elbows are diagnostic only: "
        f"mean={errors.mean():.3f}m, median={np.median(errors):.3f}m, "
        f"p95={np.percentile(errors, 95):.3f}m"
    )

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save,
            frame_indices=np.asarray(frame_indices, dtype=np.int64),
            q_traj=q_traj,
            actual_upper_body=actual,
            inferred_upper_body=inferred,
            axol_links=axol,
            elbow_errors=errors,
        )
        print(f"Saved solved trajectory to {args.save}")

    if args.smoke_test:
        return

    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError("PyVista is missing. Install with: uv sync") from exc

    lines = upper_body_lines()
    actual_points = pv.PolyData(actual[0])
    actual_bones = pv.PolyData(actual[0])
    actual_bones.lines = lines
    inferred_points = pv.PolyData(inferred[0])
    inferred_bones = pv.PolyData(inferred[0])
    inferred_bones.lines = lines
    axol_points = pv.PolyData(axol[0])
    axol_bones = pv.PolyData(axol[0])
    axol_bones.lines = AXOL_LINES

    actual_left_trail = polyline(actual[:1, UPPER_BODY_INDEX[LEFT_HAND]])
    actual_right_trail = polyline(actual[:1, UPPER_BODY_INDEX[RIGHT_HAND]])
    inferred_left_trail = polyline(inferred[:1, UPPER_BODY_INDEX[LEFT_HAND]])
    inferred_right_trail = polyline(inferred[:1, UPPER_BODY_INDEX[RIGHT_HAND]])
    axol_left_trail = polyline(axol[:1, AXOL_LEFT_GRIPPER_INDEX])
    axol_right_trail = polyline(axol[:1, AXOL_RIGHT_GRIPPER_INDEX])

    plotter = pv.Plotter(window_size=(1500, 850))
    plotter.set_background(args.background)
    plotter.add_axes()
    plotter.add_floor("-z", color="gray", lighting=False, pad=1.0)
    plotter.add_mesh(
        actual_points,
        color="crimson",
        point_size=args.point_size,
        render_points_as_spheres=True,
        label="PICO recorded joints",
    )
    plotter.add_mesh(actual_bones, color="white", line_width=args.line_width)
    plotter.add_mesh(
        inferred_points,
        color="deepskyblue",
        point_size=args.point_size,
        render_points_as_spheres=True,
        label="PICO inferred elbows",
    )
    plotter.add_mesh(inferred_bones, color="dodgerblue", line_width=args.line_width)
    plotter.add_mesh(
        axol_points,
        color="violet",
        point_size=args.point_size,
        render_points_as_spheres=True,
        label="Axol IK",
    )
    plotter.add_mesh(axol_bones, color="orchid", line_width=args.line_width)
    plotter.add_mesh(actual_left_trail, color="lime", line_width=5, label="Recorded L")
    plotter.add_mesh(actual_right_trail, color="orange", line_width=5, label="Recorded R")
    plotter.add_mesh(inferred_left_trail, color="springgreen", line_width=5, label="Inferred L")
    plotter.add_mesh(inferred_right_trail, color="gold", line_width=5, label="Inferred R")
    plotter.add_mesh(axol_left_trail, color="magenta", line_width=5, label="Axol L")
    plotter.add_mesh(axol_right_trail, color="yellow", line_width=5, label="Axol R")
    plotter.add_text("PICO recorded", position=(0.07, 0.93), viewport=True)
    plotter.add_text("Inferred elbows", position=(0.42, 0.93), viewport=True)
    plotter.add_text("Axol IK", position=(0.73, 0.93), viewport=True)
    plotter.add_legend(size=(0.22, 0.25), loc="upper right")
    plotter.camera_position = [
        (0.0, -4.0, 1.7),
        (0.0, 0.0, 0.7),
        (0.0, 0.0, 1.0),
    ]
    plotter.show(interactive_update=True, auto_close=False)

    playback_fps = float(args.fps if args.fps is not None else dataset_fps)
    frame_delay = 0.0 if playback_fps <= 0 else 1.0 / playback_fps

    while True:
        next_time = time.perf_counter()
        for local_index, frame_index in enumerate(frame_indices):
            actual_points.points = actual[local_index]
            actual_bones.points = actual[local_index]
            inferred_points.points = inferred[local_index]
            inferred_bones.points = inferred[local_index]
            axol_points.points = axol[local_index]
            axol_bones.points = axol[local_index]
            update_polyline(
                actual_left_trail,
                trail_window(actual[:, UPPER_BODY_INDEX[LEFT_HAND]], local_index, args.trail_length),
            )
            update_polyline(
                actual_right_trail,
                trail_window(actual[:, UPPER_BODY_INDEX[RIGHT_HAND]], local_index, args.trail_length),
            )
            update_polyline(
                inferred_left_trail,
                trail_window(
                    inferred[:, UPPER_BODY_INDEX[LEFT_HAND]],
                    local_index,
                    args.trail_length,
                ),
            )
            update_polyline(
                inferred_right_trail,
                trail_window(
                    inferred[:, UPPER_BODY_INDEX[RIGHT_HAND]],
                    local_index,
                    args.trail_length,
                ),
            )
            update_polyline(
                axol_left_trail,
                trail_window(axol[:, AXOL_LEFT_GRIPPER_INDEX], local_index, args.trail_length),
            )
            update_polyline(
                axol_right_trail,
                trail_window(axol[:, AXOL_RIGHT_GRIPPER_INDEX], local_index, args.trail_length),
            )
            plotter.add_text(
                f"episode={args.episode} frame={frame_index}/{len(poses) - 1} "
                f"elbow_err={errors[local_index]:.3f}m",
                name="frame_text",
                position="lower_left",
                font_size=10,
                color="white",
            )
            plotter.update()
            if frame_delay > 0:
                next_time += frame_delay
                time.sleep(max(0.0, next_time - time.perf_counter()))
        if not args.loop:
            break

    plotter.show()


if __name__ == "__main__":
    main()
