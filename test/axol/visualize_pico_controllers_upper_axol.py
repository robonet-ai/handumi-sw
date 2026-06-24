#!/usr/bin/env python3
"""Visualize controller-only PICO upper-body reconstruction and Axol IK."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

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
    infer_elbow,
    parse_axis_map,
    upper_body_lines,
)
from dexumi.robots.axol.config import KinematicsConfig
from dexumi.robots.axol.solver import KinematicsSolver
from dexumi.robots.utils import Joint, urdf_body_name


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
AXOL_LEFT_GRIPPER_INDEX = AXOL_LINK_ORDER.index(
    urdf_body_name(Joint.GRIPPER, is_left=True)
)
AXOL_RIGHT_GRIPPER_INDEX = AXOL_LINK_ORDER.index(
    urdf_body_name(Joint.GRIPPER, is_left=False)
)
AXOL_LEFT_ELBOW_INDEX = AXOL_LINK_ORDER.index(urdf_body_name(Joint.ELBOW, is_left=True))
AXOL_RIGHT_ELBOW_INDEX = AXOL_LINK_ORDER.index(urdf_body_name(Joint.ELBOW, is_left=False))
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


COMPACT_NAMES = [
    "pelvis",
    "spine",
    "chest",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
]
PELVIS = COMPACT_NAMES.index("pelvis")
SPINE = COMPACT_NAMES.index("spine")
CHEST = COMPACT_NAMES.index("chest")
NECK = COMPACT_NAMES.index("neck")
HEAD = COMPACT_NAMES.index("head")
L_SHOULDER = COMPACT_NAMES.index("left_shoulder")
L_ELBOW = COMPACT_NAMES.index("left_elbow")
L_WRIST = COMPACT_NAMES.index("left_wrist")
R_SHOULDER = COMPACT_NAMES.index("right_shoulder")
R_ELBOW = COMPACT_NAMES.index("right_elbow")
R_WRIST = COMPACT_NAMES.index("right_wrist")


def _load_fps(root: Path) -> float:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return 30.0
    with info_path.open("r", encoding="utf-8") as f:
        return float(json.load(f).get("fps", 30.0))


def _load_episode(root: Path, episode: int) -> tuple[pd.DataFrame, float]:
    parquet_files = sorted((root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")

    df = pd.concat([pd.read_parquet(path) for path in parquet_files], ignore_index=True)
    df = df[df["episode_index"] == episode].copy()
    if df.empty:
        raise ValueError(f"Episode {episode} not found in {root}")

    sort_columns = [col for col in ("index", "frame_index") if col in df.columns]
    if sort_columns:
        df.sort_values(sort_columns, inplace=True)
    return df.reset_index(drop=True), _load_fps(root)


def _stack_pose(values: np.ndarray, column: str) -> np.ndarray:
    poses = np.stack([np.asarray(value, dtype=np.float32) for value in values])
    if poses.ndim != 2 or poses.shape[1] < 7:
        raise ValueError(f"{column!r} must have shape (N, >=7); got {poses.shape}")
    return poses[:, :7]


def _controllers_to_headset_local(
    headset: np.ndarray,
    left_controller: np.ndarray,
    right_controller: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return chest-local pseudo poses using the headset as a chest anchor."""

    local_headset = np.zeros_like(headset)
    local_headset[:, 6] = 1.0
    local_left = left_controller.copy()
    local_right = right_controller.copy()
    for index in range(len(headset)):
        rotation = Rotation.from_quat(headset[index, 3:7]).as_matrix().astype(np.float32)
        local_left[index, :3] = rotation.T @ (
            left_controller[index, :3] - headset[index, :3]
        )
        local_right[index, :3] = rotation.T @ (
            right_controller[index, :3] - headset[index, :3]
        )
        local_left[index, 3:7] = left_controller[index, 3:7]
        local_right[index, 3:7] = right_controller[index, 3:7]
    return local_headset, local_left, local_right


def _normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return fallback.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _estimate_shoulders(
    anchors: np.ndarray,
    left_wrists: np.ndarray,
    right_wrists: np.ndarray,
    *,
    shoulder_width: float,
    anchor_to_shoulder: float,
    anchor_to_head: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    lateral_fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    pelvises = []
    spines = []
    chests = []
    necks = []
    heads = []
    left_shoulders = []
    right_shoulders = []
    laterals = []

    for anchor, left_wrist, right_wrist in zip(anchors, left_wrists, right_wrists):
        lateral = left_wrist - right_wrist
        lateral = lateral - up * float(np.dot(lateral, up))
        lateral = _normalize(lateral, lateral_fallback)
        shoulder_center = anchor + up * anchor_to_shoulder

        pelvises.append(anchor - up * 0.34)
        spines.append(anchor - up * 0.17)
        chests.append(anchor)
        necks.append(anchor + up * (anchor_to_head * 0.68))
        heads.append(anchor + up * anchor_to_head)
        left_shoulders.append(shoulder_center + lateral * (shoulder_width * 0.5))
        right_shoulders.append(shoulder_center - lateral * (shoulder_width * 0.5))
        laterals.append(lateral)

    return (
        np.asarray(pelvises, dtype=np.float32),
        np.asarray(spines, dtype=np.float32),
        np.asarray(chests, dtype=np.float32),
        np.asarray(necks, dtype=np.float32),
        np.asarray(heads, dtype=np.float32),
        np.asarray(left_shoulders, dtype=np.float32),
        np.asarray(right_shoulders, dtype=np.float32),
        np.asarray(laterals, dtype=np.float32),
    )


def _constrain_wrist(
    shoulder: np.ndarray,
    wrist: np.ndarray,
    *,
    arm_length: float,
    max_reach_ratio: float,
) -> np.ndarray:
    offset = wrist - shoulder
    distance = float(np.linalg.norm(offset))
    if distance < 1e-6:
        return wrist.astype(np.float32)
    max_reach = arm_length * max_reach_ratio
    if distance <= max_reach:
        return wrist.astype(np.float32)
    return (shoulder + offset / distance * max_reach).astype(np.float32)


def reconstruct_controller_upper_body(
    headset: np.ndarray,
    left_controller: np.ndarray,
    right_controller: np.ndarray,
    *,
    shoulder_width: float,
    anchor_to_shoulder: float,
    anchor_to_head: float,
    arm_length: float,
    upper_ratio: float,
    max_reach_ratio: float,
    bend_forward: float,
    bend_down: float,
    bend_side: float,
) -> tuple[np.ndarray, np.ndarray]:
    anchors = headset[:, :3].astype(np.float32)
    left_wrists = left_controller[:, :3].astype(np.float32)
    right_wrists = right_controller[:, :3].astype(np.float32)
    (
        pelvises,
        spines,
        chests,
        necks,
        heads,
        left_shoulders,
        right_shoulders,
        laterals,
    ) = _estimate_shoulders(
        anchors,
        left_wrists,
        right_wrists,
        shoulder_width=shoulder_width,
        anchor_to_shoulder=anchor_to_shoulder,
        anchor_to_head=anchor_to_head,
    )

    left_lengths = (arm_length * upper_ratio, arm_length * (1.0 - upper_ratio))
    right_lengths = left_lengths
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    compact = np.empty((len(headset), len(COMPACT_NAMES), 3), dtype=np.float32)
    smpl24 = np.zeros((len(headset), 24, 7), dtype=np.float32)

    for index in range(len(headset)):
        left_wrist = _constrain_wrist(
            left_shoulders[index],
            left_wrists[index],
            arm_length=arm_length,
            max_reach_ratio=max_reach_ratio,
        )
        right_wrist = _constrain_wrist(
            right_shoulders[index],
            right_wrists[index],
            arm_length=arm_length,
            max_reach_ratio=max_reach_ratio,
        )
        shoulder_center = 0.5 * (left_shoulders[index] + right_shoulders[index])
        wrist_center = 0.5 * (left_wrist + right_wrist)
        forward = wrist_center - shoulder_center
        forward = forward - up * float(np.dot(forward, up))
        forward = _normalize(forward, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        left_hint = forward * bend_forward + up * bend_down + laterals[index] * bend_side
        right_hint = forward * bend_forward + up * bend_down - laterals[index] * bend_side

        left_elbow = infer_elbow(
            left_shoulders[index],
            left_wrist,
            upper_length=left_lengths[0],
            forearm_length=left_lengths[1],
            bend_hint=left_hint,
        )
        right_elbow = infer_elbow(
            right_shoulders[index],
            right_wrist,
            upper_length=right_lengths[0],
            forearm_length=right_lengths[1],
            bend_hint=right_hint,
        )
        compact[index] = np.stack(
            [
                pelvises[index],
                spines[index],
                chests[index],
                necks[index],
                heads[index],
                left_shoulders[index],
                left_elbow,
                left_wrist,
                right_shoulders[index],
                right_elbow,
                right_wrist,
            ]
        )

        smpl24[index, 0, :3] = pelvises[index]
        smpl24[index, 3, :3] = spines[index]
        smpl24[index, 6, :3] = 0.5 * (spines[index] + chests[index])
        smpl24[index, 9, :3] = chests[index]
        smpl24[index, 12, :3] = necks[index]
        smpl24[index, 13, :3] = 0.5 * (necks[index] + left_shoulders[index])
        smpl24[index, 14, :3] = 0.5 * (necks[index] + right_shoulders[index])
        smpl24[index, 15, :3] = heads[index]
        smpl24[index, LEFT_SHOULDER, :3] = left_shoulders[index]
        smpl24[index, RIGHT_SHOULDER, :3] = right_shoulders[index]
        smpl24[index, LEFT_ELBOW, :3] = left_elbow
        smpl24[index, RIGHT_ELBOW, :3] = right_elbow
        smpl24[index, LEFT_WRIST, :3] = left_wrist
        smpl24[index, RIGHT_WRIST, :3] = right_wrist
        smpl24[index, LEFT_HAND, :3] = left_wrist
        smpl24[index, RIGHT_HAND, :3] = right_wrist
        smpl24[index, :, 6] = 1.0

    return compact, smpl24


def display_positions(
    positions: np.ndarray,
    *,
    first_root: np.ndarray,
    axis_map: str,
    scale: float,
    offset: np.ndarray,
) -> np.ndarray:
    transform = parse_axis_map(axis_map)
    flat = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    out = np.asarray(
        [transform(point - first_root) * scale + offset for point in flat],
        dtype=np.float32,
    )
    return out.reshape(positions.shape)


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
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--body-frame",
        choices=("headset-local", "world"),
        default="headset-local",
        help="Use headset-local for a chest-mounted PICO; world is kept for debugging.",
    )
    parser.add_argument(
        "--axis-map",
        default="x,z,y",
        help="Axis map used for retargeting PICO controller motion into Axol space.",
    )
    parser.add_argument(
        "--display-axis-map",
        default="x,z,y",
        help="Axis map used only to draw the PICO upper body beside Axol.",
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--spacing", type=float, default=1.25)
    parser.add_argument("--trail-length", type=int, default=180)
    parser.add_argument("--shoulder-width", type=float, default=0.38)
    parser.add_argument("--anchor-to-shoulder", type=float, default=0.10)
    parser.add_argument("--anchor-to-head", type=float, default=0.42)
    parser.add_argument("--arm-length", type=float, default=0.62)
    parser.add_argument("--upper-ratio", type=float, default=0.48)
    parser.add_argument("--max-reach-ratio", type=float, default=0.98)
    parser.add_argument("--bend-forward", type=float, default=0.0)
    parser.add_argument("--bend-down", type=float, default=-1.0)
    parser.add_argument("--bend-side", type=float, default=0.10)
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
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    df, dataset_fps = _load_episode(args.dataset_root, args.episode)
    left_controller = _stack_pose(
        df["observation.pico.left_controller_pose"].values,
        "observation.pico.left_controller_pose",
    )
    right_controller = _stack_pose(
        df["observation.pico.right_controller_pose"].values,
        "observation.pico.right_controller_pose",
    )
    headset = _stack_pose(
        df["observation.pico.headset_pose"].values,
        "observation.pico.headset_pose",
    )
    if args.body_frame == "headset-local":
        body_headset, body_left_controller, body_right_controller = (
            _controllers_to_headset_local(headset, left_controller, right_controller)
        )
    else:
        body_headset = headset
        body_left_controller = left_controller
        body_right_controller = right_controller

    _, body24 = reconstruct_controller_upper_body(
        body_headset,
        body_left_controller,
        body_right_controller,
        shoulder_width=args.shoulder_width,
        anchor_to_shoulder=args.anchor_to_shoulder,
        anchor_to_head=args.anchor_to_head,
        arm_length=args.arm_length,
        upper_ratio=args.upper_ratio,
        max_reach_ratio=args.max_reach_ratio,
        bend_forward=args.bend_forward,
        bend_down=args.bend_down,
        bend_side=args.bend_side,
    )

    frame_indices = list(range(args.start_frame, len(df), args.stride))
    if args.max_frames is not None:
        frame_indices = frame_indices[: args.max_frames]
    if not frame_indices:
        raise ValueError("No frames selected.")

    config = KinematicsConfig(
        pos_weight=args.pos_weight,
        ori_weight=args.ori_weight,
        elbow_weight=args.elbow_weight,
        max_joint_delta=args.max_joint_delta,
        max_reach=args.max_reach,
    )
    solver = KinematicsSolver(config=config)
    retargeter = PicoToAxolArmRetargeter(
        solver=solver,
        first_body_pose=body24[frame_indices[0]],
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
    q = settle_first_frame(retargeter, body24[frame_indices[0]], args.settle_iterations)
    solver.set_posture_pose(q)

    robot_link_names = list(solver.robot.links.names)
    axol_link_indices = [robot_link_names.index(name) for name in AXOL_LINK_ORDER]

    first_root = body_headset[frame_indices[0], :3].copy()
    human_offset = np.array([-args.spacing * 0.5, 0.0, 0.0], dtype=np.float32)
    axol_offset = np.array([args.spacing * 0.5, 0.0, 0.0], dtype=np.float32)

    human_frames = []
    axol_frames = []
    for frame_index in frame_indices:
        human_frames.append(
            display_positions(
                body24[frame_index, UPPER_BODY_JOINTS, :3],
                first_root=first_root,
                axis_map=args.display_axis_map,
                scale=args.scale,
                offset=human_offset,
            )
        )
        q = retargeter.retarget_frame(body24[frame_index], q)
        axol_frames.append(
            axol_link_positions(solver, q, axol_link_indices) + axol_offset
        )

    human = np.asarray(human_frames, dtype=np.float32)
    axol = np.asarray(axol_frames, dtype=np.float32)

    human[:, :, 2] -= float(human[0, :, 2].min())
    axol[:, :, 2] -= float(axol[0, :, 2].min())

    print(
        "PICO controller upper-body retargeting: "
        f"episode={args.episode}, frames={len(frame_indices)}, "
        f"body_frame={args.body_frame!r}, "
        f"retarget_axis_map={args.axis_map!r}, "
        f"display_axis_map={args.display_axis_map!r}, "
        f"Axol joints={solver.num_joints}"
    )
    print(
        "Controller-only upper body: "
        f"shoulder_width={args.shoulder_width:.3f}m, "
        f"arm_length={args.arm_length:.3f}m, "
        f"elbow_weight={args.elbow_weight:.3f}"
    )
    print(
        "Controller motion range: "
        f"L={np.linalg.norm(left_controller[:, :3] - left_controller[0, :3], axis=1).max():.3f}m, "
        f"R={np.linalg.norm(right_controller[:, :3] - right_controller[0, :3], axis=1).max():.3f}m"
    )

    if args.smoke_test:
        return

    import pyvista as pv

    off_screen = args.screenshot is not None
    plotter = pv.Plotter(window_size=(1500, 850), off_screen=off_screen)
    plotter.set_background(args.background)
    plotter.add_axes()
    plotter.add_floor("-z", color="gray", lighting=False, pad=1.0)

    human_points = pv.PolyData(human[0])
    human_bones = pv.PolyData(human[0])
    human_bones.lines = upper_body_lines()
    axol_points = pv.PolyData(axol[0])
    axol_bones = pv.PolyData(axol[0])
    axol_bones.lines = AXOL_LINES

    human_left_trail = polyline(human[:1, UPPER_BODY_INDEX[LEFT_HAND]])
    human_right_trail = polyline(human[:1, UPPER_BODY_INDEX[RIGHT_HAND]])
    human_left_elbow_trail = polyline(human[:1, UPPER_BODY_INDEX[LEFT_ELBOW]])
    human_right_elbow_trail = polyline(human[:1, UPPER_BODY_INDEX[RIGHT_ELBOW]])
    axol_left_trail = polyline(axol[:1, AXOL_LEFT_GRIPPER_INDEX])
    axol_right_trail = polyline(axol[:1, AXOL_RIGHT_GRIPPER_INDEX])
    axol_left_elbow_trail = polyline(axol[:1, AXOL_LEFT_ELBOW_INDEX])
    axol_right_elbow_trail = polyline(axol[:1, AXOL_RIGHT_ELBOW_INDEX])

    plotter.add_mesh(
        human_points,
        color="crimson",
        point_size=args.point_size,
        render_points_as_spheres=True,
        label="Controller joints",
    )
    plotter.add_mesh(
        human_bones,
        color="white",
        line_width=args.line_width,
        render_lines_as_tubes=True,
        label="Inferred upper body",
    )
    plotter.add_mesh(
        axol_points,
        color="deepskyblue",
        point_size=args.point_size,
        render_points_as_spheres=True,
        label="Axol links",
    )
    plotter.add_mesh(
        axol_bones,
        color="dodgerblue",
        line_width=args.line_width,
        render_lines_as_tubes=True,
        label="Axol skeleton",
    )
    plotter.add_mesh(human_left_trail, color="lime", line_width=5, label="Controller L")
    plotter.add_mesh(human_right_trail, color="orange", line_width=5, label="Controller R")
    plotter.add_mesh(
        human_left_elbow_trail,
        color="cyan",
        line_width=3,
        label="Human L elbow",
    )
    plotter.add_mesh(
        human_right_elbow_trail,
        color="magenta",
        line_width=3,
        label="Human R elbow",
    )
    plotter.add_mesh(axol_left_trail, color="springgreen", line_width=5, label="Axol L")
    plotter.add_mesh(axol_right_trail, color="gold", line_width=5, label="Axol R")
    plotter.add_mesh(
        axol_left_elbow_trail,
        color="lightskyblue",
        line_width=3,
        label="Axol L elbow",
    )
    plotter.add_mesh(
        axol_right_elbow_trail,
        color="violet",
        line_width=3,
        label="Axol R elbow",
    )
    plotter.add_text(
        "PICO controllers + inferred elbows",
        position=(0.06, 0.93),
        viewport=True,
    )
    plotter.add_text("Axol IK", position=(0.62, 0.93), viewport=True)
    plotter.add_legend(size=(0.25, 0.24), loc="upper right")
    plotter.camera_position = [
        (0.0, -4.0, 1.7),
        (0.0, 0.0, 0.7),
        (0.0, 0.0, 1.0),
    ]

    if args.screenshot is not None:
        plotter.show(auto_close=False)
        plotter.screenshot(str(args.screenshot))
        plotter.close()
        print(f"Screenshot saved to {args.screenshot}")
        return

    plotter.show(auto_close=False, interactive_update=True)
    playback_fps = float(args.fps if args.fps is not None else dataset_fps)
    frame_delay = 0.0 if playback_fps <= 0 else 1.0 / playback_fps

    while True:
        next_time = time.perf_counter()
        for local_index, frame_index in enumerate(frame_indices):
            human_points.points = human[local_index]
            human_bones.points = human[local_index]
            axol_points.points = axol[local_index]
            axol_bones.points = axol[local_index]
            update_polyline(
                human_left_trail,
                trail_window(
                    human[:, UPPER_BODY_INDEX[LEFT_HAND]],
                    local_index,
                    args.trail_length,
                ),
            )
            update_polyline(
                human_right_trail,
                trail_window(
                    human[:, UPPER_BODY_INDEX[RIGHT_HAND]],
                    local_index,
                    args.trail_length,
                ),
            )
            update_polyline(
                human_left_elbow_trail,
                trail_window(
                    human[:, UPPER_BODY_INDEX[LEFT_ELBOW]],
                    local_index,
                    args.trail_length,
                ),
            )
            update_polyline(
                human_right_elbow_trail,
                trail_window(
                    human[:, UPPER_BODY_INDEX[RIGHT_ELBOW]],
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
            update_polyline(
                axol_left_elbow_trail,
                trail_window(axol[:, AXOL_LEFT_ELBOW_INDEX], local_index, args.trail_length),
            )
            update_polyline(
                axol_right_elbow_trail,
                trail_window(axol[:, AXOL_RIGHT_ELBOW_INDEX], local_index, args.trail_length),
            )
            plotter.add_text(
                f"episode={args.episode} frame={frame_index}/{len(df) - 1}",
                name="frame_text",
                position="lower_left",
                font_size=11,
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
