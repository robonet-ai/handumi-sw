#!/usr/bin/env python3
"""Compare controller-only upper-body reconstruction methods for chest-mounted PICO."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from dexumi.retargeting.pico_upper_body import infer_elbow


NAMES = [
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
PELVIS, SPINE, CHEST, NECK, HEAD = range(5)
L_SHOULDER, L_ELBOW, L_WRIST = 5, 6, 7
R_SHOULDER, R_ELBOW, R_WRIST = 8, 9, 10

LINES = np.asarray(
    [
        2,
        PELVIS,
        SPINE,
        2,
        SPINE,
        CHEST,
        2,
        CHEST,
        NECK,
        2,
        NECK,
        HEAD,
        2,
        NECK,
        L_SHOULDER,
        2,
        L_SHOULDER,
        L_ELBOW,
        2,
        L_ELBOW,
        L_WRIST,
        2,
        NECK,
        R_SHOULDER,
        2,
        R_SHOULDER,
        R_ELBOW,
        2,
        R_ELBOW,
        R_WRIST,
    ],
    dtype=np.int_,
)

# Measured from the old full-body dataset, first frames, relative to joint 9/chest.
# Old full-body axes are mapped into the headset-local convention below by
# _old_body_offset_to_local(): local x = body left/right, local y = up, local z = back.
OLD_CHEST_OFFSETS = {
    "pelvis": np.array([-0.005, -0.293, -0.024], dtype=np.float32),
    "spine": np.array([0.0, -0.14, -0.012], dtype=np.float32),
    "chest": np.array([0.0, 0.0, 0.0], dtype=np.float32),
    "neck": np.array([0.0363, 0.1991, -0.0004], dtype=np.float32),
    "head": np.array([-0.0228, 0.2641, 0.0010], dtype=np.float32),
    "left_shoulder": np.array([0.0361, 0.1025, 0.1821], dtype=np.float32),
    "right_shoulder": np.array([0.0550, 0.1184, -0.1745], dtype=np.float32),
}


@dataclass(frozen=True)
class MethodResult:
    name: str
    skeletons: np.ndarray
    left_color: str
    right_color: str
    joint_color: str
    line_color: str


def _old_body_offset_to_local(offset: np.ndarray) -> np.ndarray:
    """Map old PICO full-body offsets to chest-local OpenXR-like coordinates.

    The old full-body initial pose had body left/right mostly on the old z-axis.
    In headset-local coordinates from the new recordings, left/right is x, up is
    y, and forward is -z. This keeps old left shoulder on local negative x.
    """

    x_old, y_old, z_old = np.asarray(offset, dtype=np.float32)
    return np.array([-z_old, y_old, x_old], dtype=np.float32)


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


def _to_headset_local(points: np.ndarray, headset_pose: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    headset_pos = np.asarray(headset_pose[:3], dtype=np.float32)
    headset_rot = Rotation.from_quat(headset_pose[3:7]).as_matrix().astype(np.float32)
    return ((points.reshape(-1, 3) - headset_pos) @ headset_rot).reshape(points.shape)


def _controllers_to_local(
    headset: np.ndarray, left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    local_left = np.empty((len(headset), 3), dtype=np.float32)
    local_right = np.empty((len(headset), 3), dtype=np.float32)
    for i in range(len(headset)):
        rot = Rotation.from_quat(headset[i, 3:7]).as_matrix().astype(np.float32)
        local_left[i] = rot.T @ (left[i, :3] - headset[i, :3])
        local_right[i] = rot.T @ (right[i, :3] - headset[i, :3])
    return local_left, local_right


def _normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return fallback.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _infer_elbow_with_hint(
    shoulder: np.ndarray,
    wrist: np.ndarray,
    *,
    upper_length: float,
    forearm_length: float,
    hint: np.ndarray,
) -> np.ndarray:
    distance = float(np.linalg.norm(wrist - shoulder))
    reach = upper_length + forearm_length
    if distance > reach * 0.995:
        direction = (wrist - shoulder) / max(distance, 1e-6)
        wrist = shoulder + direction * reach * 0.995
    return infer_elbow(
        shoulder,
        wrist,
        upper_length=upper_length,
        forearm_length=forearm_length,
        bend_hint=hint,
    )


def _skeleton_from_local(
    left_wrist: np.ndarray,
    right_wrist: np.ndarray,
    *,
    left_shoulder: np.ndarray,
    right_shoulder: np.ndarray,
    neck: np.ndarray,
    head: np.ndarray,
    pelvis: np.ndarray,
    spine: np.ndarray,
    upper_length: float,
    forearm_length: float,
    left_hint: np.ndarray,
    right_hint: np.ndarray,
) -> np.ndarray:
    frames = np.empty((len(left_wrist), len(NAMES), 3), dtype=np.float32)
    for i in range(len(left_wrist)):
        left_elbow = _infer_elbow_with_hint(
            left_shoulder,
            left_wrist[i],
            upper_length=upper_length,
            forearm_length=forearm_length,
            hint=left_hint,
        )
        right_elbow = _infer_elbow_with_hint(
            right_shoulder,
            right_wrist[i],
            upper_length=upper_length,
            forearm_length=forearm_length,
            hint=right_hint,
        )
        frames[i] = np.stack(
            [
                pelvis,
                spine,
                np.zeros(3, dtype=np.float32),
                neck,
                head,
                left_shoulder,
                left_elbow,
                left_wrist[i],
                right_shoulder,
                right_elbow,
                right_wrist[i],
            ]
        )
    return frames


def _calibrated_90_skeleton(
    left_wrist: np.ndarray,
    right_wrist: np.ndarray,
    *,
    shoulder_width: float,
    shoulder_height: float,
    neck_height: float,
    head_height: float,
) -> np.ndarray:
    left_shoulder = np.array(
        [-shoulder_width * 0.5, shoulder_height, 0.0],
        dtype=np.float32,
    )
    right_shoulder = np.array(
        [shoulder_width * 0.5, shoulder_height, 0.0],
        dtype=np.float32,
    )
    left_elbow0 = np.array(
        [
            left_shoulder[0] * 0.85 + left_wrist[0, 0] * 0.15,
            left_wrist[0, 1],
            left_shoulder[2] * 0.85 + left_wrist[0, 2] * 0.15,
        ],
        dtype=np.float32,
    )
    right_elbow0 = np.array(
        [
            right_shoulder[0] * 0.85 + right_wrist[0, 0] * 0.15,
            right_wrist[0, 1],
            right_shoulder[2] * 0.85 + right_wrist[0, 2] * 0.15,
        ],
        dtype=np.float32,
    )
    left_upper = float(np.linalg.norm(left_elbow0 - left_shoulder))
    left_fore = float(np.linalg.norm(left_wrist[0] - left_elbow0))
    right_upper = float(np.linalg.norm(right_elbow0 - right_shoulder))
    right_fore = float(np.linalg.norm(right_wrist[0] - right_elbow0))
    upper_length = max(left_upper, right_upper, 0.25)
    forearm_length = max(left_fore, right_fore, 0.25)

    frames = np.empty((len(left_wrist), len(NAMES), 3), dtype=np.float32)
    for i in range(len(left_wrist)):
        left_axis = left_wrist[i] - left_shoulder
        right_axis = right_wrist[i] - right_shoulder
        left_hint = left_elbow0 - left_shoulder
        right_hint = right_elbow0 - right_shoulder
        left_direction = _normalize(
            left_axis,
            np.array([0, 0, -1], dtype=np.float32),
        )
        right_direction = _normalize(
            right_axis,
            np.array([0, 0, -1], dtype=np.float32),
        )
        left_hint = left_hint - left_direction * float(np.dot(left_hint, left_direction))
        right_hint = right_hint - right_direction * float(
            np.dot(right_hint, right_direction)
        )
        frames[i] = np.stack(
            [
                np.array([0.0, -0.34, 0.0], dtype=np.float32),
                np.array([0.0, -0.17, 0.0], dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                np.array([0.0, neck_height, 0.0], dtype=np.float32),
                np.array([0.0, head_height, -0.02], dtype=np.float32),
                left_shoulder,
                _infer_elbow_with_hint(
                    left_shoulder,
                    left_wrist[i],
                    upper_length=upper_length,
                    forearm_length=forearm_length,
                    hint=left_hint,
                ),
                left_wrist[i],
                right_shoulder,
                _infer_elbow_with_hint(
                    right_shoulder,
                    right_wrist[i],
                    upper_length=upper_length,
                    forearm_length=forearm_length,
                    hint=right_hint,
                ),
                right_wrist[i],
            ]
        )
    return frames


def _world_naive_skeleton(
    headset: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    shoulder_width: float,
    shoulder_height: float,
    arm_length: float,
    upper_ratio: float,
) -> np.ndarray:
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    world = np.empty((len(headset), len(NAMES), 3), dtype=np.float32)
    upper_length = arm_length * upper_ratio
    forearm_length = arm_length * (1.0 - upper_ratio)
    for i in range(len(headset)):
        anchor = headset[i, :3]
        lateral = left[i, :3] - right[i, :3]
        lateral = lateral - up * float(np.dot(lateral, up))
        lateral = _normalize(lateral, np.array([-1.0, 0.0, 0.0], dtype=np.float32))
        left_shoulder = anchor + up * shoulder_height + lateral * shoulder_width * 0.5
        right_shoulder = anchor + up * shoulder_height - lateral * shoulder_width * 0.5
        left_elbow = _infer_elbow_with_hint(
            left_shoulder,
            left[i, :3],
            upper_length=upper_length,
            forearm_length=forearm_length,
            hint=-up,
        )
        right_elbow = _infer_elbow_with_hint(
            right_shoulder,
            right[i, :3],
            upper_length=upper_length,
            forearm_length=forearm_length,
            hint=-up,
        )
        world[i] = np.stack(
            [
                anchor - up * 0.34,
                anchor - up * 0.17,
                anchor,
                anchor + up * 0.20,
                anchor + up * 0.26,
                left_shoulder,
                left_elbow,
                left[i, :3],
                right_shoulder,
                right_elbow,
                right[i, :3],
            ]
        )

    local = np.empty_like(world)
    for i in range(len(headset)):
        local[i] = _to_headset_local(world[i], headset[i])
    return local


def build_methods(
    headset: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    shoulder_width: float,
    shoulder_height: float,
    arm_length: float,
    upper_ratio: float,
) -> list[MethodResult]:
    left_local, right_local = _controllers_to_local(headset, left, right)
    upper_length = arm_length * upper_ratio
    forearm_length = arm_length * (1.0 - upper_ratio)

    old = {
        name: _old_body_offset_to_local(value)
        for name, value in OLD_CHEST_OFFSETS.items()
    }
    simple_left_shoulder = np.array(
        [-shoulder_width * 0.5, shoulder_height, 0.0],
        dtype=np.float32,
    )
    simple_right_shoulder = np.array(
        [shoulder_width * 0.5, shoulder_height, 0.0],
        dtype=np.float32,
    )

    return [
        MethodResult(
            name="A world naive",
            skeletons=_world_naive_skeleton(
                headset,
                left,
                right,
                shoulder_width=shoulder_width,
                shoulder_height=shoulder_height,
                arm_length=arm_length,
                upper_ratio=upper_ratio,
            ),
            left_color="lime",
            right_color="orange",
            joint_color="crimson",
            line_color="white",
        ),
        MethodResult(
            name="B chest local",
            skeletons=_skeleton_from_local(
                left_local,
                right_local,
                left_shoulder=simple_left_shoulder,
                right_shoulder=simple_right_shoulder,
                neck=np.array([0.0, 0.20, 0.0], dtype=np.float32),
                head=np.array([0.0, 0.27, -0.02], dtype=np.float32),
                pelvis=np.array([0.0, -0.34, 0.0], dtype=np.float32),
                spine=np.array([0.0, -0.17, 0.0], dtype=np.float32),
                upper_length=upper_length,
                forearm_length=forearm_length,
                left_hint=np.array([0.0, -1.0, 0.0], dtype=np.float32),
                right_hint=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            ),
            left_color="springgreen",
            right_color="gold",
            joint_color="deepskyblue",
            line_color="dodgerblue",
        ),
        MethodResult(
            name="C old head offset",
            skeletons=_skeleton_from_local(
                left_local,
                right_local,
                left_shoulder=old["left_shoulder"],
                right_shoulder=old["right_shoulder"],
                neck=old["neck"],
                head=old["head"],
                pelvis=old["pelvis"],
                spine=old["spine"],
                upper_length=upper_length,
                forearm_length=forearm_length,
                left_hint=np.array([0.0, -1.0, 0.0], dtype=np.float32),
                right_hint=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            ),
            left_color="chartreuse",
            right_color="yellow",
            joint_color="violet",
            line_color="orchid",
        ),
        MethodResult(
            name="D first-pose 90",
            skeletons=_calibrated_90_skeleton(
                left_local,
                right_local,
                shoulder_width=shoulder_width,
                shoulder_height=shoulder_height,
                neck_height=0.20,
                head_height=0.27,
            ),
            left_color="mediumspringgreen",
            right_color="khaki",
            joint_color="tomato",
            line_color="lightgray",
        ),
    ]


def _display(values: np.ndarray, offset_x: float, scale: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    out = np.empty_like(values)
    out[..., 0] = values[..., 0] * scale + offset_x
    out[..., 1] = values[..., 2] * scale
    out[..., 2] = values[..., 1] * scale
    return out


def _polyline(points: np.ndarray) -> "pv.PolyData":
    import pyvista as pv

    if len(points) < 2:
        return pv.PolyData(points)
    poly = pv.PolyData(points)
    poly.lines = np.concatenate([[len(points)], np.arange(len(points))]).astype(np.int_)
    return poly


def _update_polyline(poly: "pv.PolyData", points: np.ndarray) -> None:
    if len(points) < 2:
        poly.points = points if len(points) else np.zeros((1, 3), dtype=np.float32)
        poly.lines = np.empty(0, dtype=np.int_)
        return
    poly.points = points
    poly.lines = np.concatenate([[len(points)], np.arange(len(points))]).astype(np.int_)


def _window(values: np.ndarray, index: int, trail_length: int) -> np.ndarray:
    start = 0 if trail_length <= 0 else max(0, index + 1 - trail_length)
    return values[start : index + 1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--spacing", type=float, default=1.15)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--trail-length", type=int, default=180)
    parser.add_argument("--shoulder-width", type=float, default=0.38)
    parser.add_argument("--shoulder-height", type=float, default=0.10)
    parser.add_argument("--arm-length", type=float, default=0.62)
    parser.add_argument("--upper-ratio", type=float, default=0.48)
    parser.add_argument("--background", default="black")
    parser.add_argument("--point-size", type=float, default=14.0)
    parser.add_argument("--line-width", type=float, default=4.0)
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    df, dataset_fps = _load_episode(args.dataset_root, args.episode)
    headset = _stack_pose(
        df["observation.pico.headset_pose"].values,
        "observation.pico.headset_pose",
    )
    left = _stack_pose(
        df["observation.pico.left_controller_pose"].values,
        "observation.pico.left_controller_pose",
    )
    right = _stack_pose(
        df["observation.pico.right_controller_pose"].values,
        "observation.pico.right_controller_pose",
    )

    frame_indices = list(range(args.start_frame, len(df), args.stride))
    if args.max_frames is not None:
        frame_indices = frame_indices[: args.max_frames]
    if not frame_indices:
        raise ValueError("No frames selected.")

    methods = build_methods(
        headset,
        left,
        right,
        shoulder_width=args.shoulder_width,
        shoulder_height=args.shoulder_height,
        arm_length=args.arm_length,
        upper_ratio=args.upper_ratio,
    )
    offsets = (np.arange(len(methods), dtype=np.float32) - (len(methods) - 1) * 0.5) * args.spacing
    selected = []
    for method, offset in zip(methods, offsets):
        display = _display(method.skeletons[frame_indices], float(offset), args.scale)
        display[:, :, 2] -= float(display[0, :, 2].min())
        selected.append(
            MethodResult(
                method.name,
                display,
                method.left_color,
                method.right_color,
                method.joint_color,
                method.line_color,
            )
        )

    print(
        f"Loaded episode={args.episode}, frames={len(df)}, selected={len(frame_indices)}, "
        f"fps={dataset_fps:g}"
    )
    for method in selected:
        first = method.skeletons[0]
        print(
            f"{method.name}: "
            f"L shoulder={np.round(first[L_SHOULDER], 3)}, "
            f"L elbow={np.round(first[L_ELBOW], 3)}, "
            f"L wrist={np.round(first[L_WRIST], 3)}"
        )

    if args.smoke_test:
        return

    import pyvista as pv

    plotter = pv.Plotter(window_size=(1600, 900), off_screen=args.screenshot is not None)
    plotter.set_background(args.background)
    plotter.add_axes()
    plotter.add_floor("-z", color="gray", lighting=False, pad=1.0)

    point_meshes = []
    bone_meshes = []
    left_trails = []
    right_trails = []
    left_elbow_trails = []
    right_elbow_trails = []
    for index, method in enumerate(selected):
        points = pv.PolyData(method.skeletons[0])
        bones = pv.PolyData(method.skeletons[0])
        bones.lines = LINES
        left_trail = _polyline(method.skeletons[:1, L_WRIST])
        right_trail = _polyline(method.skeletons[:1, R_WRIST])
        left_elbow_trail = _polyline(method.skeletons[:1, L_ELBOW])
        right_elbow_trail = _polyline(method.skeletons[:1, R_ELBOW])
        point_meshes.append(points)
        bone_meshes.append(bones)
        left_trails.append(left_trail)
        right_trails.append(right_trail)
        left_elbow_trails.append(left_elbow_trail)
        right_elbow_trails.append(right_elbow_trail)
        plotter.add_mesh(
            points,
            color=method.joint_color,
            point_size=args.point_size,
            render_points_as_spheres=True,
            label=f"{method.name} joints",
        )
        plotter.add_mesh(
            bones,
            color=method.line_color,
            line_width=args.line_width,
            render_lines_as_tubes=True,
            label=f"{method.name} bones" if index == 0 else None,
        )
        plotter.add_mesh(
            left_trail,
            color=method.left_color,
            line_width=5,
            label=f"{method.name} L",
        )
        plotter.add_mesh(
            right_trail,
            color=method.right_color,
            line_width=5,
            label=f"{method.name} R",
        )
        plotter.add_mesh(
            left_elbow_trail,
            color="cyan",
            line_width=3,
            label=f"{method.name} L elbow",
        )
        plotter.add_mesh(
            right_elbow_trail,
            color="magenta",
            line_width=3,
            label=f"{method.name} R elbow",
        )
        x_view = 0.08 + index * 0.22
        plotter.add_text(method.name, position=(x_view, 0.93), viewport=True, font_size=11)

    plotter.add_legend(size=(0.31, 0.34), loc="upper right")
    plotter.camera_position = [
        (0.0, -4.4, 1.65),
        (0.0, 0.0, 0.65),
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
    delay = 0.0 if playback_fps <= 0 else 1.0 / playback_fps
    while True:
        next_time = time.perf_counter()
        for local_index, frame in enumerate(frame_indices):
            for method_index, method in enumerate(selected):
                point_meshes[method_index].points = method.skeletons[local_index]
                bone_meshes[method_index].points = method.skeletons[local_index]
                _update_polyline(
                    left_trails[method_index],
                    _window(method.skeletons[:, L_WRIST], local_index, args.trail_length),
                )
                _update_polyline(
                    right_trails[method_index],
                    _window(method.skeletons[:, R_WRIST], local_index, args.trail_length),
                )
                _update_polyline(
                    left_elbow_trails[method_index],
                    _window(method.skeletons[:, L_ELBOW], local_index, args.trail_length),
                )
                _update_polyline(
                    right_elbow_trails[method_index],
                    _window(method.skeletons[:, R_ELBOW], local_index, args.trail_length),
                )
            plotter.add_text(
                f"episode={args.episode} frame={frame}/{len(df) - 1}",
                position="lower_left",
                color="white",
                font_size=11,
                name="frame_label",
            )
            plotter.update()
            if delay > 0:
                next_time += delay
                time.sleep(max(0.0, next_time - time.perf_counter()))
        if not args.loop:
            break

    plotter.show()


if __name__ == "__main__":
    main()
