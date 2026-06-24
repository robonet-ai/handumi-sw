#!/usr/bin/env python3
"""Compare controller-only upper-body reconstruction with PICO object trackers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from dexumi.retargeting.pico_upper_body import infer_elbow, parse_axis_map


UPPER_NAMES = [
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
UPPER_LINES = np.asarray(
    [
        2,
        0,
        1,
        2,
        1,
        2,
        2,
        1,
        3,
        2,
        3,
        4,
        2,
        4,
        5,
        2,
        1,
        6,
        2,
        6,
        7,
        2,
        7,
        8,
    ],
    dtype=np.int_,
)


def _load_fps(root: Path) -> float:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return 30.0
    with info_path.open("r", encoding="utf-8") as f:
        return float(json.load(f).get("fps", 30.0))


def _stack_pose_column(values: np.ndarray, *, width: int = 7) -> np.ndarray:
    poses = [np.asarray(value, dtype=np.float32) for value in values]
    stacked = np.stack(poses)
    if stacked.ndim != 2 or stacked.shape[1] < width:
        raise ValueError(f"Expected pose column with shape (N, >={width}); got {stacked.shape}")
    return stacked[:, :width]


def _stack_tracker_column(values: np.ndarray) -> np.ndarray:
    frames: list[np.ndarray] = []
    max_count = 0
    for value in values:
        frame = np.asarray(value, dtype=object)
        if frame.size == 0:
            arr = np.zeros((0, 7), dtype=np.float32)
        else:
            arr = np.stack([np.asarray(item, dtype=np.float32) for item in frame])
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            arr = arr[:, :7]
        frames.append(arr)
        max_count = max(max_count, len(arr))

    trackers = np.full((len(frames), max_count, 7), np.nan, dtype=np.float32)
    for index, frame in enumerate(frames):
        trackers[index, : len(frame), : frame.shape[1]] = frame
    return trackers


def load_episode(root: Path, episode: int) -> tuple[pd.DataFrame, float]:
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


def _normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return fallback.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _estimate_shoulders(
    heads: np.ndarray,
    left_wrists: np.ndarray,
    right_wrists: np.ndarray,
    *,
    shoulder_width: float,
    head_to_shoulder: float,
    head_to_chest: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    lateral_fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    chests: list[np.ndarray] = []
    necks: list[np.ndarray] = []
    left_shoulders: list[np.ndarray] = []
    right_shoulders: list[np.ndarray] = []
    laterals: list[np.ndarray] = []

    for head, left_wrist, right_wrist in zip(heads, left_wrists, right_wrists):
        lateral = left_wrist - right_wrist
        lateral = lateral - up * float(np.dot(lateral, up))
        lateral = _normalize(lateral, lateral_fallback)
        shoulder_center = head - up * head_to_shoulder

        chests.append(head - up * head_to_chest)
        necks.append(head - up * 0.12)
        left_shoulders.append(shoulder_center + lateral * (shoulder_width * 0.5))
        right_shoulders.append(shoulder_center - lateral * (shoulder_width * 0.5))
        laterals.append(lateral)

    return (
        np.asarray(chests, dtype=np.float32),
        np.asarray(necks, dtype=np.float32),
        np.asarray(left_shoulders, dtype=np.float32),
        np.asarray(right_shoulders, dtype=np.float32),
        np.asarray(laterals, dtype=np.float32),
    )


def _estimate_arm_lengths(
    shoulders: np.ndarray,
    wrists: np.ndarray,
    *,
    upper_ratio: float,
    extension_ratio: float,
    percentile: float,
) -> tuple[float, float]:
    distances = np.linalg.norm(wrists - shoulders, axis=1)
    distances = distances[np.isfinite(distances)]
    if len(distances) == 0:
        return 0.28, 0.28
    arm_length = float(np.percentile(distances, percentile) / extension_ratio)
    return arm_length * upper_ratio, arm_length * (1.0 - upper_ratio)


def reconstruct_upper_body(
    headset_pose: np.ndarray,
    left_controller_pose: np.ndarray,
    right_controller_pose: np.ndarray,
    *,
    shoulder_width: float,
    head_to_shoulder: float,
    head_to_chest: float,
    upper_ratio: float,
    extension_ratio: float,
    length_percentile: float,
    bend_forward: float,
    bend_down: float,
    bend_side: float,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    heads = headset_pose[:, :3].astype(np.float32)
    left_wrists = left_controller_pose[:, :3].astype(np.float32)
    right_wrists = right_controller_pose[:, :3].astype(np.float32)
    chests, necks, left_shoulders, right_shoulders, laterals = _estimate_shoulders(
        heads,
        left_wrists,
        right_wrists,
        shoulder_width=shoulder_width,
        head_to_shoulder=head_to_shoulder,
        head_to_chest=head_to_chest,
    )
    left_lengths = _estimate_arm_lengths(
        left_shoulders,
        left_wrists,
        upper_ratio=upper_ratio,
        extension_ratio=extension_ratio,
        percentile=length_percentile,
    )
    right_lengths = _estimate_arm_lengths(
        right_shoulders,
        right_wrists,
        upper_ratio=upper_ratio,
        extension_ratio=extension_ratio,
        percentile=length_percentile,
    )

    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    upper = np.empty((len(heads), len(UPPER_NAMES), 3), dtype=np.float32)
    for index in range(len(heads)):
        shoulder_center = 0.5 * (left_shoulders[index] + right_shoulders[index])
        wrist_center = 0.5 * (left_wrists[index] + right_wrists[index])
        forward = wrist_center - shoulder_center
        forward = forward - up * float(np.dot(forward, up))
        forward = _normalize(forward, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        left_hint = (
            forward * bend_forward
            + up * bend_down
            + laterals[index] * bend_side
        )
        right_hint = (
            forward * bend_forward
            + up * bend_down
            - laterals[index] * bend_side
        )

        left_elbow = infer_elbow(
            left_shoulders[index],
            left_wrists[index],
            upper_length=left_lengths[0],
            forearm_length=left_lengths[1],
            bend_hint=left_hint,
        )
        right_elbow = infer_elbow(
            right_shoulders[index],
            right_wrists[index],
            upper_length=right_lengths[0],
            forearm_length=right_lengths[1],
            bend_hint=right_hint,
        )
        upper[index] = np.stack(
            [
                chests[index],
                necks[index],
                heads[index],
                left_shoulders[index],
                left_elbow,
                left_wrists[index],
                right_shoulders[index],
                right_elbow,
                right_wrists[index],
            ]
        )
    return upper, left_lengths, right_lengths


def apply_display_transform(
    values: np.ndarray,
    *,
    origin: np.ndarray,
    axis_map: str,
    scale: float,
    offset: np.ndarray,
) -> np.ndarray:
    transform = parse_axis_map(axis_map)
    flat = np.asarray(values, dtype=np.float32).reshape(-1, 3)
    out = np.asarray(
        [transform(point - origin) * scale + offset for point in flat],
        dtype=np.float32,
    )
    return out.reshape(values.shape)


def polyline(points: np.ndarray) -> "pv.PolyData":
    import pyvista as pv

    valid = points[np.all(np.isfinite(points), axis=1)]
    if len(valid) < 2:
        return pv.PolyData(valid if len(valid) else np.zeros((1, 3), dtype=np.float32))
    poly = pv.PolyData(valid)
    poly.lines = np.concatenate([[len(valid)], np.arange(len(valid))]).astype(np.int_)
    return poly


def update_polyline(poly: "pv.PolyData", points: np.ndarray) -> None:
    valid = points[np.all(np.isfinite(points), axis=1)]
    if len(valid) < 2:
        poly.points = valid if len(valid) else np.zeros((1, 3), dtype=np.float32)
        poly.lines = np.empty(0, dtype=np.int_)
        return
    poly.points = valid
    poly.lines = np.concatenate([[len(valid)], np.arange(len(valid))]).astype(np.int_)


def trail(values: np.ndarray, frame: int, length: int) -> np.ndarray:
    start = 0 if length <= 0 else max(0, frame + 1 - length)
    return values[start : frame + 1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("outputs/datasets/pico_object_test"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--axis-map", default="z,x,y")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--spacing", type=float, default=1.1)
    parser.add_argument("--trail-length", type=int, default=180)
    parser.add_argument("--shoulder-width", type=float, default=0.38)
    parser.add_argument("--head-to-shoulder", type=float, default=0.22)
    parser.add_argument("--head-to-chest", type=float, default=0.45)
    parser.add_argument("--upper-ratio", type=float, default=0.44)
    parser.add_argument("--extension-ratio", type=float, default=0.92)
    parser.add_argument("--length-percentile", type=float, default=95.0)
    parser.add_argument("--bend-forward", type=float, default=0.65)
    parser.add_argument("--bend-down", type=float, default=-1.0)
    parser.add_argument("--bend-side", type=float, default=0.25)
    parser.add_argument("--background", default="black")
    parser.add_argument("--point-size", type=float, default=16.0)
    parser.add_argument("--line-width", type=float, default=4.0)
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    df, dataset_fps = load_episode(args.dataset_root, args.episode)
    left_controller = _stack_pose_column(df["observation.pico.left_controller_pose"].values)
    right_controller = _stack_pose_column(df["observation.pico.right_controller_pose"].values)
    headset = _stack_pose_column(df["observation.pico.headset_pose"].values)
    trackers = _stack_tracker_column(df["observation.pico.motion_tracker_pose"].values)
    counts = (
        np.asarray(df["observation.pico.motion_tracker_count"].values, dtype=np.int32)
        if "observation.pico.motion_tracker_count" in df
        else np.sum(np.isfinite(trackers[:, :, 0]), axis=1)
    )

    upper, left_lengths, right_lengths = reconstruct_upper_body(
        headset,
        left_controller,
        right_controller,
        shoulder_width=args.shoulder_width,
        head_to_shoulder=args.head_to_shoulder,
        head_to_chest=args.head_to_chest,
        upper_ratio=args.upper_ratio,
        extension_ratio=args.extension_ratio,
        length_percentile=args.length_percentile,
        bend_forward=args.bend_forward,
        bend_down=args.bend_down,
        bend_side=args.bend_side,
    )

    frame_indices = list(range(args.start_frame, len(df), args.stride))
    if args.max_frames is not None:
        frame_indices = frame_indices[: args.max_frames]
    if not frame_indices:
        raise ValueError("No frames selected.")

    origin = headset[frame_indices[0], :3].copy()
    upper_display = apply_display_transform(
        upper,
        origin=origin,
        axis_map=args.axis_map,
        scale=args.scale,
        offset=np.array([-args.spacing * 0.5, 0.0, 0.0], dtype=np.float32),
    )
    trackers_display = apply_display_transform(
        trackers[:, :, :3],
        origin=origin,
        axis_map=args.axis_map,
        scale=args.scale,
        offset=np.array([args.spacing * 0.5, 0.0, 0.0], dtype=np.float32),
    )

    print(
        "Loaded episode "
        f"{args.episode}: frames={len(df)}, selected={len(frame_indices)}, "
        f"fps={dataset_fps:g}, tracker_count min/max={int(np.nanmin(counts))}/{int(np.nanmax(counts))}"
    )
    print(
        "Inferred arm lengths "
        f"left upper/forearm={left_lengths[0]:.3f}/{left_lengths[1]:.3f} m, "
        f"right upper/forearm={right_lengths[0]:.3f}/{right_lengths[1]:.3f} m"
    )

    if args.smoke_test:
        return

    import pyvista as pv

    plotter = pv.Plotter(window_size=(1400, 900), off_screen=args.screenshot is not None)
    plotter.set_background(args.background)
    plotter.add_axes(line_width=2)
    plotter.add_text(
        "Mandos: upper body inferido",
        position=(40, 840),
        color="white",
        font_size=16,
    )
    plotter.add_text(
        "Object trackers: poses reales",
        position=(850, 840),
        color="white",
        font_size=16,
    )

    frame0 = frame_indices[0]
    upper_poly = pv.PolyData(upper_display[frame0])
    upper_poly.lines = UPPER_LINES
    plotter.add_mesh(
        upper_poly,
        color="white",
        line_width=args.line_width,
        render_lines_as_tubes=True,
        name="upper_lines",
    )
    plotter.add_points(
        upper_display[frame0],
        color="crimson",
        point_size=args.point_size,
        render_points_as_spheres=True,
        name="upper_points",
    )

    tracker_points = trackers_display[frame0]
    plotter.add_points(
        tracker_points,
        color="deepskyblue",
        point_size=args.point_size + 2.0,
        render_points_as_spheres=True,
        name="tracker_points",
    )
    tracker_line = pv.PolyData(tracker_points)
    if len(tracker_points) >= 2:
        tracker_line.lines = np.asarray([2, 0, 1], dtype=np.int_)
    plotter.add_mesh(
        tracker_line,
        color="deepskyblue",
        line_width=args.line_width,
        render_lines_as_tubes=True,
        name="tracker_line",
    )

    left_wrist_trail = polyline(upper_display[: frame0 + 1, 5])
    right_wrist_trail = polyline(upper_display[: frame0 + 1, 8])
    plotter.add_mesh(left_wrist_trail, color="lime", line_width=3.0, name="left_controller_trail")
    plotter.add_mesh(right_wrist_trail, color="orange", line_width=3.0, name="right_controller_trail")
    tracker_trails = []
    for tracker_index in range(trackers_display.shape[1]):
        tracker_trail = polyline(trackers_display[: frame0 + 1, tracker_index])
        tracker_trails.append(tracker_trail)
        color = "cyan" if tracker_index == 0 else "dodgerblue"
        plotter.add_mesh(tracker_trail, color=color, line_width=3.0, name=f"tracker_trail_{tracker_index}")

    floor = pv.Plane(center=(0.0, 0.0, -0.02), direction=(0, 0, 1), i_size=2.8, j_size=1.4)
    plotter.add_mesh(floor, color=(0.35, 0.35, 0.35), opacity=0.45)
    plotter.camera_position = "xy"
    plotter.camera.zoom(1.15)

    if args.screenshot is not None:
        plotter.show(auto_close=False)
        plotter.screenshot(str(args.screenshot))
        plotter.close()
        print(f"Screenshot saved to {args.screenshot}")
        return

    plotter.show(auto_close=False, interactive_update=True)
    dt = 1.0 / float(args.fps or dataset_fps)
    while True:
        for selected_index, frame in enumerate(frame_indices):
            upper_poly.points = upper_display[frame]
            plotter.remove_actor("upper_points")
            plotter.add_points(
                upper_display[frame],
                color="crimson",
                point_size=args.point_size,
                render_points_as_spheres=True,
                name="upper_points",
            )

            current_trackers = trackers_display[frame]
            plotter.remove_actor("tracker_points")
            plotter.add_points(
                current_trackers,
                color="deepskyblue",
                point_size=args.point_size + 2.0,
                render_points_as_spheres=True,
                name="tracker_points",
            )
            tracker_line.points = current_trackers
            if len(current_trackers) >= 2:
                tracker_line.lines = np.asarray([2, 0, 1], dtype=np.int_)

            update_polyline(left_wrist_trail, trail(upper_display[:, 5], frame, args.trail_length))
            update_polyline(right_wrist_trail, trail(upper_display[:, 8], frame, args.trail_length))
            for tracker_index, tracker_trail in enumerate(tracker_trails):
                update_polyline(
                    tracker_trail,
                    trail(trackers_display[:, tracker_index], frame, args.trail_length),
                )

            plotter.add_text(
                f"episode={args.episode} frame={frame}/{len(df) - 1}",
                position="lower_left",
                color="white",
                font_size=12,
                name="frame_label",
            )
            plotter.update()
            time.sleep(dt)
            if selected_index == len(frame_indices) - 1 and not args.loop:
                plotter.close()
                return


if __name__ == "__main__":
    main()
