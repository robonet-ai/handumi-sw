#!/usr/bin/env python3
"""Visualize raw PICO controller/object-tracker and HMD trajectories."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _load_fps(root: Path) -> float:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return 30.0
    with info_path.open("r", encoding="utf-8") as f:
        return float(json.load(f).get("fps", 30.0))


def load_episode(root: Path, episode: int) -> tuple[pd.DataFrame, float]:
    parquet_files = sorted((root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")

    df = pd.concat([pd.read_parquet(path) for path in parquet_files], ignore_index=True)
    if "episode_index" in df:
        df = df[df["episode_index"] == episode].copy()
    if df.empty:
        raise ValueError(f"Episode {episode} not found in {root}")

    sort_columns = [col for col in ("index", "frame_index") if col in df.columns]
    if sort_columns:
        df.sort_values(sort_columns, inplace=True)
    return df.reset_index(drop=True), _load_fps(root)


def stack_pose_column(values: np.ndarray, *, width: int = 7) -> np.ndarray:
    poses = [np.asarray(value, dtype=np.float32) for value in values]
    stacked = np.stack(poses)
    if stacked.ndim != 2 or stacked.shape[1] < width:
        raise ValueError(f"Expected pose column with shape (N, >={width}); got {stacked.shape}")
    return stacked[:, :width]


def stack_tracker_column(values: np.ndarray) -> np.ndarray:
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


def order_trackers_by_controller_side(
    trackers: np.ndarray,
    left_controller: np.ndarray,
    right_controller: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int], tuple[float, float]]:
    if trackers.shape[1] < 2:
        empty = np.full((len(trackers), 2, 7), np.nan, dtype=np.float32)
        return empty, (-1, -1), (float("nan"), float("nan"))

    positions = trackers[:, :2, :3]
    left = left_controller[:, :3]
    right = right_controller[:, :3]

    identity_left = np.linalg.norm(positions[:, 0] - left, axis=1)
    identity_right = np.linalg.norm(positions[:, 1] - right, axis=1)
    swap_left = np.linalg.norm(positions[:, 1] - left, axis=1)
    swap_right = np.linalg.norm(positions[:, 0] - right, axis=1)

    identity_cost = float(np.nanmedian(identity_left + identity_right))
    swap_cost = float(np.nanmedian(swap_left + swap_right))
    if identity_cost <= swap_cost:
        return trackers[:, [0, 1], :].copy(), (0, 1), (
            float(np.nanmedian(identity_left)),
            float(np.nanmedian(identity_right)),
        )
    return trackers[:, [1, 0], :].copy(), (1, 0), (
        float(np.nanmedian(swap_left)),
        float(np.nanmedian(swap_right)),
    )


def parse_axis_map(axis_map: str):
    tokens = [token.strip() for token in axis_map.split(",")]
    if len(tokens) != 3:
        raise ValueError("--axis-map must contain exactly three comma-separated axes")

    axes = {"x": 0, "y": 1, "z": 2}
    parsed: list[tuple[int, float]] = []
    for token in tokens:
        sign = -1.0 if token.startswith("-") else 1.0
        axis = token[1:] if token.startswith("-") else token
        if axis not in axes:
            raise ValueError(f"Unknown axis token {token!r}")
        parsed.append((axes[axis], sign))

    def transform(point: np.ndarray) -> np.ndarray:
        return np.asarray([sign * point[index] for index, sign in parsed], dtype=np.float32)

    return transform


def apply_display_transform(
    points: np.ndarray,
    *,
    origin: np.ndarray,
    axis_map: str,
    scale: float,
) -> np.ndarray:
    transform = parse_axis_map(axis_map)
    flat = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    out = np.asarray([transform(point - origin) * scale for point in flat], dtype=np.float32)
    return out.reshape(points.shape)


def polyline(points: np.ndarray) -> "pv.PolyData":
    import pyvista as pv

    valid_points: list[np.ndarray] = []
    lines: list[int] = []
    run: list[int] = []
    for point in np.asarray(points, dtype=np.float32):
        if np.all(np.isfinite(point)):
            run.append(len(valid_points))
            valid_points.append(point)
            continue
        if len(run) >= 2:
            lines.extend([len(run), *run])
        run = []
    if len(run) >= 2:
        lines.extend([len(run), *run])

    poly = pv.PolyData(
        np.asarray(valid_points, dtype=np.float32)
        if valid_points
        else np.zeros((1, 3), dtype=np.float32)
    )
    poly.lines = np.asarray(lines, dtype=np.int_)
    return poly


def update_polyline(poly: "pv.PolyData", points: np.ndarray) -> None:
    updated = polyline(points)
    poly.points = updated.points
    poly.lines = updated.lines


def valid_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    valid = points[np.all(np.isfinite(points), axis=1)]
    return valid if len(valid) else np.zeros((1, 3), dtype=np.float32)


def trail(values: np.ndarray, frame: int, length: int) -> np.ndarray:
    start = 0 if length <= 0 else max(0, frame + 1 - length)
    return values[start : frame + 1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--ee-source",
        choices=("trackers", "controllers", "both"),
        default="trackers",
        help="Which PICO inputs to show as end-effectors.",
    )
    parser.add_argument("--axis-map", default="z,x,y")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--trail-length", type=int, default=300)
    parser.add_argument("--point-size", type=float, default=18.0)
    parser.add_argument("--line-width", type=float, default=5.0)
    parser.add_argument("--background", default="black")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    df, dataset_fps = load_episode(args.dataset_root, args.episode)
    fps = float(args.fps or dataset_fps)
    max_frames = args.max_frames
    if max_frames is None and args.seconds > 0:
        max_frames = max(1, int(round(args.seconds * dataset_fps / args.stride)))

    left_controller = stack_pose_column(df["observation.pico.left_controller_pose"].values)
    right_controller = stack_pose_column(df["observation.pico.right_controller_pose"].values)
    headset = stack_pose_column(df["observation.pico.headset_pose"].values)
    trackers = stack_tracker_column(df["observation.pico.motion_tracker_pose"].values)
    ordered_trackers, mapping, median_distances = order_trackers_by_controller_side(
        trackers, left_controller, right_controller
    )

    frame_indices = list(range(args.start_frame, len(df), args.stride))
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]
    if not frame_indices:
        raise ValueError("No frames selected.")

    origin = headset[frame_indices[0], :3].copy()
    controller_points = np.stack(
        [left_controller[:, :3], right_controller[:, :3]], axis=1
    )
    tracker_points = ordered_trackers[:, :, :3]
    headset_points = headset[:, :3]

    controller_display = apply_display_transform(
        controller_points[frame_indices],
        origin=origin,
        axis_map=args.axis_map,
        scale=args.scale,
    )
    tracker_display = apply_display_transform(
        tracker_points[frame_indices],
        origin=origin,
        axis_map=args.axis_map,
        scale=args.scale,
    )
    headset_display = apply_display_transform(
        headset_points[frame_indices],
        origin=origin,
        axis_map=args.axis_map,
        scale=args.scale,
    )

    first_groups = [headset_display[0:1]]
    if args.ee_source in {"controllers", "both"}:
        first_groups.append(controller_display[0])
    if args.ee_source in {"trackers", "both"}:
        first_groups.append(valid_points(tracker_display[0]))
    all_first = np.concatenate(first_groups, axis=0)
    floor_z = float(np.nanmin(all_first[:, 2]))
    controller_display[:, :, 2] -= floor_z
    tracker_display[:, :, 2] -= floor_z
    headset_display[:, 2] -= floor_z

    print(
        "Loaded episode "
        f"{args.episode}: frames={len(df)}, selected={len(frame_indices)}, "
        f"seconds={len(frame_indices) * args.stride / dataset_fps:.2f}, fps={dataset_fps:g}"
    )
    print(
        "Object tracker side mapping: "
        f"left=tracker[{mapping[0]}] median_dist={median_distances[0]:.3f}m, "
        f"right=tracker[{mapping[1]}] median_dist={median_distances[1]:.3f}m"
    )

    if args.smoke_test:
        return

    import pyvista as pv

    plotter = pv.Plotter(window_size=(1500, 850), off_screen=args.screenshot is not None)
    plotter.set_background(args.background)
    plotter.add_axes()
    plotter.add_floor("-z", color="gray", lighting=False, pad=1.0)

    headset_poly = pv.PolyData(headset_display[0:1])

    headset_trail = polyline(headset_display[:1])

    ee_meshes: list[tuple[str, str, "pv.PolyData", "pv.PolyData", np.ndarray]] = []
    if args.ee_source in {"controllers", "both"}:
        left_controller_poly = pv.PolyData(controller_display[0, 0:1])
        right_controller_poly = pv.PolyData(controller_display[0, 1:2])
        left_controller_trail = polyline(controller_display[:1, 0])
        right_controller_trail = polyline(controller_display[:1, 1])
        ee_meshes.extend(
            [
                (
                    "EE L mando",
                    "#00ff38",
                    left_controller_poly,
                    left_controller_trail,
                    controller_display[:, 0],
                ),
                (
                    "EE R mando",
                    "#ffb000",
                    right_controller_poly,
                    right_controller_trail,
                    controller_display[:, 1],
                ),
            ]
        )
    if args.ee_source in {"trackers", "both"}:
        left_tracker_poly = pv.PolyData(valid_points(tracker_display[0, 0:1]))
        right_tracker_poly = pv.PolyData(valid_points(tracker_display[0, 1:2]))
        left_tracker_trail = polyline(tracker_display[:1, 0])
        right_tracker_trail = polyline(tracker_display[:1, 1])
        ee_meshes.extend(
            [
                (
                    "EE L tracker",
                    "#00c8ff",
                    left_tracker_poly,
                    left_tracker_trail,
                    tracker_display[:, 0],
                ),
                (
                    "EE R tracker",
                    "#ff00f5",
                    right_tracker_poly,
                    right_tracker_trail,
                    tracker_display[:, 1],
                ),
            ]
        )

    for label, color, point_poly, trail_poly, _trajectory in ee_meshes:
        plotter.add_mesh(trail_poly, color=color, line_width=args.line_width, label=label)
        plotter.add_mesh(
            point_poly,
            color=color,
            point_size=args.point_size,
            render_points_as_spheres=True,
        )

    plotter.add_mesh(
        headset_trail,
        color="#ffffff",
        line_width=max(2.0, args.line_width - 2.0),
        label="HMD",
    )
    plotter.add_mesh(
        headset_poly,
        color="#ffffff",
        point_size=args.point_size * 0.8,
        render_points_as_spheres=True,
    )

    legend_entries = [[label, color] for label, color, *_rest in ee_meshes]
    legend_entries.append(["HMD", "#ffffff"])
    plotter.add_legend(
        legend_entries,
        bcolor="black",
        border=False,
        size=(0.25, 0.18),
        loc="upper right",
    )

    text = plotter.add_text(
        f"episode={args.episode} frame={frame_indices[0]}/{len(df) - 1}",
        position="lower_left",
        color="white",
        font_size=14,
    )

    def update(local_index: int) -> None:
        frame = frame_indices[local_index]
        headset_poly.points = headset_display[local_index : local_index + 1]

        for _label, _color, point_poly, trail_poly, trajectory in ee_meshes:
            point_poly.points = valid_points(trajectory[local_index : local_index + 1])
            update_polyline(trail_poly, trail(trajectory, local_index, args.trail_length))
        update_polyline(headset_trail, trail(headset_display, local_index, args.trail_length))
        text.SetText(2, f"episode={args.episode} frame={frame}/{len(df) - 1}")

    if args.screenshot is not None:
        update(len(frame_indices) - 1)
        plotter.show(auto_close=False)
        plotter.screenshot(str(args.screenshot))
        plotter.close()
        return

    plotter.show(auto_close=False, interactive_update=True)
    local_index = 0
    while True:
        update(local_index)
        plotter.update()
        local_index += 1
        if local_index >= len(frame_indices):
            if not args.loop:
                break
            local_index = 0
        time.sleep(max(0.0, 1.0 / max(1e-6, fps)))
    plotter.show()


if __name__ == "__main__":
    main()
