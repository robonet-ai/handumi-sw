#!/usr/bin/env python3
"""View a synchronized HandUMI episode in Rerun.

Controllers, optional calibrated TCPs, HMD, cameras, canonical body, CoM,
contacts, support, provenance, timing quality, and diagnostics share explicit
``episode_frame`` and ``episode_time`` cursors. Full paths are planned once in
linear time and logged as deterministic bounded chunks.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from handumi.dataset.raw import LEFT_POSE_SLICE, RIGHT_POSE_SLICE
from handumi.dataset.reader import (
    RawEpisode,
    dataset_root_from_repo_id,
    handumi_metadata,
    load_raw_episode,
)
from handumi.calibration.control_tcp import (
    apply_controller_tcp_calibration,
    controller_tcp_calibration_from_metadata,
)
from handumi.visualization import LEFT_COLOR, RIGHT_COLOR
from handumi.visualization.body import (
    WHOLE_COM_TRAIL_PATH,
    body_frame_at,
    body_render_plan,
    full_trajectory_plan,
)
from handumi.visualization.controller_trajectory import (
    HMD_ROOT,
    LEFT_WIDTH_PATH,
    RIGHT_WIDTH_PATH,
    RenderOp,
    RerunSink,
    controller_current_plan,
    controller_path,
    initialize_rerun,
)

log = logging.getLogger("handumi.view_trajectory")

FRAME_TIMELINE = "episode_frame"
TIME_TIMELINE = "episode_time"


@dataclass(frozen=True)
class ViewerOptions:
    temporal_decimation: int = 1
    spatial_decimation_m: float = 0.0
    trail_point_cap: int = 2048
    trail_duration_s: float = 10.0


@dataclass(frozen=True)
class EpisodeRenderStats:
    frames: int
    full_trajectory_operations: int
    body_present: bool
    camera_names: tuple[str, ...]
    tcp_source: str | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id", required=True, help="LeRobot dataset repository id."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Local dataset root (default: outputs/datasets/<repo name>).",
    )
    parser.add_argument("--revision", default="main", help="Dataset revision.")
    parser.add_argument("--episode", type=int, required=True, help="Episode index.")
    parser.add_argument(
        "--spawn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Spawn the Rerun viewer; use --no-spawn for headless export.",
    )
    parser.add_argument(
        "--rrd", type=Path, default=None, help="Optional .rrd output path."
    )
    parser.add_argument(
        "--video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Download/decode and display recorded camera streams.",
    )
    parser.add_argument(
        "--trail-duration-s",
        type=float,
        default=10.0,
        help="Maximum source-time span per full-trajectory chunk; paths remain complete.",
    )
    parser.add_argument(
        "--trail-point-cap",
        type=int,
        default=2048,
        help="Maximum points per trajectory chunk (minimum 2).",
    )
    parser.add_argument(
        "--temporal-decimation",
        type=int,
        default=1,
        help="Keep every Nth sample within each valid trajectory run.",
    )
    parser.add_argument(
        "--spatial-decimation-m",
        type=float,
        default=0.0,
        help="Minimum distance between retained trajectory samples in meters.",
    )
    return parser.parse_args(argv)


def _frame_signal(
    signals: dict[str, np.ndarray], key: str, count: int
) -> np.ndarray | None:
    value = signals.get(key)
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    return array.reshape(-1) if array.size == count else None


def _tracked_mask(episode: RawEpisode, side: str) -> np.ndarray:
    count = len(episode.states)
    tracked = _frame_signal(
        episode.signals, f"observation.tracking.{side}_tracked", count
    )
    if tracked is None:
        pose_slice = LEFT_POSE_SLICE if side == "left" else RIGHT_POSE_SLICE
        return np.all(np.isfinite(episode.states[:, pose_slice]), axis=1)
    return tracked.astype(bool)


def _hmd_data(episode: RawEpisode) -> tuple[np.ndarray | None, np.ndarray | None]:
    poses = episode.signals.get("observation.tracking.hmd_pose")
    if poses is None:
        return None, None
    pose_array = np.asarray(poses, dtype=np.float32)
    if pose_array.shape != (len(episode.states), 7):
        return None, None
    validity = np.asarray(episode.signals.get("observation.valid"))
    if validity.shape == (len(episode.states), 8):
        mask = validity[:, 4].astype(bool)
    else:
        mask = np.all(np.isfinite(pose_array[:, :3]), axis=1)
    return pose_array, mask


def _controller_trajectories(
    episode: RawEpisode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, str | None]:
    raw_left = np.asarray(episode.states[:, LEFT_POSE_SLICE], dtype=np.float32)
    raw_right = np.asarray(episode.states[:, RIGHT_POSE_SLICE], dtype=np.float32)
    snapshot = handumi_metadata(episode.metadata).get("controller_tcp_calibration")
    if not isinstance(snapshot, dict):
        return raw_left, raw_right, None, None, None
    if snapshot.get("applied_to_state") is True:
        return (
            raw_left,
            raw_right,
            raw_left.copy(),
            raw_right.copy(),
            "state already stores TCP",
        )
    try:
        calibration = controller_tcp_calibration_from_metadata(snapshot)
        left_tcp, right_tcp = apply_controller_tcp_calibration(
            raw_left, raw_right, calibration
        )
    except (TypeError, ValueError) as exc:
        log.warning("Ignoring invalid dataset controller-to-TCP snapshot: %s", exc)
        return raw_left, raw_right, None, None, None
    sha256 = str(snapshot.get("sha256", "unknown"))
    return raw_left, raw_right, left_tcp, right_tcp, f"dataset snapshot sha256={sha256}"


def _widths_mm(episode: RawEpisode, side: str) -> np.ndarray:
    count = len(episode.states)
    recorded = _frame_signal(
        episode.signals, f"observation.feetech.{side}_width_mm", count
    )
    if recorded is not None:
        return np.nan_to_num(recorded.astype(np.float32), nan=0.0)
    index = 14 if side == "left" else 15
    return np.nan_to_num(episode.states[:, index] * 1000.0, nan=0.0)


def _full_trajectory_ops(
    episode: RawEpisode,
    options: ViewerOptions,
    raw_left: np.ndarray,
    raw_right: np.ndarray,
    left_tcp: np.ndarray | None,
    right_tcp: np.ndarray | None,
) -> tuple[RenderOp, ...]:
    fps = max(1.0, float(episode.fps))
    duration_frames = (
        None
        if options.trail_duration_s <= 0
        else max(1, int(round(options.trail_duration_s * fps)))
    )
    common = {
        "temporal_step": max(1, options.temporal_decimation),
        "spatial_step_m": max(0.0, options.spatial_decimation_m),
        "point_cap": max(2, options.trail_point_cap),
        "duration_frames": duration_frames,
    }
    left_valid = _tracked_mask(episode, "left")
    right_valid = _tracked_mask(episode, "right")
    operations: list[RenderOp] = []
    for side, raw, tcp, valid, color in (
        ("left", raw_left, left_tcp, left_valid, LEFT_COLOR),
        ("right", raw_right, right_tcp, right_valid, RIGHT_COLOR),
    ):
        operations.extend(
            full_trajectory_plan(
                controller_path(side, "raw_trail"),
                raw[:, :3],
                valid,
                color=(*color, 90),
                radius=0.0015,
                **common,
            )
        )
        if tcp is not None:
            operations.extend(
                full_trajectory_plan(
                    controller_path(side, "trail"),
                    tcp[:, :3],
                    valid,
                    color=color,
                    radius=0.003,
                    **common,
                )
            )
    hmd, hmd_valid = _hmd_data(episode)
    if hmd is not None:
        operations.extend(
            full_trajectory_plan(
                f"{HMD_ROOT}/trail",
                hmd[:, :3],
                hmd_valid,
                color=(120, 170, 255, 150),
                radius=0.002,
                **common,
            )
        )
    if episode.body is not None:
        signals = episode.body.signals
        whole = signals.get("observation.body.whole_com")
        valid = signals.get("observation.body.whole_com_valid")
        if whole is not None:
            operations.extend(
                full_trajectory_plan(
                    WHOLE_COM_TRAIL_PATH,
                    whole,
                    valid,
                    color=(245, 245, 245, 180),
                    radius=0.003,
                    **common,
                )
            )
    return tuple(operations)


def log_episode(
    rr: Any,
    episode: RawEpisode,
    *,
    options: ViewerOptions | None = None,
) -> EpisodeRenderStats:
    """Log a loaded episode with one synchronized cursor update per frame."""
    options = options or ViewerOptions()
    sink = RerunSink(rr)
    raw_left, raw_right, left_tcp, right_tcp, tcp_source = _controller_trajectories(
        episode
    )
    full_ops = _full_trajectory_ops(
        episode, options, raw_left, raw_right, left_tcp, right_tcp
    )
    sink.emit(full_ops)
    count = len(episode.states)
    left_valid = _tracked_mask(episode, "left")
    right_valid = _tracked_mask(episode, "right")
    hmd, hmd_valid = _hmd_data(episode)
    left_widths = _widths_mm(episode, "left")
    right_widths = _widths_mm(episode, "right")

    for frame_index in range(count):
        rr.set_time(FRAME_TIMELINE, sequence=frame_index)
        rr.set_time(TIME_TIMELINE, duration=frame_index / float(episode.fps))
        operations: list[RenderOp] = [
            RenderOp(LEFT_WIDTH_PATH, "scalars", float(left_widths[frame_index])),
            RenderOp(RIGHT_WIDTH_PATH, "scalars", float(right_widths[frame_index])),
        ]
        for side, raw, tcp, valid, color in (
            ("left", raw_left, left_tcp, left_valid, LEFT_COLOR),
            ("right", raw_right, right_tcp, right_valid, RIGHT_COLOR),
        ):
            operations.extend(
                controller_current_plan(
                    side,
                    raw[frame_index],
                    tcp_pose7=None if tcp is None else tcp[frame_index],
                    color=color,
                    tracked=bool(valid[frame_index]),
                )
            )
        if hmd is not None and hmd_valid is not None and bool(hmd_valid[frame_index]):
            point = hmd[frame_index, :3]
            if np.all(np.isfinite(point)):
                operations.append(
                    RenderOp(
                        HMD_ROOT,
                        "points3d",
                        np.asarray([point]),
                        {
                            "colors": [[120, 170, 255, 230]],
                            "radii": 0.014,
                            "labels": ["HMD"],
                        },
                    )
                )
            else:
                operations.append(RenderOp(HMD_ROOT, "clear"))
        else:
            operations.append(RenderOp(HMD_ROOT, "clear"))
        if episode.body is not None:
            operations.extend(
                body_render_plan(
                    body_frame_at(episode.body.signals, frame_index),
                    trail=None,
                    log_trail=False,
                )
            )
        for key, images in episode.images.items():
            if frame_index < len(images):
                operations.append(
                    RenderOp(key, "image", images[frame_index], {"jpeg_quality": 75})
                )
        sink.emit(operations)

    camera_names = tuple(
        key.removeprefix("observation.images.") for key in sorted(episode.images)
    )
    return EpisodeRenderStats(
        frames=count,
        full_trajectory_operations=len(full_ops),
        body_present=episode.body is not None,
        camera_names=camera_names,
        tcp_source=tcp_source,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    root = args.root or dataset_root_from_repo_id(args.repo_id)
    episode = load_raw_episode(
        repo_id=args.repo_id,
        root=root,
        revision=args.revision,
        episode=args.episode,
        download_videos=args.video,
    )
    camera_names = [
        key.removeprefix("observation.images.") for key in sorted(episode.images)
    ]
    stream = initialize_rerun(
        "handumi_view_trajectory",
        camera_names,
        fps=max(1, int(round(episode.fps))),
        spawn=args.spawn,
        recorder_status=False,
        include_quality=episode.body is not None,
        save_path=args.rrd,
        timeline=TIME_TIMELINE,
        on_error=lambda exc: log.error("Rerun initialization failed: %s", exc),
    )
    if stream is None:
        raise SystemExit("Could not initialize Rerun.")
    stats = log_episode(
        stream.rr,
        episode,
        options=ViewerOptions(
            temporal_decimation=args.temporal_decimation,
            spatial_decimation_m=args.spatial_decimation_m,
            trail_point_cap=args.trail_point_cap,
            trail_duration_s=args.trail_duration_s,
        ),
    )
    disconnect = getattr(stream.rr, "disconnect", None)
    if callable(disconnect):
        disconnect()
    log.info(
        "Rendered episode %d: %d frames, body=%s, cameras=%d, TCP=%s",
        args.episode,
        stats.frames,
        stats.body_present,
        len(stats.camera_names),
        stats.tcp_source or "unavailable (raw controller shown only)",
    )


if __name__ == "__main__":
    main()
