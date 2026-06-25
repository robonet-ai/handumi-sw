"""Write LeRobot v3.0 datasets from IK-solved joint trajectories.

The writer produces the on-disk layout expected by ``LeRobotDataset``::

    <root>/
    ├── data/
    │   └── chunk-{chunk_index:03d}/
    │       └── file-{file_index:03d}.parquet   (one per episode)
    ├── meta/
    │   ├── info.json
    │   ├── stats.json
    │   ├── tasks.parquet
    │   └── episodes/
    │       └── chunk-{chunk_index:03d}/
    │           └── file-{file_index:03d}.parquet
    └── videos/                                  (copied from source)
        └── {video_key}/
            └── chunk-{chunk_index:03d}/
                └── file-{file_index:03d}.mp4

Usage
-----
::

    from dexumi.dataset import EpisodeResult, write_dataset

    episodes = [
        EpisodeResult(
            episode_index=0,
            states=np.zeros((100, 14), dtype=np.float32),
            actions=np.zeros((100, 14), dtype=np.float32),
            task="Pick and place cube",
        ),
    ]
    write_dataset(
        output_root=Path("outputs/my-dataset"),
        source_root=Path("outputs/source-dataset"),
        source_info=source_info_dict,
        episodes=episodes,
        robot_type="bi_piper_follower",
        joint_names=["left_shoulder_pan.pos", ...],
        fps=30,
    )
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dexumi.dataset.schema import CHUNKS_SIZE, chunk_and_file


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    """IK-solved data for one episode.

    Attributes
    ----------
    episode_index:
        Original episode index (used to locate the source video files).
    states:
        Joint angles at time *t*, shape ``(T, N_joints)`` float32.
        Already has the last frame removed (T = source_frames - 1).
    actions:
        Joint angles at time *t+1*, shape ``(T, N_joints)`` float32.
    task:
        Natural-language task description for this episode.
    source_episode_index:
        Episode index in the *source* dataset.  Defaults to ``episode_index``
        when the output dataset preserves the original numbering.
    """

    episode_index: int
    states: np.ndarray
    actions: np.ndarray
    task: str
    source_episode_index: int = -1

    def __post_init__(self) -> None:
        if self.source_episode_index < 0:
            self.source_episode_index = self.episode_index


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_feature_stats(values: np.ndarray) -> dict[str, Any]:
    """Compute LeRobot-style statistics for a 2-D float array (frames × dim)."""
    flat = values.reshape(-1) if values.ndim == 1 else values.reshape(len(values), -1)
    if flat.ndim == 1:
        flat = flat[:, None]

    min_vals = flat.min(axis=0).tolist()
    max_vals = flat.max(axis=0).tolist()
    mean_vals = flat.mean(axis=0).tolist()
    std_vals = flat.std(axis=0).tolist()
    count = [int(len(flat))] * flat.shape[1] if flat.shape[1] > 1 else [int(len(flat))]
    q01 = np.percentile(flat, 1, axis=0).tolist()
    q10 = np.percentile(flat, 10, axis=0).tolist()
    q50 = np.percentile(flat, 50, axis=0).tolist()
    q90 = np.percentile(flat, 90, axis=0).tolist()
    q99 = np.percentile(flat, 99, axis=0).tolist()

    return {
        "min": min_vals,
        "max": max_vals,
        "mean": mean_vals,
        "std": std_vals,
        "count": count,
        "q01": q01,
        "q10": q10,
        "q50": q50,
        "q90": q90,
        "q99": q99,
    }


def _scalar_stats(values: np.ndarray) -> dict[str, Any]:
    """Stats for a 1-D scalar sequence (timestamp, frame_index, …)."""
    v = np.asarray(values, dtype=np.float64)
    return {
        "min": [float(v.min())],
        "max": [float(v.max())],
        "mean": [float(v.mean())],
        "std": [float(v.std())],
        "count": [int(len(v))],
        "q01": [float(np.percentile(v, 1))],
        "q10": [float(np.percentile(v, 10))],
        "q50": [float(np.percentile(v, 50))],
        "q90": [float(np.percentile(v, 90))],
        "q99": [float(np.percentile(v, 99))],
    }


def _video_placeholder_stats(shape: list[int]) -> dict[str, Any]:
    """Placeholder statistics for a video feature (pixel values in [0,1])."""
    c = shape[2] if len(shape) == 3 else 1
    return {
        "min": [[[0.0]] * c],
        "max": [[[1.0]] * c],
        "mean": [[[0.5]] * c],
        "std": [[[0.0]] * c],
        "count": [1],
        "q01": [[[0.0]] * c],
        "q10": [[[0.1]] * c],
        "q50": [[[0.5]] * c],
        "q90": [[[0.9]] * c],
        "q99": [[[1.0]] * c],
    }


def _build_info_json(
    *,
    robot_type: str,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    fps: int,
    joint_names: list[str],
    video_features: dict[str, Any],
    chunks_size: int = CHUNKS_SIZE,
) -> dict[str, Any]:
    """Build the ``info.json`` dict for a new LeRobot dataset."""

    n = len(joint_names)
    state_action_feature = {
        "dtype": "float32",
        "shape": [n],
        "names": joint_names,
    }

    features: dict[str, Any] = {
        "action": state_action_feature,
        "observation.state": state_action_feature,
    }
    features.update(video_features)
    features["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    features["frame_index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["episode_index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["task_index"] = {"dtype": "int64", "shape": [1], "names": None}

    return {
        "codebase_version": "v3.0",
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "chunks_size": chunks_size,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 500,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }


def _build_stats_json(
    *,
    all_states: np.ndarray,
    all_actions: np.ndarray,
    all_timestamps: np.ndarray,
    all_frame_indices: np.ndarray,
    all_episode_indices: np.ndarray,
    all_global_indices: np.ndarray,
    all_task_indices: np.ndarray,
    video_features: dict[str, Any],
) -> dict[str, Any]:
    """Compute global statistics across all episodes."""

    stats: dict[str, Any] = {}

    for key, vals in (
        ("observation.state", all_states),
        ("action", all_actions),
    ):
        stats[key] = _compute_feature_stats(vals)

    for key, vals in (
        ("timestamp", all_timestamps),
        ("frame_index", all_frame_indices),
        ("episode_index", all_episode_indices),
        ("index", all_global_indices),
        ("task_index", all_task_indices),
    ):
        stats[key] = _scalar_stats(vals)

    for video_key, feat in video_features.items():
        stats[video_key] = _video_placeholder_stats(feat["shape"])

    return stats


# ---------------------------------------------------------------------------
# Episodes parquet helpers
# ---------------------------------------------------------------------------


def _episode_parquet_row(
    *,
    ep: EpisodeResult,
    episode_length: int,
    dataset_from_index: int,
    dataset_to_index: int,
    fps: int,
    video_keys: list[str],
    chunks_size: int = CHUNKS_SIZE,
) -> dict[str, Any]:
    """Build one row for the episodes meta parquet."""

    chunk_idx, file_idx = chunk_and_file(ep.episode_index, chunks_size)
    meta_chunk_idx, meta_file_idx = chunk_and_file(ep.episode_index, chunks_size)

    row: dict[str, Any] = {
        "episode_index": ep.episode_index,
        "tasks": [ep.task],
        "length": episode_length,
        "data/chunk_index": chunk_idx,
        "data/file_index": file_idx,
        "dataset_from_index": dataset_from_index,
        "dataset_to_index": dataset_to_index,
    }

    for vk in video_keys:
        src_chunk, src_file = chunk_and_file(ep.source_episode_index, chunks_size)
        from_ts = 0.0
        to_ts = float(episode_length - 1) / fps
        row[f"videos/{vk}/chunk_index"] = src_chunk
        row[f"videos/{vk}/file_index"] = src_file
        row[f"videos/{vk}/from_timestamp"] = from_ts
        row[f"videos/{vk}/to_timestamp"] = to_ts

    row["meta/episodes/chunk_index"] = meta_chunk_idx
    row["meta/episodes/file_index"] = meta_file_idx

    for col_prefix, data in (
        ("observation.state", ep.states),
        ("action", ep.actions),
    ):
        ep_stats = _compute_feature_stats(data)
        for stat_name, val in ep_stats.items():
            row[f"stats/{col_prefix}/{stat_name}"] = val

    ts_arr = np.arange(episode_length, dtype=np.float32) / fps
    fi_arr = np.arange(episode_length, dtype=np.int64)
    ei_arr = np.full(episode_length, ep.episode_index, dtype=np.int64)
    gi_arr = np.arange(dataset_from_index, dataset_to_index, dtype=np.int64)
    ti_arr = np.zeros(episode_length, dtype=np.int64)

    for col_prefix, arr in (
        ("timestamp", ts_arr),
        ("frame_index", fi_arr),
        ("episode_index", ei_arr),
        ("index", gi_arr),
        ("task_index", ti_arr),
    ):
        ep_stats = _scalar_stats(arr)
        for stat_name, val in ep_stats.items():
            row[f"stats/{col_prefix}/{stat_name}"] = val

    return row


# ---------------------------------------------------------------------------
# Video copy helpers
# ---------------------------------------------------------------------------


def _copy_video_files(
    *,
    source_root: Path,
    output_root: Path,
    video_keys: list[str],
    episodes: list[EpisodeResult],
    chunks_size: int = CHUNKS_SIZE,
) -> tuple[int, int]:
    """Copy video files from source to output root."""

    copied = 0
    missing = 0

    for ep in episodes:
        src_chunk, src_file = chunk_and_file(ep.source_episode_index, chunks_size)
        dst_chunk, dst_file = chunk_and_file(ep.episode_index, chunks_size)

        for vk in video_keys:
            src_path = (
                source_root
                / "videos"
                / vk
                / f"chunk-{src_chunk:03d}"
                / f"file-{src_file:03d}.mp4"
            )
            dst_path = (
                output_root
                / "videos"
                / vk
                / f"chunk-{dst_chunk:03d}"
                / f"file-{dst_file:03d}.mp4"
            )
            if not src_path.exists():
                missing += 1
                print(
                    f"  WARNING: missing source video {src_path.relative_to(source_root)}",
                    flush=True,
                )
                continue
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)
                copied += 1

    return copied, missing


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def write_dataset(
    *,
    output_root: Path,
    source_root: Path,
    source_info: dict[str, Any],
    episodes: list[EpisodeResult],
    robot_type: str,
    joint_names: list[str],
    fps: int,
    chunks_size: int = CHUNKS_SIZE,
) -> None:
    """Write a complete LeRobot v3.0 dataset to ``output_root``.

    Existing files under ``output_root`` are overwritten.  Video files from
    ``source_root`` are copied as-is (frames beyond the last retained frame are
    simply never referenced).

    Parameters
    ----------
    output_root:
        Where to write the new dataset.
    source_root:
        Root of the source LeRobot dataset (for video copying).
    source_info:
        Parsed ``info.json`` of the source dataset.
    episodes:
        Per-episode IK results from :class:`EpisodeResult`.
    robot_type:
        String written into ``info.json`` (e.g. ``"bi_axol"``).
    joint_names:
        Ordered list of joint names for ``observation.state`` / ``action``.
    fps:
        Frames per second; used for timestamp computation.
    chunks_size:
        Number of episodes per chunk directory (default: 1000).
    """

    output_root = Path(output_root)
    source_root = Path(source_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Collect video feature definitions from source info.json
    video_features: dict[str, Any] = {
        k: v
        for k, v in source_info.get("features", {}).items()
        if v.get("dtype") == "video"
    }
    video_keys = list(video_features.keys())

    # ------------------------------------------------------------------
    # 1. Write data parquet (one per episode)
    # ------------------------------------------------------------------
    all_states_list: list[np.ndarray] = []
    all_actions_list: list[np.ndarray] = []
    all_timestamps_list: list[np.ndarray] = []
    all_frame_indices_list: list[np.ndarray] = []
    all_episode_indices_list: list[np.ndarray] = []
    all_global_indices_list: list[np.ndarray] = []
    all_task_indices_list: list[np.ndarray] = []

    # Build task index map
    task_to_idx: dict[str, int] = {}
    for ep in episodes:
        if ep.task not in task_to_idx:
            task_to_idx[ep.task] = len(task_to_idx)

    global_frame_cursor = 0
    episode_rows: list[dict[str, Any]] = []

    for ep in episodes:
        T = len(ep.states)
        assert T == len(ep.actions), "states and actions must have the same length"
        if T == 0:
            continue

        chunk_idx, file_idx = chunk_and_file(ep.episode_index, chunks_size)
        data_dir = output_root / "data" / f"chunk-{chunk_idx:03d}"
        data_dir.mkdir(parents=True, exist_ok=True)

        timestamps = np.arange(T, dtype=np.float32) / fps
        frame_indices = np.arange(T, dtype=np.int64)
        episode_indices = np.full(T, ep.episode_index, dtype=np.int64)
        global_indices = np.arange(
            global_frame_cursor, global_frame_cursor + T, dtype=np.int64
        )
        task_idx_val = task_to_idx[ep.task]
        task_indices = np.full(T, task_idx_val, dtype=np.int64)

        rows = {
            "observation.state": list(ep.states),
            "action": list(ep.actions),
            "timestamp": timestamps,
            "frame_index": frame_indices,
            "episode_index": episode_indices,
            "index": global_indices,
            "task_index": task_indices,
        }
        df = pd.DataFrame(rows)
        df.to_parquet(data_dir / f"file-{file_idx:03d}.parquet", index=False)

        # Accumulate for global stats
        all_states_list.append(ep.states)
        all_actions_list.append(ep.actions)
        all_timestamps_list.append(timestamps)
        all_frame_indices_list.append(frame_indices)
        all_episode_indices_list.append(episode_indices)
        all_global_indices_list.append(global_indices)
        all_task_indices_list.append(task_indices)

        episode_rows.append(
            _episode_parquet_row(
                ep=ep,
                episode_length=T,
                dataset_from_index=global_frame_cursor,
                dataset_to_index=global_frame_cursor + T,
                fps=fps,
                video_keys=video_keys,
                chunks_size=chunks_size,
            )
        )

        global_frame_cursor += T

    total_frames = global_frame_cursor
    total_episodes = len(episode_rows)
    total_tasks = len(task_to_idx)

    # ------------------------------------------------------------------
    # 2. Write meta/tasks.parquet
    # ------------------------------------------------------------------
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    tasks_df = pd.DataFrame(
        {"task_index": list(task_to_idx.values())},
        index=pd.Index(list(task_to_idx.keys()), name="task"),
    )
    tasks_df.to_parquet(meta_dir / "tasks.parquet")

    # ------------------------------------------------------------------
    # 3. Write meta/episodes parquet (chunked)
    # ------------------------------------------------------------------
    episodes_meta_df = pd.DataFrame(episode_rows)
    # Group by meta chunk
    for meta_chunk_idx in range(
        math.ceil(total_episodes / chunks_size) if total_episodes > 0 else 1
    ):
        start = meta_chunk_idx * chunks_size
        end = min(start + chunks_size, total_episodes)
        chunk_df = episodes_meta_df.iloc[start:end]
        ep_dir = meta_dir / "episodes" / f"chunk-{meta_chunk_idx:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        chunk_df.to_parquet(ep_dir / "file-000.parquet", index=False)

    # ------------------------------------------------------------------
    # 4. Write meta/info.json
    # ------------------------------------------------------------------
    info = _build_info_json(
        robot_type=robot_type,
        total_episodes=total_episodes,
        total_frames=total_frames,
        total_tasks=total_tasks,
        fps=fps,
        joint_names=joint_names,
        video_features=video_features,
        chunks_size=chunks_size,
    )
    with open(meta_dir / "info.json", "w") as fh:
        json.dump(info, fh, indent=4)

    # ------------------------------------------------------------------
    # 5. Write meta/stats.json
    # ------------------------------------------------------------------
    if all_states_list:
        all_states = np.concatenate(all_states_list, axis=0)
        all_actions = np.concatenate(all_actions_list, axis=0)
        all_timestamps = np.concatenate(all_timestamps_list)
        all_frame_indices = np.concatenate(all_frame_indices_list)
        all_episode_indices = np.concatenate(all_episode_indices_list)
        all_global_indices = np.concatenate(all_global_indices_list)
        all_task_indices = np.concatenate(all_task_indices_list)

        stats = _build_stats_json(
            all_states=all_states,
            all_actions=all_actions,
            all_timestamps=all_timestamps,
            all_frame_indices=all_frame_indices,
            all_episode_indices=all_episode_indices,
            all_global_indices=all_global_indices,
            all_task_indices=all_task_indices,
            video_features=video_features,
        )
        with open(meta_dir / "stats.json", "w") as fh:
            json.dump(stats, fh, indent=4)

    # ------------------------------------------------------------------
    # 6. Copy video files from source
    # ------------------------------------------------------------------
    copied_videos, missing_videos = _copy_video_files(
        source_root=source_root,
        output_root=output_root,
        video_keys=video_keys,
        episodes=episodes,
        chunks_size=chunks_size,
    )

    print(
        f"Dataset written to {output_root}\n"
        f"  Episodes: {total_episodes}  Frames: {total_frames}  "
        f"Tasks: {total_tasks}  Video keys: {len(video_keys)}  "
        f"Video files copied: {copied_videos}"
    )
    if missing_videos:
        print(
            f"  WARNING: {missing_videos} video file(s) missing in source. "
            "Reload the source dataset with dexumi.dataset.open_dataset "
            "(videos are always downloaded).",
            flush=True,
        )
