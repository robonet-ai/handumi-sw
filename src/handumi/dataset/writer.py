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

    from handumi.dataset import EpisodeResult, write_dataset

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Shared LeRobot v3.0 on-disk layout helpers
# ---------------------------------------------------------------------------

CHUNKS_SIZE = 1000


def chunk_and_file(index: int, chunks_size: int = CHUNKS_SIZE) -> tuple[int, int]:
    return index // chunks_size, index % chunks_size


def info_path(root: str | Path) -> Path:
    return Path(root) / "meta" / "info.json"


def load_info(root: str | Path) -> dict[str, Any]:
    path = info_path(root)
    with open(path) as fh:
        return json.load(fh)


def update_handumi_metadata(
    root: str | Path, metadata: dict[str, Any]
) -> dict[str, Any]:
    """Merge HandUMI-specific metadata into ``meta/info.json``."""
    path = info_path(root)
    info = load_info(root)
    current = info.get("handumi", {})
    info["handumi"] = {
        **(current if isinstance(current, dict) else {}),
        **metadata,
    }
    with open(path, "w") as fh:
        json.dump(info, fh, indent=4)
        fh.write("\n")
    return info


def _derive_handumi_metadata(
    *,
    source_info: dict[str, Any],
    explicit: dict[str, Any] | None,
) -> dict[str, Any] | None:
    source_meta = source_info.get("handumi", {})
    if not isinstance(source_meta, dict):
        source_meta = {}
    merged = dict(source_meta)
    if explicit:
        merged.update(explicit)
    merged = _raw_only_metadata(merged)
    if not merged:
        return None
    source_semantics = source_meta.get("state_semantics")
    if source_semantics is not None and "tcp" not in str(source_semantics).lower():
        merged.setdefault("source_state_semantics", source_semantics)
    merged["derived_dataset"] = True
    return merged


def _raw_only_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Drop legacy state-schema labels while preserving calibration provenance."""
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        key_l = str(key).lower()
        value_l = str(value).lower()
        if key_l in {"tracking_schema", "state_semantics"} and "tcp" in value_l:
            continue
        clean[key] = value
    return clean


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
    optional_observations: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source_episode_index < 0:
            self.source_episode_index = self.episode_index


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_feature_stats(values: np.ndarray) -> dict[str, Any]:
    """Compute LeRobot-style statistics for a 2-D float array (frames × dim)."""
    array = np.asarray(values)
    feature_width = int(np.prod(array.shape[1:])) if array.ndim > 1 else 1
    flat = array.reshape(len(array), feature_width)

    finite = np.isfinite(flat)
    count = finite.sum(axis=0).astype(int).tolist()

    def finite_stat(column: np.ndarray, operation) -> float:
        values = column[np.isfinite(column)]
        return float("nan") if len(values) == 0 else float(operation(values))

    columns = [flat[:, index] for index in range(flat.shape[1])]
    min_vals = [finite_stat(column, np.min) for column in columns]
    max_vals = [finite_stat(column, np.max) for column in columns]
    mean_vals = [finite_stat(column, np.mean) for column in columns]
    std_vals = [finite_stat(column, np.std) for column in columns]
    q01 = [
        finite_stat(column, lambda value: np.percentile(value, 1)) for column in columns
    ]
    q10 = [
        finite_stat(column, lambda value: np.percentile(value, 10))
        for column in columns
    ]
    q50 = [
        finite_stat(column, lambda value: np.percentile(value, 50))
        for column in columns
    ]
    q90 = [
        finite_stat(column, lambda value: np.percentile(value, 90))
        for column in columns
    ]
    q99 = [
        finite_stat(column, lambda value: np.percentile(value, 99))
        for column in columns
    ]

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


def _video_default_stats(shape: list[int]) -> dict[str, Any]:
    """Default statistics for a video feature with pixel values in [0, 1]."""
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


def _fixed_shape_arrow_type(dtype: np.dtype, shape: tuple[int, ...]) -> pa.DataType:
    """Return a typed Arrow scalar/fixed-list tree for one feature row."""
    arrow_type: pa.DataType = pa.from_numpy_dtype(dtype)
    for size in reversed(shape):
        arrow_type = pa.list_(arrow_type, int(size))
    return arrow_type


def _build_info_json(
    *,
    robot_type: str,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    fps: int,
    joint_names: list[str],
    video_features: dict[str, Any],
    handumi_metadata: dict[str, Any] | None = None,
    optional_features: dict[str, Any] | None = None,
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
    features.update(optional_features or {})
    features["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    features["frame_index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["episode_index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["task_index"] = {"dtype": "int64", "shape": [1], "names": None}

    info = {
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
    if handumi_metadata:
        info["handumi"] = handumi_metadata
    return info


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
    optional_values: dict[str, np.ndarray] | None = None,
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
        stats[video_key] = _video_default_stats(feat["shape"])

    for key, values in (optional_values or {}).items():
        stats[key] = _compute_feature_stats(values)

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
    source_video_refs: dict[int, dict[str, dict[str, float | int]]],
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
        source_ref = source_video_refs.get(ep.source_episode_index, {}).get(vk)
        if source_ref is None:
            src_chunk, src_file = chunk_and_file(ep.source_episode_index, chunks_size)
            from_ts = 0.0
            to_ts = float(episode_length) / fps
        else:
            src_chunk = int(source_ref["chunk_index"])
            src_file = int(source_ref["file_index"])
            from_ts = float(source_ref["from_timestamp"])
            to_ts = min(
                float(source_ref["to_timestamp"]),
                from_ts + float(episode_length) / fps,
            )
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

    for key, data in ep.optional_observations.items():
        ep_stats = _compute_feature_stats(np.asarray(data))
        for stat_name, value in ep_stats.items():
            row[f"stats/{key}/{stat_name}"] = value

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


def _load_source_video_refs(
    source_root: Path,
    video_keys: list[str],
) -> dict[int, dict[str, dict[str, float | int]]]:
    """Load per-episode video file/range references from LeRobot metadata."""
    refs: dict[int, dict[str, dict[str, float | int]]] = {}
    episode_files = sorted(
        (source_root / "meta" / "episodes").glob("chunk-*/*.parquet")
    )
    for path in episode_files:
        frame = pd.read_parquet(path)
        for _, row in frame.iterrows():
            episode_index = int(np.asarray(row.at["episode_index"]).item())
            episode_refs = refs.setdefault(episode_index, {})
            for key in video_keys:
                prefix = f"videos/{key}"
                required = (
                    f"{prefix}/chunk_index",
                    f"{prefix}/file_index",
                    f"{prefix}/from_timestamp",
                    f"{prefix}/to_timestamp",
                )
                if not all(column in frame.columns for column in required):
                    continue
                episode_refs[key] = {
                    "chunk_index": int(np.asarray(row.at[required[0]]).item()),
                    "file_index": int(np.asarray(row.at[required[1]]).item()),
                    "from_timestamp": float(np.asarray(row.at[required[2]]).item()),
                    "to_timestamp": float(np.asarray(row.at[required[3]]).item()),
                }
    return refs


def _copy_video_files(
    *,
    source_root: Path,
    output_root: Path,
    video_keys: list[str],
    episodes: list[EpisodeResult],
    source_video_refs: dict[int, dict[str, dict[str, float | int]]],
    chunks_size: int = CHUNKS_SIZE,
) -> tuple[int, int]:
    """Copy video files from source to output root."""

    copied = 0
    missing = 0

    requested: set[tuple[str, int, int]] = set()
    for ep in episodes:
        for vk in video_keys:
            source_ref = source_video_refs.get(ep.source_episode_index, {}).get(vk)
            if source_ref is None:
                src_chunk, src_file = chunk_and_file(
                    ep.source_episode_index,
                    chunks_size,
                )
            else:
                src_chunk = int(source_ref["chunk_index"])
                src_file = int(source_ref["file_index"])
            requested.add((vk, src_chunk, src_file))

    for vk, src_chunk, src_file in sorted(requested):
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
            / f"chunk-{src_chunk:03d}"
            / f"file-{src_file:03d}.mp4"
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
    handumi_metadata: dict[str, Any] | None = None,
    preserve_tracking_sidecars: bool = False,
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
    output_handumi_metadata = _derive_handumi_metadata(
        source_info=source_info,
        explicit=handumi_metadata,
    )
    if preserve_tracking_sidecars:
        output_handumi_metadata = dict(output_handumi_metadata or {})
        output_handumi_metadata["tracking_sidecar"] = {
            "schema": "handumi_tracking_sidecar_v1",
            "manifest": "raw/tracking/manifest.json",
            "derived_references": True,
        }
    source_video_refs = _load_source_video_refs(source_root, video_keys)

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
    all_optional_values: dict[str, list[np.ndarray]] = {}

    optional_keys = sorted(
        {key for episode in episodes for key in episode.optional_observations}
    )
    source_features = source_info.get("features", {})
    optional_features: dict[str, Any] = {}
    for key in optional_keys:
        if key in source_features:
            optional_features[key] = source_features[key]
            continue
        sample = next(
            np.asarray(episode.optional_observations[key])
            for episode in episodes
            if key in episode.optional_observations
        )
        optional_features[key] = {
            "dtype": str(sample.dtype),
            "shape": list(sample.shape[1:]),
            "names": None,
        }

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

        empty_body_observation: dict[str, np.ndarray] | None = None
        for key in optional_keys:
            if key not in ep.optional_observations:
                if key.startswith("observation.body."):
                    if empty_body_observation is None:
                        from handumi.body.model import CanonicalBodyFrame

                        empty_body_observation = (
                            CanonicalBodyFrame.empty().observation()
                        )
                    if key in empty_body_observation:
                        ep.optional_observations[key] = np.repeat(
                            empty_body_observation[key][None, ...], T, axis=0
                        )
                if key not in ep.optional_observations:
                    feature = optional_features[key]
                    shape = tuple(feature.get("shape", ()))
                    dtype = np.dtype(feature["dtype"])
                    fill = np.nan if np.issubdtype(dtype, np.floating) else 0
                    ep.optional_observations[key] = np.full(
                        (T, *shape), fill, dtype=dtype
                    )
            values = np.asarray(ep.optional_observations[key])
            if len(values) != T:
                raise ValueError(
                    f"Episode {ep.episode_index} feature {key!r} has {len(values)} "
                    f"rows; expected {T}."
                )

        rows = {
            "observation.state": list(ep.states),
            "action": list(ep.actions),
            "timestamp": timestamps,
            "frame_index": frame_indices,
            "episode_index": episode_indices,
            "index": global_indices,
            "task_index": task_indices,
        }
        rows.update(
            {
                key: np.asarray(ep.optional_observations[key]).tolist()
                for key in optional_keys
            }
        )
        table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
        for key in optional_keys:
            values = np.asarray(ep.optional_observations[key])
            arrow_type = _fixed_shape_arrow_type(values.dtype, values.shape[1:])
            column = pa.array(values.tolist(), type=arrow_type)
            column_index = table.schema.get_field_index(key)
            table = table.set_column(column_index, pa.field(key, arrow_type), column)
        pq.write_table(table, data_dir / f"file-{file_idx:03d}.parquet")

        # Accumulate for global stats
        all_states_list.append(ep.states)
        all_actions_list.append(ep.actions)
        all_timestamps_list.append(timestamps)
        all_frame_indices_list.append(frame_indices)
        all_episode_indices_list.append(episode_indices)
        all_global_indices_list.append(global_indices)
        all_task_indices_list.append(task_indices)
        for key in optional_keys:
            all_optional_values.setdefault(key, []).append(
                np.asarray(ep.optional_observations[key])
            )

        episode_rows.append(
            _episode_parquet_row(
                ep=ep,
                episode_length=T,
                dataset_from_index=global_frame_cursor,
                dataset_to_index=global_frame_cursor + T,
                fps=fps,
                video_keys=video_keys,
                source_video_refs=source_video_refs,
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
        handumi_metadata=output_handumi_metadata,
        optional_features=optional_features,
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
            optional_values={
                key: np.concatenate(values, axis=0)
                for key, values in all_optional_values.items()
            },
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
        source_video_refs=source_video_refs,
        chunks_size=chunks_size,
    )

    if preserve_tracking_sidecars:
        from handumi.dataset.tracking_sidecar import discover_tracking_sidecars

        sidecar_records = []
        for episode in episodes:
            for source_path in discover_tracking_sidecars(
                source_root, episode_index=episode.source_episode_index
            ):
                destination = (
                    output_root
                    / "raw"
                    / "tracking"
                    / "source"
                    / f"episode-{episode.episode_index:06d}"
                    / source_path.name
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination)
                sidecar_records.append(
                    {
                        "path": destination.relative_to(output_root).as_posix(),
                        "episode_index": episode.episode_index,
                        "source_episode_index": episode.source_episode_index,
                        "status": "derived_reference",
                    }
                )
        source_session_manifests = (
            source_root / "raw" / "tracking" / "session_manifests.jsonl"
        )
        session_manifest_reference: str | None = None
        if source_session_manifests.exists():
            destination = output_root / "raw" / "tracking" / "session_manifests.jsonl"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_session_manifests, destination)
            session_manifest_reference = destination.relative_to(output_root).as_posix()
        if sidecar_records:
            sidecar_manifest = output_root / "raw" / "tracking" / "manifest.json"
            manifest = {
                "schema": "handumi_tracking_sidecar_v1",
                "files": sidecar_records,
            }
            if session_manifest_reference is not None:
                manifest["session_manifests"] = session_manifest_reference
            sidecar_manifest.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        f"Dataset written to {output_root}\n"
        f"  Episodes: {total_episodes}  Frames: {total_frames}  "
        f"Tasks: {total_tasks}  Video keys: {len(video_keys)}  "
        f"Video files copied: {copied_videos}"
    )
    if missing_videos:
        print(
            f"  WARNING: {missing_videos} video file(s) missing in source. "
            "Reload the source dataset with handumi.dataset.open_dataset "
            "(videos are always downloaded).",
            flush=True,
        )
