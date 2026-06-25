"""Small wrappers around LeRobot dataset loading.

This module keeps LeRobot imports isolated so the base project can stay light.
Install the optional dependency with:

    GIT_LFS_SKIP_SMUDGE=1 uv sync --extra lerobot
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetDownloadResult:
    """Summary of a downloaded or loaded LeRobot dataset."""

    repo_id: str
    root: Path
    num_episodes: int
    num_frames: int
    fps: int
    features: tuple[str, ...]


def _parse_episodes(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def ensure_dataset_metadata(
    repo_id: str,
    root: str | Path,
    *,
    revision: str | None = "main",
) -> dict[str, Any]:
    """Ensure ``meta/info.json`` exists locally, downloading ``meta/`` from Hub if needed.

    Call this before reading ``total_episodes`` so episode selection matches the
    source repository.
    """
    root = Path(root)
    info_path = root / "meta" / "info.json"

    needs_download = not info_path.exists()
    if not needs_download:
        with open(info_path) as fh:
            info = json.load(fh)
        needs_download = int(info.get("total_episodes", 0)) <= 0
    else:
        info = {}

    if needs_download:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required to download dataset metadata. "
                "Install with: GIT_LFS_SKIP_SMUDGE=1 uv sync --extra lerobot"
            ) from exc

        print(f"Downloading dataset metadata for {repo_id} …")
        root.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision or "main",
            local_dir=root,
            allow_patterns=["meta/**"],
        )
        with open(info_path) as fh:
            info = json.load(fh)

    return info


def download_lerobot_dataset(
    repo_id: str,
    output_dir: str | Path,
    *,
    revision: str | None = "main",
    episodes: str | None = None,
    force_cache_sync: bool = False,
) -> DatasetDownloadResult:
    """Download or load a LeRobot dataset (data, meta, and videos)."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot is not installed. Install it with: "
            "GIT_LFS_SKIP_SMUDGE=1 uv sync --extra lerobot"
        ) from exc

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=Path(output_dir),
        revision=revision,
        episodes=_parse_episodes(episodes),
        download_videos=True,
        force_cache_sync=force_cache_sync,
    )

    return DatasetDownloadResult(
        repo_id=dataset.repo_id,
        root=Path(dataset.root),
        num_episodes=dataset.num_episodes,
        num_frames=dataset.num_frames,
        fps=dataset.fps,
        features=tuple(dataset.features.keys()),
    )


def load_lerobot_dataset(
    repo_id: str,
    root: str | Path,
    *,
    episode: int | None = None,
    episodes: list[int] | None = None,
    revision: str | None = "main",
) -> Any:
    """Load a local or remote LeRobot dataset with LeRobot's reader.

    Always downloads video files when they are missing locally.
    """
    if episode is not None and episodes is not None:
        raise ValueError("Use only one of episode or episodes.")

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot is not installed. Install it with: "
            "GIT_LFS_SKIP_SMUDGE=1 uv sync --extra lerobot"
        ) from exc

    if episodes is not None:
        resolved_episodes = episodes
    elif episode is not None:
        resolved_episodes = [episode]
    else:
        resolved_episodes = None

    return LeRobotDataset(
        repo_id=repo_id,
        root=Path(root),
        revision=revision,
        episodes=resolved_episodes,
        download_videos=True,
    )


def load_pico_body_poses(
    repo_id: str,
    root: str | Path,
    *,
    episode: int = 0,
    column: str = "observation.pico.body_joints_pose",
    revision: str | None = "main",
) -> tuple[Any, int]:
    """Load PICO body joint poses from a LeRobot dataset.

    Also ensures video files for the episode exist under ``root/videos/``.

    Returns:
        A tuple ``(poses, fps)`` where ``poses`` has shape ``(T, 24, 7)``.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required. Install with: uv sync --extra viewer") from exc

    dataset = load_lerobot_dataset(
        repo_id=repo_id,
        root=root,
        episode=episode,
        revision=revision,
    )

    if column not in dataset.features:
        available = "\n".join(f"  - {name}" for name in dataset.features)
        raise KeyError(f"Column {column!r} not found. Available features:\n{available}")

    frames = []
    for idx in range(len(dataset)):
        item = dataset.get_raw_item(idx)
        frames.append(np.asarray(item[column], dtype=np.float32))

    if not frames:
        raise ValueError(f"No frames found for episode {episode}.")

    poses = np.stack(frames, axis=0)
    if poses.ndim != 3 or poses.shape[1:] != (24, 7):
        raise ValueError(f"Expected poses with shape (T, 24, 7), got {poses.shape}.")

    return poses, int(dataset.fps)
