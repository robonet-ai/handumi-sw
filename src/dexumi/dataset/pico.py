"""Domain-specific reads from PICO body-joint columns in LeRobot datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dexumi.dataset.reader import open_dataset
from dexumi.dataset.ref import DatasetRef


def load_pico_body_poses(
    ref: DatasetRef | None = None,
    *,
    repo_id: str | None = None,
    root: str | Path | None = None,
    episode: int = 0,
    column: str = "observation.pico.body_joints_pose",
    revision: str | None = "main",
) -> tuple[Any, int]:
    """Load PICO body joint poses from a LeRobot dataset episode.

    Returns:
        A tuple ``(poses, fps)`` where ``poses`` has shape ``(T, 24, 7)``.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required. Install project dependencies with: uv sync") from exc

    dataset = open_dataset(
        ref,
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
