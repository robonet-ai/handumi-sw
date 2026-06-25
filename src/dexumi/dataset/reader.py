"""Read and cache LeRobot datasets via the upstream LeRobot reader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dexumi.dataset.ref import DatasetRef
from dexumi.dataset.schema import info_path, load_info


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


def _resolve_ref(
    ref: DatasetRef | None,
    *,
    repo_id: str | None,
    root: str | Path | None,
    revision: str | None,
) -> DatasetRef:
    if ref is not None:
        if repo_id is not None or root is not None or revision is not None:
            raise ValueError("Pass either ref or repo_id/root/revision, not both.")
        return ref
    if repo_id is None or root is None:
        raise ValueError("repo_id and root are required when ref is not provided.")
    return DatasetRef(
        repo_id=repo_id,
        root=Path(root),
        revision=revision or "main",
    )


def ensure_metadata(
    ref: DatasetRef | None = None,
    *,
    repo_id: str | None = None,
    root: str | Path | None = None,
    revision: str | None = "main",
) -> dict[str, Any]:
    """Ensure ``meta/info.json`` exists locally, downloading ``meta/`` from Hub if needed."""
    resolved = _resolve_ref(ref, repo_id=repo_id, root=root, revision=revision)
    path = info_path(resolved.root)

    needs_download = not path.exists()
    if not needs_download:
        info = load_info(resolved.root)
        needs_download = int(info.get("total_episodes", 0)) <= 0
    else:
        info = {}

    if needs_download:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required to download dataset metadata. "
                "Install project dependencies with: uv sync"
            ) from exc

        print(f"Downloading dataset metadata for {resolved.repo_id} …")
        resolved.root.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=resolved.repo_id,
            repo_type="dataset",
            revision=resolved.revision,
            local_dir=resolved.root,
            allow_patterns=["meta/**"],
        )
        info = load_info(resolved.root)

    return info


def open_dataset(
    ref: DatasetRef | None = None,
    *,
    repo_id: str | None = None,
    root: str | Path | None = None,
    episode: int | None = None,
    episodes: list[int] | None = None,
    revision: str | None = "main",
) -> Any:
    """Open a local or remote LeRobot dataset, downloading missing files as needed."""
    if episode is not None and episodes is not None:
        raise ValueError("Use only one of episode or episodes.")

    resolved = _resolve_ref(ref, repo_id=repo_id, root=root, revision=revision)

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot is not installed. Install project dependencies with: uv sync"
        ) from exc

    if episodes is not None:
        resolved_episodes = episodes
    elif episode is not None:
        resolved_episodes = [episode]
    else:
        resolved_episodes = None

    return LeRobotDataset(
        repo_id=resolved.repo_id,
        root=resolved.root,
        revision=resolved.revision,
        episodes=resolved_episodes,
        download_videos=True,
    )


def download_dataset(
    ref: DatasetRef | None = None,
    *,
    repo_id: str | None = None,
    output_dir: str | Path | None = None,
    revision: str | None = "main",
    episodes: str | None = None,
    force_cache_sync: bool = False,
) -> DatasetDownloadResult:
    """Download or load a LeRobot dataset and return a short summary."""
    resolved = _resolve_ref(
        ref,
        repo_id=repo_id,
        root=output_dir,
        revision=revision,
    )

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot is not installed. Install project dependencies with: uv sync"
        ) from exc

    dataset = LeRobotDataset(
        repo_id=resolved.repo_id,
        root=resolved.root,
        revision=resolved.revision,
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
