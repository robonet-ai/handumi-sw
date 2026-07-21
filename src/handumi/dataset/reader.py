"""Read and cache LeRobot datasets via the upstream LeRobot reader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import numpy as np

from handumi.dataset.raw import (
    TRACKING_VALIDITY_NAMES,
)
from handumi.dataset.writer import info_path, load_info
from handumi.dataset.tracking_sidecar import discover_tracking_sidecars


def dataset_root_from_repo_id(repo_id: str) -> Path:
    """Default local cache directory for a Hugging Face dataset repo id."""
    repo_name = repo_id.rstrip("/").split("/")[-1]
    if not repo_name:
        raise ValueError(f"Cannot derive dataset root from repo id {repo_id!r}.")
    return Path("outputs/datasets") / repo_name


@dataclass(frozen=True)
class DatasetRef:
    """Pointer to a LeRobot dataset on disk and/or on the Hugging Face Hub."""

    repo_id: str
    root: Path
    revision: str = "main"

    @classmethod
    def from_repo_id(
        cls,
        repo_id: str,
        *,
        root: str | Path | None = None,
        revision: str = "main",
    ) -> DatasetRef:
        resolved_root = (
            Path(root) if root is not None else dataset_root_from_repo_id(repo_id)
        )
        return cls(repo_id=repo_id, root=resolved_root, revision=revision)


@dataclass(frozen=True)
class DatasetDownloadResult:
    """Summary of a downloaded or loaded LeRobot dataset."""

    repo_id: str
    root: Path
    num_episodes: int
    num_frames: int
    fps: int
    features: tuple[str, ...]


@dataclass(frozen=True)
class CanonicalBodyEpisode:
    """Optional aligned canonical body columns for one episode."""

    signals: dict[str, np.ndarray]


@dataclass(frozen=True)
class RawEpisode:
    """Raw state plus derived diagnostics for one compact HandUMI episode."""

    states: np.ndarray
    fps: float
    signals: dict[str, np.ndarray]
    body: CanonicalBodyEpisode | None = None
    tracking_sidecars: tuple[Path, ...] = ()
    images: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def handumi_metadata(
    info_or_root: Mapping[str, object] | str | Path,
) -> dict[str, Any]:
    """Return HandUMI-specific metadata from an ``info.json`` dict or dataset root."""
    info = (
        load_info(info_or_root)
        if not isinstance(info_or_root, Mapping)
        else info_or_root
    )
    meta = info.get("handumi", {})
    return dict(meta) if isinstance(meta, dict) else {}


def recording_device(info_or_root: dict[str, Any] | str | Path) -> str | None:
    """Return the tracking device recorded in ``meta/info.json``, if available."""
    value = handumi_metadata(info_or_root).get("recording_device")
    return str(value) if value is not None else None


def validate_raw_state_metadata(info_or_root: dict[str, Any] | str | Path) -> None:
    """Require the current layout and reject ambiguous historic recordings."""
    from handumi.dataset.compatibility import validate_dataset_compatibility

    info = (
        load_info(info_or_root) if not isinstance(info_or_root, dict) else info_or_root
    )
    root = Path(info_or_root) if isinstance(info_or_root, (str, Path)) else None
    validate_dataset_compatibility(info, root=root)


def load_raw_episode_states(
    ref: DatasetRef | None = None,
    *,
    repo_id: str | None = None,
    root: str | Path | None = None,
    episode: int,
    source: str = "observation.state",
    revision: str | None = None,
) -> tuple[np.ndarray, float]:
    """Load one raw 16D HandUMI episode column as ``(states, fps)``."""
    loaded = load_raw_episode(
        ref,
        repo_id=repo_id,
        root=root,
        episode=episode,
        source=source,
        revision=revision,
    )
    return loaded.states, loaded.fps


def load_raw_episode(
    ref: DatasetRef | None = None,
    *,
    repo_id: str | None = None,
    root: str | Path | None = None,
    episode: int,
    source: str = "observation.state",
    revision: str | None = None,
    download_videos: bool = False,
) -> RawEpisode:
    """Load one validated episode, optionally decoding its recorded cameras."""
    dataset = open_dataset(
        ref,
        repo_id=repo_id,
        root=root,
        episode=episode,
        revision=revision,
        download_videos=download_videos,
    )
    info = getattr(getattr(dataset, "meta", None), "info", {})
    validate_raw_state_metadata(info if isinstance(info, dict) else {})
    metadata = handumi_metadata(info if isinstance(info, dict) else {})
    fps = float(getattr(dataset, "fps", 30) or 30)
    table = dataset.hf_dataset
    if source not in table.column_names:
        raise ValueError(f"Dataset has no {source!r} feature.")
    states = np.asarray(table[source], dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 16:
        width = states.shape[1] if states.ndim == 2 else states.shape
        raise ValueError(
            f"Expected raw HandUMI state width 16 in {source!r}, got {width}."
        )
    if len(states) == 0:
        raise ValueError(f"Episode {episode} is empty.")

    prefixes = (
        "observation.tracking.",
        "observation.feetech.",
        "observation.camera.",
        "observation.sync.",
    )
    signals: dict[str, np.ndarray] = {}
    for key in table.column_names:
        if key != "observation.valid" and not key.startswith(prefixes):
            continue
        values = np.asarray(table[key])
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        signals[key] = values
    signals = normalize_raw_signals(states, signals, metadata=metadata)
    body_signals: dict[str, np.ndarray] = {}
    for key in table.column_names:
        if key.startswith("observation.body."):
            values = table[key]
            array = np.asarray(values)
            if array.dtype == object:
                array = np.stack(
                    [
                        np.asarray(
                            value.tolist() if hasattr(value, "tolist") else value
                        )
                        for value in values
                    ]
                )
            body_signals[key] = array
    body = CanonicalBodyEpisode(body_signals) if body_signals else None
    images = _decode_episode_images(dataset, len(states)) if download_videos else {}
    dataset_root = Path(getattr(dataset, "root", root or "."))
    sidecars = discover_tracking_sidecars(dataset_root, episode_index=episode)
    return RawEpisode(
        states=states,
        fps=fps,
        signals=signals,
        body=body,
        tracking_sidecars=sidecars,
        images=images,
        metadata=dict(info) if isinstance(info, dict) else {},
    )


def _decode_episode_images(dataset: Any, frame_count: int) -> dict[str, np.ndarray]:
    """Decode available LeRobot image/video features without a second parser."""
    table = dataset.hf_dataset
    feature_keys = set(getattr(dataset, "features", {}) or {})
    feature_keys.update(getattr(table, "column_names", ()))
    image_keys = sorted(
        key for key in feature_keys if str(key).startswith("observation.images.")
    )
    decoded: dict[str, np.ndarray] = {}
    for key in image_keys:
        frames: list[np.ndarray] = []
        for index in range(frame_count):
            row = dataset[index]
            if key not in row:
                frames = []
                break
            value = row[key]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            elif hasattr(value, "numpy"):
                value = value.numpy()
            else:
                value = np.asarray(value)
            image = np.asarray(value)
            if image.ndim == 3 and image.shape[0] in (1, 3, 4):
                image = np.moveaxis(image, 0, -1)
            if image.ndim not in (2, 3):
                frames = []
                break
            if np.issubdtype(image.dtype, np.floating):
                image = np.clip(image, 0.0, 1.0)
                image = np.rint(image * 255.0).astype(np.uint8)
            elif image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            frames.append(image)
        if frames:
            decoded[key] = np.stack(frames)
    return decoded


def normalize_raw_signals(
    states: np.ndarray,
    signals: dict[str, np.ndarray],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Derive diagnostics omitted from the single compact on-disk layout."""
    states = np.asarray(states, dtype=np.float32)
    frame_count = len(states)
    normalized = {key: np.asarray(value) for key, value in signals.items()}
    metadata = metadata or {}

    validity = normalized.get("observation.valid")
    expected_validity_shape = (frame_count, len(TRACKING_VALIDITY_NAMES))
    if validity is None or np.asarray(validity).shape != expected_validity_shape:
        shape = None if validity is None else np.asarray(validity).shape
        raise ValueError(
            "Current HandUMI layout requires observation.valid shape "
            f"{expected_validity_shape}, got {shape}."
        )

    workspace = normalized.get("observation.tracking.workspace_from_device_pose")
    device_hmd = normalized.get("observation.tracking.device_hmd_pose")
    if (
        "observation.tracking.hmd_pose" not in normalized
        and workspace is not None
        and device_hmd is not None
    ):
        workspace = np.asarray(workspace)
        device_hmd = np.asarray(device_hmd)
        if workspace.shape == device_hmd.shape == (frame_count, 7):
            normalized["observation.tracking.hmd_pose"] = np.stack(
                [
                    _compose_pose7(a, b)
                    for a, b in zip(workspace, device_hmd, strict=True)
                ]
            ).astype(np.float32)

    aligned = _frame_signal(
        normalized.get("observation.tracking.aligned_time_ns"), frame_count
    )
    device_time = _frame_signal(
        normalized.get("observation.tracking.device_time_ns"), frame_count
    )
    if (
        "observation.tracking.clock_offset_ns" not in normalized
        and aligned is not None
        and device_time is not None
    ):
        valid_clock = (aligned > 0) & (device_time > 0)
        normalized["observation.tracking.clock_offset_ns"] = np.where(
            valid_clock,
            aligned.astype(np.int64) - device_time.astype(np.int64),
            0,
        )

    _restore_enabled_signals(normalized, metadata, frame_count)
    _derive_source_timing(normalized, frame_count)
    return normalized


def _restore_enabled_signals(
    signals: dict[str, np.ndarray],
    metadata: dict[str, Any],
    frame_count: int,
) -> None:
    sources = metadata.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("Current HandUMI layout requires handumi.sources metadata.")

    feetech = sources.get("feetech")
    if isinstance(feetech, dict) and "enabled" in feetech:
        signals.setdefault(
            "observation.feetech.enabled",
            np.full(frame_count, int(bool(feetech["enabled"])), dtype=np.int64),
        )

    cameras = sources.get("cameras")
    if not isinstance(cameras, dict):
        return
    for name, config in cameras.items():
        if not isinstance(config, dict) or "enabled" not in config:
            continue
        signals.setdefault(
            f"observation.camera.{name}.enabled",
            np.full(frame_count, int(bool(config["enabled"])), dtype=np.int64),
        )


def _derive_source_timing(signals: dict[str, np.ndarray], frame_count: int) -> None:
    target = _frame_signal(signals.get("observation.sync.target_time_ns"), frame_count)
    record = _frame_signal(signals.get("observation.sync.record_time_ns"), frame_count)
    if target is None or record is None:
        return

    sources: dict[str, np.ndarray] = {}
    aligned = _frame_signal(
        signals.get("observation.tracking.aligned_time_ns"), frame_count
    )
    received = _frame_signal(
        signals.get("observation.tracking.pc_monotonic_ns"), frame_count
    )
    if aligned is not None:
        sources["observation.tracking"] = (
            aligned if received is None else np.where(aligned > 0, aligned, received)
        )

    for key, value in tuple(signals.items()):
        if not key.endswith(".sample_time_ns"):
            continue
        sample = _frame_signal(value, frame_count)
        if sample is not None:
            sources[key.removesuffix(".sample_time_ns")] = sample

    missing_value_ms = np.iinfo(np.int64).max / 1e6
    for prefix, sample in sources.items():
        missing = sample <= 0
        age_ms = np.where(
            missing, missing_value_ms, np.maximum(0, record - sample) / 1e6
        )
        sync_error_ms = np.where(
            missing,
            missing_value_ms,
            np.abs(sample - target) / 1e6,
        )
        signals.setdefault(f"{prefix}.age_ms", age_ms.astype(np.float32))
        signals.setdefault(f"{prefix}.sync_error_ms", sync_error_ms.astype(np.float32))


def _frame_signal(value: Any, frame_count: int) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    return array.reshape(-1) if array.size == frame_count else None


def _compose_pose7(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(7)
    b = np.asarray(b, dtype=np.float64).reshape(7)
    qa = _normalize_quaternion(a[3:7])
    qb = _normalize_quaternion(b[3:7])
    vector = qa[:3]
    rotated = b[:3] + 2.0 * (
        qa[3] * np.cross(vector, b[:3]) + np.cross(vector, np.cross(vector, b[:3]))
    )
    ax, ay, az, aw = qa
    bx, by, bz, bw = qb
    quaternion = _normalize_quaternion(
        np.array(
            [
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ]
        )
    )
    return np.concatenate([a[:3] + rotated, quaternion])


def _normalize_quaternion(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quaternion / norm


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
    revision: str | None = None,
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
    revision: str | None = None,
    download_videos: bool = True,
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
        download_videos=download_videos,
    )


def download_dataset(
    ref: DatasetRef | None = None,
    *,
    repo_id: str | None = None,
    output_dir: str | Path | None = None,
    revision: str | None = None,
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
