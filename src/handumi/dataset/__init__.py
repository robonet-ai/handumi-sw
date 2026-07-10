"""LeRobot dataset read/write boundary for handumi."""

from typing import Any

from handumi.dataset.reader import (
    DatasetDownloadResult,
    RawEpisode,
    download_dataset,
    ensure_metadata,
    handumi_metadata,
    load_raw_episode_states,
    load_raw_episode,
    open_dataset,
    recording_device,
    validate_raw_state_metadata,
)
from handumi.dataset.raw import (
    HANDUMI_RAW_IMAGE_KEYS,
    HANDUMI_RAW_STATE_NAMES,
    HANDUMI_RAW_STATE_SIZE,
    LEFT_GRIPPER_INDEX,
    LEFT_POSE_SLICE,
    RIGHT_GRIPPER_INDEX,
    RIGHT_POSE_SLICE,
    raw_state_feature,
    validate_raw_state_shape,
)
from handumi.dataset.reader import DatasetRef, dataset_root_from_repo_id
from handumi.dataset.quality import (
    EpisodeQualityConfig,
    EpisodeQualityReport,
    QualityFinding,
    validate_episode,
    write_quality_report,
)


def __getattr__(name: str) -> Any:
    """Lazily expose writer symbols without importing pandas at package import time."""
    writer_symbols = {
        "CHUNKS_SIZE",
        "EpisodeResult",
        "chunk_and_file",
        "info_path",
        "load_info",
        "update_handumi_metadata",
        "write_dataset",
    }
    if name in writer_symbols:
        from handumi.dataset.writer import (
            CHUNKS_SIZE,
            EpisodeResult,
            chunk_and_file,
            info_path,
            load_info,
            update_handumi_metadata,
            write_dataset,
        )

        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CHUNKS_SIZE",
    "DatasetDownloadResult",
    "DatasetRef",
    "EpisodeResult",
    "EpisodeQualityConfig",
    "EpisodeQualityReport",
    "RawEpisode",
    "QualityFinding",
    "HANDUMI_RAW_IMAGE_KEYS",
    "HANDUMI_RAW_STATE_NAMES",
    "HANDUMI_RAW_STATE_SIZE",
    "LEFT_GRIPPER_INDEX",
    "LEFT_POSE_SLICE",
    "RIGHT_GRIPPER_INDEX",
    "RIGHT_POSE_SLICE",
    "chunk_and_file",
    "dataset_root_from_repo_id",
    "download_dataset",
    "ensure_metadata",
    "handumi_metadata",
    "info_path",
    "load_info",
    "load_raw_episode_states",
    "load_raw_episode",
    "open_dataset",
    "raw_state_feature",
    "recording_device",
    "update_handumi_metadata",
    "validate_raw_state_metadata",
    "validate_raw_state_shape",
    "validate_episode",
    "write_quality_report",
    "write_dataset",
]
