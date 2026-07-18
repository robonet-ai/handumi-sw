from pathlib import Path

import numpy as np
import pandas as pd

from handumi.dataset.writer import (
    EpisodeResult,
    _copy_video_files,
    _episode_parquet_row,
    _load_source_video_refs,
)


def test_converted_episode_preserves_shared_source_video_range(tmp_path: Path):
    source = tmp_path / "source"
    episodes_dir = source / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    key = "observation.images.workspace"
    prefix = f"videos/{key}"
    pd.DataFrame(
        {
            "episode_index": [0, 1],
            f"{prefix}/chunk_index": [0, 0],
            f"{prefix}/file_index": [0, 0],
            f"{prefix}/from_timestamp": [0.0, 10.0],
            f"{prefix}/to_timestamp": [10.0, 20.0],
        }
    ).to_parquet(episodes_dir / "file-000.parquet", index=False)
    video = source / "videos" / key / "chunk-000" / "file-000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")

    refs = _load_source_video_refs(source, [key])
    episode = EpisodeResult(
        episode_index=0,
        source_episode_index=1,
        states=np.zeros((9, 2), dtype=np.float32),
        actions=np.zeros((9, 2), dtype=np.float32),
        task="test",
    )
    row = _episode_parquet_row(
        ep=episode,
        episode_length=9,
        dataset_from_index=0,
        dataset_to_index=9,
        fps=10,
        video_keys=[key],
        source_video_refs=refs,
    )

    assert row[f"{prefix}/file_index"] == 0
    assert row[f"{prefix}/from_timestamp"] == 10.0
    assert row[f"{prefix}/to_timestamp"] == 10.9

    output = tmp_path / "output"
    copied, missing = _copy_video_files(
        source_root=source,
        output_root=output,
        video_keys=[key],
        episodes=[episode],
        source_video_refs=refs,
    )

    assert (copied, missing) == (1, 0)
    assert (output / "videos" / key / "chunk-000" / "file-000.mp4").is_file()
    assert not (output / "videos" / key / "chunk-000" / "file-001.mp4").exists()
