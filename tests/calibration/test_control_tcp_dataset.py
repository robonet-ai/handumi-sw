from pathlib import Path

import numpy as np
import pandas as pd

from handumi.calibration.control_tcp import load_episode_poses


def _write_pose_dataset(path: Path, column: str) -> None:
    poses = [
        np.array([index, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        for index in (1.0, 0.0)
    ]
    pd.DataFrame(
        {
            "episode_index": [0, 0],
            "frame_index": [1, 0],
            column: poses,
        }
    ).to_parquet(path)


def test_load_episode_poses_accepts_explicit_pose_column(tmp_path: Path):
    path = tmp_path / "explicit.parquet"
    _write_pose_dataset(path, "observation.tracking.left_controller_pose")

    poses = load_episode_poses(
        path,
        0,
        "left",
        column="observation.tracking.left_controller_pose",
    )

    np.testing.assert_allclose(poses[:, 0], [0.0, 1.0])


def test_load_episode_poses_falls_back_to_compact_state(tmp_path: Path):
    path = tmp_path / "compact.parquet"
    states = []
    for index in (1.0, 0.0):
        state = np.zeros(16, dtype=np.float32)
        state[3:7] = [0.0, 0.0, 0.0, 1.0]
        state[7] = index
        state[10:14] = [0.0, 0.0, 0.0, 1.0]
        states.append(state)
    pd.DataFrame(
        {
            "episode_index": [0, 0],
            "frame_index": [1, 0],
            "observation.state": states,
        }
    ).to_parquet(path)

    poses = load_episode_poses(path, 0, "right")

    np.testing.assert_allclose(poses[:, 0], [0.0, 1.0])
