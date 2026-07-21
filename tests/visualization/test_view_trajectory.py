from __future__ import annotations

import numpy as np

from handumi.body.model import CanonicalBodyFrame
from handumi.dataset.reader import CanonicalBodyEpisode, RawEpisode
from handumi.scripts.view_trajectory import (
    FRAME_TIMELINE,
    TIME_TIMELINE,
    ViewerOptions,
    log_episode,
)
from handumi.visualization.body import BODY_JOINTS_PATH, WHOLE_COM_PATH

from .test_body_visualization import synthetic_body_frame


class _Archetype:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def compress(self, **kwargs):
        return self


class _FakeRerun:
    Clear = Points3D = LineStrips3D = Scalars = Image = Mesh3D = TextDocument = (
        _Archetype
    )

    def __init__(self):
        self.frame = None
        self.time = None
        self.time_calls = []
        self.logs = []

    def set_time(self, timeline, **kwargs):
        self.time_calls.append((timeline, kwargs))
        if timeline == FRAME_TIMELINE:
            self.frame = kwargs["sequence"]
        elif timeline == TIME_TIMELINE:
            self.time = kwargs["duration"]

    def log(self, path, archetype, *, static=False):
        self.logs.append((path, self.frame, self.time, static, archetype))


def _episode(*, body: bool, images: bool = False) -> RawEpisode:
    count = 4
    states = np.zeros((count, 16), dtype=np.float32)
    states[:, 0] = np.arange(count) * 0.1
    states[:, 7] = 1.0 + np.arange(count) * 0.1
    states[:, 3:7] = [0, 0, 0, 1]
    states[:, 10:14] = [0, 0, 0, 1]
    signals = {
        "observation.tracking.left_tracked": np.ones(count, dtype=np.int64),
        "observation.tracking.right_tracked": np.ones(count, dtype=np.int64),
        "observation.feetech.left_width_mm": np.arange(count, dtype=np.float32),
        "observation.feetech.right_width_mm": np.arange(count, dtype=np.float32) + 10,
        "observation.valid": np.ones((count, 8), dtype=np.int64),
        "observation.tracking.hmd_pose": np.tile(
            np.array([0, 0, 1.6, 0, 0, 0, 1], dtype=np.float32), (count, 1)
        ),
    }
    canonical = None
    if body:
        frames = [synthetic_body_frame() for _ in range(count)]
        frames[1] = CanonicalBodyFrame.empty()
        observations = [frame.observation() for frame in frames]
        canonical = CanonicalBodyEpisode(
            {
                key: np.stack([observation[key] for observation in observations])
                for key in observations[0]
            }
        )
    image_data = (
        {"observation.images.left_wrist": np.zeros((count, 8, 10, 3), dtype=np.uint8)}
        if images
        else {}
    )
    metadata = {
        "handumi": {
            "controller_tcp_calibration": {
                "sha256": "synthetic",
                "applied_to_state": False,
                "controller_to_gripper_tcp": {
                    "left": {
                        "position": [0.1, 0.0, 0.0],
                        "quaternion": [0.0, 0.0, 0.0, 1.0],
                    },
                    "right": {
                        "position": [-0.1, 0.0, 0.0],
                        "quaternion": [0.0, 0.0, 0.0, 1.0],
                    },
                },
            }
        }
    }
    return RawEpisode(
        states=states,
        fps=10.0,
        signals=signals,
        body=canonical,
        images=image_data,
        metadata=metadata,
    )


def test_offline_frame_and_time_are_set_before_synchronized_values():
    rr = _FakeRerun()
    stats = log_episode(rr, _episode(body=True, images=True))
    assert stats.frames == 4
    assert len(rr.time_calls) == 8
    current = [entry for entry in rr.logs if not entry[3]]
    assert current
    assert all(
        frame is not None and time == frame / 10.0 for _, frame, time, _, _ in current
    )
    frame_two_paths = {
        path for path, frame, time, static, _ in current if frame == 2 and time == 0.2
    }
    assert "tracking/left/raw" in frame_two_paths
    assert BODY_JOINTS_PATH in frame_two_paths
    assert "observation.images.left_wrist" in frame_two_paths


def test_invalid_body_frame_clears_at_the_same_cursor():
    rr = _FakeRerun()
    log_episode(rr, _episode(body=True))
    clears = [
        entry
        for entry in rr.logs
        if entry[0] == WHOLE_COM_PATH and entry[1] == 1 and not entry[3]
    ]
    assert len(clears) == 1
    assert isinstance(clears[0][4], _Archetype)


def test_controller_only_episode_with_body_none_retains_legacy_entities():
    rr = _FakeRerun()
    stats = log_episode(rr, _episode(body=False))
    assert stats.body_present is False
    paths = {entry[0] for entry in rr.logs}
    assert "tracking/left/tcp" in paths
    assert "tracking/right/raw" in paths
    assert not any(path.startswith("tracking/body/") for path in paths)


def test_full_paths_are_logged_once_not_rebuilt_per_frame():
    rr = _FakeRerun()
    log_episode(
        rr,
        _episode(body=True),
        options=ViewerOptions(
            temporal_decimation=2,
            spatial_decimation_m=0.01,
            trail_point_cap=2,
            trail_duration_s=0.2,
        ),
    )
    for path in (
        "tracking/left/trail",
        "tracking/left/raw_trail",
        "tracking/body/whole_com/trail",
    ):
        logs = [entry for entry in rr.logs if entry[0] == path]
        assert len(logs) == 1
        assert logs[0][3] is True
