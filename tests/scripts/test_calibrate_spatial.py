from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from handumi.calibration.spatial import CameraIntrinsics, CharucoBoardSpec
from handumi.robots.utils import mat_to_pose7
from handumi.scripts.setup.calibrate_spatial import (
    _controller_is_stable,
    _init_rerun_view,
    _log_camera_model,
    _log_camera_pose,
    _pose_is_distinct,
    _rectification_maps,
)


def _rotation_pose(degrees: float) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("x", degrees, degrees=True).as_matrix()
    return mat_to_pose7(transform)


def test_auto_capture_requires_distinct_controller_rotation():
    accepted = [_rotation_pose(0.0)]

    assert not _pose_is_distinct(_rotation_pose(4.0), accepted)
    assert _pose_is_distinct(_rotation_pose(12.0), accepted)


def test_controller_stability_tolerates_tracking_jitter_but_not_motion():
    reference = _rotation_pose(0.0)

    assert _controller_is_stable(_rotation_pose(2.0), reference)
    assert not _controller_is_stable(_rotation_pose(10.0), reference)


def test_rerun_calibration_archetypes_are_valid():
    rr = _init_rerun_view(["left_wrist"], CharucoBoardSpec(), spawn=False)
    intrinsics = CameraIntrinsics(
        camera="left_wrist",
        width=640,
        height=480,
        matrix=np.array([[300, 0, 320], [0, 300, 240], [0, 0, 1]], dtype=float),
        distortion=np.zeros((4, 1)),
        rms_px=0.2,
        mean_error_px=0.2,
        views=15,
    )

    _log_camera_model(rr, "left_wrist", intrinsics, (255, 190, 50))
    _log_camera_pose(rr, "left_wrist", np.eye(4))

    matrix, map_x, map_y = _rectification_maps(intrinsics)
    assert matrix.shape == (3, 3)
    assert map_x.shape == (480, 640)
    assert map_y.shape == (480, 640)
