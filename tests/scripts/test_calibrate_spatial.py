from __future__ import annotations

import numpy as np

from handumi.calibration.spatial import CameraIntrinsics, CharucoBoardSpec
from handumi.scripts.setup.calibrate_spatial import (
    _init_rerun_view,
    _log_camera_model,
    _log_camera_pose,
    _rectification_maps,
)


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
        views=30,
    )

    _log_camera_model(rr, "left_wrist", intrinsics, (255, 190, 50))
    _log_camera_pose(rr, "left_wrist", np.eye(4))

    matrix, map_x, map_y = _rectification_maps(intrinsics)
    assert matrix.shape == (3, 3)
    assert map_x.shape == (480, 640)
    assert map_y.shape == (480, 640)
