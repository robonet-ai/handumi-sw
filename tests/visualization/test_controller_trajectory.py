from __future__ import annotations

import numpy as np

from handumi.body.model import CanonicalBodyFrame
from handumi.feetech import GripperWidths
from handumi.tracking.base import ControllerPairSample
from handumi.utils.trajectory import TrajectoryTrail
from handumi.visualization import LEFT_COLOR, RIGHT_COLOR
from handumi.visualization.controller_trajectory import (
    CONTROLLER_VIEW_NAME,
    LEFT_WIDTH_PATH,
    RIGHT_WIDTH_PATH,
    LiveRerunStream,
    controller_path,
    controller_render_plan,
)


def _pose(x: float) -> np.ndarray:
    return np.array([x, 0, 0, 0, 0, 0, 1], dtype=np.float32)


def test_legacy_controller_paths_view_name_and_colors_are_unchanged():
    assert CONTROLLER_VIEW_NAME == "controller_trajectory"
    assert LEFT_COLOR == (255, 190, 50)
    assert RIGHT_COLOR == (80, 220, 130)
    assert LEFT_WIDTH_PATH == "observation.feetech.left_width_mm"
    assert RIGHT_WIDTH_PATH == "observation.feetech.right_width_mm"
    assert controller_path("left", "tcp") == "tracking/left/tcp"
    assert controller_path("right", "raw_trail") == "tracking/right/raw_trail"


def test_controller_plan_retains_exact_solid_and_faint_geometry():
    trail = TrajectoryTrail(3)
    raw_trail = TrajectoryTrail(3)
    controller_render_plan("left", _pose(0), _pose(1), trail, raw_trail, LEFT_COLOR)
    plan = controller_render_plan(
        "left", _pose(2), _pose(3), trail, raw_trail, LEFT_COLOR
    )
    by_path = {operation.path: operation for operation in plan}
    assert by_path["tracking/left/tcp"].kwargs == {
        "colors": [LEFT_COLOR],
        "radii": 0.012,
    }
    assert by_path["tracking/left/raw"].kwargs == {
        "colors": [[*LEFT_COLOR, 90]],
        "radii": 0.007,
    }
    assert by_path["tracking/left/trail"].kwargs["radii"] == 0.003
    assert by_path["tracking/left/raw_trail"].kwargs["radii"] == 0.0015


class _Archetype:
    def __init__(self, *args, **kwargs):
        pass

    def compress(self, **kwargs):
        return self


class _FailingRerun:
    Clear = Points3D = LineStrips3D = Scalars = Image = Mesh3D = TextDocument = (
        _Archetype
    )

    def log(self, *args, **kwargs):
        raise RuntimeError("viewer broke")


def test_live_visualization_failure_is_nonfatal_and_disables_future_logs():
    errors = []
    stream = LiveRerunStream(_FailingRerun(), fps=30, on_error=errors.append)
    sample = ControllerPairSample.empty("meta")
    widths = GripperWidths(
        left=0,
        right=0,
        left_mm=0,
        right_mm=0,
        left_normalized=0,
        right_normalized=0,
        left_ticks=0,
        right_ticks=0,
    )
    stream.log_frame({}, sample, widths, body_frame=CanonicalBodyFrame.empty())
    stream.log_frame({}, sample, widths, body_frame=CanonicalBodyFrame.empty())
    assert stream.healthy is False
    assert len(errors) == 1
