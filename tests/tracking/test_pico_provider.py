from __future__ import annotations

import numpy as np

from handumi.calibration.control_tcp import ControllerTcpCalibration
from handumi.robots.utils import IDENTITY_POSE7
from handumi.tracking.pico import PicoTrackingProvider


class _FakeXrt:
    def get_time_stamp_ns(self) -> int:
        return 123

    def get_headset_pose(self):
        return np.array([0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def get_left_controller_pose(self):
        return np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def get_right_controller_pose(self):
        return np.array([-1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def is_body_data_available(self) -> bool:
        return False

    def get_left_hand_is_active(self) -> bool:
        return False

    def get_right_hand_is_active(self) -> bool:
        return False


def _identity_calibration() -> ControllerTcpCalibration:
    pose = IDENTITY_POSE7.astype(np.float32)
    return ControllerTcpCalibration(left=pose.copy(), right=pose.copy(), source=None)


def test_pico_provider_applies_workspace_but_preserves_device_poses():
    provider = PicoTrackingProvider(calibration=_identity_calibration())
    provider.xrt = _FakeXrt()
    table_from_pico = np.array([0.5, -0.25, 0.1, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    provider.set_workspace_from_device_pose(table_from_pico, locked=True)
    sample = provider.latest()

    np.testing.assert_allclose(sample.left_device_controller_pose[:3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(sample.left_controller_pose[:3], [1.5, 1.75, 3.1])
    np.testing.assert_allclose(sample.workspace_from_device_pose, table_from_pico)
    assert sample.left_tracked
    assert sample.streaming
