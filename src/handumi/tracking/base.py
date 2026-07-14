"""Common tracking provider contracts for HandUMI recorders."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Protocol

import numpy as np

from handumi.calibration.control_tcp import ControllerTcpCalibration
from handumi.robots.utils import IDENTITY_POSE7, pose_mul, quat_normalize


@dataclass(frozen=True)
class ControllerPairSample:
    """One normalized left/right controller sample from a tracking backend.

    ``*_controller_pose`` is a pose7 controller-anchor pose in the backend's
    normalized recording frame. ``*_tcp_pose`` is derived by applying the same
    controller->gripper-TCP calibration used by replay/conversion.
    """

    device: str
    left_controller_pose: np.ndarray
    right_controller_pose: np.ndarray
    left_tcp_pose: np.ndarray
    right_tcp_pose: np.ndarray
    left_tracked: bool
    right_tracked: bool
    left_device_tracked: bool = False
    right_device_tracked: bool = False
    left_pose_valid: bool = False
    right_pose_valid: bool = False
    hmd_pose: np.ndarray = field(
        default_factory=lambda: IDENTITY_POSE7.astype(np.float32).copy()
    )
    hmd_tracked: bool = False
    workspace_from_device_pose: np.ndarray = field(
        default_factory=lambda: IDENTITY_POSE7.astype(np.float32).copy()
    )
    device_time_ns: int = 0
    pc_monotonic_ns: int = 0
    aligned_time_ns: int = 0
    clock_offset_ns: int = 0
    clock_synced: bool = False
    connected: bool = False
    streaming: bool = False
    sequence: int = 0

    @classmethod
    def empty(cls, device: str) -> "ControllerPairSample":
        pose = IDENTITY_POSE7.astype(np.float32)
        return cls(
            device=device,
            left_controller_pose=pose.copy(),
            right_controller_pose=pose.copy(),
            left_tcp_pose=pose.copy(),
            right_tcp_pose=pose.copy(),
            left_tracked=False,
            right_tracked=False,
        )

    def tracking_frame(self) -> dict[str, np.ndarray]:
        return {
            "observation.tracking.left_controller_pose": self.left_controller_pose,
            "observation.tracking.right_controller_pose": self.right_controller_pose,
            "observation.tracking.left_tcp_pose": self.left_tcp_pose,
            "observation.tracking.right_tcp_pose": self.right_tcp_pose,
            "observation.tracking.left_tracked": np.array(
                [int(self.left_tracked)], dtype=np.int64
            ),
            "observation.tracking.right_tracked": np.array(
                [int(self.right_tracked)], dtype=np.int64
            ),
            "observation.tracking.left_device_tracked": np.array(
                [int(self.left_device_tracked)], dtype=np.int64
            ),
            "observation.tracking.right_device_tracked": np.array(
                [int(self.right_device_tracked)], dtype=np.int64
            ),
            "observation.tracking.left_pose_valid": np.array(
                [int(self.left_pose_valid)], dtype=np.int64
            ),
            "observation.tracking.right_pose_valid": np.array(
                [int(self.right_pose_valid)], dtype=np.int64
            ),
            "observation.tracking.hmd_pose": self.hmd_pose,
            "observation.tracking.hmd_tracked": np.array(
                [int(self.hmd_tracked)], dtype=np.int64
            ),
            "observation.tracking.workspace_from_device_pose": self.workspace_from_device_pose,
            "observation.tracking.device_time_ns": np.array(
                [int(self.device_time_ns)], dtype=np.int64
            ),
            "observation.tracking.pc_monotonic_ns": np.array(
                [int(self.pc_monotonic_ns)], dtype=np.int64
            ),
            "observation.tracking.aligned_time_ns": np.array(
                [int(self.aligned_time_ns)], dtype=np.int64
            ),
            "observation.tracking.clock_offset_ns": np.array(
                [int(self.clock_offset_ns)], dtype=np.int64
            ),
            "observation.tracking.clock_synced": np.array(
                [int(self.clock_synced)], dtype=np.int64
            ),
            "observation.tracking.connected": np.array(
                [int(self.connected)], dtype=np.int64
            ),
            "observation.tracking.streaming": np.array(
                [int(self.streaming)], dtype=np.int64
            ),
            "observation.tracking.sequence": np.array(
                [int(self.sequence)], dtype=np.int64
            ),
        }


class TrackingProvider(Protocol):
    device: str

    def start(self) -> None:
        """Start the tracking backend."""

    def stop(self) -> None:
        """Stop and release backend resources."""

    def latest(self) -> ControllerPairSample:
        """Return the latest normalized controller pair sample."""
        ...


class LegacyControllerProviderAdapter:
    """Explicitly retain the controller-only provider boundary during migration."""

    def __init__(self, provider: TrackingProvider) -> None:
        self.provider = provider
        self.device = provider.device

    def start(self) -> None:
        self.provider.start()

    def stop(self) -> None:
        self.provider.stop()

    def latest(self) -> ControllerPairSample:
        return self.provider.latest()

    def sample_at(self, target_time_ns: int) -> ControllerPairSample:
        sampler = getattr(self.provider, "sample_at", None)
        if callable(sampler):
            return sampler(target_time_ns)
        return self.provider.latest()


def as_pose7(value: object) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float32).reshape(-1)
    out = IDENTITY_POSE7.astype(np.float32).copy()
    n = min(len(pose), 7)
    out[:n] = pose[:n]
    out[3:7] = quat_normalize(out[3:7]).astype(np.float32)
    return out


def apply_tcp_calibration_pose7(
    left_controller_pose: np.ndarray,
    right_controller_pose: np.ndarray,
    calibration: ControllerTcpCalibration,
) -> tuple[np.ndarray, np.ndarray]:
    left = pose_mul(as_pose7(left_controller_pose), calibration.left).astype(np.float32)
    right = pose_mul(as_pose7(right_controller_pose), calibration.right).astype(
        np.float32
    )
    left[3:7] = quat_normalize(left[3:7]).astype(np.float32)
    right[3:7] = quat_normalize(right[3:7]).astype(np.float32)
    return left, right


__all__ = [
    "ControllerPairSample",
    "LegacyControllerProviderAdapter",
    "TrackingProvider",
    "apply_tcp_calibration_pose7",
    "as_pose7",
]
