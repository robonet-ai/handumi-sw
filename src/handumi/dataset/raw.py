"""HandUMI raw robot-agnostic dataset schema.

The raw state is the source-of-truth representation recorded from the wearable
HandUMI hardware before any robot-specific IK or retargeting.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from handumi.tracking.transforms import Pose

HANDUMI_RAW_STATE_NAMES: tuple[str, ...] = (
    "left_x",
    "left_y",
    "left_z",
    "left_qx",
    "left_qy",
    "left_qz",
    "left_qw",
    "right_x",
    "right_y",
    "right_z",
    "right_qx",
    "right_qy",
    "right_qz",
    "right_qw",
    "left_gripper_width",
    "right_gripper_width",
)

HANDUMI_RAW_STATE_SIZE = len(HANDUMI_RAW_STATE_NAMES)
HANDUMI_TRACKING_SCHEMA = "controller_raw_compact"
HANDUMI_CAPTURE_SCHEMA = "synchronized_sources"
HANDUMI_STATE_SEMANTICS = "workspace_controller_pose7_plus_gripper_widths"

LEFT_POSE_SLICE = slice(0, 7)
RIGHT_POSE_SLICE = slice(7, 14)
LEFT_GRIPPER_INDEX = 14
RIGHT_GRIPPER_INDEX = 15

HANDUMI_RAW_IMAGE_KEYS: tuple[str, ...] = (
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.images.workspace",
)

TRACKING_VALIDITY_NAMES: tuple[str, ...] = (
    "left_device_tracked",
    "left_pose_valid",
    "right_device_tracked",
    "right_pose_valid",
    "hmd_tracked",
    "clock_synced",
    "connected",
    "streaming",
)


def canonical_body_features() -> dict[str, dict[str, Any]]:
    """Load optional body feature definitions without a package import cycle."""
    from handumi.body.model import canonical_body_features as _features

    return _features()


def raw_state_feature() -> dict[str, Any]:
    """Return the LeRobot feature metadata for raw state/action vectors."""
    return {
        "dtype": "float32",
        "shape": [HANDUMI_RAW_STATE_SIZE],
        "names": list(HANDUMI_RAW_STATE_NAMES),
    }


def pose7_feature() -> dict[str, Any]:
    """Return feature metadata for a single pose7 vector."""
    return {
        "dtype": "float32",
        "shape": (7,),
        "names": ["x", "y", "z", "qx", "qy", "qz", "qw"],
    }


def scalar_feature(dtype: str) -> dict[str, Any]:
    """Return feature metadata for a scalar stored as a one-element array."""
    return {"dtype": dtype, "shape": (1,), "names": None}


def raw_tracking_features() -> dict[str, Any]:
    """Compact tracking schema recorded for every HandUMI tracking backend.

    Workspace controller poses already live in ``observation.state``. The
    original device poses and the transform used to create that state remain
    available for later recalibration. Processed controller->TCP poses are
    intentionally not part of the raw dataset schema.
    """
    features: dict[str, Any] = {}
    for side in ("left", "right"):
        features[f"observation.tracking.{side}_device_controller_pose"] = (
            pose7_feature()
        )
        features[f"observation.tracking.{side}_tracked"] = scalar_feature("int64")
    features["observation.tracking.device_hmd_pose"] = pose7_feature()
    features["observation.tracking.workspace_from_device_pose"] = pose7_feature()
    features["observation.tracking.device_time_ns"] = scalar_feature("int64")
    features["observation.tracking.pc_monotonic_ns"] = scalar_feature("int64")
    features["observation.tracking.aligned_time_ns"] = scalar_feature("int64")
    features["observation.tracking.sequence"] = scalar_feature("int64")
    features["observation.valid"] = {
        "dtype": "int64",
        "shape": (len(TRACKING_VALIDITY_NAMES),),
        "names": list(TRACKING_VALIDITY_NAMES),
    }
    return features


def feetech_features() -> dict[str, Any]:
    """Common Feetech gripper encoder schema."""
    features: dict[str, Any] = {}
    for side in ("left", "right"):
        features[f"observation.feetech.{side}_ticks"] = scalar_feature("int64")
        features[f"observation.feetech.{side}_width_mm"] = scalar_feature("float32")
        features[f"observation.feetech.{side}_normalized"] = scalar_feature("float32")
    for key in ("sample_time_ns", "sequence", "healthy"):
        features[f"observation.feetech.{key}"] = scalar_feature("int64")
    return features


def capture_timing_features() -> dict[str, Any]:
    """Recorder target and wall-clock timing stored on every dataset row."""
    return {
        "observation.sync.target_time_ns": scalar_feature("int64"),
        "observation.sync.record_time_ns": scalar_feature("int64"),
    }


def camera_health_features(camera_names: Sequence[str]) -> dict[str, Any]:
    """Per-camera source timestamp, sequence, and health fields."""
    features: dict[str, Any] = {}
    for name in camera_names:
        prefix = f"observation.camera.{name}"
        for key in ("sample_time_ns", "sequence", "healthy"):
            features[f"{prefix}.{key}"] = scalar_feature("int64")
    return features


def validate_raw_state_shape(
    value: Sequence[object], *, name: str = "raw state"
) -> None:
    """Raise ``ValueError`` if ``value`` is not a single 16D raw state vector."""
    if len(value) != HANDUMI_RAW_STATE_SIZE:
        raise ValueError(
            f"Expected {name} length {HANDUMI_RAW_STATE_SIZE}, got {len(value)}."
        )


def pose_to_state_vector(
    left: "Pose",
    right: "Pose",
    left_width_m: float,
    right_width_m: float,
) -> np.ndarray:
    """Assemble the 16D HandUMI raw state from calibrated left/right poses + widths.

    Backend-neutral: any tracking source (Quest, PICO) that produces workspace
    ``Pose`` values plus gripper widths feeds the same raw-state layout.
    """
    state = np.zeros(HANDUMI_RAW_STATE_SIZE, dtype=np.float32)
    state[LEFT_POSE_SLICE] = np.concatenate([left.position, left.quaternion])
    state[RIGHT_POSE_SLICE] = np.concatenate([right.position, right.quaternion])
    state[LEFT_GRIPPER_INDEX] = float(left_width_m)
    state[RIGHT_GRIPPER_INDEX] = float(right_width_m)
    return state


__all__ = [
    "HANDUMI_RAW_IMAGE_KEYS",
    "HANDUMI_RAW_STATE_NAMES",
    "HANDUMI_RAW_STATE_SIZE",
    "HANDUMI_TRACKING_SCHEMA",
    "HANDUMI_CAPTURE_SCHEMA",
    "HANDUMI_STATE_SEMANTICS",
    "LEFT_GRIPPER_INDEX",
    "LEFT_POSE_SLICE",
    "RIGHT_GRIPPER_INDEX",
    "RIGHT_POSE_SLICE",
    "TRACKING_VALIDITY_NAMES",
    "pose_to_state_vector",
    "camera_health_features",
    "canonical_body_features",
    "capture_timing_features",
    "feetech_features",
    "pose7_feature",
    "raw_state_feature",
    "raw_tracking_features",
    "scalar_feature",
    "validate_raw_state_shape",
]
