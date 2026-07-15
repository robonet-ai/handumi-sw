"""Stable HandUMI canonical body schema.

Floating-point values are unavailable by default and therefore use NaN plus
explicit masks. A zero pose is never used as an availability sentinel.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

CANONICAL_BODY_SCHEMA = "handumi_canonical_25_v1"
TRACKING_SCHEMA = "handumi_tracking_v2"
BODY_PREFIX = "observation.body"
MAX_SUPPORT_POLYGON_VERTICES = 8


@dataclass(frozen=True)
class CanonicalJoint:
    index: int
    identifier: str
    parent_index: int


_JOINT_PARENT_IDS: tuple[tuple[str, str | None], ...] = (
    ("pelvis", None),
    ("spine_lower", "pelvis"),
    ("spine_middle", "spine_lower"),
    ("spine_upper", "spine_middle"),
    ("chest", "spine_upper"),
    ("neck", "chest"),
    ("head", "neck"),
    ("left_shoulder", "chest"),
    ("left_elbow", "left_shoulder"),
    ("left_wrist", "left_elbow"),
    ("left_hand", "left_wrist"),
    ("right_shoulder", "chest"),
    ("right_elbow", "right_shoulder"),
    ("right_wrist", "right_elbow"),
    ("right_hand", "right_wrist"),
    ("left_hip", "pelvis"),
    ("left_knee", "left_hip"),
    ("left_ankle", "left_knee"),
    ("left_heel", "left_ankle"),
    ("left_foot_ball", "left_ankle"),
    ("right_hip", "pelvis"),
    ("right_knee", "right_hip"),
    ("right_ankle", "right_knee"),
    ("right_heel", "right_ankle"),
    ("right_foot_ball", "right_ankle"),
)

_INDEX_BY_ID = {identifier: index for index, (identifier, _) in enumerate(_JOINT_PARENT_IDS)}
CANONICAL_JOINTS: tuple[CanonicalJoint, ...] = tuple(
    CanonicalJoint(
        index=index,
        identifier=identifier,
        parent_index=-1 if parent is None else _INDEX_BY_ID[parent],
    )
    for index, (identifier, parent) in enumerate(_JOINT_PARENT_IDS)
)
CANONICAL_JOINT_COUNT = len(CANONICAL_JOINTS)

# PICO's public BodyTrackerRole order. Keeping the table here makes index-based
# packets self-describing and prevents accidental mapping by a guessed order.
PICO_BODY_24_SOURCE_NAMES: tuple[str, ...] = (
    "Pelvis",
    "LEFT_HIP",
    "RIGHT_HIP",
    "SPINE1",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "SPINE2",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "SPINE3",
    "LEFT_FOOT",
    "RIGHT_FOOT",
    "NECK",
    "LEFT_COLLAR",
    "RIGHT_COLLAR",
    "HEAD",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HAND",
    "RIGHT_HAND",
)


class CanonicalProvenance(IntEnum):
    UNAVAILABLE = 0
    PLATFORM_ESTIMATED = 1
    DEVICE_REPORTED = 2
    EXTERNAL_TRACKER = 3
    INFERRED = 4
    SYNTHETIC_TEST = 5
    UNKNOWN = 6


class CanonicalTrackingState(IntEnum):
    INVALID = 0
    VALID = 1
    TRACKED = 2


class CanonicalClockQuality(IntEnum):
    UNAVAILABLE = 0
    RECEIVE_ONLY = 1
    MAPPED_UNBOUNDED = 2
    SYNCHRONIZED = 3
    DIAGNOSTIC_ONLY = 4


class ComProvenance(IntEnum):
    UNAVAILABLE = 0
    KINEMATIC_INFERRED = 1
    FUSED_ESTIMATED = 2


class ComDiagnostic(IntEnum):
    UNAVAILABLE = 0
    VALID = 1
    MISSING_LANDMARKS = 2
    UNRESOLVED_MASS = 3
    EXCESSIVE_UNCERTAINTY = 4
    GROUND_UNAVAILABLE = 5
    TRAJECTORY_BOUNDARY = 6
    TIMING_INVALID = 7
    RELOCALIZATION = 8


@dataclass(frozen=True)
class CanonicalBodyFrame:
    """One aligned canonical body observation in ``handumi_world``."""

    joint_pose: np.ndarray
    position_valid: np.ndarray
    orientation_valid: np.ndarray
    tracking_state: np.ndarray
    confidence: np.ndarray
    provenance: np.ndarray
    platform_root_pose: np.ndarray
    platform_root_position_valid: np.ndarray
    platform_root_orientation_valid: np.ndarray
    whole_com: np.ndarray
    whole_com_valid: np.ndarray
    whole_com_confidence: np.ndarray
    whole_com_covariance: np.ndarray
    whole_com_provenance: np.ndarray
    whole_com_diagnostic: np.ndarray
    whole_com_unresolved_mass_fraction: np.ndarray
    whole_com_ground_projection: np.ndarray
    whole_com_ground_projection_valid: np.ndarray
    whole_com_ground_projection_covariance: np.ndarray
    whole_com_velocity: np.ndarray
    whole_com_velocity_valid: np.ndarray
    whole_com_acceleration: np.ndarray
    whole_com_acceleration_valid: np.ndarray
    whole_com_trajectory_diagnostic: np.ndarray
    segment_com: np.ndarray
    segment_com_valid: np.ndarray
    segment_com_confidence: np.ndarray
    segment_com_covariance: np.ndarray
    segment_com_provenance: np.ndarray
    segment_mass_fraction: np.ndarray
    ground_plane: np.ndarray
    contact_probability: np.ndarray
    contact_valid: np.ndarray
    contact_provenance: np.ndarray
    support_polygon: np.ndarray
    support_polygon_valid: np.ndarray
    center_of_pressure: np.ndarray
    center_of_pressure_valid: np.ndarray
    source_time_ns: np.ndarray
    mapped_time_ns: np.ndarray
    receive_time_ns: np.ndarray
    clock_offset_ns: np.ndarray
    rtt_ns: np.ndarray
    uncertainty_ns: np.ndarray
    clock_quality: np.ndarray
    source_sequence: np.ndarray

    @classmethod
    def empty(cls) -> "CanonicalBodyFrame":
        nan = np.float32(np.nan)
        return cls(
            joint_pose=np.full((CANONICAL_JOINT_COUNT, 7), nan, dtype=np.float32),
            position_valid=np.zeros(CANONICAL_JOINT_COUNT, dtype=np.int64),
            orientation_valid=np.zeros(CANONICAL_JOINT_COUNT, dtype=np.int64),
            tracking_state=np.zeros(CANONICAL_JOINT_COUNT, dtype=np.int64),
            confidence=np.full(CANONICAL_JOINT_COUNT, nan, dtype=np.float32),
            provenance=np.zeros(CANONICAL_JOINT_COUNT, dtype=np.int64),
            platform_root_pose=np.full(7, nan, dtype=np.float32),
            platform_root_position_valid=np.zeros(1, dtype=np.int64),
            platform_root_orientation_valid=np.zeros(1, dtype=np.int64),
            whole_com=np.full(3, nan, dtype=np.float32),
            whole_com_valid=np.zeros(1, dtype=np.int64),
            whole_com_confidence=np.full(1, nan, dtype=np.float32),
            whole_com_covariance=np.full((3, 3), nan, dtype=np.float32),
            whole_com_provenance=np.zeros(1, dtype=np.int64),
            whole_com_diagnostic=np.zeros(1, dtype=np.int64),
            whole_com_unresolved_mass_fraction=np.full(1, nan, dtype=np.float32),
            whole_com_ground_projection=np.full(3, nan, dtype=np.float32),
            whole_com_ground_projection_valid=np.zeros(1, dtype=np.int64),
            whole_com_ground_projection_covariance=np.full(
                (3, 3), nan, dtype=np.float32
            ),
            whole_com_velocity=np.full(3, nan, dtype=np.float32),
            whole_com_velocity_valid=np.zeros(1, dtype=np.int64),
            whole_com_acceleration=np.full(3, nan, dtype=np.float32),
            whole_com_acceleration_valid=np.zeros(1, dtype=np.int64),
            whole_com_trajectory_diagnostic=np.zeros(1, dtype=np.int64),
            segment_com=np.full((CANONICAL_JOINT_COUNT, 3), nan, dtype=np.float32),
            segment_com_valid=np.zeros(CANONICAL_JOINT_COUNT, dtype=np.int64),
            segment_com_confidence=np.full(
                CANONICAL_JOINT_COUNT, nan, dtype=np.float32
            ),
            segment_com_covariance=np.full(
                (CANONICAL_JOINT_COUNT, 3, 3), nan, dtype=np.float32
            ),
            segment_com_provenance=np.zeros(
                CANONICAL_JOINT_COUNT, dtype=np.int64
            ),
            segment_mass_fraction=np.zeros(
                CANONICAL_JOINT_COUNT, dtype=np.float32
            ),
            ground_plane=np.full(4, nan, dtype=np.float32),
            contact_probability=np.full(4, nan, dtype=np.float32),
            contact_valid=np.zeros(4, dtype=np.int64),
            contact_provenance=np.zeros(4, dtype=np.int64),
            support_polygon=np.full(
                (MAX_SUPPORT_POLYGON_VERTICES, 3), nan, dtype=np.float32
            ),
            support_polygon_valid=np.zeros(
                MAX_SUPPORT_POLYGON_VERTICES, dtype=np.int64
            ),
            center_of_pressure=np.full(3, nan, dtype=np.float32),
            center_of_pressure_valid=np.zeros(1, dtype=np.int64),
            source_time_ns=np.zeros(1, dtype=np.int64),
            mapped_time_ns=np.zeros(1, dtype=np.int64),
            receive_time_ns=np.zeros(1, dtype=np.int64),
            clock_offset_ns=np.zeros(1, dtype=np.int64),
            rtt_ns=np.zeros(1, dtype=np.int64),
            uncertainty_ns=np.zeros(1, dtype=np.int64),
            clock_quality=np.zeros(1, dtype=np.int64),
            source_sequence=np.full(1, -1, dtype=np.int64),
        )

    def observation(self) -> dict[str, np.ndarray]:
        return {
            f"{BODY_PREFIX}.joint_pose": self.joint_pose,
            f"{BODY_PREFIX}.position_valid": self.position_valid,
            f"{BODY_PREFIX}.orientation_valid": self.orientation_valid,
            f"{BODY_PREFIX}.tracking_state": self.tracking_state,
            f"{BODY_PREFIX}.confidence": self.confidence,
            f"{BODY_PREFIX}.provenance": self.provenance,
            f"{BODY_PREFIX}.platform_root_pose": self.platform_root_pose,
            f"{BODY_PREFIX}.platform_root_position_valid": self.platform_root_position_valid,
            f"{BODY_PREFIX}.platform_root_orientation_valid": self.platform_root_orientation_valid,
            f"{BODY_PREFIX}.whole_com": self.whole_com,
            f"{BODY_PREFIX}.whole_com_valid": self.whole_com_valid,
            f"{BODY_PREFIX}.whole_com_confidence": self.whole_com_confidence,
            f"{BODY_PREFIX}.whole_com_covariance": self.whole_com_covariance,
            f"{BODY_PREFIX}.whole_com_provenance": self.whole_com_provenance,
            f"{BODY_PREFIX}.whole_com_diagnostic": self.whole_com_diagnostic,
            f"{BODY_PREFIX}.whole_com_unresolved_mass_fraction": (
                self.whole_com_unresolved_mass_fraction
            ),
            f"{BODY_PREFIX}.whole_com_ground_projection": self.whole_com_ground_projection,
            f"{BODY_PREFIX}.whole_com_ground_projection_valid": (
                self.whole_com_ground_projection_valid
            ),
            f"{BODY_PREFIX}.whole_com_ground_projection_covariance": (
                self.whole_com_ground_projection_covariance
            ),
            f"{BODY_PREFIX}.whole_com_velocity": self.whole_com_velocity,
            f"{BODY_PREFIX}.whole_com_velocity_valid": self.whole_com_velocity_valid,
            f"{BODY_PREFIX}.whole_com_acceleration": self.whole_com_acceleration,
            f"{BODY_PREFIX}.whole_com_acceleration_valid": self.whole_com_acceleration_valid,
            f"{BODY_PREFIX}.whole_com_trajectory_diagnostic": self.whole_com_trajectory_diagnostic,
            f"{BODY_PREFIX}.segment_com": self.segment_com,
            f"{BODY_PREFIX}.segment_com_valid": self.segment_com_valid,
            f"{BODY_PREFIX}.segment_com_confidence": self.segment_com_confidence,
            f"{BODY_PREFIX}.segment_com_covariance": self.segment_com_covariance,
            f"{BODY_PREFIX}.segment_com_provenance": self.segment_com_provenance,
            f"{BODY_PREFIX}.segment_mass_fraction": self.segment_mass_fraction,
            f"{BODY_PREFIX}.ground_plane": self.ground_plane,
            f"{BODY_PREFIX}.contact_probability": self.contact_probability,
            f"{BODY_PREFIX}.contact_valid": self.contact_valid,
            f"{BODY_PREFIX}.contact_provenance": self.contact_provenance,
            f"{BODY_PREFIX}.support_polygon": self.support_polygon,
            f"{BODY_PREFIX}.support_polygon_valid": self.support_polygon_valid,
            f"{BODY_PREFIX}.center_of_pressure": self.center_of_pressure,
            f"{BODY_PREFIX}.center_of_pressure_valid": self.center_of_pressure_valid,
            f"{BODY_PREFIX}.source_time_ns": self.source_time_ns,
            f"{BODY_PREFIX}.mapped_time_ns": self.mapped_time_ns,
            f"{BODY_PREFIX}.receive_time_ns": self.receive_time_ns,
            f"{BODY_PREFIX}.clock_offset_ns": self.clock_offset_ns,
            f"{BODY_PREFIX}.rtt_ns": self.rtt_ns,
            f"{BODY_PREFIX}.uncertainty_ns": self.uncertainty_ns,
            f"{BODY_PREFIX}.clock_quality": self.clock_quality,
            f"{BODY_PREFIX}.source_sequence": self.source_sequence,
        }


def _feature(dtype: str, shape: tuple[int, ...], names: list[str] | None = None) -> dict[str, Any]:
    return {"dtype": dtype, "shape": shape, "names": names}


def canonical_body_features() -> dict[str, dict[str, Any]]:
    joint_names = [joint.identifier for joint in CANONICAL_JOINTS]
    features = {
        f"{BODY_PREFIX}.joint_pose": _feature("float32", (CANONICAL_JOINT_COUNT, 7)),
        f"{BODY_PREFIX}.position_valid": _feature("int64", (CANONICAL_JOINT_COUNT,), joint_names),
        f"{BODY_PREFIX}.orientation_valid": _feature(
            "int64", (CANONICAL_JOINT_COUNT,), joint_names
        ),
        f"{BODY_PREFIX}.tracking_state": _feature("int64", (CANONICAL_JOINT_COUNT,), joint_names),
        f"{BODY_PREFIX}.confidence": _feature("float32", (CANONICAL_JOINT_COUNT,), joint_names),
        f"{BODY_PREFIX}.provenance": _feature("int64", (CANONICAL_JOINT_COUNT,), joint_names),
        f"{BODY_PREFIX}.platform_root_pose": _feature("float32", (7,)),
        f"{BODY_PREFIX}.platform_root_position_valid": _feature("int64", (1,)),
        f"{BODY_PREFIX}.platform_root_orientation_valid": _feature("int64", (1,)),
        f"{BODY_PREFIX}.whole_com": _feature("float32", (3,), ["x", "y", "z"]),
        f"{BODY_PREFIX}.whole_com_valid": _feature("int64", (1,)),
        f"{BODY_PREFIX}.whole_com_confidence": _feature("float32", (1,)),
        f"{BODY_PREFIX}.whole_com_covariance": _feature("float32", (3, 3)),
        f"{BODY_PREFIX}.whole_com_provenance": _feature("int64", (1,)),
        f"{BODY_PREFIX}.whole_com_diagnostic": _feature("int64", (1,)),
        f"{BODY_PREFIX}.whole_com_unresolved_mass_fraction": _feature(
            "float32", (1,)
        ),
        f"{BODY_PREFIX}.whole_com_ground_projection": _feature(
            "float32", (3,), ["x", "y", "z"]
        ),
        f"{BODY_PREFIX}.whole_com_ground_projection_valid": _feature(
            "int64", (1,)
        ),
        f"{BODY_PREFIX}.whole_com_ground_projection_covariance": _feature(
            "float32", (3, 3)
        ),
        f"{BODY_PREFIX}.whole_com_velocity": _feature(
            "float32", (3,), ["vx", "vy", "vz"]
        ),
        f"{BODY_PREFIX}.whole_com_velocity_valid": _feature("int64", (1,)),
        f"{BODY_PREFIX}.whole_com_acceleration": _feature(
            "float32", (3,), ["ax", "ay", "az"]
        ),
        f"{BODY_PREFIX}.whole_com_acceleration_valid": _feature("int64", (1,)),
        f"{BODY_PREFIX}.whole_com_trajectory_diagnostic": _feature("int64", (1,)),
        f"{BODY_PREFIX}.segment_com": _feature("float32", (CANONICAL_JOINT_COUNT, 3)),
        f"{BODY_PREFIX}.segment_com_valid": _feature(
            "int64", (CANONICAL_JOINT_COUNT,), joint_names
        ),
        f"{BODY_PREFIX}.segment_com_confidence": _feature(
            "float32", (CANONICAL_JOINT_COUNT,), joint_names
        ),
        f"{BODY_PREFIX}.segment_com_covariance": _feature(
            "float32", (CANONICAL_JOINT_COUNT, 3, 3)
        ),
        f"{BODY_PREFIX}.segment_com_provenance": _feature(
            "int64", (CANONICAL_JOINT_COUNT,), joint_names
        ),
        f"{BODY_PREFIX}.segment_mass_fraction": _feature(
            "float32", (CANONICAL_JOINT_COUNT,), joint_names
        ),
        f"{BODY_PREFIX}.ground_plane": _feature("float32", (4,), ["nx", "ny", "nz", "d"]),
        f"{BODY_PREFIX}.contact_probability": _feature(
            "float32", (4,), ["left_heel", "left_ball", "right_heel", "right_ball"]
        ),
        f"{BODY_PREFIX}.contact_valid": _feature(
            "int64", (4,), ["left_heel", "left_ball", "right_heel", "right_ball"]
        ),
        f"{BODY_PREFIX}.contact_provenance": _feature(
            "int64", (4,), ["left_heel", "left_ball", "right_heel", "right_ball"]
        ),
        f"{BODY_PREFIX}.support_polygon": _feature(
            "float32", (MAX_SUPPORT_POLYGON_VERTICES, 3)
        ),
        f"{BODY_PREFIX}.support_polygon_valid": _feature(
            "int64", (MAX_SUPPORT_POLYGON_VERTICES,)
        ),
        f"{BODY_PREFIX}.center_of_pressure": _feature(
            "float32", (3,), ["x", "y", "z"]
        ),
        f"{BODY_PREFIX}.center_of_pressure_valid": _feature("int64", (1,)),
    }
    for key in (
        "source_time_ns",
        "mapped_time_ns",
        "receive_time_ns",
        "clock_offset_ns",
        "rtt_ns",
        "uncertainty_ns",
        "clock_quality",
        "source_sequence",
    ):
        features[f"{BODY_PREFIX}.{key}"] = _feature("int64", (1,))
    return features


def canonical_body_metadata(*, transforms: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    from handumi.body.mapping import META_TO_CANONICAL, PICO_TO_CANONICAL

    transform_table = transforms or []
    calibration_hash = hashlib.sha256(
        json.dumps(transform_table, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "tracking_schema": TRACKING_SCHEMA,
        "body_schema": CANONICAL_BODY_SCHEMA,
        "canonical_joint_table": [
            {
                "index": joint.index,
                "id": joint.identifier,
                "parent_index": joint.parent_index,
            }
            for joint in CANONICAL_JOINTS
        ],
        "source_mappings": {
            "meta_ovr_body_70_84": {
                "mapping": META_TO_CANONICAL,
                "source_pose_convention": "Unity_left_handed_x_right_y_up_z_forward",
                "hierarchy_reference": (
                    "Meta XR SDK 74 OVRSkeletonMapping/OVRHumanBodyBonesMappings"
                ),
                "platform_root_preserved_separately": True,
            },
            "pico_body_tracker_role_24": {
                "source_joint_order": list(PICO_BODY_24_SOURCE_NAMES),
                "mapping": PICO_TO_CANONICAL,
                "source_pose_convention": "PICO_SDK_right_handed_source_space",
                "hierarchy_reference": (
                    "XR-Robotics/XRoboToolkit-Unity-Client "
                    "PXR_Plugin.BodyTrackerRole"
                ),
                "platform_root_preserved_separately": False,
            },
        },
        "heel_policy": {
            "direct_source": None,
            "aligned_value": "unavailable_NaN",
            "future_estimate_provenance": "INFERRED",
        },
        "coordinate_frame": {
            "id": "handumi_world",
            "handedness": "right",
            "axes": {"x": "initial_horizontal_heading", "y": "left", "z": "up"},
            "origin": "calibrated_ground_plane",
        },
        "transforms": transform_table,
        "calibration_hash": calibration_hash,
        "estimator_version": "not_run",
        "runtime_version": "captured_in_native_sidecar",
        "missing_float_encoding": "NaN_plus_explicit_mask",
    }


__all__ = [
    "BODY_PREFIX",
    "CANONICAL_BODY_SCHEMA",
    "CANONICAL_JOINT_COUNT",
    "CANONICAL_JOINTS",
    "MAX_SUPPORT_POLYGON_VERTICES",
    "PICO_BODY_24_SOURCE_NAMES",
    "TRACKING_SCHEMA",
    "CanonicalBodyFrame",
    "CanonicalClockQuality",
    "ComDiagnostic",
    "ComProvenance",
    "CanonicalJoint",
    "CanonicalProvenance",
    "CanonicalTrackingState",
    "canonical_body_features",
    "canonical_body_metadata",
]
