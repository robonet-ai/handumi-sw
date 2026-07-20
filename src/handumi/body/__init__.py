"""Canonical human-body representation and source mappings."""

from handumi.body.com import (
    COM_ESTIMATOR_SCHEMA,
    AnthropometricTable,
    BodyProfile,
    ComEstimatorConfig,
    KinematicComEstimator,
    SegmentDefinition,
    default_anthropometric_table,
)
from handumi.body.calibration import (
    BODY_CALIBRATION_SCHEMA,
    PROFILE_SKELETON_SCHEMA,
    ProfileConstrainedSkeleton,
    ProfileNeutralCalibration,
    estimate_profile_neutral_calibration,
)
from handumi.body.mapping import canonical_body_from_packet
from handumi.body.model import (
    CANONICAL_BODY_SCHEMA,
    CANONICAL_JOINT_COUNT,
    CANONICAL_JOINTS,
    PICO_BODY_24_SOURCE_NAMES,
    CanonicalBodyFrame,
    CanonicalJoint,
    CanonicalProvenance,
    ComDiagnostic,
    ComProvenance,
    canonical_body_features,
    canonical_body_metadata,
)

__all__ = [
    "CANONICAL_BODY_SCHEMA",
    "CANONICAL_JOINT_COUNT",
    "CANONICAL_JOINTS",
    "COM_ESTIMATOR_SCHEMA",
    "BODY_CALIBRATION_SCHEMA",
    "PROFILE_SKELETON_SCHEMA",
    "PICO_BODY_24_SOURCE_NAMES",
    "CanonicalBodyFrame",
    "CanonicalJoint",
    "CanonicalProvenance",
    "ComDiagnostic",
    "ComEstimatorConfig",
    "ComProvenance",
    "AnthropometricTable",
    "BodyProfile",
    "ProfileConstrainedSkeleton",
    "ProfileNeutralCalibration",
    "KinematicComEstimator",
    "SegmentDefinition",
    "canonical_body_features",
    "canonical_body_from_packet",
    "canonical_body_metadata",
    "default_anthropometric_table",
    "estimate_profile_neutral_calibration",
]
