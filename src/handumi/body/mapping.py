"""Deterministic source-skeleton to ``handumi_canonical_25_v1`` mappings."""

from __future__ import annotations

import numpy as np

from handumi.body.model import (
    CANONICAL_JOINTS,
    PICO_BODY_24_SOURCE_NAMES,
    CanonicalBodyFrame,
    CanonicalClockQuality,
    CanonicalProvenance,
    CanonicalTrackingState,
)
from handumi.tracking.packet import (
    JointSample,
    JointTrackingState,
    SourceProvenance,
    TimestampQuality,
    TrackingPacket,
)
from handumi.tracking.transforms import (
    HandumiWorldCalibration,
    unity_position_to_handumi,
    unity_quaternion_to_handumi,
)


_CANONICAL_INDEX = {joint.identifier: joint.index for joint in CANONICAL_JOINTS}

META_TO_CANONICAL: dict[str, str] = {
    "Hips": "pelvis",
    "SpineLower": "spine_lower",
    "SpineMiddle": "spine_middle",
    "SpineUpper": "spine_upper",
    "Chest": "chest",
    "Neck": "neck",
    "Head": "head",
    "LeftArmUpper": "left_shoulder",
    "LeftArmLower": "left_elbow",
    "LeftHandWrist": "left_wrist",
    # Canonical ``*_hand`` is the distal hand endpoint used by the hand segment
    # model. Meta FullBody provides the middle fingertip directly; mapping the
    # palm here shortened a ~0.19 m hand to ~0.043 m. The raw palm remains in
    # the native sidecar.
    "LeftHandMiddleTip": "left_hand",
    "RightArmUpper": "right_shoulder",
    "RightArmLower": "right_elbow",
    "RightHandWrist": "right_wrist",
    "RightHandMiddleTip": "right_hand",
    "LeftUpperLeg": "left_hip",
    "LeftLowerLeg": "left_knee",
    "LeftFootAnkle": "left_ankle",
    "LeftFootBall": "left_foot_ball",
    "RightUpperLeg": "right_hip",
    "RightLowerLeg": "right_knee",
    "RightFootAnkle": "right_ankle",
    "RightFootBall": "right_foot_ball",
}

PICO_TO_CANONICAL: dict[str, str] = {
    "Pelvis": "pelvis",
    "SPINE1": "spine_lower",
    "SPINE2": "spine_middle",
    "SPINE3": "chest",
    "NECK": "neck",
    "HEAD": "head",
    "LEFT_SHOULDER": "left_shoulder",
    "LEFT_ELBOW": "left_elbow",
    "LEFT_WRIST": "left_wrist",
    "LEFT_HAND": "left_hand",
    "RIGHT_SHOULDER": "right_shoulder",
    "RIGHT_ELBOW": "right_elbow",
    "RIGHT_WRIST": "right_wrist",
    "RIGHT_HAND": "right_hand",
    "LEFT_HIP": "left_hip",
    "LEFT_KNEE": "left_knee",
    "LEFT_ANKLE": "left_ankle",
    "LEFT_FOOT": "left_foot_ball",
    "RIGHT_HIP": "right_hip",
    "RIGHT_KNEE": "right_knee",
    "RIGHT_ANKLE": "right_ankle",
    "RIGHT_FOOT": "right_foot_ball",
}

_PROVENANCE = {
    SourceProvenance.PLATFORM_ESTIMATED: CanonicalProvenance.PLATFORM_ESTIMATED,
    SourceProvenance.DEVICE_REPORTED: CanonicalProvenance.DEVICE_REPORTED,
    SourceProvenance.EXTERNAL_TRACKER: CanonicalProvenance.EXTERNAL_TRACKER,
    SourceProvenance.SYNTHETIC_TEST: CanonicalProvenance.SYNTHETIC_TEST,
    SourceProvenance.UNKNOWN: CanonicalProvenance.UNKNOWN,
}
_TRACKING_STATE = {
    JointTrackingState.INVALID: CanonicalTrackingState.INVALID,
    JointTrackingState.VALID: CanonicalTrackingState.VALID,
    JointTrackingState.TRACKED: CanonicalTrackingState.TRACKED,
}
_CLOCK_QUALITY = {
    TimestampQuality.UNAVAILABLE: CanonicalClockQuality.UNAVAILABLE,
    TimestampQuality.RECEIVE_ONLY: CanonicalClockQuality.RECEIVE_ONLY,
    TimestampQuality.MAPPED_UNBOUNDED: CanonicalClockQuality.MAPPED_UNBOUNDED,
    TimestampQuality.SYNCHRONIZED: CanonicalClockQuality.SYNCHRONIZED,
    TimestampQuality.DIAGNOSTIC_ONLY: CanonicalClockQuality.DIAGNOSTIC_ONLY,
}


def _meta_source_name(name: str) -> str:
    for prefix in ("FullBody_", "Body_"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _pico_source_name(joint: JointSample) -> str:
    if joint.name.startswith("PicoBody_") and 0 <= joint.index < len(
        PICO_BODY_24_SOURCE_NAMES
    ):
        return PICO_BODY_24_SOURCE_NAMES[joint.index]
    return joint.name


def _source_name_and_target(
    packet: TrackingPacket, joint: JointSample
) -> tuple[str, str | None]:
    if packet.source == "pico":
        source_name = _pico_source_name(joint)
        return source_name, PICO_TO_CANONICAL.get(source_name)
    source_name = _meta_source_name(joint.name)
    return source_name, META_TO_CANONICAL.get(source_name)


def _converted_components(
    packet: TrackingPacket,
    joint: JointSample,
    calibration: HandumiWorldCalibration,
) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    pose = np.asarray(joint.pose, dtype=np.float64)
    position_valid = bool(joint.location_flags & 0x2) and np.all(np.isfinite(pose[:3]))
    orientation_valid = bool(joint.location_flags & 0x1) and np.all(
        np.isfinite(pose[3:7])
    )
    position = np.full(3, np.nan, dtype=np.float64)
    quaternion = np.full(4, np.nan, dtype=np.float64)
    if position_valid:
        source_position = (
            unity_position_to_handumi(pose[:3])
            if packet.source == "meta_quest"
            else pose[:3]
        )
        position = calibration.apply_position(source_position)
    if orientation_valid:
        source_quaternion = (
            unity_quaternion_to_handumi(pose[3:7])
            if packet.source == "meta_quest"
            else pose[3:7]
        )
        norm = float(np.linalg.norm(source_quaternion))
        if norm <= 1e-12:
            orientation_valid = False
        else:
            quaternion = calibration.apply_orientation(source_quaternion / norm)
    return position, quaternion, bool(position_valid), bool(orientation_valid)


def _write_joint(
    frame: CanonicalBodyFrame,
    packet: TrackingPacket,
    source_joint: JointSample,
    target_index: int,
    calibration: HandumiWorldCalibration,
) -> None:
    position, quaternion, position_valid, orientation_valid = _converted_components(
        packet, source_joint, calibration
    )
    frame.joint_pose[target_index, :3] = position
    frame.joint_pose[target_index, 3:7] = quaternion
    frame.position_valid[target_index] = int(position_valid)
    frame.orientation_valid[target_index] = int(orientation_valid)
    frame.tracking_state[target_index] = int(
        _TRACKING_STATE[source_joint.tracking_state]
    )
    frame.confidence[target_index] = np.float32(source_joint.confidence)
    frame.provenance[target_index] = int(
        _PROVENANCE.get(source_joint.provenance, CanonicalProvenance.UNKNOWN)
    )


def canonical_body_from_packet(
    packet: TrackingPacket | None,
    *,
    calibration: HandumiWorldCalibration | None = None,
) -> CanonicalBodyFrame:
    """Map one source packet without fabricating missing canonical joints."""
    frame = CanonicalBodyFrame.empty()
    if packet is None:
        return frame
    calibration = calibration or HandumiWorldCalibration.identity()
    frame.ground_plane[:] = calibration.ground_plane
    timestamps = packet.timestamps
    frame.receive_time_ns[0] = timestamps.receive_time_ns
    frame.clock_offset_ns[0] = timestamps.clock_offset_ns
    frame.rtt_ns[0] = timestamps.rtt_ns
    frame.uncertainty_ns[0] = timestamps.uncertainty_ns
    frame.source_sequence[0] = -1 if packet.sequence is None else packet.sequence
    body = packet.body
    source_time_ns = (
        body.source_time_ns
        if body is not None and body.source_time_ns > 0
        else timestamps.source_time_ns
    )
    frame.source_time_ns[0] = source_time_ns
    frame.mapped_time_ns[0] = (
        source_time_ns + timestamps.clock_offset_ns
        if source_time_ns > 0 and timestamps.rtt_ns > 0
        else 0
    )
    quality = body.timestamp_quality if body is not None else timestamps.quality
    frame.clock_quality[0] = int(_CLOCK_QUALITY[quality])
    if body is None or not body.active:
        return frame
    if packet.source == "meta_quest" and body.calibration_state.lower() != "valid":
        # Retain the native packet and state in the raw sidecar, but do not
        # turn an explicitly invalid Meta skeleton into interpreted canonical,
        # CoM, contact, or support outputs.
        return frame

    for joint in body.joints:
        source_name, target = _source_name_and_target(packet, joint)
        if source_name == "Root" and packet.source == "meta_quest":
            position, quaternion, position_valid, orientation_valid = (
                _converted_components(packet, joint, calibration)
            )
            frame.platform_root_pose[:3] = position
            frame.platform_root_pose[3:7] = quaternion
            frame.platform_root_position_valid[0] = int(position_valid)
            frame.platform_root_orientation_valid[0] = int(orientation_valid)
            continue
        if target is None:
            continue
        _write_joint(frame, packet, joint, _CANONICAL_INDEX[target], calibration)

    return frame


__all__ = [
    "META_TO_CANONICAL",
    "PICO_TO_CANONICAL",
    "canonical_body_from_packet",
]
