from __future__ import annotations

import numpy as np
import pytest

from handumi.body.calibration import (
    NeutralCalibrationCapture,
    ProfileConstrainedSkeleton,
    ProfileNeutralCalibration,
    estimate_profile_neutral_calibration,
    persist_neutral_calibration_capture,
    validate_neutral_capture,
)
from handumi.body.com import BodyProfile, KinematicComEstimator
from handumi.body.model import (
    CANONICAL_JOINTS,
    CanonicalBodyFrame,
    CanonicalProvenance,
    CanonicalTrackingState,
)
from handumi.tracking.meta_quest import parse_tracking_packet
from handumi.tracking.mock_quest_sender import make_tracking_packet_fixture
from handumi.tracking.transforms import HandumiWorldCalibration

_INDEX = {joint.identifier: joint.index for joint in CANONICAL_JOINTS}


def _profile() -> BodyProfile:
    return BodyProfile(
        height_m=1.70,
        mass_kg=72.0,
        arm_span_m=1.70,
        leg_length_m=0.77,
        hand_length_m=0.18,
        foot_length_m=0.27,
        foot_width_m=0.105,
        shoulder_breadth_m=0.44,
        hip_breadth_m=0.32,
        measurement_uncertainty_m=0.005,
        mass_uncertainty_kg=1.0,
    )


def _source_frame(*, shift_x: float = 0.0) -> CanonicalBodyFrame:
    frame = CanonicalBodyFrame.empty()
    positions = {
        "pelvis": (0.00, 0.00, 1.11),
        "spine_lower": (0.00, 0.00, 1.20),
        "spine_middle": (0.00, 0.00, 1.32),
        "spine_upper": (0.00, 0.00, 1.44),
        "chest": (0.00, 0.00, 1.55),
        "neck": (0.00, 0.00, 1.86),
        "head": (0.00, 0.00, 2.04),
        "left_shoulder": (0.00, 0.20, 1.58),
        "left_elbow": (0.00, 0.42, 1.58),
        "left_wrist": (0.00, 0.62, 1.58),
        "left_hand": (0.00, 0.82, 1.58),
        "right_shoulder": (0.00, -0.20, 1.58),
        "right_elbow": (0.00, -0.42, 1.58),
        "right_wrist": (0.00, -0.62, 1.58),
        "right_hand": (0.00, -0.82, 1.58),
        "left_hip": (0.00, 0.09, 1.11),
        "left_knee": (0.00, 0.09, 0.80),
        "left_ankle": (0.00, 0.09, 0.42),
        "left_foot_ball": (0.17, 0.09, 0.34),
        "right_hip": (0.00, -0.09, 1.11),
        "right_knee": (0.00, -0.09, 0.80),
        "right_ankle": (0.00, -0.09, 0.42),
        "right_foot_ball": (0.17, -0.09, 0.34),
    }
    for name, position in positions.items():
        index = _INDEX[name]
        frame.joint_pose[index, :3] = np.asarray(position) + (shift_x, 0.0, 0.0)
        frame.joint_pose[index, 3:7] = (0.0, 0.0, 0.0, 1.0)
        frame.position_valid[index] = 1
        frame.orientation_valid[index] = 1
        frame.tracking_state[index] = int(CanonicalTrackingState.TRACKED)
        frame.confidence[index] = 1.0
        frame.provenance[index] = int(CanonicalProvenance.PLATFORM_ESTIMATED)
    return frame


def _transform_frame(frame: CanonicalBodyFrame, calibration) -> CanonicalBodyFrame:
    output = CanonicalBodyFrame.empty()
    output.ground_plane[:] = calibration.ground_plane
    for joint in CANONICAL_JOINTS:
        if not frame.position_valid[joint.index]:
            continue
        output.joint_pose[joint.index, :3] = calibration.apply_position(
            frame.joint_pose[joint.index, :3]
        )
        output.joint_pose[joint.index, 3:7] = (0.0, 0.0, 0.0, 1.0)
        output.position_valid[joint.index] = 1
        output.orientation_valid[joint.index] = 1
        output.tracking_state[joint.index] = frame.tracking_state[joint.index]
        output.confidence[joint.index] = frame.confidence[joint.index]
        output.provenance[joint.index] = frame.provenance[joint.index]
    return output


def _distance(frame: CanonicalBodyFrame, first: str, second: str) -> float:
    return float(
        np.linalg.norm(
            frame.joint_pose[_INDEX[first], :3] - frame.joint_pose[_INDEX[second], :3]
        )
    )


def _chain(frame: CanonicalBodyFrame, names: tuple[str, ...]) -> float:
    return sum(
        _distance(frame, a, b) for a, b in zip(names[:-1], names[1:], strict=True)
    )


def test_neutral_calibration_corrects_stage_floor_without_qualifying_accuracy():
    frames = [_source_frame(shift_x=0.001 * np.sin(index)) for index in range(20)]
    hmd = [np.array([0.4, -0.2, 2.02, 0.0, 0.0, 0.0, 1.0])] * len(frames)

    result = estimate_profile_neutral_calibration(
        frames,
        hmd,
        _profile(),
        source_frame="meta_right_handed_source",
    )

    assert result.source_ground_height_m == pytest.approx(0.34, abs=0.01)
    assert result.observed_stature_m == pytest.approx(1.70, abs=0.01)
    assert result.world.qualified is False
    np.testing.assert_allclose(result.world.ground_plane, [0, 0, 1, 0])
    floor_point = result.world.apply_position(
        [0.4, -0.2, result.source_ground_height_m]
    )
    assert floor_point[2] == pytest.approx(0.0, abs=1e-8)
    assert result.metadata()["limitation"].endswith("validation")


def test_profile_constraints_use_dimensions_and_mark_positions_inferred():
    profile = _profile()
    source_frames = [_source_frame() for _ in range(20)]
    hmd = [np.array([0.0, 0.0, 2.02, 0.0, 0.0, 0.0, 1.0])] * len(source_frames)
    neutral = estimate_profile_neutral_calibration(
        source_frames,
        hmd,
        profile,
        source_frame="meta_right_handed_source",
    )
    world_frames = [_transform_frame(frame, neutral.world) for frame in source_frames]
    fitter = ProfileConstrainedSkeleton(profile)
    fitter.calibrate(world_frames)

    fitted = fitter.apply(world_frames[0])

    assert _distance(fitted, "left_shoulder", "right_shoulder") == pytest.approx(0.44)
    assert _distance(fitted, "left_hip", "right_hip") == pytest.approx(0.32)
    assert fitted.joint_pose[_INDEX["left_hip"], 2] == pytest.approx(0.77)
    assert fitted.joint_pose[_INDEX["head"], 2] == pytest.approx(1.70)
    assert fitted.joint_pose[_INDEX["left_foot_ball"], 2] == pytest.approx(
        0.0, abs=1e-7
    )
    assert _chain(fitted, ("left_hip", "left_knee", "left_ankle")) == pytest.approx(
        0.69
    )
    left_reach = _chain(
        fitted, ("left_shoulder", "left_elbow", "left_wrist", "left_hand")
    )
    right_reach = _chain(
        fitted, ("right_shoulder", "right_elbow", "right_wrist", "right_hand")
    )
    assert 0.44 + left_reach + right_reach == pytest.approx(1.70)
    assert _distance(fitted, "left_wrist", "left_hand") == pytest.approx(0.18)
    assert fitted.provenance[_INDEX["left_hand"]] == int(CanonicalProvenance.INFERRED)
    assert fitted.tracking_state[_INDEX["left_hand"]] == int(
        CanonicalTrackingState.VALID
    )

    estimated = KinematicComEstimator(profile).estimate(fitted)
    assert _distance(estimated, "left_heel", "left_foot_ball") == pytest.approx(0.27)
    metadata = fitter.metadata()
    assert metadata["constraints"]["foot_width_m"] == pytest.approx(0.105)
    assert KinematicComEstimator(profile).metadata()[
        "resolved_total_mass_kg"
    ] == pytest.approx(72.0)


def test_neutral_calibration_rejects_profile_inconsistent_pose():
    frames = [_source_frame() for _ in range(20)]
    for frame in frames:
        frame.joint_pose[_INDEX["head"], 2] = 1.4
    hmd = [np.array([0.0, 0.0, 1.4, 0.0, 0.0, 0.0, 1.0])] * len(frames)
    with pytest.raises(ValueError, match="inconsistent with body profile"):
        estimate_profile_neutral_calibration(
            frames,
            hmd,
            _profile(),
            source_frame="meta_right_handed_source",
        )


def _neutral_packet(index: int, *, calibration_state: str = "Valid"):
    raw = make_tracking_packet_fixture(84, seq=index)
    raw["body"]["sourceTimeNs"] = 1_000_000_000 + index * 200_000_000
    raw["body"]["observationSequence"] = index
    raw["body"]["calibrationState"] = calibration_state
    return parse_tracking_packet(
        raw,
        pc_monotonic_ns=2_000_000_000 + index * 200_000_000,
        receive_sequence=100 + index,
    )


def test_neutral_capture_validation_requires_duration_and_valid_meta_state():
    packets = tuple(_neutral_packet(index) for index in range(16))
    hmd = tuple(np.array([0, 0, 1.7, 0, 0, 0, 1.0]) for _ in packets)
    capture = NeutralCalibrationCapture(packets, hmd, requested_duration_s=3.0)
    validate_neutral_capture(capture, min_samples=15)
    assert capture.observed_duration_s == pytest.approx(3.0)

    invalid_packets = list(packets)
    invalid_packets[5] = _neutral_packet(5, calibration_state="Invalid")
    with pytest.raises(ValueError, match="calibration state"):
        validate_neutral_capture(
            NeutralCalibrationCapture(tuple(invalid_packets), hmd, 3.0),
            min_samples=15,
        )
    with pytest.raises(ValueError, match="coverage is too short"):
        validate_neutral_capture(
            NeutralCalibrationCapture(packets[:15], hmd[:15], 4.0),
            min_samples=15,
        )


def test_neutral_capture_artifact_preserves_exact_inputs_and_runtime(tmp_path):
    packets = tuple(_neutral_packet(index) for index in range(16))
    hmd = tuple(np.array([0, 0, 1.7, 0, 0, 0, 1.0]) for _ in packets)
    capture = NeutralCalibrationCapture(packets, hmd, requested_duration_s=3.0)
    world = HandumiWorldCalibration.identity(
        source_frame="meta_right_handed_source", qualified=False
    )
    neutral = ProfileNeutralCalibration(
        world=world,
        source_ground_height_m=0.0,
        observed_stature_m=1.7,
        stature_error_m=0.0,
        sample_count=16,
        ground_sample_std_m=0.01,
        pelvis_motion_p95_m=0.01,
    )

    path, reference = persist_neutral_calibration_capture(
        tmp_path,
        capture,
        neutral,
        _profile(),
        applied_world=world,
        profile_skeleton={"schema": "test", "provenance": "INFERRED"},
        frame_epoch=0,
        frame_epoch_reason="initial_profile_neutral_calibration",
        neutral_world_applied=True,
    )

    artifact = __import__("json").loads(path.read_text())
    assert reference["path"].startswith("raw/tracking/calibration/neutral-")
    assert artifact["inputs"]["native_packets"][3]["packet"] == packets[3].raw
    assert artifact["capture"]["sample_count"] == 16
    assert artifact["runtime"]["python"]
    assert artifact["outputs"]["neutral_world_applied"] is True
    assert artifact["qualified"] is False
    assert not list(path.parent.glob("*.tmp"))
