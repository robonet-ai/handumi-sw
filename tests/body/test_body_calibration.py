from __future__ import annotations

import numpy as np
import pytest

from handumi.body.calibration import (
    ProfileConstrainedSkeleton,
    estimate_profile_neutral_calibration,
)
from handumi.body.com import BodyProfile, KinematicComEstimator
from handumi.body.model import (
    CANONICAL_JOINTS,
    CanonicalBodyFrame,
    CanonicalProvenance,
    CanonicalTrackingState,
)
from handumi.tracking.transforms import quat_rotate

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


def _transform_frame(
    frame: CanonicalBodyFrame, calibration
) -> CanonicalBodyFrame:
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
            frame.joint_pose[_INDEX[first], :3]
            - frame.joint_pose[_INDEX[second], :3]
        )
    )


def _chain(frame: CanonicalBodyFrame, names: tuple[str, ...]) -> float:
    return sum(_distance(frame, a, b) for a, b in zip(names[:-1], names[1:], strict=True))


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
    floor_point = result.world.apply_position([0.4, -0.2, result.source_ground_height_m])
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
    assert fitted.tracking_state[_INDEX["left_hand"]] == int(CanonicalTrackingState.VALID)

    estimated = KinematicComEstimator(profile).estimate(fitted)
    assert _distance(estimated, "left_heel", "left_foot_ball") == pytest.approx(0.27)
    metadata = fitter.metadata()
    assert metadata["constraints"]["foot_width_m"] == pytest.approx(0.105)
    assert KinematicComEstimator(profile).metadata()["resolved_total_mass_kg"] == pytest.approx(72.0)


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
