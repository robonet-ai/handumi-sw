from __future__ import annotations

import numpy as np
import pytest

from handumi.body.com import (
    AnthropometricTable,
    BodyProfile,
    ComEstimatorConfig,
    KinematicComEstimator,
    default_anthropometric_table,
)
from handumi.body.model import (
    CANONICAL_JOINTS,
    CanonicalBodyFrame,
    CanonicalProvenance,
    CanonicalTrackingState,
    ComDiagnostic,
    ComProvenance,
)

_INDEX = {joint.identifier: joint.index for joint in CANONICAL_JOINTS}


def _profile() -> BodyProfile:
    return BodyProfile(
        height_m=1.75,
        mass_kg=70.0,
        foot_length_m=0.25,
        foot_width_m=0.10,
        measurement_uncertainty_m=0.005,
    )


def _analytic_frame(*, time_ns: int = 1_000_000_000) -> CanonicalBodyFrame:
    frame = CanonicalBodyFrame.empty()
    positions = {
        "pelvis": (0.00, 0.00, 0.95),
        "spine_lower": (0.00, 0.00, 1.05),
        "spine_middle": (0.00, 0.00, 1.20),
        "spine_upper": (0.00, 0.00, 1.35),
        "chest": (0.00, 0.00, 1.42),
        "neck": (0.00, 0.00, 1.55),
        "head": (0.00, 0.00, 1.72),
        "left_shoulder": (0.00, 0.22, 1.45),
        "left_elbow": (0.00, 0.50, 1.25),
        "left_wrist": (0.00, 0.70, 1.05),
        "left_hand": (0.00, 0.80, 1.00),
        "right_shoulder": (0.00, -0.22, 1.45),
        "right_elbow": (0.00, -0.50, 1.25),
        "right_wrist": (0.00, -0.70, 1.05),
        "right_hand": (0.00, -0.80, 1.00),
        "left_hip": (0.00, 0.10, 0.95),
        "left_knee": (0.00, 0.10, 0.52),
        "left_ankle": (0.00, 0.10, 0.08),
        "left_foot_ball": (0.17, 0.10, 0.00),
        "right_hip": (0.00, -0.10, 0.95),
        "right_knee": (0.00, -0.10, 0.52),
        "right_ankle": (0.00, -0.10, 0.08),
        "right_foot_ball": (0.17, -0.10, 0.00),
    }
    for name, position in positions.items():
        index = _INDEX[name]
        frame.joint_pose[index, :3] = position
        frame.joint_pose[index, 3:7] = (0.0, 0.0, 0.0, 1.0)
        frame.position_valid[index] = 1
        frame.orientation_valid[index] = 1
        frame.tracking_state[index] = int(CanonicalTrackingState.TRACKED)
        frame.confidence[index] = 1.0
        frame.provenance[index] = int(CanonicalProvenance.PLATFORM_ESTIMATED)
    frame.ground_plane[:] = (0.0, 0.0, 1.0, 0.0)
    frame.mapped_time_ns[0] = time_ns
    return frame


def _reference_com(frame: CanonicalBodyFrame, table: AnthropometricTable) -> np.ndarray:
    total = np.zeros(3)
    for segment in table.segments:
        proximal = np.mean(
            [frame.joint_pose[_INDEX[name], :3] for name in segment.proximal_landmarks],
            axis=0,
        )
        distal = np.mean(
            [frame.joint_pose[_INDEX[name], :3] for name in segment.distal_landmarks],
            axis=0,
        )
        segment_com = proximal + segment.com_fraction_from_proximal * (
            distal - proximal
        )
        total += segment.mass_fraction * segment_com
    return total


def test_default_table_has_15_segments_and_conserves_total_mass():
    table = default_anthropometric_table()
    assert len(table.segments) == 15
    assert sum(segment.mass_fraction for segment in table.segments) == pytest.approx(
        1.0
    )
    assert {segment.identifier for segment in table.segments} >= {
        "head_neck",
        "trunk",
        "pelvis",
        "left_foot",
        "right_foot",
    }
    estimator_metadata = KinematicComEstimator(_profile()).metadata()
    assert estimator_metadata["resolved_total_mass_kg"] == pytest.approx(70.0)
    assert sum(estimator_metadata["segment_mass_kg"].values()) == pytest.approx(70.0)


def test_profile_and_custom_table_round_trip_through_yaml(tmp_path):
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("height_m: 1.82\nmass_kg: 81\nfoot_length_m: 0.27\n")
    profile = BodyProfile.from_yaml(profile_path)
    assert profile.height_m == pytest.approx(1.82)
    assert len(profile.metadata()["sha256"]) == 64

    table = default_anthropometric_table()
    table_path = tmp_path / "table.yaml"
    import yaml

    table_path.write_text(yaml.safe_dump(table.metadata()))
    restored = AnthropometricTable.from_yaml(table_path)
    assert restored.version == table.version
    assert restored.segments == table.segments


def test_analytic_reference_mass_weighted_com_and_ground_projection():
    frame = _analytic_frame()
    table = default_anthropometric_table()
    result = KinematicComEstimator(_profile(), table=table).estimate(frame)

    expected = _reference_com(frame, table)
    assert result.whole_com_valid[0] == 1
    assert result.whole_com_unresolved_mass_fraction[0] == pytest.approx(0.0)
    np.testing.assert_allclose(result.whole_com, expected, atol=1e-7)
    np.testing.assert_allclose(
        result.whole_com_ground_projection,
        [expected[0], expected[1], 0.0],
        atol=1e-7,
    )
    assert result.whole_com_provenance[0] == int(ComProvenance.KINEMATIC_INFERRED)
    assert np.isfinite(result.whole_com_covariance).all()
    assert np.isfinite(result.whole_com_ground_projection_covariance).all()
    segment_mask = result.segment_com_valid.astype(bool)
    assert np.all(
        result.segment_com_provenance[segment_mask]
        == int(ComProvenance.KINEMATIC_INFERRED)
    )
    assert result.center_of_pressure_valid[0] == 0
    assert np.isnan(result.center_of_pressure).all()
    assert sum(result.segment_mass_fraction) == pytest.approx(1.0)


def test_symmetric_pose_has_centered_lateral_com():
    result = KinematicComEstimator(_profile()).estimate(_analytic_frame())
    assert result.whole_com[1] == pytest.approx(0.0, abs=1e-7)


def test_rigid_transform_invariance():
    frame = _analytic_frame()
    baseline = KinematicComEstimator(_profile()).estimate(frame)
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([1.0, -2.0, 0.0])
    transformed = _analytic_frame()
    valid = transformed.position_valid.astype(bool)
    transformed.joint_pose[valid, :3] = (
        transformed.joint_pose[valid, :3] @ rotation.T + translation
    )
    result = KinematicComEstimator(_profile()).estimate(transformed)
    np.testing.assert_allclose(
        result.whole_com,
        rotation @ baseline.whole_com + translation,
        atol=1e-7,
    )


def test_support_polygon_is_invariant_to_non_horizontal_ground_plane():
    # Rotate the complete observation so the calibrated floor normal is world X.
    # A hull implementation tied to world XY collapses the foot length dimension.
    rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    estimator = KinematicComEstimator(_profile())
    for time_ns in (1_000_000_000, 1_033_333_333):
        frame = _analytic_frame(time_ns=time_ns)
        valid = frame.position_valid.astype(bool)
        frame.joint_pose[valid, :3] = frame.joint_pose[valid, :3] @ rotation.T
        frame.ground_plane[:] = (*rotation[:, 2], 0.0)
        result = estimator.estimate(frame)

    polygon = result.support_polygon[result.support_polygon_valid.astype(bool)]
    assert len(polygon) >= 4
    np.testing.assert_allclose(polygon @ rotation[:, 2], 0.0, atol=1e-7)


def test_missing_high_mass_landmark_invalidates_instead_of_renormalizing():
    frame = _analytic_frame()
    frame.position_valid[_INDEX["left_shoulder"]] = 0
    frame.joint_pose[_INDEX["left_shoulder"], :3] = np.nan
    result = KinematicComEstimator(_profile()).estimate(frame)
    assert result.whole_com_valid[0] == 0
    assert np.isnan(result.whole_com).all()
    assert result.whole_com_unresolved_mass_fraction[0] >= 0.355
    assert result.whole_com_diagnostic[0] == int(ComDiagnostic.UNRESOLVED_MASS)


def test_predicted_uncertainty_gate_invalidates_result():
    config = ComEstimatorConfig(max_com_std_m=1e-6)
    result = KinematicComEstimator(_profile(), config=config).estimate(
        _analytic_frame()
    )
    assert result.whole_com_valid[0] == 0
    assert result.whole_com_diagnostic[0] == int(ComDiagnostic.EXCESSIVE_UNCERTAINTY)


def test_low_confidence_increases_covariance_and_reaches_whole_result():
    baseline_frame = _analytic_frame()
    baseline = KinematicComEstimator(_profile()).estimate(baseline_frame)
    low_frame = _analytic_frame()
    valid = low_frame.position_valid.astype(bool)
    low_frame.confidence[valid] = 0.2
    low_frame.tracking_state[valid] = int(CanonicalTrackingState.VALID)
    low = KinematicComEstimator(
        _profile(), config=ComEstimatorConfig(max_com_std_m=1.0)
    ).estimate(low_frame)
    assert low.whole_com_valid[0] == 1
    assert low.whole_com_confidence[0] == pytest.approx(0.2)
    assert np.trace(low.whole_com_covariance) > np.trace(baseline.whole_com_covariance)


def test_contact_support_polygon_airborne_and_unilateral_states():
    estimator = KinematicComEstimator(_profile())
    estimator.estimate(_analytic_frame(time_ns=1_000_000_000))
    standing = estimator.estimate(_analytic_frame(time_ns=1_033_333_333))
    assert standing.contact_valid.tolist() == [1, 1, 1, 1]
    assert np.all(standing.contact_probability > 0.65)
    assert standing.support_polygon_valid.sum() >= 4
    assert standing.provenance[_INDEX["left_heel"]] == int(CanonicalProvenance.INFERRED)

    estimator.reset()
    airborne_frame = _analytic_frame(time_ns=1_066_666_666)
    foot_names = (
        "left_ankle",
        "left_foot_ball",
        "right_ankle",
        "right_foot_ball",
    )
    for name in foot_names:
        airborne_frame.joint_pose[_INDEX[name], 2] += 0.30
    estimator.estimate(airborne_frame)
    airborne_frame.mapped_time_ns[0] = 1_099_999_999
    airborne = estimator.estimate(airborne_frame)
    assert np.all(airborne.contact_probability < 0.01)
    assert airborne.support_polygon_valid.sum() == 0

    estimator.reset()
    first = _analytic_frame(time_ns=2_000_000_000)
    for name in ("right_ankle", "right_foot_ball"):
        first.joint_pose[_INDEX[name], 2] += 0.30
    estimator.estimate(first)
    second = _analytic_frame(time_ns=2_033_333_333)
    for name in ("right_ankle", "right_foot_ball"):
        second.joint_pose[_INDEX[name], 2] += 0.30
    unilateral = estimator.estimate(second)
    assert np.all(unilateral.contact_probability[:2] > 0.65)
    assert np.all(unilateral.contact_probability[2:] < 0.01)
    assert unilateral.support_polygon_valid.sum() == 4


def test_missing_foot_dimensions_do_not_fabricate_heel_or_support_geometry():
    profile = BodyProfile(height_m=1.75, mass_kg=70.0)
    estimator = KinematicComEstimator(profile)
    first = estimator.estimate(_analytic_frame(time_ns=1_000_000_000))
    second = estimator.estimate(_analytic_frame(time_ns=1_033_000_000))

    for frame in (first, second):
        assert not frame.position_valid[_INDEX["left_heel"]]
        assert not frame.position_valid[_INDEX["right_heel"]]
        assert frame.support_polygon_valid.sum() == 0


def test_external_contact_input_is_labeled_fused_estimated():
    estimator = KinematicComEstimator(_profile())
    result = estimator.estimate(
        _analytic_frame(time_ns=1_000_000_000),
        external_contact_probability={"left_heel": 1.0},
    )
    assert result.contact_probability[0] == pytest.approx(1.0)
    assert result.contact_provenance[0] == int(ComProvenance.FUSED_ESTIMATED)


def test_contact_rejects_large_ground_penetration():
    estimator = KinematicComEstimator(_profile())
    first = _analytic_frame(time_ns=3_000_000_000)
    for name in ("left_ankle", "left_foot_ball"):
        first.joint_pose[_INDEX[name], 2] -= 0.30
    estimator.estimate(first)
    first.mapped_time_ns[0] = 3_033_333_333
    result = estimator.estimate(first)
    assert np.all(result.contact_probability[:2] < 0.01)
    assert np.all(result.contact_probability[2:] > 0.65)
    assert result.support_polygon_valid.sum() == 4
