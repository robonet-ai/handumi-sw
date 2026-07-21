from __future__ import annotations

import numpy as np

from handumi.body.model import (
    CANONICAL_JOINTS,
    CanonicalBodyFrame,
    CanonicalProvenance,
    CanonicalTrackingState,
    ComDiagnostic,
    ComProvenance,
)
from handumi.visualization.body import (
    BODY_JOINTS_PATH,
    BODY_SKELETON_PATH,
    CENTER_OF_PRESSURE_PATH,
    COM_PROJECTION_PATH,
    CONTACTS_PATH,
    FUSED_COLOR,
    GROUND_PATH,
    KINEMATIC_COLOR,
    LEARNED_COLOR,
    PLATFORM_COLOR,
    SEGMENT_COM_PATH,
    SUPPORT_POLYGON_PATH,
    UNKNOWN_COLOR,
    WHOLE_COM_PATH,
    WHOLE_COM_TRAIL_PATH,
    BodyTrailBuffer,
    body_render_plan,
    canonical_skeleton_edges,
    chunk_trajectory,
    confidence_alpha,
    decimate_trajectory,
    full_trajectory_plan,
    ground_plane_geometry,
    provenance_style,
    support_polygon_strip,
)


def synthetic_body_frame() -> CanonicalBodyFrame:
    frame = CanonicalBodyFrame.empty()
    for joint in CANONICAL_JOINTS:
        side = -1.0 if joint.identifier.startswith("left_") else 1.0
        frame.joint_pose[joint.index, :3] = [
            0.025 * joint.index,
            0.08 * side,
            0.7 + 0.015 * joint.index,
        ]
        frame.joint_pose[joint.index, 3:] = [0.0, 0.0, 0.0, 1.0]
        frame.position_valid[joint.index] = 1
        frame.orientation_valid[joint.index] = 1
        frame.tracking_state[joint.index] = int(CanonicalTrackingState.TRACKED)
        frame.confidence[joint.index] = 0.9
        frame.provenance[joint.index] = int(CanonicalProvenance.PLATFORM_ESTIMATED)
        frame.segment_com[joint.index] = frame.joint_pose[joint.index, :3] + [
            0,
            0,
            0.01,
        ]
        frame.segment_com_valid[joint.index] = 1
        frame.segment_com_confidence[joint.index] = 0.8
        frame.segment_com_provenance[joint.index] = int(
            ComProvenance.KINEMATIC_INFERRED
        )
    frame.whole_com[:] = [0.1, 0.0, 0.95]
    frame.whole_com_valid[0] = 1
    frame.whole_com_confidence[0] = 0.85
    frame.whole_com_provenance[0] = int(ComProvenance.KINEMATIC_INFERRED)
    frame.whole_com_diagnostic[0] = int(ComDiagnostic.VALID)
    frame.whole_com_ground_projection[:] = [0.1, 0.0, 0.0]
    frame.whole_com_ground_projection_valid[0] = 1
    frame.ground_plane[:] = [0.0, 0.0, 1.0, 0.0]
    for contact_index, joint_name in enumerate(
        ("left_heel", "left_foot_ball", "right_heel", "right_foot_ball")
    ):
        joint_index = next(
            j.index for j in CANONICAL_JOINTS if j.identifier == joint_name
        )
        frame.joint_pose[joint_index, 2] = 0.0
        frame.contact_probability[contact_index] = 0.9
        frame.contact_valid[contact_index] = 1
        frame.contact_provenance[contact_index] = int(ComProvenance.KINEMATIC_INFERRED)
    frame.support_polygon[:4] = [
        [-0.1, -0.1, 0.0],
        [0.1, -0.1, 0.0],
        [0.1, 0.1, 0.0],
        [-0.1, 0.1, 0.0],
    ]
    frame.support_polygon_valid[:4] = 1
    return frame


def _op(plan, path):
    return next(operation for operation in plan if operation.path == path)


def test_stable_body_paths_are_centralized():
    assert BODY_JOINTS_PATH == "tracking/body/joints"
    assert BODY_SKELETON_PATH == "tracking/body/skeleton"
    assert SEGMENT_COM_PATH == "tracking/body/segment_com"
    assert WHOLE_COM_PATH == "tracking/body/whole_com"
    assert WHOLE_COM_TRAIL_PATH == "tracking/body/whole_com/trail"
    assert COM_PROJECTION_PATH == "tracking/body/com_projection"
    assert GROUND_PATH == "tracking/body/ground"
    assert CONTACTS_PATH == "tracking/body/contacts"
    assert SUPPORT_POLYGON_PATH == "tracking/body/support_polygon"


def test_canonical_parent_child_edges_use_model_table():
    frame = synthetic_body_frame()
    edges = canonical_skeleton_edges(frame)
    assert len(edges) == len(CANONICAL_JOINTS) - 1
    assert any(
        np.allclose(edge, [frame.joint_pose[0, :3], frame.joint_pose[1, :3]])
        for edge in edges
    )


def test_missing_parent_or_child_suppresses_only_affected_edges():
    frame = synthetic_body_frame()
    frame.position_valid[4] = 0  # chest: one parent edge and three child edges
    edges = canonical_skeleton_edges(frame)
    assert len(edges) == len(CANONICAL_JOINTS) - 1 - 4
    assert any(
        np.allclose(edge, [frame.joint_pose[15, :3], frame.joint_pose[16, :3]])
        for edge in edges
    )


def test_invalid_current_data_explicitly_clears_previous_entities():
    valid = body_render_plan(synthetic_body_frame())
    assert _op(valid, BODY_JOINTS_PATH).archetype == "points3d"
    assert _op(valid, WHOLE_COM_PATH).archetype == "points3d"
    invalid = body_render_plan(CanonicalBodyFrame.empty())
    for path in (
        BODY_JOINTS_PATH,
        BODY_SKELETON_PATH,
        SEGMENT_COM_PATH,
        WHOLE_COM_PATH,
        COM_PROJECTION_PATH,
        CENTER_OF_PRESSURE_PATH,
    ):
        assert _op(invalid, path).archetype == "clear"


def test_body_geometry_omits_labels_to_keep_the_skeleton_visible():
    plan = body_render_plan(synthetic_body_frame())
    for path in (
        BODY_JOINTS_PATH,
        SEGMENT_COM_PATH,
        WHOLE_COM_PATH,
        COM_PROJECTION_PATH,
        CONTACTS_PATH,
    ):
        assert "labels" not in _op(plan, path).kwargs


def test_nan_values_never_reach_render_archetypes():
    frame = synthetic_body_frame()
    frame.joint_pose[3, :3] = np.nan
    frame.segment_com[5] = np.nan
    frame.whole_com[:] = np.nan
    for operation in body_render_plan(frame):
        if operation.archetype in {"points3d", "line_strips3d", "mesh3d", "scalars"}:
            arrays = (
                operation.data if isinstance(operation.data, list) else [operation.data]
            )
            for array in arrays:
                assert np.all(np.isfinite(np.asarray(array, dtype=np.float64)))


def test_provenance_to_style_mapping_and_future_fallbacks():
    assert (
        provenance_style(CanonicalProvenance.PLATFORM_ESTIMATED, 1.0).color[:3]
        == PLATFORM_COLOR
    )
    assert (
        provenance_style(CanonicalProvenance.INFERRED, 1.0).color[:3] == KINEMATIC_COLOR
    )
    assert (
        provenance_style(ComProvenance.FUSED_ESTIMATED, 1.0, com=True).color[:3]
        == FUSED_COLOR
    )
    assert provenance_style(999, 1.0).color[:3] == UNKNOWN_COLOR

    class FutureLearned:
        name = "LEARNED_ESTIMATE"

        def __int__(self):
            return 999

    assert provenance_style(FutureLearned(), 1.0).color[:3] == LEARNED_COLOR


def test_confidence_maps_to_alpha_and_quality_flag():
    assert confidence_alpha(1.0) == (255, False)
    low_alpha, low = confidence_alpha(0.2)
    assert 64 < low_alpha < 255
    assert low is True
    assert confidence_alpha(np.nan) == (64, True)


def test_whole_and_segment_com_obey_individual_masks():
    frame = synthetic_body_frame()
    frame.segment_com_valid[2] = 0
    frame.whole_com_valid[0] = 0
    plan = body_render_plan(frame)
    assert len(_op(plan, SEGMENT_COM_PATH).data) == len(CANONICAL_JOINTS) - 1
    assert _op(plan, WHOLE_COM_PATH).archetype == "clear"


def test_tilted_ground_geometry_lies_on_calibrated_plane():
    plane = np.array([0.0, 1.0, 1.0, -0.5], dtype=np.float32)
    geometry = ground_plane_geometry(plane, half_extent_m=2.0)
    assert geometry is not None
    vertices, triangles = geometry
    assert triangles.shape == (2, 3)
    np.testing.assert_allclose(vertices @ plane[:3] + plane[3], 0.0, atol=1e-6)


def test_contact_and_support_masks_and_polygon_order():
    frame = synthetic_body_frame()
    frame.contact_valid[1] = 0
    frame.support_polygon_valid[2] = 0
    plan = body_render_plan(frame)
    assert len(_op(plan, CONTACTS_PATH).data) == 3
    polygon = support_polygon_strip(frame)
    assert polygon is not None
    assert polygon.shape == (4, 3)
    np.testing.assert_allclose(polygon[0], polygon[-1])
    np.testing.assert_allclose(polygon[1], frame.support_polygon[1])


def test_com_trail_preserves_invalid_gaps_and_is_bounded():
    trail = BodyTrailBuffer(4)
    for value in ([0, 0, 0], [1, 0, 0], None, [3, 0, 0], [4, 0, 0]):
        trail.append(value)
    assert trail.sample_count == 4
    assert trail.max_points == 4
    strips = trail.strips()
    assert len(strips) == 1
    np.testing.assert_allclose(strips[0][:, 0], [3, 4])


def test_decimation_is_deterministic_and_preserves_invalid_gaps():
    points = np.stack([np.arange(12), np.zeros(12), np.zeros(12)], axis=1)
    valid = np.ones(12, dtype=bool)
    valid[5:7] = False
    first = decimate_trajectory(points, valid, temporal_step=2, spatial_step_m=1.5)
    second = decimate_trajectory(points, valid, temporal_step=2, spatial_step_m=1.5)
    assert len(first) == 2
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a.indices, b.indices)
    assert first[0].indices[-1] == 4
    assert second[1].indices[0] == 7


def test_long_trajectory_is_planned_once_and_chunked_linearly():
    count = 20_000
    points = np.stack(
        [np.arange(count, dtype=np.float32), np.zeros(count), np.zeros(count)], axis=1
    )
    chunks = chunk_trajectory(
        decimate_trajectory(points, temporal_step=3), point_cap=128, duration_frames=200
    )
    assert chunks
    assert all(2 <= len(chunk.points) <= 128 for chunk in chunks)
    assert sum(len(chunk.points) for chunk in chunks) <= count + len(chunks)
    plan = full_trajectory_plan(
        "tracking/test/trail",
        points,
        None,
        color=(255, 255, 255),
        radius=0.01,
        temporal_step=3,
        spatial_step_m=0.0,
        point_cap=128,
        duration_frames=200,
    )
    assert len(plan) == 1
    assert plan[0].static is True
