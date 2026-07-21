import numpy as np

from handumi.body.mapping import canonical_body_from_packet
from handumi.body.model import (
    CANONICAL_JOINT_COUNT,
    CANONICAL_JOINTS,
    CanonicalClockQuality,
    CanonicalTrackingState,
    canonical_body_features,
    canonical_body_metadata,
)
from handumi.tracking.meta_quest import parse_tracking_packet
from handumi.tracking.mock_quest_sender import make_tracking_packet_fixture
from handumi.tracking.transforms import (
    FrameTransformGraph,
    HandumiWorldCalibration,
    NamedFrameTransform,
    Pose,
)
from handumi.tracking.pico import tracking_packet_from_pico_frame


def _meta_84_fixture():
    raw = make_tracking_packet_fixture(84, seq=41)
    names = raw["body"]["jointNames"]
    replacements = {
        0: "FullBody_Root",
        1: "FullBody_Hips",
        2: "FullBody_SpineLower",
        3: "FullBody_SpineMiddle",
        4: "FullBody_SpineUpper",
        5: "FullBody_Chest",
        6: "FullBody_Neck",
        7: "FullBody_Head",
        10: "FullBody_LeftArmUpper",
        11: "FullBody_LeftArmLower",
        18: "FullBody_LeftHandPalm",
        19: "FullBody_LeftHandWrist",
        33: "FullBody_LeftHandMiddleTip",
        44: "FullBody_RightHandPalm",
        45: "FullBody_RightHandWrist",
        59: "FullBody_RightHandMiddleTip",
        70: "FullBody_LeftUpperLeg",
        71: "FullBody_LeftLowerLeg",
        73: "FullBody_LeftFootAnkle",
        77: "FullBody_RightUpperLeg",
        78: "FullBody_RightLowerLeg",
        80: "FullBody_RightFootAnkle",
        83: "FullBody_RightFootBall",
    }
    for index, name in replacements.items():
        names[index] = name
    return raw


def test_canonical_joint_ids_and_parents_are_stable():
    assert CANONICAL_JOINT_COUNT == 25
    assert [joint.index for joint in CANONICAL_JOINTS] == list(range(25))
    assert CANONICAL_JOINTS[0].identifier == "pelvis"
    assert CANONICAL_JOINTS[0].parent_index == -1
    assert CANONICAL_JOINTS[-1].identifier == "right_foot_ball"
    assert CANONICAL_JOINTS[-1].parent_index == 22


def test_84_joint_meta_maps_deterministically_without_filling_missing_heel():
    raw = _meta_84_fixture()
    packet = parse_tracking_packet(
        raw,
        pc_monotonic_ns=2_000,
        receive_sequence=9,
        clock_offset_ns=100,
        rtt_ns=20,
    )
    frame = canonical_body_from_packet(packet)
    by_name = {joint.identifier: joint.index for joint in CANONICAL_JOINTS}

    pelvis = by_name["pelvis"]
    # Unity (x, y, z) -> HandUMI (z, -x, y).
    np.testing.assert_allclose(frame.joint_pose[pelvis, :3], [0.0, -0.01, 1.0])
    assert frame.position_valid[pelvis] == 1
    assert frame.tracking_state[pelvis] == CanonicalTrackingState.VALID
    assert frame.source_sequence[0] == 41
    assert frame.clock_quality[0] == CanonicalClockQuality.DIAGNOSTIC_ONLY
    assert frame.source_time_ns[0] == raw["body"]["sourceTimeNs"]
    assert frame.mapped_time_ns[0] == raw["body"]["sourceTimeNs"] + 100
    assert np.isnan(frame.joint_pose[by_name["left_heel"]]).all()
    assert frame.position_valid[by_name["left_heel"]] == 0
    assert frame.platform_root_position_valid[0] == 1
    # The canonical distal hand endpoint is Meta's middle fingertip, not the
    # palm. Fixture joint 33 has Unity position (0.33, 1.0, 0.0).
    np.testing.assert_allclose(
        frame.joint_pose[by_name["left_hand"], :3], [0.0, -0.33, 1.0]
    )
    np.testing.assert_allclose(
        frame.joint_pose[by_name["right_hand"], :3], [0.0, -0.59, 1.0]
    )


def test_invalid_source_joint_stays_nan_even_when_source_pose_has_numbers():
    raw = _meta_84_fixture()
    raw["body"]["jointLocationFlags"][1] = 0
    packet = parse_tracking_packet(raw, pc_monotonic_ns=1, receive_sequence=1)
    frame = canonical_body_from_packet(packet)
    assert np.isnan(frame.joint_pose[0]).all()
    assert frame.position_valid[0] == 0
    assert frame.orientation_valid[0] == 0


def test_invalid_meta_body_calibration_is_retained_raw_but_not_interpreted():
    raw = _meta_84_fixture()
    raw["body"]["calibrationState"] = "Invalid"
    packet = parse_tracking_packet(raw, pc_monotonic_ns=123, receive_sequence=1)

    frame = canonical_body_from_packet(packet)

    assert frame.receive_time_ns[0] == 123
    assert not frame.position_valid.any()
    assert np.isnan(frame.joint_pose).all()


def test_pico_public_24_joint_order_maps_to_canonical_and_leaves_heel_missing():
    poses = np.zeros((24, 7), dtype=np.float32)
    poses[:, 6] = 1.0
    packet = tracking_packet_from_pico_frame(
        {
            "observation.pico.timestamp_ns": np.array([123], dtype=np.int64),
            "observation.pico.body_joints_pose": poses,
        },
        sequence=2,
        receive_time_ns=456,
    )
    frame = canonical_body_from_packet(packet)
    by_name = {joint.identifier: joint.index for joint in CANONICAL_JOINTS}
    assert frame.position_valid[by_name["left_hip"]] == 1
    assert frame.position_valid[by_name["left_foot_ball"]] == 1
    assert frame.position_valid[by_name["left_heel"]] == 0
    assert np.isnan(frame.joint_pose[by_name["spine_upper"]]).all()


def test_world_calibration_sets_ground_heading_without_legacy_hmd_inverse():
    calibration = HandumiWorldCalibration.from_ground_heading(
        ground_origin=[1.0, 2.0, 3.0],
        ground_normal=[0.0, 0.0, 1.0],
        initial_heading=[0.0, 1.0, 0.0],
        source_frame="meta_stage_right_handed",
    )
    np.testing.assert_allclose(
        calibration.apply_position([1.0, 3.0, 3.0]), [1.0, 0.0, 0.0], atol=1e-7
    )
    assert calibration.metadata()["target"] == "handumi_world"


def test_feature_and_metadata_contract_include_masks_and_joint_table():
    features = canonical_body_features()
    assert features["observation.body.joint_pose"]["shape"] == (25, 7)
    assert features["observation.body.position_valid"]["shape"] == (25,)
    metadata = canonical_body_metadata(transforms=[])
    assert metadata["tracking_schema"] == "handumi_tracking_v2"
    assert metadata["body_schema"] == "handumi_canonical_25_v1"
    assert len(metadata["canonical_joint_table"]) == 25
    assert len(metadata["calibration_hash"]) == 64


def test_transform_graph_keeps_world_workspace_camera_and_mocap_explicit():
    graph = FrameTransformGraph()
    graph.add(
        NamedFrameTransform(
            "handumi_world", "legacy_workspace", Pose([1, 0, 0], [0, 0, 0, 1])
        )
    )
    graph.add(
        NamedFrameTransform(
            "legacy_workspace", "left_wrist_camera", Pose([0, 2, 0], [0, 0, 0, 1])
        )
    )
    graph.add(
        NamedFrameTransform("handumi_world", "mocap", Pose([0, 0, 3], [0, 0, 0, 1]))
    )
    camera_from_world = graph.resolve("handumi_world", "left_wrist_camera")
    np.testing.assert_allclose(camera_from_world.position, [1, 2, 0])
    world_from_mocap = graph.resolve("mocap", "handumi_world")
    np.testing.assert_allclose(world_from_mocap.position, [0, 0, -3])
