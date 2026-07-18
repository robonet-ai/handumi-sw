from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from handumi.calibration.spatial import (
    CameraIntrinsics,
    CharucoDetection,
    CharucoBoardSpec,
    board_from_table_pose,
    calibration_hash,
    detect_charuco,
    estimate_board_pose,
    new_spatial_calibration,
    pose7_to_dict,
    session_calibration_metadata,
    session_table_from_device,
    solve_controller_camera,
    solve_table_camera,
    solve_table_device,
    solve_table_quest,
    write_yaml,
)
from handumi.robots.utils import mat_to_pose7, pose7_to_mat


def _transform(translation, rotvec) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    value[:3, 3] = translation
    return value


def _synthetic_views() -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    controller_from_camera = _transform([0.03, -0.02, 0.08], [0.2, -0.1, 0.05])
    quest_from_board = _transform([0.4, 0.2, 0.7], [0.1, 0.2, -0.3])
    controllers: list[np.ndarray] = []
    boards: list[np.ndarray] = []
    for _ in range(24):
        quest_from_controller = _transform(
            rng.uniform(-0.3, 0.3, 3), rng.uniform(-1.2, 1.2, 3)
        )
        camera_from_board = (
            np.linalg.inv(controller_from_camera)
            @ np.linalg.inv(quest_from_controller)
            @ quest_from_board
        )
        controllers.append(mat_to_pose7(quest_from_controller))
        boards.append(mat_to_pose7(camera_from_board))
    return controllers, boards, controller_from_camera, quest_from_board


def test_board_table_frame_is_centered_and_right_handed():
    board = CharucoBoardSpec()
    board_from_table = pose7_to_mat(board_from_table_pose(board))

    np.testing.assert_allclose(board_from_table[:3, 3], [0.075, 0.105, 0.0])
    np.testing.assert_allclose(board_from_table[:3, :3], np.diag([1.0, -1.0, -1.0]))
    assert np.linalg.det(board_from_table[:3, :3]) == 1.0


def test_canonical_board_detects_24_corners_and_ids_0_through_16():
    board = CharucoBoardSpec()
    rendered = board.create().generateImage((500, 700), marginSize=20)

    detection = detect_charuco(rendered, board, min_corners=12)

    assert detection is not None
    assert detection.count == 24
    assert detection.marker_ids is not None
    assert sorted(detection.marker_ids.reshape(-1).tolist()) == list(range(17))


def test_pinhole_board_pose_uses_native_distortion_model():
    matrix = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    object_points = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.1, 0.0], [0.0, 0.1, 0.0]],
        dtype=np.float64,
    ).reshape(-1, 1, 3)
    rvec = np.array([0.1, -0.2, 0.05])
    tvec = np.array([0.02, -0.01, 0.6])
    image_points, _ = cv2.projectPoints(object_points, rvec, tvec, matrix, np.zeros(5))
    detection = CharucoDetection(
        object_points=object_points,
        image_points=image_points,
        ids=np.arange(4),
        marker_corners=(),
        marker_ids=None,
    )
    intrinsics = CameraIntrinsics(
        camera="workspace",
        width=640,
        height=480,
        matrix=matrix,
        distortion=np.zeros((5, 1)),
        rms_px=0.0,
        mean_error_px=0.0,
        views=15,
        model="pinhole",
    )

    _, error_px = estimate_board_pose(detection, intrinsics)

    assert error_px < 1e-4


def test_controller_camera_solver_recovers_mount():
    controllers, boards, expected, _ = _synthetic_views()

    solved, metrics = solve_controller_camera(controllers, boards)

    np.testing.assert_allclose(pose7_to_mat(solved), expected, atol=2e-6)
    assert metrics["translation_rms_mm"] < 0.001
    assert metrics["rotation_rms_deg"] < 0.001


def test_session_solver_recovers_table_from_device():
    controllers, boards, controller_camera, quest_board = _synthetic_views()
    board = CharucoBoardSpec()
    expected = np.linalg.inv(quest_board @ pose7_to_mat(board_from_table_pose(board)))

    solved, metrics = solve_table_device(
        controllers,
        mat_to_pose7(controller_camera),
        boards,
        board,
    )

    np.testing.assert_allclose(pose7_to_mat(solved), expected, atol=2e-6)
    assert metrics["translation_rms_mm"] < 0.001


def test_legacy_quest_table_solver_alias_still_works():
    controllers, boards, controller_camera, quest_board = _synthetic_views()
    board = CharucoBoardSpec()
    expected = np.linalg.inv(quest_board @ pose7_to_mat(board_from_table_pose(board)))

    solved, _ = solve_table_quest(
        controllers,
        mat_to_pose7(controller_camera),
        boards,
        board,
    )

    np.testing.assert_allclose(pose7_to_mat(solved), expected, atol=2e-6)


def test_fixed_workspace_camera_solver_recovers_table_pose():
    board = CharucoBoardSpec()
    table_camera = _transform([0.2, -0.4, 0.8], [2.8, 0.1, -0.2])
    table_board = np.linalg.inv(pose7_to_mat(board_from_table_pose(board)))
    camera_board = np.linalg.inv(table_camera) @ table_board

    solved, metrics = solve_table_camera(
        [mat_to_pose7(camera_board) for _ in range(5)], board
    )

    np.testing.assert_allclose(pose7_to_mat(solved), table_camera, atol=2e-6)
    assert metrics["translation_rms_mm"] < 0.001


def test_calibration_hash_is_key_order_independent():
    assert calibration_hash({"a": 1, "b": 2}) == calibration_hash({"b": 2, "a": 1})


def test_session_metadata_embeds_matching_spatial_calibration(tmp_path):
    spatial_path = tmp_path / "spatial.yaml"
    session_path = tmp_path / "session.yaml"
    spatial = new_spatial_calibration(CharucoBoardSpec())
    write_yaml(spatial_path, spatial)
    session = {
        "kind": "handumi_session_calibration",
        "created_at": "2026-07-11T00:00:00+00:00",
        "spatial_calibration_path": str(spatial_path),
        "spatial_calibration_sha256": calibration_hash(spatial),
        "board": CharucoBoardSpec().to_dict(),
        "tracking_device": "pico",
        "table_from_device": pose7_to_dict(mat_to_pose7(np.eye(4))),
        "metrics": {},
    }
    write_yaml(session_path, session)

    metadata = session_calibration_metadata(session_path)

    assert metadata is not None
    assert metadata["spatial_calibration"] == spatial
    assert metadata["workspace_frame"] == "table"
    assert metadata["tracking_device"] == "pico"
    np.testing.assert_allclose(
        session_table_from_device(session_path),
        mat_to_pose7(np.eye(4)),
    )


def test_legacy_session_metadata_accepts_table_from_quest(tmp_path):
    spatial_path = tmp_path / "spatial.yaml"
    session_path = tmp_path / "session.yaml"
    spatial = new_spatial_calibration(CharucoBoardSpec())
    write_yaml(spatial_path, spatial)
    session = {
        "kind": "handumi_session_calibration",
        "created_at": "2026-07-11T00:00:00+00:00",
        "spatial_calibration_path": str(spatial_path),
        "spatial_calibration_sha256": calibration_hash(spatial),
        "board": CharucoBoardSpec().to_dict(),
        "table_from_quest": pose7_to_dict(mat_to_pose7(np.eye(4))),
        "metrics": {},
    }
    write_yaml(session_path, session)

    metadata = session_calibration_metadata(session_path)

    assert metadata is not None
    assert metadata["tracking_device"] == "meta"
    assert metadata["table_from_device"] == session["table_from_quest"]
    np.testing.assert_allclose(
        session_table_from_device(session_path),
        mat_to_pose7(np.eye(4)),
    )
