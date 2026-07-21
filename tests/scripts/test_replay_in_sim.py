from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from handumi.scripts.replay.replay_in_sim import (
    _metadata_tcp_calibration,
    _render_task_scene,
    _resolve_gripper_openings,
    _resolved_controller_device,
    _resolved_retarget_mode,
    _resolved_tcp_calibration,
    _source_robot_from_metadata,
    _tcp_geometry_diagnostics,
    load_robot_from_table,
)


def test_recorded_normalized_grippers_take_precedence():
    states = np.zeros((2, 16), dtype=np.float32)
    states[:, 14:16] = 0.08
    recorded = np.array([[0.2, 0.7], [0.3, 0.8]], dtype=np.float32)

    openings, source = _resolve_gripper_openings(
        states, recorded, max_width_m=0.08
    )

    np.testing.assert_allclose(openings, recorded)
    assert source == "recorded Feetech normalized"


def test_grippers_fall_back_to_widths_in_meters():
    states = np.zeros((2, 16), dtype=np.float32)
    states[:, 14:16] = [[0.0, 0.033], [0.066, 0.099]]

    openings, source = _resolve_gripper_openings(
        states, None, max_width_m=0.066
    )

    np.testing.assert_allclose(openings, [[0.0, 0.5], [1.0, 1.0]], atol=1e-6)
    assert source == "state widths in meters"


def test_physical_width_gripper_retarget_overrides_recorded_percentage():
    states = np.zeros((2, 16), dtype=np.float32)
    states[:, 14:16] = [[0.024, 0.048], [0.096, 0.12]]
    recorded = np.array([[0.2, 0.7], [0.3, 0.8]], dtype=np.float32)

    openings, source = _resolve_gripper_openings(
        states,
        recorded,
        max_width_m=0.096,
        mode="physical-width",
    )

    np.testing.assert_allclose(openings, [[0.25, 0.5], [1.0, 1.0]], atol=1e-6)
    assert source == "state widths in meters"


def test_controller_device_is_read_from_dataset_metadata():
    args = Namespace(controller_device=None)
    info = {"handumi": {"recording_device": "meta"}}

    assert _resolved_controller_device(args, info) == "meta"


def test_explicit_controller_device_overrides_metadata():
    args = Namespace(controller_device="pico")
    info = {"handumi": {"recording_device": "meta"}}

    assert _resolved_controller_device(args, info) == "pico"


def test_controller_device_cannot_contradict_identity_bound_snapshot():
    args = Namespace(controller_device="pico")
    info = _metadata_calibration_info()
    info["handumi"]["controller_tcp_calibration"].update(
        {
            "schema_version": 2,
            "source_robot": "piper",
            "source_gripper": "piper_parallel_v1",
            "tracking_device": "meta",
            "controller_mount": "handumi_v1",
        }
    )

    with np.testing.assert_raises(SystemExit):
        _resolved_controller_device(args, info)


def test_auto_retarget_uses_absolute_table_for_calibrated_table_dataset():
    args = Namespace(retarget_mode="auto")
    info = {"handumi": {"tracking_workspace": "table"}}

    assert _resolved_retarget_mode(args, info) == "absolute-table"


def test_auto_retarget_falls_back_to_local_relative_without_table_frame():
    args = Namespace(retarget_mode="auto")

    assert _resolved_retarget_mode(args, {"handumi": {}}) == "local-relative"


def test_explicit_retarget_mode_overrides_dataset_metadata():
    args = Namespace(retarget_mode="anchored")
    info = {"handumi": {"tracking_workspace": "table"}}

    assert _resolved_retarget_mode(args, info) == "anchored"


def test_controller_tcp_calibration_is_loaded_from_metadata():
    info = {
        "handumi": {
            "controller_tcp_calibration": {
                "sha256": "abc123",
                "applied_to_state": False,
                "controller_to_gripper_tcp": {
                    "left": {
                        "position": [0.1, 0.2, 0.3],
                        "quaternion": [0.0, 0.0, 0.0, 1.0],
                    },
                    "right": {
                        "position": [-0.1, -0.2, -0.3],
                        "quaternion": [0.0, 0.0, 0.0, 1.0],
                    },
                },
            }
        }
    }

    resolved = _metadata_tcp_calibration(info)

    assert resolved is not None
    calibration, source = resolved
    np.testing.assert_allclose(calibration.left[:3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(calibration.right[:3], [-0.1, -0.2, -0.3])
    assert source == "legacy dataset metadata sha256=abc123"


def _metadata_calibration_info() -> dict[str, object]:
    return {
        "handumi": {
            "controller_tcp_calibration": {
                "sha256": "old-snapshot",
                "applied_to_state": False,
                "controller_to_gripper_tcp": {
                    "left": {
                        "position": [0.01, 0.02, 0.03],
                        "quaternion": [0.0, 0.0, 0.0, 1.0],
                    },
                    "right": {
                        "position": [-0.01, -0.02, -0.03],
                        "quaternion": [0.0, 0.0, 0.0, 1.0],
                    },
                },
            }
        }
    }


def test_piper_meta_configured_tcp_calibration_precedes_dataset_snapshot():
    from handumi.robots.registry import load_robot_config

    args = Namespace(
        controller_tcp_calibration=None,
        use_dataset_tcp_calibration=False,
    )
    configured = load_robot_config("piper").controller_tcp_calibrations["meta"]

    selection = _resolved_tcp_calibration(
        args,
        _metadata_calibration_info(),
        robot="piper",
        controller_device="meta",
        configured_path=configured,
        configured_gripper="piper_parallel_v1",
        configured_mount="handumi_v1",
    )

    np.testing.assert_allclose(
        selection.calibration.left[:3], [0.12068467, 0.02142489, -0.21669616]
    )
    assert selection.source.startswith("configured piper/meta:")
    assert len(selection.source.rsplit("sha256=", 1)[1]) == 64
    assert selection.source_gripper == "piper_parallel_v1"
    assert selection.controller_mount == "handumi_v1"
    assert selection.trusted_dataset_snapshot is False


def test_dataset_tcp_snapshot_can_be_requested_explicitly():
    args = Namespace(
        controller_tcp_calibration=None,
        use_dataset_tcp_calibration=True,
    )

    selection = _resolved_tcp_calibration(
        args,
        _metadata_calibration_info(),
        robot="piper",
        controller_device="meta",
        configured_path=Path("configs/calibration/meta_controller_tcp.yaml"),
    )

    np.testing.assert_allclose(selection.calibration.left[:3], [0.01, 0.02, 0.03])
    assert selection.source == "legacy dataset metadata sha256=old-snapshot"


def test_explicit_tcp_path_precedes_configured_and_dataset(tmp_path: Path):
    explicit = tmp_path / "explicit_tcp.yaml"
    explicit.write_text(
        """\
calibration:
  controller_to_gripper_tcp:
    left:
      position: [0.4, 0.5, 0.6]
      quaternion: [0.0, 0.0, 0.0, 1.0]
    right:
      position: [-0.4, -0.5, -0.6]
      quaternion: [0.0, 0.0, 0.0, 1.0]
""",
        encoding="utf-8",
    )
    args = Namespace(
        controller_tcp_calibration=explicit,
        use_dataset_tcp_calibration=False,
    )

    selection = _resolved_tcp_calibration(
        args,
        _metadata_calibration_info(),
        robot="piper",
        controller_device="meta",
        configured_path=Path("configs/calibration/meta_controller_tcp.yaml"),
    )

    np.testing.assert_allclose(selection.calibration.left[:3], [0.4, 0.5, 0.6])
    assert selection.source.startswith(f"explicit {explicit} sha256=")
    assert len(selection.source.rsplit("sha256=", 1)[1]) == 64


def test_identity_bound_dataset_snapshot_precedes_current_robot_setup():
    info = _metadata_calibration_info()
    snapshot = info["handumi"]["controller_tcp_calibration"]
    snapshot.update(
        {
            "schema_version": 2,
            "source_robot": "piper",
            "source_gripper": "piper_parallel_v1",
            "tracking_device": "meta",
            "controller_mount": "handumi_v1",
        }
    )
    args = Namespace(
        controller_tcp_calibration=None,
        use_dataset_tcp_calibration=False,
    )

    selection = _resolved_tcp_calibration(
        args,
        info,
        robot="piper",
        controller_device="meta",
        configured_path=Path("configs/calibration/meta_controller_tcp.yaml"),
        configured_gripper="piper_parallel_v1",
        configured_mount="handumi_v1",
    )

    np.testing.assert_allclose(selection.calibration.left[:3], [0.01, 0.02, 0.03])
    assert selection.trusted_dataset_snapshot is True
    assert selection.source_robot == "piper"
    assert selection.source_gripper == "piper_parallel_v1"
    assert selection.source.startswith("dataset robot-tool snapshot piper/meta")


def test_source_robot_uses_legacy_target_robot_before_replay_target():
    info = {"handumi": {"target_robot": {"name": "piper"}}}

    assert _source_robot_from_metadata(info, fallback="axol") == "piper"


def test_tcp_geometry_diagnostics_reports_offsets_z_and_synchronous_distance():
    from handumi.calibration.control_tcp import ControllerTcpCalibration

    calibration = ControllerTcpCalibration(
        left=np.array([0.0, 0.0, -0.2, 0.0, 0.0, 0.0, 1.0]),
        right=np.array([0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 1.0]),
    )
    left = np.array(
        [[0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 1.0], [0.1, 0.0, 0.0, 0, 0, 0, 1]],
        dtype=np.float32,
    )
    right = np.array(
        [[0.3, 0.0, 0.04, 0.0, 0.0, 0.0, 1.0], [0.2, 0.0, 0.01, 0, 0, 0, 1]],
        dtype=np.float32,
    )

    diagnostics = _tcp_geometry_diagnostics(calibration, left, right)

    np.testing.assert_allclose(diagnostics["offset_position_norm_m"], [0.2, 0.1])
    np.testing.assert_allclose(diagnostics["workspace_min_z_m"], [0.0, 0.01])
    np.testing.assert_allclose(
        diagnostics["same_frame_min_separation_m"],
        [np.sqrt(0.1**2 + 0.01**2)],
    )


def test_load_robot_from_table(tmp_path: Path):
    path = tmp_path / "deployment.yaml"
    path.write_text(
        """\
calibration:
  robot_from_table:
    position: [0.3, 0.0, 0.1]
    quaternion: [0.0, 0.0, 0.0, 1.0]
""",
        encoding="utf-8",
    )

    pose = load_robot_from_table(path)

    np.testing.assert_allclose(pose, [0.3, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0])


def test_load_robot_from_table_rejects_wrong_robot(tmp_path: Path):
    path = tmp_path / "deployment.yaml"
    path.write_text(
        """\
robot: axol
calibration:
  robot_from_table:
    position: [0.3, 0.0, 0.1]
    quaternion: [0.0, 0.0, 0.0, 1.0]
""",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="declares robot 'axol'; expected 'piper'"):
        load_robot_from_table(path, expected_robot="piper")


def test_absolute_table_parser_defaults_prepare_start_and_align_tools():
    from handumi.scripts.replay.replay_in_sim import build_parser

    args = build_parser().parse_args(["dataset"])

    assert args.retarget_mode == "auto"
    assert args.hide_trajectories is False
    assert args.use_dataset_tcp_calibration is False
    assert args.absolute_orientation == "relative-start"
    assert args.initial_solve_iterations == 12
    assert args.initial_position_tolerance_m == 0.01
    assert args.max_ik_position_error_m == 0.03
    assert args.max_ik_rotation_error_deg == 45.0
    assert args.table_clearance_warning_m == 0.10


def test_hide_trajectories_parser_flag():
    from handumi.scripts.replay.replay_in_sim import build_parser

    args = build_parser().parse_args(["dataset", "--hide-trajectories"])

    assert args.hide_trajectories is True


def test_render_task_scene_maps_table_bodies_into_robot_world():
    class FakeScene:
        def __init__(self):
            self.frames = []
            self.boxes = []

        def add_frame(self, name, **kwargs):
            self.frames.append((name, kwargs))
            return object()

        def add_box(self, name, **kwargs):
            self.boxes.append((name, kwargs))
            return object()

    class FakeServer:
        def __init__(self):
            self.scene = FakeScene()

    server = FakeServer()
    args = Namespace(scene="cube_in_box")
    rollout = {
        "robot_from_table_pose7": np.array(
            [[0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32
        )
    }

    _render_task_scene(server, args, rollout)

    assert len(server.scene.frames) == 2
    assert len(server.scene.boxes) == 6
    cube = next(item for item in server.scene.frames if item[0].endswith("cube"))
    np.testing.assert_allclose(cube[1]["position"], [0.3, -0.1, 0.0], atol=1e-6)
