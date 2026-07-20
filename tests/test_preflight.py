import os
import tempfile
from pathlib import Path
from unittest import mock

import yaml

from handumi.preflight import (
    CalibrationProbe,
    CameraProbe,
    CheckStatus,
    FeetechProbe,
    PreflightInventory,
    PreflightRequest,
    QuestProbe,
    atomic_update_rig,
    evaluate_preflight,
)


def _camera(name: str, *, serial: str | None = None, **overrides) -> CameraProbe:
    values = {
        "configured": f"/dev/v4l/by-id/{name}",
        "canonical_path": f"/dev/video-{name}",
        "device_class": "camera",
        "available": True,
        "readable": True,
        "writable": True,
        "serial": serial or f"camera-{name}",
        "usb_path": f"pci-1-usb-{name}",
        "frame_count": 3,
        "width": 640,
        "height": 480,
        "fps": 30.0,
    }
    values.update(overrides)
    return CameraProbe(**values)


def _feetech(side: str, servo_id: int, **overrides) -> FeetechProbe:
    values = {
        "configured": f"/dev/serial/by-id/{side}",
        "canonical_path": f"/dev/ttyUSB-{side}",
        "device_class": "serial",
        "available": True,
        "readable": True,
        "writable": True,
        "serial": f"feetech-{side}",
        "usb_path": f"pci-2-usb-{side}",
        "configured_servo_id": servo_id,
        "detected_servo_ids": (servo_id,),
        "positions_by_id": {servo_id: (1000, 1010)},
        "calibration_range_valid": True,
    }
    values.update(overrides)
    return FeetechProbe(**values)


def _quest(**overrides) -> QuestProbe:
    values = {
        "connected": True,
        "streaming": True,
        "package_identifier": "com.handumi.questapp.bodyprobe",
        "version_name": "0.1.2",
        "protocol_schema": "tracking_packet_v2",
        "protocol_version": 2,
        "foreground_worn_observed": True,
        "hmd_tracked": True,
        "left_controller_tracked": True,
        "right_controller_tracked": True,
        "body_supported": True,
        "body_enabled": True,
        "body_active": True,
        "body_calibration_state": "Valid",
        "clock_synced": True,
        "clock_rtt_ms": 4.0,
        "source_timestamp_quality": "DIAGNOSTIC_ONLY",
    }
    values.update(overrides)
    return QuestProbe(**values)


def _inventory(**overrides) -> PreflightInventory:
    values = {
        "cameras": {
            name: _camera(name) for name in ("left_wrist", "right_wrist", "workspace")
        },
        "feetech": {
            "left": _feetech("left", 0),
            "right": _feetech("right", 1),
        },
        "quest": _quest(),
        "calibrations": tuple(
            CalibrationProbe(name, Path(f"/{name}.yaml"), True, True, True, "a" * 64)
            for name in ("controller_tcp", "session_table", "body_profile", "feetech")
        ),
        "local_port_users": {65432: (), 42000: (), 8003: ()},
        "output_writable": True,
        "output_probe_path": Path("/tmp"),
        "disk_free_bytes": 20 * 1024**3,
        "dependencies": {
            "numpy": "2.0",
            "cv2": "4.10",
            "rerun_sdk": "0.26",
            "yaml": "6.0",
        },
        "python_version": (3, 12, 10),
        "platform": "Linux",
    }
    values.update(overrides)
    return PreflightInventory(**values)


def _request(**overrides) -> PreflightRequest:
    values = {
        "require_body": True,
        "expected_package": "com.handumi.questapp.bodyprobe",
        "expected_version": "0.1.2",
    }
    values.update(overrides)
    return PreflightRequest(**values)


def test_complete_inventory_passes_without_qualified_timing_claims():
    report = evaluate_preflight(_request(), _inventory())

    assert report.passed
    assert all(check.status is CheckStatus.PASS for check in report.checks)
    clock = next(check for check in report.checks if check.code == "QUEST-CLOCK")
    assert clock.evidence["source_timestamp_quality"] == "DIAGNOSTIC_ONLY"


def test_pico_receive_only_timing_passes_as_diagnostic_not_synchronized():
    quest = _quest(
        clock_synced=False, clock_rtt_ms=None, source_timestamp_quality="RECEIVE_ONLY"
    )
    request = _request(
        device="pico",
        require_body=False,
        expected_package=None,
        expected_version=None,
        require_clock_sync=False,
    )

    report = evaluate_preflight(request, _inventory(quest=quest))

    assert report.passed
    clock = next(check for check in report.checks if check.code == "QUEST-CLOCK")
    assert "no synchronization claim" in clock.summary


def test_path_that_exists_but_is_now_serial_fails_camera_class_check():
    cameras = dict(_inventory().cameras)
    cameras["left_wrist"] = _camera(
        "left_wrist",
        canonical_path="/dev/ttyUSB0",
        device_class="serial",
    )

    report = evaluate_preflight(_request(), _inventory(cameras=cameras))

    check = next(
        item for item in report.checks if item.code == "CAMERA-LEFT_WRIST-CLASS"
    )
    assert check.status is CheckStatus.FAIL
    assert "not a camera" in check.summary


def test_duplicate_camera_serial_with_distinct_topology_warns_while_busy_fails():
    cameras = dict(_inventory().cameras)
    cameras["left_wrist"] = _camera("left_wrist", serial="duplicate")
    cameras["right_wrist"] = _camera(
        "right_wrist",
        serial="duplicate",
        busy_by=("pid=42:ffmpeg",),
    )

    report = evaluate_preflight(_request(), _inventory(cameras=cameras))

    duplicate = next(
        check for check in report.checks if check.code == "CAMERA-DUPLICATE-IDENTITY"
    )
    assert duplicate.status is CheckStatus.WARN
    assert "pinned by USB by-path" in duplicate.action
    assert "CAMERA-RIGHT_WRIST-ACCESS" in {
        check.code for check in report.failures
    }


def test_duplicate_usb_topology_fails_even_when_camera_serials_differ():
    cameras = dict(_inventory().cameras)
    cameras["left_wrist"] = _camera("left_wrist", usb_path="same-usb-path")
    cameras["right_wrist"] = _camera("right_wrist", usb_path="same-usb-path")

    report = evaluate_preflight(_request(), _inventory(cameras=cameras))

    assert "CAMERA-USB-TOPOLOGY" in {check.code for check in report.failures}


def test_partial_hardware_and_wrong_feetech_class_fail_closed():
    feetech = dict(_inventory().feetech)
    feetech["left"] = _feetech("left", 0, device_class="camera")
    feetech["right"] = _feetech(
        "right",
        1,
        detected_servo_ids=(),
        positions_by_id={1: ()},
    )
    cameras = dict(_inventory().cameras)
    cameras.pop("workspace")

    report = evaluate_preflight(
        _request(), _inventory(cameras=cameras, feetech=feetech)
    )

    codes = {check.code for check in report.failures}
    assert "CAMERA-WORKSPACE-AVAILABLE" in codes
    assert "FEETECH-LEFT-CLASS" in codes
    assert "FEETECH-RIGHT-SERVO" in codes


def test_static_encoder_is_warning_but_invalid_calibration_is_failure():
    feetech = dict(_inventory().feetech)
    feetech["left"] = _feetech("left", 0, positions_by_id={0: (1000, 1000)})
    calibrations = (
        CalibrationProbe(
            "feetech",
            Path("/cache.yaml"),
            True,
            True,
            False,
            error="incomplete endpoints",
        ),
    )

    report = evaluate_preflight(
        _request(), _inventory(feetech=feetech, calibrations=calibrations)
    )

    motion = next(
        check for check in report.checks if check.code == "FEETECH-LEFT-MOTION"
    )
    assert motion.status is CheckStatus.WARN
    assert "CALIBRATION-FEETECH" in {check.code for check in report.failures}


def test_live_encoder_outside_cached_endpoints_fails_range_check():
    feetech = dict(_inventory().feetech)
    feetech["left"] = _feetech("left", 0, calibration_range_valid=False)

    report = evaluate_preflight(_request(), _inventory(feetech=feetech))

    assert "FEETECH-LEFT-RANGE" in {check.code for check in report.failures}


def test_quest_package_protocol_tracking_body_and_clock_mismatches_fail():
    quest = _quest(
        package_identifier="com.example.wrong",
        protocol_version=1,
        foreground_worn_observed=False,
        body_calibration_state="Invalid",
        clock_synced=False,
        clock_rtt_ms=None,
    )

    report = evaluate_preflight(_request(), _inventory(quest=quest))

    codes = {check.code for check in report.failures}
    assert {
        "QUEST-PACKAGE",
        "QUEST-PROTOCOL",
        "QUEST-FOREGROUND-TRACKING",
        "QUEST-CLOCK",
        "QUEST-BODY",
    } <= codes


def test_disk_dependency_port_and_runtime_failures_are_reported_together():
    inventory = _inventory(
        local_port_users={8003: ("pid=7:viser",)},
        output_writable=False,
        disk_free_bytes=1,
        dependencies={"numpy": None, "cv2": "4", "rerun_sdk": "0.26", "yaml": "6"},
        python_version=(3, 11, 9),
        platform="Darwin",
    )

    report = evaluate_preflight(_request(), inventory)

    codes = {check.code for check in report.failures}
    assert {
        "RUNTIME-PYTHON",
        "RUNTIME-PLATFORM",
        "DEPENDENCY-NUMPY",
        "OUTPUT-WRITABLE",
        "OUTPUT-DISK",
        "PORT-8003",
    } <= codes


def test_atomic_rig_update_changes_only_assignments_and_preserves_mode():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rig.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "cameras": {
                        "left_wrist": {"index_or_path": 0, "note": "keep"},
                    },
                    "feetech": {
                        "left": {"port": "/dev/ttyUSB0", "servo_id": 5},
                    },
                    "meta_quest": {"connection": {"quest_ip": "192.0.2.1"}},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        os.chmod(path, 0o640)

        atomic_update_rig(
            path,
            camera_paths={"left_wrist": "/dev/v4l/by-id/camera-left"},
            feetech_ports={"left": "/dev/serial/by-id/feetech-left"},
        )
        updated = yaml.safe_load(path.read_text(encoding="utf-8"))

        assert updated["cameras"]["left_wrist"] == {
            "index_or_path": "/dev/v4l/by-id/camera-left",
            "note": "keep",
        }
        assert updated["feetech"]["left"] == {
            "port": "/dev/serial/by-id/feetech-left",
            "servo_id": 5,
        }
        assert updated["meta_quest"]["connection"]["quest_ip"] == "192.0.2.1"
        assert path.stat().st_mode & 0o777 == 0o640
        assert list(path.parent.glob(".rig.yaml.*.tmp")) == []


def test_failed_atomic_replace_leaves_original_and_removes_temporary_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rig.yaml"
        original = "cameras:\n  left_wrist:\n    index_or_path: 0\n"
        path.write_text(original, encoding="utf-8")

        with mock.patch(
            "handumi.preflight.os.replace",
            side_effect=OSError("simulated disk failure"),
        ):
            try:
                atomic_update_rig(path, camera_paths={"left_wrist": 2})
            except OSError as exc:
                assert "simulated disk failure" in str(exc)
            else:
                raise AssertionError("atomic update should propagate replace failure")

        assert path.read_text(encoding="utf-8") == original
        assert list(path.parent.glob(".rig.yaml.*.tmp")) == []
