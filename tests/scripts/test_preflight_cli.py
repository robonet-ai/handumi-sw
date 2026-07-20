import sys
import tempfile
from pathlib import Path
from unittest import mock

import yaml

from handumi.calibration.spatial import calibration_hash
from handumi.scripts import preflight


def test_default_cli_is_read_only_and_requires_all_camera_roles():
    args = preflight.parse_args([])

    assert args.rig_config == Path("configs/rig.yaml")
    assert args.camera_names is None
    assert not args.interactive_remap
    assert not args.write_rig
    assert args.viser_port == 8003


def test_symlink_reenumerated_to_serial_is_classified_by_target(tmp_path):
    serial = tmp_path / "ttyUSB9"
    serial.touch()
    stale_camera = tmp_path / "left-wrist-camera"
    stale_camera.symlink_to(serial)

    base = preflight._device_base(str(stale_camera), {})

    assert base["available"]
    assert base["canonical_path"] == str(serial)
    assert base["device_class"] == "serial"


def test_camera_disconnect_runs_after_partial_probe_failure():
    class _Camera:
        def __init__(self, **kwargs):
            self.disconnected = False

        def connect(self):
            raise RuntimeError("busy")

        def disconnect(self):
            self.disconnected = True

    camera = _Camera()
    args = preflight.parse_args(["--camera", "left_wrist"])
    rig = {"cameras": {"left_wrist": {"index_or_path": "/dev/video0"}}}
    available = {
        "configured": "/dev/video0",
        "canonical_path": "/dev/video0",
        "device_class": "camera",
        "available": True,
        "readable": True,
        "writable": True,
        "busy_by": (),
        "serial": "camera",
        "usb_path": "usb-1",
    }
    with (
        mock.patch.object(preflight, "_device_base", return_value=available),
        mock.patch.object(preflight, "OpenCVCameraDevice", return_value=camera),
    ):
        probes = preflight._collect_cameras(args, rig, ("left_wrist",), handle_users={})

    assert camera.disconnected
    assert "busy" in (probes["left_wrist"].error or "")


def test_feetech_context_closes_after_partial_probe_failure():
    class _Bus:
        def __init__(self, **kwargs):
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.closed = True

        def scan(self, ids):
            raise RuntimeError("adapter disconnected")

    bus = _Bus()
    args = preflight.parse_args([])
    rig = {
        "feetech": {
            "left": {"port": "/dev/ttyUSB0", "servo_id": 0},
            "right": {"port": "/dev/missing", "servo_id": 1},
        }
    }
    available = {
        "configured": "/dev/ttyUSB0",
        "canonical_path": "/dev/ttyUSB0",
        "device_class": "serial",
        "available": True,
        "readable": True,
        "writable": True,
        "busy_by": (),
        "serial": "adapter",
        "usb_path": "usb-2",
    }
    missing = {
        **available,
        "configured": "/dev/missing",
        "canonical_path": None,
        "available": False,
    }
    with (
        mock.patch.object(preflight, "_device_base", side_effect=[available, missing]),
        mock.patch.object(preflight, "FeetechBus", return_value=bus),
        mock.patch.object(preflight, "load_config", side_effect=SystemExit("missing")),
    ):
        probes = preflight._collect_feetech(args, rig, handle_users={})

    assert bus.closed
    assert "adapter disconnected" in (probes["left"].error or "")


def test_session_probe_validates_device_hash_transform_and_all_camera_intrinsics(
    tmp_path,
):
    camera_entry = {
        "camera": "placeholder",
        "resolution": [640, 480],
        "model": "pinhole",
        "matrix": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
        "rms_px": 0.2,
        "mean_error_px": 0.1,
        "views": 20,
    }
    spatial = {
        "schema_version": 1,
        "kind": "handumi_spatial_calibration",
        "cameras": {
            name: {**camera_entry, "camera": name}
            for name in ("left_wrist", "right_wrist", "workspace")
        },
    }
    spatial_path = tmp_path / "spatial.yaml"
    spatial_path.write_text(yaml.safe_dump(spatial), encoding="utf-8")
    session = {
        "kind": "handumi_session_calibration",
        "tracking_device": "meta",
        "spatial_calibration_path": str(spatial_path),
        "spatial_calibration_sha256": calibration_hash(spatial),
        "table_from_device": {
            "translation_m": [0.0, 0.0, 0.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    }
    session_path = tmp_path / "session.yaml"
    session_path.write_text(yaml.safe_dump(session), encoding="utf-8")
    args = preflight.parse_args(
        ["--device", "meta", "--session-calibration", str(session_path)]
    )

    session_probe, camera_probe = preflight._session_calibration_probes(args)

    assert session_probe.valid
    assert camera_probe.valid
    pico_args = preflight.parse_args(
        ["--device", "pico", "--session-calibration", str(session_path)]
    )
    mismatched_session, _ = preflight._session_calibration_probes(pico_args)
    assert not mismatched_session.valid
    assert "mismatch" in (mismatched_session.error or "")


def test_dry_run_does_not_create_output_or_change_rig():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rig = root / "rig.yaml"
        rig.write_text(
            yaml.safe_dump(
                {
                    "cameras": {
                        name: {"index_or_path": f"/dev/{name}-missing"}
                        for name in ("left_wrist", "right_wrist", "workspace")
                    },
                    "feetech": {
                        "left": {"port": "/dev/tty-missing-left", "servo_id": 0},
                        "right": {"port": "/dev/tty-missing-right", "servo_id": 1},
                    },
                    "meta_quest": {
                        "connection": {
                            "quest_ip": "192.0.2.1",
                            "tcp_port": 65432,
                            "sync_port": 42000,
                        }
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        before = rig.read_bytes()
        output = root / "does-not-exist" / "dataset"
        calibration = root / "missing-calibration.yaml"
        argv = [
            "handumi-preflight",
            "--rig-config",
            str(rig),
            "--output-dir",
            str(output),
            "--feetech-calibration",
            str(calibration),
            "--skip-quest-probe",
            "--skip-camera-stream",
            "--skip-feetech-open",
            "--json",
        ]

        with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
            try:
                preflight.main()
            except SystemExit as exc:
                assert exc.code == 1
            else:
                raise AssertionError("missing hardware should fail preflight")

        assert not output.exists()
        assert rig.read_bytes() == before
