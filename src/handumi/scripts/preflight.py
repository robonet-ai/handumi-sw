#!/usr/bin/env python3
"""Read-only HandUMI hardware preflight with optional explicit rig remapping."""

from __future__ import annotations

import argparse
import glob
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from handumi.body.com import BodyProfile
from handumi.calibration.control_tcp import (
    calibration_path_for_robot_device,
    load_controller_tcp_calibration,
)
from handumi.calibration.spatial import (
    CameraIntrinsics,
    load_yaml as load_spatial_yaml,
    session_calibration_metadata,
    session_table_from_device,
)
from handumi.cameras.opencv import OpenCVCameraDevice
from handumi.config import DEFAULT_RIG_CONFIG
from handumi.feetech.bus import FeetechBus
from handumi.feetech.calibration import load_config, user_calibration_path
from handumi.preflight import (
    CalibrationProbe,
    CameraProbe,
    DeviceProbe,
    FeetechProbe,
    PreflightInventory,
    PreflightRequest,
    QuestProbe,
    atomic_update_rig,
    evaluate_preflight,
)
from handumi.tracking.meta_quest import MetaQuestConfig, MetaQuestReceiver


_DEPENDENCY_DISTRIBUTIONS = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "rerun_sdk": "rerun-sdk",
    "viser": "viser",
    "yaml": "PyYAML",
    "scservo_sdk": "feetech-servo-sdk",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("meta", "pico"), default="meta")
    parser.add_argument("--robot", default="piper")
    parser.add_argument("--rig-config", type=Path, default=DEFAULT_RIG_CONFIG)
    parser.add_argument(
        "--require-body", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--expected-package", default=None)
    parser.add_argument("--expected-version", default=None)
    parser.add_argument("--expected-build-id", default=None)
    parser.add_argument("--expected-protocol-schema", default="tracking_packet_v2")
    parser.add_argument("--expected-protocol-version", type=int, default=None)
    parser.add_argument("--quest-timeout-s", type=float, default=5.0)
    parser.add_argument(
        "--pico-mode",
        choices=("mandos", "object", "whole-body"),
        default="mandos",
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument(
        "--camera",
        dest="camera_names",
        action="append",
        choices=("left_wrist", "right_wrist", "workspace"),
        help="Camera role to require; repeat as needed (default: all three).",
    )
    parser.add_argument("--camera-probe-timeout-s", type=float, default=2.0)
    parser.add_argument("--feetech-motion-window-s", type=float, default=1.0)
    parser.add_argument("--feetech-calibration", type=Path, default=None)
    parser.add_argument("--session-calibration", type=Path, default=None)
    parser.add_argument("--body-profile", type=Path, default=None)
    parser.add_argument("--controller-tcp-calibration", type=Path, default=None)
    parser.add_argument(
        "--camera-calibration",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/datasets"))
    parser.add_argument("--min-free-gb", type=float, default=10.0)
    parser.add_argument("--viser-port", type=int, default=8003)
    parser.add_argument("--rerun-port", type=int, action="append", default=[])
    parser.add_argument(
        "--skip-quest-probe",
        action="store_true",
        help="Report Quest checks as failed without opening the network receiver.",
    )
    parser.add_argument(
        "--skip-camera-stream",
        action="store_true",
        help="Classify camera paths but do not open them; stream checks will fail.",
    )
    parser.add_argument(
        "--skip-feetech-open",
        action="store_true",
        help="Classify serial paths but do not open buses; servo checks will fail.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON report.")
    parser.add_argument(
        "--interactive-remap",
        action="store_true",
        help="Prompt for camera/Feetech path remapping after the read-only checks.",
    )
    parser.add_argument(
        "--write-rig",
        action="store_true",
        help="Atomically write confirmed remapping to --rig-config; requires --interactive-remap.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.write_rig and not args.interactive_remap:
        raise SystemExit("--write-rig requires --interactive-remap.")
    if args.min_free_gb <= 0:
        raise SystemExit("--min-free-gb must be greater than zero.")
    if (
        min(
            args.quest_timeout_s,
            args.camera_probe_timeout_s,
            args.feetech_motion_window_s,
        )
        < 0
    ):
        raise SystemExit("Probe timeouts/windows must not be negative.")

    rig = _load_rig(args.rig_config)
    camera_names = tuple(
        dict.fromkeys(args.camera_names or ("left_wrist", "right_wrist", "workspace"))
    )
    expected_package = args.expected_package
    if expected_package is None and args.device == "meta":
        expected_package = (
            "com.handumi.questapp.bodyprobe"
            if args.require_body
            else "com.handumi.questapp"
        )
    request = PreflightRequest(
        device=args.device,
        require_body=args.require_body,
        expected_package=expected_package,
        expected_version=args.expected_version,
        expected_build_id=args.expected_build_id,
        expected_protocol_schema=args.expected_protocol_schema,
        expected_protocol_version=(
            args.expected_protocol_version
            if args.expected_protocol_version is not None
            else 2
            if args.require_body or args.device == "pico"
            else 1
        ),
        require_clock_sync=args.device == "meta",
        camera_names=camera_names,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        local_ports=tuple(
            dict.fromkeys(
                (
                    _quest_port(rig, "tcp_port", 65432),
                    _quest_port(rig, "sync_port", 42000),
                    args.viser_port,
                    *args.rerun_port,
                )
            )
        ),
        min_disk_bytes=int(args.min_free_gb * 1024**3),
        required_dependencies=(
            "numpy",
            "cv2",
            "rerun_sdk",
            "yaml",
            "viser",
            "scservo_sdk",
        ),
    )

    configured_devices = _configured_device_paths(rig, camera_names)
    handle_users = _device_handle_users(configured_devices.values())
    cameras = _collect_cameras(args, rig, camera_names, handle_users)
    feetech = _collect_feetech(args, rig, handle_users)
    quest = (
        QuestProbe(error="Quest probe explicitly skipped")
        if args.skip_quest_probe
        else _collect_quest(args, expected_package)
    )
    output_probe_path = _nearest_existing_parent(args.output_dir)
    try:
        disk_free_bytes = shutil.disk_usage(output_probe_path).free
    except OSError:
        disk_free_bytes = 0
    inventory = PreflightInventory(
        cameras=cameras,
        feetech=feetech,
        quest=quest,
        calibrations=_collect_calibrations(args),
        local_port_users=_local_port_users(request.local_ports),
        output_writable=os.access(output_probe_path, os.W_OK | os.X_OK),
        output_probe_path=output_probe_path,
        disk_free_bytes=disk_free_bytes,
        dependencies=_dependency_versions(request.required_dependencies),
        python_version=sys.version_info[:3],
        platform=platform.system(),
    )
    report = evaluate_preflight(request, inventory)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        _print_report(report)

    if args.interactive_remap:
        camera_paths, feetech_ports = _interactive_remap(camera_names)
        if camera_paths or feetech_ports:
            print("\nProposed local rig changes:")
            for name, value in camera_paths.items():
                print(f"  camera {name}: {value}")
            for side, value in feetech_ports.items():
                print(f"  Feetech {side}: {value}")
            if not args.write_rig:
                print(
                    "Dry-run only; rerun with --interactive-remap --write-rig to persist atomically."
                )
            elif _confirm("Write these changes atomically to the local rig? [y/N]: "):
                atomic_update_rig(
                    args.rig_config,
                    camera_paths=camera_paths,
                    feetech_ports=feetech_ports,
                )
                print(f"Updated {args.rig_config}; rerun handumi-preflight.")
            else:
                print("Rig unchanged.")
    if not report.passed:
        raise SystemExit(1)


def _load_rig(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(
            f"Missing rig configuration: {path}. Copy configs/rig.example.yaml and map this machine."
        )
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"Invalid rig configuration {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid rig configuration {path}: expected a mapping.")
    return data


def _configured_device_paths(
    rig: Mapping[str, Any], camera_names: tuple[str, ...]
) -> dict[str, int | str]:
    paths: dict[str, int | str] = {}
    cameras = rig.get("cameras") or {}
    if isinstance(cameras, Mapping):
        for name in camera_names:
            entry = cameras.get(name) or {}
            if isinstance(entry, Mapping) and "index_or_path" in entry:
                paths[f"camera.{name}"] = entry["index_or_path"]
    feetech = rig.get("feetech") or {}
    if isinstance(feetech, Mapping):
        for side in ("left", "right"):
            entry = feetech.get(side) or {}
            if isinstance(entry, Mapping) and entry.get("port"):
                paths[f"feetech.{side}"] = str(entry["port"])
    return paths


def _camera_path(value: int | str) -> str:
    return f"/dev/video{value}" if isinstance(value, int) else str(value)


def _device_base(
    configured: int | str, handle_users: Mapping[str, tuple[str, ...]]
) -> dict[str, Any]:
    path = _camera_path(configured) if isinstance(configured, int) else str(configured)
    canonical = os.path.realpath(path) if os.path.lexists(path) else None
    available = canonical is not None and Path(canonical).exists()
    device_class = _classify_device(canonical or path)
    properties = _udev_properties(canonical or path) if available else {}
    busy = tuple(
        dict.fromkeys(
            (
                *handle_users.get(path, ()),
                *handle_users.get(canonical or "", ()),
            )
        )
    )
    return {
        "configured": configured,
        "canonical_path": canonical,
        "device_class": device_class,
        "available": available,
        "readable": available and os.access(canonical or path, os.R_OK),
        "writable": available and os.access(canonical or path, os.W_OK),
        "busy_by": busy,
        "serial": properties.get("ID_SERIAL_SHORT") or properties.get("ID_SERIAL"),
        "usb_path": properties.get("ID_PATH"),
    }


def _classify_device(path: str) -> str:
    name = Path(path).name
    if name.startswith("video") or Path("/sys/class/video4linux", name).exists():
        return "camera"
    if name.startswith(("ttyACM", "ttyUSB")) or Path("/sys/class/tty", name).exists():
        return "serial"
    return "unknown"


def _udev_properties(path: str) -> dict[str, str]:
    if shutil.which("udevadm") is None:
        return {}
    try:
        result = subprocess.run(
            ["udevadm", "info", "--query=property", "--name", path],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    properties = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            properties[key] = value
    return properties


def _collect_cameras(
    args: argparse.Namespace,
    rig: Mapping[str, Any],
    names: tuple[str, ...],
    handle_users: Mapping[str, tuple[str, ...]],
) -> dict[str, CameraProbe]:
    configured = rig.get("cameras") or {}
    probes: dict[str, CameraProbe] = {}
    for name in names:
        entry = configured.get(name) or {} if isinstance(configured, Mapping) else {}
        value: int | str = (
            entry.get("index_or_path", "") if isinstance(entry, Mapping) else ""
        )
        base = _device_base(value, handle_users)
        probe = CameraProbe(**base)
        if (
            args.skip_camera_stream
            or not probe.available
            or probe.device_class != "camera"
            or probe.busy_by
        ):
            probes[name] = probe
            continue
        camera = OpenCVCameraDevice(
            index_or_path=value,
            fps=args.camera_fps,
            width=args.camera_width,
            height=args.camera_height,
        )
        samples = {}
        try:
            camera.connect()
            deadline = time.monotonic() + args.camera_probe_timeout_s
            while time.monotonic() <= deadline and len(samples) < 3:
                sample = camera.sample_at()
                samples[sample.sequence] = sample
                time.sleep(0.01)
            ordered = sorted(
                samples.values(), key=lambda sample: sample.capture_time_ns
            )
            fps = None
            if (
                len(ordered) >= 2
                and ordered[-1].capture_time_ns > ordered[0].capture_time_ns
            ):
                fps = (
                    (len(ordered) - 1)
                    * 1e9
                    / (ordered[-1].capture_time_ns - ordered[0].capture_time_ns)
                )
            shape = ordered[-1].image.shape if ordered else ()
            probes[name] = replace(
                probe,
                frame_count=len(ordered),
                width=int(shape[1]) if len(shape) >= 2 else None,
                height=int(shape[0]) if len(shape) >= 2 else None,
                fps=fps,
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic boundary.
            probes[name] = replace(probe, error=f"{type(exc).__name__}: {exc}")
        finally:
            camera.disconnect()
    return probes


def _collect_feetech(
    args: argparse.Namespace,
    rig: Mapping[str, Any],
    handle_users: Mapping[str, tuple[str, ...]],
) -> dict[str, FeetechProbe]:
    root = rig.get("feetech") or {}
    baudrate = (
        int(root.get("baudrate", 1_000_000)) if isinstance(root, Mapping) else 1_000_000
    )
    protocol = int(root.get("protocol_version", 0)) if isinstance(root, Mapping) else 0
    probes: dict[str, FeetechProbe] = {}
    calibration_path = args.feetech_calibration or user_calibration_path()
    try:
        merged_config = load_config(args.rig_config, calibration_path)
    except (OSError, SystemExit, TypeError, ValueError, yaml.YAMLError):
        merged_config = None
    for side, default_id in (("left", 0), ("right", 1)):
        entry = root.get(side) or {} if isinstance(root, Mapping) else {}
        port = str(entry.get("port", "")) if isinstance(entry, Mapping) else ""
        servo_id = (
            int(entry.get("servo_id", default_id))
            if isinstance(entry, Mapping)
            else default_id
        )
        base = _device_base(port, handle_users)
        probe = FeetechProbe(**base, configured_servo_id=servo_id)
        if (
            args.skip_feetech_open
            or not probe.available
            or probe.device_class != "serial"
            or probe.busy_by
            or not (probe.readable and probe.writable)
        ):
            probes[side] = probe
            continue
        try:
            with FeetechBus(
                port=port,
                baudrate=baudrate,
                protocol_version=protocol,
            ) as bus:
                detected = tuple(bus.scan(range(0, 21)))
                positions: list[int] = []
                if servo_id in detected:
                    positions.append(bus.read_position(servo_id))
                    if args.feetech_motion_window_s > 0:
                        time.sleep(args.feetech_motion_window_s)
                        positions.append(bus.read_position(servo_id))
                calibration = (
                    None if merged_config is None else getattr(merged_config, side)
                )
                range_valid = None
                if calibration is not None and positions:
                    try:
                        for position in positions:
                            calibration.normalized_width(position)
                        range_valid = True
                    except ValueError:
                        range_valid = False
                probes[side] = replace(
                    probe,
                    detected_servo_ids=detected,
                    positions_by_id={servo_id: tuple(positions)},
                    calibration_range_valid=range_valid,
                )
        except Exception as exc:  # noqa: BLE001 - diagnostic boundary.
            probes[side] = replace(
                probe,
                positions_by_id={servo_id: ()},
                error=f"{type(exc).__name__}: {exc}",
            )
    return probes


def _collect_quest(
    args: argparse.Namespace, expected_package: str | None
) -> QuestProbe:
    if args.device != "meta":
        return _collect_pico(args)
    try:
        config = MetaQuestConfig.from_yaml(args.rig_config)
    except (OSError, SystemExit, TypeError, ValueError) as exc:
        return QuestProbe(error=str(exc))
    receiver = MetaQuestReceiver(config)
    try:
        receiver.start()
        deadline = time.monotonic() + args.quest_timeout_s
        packet = None
        manifest = None
        while time.monotonic() <= deadline:
            packet = receiver.latest_packet()
            manifest = receiver.session_manifest()
            metrics = receiver.metrics()
            if packet is not None and (manifest is not None or not args.require_body):
                break
            time.sleep(0.05)
        metrics = receiver.metrics()
        packet = receiver.latest_packet()
        manifest = receiver.session_manifest() or {}
        controllers = {
            controller.side: controller.pose.tracking_state.value == "TRACKED"
            for controller in (() if packet is None else packet.controllers)
        }
        body = None if packet is None else packet.body
        rtt_ms = metrics.get("rtt_ms")
        clock_synced = metrics.get("rtt_ns") is not None
        source_quality = None if packet is None else packet.timestamps.quality.value
        return QuestProbe(
            connected=bool(metrics.get("connected")),
            streaming=bool(metrics.get("streaming")),
            package_identifier=_manifest_text(manifest, "packageIdentifier"),
            version_name=_manifest_text(manifest, "versionName"),
            build_id=_manifest_text(manifest, "buildId"),
            source_commit=_manifest_text(manifest, "sourceCommit"),
            protocol_schema=None if packet is None else packet.schema,
            protocol_version=None if packet is None else packet.source_schema_version,
            manifest_schema=_manifest_text(manifest, "schema"),
            foreground_worn_observed=bool(
                metrics.get("streaming")
                and packet is not None
                and packet.hmd is not None
                and packet.hmd.tracking_state.value == "TRACKED"
            ),
            hmd_tracked=bool(
                packet is not None
                and packet.hmd is not None
                and packet.hmd.tracking_state.value == "TRACKED"
            ),
            left_controller_tracked=controllers.get("left", False),
            right_controller_tracked=controllers.get("right", False),
            body_supported=_manifest_bool(manifest, "bodyTrackingSupported"),
            body_enabled=_manifest_bool(manifest, "bodyTrackingEnabled"),
            body_active=None if body is None else body.active,
            body_calibration_state=(
                _manifest_text(manifest, "calibrationState")
                if body is None
                else body.calibration_state
            ),
            clock_synced=clock_synced,
            clock_rtt_ms=(
                float(rtt_ms) if clock_synced and rtt_ms is not None else None
            ),
            source_timestamp_quality=source_quality,
            error=(
                None
                if metrics.get("streaming")
                else f"No live {expected_package} frame before {args.quest_timeout_s:.1f}s timeout"
            ),
        )
    finally:
        receiver.stop()


def _collect_pico(args: argparse.Namespace) -> QuestProbe:
    """Read one existing XRoboToolkit stream without restarting its service."""
    from handumi.tracking.pico import (
        init_xrt,
        read_pico_frame,
        tracking_packet_from_pico_frame,
    )

    xrt = None
    try:
        xrt = init_xrt()
        frame = read_pico_frame(xrt, mode=args.pico_mode)
        packet = tracking_packet_from_pico_frame(
            frame,
            sequence=0,
            receive_time_ns=time.monotonic_ns(),
        )
        controllers = {
            controller.side: controller.pose.tracking_state.value == "TRACKED"
            for controller in packet.controllers
        }
        hmd_tracked = bool(
            packet.hmd is not None and packet.hmd.tracking_state.value == "TRACKED"
        )
        body = packet.body
        streaming = bool(
            hmd_tracked
            and controllers.get("left", False)
            and controllers.get("right", False)
        )
        return QuestProbe(
            connected=True,
            streaming=streaming,
            protocol_schema=packet.schema,
            protocol_version=packet.source_schema_version,
            foreground_worn_observed=streaming,
            hmd_tracked=hmd_tracked,
            left_controller_tracked=controllers.get("left", False),
            right_controller_tracked=controllers.get("right", False),
            body_supported=None if body is None else True,
            body_enabled=None if body is None else True,
            body_active=None if body is None else body.active,
            body_calibration_state=None if body is None else body.calibration_state,
            clock_synced=False,
            source_timestamp_quality=packet.timestamps.quality.value,
            error=None
            if streaming
            else "XRoboToolkit returned incomplete HMD/controller tracking",
        )
    except (OSError, RuntimeError, SystemExit, TypeError, ValueError) as exc:
        return QuestProbe(error=f"{type(exc).__name__}: {exc}")
    finally:
        if xrt is not None:
            try:
                xrt.close()
            except Exception:  # noqa: BLE001 - optional SDK cleanup.
                pass


def _manifest_text(manifest: Mapping[str, Any], key: str) -> str | None:
    value = manifest.get(key)
    text = "" if value is None else str(value).strip()
    return text or None


def _manifest_bool(manifest: Mapping[str, Any], key: str) -> bool | None:
    value = manifest.get(key)
    return value if isinstance(value, bool) else None


def _collect_calibrations(args: argparse.Namespace) -> tuple[CalibrationProbe, ...]:
    tcp_path, _ = calibration_path_for_robot_device(
        args.robot,
        args.device,
        explicit_path=args.controller_tcp_calibration,
    )
    probes = [
        _controller_tcp_probe(tcp_path),
        *_session_calibration_probes(args),
        _body_profile_probe(args.body_profile, required=args.require_body),
    ]
    probes.extend(
        _spatial_camera_probe(
            f"camera_{index}", path, camera_names=tuple(args.camera_names or ())
        )
        for index, path in enumerate(args.camera_calibration, start=1)
    )
    calibration_path = args.feetech_calibration or user_calibration_path()
    try:
        config = load_config(args.rig_config, calibration_path)
        valid = all(
            calibration.is_complete
            and calibration.closed_ticks != calibration.open_ticks
            and calibration.max_width_mm is not None
            and calibration.max_width_mm > 0
            for calibration in (config.left, config.right)
        )
        error = (
            None
            if valid
            else "left/right endpoint or maximum-width values are incomplete"
        )
    except (OSError, SystemExit, TypeError, ValueError, yaml.YAMLError) as exc:
        valid = False
        error = str(exc)
    probes.append(
        CalibrationProbe(
            name="feetech",
            path=calibration_path,
            required=True,
            exists=calibration_path.exists(),
            valid=valid,
            sha256=_sha256(calibration_path),
            error=error,
        )
    )
    return tuple(probes)


def _controller_tcp_probe(path: Path) -> CalibrationProbe:
    exists = path.is_file()
    error = None
    valid = False
    if exists:
        try:
            calibration = load_controller_tcp_calibration(path)
            valid = bool(
                calibration.left.shape == (7,)
                and calibration.right.shape == (7,)
                and all(
                    abs(float((pose[3:] ** 2).sum()) - 1.0) < 1e-3
                    for pose in (calibration.left, calibration.right)
                )
            )
            if not valid:
                error = "controller TCP poses are not finite normalized pose7 values"
        except (OSError, SystemExit, KeyError, TypeError, ValueError) as exc:
            error = str(exc)
    else:
        error = "file does not exist"
    return CalibrationProbe(
        "controller_tcp",
        path,
        True,
        exists,
        valid,
        sha256=_sha256(path),
        error=error,
    )


def _session_calibration_probes(
    args: argparse.Namespace,
) -> tuple[CalibrationProbe, CalibrationProbe]:
    path = args.session_calibration
    if path is None:
        missing = CalibrationProbe(
            "session_table", None, True, False, False, error="not configured"
        )
        cameras = CalibrationProbe(
            "camera_spatial",
            None,
            True,
            False,
            False,
            error="session calibration not configured",
        )
        return missing, cameras
    exists = path.is_file()
    metadata = None
    error = None
    valid = False
    if exists:
        try:
            metadata = session_calibration_metadata(path)
            transform = session_table_from_device(path)
            valid = bool(
                metadata is not None
                and metadata.get("tracking_device") == args.device
                and transform.shape == (7,)
                and all(float(value) == float(value) for value in transform)
            )
            if not valid:
                error = "tracking device mismatch or invalid table_from_device pose"
        except (OSError, KeyError, TypeError, ValueError) as exc:
            error = str(exc)
    else:
        error = "file does not exist"
    session = CalibrationProbe(
        "session_table",
        path,
        True,
        exists,
        valid,
        sha256=_sha256(path),
        error=error,
    )
    spatial_path = (
        None
        if metadata is None
        else Path(str(metadata.get("spatial_calibration_path", "")))
    )
    camera_names = tuple(
        dict.fromkeys(args.camera_names or ("left_wrist", "right_wrist", "workspace"))
    )
    camera_probe = _spatial_camera_probe(
        "camera_spatial", spatial_path, camera_names=camera_names
    )
    return session, camera_probe


def _body_profile_probe(path: Path | None, *, required: bool) -> CalibrationProbe:
    if path is None:
        return CalibrationProbe(
            "body_profile", None, required, False, False, error="not configured"
        )
    exists = path.is_file()
    error = None
    valid = False
    if exists:
        try:
            BodyProfile.from_yaml(path)
            valid = True
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
            error = str(exc)
    else:
        error = "file does not exist"
    return CalibrationProbe(
        "body_profile",
        path,
        required,
        exists,
        valid,
        sha256=_sha256(path),
        error=error,
    )


def _spatial_camera_probe(
    name: str, path: Path | None, *, camera_names: tuple[str, ...]
) -> CalibrationProbe:
    if path is None:
        return CalibrationProbe(name, None, True, False, False, error="not configured")
    exists = path.is_file()
    error = None
    valid = False
    if exists:
        try:
            data = load_spatial_yaml(path)
            if data.get("kind") != "handumi_spatial_calibration":
                raise ValueError("not a HandUMI spatial calibration")
            cameras = data.get("cameras") or {}
            if not isinstance(cameras, Mapping):
                raise ValueError("spatial cameras section is not a mapping")
            missing = [camera for camera in camera_names if camera not in cameras]
            if missing:
                raise ValueError(f"missing camera intrinsics: {', '.join(missing)}")
            for camera in camera_names:
                CameraIntrinsics.from_dict(dict(cameras[camera]))
            valid = True
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
            error = str(exc)
    else:
        error = "file does not exist"
    return CalibrationProbe(
        name,
        path,
        True,
        exists,
        valid,
        sha256=_sha256(path),
        error=error,
    )


def _yaml_calibration(
    name: str, path: Path | None, *, required: bool
) -> CalibrationProbe:
    if path is None:
        return CalibrationProbe(
            name, None, required, False, False, error="not configured"
        )
    exists = path.is_file()
    valid = False
    error = None
    if exists:
        try:
            valid = isinstance(
                yaml.safe_load(path.read_text(encoding="utf-8")), Mapping
            )
            if not valid:
                error = "YAML root is not a mapping"
        except (OSError, yaml.YAMLError) as exc:
            error = str(exc)
    else:
        error = "file does not exist"
    return CalibrationProbe(
        name,
        path,
        required,
        exists,
        valid,
        sha256=_sha256(path),
        error=error,
    )


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    import hashlib

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _dependency_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions = {}
    for name in names:
        distribution = _DEPENDENCY_DISTRIBUTIONS.get(name, name)
        try:
            versions[name] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.expanduser().resolve(strict=False)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _quest_port(rig: Mapping[str, Any], name: str, default: int) -> int:
    meta = rig.get("meta_quest") or {}
    connection = meta.get("connection") or {} if isinstance(meta, Mapping) else {}
    return (
        int(connection.get(name, default))
        if isinstance(connection, Mapping)
        else default
    )


def _local_port_users(ports: Iterable[int]) -> dict[int, tuple[str, ...]]:
    users = {int(port): [] for port in ports}
    if shutil.which("ss") is None:
        return {port: () for port in users}
    try:
        result = subprocess.run(
            ["ss", "-H", "-ltnup"],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {port: () for port in users}
    for line in result.stdout.splitlines():
        for port in users:
            if re.search(rf"(?:\]|:){port}\s", line):
                owner = (
                    line.split("users:", 1)[-1].strip()
                    if "users:" in line
                    else line.strip()
                )
                users[port].append(owner)
    return {port: tuple(dict.fromkeys(owners)) for port, owners in users.items()}


def _device_handle_users(paths: Iterable[int | str]) -> dict[str, tuple[str, ...]]:
    targets = set()
    for value in paths:
        path = _camera_path(value) if isinstance(value, int) else str(value)
        targets.add(path)
        if os.path.lexists(path):
            targets.add(os.path.realpath(path))
    users: dict[str, list[str]] = {target: [] for target in targets}
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            comm = (proc / "comm").read_text(encoding="utf-8").strip()
            for fd in (proc / "fd").iterdir():
                try:
                    target = os.path.realpath(fd)
                except OSError:
                    continue
                if target in users:
                    users[target].append(f"pid={proc.name}:{comm}")
        except (OSError, PermissionError):
            continue
    return {target: tuple(dict.fromkeys(values)) for target, values in users.items()}


def _print_report(report) -> None:
    for check in report.checks:
        print(f"[{check.status.value:4}] {check.code}: {check.summary}")
        if check.action:
            print(f"       Action: {check.action}")
    counts = report.as_dict()["counts"]
    print(
        "\nPreflight "
        + ("PASSED" if report.passed else "FAILED")
        + f" — {counts['PASS']} pass, {counts['WARN']} warn, {counts['FAIL']} fail, {counts['SKIP']} skip"
    )


def _device_candidates(device_class: str) -> list[DeviceProbe]:
    patterns = (
        ("/dev/v4l/by-id/*", "/dev/video*")
        if device_class == "camera"
        else ("/dev/serial/by-id/*", "/dev/ttyACM*", "/dev/ttyUSB*")
    )
    candidates = []
    seen = set()
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            canonical = os.path.realpath(path)
            if canonical in seen or _classify_device(canonical) != device_class:
                continue
            seen.add(canonical)
            candidates.append(DeviceProbe(**_device_base(path, {})))
    return candidates


def _interactive_remap(
    camera_names: tuple[str, ...],
) -> tuple[dict[str, str], dict[str, str]]:
    cameras = _device_candidates("camera")
    serials = _device_candidates("serial")
    camera_paths = {
        name: selection
        for name in camera_names
        if (selection := _choose_device(f"camera {name}", cameras)) is not None
    }
    feetech_ports = {
        side: selection
        for side in ("left", "right")
        if (selection := _choose_device(f"Feetech {side}", serials)) is not None
    }
    return camera_paths, feetech_ports


def _choose_device(label: str, candidates: list[DeviceProbe]) -> str | None:
    print(f"\nSelect {label} (Enter keeps current mapping):")
    for index, candidate in enumerate(candidates, start=1):
        print(
            f"  {index}. {candidate.configured} -> {candidate.canonical_path} "
            f"identity={candidate.identity_token or 'unknown'} usb={candidate.usb_path or 'unknown'}"
        )
    answer = input("Selection: ").strip()
    if not answer:
        return None
    try:
        candidate = candidates[int(answer) - 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(f"Invalid selection for {label}: {answer}") from exc
    return str(candidate.configured)


def _confirm(prompt: str) -> bool:
    return input(prompt).strip().lower() in {"y", "yes"}


if __name__ == "__main__":
    main()
