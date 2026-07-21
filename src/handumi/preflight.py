"""Pure release-grade hardware preflight models and evaluation.

The system-facing collector lives in :mod:`handumi.scripts.preflight`.  Keeping
the evaluation here deterministic makes missing, moved, duplicated,
misclassified, busy, or partially connected hardware testable without opening
real devices.  A preflight never creates a dataset.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

import yaml


class CheckStatus(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass(frozen=True)
class PreflightCheck:
    code: str
    status: CheckStatus
    summary: str
    action: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "status": self.status.value,
            "summary": self.summary,
            "action": self.action,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class DeviceProbe:
    configured: int | str
    canonical_path: str | None
    device_class: str
    available: bool
    readable: bool = False
    writable: bool = False
    busy_by: tuple[str, ...] = ()
    serial: str | None = None
    usb_path: str | None = None
    error: str | None = None

    @property
    def identity(self) -> str | None:
        return self.serial or self.usb_path or self.canonical_path

    @property
    def identity_token(self) -> str | None:
        if self.identity is None:
            return None
        return hashlib.sha256(self.identity.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class CameraProbe(DeviceProbe):
    frame_count: int = 0
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    luminance_mean: float | None = None
    luminance_std: float | None = None
    frame_delta_mean: float | None = None


@dataclass(frozen=True)
class FeetechProbe(DeviceProbe):
    configured_servo_id: int = 0
    detected_servo_ids: tuple[int, ...] = ()
    positions_by_id: Mapping[int, tuple[int, ...]] = field(default_factory=dict)
    calibration_range_valid: bool | None = None


@dataclass(frozen=True)
class QuestProbe:
    connected: bool = False
    streaming: bool = False
    package_identifier: str | None = None
    version_name: str | None = None
    build_id: str | None = None
    source_commit: str | None = None
    protocol_schema: str | None = None
    protocol_version: int | None = None
    manifest_schema: str | None = None
    foreground_worn_observed: bool = False
    hmd_tracked: bool = False
    left_controller_tracked: bool = False
    right_controller_tracked: bool = False
    body_supported: bool | None = None
    body_enabled: bool | None = None
    body_active: bool | None = None
    body_calibration_state: str | None = None
    clock_synced: bool = False
    clock_rtt_ms: float | None = None
    source_timestamp_quality: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class CalibrationProbe:
    name: str
    path: Path | None
    required: bool
    exists: bool
    valid: bool
    sha256: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class PreflightInventory:
    cameras: Mapping[str, CameraProbe]
    feetech: Mapping[str, FeetechProbe]
    quest: QuestProbe
    calibrations: tuple[CalibrationProbe, ...]
    local_port_users: Mapping[int, tuple[str, ...]]
    output_writable: bool
    output_probe_path: Path
    disk_free_bytes: int
    dependencies: Mapping[str, str | None]
    python_version: tuple[int, int, int]
    platform: str
    collection_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreflightRequest:
    device: str = "meta"
    require_body: bool = False
    expected_package: str | None = None
    expected_version: str | None = None
    expected_build_id: str | None = None
    expected_protocol_schema: str = "tracking_packet_v2"
    expected_protocol_version: int = 2
    require_clock_sync: bool = True
    camera_names: tuple[str, ...] = (
        "left_wrist",
        "right_wrist",
        "workspace",
    )
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    feetech_sides: tuple[str, ...] = ("left", "right")
    local_ports: tuple[int, ...] = (65432, 42000, 8003)
    min_disk_bytes: int = 10 * 1024**3
    required_dependencies: tuple[str, ...] = (
        "numpy",
        "cv2",
        "rerun_sdk",
        "yaml",
    )


@dataclass(frozen=True)
class PreflightReport:
    checks: tuple[PreflightCheck, ...]

    @property
    def passed(self) -> bool:
        return not any(check.status is CheckStatus.FAIL for check in self.checks)

    @property
    def failures(self) -> tuple[PreflightCheck, ...]:
        return tuple(check for check in self.checks if check.status is CheckStatus.FAIL)

    def as_dict(self) -> dict[str, Any]:
        counts = {
            status.value: sum(check.status is status for check in self.checks)
            for status in CheckStatus
        }
        return {
            "schema": "handumi_preflight_report_v1",
            "passed": self.passed,
            "counts": counts,
            "checks": [check.as_dict() for check in self.checks],
        }


def evaluate_preflight(
    request: PreflightRequest, inventory: PreflightInventory
) -> PreflightReport:
    """Evaluate one collected inventory without opening or mutating hardware."""
    checks: list[PreflightCheck] = []
    checks.extend(_runtime_checks(request, inventory))
    checks.extend(_quest_checks(request, inventory.quest))
    checks.extend(_camera_checks(request, inventory.cameras))
    checks.extend(_feetech_checks(request, inventory.feetech))
    checks.extend(_calibration_checks(inventory.calibrations))
    checks.extend(_storage_checks(request, inventory))
    checks.extend(_port_checks(request, inventory.local_port_users))
    for index, error in enumerate(inventory.collection_errors, start=1):
        checks.append(
            PreflightCheck(
                f"COLLECT-{index:03d}",
                CheckStatus.FAIL,
                error,
                "Resolve the collector error and rerun the read-only preflight.",
            )
        )
    return PreflightReport(tuple(checks))


def _runtime_checks(
    request: PreflightRequest, inventory: PreflightInventory
) -> list[PreflightCheck]:
    checks = []
    supported_python = inventory.python_version[:2] == (3, 12)
    checks.append(
        PreflightCheck(
            "RUNTIME-PYTHON",
            CheckStatus.PASS if supported_python else CheckStatus.FAIL,
            "Python runtime is supported."
            if supported_python
            else f"Unsupported Python {'.'.join(map(str, inventory.python_version))}.",
            "Use the locked Python 3.12 environment." if not supported_python else "",
            {"version": ".".join(map(str, inventory.python_version))},
        )
    )
    linux = inventory.platform.lower().startswith("linux")
    checks.append(
        PreflightCheck(
            "RUNTIME-PLATFORM",
            CheckStatus.PASS if linux else CheckStatus.FAIL,
            f"Runtime platform: {inventory.platform}.",
            "Recording hardware support currently requires Linux." if not linux else "",
        )
    )
    for name in request.required_dependencies:
        version = inventory.dependencies.get(name)
        checks.append(
            PreflightCheck(
                f"DEPENDENCY-{name.upper().replace('_', '-')}",
                CheckStatus.PASS if version is not None else CheckStatus.FAIL,
                f"Optional/runtime dependency {name} is available ({version})."
                if version is not None
                else f"Required dependency {name} is unavailable.",
                "Install the documented locked extras for this capture profile."
                if version is None
                else "",
            )
        )
    return checks


def _quest_checks(request: PreflightRequest, quest: QuestProbe) -> list[PreflightCheck]:
    checks = [
        PreflightCheck(
            "QUEST-CONNECTION",
            CheckStatus.PASS
            if quest.connected and quest.streaming
            else CheckStatus.FAIL,
            "Quest tracking transport is connected and producing frames."
            if quest.connected and quest.streaming
            else f"Quest tracking transport is unavailable: {quest.error or 'no live frames'}.",
            "Wear/foreground the expected app, verify the trusted-LAN IP, and check TCP 65432/UDP 42000."
            if not (quest.connected and quest.streaming)
            else "",
        )
    ]
    if request.expected_package is not None:
        package_ok = quest.package_identifier == request.expected_package
        checks.append(
            PreflightCheck(
                "QUEST-PACKAGE",
                CheckStatus.PASS if package_ok else CheckStatus.FAIL,
                f"Quest package is {quest.package_identifier or 'unreported'}."
                if package_ok
                else "Quest package identity is missing or does not match the expected build profile.",
                f"Install and foreground {request.expected_package}; do not grant BODY_TRACKING with adb shell pm grant."
                if not package_ok
                else "",
                {
                    "expected": request.expected_package,
                    "observed": quest.package_identifier,
                },
            )
        )
    if request.expected_version is not None:
        version_ok = quest.version_name == request.expected_version
        checks.append(
            PreflightCheck(
                "QUEST-VERSION",
                CheckStatus.PASS if version_ok else CheckStatus.FAIL,
                f"Quest app version is {quest.version_name or 'unreported'}.",
                f"Install the expected {request.expected_version} artifact."
                if not version_ok
                else "",
            )
        )
    if request.expected_build_id is not None:
        build_ok = quest.build_id == request.expected_build_id
        checks.append(
            PreflightCheck(
                "QUEST-BUILD",
                CheckStatus.PASS if build_ok else CheckStatus.FAIL,
                f"Quest build ID is {quest.build_id or 'unreported'}.",
                f"Install the expected build {request.expected_build_id}."
                if not build_ok
                else "",
            )
        )
    protocol_ok = (
        quest.protocol_schema == request.expected_protocol_schema
        and quest.protocol_version == request.expected_protocol_version
    )
    checks.append(
        PreflightCheck(
            "QUEST-PROTOCOL",
            CheckStatus.PASS if protocol_ok else CheckStatus.FAIL,
            (
                f"Quest protocol is {quest.protocol_schema} v{quest.protocol_version}."
                if protocol_ok
                else "Quest protocol is absent or incompatible with this recorder."
            ),
            "Use the documented matching source/APK pair." if not protocol_ok else "",
        )
    )
    tracked = (
        quest.foreground_worn_observed
        and quest.hmd_tracked
        and quest.left_controller_tracked
        and quest.right_controller_tracked
    )
    checks.append(
        PreflightCheck(
            "QUEST-FOREGROUND-TRACKING",
            CheckStatus.PASS if tracked else CheckStatus.FAIL,
            "Worn/foreground HMD and both controllers are producing tracked poses."
            if tracked
            else "Foreground/worn-headset and complete HMD/controller tracking were not observed.",
            "Wear the headset, keep the app foreground, and wake/track both controllers."
            if not tracked
            else "",
        )
    )
    clock_diagnostic_available = quest.source_timestamp_quality is not None
    clock_ok = quest.clock_synced or (
        not request.require_clock_sync and clock_diagnostic_available
    )
    checks.append(
        PreflightCheck(
            "QUEST-CLOCK",
            CheckStatus.PASS if clock_ok else CheckStatus.FAIL,
            f"Quest clock exchange is active (RTT {quest.clock_rtt_ms:.2f} ms)."
            if quest.clock_synced and quest.clock_rtt_ms is not None
            else f"Source timing diagnostic is {quest.source_timestamp_quality}; no synchronization claim is made."
            if clock_ok
            else "Quest UDP clock exchange is not synchronized.",
            "Check UDP 42000 and trusted-LAN firewall/routing. Body source timing may still be diagnostic-only."
            if not clock_ok
            else "",
            {"source_timestamp_quality": quest.source_timestamp_quality},
        )
    )
    if request.require_body:
        body_ok = bool(
            quest.body_supported
            and quest.body_enabled
            and quest.body_active
            and (quest.body_calibration_state or "").lower() == "valid"
        )
        checks.append(
            PreflightCheck(
                "QUEST-BODY",
                CheckStatus.PASS if body_ok else CheckStatus.FAIL,
                "Body tracking is supported, enabled, active, and calibration state is Valid."
                if body_ok
                else "Body tracking is inactive, unsupported, disabled, or not validly calibrated.",
                "Use the Body Probe build, accept the install-time BODY_TRACKING permission, wear the headset, and complete Meta calibration."
                if not body_ok
                else "",
                {
                    "supported": quest.body_supported,
                    "enabled": quest.body_enabled,
                    "active": quest.body_active,
                    "calibration_state": quest.body_calibration_state,
                },
            )
        )
    return checks


def _camera_checks(
    request: PreflightRequest, cameras: Mapping[str, CameraProbe]
) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    identities: dict[str, list[str]] = {}
    usb_paths: dict[str, list[str]] = {}
    for name in request.camera_names:
        camera = cameras.get(name)
        if camera is None or not camera.available:
            checks.append(
                PreflightCheck(
                    f"CAMERA-{name.upper()}-AVAILABLE",
                    CheckStatus.FAIL,
                    f"Camera {name} is missing.",
                    "Reconnect it and remap by stable camera identity, not a stale /dev path.",
                )
            )
            continue
        class_ok = camera.device_class == "camera"
        checks.append(
            PreflightCheck(
                f"CAMERA-{name.upper()}-CLASS",
                CheckStatus.PASS if class_ok else CheckStatus.FAIL,
                f"{name} resolves to device class {camera.device_class}."
                if class_ok
                else f"{name} path exists but resolves to {camera.device_class}, not a camera.",
                "Remap after USB re-enumeration; never trust path existence alone."
                if not class_ok
                else "",
                {
                    "configured": str(camera.configured),
                    "canonical": camera.canonical_path,
                    "identity_token": camera.identity_token,
                    "usb_path": camera.usb_path,
                },
            )
        )
        if camera.identity is not None:
            identities.setdefault(camera.identity, []).append(name)
        if camera.usb_path is not None:
            usb_paths.setdefault(camera.usb_path, []).append(name)
        access_ok = camera.readable and not camera.busy_by
        checks.append(
            PreflightCheck(
                f"CAMERA-{name.upper()}-ACCESS",
                CheckStatus.PASS if access_ok else CheckStatus.FAIL,
                f"Camera {name} is available to this process."
                if access_ok
                else f"Camera {name} is unreadable or already held by another process.",
                "Close the listed process or repair camera permissions."
                if not access_ok
                else "",
                {"busy_by": list(camera.busy_by), "error": camera.error},
            )
        )
        mode_ok = (
            camera.frame_count >= 2
            and camera.width == request.camera_width
            and camera.height == request.camera_height
            and camera.fps is not None
            and camera.fps >= request.camera_fps * 0.9
        )
        checks.append(
            PreflightCheck(
                f"CAMERA-{name.upper()}-STREAM",
                CheckStatus.PASS if mode_ok else CheckStatus.FAIL,
                (
                    f"Camera {name} produced {camera.frame_count} frames at "
                    f"{camera.width}x{camera.height}, {camera.fps} FPS."
                    if mode_ok
                    else f"Camera {name} did not produce the requested frame mode."
                ),
                f"Verify {request.camera_width}x{request.camera_height} at {request.camera_fps} FPS and USB bandwidth."
                if not mode_ok
                else "",
            )
        )
        content_ok = (
            camera.luminance_mean is not None
            and camera.luminance_std is not None
            and (camera.luminance_mean >= 2.0 or camera.luminance_std >= 2.0)
        )
        checks.append(
            PreflightCheck(
                f"CAMERA-{name.upper()}-CONTENT",
                CheckStatus.PASS if content_ok else CheckStatus.FAIL,
                (
                    f"Camera {name} produced non-black image content."
                    if content_ok
                    else f"Camera {name} produced only black or unusable frames."
                ),
                (
                    "Remove a lens cover, provide light, check exposure, and verify "
                    "the camera is aimed at the intended workspace."
                    if not content_ok
                    else ""
                ),
                {
                    "luminance_mean": camera.luminance_mean,
                    "luminance_std": camera.luminance_std,
                    "frame_delta_mean": camera.frame_delta_mean,
                },
            )
        )
    for identity, names in identities.items():
        if len(names) > 1:
            assigned = [cameras[name] for name in names]
            distinct_canonical = len(
                {camera.canonical_path for camera in assigned if camera.canonical_path}
            ) == len(assigned)
            distinct_topology = len(
                {camera.usb_path for camera in assigned if camera.usb_path}
            ) == len(assigned)
            safely_disambiguated = distinct_canonical and distinct_topology
            checks.append(
                PreflightCheck(
                    "CAMERA-DUPLICATE-IDENTITY",
                    CheckStatus.WARN if safely_disambiguated else CheckStatus.FAIL,
                    (
                        f"Camera assignments {', '.join(names)} report the same "
                        "vendor identity but resolve to distinct canonical devices "
                        "and USB paths."
                        if safely_disambiguated
                        else f"Camera assignments {', '.join(names)} cannot be "
                        "distinguished by identity and topology."
                    ),
                    (
                        "Keep these roles pinned by USB by-path; reconnect one at a "
                        "time before accepting any topology change."
                        if safely_disambiguated
                        else "Assign distinct left/right/workspace devices or remap "
                        "them by unique USB topology."
                    ),
                    {
                        "identity_token": hashlib.sha256(identity.encode()).hexdigest()[
                            :12
                        ],
                        "canonical_paths": {
                            name: cameras[name].canonical_path for name in names
                        },
                        "usb_paths": {name: cameras[name].usb_path for name in names},
                    },
                )
            )
    missing_topology = [
        name
        for name in request.camera_names
        if cameras.get(name) is not None and cameras[name].usb_path is None
    ]
    duplicate_topology = {
        path: names for path, names in usb_paths.items() if len(names) > 1
    }
    topology_ok = not missing_topology and not duplicate_topology
    checks.append(
        PreflightCheck(
            "CAMERA-USB-TOPOLOGY",
            CheckStatus.PASS if topology_ok else CheckStatus.FAIL,
            "Every camera has a distinct recorded USB by-path topology."
            if topology_ok
            else "Camera USB topology is missing or duplicates a physical path.",
            "Reconnect one camera at a time and remap using stable by-id plus by-path evidence."
            if not topology_ok
            else "",
            {
                "paths": {
                    name: cameras[name].usb_path
                    for name in request.camera_names
                    if name in cameras
                },
                "missing": missing_topology,
                "duplicates": duplicate_topology,
            },
        )
    )
    return checks


def _feetech_checks(
    request: PreflightRequest, feetech: Mapping[str, FeetechProbe]
) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    identities: dict[str, list[str]] = {}
    usb_paths: dict[str, list[str]] = {}
    for side in request.feetech_sides:
        probe = feetech.get(side)
        if probe is None or not probe.available:
            checks.append(
                PreflightCheck(
                    f"FEETECH-{side.upper()}-AVAILABLE",
                    CheckStatus.FAIL,
                    f"{side} Feetech adapter is missing.",
                    "Reconnect and remap the adapter by serial identity and side.",
                )
            )
            continue
        class_ok = probe.device_class == "serial"
        checks.append(
            PreflightCheck(
                f"FEETECH-{side.upper()}-CLASS",
                CheckStatus.PASS if class_ok else CheckStatus.FAIL,
                f"{side} Feetech path resolves to serial hardware."
                if class_ok
                else f"{side} Feetech path resolves to {probe.device_class}, not serial hardware.",
                "Remap the side; a camera/unknown path must never be opened as Feetech."
                if not class_ok
                else "",
                {
                    "configured": str(probe.configured),
                    "canonical": probe.canonical_path,
                    "identity_token": probe.identity_token,
                    "usb_path": probe.usb_path,
                },
            )
        )
        if probe.identity is not None:
            identities.setdefault(probe.identity, []).append(side)
        if probe.usb_path is not None:
            usb_paths.setdefault(probe.usb_path, []).append(side)
        access_ok = probe.readable and probe.writable and not probe.busy_by
        checks.append(
            PreflightCheck(
                f"FEETECH-{side.upper()}-ACCESS",
                CheckStatus.PASS if access_ok else CheckStatus.FAIL,
                f"{side} Feetech adapter is readable/writable and not busy."
                if access_ok
                else f"{side} Feetech adapter permission or handle conflict detected.",
                "Close conflicting processes or repair dialout/udev permissions."
                if not access_ok
                else "",
                {"busy_by": list(probe.busy_by), "error": probe.error},
            )
        )
        configured_id = probe.configured_servo_id
        servo_ok = configured_id in probe.detected_servo_ids
        checks.append(
            PreflightCheck(
                f"FEETECH-{side.upper()}-SERVO",
                CheckStatus.PASS if servo_ok else CheckStatus.FAIL,
                f"{side} servo ID {configured_id} responds."
                if servo_ok
                else f"Configured {side} servo ID {configured_id} did not respond.",
                "Verify side mapping, baudrate, power, cable, and servo ID."
                if not servo_ok
                else "",
                {"detected_ids": list(probe.detected_servo_ids)},
            )
        )
        positions = tuple(probe.positions_by_id.get(configured_id, ()))
        moved = len(positions) >= 2 and len(set(positions)) >= 2
        checks.append(
            PreflightCheck(
                f"FEETECH-{side.upper()}-MOTION",
                CheckStatus.PASS if moved else CheckStatus.WARN,
                f"{side} encoder motion observed: {positions}."
                if moved
                else f"{side} encoder responded but motion was not observed during the probe window.",
                "Move the gripper through a small safe stroke while rerunning preflight; full endpoint QA remains a separate hardware gate."
                if not moved
                else "",
            )
        )
        range_status = (
            CheckStatus.PASS
            if probe.calibration_range_valid is True
            else CheckStatus.FAIL
            if probe.calibration_range_valid is False
            else CheckStatus.WARN
        )
        checks.append(
            PreflightCheck(
                f"FEETECH-{side.upper()}-RANGE",
                range_status,
                f"{side} live encoder samples are compatible with cached endpoints."
                if range_status is CheckStatus.PASS
                else f"{side} live encoder range could not be verified against cached endpoints."
                if range_status is CheckStatus.WARN
                else f"{side} live encoder samples conflict with cached endpoints.",
                "Re-home only if needed, then repeat physical closed/open calibration; do not record with stale endpoints."
                if range_status is not CheckStatus.PASS
                else "",
            )
        )
    for identity, sides in identities.items():
        if len(sides) > 1:
            checks.append(
                PreflightCheck(
                    "FEETECH-DUPLICATE-IDENTITY",
                    CheckStatus.FAIL,
                    f"Feetech sides {', '.join(sides)} resolve to one adapter identity.",
                    "Assign distinct adapters/sides or explicitly configure a supported shared bus.",
                    {
                        "identity_token": hashlib.sha256(identity.encode()).hexdigest()[
                            :12
                        ]
                    },
                )
            )
    missing_topology = [
        side
        for side in request.feetech_sides
        if feetech.get(side) is not None and feetech[side].usb_path is None
    ]
    duplicate_topology = {
        path: sides for path, sides in usb_paths.items() if len(sides) > 1
    }
    topology_ok = not missing_topology and not duplicate_topology
    checks.append(
        PreflightCheck(
            "FEETECH-USB-TOPOLOGY",
            CheckStatus.PASS if topology_ok else CheckStatus.FAIL,
            "Both Feetech sides have distinct recorded USB by-path topology."
            if topology_ok
            else "Feetech USB topology is missing or maps both sides to one path.",
            "Reconnect one adapter at a time and confirm side, serial identity, and by-path."
            if not topology_ok
            else "",
            {
                "paths": {
                    side: feetech[side].usb_path
                    for side in request.feetech_sides
                    if side in feetech
                },
                "missing": missing_topology,
                "duplicates": duplicate_topology,
            },
        )
    )
    return checks


def _calibration_checks(
    calibrations: tuple[CalibrationProbe, ...],
) -> list[PreflightCheck]:
    checks = []
    for calibration in calibrations:
        ok = calibration.exists and calibration.valid
        status = (
            CheckStatus.PASS
            if ok
            else (CheckStatus.FAIL if calibration.required else CheckStatus.WARN)
        )
        checks.append(
            PreflightCheck(
                f"CALIBRATION-{calibration.name.upper().replace('_', '-')}",
                status,
                f"Calibration {calibration.name} is present and valid."
                if ok
                else f"Calibration {calibration.name} is missing or invalid.",
                "Run the documented calibration and rerun preflight. Missing measurements are never fabricated."
                if not ok
                else "",
                {
                    "path": str(calibration.path) if calibration.path else None,
                    "sha256": calibration.sha256,
                    "error": calibration.error,
                },
            )
        )
    return checks


def _storage_checks(
    request: PreflightRequest, inventory: PreflightInventory
) -> list[PreflightCheck]:
    writable = inventory.output_writable
    enough = inventory.disk_free_bytes >= request.min_disk_bytes
    return [
        PreflightCheck(
            "OUTPUT-WRITABLE",
            CheckStatus.PASS if writable else CheckStatus.FAIL,
            f"Output parent is writable: {inventory.output_probe_path}."
            if writable
            else f"Output parent is not writable: {inventory.output_probe_path}.",
            "Choose a writable output parent. Dry-run does not create a dataset."
            if not writable
            else "",
        ),
        PreflightCheck(
            "OUTPUT-DISK",
            CheckStatus.PASS if enough else CheckStatus.FAIL,
            f"Free disk: {inventory.disk_free_bytes / 1024**3:.2f} GiB.",
            f"Free at least {request.min_disk_bytes / 1024**3:.2f} GiB before recording."
            if not enough
            else "",
        ),
    ]


def _port_checks(
    request: PreflightRequest, users: Mapping[int, tuple[str, ...]]
) -> list[PreflightCheck]:
    checks = []
    for port in request.local_ports:
        owners = users.get(port, ())
        checks.append(
            PreflightCheck(
                f"PORT-{port}",
                CheckStatus.PASS if not owners else CheckStatus.FAIL,
                f"Port {port} has no conflicting local listener."
                if not owners
                else f"Port {port} is already in use by {', '.join(owners)}.",
                "Stop the stale Quest/Rerun/Viser process or select another viewer port."
                if owners
                else "",
            )
        )
    return checks


def atomic_update_rig(
    path: Path,
    *,
    camera_paths: Mapping[str, int | str] | None = None,
    feetech_ports: Mapping[str, str] | None = None,
) -> None:
    """Atomically update only local camera/Feetech assignments.

    The caller must obtain explicit operator confirmation. The temporary file
    is fsynced and replaced in the same directory; a failed write leaves the
    original rig untouched.
    """
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Rig config {path} must contain a mapping.")
    if camera_paths:
        cameras = data.setdefault("cameras", {})
        if not isinstance(cameras, dict):
            raise ValueError("Rig cameras section must be a mapping.")
        for name, value in camera_paths.items():
            entry = cameras.setdefault(name, {})
            if not isinstance(entry, dict):
                raise ValueError(f"Rig camera {name} entry must be a mapping.")
            entry["index_or_path"] = value
    if feetech_ports:
        feetech = data.setdefault("feetech", {})
        if not isinstance(feetech, dict):
            raise ValueError("Rig feetech section must be a mapping.")
        for side, value in feetech_ports.items():
            entry = feetech.setdefault(side, {})
            if not isinstance(entry, dict):
                raise ValueError(f"Rig Feetech {side} entry must be a mapping.")
            entry["port"] = value

    mode = path.stat().st_mode & 0o777
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            yaml.safe_dump(data, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp_path, mode)
        os.replace(temp_path, path)
        temp_path = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


__all__ = [
    "CalibrationProbe",
    "CameraProbe",
    "CheckStatus",
    "DeviceProbe",
    "FeetechProbe",
    "PreflightCheck",
    "PreflightInventory",
    "PreflightReport",
    "PreflightRequest",
    "QuestProbe",
    "atomic_update_rig",
    "evaluate_preflight",
]
