#!/usr/bin/env python3
"""Replay a HandUMI LeRobot episode in simulation with YAML robot configs."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from handumi.calibration.control_tcp import (
    DEFAULT_DEVICE as DEFAULT_CONTROLLER_DEVICE,
    ControllerTcpCalibration,
    apply_controller_tcp_calibration,
    calibration_path_for_device,
    controller_tcp_calibration_sha256,
    controller_tcp_calibration_from_metadata,
    is_identity_bound_controller_tcp_metadata,
    load_controller_tcp_calibration,
)
from handumi.dataset import (
    DatasetRef,
    ensure_metadata,
    handumi_metadata,
    open_dataset,
    validate_raw_state_metadata,
)
from handumi.dataset.raw import (
    HANDUMI_RAW_STATE_SIZE,
    LEFT_GRIPPER_INDEX,
    LEFT_POSE_SLICE,
    RIGHT_GRIPPER_INDEX,
    RIGHT_POSE_SLICE,
)
from handumi.retargeting.handumi_to_robot import (
    VR_TO_ROBOT,
    absolute_table_robot_target_pose7,
    local_frame_adapter,
    local_relative_robot_target_pose7,
    orientation_only_pose_adapter,
    raw_state_pose7_pair,
    raw_state_robot_target_pose7,
    retarget_anchors_from_raw_state,
)
from handumi.robots.kinematics import optimization_score_from_errors, pose_error_arrays
from handumi.robots.registry import EMBODIMENT_NAMES, load_embodiment, load_robot_config
from handumi.robots.utils import pose_mul, quat_normalize

DEFAULT_REPO_ID = "NONHUMAN-RESEARCH/handumi-dataset-v2"
DEFAULT_OUT_DIR = Path("outputs/replay_in_sim")
DEFAULT_DEPLOYMENT_CALIBRATION_DIR = Path("configs/calibration")
GRIPPER_NORMALIZED_KEYS = (
    "observation.feetech.left_normalized",
    "observation.feetech.right_normalized",
)


@dataclass(frozen=True)
class TcpCalibrationSelection:
    calibration: ControllerTcpCalibration
    source: str
    source_robot: str
    source_gripper: str
    tracking_device: str
    controller_mount: str
    trusted_dataset_snapshot: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay a raw HandUMI LeRobot episode through bimanual IK."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Local dataset root. Defaults to outputs/datasets/<repo-id suffix>.",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("-e", "--episode", type=int, default=0)
    parser.add_argument("--robot", choices=EMBODIMENT_NAMES, default="piper")
    parser.add_argument(
        "--source",
        choices=("observation.state", "action"),
        default="observation.state",
        help="Raw 16D LeRobot feature to replay.",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--retarget-mode",
        choices=("auto", "local-relative", "anchored", "absolute-table"),
        default="auto",
        help=(
            "auto selects absolute-table for datasets recorded in a calibrated table "
            "workspace, otherwise local-relative. local-relative replays frame-to-frame "
            "TCP SE(3) deltas in robot EE space. anchored preserves the older home + "
            "position-delta mode. absolute-table applies one shared table->robot "
            "transform to both TCPs."
        ),
    )
    parser.add_argument(
        "--compose-source",
        choices=("commanded", "achieved"),
        default="commanded",
        help=(
            "For --retarget-mode local-relative, compose each adapted delta on "
            "the previous commanded target or previous achieved FK pose."
        ),
    )
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=1.0,
        help="Scale local-relative translation deltas after frame adaptation.",
    )
    parser.add_argument(
        "--controller-device",
        choices=("pico", "meta"),
        default=None,
        help=(
            "Controller device. Defaults to handumi.recording_device in dataset "
            f"metadata, then {DEFAULT_CONTROLLER_DEVICE!r} for legacy datasets."
        ),
    )
    parser.add_argument(
        "--controller-tcp-calibration",
        type=Path,
        default=None,
        help=(
            "Explicit YAML with controller->HandUMI TCP transforms. Overrides both "
            "the robot/device configured calibration and dataset metadata."
        ),
    )
    parser.add_argument(
        "--use-dataset-tcp-calibration",
        action="store_true",
        help=(
            "Force the controller->TCP snapshot embedded in the dataset, including "
            "unidentified legacy snapshots, instead of normal precedence."
        ),
    )
    parser.add_argument(
        "--raw-controller-debug",
        action="store_true",
        help="Replay raw PICO controller poses without controller->TCP calibration.",
    )
    parser.add_argument(
        "--deployment-calibration",
        type=Path,
        default=None,
        help=(
            "YAML containing robot_from_table for --retarget-mode absolute-table. "
            "Defaults to configs/calibration/{robot}_table.yaml."
        ),
    )
    parser.add_argument(
        "--absolute-orientation",
        choices=("relative-start", "table-absolute"),
        default="relative-start",
        help=(
            "Orientation policy for absolute-table. relative-start aligns each "
            "HandUMI tool frame to the robot home TCP at frame 0 while preserving "
            "all subsequent wrist rotations. table-absolute requires matching, "
            "externally calibrated HandUMI and robot TCP frame conventions."
        ),
    )
    parser.add_argument(
        "--initial-solve-iterations",
        type=int,
        default=12,
        help=(
            "Unrestricted IK iterations used to prepare the absolute-table start "
            "configuration before frame 0."
        ),
    )
    parser.add_argument(
        "--initial-position-tolerance-m",
        type=float,
        default=0.01,
        help="Required maximum TCP position error before starting replay.",
    )
    parser.add_argument(
        "--gripper-max-width-m",
        type=float,
        default=None,
        help=(
            "Fallback physical opening mapped to fully open. Defaults to the robot "
            "configuration and is used only when recorded Feetech "
            "normalized fields are unavailable."
        ),
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--scene",
        default=None,
        help=(
            "Render assets/scenes/<name>/scene.xml in the calibrated table frame. "
            "This provides initial task context; recorded object motion is unavailable."
        ),
    )
    parser.add_argument(
        "--max-ik-position-error-m",
        type=float,
        default=0.03,
        help="Maximum acceptable per-frame TCP position error.",
    )
    parser.add_argument(
        "--table-clearance-warning-m",
        type=float,
        default=0.10,
        help=(
            "Warn when a table-calibrated episode never places either HandUMI TCP "
            "closer than this height to table z=0."
        ),
    )
    parser.add_argument(
        "--max-ik-rotation-error-deg",
        type=float,
        default=45.0,
        help="Maximum acceptable per-frame TCP rotation error.",
    )
    parser.add_argument(
        "--strict-ik",
        action="store_true",
        help="Fail instead of warning when an IK error threshold is exceeded.",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("-o", "--output", type=Path, default=None)
    return parser


def load_episode_states(
    args: argparse.Namespace,
) -> tuple[np.ndarray, float, dict[str, object], np.ndarray | None]:
    """Load one episode from the current 16D HandUMI raw layout."""
    ref = DatasetRef.from_repo_id(
        args.repo_id,
        root=args.dataset_root,
        revision=args.revision,
    )
    source_info = ensure_metadata(ref)
    validate_raw_state_metadata(source_info)
    dataset = open_dataset(ref, episode=args.episode)
    fps = float(getattr(dataset, "fps", 30) or 30)
    states: list[np.ndarray] = []
    normalized_grippers: list[np.ndarray] = []
    has_normalized_grippers = True

    for item in dataset:
        if args.source not in item:
            raise ValueError(f"Dataset item has no {args.source!r} feature.")
        state = np.asarray(item[args.source], dtype=np.float32).reshape(-1)
        if len(state) != HANDUMI_RAW_STATE_SIZE:
            raise ValueError(
                f"Expected HandUMI state length {HANDUMI_RAW_STATE_SIZE} "
                f"(poses + grippers) in {args.source!r}, got {len(state)}."
            )
        states.append(state)
        if all(key in item for key in GRIPPER_NORMALIZED_KEYS):
            normalized_grippers.append(
                np.asarray(
                    [
                        np.asarray(item[key], dtype=np.float32).reshape(-1)[0]
                        for key in GRIPPER_NORMALIZED_KEYS
                    ],
                    dtype=np.float32,
                )
            )
        else:
            has_normalized_grippers = False

    if not states:
        raise ValueError(f"Episode {args.episode} is empty.")
    normalized = (
        np.clip(np.stack(normalized_grippers, axis=0), 0.0, 1.0)
        if has_normalized_grippers and len(normalized_grippers) == len(states)
        else None
    )
    return np.stack(states, axis=0), fps, source_info, normalized


def _resolve_gripper_openings(
    states: np.ndarray,
    recorded_normalized: np.ndarray | None,
    *,
    max_width_m: float,
) -> tuple[np.ndarray | None, str]:
    """Return per-frame 0-1 gripper openings and their source description."""
    if recorded_normalized is not None:
        return np.clip(recorded_normalized, 0.0, 1.0), "recorded Feetech normalized"
    if states.ndim != 2 or states.shape[1] != HANDUMI_RAW_STATE_SIZE:
        raise ValueError(
            f"Expected HandUMI states shape (T, {HANDUMI_RAW_STATE_SIZE}), "
            f"got {states.shape}."
        )
    if max_width_m <= 0:
        raise SystemExit("--gripper-max-width-m must be greater than zero.")
    widths_m = states[:, [LEFT_GRIPPER_INDEX, RIGHT_GRIPPER_INDEX]]
    return np.clip(widths_m / float(max_width_m), 0.0, 1.0), "state widths in meters"


def _resolved_controller_device(
    args: argparse.Namespace,
    source_info: dict[str, object],
) -> str:
    if args.controller_device is not None:
        snapshot = _metadata_tcp_snapshot(source_info)
        if snapshot is not None and _is_trusted_tcp_snapshot(snapshot):
            recorded_device = str(snapshot["tracking_device"])
            if str(args.controller_device) != recorded_device:
                raise SystemExit(
                    f"--controller-device {args.controller_device!r} conflicts with "
                    f"the identity-bound dataset snapshot ({recorded_device!r})."
                )
        return str(args.controller_device)
    recorded = handumi_metadata(source_info).get("recording_device")
    if recorded in ("pico", "meta"):
        return str(recorded)
    return DEFAULT_CONTROLLER_DEVICE


def _resolved_retarget_mode(
    args: argparse.Namespace,
    source_info: dict[str, object],
) -> str:
    """Choose geometry-preserving replay when the dataset has a table frame."""
    requested = str(args.retarget_mode)
    if requested != "auto":
        return requested
    workspace = handumi_metadata(source_info).get("tracking_workspace")
    return "absolute-table" if workspace == "table" else "local-relative"


def _metadata_tcp_calibration(
    source_info: dict[str, object],
) -> tuple[ControllerTcpCalibration, str] | None:
    snapshot = _metadata_tcp_snapshot(source_info)
    if not isinstance(snapshot, dict) or snapshot.get("applied_to_state") is True:
        return None
    calibration = controller_tcp_calibration_from_metadata(snapshot)
    sha256 = str(snapshot.get("sha256", "unknown"))
    identity = _metadata_tcp_identity(source_info)
    if _is_trusted_tcp_snapshot(snapshot):
        source = (
            "dataset robot-tool snapshot "
            f"{identity['source_robot']}/{identity['tracking_device']} "
            f"gripper={identity['source_gripper']} "
            f"mount={identity['controller_mount']} sha256={sha256}"
        )
    else:
        source = f"legacy dataset metadata sha256={sha256}"
    return calibration, source


def _metadata_tcp_snapshot(source_info: dict[str, object]) -> dict[str, object] | None:
    snapshot = handumi_metadata(source_info).get("controller_tcp_calibration")
    return snapshot if isinstance(snapshot, dict) else None


def _metadata_tcp_identity(source_info: dict[str, object]) -> dict[str, str]:
    metadata = handumi_metadata(source_info)
    snapshot = _metadata_tcp_snapshot(source_info) or {}
    target_robot = metadata.get("target_robot")
    target_robot_name = (
        str(target_robot.get("name"))
        if isinstance(target_robot, dict) and target_robot.get("name")
        else ""
    )
    return {
        "source_robot": str(snapshot.get("source_robot") or target_robot_name),
        "source_gripper": str(snapshot.get("source_gripper") or "unknown"),
        "tracking_device": str(
            snapshot.get("tracking_device")
            or metadata.get("recording_device")
            or "unknown"
        ),
        "controller_mount": str(snapshot.get("controller_mount") or "unknown"),
    }


def _is_trusted_tcp_snapshot(snapshot: dict[str, object]) -> bool:
    return is_identity_bound_controller_tcp_metadata(snapshot)


def _source_robot_from_metadata(
    source_info: dict[str, object],
    *,
    fallback: str,
) -> str:
    identity = _metadata_tcp_identity(source_info)
    return identity["source_robot"] or fallback


def _resolved_tcp_calibration(
    args: argparse.Namespace,
    source_info: dict[str, object],
    *,
    robot: str,
    controller_device: str,
    configured_path: Path | None,
    configured_gripper: str | None = None,
    configured_mount: str | None = None,
) -> TcpCalibrationSelection:
    """Resolve explicit, trusted dataset, robot setup, then legacy calibration."""
    explicit_path = args.controller_tcp_calibration
    use_dataset = bool(getattr(args, "use_dataset_tcp_calibration", False))
    if explicit_path is not None and use_dataset:
        raise SystemExit(
            "--controller-tcp-calibration and --use-dataset-tcp-calibration "
            "are mutually exclusive."
        )
    if explicit_path is not None:
        calibration = load_controller_tcp_calibration(explicit_path)
        sha256 = controller_tcp_calibration_sha256(explicit_path)
        return TcpCalibrationSelection(
            calibration=calibration,
            source=f"explicit {explicit_path} sha256={sha256}",
            source_robot=robot,
            source_gripper=configured_gripper or "unknown",
            tracking_device=controller_device,
            controller_mount=configured_mount or "unknown",
            trusted_dataset_snapshot=False,
        )

    snapshot = _metadata_tcp_snapshot(source_info)
    metadata_calibration = _metadata_tcp_calibration(source_info)
    if use_dataset:
        if metadata_calibration is None:
            raise SystemExit("Dataset has no unapplied controller->TCP calibration snapshot.")
        calibration, source = metadata_calibration
        identity = _metadata_tcp_identity(source_info)
        return TcpCalibrationSelection(
            calibration=calibration,
            source=source,
            trusted_dataset_snapshot=bool(
                snapshot is not None and _is_trusted_tcp_snapshot(snapshot)
            ),
            **identity,
        )

    if (
        snapshot is not None
        and _is_trusted_tcp_snapshot(snapshot)
        and metadata_calibration is not None
    ):
        calibration, source = metadata_calibration
        return TcpCalibrationSelection(
            calibration=calibration,
            source=source,
            trusted_dataset_snapshot=True,
            **_metadata_tcp_identity(source_info),
        )

    if configured_path is not None:
        calibration = load_controller_tcp_calibration(configured_path)
        sha256 = controller_tcp_calibration_sha256(configured_path)
        source = (
            f"configured {robot}/{controller_device}: {configured_path} "
            f"sha256={sha256}"
        )
        return TcpCalibrationSelection(
            calibration=calibration,
            source=source,
            source_robot=robot,
            source_gripper=configured_gripper or "unknown",
            tracking_device=controller_device,
            controller_mount=configured_mount or "unknown",
            trusted_dataset_snapshot=False,
        )

    if metadata_calibration is not None:
        calibration, source = metadata_calibration
        identity = _metadata_tcp_identity(source_info)
        return TcpCalibrationSelection(
            calibration=calibration,
            source=source,
            trusted_dataset_snapshot=False,
            **identity,
        )

    fallback_path = calibration_path_for_device(controller_device)
    calibration = load_controller_tcp_calibration(fallback_path)
    sha256 = controller_tcp_calibration_sha256(fallback_path)
    return TcpCalibrationSelection(
        calibration=calibration,
        source=f"legacy device fallback {fallback_path} sha256={sha256}",
        source_robot=robot,
        source_gripper=configured_gripper or "unknown",
        tracking_device=controller_device,
        controller_mount=configured_mount or "unknown",
        trusted_dataset_snapshot=False,
    )


def _pose7_from_mapping(value: object, *, name: str) -> np.ndarray:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    try:
        position = np.asarray(value["position"], dtype=np.float32).reshape(3)
        quaternion = quat_normalize(
            np.asarray(value["quaternion"], dtype=np.float32).reshape(4)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain position[3] and quaternion[4]"
        ) from exc
    return np.concatenate([position, quaternion]).astype(np.float32)


def load_robot_from_table(path: Path) -> np.ndarray:
    """Load ``T_robot_world_table`` from a deployment calibration YAML."""
    if not path.exists():
        raise SystemExit(
            f"Missing deployment calibration: {path}\n"
            "Create it with calibration.robot_from_table position/quaternion values."
        )
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if data.get("verified") is not True:
        print(
            f"[replay] warning: deployment calibration {path} is not marked "
            "verified; absolute task placement depends on the physical rig matching it."
        )
    root = data.get("calibration", data)
    try:
        return _pose7_from_mapping(root["robot_from_table"], name="robot_from_table")
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid deployment calibration {path}: {exc}") from exc


def apply_tcp_calibration_to_states(
    states: np.ndarray,
    calibration: ControllerTcpCalibration,
    calibration_source: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    str,
]:
    """Return states whose left/right poses are calibrated gripper TCP poses."""
    raw_left: list[np.ndarray] = []
    raw_right: list[np.ndarray] = []
    for state in states:
        left, right = raw_state_pose7_pair(state)
        raw_left.append(left)
        raw_right.append(right)

    raw_left_arr = np.asarray(raw_left, dtype=np.float32)
    raw_right_arr = np.asarray(raw_right, dtype=np.float32)
    left_tcp, right_tcp = apply_controller_tcp_calibration(
        raw_left_arr,
        raw_right_arr,
        calibration,
    )

    calibrated = np.asarray(states, dtype=np.float32).copy()
    calibrated[:, LEFT_POSE_SLICE] = left_tcp
    calibrated[:, RIGHT_POSE_SLICE] = right_tcp
    return (
        calibrated,
        raw_left_arr,
        raw_right_arr,
        left_tcp,
        right_tcp,
        calibration_source,
    )


def _tcp_geometry_diagnostics(
    calibration: ControllerTcpCalibration,
    left_tcp_pose7: np.ndarray,
    right_tcp_pose7: np.ndarray,
) -> dict[str, np.ndarray]:
    """Summarize source geometry before IK or deployment transforms are applied."""
    left = np.asarray(left_tcp_pose7, dtype=np.float32)
    right = np.asarray(right_tcp_pose7, dtype=np.float32)
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("left/right calibrated TCP trajectories must have equal 2-D shapes")
    if left.shape[1] < 3 or len(left) == 0:
        raise ValueError("calibrated TCP trajectories must contain at least one position")
    separation = np.linalg.norm(left[:, :3] - right[:, :3], axis=1)
    return {
        "offset_position_norm_m": np.asarray(
            [
                np.linalg.norm(calibration.left[:3]),
                np.linalg.norm(calibration.right[:3]),
            ],
            dtype=np.float32,
        ),
        "workspace_min_z_m": np.asarray(
            [np.min(left[:, 2]), np.min(right[:, 2])],
            dtype=np.float32,
        ),
        "same_frame_min_separation_m": np.asarray(
            [np.min(separation)],
            dtype=np.float32,
        ),
    }


def _print_tcp_geometry_diagnostics(
    selection: TcpCalibrationSelection,
    diagnostics: dict[str, np.ndarray],
    *,
    workspace: str,
) -> None:
    offsets = diagnostics["offset_position_norm_m"]
    min_z = diagnostics["workspace_min_z_m"]
    min_separation = float(diagnostics["same_frame_min_separation_m"][0])
    print(
        "[replay] source tool: "
        f"robot={selection.source_robot} gripper={selection.source_gripper} "
        f"device={selection.tracking_device} mount={selection.controller_mount}"
    )
    print(f"[replay] TCP calibration: {selection.source}")
    print(
        "[replay] Controller->TCP position distance: "
        f"left={float(offsets[0]) * 100:.1f}cm "
        f"right={float(offsets[1]) * 100:.1f}cm"
    )
    print(
        f"[replay] calibrated TCP geometry in {workspace or 'tracking'} frame: "
        f"z_min left={float(min_z[0]) * 100:.1f}cm "
        f"right={float(min_z[1]) * 100:.1f}cm; "
        f"same-frame separation_min={min_separation * 100:.1f}cm"
    )


def solve_episode(args: argparse.Namespace) -> dict[str, np.ndarray]:
    runtime = load_embodiment(args.robot)
    states, fps, source_info, recorded_gripper_openings = load_episode_states(args)
    source_metadata = handumi_metadata(source_info)
    controller_device = _resolved_controller_device(args, source_info)
    requested_retarget_mode = str(args.retarget_mode)
    retarget_mode = _resolved_retarget_mode(args, source_info)
    gripper_openings, gripper_source = _resolve_gripper_openings(
        states,
        recorded_gripper_openings,
        max_width_m=(
            runtime.config.gripper_max_width_m
            if args.gripper_max_width_m is None
            else args.gripper_max_width_m
        ),
    )
    if args.raw_controller_debug:
        states_for_retarget = states
        raw_left_controller: list[np.ndarray] = []
        raw_right_controller: list[np.ndarray] = []
        for state in states:
            left, right = raw_state_pose7_pair(state)
            raw_left_controller.append(left)
            raw_right_controller.append(right)
        raw_left_controller_arr = np.asarray(raw_left_controller, dtype=np.float32)
        raw_right_controller_arr = np.asarray(raw_right_controller, dtype=np.float32)
        left_tcp_arr = raw_left_controller_arr
        right_tcp_arr = raw_right_controller_arr
        calibration_source = None
        tcp_selection = None
        tcp_diagnostics: dict[str, np.ndarray] = {}
    else:
        source_robot = _source_robot_from_metadata(
            source_info,
            fallback=args.robot,
        )
        source_config = None
        try:
            source_config = load_robot_config(source_robot)
        except ValueError as exc:
            snapshot = _metadata_tcp_snapshot(source_info)
            can_replay_without_profile = (
                args.controller_tcp_calibration is not None
                or bool(getattr(args, "use_dataset_tcp_calibration", False))
                or (snapshot is not None and _is_trusted_tcp_snapshot(snapshot))
            )
            if not can_replay_without_profile:
                raise SystemExit(
                    f"Cannot resolve source robot-tool calibration: {exc}"
                ) from exc

        tcp_selection = _resolved_tcp_calibration(
            args,
            source_info,
            robot=source_robot,
            controller_device=controller_device,
            configured_path=(
                source_config.controller_tcp_calibrations.get(controller_device)
                if source_config is not None
                else None
            ),
            configured_gripper=(
                source_config.handumi_gripper if source_config is not None else None
            ),
            configured_mount=(
                source_config.handumi_controller_mount
                if source_config is not None
                else None
            ),
        )
        (
            states_for_retarget,
            raw_left_controller_arr,
            raw_right_controller_arr,
            left_tcp_arr,
            right_tcp_arr,
            calibration_source,
        ) = apply_tcp_calibration_to_states(
            states,
            tcp_selection.calibration,
            tcp_selection.source,
        )
        tcp_diagnostics = _tcp_geometry_diagnostics(
            tcp_selection.calibration,
            left_tcp_arr,
            right_tcp_arr,
        )
        _print_tcp_geometry_diagnostics(
            tcp_selection,
            tcp_diagnostics,
            workspace=str(source_metadata.get("tracking_workspace", "")),
        )

    frame_indices = list(range(args.start_frame, len(states), args.stride))
    if args.max_frames is not None:
        frame_indices = frame_indices[: args.max_frames]
    if not frame_indices:
        raise ValueError("No frames selected for replay.")
    if source_metadata.get("tracking_workspace") == "table" and tcp_diagnostics:
        minimum_tcp_z = float(np.min(tcp_diagnostics["workspace_min_z_m"]))
        if minimum_tcp_z > args.table_clearance_warning_m:
            print(
                "[replay] warning: calibrated TCP never approaches table z=0 "
                f"(minimum={minimum_tcp_z * 100:.1f}cm). Re-run controller->TCP "
                "pivot calibration; do not compensate this with robot_from_table."
            )

    cfg = runtime.config.ik_weights
    q = runtime.config.home_q.astype(np.float32).copy()
    solver = runtime.solver_cls()
    qs: list[np.ndarray] = []
    raw_left_gt: list[np.ndarray] = []
    raw_right_gt: list[np.ndarray] = []
    left_targets: list[np.ndarray] = []
    right_targets: list[np.ndarray] = []
    left_achieved: list[np.ndarray] = []
    right_achieved: list[np.ndarray] = []
    home_left_pose7, home_right_pose7 = solver.fk_pose7(q)
    first_left_pose7, first_right_pose7 = raw_state_pose7_pair(
        states_for_retarget[frame_indices[0]]
    )
    anchors = None
    left_adapter = None
    right_adapter = None
    robot_from_table = None
    left_tool_adapter = None
    right_tool_adapter = None
    initial_solve_count = 0
    initial_max_position_error_m = 0.0
    initial_max_rotation_error_deg = 0.0
    if retarget_mode == "anchored":
        anchors = retarget_anchors_from_raw_state(
            states_for_retarget[frame_indices[0]],
            left_robot_pose7=home_left_pose7,
            right_robot_pose7=home_right_pose7,
            max_reach=cfg.max_reach,
        )
    elif retarget_mode == "local-relative":
        world_map = (
            VR_TO_ROBOT
            if controller_device == "pico"
            else np.eye(3, dtype=np.float32)
        )
        left_adapter = local_frame_adapter(
            first_left_pose7,
            home_left_pose7,
            source_world_to_robot_world=world_map,
        )
        right_adapter = local_frame_adapter(
            first_right_pose7,
            home_right_pose7,
            source_world_to_robot_world=world_map,
        )
    else:
        if source_metadata.get("tracking_workspace") != "table":
            raise SystemExit(
                "--retarget-mode absolute-table requires a dataset recorded in the "
                "calibrated table workspace."
            )
        deployment_path = args.deployment_calibration or (
            DEFAULT_DEPLOYMENT_CALIBRATION_DIR / f"{args.robot}_table.yaml"
        )
        robot_from_table = load_robot_from_table(deployment_path)
        print(
            "[replay] robot_from_table: "
            f"translation=[{robot_from_table[0]:.4f}, {robot_from_table[1]:.4f}, "
            f"{robot_from_table[2]:.4f}]m source={deployment_path}"
        )
        mapped_left, mapped_right = absolute_table_robot_target_pose7(
            states_for_retarget[frame_indices[0]],
            robot_from_table,
        )
        if args.absolute_orientation == "relative-start":
            left_tool_adapter = orientation_only_pose_adapter(
                mapped_left, home_left_pose7
            )
            right_tool_adapter = orientation_only_pose_adapter(
                mapped_right, home_right_pose7
            )

        first_left_target, first_right_target = absolute_table_robot_target_pose7(
            states_for_retarget[frame_indices[0]],
            robot_from_table,
            left_tool_adapter_pose7=left_tool_adapter,
            right_tool_adapter_pose7=right_tool_adapter,
        )
        initial_solver = runtime.solver_cls(
            config=replace(cfg, max_joint_delta=None)
        )
        for initial_solve_count in range(1, args.initial_solve_iterations + 1):
            q = initial_solver.ik(
                q,
                left_pose=(first_left_target[:3], first_left_target[3:7]),
                right_pose=(first_right_target[:3], first_right_target[3:7]),
            )
            initial_left, initial_right = initial_solver.fk_pose7(q)
            initial_errors = pose_error_arrays(
                first_left_target[None],
                first_right_target[None],
                initial_left[None],
                initial_right[None],
            )
            initial_max_position_error_m = max(
                float(initial_errors["left_pos_error_m"][0]),
                float(initial_errors["right_pos_error_m"][0]),
            )
            initial_max_rotation_error_deg = max(
                float(initial_errors["left_rot_error_deg"][0]),
                float(initial_errors["right_rot_error_deg"][0]),
            )
            if initial_max_position_error_m <= args.initial_position_tolerance_m:
                break
        if initial_max_position_error_m > args.initial_position_tolerance_m:
            raise SystemExit(
                "Unable to prepare replay start pose: maximum TCP position error "
                f"is {initial_max_position_error_m * 100:.2f} cm after "
                f"{initial_solve_count} IK iterations. Check robot_from_table, TCP "
                "calibration, or robot reachability."
            )

    start = time.perf_counter()
    for selected_index, frame_index in enumerate(frame_indices):
        state = states_for_retarget[frame_index]
        raw_left, raw_right = raw_state_pose7_pair(state)
        if retarget_mode == "anchored":
            if anchors is None:
                raise RuntimeError("Anchored retarget mode was not initialized.")
            left_pose7, right_pose7 = raw_state_robot_target_pose7(state, anchors)
            q = solver.ik(
                q,
                left_pose=(left_pose7[:3], left_pose7[3:7]),
                right_pose=(right_pose7[:3], right_pose7[3:7]),
            )
            fk_left_pose7, fk_right_pose7 = solver.fk_pose7(q)
        elif retarget_mode == "absolute-table":
            if robot_from_table is None:
                raise RuntimeError("Absolute table retarget mode was not initialized.")
            left_pose7, right_pose7 = absolute_table_robot_target_pose7(
                state,
                robot_from_table,
                left_tool_adapter_pose7=left_tool_adapter,
                right_tool_adapter_pose7=right_tool_adapter,
            )
            q = solver.ik(
                q,
                left_pose=(left_pose7[:3], left_pose7[3:7]),
                right_pose=(right_pose7[:3], right_pose7[3:7]),
            )
            fk_left_pose7, fk_right_pose7 = solver.fk_pose7(q)
        elif selected_index == 0:
            left_pose7 = home_left_pose7.copy()
            right_pose7 = home_right_pose7.copy()
            fk_left_pose7 = home_left_pose7.copy()
            fk_right_pose7 = home_right_pose7.copy()
        else:
            if left_adapter is None or right_adapter is None:
                raise RuntimeError("Local-relative retarget mode was not initialized.")
            prev_state = states_for_retarget[frame_indices[selected_index - 1]]
            prev_left, prev_right = raw_state_pose7_pair(prev_state)
            base_left = (
                left_targets[-1]
                if args.compose_source == "commanded"
                else left_achieved[-1]
            )
            base_right = (
                right_targets[-1]
                if args.compose_source == "commanded"
                else right_achieved[-1]
            )
            left_pose7 = local_relative_robot_target_pose7(
                previous_source_pose7=prev_left,
                current_source_pose7=raw_left,
                base_robot_pose7=base_left,
                adapter_rot=left_adapter,
                home_robot_pose7=home_left_pose7,
                translation_scale=args.translation_scale,
                max_reach=cfg.max_reach,
            )
            right_pose7 = local_relative_robot_target_pose7(
                previous_source_pose7=prev_right,
                current_source_pose7=raw_right,
                base_robot_pose7=base_right,
                adapter_rot=right_adapter,
                home_robot_pose7=home_right_pose7,
                translation_scale=args.translation_scale,
                max_reach=cfg.max_reach,
            )
            q = solver.ik(
                q,
                left_pose=(left_pose7[:3], left_pose7[3:7]),
                right_pose=(right_pose7[:3], right_pose7[3:7]),
            )
            fk_left_pose7, fk_right_pose7 = solver.fk_pose7(q)
        if gripper_openings is not None:
            opening = gripper_openings[frame_index]
            runtime.set_finger_positions(
                q,
                {"left": float(opening[0]), "right": float(opening[1])},
            )
        qs.append(q.copy())
        raw_left_gt.append(raw_left)
        raw_right_gt.append(raw_right)
        left_targets.append(left_pose7)
        right_targets.append(right_pose7)
        left_achieved.append(fk_left_pose7)
        right_achieved.append(fk_right_pose7)
    elapsed = time.perf_counter() - start
    target_left = np.asarray(left_targets, dtype=np.float32)
    target_right = np.asarray(right_targets, dtype=np.float32)
    achieved_left = np.asarray(left_achieved, dtype=np.float32)
    achieved_right = np.asarray(right_achieved, dtype=np.float32)
    errors = pose_error_arrays(target_left, target_right, achieved_left, achieved_right)
    all_pos_err = np.concatenate(
        [errors["left_pos_error_m"], errors["right_pos_error_m"]]
    )
    all_rot_err = np.concatenate(
        [errors["left_rot_error_deg"], errors["right_rot_error_deg"]]
    )
    score = optimization_score_from_errors(
        float(all_pos_err.mean() * 100.0),
        float(all_pos_err.max() * 100.0),
        float(all_rot_err.mean()),
        float(all_rot_err.max()),
    )

    print(
        f"[replay] robot={args.robot} episode={args.episode} frames={len(qs)} "
        f"fps={fps:g} solved={elapsed:.2f}s ({elapsed / len(qs) * 1000:.1f} ms/frame)"
    )
    print(
        f"[replay] retarget={retarget_mode} "
        f"compose={args.compose_source} translation_scale={args.translation_scale:g}"
    )
    if requested_retarget_mode == "auto":
        print(
            "[replay] retarget auto: "
            f"tracking_workspace={source_metadata.get('tracking_workspace')!r} "
            f"-> {retarget_mode}"
        )
    if calibration_source is None:
        print("[replay] input pose: raw controller DEBUG mode")
    else:
        print(f"[replay] input pose: calibrated HandUMI TCP via {calibration_source}")
    print(f"[replay] grippers: {gripper_source}")
    if retarget_mode == "absolute-table":
        print(
            "[replay] start prepared: "
            f"iterations={initial_solve_count} "
            f"pos_max={initial_max_position_error_m * 100:.2f}cm "
            f"rot_max={initial_max_rotation_error_deg:.2f}deg "
            f"orientation={args.absolute_orientation}"
        )
    print(
        "[replay] IK EE error: "
        f"pos mean={all_pos_err.mean() * 100:.2f}cm "
        f"max={all_pos_err.max() * 100:.2f}cm; "
        f"rot mean={all_rot_err.mean():.2f}deg max={all_rot_err.max():.2f}deg; "
        f"score={score:.4f}"
    )
    violations = []
    if float(all_pos_err.max()) > args.max_ik_position_error_m:
        violations.append(
            f"position {all_pos_err.max() * 100:.2f}cm > "
            f"{args.max_ik_position_error_m * 100:.2f}cm"
        )
    if float(all_rot_err.max()) > args.max_ik_rotation_error_deg:
        violations.append(
            f"rotation {all_rot_err.max():.2f}deg > "
            f"{args.max_ik_rotation_error_deg:.2f}deg"
        )
    if violations:
        message = "IK fidelity threshold exceeded: " + "; ".join(violations)
        if args.strict_ik:
            raise SystemExit(message)
        print(f"[replay] warning: {message}")
    return {
        "qpos": np.asarray(qs, dtype=np.float32),
        "raw_left_pose7_ground_truth": np.asarray(raw_left_gt, dtype=np.float32),
        "raw_right_pose7_ground_truth": np.asarray(raw_right_gt, dtype=np.float32),
        "raw_left_controller_pose7": raw_left_controller_arr[frame_indices],
        "raw_right_controller_pose7": raw_right_controller_arr[frame_indices],
        "calibrated_left_tcp_pose7": left_tcp_arr[frame_indices],
        "calibrated_right_tcp_pose7": right_tcp_arr[frame_indices],
        "target_left_pose7_robot_world": target_left,
        "target_right_pose7_robot_world": target_right,
        "achieved_left_pose7_robot_world": achieved_left,
        "achieved_right_pose7_robot_world": achieved_right,
        "left_pos_error_m": errors["left_pos_error_m"],
        "right_pos_error_m": errors["right_pos_error_m"],
        "left_rot_error_deg": errors["left_rot_error_deg"],
        "right_rot_error_deg": errors["right_rot_error_deg"],
        "optimization_score": np.asarray([score], dtype=np.float32),
        "ik_weights": np.asarray(
            [cfg.pos_weight, cfg.ori_weight, cfg.rest_weight], dtype=np.float32
        ),
        "home_left_pose7_robot_world": home_left_pose7[None],
        "home_right_pose7_robot_world": home_right_pose7[None],
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "fps": np.asarray([fps], dtype=np.float32),
        "controller_tcp_calibration": np.asarray([str(calibration_source or "")]),
        "controller_tcp_source_robot": np.asarray(
            [tcp_selection.source_robot if tcp_selection is not None else ""]
        ),
        "controller_tcp_source_gripper": np.asarray(
            [tcp_selection.source_gripper if tcp_selection is not None else ""]
        ),
        "controller_tcp_controller_mount": np.asarray(
            [tcp_selection.controller_mount if tcp_selection is not None else ""]
        ),
        "controller_tcp_trusted_dataset_snapshot": np.asarray(
            [
                tcp_selection.trusted_dataset_snapshot
                if tcp_selection is not None
                else False
            ],
            dtype=np.bool_,
        ),
        "controller_tcp_offset_position_norm_m": tcp_diagnostics.get(
            "offset_position_norm_m",
            np.asarray([], dtype=np.float32),
        ),
        "calibrated_tcp_workspace_min_z_m": tcp_diagnostics.get(
            "workspace_min_z_m",
            np.asarray([], dtype=np.float32),
        ),
        "calibrated_tcp_same_frame_min_separation_m": tcp_diagnostics.get(
            "same_frame_min_separation_m",
            np.asarray([], dtype=np.float32),
        ),
        "controller_device": np.asarray([controller_device]),
        "tracking_workspace": np.asarray(
            [str(source_metadata.get("tracking_workspace", ""))]
        ),
        "robot_from_table_pose7": np.asarray(
            [robot_from_table] if robot_from_table is not None else [],
            dtype=np.float32,
        ),
        "left_tool_adapter_pose7": np.asarray(
            [left_tool_adapter] if left_tool_adapter is not None else [],
            dtype=np.float32,
        ),
        "right_tool_adapter_pose7": np.asarray(
            [right_tool_adapter] if right_tool_adapter is not None else [],
            dtype=np.float32,
        ),
        "absolute_orientation": np.asarray([args.absolute_orientation]),
        "initial_solve_iterations": np.asarray(
            [initial_solve_count], dtype=np.int64
        ),
        "initial_max_position_error_m": np.asarray(
            [initial_max_position_error_m], dtype=np.float32
        ),
        "initial_max_rotation_error_deg": np.asarray(
            [initial_max_rotation_error_deg], dtype=np.float32
        ),
        "gripper_normalized": (
            gripper_openings[frame_indices]
            if gripper_openings is not None
            else np.empty((len(frame_indices), 0), dtype=np.float32)
        ),
        "gripper_source": np.asarray([gripper_source]),
        "raw_controller_debug": np.asarray([args.raw_controller_debug], dtype=np.bool_),
        "retarget_mode": np.asarray([retarget_mode]),
        "retarget_mode_requested": np.asarray([requested_retarget_mode]),
        "compose_source": np.asarray([args.compose_source]),
        "translation_scale": np.asarray([args.translation_scale], dtype=np.float32),
    }


def save_rollout(args: argparse.Namespace, rollout: dict[str, np.ndarray]) -> Path:
    output = args.output
    if output is None:
        output = DEFAULT_OUT_DIR / f"episode_{args.episode:06d}_{args.robot}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        repo_id=np.asarray([args.repo_id]),
        robot=np.asarray([args.robot]),
        episode=np.asarray([args.episode], dtype=np.int64),
        **rollout,
    )
    print(f"[replay] saved: {output}")
    return output


def _render_task_scene(server, args: argparse.Namespace, rollout: dict[str, np.ndarray]) -> None:
    if args.scene is None:
        return
    transforms = rollout["robot_from_table_pose7"]
    if len(transforms) != 1:
        raise SystemExit("--scene currently requires --retarget-mode absolute-table.")

    from handumi.sim.scene import load_scene

    robot_from_table = transforms[0]
    for body in load_scene(args.scene):
        body_table = np.concatenate(
            [
                body.rest_position,
                body.rest_quaternion_wxyz[[1, 2, 3, 0]],
            ]
        ).astype(np.float32)
        body_robot = pose_mul(robot_from_table, body_table)
        frame = server.scene.add_frame(
            f"/scene/{body.name}",
            position=tuple(body_robot[:3]),
            wxyz=tuple(body_robot[[6, 3, 4, 5]]),
            show_axes=False,
        )
        del frame
        for index, geom in enumerate(body.geoms):
            if geom.kind != "box":
                continue
            color = tuple(int(round(channel * 255)) for channel in geom.rgba[:3])
            server.scene.add_box(
                f"/scene/{body.name}/geom_{index}",
                dimensions=tuple(2.0 * value for value in geom.size),
                position=tuple(geom.local_position),
                wxyz=tuple(geom.local_quaternion_wxyz),
                color=color,
            )


def show_viewer(args: argparse.Namespace, rollout: dict[str, np.ndarray]) -> None:
    import viser
    import yourdfpy
    from viser.extras import ViserUrdf

    runtime = load_embodiment(args.robot)
    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/grid", width=3.0, height=3.0, cell_size=0.1)
    urdf = yourdfpy.URDF.load(
        str(runtime.urdf_path),
        mesh_dir=str(runtime.urdf_path.parent),
        load_meshes=True,
    )
    robot_view = ViserUrdf(server, urdf, root_node_name="/robot")
    _render_task_scene(server, args, rollout)
    server.scene.add_spline_catmull_rom(
        "/traj/target_left",
        positions=rollout["target_left_pose7_robot_world"][:, :3],
        color=(255, 190, 50),
        line_width=2.0,
    )
    server.scene.add_spline_catmull_rom(
        "/traj/target_right",
        positions=rollout["target_right_pose7_robot_world"][:, :3],
        color=(80, 220, 130),
        line_width=2.0,
    )
    server.scene.add_spline_catmull_rom(
        "/traj/achieved_left",
        positions=rollout["achieved_left_pose7_robot_world"][:, :3],
        color=(80, 160, 255),
        line_width=2.0,
    )
    server.scene.add_spline_catmull_rom(
        "/traj/achieved_right",
        positions=rollout["achieved_right_pose7_robot_world"][:, :3],
        color=(255, 90, 90),
        line_width=2.0,
    )
    target_left = server.scene.add_icosphere(
        "/target/left", radius=0.018, color=(255, 190, 50)
    )
    target_right = server.scene.add_icosphere(
        "/target/right", radius=0.018, color=(80, 220, 130)
    )
    achieved_left = server.scene.add_icosphere(
        "/achieved/left", radius=0.014, color=(80, 160, 255)
    )
    achieved_right = server.scene.add_icosphere(
        "/achieved/right", radius=0.014, color=(255, 90, 90)
    )
    play = server.gui.add_checkbox("Play", True)
    frame = server.gui.add_slider("Frame", 0, len(rollout["qpos"]) - 1, 1, 0)
    err_text = server.gui.add_text("EE error (cm/deg)", "-", disabled=True)

    def draw(i: int) -> None:
        robot_view.update_cfg(rollout["qpos"][i])
        target_left.position = tuple(rollout["target_left_pose7_robot_world"][i, :3])
        target_right.position = tuple(rollout["target_right_pose7_robot_world"][i, :3])
        achieved_left.position = tuple(
            rollout["achieved_left_pose7_robot_world"][i, :3]
        )
        achieved_right.position = tuple(
            rollout["achieved_right_pose7_robot_world"][i, :3]
        )
        err_text.value = (
            f"L={rollout['left_pos_error_m'][i] * 100:.1f}cm/"
            f"{rollout['left_rot_error_deg'][i]:.1f}deg "
            f"R={rollout['right_pos_error_m'][i] * 100:.1f}cm/"
            f"{rollout['right_rot_error_deg'][i]:.1f}deg"
        )

    draw(0)
    print(f"[replay] viewer: http://localhost:{server.get_port()}")
    current = 0
    while True:
        if play.value:
            current = (current + 1) % len(rollout["qpos"])
            frame.value = current
        else:
            current = int(frame.value)
        draw(current)
        time.sleep(1.0 / 30.0)


def main() -> None:
    args = build_parser().parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1.")
    if args.initial_solve_iterations < 1:
        raise ValueError("--initial-solve-iterations must be >= 1.")
    if args.initial_position_tolerance_m <= 0.0:
        raise ValueError("--initial-position-tolerance-m must be > 0.")
    if args.max_ik_position_error_m <= 0.0:
        raise ValueError("--max-ik-position-error-m must be > 0.")
    if args.max_ik_rotation_error_deg <= 0.0:
        raise ValueError("--max-ik-rotation-error-deg must be > 0.")
    if args.table_clearance_warning_m <= 0.0:
        raise ValueError("--table-clearance-warning-m must be > 0.")
    rollout = solve_episode(args)
    save_rollout(args, rollout)
    if not args.headless:
        show_viewer(args, rollout)


if __name__ == "__main__":
    main()
