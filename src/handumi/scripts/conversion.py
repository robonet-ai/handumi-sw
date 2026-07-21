#!/usr/bin/env python3
"""Convert a raw HandUMI LeRobot dataset to robot-specific joint angles.

The script loads raw HandUMI controller states (``observation.state``), applies
the same optional controller->TCP calibration and local-relative retargeting
used by ``replay_in_sim.py``, then writes a LeRobot v3.0 dataset whose
``observation.state`` and ``action`` columns contain physical robot joint
values.

All video streams are preserved unchanged (the last frame of each episode is
dropped from the parquet data to produce ``action = state[t+1]``, but the
video files themselves are copied as-is).

Usage
-----
::

    # Axol embodiment (output defaults to NONHUMAN-RESEARCH/handumi-dataset-v2-axol)
    handumi-convert \
        --repo-id NONHUMAN-RESEARCH/handumi-dataset-v2 \
        --robot axol

    # Piper embodiment, custom output repo-id, push to hub afterwards
    handumi-convert \
        --repo-id NONHUMAN-RESEARCH/handumi-dataset-v2 \
        --robot piper \
        --output-repo-id NONHUMAN-RESEARCH/my-piper-dataset \
        --push-to-hub
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

import numpy as np

from handumi.calibration.control_tcp import (
    DEFAULT_DEVICE as DEFAULT_CONTROLLER_DEVICE,
    ControllerTcpCalibration,
    apply_controller_tcp_calibration,
    calibration_path_for_device,
    controller_tcp_calibration_from_metadata,
    controller_tcp_calibration_metadata,
    is_identity_bound_controller_tcp_metadata,
    load_controller_tcp_calibration,
)
from handumi.dataset.reader import dataset_root_from_repo_id, handumi_metadata
from handumi.dataset.raw import (
    LEFT_GRIPPER_INDEX,
    LEFT_POSE_SLICE,
    RIGHT_GRIPPER_INDEX,
    RIGHT_POSE_SLICE,
)
from handumi.retargeting.handumi_to_robot import (
    local_frame_adapter,
    local_relative_robot_target_pose7,
    raw_state_pose7_pair,
    raw_state_robot_target_pose7,
    retarget_anchors_from_raw_state,
)
from handumi.robots.registry import EMBODIMENT_NAMES, load_embodiment, load_robot_config

load_dotenv()


@dataclass(frozen=True)
class ConversionTcpCalibrationSelection:
    calibration: ControllerTcpCalibration
    metadata: dict[str, Any]
    source: str

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a PICO/HandUMI LeRobot dataset to an embodiment-specific "
            "joint-angle dataset via IK retargeting."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------------
    # Source / output datasets
    # ------------------------------------------------------------------
    ds = parser.add_argument_group("Dataset I/O")
    ds.add_argument(
        "dataset",
        nargs="?",
        help="Local source dataset path or Hugging Face repo id.",
    )
    ds.add_argument(
        "--repo-id",
        default=None,
        help="Legacy Hugging Face source dataset flag.",
    )
    ds.add_argument(
        "--root",
        default=None,
        help=(
            "Local root of the source dataset. "
            "Defaults to outputs/datasets/<repo-name>."
        ),
    )
    ds.add_argument(
        "--output-repo-id",
        default=None,
        help=(
            "HuggingFace repo-id for the converted dataset. "
            "Defaults to <source-repo-id>-<embodiment> "
            "(e.g. NONHUMAN-RESEARCH/handumi-dataset-v2-piper). "
            "The local output directory is always outputs/datasets/<output-repo-name>."
        ),
    )
    ds.add_argument(
        "--revision",
        default="main",
        help="Git revision of the source dataset.",
    )
    ds.add_argument(
        "--source",
        default="observation.state",
        help="Raw 16D HandUMI feature column to convert.",
    )
    ds.add_argument(
        "--column",
        default=None,
        help="Deprecated alias for --source.",
    )
    ds.add_argument(
        "--episodes",
        default=None,
        help=(
            "Comma-separated list of episode indices to process "
            "(default: all episodes)."
        ),
    )
    ds.add_argument(
        "--task",
        default=None,
        help=(
            "Override the task description for all episodes.  "
            "When not set the script tries to read tasks from the source "
            "dataset's tasks.parquet."
        ),
    )
    ds.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the resulting dataset to the HuggingFace Hub after writing.",
    )
    ds.add_argument(
        "--hub-token",
        default=None,
        help="HuggingFace API token (uses HF_TOKEN env var if not set).",
    )
    ds.add_argument(
        "--quality-config",
        type=Path,
        default=Path("configs/quality.yaml"),
        help="Offline episode-acceptance thresholds.",
    )
    ds.add_argument(
        "--skip-quality-filter",
        action="store_true",
        help="Convert rejected episodes too; intended only for debugging.",
    )

    # ------------------------------------------------------------------
    # Embodiment selection
    # ------------------------------------------------------------------
    emb = parser.add_argument_group("Embodiment")
    embodiment_selection = emb.add_mutually_exclusive_group()
    embodiment_selection.add_argument(
        "--robot",
        dest="embodiment",
        choices=EMBODIMENT_NAMES,
        default=None,
        help="Target robot; loads its validated retargeting profile.",
    )
    embodiment_selection.add_argument(
        "--embodiment",
        dest="embodiment",
        choices=EMBODIMENT_NAMES,
        default=None,
        help=argparse.SUPPRESS,
    )
    embodiment_selection.add_argument(
        "--piper",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    # ------------------------------------------------------------------
    # Shared IK parameters
    # ------------------------------------------------------------------
    ik = parser.add_argument_group("Shared IK parameters")
    ik.add_argument("--scale", type=float, default=1.0)
    ik.add_argument(
        "--axis-map",
        default=None,
        help=(
            "PICO delta → robot delta mapping, e.g. z,x,y or z,y,-x.  "
            "Defaults to the selected embodiment's validated mapping."
        ),
    )
    ik.add_argument("--left-only", action="store_true")
    ik.add_argument("--right-only", action="store_true")
    ik.add_argument(
        "--gripper",
        type=float,
        default=1.0,
        help="Fallback gripper opening in [0, 1], used only when the recording "
        "carries no Feetech widths (e.g. --skip-feetech). Otherwise the "
        "recorded opening drives the finger joints frame by frame.",
    )
    ik.add_argument(
        "--gripper-max-width-m",
        type=float,
        default=None,
        help="HandUMI full opening (m) that maps to the robot gripper fully "
        "open; defaults to the selected robot configuration. Recorded normalized "
        "Feetech values take precedence in replay-parity conversion.",
    )
    ik.add_argument("--pos-weight", type=float, default=None)
    ik.add_argument("--ori-weight", type=float, default=None)
    ik.add_argument("--max-joint-delta", type=float, default=None)
    ik.add_argument("--max-reach", type=float, default=None)
    ik.add_argument(
        "--retarget-mode",
        choices=("local-relative", "anchored", "absolute-table"),
        default=None,
        help=(
            "Retargeting mode shared with replay_in_sim.py. Defaults to "
            "absolute-table for --robot piper and local-relative otherwise."
        ),
    )
    ik.add_argument(
        "--compose-source",
        choices=("commanded", "achieved"),
        default="commanded",
        help="For local-relative mode, compose on previous target or achieved FK.",
    )
    ik.add_argument(
        "--translation-scale",
        type=float,
        default=1.0,
        help="Scale local-relative translation deltas after frame adaptation.",
    )
    ik.add_argument(
        "--controller-device",
        choices=("pico", "meta"),
        default=None,
        help=(
            "Override the source tracking device. Defaults to dataset metadata, "
            f"then {DEFAULT_CONTROLLER_DEVICE!r} for legacy datasets."
        ),
    )
    ik.add_argument(
        "--controller-tcp-calibration",
        type=Path,
        default=None,
        help="Override the source robot/device Controller->TCP calibration.",
    )
    ik.add_argument(
        "--raw-controller-debug",
        action="store_true",
        help="Use raw controller poses directly, without controller->TCP calibration.",
    )
    ik.add_argument(
        "--deployment-calibration",
        type=Path,
        default=None,
        help=(
            "YAML containing robot_from_table for absolute-table conversion. "
            "Defaults to configs/calibration/<robot>_table.yaml."
        ),
    )
    ik.add_argument(
        "--absolute-orientation",
        choices=("relative-start", "table-absolute"),
        default="relative-start",
        help="Absolute-table tool orientation policy used by replay.",
    )
    ik.add_argument("--initial-solve-iterations", type=int, default=12)
    ik.add_argument("--initial-position-tolerance-m", type=float, default=0.01)
    ik.add_argument("--max-ik-position-error-m", type=float, default=0.03)
    ik.add_argument("--max-ik-rotation-error-deg", type=float, default=45.0)
    ik.add_argument("--table-clearance-warning-m", type=float, default=0.10)
    ik.add_argument(
        "--strict-ik",
        action="store_true",
        help="Reject an episode when replay IK fidelity thresholds are exceeded.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the conversion plan without loading episodes.",
    )

    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_output_repo_id(source_repo_id: str, embodiment: str) -> str:
    """Derive ``{namespace}/{source_name}-{embodiment}`` from the source repo id."""
    repo_id = source_repo_id.rstrip("/")
    if "/" in repo_id:
        namespace, name = repo_id.rsplit("/", 1)
        return f"{namespace}/{name}-{embodiment}"
    return f"{repo_id}-{embodiment}"


def _resolve_cli_profile(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    """Resolve robot-specific conversion defaults before loading any data."""
    requested_robot = args.embodiment or "axol"
    args.piper = bool(args.piper or requested_robot == "piper")
    args.embodiment = "piper" if args.piper else requested_robot
    if args.piper:
        if args.retarget_mode not in (None, "absolute-table"):
            parser.error("--robot piper requires --retarget-mode absolute-table.")
        args.retarget_mode = "absolute-table"
    else:
        args.embodiment = args.embodiment or "axol"
        args.retarget_mode = args.retarget_mode or "local-relative"

    if args.retarget_mode == "absolute-table":
        path = args.deployment_calibration or (
            Path("configs/calibration") / f"{args.embodiment}_table.yaml"
        )
        from handumi.scripts.replay.replay_in_sim import load_robot_from_table

        try:
            load_robot_from_table(path, expected_robot=args.embodiment)
        except SystemExit as exc:
            parser.error(str(exc))
        args.deployment_calibration = path

    if args.initial_solve_iterations < 1:
        parser.error("--initial-solve-iterations must be >= 1.")
    if args.initial_position_tolerance_m <= 0.0:
        parser.error("--initial-position-tolerance-m must be > 0.")
    if args.max_ik_position_error_m <= 0.0:
        parser.error("--max-ik-position-error-m must be > 0.")
    if args.max_ik_rotation_error_deg <= 0.0:
        parser.error("--max-ik-rotation-error-deg must be > 0.")
    if args.table_clearance_warning_m <= 0.0:
        parser.error("--table-clearance-warning-m must be > 0.")


def _deployment_calibration_metadata(args: argparse.Namespace) -> dict[str, Any] | None:
    path = args.deployment_calibration
    if path is None:
        return None
    raw = Path(path).read_bytes()
    return {
        "robot": args.embodiment,
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _parse_episode_list(value: str | None, max_episodes: int) -> list[int]:
    if value is None:
        return list(range(max_episodes))
    indices = [int(x.strip()) for x in value.split(",") if x.strip()]
    out_of_range = [idx for idx in indices if idx < 0 or idx >= max_episodes]
    if out_of_range:
        raise ValueError(
            f"Episode indices out of range [0, {max_episodes - 1}]: {out_of_range}"
        )
    return indices


def _load_source_tasks(source_root: Path) -> dict[int, str]:
    """Return a mapping ``{episode_index: task_string}`` from tasks.parquet."""
    import pandas as pd

    tasks_path = source_root / "meta" / "tasks.parquet"
    episodes_dir = source_root / "meta" / "episodes"
    task_map: dict[int, str] = {}

    if not tasks_path.exists():
        return task_map

    tasks_df = pd.read_parquet(tasks_path)
    task_idx_to_str: dict[int, str] = {
        int(row["task_index"]): str(task_str)
        for task_str, row in tasks_df.iterrows()
    }

    if not episodes_dir.exists():
        return task_map

    import glob

    for parquet_path in sorted(glob.glob(str(episodes_dir / "**/*.parquet"), recursive=True)):
        ep_df = pd.read_parquet(parquet_path)
        for _, row in ep_df.iterrows():
            ep_idx = int(row["episode_index"])
            task_list = row.get("tasks", [])
            if task_list and len(task_list) > 0:
                first_task_idx = task_list[0] if isinstance(task_list[0], (int, np.integer)) else 0
                task_map[ep_idx] = task_idx_to_str.get(int(first_task_idx), "task")
    return task_map


def _source_tcp_snapshot(source_info: dict[str, Any]) -> dict[str, Any] | None:
    snapshot = handumi_metadata(source_info).get("controller_tcp_calibration")
    if not isinstance(snapshot, dict) or snapshot.get("applied_to_state") is True:
        return None
    return dict(snapshot)


def _resolve_conversion_tcp_calibration(
    args: argparse.Namespace,
    source_info: dict[str, Any],
) -> ConversionTcpCalibrationSelection:
    """Resolve the source robot-tool transform before converting to target joints."""
    metadata = handumi_metadata(source_info)
    snapshot = _source_tcp_snapshot(source_info)
    target_robot = metadata.get("target_robot")
    legacy_source_robot = (
        str(target_robot.get("name"))
        if isinstance(target_robot, dict) and target_robot.get("name")
        else ""
    )
    source_robot = str(
        (snapshot or {}).get("source_robot")
        or legacy_source_robot
        or args.embodiment
    )
    snapshot_device = str((snapshot or {}).get("tracking_device") or "")
    if (
        args.controller_device is not None
        and snapshot is not None
        and is_identity_bound_controller_tcp_metadata(snapshot)
        and str(args.controller_device) != snapshot_device
    ):
        raise ValueError(
            f"--controller-device {args.controller_device!r} conflicts with the "
            f"identity-bound dataset snapshot ({snapshot_device!r})."
        )
    controller_device = str(
        args.controller_device
        or (snapshot or {}).get("tracking_device")
        or metadata.get("recording_device")
        or DEFAULT_CONTROLLER_DEVICE
    )
    args.controller_device = controller_device

    try:
        source_config = load_robot_config(source_robot)
    except ValueError:
        source_config = None
    source_gripper = (
        source_config.handumi_gripper if source_config is not None else None
    )
    controller_mount = (
        source_config.handumi_controller_mount if source_config is not None else None
    )

    if args.controller_tcp_calibration is not None:
        path = args.controller_tcp_calibration
        calibration = load_controller_tcp_calibration(path)
        output_metadata = controller_tcp_calibration_metadata(
            path,
            applied_to_state=True,
            source_robot=source_robot,
            source_gripper=source_gripper,
            tracking_device=controller_device,
            controller_mount=controller_mount,
        )
        return ConversionTcpCalibrationSelection(
            calibration=calibration,
            metadata=output_metadata,
            source=f"explicit {path} sha256={output_metadata['sha256']}",
        )

    if snapshot is not None and is_identity_bound_controller_tcp_metadata(snapshot):
        output_metadata = dict(snapshot)
        output_metadata["applied_to_state"] = True
        return ConversionTcpCalibrationSelection(
            calibration=controller_tcp_calibration_from_metadata(snapshot),
            metadata=output_metadata,
            source=(
                "dataset robot-tool snapshot "
                f"{snapshot['source_robot']}/{snapshot['tracking_device']} "
                f"sha256={snapshot.get('sha256', 'unknown')}"
            ),
        )

    configured_path = (
        source_config.controller_tcp_calibrations.get(controller_device)
        if source_config is not None
        else None
    )
    if configured_path is not None:
        output_metadata = controller_tcp_calibration_metadata(
            configured_path,
            applied_to_state=True,
            source_robot=source_robot,
            source_gripper=source_gripper,
            tracking_device=controller_device,
            controller_mount=controller_mount,
        )
        return ConversionTcpCalibrationSelection(
            calibration=load_controller_tcp_calibration(configured_path),
            metadata=output_metadata,
            source=(
                f"configured {source_robot}/{controller_device}: {configured_path} "
                f"sha256={output_metadata['sha256']}"
            ),
        )

    if snapshot is not None:
        output_metadata = dict(snapshot)
        output_metadata["applied_to_state"] = True
        return ConversionTcpCalibrationSelection(
            calibration=controller_tcp_calibration_from_metadata(snapshot),
            metadata=output_metadata,
            source=f"legacy dataset metadata sha256={snapshot.get('sha256', 'unknown')}",
        )

    fallback_path = calibration_path_for_device(controller_device)
    output_metadata = controller_tcp_calibration_metadata(
        fallback_path,
        applied_to_state=True,
        source_robot=source_robot,
        tracking_device=controller_device,
    )
    return ConversionTcpCalibrationSelection(
        calibration=load_controller_tcp_calibration(fallback_path),
        metadata=output_metadata,
        source=(
            f"legacy device fallback {controller_device}: {fallback_path} "
            f"sha256={output_metadata['sha256']}"
        ),
    )


def _apply_tcp_calibration_to_states(
    states: np.ndarray,
    calibration: ControllerTcpCalibration,
) -> np.ndarray:
    """Return raw states whose left/right pose slots are calibrated TCP poses."""
    raw_left: list[np.ndarray] = []
    raw_right: list[np.ndarray] = []
    for state in states:
        left, right = raw_state_pose7_pair(state)
        raw_left.append(left)
        raw_right.append(right)

    left_tcp, right_tcp = apply_controller_tcp_calibration(
        np.asarray(raw_left, dtype=np.float32),
        np.asarray(raw_right, dtype=np.float32),
        calibration,
    )
    calibrated = np.asarray(states, dtype=np.float32).copy()
    calibrated[:, LEFT_POSE_SLICE] = left_tcp
    calibrated[:, RIGHT_POSE_SLICE] = right_tcp
    return calibrated


def _solver_config(runtime, args: argparse.Namespace):
    base = runtime.config.ik_weights

    def override(name: str, fallback):
        value = getattr(args, name, None)
        return fallback if value is None else float(value)

    return runtime.config_cls(
        pos_weight=override("pos_weight", base.pos_weight),
        ori_weight=override("ori_weight", base.ori_weight),
        rest_weight=override("rest_weight", base.rest_weight),
        posture_weight=override("posture_weight", base.posture_weight),
        manipulability_weight=override(
            "manipulability_weight",
            base.manipulability_weight,
        ),
        max_joint_delta=override("max_joint_delta", base.max_joint_delta)
        if base.max_joint_delta is not None or args.max_joint_delta is not None
        else None,
        max_reach=override("max_reach", base.max_reach)
        if base.max_reach is not None or args.max_reach is not None
        else None,
    )


def solve_joint_trajectory_from_raw_states(
    *,
    args: argparse.Namespace,
    states: np.ndarray,
) -> tuple[np.ndarray, str | None]:
    """Solve joints with the same retargeting loop used by replay_in_sim.py."""
    if args.raw_controller_debug:
        states_for_retarget = np.asarray(states, dtype=np.float32)
        calibration_source = None
    else:
        selection = getattr(args, "controller_tcp_selection", None)
        if selection is None:
            calibration = load_controller_tcp_calibration(
                args.controller_tcp_calibration
            )
            calibration_source = str(args.controller_tcp_calibration)
        else:
            calibration = selection.calibration
            calibration_source = selection.source
        states_for_retarget = _apply_tcp_calibration_to_states(
            states,
            calibration,
        )

    runtime = load_embodiment(args.embodiment)
    cfg = _solver_config(runtime, args)
    q = runtime.config.home_q.astype(np.float32).copy()
    solver = runtime.solver_cls(config=cfg)
    home_left_pose7, home_right_pose7 = solver.fk_pose7(q)
    first_left_pose7, first_right_pose7 = raw_state_pose7_pair(states_for_retarget[0])

    anchors = None
    left_adapter = None
    right_adapter = None
    if args.retarget_mode == "anchored":
        anchors = retarget_anchors_from_raw_state(
            states_for_retarget[0],
            left_robot_pose7=home_left_pose7,
            right_robot_pose7=home_right_pose7,
            max_reach=cfg.max_reach,
        )
    else:
        left_adapter = local_frame_adapter(first_left_pose7, home_left_pose7)
        right_adapter = local_frame_adapter(first_right_pose7, home_right_pose7)

    qs: list[np.ndarray] = []
    left_targets: list[np.ndarray] = []
    right_targets: list[np.ndarray] = []
    left_achieved: list[np.ndarray] = []
    right_achieved: list[np.ndarray] = []

    for i, state in enumerate(states_for_retarget):
        raw_left, raw_right = raw_state_pose7_pair(state)
        if args.retarget_mode == "anchored":
            if anchors is None:
                raise RuntimeError("Anchored retarget mode was not initialized.")
            left_pose7, right_pose7 = raw_state_robot_target_pose7(state, anchors)
            q = solver.ik(
                q,
                left_pose=(left_pose7[:3], left_pose7[3:7]),
                right_pose=(right_pose7[:3], right_pose7[3:7]),
            )
            fk_left_pose7, fk_right_pose7 = solver.fk_pose7(q)
        elif i == 0:
            left_pose7 = home_left_pose7.copy()
            right_pose7 = home_right_pose7.copy()
            fk_left_pose7 = home_left_pose7.copy()
            fk_right_pose7 = home_right_pose7.copy()
        else:
            if left_adapter is None or right_adapter is None:
                raise RuntimeError("Local-relative retarget mode was not initialized.")
            prev_left, prev_right = raw_state_pose7_pair(states_for_retarget[i - 1])
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

        qs.append(q.copy())
        left_targets.append(left_pose7)
        right_targets.append(right_pose7)
        left_achieved.append(fk_left_pose7)
        right_achieved.append(fk_right_pose7)
        if (i + 1) % 100 == 0 or (i + 1) == len(states_for_retarget):
            print(f"    frame {i + 1}/{len(states_for_retarget)}", end="\r", flush=True)

    print()
    joints = np.asarray(qs, dtype=np.float32)
    _write_gripper_joints(joints, states=states, runtime=runtime, args=args)
    return joints, calibration_source


def _write_gripper_joints(
    joints: np.ndarray,
    *,
    states: np.ndarray,
    runtime,
    args: argparse.Namespace,
) -> None:
    """Drive the finger joints from the recorded HandUMI opening widths.

    ``state[14]``/``state[15]`` carry the left/right opening in meters;
    normalized by --gripper-max-width-m and scaled to each finger's URDF
    range (same mapping handumi-teleop-sim renders). Recordings without Feetech
    (widths all zero) fall back to the constant --gripper opening.
    """
    widths_m = np.asarray(states, dtype=np.float32)[
        :, [LEFT_GRIPPER_INDEX, RIGHT_GRIPPER_INDEX]
    ]
    configured_max_width = (
        runtime.config.gripper_max_width_m
        if args.gripper_max_width_m is None
        else args.gripper_max_width_m
    )
    max_w = max(float(configured_max_width), 1e-6)
    normalized = np.clip(widths_m / max_w, 0.0, 1.0)
    if not np.any(widths_m > 0):
        normalized[:] = float(np.clip(args.gripper, 0.0, 1.0))
        print(f"    no recorded widths — constant gripper opening {args.gripper:.2f}")
    for column, side in enumerate(("left", "right")):
        for finger in runtime.finger_joints.get(side, ()):
            if hasattr(finger, "closed_value") and hasattr(finger, "open_value"):
                joints[:, finger.index] = finger.closed_value + (
                    normalized[:, column] * (finger.open_value - finger.closed_value)
                )
            else:
                # Compatibility with lightweight test/runtime doubles.
                joint_index, open_value = finger
                joints[:, joint_index] = normalized[:, column] * open_value


# ---------------------------------------------------------------------------
# Per-episode IK processing
# ---------------------------------------------------------------------------


def _solve_with_replay_pipeline(
    *,
    args: argparse.Namespace,
    source_episode_index: int,
) -> dict[str, np.ndarray]:
    """Run the replay solver so absolute-table conversion has exact qpos parity."""
    from handumi.scripts.replay.replay_in_sim import (
        build_parser as build_replay_parser,
        solve_episode as solve_replay_episode,
    )

    replay_args = build_replay_parser().parse_args([])
    replay_args.repo_id = args.repo_id
    replay_args.dataset_root = args.root
    replay_args.revision = args.revision
    replay_args.episode = source_episode_index
    replay_args.robot = args.embodiment
    replay_args.source = args.source
    replay_args.retarget_mode = "absolute-table"
    replay_args.compose_source = args.compose_source
    replay_args.translation_scale = args.translation_scale
    replay_args.controller_device = args.controller_device
    replay_args.controller_tcp_calibration = args.controller_tcp_calibration
    replay_args.raw_controller_debug = args.raw_controller_debug
    replay_args.deployment_calibration = args.deployment_calibration
    replay_args.absolute_orientation = args.absolute_orientation
    replay_args.initial_solve_iterations = args.initial_solve_iterations
    replay_args.initial_position_tolerance_m = args.initial_position_tolerance_m
    replay_args.gripper_max_width_m = args.gripper_max_width_m
    replay_args.max_ik_position_error_m = args.max_ik_position_error_m
    replay_args.max_ik_rotation_error_deg = args.max_ik_rotation_error_deg
    replay_args.table_clearance_warning_m = args.table_clearance_warning_m
    replay_args.strict_ik = args.strict_ik
    return solve_replay_episode(replay_args)


def _piper_command_states_from_rollout(
    rollout: dict[str, np.ndarray],
    *,
    actuated_names: list[str] | tuple[str, ...],
    gripper_max_width_m: float,
) -> np.ndarray:
    """Return two Piper 6-DoF arms plus one physical opening per gripper."""
    qpos = np.asarray(rollout["qpos"], dtype=np.float32)
    openings = np.asarray(rollout["gripper_normalized"], dtype=np.float32)
    if openings.shape != (len(qpos), 2):
        raise ValueError(
            "Piper conversion requires replay gripper_normalized with shape "
            f"({len(qpos)}, 2), got {openings.shape}."
        )
    names = list(actuated_names)
    arm_indices = {
        side: [names.index(f"{side}_joint{joint}") for joint in range(1, 7)]
        for side in ("left", "right")
    }
    width_m = np.clip(openings, 0.0, 1.0) * np.float32(gripper_max_width_m)
    return np.column_stack(
        [
            qpos[:, arm_indices["left"]],
            width_m[:, 0],
            qpos[:, arm_indices["right"]],
            width_m[:, 1],
        ]
    ).astype(np.float32)


def _output_joint_names(
    args: argparse.Namespace,
    runtime,
) -> list[str]:
    if args.piper:
        return [
            *(f"left_joint{joint}.pos" for joint in range(1, 7)),
            "left_gripper.width_m",
            *(f"right_joint{joint}.pos" for joint in range(1, 7)),
            "right_gripper.width_m",
        ]
    return [f"{name}.pos" for name in runtime.robot.joints.actuated_names]


def process_episode(
    *,
    args: argparse.Namespace,
    states: np.ndarray,
    episode_index: int,
    source_episode_index: int,
    task: str,
) -> Any:
    """Run IK retargeting on one episode and return an EpisodeResult.

    A fresh retargeter is built for each episode so the local-relative
    calibration is relative to that episode's first frame.

    Parameters
    ----------
    args:
        Parsed CLI args.
    states:
        Raw HandUMI states of shape ``(T, 16)``.
    episode_index:
        Index in the *output* dataset.
    source_episode_index:
        Index in the *source* dataset (for video copying).
    task:
        Task description string.

    Returns
    -------
    EpisodeResult
    """
    from handumi.dataset import EpisodeResult

    if len(states) < 2:
        raise ValueError(
            f"Episode {source_episode_index} has fewer than 2 frames; "
            "cannot construct (state, action) pairs."
        )

    if args.retarget_mode == "absolute-table":
        try:
            rollout = _solve_with_replay_pipeline(
                args=args,
                source_episode_index=source_episode_index,
            )
        except SystemExit as exc:
            raise RuntimeError(str(exc)) from exc
        qpos = np.asarray(rollout["qpos"], dtype=np.float32)
        if len(qpos) != len(states):
            raise RuntimeError(
                "Replay-parity solver returned "
                f"{len(qpos)} frames for {len(states)} source frames."
            )
        if getattr(args, "piper", False):
            runtime = load_embodiment("piper")
            joint_array = _piper_command_states_from_rollout(
                rollout,
                actuated_names=runtime.robot.joints.actuated_names,
                gripper_max_width_m=runtime.config.gripper_max_width_m,
            )
        else:
            joint_array = qpos
        args.ik_reports.append(
            {
                "source_episode_index": source_episode_index,
                "qpos_sha256": hashlib.sha256(qpos.tobytes()).hexdigest(),
                "output_state_sha256": hashlib.sha256(
                    joint_array.tobytes()
                ).hexdigest(),
                "frames": len(qpos),
                "retarget_mode": str(rollout["retarget_mode"][0]),
                "max_position_error_m": float(
                    max(
                        np.max(rollout["left_pos_error_m"]),
                        np.max(rollout["right_pos_error_m"]),
                    )
                ),
                "max_rotation_error_deg": float(
                    max(
                        np.max(rollout["left_rot_error_deg"]),
                        np.max(rollout["right_rot_error_deg"]),
                    )
                ),
                "initial_solve_iterations": int(
                    rollout["initial_solve_iterations"][0]
                ),
                "gripper_source": str(rollout["gripper_source"][0]),
            }
        )
    else:
        joint_array, _ = solve_joint_trajectory_from_raw_states(
            args=args,
            states=states,
        )
    states = joint_array[:-1]               # t = 0 … T-2
    actions = joint_array[1:]               # t = 1 … T-1

    return EpisodeResult(
        episode_index=episode_index,
        states=states,
        actions=actions,
        task=task,
        source_episode_index=source_episode_index,
    )


# ---------------------------------------------------------------------------
# Hub upload
# ---------------------------------------------------------------------------


def push_to_hub(
    output_root: Path,
    repo_id: str,
    *,
    token: str | None = None,
) -> None:
    """Push the dataset directory to the HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for --push-to-hub.  "
            "Install it with: pip install huggingface_hub"
        ) from exc

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(output_root),
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Pushed dataset to https://huggingface.co/datasets/{repo_id}")


def _write_converted_dataset_readme(
    output_root: Path,
    *,
    repo_id: str,
    source_repo_id: str,
    embodiment: str,
) -> None:
    """Write a Hub card that identifies the joint-level derivation."""
    from lerobot.datasets.utils import create_lerobot_dataset_card

    info = json.loads((output_root / "meta" / "info.json").read_text())
    handumi_info = info.get("handumi", {})
    layout = str(handumi_info.get("state_layout", "full_sim_qpos"))
    if layout == "bipiper_6dof_plus_gripper_width_m_per_side":
        representation = (
            "Each side stores six replay arm joints in radians and one physical "
            "gripper opening in meters."
        )
    else:
        representation = "The state uses the selected embodiment joint layout."
    card = create_lerobot_dataset_card(
        tags=["HandUMI", embodiment, "joint-level"],
        dataset_info=info,
        license="other",
        repo_id=repo_id,
        dataset_description=(
            f"Joint-level bimanual {embodiment} dataset derived from "
            f"{source_repo_id}. {representation} observation.state[t] contains "
            "the command at t and action[t] contains the command at t+1."
        ),
        url="https://github.com/robonet-ai/handumi-sw",
    )
    card.save(output_root / "README.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _resolve_cli_profile(parser, args)
    args.ik_reports = []

    if args.left_only and args.right_only:
        parser.error("Use only one of --left-only or --right-only.")
    if args.column is not None:
        args.source = args.column

    from handumi.dataset.selection import resolve_dataset_selection

    try:
        selection = resolve_dataset_selection(
            args.dataset,
            repo_id=args.repo_id,
            root=args.root,
            revision=args.revision,
            default_repo_id="NONHUMAN-RESEARCH/handumi-dataset-v2",
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.repo_id = selection.repo_id
    args.root = selection.root

    # ------------------------------------------------------------------
    # Resolve paths and names
    # ------------------------------------------------------------------
    source_repo_id = args.repo_id
    source_root = selection.root
    output_repo_id = args.output_repo_id or _default_output_repo_id(
        source_repo_id, args.embodiment
    )
    output_root = dataset_root_from_repo_id(output_repo_id)

    print(
        "Conversion plan\n"
        f"  Source: {source_root} ({source_repo_id})\n"
        f"  Output: {output_root} ({output_repo_id})\n"
        f"  Robot profile: {args.embodiment}\n"
        f"  Retargeting: {args.retarget_mode}\n"
        f"  Episodes: {args.episodes or 'all'}"
    )
    if args.dry_run:
        return

    # ------------------------------------------------------------------
    # Ensure source metadata is available, then read episode count
    # ------------------------------------------------------------------
    from handumi.dataset import ensure_metadata, validate_raw_state_metadata

    source_info = ensure_metadata(
        repo_id=source_repo_id,
        root=source_root,
        revision=args.revision,
    )
    try:
        validate_raw_state_metadata(source_info)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.raw_controller_debug:
        try:
            args.controller_tcp_selection = _resolve_conversion_tcp_calibration(
                args,
                source_info,
            )
        except (OSError, ValueError) as exc:
            parser.error(f"Could not resolve source Controller->TCP calibration: {exc}")
        print(f"Controller->TCP: {args.controller_tcp_selection.source}")
    total_source_episodes = int(source_info.get("total_episodes", 0))
    dataset_fps = int(source_info.get("fps", 30))

    if total_source_episodes <= 0:
        parser.error(
            f"Could not determine total_episodes from "
            f"{source_root / 'meta' / 'info.json'}. "
            "Check --repo-id, --root, and --revision."
        )

    try:
        episode_indices = _parse_episode_list(args.episodes, total_source_episodes)
    except ValueError as exc:
        parser.error(str(exc))

    print(
        f"Source episodes: {total_source_episodes}  "
        f"Processing: {len(episode_indices)}  "
        f"{episode_indices[:5]}{'…' if len(episode_indices) > 5 else ''}"
    )

    # ------------------------------------------------------------------
    # Load task descriptions for each episode
    # ------------------------------------------------------------------
    source_task_map = _load_source_tasks(source_root)
    default_task = args.task or "task"

    def get_task(ep_idx: int) -> str:
        if args.task:
            return args.task
        return source_task_map.get(ep_idx, default_task)

    # ------------------------------------------------------------------
    # Process each episode
    # ------------------------------------------------------------------
    from handumi.dataset import load_raw_episode
    from handumi.dataset.quality import EpisodeQualityConfig, validate_episode

    results = []
    quality_reports = []
    quality_config = None
    if not args.skip_quality_filter:
        try:
            quality_config = EpisodeQualityConfig.from_yaml(args.quality_config)
        except (OSError, ValueError) as exc:
            parser.error(f"Could not load quality config: {exc}")

    for position, src_idx in enumerate(episode_indices, start=1):
        print(f"\nEpisode {position}/{len(episode_indices)}  (source ep {src_idx})")
        try:
            raw_episode = load_raw_episode(
                repo_id=source_repo_id,
                root=source_root,
                episode=src_idx,
                source=args.source,
                revision=args.revision,
                download_videos=True,
            )
        except Exception as exc:
            print(f"  SKIP: failed to load — {exc}", file=sys.stderr)
            continue

        raw_states = raw_episode.states
        if quality_config is not None:
            report = validate_episode(
                raw_states,
                fps=raw_episode.fps,
                signals=raw_episode.signals,
                episode_index=src_idx,
                config=quality_config,
            )
            quality_reports.append(report)
            if not report.accepted:
                reasons = ", ".join(
                    finding.code
                    for finding in report.findings
                    if finding.severity == "reject"
                )
                print(f"  SKIP: quality filter — {reasons}", file=sys.stderr)
                continue

        task = get_task(src_idx)
        try:
            result = process_episode(
                args=args,
                states=raw_states,
                episode_index=len(results),
                source_episode_index=src_idx,
                task=task,
            )
        except Exception as exc:
            print(f"  SKIP: IK failed — {exc}", file=sys.stderr)
            continue

        results.append(result)
        print(
            f"  Done: {len(result.states)} frames, "
            f"task={result.task!r}"
        )

    if not results:
        print("No episodes processed successfully. Exiting.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Determine joint names and robot_type from the embodiment spec
    # ------------------------------------------------------------------
    runtime = load_embodiment(args.embodiment)
    robot_type = runtime.config.kind
    joint_names = _output_joint_names(args, runtime)

    # ------------------------------------------------------------------
    # Write output dataset
    # ------------------------------------------------------------------
    from handumi.dataset import write_dataset

    write_dataset(
        output_root=output_root,
        source_root=source_root,
        source_info=source_info,
        episodes=results,
        robot_type=robot_type,
        joint_names=joint_names,
        fps=dataset_fps,
        handumi_metadata={
            "conversion_source": args.source,
            "retarget_mode": args.retarget_mode,
            "compose_source": args.compose_source,
            "translation_scale": float(args.translation_scale),
            "replay_qpos_parity": (
                args.retarget_mode == "absolute-table" and not args.piper
            ),
            "replay_arm_qpos_parity": args.retarget_mode == "absolute-table",
            "state_layout": (
                "bipiper_6dof_plus_gripper_width_m_per_side"
                if args.piper
                else "full_sim_qpos"
            ),
            "gripper_representation": (
                {
                    "type": "opening_width",
                    "unit": "m",
                    "max_width_m": float(runtime.config.gripper_max_width_m),
                    "source": "recorded Feetech normalized",
                }
                if args.piper
                else None
            ),
            "deployment_calibration": _deployment_calibration_metadata(args),
            "absolute_orientation": args.absolute_orientation,
            "initial_solve_iterations": int(args.initial_solve_iterations),
            "initial_position_tolerance_m": float(
                args.initial_position_tolerance_m
            ),
            "ik_fidelity": args.ik_reports,
            "controller_device": args.controller_device,
            "raw_controller_debug": bool(args.raw_controller_debug),
            "controller_tcp_calibration": (
                args.controller_tcp_selection.metadata
                if not args.raw_controller_debug
                else None
            ),
            "quality_filter_enabled": not args.skip_quality_filter,
            "quality_source_accepted": sum(
                report.accepted for report in quality_reports
            ),
            "quality_source_rejected": sum(
                not report.accepted for report in quality_reports
            ),
            "converted_source_episodes": len(results),
        },
    )
    _write_converted_dataset_readme(
        output_root,
        repo_id=output_repo_id,
        source_repo_id=source_repo_id,
        embodiment=args.embodiment,
    )

    if quality_config is not None:
        from handumi.dataset.quality import write_quality_report

        write_quality_report(
            output_root / "meta" / "source_quality.json",
            quality_reports,
            config=quality_config,
            dataset=source_repo_id,
        )

    # ------------------------------------------------------------------
    # Optional: push to Hub
    # ------------------------------------------------------------------
    if args.push_to_hub:
        push_to_hub(output_root, output_repo_id, token=args.hub_token)


if __name__ == "__main__":
    main()
