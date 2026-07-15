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
        --embodiment axol

    # Piper embodiment, custom output repo-id, push to hub afterwards
    handumi-convert \
        --repo-id NONHUMAN-RESEARCH/handumi-dataset-v2 \
        --embodiment piper \
        --output-repo-id NONHUMAN-RESEARCH/my-piper-dataset \
        --push-to-hub
"""

from __future__ import annotations

import argparse
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
        "--repo-id",
        default="NONHUMAN-RESEARCH/handumi-dataset-v2",
        help="HuggingFace repo-id of the source HandUMI dataset.",
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
    ds.add_argument(
        "--preserve-body",
        action="store_true",
        help=(
            "Copy optional aligned observation.body fields and native tracking "
            "sidecars into the derived dataset."
        ),
    )

    # ------------------------------------------------------------------
    # Embodiment selection
    # ------------------------------------------------------------------
    emb = parser.add_argument_group("Embodiment")
    emb.add_argument(
        "--embodiment",
        choices=EMBODIMENT_NAMES,
        default="axol",
        help="Target robot embodiment.",
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
        default=0.08,
        help="HandUMI full opening (m) that maps to the robot gripper fully "
        "open; recorded widths are normalized by this before scaling to the "
        "finger joint range.",
    )
    ik.add_argument("--pos-weight", type=float, default=None)
    ik.add_argument("--ori-weight", type=float, default=None)
    ik.add_argument("--max-joint-delta", type=float, default=None)
    ik.add_argument("--max-reach", type=float, default=None)
    ik.add_argument(
        "--retarget-mode",
        choices=("local-relative", "anchored"),
        default="local-relative",
        help="Retargeting mode shared with replay_in_sim.py.",
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
    max_w = max(float(args.gripper_max_width_m), 1e-6)
    normalized = np.clip(widths_m / max_w, 0.0, 1.0)
    if not np.any(widths_m > 0):
        normalized[:] = float(np.clip(args.gripper, 0.0, 1.0))
        print(f"    no recorded widths — constant gripper opening {args.gripper:.2f}")
    for column, side in enumerate(("left", "right")):
        for joint_index, open_value in runtime.finger_joints.get(side, ()):
            joints[:, joint_index] = normalized[:, column] * open_value


# ---------------------------------------------------------------------------
# Per-episode IK processing
# ---------------------------------------------------------------------------


def process_episode(
    *,
    args: argparse.Namespace,
    states: np.ndarray,
    episode_index: int,
    source_episode_index: int,
    task: str,
    optional_observations: dict[str, np.ndarray] | None = None,
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

    joint_array, _ = solve_joint_trajectory_from_raw_states(args=args, states=states)
    states = joint_array[:-1]               # t = 0 … T-2
    actions = joint_array[1:]               # t = 1 … T-1

    return EpisodeResult(
        episode_index=episode_index,
        states=states,
        actions=actions,
        task=task,
        source_episode_index=source_episode_index,
        optional_observations={
            key: np.asarray(value)[:-1]
            for key, value in (optional_observations or {}).items()
        },
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.left_only and args.right_only:
        parser.error("Use only one of --left-only or --right-only.")
    if args.column is not None:
        args.source = args.column

    # ------------------------------------------------------------------
    # Resolve paths and names
    # ------------------------------------------------------------------
    source_repo_id = args.repo_id
    source_root = Path(args.root) if args.root else dataset_root_from_repo_id(source_repo_id)
    output_repo_id = args.output_repo_id or _default_output_repo_id(
        source_repo_id, args.embodiment
    )
    output_root = dataset_root_from_repo_id(output_repo_id)

    print(f"Source  : {source_root}  ({source_repo_id})")
    print(f"Output  : {output_root}  ({output_repo_id})")
    print(f"Embodiment: {args.embodiment}")

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
                optional_observations=(
                    raw_episode.body.signals
                    if args.preserve_body and raw_episode.body is not None
                    else None
                ),
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
    joint_names = [f"{name}.pos" for name in runtime.robot.joints.actuated_names]

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
            "body_observations_preserved": bool(args.preserve_body),
        },
        preserve_tracking_sidecars=bool(args.preserve_body),
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
