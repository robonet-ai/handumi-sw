"""Build embodiment bundles for dataset conversion scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from handumi.robots.registry import load_embodiment


@dataclass(frozen=True)
class DatasetEmbodimentSpec:
    """LeRobot metadata for an IK-converted embodiment dataset."""

    robot_type: str
    joint_names: list[str]


@dataclass(frozen=True)
class EmbodimentBundle:
    """Runtime objects used by ``process_umi_to_lerobot.py``."""

    spec: DatasetEmbodimentSpec
    initial_q: np.ndarray
    retarget_frame: Callable[[np.ndarray, np.ndarray], np.ndarray]
    extract_joints: Callable[[np.ndarray], np.ndarray]


def _build_config(runtime: Any, args: Any) -> Any:
    kwargs = {
        "pos_weight": args.pos_weight,
        "ori_weight": args.ori_weight,
        "posture_weight": args.elbow_weight,
        "max_joint_delta": args.max_joint_delta,
        "max_reach": args.max_reach,
    }
    return runtime.config_cls(**kwargs)


def _build_piper_bundle(args: Any, first_body_pose: np.ndarray) -> EmbodimentBundle:
    from handumi.robots.piper.shared import (
        LEROBOT_JOINT_NAMES,
        solver_q_to_robot_state,
    )

    runtime = load_embodiment("piper")
    solver = runtime.solver_cls(config=_build_config(runtime, args))
    retargeter = runtime.retargeter_cls(
        solver=solver,
        first_body_pose=first_body_pose,
        scale=args.scale,
        axis_map=args.axis_map or runtime.default_axis_map,
        enable_left=not args.right_only,
        enable_right=not args.left_only,
        gripper=args.gripper,
    )
    initial_q = runtime.settle_first_frame(retargeter, first_body_pose, 0)

    def extract_joints(q: np.ndarray) -> np.ndarray:
        return solver_q_to_robot_state(
            q,
            left_indices=solver.left_indices,
            right_indices=solver.right_indices,
            gripper=retargeter.gripper,
        )

    return EmbodimentBundle(
        spec=DatasetEmbodimentSpec(
            robot_type="bi_piper_follower",
            joint_names=list(LEROBOT_JOINT_NAMES),
        ),
        initial_q=initial_q,
        retarget_frame=retargeter.retarget_frame,
        extract_joints=extract_joints,
    )


def _build_axol_bundle(args: Any, first_body_pose: np.ndarray) -> EmbodimentBundle:
    runtime = load_embodiment("axol")
    solver = runtime.solver_cls(config=_build_config(runtime, args))
    retargeter = runtime.retargeter_cls(
        solver=solver,
        first_body_pose=first_body_pose,
        scale=args.scale,
        axis_map=args.axis_map or runtime.default_axis_map,
        enable_left=not args.right_only,
        enable_right=not args.left_only,
        gripper=args.gripper,
    )

    workspace = getattr(args, "axol_workspace", runtime.default_workspace)
    if workspace == "front":
        runtime.move_to_front_workspace(
            retargeter,
            wrist_forward=args.axol_wrist_forward,
            wrist_height=args.axol_wrist_height,
            wrist_lateral=args.axol_wrist_lateral,
            elbow_forward=args.axol_elbow_forward,
            elbow_height=args.axol_elbow_height,
            elbow_lateral=args.axol_elbow_lateral,
        )

    initial_q = runtime.settle_first_frame(
        retargeter,
        first_body_pose,
        args.settle_iterations if workspace == "front" else 0,
    )
    if workspace == "front":
        solver.set_posture_pose(initial_q)

    def extract_joints(q: np.ndarray) -> np.ndarray:
        left, right = retargeter.split_for_sim(q)
        return np.concatenate([left, right]).astype(np.float32)

    joint_names = (
        [f"{name}.pos" for name in runtime.urdf_arm_joint_names(is_left=True)]
        + ["left_gripper.pos"]
        + [f"{name}.pos" for name in runtime.urdf_arm_joint_names(is_left=False)]
        + ["right_gripper.pos"]
    )

    return EmbodimentBundle(
        spec=DatasetEmbodimentSpec(robot_type="bi_axol", joint_names=joint_names),
        initial_q=initial_q,
        retarget_frame=retargeter.retarget_frame,
        extract_joints=extract_joints,
    )


def build_embodiment(
    embodiment: str,
    args: Any,
    first_body_pose: np.ndarray,
) -> EmbodimentBundle:
    """Build the selected embodiment bundle for dataset conversion."""
    if embodiment == "piper":
        return _build_piper_bundle(args, first_body_pose)
    if embodiment == "axol":
        return _build_axol_bundle(args, first_body_pose)
    raise ValueError(f"Unsupported embodiment: {embodiment!r}")
