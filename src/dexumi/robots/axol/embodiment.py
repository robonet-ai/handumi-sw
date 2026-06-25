"""Axol embodiment: spec, adapter, and builder for PICO → joint retargeting."""

from __future__ import annotations

from argparse import Namespace

import numpy as np

from dexumi.robots.loader import EmbodimentBundle, EmbodimentSpec

_LEFT_JOINT_NAMES = [
    "left_shoulder_1.pos",
    "left_shoulder_2.pos",
    "left_shoulder_3.pos",
    "left_elbow.pos",
    "left_wrist_1.pos",
    "left_wrist_2.pos",
    "left_wrist_3.pos",
    "left_gripper.pos",
]

_RIGHT_JOINT_NAMES = [
    "right_shoulder_1.pos",
    "right_shoulder_2.pos",
    "right_shoulder_3.pos",
    "right_elbow.pos",
    "right_wrist_1.pos",
    "right_wrist_2.pos",
    "right_wrist_3.pos",
    "right_gripper.pos",
]

SPEC = EmbodimentSpec(
    robot_type="bi_axol",
    joint_names=_LEFT_JOINT_NAMES + _RIGHT_JOINT_NAMES,
)


class _ExtractAdapter:
    """Wraps PicoToAxolArmRetargeter with an ``extract_joints`` helper."""

    def __init__(self, retargeter: object) -> None:
        self._r = retargeter  # PicoToAxolArmRetargeter

    def retarget_frame(self, body_pose: np.ndarray, q_current: np.ndarray) -> np.ndarray:
        return self._r.retarget_frame(body_pose, q_current)  # type: ignore[union-attr]

    def extract_joints(self, q: np.ndarray) -> np.ndarray:
        solver = self._r.solver  # type: ignore[union-attr]
        left_arm = q[solver.left_indices].astype(np.float32)
        right_arm = q[solver.right_indices].astype(np.float32)
        gripper = np.float32(self._r.gripper)  # type: ignore[union-attr]
        return np.concatenate([left_arm, [gripper], right_arm, [gripper]])


def build_embodiment(args: Namespace, first_body_pose: np.ndarray) -> EmbodimentBundle:
    """Instantiate the Axol solver and retargeter from parsed CLI args."""
    from dexumi.retargeting.axol_from_pico import (
        PicoToAxolArmRetargeter,
        move_retargeter_to_front_workspace,
        settle_first_frame,
    )
    from dexumi.robots.axol.config import KinematicsConfig
    from dexumi.robots.axol.solver import KinematicsSolver

    config = KinematicsConfig(
        pos_weight=args.pos_weight,
        ori_weight=args.ori_weight,
        elbow_weight=args.elbow_weight,
        max_joint_delta=args.max_joint_delta,
        max_reach=args.max_reach,
    )
    solver = KinematicsSolver(config=config)
    retargeter = PicoToAxolArmRetargeter(
        solver=solver,
        first_body_pose=first_body_pose,
        scale=args.scale,
        axis_map=args.axis_map,
        enable_left=not getattr(args, "right_only", False),
        enable_right=not getattr(args, "left_only", False),
        gripper=args.gripper,
    )

    if getattr(args, "axol_workspace", "front") == "front":
        move_retargeter_to_front_workspace(
            retargeter,
            wrist_forward=args.axol_wrist_forward,
            wrist_height=args.axol_wrist_height,
            wrist_lateral=args.axol_wrist_lateral,
            elbow_forward=args.axol_elbow_forward,
            elbow_height=args.axol_elbow_height,
            elbow_lateral=args.axol_elbow_lateral,
        )

    settle_iters = 0 if getattr(args, "axol_workspace", "front") == "rest" else args.settle_iterations
    initial_q = settle_first_frame(retargeter, first_body_pose, settle_iters)
    if getattr(args, "axol_workspace", "front") == "front":
        solver.set_posture_pose(initial_q)

    adapter = _ExtractAdapter(retargeter)
    return EmbodimentBundle(spec=SPEC, _retargeter=adapter, initial_q=initial_q)
