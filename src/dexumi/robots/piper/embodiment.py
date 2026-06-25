"""Piper embodiment: spec, adapter, and builder for PICO → joint retargeting."""

from __future__ import annotations

from argparse import Namespace

import numpy as np

from dexumi.robots.loader import EmbodimentBundle, EmbodimentSpec

_LEFT_JOINT_NAMES = [
    "left_shoulder_pan.pos",
    "left_shoulder_lift.pos",
    "left_elbow_flex.pos",
    "left_forearm_roll.pos",
    "left_wrist_flex.pos",
    "left_wrist_roll.pos",
    "left_gripper.pos",
]

_RIGHT_JOINT_NAMES = [
    "right_shoulder_pan.pos",
    "right_shoulder_lift.pos",
    "right_elbow_flex.pos",
    "right_forearm_roll.pos",
    "right_wrist_flex.pos",
    "right_wrist_roll.pos",
    "right_gripper.pos",
]

SPEC = EmbodimentSpec(
    robot_type="bi_piper_follower",
    joint_names=_LEFT_JOINT_NAMES + _RIGHT_JOINT_NAMES,
)


class _ExtractAdapter:
    """Wraps PicoToPiperArmRetargeter with an ``extract_joints`` helper."""

    def __init__(self, retargeter: object) -> None:
        self._r = retargeter  # PicoToPiperArmRetargeter

    def retarget_frame(self, body_pose: np.ndarray, q_current: np.ndarray) -> np.ndarray:
        return self._r.retarget_frame(body_pose, q_current)  # type: ignore[union-attr]

    def extract_joints(self, q: np.ndarray) -> np.ndarray:
        solver = self._r.solver  # type: ignore[union-attr]
        left_arm = q[solver.left_indices].astype(np.float32)
        right_arm = q[solver.right_indices].astype(np.float32)
        gripper = np.float32(self._r.gripper)  # type: ignore[union-attr]
        return np.concatenate([left_arm, [gripper], right_arm, [gripper]])


def build_embodiment(args: Namespace, first_body_pose: np.ndarray) -> EmbodimentBundle:
    """Instantiate the Piper solver and retargeter from parsed CLI args."""
    from dexumi.retargeting.piper_from_pico import PicoToPiperArmRetargeter
    from dexumi.robots.piper.config import KinematicsConfig
    from dexumi.robots.piper.solver import KinematicsSolver

    config = KinematicsConfig(
        pos_weight=args.pos_weight,
        ori_weight=args.ori_weight,
        elbow_weight=args.elbow_weight,
        max_joint_delta=args.max_joint_delta,
        max_reach=args.max_reach,
    )
    solver = KinematicsSolver(config=config)
    retargeter = PicoToPiperArmRetargeter(
        solver=solver,
        first_body_pose=first_body_pose,
        scale=args.scale,
        axis_map=args.axis_map,
        enable_left=not getattr(args, "right_only", False),
        enable_right=not getattr(args, "left_only", False),
        gripper=args.gripper,
    )

    q_rest = retargeter.q_rest.copy()
    adapter = _ExtractAdapter(retargeter)
    return EmbodimentBundle(spec=SPEC, _retargeter=adapter, initial_q=q_rest)
