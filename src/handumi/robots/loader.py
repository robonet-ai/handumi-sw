"""Build small robot bundles from YAML-backed robot configs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from handumi.retargeting.handumi_to_robot import raw_state_target_poses
from handumi.robots.registry import load_embodiment


@dataclass(frozen=True)
class DatasetEmbodimentSpec:
    """LeRobot metadata for an IK-converted robot dataset."""

    robot_type: str
    joint_names: list[str]


@dataclass(frozen=True)
class EmbodimentBundle:
    """Runtime objects used by dataset conversion scripts."""

    spec: DatasetEmbodimentSpec
    initial_q: np.ndarray
    retarget_frame: Callable[[np.ndarray, np.ndarray], np.ndarray]
    extract_joints: Callable[[np.ndarray], np.ndarray]


def build_embodiment(
    embodiment: str,
    args,
    first_body_pose: np.ndarray,
) -> EmbodimentBundle:
    """Build a YAML-backed bimanual IK bundle.

    The current software path expects raw HandUMI 16D frames. Older PICO
    skeleton conversion should be migrated to produce that same raw format
    before calling this boundary.
    """
    del first_body_pose
    runtime = load_embodiment(embodiment)
    config = runtime.config.ik_weights
    if hasattr(args, "pos_weight"):
        config = runtime.config_cls(
            pos_weight=float(args.pos_weight),
            ori_weight=float(getattr(args, "ori_weight", config.ori_weight)),
            rest_weight=float(getattr(args, "rest_weight", config.rest_weight)),
        )
    solver = runtime.solver_cls(config=config)
    initial_q = runtime.config.home_q.astype(np.float32).copy()

    def retarget_frame(raw_state: np.ndarray, q_current: np.ndarray) -> np.ndarray:
        left_pose, right_pose = raw_state_target_poses(raw_state)
        return solver.ik(q_current, left_pose=left_pose, right_pose=right_pose)

    def extract_joints(q: np.ndarray) -> np.ndarray:
        return np.asarray(q, dtype=np.float32)

    return EmbodimentBundle(
        spec=DatasetEmbodimentSpec(
            robot_type=runtime.config.kind,
            joint_names=[f"{name}.pos" for name in runtime.robot.joints.actuated_names],
        ),
        initial_q=initial_q,
        retarget_frame=retarget_frame,
        extract_joints=extract_joints,
    )
