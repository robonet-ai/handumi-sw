"""Robot registry backed only by ``configs/robots/*.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyroki as pk
import yaml
import yourdfpy

from handumi.robots.kinematics import BimanualKinematicsSolver, KinematicsConfig
from handumi.sim.viser_sim import ViserSim

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs" / "robots"
EMBODIMENT_NAMES: tuple[str, ...] = ("axol", "piper")


@dataclass(frozen=True)
class RobotConfig:
    kind: str
    urdf: Path
    pkg_root: Path
    ee_links: dict[str, str]
    home_q: np.ndarray
    ik_weights: KinematicsConfig


@dataclass(frozen=True)
class RobotRuntime:
    """Resolved robot config plus constructors used by scripts."""

    name: str
    config: RobotConfig
    urdf_path: Path
    robot: pk.Robot
    ee_indices: tuple[int, int]
    solver_cls: type
    config_cls: type = KinematicsConfig
    command_size: int = 0
    default_port: int = 8003
    default_axis_map: str = "x,z,y"
    default_compare_axis_maps: tuple[str, ...] = ("x,z,y",)
    default_workspace: str = "rest"
    wrist_forward: float = 0.34
    wrist_height: float = 0.24
    wrist_lateral: float = 0.23

    def urdf_arm_joint_names(self, *, is_left: bool) -> list[str]:
        side = "left" if is_left else "right"
        return [
            name
            for name in self.robot.joints.actuated_names
            if name.startswith(f"{side}_")
        ]

    def command_to_arm_q(self, command: np.ndarray) -> np.ndarray:
        names = self.urdf_arm_joint_names(is_left=True)
        return np.asarray(command[: len(names)], dtype=float)

    def make_sim(
        self,
        *,
        port: int | None = None,
        joint_names: list[str] | None = None,
        default_q: np.ndarray | None = None,
        scene_bodies: list | None = None,
    ) -> ViserSim:
        return ViserSim(
            urdf_path=self.urdf_path,
            left_joint_names=self.urdf_arm_joint_names(is_left=True),
            right_joint_names=self.urdf_arm_joint_names(is_left=False),
            command_size=self.command_size,
            arm_q_fn=self.command_to_arm_q,
            joint_names=joint_names,
            default_q=default_q,
            scene_bodies=scene_bodies,
            port=self.default_port if port is None else port,
        )

    def make_physics(self, *, scene_config=None):
        del scene_config
        return None


def yourdfpy_handler(pkg_root: str | Path):
    """Resolve ``package://PKG/rest`` relative to a configured package root."""

    root = Path(pkg_root)

    def h(fname: str) -> str:
        if fname.startswith("package://"):
            rest = fname.split("package://", 1)[1]
            direct = root / rest
            if direct.exists():
                return str(direct)
            parts = Path(rest).parts
            if len(parts) >= 2:
                fallback = root / Path(*parts[1:])
                if fallback.exists():
                    return str(fallback)
            return str(direct)
        return fname

    return h


def load_robot_config(name: str) -> RobotConfig:
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise ValueError(f"Unsupported robot {name!r}. Expected config at {path}.")
    with path.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}

    weights = data.get("ik_weights") or {}
    urdf = _resolve_path(data["urdf"])
    pkg_root = _resolve_path(data["pkg_root"])
    home_q = np.asarray(data.get("home_q") or [], dtype=np.float32)
    return RobotConfig(
        kind=str(data.get("kind") or name),
        urdf=urdf,
        pkg_root=pkg_root,
        ee_links=dict(data["ee_links"]),
        home_q=home_q,
        ik_weights=KinematicsConfig(
            pos_weight=float(weights.get("pos", 100.0)),
            ori_weight=float(weights.get("ori", 15.0)),
            rest_weight=float(weights.get("rest", 2.0)),
            posture_weight=float(weights.get("posture", 0.0)),
            manipulability_weight=float(weights.get("manipulability", 0.0)),
            max_joint_delta=(
                None
                if weights.get("max_joint_delta") is None
                else float(weights["max_joint_delta"])
            ),
            max_reach=(
                None
                if weights.get("max_reach") is None
                else float(weights["max_reach"])
            ),
        ),
    )


def load_embodiment(name: str) -> RobotRuntime:
    cfg = load_robot_config(name)
    urdf = yourdfpy.URDF.load(
        str(cfg.urdf),
        filename_handler=yourdfpy_handler(cfg.pkg_root),
        mesh_dir=str(cfg.urdf.parent),
        load_meshes=False,
    )
    robot = pk.Robot.from_urdf(urdf)
    ee_indices = (
        robot.links.names.index(cfg.ee_links["left"]),
        robot.links.names.index(cfg.ee_links["right"]),
    )
    home_q = cfg.home_q
    if home_q.size == 0:
        home_q = np.zeros(robot.joints.num_actuated_joints, dtype=np.float32)
    if len(home_q) != robot.joints.num_actuated_joints:
        raise ValueError(
            f"{name} home_q has {len(home_q)} values, expected "
            f"{robot.joints.num_actuated_joints}."
        )

    class _Solver(BimanualKinematicsSolver):
        def __init__(self, config: KinematicsConfig | None = None) -> None:
            super().__init__(
                robot=robot,
                ee_indices=ee_indices,
                home_q=home_q,
                config=config or cfg.ik_weights,
            )

    command_size = max(
        sum(j.startswith("left_") for j in robot.joints.actuated_names),
        sum(j.startswith("right_") for j in robot.joints.actuated_names),
    )
    return RobotRuntime(
        name=name,
        config=cfg,
        urdf_path=cfg.urdf,
        robot=robot,
        ee_indices=ee_indices,
        solver_cls=_Solver,
        command_size=command_size,
        default_port=8002 if name == "axol" else 8003,
    )


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


__all__ = [
    "EMBODIMENT_NAMES",
    "RobotConfig",
    "RobotRuntime",
    "load_embodiment",
    "load_robot_config",
    "yourdfpy_handler",
]
