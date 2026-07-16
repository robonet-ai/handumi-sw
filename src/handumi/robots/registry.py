"""Robot registry backed only by ``configs/robots/*.yaml``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyroki as pk
import yaml
import yourdfpy

from handumi.robots.kinematics import BimanualKinematicsSolver, KinematicsConfig

if TYPE_CHECKING:
    from handumi.sim.viser_sim import ViserSim

SOURCE_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_ROOT = (
    SOURCE_ROOT if (SOURCE_ROOT / "configs" / "robots").exists() else PACKAGE_ROOT
)
REPO_ROOT = RESOURCE_ROOT  # Backward-compatible name for callers/tests.
CONFIG_DIR = RESOURCE_ROOT / "configs" / "robots"
SIDES: tuple[str, str] = ("left", "right")


def available_robot_names() -> tuple[str, ...]:
    """Return robot names discovered from ``configs/robots/*.yaml``."""

    if not CONFIG_DIR.exists():
        return ()
    return tuple(sorted(path.stem for path in CONFIG_DIR.glob("*.yaml")))


EMBODIMENT_NAMES: tuple[str, ...] = available_robot_names()


@dataclass(frozen=True)
class GripperJointConfig:
    """One robot joint driven by a normalized HandUMI gripper opening."""

    name: str
    closed_value: float = 0.0
    open_value: float | None = None


@dataclass(frozen=True)
class GripperJointRuntime:
    """Resolved gripper joint mapping for normalized HandUMI openings."""

    index: int
    closed_value: float
    open_value: float


@dataclass(frozen=True)
class RobotArmConfig:
    """YAML-declared logical arm mapping."""

    ee_link: str
    joint_names: tuple[str, ...] = ()
    gripper_joints: tuple[GripperJointConfig, ...] = ()


@dataclass(frozen=True)
class RobotRealConfig:
    """Robot defaults for real-hardware teleop.

    Machine-local connection details (CAN ports, camera IDs, Feetech ports)
    stay in ``configs/rig.yaml``; these values describe how this robot should
    be commanded once the local rig has supplied the transport.
    """

    command_rate_hz: float = 100.0
    max_joint_speed_deg_s: float = 180.0
    home_max_joint_speed_deg_s: float = 20.0
    home_timeout_s: float = 30.0
    home_tolerance_deg: float = 3.0
    speed_percent: int = 80
    gripper_effort: int = 1000


@dataclass(frozen=True)
class RobotConfig:
    kind: str
    urdf: Path
    pkg_root: Path
    mjcf: Path | None
    mjcf_joint_map: dict[str, str]
    mjcf_joint_prefix_map: dict[str, str]
    arms: dict[str, RobotArmConfig]
    ee_links: dict[str, str]
    home_q: np.ndarray
    home_poses: dict[str, np.ndarray]
    default_home_pose: str
    ik_weights: KinematicsConfig
    gripper_max_width_m: float
    controller_tcp_calibrations: dict[str, Path]
    handumi_gripper: str | None
    handumi_controller_mount: str | None
    real: RobotRealConfig
    real_options: dict[str, Any]


@dataclass(frozen=True)
class ArmRuntime:
    """Validated arm metadata resolved against the loaded kinematic model."""

    side: str
    ee_link: str
    ee_index: int
    joint_names: tuple[str, ...]
    joint_indices: tuple[int, ...]


@dataclass(frozen=True)
class RobotRuntime:
    """Resolved robot config plus constructors used by scripts."""

    name: str
    config: RobotConfig
    urdf_path: Path
    robot: pk.Robot
    arms: dict[str, ArmRuntime]
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
    # Per side: resolved finger joints. The joint value for a given HandUMI
    # opening is interpolated from ``closed_value`` to ``open_value``.
    finger_joints: dict[str, tuple[GripperJointRuntime, ...]] = None  # type: ignore[assignment]

    @property
    def ee_indices(self) -> tuple[int, int]:
        return tuple(self.arms[side].ee_index for side in SIDES)  # type: ignore[return-value]

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self.robot.joints.actuated_names)

    def arm_joint_names(self, side: str) -> list[str]:
        return list(self.arms[side].joint_names)

    def home_q(self, name: str | None = None) -> np.ndarray:
        """Return a copy of a named safe starting pose."""
        pose_name = name or self.config.default_home_pose
        try:
            return self.config.home_poses[pose_name].astype(np.float32).copy()
        except KeyError as exc:
            available = ", ".join(sorted(self.config.home_poses))
            raise ValueError(
                f"Unknown home pose {pose_name!r} for {self.name}; use {available}."
            ) from exc

    def arm_joint_indices(self, side: str) -> list[int]:
        return list(self.arms[side].joint_indices)

    def set_finger_positions(
        self, q: np.ndarray, normalized: Mapping[str, float]
    ) -> np.ndarray:
        """Write the gripper-finger joint values for a 0-1 opening per side
        into ``q`` (in place) and return it."""
        for side, fingers in (self.finger_joints or {}).items():
            fraction = float(np.clip(normalized.get(side, 0.0), 0.0, 1.0))
            for finger in fingers:
                q[finger.index] = finger.closed_value + (
                    fraction * (finger.open_value - finger.closed_value)
                )
        return q

    def urdf_arm_joint_names(self, *, is_left: bool) -> list[str]:
        """Compatibility accessor for older callers."""
        side = "left" if is_left else "right"
        return self.arm_joint_names(side)

    def command_to_arm_q(self, command: np.ndarray) -> np.ndarray:
        names = self.arm_joint_names("left")
        return np.asarray(command[: len(names)], dtype=float)

    def mjcf_actuator_name(self, urdf_joint_name: str) -> str:
        """Map a URDF joint name to the configured MJCF actuator/joint name."""

        exact = self.config.mjcf_joint_map.get(urdf_joint_name)
        if exact is not None:
            return exact
        for source_prefix, target_prefix in self.config.mjcf_joint_prefix_map.items():
            if urdf_joint_name.startswith(source_prefix):
                return target_prefix + urdf_joint_name[len(source_prefix) :]
        return urdf_joint_name

    def load_urdf(self, *, load_meshes: bool = False) -> yourdfpy.URDF:
        return yourdfpy.URDF.load(
            str(self.urdf_path),
            filename_handler=yourdfpy_handler(self.config.pkg_root),
            mesh_dir=str(self.urdf_path.parent),
            load_meshes=load_meshes,
        )

    def make_sim(
        self,
        *,
        port: int | None = None,
        joint_names: list[str] | None = None,
        default_q: np.ndarray | None = None,
        scene_bodies: list | None = None,
    ) -> "ViserSim":
        from handumi.sim.viser_sim import ViserSim

        return ViserSim(
            urdf_path=self.urdf_path,
            filename_handler=yourdfpy_handler(self.config.pkg_root),
            left_joint_names=self.arm_joint_names("left"),
            right_joint_names=self.arm_joint_names("right"),
            command_size=self.command_size,
            arm_q_fn=lambda command: np.asarray(command, dtype=float),
            joint_names=joint_names or list(self.joint_names),
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
        raise ValueError(
            f"Unsupported robot {name!r}. Expected one of {available_robot_names()}."
        )
    with path.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}

    weights = data.get("ik_weights") or {}
    real = data.get("real") or {}
    urdf = _resolve_path(data["urdf"])
    pkg_root = _resolve_path(data["pkg_root"])
    mjcf = _resolve_path(data["mjcf"]) if data.get("mjcf") else None
    home_q = np.asarray(data.get("home_q") or [], dtype=np.float32)
    default_home_pose = str(data.get("default_home_pose") or "default")
    raw_home_poses = data.get("home_poses") or {}
    home_poses = {
        str(pose_name): np.asarray(values, dtype=np.float32)
        for pose_name, values in raw_home_poses.items()
    }
    if not home_poses:
        home_poses[default_home_pose] = home_q
    elif default_home_pose not in home_poses:
        raise ValueError(
            f"default_home_pose {default_home_pose!r} is not present in home_poses."
        )
    home_q = home_poses[default_home_pose]
    arms = _parse_arms(data)
    controller_tcp_calibrations = {
        str(device): _resolve_path(value)
        for device, value in (data.get("controller_tcp_calibrations") or {}).items()
    }
    handumi_tool = data.get("handumi_tool") or {}
    return RobotConfig(
        kind=str(data.get("kind") or name),
        urdf=urdf,
        pkg_root=pkg_root,
        mjcf=mjcf,
        mjcf_joint_map={
            str(key): str(value)
            for key, value in (data.get("mjcf_joint_map") or {}).items()
        },
        mjcf_joint_prefix_map={
            str(key): str(value)
            for key, value in (data.get("mjcf_joint_prefix_map") or {}).items()
        },
        arms=arms,
        ee_links={side: arm.ee_link for side, arm in arms.items()},
        home_q=home_q,
        home_poses=home_poses,
        default_home_pose=default_home_pose,
        gripper_max_width_m=float(data.get("gripper_max_width_m", 0.08)),
        controller_tcp_calibrations=controller_tcp_calibrations,
        handumi_gripper=(
            str(handumi_tool["gripper"]) if handumi_tool.get("gripper") else None
        ),
        handumi_controller_mount=(
            str(handumi_tool["controller_mount"])
            if handumi_tool.get("controller_mount")
            else None
        ),
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
        real=RobotRealConfig(
            command_rate_hz=float(real.get("command_rate_hz", 100.0)),
            max_joint_speed_deg_s=float(real.get("max_joint_speed_deg_s", 180.0)),
            home_max_joint_speed_deg_s=float(
                real.get("home_max_joint_speed_deg_s", 20.0)
            ),
            home_timeout_s=float(real.get("home_timeout_s", 30.0)),
            home_tolerance_deg=float(real.get("home_tolerance_deg", 3.0)),
            speed_percent=int(real.get("speed_percent", 80)),
            gripper_effort=int(real.get("gripper_effort", 1000)),
        ),
        real_options={str(key): value for key, value in real.items()},
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
    arms = _resolve_arms(name, cfg, robot)
    ee_indices = (arms["left"].ee_index, arms["right"].ee_index)
    arm_joint_indices = {side: list(arms[side].joint_indices) for side in SIDES}
    home_q = cfg.home_q
    if home_q.size == 0:
        home_q = np.zeros(robot.joints.num_actuated_joints, dtype=np.float32)
    if len(home_q) != robot.joints.num_actuated_joints:
        raise ValueError(
            f"{name} home_q has {len(home_q)} values, expected "
            f"{robot.joints.num_actuated_joints}."
        )
    for pose_name, pose_q in cfg.home_poses.items():
        if len(pose_q) != robot.joints.num_actuated_joints:
            raise ValueError(
                f"{name} home pose {pose_name!r} has {len(pose_q)} values, expected "
                f"{robot.joints.num_actuated_joints}."
            )

    class _Solver(BimanualKinematicsSolver):
        def __init__(self, config: KinematicsConfig | None = None) -> None:
            super().__init__(
                robot=robot,
                ee_indices=ee_indices,
                arm_joint_indices=arm_joint_indices,
                home_q=home_q,
                config=config or cfg.ik_weights,
            )

    command_size = max(len(arms[side].joint_names) for side in SIDES)
    finger_joints = _resolve_finger_joints(urdf, robot, cfg, arms)
    return RobotRuntime(
        name=name,
        config=cfg,
        urdf_path=cfg.urdf,
        robot=robot,
        arms=arms,
        solver_cls=_Solver,
        command_size=command_size,
        default_port=8002 if name == "axol" else 8003,
        finger_joints=finger_joints,
    )


def resolve_home_q(
    runtime: RobotRuntime,
    *,
    rig_config: Path | None = None,
    explicit_name: str | None = None,
) -> tuple[str, np.ndarray]:
    """Resolve CLI, machine-local, then embodiment default home selection."""
    name = explicit_name
    if name is None and rig_config is not None and rig_config.exists():
        with rig_config.open("r", encoding="utf-8") as handle:
            data: dict[str, Any] = yaml.safe_load(handle) or {}
        local = ((data.get("robots") or {}).get(runtime.name) or {}).get("home_pose")
        if local:
            name = str(local)
    name = name or runtime.config.default_home_pose
    return name, runtime.home_q(name)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else RESOURCE_ROOT / path


def _parse_arms(data: dict[str, Any]) -> dict[str, RobotArmConfig]:
    arms_data = data.get("arms")
    if arms_data is None:
        legacy_ee_links = data.get("ee_links")
        if not isinstance(legacy_ee_links, dict):
            raise ValueError("Robot config must define arms or legacy ee_links.")
        arms_data = {side: {"ee_link": legacy_ee_links[side]} for side in SIDES}
    if not isinstance(arms_data, dict):
        raise ValueError("arms must be a mapping.")

    arms: dict[str, RobotArmConfig] = {}
    for side in SIDES:
        raw_arm = arms_data.get(side)
        if not isinstance(raw_arm, dict):
            raise ValueError(f"arms.{side} must be a mapping.")
        ee_link = raw_arm.get("ee_link")
        if not ee_link:
            raise ValueError(f"arms.{side}.ee_link is required.")
        joint_names = tuple(str(name) for name in (raw_arm.get("joint_names") or ()))
        arms[side] = RobotArmConfig(
            ee_link=str(ee_link),
            joint_names=joint_names,
            gripper_joints=_parse_gripper_joints(raw_arm.get("gripper_joints")),
        )
    return arms


def _parse_gripper_joints(value: Any) -> tuple[GripperJointConfig, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("gripper_joints must be a list.")
    joints: list[GripperJointConfig] = []
    for item in value:
        if isinstance(item, str):
            joints.append(GripperJointConfig(name=item))
            continue
        if not isinstance(item, dict) or not item.get("name"):
            raise ValueError("Each gripper_joints entry must be a name or mapping.")
        open_value = item.get("open")
        closed_value = item.get("closed", 0.0)
        joints.append(
            GripperJointConfig(
                name=str(item["name"]),
                closed_value=float(closed_value),
                open_value=None if open_value is None else float(open_value),
            )
        )
    return tuple(joints)


def _resolve_arms(
    name: str, cfg: RobotConfig, robot: pk.Robot
) -> dict[str, ArmRuntime]:
    actuated_names = list(robot.joints.actuated_names)
    link_names = list(robot.links.names)
    arms: dict[str, ArmRuntime] = {}
    for side in SIDES:
        arm = cfg.arms[side]
        joint_names = arm.joint_names or tuple(
            joint_name
            for joint_name in actuated_names
            if joint_name.startswith(f"{side}_")
        )
        if not joint_names:
            raise ValueError(
                f"{name}: arms.{side}.joint_names is required because no "
                f"actuated joints start with {side!r}."
            )
        missing_joints = [joint for joint in joint_names if joint not in actuated_names]
        if missing_joints:
            raise ValueError(
                f"{name}: arms.{side}.joint_names not in URDF actuated joints: "
                f"{missing_joints}"
            )
        if arm.ee_link not in link_names:
            raise ValueError(
                f"{name}: arms.{side}.ee_link {arm.ee_link!r} is not a URDF link."
            )
        arms[side] = ArmRuntime(
            side=side,
            ee_link=arm.ee_link,
            ee_index=link_names.index(arm.ee_link),
            joint_names=tuple(joint_names),
            joint_indices=tuple(actuated_names.index(joint) for joint in joint_names),
        )
    return arms


def _resolve_finger_joints(
    urdf: yourdfpy.URDF,
    robot: pk.Robot,
    cfg: RobotConfig,
    arms: dict[str, ArmRuntime],
) -> dict[str, tuple[GripperJointRuntime, ...]]:
    actuated_names = list(robot.joints.actuated_names)
    fingers_by_side: dict[str, tuple[GripperJointRuntime, ...]] = {}
    for side in SIDES:
        configured = cfg.arms[side].gripper_joints
        fingers: list[GripperJointRuntime] = []
        if configured:
            for gripper_joint in configured:
                if gripper_joint.name not in actuated_names:
                    raise ValueError(
                        f"arms.{side}.gripper_joints contains non-actuated joint "
                        f"{gripper_joint.name!r}."
                    )
                open_value = (
                    gripper_joint.open_value
                    if gripper_joint.open_value is not None
                    else _joint_open_value(urdf, gripper_joint.name)
                )
                fingers.append(
                    GripperJointRuntime(
                        index=actuated_names.index(gripper_joint.name),
                        closed_value=gripper_joint.closed_value,
                        open_value=open_value,
                    )
                )
        else:
            for joint_name in arms[side].joint_names:
                joint = urdf.joint_map.get(joint_name)
                if joint is None or joint.type != "prismatic" or joint.limit is None:
                    continue
                fingers.append(
                    GripperJointRuntime(
                        index=actuated_names.index(joint_name),
                        closed_value=0.0,
                        open_value=_joint_open_value(urdf, joint_name),
                    )
                )
        fingers_by_side[side] = tuple(fingers)
    return fingers_by_side


def _joint_open_value(urdf: yourdfpy.URDF, joint_name: str) -> float:
    joint = urdf.joint_map.get(joint_name)
    if joint is None or joint.limit is None:
        raise ValueError(
            f"Cannot infer open value for {joint_name!r}; set open in YAML."
        )
    lower, upper = float(joint.limit.lower), float(joint.limit.upper)
    return upper if abs(upper) >= abs(lower) else lower


__all__ = [
    "ArmRuntime",
    "EMBODIMENT_NAMES",
    "GripperJointConfig",
    "GripperJointRuntime",
    "RobotArmConfig",
    "RobotConfig",
    "RobotRealConfig",
    "RobotRuntime",
    "available_robot_names",
    "load_embodiment",
    "load_robot_config",
    "yourdfpy_handler",
]
