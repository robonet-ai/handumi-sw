"""Physically simulated bimanual robot + task scene (real contact via MuJoCo).

Unlike :class:`~handumi.sim.viser_sim.ViserSim` (kinematics-only: it just poses
a URDF wherever it's told), this module steps real rigid-body physics: any
embodiment that ships an actuated MJCF (position servos on every joint —
see ``assets/piper/piper.xml``) can be driven by commanding those actuators
toward IK targets and letting MuJoCo's contact solver handle what happens
next.

Generic over both axes on purpose:

- **Embodiment**: nothing here hard-codes Piper. The MJCF path, per-side
  joint names, and the command -> actuator-position mapping are all passed
  in by :func:`~handumi.robots.registry.load_embodiment` (mirroring how
  :class:`~handumi.sim.viser_sim.ViserSim` is already parameterized). An
  embodiment with no MJCF (e.g. axol today) simply doesn't build a
  :class:`MujocoSim` — see ``EmbodimentRuntime.make_physics``.
- **Task scene**: props (the cube/box today, anything else later) are pure
  MJCF assets under ``assets/scenes/<name>/scene.xml`` (see
  :data:`SceneConfig`), attached into the embodiment's MJCF at a configurable
  offset. Adding a new task is adding a new asset folder, not editing this
  module. Any body with a ``<freejoint>`` is treated as dynamic and its pose
  is read back from physics every frame; bodies without one are static set
  dressing, positioned once. See :class:`SceneBody`/:class:`SceneGeom`.

This runs headless (no window): a background thread steps physics in real
time, and callers read back the resulting joint angles and scene body poses
each frame to push into :class:`~handumi.sim.viser_sim.ViserSim` for rendering,
so the web view at ``http://localhost:<port>`` is unchanged for the user.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml

_logger = logging.getLogger(__name__)

SCENES_DIR = Path(__file__).resolve().parents[3] / "assets" / "scenes"

_GEOM_KIND_BY_MJTGEOM = {
    mujoco.mjtGeom.mjGEOM_BOX: "box",
    mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
    mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
}


@dataclass(frozen=True)
class SceneConfig:
    """Which task scene asset to load, and where to place it.

    ``position`` offsets the whole scene (every body in ``scene.xml``) in
    the embodiment's world frame — the same frame IK targets are solved in.
    """

    name: str | None
    position: tuple[float, float, float]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SceneConfig":
        path = Path(path)
        if not path.exists():
            return cls(name=None, position=(0.0, 0.0, 0.0))
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        scene = data.get("scene") or {}
        name = scene.get("name")
        position = tuple(scene.get("position", (0.0, 0.0, 0.0)))
        return cls(name=name, position=position)


@dataclass(frozen=True)
class SceneGeom:
    """One geom of a scene body, in the body's local frame."""

    kind: str  # "box" | "sphere" | "cylinder" — see _GEOM_KIND_BY_MJTGEOM
    size: tuple[float, float, float]  # MuJoCo half-extent convention
    rgba: tuple[float, float, float, float]
    local_position: tuple[float, float, float]
    local_quaternion_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class SceneBody:
    """One body attached from the scene asset, with its geoms and rest pose."""

    name: str
    dynamic: bool  # True if it has a <freejoint> (physics-driven each frame)
    geoms: tuple[SceneGeom, ...]
    rest_position: tuple[float, float, float]
    rest_quaternion_wxyz: tuple[float, float, float, float]


def _resolve_scene_xml(name: str) -> Path:
    path = SCENES_DIR / name / "scene.xml"
    if not path.is_file():
        raise FileNotFoundError(
            f"No scene asset found for {name!r}; expected {path}. "
            f"Add assets/scenes/{name}/scene.xml to define it."
        )
    return path


def _build_model(
    mjcf_path: Path, scene_config: SceneConfig | None
) -> tuple[mujoco.MjModel, list[str]]:
    """Load the embodiment MJCF and attach the configured scene, if any.

    Returns the compiled model and the list of body names the scene asset
    declared (queried from its own spec before attaching, since names are
    kept verbatim — see the no-prefix ``attach`` call below).
    """
    robot_spec = mujoco.MjSpec.from_file(str(mjcf_path))
    scene_body_names: list[str] = []
    if scene_config is not None and scene_config.name is not None:
        scene_spec = mujoco.MjSpec.from_file(str(_resolve_scene_xml(scene_config.name)))
        # Every MjSpec has an implicit unnamed root "world" body — skip it,
        # we only want the scene's own top-level bodies (box, cube, ...).
        scene_body_names = [
            body.name for body in scene_spec.bodies if body.name and body.name != "world"
        ]
        frame = robot_spec.worldbody.add_frame(pos=list(scene_config.position))
        robot_spec.attach(scene_spec, frame=frame, prefix="")
    return robot_spec.compile(), scene_body_names


def _read_scene_bodies(
    model: mujoco.MjModel, data: mujoco.MjData, body_names: list[str]
) -> list[SceneBody]:
    """Introspect the compiled model for each scene body's geoms and rest pose."""
    bodies = []
    for name in body_names:
        body = model.body(name)
        dynamic = body.jntnum[0] > 0
        geoms = []
        for geom_id in range(body.geomadr[0], body.geomadr[0] + body.geomnum[0]):
            geom = model.geom(geom_id)
            kind = _GEOM_KIND_BY_MJTGEOM.get(mujoco.mjtGeom(geom.type[0]))
            if kind is None:
                _logger.warning(
                    "Scene body %r has an unsupported geom type (%s); skipping "
                    "it for Viser rendering (physics still simulates it).",
                    name,
                    geom.type[0],
                )
                continue
            geoms.append(
                SceneGeom(
                    kind=kind,
                    size=tuple(geom.size.tolist()),
                    rgba=tuple(geom.rgba.tolist()),
                    local_position=tuple(geom.pos.tolist()),
                    local_quaternion_wxyz=tuple(geom.quat.tolist()),
                )
            )
        bodies.append(
            SceneBody(
                name=name,
                dynamic=bool(dynamic),
                geoms=tuple(geoms),
                rest_position=tuple(data.xpos[body.id].tolist()),
                rest_quaternion_wxyz=tuple(data.xquat[body.id].tolist()),
            )
        )
    return bodies


class MujocoSim:
    """Headless real-time physics for one embodiment + optional task scene.

    Runs ``mj_step`` on a background thread at ``physics_hz`` (independent of
    the ~30 Hz teleop control loop); callers push actuator setpoints via
    :meth:`motion_control` and read back the simulated state via
    :meth:`get_arm_qpos` / :meth:`get_body_pose`.
    """

    def __init__(
        self,
        *,
        mjcf_path: Path,
        left_joint_names: list[str],
        right_joint_names: list[str],
        command_to_arm_q_fn: Callable[[np.ndarray], np.ndarray],
        scene_config: SceneConfig | None = None,
        physics_hz: float = 500.0,
    ) -> None:
        self._command_to_arm_q_fn = command_to_arm_q_fn

        model, scene_body_names = _build_model(mjcf_path, scene_config)
        self._model = model
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

        self.scene_bodies: list[SceneBody] = _read_scene_bodies(
            self._model, self._data, scene_body_names
        )

        self._lock = threading.Lock()
        self._physics_dt = 1.0 / physics_hz
        self._thread: threading.Thread | None = None
        self._running = False

        self._left_act_ids = np.array(
            [self._model.actuator(name).id for name in left_joint_names]
        )
        self._right_act_ids = np.array(
            [self._model.actuator(name).id for name in right_joint_names]
        )
        # Same per-side joint order as the actuators above (joint1..8: six
        # revolute arm joints + two prismatic fingers) — this is also the
        # order ViserSim's URDF-side ``robot_order`` uses, so the qpos vector
        # returned by get_arm_qpos() lines up index-for-index with no
        # remapping needed, even though the joint *names* differ (MJCF vs URDF).
        self._left_qpos_adr = np.array(
            [self._model.joint(name).qposadr[0] for name in left_joint_names]
        )
        self._right_qpos_adr = np.array(
            [self._model.joint(name).qposadr[0] for name in right_joint_names]
        )

        # Dynamic scene bodies (freejoint) — qpos address + reset pose, keyed
        # by body name, so any number of props can be tracked generically.
        self._dynamic_qpos_adr: dict[str, int] = {}
        self._dynamic_reset_qpos: dict[str, np.ndarray] = {}
        for scene_body in self.scene_bodies:
            if not scene_body.dynamic:
                continue
            body = self._model.body(scene_body.name)
            joint_id = int(body.jntadr[0])
            adr = int(self._model.joint(joint_id).qposadr[0])
            self._dynamic_qpos_adr[scene_body.name] = adr
            self._dynamic_reset_qpos[scene_body.name] = self._data.qpos[
                adr : adr + 7
            ].copy()

    async def enable(self) -> None:
        """Start the physics thread. No-op after the first call."""
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        _logger.info("MuJoCo physics thread started (%.0f Hz).", 1.0 / self._physics_dt)

    async def disable(self) -> None:
        """No-op — the daemon thread exits when the process ends."""

    async def motion_control(
        self,
        left: np.ndarray | None = None,
        right: np.ndarray | None = None,
    ) -> None:
        """Set actuator targets (position servos) for the next physics steps."""
        with self._lock:
            if left is not None:
                self._data.ctrl[self._left_act_ids] = self._command_to_arm_q_fn(left)
            if right is not None:
                self._data.ctrl[self._right_act_ids] = self._command_to_arm_q_fn(right)

    async def get_arm_qpos(self) -> np.ndarray:
        """Return the actual simulated joint positions, concatenated
        left-then-right in the same per-side joint1..8 order the constructor
        was given — matches the ``robot_order`` layout
        :class:`~handumi.sim.viser_sim.ViserSim` expects."""
        with self._lock:
            left = self._data.qpos[self._left_qpos_adr].copy()
            right = self._data.qpos[self._right_qpos_adr].copy()
        return np.concatenate([left, right]).astype(np.float32)

    async def get_body_pose(self, name: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(position_xyz, quaternion_wxyz)`` for a dynamic scene body,
        or ``None`` if ``name`` isn't a known dynamic body."""
        adr = self._dynamic_qpos_adr.get(name)
        if adr is None:
            return None
        with self._lock:
            qp = self._data.qpos[adr : adr + 7].copy()
        return qp[:3].astype(np.float32), qp[3:7].astype(np.float32)

    async def reset(self) -> None:
        """Reset the whole scene (arms to rest, dynamic bodies to their start pose)."""
        with self._lock:
            mujoco.mj_resetData(self._model, self._data)
            for name, adr in self._dynamic_qpos_adr.items():
                self._data.qpos[adr : adr + 7] = self._dynamic_reset_qpos[name]
            mujoco.mj_forward(self._model, self._data)

    def _run(self) -> None:
        next_tick = time.perf_counter()
        while self._running:
            with self._lock:
                mujoco.mj_step(self._model, self._data)
            next_tick += self._physics_dt
            remaining = next_tick - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_tick = time.perf_counter()  # fell behind; resync instead of spinning

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
