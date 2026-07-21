"""Headless MuJoCo physics for the live Viser teleop (real contact/grasp).

Loads the embodiment MJCF (``assets/piper/piper.xml``), attaches a task
scene (``assets/scenes/<name>/scene.xml``) at a configurable offset, and
steps contact physics on a background thread at a fixed real-time rate.
The caller writes actuator setpoints (IK joints + gripper opening) and
reads back the settled joint state and dynamic body poses for rendering —
so what Viser shows is what physics produced, grasping included.

Thread model: one daemon thread owns ``mj_step``; all public methods are
safe to call from the render/control loop (a lock guards MjData).
"""

# mujoco ships no py.typed/stubs (its classes come from native bindings that
# pyright cannot introspect), so attribute checks are meaningless here.
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np

from handumi.sim.scene import SCENES_DIR

log = logging.getLogger(__name__)

try:
    import mujoco  # pyright: ignore[reportMissingImports]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mujoco is required for --scene physics. Install with: uv sync"
    ) from exc


def _build_model(
    mjcf_path: Path, scene_name: str | None, scene_position
) -> tuple["mujoco.MjModel", list[str]]:
    """Compile robot MJCF (+ optional scene attached at ``scene_position``).

    Returns the model and the scene's own top-level body names (kept
    verbatim thanks to ``prefix=""``).
    """
    robot_spec = mujoco.MjSpec.from_file(str(mjcf_path))
    # The Piper link meshes interpenetrate slightly at every joint; with
    # default collision masks those internal contacts friction-lock the
    # joints (the base barely moved under 9kN·m). Put robot geoms in their
    # own collision group so they collide with the floor and the scene but
    # never with each other. (floor: contype=1; scene keeps defaults 1/1;
    # robot: contype=2, conaffinity=1 -> robot-robot 0, robot-world 1.)
    for geom in robot_spec.geoms:
        if geom.name != "floor":
            geom.contype = 2
            geom.conaffinity = 1
    scene_body_names: list[str] = []
    if scene_name is not None:
        scene_xml = SCENES_DIR / scene_name / "scene.xml"
        if not scene_xml.is_file():
            raise FileNotFoundError(f"No scene asset: {scene_xml}")
        scene_spec = mujoco.MjSpec.from_file(str(scene_xml))
        # Every MjSpec has an implicit unnamed root "world" body — skip it.
        scene_body_names = [
            body.name
            for body in scene_spec.bodies
            if body.name and body.name != "world"
        ]
        frame = robot_spec.worldbody.add_frame(pos=list(scene_position))
        robot_spec.attach(scene_spec, frame=frame, prefix="")
    return robot_spec.compile(), scene_body_names


class MujocoPhysics:
    """Background contact physics driven by named actuator setpoints."""

    def __init__(
        self,
        *,
        mjcf_path: Path,
        actuator_names: list[str],
        scene_name: str | None = None,
        scene_position=(0.0, 0.0, 0.0),
        physics_hz: float = 500.0,
    ) -> None:
        self.model, self.scene_body_names = _build_model(
            mjcf_path, scene_name, scene_position
        )
        self.data = mujoco.MjData(self.model)
        self._hz = physics_hz
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._ctrl_index = {
            name: self.model.actuator(name).id for name in actuator_names
        }
        # qpos address per actuated joint (same names as the actuators here).
        self._joint_qpos_adr = {
            name: self.model.joint(name).qposadr[0] for name in actuator_names
        }
        # Freejoint scene bodies: 7 qpos values (xyz + wxyz quat) per body.
        self._body_qpos_adr: dict[str, int] = {}
        for body_name in self.scene_body_names:
            body = self.model.body(body_name)
            if body.jntnum[0] > 0:
                joint_id = body.jntadr[0]
                if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                    self._body_qpos_adr[body_name] = self.model.jnt_qposadr[joint_id]
        mujoco.mj_forward(self.model, self.data)
        self._initial_qpos = self.data.qpos.copy()

    # ---------- lifecycle ----------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info("MuJoCo physics thread started (%.0f Hz).", self._hz)

    def close(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        step_dt = self.model.opt.timestep
        next_t = time.perf_counter()
        while not self._stop.is_set():
            with self._lock:
                mujoco.mj_step(self.model, self.data)
            next_t += step_dt
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:  # fell behind: resync instead of spiralling
                next_t = time.perf_counter()

    # ---------- control / state ----------

    def set_ctrl(self, values: dict[str, float]) -> None:
        """Write actuator setpoints by name (position servos in the MJCF)."""
        with self._lock:
            for name, value in values.items():
                index = self._ctrl_index.get(name)
                if index is not None:
                    self.data.ctrl[index] = value

    def joint_positions(self) -> dict[str, float]:
        """Settled joint positions by (actuated) joint name."""
        with self._lock:
            return {
                name: float(self.data.qpos[adr])
                for name, adr in self._joint_qpos_adr.items()
            }

    def body_pose(self, name: str) -> tuple[np.ndarray, np.ndarray] | None:
        """(position, quaternion wxyz) of a dynamic scene body, else None."""
        adr = self._body_qpos_adr.get(name)
        if adr is None:
            return None
        with self._lock:
            qpos = self.data.qpos[adr : adr + 7].copy()
        return qpos[:3], qpos[3:7]

    def reset(self) -> None:
        """Reset the world (arms + scene props) to the initial state."""
        with self._lock:
            self.data.qpos[:] = self._initial_qpos
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0.0
            mujoco.mj_forward(self.model, self.data)


__all__ = ["MujocoPhysics"]
