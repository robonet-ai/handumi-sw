"""Generic Viser-based bimanual robot simulation.

All the threading, viser setup, and joint-reordering logic lives here once.
Each embodiment supplies an ``arm_q_fn`` that maps one per-arm command vector
to the URDF actuated-joint sub-vector for that arm (see
``handumi.robots.<embodiment>.shared.command_to_arm_q``).

Use :func:`~handumi.robots.registry.load_embodiment` to construct a configured
instance via :meth:`~handumi.robots.registry.EmbodimentRuntime.make_sim`.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

_logger = logging.getLogger(__name__)

try:
    import viser
    import yourdfpy
    from viser.extras import ViserUrdf
except ImportError as e:
    raise ImportError(
        "viser is required for simulation. Install project dependencies with: uv sync"
    ) from e

from handumi.sim.mujoco_sim import SceneBody
from handumi.utils.trajectory import TrajectoryTrail

# Same palette as the Rerun controller trails in live_tracking_quest.py, so
# "which arm" reads the same color in both views.
LEFT_TCP_COLOR = (240, 189, 63)  # mustard gold
RIGHT_TCP_COLOR = (90, 200, 110)  # green

_TCP_SPHERE_RADIUS_M = 0.012
_DEFAULT_TCP_TRAIL_MAX_POINTS = 150


def _rgba_to_viser_color(rgba: tuple[float, float, float, float]) -> tuple[int, int, int]:
    return tuple(int(round(c * 255)) for c in rgba[:3])


def _add_scene_geom(server: "viser.ViserServer", path: str, geom) -> None:
    """Render one :class:`~handumi.sim.mujoco_sim.SceneGeom` as a viser
    primitive, nested under its parent body's frame so local pos/quat compose
    automatically."""
    color = _rgba_to_viser_color(geom.rgba)
    if geom.kind == "box":
        # MuJoCo box size is half-extents; viser add_box wants full dimensions.
        server.scene.add_box(
            path,
            dimensions=tuple(2.0 * s for s in geom.size),
            color=color,
            position=geom.local_position,
            wxyz=geom.local_quaternion_wxyz,
        )
    elif geom.kind == "sphere":
        server.scene.add_icosphere(
            path,
            radius=geom.size[0],
            color=color,
            position=geom.local_position,
        )
    elif geom.kind == "cylinder":
        server.scene.add_cylinder(
            path,
            radius=geom.size[0],
            height=2.0 * geom.size[1],
            color=color,
            position=geom.local_position,
            wxyz=geom.local_quaternion_wxyz,
        )
    else:
        _logger.warning("Unsupported scene geom kind %r at %s; skipping render.", geom.kind, path)


def _add_scene_body(server: "viser.ViserServer", body: SceneBody):
    """Render one :class:`~handumi.sim.mujoco_sim.SceneBody`: a parent
    frame at its rest pose plus one viser primitive per geom. Returns the
    frame handle so dynamic bodies can be repositioned each frame (static
    bodies just keep their rest pose forever)."""
    frame = server.scene.add_frame(
        f"/scene/{body.name}",
        position=body.rest_position,
        wxyz=body.rest_quaternion_wxyz,
        show_axes=False,
    )
    for i, geom in enumerate(body.geoms):
        _add_scene_geom(server, f"/scene/{body.name}/g{i}", geom)
    return frame


class ViserSim:
    """Shared async bimanual simulation backed by a viser web server.

    The interface is intentionally minimal:

    .. code-block:: python

        from handumi.robots.registry import load_embodiment

        sim = load_embodiment("axol").make_sim(port=8002)
        await sim.enable()
        await sim.motion_control(
            left=np.zeros(8, dtype=np.float32),
            right=np.zeros(8, dtype=np.float32),
        )
    """

    def __init__(
        self,
        *,
        urdf_path: Path,
        left_joint_names: list[str],
        right_joint_names: list[str],
        command_size: int,
        arm_q_fn: Callable[[np.ndarray], np.ndarray],
        joint_names: list[str] | None = None,
        default_q: np.ndarray | None = None,
        scene_bodies: list[SceneBody] | None = None,
        tcp_trail_max_points: int = _DEFAULT_TCP_TRAIL_MAX_POINTS,
        port: int,
    ) -> None:
        self._urdf_path = urdf_path
        self._left_joint_names = left_joint_names
        self._right_joint_names = right_joint_names
        self._command_size = command_size
        self._arm_q_fn = arm_q_fn
        self._joint_names = joint_names
        self._default_q = default_q
        self._scene_bodies = scene_bodies or []
        self._port = port
        self._latest_q: np.ndarray | None = None
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._last_left: np.ndarray = np.zeros(command_size, dtype=np.float32)
        self._last_right: np.ndarray = np.zeros(command_size, dtype=np.float32)
        self._latest_body_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._tcp_trail_max_points = tcp_trail_max_points
        self._latest_tcp: dict[str, np.ndarray] = {}

    def _arm_q(self, command: np.ndarray) -> np.ndarray:
        """Map one arm command (shape ``(command_size,)``) to its URDF joint sub-vector."""
        return self._arm_q_fn(command)

    async def enable(self) -> None:
        """Start the viser server thread. No-op after the first call."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        _logger.info("Simulation server started at http://localhost:%d", self._port)

    async def disable(self) -> None:
        """No-op — the daemon thread exits when the process ends."""

    async def get_positions(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return the last commanded joint positions for both arms."""
        return self._last_left.copy(), self._last_right.copy()

    async def motion_control(
        self,
        left: np.ndarray | None = None,
        right: np.ndarray | None = None,
    ) -> None:
        """Push new joint positions to the rendering thread."""
        if left is not None:
            self._last_left = np.asarray(left, dtype=np.float32)
        if right is not None:
            self._last_right = np.asarray(right, dtype=np.float32)

        q = self._build_q()
        with self._condition:
            self._latest_q = q
            self._condition.notify()

    async def set_actual_joint_positions(self, q_robot_order: np.ndarray) -> None:
        """Push already-URDF-order joint positions straight through, bypassing
        ``arm_q_fn`` — used when a physics sim (e.g. MuJoCo) is the source of
        truth for joint state instead of a directly-solved IK command."""
        with self._condition:
            self._latest_q = np.asarray(q_robot_order, dtype=np.float32)
            self._condition.notify()

    async def set_tcp_pose(self, side: str, position: np.ndarray) -> None:
        """Move the TCP marker + append to its trail (``side`` is ``"left"``
        or ``"right"``). Helps eyeball whether calibration/IK is behaving —
        the sphere should trace a path matching the real hand motion."""
        with self._condition:
            self._latest_tcp[side] = np.asarray(position, dtype=np.float32)
            self._condition.notify()

    async def set_body_pose(self, name: str, position: np.ndarray, quaternion_wxyz: np.ndarray) -> None:
        """Move a dynamic scene body prop (no-op if ``name`` wasn't configured
        as a scene body — see :attr:`ViserSim._scene_bodies`)."""
        with self._condition:
            self._latest_body_poses[name] = (
                np.asarray(position, dtype=np.float32),
                np.asarray(quaternion_wxyz, dtype=np.float32),
            )
            self._condition.notify()

    def _build_q(self) -> np.ndarray:
        """Concatenate left-arm and right-arm URDF joint vectors."""
        return np.concatenate([self._arm_q(self._last_left), self._arm_q(self._last_right)])

    def _run(self) -> None:
        """Blocking viser server loop (runs in a daemon thread)."""
        server = viser.ViserServer(port=self._port)

        urdf = yourdfpy.URDF.load(
            str(self._urdf_path), mesh_dir=str(self._urdf_path.parent)
        )
        viser_urdf = ViserUrdf(
            server,
            urdf_or_path=urdf,
            root_node_name="/robot",
            load_meshes=True,
            load_collision_meshes=False,
        )

        # Build the solver-order joint list (left then right).
        # ``_joint_names`` overrides the left-arm default when provided.
        robot_order = (
            self._joint_names or self._left_joint_names
        ) + self._right_joint_names

        viser_order = viser_urdf.get_actuated_joint_names()
        viser_to_robot: list[int] = []
        for name in viser_order:
            try:
                viser_to_robot.append(robot_order.index(name))
            except ValueError:
                viser_to_robot.append(-1)

        def _to_viser(q_robot: np.ndarray) -> np.ndarray:
            q_out = np.zeros(len(viser_order), dtype=float)
            for vi, ri in enumerate(viser_to_robot):
                if ri >= 0:
                    q_out[vi] = q_robot[ri]
            return q_out

        q0 = (
            np.asarray(self._default_q, dtype=float)
            if self._default_q is not None
            else np.zeros(len(robot_order))
        )
        viser_urdf.update_cfg(_to_viser(q0))
        server.scene.add_grid("/grid", width=2.0, height=2.0, position=(0.0, 0.0, 0.0))

        # Scene body frames, keyed by name — only dynamic ones ever get moved
        # after this; static ones (no <freejoint> in the scene asset) just
        # render once at their rest pose.
        body_frames = {
            body.name: _add_scene_body(server, body) for body in self._scene_bodies
        }

        # TCP marker + trail per side — hidden until the first real pose
        # arrives, so nothing shows at the origin before tracking starts.
        tcp_spheres = {
            side: server.scene.add_icosphere(
                f"/tcp/{side}/marker",
                radius=_TCP_SPHERE_RADIUS_M,
                color=color,
                visible=False,
            )
            for side, color in (("left", LEFT_TCP_COLOR), ("right", RIGHT_TCP_COLOR))
        }
        tcp_trails = {side: TrajectoryTrail(self._tcp_trail_max_points) for side in ("left", "right")}
        tcp_trail_handles = {
            side: server.scene.add_line_segments(
                f"/tcp/{side}/trail",
                points=np.zeros((0, 2, 3), dtype=np.float32),
                colors=color,
                line_width=2.0,
            )
            for side, color in (("left", LEFT_TCP_COLOR), ("right", RIGHT_TCP_COLOR))
        }

        @server.on_client_connect
        def _set_initial_camera(client: viser.ClientHandle) -> None:
            # Front-ish, slightly elevated view that frames both arms without
            # needing to manually orbit/zoom on every session.
            client.camera.position = (-1.6, 0.0, 1.1)
            client.camera.look_at = (0.0, 0.0, 0.5)

        while True:
            with self._condition:
                self._condition.wait()
                q = self._latest_q
                body_poses = self._latest_body_poses
                self._latest_body_poses = {}
                tcp_positions = self._latest_tcp
                self._latest_tcp = {}
            if q is not None and q.size > 0:
                viser_urdf.update_cfg(_to_viser(np.asarray(q, dtype=float)))
            for name, (position, quaternion_wxyz) in body_poses.items():
                frame = body_frames.get(name)
                if frame is not None:
                    frame.position = tuple(position.tolist())
                    frame.wxyz = tuple(quaternion_wxyz.tolist())
            for side, position in tcp_positions.items():
                sphere = tcp_spheres.get(side)
                trail = tcp_trails.get(side)
                trail_handle = tcp_trail_handles.get(side)
                if sphere is None or trail is None or trail_handle is None:
                    continue
                sphere.position = tuple(position.tolist())
                sphere.visible = True
                trail.append(position)
                points = trail.points()
                if len(points) >= 2:
                    trail_handle.points = np.stack([points[:-1], points[1:]], axis=1)
