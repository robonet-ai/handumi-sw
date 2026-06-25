"""Generic Viser-based bimanual robot simulation.

All the threading, viser setup, and joint-reordering logic lives here once.
Each embodiment supplies an ``arm_q_fn`` that maps one per-arm command vector
to the URDF actuated-joint sub-vector for that arm (see
``dexumi.robots.<embodiment>.shared.command_to_arm_q``).

Use :func:`~dexumi.robots.registry.load_embodiment` to construct a configured
instance via :meth:`~dexumi.robots.registry.EmbodimentRuntime.make_sim`.
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


class ViserSim:
    """Shared async bimanual simulation backed by a viser web server.

    The interface is intentionally minimal:

    .. code-block:: python

        from dexumi.robots.registry import load_embodiment

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
        port: int,
    ) -> None:
        self._urdf_path = urdf_path
        self._left_joint_names = left_joint_names
        self._right_joint_names = right_joint_names
        self._command_size = command_size
        self._arm_q_fn = arm_q_fn
        self._joint_names = joint_names
        self._default_q = default_q
        self._port = port
        self._latest_q: np.ndarray | None = None
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._last_left: np.ndarray = np.zeros(command_size, dtype=np.float32)
        self._last_right: np.ndarray = np.zeros(command_size, dtype=np.float32)

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

        while True:
            with self._condition:
                self._condition.wait()
                q = self._latest_q
            if q is not None and q.size > 0:
                viser_urdf.update_cfg(_to_viser(np.asarray(q, dtype=float)))
