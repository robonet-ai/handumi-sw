"""Viser-based Piper simulation for offline visualization."""

from __future__ import annotations

import logging
import threading

import numpy as np

from .shared import (
    ARM_JOINT_COUNT,
    COMMAND_SIZE,
    GRIPPER_INDEX,
    URDF_PATH,
    gripper_to_finger_positions,
    urdf_arm_joint_names,
)

_logger = logging.getLogger(__name__)


try:
    import viser
    import yourdfpy
    from viser.extras import ViserUrdf
except ImportError as e:
    raise ImportError(
        "viser is required for simulation. Install project dependencies with: uv sync"
    ) from e


class Sim:
    """Viser-based dual Piper robot simulation.

    Implements the minimal async ``enable/get_positions/motion_control`` surface
    needed for visualising joint motion without hardware.

    Each arm command is shape ``(8,)``: indices ``0..5`` are the six revolute
    arm joints in radians, index ``6`` is unused, and index ``7`` is the gripper
    opening normalized to ``[0, 1]`` (0 = closed, 1 = fully open).

    Example::

        sim = Sim(port=8003)
        await sim.enable()
        await sim.motion_control(
            left=np.zeros(8, dtype=np.float32),
            right=np.zeros(8, dtype=np.float32),
        )
    """

    def __init__(
        self,
        *,
        joint_names: list[str] | None = None,
        default_q: np.ndarray | None = None,
        port: int = 8003,
    ) -> None:
        """Construct the simulation.

        The viser server is not started until :meth:`enable` is called.

        Args:
            joint_names: Ordered list of actuated joint names matching the URDF.
                Defaults to left-then-right ``izq_joint*`` / ``der_joint*`` order.
            default_q: Initial joint configuration in radians/meters. Defaults to zeros.
            port: Port for the viser web server.
        """
        self._joint_names = joint_names
        self._default_q = default_q
        self._port = port
        self._latest_q: np.ndarray | None = None
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._last_left: np.ndarray = np.zeros(COMMAND_SIZE, dtype=np.float32)
        self._last_right: np.ndarray = np.zeros(COMMAND_SIZE, dtype=np.float32)

    async def enable(self) -> None:
        """Start the viser server thread. No-op after the first call."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        _logger.info("Simulation server started at http://localhost:%d", self._port)

    async def disable(self) -> None:
        """No-op — the daemon thread exits when the process ends."""
        pass

    async def get_positions(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return the last commanded joint positions for both arms."""
        return self._last_left.copy(), self._last_right.copy()

    async def motion_control(
        self,
        left: np.ndarray | None = None,
        right: np.ndarray | None = None,
    ) -> None:
        """Update the simulation to the given joint positions."""
        if left is not None:
            self._last_left = np.asarray(left, dtype=np.float32)
        if right is not None:
            self._last_right = np.asarray(right, dtype=np.float32)

        q = self._build_q()
        with self._condition:
            self._latest_q = q
            self._condition.notify()

    def _arm_q(self, command: np.ndarray) -> np.ndarray:
        """Build one arm's URDF joint vector from an 8-element command."""
        finger_a, finger_b = gripper_to_finger_positions(command[GRIPPER_INDEX])
        return np.concatenate(
            [
                command[:ARM_JOINT_COUNT].astype(float),
                np.array([finger_a, finger_b], dtype=float),
            ]
        )

    def _build_q(self) -> np.ndarray:
        """Build the full actuated joint vector: left arm then right arm."""
        return np.concatenate([self._arm_q(self._last_left), self._arm_q(self._last_right)])

    def _run(self) -> None:
        server = viser.ViserServer(port=self._port)

        urdf = yourdfpy.URDF.load(str(URDF_PATH), mesh_dir=str(URDF_PATH.parent))
        viser_urdf = ViserUrdf(
            server,
            urdf_or_path=urdf,
            root_node_name="/robot",
            load_meshes=True,
            load_collision_meshes=False,
        )

        robot_order = (
            self._joint_names or urdf_arm_joint_names(is_left=True)
        ) + urdf_arm_joint_names(is_left=False)

        viser_order = viser_urdf.get_actuated_joint_names()
        viser_to_robot: list[int] = []
        for name in viser_order:
            try:
                viser_to_robot.append(robot_order.index(name))
            except ValueError:
                viser_to_robot.append(-1)

        def _to_viser(q_robot: np.ndarray) -> np.ndarray:
            q_out = np.zeros(len(viser_order), dtype=float)
            for viser_index, robot_index in enumerate(viser_to_robot):
                if robot_index >= 0:
                    q_out[viser_index] = q_robot[robot_index]
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
