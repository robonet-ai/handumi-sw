"""Live robot follow-along: 16D raw state -> bimanual IK -> Viser (Phase 2B).

Consumes the same 16D HandUMI raw state the recorders emit, so any tracking
source that produces it (Quest today, PICO later) can drive the robot view.
Runs alongside Rerun: Rerun keeps cameras/series/controller trails, Viser
renders the URDF arms following your hands (http://localhost:<port>).

Frame mapping (verified against Piper FK): the dual-Piper URDF world and
``handumi_workspace`` share the same right-handed X-forward / Y-left / Z-up
convention, so positions only need a fixed translation — the workspace origin
is the HMD at reset (neck height) while the robot origin is the arm-base
plate. Orientations get one constant alignment: the gripper-TCP identity
(X-forward) maps to the Piper EE rest orientation (EE Z forward, X down).

The heavy IK stack (JAX/pyroki, ~30s JIT warmup) is imported lazily inside
:class:`RobotFollower`; importing this module stays cheap so the pure
transform below is unit-testable without JAX installed.
"""

from __future__ import annotations

import asyncio
import logging
import webbrowser

import numpy as np

from handumi.retargeting.handumi_to_robot import raw_state_target_poses

log = logging.getLogger("handumi.robot_follow")

# Identity controller orientation (gripper X-forward, workspace frame) -> Piper
# EE rest orientation (EE Z axis forward, X axis down). Columns are the EE
# frame axes expressed in the world frame; right-handed (det = +1).
WRIST_ALIGN = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)

# HandUMI gripper full opening (m) used to normalize widths into [0, 1].
DEFAULT_GRIPPER_MAX_WIDTH_M = 0.08


def raw_state_to_robot_targets(
    state: np.ndarray,
    *,
    z_lift: float,
    x_shift: float = 0.0,
    gripper_max_width_m: float = DEFAULT_GRIPPER_MAX_WIDTH_M,
) -> dict:
    """Map one 16D raw state from ``handumi_workspace`` into robot-world targets.

    Returns ``{"left"/"right": (pos, rot_3x3), "left_grip"/"right_grip": float}``
    with positions translated by ``[x_shift, 0, z_lift]``, orientations
    composed with :data:`WRIST_ALIGN`, and gripper widths normalized to [0, 1].
    Pure numpy — no solver, no JAX.
    """
    (left_pos, left_rot), (right_pos, right_rot) = raw_state_target_poses(state)
    offset = np.array([x_shift, 0.0, z_lift], dtype=np.float32)
    arr = np.asarray(state, dtype=np.float32)
    max_w = max(gripper_max_width_m, 1e-6)
    return {
        "left": (left_pos + offset, left_rot @ WRIST_ALIGN),
        "right": (right_pos + offset, right_rot @ WRIST_ALIGN),
        "left_grip": float(np.clip(arr[14] / max_w, 0.0, 1.0)),
        "right_grip": float(np.clip(arr[15] / max_w, 0.0, 1.0)),
    }


class RobotFollower:
    """Owns the IK solver + Viser sim and advances them one raw state at a time.

    Construction is slow (URDF load + JAX JIT warmup, ~30s on CPU); each
    subsequent :meth:`step` solves in ~10-20ms, well inside a 30 Hz budget.
    """

    def __init__(
        self,
        *,
        embodiment: str = "piper",
        port: int | None = None,
        z_lift: float = 0.55,
        x_shift: float = 0.0,
        gripper_max_width_m: float = DEFAULT_GRIPPER_MAX_WIDTH_M,
        open_browser: bool = True,
    ) -> None:
        from handumi.robots.registry import load_embodiment

        self._z_lift = z_lift
        self._x_shift = x_shift
        self._gripper_max_width_m = gripper_max_width_m

        log.info("Loading %s IK solver (JAX JIT warmup, ~30s on CPU)...", embodiment)
        runtime = load_embodiment(embodiment)
        self._solver = runtime.solver_cls()
        self._command_size = runtime.command_size
        self._gripper_index = runtime.command_size - 1
        self._q = np.zeros(self._solver.num_joints, dtype=np.float32)

        self._sim = runtime.make_sim(port=port)
        self._aio = asyncio.new_event_loop()
        self._aio.run_until_complete(self._sim.enable())
        resolved_port = port if port is not None else runtime.default_port
        url = f"http://localhost:{resolved_port}"
        log.info("Robot view ready: %s", url)
        if open_browser:
            webbrowser.open(url)

    def step(
        self,
        state: np.ndarray,
        *,
        left_tracked: bool,
        right_tracked: bool,
    ) -> None:
        """Solve IK toward the tracked side(s) and push the pose to Viser.

        Untracked sides get no pose target, so the rest cost holds them at the
        current joint angles instead of chasing a frozen/stale pose.
        """
        targets = raw_state_to_robot_targets(
            state,
            z_lift=self._z_lift,
            x_shift=self._x_shift,
            gripper_max_width_m=self._gripper_max_width_m,
        )
        self._q = self._solver.ik(
            self._q,
            left_pose=targets["left"] if left_tracked else None,
            right_pose=targets["right"] if right_tracked else None,
        )

        left_cmd = np.zeros(self._command_size, dtype=np.float32)
        right_cmd = np.zeros(self._command_size, dtype=np.float32)
        left_cmd[: len(self._solver.left_indices)] = self._q[self._solver.left_indices]
        right_cmd[: len(self._solver.right_indices)] = self._q[self._solver.right_indices]
        left_cmd[self._gripper_index] = targets["left_grip"]
        right_cmd[self._gripper_index] = targets["right_grip"]
        self._aio.run_until_complete(
            self._sim.motion_control(left=left_cmd, right=right_cmd)
        )

    def close(self) -> None:
        self._aio.close()
