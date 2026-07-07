"""Rolling 3D position buffer, shared by any live view that draws a trail.

Used by ``handumi.scripts.live_tracking_quest`` for the Rerun controller
trails, and by ``handumi.sim.viser_sim`` for the Viser TCP trail.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import numpy.typing as npt


class TrajectoryTrail:
    """Rolling buffer of recent 3D positions for one tracked point."""

    def __init__(self, max_points: int) -> None:
        self._points: deque[np.ndarray] = deque(maxlen=max(1, max_points))

    def append(self, position: npt.ArrayLike) -> None:
        self._points.append(np.asarray(position, dtype=np.float32).reshape(3))

    def clear(self) -> None:
        self._points.clear()

    def points(self) -> np.ndarray:
        if not self._points:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(self._points, dtype=np.float32)
