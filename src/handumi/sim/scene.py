"""Small scene primitives rendered by the Viser simulator."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SceneGeom:
    """A simple renderable geometry attached to a scene body frame."""

    kind: str
    size: tuple[float, ...]
    rgba: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    local_position: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    local_quaternion_wxyz: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )


@dataclass(frozen=True)
class SceneBody:
    """A named frame plus one or more local geometries for the Viser scene."""

    name: str
    geoms: tuple[SceneGeom, ...] = ()
    rest_position: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    rest_quaternion_wxyz: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )


__all__ = ["SceneBody", "SceneGeom"]
