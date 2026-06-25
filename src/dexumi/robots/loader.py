"""Embodiment registry: build solver + retargeter by name.

Call :func:`build_embodiment` with an embodiment name and an
``argparse.Namespace`` containing the relevant flags. You get back an
:class:`EmbodimentBundle` that exposes a uniform interface regardless of
which embodiment is selected.

Supported embodiments
---------------------
``axol``
    Bimanual Axol robot (7 DOF per arm + gripper, 16 outputs total).
    See :mod:`dexumi.robots.axol.embodiment`.

``piper``
    Bimanual Piper robot (6 DOF per arm + gripper, 14 outputs total).
    See :mod:`dexumi.robots.piper.embodiment`.
"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class EmbodimentSpec:
    """Static metadata describing the output joint space of an embodiment."""

    robot_type: str
    joint_names: list[str]

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)


@dataclass
class EmbodimentBundle:
    """Solver + retargeter pair ready for per-episode IK.

    Attributes
    ----------
    spec:
        Static embodiment metadata (robot_type, joint_names, …).
    _retargeter:
        The underlying retargeter object.  Do not access directly; use the
        public methods below.
    initial_q:
        Starting joint vector for the first frame of a new episode.
    """

    spec: EmbodimentSpec
    _retargeter: object
    initial_q: np.ndarray

    def retarget_frame(self, body_pose: np.ndarray, q_current: np.ndarray) -> np.ndarray:
        """Run one frame of IK retargeting."""
        return self._retargeter.retarget_frame(body_pose, q_current)  # type: ignore[union-attr]

    def extract_joints(self, q: np.ndarray) -> np.ndarray:
        """Extract the flat motor command from the full joint vector."""
        return self._retargeter.extract_joints(q)  # type: ignore[union-attr]


_Builder = Callable[[Namespace, np.ndarray], EmbodimentBundle]


def _load_builders() -> dict[str, _Builder]:
    from dexumi.robots.axol.embodiment import build_embodiment as build_axol
    from dexumi.robots.piper.embodiment import build_embodiment as build_piper

    return {
        "axol": build_axol,
        "piper": build_piper,
    }


_BUILDERS = _load_builders()

SUPPORTED_EMBODIMENTS: tuple[str, ...] = tuple(_BUILDERS.keys())


def build_embodiment(
    name: str,
    args: Namespace,
    first_body_pose: np.ndarray,
) -> EmbodimentBundle:
    """Build a solver + retargeter bundle for the named embodiment."""
    if name not in _BUILDERS:
        raise ValueError(
            f"Unknown embodiment {name!r}. "
            f"Supported: {', '.join(SUPPORTED_EMBODIMENTS)}"
        )
    return _BUILDERS[name](args, first_body_pose)
