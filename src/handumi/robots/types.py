"""Dependency-light robot configuration types.

Keep these declarations importable in the base recording package.  The
source-only PyRoki/JAXLS integration is loaded only when an IK runtime is
actually constructed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KinematicsConfig:
    """Position-dominant IK weights and explicit reach/change limits."""

    pos_weight: float = 100.0
    ori_weight: float = 15.0
    rest_weight: float = 2.0
    posture_weight: float = 0.0
    manipulability_weight: float = 0.0
    max_joint_delta: float | None = None
    max_reach: float | None = None


__all__ = ["KinematicsConfig"]
