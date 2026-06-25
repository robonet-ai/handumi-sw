"""Public re-exports for the Axol embodiment.

Typical usage::

    from dexumi.robots.axol import KinematicsConfig, KinematicsSolver, Sim
"""

from dexumi.robots.kinematics import KinematicsConfig
from .sim import Sim
from .solver import KinematicsSolver

__all__ = ["KinematicsConfig", "KinematicsSolver", "Sim"]
