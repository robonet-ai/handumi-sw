"""Public re-exports for the Axol embodiment.

Typical usage::

    from dexumi.robots.registry import load_embodiment

    runtime = load_embodiment("axol")
    solver = runtime.solver_cls()
    sim = runtime.make_sim()
"""

from dexumi.robots.kinematics import KinematicsConfig

from .solver import KinematicsSolver


def Sim(**kwargs):
    """Build a Viser sim for Axol (backward-compatible factory)."""
    from dexumi.robots.registry import load_embodiment

    return load_embodiment("axol").make_sim(**kwargs)


__all__ = ["KinematicsConfig", "KinematicsSolver", "Sim"]
