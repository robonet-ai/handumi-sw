"""Public re-exports for the Piper embodiment.

Typical usage::

    from dexumi.robots.registry import load_embodiment

    runtime = load_embodiment("piper")
    solver = runtime.solver_cls()
    sim = runtime.make_sim()
"""

from dexumi.robots.kinematics import KinematicsConfig

from .solver import KinematicsSolver


def Sim(**kwargs):
    """Build a Viser sim for Piper (backward-compatible factory)."""
    from dexumi.robots.registry import load_embodiment

    return load_embodiment("piper").make_sim(**kwargs)


__all__ = ["KinematicsConfig", "KinematicsSolver", "Sim"]
