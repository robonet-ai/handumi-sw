"""Collision-aware joint-space trajectory planner for Piper."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import jaxls
import numpy as np
import pyroki as pk


@functools.partial(jax.jit, static_argnames=("max_iterations",))
def solve_path_step(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    q_interp: jax.Array,
    q_current: jax.Array,
    rest_weight: float,
    limit_weight: float,
    collision_margin: float,
    collision_weight: float,
    max_iterations: int,
) -> jax.Array:
    """One IK step toward ``q_interp`` with limit + self-collision costs only."""

    JointVar = robot.joint_var_cls
    costs = [
        pk.costs.rest_cost(JointVar(0), rest_pose=q_interp, weight=rest_weight),
        pk.costs.limit_cost(robot, JointVar(0), weight=limit_weight),
        pk.costs.self_collision_cost(
            robot,
            robot_coll,
            JointVar(0),
            margin=collision_margin,
            weight=collision_weight,
        ),
    ]
    var_joints = JointVar(jnp.array([0]))
    initial_vals = jaxls.VarValues.make(
        [var_joints.with_value(q_current[jnp.newaxis, :])]
    )
    problem = jaxls.LeastSquaresProblem(costs, [var_joints])
    solution_vals = problem.analyze().solve(
        initial_vals=initial_vals,
        verbose=False,
        linear_solver="dense_cholesky",
        trust_region=jaxls.TrustRegionConfig(),
        termination=jaxls.TerminationConfig(
            max_iterations=max_iterations,
            cost_tolerance=1e-2,
        ),
    )
    return solution_vals[var_joints][0]


def plan_collision_aware_trajectory(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    q_from: np.ndarray,
    q_to: np.ndarray,
    *,
    speed: float,
    rate: float,
    min_duration: float,
    rest_weight: float = 50.0,
    limit_weight: float = 100.0,
    collision_margin: float = 0.025,
    collision_weight: float = 100.0,
    max_iterations: int = 10,
) -> list[np.ndarray]:
    """Plan a collision-aware joint-space trajectory from ``q_from`` to ``q_to``."""

    q_from = np.asarray(q_from, dtype=np.float32)
    q_to = np.asarray(q_to, dtype=np.float32)
    max_dist = float(np.max(np.abs(q_from - q_to)))
    duration = max(max_dist / speed, min_duration)
    n_steps = max(2, round(duration * rate))

    trajectory: list[np.ndarray] = []
    q = q_from.copy()
    for i in range(n_steps):
        t = (i + 1) / n_steps
        alpha = t * t * (3.0 - 2.0 * t)
        q_interp = (q_from * (1.0 - alpha) + q_to * alpha).astype(np.float32)
        result = solve_path_step(
            robot,
            robot_coll,
            jnp.asarray(q_interp),
            jnp.asarray(q),
            rest_weight,
            limit_weight,
            collision_margin,
            collision_weight,
            max_iterations,
        )
        q = np.asarray(result, dtype=np.float32)
        trajectory.append(q.copy())
    return trajectory
