import numpy as np

from handumi.robots.registry import load_embodiment
from handumi.teleop.core import TeleopController


def _pose(x, y, z):
    return np.array([x, y, z, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def test_anchor_at_current_source_starts_from_named_home():
    runtime = load_embodiment("openarmv1")
    home = runtime.home_q("down")
    controller = TeleopController(
        runtime,
        home_q=home,
        enabled_sides=("left", "right"),
        source_world_to_robot_world=np.eye(3, dtype=np.float32),
    )
    sources = {"left": _pose(0.3, 0.2, 1.0), "right": _pose(0.3, -0.2, 1.0)}

    assert controller.anchor(
        sources, {"left": True, "right": True}, ("left", "right")
    ) == (
        "left",
        "right",
    )
    step = controller.step(
        sources,
        {"left": True, "right": True},
        {"left": 0.0, "right": 1.0},
    )

    assert set(step.anchored_sides) == {"left", "right"}
    assert set(step.target_pose7) == {"left", "right"}
    assert np.all(np.isfinite(step.q))


def test_tracking_loss_clears_anchors_and_holds_feedback():
    runtime = load_embodiment("openarmv1")
    controller = TeleopController(
        runtime,
        home_q=runtime.home_q("down"),
        enabled_sides=("right",),
        source_world_to_robot_world=np.eye(3, dtype=np.float32),
    )
    held = runtime.home_q("down")
    held[runtime.arm_joint_indices("right")[0]] = 0.2

    controller.tracking_lost(held)
    step = controller.step(
        {"left": _pose(0, 0, 0), "right": _pose(0, 0, 0)},
        {"left": False, "right": True},
        {"left": 0.0, "right": 0.0},
    )

    assert not controller.active
    np.testing.assert_allclose(
        step.q[runtime.arm_joint_indices("right")],
        held[runtime.arm_joint_indices("right")],
    )
