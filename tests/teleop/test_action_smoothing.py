import numpy as np
import pytest

from handumi.teleop.common import TeleopMotionSmoother


def _pose(x, quaternion=(0.0, 0.0, 0.0, 1.0)):
    return np.array([x, 0.0, 0.0, *quaternion], dtype=np.float32)


def test_motion_smoother_filters_tcp_pose_and_joint_command_with_time():
    smoother = TeleopMotionSmoother(time_constant_s=0.1)
    tracked = {"left": True, "right": True}
    first = {"left": _pose(0.0), "right": _pose(0.0)}
    second = {"left": _pose(1.0), "right": _pose(-1.0)}

    smoother.smooth_source_poses(first, tracked, 1_000_000_000)
    actual_pose = smoother.smooth_source_poses(second, tracked, 1_100_000_000)
    smoother.reset(np.array([0.0, 0.0], dtype=np.float32))
    smoother.smooth_joint_command(np.array([0.0, 0.0], dtype=np.float32), 0.0)
    actual_joint = smoother.smooth_joint_command(
        np.array([1.0, -1.0], dtype=np.float32), 0.1
    )

    expected = 1.0 - np.exp(-1.0)
    np.testing.assert_allclose(actual_pose["left"][0], expected)
    np.testing.assert_allclose(actual_pose["right"][0], -expected)
    np.testing.assert_allclose(actual_joint, [expected, -expected])

    smoother = TeleopMotionSmoother(time_constant_s=0.0)
    target = np.array([1.0, -1.0], dtype=np.float32)

    np.testing.assert_array_equal(smoother.smooth_joint_command(target, 0.0), target)


def test_motion_smoother_keeps_anchor_pose_exact_and_uses_short_quaternion_arc():
    smoother = TeleopMotionSmoother(time_constant_s=1.0)
    anchored = _pose(0.3, (0.0, 0.0, 0.0, -1.0))
    smoother.anchor_sources({"left": anchored}, ("left",))

    actual = smoother.smooth_source_poses(
        {"left": anchored}, {"left": True}, 1_000_000_000
    )

    np.testing.assert_allclose(actual["left"], anchored)


def test_motion_smoother_deadband_holds_small_translation_and_rotation_jitter():
    smoother = TeleopMotionSmoother(
        time_constant_s=0.0,
        position_deadband_m=0.001,
        orientation_deadband_rad=np.deg2rad(1.0),
    )
    initial = _pose(0.0)
    half_angle = np.deg2rad(0.25)
    jitter = _pose(0.0005, (0.0, 0.0, np.sin(half_angle), np.cos(half_angle)))

    smoother.smooth_source_poses(
        {"left": initial}, {"left": True}, 1_000_000_000
    )
    actual = smoother.smooth_source_poses(
        {"left": jitter}, {"left": True}, 1_100_000_000
    )

    np.testing.assert_allclose(actual["left"], initial)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"time_constant_s": -0.1}, "time_constant_s"),
        ({"position_deadband_m": -0.1}, "position_deadband_m"),
        ({"orientation_deadband_rad": -0.1}, "orientation_deadband_rad"),
    ),
)
def test_motion_smoother_rejects_negative_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TeleopMotionSmoother(**kwargs)
