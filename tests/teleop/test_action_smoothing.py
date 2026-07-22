import numpy as np
import pytest

from handumi.teleop.common import JointActionSmoother


def test_joint_action_smoother_uses_exponential_average():
    smoother = JointActionSmoother(alpha=0.7)
    smoother.reset(np.array([0.0, 1.0], dtype=np.float32))

    actual = smoother.smooth(np.array([1.0, -1.0], dtype=np.float32))

    np.testing.assert_allclose(actual, [0.7, -0.4])


def test_joint_action_smoother_alpha_one_is_passthrough():
    smoother = JointActionSmoother(alpha=1.0)
    target = np.array([1.0, -1.0], dtype=np.float32)

    np.testing.assert_array_equal(smoother.smooth(target), target)


@pytest.mark.parametrize("alpha", (0.0, -0.1, 1.1))
def test_joint_action_smoother_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        JointActionSmoother(alpha=alpha)
