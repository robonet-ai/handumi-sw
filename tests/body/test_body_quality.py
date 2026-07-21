import numpy as np

from handumi.body.model import ComDiagnostic
from handumi.body.quality import SmoothedComTrajectory, TrajectoryFilterConfig


def test_polynomial_filter_recovers_velocity_and_acceleration_after_boundary():
    trajectory = SmoothedComTrajectory(
        TrajectoryFilterConfig(window_size=7, polynomial_order=2)
    )
    result = None
    for index in range(7):
        time_s = index * 0.02
        position = np.array([time_s**2, 2.0 * time_s, 3.0])
        result = trajectory.update(int((1.0 + time_s) * 1e9), position)
        if index < 6:
            assert not result.velocity_valid
            assert result.diagnostic == ComDiagnostic.TRAJECTORY_BOUNDARY
    assert result is not None
    assert result.velocity_valid
    assert result.acceleration_valid
    np.testing.assert_allclose(result.velocity, [0.24, 2.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(result.acceleration, [2.0, 0.0, 0.0], atol=1e-8)


def test_invalid_timing_and_relocalization_reset_history():
    trajectory = SmoothedComTrajectory(
        TrajectoryFilterConfig(window_size=3, max_speed_m_s=1.0)
    )
    trajectory.update(1_000_000_000, np.zeros(3))
    timing = trajectory.update(1_000_000_000, np.zeros(3))
    assert timing.diagnostic == ComDiagnostic.TIMING_INVALID
    assert not timing.velocity_valid

    trajectory.update(2_000_000_000, np.zeros(3))
    jump = trajectory.update(2_010_000_000, np.array([1.0, 0.0, 0.0]))
    assert jump.diagnostic == ComDiagnostic.RELOCALIZATION
    assert not jump.velocity_valid


def test_invalid_position_clears_filter():
    trajectory = SmoothedComTrajectory(TrajectoryFilterConfig(window_size=3))
    trajectory.update(1_000_000_000, np.zeros(3))
    invalid = trajectory.update(1_010_000_000, np.full(3, np.nan))
    assert invalid.diagnostic == ComDiagnostic.TIMING_INVALID
    assert not invalid.acceleration_valid
