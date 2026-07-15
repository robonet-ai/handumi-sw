"""Temporal quality gates for derived canonical body signals."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from handumi.body.model import ComDiagnostic


@dataclass(frozen=True)
class TrajectoryFilterConfig:
    window_size: int = 7
    polynomial_order: int = 2
    max_gap_s: float = 0.2
    max_speed_m_s: float = 8.0

    def __post_init__(self) -> None:
        if self.window_size < 3 or self.window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer of at least 3")
        if not 1 <= self.polynomial_order < self.window_size:
            raise ValueError("polynomial_order must be in [1, window_size)")
        if self.max_gap_s <= 0 or self.max_speed_m_s <= 0:
            raise ValueError("trajectory gap and speed limits must be positive")


@dataclass(frozen=True)
class TrajectoryEstimate:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    velocity_valid: bool
    acceleration_valid: bool
    diagnostic: ComDiagnostic

    @classmethod
    def invalid(cls, diagnostic: ComDiagnostic) -> "TrajectoryEstimate":
        nan3 = np.full(3, np.nan, dtype=np.float64)
        return cls(nan3.copy(), nan3.copy(), nan3.copy(), False, False, diagnostic)


class SmoothedComTrajectory:
    """Causal local-polynomial derivative estimator with strict reset gates."""

    def __init__(self, config: TrajectoryFilterConfig | None = None) -> None:
        self.config = config or TrajectoryFilterConfig()
        self._samples: deque[tuple[int, np.ndarray]] = deque(
            maxlen=self.config.window_size
        )

    def reset(self) -> None:
        self._samples.clear()

    def update(self, time_ns: int, position: np.ndarray) -> TrajectoryEstimate:
        point = np.asarray(position, dtype=np.float64).reshape(3)
        if time_ns <= 0 or not np.all(np.isfinite(point)):
            self.reset()
            return TrajectoryEstimate.invalid(ComDiagnostic.TIMING_INVALID)

        diagnostic = ComDiagnostic.TRAJECTORY_BOUNDARY
        if self._samples:
            previous_time, previous_point = self._samples[-1]
            dt_s = (int(time_ns) - previous_time) / 1e9
            if dt_s <= 0 or dt_s > self.config.max_gap_s:
                self.reset()
                diagnostic = ComDiagnostic.TIMING_INVALID
            elif np.linalg.norm(point - previous_point) / dt_s > self.config.max_speed_m_s:
                self.reset()
                diagnostic = ComDiagnostic.RELOCALIZATION

        self._samples.append((int(time_ns), point.copy()))
        if len(self._samples) < self.config.window_size:
            return TrajectoryEstimate.invalid(diagnostic)

        times_ns = np.asarray([sample[0] for sample in self._samples], dtype=np.int64)
        points = np.stack([sample[1] for sample in self._samples])
        times_s = (times_ns - times_ns[-1]).astype(np.float64) / 1e9
        design = np.vander(
            times_s,
            N=self.config.polynomial_order + 1,
            increasing=True,
        )
        coefficients, _, rank, _ = np.linalg.lstsq(design, points, rcond=None)
        if rank < self.config.polynomial_order + 1:
            self.reset()
            return TrajectoryEstimate.invalid(ComDiagnostic.TIMING_INVALID)

        acceleration = (
            2.0 * coefficients[2]
            if self.config.polynomial_order >= 2
            else np.full(3, np.nan, dtype=np.float64)
        )
        return TrajectoryEstimate(
            position=coefficients[0],
            velocity=coefficients[1],
            acceleration=acceleration,
            velocity_valid=True,
            acceleration_valid=self.config.polynomial_order >= 2,
            diagnostic=ComDiagnostic.VALID,
        )


__all__ = [
    "SmoothedComTrajectory",
    "TrajectoryEstimate",
    "TrajectoryFilterConfig",
]
