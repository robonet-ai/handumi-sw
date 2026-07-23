"""Fixed-rate joint streamer infrastructure shared by all arm drivers."""

from __future__ import annotations

import threading

import numpy as np


class AccelerationLimitedJointTrajectory:
    """Online latest-target trajectory with velocity and acceleration limits.

    The generator is deliberately not a waypoint queue. A new target replaces
    the previous one and the next sample continues from the current position
    and velocity, which is the desired behaviour for live teleoperation.
    """

    def __init__(
        self,
        position: np.ndarray,
        *,
        sample_rate_hz: float,
        max_velocity: float,
        max_acceleration: float,
    ) -> None:
        if sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be > 0")
        if max_velocity <= 0.0:
            raise ValueError("max_velocity must be > 0")
        if max_acceleration <= 0.0:
            raise ValueError("max_acceleration must be > 0")
        self.dt = 1.0 / float(sample_rate_hz)
        self.max_velocity = float(max_velocity)
        self.max_acceleration = float(max_acceleration)
        self.position = np.asarray(position, dtype=np.float64).copy()
        self.velocity = np.zeros_like(self.position)

    def set_limits(
        self,
        *,
        max_velocity: float | None = None,
        max_acceleration: float | None = None,
    ) -> None:
        if max_velocity is not None:
            if max_velocity <= 0.0:
                raise ValueError("max_velocity must be > 0")
            self.max_velocity = float(max_velocity)
        if max_acceleration is not None:
            if max_acceleration <= 0.0:
                raise ValueError("max_acceleration must be > 0")
            self.max_acceleration = float(max_acceleration)

    def reset(self, position: np.ndarray) -> None:
        self.position = np.asarray(position, dtype=np.float64).copy()
        self.velocity = np.zeros_like(self.position)

    def step(self, target: np.ndarray) -> np.ndarray:
        """Return the next fixed-rate sample toward the latest target."""
        target_f = np.asarray(target, dtype=np.float64)
        if target_f.shape != self.position.shape:
            raise ValueError(
                f"target shape {target_f.shape} does not match {self.position.shape}"
            )

        distance = target_f - self.position
        # The stopping-speed envelope starts braking early enough to arrive
        # without deliberately overshooting the latest target.
        stopping_speed = np.sqrt(2.0 * self.max_acceleration * np.abs(distance))
        desired_velocity = np.sign(distance) * np.minimum(
            self.max_velocity, stopping_speed
        )
        velocity_delta = np.clip(
            desired_velocity - self.velocity,
            -self.max_acceleration * self.dt,
            self.max_acceleration * self.dt,
        )
        next_velocity = np.clip(
            self.velocity + velocity_delta,
            -self.max_velocity,
            self.max_velocity,
        )
        displacement = 0.5 * (self.velocity + next_velocity) * self.dt
        arrived = (np.abs(displacement) >= np.abs(distance)) | (
            np.abs(distance) < 1e-12
        )
        self.position = np.where(arrived, target_f, self.position + displacement)
        self.velocity = np.where(arrived, 0.0, next_velocity)
        return self.position.copy()


def step_toward(
    current: np.ndarray,
    target: np.ndarray,
    max_step: float,
) -> np.ndarray:
    """Clamp each joint's delta to *max_step* per tick toward *target*.

    Returns a float64 array.  The caller is responsible for any further
    dtype casting (e.g. rounding to int64 for milli-degree arms).
    If *max_step* <= 0 the target is returned immediately.
    """
    current_f = np.asarray(current, dtype=np.float64)
    target_f = np.asarray(target, dtype=np.float64)
    if max_step <= 0.0:
        return target_f.copy()
    return current_f + np.clip(target_f - current_f, -max_step, max_step)


class JointStreamer:
    """Daemon thread + lock + error bookkeeping for fixed-rate arm streamers.

    Subclasses must implement :meth:`_run`.  All shared infrastructure
    (threading.Thread, threading.Lock, threading.Event, error propagation)
    lives here so individual arm drivers only contain arm-specific logic.

    Typical subclass pattern::

        class MyStreamer(JointStreamer):
            def __init__(self, arms, *, command_rate_hz, ...):
                super().__init__(command_rate_hz=command_rate_hz, thread_name="my-streamer")
                self.arms = arms
                ...

            def _run(self) -> None:
                period = 1.0 / self.command_rate_hz
                ...
    """

    def __init__(self, *, command_rate_hz: float, thread_name: str) -> None:
        if command_rate_hz <= 0.0:
            raise ValueError("command_rate_hz must be > 0")
        self.command_rate_hz = float(command_rate_hz)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=thread_name,
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"{type(self).__name__} failed") from self._error

    def _run(self) -> None:
        raise NotImplementedError


__all__ = ["AccelerationLimitedJointTrajectory", "JointStreamer", "step_toward"]
