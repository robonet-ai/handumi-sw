"""Delayed, time-driven joint command playback for real teleoperation."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class JointCommand:
    """One timestamped IK result in the PC monotonic clock domain."""

    time_s: float
    q: np.ndarray
    openings: dict[str, float]


class DelayedJointCommandBuffer:
    """Interpolate timestamped IK results at ``now - delay``.

    The buffer retains the sample immediately before the playback cursor and
    all newer samples.  Joint positions and normalized gripper openings are
    linearly interpolated.  If tracking does not provide the right-hand
    bracket in time, playback safely holds the newest available command.
    """

    def __init__(self, delay_s: float, *, max_commands: int = 128) -> None:
        if delay_s < 0.0:
            raise ValueError("delay_s must be >= 0")
        if max_commands < 2:
            raise ValueError("max_commands must be >= 2")
        self.delay_s = float(delay_s)
        self._commands: deque[JointCommand] = deque(maxlen=max_commands)
        self._lock = threading.Lock()

    def reset(
        self,
        q: np.ndarray,
        openings: Mapping[str, float],
        *,
        time_s: float,
    ) -> None:
        command = self._command(q, openings, time_s)
        with self._lock:
            self._commands.clear()
            self._commands.append(command)

    def push(
        self,
        q: np.ndarray,
        openings: Mapping[str, float],
        *,
        time_s: float,
    ) -> None:
        command = self._command(q, openings, time_s)
        with self._lock:
            if self._commands and command.time_s <= self._commands[-1].time_s:
                if command.time_s == self._commands[-1].time_s:
                    self._commands[-1] = command
                    return
                raise ValueError("joint command timestamps must be monotonic")
            self._commands.append(command)

    def sample(self, now_s: float) -> tuple[np.ndarray, dict[str, float]] | None:
        playback_s = float(now_s) - self.delay_s
        with self._lock:
            if not self._commands:
                return None
            while (
                len(self._commands) >= 3
                and self._commands[1].time_s <= playback_s
            ):
                self._commands.popleft()
            first = self._commands[0]
            if playback_s <= first.time_s or len(self._commands) == 1:
                return first.q.copy(), first.openings.copy()
            second = self._commands[1]
            if playback_s >= second.time_s:
                # With only two commands this is an underflow: holding the
                # newest target is safer than extrapolating operator motion.
                return second.q.copy(), second.openings.copy()
            fraction = (playback_s - first.time_s) / (second.time_s - first.time_s)
            q = first.q + fraction * (second.q - first.q)
            sides = first.openings.keys() | second.openings.keys()
            openings: dict[str, float] = {}
            for side in sides:
                first_value = first.openings.get(side)
                second_value = second.openings.get(side)
                if first_value is None:
                    first_value = second_value
                if second_value is None:
                    second_value = first_value
                assert first_value is not None and second_value is not None
                openings[side] = first_value + fraction * (
                    second_value - first_value
                )
            return q.astype(np.float32), openings

    @staticmethod
    def _command(
        q: np.ndarray,
        openings: Mapping[str, float],
        time_s: float,
    ) -> JointCommand:
        q_value = np.asarray(q, dtype=np.float32).copy()
        if not np.all(np.isfinite(q_value)):
            raise ValueError("joint command contains non-finite values")
        if not np.isfinite(time_s):
            raise ValueError("joint command timestamp must be finite")
        return JointCommand(
            time_s=float(time_s),
            q=q_value,
            openings={side: float(value) for side, value in openings.items()},
        )


class DelayedJointCommandPlayer:
    """Read a delayed command buffer and publish it at a fixed rate."""

    def __init__(
        self,
        write: Callable[[np.ndarray, dict[str, float]], None],
        *,
        command_rate_hz: float,
        delay_s: float,
    ) -> None:
        if command_rate_hz <= 0.0:
            raise ValueError("command_rate_hz must be > 0")
        self.command_rate_hz = float(command_rate_hz)
        self.buffer = DelayedJointCommandBuffer(delay_s)
        self._write = write
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._latest_lock = threading.Lock()
        self._latest: tuple[np.ndarray, dict[str, float]] | None = None

    def start(
        self,
        q: np.ndarray,
        openings: Mapping[str, float],
        *,
        time_s: float | None = None,
    ) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("joint command player is already running")
        self.buffer.reset(
            q,
            openings,
            time_s=time.perf_counter() if time_s is None else time_s,
        )
        self._stop.clear()
        self._error = None
        with self._latest_lock:
            self._latest = None
        self._thread = threading.Thread(
            target=self._run,
            name="handumi-delayed-joint-player",
            daemon=True,
        )
        self._thread.start()

    def push(
        self,
        q: np.ndarray,
        openings: Mapping[str, float],
        *,
        time_s: float,
    ) -> None:
        self.raise_if_failed()
        self.buffer.push(q, openings, time_s=time_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("delayed joint command player failed") from self._error

    def latest(self) -> tuple[np.ndarray, dict[str, float]] | None:
        """Return the last command successfully handed to the output callback."""
        with self._latest_lock:
            if self._latest is None:
                return None
            q, openings = self._latest
            return q.copy(), openings.copy()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        period_s = 1.0 / self.command_rate_hz
        next_tick = time.perf_counter()
        try:
            while not self._stop.is_set():
                command = self.buffer.sample(next_tick)
                if command is not None:
                    self._write(*command)
                    with self._latest_lock:
                        self._latest = (command[0].copy(), command[1].copy())
                next_tick += period_s
                remaining_s = next_tick - time.perf_counter()
                if remaining_s > 0.0:
                    self._stop.wait(remaining_s)
                else:
                    # Do not burst old commands after a scheduler stall.
                    next_tick = time.perf_counter()
        except BaseException as exc:
            self._error = exc
            self._stop.set()


__all__ = [
    "DelayedJointCommandBuffer",
    "DelayedJointCommandPlayer",
    "JointCommand",
]
