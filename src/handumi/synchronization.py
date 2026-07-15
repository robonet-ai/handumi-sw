"""Timestamp alignment and sustained sensor-health checks for recording."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from handumi.feetech import GripperSample, GripperWidths, zero_gripper_widths
from handumi.tracking.base import ControllerPairSample, TrackingProvider


@dataclass(frozen=True)
class SynchronizedGripperFrame:
    widths: GripperWidths
    frame: dict[str, np.ndarray]
    healthy_for_gate: bool


@dataclass
class SustainedHealthGate:
    """Track continuous unhealthy intervals independently for each sensor."""

    timeout_s: float
    unhealthy_since_ns: dict[str, int] = field(default_factory=dict)

    def update(
        self, states: dict[str, bool], now_ns: int
    ) -> tuple[list[str], list[str]]:
        recovered: list[str] = []
        timed_out: list[str] = []
        timeout_ns = int(self.timeout_s * 1e9)
        for name, healthy in states.items():
            if healthy:
                if name in self.unhealthy_since_ns:
                    recovered.append(name)
                self.unhealthy_since_ns.pop(name, None)
                continue
            since_ns = self.unhealthy_since_ns.setdefault(name, now_ns)
            if now_ns - since_ns >= timeout_ns:
                timed_out.append(name)
        return recovered, timed_out


def tracking_sample_at(
    tracker: TrackingProvider,
    target_time_ns: int,
) -> ControllerPairSample:
    sampler = getattr(tracker, "sample_at", None)
    return sampler(target_time_ns) if sampler is not None else tracker.latest()


def synchronized_gripper_frame(
    grippers: Any | None,
    *,
    target_time_ns: int,
    record_time_ns: int,
    stale_timeout_s: float,
    max_sync_skew_s: float,
) -> SynchronizedGripperFrame:
    """Select the aperture sample nearest the row target and describe its health."""
    enabled = grippers is not None
    sample: GripperSample | None
    if grippers is None:
        sample = GripperSample(
            widths=zero_gripper_widths(),
            sample_time_ns=target_time_ns,
            sequence=0,
            enabled=False,
        )
    else:
        sample_at = getattr(grippers, "sample_at", None)
        if sample_at is not None:
            sample = sample_at(target_time_ns)
        else:
            started_ns = time.monotonic_ns()
            widths = grippers.read_normalized_widths()
            finished_ns = time.monotonic_ns()
            sample = GripperSample(
                widths=widths,
                sample_time_ns=(started_ns + finished_ns) // 2,
                sequence=0,
            )

    if sample is None:
        sample = GripperSample(
            widths=zero_gripper_widths(),
            sample_time_ns=0,
            sequence=0,
            enabled=enabled,
        )

    missing = sample.sample_time_ns <= 0
    age_ns = (
        np.iinfo(np.int64).max
        if missing
        else max(0, record_time_ns - sample.sample_time_ns)
    )
    sync_error_ns = (
        np.iinfo(np.int64).max
        if missing
        else abs(sample.sample_time_ns - target_time_ns)
    )
    healthy = bool(
        enabled
        and not missing
        and age_ns <= int(stale_timeout_s * 1e9)
        and sync_error_ns <= int(max_sync_skew_s * 1e9)
    )
    frame = {
        "observation.feetech.sample_time_ns": _scalar_int(sample.sample_time_ns),
        "observation.feetech.sequence": _scalar_int(sample.sequence),
        "observation.feetech.healthy": _scalar_int(healthy),
    }
    return SynchronizedGripperFrame(
        widths=sample.widths,
        frame=frame,
        healthy_for_gate=healthy if enabled else True,
    )


def capture_timing_frame(
    target_time_ns: int, record_time_ns: int
) -> dict[str, np.ndarray]:
    return {
        "observation.sync.target_time_ns": _scalar_int(target_time_ns),
        "observation.sync.record_time_ns": _scalar_int(record_time_ns),
    }


def _scalar_int(value: int | bool) -> np.ndarray:
    return np.array([int(value)], dtype=np.int64)
