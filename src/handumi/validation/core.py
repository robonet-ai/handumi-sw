"""Reference-frame, synchronization, metric, and statistics primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

KNOWN_FRAMES = frozenset({"source", "handumi_world", "table", "mocap", "force_plate"})
SYNC_EVENT_SCHEMA = "handumi_sync_event_v1"


def _rotation_matrix(
    quaternion_xyzw: Sequence[float] | np.ndarray,
) -> np.ndarray:
    q = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("quaternion must be finite and non-zero")
    x, y, z, w = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class FrameTransform:
    target: str
    source: str
    matrix: np.ndarray

    def __post_init__(self) -> None:
        if self.target not in KNOWN_FRAMES or self.source not in KNOWN_FRAMES:
            raise ValueError(f"frames must be one of {sorted(KNOWN_FRAMES)}")
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("transform must be a finite 4x4 matrix")
        if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9):
            raise ValueError("transform must have homogeneous bottom row [0,0,0,1]")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("transform rotation axes are not orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError(
                "transform is reflected/left-handed; right-handed axes required"
            )
        object.__setattr__(self, "matrix", matrix)

    @classmethod
    def from_pose7(
        cls, target: str, source: str, pose7: Sequence[float]
    ) -> "FrameTransform":
        pose = np.asarray(pose7, dtype=np.float64).reshape(7)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = _rotation_matrix(pose[3:])
        matrix[:3, 3] = pose[:3]
        return cls(target, source, matrix)

    def inverse(self) -> "FrameTransform":
        return FrameTransform(self.source, self.target, np.linalg.inv(self.matrix))

    def compose(self, other: "FrameTransform") -> "FrameTransform":
        if self.source != other.target:
            raise ValueError(
                f"cannot compose {self.target}<-{self.source} with {other.target}<-{other.source}"
            )
        return FrameTransform(self.target, other.source, self.matrix @ other.matrix)

    def apply(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.shape[-1] != 3:
            raise ValueError("points must end in xyz")
        flat = values.reshape(-1, 3)
        result = flat @ self.matrix[:3, :3].T + self.matrix[:3, 3]
        return result.reshape(values.shape)


@dataclass(frozen=True)
class SyncEvent:
    sequence: int
    epoch: int
    source_time_ns: int
    host_time_ns: int
    uncertainty_ns: int
    event_type: str
    hardware_channel: str

    def __post_init__(self) -> None:
        if (
            min(
                self.sequence,
                self.epoch,
                self.source_time_ns,
                self.host_time_ns,
                self.uncertainty_ns,
            )
            < 0
        ):
            raise ValueError(
                "sync event sequence, epoch, timestamps, and uncertainty must be non-negative"
            )
        if not self.event_type or not self.hardware_channel:
            raise ValueError("sync event type and hardware channel are required")

    def record(self) -> dict[str, int | str]:
        return {"schema": SYNC_EVENT_SCHEMA, **asdict(self)}


def position_errors(estimate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    estimate_array = np.asarray(estimate, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if estimate_array.shape != reference_array.shape or estimate_array.shape[-1] != 3:
        raise ValueError("position arrays must have identical (...,3) shape")
    errors = np.linalg.norm(estimate_array - reference_array, axis=-1)
    finite = errors[np.isfinite(errors)]
    if finite.size == 0:
        raise ValueError("no finite paired positions")
    return {
        "count": float(finite.size),
        "rmse_m": float(np.sqrt(np.mean(finite**2))),
        "median_m": float(np.median(finite)),
        "p95_m": float(np.percentile(finite, 95)),
        "maximum_m": float(np.max(finite)),
        "bias_m": float(np.mean(finite)),
    }


def orientation_error_deg(
    estimate_xyzw: np.ndarray, reference_xyzw: np.ndarray
) -> np.ndarray:
    estimate = np.asarray(estimate_xyzw, dtype=np.float64)
    reference = np.asarray(reference_xyzw, dtype=np.float64)
    if estimate.shape != reference.shape or estimate.shape[-1] != 4:
        raise ValueError("orientation arrays must have identical (...,4) shape")
    estimate /= np.linalg.norm(estimate, axis=-1, keepdims=True)
    reference /= np.linalg.norm(reference, axis=-1, keepdims=True)
    dots = np.abs(np.sum(estimate * reference, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dots, -1.0, 1.0)))


def classification_metrics(
    predicted: Sequence[bool | int] | np.ndarray,
    reference: Sequence[bool | int] | np.ndarray,
) -> dict[str, float | int]:
    pred = np.asarray(predicted, dtype=bool)
    truth = np.asarray(reference, dtype=bool)
    if pred.shape != truth.shape or pred.size == 0:
        raise ValueError("classification arrays must be non-empty with identical shape")
    tp = int(np.sum(pred & truth))
    tn = int(np.sum(~pred & ~truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": (tp + tn) / pred.size,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


@dataclass(frozen=True)
class DropoutInterval:
    start_time_s: float
    end_time_s: float
    duration_s: float
    reason: str
    recovered: bool


def dropout_intervals(
    times_s: Sequence[float] | np.ndarray, reasons: Sequence[str | None]
) -> tuple[DropoutInterval, ...]:
    times = np.asarray(times_s, dtype=np.float64)
    if len(times) != len(reasons) or len(times) == 0 or np.any(np.diff(times) < 0):
        raise ValueError("times/reasons must be non-empty, equal length, and ordered")
    allowed = {
        "transport_gap",
        "invalid_tracking",
        "relocalization",
        "frame_epoch_change",
        "reference_unavailable",
    }
    intervals: list[DropoutInterval] = []
    start = 0
    while start < len(reasons):
        reason = reasons[start]
        if reason is None:
            start += 1
            continue
        if reason not in allowed:
            raise ValueError(f"unknown dropout reason: {reason}")
        end = start
        while end + 1 < len(reasons) and reasons[end + 1] == reason:
            end += 1
        sample_period = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
        end_time = float(times[end] + sample_period)
        intervals.append(
            DropoutInterval(
                start_time_s=float(times[start]),
                end_time_s=end_time,
                duration_s=end_time - float(times[start]),
                reason=reason,
                recovered=end + 1 < len(reasons) and reasons[end + 1] is None,
            )
        )
        start = end + 1
    return tuple(intervals)


def bootstrap_participant_mean(
    participant_values: Mapping[str, Sequence[float]],
    *,
    seed: int,
    samples: int = 5000,
    confidence: float = 0.95,
    minimum_participants: int = 2,
) -> dict[str, float | int]:
    if len(participant_values) < minimum_participants:
        raise ValueError(f"at least {minimum_participants} participants are required")
    if samples < 100 or not 0 < confidence < 1:
        raise ValueError("samples must be >=100 and confidence must be in (0,1)")
    participant_means = np.asarray(
        [
            np.nanmean(np.asarray(values, dtype=np.float64))
            for values in participant_values.values()
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(participant_means)):
        raise ValueError("every participant requires at least one finite observation")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(participant_means), size=(samples, len(participant_means))
    )
    distribution = np.mean(participant_means[indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "participant_count": len(participant_means),
        "observation_count": sum(len(values) for values in participant_values.values()),
        "mean": float(np.mean(participant_means)),
        "ci_low": float(np.quantile(distribution, alpha)),
        "ci_high": float(np.quantile(distribution, 1.0 - alpha)),
        "confidence": confidence,
        "seed": seed,
        "bootstrap_samples": samples,
    }


def availability(mask: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(mask, dtype=bool)
    if values.size == 0:
        raise ValueError("availability mask cannot be empty")
    return {
        "available": int(np.sum(values)),
        "total": int(values.size),
        "fraction": float(np.mean(values)),
    }


def jitter(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2 or array.shape[0] < 2:
        raise ValueError("jitter requires at least two samples")
    return float(np.sqrt(np.mean(np.var(array, axis=0))))


def temporal_offset_s(
    estimate: Sequence[float] | np.ndarray,
    reference: Sequence[float] | np.ndarray,
    sample_period_s: float,
) -> float:
    left = np.asarray(estimate, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if (
        left.shape != right.shape
        or left.ndim != 1
        or len(left) < 3
        or sample_period_s <= 0
    ):
        raise ValueError(
            "temporal offset inputs must be equal 1D arrays with >=3 samples"
        )
    correlation = np.correlate(
        left - np.mean(left), right - np.mean(right), mode="full"
    )
    lag = int(np.argmax(correlation) - (len(right) - 1))
    return lag * sample_period_s
