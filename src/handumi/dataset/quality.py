"""Offline quality checks for raw HandUMI recording episodes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from handumi.dataset.raw import (
    LEFT_GRIPPER_INDEX,
    LEFT_POSE_SLICE,
    RIGHT_GRIPPER_INDEX,
    RIGHT_POSE_SLICE,
)

Severity = Literal["reject", "warning"]


@dataclass(frozen=True)
class EpisodeQualityConfig:
    min_duration_s: float = 1.0
    max_bad_tracking_fraction: float = 0.01
    max_bad_sensor_fraction: float = 0.01
    max_bad_sync_fraction: float = 0.01
    max_sync_error_ms: float = 60.0
    max_translation_speed_m_s: float = 5.0
    max_rotation_step_deg: float = 90.0
    max_pose_freeze_s: float = 1.0
    max_signal_freeze_s: float = 0.5
    freeze_translation_epsilon_m: float = 1e-5
    freeze_rotation_epsilon_deg: float = 0.02
    aperture_range_epsilon_mm: float = 0.05
    reject_single_side_pose_freeze: bool = False
    reject_aperture_freeze: bool = False

    def __post_init__(self) -> None:
        for name in (
            "max_bad_tracking_fraction",
            "max_bad_sensor_fraction",
            "max_bad_sync_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")
        for name in (
            "min_duration_s",
            "max_sync_error_ms",
            "max_translation_speed_m_s",
            "max_rotation_step_deg",
            "max_pose_freeze_s",
            "max_signal_freeze_s",
        ):
            value = float(getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be greater than zero, got {value}.")
        for name in (
            "freeze_translation_epsilon_m",
            "freeze_rotation_epsilon_deg",
            "aperture_range_epsilon_mm",
        ):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EpisodeQualityConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        values = data.get("quality", data) or {}
        known = {item.name for item in cls.__dataclass_fields__.values()}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown quality settings: {', '.join(unknown)}")
        return cls(**values)


@dataclass(frozen=True)
class QualityFinding:
    code: str
    severity: Severity
    message: str
    metrics: dict[str, float | int | str] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeQualityReport:
    episode_index: int
    frame_count: int
    duration_s: float
    findings: tuple[QualityFinding, ...]
    metrics: dict[str, float | int | bool]

    @property
    def accepted(self) -> bool:
        return not any(finding.severity == "reject" for finding in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "status": "accepted" if self.accepted else "rejected",
            "frame_count": self.frame_count,
            "duration_s": self.duration_s,
            "metrics": self.metrics,
            "findings": [asdict(finding) for finding in self.findings],
        }


def validate_episode(
    states: np.ndarray,
    *,
    fps: float,
    signals: dict[str, np.ndarray] | None = None,
    episode_index: int = 0,
    config: EpisodeQualityConfig | None = None,
) -> EpisodeQualityReport:
    """Validate one episode and return deterministic rejection reasons."""
    cfg = config or EpisodeQualityConfig()
    states = np.asarray(states, dtype=np.float64)
    signals = signals or {}
    frame_count = len(states)
    duration_s = frame_count / max(float(fps), 1e-6)
    findings: list[QualityFinding] = []
    metrics: dict[str, float | int | bool] = {
        "diagnostics_available": bool(signals),
    }

    if states.ndim != 2 or states.shape[1] != 16:
        findings.append(
            QualityFinding(
                "invalid_state_shape",
                "reject",
                f"Expected state shape (T, 16), got {states.shape}.",
            )
        )
        return EpisodeQualityReport(
            episode_index, frame_count, duration_s, tuple(findings), metrics
        )

    if duration_s < cfg.min_duration_s:
        findings.append(
            QualityFinding(
                "episode_too_short",
                "reject",
                "Episode is shorter than the configured minimum duration.",
                {"duration_s": duration_s, "minimum_s": cfg.min_duration_s},
            )
        )
    if not np.isfinite(states).all():
        findings.append(
            QualityFinding(
                "non_finite_state",
                "reject",
                "State contains NaN or infinite values.",
            )
        )
        return EpisodeQualityReport(
            episode_index, frame_count, duration_s, tuple(findings), metrics
        )
    if frame_count < 2:
        return EpisodeQualityReport(
            episode_index, frame_count, duration_s, tuple(findings), metrics
        )

    dt = _row_deltas(signals, frame_count, fps)
    _check_tracking_fraction(signals, frame_count, cfg, findings, metrics)
    _check_sensor_health(signals, frame_count, cfg, findings, metrics)
    _check_sync_error(signals, frame_count, cfg, findings, metrics)
    _check_signal_freezes(signals, frame_count, dt, cfg, findings, metrics)
    _check_kinematics(states, dt, cfg, findings, metrics)
    _check_pose_freezes(states, dt, cfg, findings, metrics)
    _check_aperture(states, signals, cfg, findings, metrics)

    return EpisodeQualityReport(
        episode_index=episode_index,
        frame_count=frame_count,
        duration_s=duration_s,
        findings=tuple(findings),
        metrics=metrics,
    )


def write_quality_report(
    path: str | Path,
    reports: list[EpisodeQualityReport],
    *,
    config: EpisodeQualityConfig,
    dataset: str | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    accepted = sum(report.accepted for report in reports)
    payload = {
        "schema_version": 1,
        "dataset": dataset,
        "config": asdict(config),
        "summary": {
            "total": len(reports),
            "accepted": accepted,
            "rejected": len(reports) - accepted,
        },
        "episodes": [report.to_dict() for report in reports],
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def _check_tracking_fraction(
    signals: dict[str, np.ndarray],
    frame_count: int,
    cfg: EpisodeQualityConfig,
    findings: list[QualityFinding],
    metrics: dict[str, float | int | bool],
) -> None:
    left = _signal(signals, "observation.tracking.left_tracked", frame_count)
    right = _signal(signals, "observation.tracking.right_tracked", frame_count)
    if left is None or right is None:
        return
    bad = (left < 0.5) | (right < 0.5)
    fraction = float(np.mean(bad))
    metrics["bad_tracking_fraction"] = fraction
    if fraction > cfg.max_bad_tracking_fraction:
        findings.append(
            QualityFinding(
                "tracking_quality_fraction",
                "reject",
                "Controller tracking was degraded for too much of the episode.",
                {
                    "fraction": fraction,
                    "maximum": cfg.max_bad_tracking_fraction,
                },
            )
        )


def _check_sensor_health(
    signals: dict[str, np.ndarray],
    frame_count: int,
    cfg: EpisodeQualityConfig,
    findings: list[QualityFinding],
    metrics: dict[str, float | int | bool],
) -> None:
    for key in sorted(signals):
        if not key.endswith(".healthy"):
            continue
        values = _signal(signals, key, frame_count)
        if values is None:
            continue
        prefix = key.removesuffix(".healthy")
        enabled = _signal(signals, f"{prefix}.enabled", frame_count)
        mask = np.ones(frame_count, dtype=bool) if enabled is None else enabled >= 0.5
        if not np.any(mask):
            continue
        fraction = float(np.mean(values[mask] < 0.5))
        metric_name = f"bad_sensor_fraction.{prefix.removeprefix('observation.')}"
        metrics[metric_name] = fraction
        if fraction > cfg.max_bad_sensor_fraction:
            findings.append(
                QualityFinding(
                    "sensor_health_fraction",
                    "reject",
                    f"{prefix} was unhealthy for too much of the episode.",
                    {
                        "sensor": prefix,
                        "fraction": fraction,
                        "maximum": cfg.max_bad_sensor_fraction,
                    },
                )
            )


def _check_sync_error(
    signals: dict[str, np.ndarray],
    frame_count: int,
    cfg: EpisodeQualityConfig,
    findings: list[QualityFinding],
    metrics: dict[str, float | int | bool],
) -> None:
    for key in sorted(signals):
        if not key.endswith(".sync_error_ms"):
            continue
        values = _signal(signals, key, frame_count)
        if values is None:
            continue
        prefix = key.removesuffix(".sync_error_ms")
        enabled = _signal(signals, f"{prefix}.enabled", frame_count)
        mask = np.isfinite(values)
        if enabled is not None:
            mask &= enabled >= 0.5
        if not np.any(mask):
            continue
        bad_fraction = float(np.mean(values[mask] > cfg.max_sync_error_ms))
        maximum = float(np.max(values[mask]))
        metric_name = f"max_sync_error_ms.{prefix.removeprefix('observation.')}"
        metrics[metric_name] = maximum
        if bad_fraction > cfg.max_bad_sync_fraction:
            findings.append(
                QualityFinding(
                    "sensor_sync_fraction",
                    "reject",
                    f"{prefix} exceeded the synchronization tolerance too often.",
                    {
                        "sensor": prefix,
                        "fraction": bad_fraction,
                        "maximum_fraction": cfg.max_bad_sync_fraction,
                        "max_error_ms": maximum,
                    },
                )
            )


def _check_signal_freezes(
    signals: dict[str, np.ndarray],
    frame_count: int,
    dt: np.ndarray,
    cfg: EpisodeQualityConfig,
    findings: list[QualityFinding],
    metrics: dict[str, float | int | bool],
) -> None:
    keys = [
        "observation.tracking.aligned_time_ns",
        "observation.feetech.sample_time_ns",
        *sorted(
            key
            for key in signals
            if key.startswith("observation.camera.") and key.endswith(".sample_time_ns")
        ),
    ]
    for key in keys:
        values = _signal(signals, key, frame_count)
        if values is None:
            continue
        prefix = key.removesuffix(".sample_time_ns")
        enabled = _signal(signals, f"{prefix}.enabled", frame_count)
        if enabled is not None and not np.any(enabled >= 0.5):
            continue
        frozen_s = _longest_true_duration(np.diff(values) == 0, dt)
        metric_name = f"max_timestamp_freeze_s.{prefix.removeprefix('observation.')}"
        metrics[metric_name] = frozen_s
        if frozen_s > cfg.max_signal_freeze_s:
            findings.append(
                QualityFinding(
                    "source_timestamp_freeze",
                    "reject",
                    f"{prefix} repeated one source sample for too long.",
                    {
                        "sensor": prefix,
                        "duration_s": frozen_s,
                        "maximum_s": cfg.max_signal_freeze_s,
                    },
                )
            )


def _check_kinematics(
    states: np.ndarray,
    dt: np.ndarray,
    cfg: EpisodeQualityConfig,
    findings: list[QualityFinding],
    metrics: dict[str, float | int | bool],
) -> None:
    for side, pose_slice in (("left", LEFT_POSE_SLICE), ("right", RIGHT_POSE_SLICE)):
        poses = states[:, pose_slice]
        quat_norms = np.linalg.norm(poses[:, 3:7], axis=1)
        if np.any(quat_norms < 0.5):
            findings.append(
                QualityFinding(
                    "invalid_quaternion",
                    "reject",
                    f"{side} controller contains a near-zero quaternion.",
                )
            )
            continue
        speed = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1) / dt
        rotation = _rotation_steps_deg(poses[:, 3:7])
        max_speed = float(np.max(speed, initial=0.0))
        max_rotation = float(np.max(rotation, initial=0.0))
        metrics[f"max_translation_speed_m_s.{side}"] = max_speed
        metrics[f"max_rotation_step_deg.{side}"] = max_rotation
        if max_speed > cfg.max_translation_speed_m_s:
            findings.append(
                QualityFinding(
                    "translation_jump",
                    "reject",
                    f"{side} controller exceeded the plausible translation speed.",
                    {
                        "speed_m_s": max_speed,
                        "maximum_m_s": cfg.max_translation_speed_m_s,
                    },
                )
            )
        if max_rotation > cfg.max_rotation_step_deg:
            findings.append(
                QualityFinding(
                    "rotation_jump",
                    "reject",
                    f"{side} controller rotated too far in one frame.",
                    {
                        "step_deg": max_rotation,
                        "maximum_deg": cfg.max_rotation_step_deg,
                    },
                )
            )


def _check_pose_freezes(
    states: np.ndarray,
    dt: np.ndarray,
    cfg: EpisodeQualityConfig,
    findings: list[QualityFinding],
    metrics: dict[str, float | int | bool],
) -> None:
    frozen_by_side: dict[str, np.ndarray] = {}
    for side, pose_slice in (("left", LEFT_POSE_SLICE), ("right", RIGHT_POSE_SLICE)):
        poses = states[:, pose_slice]
        translation = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)
        rotation = _rotation_steps_deg(poses[:, 3:7])
        frozen = (translation <= cfg.freeze_translation_epsilon_m) & (
            rotation <= cfg.freeze_rotation_epsilon_deg
        )
        frozen_by_side[side] = frozen
        longest_s = _longest_true_duration(frozen, dt)
        metrics[f"max_pose_freeze_s.{side}"] = longest_s
        if longest_s > cfg.max_pose_freeze_s:
            severity: Severity = (
                "reject" if cfg.reject_single_side_pose_freeze else "warning"
            )
            findings.append(
                QualityFinding(
                    "single_side_pose_freeze",
                    severity,
                    f"{side} controller pose remained numerically frozen.",
                    {
                        "duration_s": longest_s,
                        "maximum_s": cfg.max_pose_freeze_s,
                    },
                )
            )

    both_s = _longest_true_duration(
        frozen_by_side["left"] & frozen_by_side["right"], dt
    )
    metrics["max_pose_freeze_s.both"] = both_s
    if both_s > cfg.max_pose_freeze_s:
        findings.append(
            QualityFinding(
                "full_pose_freeze",
                "reject",
                "Both controller poses remained numerically frozen.",
                {"duration_s": both_s, "maximum_s": cfg.max_pose_freeze_s},
            )
        )


def _check_aperture(
    states: np.ndarray,
    signals: dict[str, np.ndarray],
    cfg: EpisodeQualityConfig,
    findings: list[QualityFinding],
    metrics: dict[str, float | int | bool],
) -> None:
    enabled = _signal(signals, "observation.feetech.enabled", len(states))
    if enabled is not None and not np.any(enabled >= 0.5):
        return
    ranges_mm = {
        "left": float(np.ptp(states[:, LEFT_GRIPPER_INDEX]) * 1000.0),
        "right": float(np.ptp(states[:, RIGHT_GRIPPER_INDEX]) * 1000.0),
    }
    for side, range_mm in ranges_mm.items():
        metrics[f"aperture_range_mm.{side}"] = range_mm
    if all(value <= cfg.aperture_range_epsilon_mm for value in ranges_mm.values()):
        severity: Severity = "reject" if cfg.reject_aperture_freeze else "warning"
        findings.append(
            QualityFinding(
                "aperture_freeze",
                severity,
                "Both gripper apertures remained constant for the full episode.",
                {
                    "left_range_mm": ranges_mm["left"],
                    "right_range_mm": ranges_mm["right"],
                },
            )
        )


def _row_deltas(
    signals: dict[str, np.ndarray], frame_count: int, fps: float
) -> np.ndarray:
    fallback = 1.0 / max(float(fps), 1e-6)
    target = _signal(signals, "observation.sync.target_time_ns", frame_count)
    if target is None:
        return np.full(frame_count - 1, fallback, dtype=np.float64)
    dt = np.diff(target.astype(np.float64)) / 1e9
    dt[~np.isfinite(dt) | (dt <= 0)] = fallback
    return dt


def _rotation_steps_deg(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    normalized = quaternions / np.maximum(norms, 1e-12)
    dots = np.sum(normalized[:-1] * normalized[1:], axis=1)
    return np.degrees(2.0 * np.arccos(np.clip(np.abs(dots), 0.0, 1.0)))


def _longest_true_duration(mask: np.ndarray, dt: np.ndarray) -> float:
    longest = 0.0
    current = 0.0
    for frozen, step_s in zip(np.asarray(mask, dtype=bool), dt):
        current = current + float(step_s) if frozen else 0.0
        longest = max(longest, current)
    return longest


def _signal(
    signals: dict[str, np.ndarray], key: str, frame_count: int
) -> np.ndarray | None:
    if key not in signals:
        return None
    values = np.asarray(signals[key]).reshape(-1)
    return values.astype(np.float64) if len(values) == frame_count else None


__all__ = [
    "EpisodeQualityConfig",
    "EpisodeQualityReport",
    "QualityFinding",
    "validate_episode",
    "write_quality_report",
]
