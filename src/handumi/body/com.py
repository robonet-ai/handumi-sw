"""Uncertainty-aware kinematic whole-body CoM and contact estimation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from handumi.body.model import (
    CANONICAL_JOINTS,
    CanonicalBodyFrame,
    CanonicalProvenance,
    CanonicalTrackingState,
    ComDiagnostic,
    ComProvenance,
)
from handumi.body.quality import SmoothedComTrajectory, TrajectoryFilterConfig

COM_ESTIMATOR_SCHEMA = "handumi_kinematic_com_v1"
DEFAULT_ANTHROPOMETRIC_TABLE = "handumi_dempster_15_v1"
CONTACT_NAMES = ("left_heel", "left_ball", "right_heel", "right_ball")

_JOINT_INDEX = {joint.identifier: joint.index for joint in CANONICAL_JOINTS}


@dataclass(frozen=True)
class BodyProfile:
    height_m: float
    mass_kg: float
    arm_span_m: float | None = None
    leg_length_m: float | None = None
    foot_length_m: float | None = None
    foot_width_m: float | None = None
    shoulder_breadth_m: float | None = None
    hip_breadth_m: float | None = None
    measurement_uncertainty_m: float = 0.01
    mass_uncertainty_kg: float = 0.5
    source: str = "operator_profile"

    def __post_init__(self) -> None:
        if not 0.5 <= self.height_m <= 2.75:
            raise ValueError("height_m must be in [0.5, 2.75]")
        if not 10.0 <= self.mass_kg <= 400.0:
            raise ValueError("mass_kg must be in [10, 400]")
        for name in (
            "arm_span_m",
            "leg_length_m",
            "foot_length_m",
            "foot_width_m",
            "shoulder_breadth_m",
            "hip_breadth_m",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when supplied")
        if self.measurement_uncertainty_m < 0 or self.mass_uncertainty_kg < 0:
            raise ValueError("profile uncertainties must be non-negative")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BodyProfile":
        allowed = {field.name for field in fields(cls)}
        unknown = set(data) - allowed - {"schema"}
        if unknown:
            raise ValueError(f"Unknown body-profile fields: {sorted(unknown)}")
        return cls(**{key: value for key, value in data.items() if key in allowed})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BodyProfile":
        data = yaml.safe_load(Path(path).read_text())
        if not isinstance(data, dict):
            raise ValueError("Body profile must be a YAML mapping")
        return cls.from_dict(data)

    def metadata(self) -> dict[str, Any]:
        values = asdict(self)
        encoded = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return {
            "schema": "handumi_body_profile_v1",
            "values": values,
            "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        }


@dataclass(frozen=True)
class SegmentDefinition:
    identifier: str
    output_joint: str
    proximal_landmarks: tuple[str, ...]
    distal_landmarks: tuple[str, ...]
    mass_fraction: float
    com_fraction_from_proximal: float
    mass_fraction_std: float
    com_fraction_std: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SegmentDefinition":
        return cls(
            identifier=str(data["identifier"]),
            output_joint=str(data["output_joint"]),
            proximal_landmarks=tuple(str(value) for value in data["proximal_landmarks"]),
            distal_landmarks=tuple(str(value) for value in data["distal_landmarks"]),
            mass_fraction=float(data["mass_fraction"]),
            com_fraction_from_proximal=float(
                data["com_fraction_from_proximal"]
            ),
            mass_fraction_std=float(data.get("mass_fraction_std", 0.0)),
            com_fraction_std=float(data.get("com_fraction_std", 0.03)),
        )

    def metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data["proximal_landmarks"] = list(self.proximal_landmarks)
        data["distal_landmarks"] = list(self.distal_landmarks)
        return data


@dataclass(frozen=True)
class AnthropometricTable:
    version: str
    source: str
    segments: tuple[SegmentDefinition, ...]

    def __post_init__(self) -> None:
        if len(self.segments) == 0:
            raise ValueError("Anthropometric table must define segments")
        identifiers = [segment.identifier for segment in self.segments]
        outputs = [segment.output_joint for segment in self.segments]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Segment identifiers must be unique")
        if len(set(outputs)) != len(outputs):
            raise ValueError("Segment output joints must be unique")
        for segment in self.segments:
            joint_names = (
                segment.proximal_landmarks
                + segment.distal_landmarks
                + (segment.output_joint,)
            )
            unknown = [name for name in joint_names if name not in _JOINT_INDEX]
            if unknown:
                raise ValueError(
                    f"Segment {segment.identifier!r} uses unknown joints {unknown}"
                )
            if not 0 < segment.mass_fraction < 1:
                raise ValueError("Every segment mass fraction must be in (0, 1)")
            if not 0 <= segment.com_fraction_from_proximal <= 1:
                raise ValueError("Segment CoM fractions must be in [0, 1]")
            if segment.mass_fraction_std < 0 or segment.com_fraction_std < 0:
                raise ValueError("Segment uncertainties must be non-negative")
        total = sum(segment.mass_fraction for segment in self.segments)
        if not math.isclose(total, 1.0, abs_tol=5e-4):
            raise ValueError(f"Segment mass fractions must sum to 1.0, got {total}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnthropometricTable":
        raw_segments = data.get("segments")
        if not isinstance(raw_segments, list):
            raise ValueError("Anthropometric table requires a segments list")
        return cls(
            version=str(data["version"]),
            source=str(data.get("source", "custom")),
            segments=tuple(SegmentDefinition.from_dict(item) for item in raw_segments),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AnthropometricTable":
        data = yaml.safe_load(Path(path).read_text())
        if not isinstance(data, dict):
            raise ValueError("Anthropometric table must be a YAML mapping")
        return cls.from_dict(data)

    def metadata(self) -> dict[str, Any]:
        table = {
            "version": self.version,
            "source": self.source,
            "segments": [segment.metadata() for segment in self.segments],
        }
        encoded = json.dumps(table, sort_keys=True, separators=(",", ":"))
        return {**table, "sha256": hashlib.sha256(encoded.encode()).hexdigest()}


def _segment(
    identifier: str,
    output_joint: str,
    proximal: str | tuple[str, ...],
    distal: str | tuple[str, ...],
    mass_fraction: float,
    com_fraction: float,
) -> SegmentDefinition:
    proximal_names = (proximal,) if isinstance(proximal, str) else proximal
    distal_names = (distal,) if isinstance(distal, str) else distal
    return SegmentDefinition(
        identifier=identifier,
        output_joint=output_joint,
        proximal_landmarks=proximal_names,
        distal_landmarks=distal_names,
        mass_fraction=mass_fraction,
        com_fraction_from_proximal=com_fraction,
        mass_fraction_std=mass_fraction * 0.10,
        com_fraction_std=0.03,
    )


def default_anthropometric_table() -> AnthropometricTable:
    """Return the mass-conserving HandUMI 15-segment default table.

    Mass and endpoint coefficients follow the Dempster-style 15-segment table
    reported by Wang et al. (2022). The table is a population prior, not a
    subject-specific anatomical measurement.
    """
    segments = (
        _segment("head_neck", "head", "neck", "head", 0.081, 1.000),
        _segment(
            "trunk",
            "chest",
            ("left_shoulder", "right_shoulder"),
            ("left_hip", "right_hip"),
            0.355,
            0.500,
        ),
        _segment(
            "pelvis",
            "pelvis",
            "spine_lower",
            ("left_hip", "right_hip"),
            0.142,
            0.105,
        ),
        _segment(
            "left_upper_arm",
            "left_shoulder",
            "left_shoulder",
            "left_elbow",
            0.028,
            0.436,
        ),
        _segment(
            "right_upper_arm",
            "right_shoulder",
            "right_shoulder",
            "right_elbow",
            0.028,
            0.436,
        ),
        _segment(
            "left_forearm",
            "left_elbow",
            "left_elbow",
            "left_wrist",
            0.016,
            0.430,
        ),
        _segment(
            "right_forearm",
            "right_elbow",
            "right_elbow",
            "right_wrist",
            0.016,
            0.430,
        ),
        _segment("left_hand", "left_hand", "left_wrist", "left_hand", 0.006, 0.506),
        _segment("right_hand", "right_hand", "right_wrist", "right_hand", 0.006, 0.506),
        _segment("left_thigh", "left_hip", "left_hip", "left_knee", 0.100, 0.433),
        _segment("right_thigh", "right_hip", "right_hip", "right_knee", 0.100, 0.433),
        _segment(
            "left_shank", "left_knee", "left_knee", "left_ankle", 0.0465, 0.433
        ),
        _segment(
            "right_shank",
            "right_knee",
            "right_knee",
            "right_ankle",
            0.0465,
            0.433,
        ),
        _segment(
            "left_foot",
            "left_foot_ball",
            "left_ankle",
            "left_foot_ball",
            0.0145,
            0.500,
        ),
        _segment(
            "right_foot",
            "right_foot_ball",
            "right_ankle",
            "right_foot_ball",
            0.0145,
            0.500,
        ),
    )
    return AnthropometricTable(
        version=DEFAULT_ANTHROPOMETRIC_TABLE,
        source=(
            "Wang_et_al_2022_Dempster_style_15_segment_table;"
            "doi:10.3389/fnbot.2022.863722"
        ),
        segments=segments,
    )


@dataclass(frozen=True)
class ComEstimatorConfig:
    tracked_position_std_m: float = 0.015
    valid_position_std_m: float = 0.040
    max_com_std_m: float = 0.120
    contact_height_midpoint_m: float = 0.035
    contact_height_scale_m: float = 0.012
    contact_speed_midpoint_m_s: float = 0.20
    contact_speed_scale_m_s: float = 0.08
    contact_accept_probability: float = 0.65
    contact_max_gap_s: float = 0.20
    contact_max_speed_m_s: float = 5.0
    inferred_heel_offset_ratio: float = 0.35
    inferred_foot_width_ratio: float = 0.40
    trajectory: TrajectoryFilterConfig = TrajectoryFilterConfig()

    def __post_init__(self) -> None:
        positive = (
            self.tracked_position_std_m,
            self.valid_position_std_m,
            self.max_com_std_m,
            self.contact_height_scale_m,
            self.contact_speed_scale_m_s,
            self.contact_max_gap_s,
            self.contact_max_speed_m_s,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Estimator scales and limits must be positive")
        if not 0 <= self.contact_accept_probability <= 1:
            raise ValueError("contact_accept_probability must be in [0, 1]")


@dataclass(frozen=True)
class _Landmark:
    position: np.ndarray
    covariance: np.ndarray
    confidence: float


class KinematicComEstimator:
    """Stateful kinematic estimator; all outputs retain inferred provenance."""

    def __init__(
        self,
        profile: BodyProfile,
        *,
        table: AnthropometricTable | None = None,
        config: ComEstimatorConfig | None = None,
    ) -> None:
        self.profile = profile
        self.table = table or default_anthropometric_table()
        self.config = config or ComEstimatorConfig()
        self.trajectory = SmoothedComTrajectory(self.config.trajectory)
        self._contact_history: dict[int, tuple[int, np.ndarray]] = {}

    def reset(self) -> None:
        self.trajectory.reset()
        self._contact_history.clear()

    def metadata(self) -> dict[str, Any]:
        config = asdict(self.config)
        config["trajectory"] = asdict(self.config.trajectory)
        segment_masses = {
            segment.identifier: segment.mass_fraction * self.profile.mass_kg
            for segment in self.table.segments
        }
        return {
            "schema": COM_ESTIMATOR_SCHEMA,
            "profile": self.profile.metadata(),
            "anthropometric_table": self.table.metadata(),
            "configuration": config,
            "segment_mass_kg": segment_masses,
            "resolved_total_mass_kg": sum(segment_masses.values()),
            "provenance": "KINEMATIC_INFERRED",
            "center_of_pressure": "UNAVAILABLE_without_force_or_pressure_input",
            "contact_thresholds": "configurable_not_ground_truth_validated",
        }

    def estimate(
        self,
        frame: CanonicalBodyFrame,
        *,
        external_contact_probability: Mapping[str, float] | None = None,
    ) -> CanonicalBodyFrame:
        output = _copy_frame(frame)
        _clear_derived_outputs(output)
        self._clear_previous_inferred_heels(output)

        segment_values: list[tuple[SegmentDefinition, np.ndarray, np.ndarray, float]] = []
        missing_mass = 0.0
        for segment in self.table.segments:
            output_index = _JOINT_INDEX[segment.output_joint]
            output.segment_mass_fraction[output_index] = segment.mass_fraction
            proximal = self._landmark(output, segment.proximal_landmarks)
            distal = self._landmark(output, segment.distal_landmarks)
            if proximal is None or distal is None:
                missing_mass += segment.mass_fraction
                continue
            fraction = segment.com_fraction_from_proximal
            axis = distal.position - proximal.position
            position = proximal.position + fraction * axis
            covariance = (
                (1.0 - fraction) ** 2 * proximal.covariance
                + fraction**2 * distal.covariance
                + np.outer(axis, axis) * segment.com_fraction_std**2
            )
            confidence = min(proximal.confidence, distal.confidence)
            output.segment_com[output_index] = position
            output.segment_com_valid[output_index] = 1
            output.segment_com_confidence[output_index] = confidence
            output.segment_com_covariance[output_index] = covariance
            output.segment_com_provenance[output_index] = int(
                ComProvenance.KINEMATIC_INFERRED
            )
            segment_values.append((segment, position, covariance, confidence))

        output.whole_com_unresolved_mass_fraction[0] = missing_mass
        if missing_mass > 5e-4:
            output.whole_com_diagnostic[0] = int(ComDiagnostic.UNRESOLVED_MASS)
            self.trajectory.reset()
        else:
            self._write_whole_com(output, segment_values)

        timestamp_ns = _frame_time_ns(output)
        if output.whole_com_valid[0]:
            trajectory = self.trajectory.update(timestamp_ns, output.whole_com)
            if trajectory.velocity_valid:
                output.whole_com_velocity[:] = trajectory.velocity
                output.whole_com_velocity_valid[0] = 1
            if trajectory.acceleration_valid:
                output.whole_com_acceleration[:] = trajectory.acceleration
                output.whole_com_acceleration_valid[0] = 1
            output.whole_com_trajectory_diagnostic[0] = int(trajectory.diagnostic)
        else:
            self.trajectory.reset()

        self._infer_heels(output)
        self._estimate_contacts(
            output,
            timestamp_ns,
            external_contact_probability or {},
        )
        return output

    def _landmark(
        self, frame: CanonicalBodyFrame, names: tuple[str, ...]
    ) -> _Landmark | None:
        positions = []
        covariances = []
        confidences = []
        for name in names:
            index = _JOINT_INDEX[name]
            position = np.asarray(frame.joint_pose[index, :3], dtype=np.float64)
            if not frame.position_valid[index] or not np.all(np.isfinite(position)):
                return None
            confidence = float(frame.confidence[index])
            if not np.isfinite(confidence):
                confidence = (
                    1.0
                    if frame.tracking_state[index] == int(CanonicalTrackingState.TRACKED)
                    else 0.5
                )
            confidence = float(np.clip(confidence, 0.0, 1.0))
            base_std = (
                self.config.tracked_position_std_m
                if frame.tracking_state[index] == int(CanonicalTrackingState.TRACKED)
                else self.config.valid_position_std_m
            )
            std = math.sqrt(
                base_std**2 + self.profile.measurement_uncertainty_m**2
            ) / max(0.25, confidence)
            positions.append(position)
            covariances.append(np.eye(3) * std**2)
            confidences.append(confidence)
        count = len(positions)
        return _Landmark(
            position=np.mean(positions, axis=0),
            covariance=np.sum(covariances, axis=0) / count**2,
            confidence=min(confidences),
        )

    def _write_whole_com(
        self,
        output: CanonicalBodyFrame,
        segments: list[tuple[SegmentDefinition, np.ndarray, np.ndarray, float]],
    ) -> None:
        center = sum(
            (segment.mass_fraction * position for segment, position, _, _ in segments),
            start=np.zeros(3, dtype=np.float64),
        )
        covariance = np.zeros((3, 3), dtype=np.float64)
        confidence = 0.0
        for segment, position, segment_covariance, segment_confidence in segments:
            covariance += segment.mass_fraction**2 * segment_covariance
            covariance += segment.mass_fraction_std**2 * np.outer(
                position - center, position - center
            )
            confidence += segment.mass_fraction * segment_confidence
        max_std = math.sqrt(max(0.0, float(np.linalg.eigvalsh(covariance).max())))
        if max_std > self.config.max_com_std_m:
            output.whole_com_diagnostic[0] = int(
                ComDiagnostic.EXCESSIVE_UNCERTAINTY
            )
            return
        output.whole_com[:] = center
        output.whole_com_valid[0] = 1
        output.whole_com_confidence[0] = float(np.clip(confidence, 0.0, 1.0))
        output.whole_com_covariance[:] = covariance
        output.whole_com_provenance[0] = int(ComProvenance.KINEMATIC_INFERRED)
        output.whole_com_diagnostic[0] = int(ComDiagnostic.VALID)

        plane = _normalized_plane(output.ground_plane)
        if plane is None:
            return
        normal, offset = plane
        projection = center - normal * (float(np.dot(normal, center)) + offset)
        output.whole_com_ground_projection[:] = projection
        output.whole_com_ground_projection_valid[0] = 1
        tangent_projection = np.eye(3) - np.outer(normal, normal)
        output.whole_com_ground_projection_covariance[:] = (
            tangent_projection @ covariance @ tangent_projection.T
        )

    def _clear_previous_inferred_heels(self, output: CanonicalBodyFrame) -> None:
        for name in ("left_heel", "right_heel"):
            index = _JOINT_INDEX[name]
            if output.provenance[index] == int(CanonicalProvenance.INFERRED):
                output.joint_pose[index] = np.nan
                output.position_valid[index] = 0
                output.orientation_valid[index] = 0
                output.tracking_state[index] = int(CanonicalTrackingState.INVALID)
                output.confidence[index] = np.nan
                output.provenance[index] = int(CanonicalProvenance.UNAVAILABLE)

    def _infer_heels(self, output: CanonicalBodyFrame) -> None:
        plane = _normalized_plane(output.ground_plane)
        if plane is None:
            return
        normal, offset = plane
        for side in ("left", "right"):
            heel_index = _JOINT_INDEX[f"{side}_heel"]
            if output.position_valid[heel_index]:
                continue
            ankle_index = _JOINT_INDEX[f"{side}_ankle"]
            ball_index = _JOINT_INDEX[f"{side}_foot_ball"]
            if not output.position_valid[ankle_index] or not output.position_valid[ball_index]:
                continue
            ankle = np.asarray(output.joint_pose[ankle_index, :3], dtype=np.float64)
            ball = np.asarray(output.joint_pose[ball_index, :3], dtype=np.float64)
            forward = ball - ankle
            forward -= normal * float(np.dot(normal, forward))
            ankle_to_ball = float(np.linalg.norm(forward))
            if ankle_to_ball <= 1e-6:
                continue
            direction = forward / ankle_to_ball
            if self.profile.foot_length_m is not None:
                heel_offset = max(0.01, self.profile.foot_length_m - ankle_to_ball)
            else:
                heel_offset = self.config.inferred_heel_offset_ratio * ankle_to_ball
            heel = ankle - direction * heel_offset
            ball_height = float(np.dot(normal, ball)) + offset
            heel_height = float(np.dot(normal, heel)) + offset
            heel += normal * (ball_height - heel_height)
            output.joint_pose[heel_index, :3] = heel
            output.position_valid[heel_index] = 1
            output.tracking_state[heel_index] = int(CanonicalTrackingState.VALID)
            source_confidence = min(
                _finite_confidence(output.confidence[ankle_index]),
                _finite_confidence(output.confidence[ball_index]),
            )
            output.confidence[heel_index] = 0.7 * source_confidence
            output.provenance[heel_index] = int(CanonicalProvenance.INFERRED)

    def _estimate_contacts(
        self,
        output: CanonicalBodyFrame,
        timestamp_ns: int,
        external: Mapping[str, float],
    ) -> None:
        plane = _normalized_plane(output.ground_plane)
        if plane is None or timestamp_ns <= 0:
            self._contact_history.clear()
            return
        normal, offset = plane
        contact_indices = (
            _JOINT_INDEX["left_heel"],
            _JOINT_INDEX["left_foot_ball"],
            _JOINT_INDEX["right_heel"],
            _JOINT_INDEX["right_foot_ball"],
        )
        for contact_index, joint_index in enumerate(contact_indices):
            if not output.position_valid[joint_index]:
                self._contact_history.pop(contact_index, None)
                continue
            position = np.asarray(output.joint_pose[joint_index, :3], dtype=np.float64)
            previous = self._contact_history.get(contact_index)
            self._contact_history[contact_index] = (timestamp_ns, position.copy())
            external_value = external.get(CONTACT_NAMES[contact_index])
            if previous is None:
                self._write_external_contact(
                    output, contact_index, external_value
                )
                continue
            previous_time, previous_position = previous
            dt_s = (timestamp_ns - previous_time) / 1e9
            speed = (
                float(np.linalg.norm(position - previous_position)) / dt_s
                if 0 < dt_s <= self.config.contact_max_gap_s
                else float("inf")
            )
            if speed > self.config.contact_max_speed_m_s:
                self._write_external_contact(
                    output, contact_index, external_value
                )
                continue
            height = abs(float(np.dot(normal, position)) + offset)
            height_probability = _decreasing_logistic(
                height,
                self.config.contact_height_midpoint_m,
                self.config.contact_height_scale_m,
            )
            speed_probability = _decreasing_logistic(
                speed,
                self.config.contact_speed_midpoint_m_s,
                self.config.contact_speed_scale_m_s,
            )
            confidence = _finite_confidence(output.confidence[joint_index])
            quality_factor = 0.5 + 0.5 * confidence
            probability = height_probability * speed_probability * quality_factor
            provenance = ComProvenance.KINEMATIC_INFERRED
            if external_value is not None and np.isfinite(external_value):
                probability = 1.0 - (1.0 - probability) * (
                    1.0 - float(np.clip(external_value, 0.0, 1.0))
                )
                provenance = ComProvenance.FUSED_ESTIMATED
            output.contact_probability[contact_index] = np.clip(
                probability, 0.0, 1.0
            )
            output.contact_valid[contact_index] = 1
            output.contact_provenance[contact_index] = int(provenance)
        self._write_support_polygon(output, normal, offset)

    @staticmethod
    def _write_external_contact(
        output: CanonicalBodyFrame,
        contact_index: int,
        value: float | None,
    ) -> None:
        if value is None or not np.isfinite(value):
            return
        output.contact_probability[contact_index] = np.clip(value, 0.0, 1.0)
        output.contact_valid[contact_index] = 1
        output.contact_provenance[contact_index] = int(
            ComProvenance.FUSED_ESTIMATED
        )

    def _write_support_polygon(
        self, output: CanonicalBodyFrame, normal: np.ndarray, offset: float
    ) -> None:
        points = []
        for side_index, side in enumerate(("left", "right")):
            heel_contact = 2 * side_index
            ball_contact = heel_contact + 1
            accepted = all(
                output.contact_valid[index]
                and output.contact_probability[index]
                >= self.config.contact_accept_probability
                for index in (heel_contact, ball_contact)
            )
            if not accepted:
                continue
            heel = np.asarray(
                output.joint_pose[_JOINT_INDEX[f"{side}_heel"], :3],
                dtype=np.float64,
            )
            ball = np.asarray(
                output.joint_pose[_JOINT_INDEX[f"{side}_foot_ball"], :3],
                dtype=np.float64,
            )
            heel = _project_to_plane(heel, normal, offset)
            ball = _project_to_plane(ball, normal, offset)
            forward = ball - heel
            forward -= normal * float(np.dot(normal, forward))
            length = float(np.linalg.norm(forward))
            if length <= 1e-6:
                continue
            lateral = np.cross(normal, forward / length)
            width = (
                self.profile.foot_width_m
                if self.profile.foot_width_m is not None
                else self.config.inferred_foot_width_ratio * length
            )
            points.extend(
                (
                    heel + lateral * width / 2,
                    heel - lateral * width / 2,
                    ball + lateral * width / 2,
                    ball - lateral * width / 2,
                )
            )
        hull = _convex_hull_on_plane(points, normal)
        for index, point in enumerate(hull[: len(output.support_polygon)]):
            output.support_polygon[index] = point
            output.support_polygon_valid[index] = 1


def _copy_frame(frame: CanonicalBodyFrame) -> CanonicalBodyFrame:
    return replace(
        frame,
        **{
            field.name: np.array(getattr(frame, field.name), copy=True)
            for field in fields(CanonicalBodyFrame)
        },
    )


def _clear_derived_outputs(frame: CanonicalBodyFrame) -> None:
    nan_fields = (
        "whole_com",
        "whole_com_confidence",
        "whole_com_covariance",
        "whole_com_unresolved_mass_fraction",
        "whole_com_ground_projection",
        "whole_com_ground_projection_covariance",
        "whole_com_velocity",
        "whole_com_acceleration",
        "segment_com",
        "segment_com_confidence",
        "segment_com_covariance",
        "contact_probability",
        "support_polygon",
        "center_of_pressure",
    )
    zero_fields = (
        "whole_com_valid",
        "whole_com_provenance",
        "whole_com_diagnostic",
        "whole_com_ground_projection_valid",
        "whole_com_velocity_valid",
        "whole_com_acceleration_valid",
        "whole_com_trajectory_diagnostic",
        "segment_com_valid",
        "segment_com_provenance",
        "segment_mass_fraction",
        "contact_valid",
        "contact_provenance",
        "support_polygon_valid",
        "center_of_pressure_valid",
    )
    for name in nan_fields:
        getattr(frame, name)[...] = np.nan
    for name in zero_fields:
        getattr(frame, name)[...] = 0


def _normalized_plane(plane: np.ndarray) -> tuple[np.ndarray, float] | None:
    values = np.asarray(plane, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(values)):
        return None
    norm = float(np.linalg.norm(values[:3]))
    if norm <= 1e-12:
        return None
    return values[:3] / norm, float(values[3] / norm)


def _project_to_plane(
    point: np.ndarray, normal: np.ndarray, offset: float
) -> np.ndarray:
    return point - normal * (float(np.dot(normal, point)) + offset)


def _decreasing_logistic(value: float, midpoint: float, scale: float) -> float:
    exponent = float(np.clip((value - midpoint) / scale, -60.0, 60.0))
    return 1.0 / (1.0 + math.exp(exponent))


def _finite_confidence(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else 0.5


def _frame_time_ns(frame: CanonicalBodyFrame) -> int:
    for value in (frame.mapped_time_ns[0], frame.receive_time_ns[0]):
        if int(value) > 0:
            return int(value)
    return 0


def _convex_hull_on_plane(
    points: list[np.ndarray], normal: np.ndarray
) -> list[np.ndarray]:
    if len(points) <= 1:
        return [np.asarray(point, dtype=np.float64) for point in points]

    # Build a stable 2-D basis in the calibrated plane. Using world X/Y here
    # would collapse a valid polygon when the complete frame is rigidly rotated.
    reference_axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
    axis_u = np.cross(normal, reference_axis)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    unique = {
        (
            round(float(np.dot(point, axis_u)), 12),
            round(float(np.dot(point, axis_v)), 12),
        ): np.asarray(point, dtype=np.float64)
        for point in points
    }
    ordered = sorted(unique.items())
    if len(ordered) <= 2:
        return [item[1] for item in ordered]

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[tuple[float, float], np.ndarray]] = []
    for item in ordered:
        while len(lower) >= 2 and cross(lower[-2][0], lower[-1][0], item[0]) <= 0:
            lower.pop()
        lower.append(item)
    upper: list[tuple[tuple[float, float], np.ndarray]] = []
    for item in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2][0], upper[-1][0], item[0]) <= 0:
            upper.pop()
        upper.append(item)
    return [item[1] for item in lower[:-1] + upper[:-1]]


__all__ = [
    "COM_ESTIMATOR_SCHEMA",
    "CONTACT_NAMES",
    "DEFAULT_ANTHROPOMETRIC_TABLE",
    "AnthropometricTable",
    "BodyProfile",
    "ComEstimatorConfig",
    "KinematicComEstimator",
    "SegmentDefinition",
    "default_anthropometric_table",
]
