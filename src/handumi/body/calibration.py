"""Profile-aware neutral calibration and constrained body geometry.

This module deliberately keeps two operations separate:

* a rigid source-to-world calibration puts the estimated floor at ``z=0``;
* optional profile constraints retarget platform-estimated joint positions to
  operator-supplied dimensions and mark every changed position ``INFERRED``.

Native Quest/PICO packets remain unchanged in the tracking sidecar.  Neither
operation is ground-truth anatomical validation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, replace
from typing import Any, Sequence

import numpy as np

from handumi.body.com import BodyProfile
from handumi.body.model import (
    CANONICAL_JOINTS,
    CanonicalBodyFrame,
    CanonicalProvenance,
    CanonicalTrackingState,
)
from handumi.tracking.transforms import (
    HandumiWorldCalibration,
    quat_rotate,
)

BODY_CALIBRATION_SCHEMA = "handumi_profile_neutral_calibration_v1"
PROFILE_SKELETON_SCHEMA = "handumi_profile_constrained_skeleton_v1"

_INDEX = {joint.identifier: joint.index for joint in CANONICAL_JOINTS}
_LEFT_RIGHT_PAIRS = {
    "shoulder_breadth_m": ("left_shoulder", "right_shoulder"),
    "hip_breadth_m": ("left_hip", "right_hip"),
}
_AXIAL_CHAIN = (
    "spine_lower",
    "spine_middle",
    "spine_upper",
    "chest",
    "neck",
    "head",
)


@dataclass(frozen=True)
class ProfileNeutralCalibration:
    """One unqualified, profile-assisted neutral-pose frame calibration."""

    world: HandumiWorldCalibration
    source_ground_height_m: float
    observed_stature_m: float
    stature_error_m: float
    sample_count: int
    ground_sample_std_m: float
    pelvis_motion_p95_m: float
    method: str = "platform_feet_plus_profile_stature"

    def metadata(self) -> dict[str, Any]:
        values = {
            "schema": BODY_CALIBRATION_SCHEMA,
            "method": self.method,
            "sample_count": self.sample_count,
            "source_ground_height_m": self.source_ground_height_m,
            "observed_stature_m": self.observed_stature_m,
            "stature_error_m": self.stature_error_m,
            "ground_sample_std_m": self.ground_sample_std_m,
            "pelvis_motion_p95_m": self.pelvis_motion_p95_m,
            "qualified": False,
            "limitation": (
                "profile-assisted platform estimate; requires independent "
                "floor/anatomical validation"
            ),
            "transform": self.world.metadata(),
        }
        encoded = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return {**values, "sha256": hashlib.sha256(encoded.encode()).hexdigest()}


def estimate_profile_neutral_calibration(
    frames: Sequence[CanonicalBodyFrame],
    device_hmd_poses: Sequence[np.ndarray],
    profile: BodyProfile,
    *,
    source_frame: str,
    min_samples: int = 15,
    max_neutral_motion_m: float = 0.08,
    max_stature_error_m: float = 0.15,
) -> ProfileNeutralCalibration:
    """Estimate a level floor origin from a short upright neutral/T-pose.

    The runtime feet and ``head - measured height`` independently estimate the
    source-frame floor height.  Their combination corrects a stale Guardian /
    Stage floor without silently declaring the result ground truth.
    """

    if len(frames) != len(device_hmd_poses):
        raise ValueError("Neutral body frames and HMD poses must have equal length")
    if min_samples < 3:
        raise ValueError("min_samples must be at least 3")

    accepted: list[tuple[float, float, np.ndarray, np.ndarray]] = []
    for frame, hmd_pose in zip(frames, device_hmd_poses, strict=True):
        hmd = np.asarray(hmd_pose, dtype=np.float64).reshape(7)
        head = _point(frame, "head")
        left_foot = _point(frame, "left_foot_ball")
        right_foot = _point(frame, "right_foot_ball")
        pelvis = _point(frame, "pelvis")
        if (
            head is None
            or left_foot is None
            or right_foot is None
            or pelvis is None
            or not np.all(np.isfinite(hmd))
        ):
            continue
        foot_height = 0.5 * float(left_foot[2] + right_foot[2])
        observed_stature = float(head[2] - foot_height)
        profile_floor_height = float(head[2] - profile.height_m)
        accepted.append((foot_height, profile_floor_height, pelvis, hmd))

    if len(accepted) < min_samples:
        raise ValueError(
            f"Neutral calibration requires {min_samples} complete frames; "
            f"received {len(accepted)}"
        )

    foot_heights = np.asarray([item[0] for item in accepted], dtype=np.float64)
    profile_floor_heights = np.asarray(
        [item[1] for item in accepted], dtype=np.float64
    )
    observed_statures = foot_heights * -1.0 + np.asarray(
        [item[1] + profile.height_m for item in accepted], dtype=np.float64
    )
    observed_stature = float(np.median(observed_statures))
    stature_error = observed_stature - profile.height_m
    allowed_stature_error = max(
        float(max_stature_error_m), 8.0 * profile.measurement_uncertainty_m
    )
    if abs(stature_error) > allowed_stature_error:
        raise ValueError(
            "Neutral pose is inconsistent with body profile: observed "
            f"head-to-feet stature {observed_stature:.3f} m versus "
            f"height_m {profile.height_m:.3f} m"
        )

    pelvis_positions = np.stack([item[2] for item in accepted])
    pelvis_center = np.median(pelvis_positions, axis=0)
    pelvis_motion = np.linalg.norm(pelvis_positions - pelvis_center, axis=1)
    pelvis_motion_p95 = float(np.percentile(pelvis_motion, 95))
    if pelvis_motion_p95 > max_neutral_motion_m:
        raise ValueError(
            "Neutral calibration movement is too large: pelvis p95 motion "
            f"{pelvis_motion_p95:.3f} m exceeds {max_neutral_motion_m:.3f} m"
        )

    # Weight both independent estimates equally. The platform feet retain the
    # runtime's lower-body prior; head-height carries the operator measurement.
    ground_samples = np.concatenate((foot_heights, profile_floor_heights))
    source_ground_height = float(np.median(ground_samples))
    ground_sample_std = float(np.std(ground_samples))

    hmd_positions = np.stack([item[3][:3] for item in accepted])
    forwards = np.stack(
        [quat_rotate(item[3][3:7], (1.0, 0.0, 0.0)) for item in accepted]
    )
    heading = np.median(forwards, axis=0)
    heading[2] = 0.0
    if float(np.linalg.norm(heading)) <= 1e-6:
        raise ValueError("Neutral HMD heading is parallel to gravity")
    ground_origin = np.median(hmd_positions, axis=0)
    ground_origin[2] = source_ground_height
    world = HandumiWorldCalibration.from_ground_heading(
        ground_origin=ground_origin,
        ground_normal=(0.0, 0.0, 1.0),
        initial_heading=heading,
        source_frame=source_frame,
        qualified=False,
    )
    return ProfileNeutralCalibration(
        world=world,
        source_ground_height_m=source_ground_height,
        observed_stature_m=observed_stature,
        stature_error_m=stature_error,
        sample_count=len(accepted),
        ground_sample_std_m=ground_sample_std,
        pelvis_motion_p95_m=pelvis_motion_p95,
    )


class ProfileConstrainedSkeleton:
    """Retarget canonical positions to measured dimensions with provenance."""

    def __init__(self, profile: BodyProfile) -> None:
        self.profile = profile
        self._root_translation = np.zeros(3, dtype=np.float64)
        self._target_hip_height_m = 0.0
        self._calibrated = False
        self._observed: dict[str, float] = {}

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    def calibrate(self, frames: Sequence[CanonicalBodyFrame]) -> None:
        complete = [frame for frame in frames if self._has_required_neutral(frame)]
        if not complete:
            raise ValueError("No complete neutral frames are available for profile fitting")

        representative = _median_frame(complete)
        plane = _plane(representative.ground_plane)
        if plane is None:
            raise ValueError("Profile fitting requires a calibrated ground plane")
        normal, offset = plane
        hip_center = _pair_center(representative, "left_hip", "right_hip")
        if hip_center is None:
            raise ValueError("Profile fitting requires both hip joints")
        observed_hip_height = float(np.dot(normal, hip_center) + offset)
        target_hip_height = (
            self.profile.leg_length_m
            if self.profile.leg_length_m is not None
            else observed_hip_height
        )
        self._target_hip_height_m = target_hip_height
        self._root_translation = normal * (target_hip_height - observed_hip_height)
        self._observed = self._measure(representative, normal, offset)
        self._calibrated = True

    def apply(self, frame: CanonicalBodyFrame) -> CanonicalBodyFrame:
        if not self._calibrated:
            raise RuntimeError("ProfileConstrainedSkeleton must be calibrated first")
        output = _copy_frame(frame)
        source = np.asarray(output.joint_pose[:, :3], dtype=np.float64).copy()
        changed: set[int] = set()

        if float(np.linalg.norm(self._root_translation)) > 1e-12:
            valid = output.position_valid.astype(bool)
            output.joint_pose[valid, :3] += self._root_translation
            changed.update(np.flatnonzero(valid).tolist())
            source[valid] += self._root_translation

        original_hips = {
            side: _point(output, f"{side}_hip") for side in ("left", "right")
        }
        self._apply_pair_width(
            output,
            source,
            "left_hip",
            "right_hip",
            self.profile.hip_breadth_m,
            anchor=None,
            changed=changed,
        )
        if self.profile.leg_length_m is None:
            for side in ("left", "right"):
                _translate_from_joint_change(
                    output,
                    original_hips[side],
                    f"{side}_hip",
                    (f"{side}_knee", f"{side}_ankle", f"{side}_foot_ball"),
                    changed,
                )
        hip_center = _pair_center(output, "left_hip", "right_hip")
        torso_target = self.profile.height_m - self._target_hip_height_m
        if hip_center is not None:
            if torso_target <= 0:
                raise ValueError("height_m must exceed the fitted hip height")
            _place_chain(
                output,
                source,
                hip_center,
                _AXIAL_CHAIN,
                torso_target,
                changed,
            )

        chest = _point(output, "chest")
        source_chest = source[_INDEX["chest"]]
        source_shoulder_center = _source_pair_center(
            output, source, "left_shoulder", "right_shoulder"
        )
        original_shoulders = {
            side: _point(output, f"{side}_shoulder")
            for side in ("left", "right")
        }
        shoulder_anchor = None
        if (
            chest is not None
            and np.all(np.isfinite(source_chest))
            and source_shoulder_center is not None
        ):
            shoulder_anchor = chest + (source_shoulder_center - source_chest)
        self._apply_pair_width(
            output,
            source,
            "left_shoulder",
            "right_shoulder",
            self.profile.shoulder_breadth_m,
            anchor=shoulder_anchor,
            changed=changed,
        )
        if self.profile.arm_span_m is None:
            for side in ("left", "right"):
                _translate_from_joint_change(
                    output,
                    original_shoulders[side],
                    f"{side}_shoulder",
                    (f"{side}_elbow", f"{side}_wrist", f"{side}_hand"),
                    changed,
                )

        shoulder_width = _distance(output, "left_shoulder", "right_shoulder")
        if self.profile.arm_span_m is not None and shoulder_width is not None:
            per_side_reach = 0.5 * (self.profile.arm_span_m - shoulder_width)
            if per_side_reach <= 0:
                raise ValueError("arm_span_m must exceed shoulder breadth")
            for side in ("left", "right"):
                shoulder = _point(output, f"{side}_shoulder")
                if shoulder is None:
                    continue
                _place_arm_chain(
                    output,
                    source,
                    shoulder,
                    side,
                    per_side_reach,
                    self.profile.hand_length_m,
                    changed,
                )
        elif self.profile.hand_length_m is not None:
            for side in ("left", "right"):
                _set_distal_length(
                    output,
                    source,
                    f"{side}_wrist",
                    f"{side}_hand",
                    self.profile.hand_length_m,
                    changed,
                )

        if self.profile.leg_length_m is not None:
            for side in ("left", "right"):
                hip = _point(output, f"{side}_hip")
                if hip is None:
                    continue
                ankle_clearance = self._observed.get(
                    f"{side}_ankle_ground_height", 0.0
                )
                target_leg_chain = self.profile.leg_length_m - ankle_clearance
                if target_leg_chain <= 0:
                    raise ValueError(
                        "leg_length_m must exceed the neutral ankle ground clearance"
                    )
                source_leg_chain = _source_chain_length(
                    source,
                    (f"{side}_hip", f"{side}_knee", f"{side}_ankle"),
                )
                _place_chain(
                    output,
                    source,
                    hip,
                    (f"{side}_knee", f"{side}_ankle"),
                    target_leg_chain,
                    changed,
                )
                scale = (
                    target_leg_chain / source_leg_chain
                    if np.isfinite(source_leg_chain) and source_leg_chain > 1e-9
                    else 1.0
                )
                _move_foot_with_ankle(output, source, side, scale, changed)

        for index in changed:
            output.provenance[index] = int(CanonicalProvenance.INFERRED)
            if output.tracking_state[index] == int(CanonicalTrackingState.TRACKED):
                output.tracking_state[index] = int(CanonicalTrackingState.VALID)
            if np.isfinite(output.confidence[index]):
                output.confidence[index] *= np.float32(0.8)
        return output

    def metadata(self) -> dict[str, Any]:
        constraints = {
            name: getattr(self.profile, name)
            for name in (
                "height_m",
                "arm_span_m",
                "leg_length_m",
                "hand_length_m",
                "foot_length_m",
                "foot_width_m",
                "shoulder_breadth_m",
                "hip_breadth_m",
            )
        }
        return {
            "schema": PROFILE_SKELETON_SCHEMA,
            "calibrated": self._calibrated,
            "constraints": constraints,
            "observed_neutral_geometry_m": dict(self._observed),
            "target_hip_ground_height_m": self._target_hip_height_m,
            "root_translation_m": self._root_translation.tolist(),
            "provenance": "INFERRED",
            "raw_source_retained": "raw/tracking sidecar",
            "limitation": "profile-constrained estimate; not anatomical ground truth",
        }

    @staticmethod
    def _has_required_neutral(frame: CanonicalBodyFrame) -> bool:
        required = (
            "pelvis",
            "head",
            "left_shoulder",
            "right_shoulder",
            "left_hand",
            "right_hand",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        )
        return all(_point(frame, name) is not None for name in required)

    @staticmethod
    def _measure(
        frame: CanonicalBodyFrame, normal: np.ndarray, offset: float
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        head = _point(frame, "head")
        feet = _pair_center(frame, "left_foot_ball", "right_foot_ball")
        if head is not None and feet is not None:
            result["stature"] = float(np.dot(normal, head - feet))
        hips = _pair_center(frame, "left_hip", "right_hip")
        if hips is not None:
            result["hip_ground_height"] = float(np.dot(normal, hips) + offset)
        for label, (left, right) in _LEFT_RIGHT_PAIRS.items():
            value = _distance(frame, left, right)
            if value is not None:
                result[label.removesuffix("_m")] = value
        for side in ("left", "right"):
            result[f"{side}_arm_chain"] = _chain_length(
                frame,
                (
                    f"{side}_shoulder",
                    f"{side}_elbow",
                    f"{side}_wrist",
                    f"{side}_hand",
                ),
            )
            result[f"{side}_leg_chain"] = _chain_length(
                frame,
                (f"{side}_hip", f"{side}_knee", f"{side}_ankle"),
            )
            value = _distance(frame, f"{side}_wrist", f"{side}_hand")
            if value is not None:
                result[f"{side}_hand_length"] = value
            ankle = _point(frame, f"{side}_ankle")
            if ankle is not None:
                result[f"{side}_ankle_ground_height"] = float(
                    np.dot(normal, ankle) + offset
                )
        return {key: value for key, value in result.items() if np.isfinite(value)}

    @staticmethod
    def _apply_pair_width(
        output: CanonicalBodyFrame,
        source: np.ndarray,
        left_name: str,
        right_name: str,
        target: float | None,
        *,
        anchor: np.ndarray | None,
        changed: set[int],
    ) -> None:
        if target is None:
            if anchor is None:
                return
            current_center = _pair_center(output, left_name, right_name)
            if current_center is None:
                return
            delta = anchor - current_center
            for name in (left_name, right_name):
                index = _INDEX[name]
                output.joint_pose[index, :3] += delta
                changed.add(index)
            return
        left_index = _INDEX[left_name]
        right_index = _INDEX[right_name]
        if not output.position_valid[left_index] or not output.position_valid[right_index]:
            return
        left = source[left_index]
        right = source[right_index]
        axis = left - right
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-9:
            return
        center = anchor if anchor is not None else 0.5 * (left + right)
        direction = axis / norm
        output.joint_pose[left_index, :3] = center + 0.5 * target * direction
        output.joint_pose[right_index, :3] = center - 0.5 * target * direction
        changed.update((left_index, right_index))


def _point(frame: CanonicalBodyFrame, name: str) -> np.ndarray | None:
    index = _INDEX[name]
    point = np.asarray(frame.joint_pose[index, :3], dtype=np.float64)
    if not frame.position_valid[index] or not np.all(np.isfinite(point)):
        return None
    return point


def _plane(value: np.ndarray) -> tuple[np.ndarray, float] | None:
    plane = np.asarray(value, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(plane)):
        return None
    norm = float(np.linalg.norm(plane[:3]))
    if norm <= 1e-12:
        return None
    return plane[:3] / norm, float(plane[3] / norm)


def _pair_center(
    frame: CanonicalBodyFrame, left_name: str, right_name: str
) -> np.ndarray | None:
    left = _point(frame, left_name)
    right = _point(frame, right_name)
    return None if left is None or right is None else 0.5 * (left + right)


def _source_pair_center(
    frame: CanonicalBodyFrame,
    source: np.ndarray,
    left_name: str,
    right_name: str,
) -> np.ndarray | None:
    left_index = _INDEX[left_name]
    right_index = _INDEX[right_name]
    if not frame.position_valid[left_index] or not frame.position_valid[right_index]:
        return None
    return 0.5 * (source[left_index] + source[right_index])


def _distance(
    frame: CanonicalBodyFrame, first_name: str, second_name: str
) -> float | None:
    first = _point(frame, first_name)
    second = _point(frame, second_name)
    return None if first is None or second is None else float(np.linalg.norm(second - first))


def _chain_length(frame: CanonicalBodyFrame, names: Sequence[str]) -> float:
    total = 0.0
    for first, second in zip(names[:-1], names[1:], strict=True):
        value = _distance(frame, first, second)
        if value is None:
            return float("nan")
        total += value
    return total


def _source_chain_length(source: np.ndarray, names: Sequence[str]) -> float:
    total = 0.0
    for first, second in zip(names[:-1], names[1:], strict=True):
        vector = source[_INDEX[second]] - source[_INDEX[first]]
        length = float(np.linalg.norm(vector))
        if not np.isfinite(length):
            return float("nan")
        total += length
    return total


def _place_chain(
    output: CanonicalBodyFrame,
    source: np.ndarray,
    start: np.ndarray,
    joint_names: Sequence[str],
    target_total: float,
    changed: set[int],
) -> None:
    if target_total <= 0 or not joint_names:
        return
    parent = np.asarray(start, dtype=np.float64)
    source_parent = parent
    # The first source direction is relative to the nearest logical parent.
    first_index = _INDEX[joint_names[0]]
    first_parent_index = CANONICAL_JOINTS[first_index].parent_index
    if first_parent_index >= 0 and np.all(np.isfinite(source[first_parent_index])):
        source_parent = source[first_parent_index]
    directions: list[np.ndarray] = []
    lengths: list[float] = []
    previous_source = source_parent
    for name in joint_names:
        index = _INDEX[name]
        if not output.position_valid[index] or not np.all(np.isfinite(source[index])):
            return
        vector = source[index] - previous_source
        length = float(np.linalg.norm(vector))
        if length <= 1e-9:
            return
        directions.append(vector / length)
        lengths.append(length)
        previous_source = source[index]
    scale = target_total / sum(lengths)
    for name, direction, length in zip(joint_names, directions, lengths, strict=True):
        parent = parent + direction * (length * scale)
        index = _INDEX[name]
        output.joint_pose[index, :3] = parent
        changed.add(index)


def _place_arm_chain(
    output: CanonicalBodyFrame,
    source: np.ndarray,
    shoulder: np.ndarray,
    side: str,
    target_total: float,
    target_hand: float | None,
    changed: set[int],
) -> None:
    names = (f"{side}_elbow", f"{side}_wrist", f"{side}_hand")
    indices = [_INDEX[name] for name in names]
    shoulder_index = _INDEX[f"{side}_shoulder"]
    points = [source[shoulder_index], *(source[index] for index in indices)]
    if any(
        not output.position_valid[index] or not np.all(np.isfinite(source[index]))
        for index in indices
    ):
        return
    vectors = [points[i + 1] - points[i] for i in range(3)]
    lengths = [float(np.linalg.norm(vector)) for vector in vectors]
    if any(length <= 1e-9 for length in lengths):
        return
    if target_hand is None:
        target_lengths = [length * target_total / sum(lengths) for length in lengths]
    else:
        remaining = target_total - target_hand
        if remaining <= 0:
            raise ValueError("hand_length_m must be less than per-side arm reach")
        proximal_sum = lengths[0] + lengths[1]
        target_lengths = [
            lengths[0] * remaining / proximal_sum,
            lengths[1] * remaining / proximal_sum,
            target_hand,
        ]
    parent = np.asarray(shoulder, dtype=np.float64)
    for index, vector, target_length in zip(
        indices, vectors, target_lengths, strict=True
    ):
        parent = parent + vector / np.linalg.norm(vector) * target_length
        output.joint_pose[index, :3] = parent
        changed.add(index)


def _set_distal_length(
    output: CanonicalBodyFrame,
    source: np.ndarray,
    proximal_name: str,
    distal_name: str,
    target: float,
    changed: set[int],
) -> None:
    proximal = _point(output, proximal_name)
    distal_index = _INDEX[distal_name]
    proximal_index = _INDEX[proximal_name]
    if proximal is None or not output.position_valid[distal_index]:
        return
    vector = source[distal_index] - source[proximal_index]
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return
    output.joint_pose[distal_index, :3] = proximal + vector / norm * target
    changed.add(distal_index)


def _translate_from_joint_change(
    output: CanonicalBodyFrame,
    original_parent: np.ndarray | None,
    parent_name: str,
    descendant_names: Sequence[str],
    changed: set[int],
) -> None:
    """Preserve an unconstrained limb when its fitted parent joint moves."""
    current_parent = _point(output, parent_name)
    if original_parent is None or current_parent is None:
        return
    delta = current_parent - original_parent
    if float(np.linalg.norm(delta)) <= 1e-12:
        return
    for name in descendant_names:
        index = _INDEX[name]
        if not output.position_valid[index]:
            continue
        output.joint_pose[index, :3] += delta
        changed.add(index)


def _move_foot_with_ankle(
    output: CanonicalBodyFrame,
    source: np.ndarray,
    side: str,
    scale: float,
    changed: set[int],
) -> None:
    ankle_name = f"{side}_ankle"
    ball_name = f"{side}_foot_ball"
    ankle = _point(output, ankle_name)
    ball_index = _INDEX[ball_name]
    ankle_index = _INDEX[ankle_name]
    if ankle is None or not output.position_valid[ball_index]:
        return
    offset = source[ball_index] - source[ankle_index]
    if not np.all(np.isfinite(offset)):
        return
    output.joint_pose[ball_index, :3] = ankle + scale * offset
    changed.add(ball_index)


def _copy_frame(frame: CanonicalBodyFrame) -> CanonicalBodyFrame:
    return replace(
        frame,
        **{
            field.name: np.array(getattr(frame, field.name), copy=True)
            for field in fields(CanonicalBodyFrame)
        },
    )


def _median_frame(frames: Sequence[CanonicalBodyFrame]) -> CanonicalBodyFrame:
    output = _copy_frame(frames[0])
    poses = np.stack([frame.joint_pose for frame in frames])
    valid = np.stack([frame.position_valid.astype(bool) for frame in frames])
    for index in range(len(CANONICAL_JOINTS)):
        selected = poses[valid[:, index], index, :3]
        if len(selected):
            output.joint_pose[index, :3] = np.median(selected, axis=0)
            output.position_valid[index] = 1
    return output


__all__ = [
    "BODY_CALIBRATION_SCHEMA",
    "PROFILE_SKELETON_SCHEMA",
    "ProfileConstrainedSkeleton",
    "ProfileNeutralCalibration",
    "estimate_profile_neutral_calibration",
]
