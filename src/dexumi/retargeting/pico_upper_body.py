"""Upper-body helpers for PICO body-joint recordings.

The production retargeting path intentionally does not depend on recorded
PICO elbow joints.  They are useful for offline diagnostics, but at runtime
we infer elbows from shoulder and wrist positions so the same method can be
used with controller-only recordings.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# SMPL-like 24-joint parent chain used by SONIC / XRoboToolkit body data.
SMPL24_PARENT_INDICES: list[int] = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
]

LEFT_SHOULDER = 16
RIGHT_SHOULDER = 17
LEFT_ELBOW = 18
RIGHT_ELBOW = 19
LEFT_WRIST = 20
RIGHT_WRIST = 21
LEFT_HAND = 22
RIGHT_HAND = 23

UPPER_BODY_JOINTS: list[int] = [
    0,
    3,
    6,
    9,
    12,
    13,
    14,
    15,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_HAND,
    RIGHT_HAND,
]
UPPER_BODY_INDEX: dict[int, int] = {
    joint: index for index, joint in enumerate(UPPER_BODY_JOINTS)
}


def parse_axis_map(spec: str) -> Callable[[np.ndarray], np.ndarray]:
    """Build a vector transform from a spec like ``z,x,y`` or ``z,y,-x``."""

    axes = {"x": 0, "y": 1, "z": 2}
    parts = [part.strip().lower() for part in spec.split(",")]
    if len(parts) != 3:
        raise ValueError("--axis-map must contain exactly 3 comma-separated axes.")

    rows: list[tuple[int, int]] = []
    used: set[int] = set()
    for part in parts:
        sign = -1 if part.startswith("-") else 1
        axis = part[1:] if part.startswith("-") else part
        if axis not in axes:
            raise ValueError(f"Invalid axis {part!r}; use x, y, z, or a negated axis.")
        index = axes[axis]
        if index in used:
            raise ValueError(f"Axis {axis!r} is repeated in --axis-map.")
        used.add(index)
        rows.append((sign, index))

    def transform(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        return np.asarray([sign * vector[index] for sign, index in rows], dtype=np.float32)

    return transform


def parent_indices_to_lines(parent_indices: list[int]) -> list[int]:
    """Convert a parent-index skeleton to PyVista line-cell format."""

    cells: list[int] = []
    for child, parent in enumerate(parent_indices):
        if parent < 0:
            continue
        cells.extend([2, parent, child])
    return cells


def upper_body_lines() -> np.ndarray:
    """Return line cells for the compact upper-body skeleton."""

    cells: list[int] = []
    for child in UPPER_BODY_JOINTS:
        parent = SMPL24_PARENT_INDICES[child]
        if parent in UPPER_BODY_INDEX:
            cells.extend([2, UPPER_BODY_INDEX[parent], UPPER_BODY_INDEX[child]])
    return np.asarray(cells, dtype=np.int_)


def estimate_arm_lengths(
    poses: np.ndarray,
    *,
    shoulder_index: int,
    wrist_index: int,
    upper_ratio: float = 0.44,
    extension_ratio: float = 0.92,
    percentile: float = 95.0,
) -> tuple[float, float]:
    """Estimate upper-arm and forearm lengths from shoulder-wrist distance only.

    ``extension_ratio`` accounts for the fact that the dataset rarely contains
    a perfectly straight arm.  A high percentile avoids making the arm too short
    from bent frames.
    """

    positions = np.asarray(poses[:, :, :3], dtype=np.float32)
    distances = np.linalg.norm(
        positions[:, wrist_index] - positions[:, shoulder_index], axis=1
    )
    arm_length = float(np.percentile(distances, percentile) / extension_ratio)
    return arm_length * upper_ratio, arm_length * (1.0 - upper_ratio)


def infer_elbow(
    shoulder: np.ndarray,
    wrist: np.ndarray,
    *,
    upper_length: float,
    forearm_length: float,
    bend_hint: np.ndarray,
) -> np.ndarray:
    """Infer one elbow from shoulder/wrist using a two-sphere intersection."""

    shoulder = np.asarray(shoulder, dtype=np.float32)
    wrist = np.asarray(wrist, dtype=np.float32)
    axis = wrist - shoulder
    distance = float(np.linalg.norm(axis))
    if distance < 1e-6:
        return shoulder.copy()

    direction = axis / distance
    max_reach = upper_length + forearm_length
    min_reach = abs(upper_length - forearm_length) + 1e-4
    d = float(np.clip(distance, min_reach, max_reach - 1e-4))

    along = (upper_length**2 - forearm_length**2 + d**2) / (2.0 * d)
    height_sq = max(upper_length**2 - along**2, 0.0)
    center = shoulder + direction * along

    hint = np.asarray(bend_hint, dtype=np.float32)
    hint = hint - direction * float(np.dot(hint, direction))
    norm = float(np.linalg.norm(hint))
    if norm < 1e-6:
        hint = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        hint = hint - direction * float(np.dot(hint, direction))
        norm = max(float(np.linalg.norm(hint)), 1e-6)

    bend_direction = hint / norm
    return (center + bend_direction * np.sqrt(height_sq)).astype(np.float32)


def infer_pose_elbows(
    body_pose: np.ndarray,
    *,
    left_lengths: tuple[float, float],
    right_lengths: tuple[float, float],
    bend_forward: float = 0.65,
    bend_down: float = -1.0,
    bend_side: float = 0.25,
) -> np.ndarray:
    """Return a copy of ``body_pose`` with left/right elbows inferred."""

    inferred = np.asarray(body_pose[:, :3], dtype=np.float32).copy()
    left_hint = np.array([bend_forward, bend_down, bend_side], dtype=np.float32)
    right_hint = np.array([bend_forward, bend_down, -bend_side], dtype=np.float32)
    inferred[LEFT_ELBOW] = infer_elbow(
        inferred[LEFT_SHOULDER],
        inferred[LEFT_WRIST],
        upper_length=left_lengths[0],
        forearm_length=left_lengths[1],
        bend_hint=left_hint,
    )
    inferred[RIGHT_ELBOW] = infer_elbow(
        inferred[RIGHT_SHOULDER],
        inferred[RIGHT_WRIST],
        upper_length=right_lengths[0],
        forearm_length=right_lengths[1],
        bend_hint=right_hint,
    )
    return inferred
