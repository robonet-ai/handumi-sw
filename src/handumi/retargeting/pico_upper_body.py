"""Upper-body helpers for PICO body-joint recordings.

The production retargeting path intentionally does not depend on recorded
PICO elbow joints.  They are useful for offline visualization, but at runtime
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

