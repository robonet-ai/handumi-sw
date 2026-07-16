"""Canonical-body geometry, styling, quality signals, and trajectory planning."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields
from typing import Any, Iterable, Mapping

import numpy as np

from handumi.body.model import (
    BODY_PREFIX,
    CANONICAL_JOINTS,
    CanonicalBodyFrame,
    CanonicalClockQuality,
    CanonicalProvenance,
    CanonicalTrackingState,
    ComDiagnostic,
    ComProvenance,
)
from handumi.visualization.controller_trajectory import RenderOp

BODY_ROOT = "tracking/body"
BODY_JOINTS_PATH = f"{BODY_ROOT}/joints"
BODY_SKELETON_PATH = f"{BODY_ROOT}/skeleton"
SEGMENT_COM_PATH = f"{BODY_ROOT}/segment_com"
WHOLE_COM_PATH = f"{BODY_ROOT}/whole_com"
WHOLE_COM_TRAIL_PATH = f"{WHOLE_COM_PATH}/trail"
COM_PROJECTION_PATH = f"{BODY_ROOT}/com_projection"
COM_VERTICAL_PATH = f"{COM_PROJECTION_PATH}/vertical"
GROUND_PATH = f"{BODY_ROOT}/ground"
CONTACTS_PATH = f"{BODY_ROOT}/contacts"
SUPPORT_POLYGON_PATH = f"{BODY_ROOT}/support_polygon"
CENTER_OF_PRESSURE_PATH = f"{BODY_ROOT}/center_of_pressure"
BODY_QUALITY_ROOT = "quality/body"

PLATFORM_COLOR = (50, 220, 235)
KINEMATIC_COLOR = (245, 170, 45)
LEARNED_COLOR = (245, 70, 210)
FUSED_COLOR = (245, 245, 245)
DEVICE_REPORTED_COLOR = (90, 150, 255)
EXTERNAL_TRACKER_COLOR = (175, 120, 255)
SYNTHETIC_COLOR = (150, 235, 120)
UNKNOWN_COLOR = (150, 150, 155)

_CONTACT_JOINTS = (
    "left_heel",
    "left_foot_ball",
    "right_heel",
    "right_foot_ball",
)
_JOINT_INDEX = {joint.identifier: joint.index for joint in CANONICAL_JOINTS}


@dataclass(frozen=True)
class VisualStyle:
    color: tuple[int, int, int, int]
    low_confidence: bool
    provenance_label: str


def _enum_name(enum_type: type, value: Any, fallback: str) -> str:
    try:
        return enum_type(int(value)).name
    except (TypeError, ValueError):
        name = getattr(value, "name", None)
        return str(name) if name else fallback


def confidence_alpha(confidence: Any) -> tuple[int, bool]:
    """Map confidence deterministically to visible alpha plus a quality flag."""
    try:
        numeric = float(confidence)
    except (TypeError, ValueError):
        numeric = 0.0
    if not np.isfinite(numeric):
        numeric = 0.0
    numeric = float(np.clip(numeric, 0.0, 1.0))
    return int(round(64 + 191 * numeric)), numeric < 0.5


def provenance_style(
    provenance: Any,
    confidence: Any,
    *,
    com: bool = False,
) -> VisualStyle:
    """Return centralized colors with documented future-value fallbacks.

    A future enum whose symbolic name contains ``LEARNED`` gets magenta.
    Unknown numeric values get neutral gray instead of raising or being called
    measured.
    """
    alpha, low = confidence_alpha(confidence)
    if com:
        name = _enum_name(ComProvenance, provenance, f"UNKNOWN_{provenance}")
        colors = {
            ComProvenance.UNAVAILABLE.name: UNKNOWN_COLOR,
            ComProvenance.KINEMATIC_INFERRED.name: KINEMATIC_COLOR,
            ComProvenance.FUSED_ESTIMATED.name: FUSED_COLOR,
        }
    else:
        name = _enum_name(CanonicalProvenance, provenance, f"UNKNOWN_{provenance}")
        colors = {
            CanonicalProvenance.UNAVAILABLE.name: UNKNOWN_COLOR,
            CanonicalProvenance.PLATFORM_ESTIMATED.name: PLATFORM_COLOR,
            CanonicalProvenance.DEVICE_REPORTED.name: DEVICE_REPORTED_COLOR,
            CanonicalProvenance.EXTERNAL_TRACKER.name: EXTERNAL_TRACKER_COLOR,
            CanonicalProvenance.INFERRED.name: KINEMATIC_COLOR,
            CanonicalProvenance.SYNTHETIC_TEST.name: SYNTHETIC_COLOR,
            CanonicalProvenance.UNKNOWN.name: UNKNOWN_COLOR,
        }
    if "LEARNED" in name.upper():
        rgb = LEARNED_COLOR
    else:
        rgb = colors.get(name, UNKNOWN_COLOR)
    return VisualStyle((*rgb, alpha), low, name)


def _valid_point(value: Any, valid: Any = True) -> np.ndarray | None:
    if not bool(valid):
        return None
    try:
        point = np.asarray(value, dtype=np.float32).reshape(3)
    except (TypeError, ValueError):
        return None
    return point if np.all(np.isfinite(point)) else None


def canonical_skeleton_edges(frame: CanonicalBodyFrame) -> tuple[np.ndarray, ...]:
    """Build only parent/child edges whose current positions are both valid."""
    edges: list[np.ndarray] = []
    for child in CANONICAL_JOINTS:
        if child.parent_index < 0:
            continue
        child_point = _valid_point(
            frame.joint_pose[child.index, :3], frame.position_valid[child.index]
        )
        parent_point = _valid_point(
            frame.joint_pose[child.parent_index, :3],
            frame.position_valid[child.parent_index],
        )
        if child_point is not None and parent_point is not None:
            edges.append(np.stack([parent_point, child_point]))
    return tuple(edges)


def support_polygon_strip(frame: CanonicalBodyFrame) -> np.ndarray | None:
    """Return the stored valid hull order, explicitly closed for rendering."""
    points: list[np.ndarray] = []
    for value, valid in zip(
        frame.support_polygon, frame.support_polygon_valid, strict=True
    ):
        point = _valid_point(value, valid)
        if point is not None:
            points.append(point)
    if len(points) < 2:
        return None
    return np.asarray([*points, points[0]], dtype=np.float32)


def ground_plane_geometry(
    plane: Any,
    *,
    half_extent_m: float = 1.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Create a square mesh on ``n dot x + d = 0`` for any plane orientation."""
    try:
        value = np.asarray(plane, dtype=np.float64).reshape(4)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(value)):
        return None
    normal = value[:3]
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-9:
        return None
    normal /= norm
    offset = float(value[3]) / norm
    center = -offset * normal
    helper = np.zeros(3, dtype=np.float64)
    helper[int(np.argmin(np.abs(normal)))] = 1.0
    axis_u = np.cross(normal, helper)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    extent = max(0.01, float(half_extent_m))
    vertices = np.asarray(
        [
            center - extent * axis_u - extent * axis_v,
            center + extent * axis_u - extent * axis_v,
            center + extent * axis_u + extent * axis_v,
            center - extent * axis_u + extent * axis_v,
        ],
        dtype=np.float32,
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    return vertices, triangles


class BodyTrailBuffer:
    """Bounded CoM samples with explicit invalid gaps."""

    def __init__(self, max_points: int) -> None:
        self._samples: deque[np.ndarray | None] = deque(maxlen=max(1, max_points))

    def append(self, position: Any | None) -> None:
        point = None if position is None else _valid_point(position)
        self._samples.append(None if point is None else point.copy())

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def max_points(self) -> int:
        return int(self._samples.maxlen or 0)

    def strips(self) -> tuple[np.ndarray, ...]:
        strips: list[np.ndarray] = []
        current: list[np.ndarray] = []
        for sample in self._samples:
            if sample is None:
                if len(current) >= 2:
                    strips.append(np.asarray(current, dtype=np.float32))
                current = []
            else:
                current.append(sample)
        if len(current) >= 2:
            strips.append(np.asarray(current, dtype=np.float32))
        return tuple(strips)


def _clear(path: str) -> RenderOp:
    return RenderOp(path, "clear")


def _quality_ops(frame: CanonicalBodyFrame) -> list[RenderOp]:
    finite = lambda value: np.nan_to_num(  # noqa: E731
        np.asarray(value, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    clock_name = _enum_name(
        CanonicalClockQuality,
        frame.clock_quality[0],
        f"UNKNOWN_{frame.clock_quality[0]}",
    )
    diagnostic = _enum_name(
        ComDiagnostic,
        frame.whole_com_diagnostic[0],
        f"UNKNOWN_{frame.whole_com_diagnostic[0]}",
    )
    tracking_names = [
        _enum_name(CanonicalTrackingState, value, f"UNKNOWN_{value}")
        for value in frame.tracking_state
    ]
    provenance_names = [
        _enum_name(CanonicalProvenance, value, f"UNKNOWN_{value}")
        for value in frame.provenance
    ]
    low_count = int(
        np.count_nonzero(
            np.asarray(frame.position_valid, dtype=bool)
            & (finite(frame.confidence) < 0.5)
        )
    )
    summary = (
        f"tracking={','.join(sorted(set(tracking_names)))}; "
        f"provenance={','.join(sorted(set(provenance_names)))}; "
        f"clock={clock_name}; com_diagnostic={diagnostic}; "
        f"low_confidence_joints={low_count}"
    )
    return [
        RenderOp(f"{BODY_QUALITY_ROOT}/joints/confidence", "scalars", finite(frame.confidence)),
        RenderOp(f"{BODY_QUALITY_ROOT}/joints/tracking_state", "scalars", finite(frame.tracking_state)),
        RenderOp(f"{BODY_QUALITY_ROOT}/joints/provenance", "scalars", finite(frame.provenance)),
        RenderOp(f"{BODY_QUALITY_ROOT}/segments/confidence", "scalars", finite(frame.segment_com_confidence)),
        RenderOp(f"{BODY_QUALITY_ROOT}/segments/provenance", "scalars", finite(frame.segment_com_provenance)),
        RenderOp(f"{BODY_QUALITY_ROOT}/whole_com/confidence", "scalars", finite(frame.whole_com_confidence)),
        RenderOp(f"{BODY_QUALITY_ROOT}/whole_com/provenance", "scalars", finite(frame.whole_com_provenance)),
        RenderOp(f"{BODY_QUALITY_ROOT}/whole_com/diagnostic", "scalars", finite(frame.whole_com_diagnostic)),
        RenderOp(
            f"{BODY_QUALITY_ROOT}/whole_com/unresolved_mass_fraction",
            "scalars",
            finite(frame.whole_com_unresolved_mass_fraction),
        ),
        RenderOp(f"{BODY_QUALITY_ROOT}/contacts/probability", "scalars", finite(frame.contact_probability)),
        RenderOp(f"{BODY_QUALITY_ROOT}/contacts/provenance", "scalars", finite(frame.contact_provenance)),
        RenderOp(f"{BODY_QUALITY_ROOT}/clock_quality", "scalars", finite(frame.clock_quality)),
        RenderOp(f"{BODY_QUALITY_ROOT}/low_confidence_joint_count", "scalars", low_count),
        RenderOp(f"{BODY_QUALITY_ROOT}/state", "text_document", summary),
    ]


def body_render_plan(
    frame: CanonicalBodyFrame,
    *,
    trail: BodyTrailBuffer | None = None,
    log_trail: bool = True,
    ground_half_extent_m: float = 1.0,
) -> tuple[RenderOp, ...]:
    """Plan one canonical body frame, clearing every invalid current entity."""
    ops: list[RenderOp] = []
    joint_points: list[np.ndarray] = []
    joint_colors: list[tuple[int, int, int, int]] = []
    for joint in CANONICAL_JOINTS:
        point = _valid_point(
            frame.joint_pose[joint.index, :3], frame.position_valid[joint.index]
        )
        if point is None:
            continue
        style = provenance_style(
            frame.provenance[joint.index], frame.confidence[joint.index]
        )
        joint_points.append(point)
        joint_colors.append(style.color)
    if joint_points:
        ops.append(
            RenderOp(
                BODY_JOINTS_PATH,
                "points3d",
                np.asarray(joint_points),
                {"colors": joint_colors, "radii": 0.012},
            )
        )
    else:
        ops.append(_clear(BODY_JOINTS_PATH))

    edges = canonical_skeleton_edges(frame)
    if edges:
        edge_colors = []
        for child in CANONICAL_JOINTS:
            if child.parent_index < 0:
                continue
            child_point = _valid_point(
                frame.joint_pose[child.index, :3], frame.position_valid[child.index]
            )
            parent_point = _valid_point(
                frame.joint_pose[child.parent_index, :3],
                frame.position_valid[child.parent_index],
            )
            if child_point is not None and parent_point is not None:
                edge_colors.append(
                    provenance_style(
                        frame.provenance[child.index], frame.confidence[child.index]
                    ).color
                )
        ops.append(
            RenderOp(
                BODY_SKELETON_PATH,
                "line_strips3d",
                list(edges),
                {"colors": edge_colors, "radii": 0.004},
            )
        )
    else:
        ops.append(_clear(BODY_SKELETON_PATH))

    segment_points: list[np.ndarray] = []
    segment_colors: list[tuple[int, int, int, int]] = []
    for joint in CANONICAL_JOINTS:
        point = _valid_point(frame.segment_com[joint.index], frame.segment_com_valid[joint.index])
        if point is None:
            continue
        style = provenance_style(
            frame.segment_com_provenance[joint.index],
            frame.segment_com_confidence[joint.index],
            com=True,
        )
        segment_points.append(point)
        segment_colors.append(style.color)
    if segment_points:
        ops.append(
            RenderOp(
                SEGMENT_COM_PATH,
                "points3d",
                np.asarray(segment_points),
                {"colors": segment_colors, "radii": 0.009},
            )
        )
    else:
        ops.append(_clear(SEGMENT_COM_PATH))

    whole = _valid_point(frame.whole_com, frame.whole_com_valid[0])
    if whole is not None:
        whole_style = provenance_style(
            frame.whole_com_provenance[0],
            frame.whole_com_confidence[0],
            com=True,
        )
        ops.append(
            RenderOp(
                WHOLE_COM_PATH,
                "points3d",
                np.asarray([whole]),
                {
                    "colors": [whole_style.color],
                    "radii": 0.018,
                },
            )
        )
    else:
        ops.append(_clear(WHOLE_COM_PATH))

    if trail is not None:
        trail.append(whole)
        if log_trail:
            strips = trail.strips()
            if strips:
                ops.append(
                    RenderOp(
                        WHOLE_COM_TRAIL_PATH,
                        "line_strips3d",
                        list(strips),
                        {"colors": [[245, 245, 245, 180]], "radii": 0.003},
                    )
                )
            else:
                ops.append(_clear(WHOLE_COM_TRAIL_PATH))

    projection = _valid_point(
        frame.whole_com_ground_projection,
        frame.whole_com_ground_projection_valid[0],
    )
    if projection is not None:
        ops.append(
            RenderOp(
                COM_PROJECTION_PATH,
                "points3d",
                np.asarray([projection]),
                {
                    "colors": [[245, 245, 245, 220]],
                    "radii": 0.014,
                },
            )
        )
    else:
        ops.append(_clear(COM_PROJECTION_PATH))
    if whole is not None and projection is not None:
        ops.append(
            RenderOp(
                COM_VERTICAL_PATH,
                "line_strips3d",
                [np.stack([whole, projection])],
                {"colors": [[245, 245, 245, 150]], "radii": 0.002},
            )
        )
    else:
        ops.append(_clear(COM_VERTICAL_PATH))

    ground = ground_plane_geometry(frame.ground_plane, half_extent_m=ground_half_extent_m)
    if ground is None:
        ops.append(_clear(GROUND_PATH))
    else:
        vertices, triangles = ground
        ops.append(
            RenderOp(
                GROUND_PATH,
                "mesh3d",
                vertices,
                {"triangle_indices": triangles, "albedo_factor": [95, 110, 125, 65]},
            )
        )

    contact_points: list[np.ndarray] = []
    contact_colors: list[tuple[int, int, int, int]] = []
    for contact_index, joint_name in enumerate(_CONTACT_JOINTS):
        joint_index = _JOINT_INDEX[joint_name]
        point = _valid_point(
            frame.joint_pose[joint_index, :3],
            bool(frame.contact_valid[contact_index])
            and bool(frame.position_valid[joint_index]),
        )
        probability = float(frame.contact_probability[contact_index])
        if point is None or not np.isfinite(probability):
            continue
        probability = float(np.clip(probability, 0.0, 1.0))
        style = provenance_style(
            frame.contact_provenance[contact_index], probability, com=True
        )
        contact_points.append(point)
        contact_colors.append(style.color)
    if contact_points:
        ops.append(
            RenderOp(
                CONTACTS_PATH,
                "points3d",
                np.asarray(contact_points),
                {"colors": contact_colors, "radii": 0.016},
            )
        )
    else:
        ops.append(_clear(CONTACTS_PATH))

    polygon = support_polygon_strip(frame)
    if polygon is None:
        ops.append(_clear(SUPPORT_POLYGON_PATH))
    else:
        ops.append(
            RenderOp(
                SUPPORT_POLYGON_PATH,
                "line_strips3d",
                [polygon],
                {"colors": [[245, 245, 245, 220]], "radii": 0.005},
            )
        )

    cop = _valid_point(frame.center_of_pressure, frame.center_of_pressure_valid[0])
    if cop is None:
        ops.append(_clear(CENTER_OF_PRESSURE_PATH))
    else:
        ops.append(
            RenderOp(
                CENTER_OF_PRESSURE_PATH,
                "points3d",
                np.asarray([cop]),
                {"colors": [[255, 100, 100, 255]], "radii": 0.014},
            )
        )

    ops.extend(_quality_ops(frame))
    return tuple(ops)


def body_frame_at(signals: Mapping[str, np.ndarray], index: int) -> CanonicalBodyFrame:
    """Reconstruct one frame from reader-validated canonical episode columns."""
    empty = CanonicalBodyFrame.empty()
    values: dict[str, np.ndarray] = {}
    for definition in fields(CanonicalBodyFrame):
        default = np.asarray(getattr(empty, definition.name))
        key = f"{BODY_PREFIX}.{definition.name}"
        if key not in signals:
            values[definition.name] = default.copy()
            continue
        value = np.asarray(signals[key][index], dtype=default.dtype)
        try:
            values[definition.name] = value.reshape(default.shape).copy()
        except ValueError as exc:
            raise ValueError(
                f"Canonical body column {key!r} frame {index} has shape "
                f"{value.shape}, expected {default.shape}."
            ) from exc
    return CanonicalBodyFrame(**values)


@dataclass(frozen=True)
class TrajectoryStrip:
    points: np.ndarray
    indices: np.ndarray


def decimate_trajectory(
    points: Any,
    valid: Any | None = None,
    *,
    temporal_step: int = 1,
    spatial_step_m: float = 0.0,
) -> tuple[TrajectoryStrip, ...]:
    """Deterministically decimate contiguous valid runs in one O(n) pass."""
    positions = np.asarray(points, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected trajectory shape (n, 3), got {positions.shape}.")
    mask = np.all(np.isfinite(positions), axis=1)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool).reshape(len(positions))
    step = max(1, int(temporal_step))
    distance = max(0.0, float(spatial_step_m))
    strips: list[TrajectoryStrip] = []
    run: list[int] = []

    def finish(indices: list[int]) -> None:
        if len(indices) < 2:
            return
        candidates = [indices[0]]
        candidates.extend(i for i in indices[1:-1] if (i - indices[0]) % step == 0)
        candidates.append(indices[-1])
        selected = [candidates[0]]
        for candidate in candidates[1:-1]:
            if np.linalg.norm(positions[candidate] - positions[selected[-1]]) >= distance:
                selected.append(candidate)
        if candidates[-1] != selected[-1]:
            selected.append(candidates[-1])
        if len(selected) >= 2:
            selected_array = np.asarray(selected, dtype=np.int64)
            strips.append(
                TrajectoryStrip(positions[selected_array].copy(), selected_array)
            )

    for index, is_valid in enumerate(mask):
        if is_valid:
            run.append(index)
        else:
            finish(run)
            run = []
    finish(run)
    return tuple(strips)


def chunk_trajectory(
    strips: Iterable[TrajectoryStrip],
    *,
    point_cap: int,
    duration_frames: int | None = None,
) -> tuple[TrajectoryStrip, ...]:
    """Split long strips with one-point overlap so paths remain continuous."""
    cap = max(2, int(point_cap))
    duration = None if duration_frames is None else max(1, int(duration_frames))
    chunks: list[TrajectoryStrip] = []
    for strip in strips:
        start = 0
        count = len(strip.points)
        while start < count - 1:
            end = min(count, start + cap)
            if duration is not None:
                max_index = int(strip.indices[start]) + duration
                while end > start + 2 and int(strip.indices[end - 1]) > max_index:
                    end -= 1
            if end <= start + 1:
                end = min(count, start + 2)
            chunks.append(
                TrajectoryStrip(
                    strip.points[start:end].copy(), strip.indices[start:end].copy()
                )
            )
            if end == count:
                break
            start = end - 1
    return tuple(chunks)


def full_trajectory_plan(
    path: str,
    points: Any,
    valid: Any | None,
    *,
    color: tuple[int, ...],
    radius: float,
    temporal_step: int,
    spatial_step_m: float,
    point_cap: int,
    duration_frames: int | None,
    static: bool = True,
) -> tuple[RenderOp, ...]:
    """Plan a full gap-preserving path once, never once per cursor frame."""
    strips = chunk_trajectory(
        decimate_trajectory(
            points,
            valid,
            temporal_step=temporal_step,
            spatial_step_m=spatial_step_m,
        ),
        point_cap=point_cap,
        duration_frames=duration_frames,
    )
    if not strips:
        return (RenderOp(path, "clear", static=static),)
    return (
        RenderOp(
            path,
            "line_strips3d",
            [strip.points for strip in strips],
            {"colors": [color] * len(strips), "radii": radius},
            static=static,
        ),
    )


__all__ = [
    "BODY_JOINTS_PATH",
    "BODY_QUALITY_ROOT",
    "BODY_ROOT",
    "BODY_SKELETON_PATH",
    "BodyTrailBuffer",
    "CENTER_OF_PRESSURE_PATH",
    "COM_PROJECTION_PATH",
    "COM_VERTICAL_PATH",
    "CONTACTS_PATH",
    "FUSED_COLOR",
    "GROUND_PATH",
    "KINEMATIC_COLOR",
    "LEARNED_COLOR",
    "PLATFORM_COLOR",
    "SEGMENT_COM_PATH",
    "SUPPORT_POLYGON_PATH",
    "TrajectoryStrip",
    "UNKNOWN_COLOR",
    "VisualStyle",
    "WHOLE_COM_PATH",
    "WHOLE_COM_TRAIL_PATH",
    "body_frame_at",
    "body_render_plan",
    "canonical_skeleton_edges",
    "chunk_trajectory",
    "confidence_alpha",
    "decimate_trajectory",
    "full_trajectory_plan",
    "ground_plane_geometry",
    "provenance_style",
    "support_polygon_strip",
]
