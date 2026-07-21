"""Shared controller trajectory plans and guarded Rerun side effects.

The functions that construct :class:`RenderOp` values are deliberately pure
apart from appending to caller-owned bounded trails.  Rerun imports and logging
live behind :class:`RerunSink`, which keeps geometry/path tests independent of
the SDK and prevents a viewer failure from interrupting capture or teleop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from handumi.utils.trajectory import TrajectoryTrail
from handumi.visualization import BACKGROUND_COLOR, LEFT_COLOR, RIGHT_COLOR

CONTROLLER_VIEW_NAME = "controller_trajectory"
TRACKING_ROOT = "tracking"
LEFT_ROOT = f"{TRACKING_ROOT}/left"
RIGHT_ROOT = f"{TRACKING_ROOT}/right"
HMD_ROOT = f"{TRACKING_ROOT}/hmd"
BOUNDS_PATH = f"{TRACKING_ROOT}/bounds"
LEFT_WIDTH_PATH = "observation.feetech.left_width_mm"
RIGHT_WIDTH_PATH = "observation.feetech.right_width_mm"
RECORDING_STATUS_PATH = "recording/status"


def controller_path(side: str, entity: str) -> str:
    """Return a stable legacy controller entity path."""
    if side not in ("left", "right"):
        raise ValueError(f"Unsupported controller side {side!r}.")
    if entity not in ("tcp", "trail", "raw", "raw_trail"):
        raise ValueError(f"Unsupported controller entity {entity!r}.")
    return f"{TRACKING_ROOT}/{side}/{entity}"


@dataclass(frozen=True)
class RenderOp:
    """SDK-neutral description of one Rerun log operation."""

    path: str
    archetype: str
    data: Any = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    static: bool = False


def _finite_point(value: Any) -> np.ndarray | None:
    try:
        point = np.asarray(value, dtype=np.float32).reshape(3)
    except (TypeError, ValueError):
        return None
    return point if np.all(np.isfinite(point)) else None


def controller_render_plan(
    side: str,
    tcp_pose7: np.ndarray,
    raw_pose7: np.ndarray,
    trail: TrajectoryTrail,
    raw_trail: TrajectoryTrail,
    color: tuple[int, int, int],
    *,
    tracked: bool = True,
) -> tuple[RenderOp, ...]:
    """Plan the unchanged solid TCP and faint raw-controller visualization."""
    if not tracked:
        return ()
    tcp = _finite_point(np.asarray(tcp_pose7).reshape(-1)[:3])
    raw = _finite_point(np.asarray(raw_pose7).reshape(-1)[:3])
    if tcp is None or raw is None:
        return ()

    trail.append(tcp)
    raw_trail.append(raw)
    ops = [
        RenderOp(
            controller_path(side, "tcp"),
            "points3d",
            np.asarray([tcp]),
            {"colors": [color], "radii": 0.012},
        ),
        RenderOp(
            controller_path(side, "raw"),
            "points3d",
            np.asarray([raw]),
            {"colors": [[*color, 90]], "radii": 0.007},
        ),
    ]
    points = trail.points()
    if len(points) >= 2:
        ops.append(
            RenderOp(
                controller_path(side, "trail"),
                "line_strips3d",
                [points],
                {"colors": [color], "radii": 0.003},
            )
        )
    raw_points = raw_trail.points()
    if len(raw_points) >= 2:
        ops.append(
            RenderOp(
                controller_path(side, "raw_trail"),
                "line_strips3d",
                [raw_points],
                {"colors": [[*color, 90]], "radii": 0.0015},
            )
        )
    return tuple(ops)


def controller_current_plan(
    side: str,
    raw_pose7: np.ndarray,
    *,
    tcp_pose7: np.ndarray | None,
    color: tuple[int, int, int],
    tracked: bool,
    clear_invalid: bool = True,
) -> tuple[RenderOp, ...]:
    """Plan synchronized current points without constructing any trajectory."""
    raw = _finite_point(np.asarray(raw_pose7).reshape(-1)[:3]) if tracked else None
    tcp = (
        _finite_point(np.asarray(tcp_pose7).reshape(-1)[:3])
        if tracked and tcp_pose7 is not None
        else None
    )
    ops: list[RenderOp] = []
    if raw is None:
        if clear_invalid:
            ops.append(RenderOp(controller_path(side, "raw"), "clear"))
    else:
        ops.append(
            RenderOp(
                controller_path(side, "raw"),
                "points3d",
                np.asarray([raw]),
                {"colors": [[*color, 90]], "radii": 0.007},
            )
        )
    if tcp is None:
        if clear_invalid:
            ops.append(RenderOp(controller_path(side, "tcp"), "clear"))
    else:
        ops.append(
            RenderOp(
                controller_path(side, "tcp"),
                "points3d",
                np.asarray([tcp]),
                {"colors": [color], "radii": 0.012},
            )
        )
    return tuple(ops)


def hmd_render_plan(
    pose7: np.ndarray,
    trail: TrajectoryTrail,
    *,
    tracked: bool,
    clear_invalid: bool = False,
) -> tuple[RenderOp, ...]:
    """Plan the current HMD point and its bounded trajectory."""
    point = _finite_point(np.asarray(pose7).reshape(-1)[:3]) if tracked else None
    if point is None:
        return (RenderOp(HMD_ROOT, "clear"),) if clear_invalid else ()
    trail.append(point)
    ops = [
        RenderOp(
            HMD_ROOT,
            "points3d",
            np.asarray([point]),
            {"colors": [[120, 170, 255, 230]], "radii": 0.014, "labels": ["HMD"]},
        )
    ]
    if len(points := trail.points()) >= 2:
        ops.append(
            RenderOp(
                f"{HMD_ROOT}/trail",
                "line_strips3d",
                [points],
                {"colors": [[120, 170, 255, 150]], "radii": 0.002},
            )
        )
    return tuple(ops)


def _working_volume_corners() -> np.ndarray:
    return np.asarray(
        [
            [sx * 0.75, sy * 0.75, sz * 0.4]
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ],
        dtype=np.float32,
    )


def static_controller_ops() -> tuple[RenderOp, ...]:
    """Static style and bounds logs shared by recorder, teleop, and replay."""
    return (
        RenderOp(TRACKING_ROOT, "view_coordinates", "RIGHT_HAND_Z_UP", static=True),
        RenderOp(
            LEFT_WIDTH_PATH,
            "series_lines",
            kwargs={
                "colors": [[*LEFT_COLOR, 255]],
                "widths": [2.5],
                "names": ["left_width_mm"],
            },
            static=True,
        ),
        RenderOp(
            RIGHT_WIDTH_PATH,
            "series_lines",
            kwargs={
                "colors": [[*RIGHT_COLOR, 255]],
                "widths": [2.5],
                "names": ["right_width_mm"],
            },
            static=True,
        ),
        RenderOp(
            BOUNDS_PATH,
            "points3d",
            _working_volume_corners(),
            {"colors": [[128, 100, 100, 90]] * 8, "radii": 0.004},
            static=True,
        ),
    )


class RerunSink:
    """Translate rendering plans to the pinned Rerun SDK surface."""

    def __init__(self, rr_module: Any) -> None:
        self.rr = rr_module

    def emit(self, operations: Iterable[RenderOp]) -> None:
        for op in operations:
            self.emit_one(op)

    def emit_one(self, op: RenderOp) -> None:
        rr = self.rr
        kwargs = dict(op.kwargs)
        if op.archetype == "clear":
            archetype = rr.Clear(recursive=bool(kwargs.pop("recursive", False)))
        elif op.archetype == "points3d":
            archetype = rr.Points3D(op.data, **kwargs)
        elif op.archetype == "line_strips3d":
            archetype = rr.LineStrips3D(op.data, **kwargs)
        elif op.archetype == "mesh3d":
            archetype = rr.Mesh3D(vertex_positions=op.data, **kwargs)
        elif op.archetype == "scalars":
            archetype = rr.Scalars(op.data)
        elif op.archetype == "text_document":
            archetype = rr.TextDocument(op.data, **kwargs)
        elif op.archetype == "image":
            quality = int(kwargs.pop("jpeg_quality", 75))
            archetype = rr.Image(op.data, **kwargs).compress(jpeg_quality=quality)
        elif op.archetype == "series_lines":
            archetype = rr.SeriesLines(**kwargs)
        elif op.archetype == "view_coordinates":
            archetype = getattr(rr.ViewCoordinates, str(op.data))
        else:
            raise ValueError(f"Unknown render archetype {op.archetype!r}.")
        rr.log(op.path, archetype, static=op.static)


def build_controller_blueprint(
    rrb: Any,
    rdt: Any,
    cam_names: list[str],
    *,
    recorder_status: bool,
    include_quality: bool,
    timeline: str = "log_time",
    chart_window_s: float = 20.0,
) -> Any:
    """Build the established controller/camera/chart layout and optional quality tab."""
    recent = rrb.VisibleTimeRanges(
        rrb.VisibleTimeRange(
            timeline=timeline,
            range=rdt.TimeRange(
                start=rdt.TimeRangeBoundary.cursor_relative(seconds=-chart_window_s),
                end=rdt.TimeRangeBoundary.cursor_relative(seconds=0.0),
            ),
        )
    )
    width_chart = rrb.TimeSeriesView(
        origin="/",
        contents=[f"/{LEFT_WIDTH_PATH}", f"/{RIGHT_WIDTH_PATH}"],
        name="gripper_width_mm",
        axis_y=rrb.ScalarAxis(range=(0.0, 90.0)),
        time_ranges=recent,
        plot_legend=rrb.Corner2D.LeftTop,
    )
    charts: Any = width_chart
    if include_quality:
        quality = rrb.TimeSeriesView(
            origin="/quality/body",
            name="body_quality",
            time_ranges=recent,
            plot_legend=rrb.Corner2D.LeftTop,
        )
        charts = rrb.Tabs(width_chart, quality, active_tab=0, name="signals")
    if cam_names:
        right_column: Any = rrb.Vertical(
            rrb.Horizontal(
                *[
                    rrb.Spatial2DView(origin=f"/observation.images.{name}", name=name)
                    for name in cam_names
                ]
            ),
            charts,
            row_shares=[3, 2],
        )
    else:
        right_column = charts
    main = rrb.Horizontal(
        rrb.Spatial3DView(
            origin=f"/{TRACKING_ROOT}",
            contents=[f"/{TRACKING_ROOT}/**"],
            name=CONTROLLER_VIEW_NAME,
            background=rrb.Background(color=[*BACKGROUND_COLOR, 255]),
        ),
        right_column,
        column_shares=[2, 3],
    )
    layout: Any = main
    if recorder_status:
        status = rrb.TextDocumentView(
            origin="/recording",
            contents=[f"/{RECORDING_STATUS_PATH}"],
            name="recording_status",
        )
        layout = rrb.Vertical(main, status, row_shares=[10, 1])
    return rrb.Blueprint(
        layout,
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed"),
    )


class LiveRerunStream:
    """Guarded live Rerun stream with bounded controller/HMD/body trails."""

    def __init__(
        self,
        rr_module: Any,
        *,
        fps: int,
        trail_seconds: float = 10.0,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        from handumi.visualization.body import BodyTrailBuffer

        max_points = max(2, int(max(0.01, trail_seconds) * fps))
        self.rr = rr_module
        self.sink = RerunSink(rr_module)
        self.trails = {
            "left": TrajectoryTrail(max_points),
            "right": TrajectoryTrail(max_points),
        }
        self.raw_trails = {
            "left": TrajectoryTrail(max_points),
            "right": TrajectoryTrail(max_points),
        }
        self.hmd_trail = TrajectoryTrail(max_points)
        self.body_trail = BodyTrailBuffer(max_points)
        self.healthy = True
        self.on_error = on_error

    def _guard(self, callback: Callable[[], None]) -> None:
        if not self.healthy:
            return
        try:
            callback()
        except Exception as exc:  # Viewer failures must never own capture/control flow.
            self.healthy = False
            if self.on_error is not None:
                self.on_error(exc)

    def set_status(self, state: str, detail: str) -> None:
        self._guard(
            lambda: self.sink.emit_one(
                RenderOp(
                    RECORDING_STATUS_PATH,
                    "text_document",
                    f"# {state}\n\n{detail}",
                    {"media_type": "text/markdown"},
                )
            )
        )

    def log_frame(
        self,
        cam_frames: Mapping[str, Any],
        sample: Any,
        widths: Any,
        *,
        body_frame: Any | None = None,
    ) -> None:
        def log_all() -> None:
            from handumi.visualization.body import body_render_plan

            operations: list[RenderOp] = []
            for key, image in cam_frames.items():
                if key.startswith("observation.images."):
                    operations.append(
                        RenderOp(key, "image", image, {"jpeg_quality": 75})
                    )
            operations.extend(
                [
                    RenderOp(LEFT_WIDTH_PATH, "scalars", float(widths.left_mm)),
                    RenderOp(RIGHT_WIDTH_PATH, "scalars", float(widths.right_mm)),
                ]
            )
            for side, tcp, raw, color, tracked in (
                (
                    "left",
                    sample.left_tcp_pose,
                    sample.left_controller_pose,
                    LEFT_COLOR,
                    sample.left_tracked,
                ),
                (
                    "right",
                    sample.right_tcp_pose,
                    sample.right_controller_pose,
                    RIGHT_COLOR,
                    sample.right_tracked,
                ),
            ):
                operations.extend(
                    controller_render_plan(
                        side,
                        tcp,
                        raw,
                        self.trails[side],
                        self.raw_trails[side],
                        color,
                        tracked=bool(tracked),
                    )
                )
            operations.extend(
                hmd_render_plan(
                    sample.hmd_pose,
                    self.hmd_trail,
                    tracked=bool(sample.hmd_tracked),
                )
            )
            if body_frame is not None:
                operations.extend(body_render_plan(body_frame, trail=self.body_trail))
            self.sink.emit(operations)

        self._guard(log_all)


def initialize_rerun(
    application_id: str,
    cam_names: list[str],
    *,
    fps: int,
    spawn: bool,
    recorder_status: bool,
    include_quality: bool = True,
    save_path: str | Path | None = None,
    timeline: str = "log_time",
    recording_id: str | None = None,
    on_error: Callable[[BaseException], None] | None = None,
) -> LiveRerunStream | None:
    """Initialize one guarded stream; return ``None`` on any viewer failure."""
    try:
        import rerun as rr
        import rerun.blueprint as rrb
        import rerun.datatypes as rdt

        rr.init(application_id, recording_id=recording_id, spawn=spawn)
        if save_path is not None:
            rr.save(Path(save_path))
        sink = RerunSink(rr)
        sink.emit(static_controller_ops())
        blueprint = build_controller_blueprint(
            rrb,
            rdt,
            cam_names,
            recorder_status=recorder_status,
            include_quality=include_quality,
            timeline=timeline,
        )
        rr.send_blueprint(blueprint, make_active=True, make_default=True)
        return LiveRerunStream(rr, fps=fps, on_error=on_error)
    except Exception as exc:
        if on_error is not None:
            on_error(exc)
        return None


__all__ = [
    "BOUNDS_PATH",
    "CONTROLLER_VIEW_NAME",
    "HMD_ROOT",
    "LEFT_ROOT",
    "LEFT_WIDTH_PATH",
    "LiveRerunStream",
    "RECORDING_STATUS_PATH",
    "RIGHT_ROOT",
    "RIGHT_WIDTH_PATH",
    "RenderOp",
    "RerunSink",
    "TRACKING_ROOT",
    "build_controller_blueprint",
    "controller_current_plan",
    "controller_path",
    "controller_render_plan",
    "hmd_render_plan",
    "initialize_rerun",
    "static_controller_ops",
]
