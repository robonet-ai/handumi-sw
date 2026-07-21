"""Software-only synthetic/headless capture soak."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

from handumi.reliability import (
    BoundedLatestWorker,
    StageProfiler,
    atomic_write_json,
    process_resource_snapshot,
    resolve_capture_profile,
)


def run_soak(
    *,
    duration_s: float,
    dataset_hz: float,
    camera_fps: float,
    camera_count: int,
    output: Path,
) -> dict[str, object]:
    if duration_s <= 0 or dataset_hz <= 0 or camera_fps <= 0 or camera_count < 0:
        raise ValueError(
            "duration/rates must be positive and camera_count non-negative"
        )
    profile = resolve_capture_profile(dataset_hz, camera_fps, camera_count)
    profiler = StageProfiler()
    resource_start = process_resource_snapshot()
    resource_peak = dict(resource_start)
    encoded = hashlib.sha256()

    def consume(payload: object) -> None:
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("synthetic worker payload must be bytes-like")
        encoded.update(payload)

    workers = {
        name: BoundedLatestWorker(name, consume, maxsize=2, profiler=profiler)
        for name in ("video_encoding", "rerun", "viser", "robot_ik")
    }
    started_wall = datetime.now(UTC).isoformat()
    started = time.monotonic()
    deadline = started + duration_s
    rows = 0
    cameras = 0
    next_row = started
    next_camera = started
    sample = b"handumi-synthetic-headless-frame"
    try:
        while True:
            now = time.monotonic()
            if now >= deadline:
                break
            while now >= next_camera:
                with profiler.measure("camera_acquisition", items=camera_count):
                    cameras += camera_count
                next_camera += 1.0 / camera_fps
            if now >= next_row:
                with profiler.measure("tracking_reception_clock_alignment"):
                    pass
                with profiler.measure("body_canonicalization_estimation"):
                    pass
                with profiler.measure("camera_synchronization", items=camera_count):
                    pass
                for worker in workers.values():
                    worker.submit(sample)
                with profiler.measure("dataset_serialization_writes"):
                    encoded.update(rows.to_bytes(8, "little"))
                rows += 1
                next_row += 1.0 / dataset_hz
            current = process_resource_snapshot()
            resource_peak["rss_bytes"] = max(
                resource_peak["rss_bytes"], current["rss_bytes"]
            )
            resource_peak["file_descriptors"] = max(
                resource_peak["file_descriptors"], current["file_descriptors"]
            )
            time.sleep(min(0.002, max(0.0, min(next_row, next_camera, deadline) - now)))
    finally:
        for worker in workers.values():
            worker.close()
    elapsed = time.monotonic() - started
    resource_end = process_resource_snapshot()
    achieved_hz = rows / elapsed
    minimum_rows = math.floor(duration_s * dataset_hz * 0.98)
    maintained = rows >= minimum_rows
    result: dict[str, object] = {
        "schema": "handumi_software_soak_v1",
        "classification": "software-only synthetic/headless; no Quest, camera, thermal, radio, or physical-hardware evidence",
        "started_at": started_wall,
        "ended_at": datetime.now(UTC).isoformat(),
        "wall_clock_duration_s": elapsed,
        "profile": profile.__dict__,
        "rows": rows,
        "synthetic_camera_frames": cameras,
        "achieved_dataset_hz": achieved_hz,
        "maintained_requested_profile": maintained,
        "resources": {
            "start": resource_start,
            "end": resource_end,
            "peak": resource_peak,
        },
        "profiling": profiler.snapshot(),
        "payload_sha256": encoded.hexdigest(),
    }
    atomic_write_json(output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=1800.0)
    parser.add_argument("--dataset-hz", type=float, default=10.0)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--camera-count", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_soak(
        duration_s=args.duration_s,
        dataset_hz=args.dataset_hz,
        camera_fps=args.camera_fps,
        camera_count=args.camera_count,
        output=args.output,
    )
    print(json.dumps(result, sort_keys=True))
    if not result["maintained_requested_profile"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
