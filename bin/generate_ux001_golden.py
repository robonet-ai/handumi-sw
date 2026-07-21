#!/usr/bin/env python3
"""Generate deterministic UX-001 headless Rerun evidence.

Example:
    PYTHONPATH=src python bin/generate_ux001_golden.py /tmp/handumi-ux001/body.rrd
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from handumi.body.model import (
    CANONICAL_JOINTS,
    CanonicalBodyFrame,
    CanonicalProvenance,
    CanonicalTrackingState,
    ComDiagnostic,
    ComProvenance,
)
from handumi.dataset.reader import CanonicalBodyEpisode, RawEpisode
from handumi.scripts.view_trajectory import ViewerOptions, log_episode
from handumi.visualization.controller_trajectory import initialize_rerun


def _body_frame(frame_index: int) -> CanonicalBodyFrame:
    frame = CanonicalBodyFrame.empty()
    if frame_index == 3:  # deterministic invalid gap proving stale clearing.
        return frame
    for joint in CANONICAL_JOINTS:
        frame.joint_pose[joint.index, :3] = [
            0.02 * joint.index + 0.01 * frame_index,
            -0.15 if joint.identifier.startswith("left_") else 0.15,
            0.45 + 0.025 * joint.index,
        ]
        frame.joint_pose[joint.index, 3:] = [0, 0, 0, 1]
        frame.position_valid[joint.index] = 1
        frame.orientation_valid[joint.index] = 1
        frame.tracking_state[joint.index] = int(CanonicalTrackingState.TRACKED)
        frame.confidence[joint.index] = 0.9 - 0.01 * (joint.index % 5)
        frame.provenance[joint.index] = int(CanonicalProvenance.PLATFORM_ESTIMATED)
        frame.segment_com[joint.index] = frame.joint_pose[joint.index, :3]
        frame.segment_com_valid[joint.index] = 1
        frame.segment_com_confidence[joint.index] = 0.8
        frame.segment_com_provenance[joint.index] = int(
            ComProvenance.KINEMATIC_INFERRED
        )
    frame.whole_com[:] = [0.2 + 0.01 * frame_index, 0.0, 0.9]
    frame.whole_com_valid[0] = 1
    frame.whole_com_confidence[0] = 0.82
    frame.whole_com_provenance[0] = int(ComProvenance.KINEMATIC_INFERRED)
    frame.whole_com_diagnostic[0] = int(ComDiagnostic.VALID)
    frame.whole_com_ground_projection[:] = [frame.whole_com[0], 0.0, 0.0]
    frame.whole_com_ground_projection_valid[0] = 1
    frame.ground_plane[:] = [0, 0, 1, 0]
    contact_names = ("left_heel", "left_foot_ball", "right_heel", "right_foot_ball")
    for index, name in enumerate(contact_names):
        joint_index = next(j.index for j in CANONICAL_JOINTS if j.identifier == name)
        frame.joint_pose[joint_index, 2] = 0.0
        frame.contact_probability[index] = 0.85
        frame.contact_valid[index] = 1
        frame.contact_provenance[index] = int(ComProvenance.KINEMATIC_INFERRED)
    frame.support_polygon[:4] = [
        [0.0, -0.2, 0.0],
        [0.3, -0.2, 0.0],
        [0.3, 0.2, 0.0],
        [0.0, 0.2, 0.0],
    ]
    frame.support_polygon_valid[:4] = 1
    return frame


def synthetic_episode(*, controller_only: bool) -> RawEpisode:
    count = 8
    states = np.zeros((count, 16), dtype=np.float32)
    states[:, 0] = np.arange(count) * 0.03
    states[:, 7] = 0.5 + np.arange(count) * 0.03
    states[:, 3:7] = [0, 0, 0, 1]
    states[:, 10:14] = [0, 0, 0, 1]
    states[:, 14:16] = [0.03, 0.04]
    signals = {
        "observation.tracking.left_tracked": np.ones(count, dtype=np.int64),
        "observation.tracking.right_tracked": np.ones(count, dtype=np.int64),
        "observation.valid": np.ones((count, 8), dtype=np.int64),
        "observation.tracking.hmd_pose": np.tile(
            np.array([0.25, 0.0, 1.65, 0, 0, 0, 1], dtype=np.float32),
            (count, 1),
        ),
    }
    body = None
    if not controller_only:
        observations = [_body_frame(index).observation() for index in range(count)]
        body = CanonicalBodyEpisode(
            {
                key: np.stack([observation[key] for observation in observations])
                for key in observations[0]
            }
        )
    camera = np.zeros((count, 24, 32, 3), dtype=np.uint8)
    for index in range(count):
        camera[index, 4:12, 2 + index : 10 + index, 1] = 255
    return RawEpisode(
        states=states,
        fps=20.0,
        signals=signals,
        body=body,
        images={"observation.images.left_wrist": camera},
        metadata={
            "handumi": {
                "controller_tcp_calibration": {
                    "sha256": "ux001-synthetic-identity",
                    "applied_to_state": False,
                    "controller_to_gripper_tcp": {
                        side: {
                            "position": [0.0, 0.0, -0.15],
                            "quaternion": [0.0, 0.0, 0.0, 1.0],
                        }
                        for side in ("left", "right")
                    },
                }
            }
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--controller-only", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    episode = synthetic_episode(controller_only=args.controller_only)
    stream = initialize_rerun(
        "handumi_ux001_golden",
        ["left_wrist"],
        fps=20,
        spawn=False,
        recorder_status=False,
        include_quality=episode.body is not None,
        save_path=args.output,
        timeline="episode_time",
        recording_id="3d96495b-1f72-4b5d-9ae4-91bca52dc001",
    )
    if stream is None:
        raise SystemExit("Rerun initialization failed")
    log_episode(
        stream.rr,
        episode,
        options=ViewerOptions(
            temporal_decimation=2,
            spatial_decimation_m=0.01,
            trail_point_cap=4,
            trail_duration_s=0.2,
        ),
    )
    stream.rr.disconnect()
    print(args.output)


if __name__ == "__main__":
    main()
