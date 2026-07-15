import numpy as np
import pytest

from handumi.dataset.raw import (
    HANDUMI_CAPTURE_SCHEMA,
    HANDUMI_STATE_SEMANTICS,
    HANDUMI_TRACKING_SCHEMA,
    TRACKING_VALIDITY_NAMES,
)
from handumi.dataset.reader import (
    _compose_pose7,
    normalize_raw_signals,
    validate_raw_state_metadata,
)


def _states(frame_count: int = 2) -> np.ndarray:
    states = np.zeros((frame_count, 16), dtype=np.float32)
    states[:, 3:7] = [0.0, 0.0, 0.0, 1.0]
    states[:, 10:14] = [0.0, 0.0, 0.0, 1.0]
    states[:, 0] = np.arange(frame_count, dtype=np.float32)
    states[:, 7] = -np.arange(frame_count, dtype=np.float32)
    return states


def test_pose7_composition_rotates_local_translation():
    half_turn = np.sqrt(0.5)
    workspace_from_device = np.array(
        [1.0, 0.0, 0.0, 0.0, 0.0, half_turn, half_turn]
    )
    device_from_hmd = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    composed = _compose_pose7(workspace_from_device, device_from_hmd)

    np.testing.assert_allclose(composed[:3], [1.0, 1.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(
        np.abs(composed[3:7]), [0.0, 0.0, half_turn, half_turn], atol=1e-7
    )


def test_compact_signals_restore_metadata_and_derived_timing():
    identity = np.tile(
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        (2, 1),
    )
    target = np.array([1_000_000_000, 2_000_000_000], dtype=np.int64)
    record = np.array([1_010_000_000, 2_010_000_000], dtype=np.int64)
    validity = np.ones((2, len(TRACKING_VALIDITY_NAMES)), dtype=np.int64)
    validity[1, TRACKING_VALIDITY_NAMES.index("streaming")] = 0
    signals = {
        "observation.valid": validity,
        "observation.tracking.left_tracked": np.ones(2, dtype=np.int64),
        "observation.tracking.right_tracked": np.ones(2, dtype=np.int64),
        "observation.tracking.workspace_from_device_pose": identity.copy(),
        "observation.tracking.device_hmd_pose": identity.copy(),
        "observation.tracking.device_time_ns": np.array(
            [900_000_000, 1_900_000_000], dtype=np.int64
        ),
        "observation.tracking.pc_monotonic_ns": np.array(
            [1_002_000_000, 2_003_000_000], dtype=np.int64
        ),
        "observation.tracking.aligned_time_ns": np.array(
            [1_002_000_000, 2_003_000_000], dtype=np.int64
        ),
        "observation.feetech.sample_time_ns": np.array(
            [1_001_000_000, 2_004_000_000], dtype=np.int64
        ),
        "observation.feetech.healthy": np.zeros(2, dtype=np.int64),
        "observation.camera.left_wrist.sample_time_ns": np.array(
            [996_000_000, 2_005_000_000], dtype=np.int64
        ),
        "observation.camera.left_wrist.healthy": np.ones(2, dtype=np.int64),
        "observation.sync.target_time_ns": target,
        "observation.sync.record_time_ns": record,
    }
    metadata = {
        "tracking_schema": HANDUMI_TRACKING_SCHEMA,
        "capture_schema": HANDUMI_CAPTURE_SCHEMA,
        "state_semantics": HANDUMI_STATE_SEMANTICS,
        "sources": {
            "tracking": {"enabled": True},
            "feetech": {"enabled": False},
            "cameras": {"left_wrist": {"enabled": True}},
        },
    }

    validate_raw_state_metadata({"handumi": metadata})
    normalized = normalize_raw_signals(_states(), signals, metadata=metadata)

    np.testing.assert_array_equal(normalized["observation.valid"], validity)
    np.testing.assert_allclose(
        normalized["observation.tracking.hmd_pose"], identity
    )
    assert "observation.tracking.streaming" not in normalized
    assert "observation.tracking.left_controller_pose" not in normalized
    np.testing.assert_array_equal(normalized["observation.feetech.enabled"], [0, 0])
    np.testing.assert_array_equal(
        normalized["observation.camera.left_wrist.enabled"], [1, 1]
    )
    np.testing.assert_allclose(
        normalized["observation.tracking.sync_error_ms"], [2.0, 3.0]
    )
    np.testing.assert_allclose(
        normalized["observation.camera.left_wrist.sync_error_ms"], [4.0, 5.0]
    )
    np.testing.assert_array_equal(
        normalized["observation.tracking.clock_offset_ns"],
        [102_000_000, 103_000_000],
    )


def test_rejects_previous_handumi_tracking_layout():
    info = {
        "handumi": {
            "tracking_schema": "controller_raw_and_workspace_v3",
            "capture_schema": "synchronized_sources_v1",
            "state_semantics": HANDUMI_STATE_SEMANTICS,
        }
    }

    with pytest.raises(ValueError, match="Re-record"):
        validate_raw_state_metadata(info)
