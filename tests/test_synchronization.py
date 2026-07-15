from dataclasses import replace

from handumi.feetech import GripperSample, GripperWidths
from handumi.synchronization import (
    SustainedHealthGate,
    synchronized_gripper_frame,
    tracking_sample_at,
)
from handumi.tracking.base import ControllerPairSample


def _widths(value_mm: float) -> GripperWidths:
    return GripperWidths(
        left=value_mm / 1000.0,
        right=value_mm / 1000.0,
        left_mm=value_mm,
        right_mm=value_mm,
        left_normalized=value_mm / 80.0,
        right_normalized=value_mm / 80.0,
        left_ticks=100,
        right_ticks=100,
    )


class _BufferedGrippers:
    def sample_at(self, target_time_ns):
        return GripperSample(_widths(20.0), target_time_ns + 2_000_000, 9)


class _BufferedTracker:
    device = "meta"

    def sample_at(self, target_time_ns):
        return replace(
            ControllerPairSample.empty("meta"), aligned_time_ns=target_time_ns
        )

    def latest(self):
        raise AssertionError("sample_at should be preferred")


def test_gripper_sample_is_selected_against_common_target():
    selected = synchronized_gripper_frame(
        _BufferedGrippers(),
        target_time_ns=1_000_000_000,
        record_time_ns=1_010_000_000,
        stale_timeout_s=0.1,
        max_sync_skew_s=0.01,
    )

    assert selected.healthy_for_gate
    assert selected.widths.left_mm == 20.0
    assert selected.frame["observation.feetech.sequence"].item() == 9
    assert selected.frame["observation.feetech.healthy"].item() == 1
    assert "observation.feetech.sync_error_ms" not in selected.frame


def test_disabled_feetech_does_not_block_health_gate():
    selected = synchronized_gripper_frame(
        None,
        target_time_ns=1_000_000_000,
        record_time_ns=1_000_000_000,
        stale_timeout_s=0.1,
        max_sync_skew_s=0.01,
    )

    assert selected.healthy_for_gate
    assert selected.frame["observation.feetech.healthy"].item() == 0
    assert "observation.feetech.enabled" not in selected.frame


def test_sustained_health_gate_tracks_recovery_and_timeout():
    gate = SustainedHealthGate(timeout_s=1.0)

    assert gate.update({"camera.left": False}, 0) == ([], [])
    recovered, timed_out = gate.update({"camera.left": True}, 500_000_000)
    assert recovered == ["camera.left"]
    assert timed_out == []

    gate.update({"camera.left": False}, 1_000_000_000)
    _, timed_out = gate.update({"camera.left": False}, 2_000_000_000)
    assert timed_out == ["camera.left"]


def test_tracking_prefers_native_buffer_lookup():
    target = 123_000_000
    sample = tracking_sample_at(_BufferedTracker(), target)
    assert sample.aligned_time_ns == target
