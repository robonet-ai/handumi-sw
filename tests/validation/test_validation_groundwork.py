import json
from pathlib import Path

import numpy as np
import pytest

from handumi.validation.core import (
    FrameTransform,
    SyncEvent,
    availability,
    bootstrap_participant_mean,
    classification_metrics,
    dropout_intervals,
    jitter,
    orientation_error_deg,
    position_errors,
    temporal_offset_s,
)
from handumi.validation.report import VALIDATION_STATUS, generate_report

FIXTURE = Path(__file__).parents[1] / "fixtures" / "validation" / "synthetic-known.json"


def test_known_transform_round_trip_composition_and_handedness():
    fixture = json.loads(FIXTURE.read_text())
    spec = fixture["transform"]
    transform = FrameTransform.from_pose7(spec["target"], spec["source"], spec["pose7"])
    points = np.asarray(fixture["points_mocap"])
    expected = np.asarray(fixture["points_handumi_world"])
    np.testing.assert_allclose(transform.apply(points), expected)
    identity = transform.compose(transform.inverse())
    np.testing.assert_allclose(identity.matrix, np.eye(4), atol=1e-9)
    reflected = np.eye(4)
    reflected[0, 0] = -1
    with pytest.raises(ValueError, match="left-handed"):
        FrameTransform("table", "mocap", reflected)


def test_sync_event_schema_requires_sequence_epoch_timestamps_and_uncertainty():
    event = SyncEvent(3, 1, 100, 110, 5, "led_rise", "gpio0")
    assert event.record()["schema"] == "handumi_sync_event_v1"
    with pytest.raises(ValueError):
        SyncEvent(-1, 0, 0, 0, 0, "led", "gpio0")


def test_known_position_orientation_availability_classification_and_jitter_metrics():
    reference = np.zeros((4, 3))
    estimate = reference + np.asarray([0.01, 0.0, 0.0])
    assert position_errors(estimate, reference)["rmse_m"] == pytest.approx(0.01)
    quaternions = np.tile([0.0, 0.0, 0.0, 1.0], (4, 1))
    np.testing.assert_allclose(orientation_error_deg(quaternions, quaternions), 0)
    assert availability(np.asarray([1, 0, 1]))["fraction"] == pytest.approx(2 / 3)
    assert classification_metrics([1, 0, 1], [1, 0, 0])["accuracy"] == pytest.approx(
        2 / 3
    )
    assert jitter(np.asarray([[0.0, 0.0], [0.02, 0.0]])) == pytest.approx(0.0070710678)


def test_known_offset_and_reason_specific_dropout_recovery():
    reference = np.asarray([0, 0, 1, 0, 0, 0, 0, 0], dtype=float)
    estimate = np.roll(reference, 2)
    assert temporal_offset_s(estimate, reference, 0.01) == pytest.approx(0.02)
    fixture = json.loads(FIXTURE.read_text())
    intervals = dropout_intervals(np.arange(6) * 0.1, fixture["dropout_reasons"])
    assert [(item.reason, item.duration_s, item.recovered) for item in intervals] == [
        ("transport_gap", pytest.approx(0.2), True),
        ("invalid_tracking", pytest.approx(0.1), True),
    ]


def test_participant_cluster_bootstrap_is_deterministic_and_guarded():
    values = {"synthetic-a": [1.0, 2.0], "synthetic-b": [3.0, 4.0]}
    first = bootstrap_participant_mean(values, seed=42, samples=1000)
    second = bootstrap_participant_mean(values, seed=42, samples=1000)
    assert first == second
    assert first["participant_count"] == 2
    with pytest.raises(ValueError, match="at least 2"):
        bootstrap_participant_mean({"one": [1.0]}, seed=1)


def test_reports_are_reproducible_and_explicitly_unvalidated(tmp_path: Path):
    paths = generate_report(
        tmp_path / "report",
        metrics={"position_rmse_m": 0.01, "missing_frames": 2},
        configuration={"seed": 42},
        inputs=[FIXTURE],
    )
    evidence = json.loads(paths["json"].read_text())
    assert evidence["validation_status"] == VALIDATION_STATUS
    assert "not scientifically validated" in paths["markdown"].read_text()
    assert paths["csv"].read_text().startswith("metric,value")
