import io
import json

from handumi.tracking.meta_quest import parse_frame
from handumi.tracking.quest_probe import (
    PROBE_SCHEMA,
    AdbHealthSampler,
    ProbeCapture,
    analyze_probe_records,
)


def _record(seq: int, receive_ns: int, device_ns: int, *, offset: int = 10, rtt: int = 4):
    return {
        "probe_schema": PROBE_SCHEMA,
        "pc_receive_time_ns": receive_ns,
        "sync": {
            "clock_synced": True,
            "clock_offset_ns": offset,
            "rtt_ns": rtt,
        },
        "packet": {"seq": seq, "ovrTimeNs": device_ns},
    }


def test_capture_preserves_raw_packet_and_adds_host_diagnostics():
    stream = io.StringIO()
    raw = {"seq": 7, "body": {"xrTime": 123, "joints": [{"flags": 3}]}}
    frame = parse_frame(raw, pc_monotonic_ns=456)
    capture = ProbeCapture(
        stream=stream,
        metrics_provider=lambda: {"offset_ns": 20, "rtt_ns": 8},
    )

    capture.record(frame)

    envelope = json.loads(stream.getvalue())
    assert envelope["packet"] == raw
    assert envelope["pc_receive_time_ns"] == 456
    assert envelope["sync"] == {
        "clock_synced": True,
        "clock_offset_ns": 20,
        "rtt_ns": 8,
    }


def test_capture_stores_manifest_separately_without_altering_raw_envelope():
    stream = io.StringIO()
    manifests = io.StringIO()
    packet = {
        "packetType": "session_manifest",
        "sessionId": "session-1",
        "requestedJointSet": "FullBody",
    }
    capture = ProbeCapture(stream=stream, manifest_stream=manifests)

    capture.record_raw(packet, 123, 4)

    assert json.loads(stream.getvalue())["packet"] == packet
    assert json.loads(manifests.getvalue()) == packet
    assert capture.manifest_count == 1


def test_analysis_reports_sequence_loss_and_timing_rates():
    records = [
        _record(10, 1_000_000_000, 2_000_000_000),
        _record(11, 1_010_000_000, 2_010_000_000),
        _record(13, 1_020_000_000, 2_020_000_000),
    ]

    summary = analyze_probe_records(records)

    assert summary["packet_count"] == 3
    assert summary["receive_rate_hz"] == 100.0
    assert summary["device_rate_hz"] == 100.0
    assert summary["source_sequence"]["missing_packets"] == 1
    assert summary["source_sequence"]["loss_fraction"] == 0.25
    assert summary["receive_interarrival_ns"]["standard_deviation"] == 0.0
    assert summary["mapped_sample_age_ns"]["median"] == -1_000_000_010


def test_analysis_does_not_claim_loss_measurement_without_sender_sequence():
    summary = analyze_probe_records(
        [
            {
                "pc_receive_time_ns": 1,
                "sync": {},
                "packet": {"ovrTimeNs": 2},
            }
        ]
    )

    assert summary["source_sequence"]["available"] is False
    assert summary["source_sequence"]["missing_packets"] is None
    assert summary["source_sequence"]["loss_fraction"] is None


def test_analysis_excludes_offsets_until_clock_is_synced():
    summary = analyze_probe_records(
        [
            {
                "pc_receive_time_ns": 100,
                "sync": {
                    "clock_synced": False,
                    "clock_offset_ns": 0,
                    "rtt_ns": None,
                },
                "packet": {"ovrTimeNs": 50},
            }
        ]
    )

    assert summary["clock_offset_ns"]["sample_count"] == 0
    assert summary["mapped_sample_age_ns"]["sample_count"] == 0


def test_analysis_reports_body_sample_rate_when_diagnostic_field_is_present():
    records = [
        {"packet": {"body": {"sourceTimeNs": 1_000_000_000}}},
        {"packet": {"body": {"sourceTimeNs": 1_000_000_000}}},
        {"packet": {"body": {"sourceTimeNs": 1_020_000_000}}},
        {"packet": {"body": {"sourceTimeNs": 1_040_000_000}}},
    ]

    summary = analyze_probe_records(records)

    assert summary["body_update_rate_hz"] == 50.0


def test_analysis_reads_compact_body_joint_flags():
    summary = analyze_probe_records(
        [
            {
                "packet": {
                    "seq": 1,
                    "body": {
                        "active": True,
                        "activeJointSet": "FullBody",
                        "jointCount": 2,
                        "jointNames": ["Root", "Hips"],
                        "jointLocationFlags": [15, 3],
                        "jointPoses": [0, 0, 0, 0, 0, 0, 1] * 2,
                    },
                }
            }
        ]
    )

    assert summary["body"]["joints"][0]["name"] == "Root"
    assert summary["body"]["joints"][0]["pose_tracked_fraction"] == 1.0
    assert summary["body"]["joints"][1]["pose_valid_fraction"] == 1.0


def test_analysis_excludes_manifest_and_summarizes_body_flags_and_states():
    joints = [
        {"index": 0, "name": "Root", "locationFlags": 15},
        {"index": 1, "name": "Hips", "locationFlags": 3},
    ]
    records = [
        {
            "pc_receive_time_ns": 90,
            "packet": {"packetType": "session_manifest", "sessionId": "s1"},
        },
        {
            "capture_index": 1,
            "pc_receive_time_ns": 1_100,
            "sync": {
                "clock_synced": True,
                "clock_offset_ns": 100,
                "rtt_ns": 4,
            },
            "packet": {
                "packetType": "body_pose",
                "seq": 0,
                "ovrTimeNs": 1_000,
                "body": {
                    "active": True,
                    "activeJointSet": "FullBody",
                    "jointCount": 84,
                    "sourceTimeNs": 950,
                    "confidence": 0.8,
                    "calibrationState": "Valid",
                    "fidelity": "High",
                    "skeletonRevision": 7,
                    "joints": joints,
                },
            },
        },
        {
            "capture_index": 2,
            "pc_receive_time_ns": 1_200,
            "sync": {
                "clock_synced": True,
                "clock_offset_ns": 100,
                "rtt_ns": 4,
            },
            "packet": {
                "packetType": "body_pose",
                "seq": 1,
                "ovrTimeNs": 1_100,
                "body": {
                    "active": False,
                    "activeJointSet": "None",
                    "jointCount": 0,
                    "sourceTimeNs": 0,
                    "confidence": 0.0,
                    "calibrationState": "Invalid",
                    "fidelity": "Unknown",
                    "skeletonRevision": 0,
                    "joints": [],
                },
            },
        },
    ]

    summary = analyze_probe_records(records)

    assert summary["manifest_count"] == 1
    assert summary["packet_count"] == 2
    assert summary["body"]["active_pose_packets"] == 1
    assert summary["body"]["inactive_pose_packets"] == 1
    assert summary["body"]["joint_count_counts"] == {"0": 1, "84": 1}
    assert summary["body"]["calibration_state_counts"] == {
        "Invalid": 1,
        "Valid": 1,
    }
    assert len(summary["body"]["active_intervals"]) == 2
    root = summary["body"]["joints"][0]
    hips = summary["body"]["joints"][1]
    assert root["pose_valid_fraction"] == 1.0
    assert root["pose_tracked_fraction"] == 1.0
    assert hips["pose_valid_fraction"] == 1.0
    assert hips["pose_tracked_fraction"] == 0.0
    assert summary["mapped_body_sample_age_ns"]["median"] == 50.0


def test_analysis_distinguishes_sequence_duplicates_resets_and_reordering():
    records = [
        _record(10, 1, 1),
        _record(12, 2, 2),
        _record(12, 3, 3),
        _record(11, 4, 4),
        _record(0, 5, 5),
    ]

    sequence = analyze_probe_records(records)["source_sequence"]

    assert sequence["missing_packets"] == 1
    assert sequence["duplicates"] == 1
    assert sequence["out_of_order"] == 1
    assert sequence["resets"] == 1


def test_analysis_preserves_70_and_84_joint_set_transitions():
    records = []
    for sequence, count, joint_set in ((0, 70, "UpperBody"), (1, 84, "FullBody")):
        records.append(
            {
                "packet": {
                    "seq": sequence,
                    "body": {
                        "active": True,
                        "activeJointSet": joint_set,
                        "jointCount": count,
                        "sourceTimeNs": sequence + 1,
                        "joints": [
                            {"index": index, "locationFlags": 0}
                            for index in range(count)
                        ],
                    },
                }
            }
        )

    body = analyze_probe_records(records)["body"]

    assert body["joint_count_counts"] == {"70": 1, "84": 1}
    assert body["joint_set_counts"] == {"FullBody": 1, "UpperBody": 1}
    assert len(body["joints"]) == 84


def test_adb_health_sampler_records_tool_failure_instead_of_crashing(tmp_path):
    sampler = AdbHealthSampler(
        output_path=tmp_path / "health.jsonl",
        logcat_path=tmp_path / "logcat.txt",
        adb_path=str(tmp_path / "missing-adb"),
    )

    sample = sampler.sample()

    assert sample["record_type"] == "adb_health"
    assert sample["battery"]["returncode"] is None
    assert sample["thermal"]["stderr"]
