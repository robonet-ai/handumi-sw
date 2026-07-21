import io
import json
import socket
import time
import unittest

import numpy as np

from handumi.robots.utils import IDENTITY_POSE7
from handumi.tracking import mock_quest_sender as mock
from handumi.tracking.base import (
    ControllerPairSample,
    LegacyControllerProviderAdapter,
)
from handumi.tracking.meta_quest import (
    MetaQuestConfig,
    MetaQuestReceiver,
    parse_tracking_packet,
)
from handumi.tracking.packet import (
    JointTrackingState,
    PacketLossReason,
    SourceProvenance,
    TimestampQuality,
    TrackingPacketStream,
    drain_tracking_packets_jsonl,
)
from handumi.tracking.pico import tracking_packet_from_pico_frame


class TrackingPacketContractTest(unittest.TestCase):
    def test_compact_84_joint_meta_packet_preserves_flags_and_unknown_fields(self):
        raw = mock.make_tracking_packet_fixture(84, seq=17)
        raw["body"]["jointPoses"][0] = float("nan")

        packet = parse_tracking_packet(
            raw,
            pc_monotonic_ns=10_000,
            receive_sequence=5,
            clock_offset_ns=1_000,
            rtt_ns=6_000,
        )

        self.assertEqual(packet.source_schema_version, 2)
        self.assertEqual(packet.sequence, 17)
        self.assertEqual(packet.receive_sequence, 5)
        self.assertEqual(packet.timestamps.quality, TimestampQuality.DIAGNOSTIC_ONLY)
        self.assertEqual(packet.timestamps.uncertainty_ns, 3_000)
        self.assertIs(packet.raw, raw)
        self.assertTrue(packet.raw["unknownFixtureField"]["preserved"])
        self.assertIsNotNone(packet.body)
        assert packet.body is not None
        self.assertEqual(packet.body.joint_count, 84)
        self.assertEqual(len(packet.body.joints), 84)
        self.assertEqual(packet.body.joints[0].location_flags, 15)
        self.assertEqual(
            packet.body.joints[0].tracking_state, JointTrackingState.TRACKED
        )
        self.assertEqual(packet.body.joints[1].location_flags, 3)
        self.assertEqual(packet.body.joints[1].tracking_state, JointTrackingState.VALID)
        self.assertTrue(np.isnan(packet.body.joints[0].pose[0]))
        self.assertEqual(packet.body.provenance, SourceProvenance.PLATFORM_ESTIMATED)

    def test_verbose_70_joint_and_inactive_packets_are_supported(self):
        verbose = mock.make_tracking_packet_fixture(70, seq=1)
        body = verbose["body"]
        body["joints"] = [
            {
                "index": index,
                "name": f"Verbose_{index}",
                "locationFlags": 3,
                "position": {"x": 0, "y": 1, "z": 0},
                "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
            }
            for index in range(70)
        ]
        body.pop("jointPoses")
        body.pop("jointLocationFlags")
        packet = parse_tracking_packet(verbose, pc_monotonic_ns=1, receive_sequence=1)
        assert packet.body is not None
        self.assertEqual(len(packet.body.joints), 70)

        inactive = mock.make_tracking_packet_fixture(84, seq=2, active=False)
        packet = parse_tracking_packet(inactive, pc_monotonic_ns=2, receive_sequence=2)
        assert packet.body is not None
        self.assertFalse(packet.body.active)
        self.assertEqual(packet.body.joint_count, 0)
        self.assertEqual(packet.body.joints, ())

    def test_legacy_packet_normalizes_as_source_version_one(self):
        raw = mock._make_frame(0, time.monotonic(), 0)
        packet = parse_tracking_packet(raw, pc_monotonic_ns=1, receive_sequence=3)
        self.assertEqual(packet.source_schema_version, 1)
        self.assertIsNone(packet.sequence)
        self.assertIsNone(packet.body)
        self.assertEqual(packet.timestamps.quality, TimestampQuality.MAPPED_UNBOUNDED)


class PacketStreamTest(unittest.TestCase):
    def test_bounded_stream_explicitly_counts_overflow(self):
        stream = TrackingPacketStream(max_packets=3)
        for seq in range(10_000):
            stream.publish(
                parse_tracking_packet(
                    mock.make_tracking_packet_fixture(70, seq=seq),
                    pc_monotonic_ns=seq + 1,
                    receive_sequence=seq + 1,
                )
            )
        stats = stream.stats()
        self.assertEqual(stats.accepted, 10_000)
        self.assertEqual(stats.queued, 3)
        self.assertEqual(stats.dropped[PacketLossReason.QUEUE_OVERFLOW.value], 9_997)
        self.assertEqual(
            [packet.sequence for packet in stream.drain()], [9_997, 9_998, 9_999]
        )

    def test_jsonl_writer_drains_fifo_once_and_preserves_unknown_fields(self):
        stream = TrackingPacketStream(max_packets=4)
        for seq in (7, 8):
            raw = mock.make_tracking_packet_fixture(84, seq=seq)
            raw["futureField"] = {"kept": seq}
            stream.publish(
                parse_tracking_packet(
                    raw, pc_monotonic_ns=seq, receive_sequence=seq + 100
                )
            )

        output = io.StringIO()
        self.assertEqual(drain_tracking_packets_jsonl(stream, output), 2)
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([record["sequence"] for record in records], [7, 8])
        self.assertEqual(records[0]["packet"]["futureField"], {"kept": 7})
        self.assertEqual(stream.stats().queued, 0)
        self.assertEqual(drain_tracking_packets_jsonl(stream, output), 0)

    def test_receiver_reports_version_sequence_and_queue_diagnostics(self):
        receiver = MetaQuestReceiver(
            MetaQuestConfig(quest_ip="127.0.0.1", packet_queue_size=2)
        )
        receiver._handle_message(mock.make_tracking_packet_fixture(84, seq=1))
        receiver._handle_message(mock.make_tracking_packet_fixture(84, seq=3))
        receiver._handle_message(mock.make_tracking_packet_fixture(84, seq=3))
        receiver._handle_message(mock.make_tracking_packet_fixture(84, seq=2))
        receiver._handle_message(
            {"schema": "tracking_packet_v99", "sourceSchemaVersion": 99}
        )

        stats = receiver.packet_stream_stats()
        self.assertEqual(stats["accepted"], 4)
        self.assertEqual(stats["queued"], 2)
        self.assertEqual(stats["dropped"][PacketLossReason.QUEUE_OVERFLOW.value], 2)
        self.assertEqual(stats["diagnostics"][PacketLossReason.SEQUENCE_GAP.value], 1)
        self.assertEqual(stats["diagnostics"][PacketLossReason.DUPLICATE.value], 1)
        self.assertEqual(stats["diagnostics"][PacketLossReason.OUT_OF_ORDER.value], 1)
        self.assertEqual(
            stats["diagnostics"][PacketLossReason.UNSUPPORTED_VERSION.value], 1
        )

    def test_receiver_survives_malformed_json_and_nonnumeric_version(self):
        receiver = MetaQuestReceiver(MetaQuestConfig(quest_ip="127.0.0.1"))
        local, remote = socket.socketpair()
        try:
            receiver._running = True
            remote.sendall(b"{not-json}\n")
            remote.shutdown(socket.SHUT_WR)
            receiver._tcp_recv_loop(local)
        finally:
            receiver._running = False
            local.close()
            remote.close()

        receiver._handle_message(
            {"schema": "tracking_packet_v2", "sourceSchemaVersion": "bad"}
        )
        diagnostics = receiver.packet_stream_stats()["diagnostics"]
        self.assertEqual(diagnostics[PacketLossReason.MALFORMED_FRAME.value], 1)
        self.assertEqual(diagnostics[PacketLossReason.UNSUPPORTED_VERSION.value], 1)


class PicoNormalizationTest(unittest.TestCase):
    def test_pico_body_hands_and_trackers_share_common_packet(self):
        frame = {
            "observation.pico.timestamp_ns": np.array([123], dtype=np.int64),
            # An identity pose at the source origin is valid; all-zero arrays
            # are the PICO unavailable sentinel.
            "observation.pico.headset_pose": np.array([0, 0, 0, 0, 0, 0, 1]),
            "observation.pico.left_controller_pose": np.array([-1, 1, 0, 0, 0, 0, 1]),
            "observation.pico.right_controller_pose": np.array([1, 1, 0, 0, 0, 0, 1]),
            "observation.pico.body_joints_pose": np.tile(
                np.array([[0, 1, 0, 0, 0, 0, 1]], dtype=np.float32), (24, 1)
            ),
            "observation.pico.left_hand_pose": np.tile(
                np.array([[0, 1, 0, 0, 0, 0, 1]], dtype=np.float32), (27, 1)
            ),
            "observation.pico.right_hand_pose": np.zeros((27, 7), dtype=np.float32),
            "observation.pico.motion_tracker_pose": np.array(
                [[0.5, 1, 0, 0, 0, 0, 1]], dtype=np.float32
            ),
            "observation.pico.motion_tracker_velocity": np.ones(
                (1, 6), dtype=np.float32
            ),
            "observation.pico.motion_tracker_accel": np.ones((1, 6), dtype=np.float32),
            "observation.pico.motion_tracker_count": np.array([1], dtype=np.int64),
            "observation.pico.motion_tracker_serial_hash": np.array(
                [99], dtype=np.int64
            ),
        }
        packet = tracking_packet_from_pico_frame(frame, sequence=7, receive_time_ns=456)
        assert packet.body is not None
        self.assertEqual(packet.body.joint_count, 24)
        assert packet.hmd is not None
        self.assertEqual(packet.hmd.tracking_state, JointTrackingState.TRACKED)
        self.assertEqual(len(packet.hands), 2)
        self.assertTrue(packet.hands[0].active)
        self.assertFalse(packet.hands[1].active)
        self.assertEqual(len(packet.external_trackers), 1)
        self.assertEqual(packet.external_trackers[0].tracker_id, "99")
        self.assertEqual(packet.timestamps.quality, TimestampQuality.RECEIVE_ONLY)


class _LegacyProvider:
    device = "test"

    def start(self):
        pass

    def stop(self):
        pass

    def latest(self):
        return ControllerPairSample.empty(self.device)


class LegacyAdapterTest(unittest.TestCase):
    def test_adapter_keeps_latest_semantics(self):
        adapter = LegacyControllerProviderAdapter(_LegacyProvider())
        sample = adapter.latest()
        self.assertEqual(sample.device, "test")
        self.assertEqual(sample.left_controller_pose.tolist(), IDENTITY_POSE7.tolist())


if __name__ == "__main__":
    unittest.main()
