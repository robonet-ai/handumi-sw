import socket
import threading
import time
import unittest

import numpy as np

from handumi.calibration.control_tcp import ControllerTcpCalibration
from handumi.robots.utils import IDENTITY_POSE7
from handumi.tracking import (
    MetaQuestConfig,
    MetaQuestReceiver,
    MetaQuestTrackingProvider,
    parse_frame,
)
from handumi.tracking import mock_quest_sender as mock


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _tracked_frame():
    return parse_frame(
        {
            "hmdPosition": {"x": 0, "y": 1.1, "z": 0},
            "hmdRotation": {"x": 0, "y": 0, "z": 0, "w": 1},
            "leftControllerPosition": {"x": -0.2, "y": 0.9, "z": 0.3},
            "leftControllerRotation": {"x": 0, "y": 0, "z": 0, "w": 1},
            "leftTracked": True,
            "leftValid": True,
            "rightControllerPosition": {"x": 0.2, "y": 0.9, "z": 0.3},
            "rightControllerRotation": {"x": 0, "y": 0, "z": 0, "w": 1},
            "rightTracked": True,
            "rightValid": True,
        },
        pc_monotonic_ns=time.monotonic_ns(),
    )


def _identity_calibration() -> ControllerTcpCalibration:
    return ControllerTcpCalibration(
        left=IDENTITY_POSE7.copy(),
        right=IDENTITY_POSE7.copy(),
        source=None,
    )


class ParseFrameTest(unittest.TestCase):
    """Parsing of the YubiQuestApp flat wire format (dict-shaped vectors)."""

    def test_full_frame(self):
        frame = parse_frame(
            {
                "ovrTimeNs": 123456789,
                "deltaTime": 0.0139,
                "hmdPosition": {"x": 0, "y": 1.1, "z": 0},
                "hmdRotation": {"x": 0, "y": 0, "z": 0, "w": 1},
                "leftControllerPosition": {"x": -0.2, "y": 0.95, "z": 0.3},
                "leftControllerRotation": {"x": 0.1, "y": 0.2, "z": 0.3, "w": 0.9},
                "leftTracked": True,
                "leftValid": True,
                "leftJoystick": {"x": 0.1, "y": -0.2},
                "leftThumbstickClick": True,
                "leftTriggerPressed": True,
                "buttonXPressed": True,
                "buttonYPressed": False,
                "rightTracked": False,
                "rightValid": False,
            },
            pc_monotonic_ns=999,
        )
        self.assertEqual(frame.device_time_ns, 123456789)
        self.assertEqual(frame.pc_monotonic_ns, 999)
        self.assertAlmostEqual(frame.delta_time_s, 0.0139, places=5)
        self.assertTrue(frame.hmd.tracked)
        self.assertTrue(frame.left.tracked)
        self.assertTrue(frame.left.valid)
        self.assertTrue(np.allclose(frame.left.position, [-0.2, 0.95, 0.3], atol=1e-6))
        # Analog trigger is reported as 1.0 from the pressed flag (no analog wire value).
        self.assertEqual(frame.left.buttons.trigger, 1.0)
        self.assertTrue(frame.left.buttons.primary)  # buttonXPressed
        self.assertEqual(frame.left.buttons.thumbstick, (0.1, -0.2))
        self.assertTrue(frame.left.buttons.thumbstick_click)
        self.assertFalse(frame.right.tracked)

    def test_missing_fields_get_safe_defaults(self):
        frame = parse_frame({}, pc_monotonic_ns=1)
        self.assertEqual(frame.seq, 0)
        self.assertEqual(frame.device_time_ns, 0)
        self.assertFalse(frame.hmd.tracked)
        self.assertFalse(frame.left.tracked)
        self.assertFalse(frame.left.valid)
        self.assertEqual(frame.left.position.tolist(), [0.0, 0.0, 0.0])
        # Quaternion defaults to identity, not zeros.
        self.assertEqual(frame.right.quaternion.tolist(), [0.0, 0.0, 0.0, 1.0])
        self.assertNotEqual(frame.delta_time_s, frame.delta_time_s)  # NaN

    def test_quaternion_xyzw_order_preserved(self):
        frame = parse_frame(
            {"leftControllerRotation": {"x": 0.11, "y": 0.22, "z": 0.33, "w": 0.44}},
            pc_monotonic_ns=0,
        )
        self.assertTrue(
            np.allclose(frame.left.quaternion, [0.11, 0.22, 0.33, 0.44], atol=1e-6)
        )


class TrackingFreshnessTest(unittest.TestCase):
    def setUp(self):
        self.config = MetaQuestConfig(
            quest_ip="127.0.0.1",
            frame_stale_timeout_s=0.05,
        )

    def test_receiver_marks_old_frame_as_not_streaming(self):
        receiver = MetaQuestReceiver(self.config)
        with receiver._lock:
            receiver._connected = True
            receiver._latest = _tracked_frame()
            receiver._last_frame_mono = time.monotonic() - 0.10

        metrics = receiver.metrics()

        self.assertTrue(metrics["connected"])
        self.assertFalse(metrics["streaming"])
        self.assertGreater(metrics["last_frame_age_s"], 0.05)

    def test_provider_does_not_expose_cached_stale_pose_as_tracked(self):
        provider = MetaQuestTrackingProvider(
            config=self.config,
            calibration=_identity_calibration(),
        )
        with provider.receiver._lock:
            provider.receiver._connected = True
            provider.receiver._latest = _tracked_frame()
            provider.receiver._last_frame_mono = time.monotonic() - 0.10

        sample = provider.latest()

        self.assertFalse(sample.left_tracked)
        self.assertFalse(sample.right_tracked)

    def test_provider_does_not_expose_cached_pose_after_disconnect(self):
        provider = MetaQuestTrackingProvider(
            config=self.config,
            calibration=_identity_calibration(),
        )
        with provider.receiver._lock:
            provider.receiver._connected = False
            provider.receiver._latest = _tracked_frame()
            provider.receiver._last_frame_mono = time.monotonic()

        sample = provider.latest()

        self.assertFalse(sample.left_tracked)
        self.assertFalse(sample.right_tracked)

    def test_provider_exposes_fresh_valid_pose_as_tracked(self):
        provider = MetaQuestTrackingProvider(
            config=self.config,
            calibration=_identity_calibration(),
        )
        with provider.receiver._lock:
            provider.receiver._connected = True
            provider.receiver._latest = _tracked_frame()
            provider.receiver._last_frame_mono = time.monotonic()

        sample = provider.latest()

        self.assertTrue(sample.left_tracked)
        self.assertTrue(sample.right_tracked)
        self.assertTrue(sample.left_device_tracked)
        self.assertTrue(sample.left_pose_valid)
        self.assertTrue(sample.hmd_tracked)
        self.assertEqual(sample.hmd_pose.shape, (7,))

    def test_receiver_selects_native_frame_nearest_aligned_pc_time(self):
        receiver = MetaQuestReceiver(self.config)
        first = parse_frame(
            {"ovrTimeNs": 1_000, "leftTracked": True}, pc_monotonic_ns=10_000
        )
        second = parse_frame(
            {"ovrTimeNs": 2_000, "leftTracked": True}, pc_monotonic_ns=20_000
        )
        with receiver._lock:
            receiver._frames.extend((first, second))
            receiver._offset_ns = 100_000
            receiver._rtt_ns = 1_000

        selected = receiver.aligned_at(102_100)

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.frame.device_time_ns, 2_000)
        self.assertEqual(selected.aligned_time_ns, 102_000)
        self.assertTrue(selected.clock_synced)


class PipeSmokeTest(unittest.TestCase):
    """End-to-end: mock Quest (TCP + UDP) -> receiver, on the loopback."""

    def test_receiver_streams_and_syncs(self):
        tcp_port = _free_port()
        sync_port = _free_port()
        skew_ns = int(5e9)
        stop = threading.Event()

        udp = threading.Thread(
            target=mock._udp_sync_server,
            args=("127.0.0.1", sync_port, skew_ns, stop),
            daemon=True,
        )
        udp.start()

        def tcp_server():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", tcp_port))
            s.listen(1)
            s.settimeout(0.5)
            while not stop.is_set():
                try:
                    conn, addr = s.accept()
                except socket.timeout:
                    continue
                mock._serve_client(conn, addr, 120.0, skew_ns, stop)
            s.close()

        threading.Thread(target=tcp_server, daemon=True).start()

        cfg = MetaQuestConfig(
            quest_ip="127.0.0.1", tcp_port=tcp_port, sync_port=sync_port
        )
        rx = MetaQuestReceiver(cfg)
        rx.start()
        try:
            deadline = time.monotonic() + 6.0
            while time.monotonic() < deadline:
                m = rx.metrics()
                frame = rx.latest()
                # Wait for a measurable fps (>=2 frames in the window) and a
                # completed UDP sync, not just the first frame.
                if frame is not None and m["fps"] > 10.0 and rx._rtt_ns is not None:
                    break
                time.sleep(0.05)

            m = rx.metrics()
            frame = rx.latest()
            self.assertIsNotNone(frame, "no frame received from mock")
            assert frame is not None
            self.assertTrue(m["connected"])
            self.assertTrue(m["streaming"])
            self.assertGreater(m["fps"], 10.0)
            self.assertTrue(frame.left.tracked)
            self.assertGreater(frame.device_time_ns, 0)
            self.assertGreater(frame.pc_monotonic_ns, 0)
            # offset = pc - device ~= -skew (device clock runs +skew ahead).
            self.assertAlmostEqual(m["offset_s"], -5.0, delta=0.2)

            # Poses move over time.
            frame = rx.latest()
            assert frame is not None
            p1 = frame.left.position.copy()
            time.sleep(0.3)
            frame = rx.latest()
            assert frame is not None
            p2 = frame.left.position
            self.assertTrue((abs(p1 - p2) > 1e-4).any(), "controller pose did not move")
        finally:
            rx.stop()
            stop.set()


if __name__ == "__main__":
    unittest.main()
