import os
import socket
import tempfile
import threading
import time
import unittest

import numpy as np

from handumi.scripts.record_handumi_quest import (
    WorkspaceState,
    build_features,
    build_observation,
    record_episode,
)
from handumi.dataset.raw import HANDUMI_RAW_STATE_SIZE
from handumi.feetech import GripperWidths
from handumi.tracking.meta_quest import parse_frame
from handumi.tracking.transforms import (
    MountingOffsets,
    WorkspaceCalibration,
    unity_pose_to_handumi,
)


def _widths() -> GripperWidths:
    return GripperWidths(
        left=0.011, right=0.022, left_mm=11.0, right_mm=22.0,
        left_normalized=0.13, right_normalized=0.26, left_ticks=1500, right_ticks=1600,
    )


def _frame(primary_left=False, hmd_tracked=True):
    msg = {
        "ovrTimeNs": 12345,
        "leftControllerPosition": {"x": -0.2, "y": 0.95, "z": 0.3},
        "leftControllerRotation": {"x": 0, "y": 0, "z": 0, "w": 1},
        "leftTracked": True, "leftValid": True, "buttonXPressed": primary_left,
        "rightControllerPosition": {"x": 0.2, "y": 0.95, "z": 0.3},
        "rightControllerRotation": {"x": 0, "y": 0, "z": 0, "w": 1},
        "rightTracked": True, "rightValid": True,
    }
    if hmd_tracked:
        msg["hmdPosition"] = {"x": 0.0, "y": 1.1, "z": 0.2}
        msg["hmdRotation"] = {"x": 0, "y": 0, "z": 0, "w": 1}
    return parse_frame(msg, pc_monotonic_ns=999)


class FeaturesTest(unittest.TestCase):
    def test_schema(self):
        f = build_features(["left_wrist", "right_wrist"], 640, 480, use_videos=True)
        self.assertIn("observation.images.left_wrist", f)
        # Shape is a tuple so it matches numpy value.shape in LeRobot validation.
        self.assertEqual(f["observation.state"]["shape"], (HANDUMI_RAW_STATE_SIZE,))
        self.assertIn("observation.feetech.left_ticks", f)
        self.assertEqual(f["observation.quest.left_controller_pose"]["shape"], (7,))
        self.assertEqual(f["observation.quest.device_time_ns"]["dtype"], "int64")
        self.assertIn("observation.quest.pc_monotonic_ns", f)
        self.assertIn("observation.quest.seq", f)


class BuildObservationTest(unittest.TestCase):
    def test_tracked_frame_identity_calibration(self):
        obs = build_observation(
            _frame(),
            mounts=MountingOffsets.identity(),
            workspace=WorkspaceCalibration.identity(),
            widths=_widths(),
        )
        state = obs["observation.state"]
        self.assertEqual(state.shape, (HANDUMI_RAW_STATE_SIZE,))
        # Identity calibration => state poses equal the Unity-converted poses.
        left = unity_pose_to_handumi([-0.2, 0.95, 0.3], [0, 0, 0, 1])
        self.assertTrue(np.allclose(state[0:3], left.position, atol=1e-6))
        self.assertTrue(np.allclose(state[14], 0.011, atol=1e-6))
        self.assertTrue(np.allclose(state[15], 0.022, atol=1e-6))
        self.assertEqual(int(obs["observation.quest.left_tracked"][0]), 1)
        self.assertEqual(int(obs["observation.quest.device_time_ns"][0]), 12345)
        self.assertEqual(int(obs["observation.quest.pc_monotonic_ns"][0]), 999)
        # yubi legacy TCP/JSON carries no seq, so it defaults to 0.
        self.assertEqual(int(obs["observation.quest.seq"][0]), 0)
        self.assertEqual(obs["observation.quest.left_controller_pose"].shape, (7,))

    def test_missing_frame_is_zero_filled(self):
        obs = build_observation(
            None,
            mounts=MountingOffsets.identity(),
            workspace=WorkspaceCalibration.identity(),
            widths=_widths(),
        )
        self.assertEqual(int(obs["observation.quest.left_tracked"][0]), 0)
        self.assertEqual(int(obs["observation.quest.device_time_ns"][0]), 0)
        # Identity pose => zero position, identity quaternion.
        pose = obs["observation.quest.left_controller_pose"]
        self.assertTrue(np.allclose(pose, [0, 0, 0, 0, 0, 0, 1], atol=1e-6))
        # Feetech width still recorded from the encoder.
        self.assertTrue(np.allclose(obs["observation.state"][14], 0.011, atol=1e-6))


class WorkspaceStateTest(unittest.TestCase):
    def test_auto_init_on_first_tracked_frame(self):
        ws = WorkspaceState()
        self.assertFalse(ws.is_set)
        frame = _frame()
        ws.update(frame)
        self.assertTrue(ws.is_set)
        # The HMD reference maps to the workspace origin.
        ref = unity_pose_to_handumi(frame.hmd.position, frame.hmd.quaternion)
        self.assertTrue(np.allclose(ws.calibration.apply(ref).as_matrix(), np.eye(4), atol=1e-6))

    def test_no_init_without_hmd(self):
        ws = WorkspaceState()
        ws.update(_frame(hmd_tracked=False))
        self.assertFalse(ws.is_set)

    def test_reset_on_left_x_edge(self):
        ws = WorkspaceState()
        ws.update(_frame())  # auto-init
        first = ws.calibration.workspace_from_quest.position.copy()
        # Rising edge of left X with the HMD elsewhere -> new reset.
        moved = parse_frame(
            {
                "hmdPosition": {"x": 0.5, "y": 1.3, "z": 0.4},
                "hmdRotation": {"x": 0, "y": 0, "z": 0, "w": 1},
                "leftControllerPosition": {"x": 0, "y": 0, "z": 0},
                "leftControllerRotation": {"x": 0, "y": 0, "z": 0, "w": 1},
                "leftTracked": True, "leftValid": True, "buttonXPressed": True,
                "rightTracked": True, "rightValid": True,
            },
            pc_monotonic_ns=1,
        )
        ws.update(moved)
        second = ws.calibration.workspace_from_quest.position
        self.assertFalse(np.allclose(first, second))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class RecordEpisodeIntegrationTest(unittest.TestCase):
    """Record a real LeRobot episode from the mock Quest (no cameras/video)."""

    def test_records_16d_state_and_quest_features(self):
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            self.skipTest("lerobot not installed")

        from handumi.tracking import mock_quest_sender as mock
        from handumi.tracking.meta_quest import MetaQuestConfig, MetaQuestReceiver
        from handumi.tracking.transforms import MountingOffsets

        tcp_port, sync_port = _free_port(), _free_port()
        stop = threading.Event()
        threading.Thread(
            target=mock._udp_sync_server,
            args=("127.0.0.1", sync_port, int(5e9), stop), daemon=True,
        ).start()

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
                mock._serve_client(conn, addr, 120.0, int(5e9), stop)
            s.close()

        threading.Thread(target=tcp_server, daemon=True).start()

        rx = MetaQuestReceiver(MetaQuestConfig("127.0.0.1", tcp_port, sync_port))
        rx.start()
        base = tempfile.mkdtemp()
        try:
            time.sleep(0.3)
            features = build_features([], 640, 480, use_videos=False)
            dataset = LeRobotDataset.create(
                repo_id="local/quest_test",
                fps=30,
                root=os.path.join(base, "ds"),
                robot_type="handumi_raw",
                features=features,
                use_videos=False,
                image_writer_processes=0,
                image_writer_threads=1,
            )
            workspace = WorkspaceState()
            n_frames, status = record_episode(
                dataset=dataset, cameras=[], cam_names=[], receiver=rx,
                mounts=MountingOffsets.identity(), workspace=workspace, grippers=None,
                episode_time_s=0.4, fps=30, task="test", cam_width=640, cam_height=480,
                button_control=False, stop_event=threading.Event(),
            )
            self.assertGreater(n_frames, 0)
            self.assertEqual(status, "recorded")
            self.assertTrue(workspace.is_set)
            dataset.save_episode()
            dataset.finalize()
            self.assertEqual(dataset.num_episodes, 1)
            self.assertEqual(dataset.num_frames, n_frames)
            self.assertIn("observation.quest.seq", dataset.features)
            self.assertIn("observation.state", dataset.features)
        finally:
            rx.stop()
            stop.set()
            import shutil

            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
