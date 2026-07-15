import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from handumi.scripts.setup.calibrate_tcp_offset import solve_pivot
from handumi.calibration.control_tcp import controller_tcp_calibration_metadata
from handumi.tracking.transforms import quat_to_matrix


def _rand_rotations(rng, n):
    quats = rng.standard_normal((n, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    return np.asarray([quat_to_matrix(q) for q in quats])


class SolvePivotTest(unittest.TestCase):
    def test_recovers_known_offset(self):
        rng = np.random.default_rng(7)
        t_true = np.array([0.14, -0.02, 0.05])
        c_true = np.array([0.3, 0.1, -0.4])
        R = _rand_rotations(rng, 200)
        # p_i = c - R_i @ t  (tip pinned at c)
        P = c_true - np.einsum("nij,j->ni", R, t_true)
        t, c, rms = solve_pivot(P, R)
        self.assertTrue(np.allclose(t, t_true, atol=1e-6))
        self.assertTrue(np.allclose(c, c_true, atol=1e-6))
        self.assertLess(rms, 1e-6)

    def test_noise_reflected_in_rms(self):
        rng = np.random.default_rng(8)
        t_true = np.array([0.14, 0.0, 0.0])
        R = _rand_rotations(rng, 300)
        P = -np.einsum("nij,j->ni", R, t_true) + rng.normal(0, 0.002, (300, 3))
        t, _, rms = solve_pivot(P, R)
        self.assertTrue(np.allclose(t, t_true, atol=0.005))
        self.assertGreater(rms, 0.0005)
        self.assertLess(rms, 0.01)


class CalibrationMetadataTest(unittest.TestCase):
    def test_snapshot_contains_exact_offsets_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta_controller_tcp.yaml"
            path.write_text(
                """calibration:
  controller_to_gripper_tcp:
    left:
      position: [0.1, 0.2, 0.3]
      quaternion: [0.0, 0.0, 0.0, 1.0]
    right:
      position: [-0.1, 0.2, 0.3]
      quaternion: [0.0, 0.0, 0.0, 1.0]
"""
            )
            metadata = controller_tcp_calibration_metadata(
                path,
                applied_to_state=False,
                source_robot="piper",
                source_gripper="piper_parallel_v1",
                tracking_device="meta",
                controller_mount="handumi_v1",
            )

        self.assertEqual(metadata["schema_version"], 2)
        self.assertFalse(metadata["applied_to_state"])
        self.assertEqual(metadata["source_robot"], "piper")
        self.assertEqual(metadata["source_gripper"], "piper_parallel_v1")
        self.assertEqual(metadata["tracking_device"], "meta")
        self.assertEqual(metadata["controller_mount"], "handumi_v1")
        self.assertEqual(len(metadata["sha256"]), 64)
        self.assertEqual(
            metadata["controller_to_gripper_tcp"]["left"]["position"],
            [0.10000000149011612, 0.20000000298023224, 0.30000001192092896],
        )
        json.dumps(metadata)


if __name__ == "__main__":
    unittest.main()
