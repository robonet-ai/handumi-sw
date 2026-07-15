import unittest
from pathlib import Path

import numpy as np

from handumi.retargeting.handumi_to_robot import quaternion_xyzw_to_matrix
from handumi.tracking.transforms import (
    MountingOffsets,
    Pose,
    WorkspaceCalibration,
    gripper_pose_in_workspace,
    matrix_to_quat,
    quat_conjugate,
    quat_multiply,
    quat_normalize,
    quat_rotate,
    quat_to_matrix,
    unity_position_to_handumi,
    unity_pose_to_handumi,
    unity_quaternion_to_handumi,
)

CALIBRATION_DIR = Path(__file__).resolve().parents[2] / "configs" / "calibration"


def _rand_quat(rng) -> np.ndarray:
    q = rng.standard_normal(4)
    return quat_normalize(q)


def _rand_pose(rng) -> Pose:
    return Pose(rng.standard_normal(3), _rand_quat(rng))


class QuaternionTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_normalize_zero_is_identity(self):
        self.assertTrue(np.allclose(quat_normalize([0, 0, 0, 0]), [0, 0, 0, 1]))

    def test_multiply_identity(self):
        q = _rand_quat(self.rng)
        ident = [0, 0, 0, 1]
        self.assertTrue(np.allclose(quat_multiply(q, ident), q))
        self.assertTrue(np.allclose(quat_multiply(ident, q), q))

    def test_conjugate_is_inverse(self):
        q = _rand_quat(self.rng)
        self.assertTrue(np.allclose(quat_multiply(q, quat_conjugate(q)), [0, 0, 0, 1]))

    def test_rotate_matches_matrix(self):
        for _ in range(5):
            q = _rand_quat(self.rng)
            v = self.rng.standard_normal(3)
            self.assertTrue(np.allclose(quat_rotate(q, v), quat_to_matrix(q) @ v))

    def test_matrix_quat_round_trip(self):
        for _ in range(5):
            q = _rand_quat(self.rng)
            r = quat_to_matrix(q)
            self.assertTrue(np.allclose(quat_to_matrix(matrix_to_quat(r)), r))

    def test_matrix_matches_retargeting_convention(self):
        # transforms.quat_to_matrix must agree with the existing repo helper.
        for _ in range(5):
            q = _rand_quat(self.rng)
            self.assertTrue(
                np.allclose(quat_to_matrix(q), quaternion_xyzw_to_matrix(q), atol=1e-5)
            )


class PoseTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def test_compose_matches_matrix_product(self):
        a, b = _rand_pose(self.rng), _rand_pose(self.rng)
        self.assertTrue(
            np.allclose(a.compose(b).as_matrix(), a.as_matrix() @ b.as_matrix())
        )

    def test_inverse_round_trip(self):
        p = _rand_pose(self.rng)
        ident = Pose.identity()
        self.assertTrue(np.allclose(p.compose(p.inverse()).as_matrix(), ident.as_matrix()))
        self.assertTrue(np.allclose(p.inverse().compose(p).as_matrix(), ident.as_matrix()))

    def test_matrix_round_trip(self):
        p = _rand_pose(self.rng)
        self.assertTrue(np.allclose(Pose.from_matrix(p.as_matrix()).as_matrix(), p.as_matrix()))

    def test_matmul_operator(self):
        a, b = _rand_pose(self.rng), _rand_pose(self.rng)
        self.assertTrue(np.allclose((a @ b).as_matrix(), a.compose(b).as_matrix()))


class UnityConversionTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2)

    def test_position_axes(self):
        self.assertTrue(np.allclose(unity_position_to_handumi([1, 0, 0]), [0, -1, 0]))
        self.assertTrue(np.allclose(unity_position_to_handumi([0, 1, 0]), [0, 0, 1]))
        self.assertTrue(np.allclose(unity_position_to_handumi([0, 0, 1]), [1, 0, 0]))

    def test_quaternion_consistent_with_position_map(self):
        # Converting a rotated point must equal rotating the converted point
        # under the converted quaternion: M(R v) == R' (M v).
        for _ in range(10):
            q = _rand_quat(self.rng)
            v = self.rng.standard_normal(3)
            r_unity = quat_to_matrix(q)
            r_handumi = quat_to_matrix(unity_quaternion_to_handumi(q))
            lhs = unity_position_to_handumi(r_unity @ v)
            rhs = r_handumi @ unity_position_to_handumi(v)
            self.assertTrue(np.allclose(lhs, rhs, atol=1e-9))

    def test_identity_unity_quaternion(self):
        q = unity_quaternion_to_handumi([0, 0, 0, 1])
        self.assertTrue(np.allclose(quat_to_matrix(q), np.eye(3)))

    def test_pose_helper_matches_parts(self):
        pos = self.rng.standard_normal(3)
        quat = _rand_quat(self.rng)
        pose = unity_pose_to_handumi(pos, quat)
        self.assertTrue(np.allclose(pose.position, unity_position_to_handumi(pos)))
        self.assertTrue(np.allclose(pose.quaternion, unity_quaternion_to_handumi(quat)))


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)

    def test_reference_maps_to_origin(self):
        ref = _rand_pose(self.rng)
        ws = WorkspaceCalibration.from_reference(ref)
        self.assertTrue(np.allclose(ws.apply(ref).as_matrix(), np.eye(4), atol=1e-9))

    def test_relative_pose_preserved(self):
        ref = _rand_pose(self.rng)
        p1, p2 = _rand_pose(self.rng), _rand_pose(self.rng)
        ws = WorkspaceCalibration.from_reference(ref)
        rel_raw = p1.inverse().compose(p2)
        rel_ws = ws.apply(p1).inverse().compose(ws.apply(p2))
        self.assertTrue(np.allclose(rel_raw.as_matrix(), rel_ws.as_matrix(), atol=1e-9))

    def test_identity_is_passthrough(self):
        p = _rand_pose(self.rng)
        self.assertTrue(
            np.allclose(WorkspaceCalibration.identity().apply(p).as_matrix(), p.as_matrix())
        )


class MountingOffsetsTest(unittest.TestCase):
    def test_identity_defaults(self):
        m = MountingOffsets.identity()
        self.assertTrue(np.allclose(m.left.as_matrix(), np.eye(4)))
        self.assertTrue(np.allclose(m.right.as_matrix(), np.eye(4)))

    def test_from_dict(self):
        m = MountingOffsets.from_dict(
            {"left": {"position": [0.1, 0.0, 0.0], "quaternion": [0, 0, 0, 1]}}
        )
        self.assertTrue(np.allclose(m.left.position, [0.1, 0.0, 0.0]))
        self.assertTrue(np.allclose(m.right.as_matrix(), np.eye(4)))  # missing -> identity

    def test_meta_calibration_file_keeps_mirror_invariant(self):
        # The two HandUMI mounts are physical mirror-image twins across the
        # Y=0 plane. The pivot-calibrated translation is projected onto that
        # known symmetry, so right must be the exact mirror of left: position
        # (x, -y, z), quaternion (-x, y, -z, w). A broken mirror once showed
        # up as ~12cm of unwanted lateral offset between the arms. (The pico
        # file holds independent per-side measurements and is not checked.)
        from handumi.calibration.control_tcp import load_controller_tcp_calibration

        calibration = load_controller_tcp_calibration(
            CALIBRATION_DIR / "meta_controller_tcp.yaml"
        )
        lx_p, ly_p, lz_p = calibration.left[:3]
        self.assertTrue(
            np.allclose(calibration.right[:3], [lx_p, -ly_p, lz_p], atol=1e-6)
        )
        self.assertAlmostEqual(
            float(np.linalg.norm(calibration.left[:3])), 0.25, delta=0.01
        )
        lx, ly, lz, lw = calibration.left[3:7]
        self.assertAlmostEqual(float(np.linalg.norm(calibration.left[3:7])), 1.0, places=4)
        self.assertTrue(
            np.allclose(calibration.right[3:7], [-lx, ly, -lz, lw], atol=1e-6)
        )


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(4)

    def test_identity_calibration_equals_unity_conversion(self):
        pos = self.rng.standard_normal(3)
        quat = _rand_quat(self.rng)
        out = gripper_pose_in_workspace(
            pos, quat,
            mounting_offset=Pose.identity(),
            workspace=WorkspaceCalibration.identity(),
        )
        self.assertTrue(np.allclose(out.as_matrix(), unity_pose_to_handumi(pos, quat).as_matrix()))

    def test_full_pipeline_order(self):
        pos = self.rng.standard_normal(3)
        quat = _rand_quat(self.rng)
        mount = _rand_pose(self.rng)
        ref = _rand_pose(self.rng)
        ws = WorkspaceCalibration.from_reference(ref)

        out = gripper_pose_in_workspace(pos, quat, mounting_offset=mount, workspace=ws)
        expected = ws.apply(unity_pose_to_handumi(pos, quat).compose(mount))
        self.assertTrue(np.allclose(out.as_matrix(), expected.as_matrix(), atol=1e-9))


if __name__ == "__main__":
    unittest.main()
