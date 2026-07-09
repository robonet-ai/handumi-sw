"""Pose algebra for `[x, y, z, qx, qy, qz, qw]` transforms."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

IDENTITY_POSE7 = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        return np.array([0, 0, 0, 1], dtype=np.float32)
    return (q / n).astype(np.float32)


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def _quat_mul_raw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.asarray(a, dtype=np.float32)
    bx, by, bz, bw = np.asarray(b, dtype=np.float32)
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return quat_normalize(_quat_mul_raw(a, b))


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    v = np.asarray(v, dtype=np.float32)
    vq = np.array([v[0], v[1], v[2], 0], dtype=np.float32)
    return _quat_mul_raw(_quat_mul_raw(q, vq), quat_conj(q))[:3]


def pose_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose transforms: `T_out = T_a @ T_b`."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    out = np.zeros(7, dtype=np.float32)
    out[:3] = a[:3] + quat_rotate(a[3:], b[:3])
    out[3:] = quat_mul(a[3:], b[3:])
    return out


def pose_inv(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32)
    q_inv = quat_conj(quat_normalize(p[3:]))
    out = np.zeros(7, dtype=np.float32)
    out[:3] = quat_rotate(q_inv, -p[:3])
    out[3:] = q_inv
    return out


def pose_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return `target` expressed in `source`'s local frame: `inv(source) @ target`."""
    return pose_mul(pose_inv(source), target)


def delta_pose(prev: np.ndarray | None, cur: np.ndarray) -> np.ndarray:
    """Frame-to-frame delta: current pose expressed in the previous pose frame."""
    if prev is None:
        return IDENTITY_POSE7.copy()

    prev = np.asarray(prev, dtype=np.float32)
    cur = np.asarray(cur, dtype=np.float32)
    cur_q = quat_normalize(cur[3:])
    prev_q = quat_normalize(prev[3:])
    if float(np.dot(cur_q, prev_q)) < 0.0:
        cur_q = -cur_q

    out = np.zeros(7, dtype=np.float32)
    out[:3] = quat_rotate(quat_conj(prev_q), cur[:3] - prev[:3])
    out[3:] = quat_mul(quat_conj(prev_q), cur_q)
    return out


def pose7_to_mat(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    mat = np.zeros(pose.shape[:-1] + (4, 4), dtype=np.float32)
    mat[..., :3, :3] = Rotation.from_quat(pose[..., 3:7]).as_matrix()
    mat[..., :3, 3] = pose[..., :3]
    mat[..., 3, 3] = 1.0
    return mat


def mat_to_pose7(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    pose = np.zeros(mat.shape[:-2] + (7,), dtype=np.float32)
    pose[..., :3] = mat[..., :3, 3]
    pose[..., 3:7] = Rotation.from_matrix(mat[..., :3, :3]).as_quat()
    return pose


def _normalize_vec(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.maximum(norm, eps)


def mat_to_rot6d(mat: np.ndarray) -> np.ndarray:
    """UMI/exUMI row-major 6D rotation representation."""
    mat = np.asarray(mat, dtype=np.float32)
    return mat[..., :2, :].copy().reshape(mat.shape[:-2] + (6,))


def rot6d_to_mat(d6: np.ndarray) -> np.ndarray:
    """Inverse of `mat_to_rot6d`, matching the UMI/exUMI row-major convention."""
    d6 = np.asarray(d6, dtype=np.float32)
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = _normalize_vec(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = _normalize_vec(b2)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2).astype(np.float32)


def mat_to_pose10d(mat: np.ndarray) -> np.ndarray:
    """Return position + 6D rotation. UMI calls this pose10d once gripper is added."""
    mat = np.asarray(mat, dtype=np.float32)
    pos = mat[..., :3, 3]
    rot6d = mat_to_rot6d(mat[..., :3, :3])
    return np.concatenate([pos, rot6d], axis=-1).astype(np.float32)


def pose7_to_pose10d(pose: np.ndarray) -> np.ndarray:
    return mat_to_pose10d(pose7_to_mat(pose))


def pose_sequence_to_pose10d(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float32)
    flat = poses.reshape((-1, 7))
    pose9 = pose7_to_pose10d(flat)
    return pose9.reshape(poses.shape[:-1] + (9,))
