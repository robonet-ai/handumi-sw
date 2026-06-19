#!/usr/bin/env python3
"""
IK replay del robot Axol desde un dataset Pico (modo delta-pose relativo).

Lee un episodio, convierte las poses de los controllers Pico en deltas
relativos al primer frame (anclaje relativo), los aplica sobre la pose home
del EE (FK en q=0) en el frame del robot y anima el resultado.

Uso
───
    uv run python test/test_ik_axol.py datasets/my_dataset
    uv run python test/test_ik_axol.py datasets/my_dataset --episode 0
    uv run python test/test_ik_axol.py datasets/my_dataset --speed 0.5

    # Corrección de ejes Pico → robot (sólo rotación; la traslación se ignora):
    uv run python test/test_ik_axol.py datasets/my_dataset \\
        --axes-rotation 0.0 0.0 0.0 1.0   # qx qy qz qw

    # Frame de calibración distinto al 0:
    uv run python test/test_ik_axol.py datasets/my_dataset --calib-frame 10

    # Sólo visor URDF en MuJoCo (sin dataset):
    uv run python test/test_ik_axol.py --urdf
    uv run python test/test_ik_axol.py --urdf --animate
    uv run python test/test_ik_axol.py --urdf --collision

Estrategia de transformación Pico → robot
──────────────────────────────────────────
Se usa delta-pose tracking (anclaje relativo):

    T_robot_ee(t) = T_robot_ee(0) · T_pico(0)⁻¹ · T_pico(t)

Donde T_robot_ee(0) es la pose home del EE (FK en q=0).  El movimiento del
controller se expresa como delta respecto al frame de calibración (calib_frame,
default=0) y se aplica sobre la pose home del EE en el robot.  Esto elimina
la dependencia de la posición absoluta del headset.

Si los ejes del Pico y del robot no coinciden (VR usa Y-arriba, robots Z-arriba),
usa --axes-rotation para rotar las poses del Pico antes de calcular los deltas.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


# ── Carga del dataset ──────────────────────────────────────────────────────────

def load_episode(
    root: Path,
    episode: int | None,
) -> tuple[int, float, np.ndarray, np.ndarray, np.ndarray | None]:
    """Carga un episodio del dataset LeRobot (parquet).

    Returns
    -------
    episode_id  : int
    fps         : float
    left_poses  : (N, 7)     [x, y, z, qx, qy, qz, qw] — controller izquierdo
    right_poses : (N, 7)     [x, y, z, qx, qy, qz, qw] — controller derecho
    body_poses  : (N, 24, 7) o None  — body joints del Pico (para hints de codo)
    """
    parquet_files = sorted((root / "data").rglob("*.parquet"))
    if not parquet_files:
        sys.exit(f"[ERROR] No se encontraron .parquet en {root / 'data'}")

    df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
    df.sort_values("index", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if episode is None:
        episode = int(df["episode_index"].iloc[0])
    mask = df["episode_index"] == episode
    if not mask.any():
        sys.exit(f"[ERROR] El episodio {episode} no existe en el dataset.")
    df = df[mask]

    fps = 30.0
    info_path = root / "meta" / "info.json"
    if info_path.exists():
        fps = float(json.loads(info_path.read_text()).get("fps", 30))

    left  = np.stack(df["observation.pico.left_controller_pose"].values).astype(np.float32)
    right = np.stack(df["observation.pico.right_controller_pose"].values).astype(np.float32)

    body: np.ndarray | None = None
    if "observation.pico.body_joints_pose" in df.columns:
        try:
            raw = np.stack(df["observation.pico.body_joints_pose"].values).astype(np.float32)
            body = raw.reshape(len(df), 24, 7)
        except Exception:
            pass

    return episode, fps, left, right, body


# ── Transformación de frames Pico → robot (delta-pose relativo) ───────────────

def compute_relative_poses(
    poses: np.ndarray,
    ref_idx: int = 0,
) -> np.ndarray:
    """Convierte poses absolutas del Pico en deltas relativos al frame ref_idx.

    Parameters
    ----------
    poses   : (N, 7) [x, y, z, qx, qy, qz, qw] en frame Pico
    ref_idx : índice del frame de referencia (calibración), normalmente 0

    Returns
    -------
    delta_poses : (N, 7) — pose relativa: delta_pos expresado en el frame de
                  referencia, delta_rot como cuaternión relativo [qx,qy,qz,qw]
    """
    ref_pos = poses[ref_idx, :3].astype(np.float64)
    ref_rot = Rotation.from_quat(poses[ref_idx, 3:])
    ref_rot_inv = ref_rot.inv()

    out = np.zeros_like(poses, dtype=np.float32)
    for i in range(len(poses)):
        pos_i = poses[i, :3].astype(np.float64)
        rot_i = Rotation.from_quat(poses[i, 3:])

        delta_pos = ref_rot_inv.apply(pos_i - ref_pos)
        delta_rot = ref_rot_inv * rot_i

        out[i, :3] = delta_pos.astype(np.float32)
        out[i, 3:] = delta_rot.as_quat().astype(np.float32)

    return out


def apply_axes_rotation(poses: np.ndarray, R_axes: np.ndarray) -> np.ndarray:
    """Aplica una rotación fija de ejes (3×3) a las poses del Pico.

    Rota tanto la posición como la orientación por R_axes.  Úsala para
    alinear la convención de ejes del Pico con la del robot **antes** de
    calcular los deltas relativos con ``compute_relative_poses``.

    Parameters
    ----------
    poses  : (N, 7) [x, y, z, qx, qy, qz, qw]
    R_axes : (3, 3) rotación fija Pico → robot axes

    Returns
    -------
    (N, 7) poses con posición y orientación rotadas
    """
    out = poses.copy()
    R_rot = Rotation.from_matrix(R_axes)
    for i in range(len(poses)):
        out[i, :3] = (R_axes @ poses[i, :3].astype(np.float64)).astype(np.float32)
        rot_i = Rotation.from_quat(poses[i, 3:])
        out[i, 3:] = (R_rot * rot_i).as_quat().astype(np.float32)
    return out


def relative_to_robot_pose(
    delta_pos: np.ndarray,
    delta_rot: np.ndarray,
    home_pos: np.ndarray,
    home_rot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Aplica un delta de pose (relativo a calibración) sobre la pose home del EE.

    Fórmula:
        pos_robot = home_pos + home_rot @ delta_pos
        rot_robot = home_rot @ delta_rot

    Parameters
    ----------
    delta_pos : (3,)   delta de traslación en frame de calibración
    delta_rot : (3,3)  delta de rotación como matriz
    home_pos  : (3,)   posición home del EE en frame robot (FK en q=0)
    home_rot  : (3,3)  orientación home del EE en frame robot (FK en q=0)

    Returns
    -------
    pos_robot : (3,)  float32
    rot_robot : (3,3) float32
    """
    pos_robot = home_pos + home_rot @ delta_pos
    rot_robot = home_rot @ delta_rot
    return pos_robot.astype(np.float32), rot_robot.astype(np.float32)


# ── Solución IK ────────────────────────────────────────────────────────────────

# Índices de body_joints_pose para los codos según la convención Pico OpenXR.
# Ajusta con --left-elbow-joint / --right-elbow-joint si tu SDK usa otros índices.
_BODY_LEFT_ELBOW  = 9
_BODY_RIGHT_ELBOW = 13


def solve_episode(
    solver,
    left: np.ndarray,
    right: np.ndarray,
    body: np.ndarray | None,
    R_axes: np.ndarray | None = None,
    *,
    left_elbow_idx: int  = _BODY_LEFT_ELBOW,
    right_elbow_idx: int = _BODY_RIGHT_ELBOW,
    use_elbow_hints: bool = True,
    calib_frame: int = 0,
    position_only: bool = False,
) -> np.ndarray:
    """Resuelve IK frame a frame usando delta-pose relativo.

    Estrategia: cada pose del controller se convierte en un delta respecto al
    frame de calibración (``calib_frame``) y se aplica sobre la pose home del
    EE dada por FK(q=0).  Esto hace que el IK sea independiente de la posición
    absoluta del headset.

    Parameters
    ----------
    solver            : KinematicsSolver
    left, right       : (N, 7) poses absolutas de los controllers en frame Pico
    body              : (N, 24, 7) o None — poses de joints corporales del Pico
    R_axes            : (3, 3) o None — rotación fija de ejes Pico → robot
                        (ver ``apply_axes_rotation``); None = identidad
    left_elbow_idx    : fila de body_joints_pose que corresponde al codo izquierdo
    right_elbow_idx   : fila de body_joints_pose que corresponde al codo derecho
    use_elbow_hints   : si True y body disponible, pasa hints de codo al solver
    calib_frame       : índice del frame de referencia para los deltas (default 0)

    Returns
    -------
    joint_traj : (N, solver.num_joints)  ángulos en radianes
    """
    N = len(left)
    joint_traj = np.zeros((N, solver.num_joints), dtype=np.float32)
    q = np.zeros(solver.num_joints, dtype=np.float32)

    # Pose home del EE (FK en q=0) — punto de anclaje del movimiento relativo
    q0 = np.zeros(solver.num_joints, dtype=np.float32)
    ee_L0, ee_R0 = solver.fk(q0)
    home_pos_L = np.asarray(ee_L0.translation(), dtype=np.float64)
    home_rot_L = np.asarray(ee_L0.rotation().as_matrix(), dtype=np.float64)
    home_pos_R = np.asarray(ee_R0.translation(), dtype=np.float64)
    home_rot_R = np.asarray(ee_R0.rotation().as_matrix(), dtype=np.float64)

    # Alinear ejes Pico → robot antes de calcular deltas (si se provee R_axes)
    left_in  = apply_axes_rotation(left,  R_axes) if R_axes is not None else left
    right_in = apply_axes_rotation(right, R_axes) if R_axes is not None else right

    # Deltas relativos al frame de calibración
    delta_left  = compute_relative_poses(left_in,  calib_frame)
    delta_right = compute_relative_poses(right_in, calib_frame)

    # Deltas del codo — misma lógica, anclados al home del EE correspondiente
    delta_elbow_L: np.ndarray | None = None
    delta_elbow_R: np.ndarray | None = None
    if use_elbow_hints and body is not None:
        body_L = body[:, left_elbow_idx, :]
        body_R = body[:, right_elbow_idx, :]
        if R_axes is not None:
            body_L = apply_axes_rotation(body_L, R_axes)
            body_R = apply_axes_rotation(body_R, R_axes)
        delta_elbow_L = compute_relative_poses(body_L, calib_frame)
        delta_elbow_R = compute_relative_poses(body_R, calib_frame)

    print(
        f"Resolviendo IK para {N} frames "
        f"(modo relativo, calib_frame={calib_frame})…",
        flush=True,
    )
    t0 = time.monotonic()

    for i in range(N):
        dL_pos = delta_left[i, :3].astype(np.float64)
        dR_pos = delta_right[i, :3].astype(np.float64)
        if position_only:
            # Modo "solo posición relativa": conserva la orientación home del EE.
            dL_rot = np.eye(3, dtype=np.float64)
            dR_rot = np.eye(3, dtype=np.float64)
        else:
            dL_rot = Rotation.from_quat(delta_left[i, 3:]).as_matrix()
            dR_rot = Rotation.from_quat(delta_right[i, 3:]).as_matrix()

        lp, lR = relative_to_robot_pose(dL_pos, dL_rot, home_pos_L, home_rot_L)
        rp, rR = relative_to_robot_pose(dR_pos, dR_rot, home_pos_R, home_rot_R)

        elbow_L: np.ndarray | None = None
        elbow_R: np.ndarray | None = None
        if delta_elbow_L is not None:
            dEL_pos = delta_elbow_L[i, :3].astype(np.float64)
            elbow_L, _ = relative_to_robot_pose(dEL_pos, np.eye(3), home_pos_L, home_rot_L)
        if delta_elbow_R is not None:
            dER_pos = delta_elbow_R[i, :3].astype(np.float64)
            elbow_R, _ = relative_to_robot_pose(dER_pos, np.eye(3), home_pos_R, home_rot_R)

        q = solver.ik(
            q,
            left_pose=(lp, lR),
            right_pose=(rp, rR),
            left_elbow_pos=elbow_L,
            right_elbow_pos=elbow_R,
        )
        joint_traj[i] = q

        if (i + 1) % 50 == 0 or i == N - 1:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            print(f"  {i + 1:4d}/{N}  ({rate:.1f} frames/s)", end="\r", flush=True)

    print(f"\nIK completado en {time.monotonic() - t0:.1f} s")
    return joint_traj


# ── MuJoCo / URDF ──────────────────────────────────────────────────────────────

_AXOL_SCENE_XML = """\
<mujoco model="axol_scene">
  <compiler angle="radian" autolimits="true"/>
  <option gravity="0 0 -9.81"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072"/>
    <texture name="grid" type="2d" builtin="checker"
             rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3"
             width="512" height="512" mark="edge" markrgb="0.8 0.8 0.8"/>
    <material name="grid" texture="grid" texrepeat="4 4" texuniform="true"
              reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="2 2 0.05" material="grid"
          condim="3" friction="1 0.005 0.0001"/>
    <frame name="robot_mount" pos="0 0 0"/>
  </worldbody>
</mujoco>
"""


def _axol_urdf_path() -> Path:
    """Ruta al URDF del Axol (paquete instalado o árbol fuente)."""
    try:
        from dexumi.robots.utils import URDF_PATH
        return URDF_PATH
    except ImportError:
        return _REPO_ROOT / "assets" / "meshes" / "axol" / "urdf" / "axol.urdf"


def _mujoco_urdf_xml(urdf_path: Path, *, collision: bool = False) -> str:
    """Prepara el URDF para MuJoCo (rutas de mesh y, opcionalmente, sólo colisión)."""
    text = urdf_path.read_text()
    if collision:
        text = re.sub(r"<visual>.*?</visual>\s*", "", text, flags=re.DOTALL)
    mesh_dir = urdf_path.parent / "meshes"

    def _mesh_path(match: re.Match[str]) -> str:
        return str(mesh_dir / match.group(1)).replace("\\", "/")

    return re.sub(r'package://assembly/meshes/([^"]+)', _mesh_path, text)


def load_axol_mujoco(*, collision: bool = False) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Carga el Axol en MuJoCo dentro de la escena definida en ``_AXOL_SCENE_XML``."""
    urdf_path = _axol_urdf_path()
    if not urdf_path.is_file():
        sys.exit(f"[ERROR] URDF no encontrado en {urdf_path}")

    scene_spec = mujoco.MjSpec.from_string(_AXOL_SCENE_XML)
    robot_spec = mujoco.MjSpec.from_string(
        _mujoco_urdf_xml(urdf_path, collision=collision),
    )
    mount = scene_spec.worldbody.frames[0]
    scene_spec.attach(robot_spec, frame=mount, prefix="")
    model = scene_spec.compile()
    return model, mujoco.MjData(model)


def _hinge_joint_names(model: mujoco.MjModel) -> list[str]:
    names: list[str] = []
    for i in range(model.njnt):
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            names.append(name)
    return names


def _apply_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: list[str],
    q: np.ndarray,
) -> None:
    """Escribe ángulos en ``data.qpos`` por nombre de joint (orden arbitrario)."""
    for name, val in zip(joint_names, q, strict=True):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            sys.exit(f"[ERROR] Joint no encontrado en el modelo MuJoCo: {name}")
        data.qpos[model.jnt_qposadr[jid]] = val
    mujoco.mj_forward(model, data)


def _run_mujoco_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    update: Callable[[], None] | None = None,
) -> None:
    """Bucle del visor pasivo MuJoCo; ``update`` se llama en cada iteración.

    Las modificaciones a ``data`` se realizan dentro de ``viewer.lock()``
    para evitar condiciones de carrera con el hilo de rendering de MuJoCo 3.x.
    """
    print("\nVisor MuJoCo abierto. Cierra la ventana para salir.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if update is not None:
                with viewer.lock():
                    update()
            viewer.sync()
            time.sleep(0.001)


# ── Animación IK replay ────────────────────────────────────────────────────────

def animate_ik_replay(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    solver_joint_names: list[str],
    joint_traj: np.ndarray,
    fps: float,
    speed: float,
) -> None:
    """Anima la trayectoria IK en el visor MuJoCo.

    Parameters
    ----------
    solver_joint_names : nombres de joints en el mismo orden que joint_traj
    joint_traj         : (N, num_joints) ángulos en radianes
    fps                : FPS del dataset (determina la duración de cada frame)
    speed              : factor de velocidad (1.0 = tiempo real)
    """
    n_frames = len(joint_traj)
    frame_dt = 1.0 / (fps * max(speed, 1e-6))
    # t_last se inicializa a None; se fija en la primera llamada a update()
    # para no consumir frames durante la apertura del visor.
    state: dict[str, int | float | None] = {"frame": 0, "t_last": None}

    def update() -> None:
        now = time.monotonic()
        if state["t_last"] is None:
            state["t_last"] = now
        if now - state["t_last"] >= frame_dt:  # type: ignore[operator]
            state["frame"] = (state["frame"] + 1) % n_frames  # type: ignore[operator]
            state["t_last"] = now
        _apply_qpos(model, data, solver_joint_names, joint_traj[state["frame"]])  # type: ignore[index]

    _apply_qpos(model, data, solver_joint_names, joint_traj[0])
    _run_mujoco_viewer(model, data, update=update)


# ── Visor URDF (modo debug) ────────────────────────────────────────────────────

def _joint_limit_trajectory(
    model: mujoco.MjModel,
    loop_time: float,
) -> tuple[list[str], np.ndarray]:
    """Trayectoria que interpola cada hinge entre sus límites en un ciclo."""
    joint_names = _hinge_joint_names(model)
    via = np.zeros((len(joint_names), 3), dtype=np.float64)
    idx = 0
    for i in range(model.njnt):
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        lo, hi = model.jnt_range[i]
        via[idx] = (lo, hi, lo)
        idx += 1

    times = np.linspace(0.0, 1.0, int(loop_time * 100))
    bins = np.arange(3) / 2.0
    inds = np.digitize(times, bins, right=True)
    inds[inds == 0] = 1
    alphas = (bins[inds] - times) / (bins[inds] - bins[inds - 1])
    traj = alphas[None, :] * via[:, inds - 1] + (1.0 - alphas)[None, :] * via[:, inds]
    return joint_names, traj


def visualize_urdf(
    *,
    collision: bool = False,
    animate: bool = False,
    configuration: list[float] | None = None,
) -> None:
    """Abre el visor MuJoCo del URDF del Axol (modo de depuración, sin IK)."""
    model, data = load_axol_mujoco(collision=collision)

    print(f"Bodies : {model.nbody}")
    hinge_names = _hinge_joint_names(model)
    print(f"Joints : {model.njnt}  (hinge: {len(hinge_names)})")
    for jid in range(model.njnt):
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint_{jid}"
        lo, hi = model.jnt_range[jid]
        print(f"  {name:20s}  [{lo:+.3f}, {hi:+.3f}] rad")

    if configuration is not None:
        if len(configuration) != len(hinge_names):
            sys.exit(
                f"[ERROR] --configuration requiere {len(hinge_names)} valores "
                f"(uno por joint hinge), recibidos {len(configuration)}."
            )
        _apply_qpos(model, data, hinge_names, np.asarray(configuration, dtype=np.float64))

    update: Callable[[], None] | None = None
    if animate:
        loop_time = 6.0
        traj_names, traj = _joint_limit_trajectory(model, loop_time)

        def update() -> None:  # noqa: F811
            frame = int(100.0 * (time.time() % loop_time))
            _apply_qpos(model, data, traj_names, traj[:, frame])

    _run_mujoco_viewer(model, data, update=update)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── IK replay ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "dataset", nargs="?", type=Path,
        help="Ruta raíz del dataset (p.ej. datasets/my_dataset).",
    )
    parser.add_argument(
        "--episode", "-e", type=int, default=None,
        help="Episodio a reproducir (default: primero disponible).",
    )
    parser.add_argument(
        "--speed", "-s", type=float, default=1.0,
        help="Factor de velocidad de la animación (default: 1.0 = tiempo real).",
    )
    parser.add_argument(
        "--axes-rotation",
        nargs=4, type=float,
        metavar=("QX", "QY", "QZ", "QW"),
        default=None,
        help=(
            "Rotación fija de ejes Pico → robot como cuaternión [qx qy qz qw]. "
            "Se aplica a las poses del Pico ANTES de calcular los deltas relativos. "
            "Úsala si el eje Y del Pico corresponde al eje Z del robot, etc. "
            "Si no se pasa, se asume identidad (mismos ejes)."
        ),
    )
    parser.add_argument(
        "--calib-frame", type=int, default=0,
        help=(
            "Índice del frame de calibración (referencia para los deltas relativos). "
            "Default: 0 (primer frame del episodio)."
        ),
    )
    parser.add_argument(
        "--no-elbow-hints", action="store_true",
        help="Desactiva el uso de body_joints_pose como hints de codo en el IK.",
    )
    parser.add_argument(
        "--position-only", action="store_true",
        help=(
            "Usa únicamente deltas de posición relativa de los controllers; "
            "la orientación objetivo de cada EE se mantiene en la orientación home."
        ),
    )
    parser.add_argument(
        "--ik-profile",
        choices=("retarget", "default"),
        default="retarget",
        help=(
            "Perfil de pesos del IK. "
            "'retarget' (default) reduce penalizaciones para que siga mejor "
            "trayectorias humanas; 'default' usa KinematicsConfig base."
        ),
    )
    parser.add_argument(
        "--left-elbow-joint", type=int, default=_BODY_LEFT_ELBOW,
        help=f"Índice del codo izquierdo en body_joints_pose (default: {_BODY_LEFT_ELBOW}).",
    )
    parser.add_argument(
        "--right-elbow-joint", type=int, default=_BODY_RIGHT_ELBOW,
        help=f"Índice del codo derecho en body_joints_pose (default: {_BODY_RIGHT_ELBOW}).",
    )

    # ── Visor URDF ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--urdf", action="store_true",
        help="Modo visor URDF sin IK (no requiere dataset).",
    )
    parser.add_argument(
        "--animate", action="store_true",
        help="[--urdf] Anima los joints entre sus límites.",
    )
    parser.add_argument(
        "--collision", action="store_true",
        help="[--urdf] Mostrar geometría de colisión.",
    )
    parser.add_argument(
        "--configuration", nargs="+", type=float, metavar="Q",
        help="[--urdf] Configuración inicial (un valor por joint actuado, en orden URDF).",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # ── Modo visor URDF ────────────────────────────────────────────────────────
    if args.urdf:
        visualize_urdf(
            collision=args.collision,
            animate=args.animate,
            configuration=args.configuration,
        )
        return

    # ── Modo IK replay ─────────────────────────────────────────────────────────
    if args.dataset is None:
        sys.exit("[ERROR] Proporciona la ruta del dataset o usa --urdf.\n"
                 "Ejemplo: uv run python test/test_ik_axol.py datasets/my_dataset")

    root = args.dataset.resolve()
    if not root.exists():
        sys.exit(f"[ERROR] No existe el directorio: {root}")

    # Cargar solver (incluye JIT warmup)
    from dexumi.robots.axol import KinematicsConfig, KinematicsSolver

    if args.ik_profile == "retarget":
        # Para replay humano -> robot conviene relajar costos que tienden a inmovilizar
        # el sistema (límites/colisión/manipulabilidad) en targets lejanos.
        cfg = KinematicsConfig(
            pos_weight=80.0,
            ori_weight=0.0,
            elbow_weight=5.0,
            rest_weight=0.5,
            posture_weight=0.0,
            manipulability_weight=0.0,
            limit_weight=0.0,
            self_collision_weight=0.0,
            max_iterations=40,
            cost_tolerance=1e-6,
            max_joint_delta=0.2,
        )
    else:
        cfg = KinematicsConfig()

    print("Cargando KinematicsSolver (JIT warmup incluido)…")
    solver = KinematicsSolver(cfg)
    print(f"Joints actuados : {solver.num_joints}")
    print(f"Orden joints    : {solver.joint_names}")
    print(f"Perfil IK       : {args.ik_profile}")

    # Cargar episodio
    episode_id, fps, left, right, body = load_episode(root, args.episode)
    N = len(left)
    print(f"\nDataset  : {root.name}")
    print(f"Episodio : {episode_id}  |  {N} frames @ {fps:.0f} fps")
    print(f"Body tracking: {'disponible' if body is not None else 'no disponible'}")
    print(f"Left  x=[{left[:,0].min():.3f}, {left[:,0].max():.3f}]  "
          f"y=[{left[:,1].min():.3f}, {left[:,1].max():.3f}]  "
          f"z=[{left[:,2].min():.3f}, {left[:,2].max():.3f}]")
    print(f"Right x=[{right[:,0].min():.3f}, {right[:,0].max():.3f}]  "
          f"y=[{right[:,1].min():.3f}, {right[:,1].max():.3f}]  "
          f"z=[{right[:,2].min():.3f}, {right[:,2].max():.3f}]")

    # Rotación de ejes Pico → robot (opcional)
    R_axes: np.ndarray | None = None
    if args.axes_rotation is not None:
        qx, qy, qz, qw = args.axes_rotation
        R_axes = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        print(f"\nRotación de ejes : q=[{qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f}]")
    else:
        print("\nRotación de ejes : identidad (usa --axes-rotation si los ejes difieren)")

    print(f"Frame calibración: {args.calib_frame}")
    print(f"Modo orientación : {'solo posición relativa' if args.position_only else 'posición + orientación relativa'}")

    # Diagnóstico previo: rango de movimiento relativo en datos crudos
    left_tmp  = apply_axes_rotation(left,  R_axes) if R_axes is not None else left
    right_tmp = apply_axes_rotation(right, R_axes) if R_axes is not None else right
    dl = compute_relative_poses(left_tmp,  args.calib_frame)
    dr = compute_relative_poses(right_tmp, args.calib_frame)
    dl_range = dl[:, :3].max(0) - dl[:, :3].min(0)
    dr_range = dr[:, :3].max(0) - dr[:, :3].min(0)
    print(f"\nRango delta izquierdo (m) : x={dl_range[0]:.4f}  y={dl_range[1]:.4f}  z={dl_range[2]:.4f}")
    print(f"Rango delta derecho   (m) : x={dr_range[0]:.4f}  y={dr_range[1]:.4f}  z={dr_range[2]:.4f}")
    if dl_range.max() < 5e-3 and dr_range.max() < 5e-3:
        print("[WARN] Los deltas de posición son menores de 5 mm — "
              "el episodio puede no tener movimiento real o el calib_frame es incorrecto.")

    # Resolver IK para todos los frames
    joint_traj = solve_episode(
        solver, left, right, body, R_axes,
        left_elbow_idx=args.left_elbow_joint,
        right_elbow_idx=args.right_elbow_joint,
        use_elbow_hints=not args.no_elbow_hints,
        calib_frame=args.calib_frame,
        position_only=args.position_only,
    )

    # Diagnóstico rápido de la trayectoria IK
    q_std = joint_traj.std(axis=0)
    q_range = joint_traj.max(axis=0) - joint_traj.min(axis=0)
    print("\nMovimiento IK por joint (rango en rad):")
    for jname, rng, std in zip(solver.joint_names, q_range, q_std):
        flag = "  ← PLANO" if rng < 1e-3 else ""
        print(f"  {jname:20s}  rango={rng:+.4f}  std={std:.4f}{flag}")
    if q_range.max() < 1e-3:
        print("[WARN] La trayectoria IK no tiene variación — revisa el dataset y el transform.")

    # Animar en el visor MuJoCo
    model, data = load_axol_mujoco()
    animate_ik_replay(model, data, solver.joint_names, joint_traj, fps, args.speed)


if __name__ == "__main__":
    main()
