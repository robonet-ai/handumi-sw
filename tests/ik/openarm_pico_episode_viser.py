#!/usr/bin/env python3
"""Replay one PICO controller episode through OpenArm IK in Viser.

This is an intentionally small manual test. It downloads only the manifest and
episode parquet data (never the video), and reads only the two raw PICO
controller pose columns. The canonical controller-to-TCP YAML is applied before
retargeting to OpenArm.
"""

from __future__ import annotations

import argparse
import json
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from handumi.calibration.control_tcp import load_controller_tcp_calibration
from handumi.retargeting.handumi_to_robot import (
    VR_TO_ROBOT,
    local_frame_adapter,
    local_relative_robot_target_pose7,
)
from handumi.robots.kinematics import pose_error_arrays
from handumi.robots.registry import load_embodiment
from handumi.robots.utils import pose_mul

DEFAULT_REPO_ID = "NONHUMAN-RESEARCH/pico-dataset-tcp-01"
DEFAULT_TCP_CALIBRATION = Path("configs/calibration/pico_controller_tcp.yaml")
LEFT_CONTROLLER = "observation.pico.left_controller_pose"
RIGHT_CONTROLLER = "observation.pico.right_controller_pose"
SIDES = ("left", "right")


@dataclass(frozen=True)
class ControllerEpisode:
    episode: int
    fps: float
    controller_pose7: dict[str, np.ndarray]
    tcp_pose7: dict[str, np.ndarray]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show one pico-dataset-tcp-01 episode driving OpenArm IK in Viser."
    )
    parser.add_argument("-e", "--episode", type=int, default=3)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--tcp-calibration",
        type=Path,
        default=DEFAULT_TCP_CALIBRATION,
        help="Canonical PICO controller-to-TCP YAML.",
    )
    parser.add_argument("--home-pose", default="down")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit frames after stride; useful for a quick smoke test.",
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Solve and print IK errors without starting Viser.",
    )
    return parser


def _download(repo_id: str, revision: str, filename: str) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=filename,
        )
    )


def load_controller_episode(
    repo_id: str, revision: str, episode: int, tcp_calibration: Path
) -> ControllerEpisode:
    manifest_path = _download(repo_id, revision, "manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {
        int(item["episode_index"]): item for item in manifest.get("episodes", [])
    }
    if episode not in entries:
        available = ", ".join(str(index) for index in sorted(entries))
        raise ValueError(f"Episode {episode} does not exist; available: {available}.")

    entry = entries[episode]
    parquet_path = _download(repo_id, revision, str(entry["data"]))
    table = pq.read_table(parquet_path, columns=[LEFT_CONTROLLER, RIGHT_CONTROLLER])

    controller = {
        "left": np.asarray(table[LEFT_CONTROLLER].to_pylist(), dtype=np.float32),
        "right": np.asarray(table[RIGHT_CONTROLLER].to_pylist(), dtype=np.float32),
    }
    calibration = load_controller_tcp_calibration(tcp_calibration)
    tcp: dict[str, np.ndarray] = {}
    for side in SIDES:
        poses = controller[side]
        if poses.ndim != 2 or poses.shape[1] != 7 or not np.isfinite(poses).all():
            raise ValueError(
                f"Invalid {side} controller poses: expected finite (N, 7), got {poses.shape}."
            )
        offset = np.asarray(getattr(calibration, side), dtype=np.float32)
        if offset.shape != (7,) or not np.isfinite(offset).all():
            raise ValueError(
                f"Invalid {side} transform in {tcp_calibration}: expected a finite pose7."
            )
        tcp[side] = np.stack([pose_mul(pose, offset) for pose in poses]).astype(
            np.float32
        )

    if len(controller["left"]) != len(controller["right"]):
        raise ValueError("Left and right controller streams have different lengths.")
    return ControllerEpisode(
        episode=episode,
        fps=float(manifest.get("fps", 30)),
        controller_pose7=controller,
        tcp_pose7=tcp,
    )


def solve_episode(
    episode: ControllerEpisode,
    *,
    home_pose: str,
    stride: int,
    max_frames: int | None,
) -> dict[str, np.ndarray]:
    if stride < 1:
        raise ValueError("--stride must be >= 1.")
    if max_frames is not None and max_frames < 1:
        raise ValueError("--max-frames must be >= 1.")

    runtime = load_embodiment("openarmv1")
    solver = runtime.solver_cls()
    q = runtime.home_q(home_pose)
    home_left, home_right = solver.fk_pose7(q)
    homes = {"left": home_left, "right": home_right}
    indices = np.arange(0, len(episode.tcp_pose7["left"]), stride, dtype=np.int64)
    if max_frames is not None:
        indices = indices[:max_frames]
    if len(indices) == 0:
        raise ValueError("No frames selected.")

    first = {side: episode.tcp_pose7[side][indices[0]] for side in SIDES}
    adapters = {
        side: local_frame_adapter(
            first[side], homes[side], source_world_to_robot_world=VR_TO_ROBOT
        )
        for side in SIDES
    }
    targets = {side: [homes[side].copy()] for side in SIDES}
    achieved = {side: [homes[side].copy()] for side in SIDES}
    qpos = [q.copy()]

    start = time.perf_counter()
    for previous_index, current_index in zip(indices[:-1], indices[1:], strict=True):
        frame_targets = {}
        for side in SIDES:
            frame_targets[side] = local_relative_robot_target_pose7(
                previous_source_pose7=episode.tcp_pose7[side][previous_index],
                current_source_pose7=episode.tcp_pose7[side][current_index],
                base_robot_pose7=targets[side][-1],
                adapter_rot=adapters[side],
                home_robot_pose7=homes[side],
                max_reach=runtime.config.ik_weights.max_reach,
            )
        q = solver.ik(
            q,
            left_pose=(frame_targets["left"][:3], frame_targets["left"][3:]),
            right_pose=(frame_targets["right"][:3], frame_targets["right"][3:]),
        )
        fk_left, fk_right = solver.fk_pose7(q)
        qpos.append(q.copy())
        for side, fk in (("left", fk_left), ("right", fk_right)):
            targets[side].append(frame_targets[side])
            achieved[side].append(fk)

    elapsed = time.perf_counter() - start
    target_left = np.asarray(targets["left"], dtype=np.float32)
    target_right = np.asarray(targets["right"], dtype=np.float32)
    achieved_left = np.asarray(achieved["left"], dtype=np.float32)
    achieved_right = np.asarray(achieved["right"], dtype=np.float32)
    errors = pose_error_arrays(
        target_left, target_right, achieved_left, achieved_right
    )
    all_position = np.concatenate(
        [errors["left_pos_error_m"], errors["right_pos_error_m"]]
    )
    all_rotation = np.concatenate(
        [errors["left_rot_error_deg"], errors["right_rot_error_deg"]]
    )
    print(
        f"[openarm-pico] episode={episode.episode} frames={len(qpos)} "
        f"fps={episode.fps:g} solved={elapsed:.2f}s"
    )
    print(
        "[openarm-pico] IK error: "
        f"position mean={all_position.mean() * 100:.2f}cm "
        f"max={all_position.max() * 100:.2f}cm; "
        f"rotation mean={all_rotation.mean():.2f}deg "
        f"max={all_rotation.max():.2f}deg"
    )
    return {
        "qpos": np.asarray(qpos, dtype=np.float32),
        "target_left": target_left,
        "target_right": target_right,
        "achieved_left": achieved_left,
        "achieved_right": achieved_right,
        **errors,
        "fps": np.asarray([episode.fps / stride], dtype=np.float32),
    }


def show_viewer(
    rollout: dict[str, np.ndarray], *, port: int, speed: float, open_browser: bool
) -> None:
    import viser
    from viser.extras import ViserUrdf

    if speed <= 0.0:
        raise ValueError("--speed must be > 0.")
    runtime = load_embodiment("openarmv1")
    server = viser.ViserServer(port=port)
    server.scene.add_grid("/grid", width=3.0, height=3.0, cell_size=0.1)
    robot_view = ViserUrdf(
        server,
        runtime.load_urdf(load_meshes=True),
        root_node_name="/robot",
    )
    colors = {
        "target_left": (255, 190, 50),
        "target_right": (80, 220, 130),
        "achieved_left": (80, 160, 255),
        "achieved_right": (255, 90, 90),
    }
    for key, color in colors.items():
        server.scene.add_spline_catmull_rom(
            f"/trajectory/{key}",
            positions=rollout[key][:, :3],
            color=color,
            line_width=2.0,
        )
    target_frames = {
        side: server.scene.add_frame(
            f"/target/{side}", axes_length=0.08, axes_radius=0.003
        )
        for side in SIDES
    }
    achieved_markers = {
        side: server.scene.add_icosphere(
            f"/achieved/{side}",
            radius=0.014,
            color=colors[f"achieved_{side}"],
        )
        for side in SIDES
    }
    play = server.gui.add_checkbox("Play", True)
    frame = server.gui.add_slider("Frame", 0, len(rollout["qpos"]) - 1, 1, 0)
    error_text = server.gui.add_text("IK error (cm / deg)", "-", disabled=True)

    def draw(index: int) -> None:
        robot_view.update_cfg(rollout["qpos"][index])
        for side in SIDES:
            target = rollout[f"target_{side}"][index]
            achieved = rollout[f"achieved_{side}"][index]
            target_frames[side].position = tuple(target[:3])
            target_frames[side].wxyz = tuple(target[[6, 3, 4, 5]])
            achieved_markers[side].position = tuple(achieved[:3])
        error_text.value = (
            f"L {rollout['left_pos_error_m'][index] * 100:.1f} / "
            f"{rollout['left_rot_error_deg'][index]:.1f} | "
            f"R {rollout['right_pos_error_m'][index] * 100:.1f} / "
            f"{rollout['right_rot_error_deg'][index]:.1f}"
        )

    @server.on_client_connect
    def _set_camera(client: viser.ClientHandle) -> None:
        client.camera.position = (-1.4, 0.0, 0.9)
        client.camera.look_at = (0.2, 0.0, 0.35)

    url = f"http://localhost:{server.get_port()}"
    print(f"[openarm-pico] viewer: {url} (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    current = 0
    draw(current)
    interval = 1.0 / (float(rollout["fps"][0]) * speed)
    try:
        while True:
            if play.value:
                current = (current + 1) % len(rollout["qpos"])
                frame.value = current
            else:
                current = int(frame.value)
            draw(current)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[openarm-pico] stopped")


def main() -> None:
    args = build_parser().parse_args()
    episode = load_controller_episode(
        args.repo_id, args.revision, args.episode, args.tcp_calibration
    )
    print(f"[openarm-pico] controller->TCP: {args.tcp_calibration}")
    rollout = solve_episode(
        episode,
        home_pose=args.home_pose,
        stride=args.stride,
        max_frames=args.max_frames,
    )
    if not args.headless:
        show_viewer(
            rollout,
            port=args.port,
            speed=args.speed,
            open_browser=not args.no_browser,
        )


if __name__ == "__main__":
    main()
