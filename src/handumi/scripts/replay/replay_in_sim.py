#!/usr/bin/env python3
"""Replay a HandUMI LeRobot episode in simulation with YAML robot configs."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from handumi.calibration.control_tcp import (
    DEFAULT_CALIBRATION as DEFAULT_CONTROLLER_TCP_CALIBRATION,
    apply_controller_tcp_calibration,
    load_controller_tcp_calibration,
)
from handumi.dataset.raw import LEFT_POSE_SLICE, RIGHT_POSE_SLICE
from handumi.dataset.reader import open_dataset
from handumi.dataset.ref import DatasetRef
from handumi.retargeting.handumi_to_robot import (
    local_frame_adapter,
    local_relative_robot_target_pose7,
    raw_state_pose7_pair,
    raw_state_robot_target_pose7,
    retarget_anchors_from_raw_state,
)
from handumi.robots.kinematics import optimization_score_from_errors, pose_error_arrays
from handumi.robots.registry import EMBODIMENT_NAMES, load_embodiment

DEFAULT_REPO_ID = "NONHUMAN-RESEARCH/handumi-dataset-v2"
DEFAULT_OUT_DIR = Path("outputs/replay_in_sim")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay a raw HandUMI LeRobot episode through bimanual IK."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Local dataset root. Defaults to outputs/datasets/<repo-id suffix>.",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("-e", "--episode", type=int, default=0)
    parser.add_argument("--robot", choices=EMBODIMENT_NAMES, default="piper")
    parser.add_argument(
        "--source",
        choices=("observation.state", "action"),
        default="observation.state",
        help="Raw 16D LeRobot feature to replay.",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--retarget-mode",
        choices=("local-relative", "anchored"),
        default="local-relative",
        help=(
            "local-relative replays frame-to-frame TCP SE(3) deltas in robot EE "
            "space. anchored preserves the older home + absolute position-delta mode."
        ),
    )
    parser.add_argument(
        "--compose-source",
        choices=("commanded", "achieved"),
        default="commanded",
        help=(
            "For --retarget-mode local-relative, compose each adapted delta on "
            "the previous commanded target or previous achieved FK pose."
        ),
    )
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=1.0,
        help="Scale local-relative translation deltas after frame adaptation.",
    )
    parser.add_argument(
        "--controller-tcp-calibration",
        type=Path,
        default=DEFAULT_CONTROLLER_TCP_CALIBRATION,
        help=(
            "YAML with PICO controller->HandUMI TCP transforms. "
            "Applied by default before retargeting."
        ),
    )
    parser.add_argument(
        "--raw-controller-debug",
        action="store_true",
        help="Replay raw PICO controller poses without controller->TCP calibration.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("-o", "--output", type=Path, default=None)
    return parser


def load_episode_states(args: argparse.Namespace) -> tuple[np.ndarray, float]:
    ref = DatasetRef.from_repo_id(
        args.repo_id,
        root=args.dataset_root,
        revision=args.revision,
    )
    dataset = open_dataset(ref, episode=args.episode)
    fps = float(getattr(dataset, "fps", 30) or 30)
    states: list[np.ndarray] = []
    for item in dataset:
        if args.source not in item:
            raise ValueError(f"Dataset item has no {args.source!r} feature.")
        states.append(np.asarray(item[args.source], dtype=np.float32))
    if not states:
        raise ValueError(f"Episode {args.episode} is empty.")
    return np.stack(states, axis=0), fps


def apply_tcp_calibration_to_states(
    states: np.ndarray,
    calibration_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Path | None,
]:
    """Return states whose left/right poses are calibrated gripper TCP poses."""
    raw_left: list[np.ndarray] = []
    raw_right: list[np.ndarray] = []
    for state in states:
        left, right = raw_state_pose7_pair(state)
        raw_left.append(left)
        raw_right.append(right)

    raw_left_arr = np.asarray(raw_left, dtype=np.float32)
    raw_right_arr = np.asarray(raw_right, dtype=np.float32)
    calibration = load_controller_tcp_calibration(calibration_path)
    left_tcp, right_tcp = apply_controller_tcp_calibration(
        raw_left_arr,
        raw_right_arr,
        calibration,
    )

    calibrated = np.asarray(states, dtype=np.float32).copy()
    calibrated[:, LEFT_POSE_SLICE] = left_tcp
    calibrated[:, RIGHT_POSE_SLICE] = right_tcp
    return (
        calibrated,
        raw_left_arr,
        raw_right_arr,
        left_tcp,
        right_tcp,
        calibration.source,
    )


def solve_episode(args: argparse.Namespace) -> dict[str, np.ndarray]:
    runtime = load_embodiment(args.robot)
    states, fps = load_episode_states(args)
    if args.raw_controller_debug:
        states_for_retarget = states
        raw_left_controller: list[np.ndarray] = []
        raw_right_controller: list[np.ndarray] = []
        for state in states:
            left, right = raw_state_pose7_pair(state)
            raw_left_controller.append(left)
            raw_right_controller.append(right)
        raw_left_controller_arr = np.asarray(raw_left_controller, dtype=np.float32)
        raw_right_controller_arr = np.asarray(raw_right_controller, dtype=np.float32)
        left_tcp_arr = raw_left_controller_arr
        right_tcp_arr = raw_right_controller_arr
        calibration_source = None
    else:
        (
            states_for_retarget,
            raw_left_controller_arr,
            raw_right_controller_arr,
            left_tcp_arr,
            right_tcp_arr,
            calibration_source,
        ) = apply_tcp_calibration_to_states(states, args.controller_tcp_calibration)

    frame_indices = list(range(args.start_frame, len(states), args.stride))
    if args.max_frames is not None:
        frame_indices = frame_indices[: args.max_frames]
    if not frame_indices:
        raise ValueError("No frames selected for replay.")

    cfg = runtime.config.ik_weights
    q = runtime.config.home_q.astype(np.float32).copy()
    solver = runtime.solver_cls()
    qs: list[np.ndarray] = []
    raw_left_gt: list[np.ndarray] = []
    raw_right_gt: list[np.ndarray] = []
    left_targets: list[np.ndarray] = []
    right_targets: list[np.ndarray] = []
    left_achieved: list[np.ndarray] = []
    right_achieved: list[np.ndarray] = []
    home_left_pose7, home_right_pose7 = solver.fk_pose7(q)
    first_left_pose7, first_right_pose7 = raw_state_pose7_pair(
        states_for_retarget[frame_indices[0]]
    )
    anchors = None
    left_adapter = None
    right_adapter = None
    if args.retarget_mode == "anchored":
        anchors = retarget_anchors_from_raw_state(
            states_for_retarget[frame_indices[0]],
            left_robot_pose7=home_left_pose7,
            right_robot_pose7=home_right_pose7,
            max_reach=cfg.max_reach,
        )
    else:
        left_adapter = local_frame_adapter(first_left_pose7, home_left_pose7)
        right_adapter = local_frame_adapter(first_right_pose7, home_right_pose7)

    start = time.perf_counter()
    for selected_index, frame_index in enumerate(frame_indices):
        state = states_for_retarget[frame_index]
        raw_left, raw_right = raw_state_pose7_pair(state)
        if args.retarget_mode == "anchored":
            if anchors is None:
                raise RuntimeError("Anchored retarget mode was not initialized.")
            left_pose7, right_pose7 = raw_state_robot_target_pose7(state, anchors)
            q = solver.ik(
                q,
                left_pose=(left_pose7[:3], left_pose7[3:7]),
                right_pose=(right_pose7[:3], right_pose7[3:7]),
            )
            fk_left_pose7, fk_right_pose7 = solver.fk_pose7(q)
        elif selected_index == 0:
            left_pose7 = home_left_pose7.copy()
            right_pose7 = home_right_pose7.copy()
            fk_left_pose7 = home_left_pose7.copy()
            fk_right_pose7 = home_right_pose7.copy()
        else:
            if left_adapter is None or right_adapter is None:
                raise RuntimeError("Local-relative retarget mode was not initialized.")
            prev_state = states_for_retarget[frame_indices[selected_index - 1]]
            prev_left, prev_right = raw_state_pose7_pair(prev_state)
            base_left = (
                left_targets[-1]
                if args.compose_source == "commanded"
                else left_achieved[-1]
            )
            base_right = (
                right_targets[-1]
                if args.compose_source == "commanded"
                else right_achieved[-1]
            )
            left_pose7 = local_relative_robot_target_pose7(
                previous_source_pose7=prev_left,
                current_source_pose7=raw_left,
                base_robot_pose7=base_left,
                adapter_rot=left_adapter,
                home_robot_pose7=home_left_pose7,
                translation_scale=args.translation_scale,
                max_reach=cfg.max_reach,
            )
            right_pose7 = local_relative_robot_target_pose7(
                previous_source_pose7=prev_right,
                current_source_pose7=raw_right,
                base_robot_pose7=base_right,
                adapter_rot=right_adapter,
                home_robot_pose7=home_right_pose7,
                translation_scale=args.translation_scale,
                max_reach=cfg.max_reach,
            )
            q = solver.ik(
                q,
                left_pose=(left_pose7[:3], left_pose7[3:7]),
                right_pose=(right_pose7[:3], right_pose7[3:7]),
            )
            fk_left_pose7, fk_right_pose7 = solver.fk_pose7(q)
        qs.append(q.copy())
        raw_left_gt.append(raw_left)
        raw_right_gt.append(raw_right)
        left_targets.append(left_pose7)
        right_targets.append(right_pose7)
        left_achieved.append(fk_left_pose7)
        right_achieved.append(fk_right_pose7)
    elapsed = time.perf_counter() - start
    target_left = np.asarray(left_targets, dtype=np.float32)
    target_right = np.asarray(right_targets, dtype=np.float32)
    achieved_left = np.asarray(left_achieved, dtype=np.float32)
    achieved_right = np.asarray(right_achieved, dtype=np.float32)
    errors = pose_error_arrays(target_left, target_right, achieved_left, achieved_right)
    all_pos_err = np.concatenate(
        [errors["left_pos_error_m"], errors["right_pos_error_m"]]
    )
    all_rot_err = np.concatenate(
        [errors["left_rot_error_deg"], errors["right_rot_error_deg"]]
    )
    score = optimization_score_from_errors(
        float(all_pos_err.mean() * 100.0),
        float(all_pos_err.max() * 100.0),
        float(all_rot_err.mean()),
        float(all_rot_err.max()),
    )

    print(
        f"[replay] robot={args.robot} episode={args.episode} frames={len(qs)} "
        f"fps={fps:g} solved={elapsed:.2f}s ({elapsed / len(qs) * 1000:.1f} ms/frame)"
    )
    print(
        f"[replay] retarget={args.retarget_mode} "
        f"compose={args.compose_source} translation_scale={args.translation_scale:g}"
    )
    if calibration_source is None:
        print("[replay] input pose: raw controller DEBUG mode")
    else:
        print(f"[replay] input pose: calibrated HandUMI TCP via {calibration_source}")
    print(
        "[replay] IK EE error: "
        f"pos mean={all_pos_err.mean() * 100:.2f}cm "
        f"max={all_pos_err.max() * 100:.2f}cm; "
        f"rot mean={all_rot_err.mean():.2f}deg max={all_rot_err.max():.2f}deg; "
        f"score={score:.4f}"
    )
    return {
        "qpos": np.asarray(qs, dtype=np.float32),
        "raw_left_pose7_ground_truth": np.asarray(raw_left_gt, dtype=np.float32),
        "raw_right_pose7_ground_truth": np.asarray(raw_right_gt, dtype=np.float32),
        "raw_left_controller_pose7": raw_left_controller_arr[frame_indices],
        "raw_right_controller_pose7": raw_right_controller_arr[frame_indices],
        "calibrated_left_tcp_pose7": left_tcp_arr[frame_indices],
        "calibrated_right_tcp_pose7": right_tcp_arr[frame_indices],
        "target_left_pose7_robot_world": target_left,
        "target_right_pose7_robot_world": target_right,
        "achieved_left_pose7_robot_world": achieved_left,
        "achieved_right_pose7_robot_world": achieved_right,
        "left_pos_error_m": errors["left_pos_error_m"],
        "right_pos_error_m": errors["right_pos_error_m"],
        "left_rot_error_deg": errors["left_rot_error_deg"],
        "right_rot_error_deg": errors["right_rot_error_deg"],
        "optimization_score": np.asarray([score], dtype=np.float32),
        "ik_weights": np.asarray(
            [cfg.pos_weight, cfg.ori_weight, cfg.rest_weight], dtype=np.float32
        ),
        "home_left_pose7_robot_world": home_left_pose7[None],
        "home_right_pose7_robot_world": home_right_pose7[None],
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "fps": np.asarray([fps], dtype=np.float32),
        "controller_tcp_calibration": np.asarray([str(calibration_source or "")]),
        "raw_controller_debug": np.asarray([args.raw_controller_debug], dtype=np.bool_),
        "retarget_mode": np.asarray([args.retarget_mode]),
        "compose_source": np.asarray([args.compose_source]),
        "translation_scale": np.asarray([args.translation_scale], dtype=np.float32),
    }


def save_rollout(args: argparse.Namespace, rollout: dict[str, np.ndarray]) -> Path:
    output = args.output
    if output is None:
        output = DEFAULT_OUT_DIR / f"episode_{args.episode:06d}_{args.robot}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        repo_id=np.asarray([args.repo_id]),
        robot=np.asarray([args.robot]),
        episode=np.asarray([args.episode], dtype=np.int64),
        **rollout,
    )
    print(f"[replay] saved: {output}")
    return output


def show_viewer(args: argparse.Namespace, rollout: dict[str, np.ndarray]) -> None:
    import viser
    import yourdfpy
    from viser.extras import ViserUrdf

    runtime = load_embodiment(args.robot)
    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/grid", width=3.0, height=3.0, cell_size=0.1)
    urdf = yourdfpy.URDF.load(
        str(runtime.urdf_path),
        mesh_dir=str(runtime.urdf_path.parent),
        load_meshes=True,
    )
    robot_view = ViserUrdf(server, urdf, root_node_name="/robot")
    server.scene.add_spline_catmull_rom(
        "/traj/target_left",
        positions=rollout["target_left_pose7_robot_world"][:, :3],
        color=(255, 190, 50),
        line_width=2.0,
    )
    server.scene.add_spline_catmull_rom(
        "/traj/target_right",
        positions=rollout["target_right_pose7_robot_world"][:, :3],
        color=(80, 220, 130),
        line_width=2.0,
    )
    server.scene.add_spline_catmull_rom(
        "/traj/achieved_left",
        positions=rollout["achieved_left_pose7_robot_world"][:, :3],
        color=(80, 160, 255),
        line_width=2.0,
    )
    server.scene.add_spline_catmull_rom(
        "/traj/achieved_right",
        positions=rollout["achieved_right_pose7_robot_world"][:, :3],
        color=(255, 90, 90),
        line_width=2.0,
    )
    target_left = server.scene.add_icosphere(
        "/target/left", radius=0.018, color=(255, 190, 50)
    )
    target_right = server.scene.add_icosphere(
        "/target/right", radius=0.018, color=(80, 220, 130)
    )
    achieved_left = server.scene.add_icosphere(
        "/achieved/left", radius=0.014, color=(80, 160, 255)
    )
    achieved_right = server.scene.add_icosphere(
        "/achieved/right", radius=0.014, color=(255, 90, 90)
    )
    play = server.gui.add_checkbox("Play", True)
    frame = server.gui.add_slider("Frame", 0, len(rollout["qpos"]) - 1, 1, 0)
    err_text = server.gui.add_text("EE error (cm/deg)", "-", disabled=True)

    def draw(i: int) -> None:
        robot_view.update_cfg(rollout["qpos"][i])
        target_left.position = tuple(rollout["target_left_pose7_robot_world"][i, :3])
        target_right.position = tuple(rollout["target_right_pose7_robot_world"][i, :3])
        achieved_left.position = tuple(
            rollout["achieved_left_pose7_robot_world"][i, :3]
        )
        achieved_right.position = tuple(
            rollout["achieved_right_pose7_robot_world"][i, :3]
        )
        err_text.value = (
            f"L={rollout['left_pos_error_m'][i] * 100:.1f}cm/"
            f"{rollout['left_rot_error_deg'][i]:.1f}deg "
            f"R={rollout['right_pos_error_m'][i] * 100:.1f}cm/"
            f"{rollout['right_rot_error_deg'][i]:.1f}deg"
        )

    draw(0)
    print(f"[replay] viewer: http://localhost:{server.get_port()}")
    current = 0
    while True:
        if play.value:
            current = (current + 1) % len(rollout["qpos"])
            frame.value = current
        else:
            current = int(frame.value)
        draw(current)
        time.sleep(1.0 / 30.0)


def main() -> None:
    args = build_parser().parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1.")
    rollout = solve_episode(args)
    save_rollout(args, rollout)
    if not args.headless:
        show_viewer(args, rollout)


if __name__ == "__main__":
    main()
