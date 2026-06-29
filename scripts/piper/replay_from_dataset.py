#!/usr/bin/env python3
"""Replay a Piper LeRobot trajectory on one or two real Piper arms.

The Piper datasets produced by ``scripts/process_umi_to_lerobot.py`` store
physical robot units:

    [left j1..j6 radians, left gripper meters,
     right j1..j6 radians, right gripper meters]

The Piper SDK expects ``JointCtrl`` in 0.001 degrees and ``GripperCtrl`` in
0.001 mm, so this script performs that conversion immediately before sending
commands over CAN.

python scripts/piper/replay_from_dataset.py \
  --repo-id NONHUMAN-RESEARCH/handumi-dataset-v2-piper \
  --episode 5 \
  --yes
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from handumi.dataset import (  # noqa: E402
    dataset_root_from_repo_id,
    load_info,
    open_dataset,
)
from handumi.robots.piper.shared import (  # noqa: E402
    GRIPPER_STROKE_M,
    JOINT_LIMITS_RAD,
    LEROBOT_JOINT_NAMES,
    robot_arm_to_sdk_joint_ctrl,
    robot_gripper_to_sdk_ctrl,
)

try:
    from piper_sdk import C_PiperInterface_V2
except ImportError:  # pragma: no cover - depends on optional robot SDK
    C_PiperInterface_V2 = None


TrajectorySource = Literal["state-action", "state", "action"]

DEFAULT_REPO_ID = "NONHUMAN-RESEARCH/handumi-dataset-v2-piper"


@dataclass(frozen=True)
class DatasetInfo:
    fps: int
    chunks_size: int
    robot_type: str
    joint_names: list[str]
    total_episodes: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay a Piper IK-converted LeRobot dataset on real Piper arms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face repo id of the Piper LeRobot dataset.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Local dataset root. Defaults to outputs/datasets/<repo-id suffix>.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Dataset revision on the Hugging Face Hub.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to replay from the dataset.",
    )
    parser.add_argument(
        "--trajectory-source",
        choices=("state-action", "state", "action"),
        default="state-action",
        help=(
            "Which parquet vectors to replay. state-action reconstructs the "
            "IK sequence as observation.state[0] followed by every action."
        ),
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Maximum number of trajectory frames to replay after --start-index.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Override dataset FPS.")
    parser.add_argument(
        "--rate-scale",
        type=float,
        default=1.0,
        help="Playback speed multiplier. 0.5 is half speed, 2.0 is double speed.",
    )
    parser.add_argument(
        "--arms",
        choices=("both", "left", "right"),
        default="both",
        help="Select which physical arm(s) receive commands.",
    )
    parser.add_argument(
        "--left-port",
        default=os.environ.get("LEFT_ROBOT_ARM_PORT", "can0"),
        help="CAN port for the left Piper arm.",
    )
    parser.add_argument(
        "--right-port",
        default=os.environ.get("RIGHT_ROBOT_ARM_PORT", "can1"),
        help="CAN port for the right Piper arm.",
    )
    parser.add_argument(
        "--speed-percent",
        type=int,
        default=30,
        help="Piper MotionCtrl_2 speed percentage.",
    )
    parser.add_argument(
        "--gripper-effort",
        type=int,
        default=1000,
        help="Piper GripperCtrl effort in 0.001 N/m units.",
    )
    parser.add_argument(
        "--no-gripper",
        action="store_true",
        help="Send only joint commands and leave grippers untouched.",
    )
    parser.add_argument(
        "--ramp-duration",
        type=float,
        default=3.0,
        help="Seconds used to interpolate from current feedback to the first target.",
    )
    parser.add_argument(
        "--max-step-rad",
        type=float,
        default=0.50,
        help="Reject consecutive joint jumps larger than this unless allowed.",
    )
    parser.add_argument(
        "--allow-large-steps",
        action="store_true",
        help="Do not reject large consecutive trajectory jumps.",
    )
    parser.add_argument(
        "--allow-clipping",
        action="store_true",
        help="Clip out-of-range joints/gripper instead of failing validation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load, validate, and print SDK conversions without connecting to CAN.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive REPLAY confirmation for real robot execution.",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=8.0,
        help="Seconds to wait for EnablePiper() during connection.",
    )
    return parser


def load_dataset_info(root: Path) -> DatasetInfo:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing dataset metadata: {info_path}")

    raw = load_info(root)

    feature = raw.get("features", {}).get("observation.state")
    if not feature:
        raise ValueError("Dataset has no observation.state feature in meta/info.json.")

    names = list(feature.get("names") or [])
    return DatasetInfo(
        fps=int(raw.get("fps", 30)),
        chunks_size=int(raw.get("chunks_size", 1000)),
        robot_type=str(raw.get("robot_type", "")),
        joint_names=names,
        total_episodes=int(raw.get("total_episodes", 0)),
    )


def validate_dataset_info(info: DatasetInfo) -> None:
    if info.robot_type != "bi_piper_follower":
        raise ValueError(
            f"Expected robot_type='bi_piper_follower', got {info.robot_type!r}."
        )
    if info.joint_names != list(LEROBOT_JOINT_NAMES):
        raise ValueError(
            "Dataset joint names do not match the Piper replay layout.\n"
            f"Expected: {LEROBOT_JOINT_NAMES}\n"
            f"Got     : {info.joint_names}"
        )


def load_episode_vectors(
    root: Path,
    *,
    repo_id: str,
    episode: int,
    source: TrajectorySource,
    revision: str,
) -> tuple[np.ndarray, int]:
    dataset = open_dataset(
        repo_id=repo_id,
        root=root,
        episode=episode,
        revision=revision,
    )

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for idx in range(len(dataset)):
        item = dataset.get_raw_item(idx)
        if "observation.state" not in item or "action" not in item:
            raise ValueError(
                "Dataset episode must contain observation.state and action columns."
            )
        states.append(np.asarray(item["observation.state"], dtype=np.float32))
        actions.append(np.asarray(item["action"], dtype=np.float32))

    if not states:
        raise ValueError(f"No frames found for episode {episode}.")

    states_arr = np.stack(states, axis=0)
    actions_arr = np.stack(actions, axis=0)

    if source == "state-action":
        trajectory = np.vstack([states_arr[:1], actions_arr])
    elif source == "state":
        trajectory = states_arr
    elif source == "action":
        trajectory = actions_arr
    else:
        raise ValueError(f"Unsupported trajectory source: {source!r}")

    if trajectory.ndim != 2 or trajectory.shape[1] != 14:
        raise ValueError(f"Expected trajectory shape (T, 14), got {trajectory.shape}.")
    return trajectory, int(dataset.fps)


def crop_trajectory(trajectory: np.ndarray, start_index: int, frames: int | None) -> np.ndarray:
    if start_index < 0:
        raise ValueError("--start-index must be >= 0.")
    if start_index >= len(trajectory):
        raise ValueError(
            f"--start-index {start_index} is outside trajectory length {len(trajectory)}."
        )
    end = None if frames is None else start_index + max(0, frames)
    cropped = trajectory[start_index:end]
    if len(cropped) == 0:
        raise ValueError("Selected trajectory slice has zero frames.")
    return cropped


def _arm_joint_slice(side: Literal["left", "right"]) -> slice:
    return slice(0, 6) if side == "left" else slice(7, 13)


def _arm_gripper_index(side: Literal["left", "right"]) -> int:
    return 6 if side == "left" else 13


def validate_and_prepare_trajectory(
    trajectory: np.ndarray,
    *,
    allow_clipping: bool,
    max_step_rad: float,
    allow_large_steps: bool,
) -> np.ndarray:
    if not np.isfinite(trajectory).all():
        raise ValueError("Trajectory contains NaN or infinite values.")

    prepared = trajectory.astype(np.float32, copy=True)
    for side in ("left", "right"):
        joint_slice = _arm_joint_slice(side)
        q = prepared[:, joint_slice]
        low = JOINT_LIMITS_RAD[:, 0]
        high = JOINT_LIMITS_RAD[:, 1]
        below = q < (low - 1e-3)
        above = q > (high + 1e-3)
        if (below | above).any():
            bad_count = int((below | above).sum())
            msg = (
                f"{side} arm has {bad_count} joint values outside Piper limits "
                f"{JOINT_LIMITS_RAD.tolist()}."
            )
            if not allow_clipping:
                raise ValueError(msg + " Re-run with --allow-clipping to clamp them.")
            print(f"WARNING: {msg} Clipping to Piper limits.")
            prepared[:, joint_slice] = np.clip(q, low, high)

        gripper_idx = _arm_gripper_index(side)
        grip = prepared[:, gripper_idx]
        out = (grip < -1e-4) | (grip > GRIPPER_STROKE_M + 1e-4)
        if out.any():
            msg = (
                f"{side} gripper has {int(out.sum())} values outside "
                f"[0, {GRIPPER_STROKE_M:.3f}] meters."
            )
            if not allow_clipping:
                raise ValueError(msg + " Re-run with --allow-clipping to clamp them.")
            print(f"WARNING: {msg} Clipping to gripper stroke.")
            prepared[:, gripper_idx] = np.clip(grip, 0.0, GRIPPER_STROKE_M)

    if len(prepared) > 1 and not allow_large_steps:
        left_steps = np.abs(np.diff(prepared[:, 0:6], axis=0))
        right_steps = np.abs(np.diff(prepared[:, 7:13], axis=0))
        max_step = float(max(left_steps.max(initial=0.0), right_steps.max(initial=0.0)))
        if max_step > max_step_rad:
            raise ValueError(
                f"Trajectory has a max consecutive joint jump of {max_step:.3f} rad, "
                f"greater than --max-step-rad {max_step_rad:.3f}. Use "
                "--allow-large-steps only after inspecting the dataset."
            )

    return prepared


class PiperArm:
    def __init__(
        self,
        *,
        name: Literal["left", "right"],
        port: str,
        speed_percent: int,
        gripper_effort: int,
        no_gripper: bool,
        dry_run: bool,
        connect_timeout: float,
    ) -> None:
        self.name = name
        self.port = port
        self.speed_percent = int(np.clip(speed_percent, 0, 100))
        self.gripper_effort = int(gripper_effort)
        self.no_gripper = no_gripper
        self.dry_run = dry_run
        self.connect_timeout = connect_timeout
        self.piper: Any | None = None

    def connect(self) -> None:
        if self.dry_run:
            print(f"[{self.name}] dry-run: not connecting to {self.port}")
            return
        if C_PiperInterface_V2 is None:
            raise ImportError(
                "piper_sdk is not installed. Install the optional Piper dependencies."
            )

        print(f"[{self.name}] connecting on {self.port}")
        self.piper = C_PiperInterface_V2(self.port)
        self.piper.ConnectPort()
        time.sleep(0.2)

        status = self.piper.GetArmStatus().arm_status
        print(
            f"[{self.name}] status: motion_status={status.motion_status} "
            f"ctrl_mode={status.ctrl_mode}"
        )
        if status.motion_status != 0:
            print(f"[{self.name}] resuming from non-idle motion status")
            self.piper.EmergencyStop(0x02)
            time.sleep(0.2)
        if status.ctrl_mode == 2:
            print(f"[{self.name}] teaching mode detected; sending resume command")
            self.piper.EmergencyStop(0x02)
            time.sleep(0.2)

        deadline = time.monotonic() + self.connect_timeout
        while not self.piper.EnablePiper():
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"[{self.name}] EnablePiper() timed out on {self.port}."
                )
            time.sleep(0.01)

        self.piper.MotionCtrl_2(0x01, 0x01, self.speed_percent, 0x00)
        print(f"[{self.name}] enabled in joint mode at {self.speed_percent}% speed")

    def current_state(self) -> np.ndarray:
        if self.dry_run or self.piper is None:
            return np.zeros(7, dtype=np.float32)

        joint_state = self.piper.GetArmJointMsgs().joint_state
        joints_sdk = np.array(
            [
                joint_state.joint_1,
                joint_state.joint_2,
                joint_state.joint_3,
                joint_state.joint_4,
                joint_state.joint_5,
                joint_state.joint_6,
            ],
            dtype=np.float32,
        )
        joints_rad = np.deg2rad(joints_sdk / 1000.0)
        gripper_sdk = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle
        gripper_m = float(gripper_sdk) / 1_000_000.0
        return np.concatenate([joints_rad, np.array([gripper_m], dtype=np.float32)])

    def command(self, arm_state: np.ndarray) -> None:
        joints = robot_arm_to_sdk_joint_ctrl(arm_state[:6])
        gripper = robot_gripper_to_sdk_ctrl(float(arm_state[6]))

        if self.dry_run:
            return
        assert self.piper is not None
        self.piper.JointCtrl(*joints)
        if not self.no_gripper:
            self.piper.GripperCtrl(gripper, self.gripper_effort, 0x01, 0)

    def print_conversion_sample(self, arm_state: np.ndarray) -> None:
        joints = robot_arm_to_sdk_joint_ctrl(arm_state[:6])
        gripper = robot_gripper_to_sdk_ctrl(float(arm_state[6]))
        print(
            f"[{self.name}] sample radians/m -> SDK: "
            f"{np.round(arm_state, 4).tolist()} -> joints={joints}, gripper={gripper}"
        )

    def disconnect(self) -> None:
        if self.dry_run or self.piper is None:
            return
        try:
            self.piper.DisconnectPort()
            print(f"[{self.name}] disconnected")
        except Exception as exc:  # pragma: no cover - best effort robot cleanup
            print(f"[{self.name}] warning: disconnect failed: {exc}", file=sys.stderr)


def selected_sides(arms: str) -> list[Literal["left", "right"]]:
    if arms == "both":
        return ["left", "right"]
    if arms == "left":
        return ["left"]
    if arms == "right":
        return ["right"]
    raise ValueError(f"Unsupported arms selection: {arms!r}")


def arm_state_from_frame(frame: np.ndarray, side: Literal["left", "right"]) -> np.ndarray:
    if side == "left":
        return frame[:7]
    return frame[7:14]


def confirm_real_replay(args: argparse.Namespace, frame_count: int, fps: float) -> None:
    if args.dry_run or args.yes:
        return
    print()
    print("About to command real Piper hardware.")
    print(f"  Arms    : {args.arms}")
    print(f"  Ports   : left={args.left_port} right={args.right_port}")
    print(f"  Frames  : {frame_count}")
    print(f"  Rate    : {fps:.2f} Hz")
    print(f"  Dataset : {args.dataset_root}")
    answer = input("Type REPLAY to continue: ").strip()
    if answer != "REPLAY":
        raise RuntimeError("Replay cancelled by user.")


def ramp_to_first_frame(
    arms: dict[str, PiperArm],
    first_frame: np.ndarray,
    *,
    fps: float,
    duration: float,
) -> None:
    if duration <= 0:
        for side, arm in arms.items():
            arm.command(arm_state_from_frame(first_frame, side))  # type: ignore[arg-type]
        return

    steps = max(1, int(round(duration * fps)))
    current = {
        side: arm.current_state()
        for side, arm in arms.items()
    }
    targets = {
        side: arm_state_from_frame(first_frame, side)  # type: ignore[arg-type]
        for side in arms
    }

    print(f"Ramping to first frame over {duration:.2f}s ({steps} steps)")
    period = 1.0 / fps
    for i in range(1, steps + 1):
        alpha = i / steps
        start_time = time.monotonic()
        for side, arm in arms.items():
            state = (1.0 - alpha) * current[side] + alpha * targets[side]
            arm.command(state)
        elapsed = time.monotonic() - start_time
        time.sleep(max(0.0, period - elapsed))


def replay_trajectory(
    arms: dict[str, PiperArm],
    trajectory: np.ndarray,
    *,
    fps: float,
) -> None:
    period = 1.0 / fps
    next_tick = time.monotonic()
    last_print = 0.0

    print(f"Replaying {len(trajectory)} frames at {fps:.2f} Hz")
    for frame_idx, frame in enumerate(trajectory):
        for side, arm in arms.items():
            arm.command(arm_state_from_frame(frame, side))  # type: ignore[arg-type]

        now = time.monotonic()
        if now - last_print >= 1.0 or frame_idx == len(trajectory) - 1:
            print(f"  frame {frame_idx + 1}/{len(trajectory)}", end="\r", flush=True)
            last_print = now

        next_tick += period
        time.sleep(max(0.0, next_tick - time.monotonic()))
    print()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_root = (
        Path(args.dataset_root).expanduser().resolve()
        if args.dataset_root is not None
        else dataset_root_from_repo_id(args.repo_id).resolve()
    )
    info = load_dataset_info(dataset_root)
    validate_dataset_info(info)
    if args.episode < 0 or args.episode >= info.total_episodes:
        parser.error(
            f"--episode {args.episode} is outside dataset range "
            f"[0, {info.total_episodes})."
        )

    trajectory, dataset_fps = load_episode_vectors(
        dataset_root,
        repo_id=args.repo_id,
        episode=args.episode,
        source=args.trajectory_source,
        revision=args.revision,
    )

    base_fps = float(args.fps if args.fps is not None else dataset_fps)
    if base_fps <= 0:
        parser.error("FPS must be positive.")
    if args.rate_scale <= 0:
        parser.error("--rate-scale must be positive.")
    replay_fps = base_fps * float(args.rate_scale)
    trajectory = crop_trajectory(trajectory, args.start_index, args.frames)
    trajectory = validate_and_prepare_trajectory(
        trajectory,
        allow_clipping=args.allow_clipping,
        max_step_rad=args.max_step_rad,
        allow_large_steps=args.allow_large_steps,
    )

    print(f"Dataset       : {dataset_root} ({args.repo_id})")
    print(f"Episode       : {args.episode}")
    print(f"Trajectory    : {trajectory.shape[0]} frames x {trajectory.shape[1]} values")
    print(f"Units         : joints=radians, gripper=meters")
    print(f"Playback      : source FPS {base_fps:.2f}, replay FPS {replay_fps:.2f}")
    print(f"Joint min/max : {trajectory[:, [0,1,2,3,4,5,7,8,9,10,11,12]].min():.3f} / "
          f"{trajectory[:, [0,1,2,3,4,5,7,8,9,10,11,12]].max():.3f} rad")
    print(f"Gripper range : {trajectory[:, [6, 13]].min():.4f} / "
          f"{trajectory[:, [6, 13]].max():.4f} m")

    confirm_real_replay(args, len(trajectory), replay_fps)

    sides = selected_sides(args.arms)
    arm_by_side: dict[str, PiperArm] = {}
    for side in sides:
        port = args.left_port if side == "left" else args.right_port
        arm_by_side[side] = PiperArm(
            name=side,
            port=port,
            speed_percent=args.speed_percent,
            gripper_effort=args.gripper_effort,
            no_gripper=args.no_gripper,
            dry_run=args.dry_run,
            connect_timeout=args.connect_timeout,
        )
        arm_by_side[side].print_conversion_sample(arm_state_from_frame(trajectory[0], side))

    try:
        for arm in arm_by_side.values():
            arm.connect()
        ramp_to_first_frame(
            arm_by_side,
            trajectory[0],
            fps=replay_fps,
            duration=args.ramp_duration,
        )
        replay_trajectory(arm_by_side, trajectory, fps=replay_fps)
    except KeyboardInterrupt:
        print("\nReplay interrupted by user.")
    finally:
        for arm in arm_by_side.values():
            arm.disconnect()


if __name__ == "__main__":
    main()
