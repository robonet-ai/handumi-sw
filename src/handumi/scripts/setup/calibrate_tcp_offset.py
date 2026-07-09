"""Calibrate controller -> physical HandUMI gripper TCP transforms.

The shared calibration file format is:

    configs/calibration/{device}_controller_tcp.yaml

where ``device`` is ``pico`` or ``meta``. The important transform is always:

    T_world_tcp = T_world_controller @ T_controller_tcp

PICO calibration uses recorded pose7 data from Parquet/CSV. Meta Quest pivot
calibration samples the live Quest stream, matching the previous
``handumi-calibrate-tcp-offset`` behavior.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from handumi.calibration.control_tcp import (
    DEFAULT_PARQUET,
    SIDES,
    SUPPORTED_DEVICES,
    calibration_path_for_device,
    existing_or_identity,
    load_controller_tcp_calibration,
    load_csv_poses,
    load_episode_poses,
    solve_orientation_offset,
    solve_pivot_offset,
    write_controller_tcp_calibration,
)
from handumi.devices.meta_quest import MetaQuestConfig, MetaQuestReceiver
from handumi.devices.transforms import unity_pose_to_handumi
from handumi.robots.utils import IDENTITY_POSE7, pose_inv, quat_normalize

COMMANDS = {"pivot", "orient", "inspect"}
DEFAULT_META_TRACKING_CONFIG = Path("configs/tracking_meta_quest.yaml")


def _device(args: argparse.Namespace) -> str:
    return args.device_local or args.device


def _output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output
    return calibration_path_for_device(_device(args))


def _load_input_poses(args: argparse.Namespace, side: str) -> np.ndarray:
    if args.csv is not None:
        return load_csv_poses(args.csv, side)
    if args.episode is None:
        raise SystemExit("Use --episode with --parquet, or pass --csv")
    return load_episode_poses(args.parquet, args.episode, side, column=args.column)


def _pose7_from_meta_stream(args: argparse.Namespace) -> np.ndarray:
    config = MetaQuestConfig.from_yaml(args.tracking_config)
    if args.quest_ip is not None:
        config = MetaQuestConfig(
            quest_ip=args.quest_ip,
            tcp_port=config.tcp_port,
            sync_port=config.sync_port,
            connect_retry_s=config.connect_retry_s,
        )

    receiver = MetaQuestReceiver(config)
    receiver.start()
    print(f"Connecting to Quest at {config.quest_ip}:{config.tcp_port} ...")
    print(f"\nPin the {args.side.upper()} gripper TIP on a fixed point, then keep it")
    print("pinned while rotating the device in all directions.")
    input("Press Enter to start sampling... ")

    poses: list[np.ndarray] = []
    deadline = time.monotonic() + args.duration_s
    period = 1.0 / args.rate_hz
    try:
        while time.monotonic() < deadline:
            frame = receiver.latest()
            if frame is not None:
                controller = getattr(frame, args.side)
                if controller.tracked and controller.valid:
                    pose = unity_pose_to_handumi(controller.position, controller.quaternion)
                    poses.append(
                        np.asarray(
                            [*pose.position, *quat_normalize(pose.quaternion)],
                            dtype=np.float32,
                        )
                    )
            remaining = deadline - time.monotonic()
            print(f"\r  sampling... {remaining:5.1f}s left, {len(poses)} samples", end="")
            time.sleep(period)
    finally:
        receiver.stop()
    print()

    if len(poses) < 50:
        raise SystemExit(f"Only {len(poses)} tracked samples; check trk=1 and retry.")
    return np.stack(poses, axis=0)


def _pose_from_dict(raw: dict | None) -> np.ndarray:
    pose = IDENTITY_POSE7.copy()
    raw = raw or {}
    if "position" in raw:
        pose[:3] = np.asarray(raw["position"], dtype=np.float32)
    if "quaternion" in raw:
        pose[3:] = quat_normalize(np.asarray(raw["quaternion"], dtype=np.float32))
    return pose


def _seed_from_meta_tracking_config(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text()) or {}
    mapping = (data.get("calibration") or {}).get("controller_to_gripper_tcp")
    if not mapping:
        return None
    return _pose_from_dict(mapping.get("left")), _pose_from_dict(mapping.get("right"))


def _existing_or_seeded(args: argparse.Namespace, output: Path) -> tuple[np.ndarray, np.ndarray]:
    if output.exists():
        return existing_or_identity(output)
    if _device(args) == "meta":
        seeded = _seed_from_meta_tracking_config(args.tracking_config)
        if seeded is not None:
            return seeded
    return existing_or_identity(output)


def _save_side_pose(args: argparse.Namespace, side_pose: np.ndarray, *, update_rotation: bool) -> Path:
    output = _output_path(args)
    left, right = _existing_or_seeded(args, output)
    target = left if args.side == "left" else right
    if update_rotation:
        target[3:] = quat_normalize(side_pose[3:])
    else:
        target[:3] = side_pose[:3]
    write_controller_tcp_calibration(output, left=left, right=right)
    return output


def _print_pivot_report(device: str, side: str, result, output: Path) -> None:
    print(f"[{device}-tcp] side={side} samples={result.num_samples}")
    print(
        f"[{device}-tcp] controller->TCP position (m):",
        np.array2string(result.position, precision=5, suppress_small=True),
    )
    print(
        f"[{device}-tcp] fixed TCP point in tracking world (m):",
        np.array2string(result.pivot_world, precision=5, suppress_small=True),
    )
    print(
        f"[{device}-tcp] residual rms={result.rms_error * 100:.2f}cm "
        f"max={result.max_error * 100:.2f}cm condition={result.condition:.1f}"
    )
    if result.rms_error > 0.02 or result.max_error > 0.04:
        print(f"[{device}-tcp] WARNING: high residual; the tip probably slipped.")
    if result.condition > 500:
        print(f"[{device}-tcp] WARNING: weak rotation diversity; rotate through more poses.")
    print(f"[{device}-tcp] wrote: {output}")


def pivot_main(args: argparse.Namespace) -> None:
    device = _device(args)
    poses = _pose7_from_meta_stream(args) if device == "meta" else _load_input_poses(args, args.side)
    result = solve_pivot_offset(poses)
    side_pose = IDENTITY_POSE7.copy()
    side_pose[:3] = result.position
    output = _save_side_pose(args, side_pose, update_rotation=False)
    _print_pivot_report(device, args.side, result, output)


def orient_main(args: argparse.Namespace) -> None:
    poses = _load_input_poses(args, args.side)
    quat = quat_normalize(np.asarray(args.tcp_quat_world, dtype=np.float32))
    offset_quat = solve_orientation_offset(poses, quat)
    side_pose = IDENTITY_POSE7.copy()
    side_pose[3:] = offset_quat
    output = _save_side_pose(args, side_pose, update_rotation=True)
    device = _device(args)
    print(f"[{device}-tcp] side={args.side} controller->TCP quaternion xyzw:")
    print("          ", np.array2string(offset_quat, precision=5, suppress_small=True))
    print(f"[{device}-tcp] wrote: {output}")


def inspect_main(args: argparse.Namespace) -> None:
    path = args.path or _output_path(args)
    calibration = load_controller_tcp_calibration(path)
    print(f"[tcp] loaded: {path}")
    for side, pose in (("left", calibration.left), ("right", calibration.right)):
        inv_pose = pose_inv(pose)
        print(f"  {side}:")
        print("    controller->tcp:", np.array2string(pose, precision=5, suppress_small=True))
        print("    tcp->controller:", np.array2string(inv_pose, precision=5, suppress_small=True))


def add_device_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=SUPPORTED_DEVICES, default=None, dest="device_local")


def add_common_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("-e", "--episode", type=int)
    parser.add_argument("--csv", type=Path, help="CSV with x,y,z,qx,qy,qz,qw and optional side")
    parser.add_argument("--column", help="Override parquet pose column")
    parser.add_argument("--side", choices=SIDES, required=True)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--tracking-config", type=Path, default=DEFAULT_META_TRACKING_CONFIG)
    parser.add_argument("--quest-ip", type=str, default=None)
    parser.add_argument("--duration-s", type=float, default=25.0)
    parser.add_argument("--rate-hz", type=float, default=30.0)
    add_device_arg(parser)


def _argv_with_default_command(argv: list[str]) -> list[str]:
    if not argv or "-h" in argv or "--help" in argv:
        return argv
    if any(arg in COMMANDS for arg in argv):
        return argv
    return ["pivot", *argv]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", choices=SUPPORTED_DEVICES, default="meta")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pivot = sub.add_parser(
        "pivot",
        help="Estimate controller->TCP translation by keeping the gripper tip fixed.",
    )
    add_common_input_args(pivot)
    pivot.set_defaults(func=pivot_main)

    orient = sub.add_parser(
        "orient",
        help="Estimate controller->TCP rotation from a known TCP world orientation.",
    )
    add_common_input_args(orient)
    orient.add_argument(
        "--tcp-quat-world",
        nargs=4,
        type=float,
        metavar=("QX", "QY", "QZ", "QW"),
        required=True,
        help="Desired TCP orientation in the same world frame as the recorded controller poses.",
    )
    orient.set_defaults(func=orient_main)

    inspect = sub.add_parser("inspect", help="Print a calibration YAML.")
    inspect.add_argument("path", type=Path, nargs="?")
    inspect.add_argument("-o", "--output", type=Path, default=None)
    add_device_arg(inspect)
    inspect.set_defaults(func=inspect_main)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args(_argv_with_default_command(sys.argv[1:]))
    args.func(args)


if __name__ == "__main__":
    main()
