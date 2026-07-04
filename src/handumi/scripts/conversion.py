#!/usr/bin/env python3
"""Convert a PICO/HandUMI LeRobot dataset to robot-specific joint angles.

The script loads a source LeRobot dataset that contains PICO body-joint poses
(``observation.pico.body_joints_pose``), runs inverse kinematics for the chosen
embodiment, and writes a new LeRobot v3.0 dataset whose ``observation.state``
and ``action`` columns contain physical robot joint values. For Piper this is
``[j1..j6 radians, gripper meters]`` per arm; SDK integer command units are
derived later at replay/control time.

All video streams are preserved unchanged (the last frame of each episode is
dropped from the parquet data to produce ``action = state[t+1]``, but the
video files themselves are copied as-is).

Usage
-----
::

    # Axol embodiment (output defaults to NONHUMAN-RESEARCH/handumi-dataset-v2-axol)
    handumi-convert \
        --repo-id NONHUMAN-RESEARCH/handumi-dataset-v2 \
        --embodiment axol

    # Piper embodiment, custom output repo-id, push to hub afterwards
    handumi-convert \
        --repo-id NONHUMAN-RESEARCH/handumi-dataset-v2 \
        --embodiment piper \
        --output-repo-id NONHUMAN-RESEARCH/my-piper-dataset \
        --push-to-hub
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

import numpy as np

from handumi.dataset.ref import dataset_root_from_repo_id

load_dotenv()

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a PICO/HandUMI LeRobot dataset to an embodiment-specific "
            "joint-angle dataset via IK retargeting."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------------
    # Source / output datasets
    # ------------------------------------------------------------------
    ds = parser.add_argument_group("Dataset I/O")
    ds.add_argument(
        "--repo-id",
        default="NONHUMAN-RESEARCH/handumi-dataset-v2",
        help="HuggingFace repo-id of the source HandUMI dataset.",
    )
    ds.add_argument(
        "--root",
        default=None,
        help=(
            "Local root of the source dataset. "
            "Defaults to outputs/datasets/<repo-name>."
        ),
    )
    ds.add_argument(
        "--output-repo-id",
        default=None,
        help=(
            "HuggingFace repo-id for the converted dataset. "
            "Defaults to <source-repo-id>-<embodiment> "
            "(e.g. NONHUMAN-RESEARCH/handumi-dataset-v2-piper). "
            "The local output directory is always outputs/datasets/<output-repo-name>."
        ),
    )
    ds.add_argument(
        "--revision",
        default="main",
        help="Git revision of the source dataset.",
    )
    ds.add_argument(
        "--column",
        default="observation.pico.body_joints_pose",
        help="Feature column containing PICO body joint poses.",
    )
    ds.add_argument(
        "--episodes",
        default=None,
        help=(
            "Comma-separated list of episode indices to process "
            "(default: all episodes)."
        ),
    )
    ds.add_argument(
        "--task",
        default=None,
        help=(
            "Override the task description for all episodes.  "
            "When not set the script tries to read tasks from the source "
            "dataset's tasks.parquet."
        ),
    )
    ds.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the resulting dataset to the HuggingFace Hub after writing.",
    )
    ds.add_argument(
        "--hub-token",
        default=None,
        help="HuggingFace API token (uses HF_TOKEN env var if not set).",
    )

    # ------------------------------------------------------------------
    # Embodiment selection
    # ------------------------------------------------------------------
    emb = parser.add_argument_group("Embodiment")
    emb.add_argument(
        "--embodiment",
        choices=("axol", "piper"),
        default="axol",
        help="Target robot embodiment.",
    )

    # ------------------------------------------------------------------
    # Shared IK parameters
    # ------------------------------------------------------------------
    ik = parser.add_argument_group("Shared IK parameters")
    ik.add_argument("--scale", type=float, default=1.0)
    ik.add_argument(
        "--axis-map",
        default=None,
        help=(
            "PICO delta → robot delta mapping, e.g. z,x,y or z,y,-x.  "
            "Defaults to the selected embodiment's validated mapping."
        ),
    )
    ik.add_argument("--left-only", action="store_true")
    ik.add_argument("--right-only", action="store_true")
    ik.add_argument(
        "--gripper",
        type=float,
        default=1.0,
        help="Constant gripper value in [0, 1] written for every frame.",
    )
    ik.add_argument("--pos-weight", type=float, default=50.0)
    ik.add_argument("--ori-weight", type=float, default=0.0)
    ik.add_argument("--elbow-weight", type=float, default=5.0)
    ik.add_argument("--max-joint-delta", type=float, default=0.35)
    ik.add_argument("--max-reach", type=float, default=0.8)

    # ------------------------------------------------------------------
    # Axol-specific parameters
    # ------------------------------------------------------------------
    axol = parser.add_argument_group("Axol-specific parameters")
    axol.add_argument(
        "--axol-workspace",
        choices=("front", "rest"),
        default="front",
        help="Use a front/chest initial workspace or the raw URDF rest pose.",
    )
    axol.add_argument("--axol-wrist-forward", type=float, default=-0.34)
    axol.add_argument("--axol-wrist-height", type=float, default=0.58)
    axol.add_argument("--axol-wrist-lateral", type=float, default=0.23)
    axol.add_argument("--axol-elbow-forward", type=float, default=-0.16)
    axol.add_argument("--axol-elbow-height", type=float, default=0.68)
    axol.add_argument("--axol-elbow-lateral", type=float, default=0.20)
    axol.add_argument(
        "--settle-iterations",
        type=int,
        default=20,
        help="IK iterations on the first frame before episode processing starts.",
    )

    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_output_repo_id(source_repo_id: str, embodiment: str) -> str:
    """Derive ``{namespace}/{source_name}-{embodiment}`` from the source repo id."""
    repo_id = source_repo_id.rstrip("/")
    if "/" in repo_id:
        namespace, name = repo_id.rsplit("/", 1)
        return f"{namespace}/{name}-{embodiment}"
    return f"{repo_id}-{embodiment}"


def _parse_episode_list(value: str | None, max_episodes: int) -> list[int]:
    if value is None:
        return list(range(max_episodes))
    indices = [int(x.strip()) for x in value.split(",") if x.strip()]
    out_of_range = [idx for idx in indices if idx < 0 or idx >= max_episodes]
    if out_of_range:
        raise ValueError(
            f"Episode indices out of range [0, {max_episodes - 1}]: {out_of_range}"
        )
    return indices


def _load_source_tasks(source_root: Path) -> dict[int, str]:
    """Return a mapping ``{episode_index: task_string}`` from tasks.parquet."""
    import pandas as pd

    tasks_path = source_root / "meta" / "tasks.parquet"
    episodes_dir = source_root / "meta" / "episodes"
    task_map: dict[int, str] = {}

    if not tasks_path.exists():
        return task_map

    tasks_df = pd.read_parquet(tasks_path)
    task_idx_to_str: dict[int, str] = {
        int(row["task_index"]): str(task_str)
        for task_str, row in tasks_df.iterrows()
    }

    if not episodes_dir.exists():
        return task_map

    import glob

    for parquet_path in sorted(glob.glob(str(episodes_dir / "**/*.parquet"), recursive=True)):
        ep_df = pd.read_parquet(parquet_path)
        for _, row in ep_df.iterrows():
            ep_idx = int(row["episode_index"])
            task_list = row.get("tasks", [])
            if task_list and len(task_list) > 0:
                first_task_idx = task_list[0] if isinstance(task_list[0], (int, np.integer)) else 0
                task_map[ep_idx] = task_idx_to_str.get(int(first_task_idx), "task")
    return task_map


# ---------------------------------------------------------------------------
# Per-episode IK processing
# ---------------------------------------------------------------------------


def process_episode(
    *,
    args: argparse.Namespace,
    poses: np.ndarray,
    episode_index: int,
    source_episode_index: int,
    task: str,
) -> object:
    """Run IK retargeting on one episode and return an EpisodeResult.

    A fresh retargeter is built for each episode so the calibration
    is relative to that episode's first frame.

    Parameters
    ----------
    args:
        Parsed CLI args.
    poses:
        PICO body poses of shape ``(T, 24, 7)``.
    episode_index:
        Index in the *output* dataset.
    source_episode_index:
        Index in the *source* dataset (for video copying).
    task:
        Task description string.

    Returns
    -------
    EpisodeResult
    """
    from handumi.dataset import EpisodeResult
    from handumi.robots.loader import build_embodiment

    if len(poses) < 2:
        raise ValueError(
            f"Episode {source_episode_index} has fewer than 2 frames; "
            "cannot construct (state, action) pairs."
        )

    bundle = build_embodiment(args.embodiment, args, poses[0])

    q = bundle.initial_q.copy()
    q_list: list[np.ndarray] = []
    for i, body_pose in enumerate(poses):
        q = bundle.retarget_frame(body_pose, q)
        q_list.append(bundle.extract_joints(q))
        if (i + 1) % 100 == 0 or (i + 1) == len(poses):
            print(f"    frame {i + 1}/{len(poses)}", end="\r", flush=True)

    print()  # newline after progress

    joint_array = np.stack(q_list, axis=0)  # (T, N_joints)
    states = joint_array[:-1]               # t = 0 … T-2
    actions = joint_array[1:]               # t = 1 … T-1

    return EpisodeResult(
        episode_index=episode_index,
        states=states,
        actions=actions,
        task=task,
        source_episode_index=source_episode_index,
    )


# ---------------------------------------------------------------------------
# Hub upload
# ---------------------------------------------------------------------------


def push_to_hub(
    output_root: Path,
    repo_id: str,
    *,
    token: str | None = None,
) -> None:
    """Push the dataset directory to the HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for --push-to-hub.  "
            "Install it with: pip install huggingface_hub"
        ) from exc

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(output_root),
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Pushed dataset to https://huggingface.co/datasets/{repo_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.left_only and args.right_only:
        parser.error("Use only one of --left-only or --right-only.")

    # ------------------------------------------------------------------
    # Resolve paths and names
    # ------------------------------------------------------------------
    source_repo_id = args.repo_id
    source_root = Path(args.root) if args.root else dataset_root_from_repo_id(source_repo_id)
    output_repo_id = args.output_repo_id or _default_output_repo_id(
        source_repo_id, args.embodiment
    )
    output_root = dataset_root_from_repo_id(output_repo_id)

    print(f"Source  : {source_root}  ({source_repo_id})")
    print(f"Output  : {output_root}  ({output_repo_id})")
    print(f"Embodiment: {args.embodiment}")

    # ------------------------------------------------------------------
    # Ensure source metadata is available, then read episode count
    # ------------------------------------------------------------------
    from handumi.dataset import ensure_metadata

    source_info = ensure_metadata(
        repo_id=source_repo_id,
        root=source_root,
        revision=args.revision,
    )
    total_source_episodes = int(source_info.get("total_episodes", 0))
    dataset_fps = int(source_info.get("fps", 30))

    if total_source_episodes <= 0:
        parser.error(
            f"Could not determine total_episodes from "
            f"{source_root / 'meta' / 'info.json'}. "
            "Check --repo-id, --root, and --revision."
        )

    try:
        episode_indices = _parse_episode_list(args.episodes, total_source_episodes)
    except ValueError as exc:
        parser.error(str(exc))

    print(
        f"Source episodes: {total_source_episodes}  "
        f"Processing: {len(episode_indices)}  "
        f"{episode_indices[:5]}{'…' if len(episode_indices) > 5 else ''}"
    )

    # ------------------------------------------------------------------
    # Load task descriptions for each episode
    # ------------------------------------------------------------------
    source_task_map = _load_source_tasks(source_root)
    default_task = args.task or "task"

    def get_task(ep_idx: int) -> str:
        if args.task:
            return args.task
        return source_task_map.get(ep_idx, default_task)

    # ------------------------------------------------------------------
    # Process each episode
    # ------------------------------------------------------------------
    from handumi.dataset import load_pico_body_poses

    results = []
    for out_idx, src_idx in enumerate(episode_indices):
        print(f"\nEpisode {out_idx + 1}/{len(episode_indices)}  (source ep {src_idx})")
        try:
            poses, ep_fps = load_pico_body_poses(
                repo_id=source_repo_id,
                root=source_root,
                episode=src_idx,
                column=args.column,
                revision=args.revision,
            )
        except Exception as exc:
            print(f"  SKIP: failed to load — {exc}", file=sys.stderr)
            continue

        task = get_task(src_idx)
        try:
            result = process_episode(
                args=args,
                poses=poses,
                episode_index=out_idx,
                source_episode_index=src_idx,
                task=task,
            )
        except Exception as exc:
            print(f"  SKIP: IK failed — {exc}", file=sys.stderr)
            continue

        results.append(result)
        print(
            f"  Done: {len(result.states)} frames, "
            f"task={result.task!r}"
        )

    if not results:
        print("No episodes processed successfully. Exiting.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Determine joint names and robot_type from the embodiment spec
    # ------------------------------------------------------------------
    from handumi.robots.loader import build_embodiment

    # Build a minimal dummy bundle just to read the spec (no compute)
    dummy_pose = np.zeros((24, 7), dtype=np.float32)
    dummy_bundle = build_embodiment(args.embodiment, args, dummy_pose)
    spec = dummy_bundle.spec

    # ------------------------------------------------------------------
    # Write output dataset
    # ------------------------------------------------------------------
    from handumi.dataset import write_dataset

    write_dataset(
        output_root=output_root,
        source_root=source_root,
        source_info=source_info,
        episodes=results,
        robot_type=spec.robot_type,
        joint_names=spec.joint_names,
        fps=dataset_fps,
    )

    # ------------------------------------------------------------------
    # Optional: push to Hub
    # ------------------------------------------------------------------
    if args.push_to_hub:
        push_to_hub(output_root, output_repo_id, token=args.hub_token)


if __name__ == "__main__":
    main()
