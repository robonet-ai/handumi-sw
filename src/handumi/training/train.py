"""Train a policy on a recorded HandUMI dataset.

Thin wrapper over ``lerobot-train``: handumi owns no training loop, optimizer,
or logging code. This entrypoint only resolves *which* dataset to train on
(a local ``outputs/<timestamp>/`` folder from the recorders) and *which*
config to start from (``configs/train/<policy>.yaml``), then hands everything
to lerobot's own CLI. Any extra flags after the known ones pass straight
through to lerobot-train, and being later on the command line they override
the YAML defaults.

Usage:

    # Train ACT on the most recent recording, logging to wandb:
    handumi-train --latest

    # Explicit dataset, override steps:
    handumi-train --dataset outputs/20260707_143000 --steps=50000

    # Another policy config (configs/train/<name>.yaml):
    handumi-train --latest --policy act

Checkpoints land in lerobot's default ``outputs/train/<date>/<time>_<job>/``
(gitignored along with the rest of outputs/).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Must match the recorders' --repo-id default (record_handumi_quest.py).
DEFAULT_REPO_ID = "local/handumi_quest"
DATASETS_DIR = Path("outputs")
TRAIN_CONFIGS_DIR = Path("configs/train")


def _latest_dataset(datasets_dir: Path = DATASETS_DIR) -> Path:
    """Most recently recorded dataset: outputs/<timestamp>/ folders holding a
    LeRobotDataset (identified by meta/info.json)."""
    candidates = sorted(
        d
        for d in (datasets_dir.iterdir() if datasets_dir.is_dir() else [])
        if (d / "meta" / "info.json").is_file()
    )
    if not candidates:
        raise SystemExit(
            f"No datasets found under {datasets_dir}/ — record one first "
            "(handumi-record-quest) or pass --dataset <path>."
        )
    return candidates[-1]


def _config_flags(config_path: Path) -> list[str]:
    """Flatten configs/train/<policy>.yaml (draccus dotted keys) into
    --key=value flags for lerobot-train."""
    if not config_path.is_file():
        raise SystemExit(f"Training config not found: {config_path}")
    entries = yaml.safe_load(config_path.read_text()) or {}
    flags = []
    for key, value in entries.items():
        if isinstance(value, bool):
            value = "true" if value else "false"
        flags.append(f"--{key}={value}")
    return flags


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", type=Path, help="Local dataset folder (outputs/<ts>)")
    src.add_argument(
        "--latest", action="store_true", help="Train on the newest dataset in outputs/"
    )
    p.add_argument(
        "--policy",
        default="act",
        help="Training config name: configs/train/<policy>.yaml (default: act)",
    )
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Dataset repo id")
    args, passthrough = p.parse_known_args()

    dataset_root = (args.dataset or _latest_dataset()).resolve()
    if not (dataset_root / "meta" / "info.json").is_file():
        raise SystemExit(f"Not a LeRobotDataset (no meta/info.json): {dataset_root}")

    argv = [
        "lerobot-train",
        *_config_flags(TRAIN_CONFIGS_DIR / f"{args.policy}.yaml"),
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={dataset_root}",
        *passthrough,  # last wins: user flags override the YAML defaults
    ]
    print(f"Dataset: {dataset_root}")
    print(f"Running: {' '.join(argv)}")

    # Import lazily: torch et al. only load once we actually train.
    from lerobot.scripts.lerobot_train import main as lerobot_train_main

    sys.argv = argv
    lerobot_train_main()


if __name__ == "__main__":
    main()
