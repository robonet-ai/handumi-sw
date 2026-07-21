#!/usr/bin/env python3
"""Validate raw HandUMI episodes and write an auditable quality report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from handumi.dataset import ensure_metadata, load_raw_episode
from handumi.dataset.quality import (
    EpisodeQualityConfig,
    validate_episode,
    write_quality_report,
)
from handumi.dataset.selection import resolve_dataset_selection


def build_parser(*, show_advanced: bool = False) -> argparse.ArgumentParser:
    def advanced(text: str) -> str:
        return text if show_advanced else argparse.SUPPRESS

    parser = argparse.ArgumentParser(
        description="Run offline tracking, synchronization, and sensor-health checks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        help="Local dataset path or Hugging Face repo id.",
    )
    parser.add_argument("--help-advanced", action="store_true", help="Show expert options.")
    parser.add_argument("--revision", default="main", help=advanced("Hub dataset revision."))
    parser.add_argument("--source", default="observation.state", help=advanced("Feature to validate."))
    parser.add_argument(
        "--episodes",
        default=None,
        help="Comma-separated source episode indices; defaults to every episode.",
    )
    parser.add_argument(
        "--quality-config",
        type=Path,
        default=Path("configs/quality.yaml"),
        help=advanced("Quality threshold configuration."),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=advanced("Defaults to DATASET/meta/handumi_quality.json."),
    )
    parser.add_argument(
        "--strict",
        dest="fail_on_reject",
        action="store_true",
        help="Exit with status 2 when any episode is rejected.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the validation plan without loading episodes.",
    )
    return parser


def main() -> None:
    raw_argv = list(sys.argv[1:])
    show_advanced = "--help-advanced" in raw_argv
    raw_argv = [value for value in raw_argv if value != "--help-advanced"]
    parser = build_parser(show_advanced=show_advanced)
    if show_advanced:
        parser.print_help()
        return
    args = parser.parse_args(raw_argv)
    try:
        selection = resolve_dataset_selection(
            args.dataset,
            revision=args.revision,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.repo_id = selection.repo_id
    root = selection.root
    print(
        "Validation plan\n"
        f"  Dataset: {root}\n"
        f"  Repository: {args.repo_id}\n"
        f"  Episodes: {args.episodes or 'all'}\n"
        f"  Strict: {'yes' if args.fail_on_reject else 'no'}"
    )
    if args.dry_run:
        return
    info = ensure_metadata(
        repo_id=args.repo_id,
        root=root,
        revision=args.revision,
    )
    total = int(info.get("total_episodes", 0))
    indices = _episode_indices(args.episodes, total)
    config = EpisodeQualityConfig.from_yaml(args.quality_config)
    reports = []

    for position, episode_index in enumerate(indices, start=1):
        print(f"Episode {position}/{len(indices)} (source {episode_index})")
        loaded = load_raw_episode(
            repo_id=args.repo_id,
            root=root,
            episode=episode_index,
            source=args.source,
            revision=args.revision,
        )
        report = validate_episode(
            loaded.states,
            fps=loaded.fps,
            signals=loaded.signals,
            episode_index=episode_index,
            config=config,
        )
        reports.append(report)
        codes = ", ".join(finding.code for finding in report.findings) or "clean"
        print(f"  {'ACCEPT' if report.accepted else 'REJECT'}: {codes}")

    report_path = args.report or root / "meta" / "handumi_quality.json"
    write_quality_report(
        report_path,
        reports,
        config=config,
        dataset=args.repo_id,
    )
    accepted = sum(report.accepted for report in reports)
    print(
        f"Report: {report_path}\n"
        f"Accepted: {accepted}  Rejected: {len(reports) - accepted}"
    )
    if args.fail_on_reject and accepted != len(reports):
        raise SystemExit(2)


def _episode_indices(value: str | None, total: int) -> list[int]:
    if total <= 0:
        raise SystemExit("Dataset metadata reports no episodes.")
    if value is None:
        return list(range(total))
    indices = [int(part.strip()) for part in value.split(",") if part.strip()]
    invalid = [index for index in indices if index < 0 or index >= total]
    if invalid:
        raise SystemExit(f"Episode indices out of range: {invalid}")
    return indices


if __name__ == "__main__":
    main()
