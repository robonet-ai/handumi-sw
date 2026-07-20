#!/usr/bin/env python3
"""Fail closed when a HandUMI wheel leaks source-only release dependencies."""

from __future__ import annotations

import argparse
import email
import zipfile
from pathlib import Path

FORBIDDEN_REQUIREMENTS = (
    "git+",
    "jaxls",
    "lerobot",
    "openarm_can",
    "piper_sdk",
    "pyroki",
    "torch",
)
REQUIRED_MEMBERS = (
    "handumi/py.typed",
    "handumi/configs/body_profile.example.yaml",
    "handumi/configs/rig.example.yaml",
    "handumi/configs/robots/piper.yaml",
    "handumi/assets/openarm/LICENSE.openarm",
    "handumi/assets/trlc-dk1/LICENSE.trlc-dk1",
)
REQUIRED_COMMANDS = (
    "handumi-record",
    "handumi-teleop-sim",
    "handumi-replay-in-sim",
    "handumi-preflight",
)
FORBIDDEN_MEMBERS = (
    "handumi/assets/r1lite/",
    "handumi/configs/robots/r1lite.yaml",
    "handumi/configs/calibration/r1lite_table.yaml",
)


def verify(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        metadata_name = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        entry_points_name = next(
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        )
        metadata = email.message_from_bytes(archive.read(metadata_name))
        requirements = metadata.get_all("Requires-Dist", [])
        entry_points = archive.read(entry_points_name).decode()

    missing = [name for name in REQUIRED_MEMBERS if name not in names]
    if missing:
        raise SystemExit(f"Wheel is missing required resources: {missing}")
    if not any(name.endswith(".dist-info/licenses/NOTICE") for name in names):
        raise SystemExit("Wheel is missing the release NOTICE")
    forbidden_members = [
        name
        for name in names
        if any(
            name == forbidden or name.startswith(forbidden)
            for forbidden in FORBIDDEN_MEMBERS
        )
    ]
    if forbidden_members:
        raise SystemExit(
            "Wheel contains R1 Lite resources without redistribution terms: "
            f"{sorted(forbidden_members)}"
        )
    leaked = [
        requirement
        for requirement in requirements
        if any(value in requirement.lower() for value in FORBIDDEN_REQUIREMENTS)
    ]
    if leaked:
        raise SystemExit(f"Wheel metadata contains source-only dependencies: {leaked}")
    missing_commands = [
        command for command in REQUIRED_COMMANDS if f"{command} =" not in entry_points
    ]
    if missing_commands:
        raise SystemExit(f"Wheel is missing console entry points: {missing_commands}")
    print(
        f"verified {path}: {len(names)} members, "
        f"{len(requirements)} bounded requirements"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    verify(args.wheel)


if __name__ == "__main__":
    main()
