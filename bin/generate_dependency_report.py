#!/usr/bin/env python3
"""Inventory installed distribution metadata and bundled license evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_values(message: metadata.PackageMetadata, key: str) -> list[str]:
    values = message.get_all(key) or []
    return sorted({value.strip() for value in values if value.strip()})


def _license_files(distribution: metadata.Distribution) -> list[dict[str, Any]]:
    declared = set(_metadata_values(distribution.metadata, "License-File"))
    candidates: list[Path] = []
    for item in distribution.files or ():
        item_path = Path(str(item))
        filename = item_path.name.lower()
        if str(item) in declared or filename.startswith(
            ("license", "licence", "copying", "notice")
        ):
            candidates.append(item_path)

    evidence: list[dict[str, Any]] = []
    for relative in sorted(set(candidates), key=str):
        resolved = Path(str(distribution.locate_file(relative)))
        if not resolved.is_file():
            continue
        evidence.append(
            {
                "path": str(relative),
                "bytes": resolved.stat().st_size,
                "sha256": _sha256(resolved),
            }
        )
    return evidence


def _summarize_license(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split())
    if len(normalized) <= 240:
        return normalized
    return f"{normalized[:237]}..."


def build_report() -> dict[str, Any]:
    packages: list[dict[str, Any]] = []
    for distribution in metadata.distributions():
        package_metadata = distribution.metadata
        name = package_metadata.get("Name")
        if not name:
            continue
        packages.append(
            {
                "name": name,
                "version": distribution.version,
                "license_expression": package_metadata.get("License-Expression"),
                "license_metadata": _summarize_license(
                    package_metadata.get("License")
                ),
                "license_classifiers": [
                    value
                    for value in _metadata_values(package_metadata, "Classifier")
                    if value.startswith("License ::")
                ],
                "license_files": _license_files(distribution),
                "project_urls": _metadata_values(package_metadata, "Project-URL"),
            }
        )

    packages.sort(key=lambda package: package["name"].casefold())
    names = {package["name"].casefold() for package in packages}
    if "handumi" not in names:
        raise SystemExit(
            "handumi is not installed; run this script with the isolated "
            "wheel environment's Python interpreter"
        )

    return {
        "schema": "handumi.dependency-license-inventory.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "scope": "installed bounded HandUMI wheel environment",
        "limitations": [
            "Metadata and bundled files are evidence inputs, not legal approval.",
            "Source-only uv groups and system libraries are outside this report.",
            "Transitive native-library notices may require separate review.",
        ],
        "packages": packages,
    }


def write_atomic(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_report()
    write_atomic(args.output, report)
    print(f"wrote {args.output} with {len(report['packages'])} packages")


if __name__ == "__main__":
    main()
