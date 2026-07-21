#!/usr/bin/env python3
"""Fail closed on dependency or bundled robot-asset inventory drift."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _tree_hash(root: Path, paths: list[str]) -> tuple[str, list[str]]:
    files: list[Path] = []
    for item in paths:
        candidate = root / item
        if candidate.is_dir():
            files.extend(path for path in candidate.rglob("*") if path.is_file())
        elif candidate.is_file():
            files.append(candidate)
        else:
            raise ValueError(f"asset manifest path is missing: {item}")
    lines = []
    relative_files = []
    for path in sorted(set(files)):
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {relative}\n")
        relative_files.append(relative)
    return hashlib.sha256("".join(lines).encode()).hexdigest(), relative_files


def verify(root: Path) -> dict[str, Any]:
    asset_path = root / "release" / "asset-manifest.json"
    dependency_path = root / "release" / "dependency-source-inventory.json"
    assets = json.loads(asset_path.read_text())
    dependencies = json.loads(dependency_path.read_text())
    mapped_roots = {f"assets/{item['id']}" for item in assets["asset_sets"]}
    actual_roots = {
        path.relative_to(root).as_posix()
        for path in (root / "assets").iterdir()
        if path.is_dir() and path.name != "scenes"
    }
    if mapped_roots != actual_roots:
        raise ValueError(
            f"unmapped robot asset roots: {sorted(actual_roots ^ mapped_roots)}"
        )
    file_count = 0
    for item in assets["asset_sets"]:
        digest, files = _tree_hash(root, item["paths"])
        file_count += len(files)
        if digest != item["tree_sha256"]:
            raise ValueError(
                f"asset drift for {item['id']}: expected {item['tree_sha256']}, got {digest}"
            )
        if item["distributed"] and (not item["license"] or not item["notice"]):
            raise ValueError(
                f"distributed asset {item['id']} lacks license/NOTICE mapping"
            )
        if not item["distributed"] and item["license"] is None:
            pyproject = (root / "pyproject.toml").read_text()
            if f"assets/{item['id']}/**" not in pyproject:
                raise ValueError(
                    f"unlicensed asset {item['id']} is not excluded from distributions"
                )
    required = {"jaxls", "pyroki", "piper_sdk", "openarm_can"}
    names = {item["name"] for item in dependencies["dependencies"]}
    if not required <= names:
        raise ValueError(
            f"source dependency inventory is missing {sorted(required - names)}"
        )
    if not dependencies["cpu_default"] or dependencies["cuda_optional_group"] != "cuda":
        raise ValueError("CPU-default/CUDA-optional policy drifted")
    return {
        "asset_sets": len(assets["asset_sets"]),
        "mapped_files": file_count,
        "dependencies": len(names),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(verify(args.root.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
