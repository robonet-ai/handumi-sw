"""Shared LeRobot v3.0 on-disk layout helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CHUNKS_SIZE = 1000


def chunk_and_file(index: int, chunks_size: int = CHUNKS_SIZE) -> tuple[int, int]:
    return index // chunks_size, index % chunks_size


def info_path(root: str | Path) -> Path:
    return Path(root) / "meta" / "info.json"


def load_info(root: str | Path) -> dict[str, Any]:
    path = info_path(root)
    with open(path) as fh:
        return json.load(fh)
