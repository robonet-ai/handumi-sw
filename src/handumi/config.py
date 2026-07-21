"""Shared loading for the machine-local HandUMI rig configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_RIG_CONFIG = Path("configs/rig.yaml")
_PACKAGE_ROOT = Path(__file__).resolve().parent
EXAMPLE_RIG_CONFIG = (
    Path("configs/rig.example.yaml")
    if Path("configs/rig.example.yaml").exists()
    else _PACKAGE_ROOT / "configs" / "rig.example.yaml"
)


def load_rig_section(path: Path, section: str) -> dict[str, Any]:
    """Load one mapping from the unified rig YAML."""
    if not path.exists():
        raise SystemExit(
            f"Missing rig configuration: {path}.\n"
            f"Create it with: cp {EXAMPLE_RIG_CONFIG} {DEFAULT_RIG_CONFIG}"
        )
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    value = data.get(section)
    if not isinstance(value, dict):
        raise SystemExit(f"Missing or invalid '{section}' section in {path}.")
    return value
