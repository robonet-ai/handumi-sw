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
    data = load_rig_config(path)
    value = data.get(section)
    if not isinstance(value, dict):
        raise SystemExit(f"Missing or invalid '{section}' section in {path}.")
    return value


def load_rig_config(path: Path = DEFAULT_RIG_CONFIG) -> dict[str, Any]:
    """Load the complete rig mapping so optional UX defaults can coexist."""
    if not path.exists():
        raise SystemExit(
            f"Missing rig configuration: {path}.\n"
            f"Create it with: cp {EXAMPLE_RIG_CONFIG} {DEFAULT_RIG_CONFIG}"
        )
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid rig configuration mapping: {path}.")
    return data


def load_optional_rig_section(
    path: Path,
    section: str,
) -> dict[str, Any]:
    """Return an optional rig section without making old rig files invalid."""
    value = load_rig_config(path).get(section, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SystemExit(f"Invalid '{section}' section in {path}; expected a mapping.")
    return value
