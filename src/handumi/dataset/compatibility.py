"""Authoritative HandUMI packet/dataset compatibility and recovery policy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from handumi.dataset.raw import (
    HANDUMI_CAPTURE_SCHEMA,
    HANDUMI_STATE_SEMANTICS,
    HANDUMI_TRACKING_SCHEMA,
)

POLICY_SCHEMA = "handumi_dataset_compatibility_v1"
PRECOMPACT_TRACKING_SCHEMA = "controller_raw_and_workspace_v3"
PRECOMPACT_CAPTURE_SCHEMA = "synchronized_sources_v1"
PRECOMPACT_CUTOFF = "2026-07-10T00:00:00Z"


class DatasetCompatibilityError(ValueError):
    """Dataset layout or episode state is not safe to interpret."""


@dataclass(frozen=True)
class CompatibilityDecision:
    supported: bool
    layout: str
    body_optional: bool
    migration: str


def classify_metadata(info: Mapping[str, Any]) -> CompatibilityDecision:
    handumi = info.get("handumi", {})
    metadata = handumi if isinstance(handumi, Mapping) else {}
    actual = (
        str(metadata.get("tracking_schema", "")),
        str(metadata.get("capture_schema", "")),
        str(metadata.get("state_semantics", "")),
    )
    current = (
        HANDUMI_TRACKING_SCHEMA,
        HANDUMI_CAPTURE_SCHEMA,
        HANDUMI_STATE_SEMANTICS,
    )
    if actual == current:
        return CompatibilityDecision(
            supported=True,
            layout="lerobot-v3-compact",
            body_optional=True,
            migration="none",
        )
    if actual[:2] == (PRECOMPACT_TRACKING_SCHEMA, PRECOMPACT_CAPTURE_SCHEMA):
        raise DatasetCompatibilityError(
            "Unsupported pre-compact HandUMI layout recorded before cutoff "
            f"{PRECOMPACT_CUTOFF}. Its duplicated workspace/raw-controller fields "
            "do not identify a single unambiguous state meaning, so HandUMI will not "
            "guess or fabricate a conversion. Preserve the original recording and "
            "re-record with current handumi-record; contact a maintainer if a "
            "Re-record with current handumi-record; contact a maintainer if a "
            "format-specific, evidence-backed importer is required."
        )
    raise DatasetCompatibilityError(
        "Unsupported or missing HandUMI schema versions. Expected "
        f"tracking={HANDUMI_TRACKING_SCHEMA!r}, capture={HANDUMI_CAPTURE_SCHEMA!r}, "
        f"state={HANDUMI_STATE_SEMANTICS!r}; got {actual!r}. No heuristic migration "
        "was attempted."
    )


def validate_episode_state(root: Path) -> str:
    """Reject partial episodes; identify stale staging as recoverable evidence."""
    manifest_path = root / "session-manifest.json"
    if not manifest_path.exists():
        if "handumi-inprogress" in root.name:
            return "recoverable-staged"
        return "legacy-complete-without-session-manifest"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetCompatibilityError(f"Corrupt session manifest: {exc}") from exc
    status = str(manifest.get("completion_status", ""))
    if status == "complete":
        return status
    if status == "incomplete":
        raise DatasetCompatibilityError(
            "Interrupted episode is incomplete and cannot be opened as a dataset. "
            "Preserve it for forensic recovery; use the recovery command/policy."
        )
    if status == "rejected":
        raise DatasetCompatibilityError(
            "Episode was explicitly rejected and cannot be opened as complete data."
        )
    raise DatasetCompatibilityError(f"Unknown session completion status: {status!r}")


def validate_dataset_compatibility(
    info: Mapping[str, Any], *, root: Path | None = None
) -> CompatibilityDecision:
    if root is not None:
        validate_episode_state(root)
    return classify_metadata(info)
