"""One user-facing dataset argument for local paths and Hub repositories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from handumi.dataset.reader import dataset_root_from_repo_id


@dataclass(frozen=True)
class DatasetSelection:
    repo_id: str
    root: Path
    revision: str
    local: bool


def resolve_dataset_selection(
    dataset: str | Path,
    *,
    revision: str = "main",
) -> DatasetSelection:
    """Resolve the canonical positional DATASET as a local path or Hub id."""
    value = str(dataset)
    candidate = Path(value).expanduser()
    is_local = (
        candidate.exists()
        or candidate.parent.exists()
        or value.startswith((".", "/", "~"))
    )
    if is_local:
        local_name = candidate.name or candidate.resolve().name
        return DatasetSelection(
            repo_id=f"local/{local_name}",
            root=candidate,
            revision=revision,
            local=True,
        )
    return DatasetSelection(
        repo_id=value,
        root=dataset_root_from_repo_id(value),
        revision=revision,
        local=False,
    )
