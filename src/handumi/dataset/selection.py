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
    dataset: str | Path | None,
    *,
    repo_id: str | None,
    root: str | Path | None,
    revision: str = "main",
    default_repo_id: str,
) -> DatasetSelection:
    """Resolve positional DATASET while preserving legacy flags."""
    if dataset is not None and (repo_id is not None or root is not None):
        raise ValueError("Use positional DATASET or --repo-id/--root, not both.")

    if dataset is not None:
        value = str(dataset)
        candidate = Path(value).expanduser()
        is_local = candidate.exists() or value.startswith((".", "/", "~"))
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

    resolved_repo_id = repo_id or default_repo_id
    resolved_root = (
        Path(root).expanduser()
        if root is not None
        else dataset_root_from_repo_id(resolved_repo_id)
    )
    return DatasetSelection(
        repo_id=resolved_repo_id,
        root=resolved_root,
        revision=revision,
        local=root is not None,
    )
