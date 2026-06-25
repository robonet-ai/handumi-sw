"""Dataset location references shared by readers and writers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetRef:
    """Pointer to a LeRobot dataset on disk and/or on the Hugging Face Hub."""

    repo_id: str
    root: Path
    revision: str = "main"

    @classmethod
    def from_repo_id(
        cls,
        repo_id: str,
        *,
        root: str | Path | None = None,
        revision: str = "main",
    ) -> DatasetRef:
        resolved_root = Path(root) if root is not None else dataset_root_from_repo_id(repo_id)
        return cls(repo_id=repo_id, root=resolved_root, revision=revision)


def dataset_root_from_repo_id(repo_id: str) -> Path:
    """Default local cache directory for a Hugging Face dataset repo id."""
    repo_name = repo_id.rstrip("/").split("/")[-1]
    if not repo_name:
        raise ValueError(f"Cannot derive dataset root from repo id {repo_id!r}.")
    return Path("outputs/datasets") / repo_name
