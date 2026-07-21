from pathlib import Path

import pytest

from handumi.dataset.selection import resolve_dataset_selection


def test_positional_local_dataset_is_resolved_without_repo_flags(tmp_path: Path):
    dataset = tmp_path / "capture"
    dataset.mkdir()

    selected = resolve_dataset_selection(
        dataset,
        repo_id=None,
        root=None,
        default_repo_id="org/default",
    )

    assert selected.local
    assert selected.root == dataset
    assert selected.repo_id == "local/capture"


def test_positional_hub_dataset_uses_repo_id():
    selected = resolve_dataset_selection(
        "org/capture",
        repo_id=None,
        root=None,
        default_repo_id="org/default",
    )

    assert not selected.local
    assert selected.repo_id == "org/capture"
    assert selected.root.name == "capture"


def test_positional_dataset_cannot_be_mixed_with_legacy_flags(tmp_path: Path):
    with pytest.raises(ValueError, match="positional DATASET"):
        resolve_dataset_selection(
            tmp_path,
            repo_id="org/capture",
            root=None,
            default_repo_id="org/default",
        )


def test_current_directory_has_a_stable_local_repo_id(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    selected = resolve_dataset_selection(
        ".",
        repo_id=None,
        root=None,
        default_repo_id="org/default",
    )

    assert selected.repo_id == f"local/{tmp_path.name}"
