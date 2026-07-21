from pathlib import Path

from handumi.dataset.selection import resolve_dataset_selection


def test_positional_local_dataset_is_resolved_without_repo_flags(tmp_path: Path):
    dataset = tmp_path / "capture"
    dataset.mkdir()

    selected = resolve_dataset_selection(dataset)

    assert selected.local
    assert selected.root == dataset
    assert selected.repo_id == "local/capture"


def test_positional_hub_dataset_uses_repo_id():
    selected = resolve_dataset_selection("org/capture")

    assert not selected.local
    assert selected.repo_id == "org/capture"
    assert selected.root.name == "capture"


def test_nonexistent_dataset_under_existing_parent_is_local(tmp_path: Path):
    dataset = tmp_path / "future-capture"

    selected = resolve_dataset_selection(dataset)

    assert selected.local
    assert selected.root == dataset


def test_current_directory_has_a_stable_local_repo_id(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    selected = resolve_dataset_selection(".")

    assert selected.repo_id == f"local/{tmp_path.name}"
