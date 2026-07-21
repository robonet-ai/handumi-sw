import json
from pathlib import Path

import pytest

from handumi.dataset.compatibility import (
    DatasetCompatibilityError,
    classify_metadata,
    validate_episode_state,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "compatibility" / "golden-cases.json"


def _info(dataset: dict[str, object]) -> dict[str, object]:
    return {"handumi": dataset}


def test_all_supported_golden_datasets_preserve_optional_body_without_fabrication():
    fixture = json.loads(FIXTURE.read_text())
    assert all(value is False for value in fixture["privacy"].values())
    datasets = [case["dataset"] for case in fixture["cases"] if "dataset" in case]
    for dataset in datasets:
        decision = classify_metadata(_info(dataset))
        assert decision.supported
        assert decision.body_optional
    controller = datasets[0]
    assert controller["body"] is None
    optional = next(dataset for dataset in datasets if "optional" in dataset)
    assert all(value is None for value in optional["optional"].values())


def test_precompact_cutoff_is_exact_actionable_and_never_heuristic():
    with pytest.raises(DatasetCompatibilityError, match="2026-07-10.*will not guess"):
        classify_metadata(
            _info(
                {
                    "tracking_schema": "controller_raw_and_workspace_v3",
                    "capture_schema": "synchronized_sources_v1",
                    "state_semantics": "ambiguous",
                }
            )
        )


def test_interrupted_rejected_recoverable_and_corrupt_states(tmp_path: Path):
    staged = tmp_path / ".episode.handumi-inprogress-id"
    staged.mkdir()
    assert validate_episode_state(staged) == "recoverable-staged"

    for status in ("incomplete", "rejected"):
        root = tmp_path / status
        root.mkdir()
        (root / "session-manifest.json").write_text(
            json.dumps({"completion_status": status})
        )
        with pytest.raises(DatasetCompatibilityError, match=status):
            validate_episode_state(root)

    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / "session-manifest.json").write_text("{")
    with pytest.raises(DatasetCompatibilityError, match="Corrupt"):
        validate_episode_state(corrupt)
