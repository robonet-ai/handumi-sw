import json

import numpy as np

from handumi.dataset.quality import (
    EpisodeQualityConfig,
    validate_episode,
    write_quality_report,
)


FPS = 30.0
FRAMES = 60


def _states() -> np.ndarray:
    states = np.zeros((FRAMES, 16), dtype=np.float32)
    states[:, 0] = np.linspace(0.0, 0.20, FRAMES)
    states[:, 7] = np.linspace(0.0, -0.20, FRAMES)
    states[:, 3:7] = [0.0, 0.0, 0.0, 1.0]
    states[:, 10:14] = [0.0, 0.0, 0.0, 1.0]
    states[:, 14] = np.linspace(0.01, 0.05, FRAMES)
    states[:, 15] = np.linspace(0.02, 0.06, FRAMES)
    return states


def _signals() -> dict[str, np.ndarray]:
    target = np.arange(FRAMES, dtype=np.int64) * 33_333_333 + 1_000_000_000
    ones = np.ones(FRAMES, dtype=np.int64)
    zeros = np.zeros(FRAMES, dtype=np.float32)
    return {
        "observation.sync.target_time_ns": target,
        "observation.tracking.aligned_time_ns": target.copy(),
        "observation.tracking.left_tracked": ones.copy(),
        "observation.tracking.right_tracked": ones.copy(),
        "observation.tracking.sync_error_ms": zeros.copy(),
        "observation.feetech.sample_time_ns": target.copy(),
        "observation.feetech.enabled": ones.copy(),
        "observation.feetech.healthy": ones.copy(),
        "observation.feetech.sync_error_ms": zeros.copy(),
        "observation.camera.left_wrist.sample_time_ns": target.copy(),
        "observation.camera.left_wrist.enabled": ones.copy(),
        "observation.camera.left_wrist.healthy": ones.copy(),
        "observation.camera.left_wrist.sync_error_ms": zeros.copy(),
    }


def _reject_codes(report) -> set[str]:
    return {finding.code for finding in report.findings if finding.severity == "reject"}


def test_clean_episode_is_accepted():
    report = validate_episode(_states(), fps=FPS, signals=_signals())

    assert report.accepted
    assert not _reject_codes(report)
    assert report.metrics["bad_tracking_fraction"] == 0.0


def test_repeated_short_tracking_losses_are_rejected_by_total_fraction():
    signals = _signals()
    signals["observation.tracking.left_tracked"][[5, 20, 40]] = 0

    report = validate_episode(_states(), fps=FPS, signals=signals)

    assert "tracking_quality_fraction" in _reject_codes(report)


def test_sensor_health_and_sync_fraction_are_checked():
    signals = _signals()
    signals["observation.camera.left_wrist.healthy"][:3] = 0
    signals["observation.feetech.sync_error_ms"][:3] = 100.0

    report = validate_episode(_states(), fps=FPS, signals=signals)

    assert {"sensor_health_fraction", "sensor_sync_fraction"} <= _reject_codes(report)


def test_frozen_source_timestamp_is_rejected():
    signals = _signals()
    signals["observation.camera.left_wrist.sample_time_ns"][:] = 42

    report = validate_episode(_states(), fps=FPS, signals=signals)

    assert "source_timestamp_freeze" in _reject_codes(report)


def test_full_pose_freeze_is_rejected_but_constant_aperture_is_only_warning():
    states = _states()
    states[:, :14] = states[0, :14]
    states[:, 14:] = states[0, 14:]

    report = validate_episode(states, fps=FPS, signals=_signals())

    assert "full_pose_freeze" in _reject_codes(report)
    assert any(
        finding.code == "aperture_freeze" and finding.severity == "warning"
        for finding in report.findings
    )


def test_implausible_translation_and_rotation_are_rejected():
    states = _states()
    states[30, 0] += 2.0
    states[30, 10:14] = [1.0, 0.0, 0.0, 0.0]

    report = validate_episode(states, fps=FPS, signals=_signals())

    assert {"translation_jump", "rotation_jump"} <= _reject_codes(report)


def test_quality_report_is_machine_readable(tmp_path):
    config = EpisodeQualityConfig()
    report = validate_episode(
        _states(), fps=FPS, signals=_signals(), episode_index=7, config=config
    )

    path = write_quality_report(
        tmp_path / "quality.json", [report], config=config, dataset="local/test"
    )
    payload = json.loads(path.read_text())

    assert payload["summary"] == {"accepted": 1, "rejected": 0, "total": 1}
    assert payload["episodes"][0]["episode_index"] == 7


def test_invalid_fraction_configuration_is_rejected():
    try:
        EpisodeQualityConfig(max_bad_tracking_fraction=1.1)
    except ValueError as exc:
        assert "max_bad_tracking_fraction" in str(exc)
    else:
        raise AssertionError("invalid quality fraction was accepted")
