import json
from unittest import mock

import numpy as np
import pandas as pd

from handumi.body.model import (
    CanonicalBodyFrame,
    CanonicalProvenance,
    canonical_body_features,
)
from handumi.dataset.reader import load_raw_episode
from handumi.dataset.raw import (
    HANDUMI_CAPTURE_SCHEMA,
    HANDUMI_STATE_SEMANTICS,
    HANDUMI_TRACKING_SCHEMA,
    TRACKING_VALIDITY_NAMES,
)
from handumi.dataset.tracking_sidecar import (
    FrameEpochTracker,
    TrackingSidecarWriter,
    discover_tracking_sidecars,
    load_tracking_sidecar,
)
from handumi.dataset.writer import EpisodeResult, load_info, write_dataset
from handumi.scripts.record import build_features
from handumi.tracking.meta_quest import parse_tracking_packet
from handumi.tracking.mock_quest_sender import make_tracking_packet_fixture


def _packet(seq: int = 1):
    raw = make_tracking_packet_fixture(84, seq=seq)
    return raw, parse_tracking_packet(
        raw,
        pc_monotonic_ns=10_000 + seq,
        receive_sequence=100 + seq,
        clock_offset_ns=50,
        rtt_ns=10,
    )


def _current_dataset_info() -> dict:
    return {
        "handumi": {
            "tracking_schema": HANDUMI_TRACKING_SCHEMA,
            "capture_schema": HANDUMI_CAPTURE_SCHEMA,
            "state_semantics": HANDUMI_STATE_SEMANTICS,
            "sources": {"feetech": {"enabled": False}, "cameras": {}},
        }
    }


def test_84_joint_sidecar_round_trip_preserves_raw_and_typed_arrays(tmp_path):
    raw, packet = _packet(7)
    raw["futureField"] = {"unknown": [1, 2, 3]}
    writer = TrackingSidecarWriter(tmp_path)
    writer.start_episode(0)
    writer.append_packets([packet])
    path = writer.finish_episode(status="recorded")
    assert path is not None

    reconstructed = load_tracking_sidecar(path)
    assert reconstructed == [raw]
    table = pd.read_parquet(path)
    assert table.iloc[0]["body_joint_count"] == 84
    assert len(table.iloc[0]["body_joint_poses"]) == 84
    assert (
        list(table.iloc[0]["body_joint_location_flags"])
        == raw["body"]["jointLocationFlags"]
    )
    assert discover_tracking_sidecars(tmp_path, episode_index=0) == (path,)


def test_sidecar_labels_new_epoch_after_source_restart(tmp_path):
    _, first = _packet(7)
    _, restarted = _packet(2)
    writer = TrackingSidecarWriter(tmp_path)
    writer.set_frame_calibration(
        {"calibration_hash": "a" * 64}, reason="initial_calibration"
    )
    writer.start_episode(0)
    writer.append_packets([first, restarted])
    event = writer.consume_frame_epoch_change()
    assert event is not None
    assert event.index == 1
    assert event.reason == "source_sequence_restarted"
    path = writer.finish_episode(status="discarded")
    assert path is not None
    table = pd.read_parquet(path)
    assert table["frame_epoch"].tolist() == [0, 1]
    assert table.iloc[1]["frame_calibration_hash"] == "a" * 64
    manifest = json.loads((tmp_path / "raw/tracking/manifest.json").read_text())
    assert manifest["frame_epochs"][-1]["reason"] == "source_sequence_restarted"


def test_frame_epoch_tracker_marks_reconnect_and_calibration_change():
    tracker = FrameEpochTracker()
    tracker.observe_connection_count(1)
    assert tracker.consume_change() is None
    tracker.observe_connection_count(2)
    reconnect = tracker.consume_change()
    assert reconnect is not None
    assert reconnect.reason == "tracking_transport_reconnected"
    tracker.set_calibration("first", reason="initial")
    tracker.set_calibration("second", reason="recalibrated")
    event = tracker.consume_change()
    assert event is not None
    assert event.reason == "recalibrated"
    assert tracker.index == 2


def test_interrupted_jsonl_is_recovered_without_dropping_packets(tmp_path):
    _, packet = _packet(9)
    writer = TrackingSidecarWriter(tmp_path)
    writer.start_episode(3)
    writer.append_packets([packet])
    assert writer._journal is not None
    writer._journal.flush()
    writer._journal.write('{"truncated":')
    writer._journal.flush()
    writer._journal.close()
    writer._journal = None

    recovered_writer = TrackingSidecarWriter(tmp_path)
    paths = discover_tracking_sidecars(tmp_path, episode_index=3)
    assert len(paths) == 1
    assert "interrupted" in paths[0].parts
    assert load_tracking_sidecar(paths[0])[0]["seq"] == 9
    manifest = json.loads((tmp_path / "raw/tracking/manifest.json").read_text())
    assert manifest["files"][0]["status"] == "interrupted"
    assert manifest["files"][0]["recovery_truncated_lines"] == 1
    recovered_writer.close()


def test_sidecar_streams_multiple_parquet_row_groups_exactly_once(tmp_path):
    packets = []
    for sequence in range(1030):
        raw = make_tracking_packet_fixture(70, seq=sequence, active=False)
        packets.append(
            parse_tracking_packet(
                raw,
                pc_monotonic_ns=sequence + 1,
                receive_sequence=sequence + 1,
            )
        )
    writer = TrackingSidecarWriter(tmp_path)
    writer.start_episode(0)
    assert writer.append_packets(packets) == 1030
    path = writer.finish_episode(status="recorded")
    assert path is not None
    table = pd.read_parquet(path, columns=["source_sequence", "receive_sequence"])
    assert table["source_sequence"].tolist() == list(range(1030))
    assert table["receive_sequence"].tolist() == list(range(1, 1031))


def test_provider_session_manifest_is_stored_once_and_referenced(tmp_path):
    _, packet = _packet(12)

    class Provider:
        def session_manifest(self):
            return {
                "packetType": "session_manifest",
                "sessionId": "session-12",
                "build": {"commit": "abc123"},
            }

        def drain_packets(self):
            return [packet]

    writer = TrackingSidecarWriter(tmp_path)
    writer.start_episode(0)
    assert writer.drain_provider(Provider()) == 1
    assert writer.drain_provider(Provider()) == 1
    writer.finish_episode(status="recorded")

    records = (
        (tmp_path / "raw/tracking/session_manifests.jsonl").read_text().splitlines()
    )
    assert len(records) == 1
    assert json.loads(records[0])["sessionId"] == "session-12"
    manifest = json.loads((tmp_path / "raw/tracking/manifest.json").read_text())
    assert manifest["session_manifests"] == "raw/tracking/session_manifests.jsonl"


def test_derived_writer_preserves_optional_body_and_sidecar_when_requested(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    _, packet = _packet(11)
    sidecar = TrackingSidecarWriter(source_root)
    sidecar.start_episode(0)
    sidecar.append_packets([packet])
    sidecar.drain_provider(
        type(
            "Provider",
            (),
            {
                "session_manifest": lambda self: {"sessionId": "derived-source"},
                "drain_packets": lambda self: [],
            },
        )()
    )
    sidecar.finish_episode(status="recorded")

    body = np.full((2, 25, 7), np.nan, dtype=np.float32)
    body[:, 0] = [0, 0, 1, 0, 0, 0, 1]
    features = canonical_body_features()
    source_info = {
        "features": {
            "observation.body.joint_pose": features["observation.body.joint_pose"]
        },
        "handumi": {
            "tracking_schema": "handumi_tracking_v2",
            "body_schema": "handumi_canonical_25_v1",
        },
    }
    episode = EpisodeResult(
        episode_index=0,
        states=np.zeros((2, 3), dtype=np.float32),
        actions=np.ones((2, 3), dtype=np.float32),
        task="test",
        optional_observations={"observation.body.joint_pose": body},
    )
    output = tmp_path / "derived"
    write_dataset(
        output_root=output,
        source_root=source_root,
        source_info=source_info,
        episodes=[episode],
        robot_type="test_robot",
        joint_names=["a", "b", "c"],
        fps=30,
        preserve_tracking_sidecars=True,
    )

    table = pd.read_parquet(output / "data/chunk-000/file-000.parquet")
    assert len(table.iloc[0]["observation.state"]) == 3
    restored_body = np.stack(
        [np.stack(value) for value in table["observation.body.joint_pose"]]
    )
    assert restored_body.shape == (2, 25, 7)
    assert np.isnan(restored_body[:, 1:]).all()
    info = load_info(output)
    assert info["features"]["observation.body.joint_pose"]["shape"] == [25, 7]
    assert info["handumi"]["tracking_schema"] == "handumi_tracking_v2"
    assert discover_tracking_sidecars(output, episode_index=0)
    manifest = json.loads((output / "raw/tracking/manifest.json").read_text())
    assert manifest["session_manifests"] == "raw/tracking/session_manifests.jsonl"
    assert (
        json.loads((output / manifest["session_manifests"]).read_text().strip())[
            "sessionId"
        ]
        == "derived-source"
    )


def test_controller_only_recording_loads_none_instead_of_invented_pose(tmp_path):
    class Table:
        column_names = ["observation.state", "observation.valid"]

        def __getitem__(self, key):
            if key == "observation.state":
                return np.zeros((2, 16), dtype=np.float32)
            return np.ones((2, len(TRACKING_VALIDITY_NAMES)), dtype=np.int64)

    class Dataset:
        fps = 30
        hf_dataset = Table()
        root = tmp_path
        meta = type("Meta", (), {"info": _current_dataset_info()})()

    with mock.patch("handumi.dataset.reader.open_dataset", return_value=Dataset()):
        episode = load_raw_episode(
            repo_id="local/old", root=tmp_path, episode=0, download_videos=False
        )
    assert episode.states.shape == (2, 16)
    assert episode.body is None
    assert episode.tracking_sidecars == ()


def test_mixed_derived_dataset_marks_old_episode_body_unavailable(tmp_path):
    body = np.full((2, 25, 7), np.nan, dtype=np.float32)
    body[:, 0] = [0, 0, 1, 0, 0, 0, 1]
    new_episode = EpisodeResult(
        episode_index=0,
        states=np.zeros((2, 2), dtype=np.float32),
        actions=np.zeros((2, 2), dtype=np.float32),
        task="new",
        optional_observations={"observation.body.joint_pose": body},
    )
    old_episode = EpisodeResult(
        episode_index=1,
        states=np.zeros((2, 2), dtype=np.float32),
        actions=np.zeros((2, 2), dtype=np.float32),
        task="old",
    )
    output = tmp_path / "mixed"
    write_dataset(
        output_root=output,
        source_root=tmp_path,
        source_info={"features": {}},
        episodes=[new_episode, old_episode],
        robot_type="test",
        joint_names=["a", "b"],
        fps=30,
    )
    old_table = pd.read_parquet(output / "data/chunk-000/file-001.parquet")
    old_body = np.stack(
        [np.stack(value) for value in old_table["observation.body.joint_pose"]]
    )
    assert np.isnan(old_body).all()


def test_reader_discovers_optional_body_columns(tmp_path):
    class Table:
        column_names = [
            "observation.state",
            "observation.valid",
            "observation.body.joint_pose",
        ]

        def __getitem__(self, key):
            if key == "observation.state":
                return np.zeros((2, 16), dtype=np.float32)
            if key == "observation.valid":
                return np.ones((2, len(TRACKING_VALIDITY_NAMES)), dtype=np.int64)
            return [np.zeros((25, 7), dtype=np.float32) for _ in range(2)]

    class Dataset:
        fps = 30
        hf_dataset = Table()
        root = tmp_path
        meta = type("Meta", (), {"info": _current_dataset_info()})()

    with mock.patch("handumi.dataset.reader.open_dataset", return_value=Dataset()):
        episode = load_raw_episode(
            repo_id="local/new", root=tmp_path, episode=0, download_videos=False
        )
    assert episode.body is not None
    assert episode.body.signals["observation.body.joint_pose"].shape == (2, 25, 7)


def test_lerobot_round_trip_accepts_nan_body_arrays_and_masks(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "lerobot"
    features = build_features([], 64, 48, False)
    dataset = LeRobotDataset.create(
        repo_id="local/data001-test",
        fps=30,
        root=root,
        robot_type="handumi_raw",
        features=features,
        use_videos=False,
    )
    frame: dict[str, object] = {
        key: np.zeros(tuple(feature["shape"]), dtype=np.dtype(feature["dtype"]))
        for key, feature in features.items()
    }
    body = CanonicalBodyFrame.empty()
    body.joint_pose[0] = [0, 0, 1, 0, 0, 0, 1]
    body.position_valid[0] = 1
    body.orientation_valid[0] = 1
    provenance_values = [int(value) for value in CanonicalProvenance]
    body.provenance[:] = np.resize(provenance_values, 25)
    frame.update(body.observation())
    state = np.asarray(frame["observation.state"])
    state[3:7] = [0, 0, 0, 1]
    state[10:14] = [0, 0, 0, 1]
    frame["action"] = state.copy()
    frame["task"] = "schema round trip"
    dataset.add_frame(frame)
    dataset.save_episode()
    dataset.finalize()

    table = pd.read_parquet(root / "data/chunk-000/file-000.parquet")
    pose = np.stack(table["observation.body.joint_pose"].iloc[0])
    assert pose.shape == (25, 7)
    np.testing.assert_allclose(pose[0], [0, 0, 1, 0, 0, 0, 1])
    assert np.isnan(pose[1:]).all()
    stored_provenance = np.asarray(
        table["observation.body.provenance"].iloc[0], dtype=np.int64
    )
    assert set(stored_provenance) == set(provenance_values)
