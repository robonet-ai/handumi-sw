"""Versioned, loss-aware tracking packet contract."""

from __future__ import annotations

import json
import threading
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, TextIO


TRACKING_PACKET_SCHEMA = "tracking_packet_v2"
TRACKING_PACKET_VERSION = 2


class SourceProvenance(str, Enum):
    PLATFORM_ESTIMATED = "PLATFORM_ESTIMATED"
    DEVICE_REPORTED = "DEVICE_REPORTED"
    EXTERNAL_TRACKER = "EXTERNAL_TRACKER"
    SYNTHETIC_TEST = "SYNTHETIC_TEST"
    UNKNOWN = "UNKNOWN"


class JointTrackingState(str, Enum):
    INVALID = "INVALID"
    VALID = "VALID"
    TRACKED = "TRACKED"


class TimestampQuality(str, Enum):
    UNAVAILABLE = "UNAVAILABLE"
    RECEIVE_ONLY = "RECEIVE_ONLY"
    MAPPED_UNBOUNDED = "MAPPED_UNBOUNDED"
    SYNCHRONIZED = "SYNCHRONIZED"
    DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"


class PacketLossReason(str, Enum):
    QUEUE_OVERFLOW = "QUEUE_OVERFLOW"
    MALFORMED_FRAME = "MALFORMED_FRAME"
    UNSUPPORTED_VERSION = "UNSUPPORTED_VERSION"
    SEQUENCE_GAP = "SEQUENCE_GAP"
    DUPLICATE = "DUPLICATE"
    OUT_OF_ORDER = "OUT_OF_ORDER"


@dataclass(frozen=True)
class TrackingTimestamps:
    source_time_ns: int = 0
    source_time_domain: str = ""
    mapped_pc_monotonic_ns: int = 0
    receive_time_ns: int = 0
    clock_offset_ns: int = 0
    rtt_ns: int = 0
    uncertainty_ns: int = 0
    quality: TimestampQuality = TimestampQuality.UNAVAILABLE


@dataclass(frozen=True)
class PoseChannel:
    pose: tuple[float, float, float, float, float, float, float]
    tracking_state: JointTrackingState
    location_flags: int = 0
    confidence: float = 0.0
    provenance: SourceProvenance = SourceProvenance.UNKNOWN


@dataclass(frozen=True)
class ControllerChannel:
    side: str
    pose: PoseChannel
    buttons: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JointSample:
    index: int
    name: str
    pose: tuple[float, float, float, float, float, float, float]
    location_flags: int
    tracking_state: JointTrackingState
    confidence: float
    provenance: SourceProvenance


@dataclass(frozen=True)
class BodyChannel:
    active: bool
    requested_joint_set: str
    active_joint_set: str
    joint_count: int
    joints: tuple[JointSample, ...]
    confidence: float
    calibration_state: str
    fidelity: str
    skeleton_revision: int
    source_time_ns: int
    source_time_domain: str
    timestamp_quality: TimestampQuality
    provenance: SourceProvenance
    observation_sequence: int | None = None
    is_new_observation: bool | None = None


@dataclass(frozen=True)
class HandChannel:
    side: str
    active: bool
    joints: tuple[JointSample, ...]
    provenance: SourceProvenance


@dataclass(frozen=True)
class ExternalTrackerChannel:
    tracker_id: str
    pose: PoseChannel
    velocity: tuple[float, ...] = ()
    acceleration: tuple[float, ...] = ()


@dataclass(frozen=True)
class TrackingPacket:
    schema: str
    source_schema_version: int
    source: str
    sequence: int | None
    receive_sequence: int
    coordinate_space: str
    timestamps: TrackingTimestamps
    hmd: PoseChannel | None = None
    controllers: tuple[ControllerChannel, ...] = ()
    body: BodyChannel | None = None
    hands: tuple[HandChannel, ...] = ()
    external_trackers: tuple[ExternalTrackerChannel, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PacketStreamStats:
    accepted: int
    delivered: int
    queued: int
    dropped: Mapping[str, int]


class TrackingPacketStream:
    """Bounded FIFO that accounts for every packet it cannot retain."""

    def __init__(self, max_packets: int = 2048) -> None:
        if max_packets <= 0:
            raise ValueError("max_packets must be positive")
        self._queue: deque[TrackingPacket] = deque()
        self._max_packets = int(max_packets)
        self._latest: TrackingPacket | None = None
        self._accepted = 0
        self._delivered = 0
        self._dropped: Counter[str] = Counter()
        self._lock = threading.Lock()

    def publish(self, packet: TrackingPacket) -> None:
        with self._lock:
            self._accepted += 1
            self._latest = packet
            if len(self._queue) >= self._max_packets:
                self._queue.popleft()
                self._dropped[PacketLossReason.QUEUE_OVERFLOW.value] += 1
            self._queue.append(packet)

    def record_drop(self, reason: PacketLossReason, count: int = 1) -> None:
        if count <= 0:
            return
        with self._lock:
            self._dropped[reason.value] += int(count)

    def latest(self) -> TrackingPacket | None:
        with self._lock:
            return self._latest

    def drain(self, max_packets: int | None = None) -> list[TrackingPacket]:
        with self._lock:
            count = len(self._queue)
            if max_packets is not None:
                count = min(count, max(0, int(max_packets)))
            packets = [self._queue.popleft() for _ in range(count)]
            self._delivered += len(packets)
            return packets

    def stats(self) -> PacketStreamStats:
        with self._lock:
            return PacketStreamStats(
                accepted=self._accepted,
                delivered=self._delivered,
                queued=len(self._queue),
                dropped=dict(self._dropped),
            )


class PacketTrackingProvider(Protocol):
    device: str

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def latest_packet(self) -> TrackingPacket | None: ...

    def drain_packets(self, max_packets: int | None = None) -> list[TrackingPacket]: ...


def json_safe_value(value: Any) -> Any:
    """Convert packet source values, including numpy-like arrays, to JSON values."""
    if isinstance(value, Mapping):
        return {str(key): json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe_value(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe_value(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def tracking_packet_record(packet: TrackingPacket) -> dict[str, Any]:
    """Return the stable lossless JSON envelope for one normalized packet."""
    return {
        "recordType": "tracking_packet",
        "schema": packet.schema,
        "sourceSchemaVersion": packet.source_schema_version,
        "source": packet.source,
        "sequence": packet.sequence,
        "receiveSequence": packet.receive_sequence,
        "packet": json_safe_value(packet.raw),
    }


def drain_tracking_packets_jsonl(
    stream: TrackingPacketStream,
    output: TextIO,
    max_packets: int | None = None,
) -> int:
    """Drain accepted packets in FIFO order into a loss-auditable JSONL stream.

    The original source mapping is nested unchanged so additive fields survive.
    ``receiveSequence`` provides an always-present workstation ordering key even
    for legacy senders that do not publish their own sequence.
    """
    packets = stream.drain(max_packets)
    for packet in packets:
        record = tracking_packet_record(packet)
        output.write(json.dumps(record, separators=(",", ":"), allow_nan=True))
        output.write("\n")
    return len(packets)


def tracking_state_from_location_flags(flags: int) -> JointTrackingState:
    if flags & 0xC == 0xC:
        return JointTrackingState.TRACKED
    if flags & 0x3 == 0x3:
        return JointTrackingState.VALID
    return JointTrackingState.INVALID


__all__ = [
    "BodyChannel",
    "ControllerChannel",
    "ExternalTrackerChannel",
    "HandChannel",
    "JointSample",
    "JointTrackingState",
    "PacketLossReason",
    "PacketStreamStats",
    "PacketTrackingProvider",
    "PoseChannel",
    "SourceProvenance",
    "TimestampQuality",
    "TRACKING_PACKET_SCHEMA",
    "TRACKING_PACKET_VERSION",
    "TrackingPacket",
    "TrackingPacketStream",
    "TrackingTimestamps",
    "tracking_state_from_location_flags",
    "drain_tracking_packets_jsonl",
    "json_safe_value",
    "tracking_packet_record",
]
