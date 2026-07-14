"""Capture and analyze raw Quest tracking packets for platform qualification.

This is investigation tooling, not the production body schema. Sender payloads
are preserved unchanged so runtime evidence can be reanalyzed after the wire
contract evolves. All Quest poses, including body joints, are runtime estimates.
This module does not estimate or report anatomical center of mass.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from handumi.tracking.meta_quest import MetaQuestConfig, MetaQuestReceiver, QuestFrame

PROBE_SCHEMA = "handumi_quest_probe_v1"


def _finite_number(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(float(value)) else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _nested_int(packet: Mapping[str, Any], paths: tuple[tuple[str, ...], ...]) -> int | None:
    for path in paths:
        value: object = packet
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                break
            value = value[key]
        else:
            number = _finite_number(value)
            if number is not None:
                return int(number)
    return None


def _body_sample_time_ns(packet: Mapping[str, Any]) -> int | None:
    """Read diagnostic body time without declaring a production wire schema."""
    return _nested_int(
        packet,
        (
            ("body", "sourceTimeNs"),
            ("body", "sampleTimeNs"),
            ("body", "xrTime"),
            ("bodySampleTimeNs",),
            ("bodyXrTime",),
        ),
    )


@dataclass
class ProbeCapture:
    """Append-only JSONL writer called from ``MetaQuestReceiver``'s RX thread."""

    stream: TextIO
    manifest_stream: TextIO | None = None
    metrics_provider: Callable[[], Mapping[str, Any]] = field(default=lambda: {})
    flush_every: int = 1
    count: int = 0
    manifest_count: int = 0

    def record(self, frame: QuestFrame) -> None:
        self.record_raw(
            frame.raw,
            int(frame.pc_monotonic_ns),
            int(frame.receive_sequence),
        )

    def record_raw(
        self,
        packet: dict[str, Any],
        pc_receive_time_ns: int,
        pc_receive_sequence: int,
    ) -> None:
        metrics = self.metrics_provider()
        envelope = {
            "probe_schema": PROBE_SCHEMA,
            "capture_index": self.count,
            "pc_receive_sequence": int(pc_receive_sequence),
            "pc_receive_time_ns": int(pc_receive_time_ns),
            "sync": {
                "clock_synced": metrics.get("rtt_ns") is not None,
                "clock_offset_ns": _finite_number(metrics.get("offset_ns")),
                "rtt_ns": _finite_number(metrics.get("rtt_ns")),
            },
            "packet": packet,
        }
        self.stream.write(json.dumps(envelope, separators=(",", ":"), allow_nan=False))
        self.stream.write("\n")
        self.count += 1
        if packet.get("packetType") == "session_manifest":
            self.manifest_count += 1
            if self.manifest_stream is not None:
                self.manifest_stream.write(
                    json.dumps(packet, separators=(",", ":"), allow_nan=False)
                )
                self.manifest_stream.write("\n")
                self.manifest_stream.flush()
        if self.flush_every > 0 and self.count % self.flush_every == 0:
            self.stream.flush()


@dataclass
class AdbHealthSampler:
    """Periodically preserve raw Quest health/lifecycle evidence during a run."""

    output_path: Path
    logcat_path: Path
    adb_path: str = "adb"
    package: str = "com.handumi.questapp.bodyprobe"
    interval_s: float = 5.0
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _logcat_process: subprocess.Popen[str] | None = field(default=None, init=False)
    _logcat_stream: TextIO | None = field(default=None, init=False)

    def start(self) -> None:
        if self.interval_s <= 0:
            raise ValueError("ADB health interval must be greater than zero")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        try:
            self._logcat_stream = self.logcat_path.open("w", encoding="utf-8")
            self._logcat_process = subprocess.Popen(  # noqa: S603 - explicit adb binary
                [
                    self.adb_path,
                    "logcat",
                    "-v",
                    "epoch",
                    "Unity:V",
                    "OVRPlugin:V",
                    "ActivityManager:I",
                    "ActivityTaskManager:I",
                    "AndroidRuntime:E",
                    "*:S",
                ],
                stdout=self._logcat_stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as exc:
            if self._logcat_stream is not None:
                self._logcat_stream.close()
                self._logcat_stream = None
            self._write_sample({"logcat_start_error": str(exc)})
        self._thread = threading.Thread(
            target=self._run,
            name="quest_adb_health",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_s + 1.0))
            self._thread = None
        if self._logcat_process is not None:
            self._logcat_process.terminate()
            try:
                self._logcat_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._logcat_process.kill()
                self._logcat_process.wait(timeout=2.0)
            self._logcat_process = None
        if self._logcat_stream is not None:
            self._logcat_stream.close()
            self._logcat_stream = None

    def sample(self) -> dict[str, Any]:
        pid_result = self._adb_shell("pidof", self.package)
        pid = pid_result["stdout"].strip().split()
        pid_value = pid[0] if pid else ""
        return {
            "record_type": "adb_health",
            "pc_time_utc": datetime.now(UTC).isoformat(),
            "pc_monotonic_ns": time.monotonic_ns(),
            "package": self.package,
            "pid": pid_value or None,
            "battery": self._adb_shell("dumpsys", "battery"),
            "thermal": self._adb_shell("dumpsys", "thermalservice"),
            "memory": self._adb_shell("dumpsys", "meminfo", self.package),
            "cpu": self._adb_shell("top", "-b", "-n", "1", "-p", pid_value)
            if pid_value
            else {"returncode": None, "stdout": "", "stderr": "process not running"},
            "lifecycle": self._adb_shell("dumpsys", "activity", "activities"),
            "process_lookup": pid_result,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            self._write_sample(self.sample())
            self._stop.wait(self.interval_s)

    def _write_sample(self, sample: Mapping[str, Any]) -> None:
        with self.output_path.open("a", encoding="utf-8") as stream:
            json.dump(sample, stream, separators=(",", ":"), allow_nan=False)
            stream.write("\n")

    def _adb_shell(self, *args: str) -> dict[str, Any]:
        try:
            result = subprocess.run(  # noqa: S603 - explicit adb binary and argv
                [self.adb_path, "shell", *args],
                capture_output=True,
                text=True,
                timeout=10.0,
                check=False,
            )
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"returncode": None, "stdout": "", "stderr": str(exc)}


def iter_probe_records(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed probe JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Probe record at {path}:{line_number} is not an object")
            yield record


def _rate_hz(times_ns: list[int]) -> float | None:
    if len(times_ns) < 2 or times_ns[-1] <= times_ns[0]:
        return None
    return (len(times_ns) - 1) * 1e9 / (times_ns[-1] - times_ns[0])


def analyze_probe_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    receive_times: list[int] = []
    device_times: list[int] = []
    body_times: list[int] = []
    offsets: list[float] = []
    rtts: list[float] = []
    sample_ages: list[float] = []
    body_sample_ages: list[float] = []
    source_sequences: list[int] = []
    manifests: list[Mapping[str, Any]] = []
    joint_sets: Counter[str] = Counter()
    joint_counts: Counter[int] = Counter()
    calibration_states: Counter[str] = Counter()
    fidelity_states: Counter[str] = Counter()
    confidences: list[float] = []
    skeleton_revisions: list[int] = []
    joint_stats: dict[tuple[int, str], Counter[str]] = {}
    body_intervals: list[dict[str, Any]] = []
    active_interval: dict[str, Any] | None = None
    body_record_count = 0
    body_active_count = 0
    missing_body_source_time = 0
    packet_count = 0

    for record in records:
        packet = record.get("packet")
        if not isinstance(packet, Mapping):
            continue
        if packet.get("packetType") == "session_manifest":
            manifests.append(packet)
            continue
        packet_count += 1
        receive_time = _finite_number(record.get("pc_receive_time_ns"))
        if receive_time is not None:
            receive_times.append(int(receive_time))
        device_time = _nested_int(packet, (("ovrTimeNs",), ("deviceTimeNs",)))
        if device_time is not None and device_time > 0:
            device_times.append(device_time)
        body_time = _body_sample_time_ns(packet)
        if body_time is not None and body_time > 0 and (
            not body_times or body_time != body_times[-1]
        ):
            body_times.append(body_time)
        if "seq" in packet:
            sequence = _finite_number(packet.get("seq"))
            if sequence is not None:
                source_sequences.append(int(sequence))
        sync = record.get("sync")
        clock_synced = False
        offset: int | float | None = None
        if isinstance(sync, Mapping):
            offset = _finite_number(sync.get("clock_offset_ns"))
            rtt = _finite_number(sync.get("rtt_ns"))
            clock_synced = sync.get("clock_synced") is True
            if clock_synced and offset is not None:
                offsets.append(float(offset))
            if clock_synced and rtt is not None:
                rtts.append(float(rtt))
            if (
                clock_synced
                and receive_time is not None
                and device_time is not None
                and offset is not None
            ):
                sample_ages.append(float(receive_time - (device_time + offset)))
            if (
                clock_synced
                and receive_time is not None
                and body_time is not None
                and body_time > 0
                and offset is not None
            ):
                body_sample_ages.append(float(receive_time - (body_time + offset)))

        body = packet.get("body")
        if not isinstance(body, Mapping):
            continue
        body_record_count += 1
        is_active = body.get("active") is True
        if is_active:
            body_active_count += 1
        if body_time is None or body_time <= 0:
            missing_body_source_time += 1

        capture_index = _nested_int(record, (("capture_index",),))
        if active_interval is None or active_interval["active"] != is_active:
            if active_interval is not None:
                body_intervals.append(active_interval)
            active_interval = {
                "active": is_active,
                "start_capture_index": capture_index,
                "end_capture_index": capture_index,
                "start_pc_receive_time_ns": int(receive_time)
                if receive_time is not None
                else None,
                "end_pc_receive_time_ns": int(receive_time)
                if receive_time is not None
                else None,
                "pose_samples": 1,
            }
        else:
            active_interval["end_capture_index"] = capture_index
            active_interval["end_pc_receive_time_ns"] = (
                int(receive_time) if receive_time is not None else None
            )
            active_interval["pose_samples"] += 1

        active_joint_set = body.get("activeJointSet")
        if isinstance(active_joint_set, str):
            joint_sets[active_joint_set] += 1
        joint_count = _finite_number(body.get("jointCount"))
        if joint_count is not None:
            joint_counts[int(joint_count)] += 1
        calibration = body.get("calibrationState")
        if isinstance(calibration, str):
            calibration_states[calibration] += 1
        fidelity = body.get("fidelity")
        if isinstance(fidelity, str):
            fidelity_states[fidelity] += 1
        confidence = _finite_number(body.get("confidence"))
        if confidence is not None:
            confidences.append(float(confidence))
        revision = _finite_number(body.get("skeletonRevision"))
        if revision is not None:
            skeleton_revisions.append(int(revision))

        joints = body.get("joints")
        if not isinstance(joints, list):
            compact_flags = body.get("jointLocationFlags")
            compact_names = body.get("jointNames")
            if not isinstance(compact_flags, list):
                continue
            joints = [
                {
                    "index": index,
                    "name": (
                        compact_names[index]
                        if isinstance(compact_names, list)
                        and index < len(compact_names)
                        else f"Joint_{index}"
                    ),
                    "locationFlags": flags,
                }
                for index, flags in enumerate(compact_flags)
            ]
        for fallback_index, joint in enumerate(joints):
            if not isinstance(joint, Mapping):
                continue
            index = _nested_int(joint, (("index",),))
            if index is None:
                index = fallback_index
            raw_name = joint.get("name")
            name = raw_name if isinstance(raw_name, str) else f"Joint_{index}"
            stats = joint_stats.setdefault((index, name), Counter())
            stats["samples"] += 1
            flags = _finite_number(
                joint.get("locationFlags", joint.get("flags"))
            )
            if flags is None:
                continue
            flag_bits = int(flags)
            stats["flag_samples"] += 1
            stats["orientation_valid"] += int(bool(flag_bits & 0x1))
            stats["position_valid"] += int(bool(flag_bits & 0x2))
            stats["orientation_tracked"] += int(bool(flag_bits & 0x4))
            stats["position_tracked"] += int(bool(flag_bits & 0x8))
            stats["pose_valid"] += int((flag_bits & 0x3) == 0x3)
            stats["pose_tracked"] += int((flag_bits & 0xC) == 0xC)

    if active_interval is not None:
        body_intervals.append(active_interval)

    gaps = duplicates = resets = out_of_order = 0
    for previous, current in zip(source_sequences, source_sequences[1:], strict=False):
        delta = current - previous
        if delta > 1:
            gaps += delta - 1
        elif delta == 0:
            duplicates += 1
        elif delta < 0:
            if current in (0, 1) and previous > current + 1:
                resets += 1
            else:
                out_of_order += 1

    expected = len(source_sequences) + gaps
    loss_fraction = gaps / expected if expected else None
    interarrival = [
        float(current - previous)
        for previous, current in zip(receive_times, receive_times[1:], strict=False)
        if current > previous
    ]
    joint_percentages: list[dict[str, Any]] = []
    for (index, name), stats in sorted(joint_stats.items()):
        denominator = stats["flag_samples"]

        def percentage(key: str) -> float | None:
            return stats[key] / denominator if denominator else None

        joint_percentages.append(
            {
                "index": index,
                "name": name,
                "samples": stats["samples"],
                "flag_samples": denominator,
                "orientation_valid_fraction": percentage("orientation_valid"),
                "position_valid_fraction": percentage("position_valid"),
                "orientation_tracked_fraction": percentage("orientation_tracked"),
                "position_tracked_fraction": percentage("position_tracked"),
                "pose_valid_fraction": percentage("pose_valid"),
                "pose_tracked_fraction": percentage("pose_tracked"),
            }
        )

    revision_changes = sum(
        current != previous
        for previous, current in zip(
            skeleton_revisions, skeleton_revisions[1:], strict=False
        )
    )
    return {
        "probe_schema": PROBE_SCHEMA,
        "manifest_count": len(manifests),
        "session_manifests": list(manifests),
        "packet_count": packet_count,
        "receive_rate_hz": _rate_hz(receive_times),
        "device_rate_hz": _rate_hz(device_times),
        "body_update_rate_hz": _rate_hz(body_times),
        "receive_interarrival_ns": {
            "sample_count": len(interarrival),
            "median": statistics.median(interarrival) if interarrival else None,
            "p95": _percentile(interarrival, 0.95),
            "standard_deviation": (
                statistics.pstdev(interarrival) if len(interarrival) > 1 else None
            ),
            "maximum": max(interarrival) if interarrival else None,
        },
        "source_sequence": {
            "available": bool(source_sequences),
            "sample_count": len(source_sequences),
            "missing_packets": gaps if source_sequences else None,
            "loss_fraction": loss_fraction,
            "duplicates": duplicates if source_sequences else None,
            "resets": resets if source_sequences else None,
            "out_of_order": out_of_order if source_sequences else None,
            "resets_or_out_of_order": resets + out_of_order
            if source_sequences
            else None,
        },
        "clock_offset_ns": {
            "sample_count": len(offsets),
            "median": statistics.median(offsets) if offsets else None,
            "standard_deviation": statistics.pstdev(offsets) if len(offsets) > 1 else None,
            "minimum": min(offsets) if offsets else None,
            "maximum": max(offsets) if offsets else None,
        },
        "rtt_ns": {
            "sample_count": len(rtts),
            "median": statistics.median(rtts) if rtts else None,
            "p95": _percentile(rtts, 0.95),
            "maximum": max(rtts) if rtts else None,
        },
        "mapped_sample_age_ns": {
            "sample_count": len(sample_ages),
            "median": statistics.median(sample_ages) if sample_ages else None,
            "p95": _percentile(sample_ages, 0.95),
            "minimum": min(sample_ages) if sample_ages else None,
            "maximum": max(sample_ages) if sample_ages else None,
        },
        "mapped_body_sample_age_ns": {
            "sample_count": len(body_sample_ages),
            "median": statistics.median(body_sample_ages)
            if body_sample_ages
            else None,
            "p95": _percentile(body_sample_ages, 0.95),
            "minimum": min(body_sample_ages) if body_sample_ages else None,
            "maximum": max(body_sample_ages) if body_sample_ages else None,
        },
        "body": {
            "record_count": body_record_count,
            "active_pose_packets": body_active_count,
            "inactive_pose_packets": body_record_count - body_active_count,
            "active_fraction": body_active_count / body_record_count
            if body_record_count
            else None,
            "missing_source_time_packets": missing_body_source_time,
            "joint_set_counts": dict(sorted(joint_sets.items())),
            "joint_count_counts": {
                str(key): value for key, value in sorted(joint_counts.items())
            },
            "confidence": {
                "sample_count": len(confidences),
                "median": statistics.median(confidences) if confidences else None,
                "minimum": min(confidences) if confidences else None,
                "maximum": max(confidences) if confidences else None,
            },
            "calibration_state_counts": dict(sorted(calibration_states.items())),
            "fidelity_counts": dict(sorted(fidelity_states.items())),
            "skeleton_revision": {
                "sample_count": len(skeleton_revisions),
                "changes": revision_changes,
                "minimum": min(skeleton_revisions) if skeleton_revisions else None,
                "maximum": max(skeleton_revisions) if skeleton_revisions else None,
            },
            "active_intervals": body_intervals,
            "joints": joint_percentages,
        },
        "limitations": [
            "Quest poses are platform-provided estimates, not direct measurements.",
            "Packet timing and loss statistics do not establish pose accuracy.",
            "Anatomical center of mass is not measured or estimated by this probe.",
        ],
    }


def analyze_probe_file(path: str | Path) -> dict[str, Any]:
    return analyze_probe_records(iter_probe_records(path))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def capture(args: argparse.Namespace) -> int:
    if args.duration_s <= 0:
        raise SystemExit("--duration-s must be greater than zero")
    if args.flush_every < 0:
        raise SystemExit("--flush-every must be zero or greater")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "quest_packets.jsonl"
    manifest_path = output_dir / "session_manifests.jsonl"
    health_path = output_dir / "quest_health.jsonl"
    logcat_path = output_dir / "quest_logcat.txt"
    health_sampler = (
        AdbHealthSampler(
            output_path=health_path,
            logcat_path=logcat_path,
            adb_path=args.adb_path,
            package=args.adb_package,
            interval_s=args.adb_interval_s,
        )
        if args.adb_health
        else None
    )

    config = MetaQuestConfig.from_yaml(args.config)
    with (
        raw_path.open("w", encoding="utf-8") as stream,
        manifest_path.open("w", encoding="utf-8") as manifest_stream,
    ):
        session = ProbeCapture(
            stream=stream,
            manifest_stream=manifest_stream,
            flush_every=args.flush_every,
        )
        receiver = MetaQuestReceiver(config, on_raw_message=session.record_raw)
        session.metrics_provider = receiver.metrics
        started = datetime.now(UTC)
        if health_sampler is not None:
            health_sampler.start()
        receiver.start()
        try:
            deadline = time.monotonic() + args.duration_s
            while time.monotonic() < deadline:
                time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
        except KeyboardInterrupt:
            pass
        finally:
            receiver.stop()
            if health_sampler is not None:
                health_sampler.stop()

    context = {
        "probe_schema": PROBE_SCHEMA,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "configured_duration_s": args.duration_s,
        "quest_ip": config.quest_ip,
        "tcp_port": config.tcp_port,
        "sync_port": config.sync_port,
        "raw_packets": raw_path.name,
        "session_manifests": manifest_path.name,
        "manifest_count": session.manifest_count,
        "adb_health_enabled": health_sampler is not None,
        "quest_health": health_path.name if health_sampler is not None else None,
        "quest_logcat": logcat_path.name if health_sampler is not None else None,
    }
    summary = analyze_probe_file(raw_path)
    _write_json(output_dir / "capture_context.json", context)
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


def analyze(args: argparse.Namespace) -> int:
    summary = analyze_probe_file(args.input)
    if args.output is not None:
        _write_json(Path(args.output), summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    capture_parser = commands.add_parser("capture", help="Capture raw Quest packets")
    capture_parser.add_argument(
        "--config", type=Path, default=Path("configs/tracking_meta_quest.yaml")
    )
    capture_parser.add_argument(
        "--adb-health",
        action="store_true",
        help="Capture Quest battery, thermal, CPU, memory, lifecycle, and logcat evidence",
    )
    capture_parser.add_argument("--adb-path", default="adb")
    capture_parser.add_argument(
        "--adb-package", default="com.handumi.questapp.bodyprobe"
    )
    capture_parser.add_argument("--adb-interval-s", type=float, default=5.0)
    capture_parser.add_argument("--duration-s", type=float, default=60.0)
    capture_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New or existing directory for JSONL and summaries",
    )
    capture_parser.add_argument(
        "--flush-every",
        type=int,
        default=1,
        help="Flush after this many packets; zero flushes only when closed",
    )
    capture_parser.set_defaults(func=capture)

    analyze_parser = commands.add_parser("analyze", help="Analyze captured JSONL")
    analyze_parser.add_argument("input", type=Path)
    analyze_parser.add_argument("--output", type=Path)
    analyze_parser.set_defaults(func=analyze)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
