"""Machine- and human-readable TEST-001 evidence reports."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from pathlib import Path
from typing import Mapping

from handumi.reliability import atomic_write_json, sha256_file

REPORT_SCHEMA = "handumi_test001_report_v1"
VALIDATION_STATUS = "not scientifically validated"


def generate_report(
    output_dir: Path,
    *,
    metrics: Mapping[str, float | int | str],
    configuration: Mapping[str, object],
    inputs: list[Path],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_hashes = [{"name": path.name, "sha256": sha256_file(path)} for path in inputs]
    config_payload = json.dumps(
        configuration, sort_keys=True, separators=(",", ":")
    ).encode()
    evidence: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "validation_status": VALIDATION_STATUS,
        "metrics": dict(metrics),
        "configuration": dict(configuration),
        "configuration_sha256": hashlib.sha256(config_payload).hexdigest(),
        "inputs": input_hashes,
        "runtime_versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    json_path = output_dir / "evidence.json"
    atomic_write_json(json_path, evidence)
    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value"])
        for key, value in sorted(metrics.items()):
            writer.writerow([key, value])
    markdown_path = output_dir / "report.md"
    lines = [
        "# HandUMI TEST-001 synthetic evidence",
        "",
        f"Status: **{VALIDATION_STATUS}**.",
        "",
        "This software report is not participant, mocap, force-plate, medical, ergonomic, or safety evidence.",
        "",
        "## Metrics",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in sorted(metrics.items()))
    markdown_path.write_text("\n".join(lines) + "\n")
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}
