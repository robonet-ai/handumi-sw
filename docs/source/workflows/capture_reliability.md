# Capture reliability and recovery

HandUMI capture is a research-preview workflow. Its timing, body-derived
signals, contact/support estimates, and synchronization diagnostics are not
scientifically validated and must never be used as physical safety interlocks.

## Capture profiles

The software recognizes two three-camera profiles. `10 Hz` dataset rows with
`30 FPS` cameras is the supported software-only profile. Requested `30 Hz`
dataset rows with three `30 FPS` cameras is experimental until hardware and
long-soak evidence establish adequate margin. Other combinations are also
experimental. The recorder warns before capture and rejects a timed episode
when it produces fewer than 98 percent of the requested rows.

These classifications describe synthetic/headless software evidence only.
They are not Quest, camera, encoder, thermal, radio, robot, or laboratory
evidence.

## Storage and finalization policy

The recorder checks free space before starting and periodically during each
episode. `--minimum-free-gb` sets the reserve (2 GiB by default), and
`--disk-check-interval-s` controls the periodic check. Disk-full, short-write,
encoder, serialization, and finalization failures fail closed.

New datasets are written below a hidden
`.NAME.handumi-inprogress-SESSION` directory. A successful close validates the
dataset, computes file checksums, writes `session-manifest.json`, fsyncs it,
and atomically renames the directory to the requested destination when the
filesystem supports atomic rename. Failed finalization is renamed to
`handumi-rejected`; stale staging directories found on the next run are
renamed to `handumi-recovered`. Rejected and recovered directories remain
for forensic review and are never presented as complete datasets.

The publishable session manifest contains relative file names, hashes,
aggregate profiling/drop data, logical configuration hashes, schema/runtime
versions, and completion status. It intentionally excludes network addresses,
USB serials, participant identifiers, room imagery, and absolute paths.

## Profiling and soak command

Per-stage monotonic latency, throughput, queue-depth, drop, timeout, and
failure counters cover tracking/clock alignment, body work, camera acquisition
and synchronization, video/dataset writes, Rerun, Viser, IK, finalization, and
checksums. Nonessential workers use bounded latest-value queues and cannot
block dataset capture indefinitely.

Run the software-only soak with machine-readable output:

```bash
uv run --locked handumi-soak \
  --duration-s 1800 --dataset-hz 10 --camera-fps 30 --camera-count 3 \
  --output /tmp/handumi-soak-10hz.json
```

Evaluate the requested 30 Hz profile separately by changing
`--dataset-hz 30`. A successful command exits zero only when the requested row
rate is maintained. Retain the JSON command context, source commit, duration,
rows, drops, maximum queue depths, resource start/end/peak values, sizes, and
SHA-256 values with release evidence.
