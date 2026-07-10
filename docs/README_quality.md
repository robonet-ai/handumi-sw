# Recording Quality

HandUMI protects a recording in two stages. The online gate stops sustained
hardware failures; the offline validator rejects shorter or intermittent
defects that only become clear after inspecting the full episode.

## Synchronized Capture

Every recorder row has a shared `observation.sync.target_time_ns`. Sources are
selected from their native-rate buffers against that target:

- Quest uses `ovrTimeNs` plus the median UDP clock offset. Before the first
  accepted clock exchange, it falls back to the PC receive timestamp.
- OpenCV cameras preserve the timestamp assigned by their hardware read loop.
- Feetech apertures are sampled continuously at 100 Hz and timestamped at the
  midpoint of each serial read.

The default target is 40 ms behind real time, which gives asynchronous sensors
time to populate their buffers. Each source stores `sample_time_ns`, `age_ms`,
`sync_error_ms`, and `healthy`. Quest additionally stores device/receive times,
clock offset, clock sync state, raw `tracked`/`valid` flags, HMD pose, and the
workspace transform.

## Online Health Gate

Recording waits for two fresh, valid controllers. The current episode is
discarded when any of these conditions remains active for its configured
timeout:

- Either controller is untracked, invalid, stale, or too far from the row
  target for more than `--tracking-loss-timeout-s` (default 1 second).
- A camera frame is stale or outside `--max-sync-skew-s` for more than
  `--sensor-loss-timeout-s`.
- The Feetech stream is stale or outside the synchronization tolerance for
  more than `--sensor-loss-timeout-s`.

Disabled Feetech sensing is recorded with `enabled=0` and does not block the
gate. Short failures remain in the raw dataset with `healthy=0`; they are not
silently replaced with apparently healthy measurements.

## Offline Validation

```bash
handumi-validate \
  --repo-id local/handumi-demo \
  --root outputs/datasets/handumi-demo
```

The command writes `meta/handumi_quality.json`. Raw parquet and video files are
never moved or deleted. Every episode receives deterministic findings and an
`accepted` or `rejected` status.

Hard rejection checks include:

- Minimum episode duration.
- Total fraction of bad controller tracking, including repeated short losses.
- Camera and encoder health fractions.
- Per-source synchronization error fraction.
- Repeated source timestamps and full controller-pose freezes.
- Implausible translation speed and rotation steps over 90 degrees.
- Invalid quaternion, NaN, or infinite state values.

A single stationary hand and constant aperture are warnings by default because
they can be valid for unimanual or holding tasks. They can be promoted to hard
rejections in `configs/quality.yaml`.

`handumi-convert` applies this validator automatically and excludes rejected
episodes from the robot-specific output. It stores the source decisions in
`meta/source_quality.json`. Use `--skip-quality-filter` only when diagnosing a
known bad recording.
