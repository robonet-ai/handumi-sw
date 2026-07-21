# Record Demonstrations

Recording uses HandUMI directly and does not require a robot arm. The raw
controller poses, camera streams, gripper widths, and calibration metadata
remain available for later retargeting to any supported embodiment.

## Before Recording

Confirm that:

- Both gripper widths respond correctly from closed to fully open.
- Both controllers report valid tracking.
- Camera intrinsics and controller-camera mounts still match the hardware.
- The current session/table calibration was created for the same `--device`
  and has been visualized in Rerun.
- The Controller-to-TCP calibration matches the installed HandUMI tool.

See [Setup and Calibration](setup.md) if any check fails.

Start with a short pilot:

```bash
handumi doctor --device meta
handumi record outputs/datasets/handumi-demo \
  --task "pick and place" \
  --session-calibration outputs/calibration/session.yaml \
  --cameras left_wrist,right_wrist,workspace \
  --rerun --clap-control \
  --episodes 3 \
  --episode-time-s 30
```

The usual device, cameras, resolution, FPS and target robot belong in the
optional `recording:` section of `configs/rig.yaml`. CLI values override those
defaults. Add `--dry-run` to resolve the complete plan, probe the encoder and
exit before opening any hardware.

Use `--device pico` and a PICO-created `--session-calibration` for PICO. Add
`--push-to-hub` only after confirming the pilot locally.

Do not connect or configure a robot arm for this step. A target embodiment can
be selected later during conversion or replay without modifying the raw
recording.

## Resume a Recording

Append more episodes to a finalized local dataset without repeating its
recording configuration:

```bash
handumi record outputs/datasets/handumi-demo --resume \
  --episodes 20 \
  --task "pick and place"
```

`--episodes` is the number of additional episodes, not the new total.
Resume requires an intact dataset from a previous graceful finalization and
loads the device, cameras, FPS, image format, calibrations, Feetech state and
robot profile from its `meta/info.json` snapshot. Explicit incompatible
overrides are rejected before hardware starts, including FPS, cameras, image
shapes, tracking schemas, calibrations, or target-robot metadata. The task text
may change so the same dataset can contain multiple tasks.

## Streaming Video Encoding

Video is encoded continuously while an episode is recorded. HandUMI probes the
local PyAV/FFmpeg encoders with a real MP4 before starting the tracking and
camera hardware, then reports the concrete selection, for example:

```text
Encoder: h264_nvenc (hardware, streaming, codec-managed threads).
```

The default `--encoder auto` tries a working hardware encoder first (NVIDIA
NVENC, Intel/AMD VAAPI or Quick Sync, or macOS VideoToolbox) and falls back to
H.264 on CPU. CPU encoding reserves one logical core and limits the threads
assigned to each camera so encoding does not starve capture.

Use `--encoder cpu` to force software encoding or `--encoder gpu` to require
hardware acceleration. `--vcodec <codec>` remains an advanced explicit
override; do not combine an explicit codec with `--encoder cpu` or
`--encoder gpu`.

Streaming writes frames directly to MP4 and calculates image statistics as
frames arrive instead of writing and rereading temporary PNG files. If an
encoder crashes, its queue overflows, a video is empty, or its frame count does
not match the episode, HandUMI discards the episode before appending its rows to
Parquet. `--encoder-threads` and `--encoder-queue-size` are advanced diagnostic
overrides; increasing the queue does not fix an encoder that is consistently
slower than capture.

## Controls

- Right double clap: start or save the current episode.
- Left double clap: discard and restart the current episode.
- `Esc` or `Ctrl+C`: discard an active partial episode and stop.
- `--space-start`: allow keyboard start when clap control is unavailable.

The recorder waits for valid controllers and discards an episode after sustained tracking, camera, or encoder failure.

:::{dropdown} Synchronization and health gates
Every row uses one shared `observation.sync.target_time_ns`. Cameras, tracking,
and Feetech readings are selected from their native buffers against that
target. The default target is 40 ms behind real time (`--sync-lag-s 0.04`).

An episode is discarded after sustained controller loss
(`--tracking-loss-timeout-s`, default 1 second), or sustained camera/encoder
failure (`--sensor-loss-timeout-s`, default 1 second). Sources must also remain
inside `--max-sync-skew-s`.

Short failures remain visible in the raw dataset through timestamps and
`healthy` flags; they are not silently replaced. Use these options only when
diagnosing a known sensor-latency problem:

```bash
handumi record --help-advanced
```
:::

## Validate the Pilot

```bash
handumi validate \
  outputs/datasets/handumi-demo --strict
```

Review `meta/handumi_quality.json`. Fix rejected captures before increasing `--episodes`.

Hard rejection checks include insufficient duration, excessive tracking loss,
unhealthy cameras or encoders, synchronization errors, frozen source
timestamps or poses, implausible translation/rotation jumps, and invalid state
values. A stationary hand or constant gripper width is only a warning by
default. Thresholds live in `configs/quality.yaml`.

Common additions:

- `--pico-wifi`: stream PICO over Wi-Fi.
- `--skip-feetech`: record without gripper widths.
- `--dataset-license <id>`: set the dataset-card license.
- `--no-video`: store image frames instead of encoded video.
- `--encoder cpu`: force H.264 software encoding for reproducible CPU testing.
- `--encoder gpu`: require hardware encoding instead of falling back to CPU.

Run `handumi record --help` for the normal interface or
`handumi record --help-advanced` for synchronization, hardware and encoder
diagnostic overrides. Physical camera IDs belong only in `configs/rig.yaml`.
