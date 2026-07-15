# Record Demonstrations

Recording uses HandUMI directly and does not require a robot arm. The raw
controller poses, camera streams, gripper widths, and calibration metadata
remain available for later retargeting to any supported embodiment.

## Before Recording

Confirm that:

- Both gripper widths respond correctly from closed to fully open.
- Both controllers report valid tracking.
- Camera intrinsics and controller-camera mounts still match the hardware.
- The current session/table calibration has been visualized in Rerun.
- The Controller-to-TCP calibration matches the installed HandUMI tool.

See [Setup and Calibration](setup.md) if any check fails.

Start with a short pilot:

```bash
handumi-record \
  --device meta \
  --repo-id your-name/handumi-demo \
  --output-dir outputs/datasets/handumi-demo \
  --task "pick and place" \
  --session-calibration outputs/calibration/session.yaml \
  --wrist-cameras --workspace-camera \
  --rerun --clap-control \
  --num-episodes 3 \
  --episode-time-s 30
```

Use `--device pico` for PICO. Add `--push-to-hub` only after confirming the pilot locally.

Do not connect or configure a robot arm for this step. A target embodiment can
be selected later during conversion or replay without modifying the raw
recording.

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
handumi-record --help
```
:::

## Validate the Pilot

```bash
handumi-validate \
  --root outputs/datasets/handumi-demo \
  --fail-on-reject
```

Review `meta/handumi_quality.json`. Fix rejected captures before increasing `--num-episodes`.

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

Run `handumi-record --help` for advanced camera and synchronization options.
