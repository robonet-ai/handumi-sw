# Quality Assurance

Review every recording before publishing or converting it. Start with visual
replay, then run automated validation and inspect the captured signals.

## 1. Replay and Inspect

For local recordings, pass the local root as `DATASET`; no dataset is
downloaded:

```bash
JAX_PLATFORMS=cpu handumi replay \
  outputs/20260714_224135 \
  --robot openarmv1 \
  --episode 0
```

See [Replay a Local Recording in Simulation](replay_in_sim.md) for the current
OpenArm v1 and TRLC-DK1 commands, calibration semantics, measured IK results,
and Viser mesh troubleshooting.

Choose the target robot explicitly. Piper is a currently available example:

```bash
TARGET_ROBOT=piper
handumi replay \
  your-name/handumi-demo \
  --robot "$TARGET_ROBOT"
```

In Viser, check the bimanual geometry, table alignment, motion continuity, and
unreachable poses. Use `--headless` for automated checks and `--strict-ik` to
fail when IK error exceeds the configured limits.
Add `--hide-trajectories` to show only the robot and scene without the target
and achieved TCP paths.

Table-calibrated datasets preserve recorded bimanual geometry automatically.

:::{dropdown} Absolute-table replay and calibration precedence
For an explicit geometry-preserving replay:

```bash
handumi replay your-name/handumi-demo \
  --robot "$TARGET_ROBOT" \
  --retarget-mode absolute-table \
  --deployment-calibration "configs/calibration/${TARGET_ROBOT}_table.yaml"
```

`absolute-table` applies `robot_from_table` to both TCP trajectories, preserving
their bimanual separation. By default, replay aligns each tool orientation on
the first frame and preserves subsequent wrist rotations. Use
`--absolute-orientation table-absolute` only when the HandUMI and robot TCP
frames were externally calibrated.

Controller-to-TCP calibration is selected in this order:

1. Explicit `--controller-tcp-calibration`.
2. Identity-bound snapshot stored in the dataset.
3. Robot/device calibration from `configs/robots/*.yaml`.
4. Device fallback for legacy data.

Replay prints the calibration source and hash, TCP distances, minimum height,
bimanual separation, table-to-robot transform, and IK errors.
:::

Offline playback of a dataset on physical arms is not currently exposed.
`handumi teleop real` consumes live HandUMI motion and is not a recorded-dataset
replay command.

## 2. Run Automated Validation

```bash
handumi validate \
  outputs/datasets/handumi-demo --strict
```

The report is written to `meta/handumi_quality.json`. Review rejected episodes
for tracking loss, stale sensors, synchronization errors, frozen poses, motion
jumps, or invalid duration. Rejected episodes are excluded automatically during
conversion.

## 3. Inspect Captured Signals

Raw datasets preserve the information needed to validate, recalibrate, or
retarget a capture:

```text
observation.images.left_wrist
observation.images.right_wrist
observation.images.workspace
observation.state                  # controller poses + gripper widths
observation.feetech.*              # ticks, width, time, health
observation.tracking.*             # device poses, validity, aligned time
observation.sync.*                 # shared target and record times
observation.camera.<name>.*        # sample time and health
```

`observation.state[14:16]` stores left/right gripper widths in meters. Tool,
controller mount, calibration hashes, source enablement, and coordinate layout
are stored in metadata. Raw controller poses remain unchanged so the same
capture can be checked against another supported robot.

## 4. Convert and Check Target Motion

Conversion creates a target-specific dataset while preserving the raw source.
For Piper, use the validated `--robot piper` profile. It runs the same
`absolute-table` solver as replay, validates `configs/calibration/piper_table.yaml`
for the selected robot, and converts the replay result to physical Piper commands:

```bash
JAX_PLATFORMS=cpu handumi convert \
  outputs/datasets/handumi-demo \
  --robot piper \
  --output your-name/handumi-demo-piper
```

The Piper state has 14 physical commands: six replay arm joints in radians
plus one gripper opening in meters per side. Its pairs are
`observation.state[t] = command[t]` and `action[t] = command[t+1]`. The two
mirrored URDF finger joints are reconstructed from the single opening only when
rendering simulation. Other embodiments use the same `--robot <name>` interface;
absolute-table support requires their corresponding
`configs/calibration/<name>_table.yaml` file.

Replay and validate the converted motion before using it with a robot-specific
integration. See [Add a New Robot Embodiment](../development/new_embodiment.md)
when adding another simulation model or hardware backend.

## 5. Publish Accepted Data

Upload only after the replay and validation checks pass:

```bash
hf auth login
huggingface-cli upload your-name/handumi-demo \
  outputs/datasets/handumi-demo --repo-type dataset
```
