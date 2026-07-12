# HandUMI Software

Ultima modificacion: 2026-07-11 20:39:48 -05 -0500

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache 2.0"></a>
  <a href="https://github.com/BrikHMP18/HandUMI"><img src="https://img.shields.io/badge/Hardware-HandUMI-4c8bf5.svg" alt="HandUMI hardware"></a>
</p>

[HandUMI](https://github.com/BrikHMP18/HandUMI) is a hand-worn interface for
collecting robot-free bimanual demonstrations. This repository contains its
data-collection, calibration, validation, replay, and robot-retargeting
software.

It records synchronized [LeRobot](https://github.com/huggingface/lerobot)-compatible
datasets:

```text
left/right wrist cameras
+ left/right Feetech gripper widths
+ VR tracking poses (PICO / Meta Quest)
-> raw HandUMI LeRobot dataset
```

The usual flow is:

```text
record data -> optionally push to Hugging Face -> convert to robot joints or replay in sim
```

The raw dataset remains robot-agnostic. Controller-to-TCP calibration and the
intended robot configuration are fingerprinted in dataset metadata so later
conversion remains reproducible.

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python >= 3.12.

```bash
git clone https://github.com/leoperezz/handumi-sw.git
cd handumi-sw
bash install.sh              # PICO support included
# bash install.sh --skip-xrt # Meta Quest only
source .venv/bin/activate
```

Check:

```bash
python --version
handumi-record --help
```

`install.sh` creates the virtual environment, runs `uv sync`, and builds the
XRoboToolkit SDK needed for PICO. Use `--skip-xrt` when the setup only uses
Meta Quest. It also creates the ignored machine-local `configs/rig.yaml` from
`configs/rig.example.yaml` without overwriting an existing rig configuration.

## Setup

Before recording, configure and calibrate the hardware once:

- [docs/README_gripper_width.md](docs/README_gripper_width.md) - Feetech/camera
  ports, servo homing, and gripper-width calibration.
- [docs/README_quest.md](docs/README_quest.md) - Meta Quest setup.
- [docs/README_pico.md](docs/README_pico.md) - PICO setup.
- [docs/README_tcp_offset.md](docs/README_tcp_offset.md) - controller to gripper-TCP offset.

## Simulation Teleop (no recording)

Run the app on your VR headset. Then run `handumi-teleop-sim` to open Viser
with the robot IK-following your HandUMI motion in real time, plus a Rerun
view of the calibrated TCP trails (`--no-rerun` to disable). Same calibration
and retargeting the replay uses, so what you see is what a recording would
replay — handy before a session to check tracking health and TCP calibration:

```bash
handumi-teleop-sim --device meta            # or --device pico
```

Teleop controls: a **double clap on either gripper** (close/open twice)
anchors or re-anchors the enabled, tracked arms so the current HandUMI pose
maps to the robot home and the robot follows from there. Pass `--space-start`
if you also want the keyboard Space key to start both idle arms at once. Arms
stay parked at home until their first anchor. Spoken feedback; `--no-sounds`
to mute. In the recorder below, the double clap re-centers the workspace and
starts/stops episodes.

For a full pick-and-place rehearsal with a task scene and real contact
physics (MuJoCo: the cube is graspable, driven by your Feetech opening):

```bash
handumi-teleop-sim --device meta --scene cube_in_box
```

`--scene <name>` loads `assets/scenes/<name>/scene.xml`, placed per
`configs/scene.yaml`. With a robot that declares an `mjcf` (Piper), the
scene runs under MuJoCo contact physics; otherwise it renders statically.
Optional `--anchor-z <m>`: anchor with the tip resting on the table to pin
absolute heights to the sim table (see `handumi-teleop-sim --help`).

## Record Data

Use `handumi-record` ([src/handumi/scripts/record.py](src/handumi/scripts/record.py))
with `--device pico` or `--device meta`.

Example with the common flags:

```bash
handumi-record \
  --device pico \
  --repo-id your-name/handumi-demo \
  --output-dir outputs/datasets/handumi-demo \
  --task "pick and place with HandUMI" \
  --robot piper \
  --wrist-cameras --workspace-camera \
  --num-episodes 10 \
  --episode-time-s 30 \
  --fps 30 \
  --cam-width 640 \
  --cam-height 480
```

For Meta Quest:

```bash
handumi-record \
  --device meta \
  --repo-id your-name/handumi-demo \
  --output-dir outputs/datasets/handumi-demo \
  --task "pick and place with HandUMI" \
  --robot piper \
  --wrist-cameras --workspace-camera \
  --clap-control \
  --num-episodes 10 \
  --fps 30
```

Useful options:

- `--rig-config` selects the local camera, Feetech, and Meta Quest settings
  (default: `configs/rig.yaml`).
- No camera-selection flag records both wrist cameras. Use
  `--wrist-cameras --workspace-camera` for all three, `--workspace-camera`
  for only the workspace view, or `--only-left-camera` /
  `--only-right-camera` for one wrist view.
- `--robot piper` records the intended embodiment and an exact snapshot of its
  robot configuration. The raw trajectories remain robot-agnostic.
- `--controller-tcp-calibration` selects the physical HandUMI mount offset to
  snapshot in metadata; raw controller poses remain unchanged.
- `--clap-control` starts or stops an episode by squeezing either the left or
  right gripper twice within 1.6 seconds.
- `--push-to-hub` pushes the dataset after recording.
- `--skip-feetech` records with zero-filled gripper widths.
- `--pico-wifi` uses PICO over Wi-Fi instead of ADB.
- `--manual-control` lets PICO buttons start/repeat/finish episodes.
- `--tracking-loss-timeout-s` sets how long tracking may remain lost before
  the current episode is discarded (default: 1 second).
- `--sync-lag-s` selects samples from the native sensor buffers against one
  shared target timestamp (default: 40 ms behind real time).
- `--sensor-loss-timeout-s` discards an episode after sustained camera or
  encoder health failure (default: 1 second).
- `--no-video` stores image frames instead of encoded video.

By default, each episode starts when you press ENTER in the terminal. Recording
then waits for fresh, valid poses from both controllers. If either controller
remains untracked beyond the loss timeout, the whole episode is discarded.
Camera capture timestamps, Quest clock alignment, encoder timestamps, source
age, synchronization error, and health flags are stored on every row.

## Validate Recordings

Run offline validation before training or conversion:

```bash
handumi-validate \
  --repo-id your-name/handumi-demo \
  --root outputs/datasets/handumi-demo
```

This writes `meta/handumi_quality.json` without deleting raw data. It rejects
episodes with excessive tracking loss, stale sensors, synchronization errors,
source or pose freezes, implausible translation jumps, rotations over 90
degrees per frame, or insufficient duration. Thresholds are in
`configs/quality.yaml`; see [docs/README_quality.md](docs/README_quality.md).

## Push to Hugging Face

If the dataset was not recorded with `--push-to-hub`, upload the local folder:

```bash
huggingface-cli login
huggingface-cli upload your-name/handumi-demo \
  outputs/datasets/handumi-demo --repo-type dataset
```

## Convert to Robot Joints

`handumi-convert`
([src/handumi/scripts/conversion.py](src/handumi/scripts/conversion.py))
converts the raw 16D HandUMI dataset into a robot-specific joint dataset using
the robot configuration in `configs/robots/`.

Conversion runs the same offline quality filter by default, skips rejected
episodes, and writes `meta/source_quality.json` in the converted dataset.
`--skip-quality-filter` is available only for debugging bad captures.

Minimal conversion:

```bash
handumi-convert --repo-id your-name/handumi-demo
```

The default embodiment is `axol`. To convert for Piper, use:

```bash
handumi-convert \
  --repo-id your-name/handumi-demo \
  --embodiment piper
```

Robot configs live in `configs/robots/`, for example
[configs/robots/piper.yaml](configs/robots/piper.yaml) and
[configs/robots/axol.yaml](configs/robots/axol.yaml).

Add `--push-to-hub` to upload the converted dataset.

## Replay in Simulation

To inspect how a recorded dataset moves the robot in simulation with
`handumi-replay-in-sim`
([src/handumi/scripts/replay/replay_in_sim.py](src/handumi/scripts/replay/replay_in_sim.py)):

```bash
handumi-replay-in-sim --repo-id your-name/handumi-demo
```

This opens a local Viser viewer and saves a rollout under `outputs/replay_in_sim/`.
The default robot is `piper`; choose another configured robot with `--robot axol`.

Headless example:

```bash
handumi-replay-in-sim \
  --repo-id your-name/handumi-demo \
  --headless
```

## Train

Training is out of scope for this repo — HandUMI produces LeRobot-compatible
datasets, so train with [lerobot](https://github.com/huggingface/lerobot)
directly (already a dependency):

```bash
lerobot-train \
  --dataset.repo_id=<repo-id> --dataset.root=outputs/datasets/<name> \
  --policy.type=act --wandb.enable=true
```

## Dataset Fields

Raw HandUMI datasets include:

```text
observation.images.left_wrist
observation.images.right_wrist
observation.images.workspace            # when --workspace-camera is enabled
observation.state                  # float32[16]
action                             # float32[16]
observation.feetech.left_ticks
observation.feetech.right_ticks
observation.feetech.left_width_mm
observation.feetech.right_width_mm
observation.feetech.left_normalized
observation.feetech.right_normalized
observation.feetech.sample_time_ns
observation.feetech.healthy
observation.tracking.left_tracked
observation.tracking.right_tracked
observation.tracking.left_device_tracked
observation.tracking.left_pose_valid
observation.tracking.hmd_pose
observation.tracking.aligned_time_ns
observation.tracking.clock_synced
observation.tracking.streaming
observation.sync.target_time_ns
observation.sync.record_time_ns
observation.camera.<name>.sample_time_ns
observation.camera.<name>.healthy
```

`observation.state[14]` and `observation.state[15]` are the left/right gripper
widths in meters. Camera, Feetech, and tracking diagnostics also include
`age_ms` and `sync_error_ms`.

## More Docs

- [docs/add_new_embodiment.md](docs/add_new_embodiment.md) - add a new robot
  embodiment.
- [docs/README_gripper_width.md](docs/README_gripper_width.md) - gripper and camera setup.
- [docs/README_quest.md](docs/README_quest.md) - Meta Quest setup.
- [docs/README_pico.md](docs/README_pico.md) - PICO setup.
- [docs/README_tcp_offset.md](docs/README_tcp_offset.md) - controller to gripper-TCP offset.
- [docs/README_quality.md](docs/README_quality.md) - synchronization, sensor health,
  and offline episode filtering.
- [docs/calibration_plan.md](docs/calibration_plan.md) - portable Quest/table
  calibration plan and acceptance criteria.

## References and Acknowledgments

- UMI: Chi et al., "Universal Manipulation Interface: In-The-Wild Robot
  Teaching Without In-The-Wild Robots," RSS 2024.
  [Project](https://umi-gripper.github.io/) ·
  [Paper](https://arxiv.org/abs/2402.10329)
- YUBI: Ohkawa et al., "YUBI: Yielding Universal Bidigital Interface for
  Bimanual Dexterous Manipulation at Scale," 2026.
  [Project](https://yubi.airoa.io/) ·
  [Paper](https://arxiv.org/abs/2606.10244) ·
  [Software](https://github.com/airoa-org/yubi-sw)
- Meta Quest support uses YubiQuestApp and adapts the yubi-sw protocol and
  coordinate conversion. PICO support uses
  [XRoboToolkit](https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind).
- Core software: [LeRobot](https://github.com/huggingface/lerobot),
  [PyRoki](https://github.com/chungmin99/pyroki),
  [Viser](https://github.com/nerfstudio-project/viser),
  [Rerun](https://github.com/rerun-io/rerun), and
  [MuJoCo](https://github.com/google-deepmind/mujoco).
- Robot assets: [Almond Axol](https://github.com/almond-bot/axol) and
  [AgileX Piper ROS](https://github.com/agilexrobotics/piper_ros), both MIT.

HandUMI is not affiliated with or endorsed by Meta, PICO, AgileX, AIRoA/YUBI,
Almond, or Hugging Face. All trademarks belong to their respective owners.

## Team

- **Project lead and original hardware design:**
  [BrikHMP18](https://github.com/BrikHMP18)
- **Core software contributors:**
  [Leonardo Pérez](https://github.com/leoperezz),
  [Raul Bastidas](https://github.com/RAUL-BASTIDAS),
  [Mitshell Ramos](https://github.com/mbrq13), and
  [Alvaro Mendoza-Li](https://github.com/alvax64)
- **Core hardware contributors:**
  [Alvaro Mendoza-Li](https://github.com/alvax64) and
  [Bryan Bastidas](https://github.com/BryanB72)
- **IK and teleoperation explorations:**
  Raul Bastidas developed the initial
  [handumi-IK](https://github.com/raulbastidas1203/handumi-IK) experiments.

## Safety

This is research software. Preview and validate trajectories before commanding
physical robots, keep an emergency stop accessible, and enforce the robot's
joint, velocity, acceleration, workspace, and collision limits. The software
is provided without warranty.

## License

Original HandUMI software and documentation in this repository are licensed
under the [Apache License 2.0](LICENSE). Third-party software and robot assets
remain under their respective licenses, listed at the end of [LICENSE](LICENSE).

This license does not automatically apply to datasets recorded with HandUMI,
the separate HandUMI hardware repository, headset applications, robot
firmware, or trademarks.
