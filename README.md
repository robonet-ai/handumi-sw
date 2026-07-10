# HandUMI Software

HandUMI records bimanual demonstrations as LeRobot-compatible datasets:

```text
left/right wrist cameras
+ left/right Feetech gripper widths
+ optional VR tracking poses (PICO / Meta Quest)
-> raw HandUMI LeRobot dataset
```

The usual flow is:

```text
record data -> optionally push to Hugging Face -> convert to robot joints or replay in sim
```

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python >= 3.12.

```bash
git clone <repo-url> handumi-sw
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
Meta Quest.

## Setup

Before recording, configure and calibrate the hardware once:

- [docs/README_gripper_width.md](docs/README_gripper_width.md) - Feetech/camera
  ports, servo homing, and gripper-width calibration.
- [docs/README_quest.md](docs/README_quest.md) - Meta Quest setup.
- [docs/README_pico.md](docs/README_pico.md) - PICO setup.
- [docs/README_tcp_offset.md](docs/README_tcp_offset.md) - controller to gripper-TCP offset.

## Live Preview (no recording)

Run the app on your VR headset. Then run `handumi-live` to open Viser with the robot IK-following your HandUMI motion in
real time, plus a Rerun view of the calibrated TCP trails (`--no-rerun` to
disable). Same calibration + retargeting the replay uses, so what you see is
what a recording would replay — handy before a session to check tracking
health and TCP calibration:

```bash
handumi-live --device meta            # or --device pico
```

Per-arm controls: both arms start disengaged at home — press **X** to
anchor the left arm to your current left-hand pose, **A** for the right
(re-press to re-anchor). A **double clap on one gripper** sends that arm
back to home, disengaged (hands-free). Spoken feedback; `--no-sounds` to
mute. In the recorder below, the double clap instead starts/stops episodes.

## Record Data

Use `handumi-record` ([src/handumi/scripts/record.py](src/handumi/scripts/record.py))
with `--device pico` or `--device meta`.

Example with the common flags:

```bash
handumi-record \
  --device pico \
  --repo-id NONHUMAN-RESEARCH/handumi-demo \
  --output-dir outputs/datasets/handumi-demo \
  --task "pick and place with HandUMI" \
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
  --repo-id NONHUMAN-RESEARCH/handumi-demo \
  --output-dir outputs/datasets/handumi-demo \
  --task "pick and place with HandUMI" \
  --num-episodes 10 \
  --episode-time-s 30 \
  --fps 30
```

Useful options:

- `--push-to-hub` pushes the dataset after recording.
- `--skip-feetech` records with zero-filled gripper widths.
- `--pico-wifi` uses PICO over Wi-Fi instead of ADB.
- `--manual-control` lets PICO buttons start/repeat/finish episodes.
- `--no-video` stores image frames instead of encoded video.

By default, each episode starts when you press ENTER in the terminal.

## Push to Hugging Face

If the dataset was not recorded with `--push-to-hub`, upload the local folder:

```bash
huggingface-cli login
huggingface-cli upload NONHUMAN-RESEARCH/handumi-demo \
  outputs/datasets/handumi-demo --repo-type dataset
```

## Convert to Robot Joints

`handumi-convert`
([src/handumi/scripts/conversion.py](src/handumi/scripts/conversion.py))
converts the raw 16D HandUMI dataset into a robot-specific joint dataset using
the robot configuration in `configs/robots/`.

Minimal conversion:

```bash
handumi-convert --repo-id NONHUMAN-RESEARCH/handumi-demo
```

The default embodiment is `axol`. To convert for Piper, use:

```bash
handumi-convert \
  --repo-id NONHUMAN-RESEARCH/handumi-demo \
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
handumi-replay-in-sim --repo-id NONHUMAN-RESEARCH/handumi-demo
```

This opens a local Viser viewer and saves a rollout under `outputs/replay_in_sim/`.
The default robot is `piper`; choose another configured robot with `--robot axol`.

Headless example:

```bash
handumi-replay-in-sim \
  --repo-id NONHUMAN-RESEARCH/handumi-demo \
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
observation.state                  # float32[16]
action                             # float32[16]
observation.feetech.left_ticks
observation.feetech.right_ticks
observation.feetech.left_width_mm
observation.feetech.right_width_mm
observation.feetech.left_normalized
observation.feetech.right_normalized
```

`observation.state[14]` and `observation.state[15]` are the left/right gripper
widths in meters.

## More Docs

- [docs/add_new_embodiment.md](docs/add_new_embodiment.md) - add a new robot
  embodiment.
- [docs/README_gripper_width.md](docs/README_gripper_width.md) - gripper and camera setup.
- [docs/README_quest.md](docs/README_quest.md) - Meta Quest setup.
- [docs/README_pico.md](docs/README_pico.md) - PICO setup.
- [docs/README_tcp_offset.md](docs/README_tcp_offset.md) - controller to gripper-TCP offset.
