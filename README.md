# HandUMI Software

Record HandUMI bimanual raw demonstrations as LeRobot-compatible datasets.

```text
left/right wrist cameras
+ left/right Feetech gripper encoder widths
+ optional VR tracking poses (Meta Quest / PICO)
-> HandUMI raw LeRobot dataset
```

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.12.

```bash
git clone <repo-url> handumi-sw
cd handumi-sw
bash install.sh              # Meta Quest only: bash install.sh --skip-xrt
source .venv/bin/activate
```

`install.sh` creates the venv, runs `uv sync`, fetches/builds the XRoboToolkit
native SDK (needed for PICO), and installs `xrobotoolkit_sdk`. Re-run it safely
after pulling changes.

Use `--skip-xrt` if you only track with **Meta Quest** — XRoboToolkit is PICO-only
tracking software, so building it is wasted time/dependencies on a Quest-only
setup (see [README_quest.md](README_quest.md)). Without the flag, install.sh
builds it for PICO support (see [README_pico.md](README_pico.md)).

Check:

```bash
python --version
handumi-record-pico --help
```

## Setup (one-time)

Do these before teleoperating or recording:

- **[README_gripper.md](README_gripper.md)** — Feetech + camera ports, servo
  homing, and gripper-width calibration.
- **[README_quest.md](README_quest.md)** — Meta Quest tracking (Phase 2): install
  the YubiQuestApp, find the Quest IP, and smoke-test the pose stream.

Everything below assumes that setup is done.

## Teleoperate (live monitor, no saving)

```bash
python -m handumi.capture.teleoperate_handumi \
  --fps 30
```

Streams cameras and gripper widths to Rerun without saving data. Start with the
grippers closed so the encoder unwrap anchors correctly.

## Record

Two recorders share the same 16D HandUMI raw state + Feetech width; pick the one
matching your tracking source.

### PICO / XRoboToolkit

```bash
handumi-record-pico \
  --repo-id local/handumi_width_test \
  --output-dir outputs/datasets/handumi_width_test \
  --task "gripper width hardware test" \
  --num-episodes 1 \
  --episode-time-s 20 \
  --fps 30
```

### Meta Quest (Phase 2)

First set up and smoke-test per [README_quest.md](README_quest.md), then:

```bash
# tracking check only (no cameras/sim): expect fps ~120 and both trk=1;
# trk=0 = controllers asleep or out of the headset cameras' view
python -m handumi.tracking.meta_quest --config configs/tracking_meta_quest.yaml

# live visualization — Rerun 3D trajectory (uses quest_ip from the config)
handumi-live-tracking-quest

# same, plus a live Piper robot following your hands via IK
# (Viser at http://localhost:8003; first launch JIT-compiles for ~30s)
handumi-live-tracking-quest --robot piper

# add a task scene with real physics (assets/scenes/cube_in_box)
handumi-live-tracking-quest --robot piper --scene cube_in_box

# record a dataset (16D state + observation.quest.* poses/clocks),
# hands-free: double-clap starts/stops each episode, voice announcements,
# saved to outputs/<timestamp>/
handumi-record-quest --robot piper --scene cube_in_box --clap-control
```

Add `--skip-cameras` / `--skip-feetech` to run without that hardware. Controls
(no headset UI): **left X** resets the workspace on the current HMD pose;
**right A** starts/stops an episode with `--button-control`; a double clap
(close both grippers twice within ~1.2s) does the same with `--clap-control`.

## Train

Datasets in `outputs/` train directly with lerobot (config in
`configs/train/act.yaml`, checkpoints in `outputs/train/`):

```bash
handumi-train --latest                              # newest dataset, ACT + wandb
handumi-train --dataset outputs/<ts> --steps=50000  # explicit dataset, overrides
```

## Inspect Dataset

```bash
lerobot-dataset-viz \
  --repo-id local/handumi_width_test \
  --root outputs/datasets/handumi_width_test \
  --episode-index 0
```

## Upload to Hugging Face

A recorded dataset is a plain folder, so push it with the Hugging Face CLI:

```bash
huggingface-cli login   # once, with a write token
huggingface-cli upload NONHUMAN-RESEARCH/handumi-dataset-v2 \
  outputs/datasets/handumi_quest_test --repo-type dataset
```

## Dataset Fields

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

`observation.state[14]` and `observation.state[15]` are left/right gripper width
in meters.

## Docs

- [README_gripper.md](README_gripper.md) — gripper + camera setup and calibration
- [README_quest.md](README_quest.md) — Meta Quest tracking setup (Phase 2)
- [docs/architecture.md](docs/architecture.md)
- [docs/phase-2-motion-tracking.md](docs/phase-2-motion-tracking.md) — Meta Quest
  motion tracking (body-worn, no-UI), Rerun trajectory rendering, yubi-sw/axol-vr
  references
- [docs/add-new-embodiment.md](docs/add-new-embodiment.md)
