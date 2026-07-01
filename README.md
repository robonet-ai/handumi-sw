# HandUMI Software

Record HandUMI bimanual raw demonstrations as LeRobot-compatible datasets.

```text
left/right wrist cameras
+ left/right Feetech gripper encoder widths
+ optional VR tracking poses (Meta Quest / PICO)
-> HandUMI raw LeRobot dataset
```

## Install

```bash
git clone <repo-url> handumi-sw
cd handumi-sw
uv sync --python "$(command -v python3.12)"
source .venv/bin/activate
```

Check:

```bash
python --version
PYTHONPATH=src python scripts/record_handumi_pico.py --help
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
PYTHONPATH=src python -m handumi.capture.teleoperate_handumi \
  --feetech-config configs/feetech.yaml \
  --fps 30
```

Streams cameras and gripper widths to Rerun without saving data. Start with the
grippers closed so the encoder unwrap anchors correctly.

## Record

Two recorders share the same 16D HandUMI raw state + Feetech width; pick the one
matching your tracking source.

### PICO / XRoboToolkit

```bash
PYTHONPATH=src python scripts/record_handumi_pico.py \
  --feetech-config configs/feetech.yaml \
  --repo-id local/handumi_width_test \
  --output-dir outputs/datasets/handumi_width_test \
  --task "gripper width hardware test" \
  --num-episodes 1 \
  --episode-time-s 20 \
  --fps 30
```

Or use the launcher (wraps the PICO recorder):

```bash
bash bin/record.sh \
  --repo-id local/handumi_width_test \
  --output-dir outputs/datasets/handumi_width_test \
  --task "gripper width hardware test" \
  --num-episodes 1 \
  --episode-time-s 20
```

### Meta Quest (Phase 2)

First set up and smoke-test per [README_quest.md](README_quest.md), then:

```bash
# live visualization — Rerun 3D trajectory (uses quest_ip from the config)
PYTHONPATH=src python scripts/live_tracking.py

# record a dataset (16D state + observation.quest.* poses/clocks)
PYTHONPATH=src python scripts/record_handumi_quest.py \
  --feetech-config configs/feetech.yaml \
  --repo-id local/handumi_quest_test \
  --output-dir outputs/datasets/handumi_quest_test \
  --task "quest tracking test" --num-episodes 1 --episode-time-s 20
```

Move the controllers and their trajectories draw in the Rerun 3D view. Add
`--skip-cameras` / `--skip-feetech` to run without that hardware.

Controls (no headset UI — feedback is on the workstation): **left X** resets the
workspace on the current HMD pose (also auto-set on the first tracked frame);
**right A** starts/stops an episode when recording with `--button-control`.

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
