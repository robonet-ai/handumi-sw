# HandUMI Software

Software stack for recording HandUMI bimanual raw demonstrations as
LeRobot-compatible datasets.

HandUMI records data without a robot in the collection loop:

```text
left/right wrist cameras
+ left/right Feetech gripper encoder widths
+ optional left/right VR tracking poses
-> HandUMI raw LeRobot dataset
```

Robot-specific datasets for Piper, Axol, and other embodiments are derived
later through offline retargeting / IK.

## Requirements

- Linux workstation with USB access.
- Python 3.12.
- `uv` installed.
- Two USB wrist cameras.
- Two Feetech servos used as gripper encoders.
- Optional: PICO / Meta Quest tracking for later capture stages.

## Installation

```bash
git clone <repo-url> handumi-sw
cd handumi-sw
uv sync --python "$(command -v python3.12)"
source .venv/bin/activate
```

Verify:

```bash
python --version
PYTHONPATH=src python scripts/record_handumi.py --help
```

## Checkpoint 1: Cameras + Feetech Width

This is the first hardware target:

```text
left USB wrist camera
right USB wrist camera
left Feetech servo encoder  -> left gripper opening
right Feetech servo encoder -> right gripper opening
LeRobotDataset output
```

PICO / Meta Quest tracking is optional and disabled by default for this
checkpoint.

### 1. Feetech Ports And IDs

Identify serial ports by unplugging/plugging one gripper adapter at a time:

```bash
while true; do
  clear
  echo "=== $(date) ==="
  ls -l /dev/serial/by-id 2>/dev/null || true
  ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true
  sleep 2
done
```

Use `/dev/serial/by-id/...` when available; otherwise use `/dev/ttyACM*` or
`/dev/ttyUSB*`.

Scan servos:

```bash
PYTHONPATH=src python scripts/setup/scan_feetech.py \
  --all-ports \
  --start-id 0 \
  --end-id 20
```

Expected convention:

```text
left gripper  -> servo ID 0
right gripper -> servo ID 1
```

Assign IDs one side at a time if needed:

Left gripper:

```bash
PYTHONPATH=src python scripts/setup/write_feetech_id.py \
  --port /dev/SERIAL_LEFT \
  --current-id <current_left_id> \
  --new-id 0
```

Right gripper:

```bash
PYTHONPATH=src python scripts/setup/write_feetech_id.py \
  --port /dev/SERIAL_RIGHT \
  --current-id <current_right_id> \
  --new-id 1
```

Verify:

```bash
PYTHONPATH=src python scripts/setup/scan_feetech.py \
  --all-ports \
  --start-id 0 \
  --end-id 20
```

Save mapping:

```bash
PYTHONPATH=src python scripts/setup/save_gripper_config.py \
  --left-id 0 \
  --right-id 1 \
  --left-port /dev/SERIAL_LEFT \
  --right-port /dev/SERIAL_RIGHT
```

Check encoder ticks:

```bash
PYTHONPATH=src python scripts/setup/monitor_gripper_ticks.py \
  --port-id /dev/SERIAL_LEFT 0 \
  --port-id /dev/SERIAL_RIGHT 1
```

### 2. Calibrate Gripper Opening

Measure max opening in millimeters, then run:

```bash
PYTHONPATH=src python scripts/setup/calibrate_gripper_width.py \
  --max-width-mm 80
```

The command asks you to close both grippers, then open both grippers. It stores:

```text
closed_ticks
open_ticks
max_width_mm
```

Per frame, HandUMI records raw ticks, normalized width, width in mm, and state
width in meters.

### 3. Cameras

Identify cameras by unplugging/plugging one camera at a time:

```bash
while true; do
  clear
  echo "=== $(date) ==="
  v4l2-ctl --list-devices
  sleep 2
done
```

Scan OpenCV indices:

```bash
PYTHONPATH=src python scripts/setup/scan_cameras.py
```

Use camera IDs in this order:

```text
first --cam-ids value  -> observation.images.left_wrist
second --cam-ids value -> observation.images.right_wrist
```

### 4. Live Monitor

Before recording, run the live Rerun monitor:

```bash
PYTHONPATH=src python -m handumi.capture.teleoperate_handumi \
  --cam-ids 0 2 \
  --feetech-config configs/feetech.yaml \
  --fps 30
```

This does not save data. It streams:

```text
left/right wrist camera images
left/right raw Feetech ticks
left/right normalized gripper opening
left/right gripper opening in mm
```

Use it to verify that camera assignment, servo IDs, ports, and calibration are
correct before recording.

### 5. Record Dataset

```bash
PYTHONPATH=src python scripts/record_handumi.py \
  --cam-ids 0 2 \
  --feetech-config configs/feetech.yaml \
  --repo-id local/handumi_width_test \
  --output-dir outputs/datasets/handumi_width_test \
  --task "gripper width hardware test" \
  --num-episodes 1 \
  --episode-time-s 20 \
  --fps 30
```

Equivalent launcher:

```bash
bash bin/record.sh \
  --cam-ids 0 2 \
  --repo-id local/handumi_width_test \
  --output-dir outputs/datasets/handumi_width_test \
  --task "gripper width hardware test" \
  --num-episodes 1 \
  --episode-time-s 20
```

The raw dataset stores:

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

`observation.state[14]` and `observation.state[15]` are the calibrated left/right
gripper widths in meters.

### 6. Inspect With LeRobot

```bash
lerobot-dataset-viz \
  --repo-id local/handumi_width_test \
  --root outputs/datasets/handumi_width_test \
  --episode-index 0
```

## Record With Tracking

After the hardware checkpoint works, enable PICO streams with:

```bash
PYTHONPATH=src python scripts/record_handumi.py \
  --use-pico \
  --pico-mandos \
  --cam-ids 0 2 \
  --feetech-config configs/feetech.yaml \
  --repo-id local/handumi_pico_test \
  --output-dir outputs/datasets/handumi_pico_test
```

## Retarget / Replay

Convert a HandUMI source dataset to a robot-specific dataset:

```bash
bash bin/process_handumi_to_lerobot.sh \
  --embodiment piper \
  --output-name handumi-dataset-v2-piper \
  --output-root outputs/datasets/handumi-dataset-v2-piper
```

Inspect retargeting:

```bash
python scripts/replay_pico_ik.py --embodiment piper --episode 0 --visualize
python scripts/compare_axis.py --embodiment axol --episode 0
```

Replay a Piper robot-specific dataset:

```bash
bash bin/piper/replay_from_dataset.sh --episode 0 --dry-run
```

## Project Layout

```text
.
├── assets/                  # Robot URDFs and meshes
├── bin/                     # Shell launchers
├── configs/                 # Camera, Feetech, tracking configs
├── docs/                    # Architecture and embodiment guide
├── scripts/                 # Manual hardware and pipeline scripts
├── src/handumi/             # Core package
├── tests/                   # Automated tests
└── utils/                   # Upload helpers
```

```text
src/handumi/
├── capture/                 # HandUMI raw recorder
├── cameras/                 # USB wrist cameras
├── dataset/                 # Raw schema, LeRobot IO, conversion
├── feetech/                 # Feetech encoder bus/calibration/gripper widths
├── replay/                  # PICO IK replay and robot replay
├── retargeting/             # Raw/PICO poses to robot targets
├── robots/                  # Piper/Axol embodiment registry and IK specs
└── tracking/                # PICO / tracker backends
```

## Docs

| Doc | Description |
|-----|-------------|
| [docs/architecture.md](docs/architecture.md) | System architecture, raw schema, configs, and manual scripts |
| [docs/phase-2-motion-tracking.md](docs/phase-2-motion-tracking.md) | Meta Quest/WebXR tracking and live Viser plan |
| [docs/add-new-embodiment.md](docs/add-new-embodiment.md) | How to add a new robot embodiment |
