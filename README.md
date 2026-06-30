# HandUMI Software

Record HandUMI bimanual raw demonstrations as LeRobot-compatible datasets.

```text
left/right wrist cameras
+ left/right Feetech gripper encoder widths
+ optional VR tracking poses
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
PYTHONPATH=src python scripts/record_handumi.py --help
```

## Hardware Setup

### 1. Identify Ports

```bash
PYTHONPATH=src python scripts/setup/setup_ports.py
```

Connect/disconnect one device at a time and note the changed port.
Use `Ctrl+C` to stop.

The Feetech section shows each serial port and detected servo IDs:

```text
/dev/ttyACM0: ids=[0]
/dev/ttyACM1: ids=[1]
```

Edit `configs/feetech.yaml`:

```yaml
left:
  servo_id: 0
  port: /dev/ttyACM0
right:
  servo_id: 1
  port: /dev/ttyACM1
```

Edit `configs/cameras.yaml`:

```yaml
left_wrist:
  index_or_path: 0
right_wrist:
  index_or_path: 2
```

### 2. Check Feetech Ticks

```bash
PYTHONPATH=src python scripts/setup/calibrate_grippers.py monitor
```

Open/close each gripper and confirm `ticks` changes.

### 3. Home Servos (centre the encoder range)

The Feetech encoder reports position modulo 4096 and wraps at the 0/4095 seam.
If a gripper's travel crosses that seam, the width readout flips or saturates.
Homing stores a correction so the current shaft angle reads 2048 (centre):

```bash
PYTHONPATH=src python scripts/setup/home_servos.py              # both sides
PYTHONPATH=src python scripts/setup/home_servos.py --side right # one side
```

Hold the gripper at **mid-travel** (half open, ~2040 ticks) and press ENTER so
the full range sits clear of the seam. The script reads the position back and
reports `OK` / `CHECK`. Always re-calibrate afterwards (closed/open shift).

A software unwrap in `handumi.feetech.gripper` also tracks wraps continuously,
so even an un-homed range is fine as long as recording **starts with the
grippers roughly closed** (away from the seam).

### 4. Calibrate Gripper Width

```bash
PYTHONPATH=src python scripts/setup/calibrate_grippers.py calibrate
PYTHONPATH=src python scripts/setup/calibrate_grippers.py calibrate --side right
```

For each side:

```text
enter max opening in mm
open gripper fully while watching live ticks, press ENTER
close gripper fully while watching live ticks, press ENTER
```

Use `--side left|right` to recalibrate one gripper without disturbing the other.
This updates `configs/feetech.yaml`.

### 5. Live Monitor

```bash
PYTHONPATH=src python -m handumi.capture.teleoperate_handumi \
  --feetech-config configs/feetech.yaml \
  --fps 30
```

Streams cameras and gripper widths to Rerun without saving data. Start with the
grippers closed so the encoder unwrap anchors correctly.

### 6. Record

```bash
PYTHONPATH=src python scripts/record_handumi.py \
  --feetech-config configs/feetech.yaml \
  --repo-id local/handumi_width_test \
  --output-dir outputs/datasets/handumi_width_test \
  --task "gripper width hardware test" \
  --num-episodes 1 \
  --episode-time-s 20 \
  --fps 30
```

Or:

```bash
bash bin/record.sh \
  --repo-id local/handumi_width_test \
  --output-dir outputs/datasets/handumi_width_test \
  --task "gripper width hardware test" \
  --num-episodes 1 \
  --episode-time-s 20
```

## Inspect Dataset

```bash
lerobot-dataset-viz \
  --repo-id local/handumi_width_test \
  --root outputs/datasets/handumi_width_test \
  --episode-index 0
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

## Motion Tracking (Phase 2)

Phase 2 adds Meta Quest controller tracking. The Quest is **body-worn on the
neck as a tracking base** (no headset UI) with a controller mounted on each
gripper; the two gripper wrist cameras are the only cameras. The transport is a
**yubi-style native Quest app streaming poses over TCP/JSON** (plus UDP
time-sync) — not WebXR. Phase 2A focuses on the motion tracking itself:
receiving controller poses in Python, calibrating them with unit-tested
transforms, merging Feetech width into the 16D raw state, and rendering a **live
3D controller trajectory in Rerun** (alongside the cameras and gripper-width
series). The Viser 3D robot follow-along is deferred to Phase 2B. See
[docs/phase-2-motion-tracking.md](docs/phase-2-motion-tracking.md); `../yubi-sw`
is the primary reference (`../axol-vr` is secondary, for state-machine logic).

## Docs

- [docs/architecture.md](docs/architecture.md)
- [docs/phase-2-motion-tracking.md](docs/phase-2-motion-tracking.md) — Meta Quest
  motion tracking (body-worn, no-UI), Rerun trajectory rendering, yubi-sw/axol-vr
  references
- [docs/add-new-embodiment.md](docs/add-new-embodiment.md)
