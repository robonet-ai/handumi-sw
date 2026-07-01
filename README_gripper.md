# Gripper Setup (Feetech + Cameras)

One-time hardware setup: identify serial ports, home the Feetech servos, and
calibrate gripper width. Do this before teleoperating or recording. Once done,
the run commands live in the main [README.md](README.md).

## 1. Identify Ports

```bash
python scripts/setup/setup_ports.py
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

## 2. Check Feetech Ticks

```bash
python scripts/setup/calibrate_grippers.py monitor
```

Open/close each gripper and confirm `ticks` changes.

## 3. Home Servos (centre the encoder range)

The Feetech encoder reports position modulo 4096 and wraps at the 0/4095 seam.
If a gripper's travel crosses that seam, the width readout flips or saturates.
Homing stores a correction so the current shaft angle reads 2048 (centre):

```bash
python scripts/setup/home_servos.py              # both sides
python scripts/setup/home_servos.py --side right # one side
```

Hold the gripper at **mid-travel** (half open, ~2040 ticks) and press ENTER so
the full range sits clear of the seam. The script reads the position back and
reports `OK` / `CHECK`. Always re-calibrate afterwards (closed/open shift).

A software unwrap in `handumi.feetech.gripper` also tracks wraps continuously,
so even an un-homed range is fine as long as recording **starts with the
grippers roughly closed** (away from the seam).

## 4. Calibrate Gripper Width

```bash
python scripts/setup/calibrate_grippers.py calibrate
python scripts/setup/calibrate_grippers.py calibrate --side right
```

For each side:

```text
enter max opening in mm
open gripper fully while watching live ticks, press ENTER
close gripper fully while watching live ticks, press ENTER
```

Use `--side left|right` to recalibrate one gripper without disturbing the other.
This updates `configs/feetech.yaml`.

Setup done — head back to [README.md](README.md) to teleoperate and record.
