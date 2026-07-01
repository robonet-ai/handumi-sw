# Gripper Setup (Feetech + Cameras)

One-time **per-laptop** hardware setup before teleoperating or recording: serial
ports, servo homing, gripper-width calibration. Run commands live in
[README.md](README.md).

> **Where calibration is stored.** Ports and tick ranges are machine-specific, so
> they live in a per-user cache — `~/.cache/handumi/feetech.yaml` (or
> `$XDG_CACHE_HOME/...`), **not** in git. The setup tools seed it from the tracked
> template `configs/feetech.yaml` on first run and write back to it. Homing itself
> is stored in the servo's EEPROM (persists across power cycles and laptops). Pass
> `--config` to any setup tool to override the path.
>
> **Once the cache exists, editing `configs/feetech.yaml` has no effect** — every
> setup tool prints `Using config: <path>` first; always edit *that* path.

## 1. Identify Ports

```bash
python scripts/setup/setup_ports.py
```

Connect/disconnect one device at a time and note the changed port. `Ctrl+C` to
stop. The Feetech section shows each serial port and detected servo IDs, and
prints the cache file to edit:

```text
/dev/ttyACM0: ids=[0]
/dev/ttyACM1: ids=[1]

Edit servo_id/port in: /home/you/.cache/handumi/feetech.yaml
```

Set each `servo_id`/`port` in that cache file:

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

The encoder wraps at the 0/4095 seam; travel crossing it makes the width readout
flip or saturate. Homing stores a correction so the current shaft angle reads
2048 (centre), clearing the range of the seam:

```bash
python scripts/setup/home_servos.py              # both sides
python scripts/setup/home_servos.py --side right # one side
```

Hold the gripper at **mid-travel** (~2040 ticks), press ENTER; the script reports
`OK` / `CHECK`. Re-calibrate afterwards.

A software unwrap in `handumi.feetech.gripper` also tracks wraps continuously, so
an un-homed range is fine as long as recording **starts with the grippers roughly
closed**.

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
This writes to the per-user cache (`~/.cache/handumi/feetech.yaml`).

Setup done — head back to [README.md](README.md) to teleoperate and record.
