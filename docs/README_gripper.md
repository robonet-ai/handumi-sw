# Gripper Setup (Feetech + Cameras)

One-time **per-laptop** hardware setup before teleoperating or recording:
serial ports, camera indices, servo homing. Width calibration lives in
[README_calibration.md](README_calibration.md).

Ports (`servo_id`/`port`, camera `index_or_path`) are wiring — committed in
`configs/feetech.yaml` / `configs/cameras.yaml`; edit them directly. Homing
is stored in the servo's EEPROM (persists across power cycles and laptops).

## 1. Identify Ports

```bash
handumi-setup-ports
```

Connect/disconnect one device at a time and note the changed port. `Ctrl+C` to
stop. The Feetech section shows each serial port and detected servo IDs:

```text
/dev/ttyACM0: ids=[0]
/dev/ttyACM1: ids=[1]
```

Edit `configs/feetech.yaml` (committed, machine-specific — same idea as
`configs/cameras.yaml`):

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
handumi-calibrate-grippers monitor
```

Open/close each gripper and confirm `ticks` changes.

## 3. Home Servos (centre the encoder range)

The encoder wraps at the 0/4095 seam; travel crossing it makes the width readout
flip or saturate. Homing stores a correction so the current shaft angle reads
2048 (centre), clearing the range of the seam:

```bash
handumi-home-servos              # both sides
handumi-home-servos --side right # one side
```

Hold the gripper at **mid-travel** (~2040 ticks), press ENTER; the script reports
`OK` / `CHECK`. Re-calibrate afterwards.

A software unwrap in `handumi.feetech.gripper` also tracks wraps continuously, so
an un-homed range is fine as long as recording **starts with the grippers roughly
closed**.

## 4. Calibrate Gripper Width

```bash
handumi-calibrate-grippers calibrate            # both sides
handumi-calibrate-grippers calibrate --side right
```

For each side: enter the max opening in mm, open fully (ENTER), close fully
(ENTER). This writes to the per-user cache
(`~/.cache/handumi/calibration.yaml`), never to the repo.

Setup done — head back to [README.md](README.md) to record. The other
calibration (controller → gripper TCP, once per mount design) is in
[README_calibration.md](README_calibration.md).
