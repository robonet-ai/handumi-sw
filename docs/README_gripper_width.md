# Gripper Setup (Feetech + Cameras)

One-time **per-laptop** hardware setup before teleoperating or recording:
serial ports, camera indices, servo homing, and width calibration.

Ports (`servo_id`/`port`, camera `index_or_path`) are machine-local wiring in
the ignored `configs/rig.yaml`; `install.sh` creates it from
`configs/rig.example.yaml`. Homing is stored in the servo's EEPROM and
persists across power cycles and laptops.

## 1. Identify Ports

```bash
handumi-setup-ports
```

Leave the command running while you plug and unplug hardware. It refreshes when
Linux reports a USB, serial, or camera device change. Connect or disconnect
**one physical device at a time**, then note which line changed. `Ctrl+C` stops
the command.

You are collecting two kinds of wiring information:

- Feetech grippers: serial device path plus servo ID.
- Cameras: video device path or camera index.

If the output does not show the devices you expect, see
[Troubleshooting](#troubleshooting).

If udev event monitoring is unavailable, the command falls back to polling. You
can force polling with `handumi-setup-ports --poll`.

### Servo IDs

Each Feetech servo has an internal ID stored in EEPROM. You only need to change
it when preparing a new servo or when two servos that share one bus have the
same ID. Servos with the same ID cannot share the same `port`.

Connect one servo at a time and assign the desired ID:

```bash
handumi-set-servo-id --port /dev/ttyUSB0 --new-id 0   
handumi-set-servo-id --port /dev/ttyUSB0 --new-id 1   
```

Then run `handumi-setup-ports` again and copy the detected `servo_id`/`port`
pairs into `configs/rig.yaml`.

### Feetech grippers

The Feetech section lists serial ports and the servo IDs found on each port:

```text
/dev/ttyACM0: ids=[0]
/dev/ttyACM1: ids=[1]
```

For each physical gripper, write down:

- which side it belongs to: `left` or `right`
- the serial port: `/dev/ttyACM0`, `/dev/ttyUSB0`, etc.
- the detected servo ID from `ids=[...]`

Then edit the `feetech` section in `configs/rig.yaml`. Put the detected serial
port in `port` and the detected ID in `servo_id`:

```yaml
feetech:
  left:
    servo_id: 0
    port: /dev/ttyACM0
  right:
    servo_id: 1
    port: /dev/ttyACM1
```

If both grippers are connected through the same Feetech bus adapter, they may
use the same `port` and different `servo_id` values:

```yaml
feetech:
  left:
    servo_id: 0
    port: /dev/ttyUSB0
  right:
    servo_id: 1
    port: /dev/ttyUSB0
```

If each gripper has its own USB adapter, they usually use different `port`
values.

### Cameras

The camera section lists USB camera devices. A single camera often exposes two
`/dev/video*` nodes:

```text
USB Camera: USB Camera (...):
  /dev/video2
  /dev/video3
USB Camera: USB Camera (...):
  /dev/video4
  /dev/video5
```

Use the first `/dev/video*` node for each physical camera unless testing shows
the second one is the usable stream. In the example above, if `/dev/video2` is
the left wrist camera and `/dev/video4` is the right wrist camera, edit
the `cameras` section in `configs/rig.yaml` like this:

```yaml
cameras:
  left_wrist:
    index_or_path: /dev/video2
  right_wrist:
    index_or_path: /dev/video4
```

`index_or_path` can be either a numeric OpenCV index such as `0` or an explicit
device path such as `/dev/video2`. Prefer `/dev/video*` paths during setup
because they match the `handumi-setup-ports` output directly.

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

Hold the gripper at **mid-travel**, press ENTER; the script reports
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
[README_tcp_offset.md](README_tcp_offset.md).

## Troubleshooting

### Feetech serial ports shows `none`

`Feetech serial ports -> none` means Linux did not create any `/dev/ttyACM*`
or `/dev/ttyUSB*` serial device. This is different from camera devices, which
show up as `/dev/video*`.

First check whether the serial device node exists:

```bash
ls /dev/ttyACM* /dev/ttyUSB*
```

If no devices exist, check whether the USB adapter itself is visible:

```bash
lsusb
journalctl -k -f
```

Common Feetech USB serial adapters show up as QinHeng/CH34x devices such as
`1a86:55d3` or `1a86:7523`. If `lsusb` shows the adapter but there is still no
`/dev/ttyUSB*`, the USB cable and hub are probably fine, but the serial driver
did not bind.

On Arch Linux, this often happens right after a system update: the running
kernel and installed module tree no longer match. Check:

```bash
uname -r
modinfo ch341
ls /usr/lib/modules/$(uname -r)
```

If `modinfo ch341` fails or `/usr/lib/modules/$(uname -r)` is missing, reboot:

```bash
sudo reboot
```

After rebooting, reconnect the Feetech adapter and check again:

```bash
ls /dev/ttyACM* /dev/ttyUSB*
handumi-setup-ports
```

If `/dev/ttyUSB0` exists but `handumi-setup-ports` reports it as unavailable
or permission denied, add your user to the serial device group shown by the
script. On Arch this is usually `uucp`; on Debian/Ubuntu it is usually
`dialout`:

```bash
sudo usermod -aG uucp $USER
```

Then log out and back in.

### Feetech port exists but no IDs are detected

If `handumi-setup-ports` shows a serial port but `ids=[]`, the adapter opened
successfully but no servo replied. Check:

- the gripper has power
- TX/RX/GND wiring between the Feetech adapter and servo
- the configured baudrate is still `1000000`
- the servo ID is outside the default scan range

To scan a wider ID range:

```bash
handumi-setup-ports --start-id 0 --end-id 253
```

### Gripper side is swapped

If the right gripper changes when you move the left gripper, swap the `left`
and `right` entries under `feetech` in `configs/rig.yaml`. Keep the `servo_id`
and `port` together as a pair.

### Camera side is swapped

If the right camera preview shows the left wrist, swap the `index_or_path`
values for `left_wrist` and `right_wrist` under `cameras` in
`configs/rig.yaml`.

### Camera appears twice

Many USB cameras expose two `/dev/video*` nodes. Usually the first node in the
pair is the video stream. If the configured camera fails to open, try the second
node from the same camera block.

### Ticks do not change

If `handumi-calibrate-grippers monitor` connects but `ticks` stay constant while
you move the gripper, confirm that the `feetech` section in `configs/rig.yaml`
points to the servo for that gripper. If the correct servo is selected,
re-home the servo and recalibrate width.

### Width flips or saturates near open/close

The encoder range may be crossing the 0/4095 wrap point. Run
`handumi-home-servos` with the gripper held at mid-travel, then run width
calibration again.
