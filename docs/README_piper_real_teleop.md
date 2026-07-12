# Piper Real Teleop

Real Piper teleop uses the same tracking, controller-to-TCP calibration,
retargeting, and PyRoki IK path as `handumi-teleop-sim`, then streams the
resulting Piper arm joint targets over CAN.

At the moment, `handumi-teleop-real` supports the AgileX Piper backend:

```bash
handumi-teleop-real --device pico --robot piper
```

## Prerequisites

Install the optional Piper dependency:

```bash
uv sync --extra piper
```

Finish the normal HandUMI setup first:

- [README_hardware_setup.md](README_hardware_setup.md) - guided Piper CAN,
  Feetech, and PICO hardware setup.
- [README_gripper_width.md](README_gripper_width.md) - servo IDs, servo
  homing, gripper-width calibration, and cameras.
- [README_pico.md](README_pico.md) or [README_quest.md](README_quest.md) -
  tracking device setup.
- [README_tcp_offset.md](README_tcp_offset.md) - controller-to-gripper TCP
  calibration.

Preview the motion in simulation before commanding hardware:

```bash
handumi-teleop-sim --device pico --robot piper
```

## Guided Hardware Setup

For a first Piper setup, run:

```bash
handumi-setup-hardware --robot piper --device pico
```

The wizard creates `configs/rig.yaml` if it does not exist, then writes the
machine-local hardware mapping into that file.

It performs these steps, described in detail in
[README_hardware_setup.md](README_hardware_setup.md):

1. Maps Piper CAN adapters by reconnecting the right arm first, then the left
   arm.
2. Checks and repairs the configured CAN interfaces with the requested bitrate
   and restart setting.
3. Maps Feetech gripper adapters by reconnecting the right gripper first, then
   the left gripper.
4. Prepares the PICO USB/ADB session when `--device pico` is used.

The setup may ask for `sudo` to bring CAN interfaces up or to add your user to
the serial-device group. If it updates serial permissions, close the session and
log in again before re-running the command.

Useful options:

```bash
handumi-setup-hardware --robot piper --device pico --skip-can-map
handumi-setup-hardware --robot piper --device pico --skip-feetech-map
handumi-setup-hardware --robot piper --device pico --skip-pico
handumi-setup-hardware --robot piper --device pico --bitrate 1000000 --restart-ms 100
```

- `--skip-can-map` keeps the existing CAN mapping in `configs/rig.yaml`.
- `--skip-can-repair` skips the CAN `ip link` repair step.
- `--skip-feetech-map` keeps the existing Feetech mapping.
- `--feetech-start-id` and `--feetech-end-id` control the servo ID scan range.
- `--skip-pico` skips PICO ADB preparation.
- `--skip-adb-check` keeps PICO setup from checking `adb devices`.
- `--rig-config <path>` uses a different local rig YAML file.

## Manual Rig Configuration

If you prefer to edit the rig file manually, start from the example:

```bash
cp configs/rig.example.yaml configs/rig.yaml
```

Set the Piper CAN section:

```yaml
robots:
  piper:
    can:
      bitrate: 1000000
      restart_ms: 100
      left_port: can0
      right_port: can1
```

The same `configs/rig.yaml` also stores Feetech and tracking settings. Keep
those sections aligned with [README_gripper_width.md](README_gripper_width.md)
and the tracking-device README.

To inspect CAN state manually:

```bash
ip -details link show can0
ip -details link show can1
```

The expected state is up, linked, configured at the rig bitrate, and not
`BUS-OFF`.

## Run

PICO:

```bash
handumi-teleop-real --device pico --robot piper
```

Meta Quest:

```bash
handumi-teleop-real --device meta --robot piper
```

Common options:

```bash
handumi-teleop-real --device pico --robot piper --side right
handumi-teleop-real --device pico --robot piper --space-start
handumi-teleop-real --device pico --robot piper --duration-s 60
handumi-teleop-real --device pico --robot piper --skip-can-repair
```

- `--side left|right|both` selects the controlled arm side.
- `--space-start` lets the keyboard Space key start idle arms.
- `--skip-feetech` disables gripper reading and double-clap start/reset; combine
  it with `--space-start`.
- `--controller-tcp-calibration <path>` overrides the default calibration file.
- `--pico-wifi` uses PICO over Wi-Fi instead of USB/ADB.
- `--skip-can-repair` leaves CAN setup to the user.

Runtime behavior:

- Tracking starts before the real arms move.
- The Piper arms home slowly to the configured start pose.
- Arms remain idle at home until a double clap starts them.
- A double clap while teleop is active clears anchors and returns enabled arms
  home; double clap again to start teleop from a fresh reference.
- If tracking is lost, anchors are cleared and the current joint target is
  held until tracking recovers.

## Troubleshooting

### `piper_sdk` import error

Install the Piper optional dependencies:

```bash
uv sync --extra piper
```

### Missing `configs/rig.yaml`

Run the guided setup or copy the example:

```bash
handumi-setup-hardware --robot piper --device pico
```

### CAN is down, missing, or `BUS-OFF`

Run the setup again without `--skip-can-repair`, then verify wiring and power:

```bash
handumi-setup-hardware --robot piper --device pico --skip-feetech-map --skip-pico
```

If `left_port` and `right_port` are the same, re-run the CAN mapping wizard.

### Feetech permission error

Let `handumi-setup-hardware` update the serial group, then close the session and
log in again. On Ubuntu this group is usually `dialout`; on Arch it is usually
`uucp`.

### PICO does not stream

Check that the XRoboToolkit PC service is running, the headset app is streaming,
and USB debugging is authorized. For USB mode, the headset app should target
`127.0.0.1:63901` after ADB reverse is prepared.

### Controllers are not tracked

Wake the controllers, make sure the PICO/Quest app is streaming controller
poses, and start teleop only after both enabled controllers report valid
tracking.

## Safety

Start with one arm when testing new hardware:

```bash
handumi-teleop-real --device pico --robot piper --side right
```

Keep the workspace clear, keep an emergency stop accessible, and validate the
same setup in `handumi-teleop-sim` before moving the physical robot.
