# OpenArm v1 Hardware Setup

Ultima modificacion: 2026-07-15 18:53:06 -05 -0500

This procedure prepares two physical OpenArm v1 arms for HandUMI real
teleoperation. It follows the official
[OpenArm v1 motor configuration guide](https://docs.openarm.dev/1.0/software/setup/motor-config/)
and adds the HandUMI-specific CAN mapping, Controller-to-TCP calibration, safe
startup home, and PICO activation checks.

## Safety and prerequisites

Before enabling or calibrating motors:

- Clear the complete arm workspace and keep the emergency stop reachable.
- Connect and power both arms, but stop every teleoperation process.
- Install `openarm-can-cli`, the OpenArm v1 zero-position calibrator, and the
  Python bindings with `uv sync --extra openarm`.
- Provide two SocketCAN interfaces, normally `can0` and `can1`.
- Configure motor IDs J1-J8 as send IDs `0x01-0x08`, with receive IDs
  `0x11-0x18`, on each arm.
- Configure every motor's internal data baudrate to 5 Mbps. HandUMI uses
  CAN-FD with a 1 Mbps nominal bitrate and 5 Mbps data bitrate.
- Prepare the correct Controller-to-TCP calibration for the tracking device.
  PICO and Meta calibrations are not interchangeable.

Mechanical-zero calibration moves joints automatically against their physical
stops. It is different from Controller-to-TCP calibration and must never run
while teleoperation is active.

## Install and map the two CAN interfaces

Run the guided setup. For a PICO rig without Feetech gripper sensing:

```bash
uv sync --extra openarm
uv run handumi-setup-hardware \
  --robot openarmv1 \
  --device pico \
  --skip-feetech-map \
  --skip-feetech-calibration \
  --controller-tcp-calibration /absolute/path/to/pico_controller_tcp.yaml
```

Unlike the Piper wizard, the OpenArm wizard does not require unplugging CAN
adapters. It lists the detected interfaces and asks which one is physically
connected to the right arm and which one is connected to the left arm. The
answer is stored under `robots.openarmv1.can` in `configs/rig.yaml`.

For example:

```yaml
robots:
  openarmv1:
    can:
      fd: true
      bitrate: 1000000
      dbitrate: 5000000
      right_port: can0
      left_port: can1
```

The setup then:

1. Configures both selected interfaces as CAN-FD `1M/5M`.
2. Verifies that the links are up and not bus-off.
3. Queries J1-J8 on each arm without commanding motion.
4. Stops if any ID is missing or cannot communicate at the configured rate.

After the mapping is saved, reuse it with `--skip-can-map`.

## Verify CAN and all motors manually

The normal state is `ERROR-ACTIVE`, not `ERROR-PASSIVE` or `BUS-OFF`:

```bash
ip -details link show can0
ip -details link show can1
```

Expected values include:

```text
state ERROR-ACTIVE
bitrate 1000000
dbitrate 5000000
```

Restore an interface that is down or was left at 10 Mbps by `discover`:

```bash
openarm-can-cli -i can0 can_configure
openarm-can-cli -i can1 can_configure
```

Verify all eight IDs on each mapped arm:

```bash
openarm-can-cli -i can0 show_param --id 1,2,3,4,5,6,7,8
openarm-can-cli -i can1 show_param --id 1,2,3,4,5,6,7,8
```

There must be eight responses per arm and no `NO RESPONSE FROM MOTOR`. J1-J7
must use MIT control; the HandUMI gripper J8 uses POS_FORCE.

### Motor internal baudrate recovery

Successful J1-J8 communication while the interface is at CAN-FD `1M/5M`
confirms that the active motor baudrate matches the rig. If a motor is not
found, use `discover` only as a diagnostic and immediately restore the
interface afterward:

```bash
openarm-can-cli -i can0 discover --full-scan -m 32
openarm-can-cli -i can0 can_configure
```

Repeat for `can1` only if required. `discover` scans several rates and leaves
the interface at 10 Mbps; it does not mean the normal HandUMI rate is 10 Mbps.

If a motor is confirmed at the wrong internal rate, follow the official guide
and change only that motor. For example, motor ID 1 on `can0`:

```bash
openarm-can-cli -i can0 change_baud --baudrate 5000000 --canid 1 --save
openarm-can-cli -i can0 can_configure
```

Flash writes are limited. Do not run `change_baud --save` routinely or from a
startup script. Recheck the motor after a power cycle.

## Calibrate mechanical zero

Calibrate one arm at a time. Before each run, place that arm approximately in
the official zero posture, close its gripper, clear the workspace, and prepare
the emergency stop.

For a rig mapped as right=`can0`, left=`can1`, calibrate the right arm first:

```bash
uv run handumi-setup-hardware \
  --robot openarmv1 \
  --device pico \
  --skip-can-map \
  --skip-feetech-map \
  --skip-feetech-calibration \
  --skip-pico \
  --calibrate-openarm-zero \
  --openarm-zero-side right \
  --controller-tcp-calibration /absolute/path/to/pico_controller_tcp.yaml
```

Type `CALIBRATE RIGHT` only after checking the physical arm. Wait for the
calibrator to finish and disable the motors. Then repeat for the left arm:

```bash
uv run handumi-setup-hardware \
  --robot openarmv1 \
  --device pico \
  --skip-can-map \
  --skip-feetech-map \
  --skip-feetech-calibration \
  --skip-pico \
  --calibrate-openarm-zero \
  --openarm-zero-side left \
  --controller-tcp-calibration /absolute/path/to/pico_controller_tcp.yaml
```

Type `CALIBRATE LEFT`. Do not use `openarm-can-cli set_zero` as a substitute:
that command writes the current position as zero without running the official
automatic mechanical-stop sequence.

## Measure each physical J8 range

The nominal OpenArm v1 range is closed=`0 rad`, open=`-60 degrees`, but the
useful physical endpoints can differ slightly after assembly and zero
calibration. HandUMI measures one gripper at a time while J8 remains disabled;
the operator places the jaws at their useful closed and open positions and the
SDK only reads feedback. J1-J7 are not registered or commanded.

Stop teleoperation, remove every object from the jaws, and calibrate the right
gripper:

```bash
uv run handumi-calibrate-openarm-grippers --side right
```

After typing `CALIBRATE RIGHT J8`, manually close the jaws until they just
touch and press Enter, then manually open them fully and press Enter. Repeat for
the left gripper:

```bash
uv run handumi-calibrate-openarm-grippers --side left
```

The independent endpoints are saved in
`~/.cache/handumi/openarmv1_grippers.yaml`. Real teleoperation loads this file
automatically. This calibration does not overwrite motor zero, the Feetech
encoder calibration, or Controller-to-TCP calibration.

## First real teleoperation

Start with a single side if the rig has not been validated before. The real
backend reads the current joints, moves slowly to the selected home, and only
then allows tracking to be anchored.

OpenArm provides three named startup poses:

- `down`: all arm joints at zero, intended for setup diagnostics.
- `arms_90`: both J4 joints at 90 degrees, also intended for diagnostics.
- `forward_open`: the default real-teleoperation pose. J2 spreads both elbows,
  J3 brings the hands toward the working area, and J4 points the arms forward.

When moving to `forward_open`, the backend first moves J1-J3 slowly while
holding the measured J4-J7 posture. It bends J4 to 90 degrees only after both
arms have lateral clearance from the center structure.

For both arms, PICO, the collision-safe default pose, and no Feetech sensors:

```bash
uv run handumi-teleop-real \
  --device pico \
  --robot openarmv1 \
  --side both \
  --home-pose forward_open \
  --space-start \
  --skip-feetech \
  --controller-tcp-calibration /absolute/path/to/pico_controller_tcp.yaml
```

Wait for:

```text
Real openarmv1 is at home
```

With both PICO controllers tracked, focus the terminal and press Space once,
without Enter. Successful activation prints:

```text
Space pressed; starting left/right
left arm anchored
right arm anchored
Teleop timer started
```

If PICO tracking is lost, Space cannot anchor the arms until tracking recovers.
The CAN streamer may still be sending hold commands; CAN traffic alone does not
prove that teleoperation is anchored.

## View real cameras inside the PICO

HandUMI uses the same XRoboToolkit Remote Vision protocol as the deployment
bridge in `wbcd-icra-2026-deployment`: commands travel through port `13579`
and length-prefixed H.264 video travels through port `12345`. Over USB the
directions are intentionally different:

```text
PICO command client -> adb reverse 13579 -> HandUMI bridge
HandUMI H.264 sender -> adb forward 12345 -> PICO decoder
```

The bridge uses the existing `ZEDMINI` source in the XRoboToolkit app. It does
not install or replace `video_source.yml`. This is an optional standalone
context-camera experiment: it does not import the robot backend, enable CAN,
or change the normal `handumi-teleop-real` command.

Close XRoboToolkit before creating the tunnels because the running app normally
owns device port `13579`. Also close OBS or any viewer that owns the selected
camera. Test video without enabling the robot:

```bash
uv run handumi-pico-camera \
  --camera /dev/video2 \
  --input-format mjpeg \
  --input-size 1280x720 \
  --fps 30 \
  --eye-y-offset 48
```

The default `--eye-y-offset 48` moves the complete image slightly downward in
both eyes. Use `0` for a centered image, a larger positive value to lower it
further, or a negative value to raise it.

When the terminal prints that Remote Vision is ready, open XRoboToolkit and use:

```text
Remote Vision source: ZEDMINI
Camera source IP: 127.0.0.1
Listen: enabled
```

One camera is fitted into each eye without stretching its aspect ratio. Up to
three cameras can use the deployment layout, with context in the center and
wrist cameras at the sides:

```bash
uv run handumi-pico-camera \
  --camera /dev/video2 \
  --left-camera /dev/video4 \
  --right-camera /dev/video6
```

After validating video alone, leave that terminal running and start normal
teleoperation in another terminal. The camera process is robot-independent:

```bash
uv run handumi-teleop-real \
  --device pico \
  --robot openarmv1 \
  --side both \
  --home-pose forward_open \
  --space-start \
  --skip-feetech \
  --controller-tcp-calibration configs/calibration/pico_controller_tcp.yaml
```

For three cameras add `--left-camera` and `--right-camera` to the standalone
`handumi-pico-camera` process. It reads one frame from every configured camera
before opening the stream, so a missing, busy, or incompatible camera fails
without involving the robot. If the stream stops, teleoperation continues and
the operator must decide whether remote operation remains safe.

## Troubleshooting

| Symptom | Meaning and action |
| --- | --- |
| CAN LEDs red after a program error | The fail-safe disabled the motors. Read the preceding exception before restarting. |
| Interface is `DOWN`, at 10 Mbps, or bus-off | Run `openarm-can-cli -i <port> can_configure`, then verify `1M/5M`. |
| Fewer than eight motors respond | Do not calibrate or teleoperate. Check power, CAN wiring, IDs `1-8`, and each motor's internal 5 Mbps rate. |
| `motor not in POS_FORCE mode` | J8 is not configured for the gripper position/force command expected by HandUMI. |
| `following error` | The measured joint did not follow the safe command. Do not increase the limit blindly; inspect the reported side and joint. |
| `home timeout` | A joint did not enter the home tolerance. The exception reports side, joint, measured value, and target. |
| Home is mechanically crooked | Recheck mechanical-zero calibration and assembly. Controller-to-TCP calibration is not used during startup home. |
| Space does nothing | Confirm `PICO controller data is available`, both controls are tracked, the terminal is focused, and `--space-start` is present. |
| Remote Vision connects but stays blank | Restart the bridge with the current code and press `Listen` again. Check `adb logcat` for decoder errors; XRoboToolkit requires 3-byte H.264 NALs to remain grouped inside the 4-byte Annex-B decoder unit. |
