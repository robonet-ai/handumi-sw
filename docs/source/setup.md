# HandUMI Setup and Calibration

Complete this page before recording. No robot arm is required: these steps
configure HandUMI, its tracking device, cameras, grippers, and workspace.
Some calibrations are permanent for one physical assembly; the table/session
alignment must be checked each session.

| Calibration | Repeat when |
| --- | --- |
| Servo homing and opening width | Servo, linkage, or gripper geometry changes |
| Camera intrinsics | Camera, resolution, or focus changes |
| Controller-to-camera mount | A controller or wrist camera mount moves |
| Controller-to-TCP | The controller/gripper mount or physical tool changes |
| Table/session frame | Each session, relocalization, or tracking reset |

## 1. Map HandUMI Hardware

`install.sh` creates the ignored machine-local `configs/rig.yaml`. Inspect the
connected cameras and Feetech adapters:

```bash
handumi-setup-ports
```

Reconnect one physical device at a time and assign its port under `cameras`
or `feetech` in `configs/rig.yaml`. Robot-arm buses do not belong in this
recording setup; configure them only for real-robot teleoperation.

Set new Feetech IDs only when required:

```bash
handumi-set-servo-id --port /dev/ttyUSB0 --new-id 0
handumi-set-servo-id --port /dev/ttyUSB0 --new-id 1
```

:::{dropdown} Hardware mapping details
Two grippers may share one serial `port` only when they use different
`servo_id` values. With separate USB adapters, each side normally has its own
port.

A USB camera commonly exposes two `/dev/video*` nodes. Start with the first
node reported for each physical camera and confirm the stream. Map
`left_wrist`, `right_wrist`, and `workspace` explicitly in `configs/rig.yaml`.

Keep these machine-local paths in `configs/rig.yaml`; do not commit them as
portable project configuration.
:::

## 2. Calibrate the Grippers

First confirm that both encoders change smoothly while opening and closing:

```bash
handumi-calibrate-grippers monitor
```

Home each servo with the gripper held at **mid-travel**. This centers the
encoder range and avoids crossing the 0/4095 wrap point:

```bash
handumi-home-servos
handumi-home-servos --side right  # one side only
```

Then calibrate the physical opening width:

```bash
handumi-calibrate-grippers calibrate
handumi-calibrate-grippers calibrate --side right
```

For each side, enter the maximum opening in millimeters, place the gripper fully
open and press Enter, then fully close it and press Enter. The result is stored
in `~/.cache/handumi/calibration.yaml`. Open and close each gripper again with
`monitor` and confirm that width increases toward fully open without flipping
or saturating.

## 3. Connect Tracking

### Meta Quest

Enable Developer Mode, connect the headset over USB, authorize `adb`, and
install [HandUMI Quest App](https://github.com/robonet-ai/handumi-quest-app/releases):

```bash
wget https://github.com/robonet-ai/handumi-quest-app/releases/download/v0.2.1/handumi-quest-app-v0.2.1.apk
adb install -r handumi-quest-app-v0.2.1.apk
adb shell ip route  # find the address after "src"
```

Set that address as `meta_quest.connection.quest_ip` in `configs/rig.yaml`.
Launch the app from Library → Unknown Sources and keep it in the foreground.

```bash
python -m handumi.tracking.meta_quest --config configs/rig.yaml
```

A healthy stream reports steady FPS and both controllers tracked.

### PICO

Install the [XRoboToolkit PC Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases)
and follow the current [XR Robotics headset instructions](https://github.com/XR-Robotics).
Start the PC service, then launch streaming:

```bash
bash /opt/apps/roboticsservice/runService.sh
```

Use `127.0.0.1:63901` for USB or the workstation IP with `--pico-wifi`.

Smoke-test a short capture before calibration:

```bash
handumi-record --device pico --skip-feetech \
  --repo-id local/pico-smoke \
  --output-dir outputs/datasets/pico-smoke \
  --task "pico smoke" --num-episodes 1 --episode-time-s 10
```

Healthy output reports `xrobotoolkit_sdk initialised` without repeated
`still waiting for PICO data` messages.

## 4. Calibrate Cameras and Workspace

Fix the 5 × 7 ChArUco board flat at its marked table position, with IDs 15 and
16 nearest the operator. Its center defines the table origin: +X right, +Y away,
and +Z up.

### Camera Intrinsics

```bash
handumi-calibrate-spatial intrinsics --camera left_wrist
handumi-calibrate-spatial intrinsics --camera right_wrist
handumi-calibrate-spatial intrinsics --camera workspace
```

Move the board throughout each image and vary distance and inclination. The
tool automatically accepts a distinct valid view every two seconds. Repeat
after changing camera, resolution, or focus.

### Controller-to-Camera Mounts

Keep the board fixed. Move the complete HandUMI through varied roll, pitch, and
yaw poses, pausing briefly for each automatic capture. Keep the controller
tracking ring visible to the headset.

Choose the tracking device explicitly. Global options such as `--device`,
`--pico-wifi`, and `--quest-ip` come before the subcommand.

Meta Quest:

```bash
handumi-calibrate-spatial --device meta mount --side left
handumi-calibrate-spatial --device meta mount --side right
```

PICO:

```bash
handumi-calibrate-spatial --device pico --pico-mode mandos mount --side left
handumi-calibrate-spatial --device pico --pico-mode mandos mount --side right
```

PICO calibration relies on live XRoboToolkit snapshots, so hold the HandUMI
steady while each view is accepted. Use `--pico-wifi` for a wireless PICO setup.

Repeat only if a controller or wrist-camera mount moves.

### Session/Table Frame

With the board still at its marked position and the headset fixed as it will be
during recording, solve the table frame for the same tracking device.

```bash
handumi-calibrate-spatial --device meta session --side left
handumi-calibrate-spatial --device meta visualize
```

For PICO:

```bash
handumi-calibrate-spatial --device pico --pico-mode mandos session --side left
handumi-calibrate-spatial --device pico --pico-mode mandos visualize
```

Inspect all cameras and both TCP trails in Rerun. The table surface must align
with `z=0`. If only the workspace-camera stage fails, retry it with:

```bash
handumi-calibrate-spatial workspace
```

Remove the board without moving the table, cameras, or headset. Repeat the
session calibration after relocalization or a tracking reset. The saved
`outputs/calibration/session.yaml` records `tracking_device` and
`table_from_device`; use it only with the same `--device`.

## 5. Calibrate the HandUMI Tool Tip

Controller-to-TCP reconstructs the physical HandUMI tool-tip pose from each
tracked controller. It is a property of the HandUMI gripper/tool and controller
mount, not of a connected robot arm. Recalibrate only when that physical
assembly changes. Fix the tip in a firm indentation. For 25 seconds, keep it
fixed while rotating the tracked assembly through varied orientations. If the
same calibrated HandUMI tool is used for another robot, validate and copy the
result to that robot's identity-bound calibration path; the wizard never
silently assumes two physical tool assemblies are identical.

:::{dropdown} Capture and fit both sides
Select the tracking device:

```bash
TRACKING_DEVICE=meta   # or pico
```

Capture the left side:

```bash
LEFT_RUN="outputs/tcp_pivot_left_$(date +%Y%m%d_%H%M%S)"
handumi-record --device "$TRACKING_DEVICE" --skip-feetech --only-left-camera \
  --repo-id local/tcp_pivot_left --output-dir "$LEFT_RUN" \
  --task "tcp pivot left" --num-episodes 1 --episode-time-s 25 \
  --tracking-loss-timeout-s 3 --no-sounds

handumi-calibrate-tcp-offset pivot --device "$TRACKING_DEVICE" --side left \
  --parquet "$LEFT_RUN/data/chunk-000/file-000.parquet" --episode 0 \
  --output outputs/calibration/controller_tcp_candidate.yaml
```

Repeat for the right side:

```bash
RIGHT_RUN="outputs/tcp_pivot_right_$(date +%Y%m%d_%H%M%S)"
handumi-record --device "$TRACKING_DEVICE" --skip-feetech --only-right-camera \
  --repo-id local/tcp_pivot_right --output-dir "$RIGHT_RUN" \
  --task "tcp pivot right" --num-episodes 1 --episode-time-s 25 \
  --tracking-loss-timeout-s 3 --no-sounds

handumi-calibrate-tcp-offset pivot --device "$TRACKING_DEVICE" --side right \
  --parquet "$RIGHT_RUN/data/chunk-000/file-000.parquet" --episode 0 \
  --output outputs/calibration/controller_tcp_candidate.yaml
```
:::

Inspect the result:

```bash
handumi-calibrate-tcp-offset inspect \
  outputs/calibration/controller_tcp_candidate.yaml
```

Accept a fit when RMS is below 0.50 cm, maximum error is below 1.00 cm, and
condition is below 500. High RMS means the tip probably slipped; a high
condition number means the capture lacked rotational variety.

:::{dropdown} Verify and promote the calibration
Before editing the project calibration, repeat a short pivot capture. Rotate
around a stationary tip and confirm the reported residual stays within the
acceptance limits. Touch the same point with both tips and confirm their
calibrated positions coincide; touching the table should place both tips near
`z=0` after session calibration.

Pivot fitting calibrates translation, not orientation. Preserve the official
quaternions and symmetrize only the measured positions:

```text
x = (left.x + right.x) / 2
y = (left.y - right.y) / 2
z = (left.z + right.z) / 2
left.position  = [x,  y, z]
right.position = [x, -y, z]
```

Update only `position` in `configs/calibration/${TRACKING_DEVICE}_controller_tcp.yaml`
or the robot-specific calibration file declared in `configs/robots/<robot>.yaml`,
then run:

```bash
uv run pytest -q tests/tracking/test_transforms.py \
  tests/scripts/test_replay_in_sim.py
```
:::

## 6. Run the Recording Preflight

Run one read-only preflight immediately before creating a dataset. For the
Meta controller profile:

```bash
handumi-preflight \
  --device meta \
  --expected-package com.handumi.questapp \
  --expected-version 0.2.1 \
  --session-calibration outputs/calibration/session.yaml \
  --controller-tcp-calibration configs/calibration/meta_controller_tcp.yaml \
  --output-dir outputs/datasets
```

For the separate body diagnostic profile, require packet v2 and the measured
profile explicitly:

```bash
handumi-preflight \
  --device meta --require-body \
  --expected-package com.handumi.questapp.bodyprobe \
  --expected-version 0.1.2 \
  --expected-protocol-version 2 \
  --session-calibration outputs/calibration/session.yaml \
  --body-profile configs/body_profile.yaml \
  --controller-tcp-calibration configs/calibration/meta_controller_tcp.yaml \
  --output-dir outputs/datasets
```

For PICO, start the pinned XRoboToolkit PC service first. The preflight reads an
existing stream without restarting the service:

```bash
handumi-preflight \
  --device pico --pico-mode mandos \
  --expected-protocol-version 2 \
  --session-calibration outputs/calibration/session.yaml \
  --controller-tcp-calibration configs/calibration/pico_controller_tcp.yaml \
  --output-dir outputs/datasets
```

The preflight verifies Quest/PICO transport, expected build/protocol fields,
foreground HMD/controller tracking, body activation/calibration when required,
clock diagnostics, camera identity/class/mode/frame production, duplicate
camera identities and USB topology, Feetech identity/side/servo/motion/range,
calibration files, dependencies, output writability/free space, and local
Quest/Rerun/Viser process or port conflicts. It resolves symlinks and checks
device class, so a stale wrist-camera path that now points to a serial adapter
fails even though the path exists.

Normal operation is read-only and never creates a dataset. To diagnose a USB
re-enumeration interactively, first preview a remap:

```bash
handumi-preflight ... --interactive-remap
```

Only after reviewing the stable identity and USB path, opt into the atomic
machine-local write:

```bash
handumi-preflight ... --interactive-remap --write-rig
```

The write updates only camera/Feetech assignments in ignored
`configs/rig.yaml`; a failed replacement leaves the original intact. Rerun the
read-only preflight after any remap. Encoder motion here is a connectivity
check, not evidence that the full physical endpoint or 0–66 mm hardware gate
passed.

Next: [Record Demonstrations](record.md).
