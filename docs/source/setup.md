# HandUMI Setup and Calibration

Ultima modificacion: 2026-07-15 11:26:49 -05 -0500

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
and follow the current [XR Robotics headset instructions](https://github.com/XR-Robotics#-get-started).
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

```bash
handumi-calibrate-spatial mount --side left
handumi-calibrate-spatial mount --side right
```

Repeat only if a controller or wrist-camera mount moves.

### Session/Table Frame

With the board still at its marked position and the headset fixed as it will be
during recording:

```bash
handumi-calibrate-spatial session --side left
handumi-calibrate-spatial visualize
```

Inspect all cameras and both TCP trails in Rerun. The table surface must align
with `z=0`. If only the workspace-camera stage fails, retry it with:

```bash
handumi-calibrate-spatial workspace
```

Remove the board without moving the table, cameras, or headset. Repeat the
session calibration after relocalization or a tracking reset.

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
Capture the left side:

```bash
LEFT_RUN="outputs/tcp_pivot_left_$(date +%Y%m%d_%H%M%S)"
handumi-record --device meta --skip-feetech --only-left-camera \
  --repo-id local/tcp_pivot_left --output-dir "$LEFT_RUN" \
  --task "tcp pivot left" --num-episodes 1 --episode-time-s 25 \
  --tracking-loss-timeout-s 3 --no-sounds

handumi-calibrate-tcp-offset pivot --device meta --side left \
  --parquet "$LEFT_RUN/data/chunk-000/file-000.parquet" --episode 0 \
  --output outputs/calibration/controller_tcp_candidate.yaml
```

Repeat for the right side:

```bash
RIGHT_RUN="outputs/tcp_pivot_right_$(date +%Y%m%d_%H%M%S)"
handumi-record --device meta --skip-feetech --only-right-camera \
  --repo-id local/tcp_pivot_right --output-dir "$RIGHT_RUN" \
  --task "tcp pivot right" --num-episodes 1 --episode-time-s 25 \
  --tracking-loss-timeout-s 3 --no-sounds

handumi-calibrate-tcp-offset pivot --device meta --side right \
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

Update only `position` in `configs/calibration/meta_controller_tcp.yaml`, then
run:

```bash
uv run pytest -q tests/tracking/test_transforms.py \
  tests/scripts/test_replay_in_sim.py
```
:::

Next: [Record Demonstrations](record.md).
