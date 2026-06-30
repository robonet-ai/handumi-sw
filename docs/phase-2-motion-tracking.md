# Phase 2 Motion Tracking

This document maps the next HandUMI phase after the cameras + Feetech
checkpoint works on physical hardware.

The reference implementation for the Meta Quest path is `../axol-vr`. HandUMI
should reuse the same architectural pattern, not the Axol-specific data
contract.

```text
HandUMI gripper motion from Meta Quest controller
+ gripper opening from Feetech encoder
        |
        v
Python live state
        |
        v
Rerun for cameras/scalars + Viser for 3D robot/workspace
        |
        v
same state contract can later be recorded as raw LeRobot data
```

## Current Status

Already in `handumi-sw`:

- `python -m handumi.capture.teleoperate_handumi` streams USB cameras + Feetech width to Rerun.
- `src/handumi/robots/sim.py` contains `ViserSim`.
- `src/handumi/robots/registry.py` exposes `runtime.make_sim()`.
- `src/handumi/replay/pico_ik.py --visualize` can update a Viser robot.
- `src/handumi/retargeting/compare_axis.py` uses Viser diagnostics.

Missing:

- Meta Quest tracking backend.
- Quest Browser WebXR client.
- Python WebSocket receiver for Quest frames.
- Live loop that merges Quest pose + Feetech width and updates Viser.
- Recording path that writes Quest poses into the 16D HandUMI raw state.

## Decision

Use `../axol-vr` as the primary guide.

Keep from `axol-vr`:

- Vite + React + `@react-three/xr` app opened in Quest Browser.
- WebSocket connection from headset to Python.
- Per-XR-frame streaming from browser to workstation.
- Headset HUD for connection/recording state.
- Server-pushed state feedback such as `saving` and `error`.
- Button edge handling for reset/start/stop.

Change for HandUMI:

- Use `XRInputSource.gripSpace` for gripper pose, not `targetRaySpace`.
- Feetech is the only source of gripper opening. Quest trigger values are UI
  controls only.
- Elbows/body tracking are optional diagnostics, not required fields.
- Payload names should be HandUMI names: `left.pose`, `right.pose`, `state`,
  `buttons`, not `l_ee`, `r_ee`, `l_grip`, `r_grip`.
- Python receiver lives in `handumi.tracking.meta_quest`, not in an Axol SDK.
- The live robot view is Viser; camera/scalar monitoring remains Rerun.

## Target Flow

```text
Quest controller mounted on each HandUMI gripper
  WebXR gripSpace pose per side
  Quest buttons for reset/start/stop/mode
        |
        v
Quest Browser WebSocket client
        |
        v
Python WebSocket server
        |
        v
handumi.tracking.meta_quest
        |
        v
HandUMI raw state
  left  [x, y, z, qx, qy, qz, qw]
  right [x, y, z, qx, qy, qz, qw]
  left_width_m  from Feetech
  right_width_m from Feetech
        |
        v
retargeting / IK
        |
        v
ViserSim robot update
```

Live visualization command:

```bash
handumi-live-viser \
  --tracking-backend meta_quest \
  --feetech-config configs/feetech.yaml \
  --embodiment piper \
  --port 8000
```

Recording remains a separate mode:

```bash
PYTHONPATH=src python scripts/record_handumi.py \
  --config configs/handumi.yaml \
  --tracking-backend meta_quest
```

## WebXR Contract

Use:

```text
referenceSpace = local-floor
poseSpace      = XRInputSource.gripSpace
units          = meters
quaternion     = [x, y, z, w]
```

WebXR coordinates:

```text
+X = right
+Y = up
-Z = forward
```

HandUMI internal pose layout:

```text
[x, y, z, qx, qy, qz, qw]
```

The Python side should preserve the WebXR pose first, then apply explicit
calibration transforms:

```text
quest_local_floor
  -> handumi_workspace
  -> robot_base_left / robot_base_right
```

Do not hide axis fixes inside the WebXR client. Axis transforms should be
testable Python code.

## WebSocket Payload

Initial headset to Python message:

```json
{
  "type": "handumi_xr_frame",
  "seq": 1234,
  "t_ms": 456789.12,
  "space": "local-floor",
  "state": "teleop",
  "left": {
    "tracked": true,
    "pose": {
      "position": [0.12, 1.24, -0.48],
      "quaternion": [0.01, 0.72, 0.02, 0.69]
    },
    "buttons": {
      "trigger": 0.0,
      "grip": 0.0,
      "thumbstick": [0.0, 0.0],
      "x": false,
      "y": false
    },
    "profiles": ["meta-quest-touch-plus", "oculus-touch-v3", "oculus-touch"]
  },
  "right": {
    "tracked": true,
    "pose": {
      "position": [0.20, 1.24, -0.50],
      "quaternion": [0.0, 0.70, 0.0, 0.71]
    },
    "buttons": {
      "trigger": 0.0,
      "grip": 0.0,
      "thumbstick": [0.0, 0.0],
      "a": false,
      "b": false
    },
    "profiles": ["meta-quest-touch-plus", "oculus-touch-v3", "oculus-touch"]
  }
}
```

Rules:

- If `gripSpace` pose is unavailable, set `tracked: false`.
- Keep `profiles` for debugging controller variants.
- Keep `trigger` and `grip` values for UI/debug only.
- Do not use `trigger` or `grip` as gripper width.
- Gripper width comes from Feetech in Python and is merged after this frame is
  received.

## Button Mapping

Follow the same state-machine idea as `axol-vr`, with HandUMI semantics:

```text
left X   = reset workspace calibration / pose offset
left Y   = exit XR session
right A  = start/stop recording when recording mode is enabled
right B  = toggle teleop/data_collection mode
both grip buttons = enable/disable tracking lock, optional
```

Gamepad layout should use WebXR `xr-standard`:

```text
button 0 = trigger
button 1 = squeeze / grip
button 3 = thumbstick press
axes 2/3 = thumbstick x/y
```

Keep server-pushed states:

```text
teleop
data_collection
recording
saving
error
```

`saving` and `error` should be controlled by Python so the headset UI cannot
start another recording while an episode is being written.

## Mapping From axol-vr

Reference files:

```text
../axol-vr/app/src/App.tsx
../axol-vr/packages/axol-vr-client/src/AxolVRClient.tsx
../axol-vr/packages/axol-vr-client/src/useAxolVRClient.ts
../axol-vr/packages/axol-vr-client/src/types.ts
```

Port to HandUMI:

```text
AxolVRClient.tsx
  -> HandUMIXRClient.tsx
  - read left/right XRInputSource each frame
  - use source.gripSpace
  - send handumi_xr_frame JSON
  - keep button edge logic

useAxolVRClient.ts
  -> useHandUMIQuestClient.ts
  - keep reconnect logic
  - keep server state feedback
  - keep wss://hostname:port/ws

types.ts
  -> types.ts
  - rename AxolState to HandUMIQuestState
  - rename AxolPoseData to HandUMIXRFrame

App.tsx
  -> Quest browser app
  - keep host input + Start button + headset HUD
  - replace Almond/Axol labels with HandUMI labels
```

Do not port:

```text
targetRaySpace as controller pose
required elbow/body tracking
l_grip / r_grip as gripper opening
Axol-specific SDK naming
Axol-specific robot assumptions
```

## Proposed Files

```text
clients/meta_quest_webxr/
  package.json
  index.html
  src/
    App.tsx
    HandUMIXRClient.tsx
    useHandUMIQuestClient.ts
    types.ts

src/handumi/tracking/
  meta_quest.py          # receiver, frame model, latest-frame buffer
  transforms.py          # explicit frame/calibration transforms

src/handumi/capture/
  live_viser.py          # Quest + Feetech + IK + Viser loop

configs/
  tracking_meta_quest.yaml
```

Python entry points:

```text
handumi-live-viser
PYTHONPATH=src python scripts/record_handumi.py --tracking-backend meta_quest
```

Direct dependencies to add when implementing:

```text
viser
websockets
```

`viser` should be direct because HandUMI imports it directly. `websockets` is
the simplest Python server dependency for the Quest receiver.

## HTTPS / WSS Requirement

Quest Browser WebXR should be treated as a secure-context workflow.

Use one of these:

```text
local HTTPS dev server + trusted certificate
deployed HTTPS static client + WSS Python endpoint
reverse proxy that terminates TLS and forwards to Python
```

The `axol-vr` client already assumes:

```text
wss://hostname:port/ws
```

HandUMI should keep that convention.

## Live Viser Loop

Loop:

1. Receive latest Quest left/right `gripSpace` poses.
2. Read Feetech left/right gripper width.
3. Build the 16D HandUMI state:

```text
0   left_x
1   left_y
2   left_z
3   left_qx
4   left_qy
5   left_qz
6   left_qw
7   right_x
8   right_y
9   right_z
10  right_qx
11  right_qy
12  right_qz
13  right_qw
14  left_gripper_width_m
15  right_gripper_width_m
```

4. Apply workspace calibration and axis transform.
5. Convert raw state into left/right target poses.
6. Solve IK for the selected embodiment.
7. Insert gripper width into the command vector.
8. Call `ViserSim.motion_control(left=..., right=...)`.
9. Log tracking FPS, dropped frames, Feetech width, and camera frames to Rerun.

Recommended split:

```text
Rerun = USB cameras + Feetech ticks/mm/m + tracking FPS
Viser = robot pose + controller frames + workspace axes
```

## PICO Compatibility

PICO should match the same internal contract:

```text
[x, y, z, qx, qy, qz, qw]
```

Current PICO path:

```text
src/handumi/tracking/pico.py
read_pico_frame()
observation.pico.left_controller_pose
observation.pico.right_controller_pose
```

Retargeting already expects `[qx, qy, qz, qw]` and converts with
`quaternion_xyzw_to_matrix()` in:

```text
src/handumi/retargeting/handumi_to_robot.py
```

The Meta Quest backend should normalize to this same shape.

## Acceptance Criteria

Phase 2 is ready when:

- Quest Browser connects to `handumi-live-viser` over WSS.
- Moving the left/right HandUMI grippers updates left/right frames in Viser.
- Opening/closing each physical gripper updates Feetech width in Rerun.
- The Viser robot follows the controller-mounted grippers.
- Reset/calibration is explicit and repeatable.
- `PYTHONPATH=src python scripts/record_handumi.py --tracking-backend meta_quest` writes the same 16D raw state
  layout used by the current dataset schema.
- No gripper width is taken from Quest trigger values.

## References

- Local guide: `../axol-vr`
- WebXR Device API: https://www.w3.org/TR/webxr/
- WebXR Gamepads Module: https://www.w3.org/TR/webxr-gamepads-module-1/
- Meta WebXR overview: https://developers.meta.com/horizon/documentation/web/webxr-overview/
- Meta WebXR controllers guide: https://developers.meta.com/horizon/documentation/web/webxr-first-steps-chapter2/
- WebXR input profiles: https://github.com/immersive-web/webxr-input-profiles
- Viser docs: https://viser.studio/
