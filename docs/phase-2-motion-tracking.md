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
Rerun for cameras/scalars + live 3D controller trajectory
   (Viser 3D robot follow-along is DEFERRED — see Phase 2 Scope)
        |
        v
same state contract can later be recorded as raw LeRobot data
```

## Wearable Form Factor

HandUMI is a portable, body-worn rig (same idea as the "Portable YUBI" setup).
The Meta Quest is **not worn on the head and renders no operator-facing UI**.

```text
Meta Quest HMD  -> mounted on the neck / upper chest, facing forward-down
                   used only as a 6DoF inside-out tracking base
Quest controllers -> one rigidly mounted on each HandUMI gripper
Wrist cameras   -> two gripper-mounted USB cameras only
Laptop          -> in a backpack, runs the Python server + Rerun + Viser
```

Consequences for this phase:

- **No headset HUD.** The operator cannot see the Quest display, so there is no
  in-VR UI. All operator feedback (connection, mode, recording, saving, error)
  is surfaced on the workstation (Rerun / terminal) and, optionally, controller
  haptics.
- **No front / torso camera this phase.** Only the two gripper-mounted wrist
  cameras are used. A front camera is a future addition, not part of Phase 2.
- **Native app, no immersive session needed.** The yubi-style Quest app reads
  OVR controller poses and streams them; it does not need an immersive WebXR
  render session. Launch the app once, mount the headset on the neck, and it
  keeps streaming. (This is a key reason native beats WebXR here — see
  [Decision](#decision).)
- **Headset cameras must see the controllers.** Mounted on the neck facing the
  workspace in front of the body, the Quest's inside-out cameras keep the
  grippers in view; controller visibility drives tracking quality.
- **Headset pose becomes the body/chest reference frame**, not an ignored
  field. It anchors the `quest_tracking_space -> handumi_workspace` calibration
  to the operator's torso instead of their eyes.

## Phase 2 Scope

Get the **motion tracking right first**. Viser is the 3D viewer that shows the
*robot* moving (IK-driven follow-along of the controller poses); it is useful
but **out of scope for now** and deferred to a later sub-phase.

```text
Phase 2A  (THIS phase — focus here)
  - Meta Quest HMD + controllers stream poses into Python
  - explicit, tested coordinate / calibration transforms
  - merge with Feetech width into the 16D HandUMI raw state
  - Rerun shows: gripper wrist cameras, Feetech width series,
    and a live 3D trajectory of each controller
  - recording writes the same 16D raw state layout

Phase 2B  (DEFERRED — do not build yet)
  - retargeting / IK on the live tracked poses
  - ViserSim robot follow-along ("see the robot move in 3D")
```

**[2B] status: implemented** — not as a separate `live_viser.py` script but as
the `--robot piper` flag on `handumi.capture.live_tracking_quest`, backed by
`handumi.capture.robot_follow.RobotFollower` (16D raw state -> IK -> ViserSim).
Sections below tagged **[2B]** describe the original plan; where they name
`live_viser.py`, read "`live_tracking_quest --robot piper`".

## Phase 2A — Three Steps

Build the motion tracking in three sequential steps, using `../yubi-sw`
(`airoa_quest/`) as the guide. Each step has a demonstrable outcome; do not
start a step until the previous one's "Done when" holds.

The headset app is the **prebuilt YubiQuestApp** (sideloaded from yubi-sw — see
[The Quest app](#the-quest-app)); we do not build one. Each Python step is
developed first against `mock_quest_sender.py` (which emits the YubiQuestApp wire
format), then validated against the real app — so the app does not block
progress.

```text
Step 1  The pose pipe       — get controller numbers into Python
Step 2  The transforms      — turn raw Unity poses into HandUMI-frame poses
Step 3  The live view       — merge Feetech + draw the Rerun 3D trajectory
```

### Step 1 — The pose pipe (receiver + contract)

- **Goal:** real controller poses flow into Python with timing + tracking flags.
- **Build:** `handumi/tracking/meta_quest.py` (TCP/JSON newline framing, frame
  dataclass model, latest-frame buffer, UDP NTP-style time-sync, `connected` /
  `streaming` / `fps` / `last_frame_age` diagnostics) and
  `handumi/tracking/mock_quest_sender.py`.
- **yubi-sw guide:** `airoa_quest_bridge/transport/tcp_json.py` (framing, UDP
  sync, metrics) and `QuestController.msg` / `QuestHmd.msg` (frame fields).
- **Done when:** against the mock, frames parse and fps/diagnostics print;
  against the real app, moving each controller changes raw `position` /
  `quaternion` and `tracked` / `valid`, and every sample carries both
  `device_time_ns` and `pc_monotonic_ns`. No transforms yet.
- **Status: implemented.** Validate with the mock (two terminals):

  ```bash
  python -m handumi.tracking.mock_quest_sender
  python -m handumi.tracking.meta_quest --quest-ip 127.0.0.1
  ```

  Tests: `python -m unittest discover -s tests/tracking`. For the real headset,
  point `--quest-ip` (or `configs/tracking_meta_quest.yaml`) at the Quest LAN IP.

### Step 2 — The transforms (tested calibration)

- **Goal:** raw Unity poses become correct left/right HandUMI-frame poses.
- **Build:** `handumi/tracking/transforms.py` (Unity→right-handed conversion,
  `handumi_workspace` calibration / reset offset, fixed controller→gripper-TCP
  mounting offset) and `tests/tracking/test_transforms.py`.
- **yubi-sw guide:** `_unity_pose_to_ros` plus the compose/invert quaternion
  helpers in `airoa_quest_bridge/quest_bridge_node.py`.
- **Done when:** unit tests pass (known Unity inputs → expected outputs, plus
  round-trips); a captured reset re-centers the workspace; left/right poses sit
  correctly relative to the body when the receiver is fed real frames.
- **Status: implemented.** `Pose` (compose/inverse/matrix), Unity→right-handed
  conversion (yubi mapping, verified consistent with the position map),
  `MountingOffsets` (controller→gripper-TCP, loads from
  `configs/tracking_meta_quest.yaml`), `WorkspaceCalibration.from_reference`
  (reset re-centering), and `gripper_pose_in_workspace` (full pipeline).
  Tests: `python -m unittest discover -s tests/tracking` (22 transform tests).
  Mounting offset in `configs/tracking_meta_quest.yaml`: position from CAD
  (X only so far), rotation measured live with
  `python scripts/setup/print_controller_pose.py` using the two-stance
  method (see the script docstring): read the quaternion with the bare
  controller held naturally (`q_A`), then mounted in the HandUMI pointing
  the same way (`q_B`); the offset is `conj(q_B) * q_A`. Needed because
  the controller mounts vertically, not in its natural handheld grip.

### Step 3 — The live view (Feetech merge + Rerun trajectory)

- **Goal:** the visible milestone — move the grippers, see the trajectory.
- **Build:** `handumi/capture/live_tracking_quest.py` (merge calibrated Quest poses +
  Feetech width into the 16D raw state; Rerun blueprint = wrist cameras +
  Feetech series + 3D controller trajectory with rolling trails) and the
  dedicated recorder `handumi/capture/record_handumi_quest.py`.
- **yubi-sw guide:** the Rerun 3D trajectory image is the target look (see
  [Rerun 3D Trajectory View](#rerun-3d-trajectory-view)).
- **Done when:** moving each gripper draws its trail in the Rerun 3D view;
  opening/closing updates Feetech width; recording writes the 16D layout; no
  gripper width comes from Quest triggers. (This is the Phase 2A acceptance
  bar.)
- **Status: live view implemented.** `handumi/capture/live_tracking_quest.py` +
  `handumi.capture.live_tracking_quest`: receiver → `gripper_pose_in_workspace` → 16D state,
  with a Rerun blueprint = wrist cameras + Feetech width series + a 3D
  `Spatial3DView` of each controller (axes + tip + rolling trail, left cyan /
  right magenta). Left **X** resets the workspace on the HMD pose (auto-inits on
  first tracked frame). Pure logic (`pose_to_state_vector`, `TrajectoryTrail`,
  calibration) is unit-tested; the full loop has a headless smoke test against
  the mock. Dry run:

  ```bash
  python -m handumi.tracking.mock_quest_sender
  python -m handumi.capture.live_tracking_quest --skip-cameras --skip-feetech
  ```

  **Recording: implemented** as a dedicated script (the PICO and Quest record
  paths are split, not flag-toggled):

  ```bash
  python -m handumi.capture.record_handumi_quest --quest-ip <QUEST_LAN_IP>
  ```

  `handumi/capture/record_handumi_quest.py` writes the same 16D raw state plus
  per-frame Quest metadata (`observation.quest.*`: calibrated controller/HMD
  poses, `tracked` flags, `device_time_ns` + `pc_monotonic_ns` + `seq` for
  offline alignment). Left **X** resets the workspace; right **A** starts/stops
  with `--button-control`. Covered by unit tests plus an end-to-end test that
  records a real LeRobot episode from the mock. The PICO recorder is now
  `scripts/record_handumi_pico.py`.

## Reference: yubi-sw motion tracking

`../yubi-sw` (airoa-org `yubi`) is the **primary reference for the motion
tracking part** — the Rerun trajectory image came from that project. Its
`airoa_quest/` package is a clean, production-grade Quest → workstation pose
pipeline. Study it alongside `../axol-vr`.

Patterns to reuse from yubi-sw:

- **Boundary coordinate conversion in tested Python.** The Quest streams Unity
  coordinates (X right, Y up, Z forward, left-handed); yubi converts to a
  right-handed robot frame *at the receive boundary*, not on the device:

  ```text
  position:    Unity(x, y, z)     -> (z, -x, y)
  quaternion:  Unity(x, y, z, w)  -> (z, -x, y, -w)
  ```

  HandUMI's `transforms.py` should do the analogous Quest→`handumi_workspace`
  conversion the same way: explicit and unit-tested (`quest_bridge_node`'s
  `_unity_pose_to_ros` is the model).

- **`tracked` / `valid` per-controller flags.** OVR reports whether each
  controller pose is currently tracked and valid. Carry these through so a
  dropped/occluded controller marks the pose unreliable instead of freezing the
  robot at a stale pose.

- **Timestamp metadata for offline alignment.** Every sample carries
  `device_time_ns` (Quest monotonic clock) and `pc_monotonic_ns` (PC receive),
  plus an NTP-style UDP time-sync side channel that estimates
  `offset = median(pc_monotonic_ns - device_time_ns)`. This lets Quest poses be
  re-aligned with camera/Feetech frames in post-processing. HandUMI should
  record both clocks per tracked frame.

- **Diagnostics on the workstation, not the headset.** yubi publishes
  `connected` / `streaming` / `fps` / `last_frame_age` / `*_tracked` /
  `*_valid` on a health channel. Because the HandUMI operator has no headset
  display, this is exactly the feedback surface we need (Rerun / terminal).

- **Controller→mount offset.** yubi composes a fixed controller→`hand_root`
  transform to get the actual hand frame from the raw OVR controller anchor.
  HandUMI's controller is rigidly bolted to the gripper, so the equivalent
  fixed **controller→gripper-TCP offset** belongs in `transforms.py`.

**Decided:** HandUMI uses the **prebuilt YubiQuestApp over TCP/JSON** (plus UDP
time-sync), not the `axol-vr` WebXR/WSS path. We do not build a headset app — we
reuse yubi's and parse its exact wire format. See [The Quest app](#the-quest-app),
[Decision](#decision), and [Transport & Network](#transport--network). `axol-vr`
survives only as a secondary reference for button/state-machine logic.

## The Quest app

We **reuse the prebuilt YubiQuestApp** from
[yubi-sw](https://github.com/airoa-org/yubi-sw) rather than building our own. It
streams OVR controller/HMD poses in the legacy TCP/JSON format documented in
[TCP/JSON Payload](#tcpjson-payload); our receiver parses exactly those keys and
`mock_quest_sender.py` emits them, so the Python side runs identically against
the mock and the real headset.

- **Install:** sideload the APK over USB with `adb` (Developer Mode required).
  Full step-by-step is in the project README, *Motion Tracking (Phase 2)*.
- **Compatibility:** our transport already mirrors yubi's (PC dials the Quest,
  ports `65432`/`42000`, `<B Q>`/`<B Q Q>` UDP sync), so it lines up out of the
  box. Verify field names on the first real run — the v0.1.0 APK is the source
  of truth for the wire format.
- **Building our own** HandUMI app (Unity/OVRPlugin) emitting the same contract
  is a possible future step (`clients/quest_app/`), not needed for Phase 2A.

## Current Status

Already in `handumi-sw`:

- `python -m handumi.capture.teleoperate_handumi` streams USB cameras + Feetech width to Rerun.
- `src/handumi/sim/viser_sim.py` contains `ViserSim`.
- `src/handumi/robots/registry.py` exposes `runtime.make_sim()`.
- `src/handumi/replay/pico_ik.py --visualize` can update a Viser robot.
- `src/handumi/retargeting/compare_axis.py` uses Viser diagnostics.

Implemented in Phase 2A (against the mock; pending validation on real hardware):

- Python TCP/JSON receiver + UDP time-sync (`handumi.tracking.meta_quest`).
- Tested calibration transforms (`handumi.tracking.transforms`).
- Live loop merging Quest pose + Feetech width into the 16D raw state and a
  Rerun 3D trajectory (`handumi.capture.live_tracking_quest`).
- Quest recorder writing the 16D state + `observation.quest.*`
  (`handumi.capture.record_handumi_quest`).

Remaining:

- Sideload the prebuilt YubiQuestApp and validate the live path on the headset.
- (Optional, later) a dedicated HandUMI Quest app.

## Decision

**Transport: go the yubi-sw way — a native Quest app that streams poses over
TCP/JSON with a UDP time-sync side channel.** `../yubi-sw` is the primary guide
for the Quest→workstation pipeline. `../axol-vr` is kept only as a *secondary*
reference for the button state-machine and reconnect ideas.

Why native-app over WebXR (`axol-vr`):

- The Quest is **neck-mounted with no display in use**. WebXR forces an
  immersive render session for nobody to watch; a native app streams poses
  without rendering, which fits an always-on body-worn tracker.
- Native gives **real clock alignment** (`device_time_ns` + `pc_monotonic_ns` +
  UDP sync) so Quest poses line up with camera/Feetech frames in the dataset.
- Native exposes OVR **`tracked`/`valid`** flags and a clean diagnostics channel
  — the feedback surface we need without a headset HUD.
- No HTTPS/WSS secure-context plumbing; a plain TCP socket is enough on the LAN.

Keep from `yubi-sw`:

- The **prebuilt YubiQuestApp** as the pose source (sideloaded, not built).
- TCP/JSON per-sample stream + UDP NTP-style time-sync.
- Unity→robot coordinate conversion done in **tested Python at the boundary**.
- Per-controller `tracked`/`valid` flags.
- `device_time_ns` + `pc_monotonic_ns` per sample for offline alignment.
- Diagnostics (`connected`/`streaming`/`fps`/`*_tracked`/`*_valid`) surfaced on
  the workstation.

Keep from `axol-vr` (secondary):

- Button edge handling and the reset/start/stop state-machine semantics.
- Server-owned states such as `saving` and `error`.

HandUMI specifics:

- **No headset HUD.** The Quest is body-mounted; all feedback goes to Rerun /
  terminal (and optionally controller haptics). See
  [Wearable Form Factor](#wearable-form-factor).
- Use each controller's **grip/anchor pose** mounted on the gripper; a fixed
  controller→gripper-TCP offset is applied in `transforms.py`.
- Feetech is the only source of gripper opening. Quest trigger/grip values are
  UI controls only.
- Keep `hmd.pose` as the body/chest reference frame for calibration.
- Python receiver lives in `handumi.tracking.meta_quest`.
- Camera/scalar monitoring + the live 3D controller trajectory are in Rerun.
  Viser robot follow-along is **[2B]** deferred.

## Target Flow

```text
Meta Quest HMD body-mounted on the neck (tracking base, no display use)
Quest controller mounted on each HandUMI gripper
  OVR anchor pose per side (Unity coords)
  Quest buttons for reset/start/stop/mode
        |
        v
Native Quest app (OVRPlugin)
  TCP/JSON pose stream + UDP time-sync
        |
        v
Python TCP receiver (handumi.tracking.meta_quest)
  + UDP NTP-style offset estimation
        |
        v
transforms.py  (Unity -> handumi_workspace, tested)
        |
        v
HandUMI raw state
  left  [x, y, z, qx, qy, qz, qw]
  right [x, y, z, qx, qy, qz, qw]
  left_width_m  from Feetech
  right_width_m from Feetech
        |
        v
Rerun: cameras + Feetech + 3D controller trajectory   (Phase 2A)
        |
        v
[2B] retargeting / IK -> ViserSim robot update          (deferred)
```

Phase 2A live tracking script (Rerun only — no robot/Viser):

```bash
python -m handumi.capture.live_tracking_quest \
  --quest-ip <QUEST_LAN_IP> \
  --feetech-config configs/feetech.yaml
```

**[2B, deferred]** live robot view (adds IK + Viser follow-along):

```bash
python scripts/live_viser.py \
  --quest-ip <QUEST_LAN_IP> \
  --feetech-config configs/feetech.yaml \
  --embodiment piper
```

Recording is a separate, dedicated script:

```bash
python -m handumi.capture.record_handumi_quest \
  --quest-ip <QUEST_LAN_IP> \
  --feetech-config configs/feetech.yaml
```

## Tracking Contract

The Quest app streams **raw Unity / OVR poses on the wire**. All axis fixes and
calibration happen in Python — never on the device.

```text
trackingSpace  = OVRCameraRig.trackingSpace (stage / local-floor equivalent)
poseSpace      = OVR controller anchor (the gripper-mounted controller)
units          = meters
quaternion     = [x, y, z, w]
```

Unity coordinates on the wire (left-handed):

```text
+X = right
+Y = up
+Z = forward
```

Python converts at the receive boundary to a right-handed frame (the yubi
mapping), then applies HandUMI calibration:

```text
position:    Unity(x, y, z)     -> (z, -x, y)
quaternion:  Unity(x, y, z, w)  -> (z, -x, y, -w)

quest_tracking_space
  -> handumi_workspace               (workspace calibration / reset offset)
  -> controller -> gripper-TCP        (fixed mounting offset)
  -> [2B] robot_base_left / robot_base_right
```

HandUMI internal pose layout stays `[x, y, z, qx, qy, qz, qw]`.

Do not hide axis fixes inside the Quest app. Every transform is testable Python
code in `transforms.py`.

## TCP/JSON Payload

This is the **YubiQuestApp legacy wire format** (the prebuilt app we reuse). The
app sends one **newline-delimited JSON object per sample** over TCP. Vectors are
`{x,y,z}` / `{x,y,z,w}` objects in **Unity** coordinates; Python converts and
timestamps on receive. `handumi.tracking.meta_quest.parse_frame` reads exactly
these keys (and `mock_quest_sender.py` emits them).

```json
{
  "ovrTimeNs": 123456789012345,
  "deltaTime": 0.0111,
  "hmdPosition": {"x": 0.02, "y": 1.10, "z": 0.05},
  "hmdRotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
  "leftControllerPosition": {"x": -0.18, "y": 0.95, "z": 0.32},
  "leftControllerRotation": {"x": 0.01, "y": 0.72, "z": 0.02, "w": 0.69},
  "leftTracked": true,
  "leftValid": true,
  "leftJoystick": {"x": 0.0, "y": 0.0},
  "leftThumbstickClick": false,
  "leftTriggerPressed": false,
  "leftGripPressed": false,
  "buttonXPressed": false,
  "buttonYPressed": false,
  "rightControllerPosition": {"x": 0.20, "y": 0.95, "z": 0.30},
  "rightControllerRotation": {"x": 0.0, "y": 0.70, "z": 0.0, "w": 0.71},
  "rightTracked": true,
  "rightValid": true,
  "rightJoystick": {"x": 0.0, "y": 0.0},
  "rightThumbstickClick": false,
  "rightTriggerPressed": false,
  "rightGripPressed": false,
  "buttonAPressed": false,
  "buttonBPressed": false,
  "hmdBattPct": 87,
  "leftBattPct": 90,
  "rightBattPct": 92,
  "hmdCharging": false
}
```

Field notes (how `parse_frame` maps them):

- Positions are meters, rotations `[x, y, z, w]`, both **Unity left-handed**.
  Python converts at the boundary (`transforms.unity_pose_to_handumi`).
- `ovrTimeNs` → `device_time_ns` (Quest monotonic clock). Python adds
  `pc_monotonic_ns = time.monotonic_ns()` on receive; the UDP sync channel gives
  the offset between them.
- `deltaTime` → `delta_time_s`. There is **no `seq`** in the legacy format, so
  `seq` defaults to 0.
- `buttonX/Y/A/BPressed` → `primary`/`secondary` (X/Y left, A/B right). Trigger
  and grip arrive as *pressed booleans only* (no analog), surfaced as 0.0/1.0.
- `*Tracked` = OVR is tracking this device; `*Valid` = pose usable this frame.
  When either is false, the pose is treated as unreliable downstream.

Rules:

- **Do not use trigger/grip as gripper width.** Width comes from Feetech in
  Python and is merged after the frame is received.
- The `state` machine is owned by Python (see [Button Mapping](#button-mapping));
  the app forwards raw button states and Python derives transitions.

## Button Mapping

Follow the same state-machine idea as `axol-vr`, with HandUMI semantics:

```text
left X   = reset workspace calibration / pose offset
left Y   = exit XR session
right A  = start/stop recording when recording mode is enabled
right B  = toggle teleop/data_collection mode
both grip buttons = enable/disable tracking lock, optional
```

Buttons are pressed by feel — the operator cannot see a headset menu — so each
state change must be confirmed on the workstation (Rerun / terminal), and
optionally via controller haptics, instead of an in-VR HUD.

These map onto the per-controller `buttons` block in the TCP/JSON payload:

```text
primary           = X (left) / A (right)   -> reset / start-stop
secondary         = Y (left) / B (right)   -> exit / toggle mode
trigger, grip     = analog 0..1            -> UI/debug only (never width)
thumbstick[x, y]  = analog axes            -> optional
thumbstick_click  = bool                   -> optional
```

The app forwards raw button states; **Python owns the state machine** and does
the edge detection, so the device never decides recording state.

Keep Python-owned states:

```text
teleop
data_collection
recording
saving
error
```

`saving` and `error` should be controlled by Python so the client cannot start
another recording while an episode is being written. Since there is no headset
UI, Python is the single source of truth for state and surfaces it on the
workstation.

## Porting Plan

Primary reference is `../yubi-sw` (`airoa_quest/`). Map its ROS-based pipeline
onto HandUMI's ROS-free Python:

```text
airoa_quest_bridge/transport/tcp_json.py
  -> handumi/tracking/meta_quest.py  (transport half)
  - keep newline-delimited TCP/JSON framing
  - keep the UDP NTP-style time-sync loop + median offset
  - keep connect/retry + fps / last_frame_age metrics
  - expose a latest-frame buffer instead of ROS publishers

airoa_quest_bridge/quest_bridge_node.py (_unity_pose_to_ros etc.)
  -> handumi/tracking/transforms.py
  - port the Unity->right-handed conversion verbatim, with unit tests
  - add handumi_workspace calibration + controller->gripper-TCP offset

QuestController.msg / QuestHmd.msg
  -> the TCP/JSON Payload above (dataclass frame model in meta_quest.py)
  - keep tracked/valid, device_time_ns, seq
```

Secondary reference is `../axol-vr`, **for logic only** (it is a WebXR app we
are not building):

```text
AxolVRClient.tsx button edge logic + AxolState machine
  -> Python state machine in the live loop
  - reset / start-stop / toggle-mode edge detection
  - server-owned saving / error states

useAxolVRClient.ts reconnect logic
  -> already covered by the yubi connect/retry transport
```

Do not build / do not port:

```text
WebXR / @react-three/xr browser client
immersive XR session + in-VR HUD / help / countdown / exit
targetRaySpace / gripSpace as the *only* pose source (use the OVR anchor)
required elbow/body tracking
trigger / grip as gripper opening
WSS / HTTPS secure-context plumbing
ROS 2 message/topic machinery from yubi (keep the patterns, drop rclpy)
```

The Quest **app itself** is the prebuilt **YubiQuestApp** (sideloaded), which
already emits the TCP/JSON above. Phase 2A's Python is developed against the
contract with a mock sender, then the real app is swapped in without changing
Python.

## Proposed Files

```text
src/handumi/tracking/
  meta_quest.py          # TCP/JSON receiver + UDP time-sync, frame model,
                         #   latest-frame buffer, tracked/valid + both clocks
  transforms.py          # Unity->workspace + mounting offset (unit-tested)
  mock_quest_sender.py   # emits the TCP/JSON contract for offline dev/tests

src/handumi/capture/
  live_tracking_quest.py          # Phase 2A: Quest + Feetech + Rerun trajectory (no robot)
  record_handumi_quest.py   # Phase 2A: dataset recorder (16D state + quest meta)
  record_handumi_pico.py    # PICO recorder (renamed from record_handumi.py)
  live_viser.py             # [2B] adds IK + ViserSim robot follow-along

scripts/
  live_tracking_quest.py          record_handumi_quest.py   record_handumi_pico.py

tests/tracking/   tests/capture/
  test_meta_quest.py  test_transforms.py  test_live_tracking_quest.py
  test_record_handumi_quest.py

configs/
  tracking_meta_quest.yaml   # quest_ip, tcp_port, sync_port, calibration

# Headset app: reuse the prebuilt YubiQuestApp (sideload). Building a dedicated
# clients/quest_app/ (Unity/OVRPlugin) is an optional future step.
```

Phase 2A is implemented end-to-end against the mock: `meta_quest.py`,
`transforms.py`, `mock_quest_sender.py`, `live_tracking_quest.py`,
`record_handumi_quest.py`, and their tests. `live_viser.py` is **[2B]**. The
headset app is the sideloaded YubiQuestApp (see [The Quest app](#the-quest-app)).

Python scripts:

```text
python -m handumi.capture.live_tracking_quest            # Phase 2A (visualize)
python -m handumi.capture.record_handumi_quest     # Phase 2A (record)
python scripts/live_viser.py               # [2B]
```

Direct dependencies to add when implementing:

```text
(none new for Phase 2A — TCP/UDP use the Python stdlib `socket`)
viser   # [2B] only, for the robot view
```

The Quest receiver needs no `websockets`/TLS dependency: a plain stdlib TCP
socket plus a UDP sync socket is enough on the LAN.

## Transport & Network

Plain LAN sockets, no TLS:

```text
TCP  : per-sample newline-delimited JSON pose stream
UDP  : NTP-style time-sync (PC pings, Quest echoes its clock)
```

Connection model (following yubi): the **PC dials the Quest** — the Quest app is
the TCP server, the workstation connects to `quest_ip:tcp_port` and binds the
UDP `sync_port`. Defaults mirror yubi (`tcp_port: 65432`, `sync_port: 42000`);
all configurable in `configs/tracking_meta_quest.yaml`.

```text
PC  --TCP connect-->  quest_ip:tcp_port     (pose stream in)
PC  <--UDP echo-->    quest_ip:sync_port    (clock offset)
```

Notes:

- The Quest needs a reachable LAN IP. On DHCP it can change; pin it or read it
  from the headset before a session.
- `pc_monotonic_ns` is stamped on receive; `device_time_ns` comes from the app;
  `offset = median(pc_monotonic_ns - device_time_ns)` aligns them.
- No secure context / certificates are required (this was a WebXR-only
  constraint that no longer applies).

## Live Tracking Loop

Loop. Steps 1–3, 4, 9, 10 are Phase 2A (build now). Steps 5–8 are **[2B]
deferred** (IK + Viser robot follow-along — do not build yet).

1. Receive latest Quest left/right controller poses (with `tracked`/`valid`).
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
5. **[2B]** Convert raw state into left/right target poses.
6. **[2B]** Solve IK for the selected embodiment.
7. **[2B]** Insert gripper width into the command vector.
8. **[2B]** Call `ViserSim.motion_control(left=..., right=...)`.
9. Log tracking FPS, dropped frames, Feetech width, and camera frames to Rerun.
10. Log the left/right controller poses to a Rerun 3D view so the operator can
    see the trajectory traced by each gripper.

For Phase 2A the loop ends at step 4 + the Rerun logging (9, 10): tracked poses
are calibrated and visualized, but no robot is driven yet.

Recommended split:

```text
Rerun = USB gripper cameras
      + Feetech ticks/mm/m
      + tracking FPS / dropped frames
      + 3D controller trajectory view        (Phase 2A)
Viser = robot pose + controller frames + workspace axes   [2B, deferred]
```

## Rerun 3D Trajectory View

Because the operator has no headset display, Rerun on the workstation is the
primary spatial feedback. In addition to the camera and gripper-width panels,
the Rerun blueprint must include a **3D view that shows the path traced by each
controller**, similar to a UMI-style trajectory plot:

```text
observation.tracking.left_pose   -> position + orientation axes (left)
observation.tracking.right_pose  -> position + orientation axes (right)
observation.tracking.left_trail  -> growing 3D line strip of left positions
observation.tracking.right_trail -> growing 3D line strip of right positions
```

Conventions:

- Use a fixed-length world grid so the trajectory has a stable reference.
- Color left and right consistently with the existing Feetech series
  (left = cyan, right = magenta) so panels read together.
- Log poses in the calibrated `handumi_workspace` frame, not raw
  `quest_local_floor`, so the trail matches the Viser robot view.
- Cap the trail to a rolling window (e.g. last N seconds) so the line strip
  does not grow unbounded during long sessions.

The blueprint extends the current `teleoperate_handumi` layout (wrist cameras +
gripper-width / normalized / ticks time series) with this `Spatial3DView`.

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

Phase 2A (motion tracking) is ready when:

- The Quest client connects to the workstation with the headset neck-mounted
  (no operator HUD required).
- Each controller pose arrives in Python with `tracked`/`valid` flags and both
  `device_time_ns` and `pc_monotonic_ns` timestamps.
- Coordinate / calibration transforms live in unit-tested Python.
- Rerun shows a 3D view tracing each controller's trajectory, plus the two
  gripper wrist cameras and the Feetech width series.
- Moving each gripper visibly draws its trajectory in the Rerun 3D view.
- Opening/closing each physical gripper updates Feetech width in Rerun.
- Reset/calibration is explicit and repeatable, with confirmation surfaced on
  the workstation (no headset UI).
- `python -m handumi.capture.record_handumi_quest` writes the same 16D
  raw state layout used by the current dataset schema.
- No gripper width is taken from Quest trigger values.

Phase 2B (deferred) adds: the Viser robot follows the controller-mounted
grippers via live IK.

## References

- Wearable form factor inspiration: "Portable YUBI" body-worn rig (neck-mounted
  HMD as tracker, gripper-mounted controllers + cameras).
- **Primary** motion-tracking reference: `../yubi-sw` (airoa-org `yubi`),
  `airoa_quest/airoa_quest_bridge/` — native Quest→PC TCP/JSON pose pipeline,
  Unity→robot coordinate conversion, `tracked`/`valid` flags, device/PC clock
  alignment (UDP NTP-style sync).
- Meta XR / OVRPlugin (native app, controller poses + OVR tracking space):
  https://developers.meta.com/horizon/documentation/unity/
- **Secondary** reference (logic only, not the chosen transport): `../axol-vr`
  — button state-machine + reconnect ideas, ported to Python. WebXR links below
  are background only since HandUMI does not use the browser/WebXR path:
  - WebXR Device API: https://www.w3.org/TR/webxr/
  - WebXR input profiles: https://github.com/immersive-web/webxr-input-profiles
- Viser docs (for [2B] only): https://viser.studio/
