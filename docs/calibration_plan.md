# HandUMI spatial calibration

Use this before production collection. The Quest must be rigidly fixed to the
chest; it may move with the operator after the session calibration.
Controller-to-TCP pivot calibration remains a separate prerequisite; see
`docs/README_tcp_offset.md`.

## Board

The canonical printed board is:

- 5 x 7 squares, portrait.
- OpenCV `DICT_5X5_100`, marker IDs 0-16.
- 30.00 mm squares and 15.00 mm markers.
- Current OpenCV pattern (`legacy_pattern: false`).

Print at 100% / Actual size and measure several squares. Accept 30.00 +/- 0.20
mm without unequal X/Y scaling. Mount it flat on a rigid matte surface.

Place it vertically as printed, with marker IDs 15 and 16 closest to the
operator. HandUMI defines the board center as the table origin: +X right, +Y
away from the operator, +Z upward.

## 1. Inspect

With the board visible in a wrist camera:

```bash
handumi-calibrate-spatial --rig-config configs/rig.yaml inspect-board \
  --camera left_wrist
```

The overlay must identify at least 12 ChArUco corners. Space accepts a view; Q
exits.

## 2. Camera intrinsics

Do this once for every camera, and again if its resolution or focus changes.
Move the board through the image, distance, roll, pitch and yaw. Capture 30
sharp views at the production resolution:

```bash
handumi-calibrate-spatial intrinsics --camera left_wrist --views 30
handumi-calibrate-spatial intrinsics --camera right_wrist --views 30
handumi-calibrate-spatial intrinsics --camera workspace --views 30
```

Results accumulate in `outputs/calibration/spatial.yaml`. Target mean
reprojection error <= 0.5 px; the command refuses values above 0.8 px.

## 3. Controller-to-camera mounts

Keep the board fixed. Move each complete HandUMI through at least 24 varied
poses; do not move the controller relative to its wrist camera.

```bash
handumi-calibrate-spatial mount --side left --views 24
handumi-calibrate-spatial mount --side right --views 24
```

Each Space press stores one camera frame and the nearest native Quest pose.
Views with fewer than 12 corners, invalid tracking, or more than 20 ms skew are
rejected. Recalibrate after changing either physical mount.

## 4. Session table frame

At the beginning of a collection session, place the board in its marked table
position and capture 8-12 views:

```bash
handumi-calibrate-spatial session --side left --views 10
handumi-calibrate-spatial verify --side right --views 5
```

`session` also captures five fixed workspace-camera views and writes its table
pose to `outputs/calibration/session.yaml`. Verification with the other wrist
should pass <= 3 mm and <= 1 degree. Inspect the result before removing the
board:

```bash
handumi-calibrate-spatial visualize
```

Rerun shows the calibrated table, three camera feeds/frustums and 10-second
controller trails (left yellow, right green). The board may then be removed as
long as the table, cameras, physical mounts and Quest tracking origin do not
change. Recalibrate after Quest relocalization, tracking reset, mount slippage,
or failed verification.

## 5. Record

```bash
handumi-record --device meta --clap-control \
  --wrist-cameras --workspace-camera \
  --session-calibration outputs/calibration/session.yaml \
  --robot piper --num-episodes 10
```

Double-squeezing either gripper starts or stops an episode. With a session
calibration, this gesture does not recenter on the HMD. The dataset stores raw
Quest poses, table-frame poses, camera/encoder data, synchronization health,
the complete spatial/session calibration, hashes, TCP calibration and target
robot configuration.

Run a 10-episode pilot and require every episode to pass `handumi-validate`
before production. For fine manipulation, also touch known table points with
both TCPs and require <= 2 mm height error and <= 3 mm inter-hand disagreement.
