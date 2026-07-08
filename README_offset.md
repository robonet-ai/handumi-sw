# HandUMI Calibration Guide (controller → TCP → robot world)

Three calibrations make teleop absolute: real HandUMI motion replays 1:1 on
the simulated arms, and a task scene (`--scene cube_in_box`) stays aligned
between sim and reality. They also validate the TCP→joints IK conversion
used later to post-process datasets to joint level — if the live mirror in
Viser is right, that conversion is right.

**Run them in this order.** Each one assumes the previous is done.

| # | What | Tool | Fixes | How often |
|---|------|------|-------|-----------|
| 1 | Mount rotation | `handumi-print-controller-pose` | Recorded orientations + IK wrist angles | Once (redo only if the mount changes) |
| 2 | Mount position | `handumi-calibrate-tcp-offset` | TCP position, especially under wrist rotations | Once (same) |
| 3 | Workspace → robot world | `handumi-calibrate-workspace` | Sim/real scene alignment (Viser mirror only) | Per session |

Calibrations 1-2 affect the **recorded data** — get them right before
collecting datasets. Calibration 3 only affects the **live Viser mirror**;
raw datasets are stored in the workspace frame and don't depend on it.

Prerequisites: Quest streaming (`README_quest.md` steps 1-5), controllers
awake and visible to the headset cameras (`trk=1` in
`python -m handumi.tracking.meta_quest --config configs/tracking_meta_quest.yaml`).

---

## 1. Mount rotation (controller → gripper TCP orientation)

The controller sits in the HandUMI mount at an arbitrary tilt. Without this,
every recorded orientation is rotated and the IK bends the Piper wrist the
wrong way.

Two-stance method — the *difference* between the stances cancels the
tracking frame's arbitrary yaw, so neither stance needs to be aligned with
anything:

1. Start the printer:

   ```bash
   handumi-print-controller-pose
   ```

2. **Stance A** — hold the BARE left controller (out of the HandUMI) in a
   natural handheld grip, pointing forward (toward the arms). Hold still
   2-3s, note the printed quaternion `q_A`.
3. **Stance B** — mount it in the HandUMI and hold the device pointing the
   SAME forward direction. Note `q_B`.
4. The offset is `conj(q_B) * q_A` (use `quat_multiply` / `quat_conjugate`
   from `handumi.tracking.transforms`).
5. The right side must be the exact Y-mirror of the left (the two devices
   are physical mirror twins): `(-x, y, -z, w)` of the left quaternion.
   Don't measure both sides independently — measure left, mirror it.
6. Paste both into `configs/tracking_meta_quest.yaml` →
   `calibration.controller_to_gripper_tcp.<side>.quaternion`.

---

## 2. Mount position (controller → gripper TCP translation)

The gripper tip sits ~14cm from the controller's tracking anchor. Without
this, wrist rotations put the estimated tip several cm off — fatal for
picks.

Pivot calibration — pin the tip, rotate the device, least-squares solves the
offset:

1. Mark a fixed point on the table (pencil dot, tape corner).
2. Run:

   ```bash
   handumi-calibrate-tcp-offset --side left
   ```

3. Rest the LEFT gripper tip on the mark, press Enter.
4. **Without letting the tip slip**, rotate the whole device through as many
   orientations as you can for the 25s window (tilt, roll, yaw — more
   variety = better conditioning).
5. Read the output:
   - `RMS residual < 5 mm` → good. Higher → the tip slipped; re-run.
   - It prints the YAML `position` for this side plus the Y-mirror for the
     other side.
6. Repeat with `--side right` and check it matches the printed mirror
   (within a few mm). If it does, keep the mirrored pair (average if you
   want): left `[x, y, z]`, right `[x, -y, z]`.
7. Paste into `configs/tracking_meta_quest.yaml` →
   `calibration.controller_to_gripper_tcp.<side>.position`.

### Checkpoint for 1+2 (do this before step 3)

```bash
handumi-live-tracking-quest --skip-cameras --skip-feetech
```

In Rerun each hand now draws two trails: **faint = raw controller anchor,
solid = estimated gripper TCP**.

- Rotate the wrist without moving your hand: the faint trail barely moves,
  the solid trail sweeps an arc (the real tip). Wrong arc radius → redo 2.
- Draw a square in the air: both trails must show the same square shape.
  Deformed/rotated solid trail → redo 1.

---

## 3. Workspace → robot world (sim/real scene alignment)

Maps your real working volume into the Piper's world so the sim scene sits
exactly where the real one does. Per-session: the workspace origin is the
HMD pose at the last reset, so it depends on where you stood — recalibrate
at the start of each session, and don't reset the workspace mid-session
(if you do, redo this step; it takes ~20s).

1. Place the real task scene at coordinates you know in the robot world.
   Example: `configs/scene.yaml` puts the box at `[0.35, 0.0, 0.0]` →
   place the real box 35cm in front of the midpoint between the two arm
   base plates, at base height.
2. Wear the Quest as in a real session (on the neck), stand in your normal
   operating spot, and run:

   ```bash
   handumi-calibrate-workspace --side right
   ```

   It sets the workspace origin from the first tracked HMD pose — don't
   move from your spot afterwards.
3. When prompted, type the robot-world coordinates of the reference point
   (e.g. `0.35 0.0 0.0`), then touch it with the RIGHT gripper tip, hold
   still, press Enter (it averages ~1s of samples).
4. Recommended: add a second point (e.g. a second mark at `0.35 0.2 0.0`)
   so the fit also solves the yaw (your body's rotation relative to the
   arms). One point fixes translation only (yaw = 0).
5. Press Enter on an empty line to finish; paste the printed block into
   `configs/teleop.yaml`.

### Final checkpoint (validates the whole TCP→joints chain)

```bash
handumi-live-tracking-quest --robot piper --scene cube_in_box
```

- Touch the real box corner with the real gripper tip → the TCP sphere in
  Viser must touch the sim box corner. Off by a constant shift → redo 3.
- Do a full real pick & place → the Piper must repeat it on the sim cube.
- If the TCP sphere drifts far from the rendered arm tip during motion,
  the IK target is outside the Piper's reach — usually a bad step-3
  translation sending your hands too far/high.

---

## Troubleshooting

- **Solid/faint Rerun trails differ in shape** → mount calibration (1-2).
- **Everything shifted by a constant amount in Viser** → workspace (3).
- **Sphere detaches from the arm while moving** → target out of reach:
  bad step 3, or you're working outside the Piper's envelope.
- **`Only N tracked samples`** → controllers asleep / out of camera view;
  check `trk=1` first.
- **High pivot RMS** → the tip slipped; use a dimple/cradle the tip can't
  slide out of and re-run.
