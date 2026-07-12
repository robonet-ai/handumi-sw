# Controller → Gripper TCP Offset

Ultima modificacion: 2026-07-11 19:51:01 -05 -0500

The fixed pose of the mounted controller relative to the gripper tip.
Recordings store raw controller poses; this transform
(`T_world_tcp = T_world_controller @ T_controller_tcp`) is applied
post-hoc by replay/conversion. Stored per device in
`configs/calibration/{meta,pico}_controller_tcp.yaml` (committed — it is a
property of the mount design). Redo only when the 3D-printed mount changes.

Gripper-width calibration is separate:
[README_gripper_width.md](README_gripper_width.md).

## Calibrate

**Translation** — pivot method: pin the gripper TIP on a fixed point and
rotate the whole device in all directions for ~25s while recording:

```bash
handumi-record --device meta --skip-feetech \
  --repo-id local/tcp_pivot_left --output-dir outputs/datasets/tcp_pivot_left \
  --task "tcp pivot left" --num-episodes 1 --episode-time-s 25

handumi-calibrate-tcp-offset pivot --device meta --side left \
  --parquet outputs/datasets/tcp_pivot_left/data/chunk-000/file-000.parquet
```

RMS residual **< 5 mm** = good; higher = the tip slipped, re-record.
Repeat per side.

**Rotation** — record a short clip holding the gripper in a known world
orientation, then:

```bash
handumi-calibrate-tcp-offset orient --device meta --side left \
  --parquet <recording>.parquet --tcp-quat-world <qx> <qy> <qz> <qw>
```

Both subcommands write the YAML in `configs/calibration/` directly.
Inspect it anytime:

```bash
handumi-calibrate-tcp-offset inspect --device meta
```

## Verify

Live (fastest): `handumi-teleop-sim --device meta` — the robot follows you in
Viser through the same calibration + IK the replay uses.

- Wrist-only rotations about a still tip → the sim TCP stays nearly still
  (sweeping arc = translation wrong).
- A square drawn in the air → same square in sim, not rotated/sheared
  (else rotation wrong).

Then confirm on a recording: `handumi-record` a short episode and
`handumi-replay-in-sim` it — same checks, plus per-frame EE errors.

## Troubleshooting

- **High pivot RMS** → tip slipped; use a dimple/cradle, re-record.
- **Replay/live rotated or sheared** → rotation offset.
- **Tip sweeps during wrist-only rotations** → translation offset.
- **Widths stuck at 0 / not moving** → gripper calibration missing or
  ports wrong ([README_gripper_width.md](README_gripper_width.md)).
- **`trk=0` / frozen poses** → controllers asleep or out of the headset
  cameras' view.
