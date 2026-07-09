# HandUMI Calibration Guide (controller → gripper TCP)

The controller sits in the HandUMI mount at a fixed pose relative to the
gripper tip (the TCP). Recordings store **raw controller poses**; the
controller → TCP transform is applied afterwards (replay, conversion), read
from:

```text
configs/calibration/{pico,meta}_controller_tcp.yaml
```

Calibrating that transform well matters because it is baked into every
converted/replayed trajectory: a wrong rotation twists every wrist motion,
a wrong translation shifts the tip by centimeters exactly during grasps.

The flow is **record → solve → verify**, all post-hoc — no live tooling
needed:

| Step | What | Tool |
|------|------|------|
| 1 | Record a pivot episode per side | `handumi-record` |
| 2 | Solve the translation | `handumi-calibrate-tcp-offset pivot` |
| 3 | Solve the rotation | `handumi-calibrate-tcp-offset orient` |
| 4 | Verify by replaying through IK | `handumi-replay-in-sim` |

Redo only when the physical mount changes. The two HandUMI devices are
mirror twins, so left/right results must be Y-mirrors of each other —
position `(x, -y, z)`, quaternion `(-x, y, -z, w)`; a big asymmetry means
one measurement is bad.

Prerequisites: tracking streaming and `trk=1` / non-zero poses
([README_quest.md](README_quest.md) or [README_pico.md](README_pico.md)),
plus camera/Feetech setup if you record with them (not required here:
`--skip-feetech` is fine for calibration episodes).

---

## 1. Record a pivot episode

Pivot calibration: pin the gripper TIP on a fixed point (a pencil dot, or
better a small dimple/cradle the tip can't slide out of) and, without
letting the tip slip, rotate the whole device through as many orientations
as you can for ~25s. Every recorded controller pose then satisfies
`p_i + R_i @ t = fixed point`, which pins down `t` — the tip position in
the controller frame.

```bash
handumi-record --device meta --skip-feetech \
  --repo-id local/tcp_pivot_left \
  --output-dir outputs/datasets/tcp_pivot_left \
  --task "tcp pivot left" --num-episodes 1 --episode-time-s 25
```

Record one episode per side (repeat with `tcp_pivot_right`, pinning the
right gripper). More orientation variety = better conditioning.

## 2. Solve the translation

```bash
handumi-calibrate-tcp-offset pivot --device meta --side left \
  --parquet outputs/datasets/tcp_pivot_left/data/chunk-000/file-000.parquet
```

(`--csv`/`--episode` are alternative inputs — see `--help`). It solves the
least-squares pivot fit and writes the `position` into
`configs/calibration/meta_controller_tcp.yaml`. Check the reported RMS
residual: **< 5 mm is good**; higher means the tip slipped — re-record.
Repeat for `--side right` and check the Y-mirror symmetry.

## 3. Solve the rotation

```bash
handumi-calibrate-tcp-offset orient --device meta --side left \
  --parquet outputs/datasets/<recording>/data/chunk-000/file-000.parquet \
  --tcp-quat-world <qx> <qy> <qz> <qw>
```

Record a short clip holding the gripper in a known world orientation (e.g.
pointing straight forward, level), then pass that orientation. Alternative:
the two-stance method with `handumi-print-controller-pose` (hold the bare
controller naturally, then mounted, both pointing the same way; the offset
is `conj(q_mounted) * q_bare`) — paste the result into the YAML manually.

Inspect the final file at any time:

```bash
handumi-calibrate-tcp-offset inspect --device meta
```

## 4. Verify

Record a short natural episode (wrist rotations, a square drawn in the
air), then replay it through bimanual IK in Viser:

```bash
handumi-record --device meta --skip-feetech \
  --repo-id local/tcp_verify --output-dir outputs/datasets/tcp_verify \
  --task "tcp verify" --num-episodes 1 --episode-time-s 15

handumi-replay-in-sim --repo-id local/tcp_verify \
  --dataset-root outputs/datasets/tcp_verify --robot piper
```

The replayed TCP motion must match what the real gripper tip did:

- Pure wrist rotations about a still tip should keep the sim TCP nearly
  still (a sweeping arc = wrong translation, step 2).
- Shapes (the air-square) must come out the same shape, not rotated or
  sheared (wrong rotation, step 3).

Faster iteration: `handumi-live --device meta` runs the same
calibration + retargeting + IK pipeline live (Viser robot + Rerun TCP
trails, no recording) — do the wrist-rotation and air-square checks in
real time, then confirm with a recorded replay.

## Troubleshooting

- **High pivot RMS** → the tip slipped; use a dimple/cradle and re-record.
- **Left/right not Y-mirrored** → one side's measurement is bad; redo the
  worse (higher-RMS) side.
- **Replay looks rotated/sheared** → rotation offset (step 3).
- **Replay tip sweeps during wrist-only rotations** → translation offset
  (step 2).
- **`trk=0` / frozen poses while recording** → controllers asleep or out
  of the headset cameras' view.
