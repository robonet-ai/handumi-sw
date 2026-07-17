# Replay a Local Recording in Simulation

`handumi-replay-in-sim` retargets one raw HandUMI episode to a configured
bimanual robot and displays the target and achieved TCP trajectories in Viser.
The source recording remains robot agnostic: the robot model, IK profile, and
table placement are selected when replay starts.

## Use a Local Dataset

Pass the recording directory with `--dataset-root`. A local replay does not
need `--repo-id` and does not download data:

```bash
JAX_PLATFORMS=cpu uv run handumi-replay-in-sim \
  --dataset-root outputs/20260714_224135 \
  --robot openarmv1 \
  --episode 0
```

Change `--episode` to select another episode. Add `--headless` when only the
IK result and saved NPZ are needed. Without it, open the URL printed by Viser,
normally <http://localhost:8080>.

`JAX_PLATFORMS=cpu` is recommended on workstations that have the JAX CUDA
plugin but not a working CUPTI installation. Without it, JAX can print an
`Unable to load cuPTI` traceback and then continue on CPU; the message does not
mean that replay or IK failed.

## Absolute-table Retargeting

Recordings captured in the calibrated table workspace normally select
`absolute-table` automatically. The explicit form is useful when auditing a
new embodiment:

```bash
JAX_PLATFORMS=cpu uv run handumi-replay-in-sim \
  --dataset-root outputs/20260714_224135 \
  --robot openarmv1 \
  --episode 0 \
  --retarget-mode absolute-table \
  --deployment-calibration configs/calibration/openarmv1_table.yaml
```

`robot_from_table` places the demonstrated table frame in the robot world. It
does **not** move the robot base. For OpenArm v1, for example, the URDF pedestal
remains fixed to world `Z=0`, the shoulder mounts are at `Z=0.698 m`, and the
calibration's `Z=0.28755 m` is the provisional table-plane height.

Keep `verified: false` for simulation-derived transforms. Measure the physical
table pose and change it to `true` before relying on absolute placement on real
hardware. Do not use `robot_from_table` to compensate for an incorrect
Controller-to-TCP calibration.

## OpenArm v1

The current OpenArm profile uses a larger offline-only joint step than live
teleoperation:

```yaml
replay:
  max_joint_delta: 0.35
```

This does not change the real OpenArm command rate, speed limits, watchdog, or
following-error checks. The simulation URDF also keeps approximately `0.48 mm`
of clearance between the finger collision meshes at the closed `0 mm`
position. The real backend retains its native closed/open motor calibration.

For `outputs/20260714_224135`, the provisional rigid table transform produces:

| Episode | Maximum TCP position error | Result against 3 cm threshold |
| --- | ---: | --- |
| 0 | 2.92 cm | Pass |
| 1 | 4.71 cm | Review unreachable segment |
| 2 | 4.34 cm | Review unreachable segment |

Do not simply reduce the table translation in X to hide those peaks. In the
same recording, values at or below `X=0.168 m` make episode 0 cross a singular
branch and create 17--20 cm errors. A future reach limiter or workspace scaling
policy is preferable to distorting the measured table transform.

## TRLC-DK1

TRLC-DK1 currently supports bimanual kinematic replay in simulation. It does
not yet provide a HandUMI real-hardware backend.

```bash
JAX_PLATFORMS=cpu uv run handumi-replay-in-sim \
  --dataset-root outputs/20260714_224135 \
  --robot trlc_dk1 \
  --episode 0 \
  --retarget-mode absolute-table \
  --deployment-calibration configs/calibration/trlc_dk1_table.yaml
```

The bimanual URDF uses two namespaced DK1 followers with a provisional `0.60 m`
base separation. The table transform is also provisional. On episode 0 of the
recording above, the current profile produced `0.22 cm` maximum position error
and `22.19 degrees` maximum orientation error.

TRLC meshes use paths such as `meshes/visual/base_link.glb`, resolved relative
to `assets/trlc-dk1`. If Viser prints `Can't find meshes/...` and shows only
trajectory lines, update the checkout and restart the replay process so the
URDF is loaded again.

## Reading the Diagnostics

Replay prints the source tool identity and calibration hash before solving.
Seeing `source tool: robot=piper` while replaying OpenArm or TRLC is expected
when Piper was the physical tool used to make the recording. The identity-bound
Controller-to-TCP snapshot reconstructs the demonstrated Piper TCP; the target
embodiment is applied afterward.

Important output fields are:

- `start prepared`: initial solve iterations and first-frame error;
- `IK EE error`: mean and maximum position/orientation error over both arms;
- `max_joint_delta`: the offline joint-step limit selected for the embodiment;
- `saved`: the NPZ containing targets, achieved TCP poses, errors, and qpos.

Use `--strict-ik` in automated validation. It exits when the maximum position
or orientation error exceeds the selected thresholds:

```bash
JAX_PLATFORMS=cpu uv run handumi-replay-in-sim \
  --dataset-root outputs/20260714_224135 \
  --robot trlc_dk1 \
  --episode 0 \
  --headless \
  --strict-ik
```
