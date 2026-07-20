# Body and Trajectory Visualization

HandUMI uses one Rerun layout for live recording, controller-only teleoperation,
and offline episode inspection. The established `controller_trajectory` 3D
view, left/right controller paths, camera row, gripper-width chart, working
bounds, colors, and recorder status remain available while canonical body and
quality entities appear beneath the same hierarchy.

## Live Recording

Enable the recorder-owned Rerun stream with `--rerun`. When a body profile is
provided, the exact aligned `CanonicalBodyFrame` written to each dataset row is
also sent to Rerun; no second tracking provider or headset connection is
opened.

```bash
handumi-record \
  --device meta \
  --repo-id your-name/full-body-demo \
  --output-dir outputs/datasets/full-body-demo \
  --body-profile configs/body_profile.yaml \
  --wrist-cameras --workspace-camera \
  --rerun
```

The recent controller, HMD, and whole-body CoM trails use bounded memory.
Live camera/body delivery uses a bounded background queue and drops stale
viewer frames if rendering falls behind, while every dataset row is still
recorded normally. Rerun failures disable only the viewer: recording,
tracking, cameras, timing, protocols, state/action arrays, and episode gates
continue unchanged. Geometry and cameras begin updating after the episode
recording gate opens, not while the recorder is waiting at the prompt.

Body-enabled Meta recording freezes one common workspace transform for the
controllers, TCPs, and canonical body so an accidental X-button press cannot
move only the controllers. With a body profile, the required upright neutral
dwell places the experimental ground at Rerun `z=0`; profile-constrained joints
are amber because their positions are inferred. For table alignment, pass the
session calibration produced by `handumi-calibrate-spatial`:

```bash
handumi-record ... \
  --session-calibration outputs/calibration/session.yaml \
  --controller-tcp-calibration configs/calibration/meta_controller_tcp.yaml
```

The session transform aligns the coordinate frames. The controller-to-TCP
pivot calibration aligns each rendered TCP with the physical HandUMI tip;
body wrist estimates are not used as a substitute for that physical
calibration.

`handumi-teleop-sim` still uses controller/TCP-only visualization because that
pipeline does not expose an aligned canonical body frame. It never fabricates
body data.

## Recorded Episode Viewer

Open one validated raw episode on explicit `episode_frame` and `episode_time`
timelines:

```bash
handumi-view-trajectory \
  --repo-id your-name/full-body-demo \
  --root outputs/datasets/full-body-demo \
  --episode 0
```

Add `--video` to download/decode recorded camera streams. Create a headless
Rerun recording without opening a window:

```bash
handumi-view-trajectory \
  --repo-id your-name/full-body-demo \
  --root outputs/datasets/full-body-demo \
  --episode 0 \
  --no-spawn --rrd /tmp/handumi-episode-000.rrd
```

The viewer calls `load_raw_episode`; normal metadata/schema validation is not
bypassed. Controller-only episodes (`body=None`) open with the legacy
controller hierarchy and no invented skeleton. A calibrated TCP is derived
only from the dataset's stored controller-to-TCP snapshot. If that snapshot is
absent or invalid, the raw controller is shown and the TCP is reported as
unavailable.

Full paths are built once in O(n), retain invalid gaps, and are split into
bounded chunks. Control their density and chunk sizes with:

```bash
handumi-view-trajectory ... \
  --temporal-decimation 2 \
  --spatial-decimation-m 0.005 \
  --trail-point-cap 2048 \
  --trail-duration-s 10
```

Temporal decimation retains deterministic samples within each contiguous valid
run; spatial decimation drops points closer than the threshold while retaining
run endpoints. `--trail-point-cap` bounds each Rerun line strip, and
`--trail-duration-s` bounds its source-time span without truncating the full
episode.

## Entity Hierarchy and Visibility

Use Rerun's entity tree and blueprint panel to toggle controllers, faint raw
controller anchors, body layers, cameras, and charts. Stable body entities are:

```text
/tracking/body/joints
/tracking/body/skeleton
/tracking/body/segment_com
/tracking/body/whole_com
/tracking/body/whole_com/trail
/tracking/body/com_projection
/tracking/body/com_projection/vertical
/tracking/body/ground
/tracking/body/contacts
/tracking/body/support_polygon
/quality/body/...
```

Left controllers remain yellow and right controllers green. Platform-estimated
joints are cyan; device-reported observations are blue; external trackers are
violet; kinematic inference is amber; future learned estimates are magenta;
and fused CoM/contact output is white. Unknown future numeric provenance uses a
neutral gray fallback and is never relabeled as measured. Lower confidence
reduces alpha and increments the synchronized low-confidence quality signal.
Joint, segment, contact, and CoM point labels are intentionally hidden in the
3D view so they cannot cover the skeleton; the same provenance and diagnostic
details remain available under `/quality/body`.

Tracking state, joint/segment provenance and confidence, CoM provenance and
diagnostics, unresolved mass, contact probabilities/provenance, and clock
quality are logged under `/quality/body`. Current enum values are named in the
quality summary; unknown future values remain visible as `UNKNOWN_<value>`
instead of crashing the viewer.

## Invalid Data and Scientific Meaning

Every layer obeys its own validity mask and finite-value check. Invalid current
joints, edges, segment CoMs, whole CoM, projection, contacts, support polygon,
ground, and CoP are actively cleared. No NaN is sent to a Rerun geometry
archetype, no pelvis is substituted for CoM, and no previous valid pose is
carried forward. CoM trails insert a gap when CoM is invalid, so replay never
connects across a dropout.

The translucent ground mesh is constructed from the calibrated plane equation.
Profile-neutral calibration produces a horizontal `z=0` plane; explicit
external/session calibrations may intentionally produce another plane.
`com_projection` is the orthogonal
**ground-projected CoM**, not center of pressure (CoP). `center_of_pressure` is
displayed only when its explicit validity mask is true; without force/pressure
input it remains absent.

Body source timing and clock quality remain diagnostic-only where recorded as
such. Viewer cursor alignment means the logged controller, body, quality, and
camera values came from the same dataset row; it is not a claim of
synchronization-grade headset timing. The body and CoM model is an engineering
estimate based on platform kinematics and population priors, not an anatomical
accuracy claim.

## Deterministic Headless Fixture

Regenerate the synthetic body/CoM evidence recording without touching captured
hardware artifacts:

```bash
PYTHONPATH=src python bin/generate_ux001_golden.py \
  /tmp/handumi-ux001/body.rrd

PYTHONPATH=src python bin/generate_ux001_golden.py \
  /tmp/handumi-ux001/controller-only.rrd --controller-only
```
