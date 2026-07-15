# HandUMI canonical body and tracking sidecars

DATA-001 adds optional full-body data without changing the established
`observation.state: float32[16]` and `action: float32[16]` contract.

## Two-rate storage

Aligned LeRobot rows contain `observation.body.*` fields using
`handumi_canonical_25_v1`. These are convenient 30 Hz observations, not the
native source of truth. Every accepted native tracking packet is drained into
`raw/tracking` Parquet sidecars described by `raw/tracking/manifest.json`.
Sidecars retain source joint arrays, flags, timing, calibration/fidelity state,
provenance, and the original source JSON, including unknown future fields.

Each episode is first appended to an fsynced
`raw/tracking/inprogress/*.jsonl.inprogress` journal. Saving the episode
atomically publishes its Parquet file. Discarded and interrupted attempts are
published under their own directories instead of silently deleting evidence.

## Canonical model

`handumi_canonical_25_v1` contains pelvis; lower/middle/upper spine; chest;
neck; head; bilateral shoulder, elbow, wrist, hand; and bilateral hip, knee,
ankle, heel, and foot ball. The stable identifiers and parent indices are
stored in `meta/info.json`. Platform root is separate.

All poses are meters and right-handed normalized `xyzw` quaternions in
`handumi_world`: X initial horizontal heading, Y left, Z up, with origin on the
calibrated ground plane. Meta Unity poses cross an explicit handedness boundary
before the ground/heading transform. The legacy HMD-centered workspace is not
used for body dynamics. Camera, legacy workspace, and optional mocap transforms
remain named edges in an explicit transform graph.

Unavailable floating-point values are NaN and have zero validity masks. Invalid
source joints are never replaced with cached values. With a measured body
profile, `handumi_kinematic_com_v1` produces covariance-bearing segment/whole
CoM, smoothed derivatives, inferred heel/contact probabilities, and a support
polygon. These remain explicitly inferred. Without a profile they stay
unavailable, and center of pressure always stays unavailable without force or
pressure measurements.

The PICO 24-joint index table follows the public `BodyTrackerRole` order in the
[XRoboToolkit Unity client](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/blob/main/PICO%20Unity%20Integration%20SDK/Runtime/Scripts/PXR_Plugin.cs).
Meta mappings follow the Meta XR SDK 74 `OVRSkeletonMapping` and
`OVRHumanBodyBonesMappings` hierarchy shipped with the Quest application.

## Readers and conversion

`load_raw_episode()` returns `body=None` for current controller-only recordings.
It never constructs empty body poses when body columns are absent. Body-enabled
datasets return a typed optional body group and episode sidecar paths. Historic
pre-compact HandUMI layouts remain subject to the mainline reader's explicit
re-recording requirement.

Derived robot datasets preserve the original behavior by default. Pass
`handumi-convert --preserve-body` to carry aligned body columns and native
sidecars into the derived dataset. The last aligned body row is removed with
the last source state so it stays synchronized with the existing `(state_t,
action_t+1)` conversion.
