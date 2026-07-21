# TEST-001 preregistration and laboratory protocol template

Status: template only—HandUMI is **not scientifically validated** until an
approved, completed laboratory study supplies retained evidence.

## Preregistration

- State primary and secondary hypotheses before collection.
- Define source, HandUMI world, table, mocap, and force-plate frames, their
  handedness/axes, calibration method, transform uncertainty, and acceptance.
- Name primary and secondary metrics: joint/segment position and orientation,
  availability, CoM, contact/support classification, temporal offset, jitter,
  dropout frequency/duration, and recovery time.
- Define exclusions before analysis: consent withdrawal, reference-system
  failure, corrupted synchronization, protocol deviation, and minimum usable
  duration. Never exclude solely because HandUMI error is high.
- Define transport-gap, invalid-tracking, relocalization, frame-epoch-change,
  and unavailable-reference dropout policies separately.
- Aggregate within participants first. State stratification/cluster bootstrap,
  deterministic seed, confidence interval, missing-data reporting, minimum
  participant/sample guards, and multiple-comparison correction.

## Synchronized acquisition runbook

1. Obtain ethics/privacy approval and informed consent; assign study IDs in the
   laboratory system, never in publishable HandUMI manifests.
2. Calibrate mocap, force plates, table, headset/controller rigid clusters, and
   all transforms. Verify right-handed axes, composition, and round trips.
3. Configure a hardware-visible event (LED/photodiode and/or electrical pulse)
   observed by video/tracking/reference systems. Record source/host timestamps,
   uncertainty, sequence, and epoch using `handumi_sync_event_v1`.
4. Capture the preregistered static/dynamic motions, reference availability,
   raw native-rate sidecars, frame epochs, and all dropout reasons.
5. Preserve raw evidence under the approved retention/access policy. Review for
   faces, room imagery, identifiers, device serials, and private paths before
   publication; honor consent withdrawal and deletion requirements.
6. Run the hashed configuration/input report pipeline. Review participant-level
   distributions and missingness. Software golden results do not substitute for
   mocap, force-plate, participant, or hardware evidence.

Force plates do not directly provide anatomical 3D CoM. Estimated balance,
contact, CoM, and support state must never act as a physical safety interlock.
