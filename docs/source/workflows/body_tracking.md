# Body Tracking and Kinematic CoM

HandUMI records Quest/PICO body tracking without changing the established
16-element controller state and action. Canonical 30 Hz body observations use
`handumi_canonical_25_v1`; every accepted native body packet remains available
in the recoverable `raw/tracking` Parquet sidecars.

## Enable CoM and Contact Estimation

Anatomical profile inputs are never guessed. Copy and measure the example:

```bash
cp configs/body_profile.example.yaml configs/body_profile.yaml
```

At minimum, set `height_m` and `mass_kg`, then record with:

```bash
handumi-record \
  --device meta \
  --body-profile configs/body_profile.yaml \
  --repo-id your-name/full-body-demo \
  --output-dir outputs/datasets/full-body-demo
```

At the first episode start, stand upright with both feet on the physical floor
and hold a neutral or T-pose for the default three-second calibration dwell.
HandUMI combines the platform foot estimate with `head - height_m`, rejects
motion or a profile-inconsistent pose, and locks the body and controllers into
one Z-up world whose experimental floor is `z=0`. Change the dwell only when
needed with `--body-neutral-calibration-s`.

This corrects stale Quest Guardian/Stage floor height, but it is still a
profile-assisted platform estimate. A motion-capture/physical-floor reference
is required before making an accuracy claim.

For a quick session, `--body-height-m` and `--body-mass-kg` may be supplied
together. A YAML profile additionally constrains the derived skeleton:

- `height_m` sets floor-to-head stature and `leg_length_m` sets the standing
  floor-to-hip-joint-center target while preserving the platform's neutral
  ankle clearance;
- `arm_span_m`, `shoulder_breadth_m`, and `hip_breadth_m` set bilateral
  geometry;
- `hand_length_m` sets wrist-crease to middle-finger-tip length;
- `foot_length_m` and `foot_width_m` set inferred heel/support geometry;
- `mass_kg` sets segment masses; uncertainties are retained in metadata and
  propagated where the estimator is sensitive to them.

Changed joint positions are labeled `INFERRED`, rendered amber, and never
relabeled as platform measurements. The raw platform skeleton remains in the
tracking sidecar. A custom versioned segment table can be selected with
`--anthropometric-table`.

Measure `hand_length_m` physically with the hand flat, from wrist crease to
middle-finger tip. Meta FullBody's middle-tip joint is used when that value is
absent; in the 2026-07-17 diagnostic capture it reported about 0.188 m on each
side. That number is useful runtime evidence, not the wearer's physical
ground-truth measurement.

For the remaining optional fields, measure arm span from middle fingertip to
middle fingertip in a level T-pose, shoulder breadth between the acromion
landmarks, leg length from the floor to the estimated hip-joint center while
standing, and hip breadth between estimated left/right hip-joint centers.
These surface measurements are constraints on an inferred joint model, not
direct observations of internal anatomy.

## Output Contract

The estimator uses a mass-conserving 15-segment model: head/neck, trunk,
pelvis, and bilateral upper arm, forearm, hand, thigh, shank, and foot. It
stores:

- segment and whole-body CoM, masks, confidence, covariance, and provenance;
- orthogonal CoM projection onto the calibrated ground plane;
- velocity and acceleration only after a causal local-polynomial window is
  complete and timing/relocalization gates pass;
- heel/foot-ball contact probabilities and a support polygon constructed only
  from accepted support feet;
- the exact profile, parameter table, configuration, hashes, and estimator
  version in dataset metadata.

Missing required landmarks invalidate whole-body CoM. Visible segment mass is
never renormalized to hide a dropout. Invalid floating-point outputs remain
NaN with zero masks, and previously estimated poses are never reused.

All profile-constrained joint positions are labeled `INFERRED`; all derived
CoM results are labeled `KINEMATIC_INFERRED`. Contact becomes
`FUSED_ESTIMATED` only when an external contact probability is explicitly
provided. Center of pressure stays unavailable unless a force plate or
pressure insole supplies the required measurements.

Use `handumi-record --rerun` for the synchronized live body view, or
`handumi-view-trajectory` for recorded episodes. See
[Body and Trajectory Visualization](visualization.md) for entity paths,
provenance colors, invalid/stale-clearing rules, and headless `.rrd` export.

## Limitations

The Meta middle fingertip, all other body joints, and the neutral floor remain
platform estimates. The default segment coefficients are a population prior and can be biased by
body composition and population differences. Kinematic foot-contact thresholds
are configurable heuristics. Quest lower-body errors directly affect CoM and
support estimates. These outputs are engineering estimates, not anatomical
accuracy evidence; the project ground-truth validation phase must qualify them.

The default mass-conserving coefficients follow the 15-segment table reported
by [Wang et al. (2022)](https://doi.org/10.3389/fnbot.2022.863722). Custom
tables can encode another de Leva/Zatsiorsky-Seluyanov-style population model
without changing the estimator implementation.
