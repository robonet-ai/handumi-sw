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

For a quick session, `--body-height-m` and `--body-mass-kg` may be supplied
together. Optional measured foot length and width improve inferred heel and
support-foot geometry. A custom versioned segment table can be selected with
`--anthropometric-table`.

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

All derived results are labeled `KINEMATIC_INFERRED`. Contact becomes
`FUSED_ESTIMATED` only when an external contact probability is explicitly
provided. Center of pressure stays unavailable unless a force plate or
pressure insole supplies the required measurements.

## Limitations

The default segment coefficients are a population prior and can be biased by
body composition and population differences. Kinematic foot-contact thresholds
are configurable heuristics. Quest lower-body errors directly affect CoM and
support estimates. These outputs are engineering estimates, not anatomical
accuracy evidence; the project ground-truth validation phase must qualify them.

The default mass-conserving coefficients follow the 15-segment table reported
by [Wang et al. (2022)](https://doi.org/10.3389/fnbot.2022.863722). Custom
tables can encode another de Leva/Zatsiorsky-Seluyanov-style population model
without changing the estimator implementation.
