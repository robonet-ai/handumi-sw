# Research-Preview Release Checklist

HandUMI may be released only as a **research preview** until TEST-001 passes
independent ground-truth validation. Full-body pose, CoM, contact, support,
profile-constrained geometry, and platform body timing must remain labeled as
experimental estimates.

## Required before a source release

- Review the exact workstation and Quest source commits; exclude Unity temp
  files, datasets, calibration caches, device identifiers, private network
  addresses, signing material, and participant captures.
- Run `uv lock --check`, `uv sync --locked --dev`, and
  `uv run --locked pytest -q` on Python 3.12.
- Build documentation with warnings treated as errors and build both
  distributions with `uv build`.
- Inspect the wheel/sdist contents, SHA-256 hash every published artifact, and
  retain the exact dependency lock and Quest APK build manifest.
- Run Android-target Unity EditMode and PlayMode tests and the worn-headset
  body-probe runbook from the Quest repository. APKs require their own hash,
  signer, source commit, Unity/Meta/OpenXR versions, and device/runtime record.
- Verify a neutral body-profile dwell in live Rerun and a physical full-stroke
  sweep for both grippers. Do not substitute offline replay or a plausible
  screenshot for these hardware checks.

## Distribution boundary

The source lock intentionally pins `jaxls`, `pyroki`, and optional robot SDKs
to immutable Git commits. Those integrations live in source-only uv dependency
groups and are deliberately absent from wheel metadata. The bounded wheel can
therefore be installed without Git or the heavyweight IK/recording stack, but
it is still only a GitHub/source release candidate: the source-only integration
policy, all runtime modes, asset rights, project ownership, and publication
authority have not passed their release gates. Do not enable PyPI trusted
publishing merely because the metadata has no direct references.

The tracked R1 Lite URDF and meshes have no documented redistribution grant.
The build configuration must exclude the model and its runtime configuration
from wheel and sdist artifacts until legal review records compatible terms.

The default uv environment resolves LeRobot's PyTorch dependencies from the
official CPU-only index. This keeps recording/validation/CI installs bounded;
accelerator environments must be managed and validated separately.

## Evidence and claims

Software tests establish contract behavior, not human-body accuracy. A release
note must enumerate passed software/Quest/HIL checks and every skipped or failed
gate. TEST-001 mocap/force-plate results are required before any anatomical,
timing, balance, contact, support, medical, ergonomic, or safety claim.

See `SECURITY.md` for private disclosure and sensitive-data handling and
`CONTRIBUTING.md` for contributor validation expectations.
