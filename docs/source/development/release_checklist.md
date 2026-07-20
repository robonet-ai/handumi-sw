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

The current lock intentionally pins `jaxls`, `pyroki`, and optional
`piper_sdk` to immutable Git commits. The
[Python packaging specification](https://packaging.python.org/en/latest/specifications/version-specifiers/#direct-references)
says public indexes should not accept distributions whose metadata contains
direct references, so the current wheel is a source/GitHub-release artifact,
not a PyPI-ready artifact.
Move those dependencies to published releases or a clearly documented optional
integration before enabling PyPI trusted publishing.

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
