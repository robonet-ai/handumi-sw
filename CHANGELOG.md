# Changelog

All notable changes will be documented here. Until a maintainer approves and
tags a release, entries remain under Unreleased and artifacts remain unsigned
research-preview candidates.

## Unreleased

### Added

- Packet-v2 Meta Quest full-body acquisition with a canonical 25-joint schema,
  raw native-rate sidecars, masks, confidence, provenance, clock diagnostics,
  frame epochs, and neutral/profile calibration artifacts.
- Experimental kinematic segment/whole-body CoM, contact, and support outputs,
  with live/offline Rerun visualization and explicit invalid-data clearing.
- Recorder-owned, non-fatal Piper Viser/IK visualization using the same aligned
  capture and one set of hardware providers.
- Hardware preflight checks for tracking, cameras, Feetech devices,
  calibrations, ports, storage, optional dependencies, and local rig remapping.
- Pinned CI, installed-wheel smoke tests, headless Rerun exports, checksums,
  dependency/license inventory, CycloneDX SBOM, vulnerability audit, and
  full-history secret scanning.

### Changed

- The bounded wheel excludes source-only Git integrations and heavyweight
  recording/IK dependencies; source installs retain pinned uv groups.
- R1 Lite assets and configuration are excluded from built distributions until
  compatible redistribution terms are documented.
- Body/profile estimates and operational documentation use research-preview
  claims pending TEST-001 ground-truth validation.

### Security

- Rerun and Viser default to loopback; LAN exposure is explicit.
- Quest TCP/UDP transport is documented as unauthenticated plaintext for a
  trusted, isolated local network.
- Privacy, retention, responsible disclosure, and independent physical-robot
  safety boundaries are documented in SECURITY.md.

### Known release blockers

- Asset/legal approval, repository ownership, Quest signing/build approval,
  hardware validation and 30-minute soak, hosted CI, and TEST-001 laboratory
  evidence remain incomplete.
