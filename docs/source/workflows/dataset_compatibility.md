# Dataset compatibility and recovery policy

This is the authoritative compatibility policy for the HandUMI research
preview. Readers, validation, conversion, replay, Rerun, and robot replay all
enter through the same validated reader.

The supported dataset layout is LeRobot v3 with tracking schema
`controller_raw_compact`, capture schema `synchronized_sources`, and state
semantics `workspace_controller_pose7_plus_gripper_widths`. Both
`observation.state` and `action` remain `float32[16]`. Controller-only datasets
remain supported with `body=None`. The canonical 25-joint body fields, masks,
confidence, provenance, optional CoM/contact/support estimates, frame epochs,
and native-rate raw sidecars are additive. Missing optional values remain
missing; readers never fabricate them.

Packet-v2 is the capture contract. The legacy controller packet accepted by
the network adapter remains controller-only and is normalized without adding
body fields. Unknown packet/schema versions fail explicitly.

The pre-compact combination `controller_raw_and_workspace_v3` plus
`synchronized_sources_v1` is rejected with cutoff `2026-07-10T00:00:00Z`.
Evidence in that layout does not identify one unambiguous meaning for the
duplicated raw/workspace controller state. A generic importer would have to
guess semantics, so none is provided. Preserve such recordings and re-record
with the current recorder. A future importer must be format-specific and backed
by provenance evidence and golden conversion results.

An episode is complete only when its session manifest says `complete` (or it
is a supported older compact dataset created before session manifests). A
staging directory is recoverable forensic evidence, never complete data.
`incomplete`, `rejected`, corrupt, and unknown states fail closed. Recovery
preserves staging contents and changes their classification; it never promotes
partial rows as complete.
